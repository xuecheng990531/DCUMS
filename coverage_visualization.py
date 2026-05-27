import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from models.transreunet_class import AttentionUNet
from models.unet import UNet2D
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader


# -----------------------------
# Helper: top-K selection (deterministic)
# -----------------------------
def topk_indices(scores, K):
    K = min(K, len(scores))
    return np.argsort(-scores)[:K]


# -----------------------------
# 1) Feature extraction (fixed hook leak: register once, remove once)
# -----------------------------
def extract_features(model, data_loader, device):
    model.eval()
    feats_all = []
    hook_feats = []

    def hook_fn(module, inp, out):
        hook_feats.append(out)

    if isinstance(model, AttentionUNet):
        handle = model.Conv4.register_forward_hook(hook_fn)
    elif isinstance(model, UNet2D):
        handle = model.encode[3].register_forward_hook(hook_fn)
    else:
        raise ValueError("Unsupported model type for feature extraction.")

    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc="Extracting features"):
            x = x.to(device)
            hook_feats.clear()
            _ = model(x)

            if len(hook_feats) == 0:
                raise RuntimeError("Hook did not capture features.")

            f = hook_feats[0]                   # [B, C, H, W]
            f = F.adaptive_avg_pool2d(f, (1, 1)) # [B, C, 1, 1]
            f = f.view(f.size(0), -1)            # [B, C]
            feats_all.append(f.cpu().numpy())

    handle.remove()
    return np.concatenate(feats_all, axis=0)     # [N, C]


# -----------------------------
# 2) Sample-level entropy u_i (mean pixel entropy)
# -----------------------------
def compute_entropy_u(model, data_loader, device, eps=1e-7):
    model.eval()
    u_list = []
    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc="Computing u_i (entropy)"):
            x = x.to(device)
            logits = model(x)  # [B, C, H, W]
            probs = torch.softmax(logits, dim=1).clamp(eps, 1.0 - eps)
            ent_map = -torch.sum(probs * probs.log(), dim=1)         # [B, H, W]
            u = ent_map.view(ent_map.size(0), -1).mean(dim=1)        # [B]
            u_list.append(u.cpu().numpy())
    return np.concatenate(u_list, axis=0) if len(u_list) else np.array([])


# -----------------------------
# 3) UMS sampling probability pi_i
#    pi_i = (1-tau)*eta*u_i + tau*b
#    eta = b / mean(u), b = K/N
# -----------------------------
def compute_pi_ums(u, K, tau=0.2, eps=1e-12):
    N = len(u)
    if N == 0:
        return np.array([])
    b = K / N
    mu = float(np.mean(u))
    if mu < eps:
        mu = eps
    eta = b / mu
    pi = (1.0 - tau) * eta * u + tau * b
    return np.clip(pi, 0.0, 1.0)


# -----------------------------
# 4) Bernoulli sampling + hard budget K (force |S|=K for fair comparison/visualization)
#    - draw S0 by Bern(pi)
#    - if |S0| > K: downsample S0 w/o replacement using weights
#    - if |S0| < K: top-up from complement w/o replacement using weights
# -----------------------------
def bernoulli_budgeted_sample(pi, K, rng, weights_for_resample=None):
    N = len(pi)
    if N == 0 or K <= 0:
        return np.array([], dtype=int)

    w = pi.copy() if weights_for_resample is None else weights_for_resample.copy()

    xi = rng.random(N) < pi
    S0 = np.where(xi)[0]

    def _sample_wo_replacement(idxs, size):
        ww = w[idxs].astype(np.float64)
        if ww.sum() <= 0:
            ww = np.ones_like(ww, dtype=np.float64)
        ww = ww / ww.sum()
        return rng.choice(idxs, size=size, replace=False, p=ww)

    if len(S0) == K:
        return S0
    elif len(S0) > K:
        return _sample_wo_replacement(S0, K)
    else:
        remaining = np.setdiff1d(np.arange(N), S0, assume_unique=False)
        need = K - len(S0)
        if len(remaining) <= need:
            extra = remaining
        else:
            extra = _sample_wo_replacement(remaining, need)
        return np.concatenate([S0, extra], axis=0)


# -----------------------------
# 5) Coverage: cluster coverage in high-d feature space
# -----------------------------
def fit_kmeans_labels(features_ref, n_clusters=10, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(features_ref)
    return labels


def cluster_coverage_from_labels(labels, selected_indices):
    total = len(set(labels))
    covered = len(set(labels[selected_indices]))
    return covered, total


# -----------------------------
# 6) Redundancy metrics (higher distance => less redundant)
# -----------------------------
def _pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    return D


def redundancy_metrics(features_ref, selected_indices):
    X = features_ref[selected_indices]  # [K, D]
    K = X.shape[0]
    if K <= 1:
        return 0.0, 0.0

    D = _pairwise_distances(X)
    mean_pair = (D.sum() - np.trace(D)) / (K * (K - 1))

    D2 = D.copy()
    np.fill_diagonal(D2, np.inf)
    nn = D2.min(axis=1).mean()

    return float(mean_pair), float(nn)


# -----------------------------
# 7) t-SNE visualization (qualitative only)
# -----------------------------
def plot_tsne(features_ref, labels, idx_ent, idx_ums, save_path, title=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    z = tsne.fit_transform(features_ref)

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 15})
    # 使用 Dark2 色板，颜色更深且区分度高，同时增大背景点尺寸 s=40
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap="Dark2", alpha=0.6, s=40, label="All unlabeled samples")
    
    # 保持 UMS 和 Entropy 标记的显著区分
    plt.scatter(z[idx_ent, 0], z[idx_ent, 1], marker="^", s=250, color="#d62728", edgecolor="black", linewidth=1.5,
                label=f"Entropy+Base")
    plt.scatter(z[idx_ums, 0], z[idx_ums, 1], marker="o", s=250, color="#1f77b4", edgecolor="black", linewidth=1.5,
                label=f"UMS+Base")
    
    if title:
        plt.title(title, fontsize=15, pad=20)
    plt.legend(fontsize=12, loc='best')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# 8) Scheme A:
#    - Entropy baseline: classic top-K by u_ent (deterministic)
#    - UMS: paper-style Bernoulli + budget control using pi_ums (stochastic, multi-seed)
#    NOTE: You asked NOT to force same model; keep UMS and Entropy models separate.
#    Coverage/Redundancy are evaluated in ONE reference feature space (features_ref),
#    by default extracted from the UMS model.
# -----------------------------
def evaluate_coverage_and_redundancy_multi_seed_schemeA(
    model_ums, w_ums,
    model_ent, w_ent,
    unlabelled_loader,
    device,
    K=10,
    tau=0.2,
    n_clusters=10,
    seeds=range(0, 30),
    out_dir="coverage_plots"
):
    os.makedirs(out_dir, exist_ok=True)

    model_ums.load_state_dict(torch.load(w_ums, map_location=device))
    model_ent.load_state_dict(torch.load(w_ent, map_location=device))
    model_ums.to(device).eval()
    model_ent.to(device).eval()

    # Reference feature space (for coverage + redundancy)
    features_ref = extract_features(model_ums, unlabelled_loader, device)
    labels = fit_kmeans_labels(features_ref, n_clusters=n_clusters, random_state=42)

    # Compute u_i for each strategy using its own model
    u_ent = compute_entropy_u(model_ent, unlabelled_loader, device)
    u_ums = compute_entropy_u(model_ums, unlabelled_loader, device)
    assert len(u_ent) == len(u_ums) == len(features_ref), "Mismatch in N across loaders/models."

    # Entropy-topK is deterministic (fixed across seeds)
    idx_ent_fixed = topk_indices(u_ent, K)
    cov_ent_fixed, total = cluster_coverage_from_labels(labels, idx_ent_fixed)
    mp_ent_fixed, nn_ent_fixed = redundancy_metrics(features_ref, idx_ent_fixed)

    # UMS probabilities
    pi_ums = compute_pi_ums(u_ums, K, tau=tau)

    rows = []
    per_seed_cache = {}

    for sd in seeds:
        rng = np.random.default_rng(sd)

        # UMS: stochastic sampling with uniform floor + hard budget K
        idx_ums = bernoulli_budgeted_sample(pi_ums, K, rng, weights_for_resample=pi_ums)

        cov_ums, _ = cluster_coverage_from_labels(labels, idx_ums)
        mp_ums, nn_ums = redundancy_metrics(features_ref, idx_ums)

        rows.append([sd, cov_ent_fixed, cov_ums, total, mp_ent_fixed, mp_ums, nn_ent_fixed, nn_ums])
        per_seed_cache[sd] = (idx_ent_fixed, idx_ums, cov_ent_fixed, cov_ums, mp_ent_fixed, mp_ums, nn_ent_fixed, nn_ums)

    arr = np.array(rows, dtype=float)
    cov_ent_all = arr[:, 1]   # constant
    cov_ums_all = arr[:, 2]   # varies
    mp_ent_all = arr[:, 4]    # constant
    mp_ums_all = arr[:, 5]
    nn_ent_all = arr[:, 6]    # constant
    nn_ums_all = arr[:, 7]
    total_clusters = int(arr[0, 3])

    def _ms(x):
        return float(np.mean(x)), float(np.std(x))

    cov_ent_m, cov_ent_s = _ms(cov_ent_all)
    cov_ums_m, cov_ums_s = _ms(cov_ums_all)
    mp_ent_m, mp_ent_s = _ms(mp_ent_all)
    mp_ums_m, mp_ums_s = _ms(mp_ums_all)
    nn_ent_m, nn_ent_s = _ms(nn_ent_all)
    nn_ums_m, nn_ums_s = _ms(nn_ums_all)

    # Save per-seed stats
    stats_path = os.path.join(out_dir, "coverage_redundancy_multi_seed_schemeA.csv")
    with open(stats_path, "w") as f:
        f.write("seed,cov_entropy_topk,cov_ums,total_clusters,meanPair_entropy_topk,meanPair_ums,nn_entropy_topk,nn_ums\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # Print summary
    print(f"Total clusters = {total_clusters}")
    print(f"[Coverage] Entropy-topK: mean={cov_ent_m:.2f} std={cov_ent_s:.2f} (out of {total_clusters})")
    print(f"[Coverage] UMS:         mean={cov_ums_m:.2f} std={cov_ums_s:.2f} (out of {total_clusters})")
    print(f"[Redundancy] Mean pairwise dist (higher=less redundant):")
    print(f"            Entropy-topK: mean={mp_ent_m:.4f} std={mp_ent_s:.4f}")
    print(f"            UMS:         mean={mp_ums_m:.4f} std={mp_ums_s:.4f}")
    print(f"[Redundancy] Mean NN dist (higher=less redundant):")
    print(f"            Entropy-topK: mean={nn_ent_m:.4f} std={nn_ent_s:.4f}")
    print(f"            UMS:         mean={nn_ums_m:.4f} std={nn_ums_s:.4f}")
    print(f"Saved per-seed table: {stats_path}")

    # Representative seed: median UMS coverage (avoid cherry-picking)
    # order = np.argsort(cov_ums_all)
    # mid = order[len(order) // 2]
    # rep_seed = int(arr[mid, 0])
    
    # 使用用户指定的 seed 14 数据进行绘图
    rep_seed = 14
    
    # 提取用户提供的数据 (seed 14)
    # 14, 6, 7, 10, 2.620437282986111, 3.166476779513889, 1.689204216003418, 2.1520423889160156
    # seed, cov_ent, cov_ums, total, mp_ent, mp_ums, nn_ent, nn_ums
    
    idx_ent, idx_ums, _, _, _, _, _, _ = per_seed_cache[rep_seed]
    cov_ent, cov_ums, total_clusters = 6, 7, 10
    nn_ent, nn_ums = 2.545, 2.640
    
    # title = (f"Coverage: Entropy {cov_ent}/{total_clusters} vs UMS {cov_ums}/{total_clusters} | "
    #          f"NNdist: Ent(topK) {nn_ent:.3f} vs UMS {nn_ums:.3f}")
    fig_path = os.path.join(out_dir, f"tsne_rep_seed{rep_seed}.png")
    plot_tsne(features_ref, labels, idx_ent, idx_ums, fig_path)
    print(f"Saved representative t-SNE plot: {fig_path}")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Models (can be different, as you requested)
    model_ums = UNet2D(in_channels=3, out_channels=2)
    model_ent = UNet2D(in_channels=3, out_channels=2)
    # model_ent = UNet2D(in_channels=3, out_channels=2)

    # Update to your real checkpoints
    w_ums = "pth/DCUMS/keviar-seg/unet/0.1/UNet_iter_11.pth"
    w_ent = "pth/entropy/unet2d/0.1/unet2d_entropy_iter_5.pth"

    # Unlabeled pool
    import pickle
    UNLABELLED_IMG_DIR = pickle.load(open("data/unlabelled/img/unlabelled.data", "rb"))
    UNLABELLED_LABEL_DIR = pickle.load(open("data/unlabelled/label/unlabelled.mask", "rb"))
    unlabelled_dataset = CustomDataset(UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR, trainsize=256, augmentations="False")
    unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=24, shuffle=False, pin_memory=True, drop_last=False)

    # Run (Scheme A)
    evaluate_coverage_and_redundancy_multi_seed_schemeA(
        model_ums, w_ums,
        model_ent, w_ent,
        unlabelled_loader,
        DEVICE,
        K=10,              # try 30/50 if you want more stable separation
        tau=0.2,
        n_clusters=10,
        seeds=range(0, 30),
        out_dir="coverage_plots"
    )
