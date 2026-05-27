# 基于熵值（Entropy）的主动学习采样主程序
# - 以 mc_dropout_sampling.py 为模板
# - 采样策略：Entropy Sampling (Predictive Entropy)

import os
import time
import logging
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import swanlab

import matplotlib

matplotlib.use('Agg')

from model_train_val_class import *
from utils.loss import *
from utils.metrics import *
from utils.utils import *
from models.unet import UNet2D
from utils.dataset import CustomDataset

# Optimizer fallback
try:
    from torch.optim.adamw import AdamW
except Exception:
    try:
        from torch.optim import AdamW  # type: ignore
    except Exception:
        from torch.optim.adam import Adam as AdamW


def add_annotated_sample(
    to_be_annotated_index,
    labelled_img_paths,
    labelled_mask_paths,
    unlabelled_img_paths,
    unlabelled_mask_paths,
):
    samples = [unlabelled_img_paths[i] for i in to_be_annotated_index]
    masks = [unlabelled_mask_paths[i] for i in to_be_annotated_index]

    new_labelled_img_paths = labelled_img_paths + samples
    new_labelled_mask_paths = labelled_mask_paths + masks

    new_unlabelled_img_paths = list(np.delete(unlabelled_img_paths, to_be_annotated_index))
    new_unlabelled_mask_paths = list(np.delete(unlabelled_mask_paths, to_be_annotated_index))

    return (
        new_labelled_img_paths,
        new_labelled_mask_paths,
        new_unlabelled_img_paths,
        new_unlabelled_mask_paths,
    )


@torch.no_grad()
def entropy_uncertainty_scores_from_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    eps: float = 1e-7,
):
    """对未标注池计算基于熵值（Entropy）的不确定性分数。

    Args:
        data_loader: 未标注集 loader
        model: 分割模型，输出 logits: [B,C,H,W]
        device: 'cuda' or 'cpu'
        eps: 防止 log(0)

    Returns:
        scores: np.ndarray shape [N]
        overall_uncertainty: float
    """

    model.eval()
    scores_all = []
    overall_uncertainty = 0.0

    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)

        logits = model(data)
        probs = torch.softmax(logits, dim=1).clamp(eps, 1.0 - eps)

        # H[p] = -sum(p * log(p))
        entropy_map = -(probs * probs.log()).sum(dim=1)  # [B,H,W]

        batch_scores = entropy_map.view(entropy_map.size(0), -1).mean(dim=1)  # [B]
        scores_all.append(batch_scores.detach().cpu().numpy())
        overall_uncertainty += float(entropy_map.sum().item())

        # 清理显存
        del data, logits, probs, entropy_map, batch_scores
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(scores_all) == 0:
        return np.empty((0,), dtype=np.float32), 0.0

    scores = np.concatenate(scores_all, axis=0)
    return scores.astype(np.float32), overall_uncertainty


def main():
    # 载入预先划分好的数据路径
    TRAIN_IMG_DIR = pickle.load(open('data/train_val/img/' + 'train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data/train_val/label/' + 'train_val.mask', 'rb'))
    UNLABELLED_IMG_DIR = pickle.load(open('data/unlabelled/img/' + 'unlabelled.data', 'rb'))
    UNLABELLED_LABEL_DIR = pickle.load(open('data/unlabelled/label/' + 'unlabelled.mask', 'rb'))
    TEST_IMG_DIR = pickle.load(open('data/test/img/' + 'test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data/test/label/' + 'test.mask', 'rb'))

    # =================== Hyper-params ===================
    nb_experiments = 1
    nb_active_learning_iter = 12
    AL_BUDGET_RATIO = 0.1
    MIN_ADD = 1

    max_iters = 1000
    VAL_INTERVAL = 200

    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    TRAIN_BATCH_SIZE = 12
    UNLAB_BATCH_SIZE = 12  # 熵值采样不需要多次前向，可以适当调大 batch_size

    # =================== SwanLab ===================
    swanlab.init(
        project="entropy_active_loop",
        workspace="xuecheng",
        name=f"0120attentionunet—entropy%",
        config={
            "iters": max_iters,
            "lr": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "unlab_batch_size": UNLAB_BATCH_SIZE,
            "budget_ratio": AL_BUDGET_RATIO,
        },
    )

    # =================== Logging ===================
    learning_type = 'entropy'
    model_name = 'attention'
    dataname = 'seg'
    os.makedirs(f'experiment_logs/{dataname}/{model_name}', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/{dataname}/{model_name}/{learning_type}_learning_output.txt',
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # =================== Model ===================
    # model = UNet2D(in_channels=3, out_channels=2).to(DEVICE)
    model = AttentionUNet(output_ch=2).to(DEVICE)
    model_name = model.__class__.__name__.lower()
    loss_fn = Softmax_CE_DiceLoss_reduction()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    overall_start = time.time()
    logging.info("=> GO GO GO 出发咯 (Entropy Sampling)")

    # 初始化全局测试指标记录器
    test_metrics_history = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
    }

    # =================== 多次实验循环 ===================
    for r in range(nb_experiments):
        labelled_img_paths, labelled_mask_paths = TRAIN_IMG_DIR, TRAIN_LABEL_DIR
        unlabelled_img_paths, unlabelled_mask_paths = UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR

        # 为每次独立实验分配训练验证集
        labelled_img_paths, val_img_paths, labelled_mask_paths, val_mask_paths = train_test_split(
            labelled_img_paths,
            labelled_mask_paths,
            test_size=0.2,
            random_state=41,
        )

        # ========== 主动学习迭代 ==========
        for i in range(nb_active_learning_iter):
            N_unlabeled = len(unlabelled_img_paths)
            K = max(MIN_ADD, int(AL_BUDGET_RATIO * N_unlabeled))
            K = min(K, N_unlabeled)

            logging.info(
                "AL Round %d | Unlabeled: %d | Entropy Budget: %d",
                i + 1,
                N_unlabeled,
                K,
            )

            # ===== Entropy Sampling =====
            unlabelled_dataset = CustomDataset(
                unlabelled_img_paths,
                unlabelled_mask_paths,
                trainsize=256,
                augmentations='False',
            )
            unlabelled_loader = DataLoader(
                unlabelled_dataset,
                batch_size=UNLAB_BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )

            logging.info("计算未标注池熵值不确定性 ...")
            uncertainty_scores, overall_uncertainty = entropy_uncertainty_scores_from_loader(
                unlabelled_loader,
                model,
                DEVICE,
            )

            if uncertainty_scores.shape[0] != N_unlabeled:
                raise RuntimeError(
                    f"Uncertainty scores length mismatch: got {uncertainty_scores.shape[0]}, expected {N_unlabeled}"
                )

            # Top-K 选取（分数越大越不确定）
            selected_idx = np.argsort(-uncertainty_scores)[:K]
            logging.info(
                "Top-%d uncertainty scores: min=%.6f, mean=%.6f, max=%.6f",
                K,
                float(uncertainty_scores[selected_idx].min()) if K > 0 else 0.0,
                float(uncertainty_scores[selected_idx].mean()) if K > 0 else 0.0,
                float(uncertainty_scores[selected_idx].max()) if K > 0 else 0.0,
            )

            swanlab.log(
                {
                    "al/unlabeled_pool": N_unlabeled,
                    "al/budget": K,
                    "al/overall_uncertainty": float(overall_uncertainty),
                    "al/uncertainty_mean": float(uncertainty_scores.mean()) if N_unlabeled > 0 else 0.0,
                    "al/uncertainty_max": float(uncertainty_scores.max()) if N_unlabeled > 0 else 0.0,
                }
            )

            labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths = (
                add_annotated_sample(
                    selected_idx,
                    labelled_img_paths,
                    labelled_mask_paths,
                    unlabelled_img_paths,
                    unlabelled_mask_paths,
                )
            )

            # ===== Dataset =====
            train_dataset = CustomDataset(
                labelled_img_paths,
                labelled_mask_paths,
                trainsize=256,
                augmentations='True',
            )
            val_dataset = CustomDataset(
                val_img_paths,
                val_mask_paths,
                trainsize=256,
                augmentations='False',
            )

            train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

            # ===== Train =====
            train_loader_iter = iter(train_loader)
            best_dice = -1.0
            start = time.time()

            for it in range(1, max_iters + 1):
                try:
                    loss, iou, dice, train_loader_iter, class_quality = train(
                        train_loader,
                        train_loader_iter,
                        model,
                        loss_fn,
                        optimizer,
                        DEVICE,
                        al_round=i,
                    )

                    logging.info(
                        "Iter %d/%d, Train Loss: %.4f, IoU: %.4f, Dice: %.4f",
                        it,
                        max_iters,
                        loss,
                        iou,
                        dice,
                    )

                    do_validate = (it % VAL_INTERVAL == 0) or (it == max_iters)
                    if do_validate:
                        logging.info("验证集上评估模型 (after iter %d)", it)
                        val_log = validate(val_loader, model, loss_fn, DEVICE)
                        logging.info(
                            "Validation Loss: %.4f, IoU: %.4f, Dice: %.4f",
                            val_log['loss'],
                            val_log['iou'],
                            val_log['dice'],
                        )

                        if val_log['dice'] > best_dice:
                            model_path = (
                                f'pth/entropy/{model_name}/{AL_BUDGET_RATIO}/'
                                f'{model_name}_entropy_iter_{i}.pth'
                            )
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            torch.save(model.state_dict(), model_path)
                            best_dice = val_log['dice']
                            logging.info("=> 模型参数更新，保存 (iter %d)", it)

                    if it % 100 == 0 and DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"训练迭代 {it} 出错: {str(e)}")
                    continue

            end = time.time()
            logging.info(
                'Training and (periodic) validation for active learning iteration %d has taken %.2f minutes',
                i,
                (end - start) / 60,
            )

            # ===== Test =====
            test_dataset = CustomDataset(
                TEST_IMG_DIR,
                TEST_LABEL_DIR,
                trainsize=256,
                augmentations='False',
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
            )

            logging.info("------------ 加载最佳模型，路径：%s -----------\n", model_path)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            with torch.no_grad():
                all_dice, all_precision, all_recall, all_iou = [], [], [], []
                for batch_idx, (data, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    data = data.to(DEVICE)
                    labels = labels.to(DEVICE)

                    preds = model(data)

                    batch_dice = dice_coef_softmax(preds, labels)
                    batch_precision = precision_softmax(preds, labels)
                    batch_recall = recall_softmax(preds, labels)
                    batch_iou = iou_score_softmax(preds, labels)

                    all_dice.append(float(batch_dice))
                    all_precision.append(float(batch_precision))
                    all_recall.append(float(batch_recall))
                    all_iou.append(float(batch_iou))

                    del data, labels, preds
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()

            all_dice = np.array(all_dice)
            all_precision = np.array(all_precision)
            all_recall = np.array(all_recall)
            all_iou = np.array(all_iou)

            mean_dice = all_dice.mean()
            mean_precision = all_precision.mean()
            mean_recall = all_recall.mean()
            mean_iou = all_iou.mean()

            test_metrics_history['dice'].append(mean_dice)
            test_metrics_history['iou'].append(mean_iou)
            test_metrics_history['precision'].append(mean_precision)
            test_metrics_history['recall'].append(mean_recall)

            logging.info(
                "------------ 主动学习轮次 %d 测试结果（累计训练样本: %d）-----------",
                i + 1,
                len(labelled_img_paths),
            )
            logging.info(
                'Test Dice: %.4f | IoU: %.4f | Precision: %.4f | Recall: %.4f',
                mean_dice,
                mean_iou,
                mean_precision,
                mean_recall,
            )

            swanlab.log(
                {
                    "test/dice": mean_dice,
                    "test/iou": mean_iou,
                    "test/precision": mean_precision,
                    "test/recall": mean_recall,
                    "train_set_size": len(labelled_img_paths),
                }
            )

    logging.info("\n===== 主动学习全过程测试性能趋势 =====")
    for idx, (dice, iou, prec, rec) in enumerate(
        zip(
            test_metrics_history['dice'],
            test_metrics_history['iou'],
            test_metrics_history['precision'],
            test_metrics_history['recall'],
        ),
        1,
    ):
        logging.info(
            "轮次 %d: Dice=%.4f, IoU=%.4f, Prec=%.4f, Rec=%.4f",
            idx,
            dice,
            iou,
            prec,
            rec,
        )

    overall_end = time.time()
    logging.info(
        'Learning for %d experiments has taken %.2f minutes',
        r + 1,
        (overall_end - overall_start) / 60,
    )


if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
