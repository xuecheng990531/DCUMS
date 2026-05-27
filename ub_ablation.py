import os
import time
import logging
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import matplotlib
matplotlib.use('Agg')
from models.transreunet_class import *
from utils.dataset import CustomDataset
from utils.loss import Softmax_CE_DiceLoss_reduction
from utils.metrics import *
import math

try:
    from torch.optim.adamw import AdamW
except ImportError:
    from torch.optim.adam import Adam as AdamW

def dcus_entropy_scores_from_loader(data_loader, model, device, al_round, warmup_rounds=2, ub=0.2, alpha=0.3, eps=1e-7):
    model.eval()
    scores_all = []
    overall_uncertainty = 0.0

    with torch.no_grad():
        quality = model.class_quality.to(device).clamp(0.0, 1.0)

        if al_round < warmup_rounds:
            class_weights = torch.ones_like(quality)
        else:
            difficulty = 1.0 - quality
            gamma = math.exp(1.0 / alpha) - 1.0
            class_weights = 1.0 + alpha * ub * torch.log(1.0 + gamma * difficulty)

        for batch_idx, (data, _) in enumerate(data_loader):
            try:
                data = data.to(device)
                logits = model(data)
                probs = torch.softmax(logits, dim=1)
                probs = probs.clamp(eps, 1.0 - eps)
                entropy_map = -torch.sum(probs * probs.log(), dim=1)
                weight_map = (probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
                weighted_entropy = entropy_map * weight_map
                batch_scores = weighted_entropy.view(weighted_entropy.size(0), -1).mean(dim=1)
                scores_all.append(batch_scores.cpu().numpy())
                overall_uncertainty += weighted_entropy.sum().item()
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

    if len(scores_all) == 0:
        return np.empty((0,), dtype=np.float32), 0.0

    uncertainty_scores = np.concatenate(scores_all, axis=0)
    return uncertainty_scores, overall_uncertainty


def get_class_weights(class_quality, ub, alpha, eps=1e-6):
    reverse_q = 1 - class_quality
    b = math.exp(1.0 / alpha) - 1
    return 1 + alpha * torch.log(b * reverse_q + 1 + eps) * ub


def train_step_ub(data, labels, model, loss_fn, optimizer, device, al_round, ub, warmup_rounds=2, scaler=None):
    model.train()
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
        preds, _ = model(data, return_quality=True)

        if al_round >= warmup_rounds:
            model.update_boundary_quality(preds.detach(), labels, radius=4)

        use_dcus = (al_round is not None) and (al_round >= warmup_rounds)
        with torch.no_grad():
            if not use_dcus:
                difficulty_map = torch.ones(preds.size(0), preds.size(2), preds.size(3), device=preds.device)
            else:
                quality = model.class_quality.clamp(0.0, 1.0)
                class_weights = get_class_weights(class_quality=quality, ub=ub, alpha=0.3)
                probs = torch.softmax(preds, dim=1)
                difficulty_map = (probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)

        raw_loss = loss_fn(preds, labels, reduction='none')
        if not use_dcus:
            loss = raw_loss.mean()
        else:
            loss = (raw_loss * difficulty_map).mean()

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return (
        loss.item(),
        float(iou_score_softmax(preds, labels)),
        float(dice_coef_softmax(preds, labels)),
        float(precision_softmax(preds, labels)),
        float(recall_softmax(preds, labels)),
        model.class_quality.detach().cpu()
    )


def validate(val_loader, model, loss_fn, device):
    from collections import OrderedDict
    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
        def update(self, val, n=1):
            val = float(val); self.val = val; self.sum += val * n; self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0.0

    losses, ious, dices, precs, recs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data, return_quality=True)[0]
            val_loss = loss_fn(preds, labels)
            batch_size = data.size(0)
            losses.update(val_loss.item(), batch_size)
            ious.update(float(iou_score_softmax(preds, labels)), batch_size)
            dices.update(float(dice_coef_softmax(preds, labels)), batch_size)
            precs.update(float(precision_softmax(preds, labels)), batch_size)
            recs.update(float(recall_softmax(preds, labels)), batch_size)

    return OrderedDict([
        ('loss', losses.avg), ('iou', ious.avg), ('dice', dices.avg),
        ('precision', precs.avg), ('recall', recs.avg),
    ])


def add_annotated_sample(to_be_annotated_index, labelled_img_paths, labelled_mask_paths,
                         unlabelled_img_paths, unlabelled_mask_paths):
    samples_to_be_annotated = [unlabelled_img_paths[i] for i in to_be_annotated_index]
    annotation = [unlabelled_mask_paths[i] for i in to_be_annotated_index]
    new_labelled_img_paths = labelled_img_paths + samples_to_be_annotated
    new_labelled_mask_paths = labelled_mask_paths + annotation
    new_unlabelled_img_paths = list(np.delete(unlabelled_img_paths, to_be_annotated_index))
    new_unlabelled_mask_paths = list(np.delete(unlabelled_mask_paths, to_be_annotated_index))
    return new_labelled_img_paths, new_labelled_mask_paths, new_unlabelled_img_paths, new_unlabelled_mask_paths


def main():
    TRAIN_IMG_DIR = pickle.load(open('data/train_val/img/' + 'train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data/train_val/label/' + 'train_val.mask', 'rb'))
    UNLABELLED_IMG_DIR = pickle.load(open('data/unlabelled/img/' + 'unlabelled.data', 'rb'))
    UNLABELLED_LABEL_DIR = pickle.load(open('data/unlabelled/label/' + 'unlabelled.mask', 'rb'))
    TEST_IMG_DIR = pickle.load(open('data/test/img/' + 'test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data/test/label/' + 'test.mask', 'rb'))

    # 重写路径：数据已移至 /icislab/volume1/lxc/polyp_data/kvasir/
    for lst in [TRAIN_IMG_DIR, TRAIN_LABEL_DIR, UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR, TEST_IMG_DIR, TEST_LABEL_DIR]:
        for i in range(len(lst)):
            lst[i] = lst[i].replace('data/Kvasir-SEG/', '/icislab/volume1/lxc/polyp_data/kvasir/')

    # =================== 超参数 ===================
    sampling_mode = 'dcums'
    nb_experiments = 1
    nb_active_learning_iter = 12
    AL_BUDGET_RATIO = 0.15
    MIN_ADD = 1
    max_iters = 800
    VAL_INTERVAL = 200
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 12

    # ub 消融参数
    UB = 0.4


    model = AttentionUNet(output_ch=2).to(DEVICE)
    model_name = model.__class__.__name__.lower()
    dataname = 'fives' if any('fives_data' in path for path in TRAIN_IMG_DIR) else 'Keviar-seg'
    model_name = 'DCUMS' if model_name == 'attentionunet' else 'UNet'
    print(f"Detected dataname: {dataname}")

    swanlab.init(
        project="ub_ablation",
        name=f"{model_name}_ub{UB}_{dataname}",
        workspace="xuecheng",
        config={
            "iters": max_iters,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "al_budget_ratio": AL_BUDGET_RATIO,
            "sampling_mode": sampling_mode,
            "ub": UB,
        }
    )

    loss_fn = Softmax_CE_DiceLoss_reduction()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    os.makedirs(f'experiment_logs/ub_ablation/ub{UB}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/ub_ablation/ub{UB}/output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    overall_start = time.time()
    test_metrics_history = {'dice': [], 'iou': [], 'precision': [], 'recall': []}

    for r in range(nb_experiments):
        labelled_img_paths, labelled_mask_paths = TRAIN_IMG_DIR, TRAIN_LABEL_DIR
        unlabelled_img_paths, unlabelled_mask_paths = UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR
        test_img_paths, test_mask_paths = TEST_IMG_DIR, TEST_LABEL_DIR

        labelled_img_paths, val_img_paths, labelled_mask_paths, val_mask_paths = train_test_split(
            labelled_img_paths, labelled_mask_paths, test_size=0.2, random_state=41
        )

        for i in range(nb_active_learning_iter):
            N_unlabeled = len(unlabelled_img_paths)
            K = int(round(AL_BUDGET_RATIO * N_unlabeled))
            K = max(MIN_ADD, K)
            K = min(K, N_unlabeled)

            logging.info("AL Round %d | Unlabeled pool size: %d | Budget: %d | Mode: %s | ub: %.2f",
                         i + 1, N_unlabeled, K, sampling_mode, UB)

            model_path = f'ablations/ub_ablation/ub{UB}/{dataname}/{model_name}/iter_{i}.pth'

            if i == 0:
                print(f"Loading base model for {model_name}...")
                state_dict = torch.load(
                    "pth/DCUMS/keviar-seg/attentionunet/0.1/DCUMS_iter_0.pth",
                    weights_only=True,
                    map_location=DEVICE
                )
                model.load_state_dict(state_dict)
                print("Base model loaded.")

            unlabelled_dataset = CustomDataset(
                unlabelled_img_paths, unlabelled_mask_paths, trainsize=256, augmentations='False'
            )
            unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

            uncertainty_scores, overall_uncertainty_score = dcus_entropy_scores_from_loader(
                unlabelled_loader, model, DEVICE, al_round=i, warmup_rounds=2, ub=UB
            )

            budget = K / N_unlabeled
            tau = 0.2
            mean_uncertainty = np.mean(uncertainty_scores)
            if mean_uncertainty == 0:
                mean_uncertainty = 1e-8
            eta = budget / mean_uncertainty
            probs = np.maximum((1 - tau) * eta * uncertainty_scores + tau * budget, 0.0)
            sampling_probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(N_unlabeled) / N_unlabeled

            to_be_annotated_indices = np.random.choice(np.arange(N_unlabeled), size=K, replace=False, p=sampling_probs)
            logging.info("第 %d 次主动学习迭代的整体不确定性得分为 %f", i + 1, overall_uncertainty_score)

            labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths = add_annotated_sample(
                to_be_annotated_indices, labelled_img_paths, labelled_mask_paths,
                unlabelled_img_paths, unlabelled_mask_paths
            )

            train_dataset = CustomDataset(labelled_img_paths, labelled_mask_paths, trainsize=256, augmentations='True')
            val_dataset = CustomDataset(val_img_paths, val_mask_paths, trainsize=256, augmentations='False')
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

            train_loader_iter = iter(train_loader)
            best_iou = -1.0
            start = time.time()
            pbar = tqdm(total=max_iters, desc=f"AL Round {i+1} Training (ub={UB})", unit="iter")

            for it in range(1, max_iters + 1):
                try:
                    data, labels = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    data, labels = next(train_loader_iter)

                loss, iou, dice, prec, rec, class_quality = train_step_ub(
                    data, labels, model, loss_fn, optimizer, DEVICE, al_round=i, ub=UB
                )

                logging.info("Iter %d/%d, Train Loss: %.4f, IoU: %.4f, Dice: %.4f", it, max_iters, loss, iou, dice)

                do_validate = (it % VAL_INTERVAL == 0) or (it == max_iters)
                if do_validate:
                    val_log = validate(val_loader, model, loss_fn, DEVICE)
                    logging.info("Validation Loss: %.4f, IoU: %.4f, Dice: %.4f", val_log['loss'], val_log['iou'], val_log['dice'])

                    if val_log['iou'] > best_iou:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        torch.save(model.state_dict(), model_path)
                        best_iou = val_log['iou']
                        logging.info("=> 模型参数更新 (best IoU: %.4f)，保存 (iter %d)", best_iou, it)

                pbar.set_postfix({'Loss': f'{loss:.4f}', 'IoU': f'{iou:.4f}', 'best_iou': f'{best_iou:.4f}'})
                pbar.update(1)

                if it % 100 == 0 and DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            pbar.close()
            end = time.time()
            logging.info('Training for AL round %d took %.2f minutes', i, (end - start) / 60)

            # =================== 测试集评估 ===================
            logging.info("------------ 第 %d 次主动学习完成，开始测试 -----------\n", i + 1)

            try:
                test_dataset = CustomDataset(test_img_paths, test_mask_paths, trainsize=448, augmentations='False')
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

                logging.info("------------ 加载最佳模型，路径：%s -----------\n", model_path)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()

                with torch.no_grad():
                    all_dice, all_precision, all_recall, all_iou = [], [], [], []
                    for data, labels in tqdm(test_loader, total=len(test_loader)):
                        data, labels = data.to(DEVICE), labels.to(DEVICE)
                        preds = model(data)
                        all_dice.append(float(dice_coef_softmax(preds, labels)))
                        all_precision.append(float(precision_softmax(preds, labels)))
                        all_recall.append(float(recall_softmax(preds, labels)))
                        all_iou.append(float(iou_score_softmax(preds, labels)))
                        del data, labels, preds
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()

                mean_dice = np.array(all_dice).mean()
                mean_precision = np.array(all_precision).mean()
                mean_recall = np.array(all_recall).mean()
                mean_iou = np.array(all_iou).mean()

                test_metrics_history['dice'].append(mean_dice)
                test_metrics_history['iou'].append(mean_iou)
                test_metrics_history['precision'].append(mean_precision)
                test_metrics_history['recall'].append(mean_recall)

                logging.info("------------ 主动学习轮次 %d 测试结果（累计训练样本: %d）-----------", i + 1, len(labelled_img_paths))
                logging.info('Test Dice: %.4f | IoU: %.4f | Precision: %.4f | Recall: %.4f',
                             mean_dice, mean_iou, mean_precision, mean_recall)

                swanlab.log({
                    "test/dice": mean_dice, "test/iou": mean_iou,
                    "test/precision": mean_precision, "test/recall": mean_recall,
                    "train_set_size": len(labelled_img_paths)
                })
            except Exception as e:
                logging.error(f"测试阶段出错: {str(e)}")
                if test_metrics_history['dice']:
                    mean_dice = test_metrics_history['dice'][-1]
                    mean_iou = test_metrics_history['iou'][-1]
                    mean_precision = test_metrics_history['precision'][-1]
                    mean_recall = test_metrics_history['recall'][-1]
                else:
                    mean_dice = mean_iou = mean_precision = mean_recall = 0.0
                continue

    logging.info("\n===== 主动学习全过程测试性能趋势 =====")
    for idx, (dice, iou, prec, rec) in enumerate(zip(
        test_metrics_history['dice'], test_metrics_history['iou'],
        test_metrics_history['precision'], test_metrics_history['recall']
    ), 1):
        logging.info("轮次 %d: Dice=%.4f, IoU=%.4f, Prec=%.4f, Rec=%.4f", idx, dice, iou, prec, rec)

    overall_end = time.time()
    logging.info('Total time: %.2f minutes', (overall_end - overall_start) / 60)


if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
