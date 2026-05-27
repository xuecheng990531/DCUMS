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
from model_train_val_class import *
from utils.loss import *
from utils.metrics import *
from utils.utils import *
from models.unet import UNet2D
from utils.dataset import CustomDataset
import math

# 修复torch.optim.AdamW导入问题
try:
    from torch.optim.adamw import AdamW
except ImportError:
    from torch.optim.adam import Adam as AdamW

def dcus_entropy_scores_from_loader(data_loader, model, device, al_round, warmup_rounds=2, ub=0.2, alpha=0.3, eps=1e-7):
    """
    使用熵采样 + 边界质量代理计算不确定性得分
    """
    model.eval()
    scores_all = []
    overall_uncertainty = 0.0

    with torch.no_grad():
        # 获取类别质量代理
        quality = model.class_quality.to(device).clamp(0.0, 1.0)

        if al_round < warmup_rounds:
            # 冷启动阶段：不使用质量代理权重
            class_weights = torch.ones_like(quality)
        else:
            print("边界质量代理已开启")
            logging.info("边界质量代理已开启")
            # 质量越低，难度越高
            difficulty = 1.0 - quality
            gamma = math.exp(1.0 / alpha) - 1.0
            # 计算类别权重：1 + alpha * ub * log(1 + gamma * difficulty)
            class_weights = 1.0 + alpha * ub * torch.log(1.0 + gamma * difficulty + eps)

        for batch_idx, (data, _) in enumerate(data_loader):
            try:
                data = data.to(device)                 
                logits = model(data)                  

                probs = torch.softmax(logits, dim=1)   
                probs = probs.clamp(eps, 1.0 - eps)
                
                # 1. 计算熵图 (Entropy Map)
                entropy_map = -torch.sum(probs * torch.log(probs), dim=1) # [B, H, W]
                
                # 2. 计算权重图 (Weight Map)
                # 使用预测概率对类别权重进行加权，得到每个像素的权重
                weight_map = (probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)  # [B, H, W]
                
                # 3. 加权熵
                weighted_entropy = entropy_map * weight_map  # [B, H, W]

                # 对整张图像取平均作为样本得分
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

def add_annotated_sample(to_be_annotated_index,
                         labelled_img_paths, labelled_mask_paths,
                         unlabelled_img_paths, unlabelled_mask_paths):
    samples_to_be_annotated = [unlabelled_img_paths[i] for i in to_be_annotated_index]
    annotation = [unlabelled_mask_paths[i] for i in to_be_annotated_index]

    new_labelled_img_paths = labelled_img_paths + samples_to_be_annotated
    new_labelled_mask_paths = labelled_mask_paths + annotation

    new_unlabelled_img_paths = list(np.delete(unlabelled_img_paths, to_be_annotated_index))
    new_unlabelled_mask_paths = list(np.delete(unlabelled_mask_paths, to_be_annotated_index))

    return new_labelled_img_paths, new_labelled_mask_paths, new_unlabelled_img_paths, new_unlabelled_mask_paths

def main():
    # 载入数据路径
    TRAIN_IMG_DIR = pickle.load(open('fives_data/train_val/img/train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('fives_data/train_val/label/train_val.mask', 'rb'))
    UNLABELLED_IMG_DIR = pickle.load(open('fives_data/unlabelled/img/unlabelled.data', 'rb'))
    UNLABELLED_LABEL_DIR = pickle.load(open('fives_data/unlabelled/label/unlabelled.mask', 'rb'))
    TEST_IMG_DIR = pickle.load(open('fives_data/test/img/test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('fives_data/test/label/test.mask', 'rb'))

    # =================== 超参数 ===================
    nb_experiments = 1
    nb_active_learning_iter = 12
    AL_BUDGET_RATIO = 0.1
    MIN_ADD = 1
    max_iters = 2000
    VAL_INTERVAL = 100
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 24
    WARMUP_ROUNDS = 2

    # 初始化模型
    model = UNet2D(in_channels=3, out_channels=2).to(DEVICE)
    model_name = "UNet_Entropy_Boundary"
    dataname = 'fives'
    
    print(f"Model: {model_name}, Dataset: {dataname}")

    # =================== Swanlab 配置 ===================
    swanlab.init(
        project="active_loop",
        workspace="xuecheng",
        name=f"unet_entropy_BQP",
        config={
            "iters": max_iters,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "al_budget_ratio": AL_BUDGET_RATIO,
            "strategy": "Entropy + Boundary Quality Proxy"
        }
    )
    
    loss_fn = Softmax_CE_DiceLoss_reduction()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # =================== 日志配置 ===================
    os.makedirs(f'experiment_logs/{dataname}/{model_name}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/{dataname}/{model_name}/active_learning_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

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
            K = max(MIN_ADD, int(round(AL_BUDGET_RATIO * N_unlabeled)))
            K = min(K, N_unlabeled)

            logging.info("AL Round %d | Unlabeled pool size: %d | Budget: %d", i + 1, N_unlabeled, K)

            model_path = f'ablations/{model_name}/{AL_BUDGET_RATIO}/{model_name}_iter_{i}.pth'

            if i == 0:
                print(f"Loading base model from pth/DCUMS/keviar-seg/attentionunet/0.1/UNet.pth ...")
                state_dict = torch.load("pth/DCUMS/keviar-seg/attentionunet/0.1/UNet.pth", map_location=DEVICE, weights_only=True)
                model.load_state_dict(state_dict)
                print("Base model loaded.")

            logging.info("\n--------- 第 %d 次主动学习训练开始 ----------", i + 1)

            unlabelled_dataset = CustomDataset(unlabelled_img_paths, unlabelled_mask_paths, trainsize=256, augmentations='False')
            unlabelled_loader = DataLoader(unlabelled_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)

            logging.info("计算未标注集的不确定性 (Entropy + Boundary Quality) ...")
            uncertainty_scores, overall_uncertainty_score = dcus_entropy_scores_from_loader(
                unlabelled_loader, model, DEVICE, al_round=i, warmup_rounds=WARMUP_ROUNDS
            )

            # 均匀混合采样逻辑
            budget = K / N_unlabeled 
            tau = 0.2
            mean_uncertainty = np.mean(uncertainty_scores) if len(uncertainty_scores) > 0 else 1e-8
            eta = budget / (mean_uncertainty + 1e-8)
            probs = np.maximum((1 - tau) * eta * uncertainty_scores + tau * budget, 0.0)
            sampling_probs = probs / (np.sum(probs) + 1e-10) if np.sum(probs) > 0 else np.ones(N_unlabeled) / N_unlabeled

            to_be_annotated_indices = np.random.choice(np.arange(N_unlabeled), size=K, replace=False, p=sampling_probs)

            labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths = add_annotated_sample(
                to_be_annotated_indices, labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths
            )

            logging.info("第 %d 次主动学习迭代的整体不确定性得分为 %f", i+1, overall_uncertainty_score)
            logging.info("当前训练集包含 %d 个样本\n", len(labelled_img_paths))

            train_dataset = CustomDataset(labelled_img_paths, labelled_mask_paths, trainsize=256, augmentations='True')
            val_dataset = CustomDataset(val_img_paths, val_mask_paths, trainsize=256, augmentations='False')
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)

            train_loader_iter = iter(train_loader)
            best_iou = -1.0
            start = time.time()
            pbar = tqdm(total=max_iters, desc=f"AL Round {i+1} Training", unit="iter")

            for it in range(1, max_iters + 1):
                try:
                    loss, iou, dice, train_loader_iter, _ = train(
                        train_loader, train_loader_iter, model, loss_fn, optimizer, DEVICE, al_round=i
                    )
                    if it % VAL_INTERVAL == 0 or it == max_iters:
                        val_log = validate(val_loader, model, loss_fn, DEVICE)
                        if val_log['iou'] > best_iou:
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            torch.save(model.state_dict(), model_path)
                            best_iou = val_log['iou']
                    pbar.set_postfix({'Loss': f'{loss:.4f}', 'IoU': f'{iou:.4f}', 'best_iou': f'{best_iou:.4f}'})
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"训练迭代 {it} 出错: {str(e)}")
                    continue
                if it % 100 == 0 and DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            pbar.close()
            logging.info('AL Round %d training took %.2f minutes', i + 1, (time.time() - start) / 60)

            # 测试集评估
            try:
                test_dataset = CustomDataset(test_img_paths, test_mask_paths, trainsize=448, augmentations='False')
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    all_dice, all_iou, all_prec, all_rec = [], [], [], []
                    for data, labels in test_loader:
                        data, labels = data.to(DEVICE), labels.to(DEVICE)
                        preds = model(data)
                        all_dice.append(float(dice_coef_softmax(preds, labels)))
                        all_iou.append(float(iou_score_softmax(preds, labels)))
                        all_prec.append(float(precision_softmax(preds, labels)))
                        all_rec.append(float(recall_softmax(preds, labels)))
                
                m_dice, m_iou, m_prec, m_rec = np.mean(all_dice), np.mean(all_iou), np.mean(all_prec), np.mean(all_rec)
                test_metrics_history['dice'].append(m_dice)
                test_metrics_history['iou'].append(m_iou)
                test_metrics_history['precision'].append(m_prec)
                test_metrics_history['recall'].append(m_rec)

                logging.info('Test Results | Dice: %.4f | IoU: %.4f', m_dice, m_iou)
                swanlab.log({"test/dice": m_dice, "test/iou": m_iou, "train_set_size": len(labelled_img_paths)})
            except Exception as e:
                logging.error(f"测试阶段出错: {str(e)}")

    logging.info("\n===== 主动学习全过程测试性能趋势 =====")
    for idx, (dice, iou) in enumerate(zip(test_metrics_history['dice'], test_metrics_history['iou']), 1):
        logging.info("轮次 %d: Dice=%.4f, IoU=%.4f", idx, dice, iou)

if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
