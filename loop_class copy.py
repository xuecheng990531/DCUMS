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
import models.unet as unet_mod
from models.transreunet_class import *
from utils.dataset import CustomDataset
import math

# 修复torch.optim.AdamW导入问题
try:
    # 部分环境的类型桩(Pylance)不从 torch.optim 直接导出 AdamW，这里改为显式子模块导入
    from torch.optim.adamw import AdamW
except ImportError:
    # 降级到Adam，如果AdamW不可用
    from torch.optim.adam import Adam as AdamW

def dcus_entropy_scores_from_loader(data_loader, model, device, al_round, warmup_rounds=2, ub=0.2, alpha=0.3, eps=1e-7):

    model.eval()
    scores_all = []
    overall_uncertainty = 0.0

    with torch.no_grad():

        quality = model.class_quality.to(device).clamp(0.0, 1.0)

        if al_round < warmup_rounds:
            # ===== 冷启动阶段：禁用 difficulty =====
            class_weights = torch.ones_like(quality)
        else:
            print("边界代理已开启")
            logging.info("边界代理已开启")
            difficulty = 1.0 - quality
            gamma = math.exp(1.0 / alpha) - 1.0
            class_weights = 1.0 + alpha * ub * torch.log(1.0 + gamma * difficulty)

        for batch_idx, (data, _) in enumerate(data_loader):
            try:
                data = data.to(device)                 
                logits  = model(data)                  

                probs = torch.softmax(logits, dim=1)   
                probs = probs.clamp(eps, 1.0 - eps)
                entropy_map = -torch.sum(probs * probs.log(), dim=1)
                ''' old code
                # pred_cls = probs.argmax(dim=1)
                # weight_map = class_weights[pred_cls]
                # weighted_entropy = entropy_map * weight_map           
                '''
                weight_map = (probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)  # [B,H,W]
                weighted_entropy = entropy_map * weight_map  # [B,H,W]

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
    """
    将选中的样本从未标注集移入标注集。
    """
    samples_to_be_annotated = [unlabelled_img_paths[i] for i in to_be_annotated_index]
    annotation = [unlabelled_mask_paths[i] for i in to_be_annotated_index]

    new_labelled_img_paths = labelled_img_paths + samples_to_be_annotated
    new_labelled_mask_paths = labelled_mask_paths + annotation

    new_unlabelled_img_paths = list(np.delete(unlabelled_img_paths, to_be_annotated_index))
    new_unlabelled_mask_paths = list(np.delete(unlabelled_mask_paths, to_be_annotated_index))

    return new_labelled_img_paths, new_labelled_mask_paths, new_unlabelled_img_paths, new_unlabelled_mask_paths


def main():
    """
    主动学习主循环：
      - random_bool: True 随机采样; False 熵不确定性采样
      - nb_experiments: 重复实验次数
      - nb_active_learning_iter: 主动学习轮数
      - nb_active_learning_iter_size: 每轮新增标注样本数
    """

    # 载入预先划分好的数据路径（data_split.ipynb 中生成）
    TRAIN_IMG_DIR = pickle.load(open('data/train_val/img/' + 'train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data/train_val/label/' + 'train_val.mask', 'rb'))
    UNLABELLED_IMG_DIR = pickle.load(open('data/unlabelled/img/' + 'unlabelled.data', 'rb'))
    UNLABELLED_LABEL_DIR = pickle.load(open('data/unlabelled/label/' + 'unlabelled.mask', 'rb'))
    TEST_IMG_DIR = pickle.load(open('data/test/img/' + 'test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data/test/label/' + 'test.mask', 'rb'))

    # =================== 超参数 ===================
    random_bool = False                  # False: 主动学习；True: 随机学习
    nb_experiments = 1                   # 实验次数
    nb_active_learning_iter = 12         # 主动学习轮数
    AL_BUDGET_RATIO = 0.1              # 每一轮从未标注池中选 5%
    MIN_ADD = 1                         # 每轮至少选 1 个样本
    max_iters = 1000                          # max_iters
    VAL_INTERVAL = 200                     # 每隔多少个迭代做一次验证

    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 24

    # model = AttentionUNet(output_ch=2).to(DEVICE)
    model=TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=4, mlp_dim=512, block_num=4, patch_dim=16, class_num=2).to(DEVICE)
    model_name = model.__class__.__name__.lower()
    dataname = 'fives' if any('fives_data' in path for path in TRAIN_IMG_DIR) else 'Keviar-seg'
    # model_name='DCUMS' if model_name=='attentionunet' else 'UNet'
    print(f"Detected dataname: {dataname}")
    logging.info("=> GO GO GO 出发咯")

    # =================== Swanlab 配置 ===================
    swanlab.init(
        project="active_loop",
        workspace="xuecheng",
        name=f"transunet",
        config={
            "iters": max_iters,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "al_budget_ratio": AL_BUDGET_RATIO,
        }
    )
    
    # =================== 模型、损失、优化器 ===================
   

    loss_fn = Softmax_CE_DiceLoss_reduction()
    # 使用我们定义的AdamW或Adam
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
        )
    
    # 初始化 AMP GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    # =================== 日志配置 ===================
    learning_type = 'random' if random_bool else 'active'
    os.makedirs(f'experiment_logs/{learning_type}_learning_results', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/{dataname}/{model_name}/{learning_type}_learning_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    

    overall_start = time.time()

    # 初始化全局测试指标记录器
    test_metrics_history = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }

    # =================== 多次实验循环 default 1===================
    r = -1
    for r in range(nb_experiments):
        labelled_img_paths, labelled_mask_paths = TRAIN_IMG_DIR, TRAIN_LABEL_DIR
        unlabelled_img_paths, unlabelled_mask_paths = UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR
        test_img_paths, test_mask_paths = TEST_IMG_DIR, TEST_LABEL_DIR

        # 为每次独立实验分配训练验证集
        labelled_img_paths, val_img_paths, labelled_mask_paths, val_mask_paths = train_test_split(
            labelled_img_paths, labelled_mask_paths,
            test_size=0.2,
            random_state=41
        )

        # ========== 主动学习迭代 ==========
        for i in range(nb_active_learning_iter):
            N_unlabeled = len(unlabelled_img_paths)

            K = int(round(AL_BUDGET_RATIO * N_unlabeled))
            K = max(MIN_ADD, K)
            K = min(K, N_unlabeled)

            logging.info(
                "AL Round %d | Unlabeled pool size: %d | Budget (10%%): %d",
                i + 1, N_unlabeled, K
            )

            if model_name == 'attentionunet':
                model_path = f'pth/DCUMS/cvcs/attentionunet/tau0/{model_name}_iter_{i}.pth'
                data_type = 'uncertain'
            elif model_name == 'transunet':
                model_path = f'pth/DCUMS/cvcs/transunet/tau0/{model_name}_iter_{i}.pth'
                data_type = 'uncertain'
            else:
                model_path = f'pth/DCUMS/cvcs/unet/tau0/{model_name}_iter_{i}.pth'
                data_type = 'uncertain'

            # 第 0 轮：从 base 模型开始
            if i == 0:
                print(f"Loading base model for {model_name}...")
                if model_name == 'attentionunet':
                    state_dict = torch.load(
                        "pth/DCUMS/cvcs/attentionunet/0.1/DCUMS_iter_0.pth",
                        weights_only=True
                    )
                elif model_name == 'transunet':
                    state_dict = torch.load(
                        "pth/DCUMS/cvcs/transunet/aaaaa.pth",
                        weights_only=True
                    )
                else:
                    state_dict = torch.load(
                        'pth/DCUMS/fives/unet2d.pth',
                        map_location='cuda',
                        weights_only=True
                    )

                model.load_state_dict(state_dict)
                print("Base model loaded.")
            else:
                # 加载上一轮的最佳模型
                prev_round_model_path = f'pth/DCUMS/cvcs/{model_name.lower()}/tau0/{model_name}_iter_{i-1}.pth'
                if os.path.exists(prev_round_model_path):
                    logging.info("Reloading best model from iteration %d", i)
                    model.load_state_dict(torch.load(prev_round_model_path, map_location=DEVICE))

            logging.info("\n--------- 第 %d 次主动学习训练开始 ----------", i + 1)

            # =================== 构建未标注集 Loader ===================
            unlabelled_dataset = CustomDataset(
                unlabelled_img_paths, unlabelled_mask_paths,
                trainsize=256, augmentations='False'
            )
            unlabelled_loader = DataLoader(
                unlabelled_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )

            # =================== 采样：随机 / 熵不确定性 ===================
            if random_bool:
                to_be_added_random = np.random.choice(
                    N_unlabeled,
                    K,
                    replace=False
                )
                labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths = add_annotated_sample(
                    to_be_added_random,
                    labelled_img_paths, labelled_mask_paths,
                    unlabelled_img_paths, unlabelled_mask_paths
                )
            else:
                logging.info("计算未标注集的不确定性 ...")
                uncertainty_scores, overall_uncertainty_score = dcus_entropy_scores_from_loader(
                    unlabelled_loader, model, DEVICE,al_round=i,warmup_rounds=2
                )

                # =================== 均匀混合采样 ==================

                budget = K / N_unlabeled 
                tau = 0.05  # 降低 tau 值以启用不确定性采样，0.05 表示 95% 权重给不确定性，5% 给随机

                mean_uncertainty = np.mean(uncertainty_scores)
                if mean_uncertainty == 0:
                    mean_uncertainty = 1e-8

                eta = budget / mean_uncertainty

                probs = np.maximum(
                    (1 - tau) * eta * uncertainty_scores + tau * budget,
                    0.0
                )

                if np.sum(probs) == 0:
                    sampling_probs = np.ones(N_unlabeled) / N_unlabeled
                else:
                    sampling_probs = probs / np.sum(probs)

                to_be_annotated_indices = np.random.choice(
                    np.arange(N_unlabeled),
                    size=K,
                    replace=False,
                    p=sampling_probs
                )

                labelled_img_paths, labelled_mask_paths, \
                unlabelled_img_paths, unlabelled_mask_paths = add_annotated_sample(
                    to_be_annotated_indices,
                    labelled_img_paths, labelled_mask_paths,
                    unlabelled_img_paths, unlabelled_mask_paths
                )

                os.makedirs('data/update_data_pool', exist_ok=True)
                np.save('data/update_data_pool/labelled_img_paths.npy', labelled_img_paths)
                np.save('data/update_data_pool/labelled_mask_paths.npy', labelled_mask_paths)
                np.save('data/update_data_pool/unlabelled_img_paths.npy', unlabelled_img_paths)
                np.save('data/update_data_pool/unlabelled_mask_paths.npy', unlabelled_mask_paths)
                logging.info("第 %d 次主动学习迭代的整体不确定性得分为 %f", i+1, overall_uncertainty_score)
                logging.info(
                    "Top-%d uncertainty scores: min=%.6f, mean=%.6f, max=%.6f",
                    K,
                    float(uncertainty_scores[to_be_annotated_indices].min()) if K > 0 else 0.0,
                    float(uncertainty_scores[to_be_annotated_indices].mean()) if K > 0 else 0.0,
                    float(uncertainty_scores[to_be_annotated_indices].max()) if K > 0 else 0.0,
                )

            logging.info("本次迭代新增了 %d 个 %s 样本到训练集", K, data_type)
            logging.info("当前训练集包含 %d 个样本\n", len(labelled_img_paths))

            # =================== 重构训练 / 验证集 ===================
            train_dataset = CustomDataset(
                labelled_img_paths, labelled_mask_paths,
                trainsize=256,
                augmentations='True'
            )
            val_dataset = CustomDataset(
                val_img_paths, val_mask_paths,
                trainsize=256,
                augmentations='False'
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=True,
                drop_last=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )

            # =================== 训练 + 周期验证（按迭代次数） ===================
            train_loader_iter = iter(train_loader)

            # 仅当验证集 IoU 创新高时才保存模型
            # 设为 -1.0 确保第一次验证一定会保存，避免后续测试阶段找不到模型文件
            best_iou = -1.0
            trigger = 0
            start = time.time()

            pbar = tqdm(total=max_iters, desc=f"AL Round {i+1} Training", unit="iter")

            for it in range(1, max_iters + 1):
                try:
                    # 一次“迭代”：从 train_loader_iter 取一个 batch 并更新参数
                    train_res = train(
                        train_loader, train_loader_iter,
                        model, loss_fn, optimizer, DEVICE,al_round=i,
                        scaler=scaler
                    )
                    loss, iou, dice, prec, rec, train_loader_iter, class_quality = train_res

                    logging.info("Iter %d/%d, Train Loss: %.4f, IoU: %.4f, Dice: %.4f",
                                 it, max_iters, loss, iou, dice)

         
                    # 按“迭代次数”决定是否验证：每 VAL_INTERVAL 次验证一次，最后一次必验证
                    do_validate = (it % VAL_INTERVAL == 0) or (it == max_iters)

                    if do_validate:
                        logging.info("验证集上评估模型 (after iter %d)", it)
                        val_log = validate(val_loader, model, loss_fn, DEVICE)
                        
                        logging.info("Validation Loss: %.4f, IoU: %.4f, Dice: %.4f",val_log['loss'], val_log['iou'], val_log['dice'])

                        # 仅在获得最好的 IoU 指标之后再保存模型
                        if val_log['iou'] > best_iou:
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            torch.save(model.state_dict(), model_path)
                            best_iou = val_log['iou']
                            logging.info("=> 模型参数更新 (best IoU: %.4f)，保存 (iter %d)", best_iou, it)
                            trigger = 0

                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f'{loss:.4f}',
                        'IoU': f'{iou:.4f}',
                        'best_iou': f'{best_iou:.4f}'
                    })
                    pbar.update(1)

                except Exception as e:
                    logging.error(f"训练迭代 {it} 出错: {str(e)}")
                    continue  # 继续下一次迭代

                # 定期清理GPU内存
                if it % 100 == 0 and DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            pbar.close()
            end = time.time()
            logging.info(
                'Training and (periodic) validation for active learning iteration %d has taken %.2f minutes',
                i, (end - start) / 60
            )

            # =================== 测试集评估 ===================
            logging.info("------------ 第 %d 次主动学习完成，开始测试 -----------\n", i + 1)

            try:
                test_dataset = CustomDataset(
                    test_img_paths, test_mask_paths,trainsize=448, augmentations='False'
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False
                )

                logging.info("------------ 加载最佳模型，路径：%s -----------\n", model_path)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()

                # =================== 上传验证集可视化到 SwanLab（不保存本地） ===================
                logging.info("------------ 上传验证集预测可视化到 SwanLab -----------")
                model.eval()

                with torch.no_grad():
                    all_dice, all_precision, all_recall, all_iou = [], [], [], []

                    for batch_idx, (data, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                        data = data.to(DEVICE)
                        labels = labels.to(DEVICE)

                        preds = model(data)  # [B, 2, H, W]

                        # 指标
                        batch_dice = dice_coef_softmax(preds, labels)
                        batch_precision = precision_softmax(preds, labels)
                        batch_recall = recall_softmax(preds, labels)
                        batch_iou= iou_score_softmax(preds, labels)

                        all_dice.append(float(batch_dice))
                        all_precision.append(float(batch_precision))
                        all_recall.append(float(batch_recall))
                        all_iou.append(float(batch_iou))

                        # 清理GPU内存
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

                # 记录到全局历史
                test_metrics_history['dice'].append(mean_dice)
                test_metrics_history['iou'].append(mean_iou)
                test_metrics_history['precision'].append(mean_precision)
                test_metrics_history['recall'].append(mean_recall)

                logging.info("------------ 主动学习轮次 %d 测试结果（累计训练样本: %d）-----------", i + 1, len(labelled_img_paths))
                logging.info('Test Dice: %.4f | IoU: %.4f | Precision: %.4f | Recall: %.4f',
                             mean_dice, mean_iou, mean_precision, mean_recall)

                # 统一记录到 SwanLab（不带 iter 前缀）
                swanlab.log({
                    "test/dice": mean_dice,
                    "test/iou": mean_iou,
                    "test/precision": mean_precision,
                    "test/recall": mean_recall,
                    "train_set_size": len(labelled_img_paths)  # 可选：记录当前训练集大小
                })
            except Exception as e:
                logging.error(f"测试阶段出错: {str(e)}")
                # 如果测试失败，使用上一轮的结果或默认值
                if test_metrics_history['dice']:
                    # 使用上一轮的值
                    mean_dice = test_metrics_history['dice'][-1]
                    mean_iou = test_metrics_history['iou'][-1]
                    mean_precision = test_metrics_history['precision'][-1]
                    mean_recall = test_metrics_history['recall'][-1]
                else:
                    # 使用默认值
                    mean_dice = 0.0
                    mean_iou = 0.0
                    mean_precision = 0.0
                    mean_recall = 0.0
                continue  # 继续下一轮主动学习

    # 所有实验结束后打印完整趋势
    logging.info("\n===== 主动学习全过程测试性能趋势 =====")
    for idx, (dice, iou, prec, rec) in enumerate(zip(
        test_metrics_history['dice'],
        test_metrics_history['iou'],
        test_metrics_history['precision'],
        test_metrics_history['recall']
    ), 1):
        logging.info("轮次 %d: Dice=%.4f, IoU=%.4f, Prec=%.4f, Rec=%.4f", idx, dice, iou, prec, rec)

    overall_end = time.time()
    logging.info(
        'Learning for %d experiments has taken %.2f minutes',
        r + 1, (overall_end - overall_start) / 60
    )


if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
