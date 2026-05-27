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
# from models.unet import *
from models.transreunet_class import *
from utils.dataset import CustomDataset
import math

# 修复torch.optim.AdamW导入问题
try:
    from torch.optim import AdamW
except ImportError:
    from torch.optim import Adam as AdamW  # 降级到Adam，如果AdamW不可用

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
    nb_active_learning_iter = 15         # 主动学习轮数
    nb_active_learning_iter_size = 15    # 每轮新增样本数
    max_iters = 2000                          # max_iters
    VAL_INTERVAL = 100                     # 每隔多少个迭代做一次验证

    LEARNING_RATE = 1e-4
    # EARLY_STOP = 25                      # 验证次数早停阈值
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 12

    # =================== Swanlab 配置 ===================
    swanlab.init(
        project="active_loop",
        workspace="xuecheng",
        config={
            "iters": max_iters,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
        }
    )

    # =================== 日志配置 ===================
    learning_type = 'random' if random_bool else 'active'
    os.makedirs(f'experiment_logs/{learning_type}_learning_results', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/{learning_type}_learning_results/{learning_type}_learning_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # =================== 模型、损失、优化器 ===================
    model = AttentionUNet(output_ch=2).to(DEVICE)
    logging.info("=> GO GO GO 出发咯")

    loss_fn = Softmax_CE_DiceLoss_reduction()
    # 使用我们定义的AdamW或Adam
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
        )

    overall_start = time.time()

    # 初始化全局测试指标记录器
    test_metrics_history = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }

    # =================== 多次实验循环 default 1===================
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
            if random_bool:
                model_path = f'pth/random_trained/2DUNET_experiment_{r+1}_iter_{i}.pth'
                results_path = 'random_learning_results'
                data_type = 'random'
            else:
                model_path = f'pth/class_active_trained/2DUNET_experiment_{r+1}_iter_{i}.pth'
                results_path = 'active_learning_results'
                data_type = 'uncertain'

            path = os.path.join(results_path, f"experiment_{r+1}", f"iteration_{i}")
            os.makedirs(path, exist_ok=True)

            # 第 0 轮：从 base 模型开始
            if i == 0:
                model.load_state_dict(torch.load('pth/base_trained/TransresUNet_cls.pth',weights_only=True))

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
                    len(unlabelled_img_paths),
                    nb_active_learning_iter_size,
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

            # =================== 伯努利随机采样（Active Inference） ===================
                N_unlabeled = len(unlabelled_img_paths)
                K = nb_active_learning_iter_size
                tau = 0.2  # 混合参数，与原文保持一致
                mean_u = np.mean(uncertainty_scores)
                if mean_u == 0:
                    mean_u = 1e-8
                #计算
                b = K / N_unlabeled
                eta = b / mean_u
                pi_base = eta * uncertainty_scores
                pi_mix = (1 - tau) * pi_base + tau * b
                # 确保概率在 [0,1] 区间
                pi_mix = np.clip(pi_mix, 0.0, 1.0)
                # 伯努利采样：每个样本独立以概率 pi_mix 被选中
                bernoulli_probs = np.random.random(size=N_unlabeled)
                to_be_annotated_indices = np.where(bernoulli_probs <= pi_mix)[0]
                # 注意：选中的样本数可能不等于 K，但期望值为 K
                logging.info("伯努利采样选中 %d 个样本（期望值 %d）", len(to_be_annotated_indices), K)
                # 控制选中样本数不超过 K
                if len(to_be_annotated_indices) > K:
                    # 如果选中数 > K，则选择 π_mix 最大的 K 个样本（确定性选择）
                    # 获取选中样本对应的概率
                    selected_probs = pi_mix[to_be_annotated_indices]
                    # 按概率降序排序，选择前 K 个
                    sorted_indices = np.argsort(selected_probs)[::-1]  # 降序
                    top_k_indices = sorted_indices[:K]
                    to_be_annotated_indices = to_be_annotated_indices[top_k_indices]
                    logging.info("选中数超过 K，选择 π_mix 最大的 %d 个样本", K)
                elif len(to_be_annotated_indices) < K:
                    # 如果选中数 < K，从剩余样本中按 π_mix 补足
                    remaining_indices = np.setdiff1d(np.arange(N_unlabeled), to_be_annotated_indices)
                    if len(remaining_indices) > 0:
                        remaining_probs = pi_mix[remaining_indices]
                        # 按概率降序排序，选择前 (K - len(to_be_annotated_indices)) 个
                        needed = K - len(to_be_annotated_indices)
                        sorted_indices = np.argsort(remaining_probs)[::-1]
                        top_needed_indices = sorted_indices[:needed]
                        additional_indices = remaining_indices[top_needed_indices]
                        to_be_annotated_indices = np.concatenate([to_be_annotated_indices, additional_indices])
                        logging.info("选中数不足 K，从剩余样本中补足 %d 个高概率样本", needed)
                # 如果选中数 == K，直接使用



                labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths = add_annotated_sample(
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
                logging.info("样本索引 %s 被选为高不确定性样本", str(to_be_annotated_indices))

            logging.info("本次迭代新增了 %d 个样本到训练集", len(to_be_annotated_indices))
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

            best_dice = 0.0
            trigger = 0
            start = time.time()

            for it in range(1, max_iters + 1):
                try:
                    # 一次“迭代”：从 train_loader_iter 取一个 batch 并更新参数
                    loss, iou, dice, train_loader_iter, class_quality= train(
                        train_loader, train_loader_iter,
                        model, loss_fn, optimizer, DEVICE,al_round=i,
                    )

                    logging.info("Iter %d/%d, Train Loss: %.4f, IoU: %.4f, Dice: %.4f",
                                 it, max_iters, loss, iou, dice)

                    # swanlab.log({
                    #     f"iter_{i+1}/train/loss": float(loss),
                    #     f"iter_{i+1}/train/iou": float(iou),
                    #     f"iter_{i+1}/train/dice": float(dice),
                    # })


                    # 按“迭代次数”决定是否验证：每 VAL_INTERVAL 次验证一次，最后一次必验证
                    do_validate = (it % VAL_INTERVAL == 0) or (it == max_iters)

                    if do_validate:
                        logging.info("验证集上评估模型 (after iter %d)", it)
                        val_log = validate(val_loader, model, loss_fn, DEVICE)
                        
                        logging.info("Validation Loss: %.4f, IoU: %.4f, Dice: %.4f",val_log['loss'], val_log['iou'], val_log['dice'])

                        # swanlab.log({
                        #     f"iter_{i+1}/val/loss": float(val_log['loss']),
                        #     f"iter_{i+1}/val/iou": float(val_log['iou']),
                        #     f"iter_{i+1}/val/dice": float(val_log['dice'])
                        # })

                        # 以“验证次数”为单位的 early stopping 计数
                        trigger += 1

                        if val_log['dice'] > best_dice:
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            torch.save(model.state_dict(), model_path)
                            best_dice = val_log['dice']
                            logging.info("=> 模型参数更新，保存 (iter %d)", it)
                            trigger = 0

                        # if trigger >= EARLY_STOP:
                        #     logging.info(
                        #         "=> 早停 (no improvement for %d validations)",
                        #         EARLY_STOP
                        #     )
                        #     break
                except Exception as e:
                    logging.error(f"训练迭代 {it} 出错: {str(e)}")
                    continue  # 继续下一次迭代

                # 定期清理GPU内存
                if it % 100 == 0 and DEVICE == 'cuda':
                    torch.cuda.empty_cache()

            end = time.time()
            logging.info(
                'Training and (periodic) validation for active learning iteration %d has taken %.2f minutes',
                i, (end - start) / 60
            )

            # =================== 测试集评估 ===================
            logging.info("------------ 第 %d 次主动学习完成，开始测试 -----------\n", i + 1)

            try:
                test_dataset = CustomDataset(
                    test_img_paths, test_mask_paths,trainsize=256, augmentations='False'
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
