"""
计算模型在测试集上的置信区间
使用正态近似法（Normal Approximation），这是最快且最简单的方法。
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
from models.transreunet_class import AttentionUNet  
from models.unet import UNet2D
from utils.dataset import CustomDataset
from utils.metrics import dice_coef_softmax, precision_softmax, recall_softmax, iou_score_softmax


def compute_normal_ci(model, data_loader, device):
    """
    使用正态分布近似法计算置信区间
    """
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="推理测试集"):
            data = data.to(device)
            labels = labels.to(device)
            preds = model(data)
            
            # 记录每个 batch 的指标
            all_metrics['dice'].append(dice_coef_softmax(preds, labels).item())
            all_metrics['iou'].append(iou_score_softmax(preds, labels).item())
            all_metrics['precision'].append(precision_softmax(preds, labels).item())
            all_metrics['recall'].append(recall_softmax(preds, labels).item())
            
            del data, labels, preds
    
    results = {}
    n = len(all_metrics['dice'])
    
    for name, values in all_metrics.items():
        vals = np.array(values)
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        
        # 计算 95% 置信区间
        # sem 是标准误差 (Standard Error of the Mean)
        sem = std / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
        
        results[name] = {
            'mean': mean,
            'std_err': sem, # 标准误差
            'ci_low': ci_low,
            'ci_high': ci_high,
            'plus_minus': ci_high - mean # 即 ± 后面的数值
        }
        
    return results


def print_results(results):
    print("\n" + "="*60)
    print("置信区间分析结果 (95% Confidence Interval)")
    print("="*60)
    
    mapping = {
        
        'iou': 'IoU',
        'dice': 'Dice',
        'recall': 'Recall',
        'precision': 'Precision',
            }
    
    for key in ['iou','dice', 'recall',  'precision']:
        res = results[key]
        name = mapping[key]
        # 格式化为： 均值 ± 误差范围
        print(f"{name}: ${res['mean']:.4f}\\pm{res['plus_minus']:.4f}$")
    
    print("\n" + "="*60)


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 4
    TRAIN_SIZE = 448
    MODEL_PATH = 'pth/DCUMS/keviar-seg/attentionunet/0.15/attentionunet_experiment_1_iter_14.pth'
    
    TEST_IMG_DIR = pickle.load(open('data/train_val/img/train_val.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data/train_val/label/train_val.mask', 'rb'))

    # model = UNet2D(in_channels=3, out_channels=2).to(DEVICE)
    model=AttentionUNet(output_ch=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    
    test_dataset = CustomDataset(TEST_IMG_DIR, TEST_LABEL_DIR, trainsize=TRAIN_SIZE, augmentations='False')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("开始计算置信区间...")
    results = compute_normal_ci(model, test_loader, DEVICE)
    print_results(results)


if __name__ == '__main__':
    main()
