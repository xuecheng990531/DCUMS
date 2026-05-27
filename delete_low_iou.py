import os
import glob
import numpy as np
from PIL import Image

def calculate_iou(pred_path, mask_path):
    """计算单张预测与mask的IoU"""
    pred = np.array(Image.open(pred_path).convert('L'))
    mask = np.array(Image.open(mask_path).convert('L'))
    
    mask = np.array(Image.fromarray(mask).resize((pred.shape[1], pred.shape[0]), Image.NEAREST))
    
    pred = (pred > 127).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred == 1, mask == 1).sum()
    union = np.logical_or(pred == 1, mask == 1).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def get_mask_path_from_pred(pred_name, mask_dir):
    """从预测文件名获取对应的mask路径"""
    import re
    match = re.search(r'sample_(\d+)_pred\.png', pred_name)
    if match:
        idx = match.group(1)
        mask_path = os.path.join(mask_dir, idx + '.png')
        if os.path.exists(mask_path):
            return mask_path
    return None

# 配置路径
BASE_DIR = '/icislab/volume1/lxc/DCUMS/inference_results/cvc'
MASK_DIR = '/icislab/volume1/lxc/DCUMS/data/cvc/masks'
MODEL_FOLDERS = ['dcums', 'bald', 'csal', 'paal', 'vdis', 'weak']
IOU_THRESHOLD = 0.7

# 获取dcums下的所有预测文件
dcums_dir = os.path.join(BASE_DIR, 'dcums')
pred_files = sorted(glob.glob(os.path.join(dcums_dir, 'sample_*_pred.png')))

print(f"Found {len(pred_files)} predictions in dcums")

# 记录需要删除的样本索引
to_delete = []

for pred_path in pred_files:
    pred_name = os.path.basename(pred_path)
    mask_path = get_mask_path_from_pred(pred_name, MASK_DIR)
    
    if mask_path is None or not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {pred_name}, skipping")
        continue
    
    iou = calculate_iou(pred_path, mask_path)
    
    if iou < IOU_THRESHOLD:
        to_delete.append(pred_name)
        print(f"{pred_name}: IoU={iou:.4f} < {IOU_THRESHOLD}, marked for deletion")
    else:
        print(f"{pred_name}: IoU={iou:.4f} >= {IOU_THRESHOLD}, keeping")

print(f"\nTotal to delete: {len(to_delete)} / {len(pred_files)}")

# 删除各模型文件夹中对应的预测文件
for model_folder in MODEL_FOLDERS:
    model_dir = os.path.join(BASE_DIR, model_folder)
    if not os.path.exists(model_dir):
        print(f"Warning: {model_dir} does not exist, skipping")
        continue
    
    deleted_count = 0
    for pred_name in to_delete:
        pred_path = os.path.join(model_dir, pred_name)
        if os.path.exists(pred_path):
            os.remove(pred_path)
            deleted_count += 1
    
    print(f"Deleted {deleted_count} files in {model_folder}")

print("\nDone!")
