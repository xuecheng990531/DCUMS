import os
import glob
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from models.transreunet_class import AttentionUNet
from models.unet import UNet2D
from utils.metrics import iou_score_softmax, dice_coef_softmax, precision_softmax, recall_softmax
from utils.five_dataset import CustomDataset

def load_paired_data(img_dir, mask_dir):
    """Load paired image and mask paths based on filenames"""
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
    mask_paths = []
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        mask_path = os.path.join(mask_dir, name + '.png')
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            mask_path = os.path.join(mask_dir, name + '.jpg')
            if os.path.exists(mask_path):
                mask_paths.append(mask_path)
            else:
                mask_path = os.path.join(mask_dir, name + ext)
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                else:
                    mask_paths.append(None)
    
    valid_pairs = [(img, mask) for img, mask in zip(img_paths, mask_paths) if mask is not None and os.path.exists(mask)]
    img_paths = [p[0] for p in valid_pairs]
    mask_paths = [p[1] for p in valid_pairs]
    
    return img_paths, mask_paths

def inference():
    os.makedirs('inference_results/cvc', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='inference_results/cvc/inference_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    TRAINSIZE = 256

    MODEL_PATH = 'pth/weak-anno/0.15/2DUNET_experiment_1_iter_14.pth'
    IMG_DIR = 'data/cvc/images'
    MASK_DIR = 'data/cvc/masks'
    OUTPUT_DIR = 'inference_results/cvc/weak'

    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Loading model from: {MODEL_PATH}")
    
    # model = AttentionUNet(img_ch=3, output_ch=2).to(DEVICE)
    model = UNet2D(in_channels=3, out_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logging.info("Model loaded successfully")

    img_paths, mask_paths = load_paired_data(IMG_DIR, MASK_DIR)
    logging.info(f"Found {len(img_paths)} image-mask pairs")

    filenames = [os.path.basename(p) for p in img_paths]
    for i, f in enumerate(filenames):
        name, ext = os.path.splitext(f)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            ext = '.png'
        filenames[i] = name + '_pred.png'

    dataset = CustomDataset(img_paths, mask_paths, trainsize=TRAINSIZE, augmentations='False')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="Inference")):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            
            preds = model(imgs)
            
            prob = torch.softmax(preds, dim=1)
            preds_arg = prob.argmax(dim=1, keepdim=True)
            preds_vis = preds_arg.float()

            batch_ious = []
            batch_dices = []
            batch_precisions = []
            batch_recalls = []

            for sample_idx in range(imgs.size(0)):
                pred_sample = preds[sample_idx:sample_idx+1]
                mask_sample = masks[sample_idx:sample_idx+1]
                d = dice_coef_softmax(pred_sample, mask_sample)
                iou = iou_score_softmax(pred_sample, mask_sample)
                p = precision_softmax(pred_sample, mask_sample)
                r = recall_softmax(pred_sample, mask_sample)
                batch_ious.append(float(iou))
                batch_dices.append(float(d))
                batch_precisions.append(float(p))
                batch_recalls.append(float(r))

            all_dice.extend(batch_dices)
            all_iou.extend(batch_ious)
            all_precision.extend(batch_precisions)
            all_recall.extend(batch_recalls)

            for i in range(preds_arg.size(0)):
                idx = batch_idx * BATCH_SIZE + i
                if idx < len(img_paths):
                    pred_np = preds_arg[i, 0].cpu().numpy().astype(np.uint8)
                    pred_img = Image.fromarray(pred_np * 255)
                    pred_img.save(os.path.join(OUTPUT_DIR, filenames[idx]))

    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)

    logging.info("=" * 50)
    logging.info("Inference Results:")
    logging.info(f"Dice: {mean_dice:.4f}")
    logging.info(f"IoU: {mean_iou:.4f}")
    logging.info(f"Precision: {mean_precision:.4f}")
    logging.info(f"Recall: {mean_recall:.4f}")
    logging.info("=" * 50)

    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"Dice: {mean_dice:.4f}")
    print(f"IoU: {mean_iou:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print("=" * 50)

    metrics = {
        'dice': mean_dice,
        'iou': mean_iou,
        'precision': mean_precision,
        'recall': mean_recall
    }
    
    return metrics

if __name__ == '__main__':
    metrics = inference()
