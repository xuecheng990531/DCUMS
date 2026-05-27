import os
import time
from collections import OrderedDict
import logging
import swanlab
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np

from utils.metrics import *
from utils.utils import *
from models.trans_new import TransUNet
from utils.dataset import CustomDataset
from utils.loss import *
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

def get_class_weights(class_quality, ub, alpha, eps=1e-6):
    reverse_q = 1 - class_quality
    b = math.exp(1.0 / alpha) - 1
    return 1 + alpha * torch.log(b * reverse_q + 1 + eps) * ub

def train_step(
    data, labels, model, loss_fn, optimizer, device,
    al_round,
    warmup_rounds=2,
    scaler=None
):
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
                difficulty_map = torch.ones(
                    preds.size(0), preds.size(2), preds.size(3),
                    device=preds.device
                )
            else:
                quality = model.class_quality.clamp(0.0, 1.0)
                class_weights = get_class_weights(
                    class_quality=quality,
                    ub=0.2,
                    alpha=0.3
                )
                probs = torch.softmax(preds, dim=1)
                difficulty_map = (
                    probs * class_weights.view(1, -1, 1, 1)
                ).sum(dim=1)

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

    iou_score = float(iou_score_softmax(preds, labels))
    dice_score = float(dice_coef_softmax(preds, labels))
    precision_score = float(precision_softmax(preds, labels))
    recall_score = float(recall_softmax(preds, labels))

    return (
        loss.item(),
        iou_score,
        dice_score,
        precision_score,
        recall_score,
        model.class_quality.detach().cpu()
    )

def validate(val_loader, model, loss_fn, device):
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    precs = AverageMeter()
    recs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            data = data.to(device)
            labels = labels.to(device)

            y = model(data, return_quality=True)
            preds = y[0]

            val_loss = loss_fn(preds, labels)
            val_iou = float(iou_score_softmax(preds, labels))
            val_dice = float(dice_coef_softmax(preds, labels))
            val_prec = float(precision_softmax(preds, labels))
            val_rec = float(recall_softmax(preds, labels))

            batch_size = data.size(0)
            losses.update(val_loss.item(), batch_size)
            ious.update(val_iou, batch_size)
            dices.update(val_dice, batch_size)
            precs.update(val_prec, batch_size)
            recs.update(val_rec, batch_size)

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('precision', precs.avg),
        ('recall', recs.avg),
    ])

    return log

def main():
    TOTAL_ITRS = 2000
    VAL_INTERVAL = 40
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 12

    # EMA 动量参数消融 (m ∈ {0.9, 0.99, 0.999})
    BASE_MOMENTUM = 0.9

    swanlab.init(
        project="momentum_ablation",
        name=f"TransResUNet_m{BASE_MOMENTUM}",
        workspace="xuecheng",
        config={
            "total_itrs": TOTAL_ITRS,
            "val_interval": VAL_INTERVAL,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "base_momentum": BASE_MOMENTUM,
        }
    )

    os.makedirs(f'experiment_logs/momentum_ablation/m{BASE_MOMENTUM}', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'experiment_logs/momentum_ablation/m{BASE_MOMENTUM}/training_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    TRAIN_IMG_DIR = pickle.load(open('data/train_val/img/' + 'train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data/train_val/label/' + 'train_val.mask', 'rb'))

    # 重写路径：数据已移至 /icislab/volume1/lxc/polyp_data/kvasir/
    TRAIN_IMG_DIR = [p.replace('data/Kvasir-SEG/', '/icislab/volume1/lxc/polyp_data/kvasir/') for p in TRAIN_IMG_DIR]
    TRAIN_LABEL_DIR = [p.replace('data/Kvasir-SEG/', '/icislab/volume1/lxc/polyp_data/kvasir/') for p in TRAIN_LABEL_DIR]

    from sklearn.model_selection import train_test_split
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        test_size=0.2,
        random_state=41
    )
    logging.info(f"train_num: {len(train_img_paths)}")
    logging.info(f"val_num: {len(val_img_paths)}")

    model = TransUNet(
        img_dim=256, in_channels=3, out_channels=128, head_num=4, mlp_dim=512,
        block_num=4, patch_dim=16, class_num=2, base_momentum=BASE_MOMENTUM
    ).to(DEVICE)
    modelname = model.__class__.__name__.lower()
    loss_fn = Softmax_CE_DiceLoss_reduction()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    train_dataset = CustomDataset(
        train_img_paths, train_mask_paths,
        trainsize=256, augmentations="True"
    )
    val_dataset = CustomDataset(
        val_img_paths, val_mask_paths,
        trainsize=256, augmentations="False"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False
    )

    best_iou = 0.0
    start = time.time()

    train_loader_iter = iter(train_loader)
    pbar = tqdm(total=TOTAL_ITRS, desc=f"Training (m={BASE_MOMENTUM})", unit="iter")

    quality_history = []

    for iteration in range(1, TOTAL_ITRS + 1):
        try:
            data, labels = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            data, labels = next(train_loader_iter)

        loss, iou, dice, prec, rec, class_quality = train_step(
            data=data,
            labels=labels,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE,
            al_round=3,
        )

        swanlab.log({
            "train/loss": loss,
            "train/iou": iou,
            "train/dice": dice,
            "train/precision": prec,
            "train/recall": rec,
        }, step=iteration)

        if iteration % VAL_INTERVAL == 0 or iteration == TOTAL_ITRS:
            logging.info(f"Validation at iteration {iteration}")
            val_log = validate(val_loader, model, loss_fn, DEVICE)
            logging.info(
                f"[Iter {iteration}] Val Loss: {val_log['loss']:.4f}, "
                f"IoU: {val_log['iou']:.4f}, Dice: {val_log['dice']:.4f}, "
                f"Prec: {val_log['precision']:.4f}, Rec: {val_log['recall']:.4f}"
            )

            swanlab.log({
                "val/loss": val_log['loss'],
                "val/iou": val_log['iou'],
                "val/dice": val_log['dice'],
                "val/precision": val_log['precision'],
                "val/recall": val_log['recall']
            }, step=iteration)

            quality_history.append(class_quality.tolist())
            swanlab.log({
                "ema/class_quality_bg": class_quality[0].item(),
                "ema/class_quality_fg": class_quality[1].item(),
            }, step=iteration)

            if val_log['iou'] > best_iou:
                best_iou = val_log['iou']
                logging.info(f"=> new best IoU: {best_iou:.4f} at iteration {iteration}")

        pbar.set_postfix({
            'Loss': f'{loss:.4f}',
            'IoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}',
            'Prec': f'{prec:.4f}',
            'Rec': f'{rec:.4f}',
            'best_iou': f'{best_iou:.4f}'
        })
        pbar.update(1)

        torch.cuda.empty_cache()

    pbar.close()
    end = time.time()
    logging.info(f"Training completed in {(end - start)/60:.2f} minutes")
    logging.info(f"Best validation IoU: {best_iou:.4f}")
    logging.info(f"Final class_quality: {class_quality.tolist()}")

    np.save(f'experiment_logs/momentum_ablation/m{BASE_MOMENTUM}/quality_history.npy', np.array(quality_history))

    print("\n" + "="*50)
    print(f"Momentum m = {BASE_MOMENTUM} 消融实验结果")
    print("="*50)
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Best Val Dice: {val_log['dice']:.4f}")
    print(f"Final class_quality (background): {class_quality[0]:.4f}")
    print(f"Final class_quality (foreground): {class_quality[1]:.4f}")
    print(f"Training time: {(end - start)/60:.2f} minutes")
    print("="*50)

if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
