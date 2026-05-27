# 加入dcussample采样方式，需要计算类别难度


import os
import time
from collections import OrderedDict
import logging
from sklearn.model_selection import train_test_split
import swanlab
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle

from utils.metrics import *
from utils.utils import *
# from models.transreunet_class import *
# from models.unet import UNet2D
from models.trans_new import TransUNet
from utils.dataset import CustomDataset
from utils.loss import *
import math
# from torchvision.utils import save_image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # val 可以是 tensor 或 float，统一转成 float
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

def get_class_weights(class_quality, ub, alpha, eps=1e-6):
    reverse_q = 1 - class_quality
    b = math.exp(1.0 / alpha) - 1
    return 1 + alpha * torch.log(b * reverse_q + 1 + eps) * ub


def train(train_loader, train_loader_iter,
          model, loss_fn, optimizer, device,al_round, scaler=None):
    """
    进行“一次迭代”的训练：
      - 从 train_loader_iter 里取一个 batch
      - 如果迭代器耗尽，重新从 train_loader 构建迭代器
      - 调用 train_step 完成一次更新
    返回：
      loss, iou, dice, 新的 train_loader_iter
    """
    try:
        data, labels = next(train_loader_iter)
    except StopIteration:
        # dataloader 走完了，重新开始
        train_loader_iter = iter(train_loader)
        data, labels = next(train_loader_iter)

    loss, iou, dice, prec, rec, quality = train_step(data, labels, model, loss_fn, optimizer, device,al_round, scaler=scaler)
    return loss, iou, dice, prec, rec, train_loader_iter, quality


def train_step(
    data, labels, model, loss_fn, optimizer, device,
    al_round,
    warmup_rounds=2,
    scaler=None
):
    model.train()
    data, labels = data.to(device), labels.to(device)

    optimizer.zero_grad()

    # 使用 autocast 进行混合精度训练
    with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
        preds, _ = model(data, return_quality=True)  # [B,2,H,W]

        # =====================================================
        # 1) EMA 更新 difficulty（❗仅在 warm-up 之后）
        # =====================================================
        if al_round >= warmup_rounds:
            model.update_boundary_quality(preds.detach(), labels,radius=4)

        # =====================================================
        # 2) 计算 difficulty map
        #    - warm-up: 全 1（等价于普通训练）
        #    - DCUS:    按论文定义
        # =====================================================
        use_dcus = (al_round is not None) and (al_round >= warmup_rounds)
        with torch.no_grad():
            
            if not use_dcus:
                # 不使用 difficulty
                difficulty_map = torch.ones(
                    preds.size(0), preds.size(2), preds.size(3),
                    device=preds.device
                )
            else:
                quality = model.class_quality.clamp(0.0, 1.0)  # [C]
                class_weights = get_class_weights(
                    class_quality=quality,
                    ub=0.2,
                    alpha=0.3
                )  # [C]

                probs = torch.softmax(preds, dim=1)  # [B,C,H,W]
                difficulty_map = (
                    probs * class_weights.view(1, -1, 1, 1)
                ).sum(dim=1)  # [B,H,W]

        # =====================================================
        # 3) Loss modulation
        # =====================================================
        raw_loss = loss_fn(preds, labels, reduction='none')  # [B,H,W]

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
    """
    完整跑一遍验证集：
      - 遍历 val_loader
      - 计算平均 loss / IoU / Dice
    不做任何可视化和文件保存，只返回指标
    """
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
            preds=y[0]
            # aux=y[2]
            # save_image(aux["att2"]["struct_map"], f"preds_{batch_idx}.png")
            # 

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
    # ========= 超参数（按迭代数） =========
    TOTAL_ITRS = 2000          # 总迭代次数（梯度更新步数）
    VAL_INTERVAL = 40          # 每多少次迭代做一次验证
    EARLY_STOP_PATIENCE = 10   # 验证无提升的次数（按"验证轮次"计数）达到此值时早停
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 12

    # EMA 动量参数消融 (m ∈ {0.9, 0.99, 0.999})
    BASE_MOMENTUM = 0.999

    swanlab.init(
        project="momentum_ablation",
        name=f"TransResUNet_m{BASE_MOMENTUM}",
        workspace="xuecheng",
        config={
            "total_itrs": TOTAL_ITRS,
            "val_interval": VAL_INTERVAL,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "base_momentum": BASE_MOMENTUM,
        }
    )

    # ========= 日志设置 =========
    os.makedirs('experiment_logs/model_logs/base_training_results', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='experiment_logs/model_logs/base_training_results/five_base_training_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # ========= 加载数据路径 =========
    TRAIN_IMG_DIR = pickle.load(open('data/cvc/train_val/img/train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data/cvc/train_val/label/train_val.mask', 'rb'))

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        test_size=0.2,
        random_state=41
    )
    logging.info("train_num:%s" % str(len(train_img_paths)))
    logging.info("val_num:%s" % str(len(val_img_paths)))

    # ========= 模型 / 损失 / 优化器 =========
    # model = AttentionUNet(img_ch=3, output_ch=2).to(DEVICE)
    model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=4, mlp_dim=512, block_num=4, patch_dim=16, class_num=2, base_momentum=BASE_MOMENTUM).to(DEVICE) # 假设 2 分类
    # model = UNet2D(in_channels=3, out_channels=2).to(DEVICE)
    modelname=model.__class__.__name__.lower()
    loss_fn = Softmax_CE_DiceLoss_reduction()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # ========= Dataset & Dataloader =========
    train_dataset = CustomDataset(
        train_img_paths,
        train_mask_paths,
        trainsize=256,
        augmentations="True"
    )

    val_dataset = CustomDataset(
        val_img_paths,
        val_mask_paths,
        trainsize=256,
        augmentations="False"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    # ========= 迭代级训练总控 =========
    best_iou = 0.0
    trigger = 0                 # 早停计数（验证轮数）
    start = time.time()

    train_loader_iter = iter(train_loader)  # 用迭代器循环使用 train_loader
    pbar = tqdm(total=TOTAL_ITRS, desc="Training", unit="iter")

    for iteration in range(1, TOTAL_ITRS + 1):
        # ---- 1. 进行一次“迭代级”的 train ----
        loss, iou, dice, prec, rec, train_loader_iter, class_quality = train(
            train_loader=train_loader,
            train_loader_iter=train_loader_iter,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE,
            al_round=-1 #单独训练模型的时候设置为-1，不更新ema，
        )

        # 单步训练日志（每次迭代都记）
        swanlab.log({
            "train/loss": loss,
            "train/iou": iou,
            "train/dice": dice,
            "train/precision": prec,
            "train/recall": rec,
        }, step=iteration)

        # early_stop = False

        # ---- 2. 按迭代数决定是否做一次完整验证 ----
        if (iteration % VAL_INTERVAL == 0) or (iteration == TOTAL_ITRS):
            logging.info(f"Validation at iteration {iteration}")
            val_log = validate(val_loader, model, loss_fn, DEVICE)
            logging.info(
                f"[Iter {iteration}] Val Loss: {val_log['loss']:.4f}, "
                f"IoU: {val_log['iou']:.4f}, Dice: {val_log['dice']:.4f}, "
                f"Prec: {val_log['precision']:.4f}, Rec: {val_log['recall']:.4f}"
            )

            # 打到 swanlab 上（验证阶段单独加几个 key）
            swanlab.log({
                "val/loss": val_log['loss'],
                "val/iou": val_log['iou'],
                "val/dice": val_log['dice'],
                "val/precision": val_log['precision'],
                "val/recall": val_log['recall']
            }, step=iteration)


            # ---- 3. 早停 & 保存最优模型逻辑（按“验证轮次”）----
            if val_log['iou'] > best_iou:
                os.makedirs('pth/DCUMS/keviar-seg', exist_ok=True)
                torch.save(model.state_dict(), f'pth/DCUMS/keviar-seg/aaaaa.pth')
                best_iou = val_log['iou']
                logging.info(
                    f"=> best model has been saved (iter: {iteration}, iou: {best_iou:.4f})"
                )
                trigger = 0  # 验证集效果提升，重置早停计数器
            else:
                trigger += 1  # 验证集效果未提升，累加早停计数器

            # if trigger >= EARLY_STOP_PATIENCE:
            #     logging.info(f"=> early stopping (iter: {iteration})")
            #     early_stop = True

        # ---- 4. 更新进度条 & 显示当前训练指标 ----
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

        # if early_stop:
        #     break

    pbar.close()
    end = time.time()
    logging.info(
        f"Training and validation has taken {(end - start)/60:.2f} minutes to complete"
    )
    logging.info(f"Best validation iou: {best_iou:.4f}")


if __name__ == '__main__':
    swanlab.login(api_key="Fw4gmSiE7N3TbkPV8YFBN")
    main()
