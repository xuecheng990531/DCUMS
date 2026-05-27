import os
import logging
import pickle
from tqdm import tqdm
from utils.metrics import *
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from models.transreunet_class import AttentionUNet
from utils.dataset import CustomDataset
from model_train_val import AverageMeter

def main():
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------------------- Logging setup --------------------
    os.makedirs('model_logs/base_test_results/output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='model_logs/base_test_results/base_test_output.txt',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # -------------------- Load test data --------------------
    TEST_IMG_DIR = pickle.load(open('data/test/img/test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data/test/label/test.mask', 'rb'))
    logging.info(f"Number of test images: {len(TEST_IMG_DIR)}")

    # -------------------- Load model --------------------
    model = AttentionUNet(output_ch=2).to(DEVICE)
    model_path = 'pth/class_active_trained/2DUNET_experiment_1_iter_4.pth'
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    logging.info(f"Loaded trained model from {model_path}")

    # -------------------- Dataset & Loader --------------------
    test_dataset = CustomDataset(TEST_IMG_DIR, TEST_LABEL_DIR,
                                trainsize=256,
                                augmentations='False')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True)

    # -------------------- Inference --------------------
    dices=AverageMeter()
    ious=AverageMeter()
    precs=AverageMeter()
    recs=AverageMeter()


    with torch.no_grad():
        for batch_idx, (imgs, masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds,score = model(imgs,return_quality=True)
            prob = torch.softmax(preds, dim=1)
            preds_arg = prob.argmax(dim=1, keepdim=True)  # 预测的分割结果 (B,1,H,W)
            preds_to_save = preds_arg.float()

            path = 'model_logs/base_test_results/output'
            os.makedirs(path, exist_ok=True)

            # ====== 这里开始：同时保存 image / gt / pred 的横向拼接结果 ======
            for i in range(preds_to_save.size(0)):
                # 原图 (C,H,W)
                img = imgs[i].detach().cpu()

                # GT mask，可能是 (H,W) 或 (1,H,W)
                gt = masks[i].detach().cpu()
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)  # -> (1,H,W)
                gt = gt.float()

                # 预测 mask (1,H,W)
                pred = preds_arg[i].detach().cpu().float()  # (1,H,W)

                # 为了能和原图拼接，把 mask 变成 3 通道
                if img.size(0) == 3:
                    gt_vis = gt.repeat(3, 1, 1)     # (3,H,W)
                    pred_vis = pred.repeat(3, 1, 1) # (3,H,W)
                else:
                    # 如果原图不是 3 通道，就都保持同样通道数（按需要自己调整）
                    gt_vis = gt
                    pred_vis = pred

                # 按宽度方向拼接： [image | gt | pred]
                concat = torch.cat([img, gt_vis, pred_vis], dim=2)  # dim=2 是 W 方向

                save_image(concat, f'{path}/compare_batch_{batch_idx}_img_{i}.png')
            # ====== 这里结束：拼接保存 ======

            # 下面是原来的指标计算，不动
            d = dice_coef_softmax(preds, masks)     # 整个 batch 的平均
            iou = iou_score_softmax(preds, masks)
            p = precision_softmax(preds, masks)
            r = recall_softmax(preds, masks)

            dices.update(float(d))
            ious.update(float(iou))
            precs.update(float(p))
            recs.update(float(r))


        # -------------------- Aggregate Metrics --------------------
        logging.info("========== 测试结果 ==========")
        logging.info(f"Mean Dice: {dices.avg:.4f}")
        logging.info(f"Mean IoU: {ious.avg:.4f}")
        logging.info(f"Mean Precision: {precs.avg:.4f}")
        logging.info(f"Mean Recall: {recs.avg:.4f}")
        logging.info("========== 测试结果 ==========")
        


if __name__ == '__main__':
    main()
