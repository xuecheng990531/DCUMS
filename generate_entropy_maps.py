import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import shutil

from models.unet import UNet2D
from models.transreunet_class import AttentionUNet
from utils.dataset import CustomDataset

def save_entropy_map(entropy_map, save_path):
    """将熵图保存为热力图，不显示 colorbar 和坐标轴"""
    plt.figure(figsize=(8, 8))
    plt.imshow(entropy_map, cmap='jet')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_prediction(probs, save_path):
    """将预测结果保存为二值图"""
    pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
    pred_img.save(save_path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型路径
    unet_path = 'pth/entropy/unet2d/0.05/unet2d_entropy_iter_1.pth'
    att_unet_path = 'pth/DCUMS/fives/attentionunet/0.08/attentionunet_iter_14.pth'
    
    # 加载测试集路径
    try:
        test_img_paths = pickle.load(open('fives_data/test/img/test.data', 'rb'))
        test_mask_paths = pickle.load(open('fives_data/test/label/test.mask', 'rb'))
    except FileNotFoundError:
        print("Error: Could not find test data pickle files.")
        return

    # 初始化模型
    unet = UNet2D(in_channels=3, out_channels=2).to(device)
    att_unet = AttentionUNet(img_ch=3, output_ch=2).to(device)
    
    # 加载权重
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    att_unet.load_state_dict(torch.load(att_unet_path, map_location=device))
    
    unet.eval()
    att_unet.eval()
    
    # 数据加载
    dataset = CustomDataset(test_img_paths, test_mask_paths, trainsize=256, augmentations='False')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = 'inference_results/entropy_maps_top20_low'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting inference on {len(test_img_paths)} images to find top 20 lowest entropy samples for AttentionUNet...")
    
    results = []
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(loader)):
            image = image.to(device)
            
            # AttentionUNet 预测
            logits_att = att_unet(image)
            probs_att = torch.softmax(logits_att, dim=1)
            # 计算平均熵作为排序依据
            entropy_att_map = -torch.sum(probs_att * torch.log(probs_att + 1e-7), dim=1).squeeze()
            avg_entropy = entropy_att_map.mean().item()
            
            results.append({
                'index': i,
                'avg_entropy': avg_entropy,
                'entropy_att_map': entropy_att_map.cpu().numpy(),
                'image_tensor': image.cpu()
            })

    # 按平均熵升序排序（最低的前20个）
    results.sort(key=lambda x: x['avg_entropy'])
    top_20 = results[:20]
    
    print(f"Saving top 20 lowest entropy samples...")
    
    with torch.no_grad():
        for item in tqdm(top_20):
            i = item['index']
            entropy_att = item['entropy_att_map']
            image_tensor = item['image_tensor'].to(device)
            
            # 重新计算 UNet 的熵图以便对比
            logits_unet = unet(image_tensor)
            probs_unet = torch.softmax(logits_unet, dim=1)
            entropy_unet = -torch.sum(probs_unet * torch.log(probs_unet + 1e-7), dim=1).squeeze().cpu().numpy()
            
            # 重新获取 AttentionUNet 的概率图
            logits_att = att_unet(image_tensor)
            probs_att = torch.softmax(logits_att, dim=1)
            
            img_name = os.path.basename(test_img_paths[i]).split('.')[0]
            
            # 保存熵图
            save_entropy_map(entropy_unet, f"{output_dir}/{img_name}_unet_entropy.png")
            save_entropy_map(entropy_att, f"{output_dir}/{img_name}_attunet_entropy.png")
            
            # 保存预测结果
            save_prediction(probs_unet, f"{output_dir}/{img_name}_unet_pred.png")
            save_prediction(probs_att, f"{output_dir}/{img_name}_attunet_pred.png")
            
            # 保存原图和GT
            orig_img = Image.open(test_img_paths[i]).resize((256, 256))
            orig_img.save(f"{output_dir}/{img_name}_orig.png")
            
            # 直接复制原始 GT 文件并缩放，不进行二值化处理
            gt_img_raw = Image.open(test_mask_paths[i]).resize((256, 256))
            gt_img_raw.save(f"{output_dir}/{img_name}_gt.png")

    print(f"Top 20 results saved to {output_dir}")

if __name__ == '__main__':
    main()
