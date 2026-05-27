import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pickle
from torchvision.transforms import ToPILImage
from dataset import CustomDataset
from metrics import dice_coef_softmax, iou_score_softmax
from scipy import ndimage


import torch
import torch.nn as nn
import torch.nn.functional as F


def get_boundary_mask(mask, radius=2):
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1),
                        device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=radius) > 0
    eroded  = F.conv2d(mask, kernel, padding=radius) == kernel.numel()
    return (dilated ^ eroded).squeeze()

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1,base_momentum=0.999, quality_xi=0.6):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # ===== 类别质量的 EMA 估计 =====
        self.register_buffer('class_quality', torch.zeros(output_ch))
        self.register_buffer('class_momentum',
                             torch.ones(output_ch) * base_momentum)
        self.base_momentum = base_momentum
        self.quality_xi = quality_xi



    def forward(self, x, return_quality=False):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        # ========== 新增：根据标志返回 quality ==========
        if return_quality:
            return out, self.class_quality
        else:
            return out
    
    @torch.no_grad()
    def update_boundary_quality(self, logits, labels,
                            ignore_index=-1, radius=2):
        """
        Boundary-aware, error-based quality proxy
        """
        B, C, H, W = logits.shape
        device = logits.device

        probs = torch.softmax(logits, dim=1)
        labels = labels.long().to(device)

        batch_sum   = torch.zeros(C, device=device)
        batch_count = torch.zeros(C, device=device)

        for b in range(B):
            lbl = labels[b]
            valid = (lbl != ignore_index) & (lbl >= 0) & (lbl < C)
            if valid.sum() == 0:
                continue

            for c in range(C):
                mask_c = (lbl == c)
                if not mask_c.any():
                    continue

                # === 1) boundary mask ===
                boundary = get_boundary_mask(mask_c, radius=radius)
                if boundary.sum() == 0:
                    boundary = mask_c

                # === 2) p(gt) on boundary ===
                p_gt = probs[b, c][boundary]

                # === 3) error-based quality ===
                # mean(p_gt) ^ xi
                q_img_c = p_gt.mean().pow(self.quality_xi)

                batch_sum[c] += q_img_c
                batch_count[c] += 1.0

        present = batch_count > 0
        if not present.any():
            return

        avg_q = batch_sum / (batch_count + 1e-6)

        # ===== EMA 更新 =====
        m = self.class_momentum
        self.class_quality[present] = (
            m[present] * self.class_quality[present]
            + (1.0 - m[present]) * avg_q[present]
        )

        # ===== momentum 衰减 =====
        self.class_momentum[present] = self.base_momentum
        self.class_momentum[~present] *= self.base_momentum


def generate_xor_visualization(model_path, test_img_paths, test_mask_paths, output_dir, num_samples=5):
    """
    Generate XOR visualization results and save as images
    
    Parameters:
        model_path: Path to the pretrained model
        test_img_paths: List of test image paths
        test_mask_paths: List of test mask paths
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    model = AttentionUNet(output_ch=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded pretrained model: {model_path}")
    
    # 创建数据集和数据加载器
    test_dataset = CustomDataset(test_img_paths, test_mask_paths, trainsize=256, augmentations='False')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 转换器，用于将tensor转换为PIL图像
    to_pil = ToPILImage()
    
    # 处理每个样本
    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # 为每个样本创建单独的文件夹
            sample_dir = os.path.join(output_dir, f'sample_{i+1}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 将数据移到设备
            img = img.to(device)
            mask = mask.to(device)
            
            # 模型预测
            pred_logits, _ = model(img, return_quality=True)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_mask = torch.argmax(pred_probs, dim=1).float()
            
            # 获取前景概率
            foreground_prob = pred_probs[0, 1].cpu().numpy()
            
            # 将数据移回CPU并转换为numpy
            img_np = img[0].cpu().numpy()
            mask_np = mask[0].cpu().numpy()
            pred_np = pred_mask[0].cpu().numpy()
            
            # 反归一化图像（用于显示）
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_display = img_np * std + mean
            img_display = np.clip(img_display, 0, 1)
            img_display = np.transpose(img_display, (1, 2, 0))
            
            # 计算XOR结果（预测与真实掩码的异或）
            xor_result = np.logical_xor(mask_np.astype(bool), pred_np.astype(bool)).astype(float)
            
            # 生成XOR概率图 - 只显示XOR区域内的概率分布
            xor_probability = foreground_prob * xor_result
            
            # 对XOR结果进行膨胀腐蚀操作
            # 定义结构元素（3x3的方形结构）
            structure = np.ones((3, 3), dtype=bool)
            
            # 先进行膨胀操作，扩大差异区域
            dilated_xor = ndimage.binary_dilation(xor_result.astype(bool), structure=structure, iterations=1).astype(float)
            
            # 再进行腐蚀操作，去除小的噪声
            eroded_xor = ndimage.binary_erosion(dilated_xor, structure=structure, iterations=1).astype(float)
            
            # 也可以尝试先腐蚀后膨胀（开运算）去除小噪声，再膨胀（闭运算）填充小孔
            opened_xor = ndimage.binary_opening(xor_result.astype(bool), structure=structure).astype(float)
            closed_xor = ndimage.binary_closing(opened_xor, structure=structure).astype(float)
            
            # 计算评估指标
            dice = dice_coef_softmax(pred_logits, mask)
            iou = iou_score_softmax(pred_logits, mask)
            
            # 保存输入图像
            input_img_fig, input_img_ax = plt.subplots(1, 1, figsize=(8, 8))
            input_img_ax.imshow(img_display)
            input_img_ax.axis('off')
            input_img_path = os.path.join(sample_dir, 'input_image.png')
            input_img_fig.savefig(input_img_path, dpi=300, bbox_inches='tight')
            plt.close(input_img_fig)
            print(f"Saved input image: {input_img_path}")
            
            # 保存真实掩码
            gt_mask_fig, gt_mask_ax = plt.subplots(1, 1, figsize=(8, 8))
            gt_mask_ax.imshow(mask_np, cmap='gray')
            gt_mask_ax.axis('off')
            gt_mask_path = os.path.join(sample_dir, 'ground_truth_mask.png')
            gt_mask_fig.savefig(gt_mask_path, dpi=300, bbox_inches='tight')
            plt.close(gt_mask_fig)
            print(f"Saved ground truth mask: {gt_mask_path}")
            
            # 保存预测掩码
            pred_mask_fig, pred_mask_ax = plt.subplots(1, 1, figsize=(8, 8))
            pred_mask_ax.imshow(pred_np, cmap='gray')
            pred_mask_ax.axis('off')
            pred_mask_path = os.path.join(sample_dir, 'predicted_mask.png')
            pred_mask_fig.savefig(pred_mask_path, dpi=300, bbox_inches='tight')
            plt.close(pred_mask_fig)
            print(f"Saved predicted mask: {pred_mask_path}")
            
            # 保存预测概率图
            prob_fig, prob_ax = plt.subplots(1, 1, figsize=(8, 8))
            im = prob_ax.imshow(foreground_prob, cmap='hot', vmin=0, vmax=1)
            prob_ax.axis('off')
            plt.colorbar(im, ax=prob_ax, fraction=0.046, pad=0.04)
            prob_path = os.path.join(sample_dir, 'probability_map.png')
            prob_fig.savefig(prob_path, dpi=300, bbox_inches='tight')
            plt.close(prob_fig)
            print(f"Saved probability map: {prob_path}")
            
            # 保存XOR概率图
            xor_prob_fig, xor_prob_ax = plt.subplots(1, 1, figsize=(8, 8))
            xor_prob_im = xor_prob_ax.imshow(xor_probability, cmap='hot', vmin=0, vmax=1)
            xor_prob_ax.axis('off')
            plt.colorbar(xor_prob_im, ax=xor_prob_ax, fraction=0.046, pad=0.04)
            xor_prob_path = os.path.join(sample_dir, 'xor_probability_map.png')
            xor_prob_fig.savefig(xor_prob_path, dpi=300, bbox_inches='tight')
            plt.close(xor_prob_fig)
            print(f"Saved XOR probability map: {xor_prob_path}")
            
            # 保存XOR结果
            xor_fig, xor_ax = plt.subplots(1, 1, figsize=(8, 8))
            xor_ax.imshow(xor_result, cmap='Reds')
            xor_ax.axis('off')
            xor_path = os.path.join(sample_dir, 'xor_result.png')
            xor_fig.savefig(xor_path, dpi=300, bbox_inches='tight')
            plt.close(xor_fig)
            print(f"Saved XOR result: {xor_path}")
            
            # 保存膨胀后的XOR结果
            dilated_fig, dilated_ax = plt.subplots(1, 1, figsize=(8, 8))
            dilated_ax.imshow(dilated_xor, cmap='Reds')
            dilated_ax.axis('off')
            dilated_path = os.path.join(sample_dir, 'dilated_xor_result.png')
            dilated_fig.savefig(dilated_path, dpi=300, bbox_inches='tight')
            plt.close(dilated_fig)
            print(f"Saved dilated XOR result: {dilated_path}")
            
            # 保存腐蚀后的XOR结果
            eroded_fig, eroded_ax = plt.subplots(1, 1, figsize=(8, 8))
            eroded_ax.imshow(eroded_xor, cmap='Reds')
            eroded_ax.axis('off')
            eroded_path = os.path.join(sample_dir, 'eroded_xor_result.png')
            eroded_fig.savefig(eroded_path, dpi=300, bbox_inches='tight')
            plt.close(eroded_fig)
            print(f"Saved eroded XOR result: {eroded_path}")
            
            # 保存开运算+闭运算后的XOR结果
            closed_fig, closed_ax = plt.subplots(1, 1, figsize=(8, 8))
            closed_ax.imshow(closed_xor, cmap='Reds')
            closed_ax.axis('off')
            closed_path = os.path.join(sample_dir, 'closed_xor_result.png')
            closed_fig.savefig(closed_path, dpi=300, bbox_inches='tight')
            plt.close(closed_fig)
            print(f"Saved closed XOR result: {closed_path}")
            
            # 保存叠加图
            overlay_fig, overlay_ax = plt.subplots(1, 1, figsize=(8, 8))
            overlay_ax.imshow(img_display)
            
            # 创建彩色掩码
            # 真实掩码（绿色）
            green_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
            green_mask[:, :, 1] = mask_np  # 绿色通道
            
            # 预测掩码（红色）
            red_mask = np.zeros((pred_np.shape[0], pred_np.shape[1], 3))
            red_mask[:, :, 0] = pred_np  # 红色通道
            
            # XOR结果（黄色）
            yellow_mask = np.zeros((xor_result.shape[0], xor_result.shape[1], 3))
            yellow_mask[:, :, 0] = xor_result  # 红色通道
            yellow_mask[:, :, 1] = xor_result  # 绿色通道
            
            # 叠加显示
            alpha = 0.4
            overlay_ax.imshow(green_mask, alpha=alpha)
            overlay_ax.imshow(red_mask, alpha=alpha)
            overlay_ax.imshow(yellow_mask, alpha=alpha*1.5)  # XOR结果更明显
            
            overlay_ax.axis('off')
            
            # 添加图例
            green_patch = mpatches.Patch(color='green', label='Ground Truth')
            red_patch = mpatches.Patch(color='red', label='Prediction')
            yellow_patch = mpatches.Patch(color='yellow', label='Difference')
            overlay_ax.legend(handles=[green_patch, red_patch, yellow_patch],
                            loc='upper right', bbox_to_anchor=(1.1, 1))
            
            overlay_path = os.path.join(sample_dir, 'overlay_display.png')
            overlay_fig.savefig(overlay_path, dpi=300, bbox_inches='tight')
            plt.close(overlay_fig)
            print(f"Saved overlay display: {overlay_path}")
            
            # 创建综合可视化图
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            # 第一行：原始图像、真实掩码、预测掩码
            axes[0, 0].imshow(img_display)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth Mask')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(pred_np, cmap='gray')
            axes[0, 2].set_title('Predicted Mask')
            axes[0, 2].axis('off')
            
            # 第二行：前景概率、XOR概率图、原始XOR结果
            im = axes[1, 0].imshow(foreground_prob, cmap='hot', vmin=0, vmax=1)
            axes[1, 0].set_title('Probability Map')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            xor_prob_im = axes[1, 1].imshow(xor_probability, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_title('XOR Probability Map')
            axes[1, 1].axis('off')
            plt.colorbar(xor_prob_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            axes[1, 2].imshow(xor_result, cmap='Reds')
            axes[1, 2].set_title('Original XOR Result')
            axes[1, 2].axis('off')
            
            # 第三行：膨胀后XOR结果、腐蚀后XOR结果、开闭运算后XOR结果
            axes[2, 0].imshow(dilated_xor, cmap='Reds')
            axes[2, 0].set_title('Dilated XOR Result')
            axes[2, 0].axis('off')
            
            axes[2, 1].imshow(eroded_xor, cmap='Reds')
            axes[2, 1].set_title('Eroded XOR Result')
            axes[2, 1].axis('off')
            
            axes[2, 2].imshow(closed_xor, cmap='Reds')
            axes[2, 2].set_title('Opened+Closed XOR Result')
            axes[2, 2].axis('off')
            
            # 第四行：重叠显示
            # 重叠显示：原始图像 + 真实掩码（绿色） + 预测掩码（红色） + XOR结果（黄色）
            axes[3, 0].imshow(img_display)
            axes[3, 0].imshow(green_mask, alpha=alpha)
            axes[3, 0].imshow(red_mask, alpha=alpha)
            axes[3, 0].imshow(yellow_mask, alpha=alpha*1.5)  # XOR结果更明显
            axes[3, 0].set_title('Overlay Display')
            axes[3, 0].axis('off')
            
            # 添加图例
            axes[3, 0].legend(handles=[green_patch, red_patch, yellow_patch],
                             loc='upper right', bbox_to_anchor=(1.1, 1))
            
            # 隐藏第四行的其他两个子图
            axes[3, 1].axis('off')
            axes[3, 2].axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存综合可视化结果
            comprehensive_path = os.path.join(sample_dir, 'comprehensive_visualization.png')
            plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comprehensive visualization: {comprehensive_path}")
            print(f"All results for sample {i+1} saved to: {sample_dir}")

def main():
    # 设置路径
    model_path = 'pth/class_active_trained/2DUNET_experiment_1_iter_14.pth'  # 默认使用这个模型
    test_img_paths = pickle.load(open('data/test/img/test.data', 'rb'))
    test_mask_paths = pickle.load(open('data/test/label/test.mask', 'rb'))
    output_dir = 'xor_visualization_results'
    
    # 生成可视化结果
    print("Starting to generate XOR visualization results...")
    generate_xor_visualization(model_path, test_img_paths, test_mask_paths, output_dir, num_samples=5)
    print("XOR visualization results generation completed!")

if __name__ == '__main__':
    main()