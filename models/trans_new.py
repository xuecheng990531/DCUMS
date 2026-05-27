import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# 保持原有相对导入
from .VIT import ViT
from .decoder import *

def get_boundary_mask(mask, radius=2):
    """从 transreunet_class.py 提取的边界计算工具"""
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1),
                        device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=radius) > 0
    eroded  = F.conv2d(mask, kernel, padding=radius) == kernel.numel()
    return (dilated ^ eroded).squeeze()

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        # 修正：将 stride 应用于 conv2
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)   
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        x = x + x_down
        x = self.relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(
            self.vit_img_dim, out_channels * 8, out_channels * 8,
            head_num, mlp_dim, block_num,
            patch_dim=1,
            classification=False,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x, return_attention=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        if return_attention:
            x, vit_attn = self.vit(x, return_attention=True)
        else:
            x = self.vit(x)
            vit_attn = None

        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        final = self.relu(x)

        if return_attention:
            return final, x1, x2, x3, vit_attn
        return final, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2, ecb=False)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels, ecb=False)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2), ecb=False)
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8), ecb=False)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        # 移除 Sigmoid，返回 Logits 保持一致
        return x

class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num,
                 base_momentum=0.999, quality_xi=0.6,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.encoder = Encoder(
            img_dim, in_channels, out_channels,
            head_num, mlp_dim, block_num, patch_dim,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )

        self.decoder = Decoder(out_channels, class_num)

        self.register_buffer('class_quality', torch.zeros(class_num))
        self.register_buffer('class_momentum', torch.ones(class_num) * base_momentum)
        self.base_momentum = base_momentum
        self.quality_xi = quality_xi
        
    def forward(self, x, return_quality=False, return_attention=False):
        if return_attention:
            final, x1, x2, x3, vit_attn = self.encoder(x, return_attention=True)
            out = self.decoder(final, x1, x2, x3)

            if return_quality:
                return out, self.class_quality, vit_attn
            else:
                return out, vit_attn

        final, x1, x2, x3 = self.encoder(x, return_attention=False)
        out = self.decoder(final, x1, x2, x3)

        if return_quality:
            return out, self.class_quality
        else:
            return out

    @torch.no_grad()
    def update_boundary_quality(self, logits, labels, ignore_index=-1, radius=2):
        """完全对齐 AttentionUNet 的质量更新方法"""
        B, C, H, W = logits.shape
        device = logits.device

        # 使用 softmax 转换概率 (如果是 C=1 二分类，通常原框架会设 output_ch=2)
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

                # 1) boundary mask
                boundary = get_boundary_mask(mask_c, radius=radius)
                if boundary.sum() == 0:
                    boundary = mask_c

                # 2) p(gt) on boundary
                p_gt = probs[b, c][boundary]

                # 3) error-based quality
                q_img_c = p_gt.mean().pow(self.quality_xi)

                batch_sum[c] += q_img_c
                batch_count[c] += 1.0

        present = batch_count > 0
        if not present.any():
            return

        avg_q = batch_sum / (batch_count + 1e-6)

        # EMA 更新
        m = self.class_momentum
        self.class_quality[present] = (
            m[present] * self.class_quality[present]
            + (1.0 - m[present]) * avg_q[present]
        )

        # momentum 衰减
        self.class_momentum[present] = self.base_momentum
        self.class_momentum[~present] *= self.base_momentum


if __name__ == '__main__':
    # 对齐 transreunet_class.py 的调用方式
    model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=4, mlp_dim=512, block_num=4, patch_dim=16, class_num=2).cuda() # 假设 2 分类
    
    # x = torch.randn(2, 3, 256, 256).cuda()
    
    # # 得到 logits 和 quality
    # logits, quality = model(x, return_quality=True)
    # print("Logits shape:", logits.shape)
    # print("Class quality:", quality)
    print(model.__class__.__name__.lower())