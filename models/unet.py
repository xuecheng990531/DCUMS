import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf
import torch.nn.functional as F

def get_boundary_mask(mask, radius=2):
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1),
                        device=mask.device)
    dilated = F.conv2d(mask, kernel, padding=radius) > 0
    eroded  = F.conv2d(mask, kernel, padding=radius) == kernel.numel()
    return (dilated ^ eroded).squeeze()
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # 3x3 kernel, stride 1, padding same
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        return self.conv(x)

# 2D UNET MODEL
class UNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], base_momentum=0.999, quality_xi=0.6):
        super(UNet2D, self).__init__()
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.2)
        
        # ===== 类别质量的 EMA 估计 =====
        self.register_buffer('class_quality', torch.zeros(out_channels))
        self.register_buffer('class_momentum',
                             torch.ones(out_channels) * base_momentum)
        self.base_momentum = base_momentum
        self.quality_xi = quality_xi

        # Down part of UNet - encoder
        for feature in features:                            # adds 4 convblocks
            self.encode.append(DoubleConvBlock(in_channels, feature))   
            in_channels = feature

        # Up part of UNet - decoder
        for feature in reversed(features):
            self.decode.append(nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=4, padding = 1, stride=2,        # upsample
                )
            )
            self.decode.append(DoubleConvBlock(feature*2, feature))     # adds 4 conv blocks using reversed features

        self.bottleneck = DoubleConvBlock(features[-1], features[-1]*2)           # bottleneck
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)                
    
    def forward(self, x, return_quality=False):
        skip_connections = []                                   # skip connections skip some layer in the  network and feeds the output of one layer as the input to the next layers (instead of only one)
        for down in self.encode:                                # for every conv block
            x = down(x)
            skip_connections.append(x)                          # append result to skip_connections list
            x = self.pool(x)                                    # apply maxpooling after double conv
            x = self.dropout(x)                                 # apply dropout after every pooling layer
         
        x = self.bottleneck(x)
        x = self.dropout(x)
        skip_connections = skip_connections[::-1]               # reverse the order of skip connections
  
        for idx in range(0, len(self.decode), 2):               # step of 2
            x = self.decode[idx](x)                             # conv transpose
            skip_connection = skip_connections[idx//2]          # step by 2
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decode[idx+1](concat_skip)
            #x = self.dropout(x)                                 # apply dropout after conv block
        out = self.final_conv(x)                     # final 1x1 conv
        
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
    
if __name__ == "__main__":
    x = torch.randn((3, 3, 160, 160))          # batch size 3, 4 channels, 160x160 image
    model = UNet2D(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)                         # should be (3, 3, 160, 160)