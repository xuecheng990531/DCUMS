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


     
if __name__ == "__main__":
    model = AttentionUNet(img_ch=3, output_ch=2)
    x = torch.randn((2, 3, 256, 256))
    y = model(x,return_quality=True)
    print(y[0].shape, y[1])  # 输出形状和类别质量参数