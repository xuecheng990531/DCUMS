import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def make_one_hot(labels, classes):
    """
    labels: [B, H, W] (LongTensor, 每个像素是 [0, classes-1] 或 ignore_index)
    返回:  [B, classes, H, W]
    """
    b, h, w = labels.size()
    one_hot = torch.zeros(b, classes, h, w, device=labels.device, dtype=torch.float)
    # ignore_index 处0即可
    valid_mask = (labels >= 0) & (labels < classes)
    labels_valid = labels.clone()
    labels_valid[~valid_mask] = 0
    return one_hot.scatter_(1, labels_valid.unsqueeze(1), 1)

class SoftmaxDiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(SoftmaxDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if target.dim() == 4 and target.size(1) == 1:
            target = target[:, 0, :, :]
        target = target.long()
        num_classes = output.size(1)
        valid_mask = (target != self.ignore_index)  # [B, H, W]

        target_1h = make_one_hot(target, classes=num_classes)
        for c in range(num_classes):
            target_1h[:,c,:,:] *= valid_mask

        probs = F.softmax(output, dim=1)
        probs = probs * valid_mask.unsqueeze(1)

        probs_flat = probs.contiguous().view(probs.size(0), -1)
        target_flat = target_1h.contiguous().view(target_1h.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)

        loss = 1 - dice.mean()
        return loss

class Softmax_CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1., reduction='mean', ignore_index=255, weight=None):
        super(Softmax_CE_DiceLoss, self).__init__()
        self.dice = SoftmaxDiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, ignore_index=ignore_index
        )
    
    def forward(self, output, target):
        if target.dim() == 4 and target.size(1) == 1:
            target_ce = target[:, 0, :, :].long()
        else:
            target_ce = target.long()

        ce_loss   = self.cross_entropy(output, target_ce)
        dice_loss = self.dice(output, target_ce)
        return ce_loss + dice_loss
    

class Softmax_CE_DiceLoss_reduction(nn.Module):
    def __init__(self, smooth=1., reduction='mean', ignore_index=255, weight=None):
        super(Softmax_CE_DiceLoss_reduction, self).__init__()
        self.dice = SoftmaxDiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, reduction='none', ignore_index=ignore_index # 关键：设置 reduction='none'
        )
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target, reduction='mean'):
        # 允许在 forward 时动态指定 reduction
        reduction = reduction or self.reduction

        # 处理 target 格式
        if target.dim() == 4 and target.size(1) == 1:
            target_ce = target[:, 0, :, :].long()
        else:
            target_ce = target.long()

        # 计算逐像素的 CE 损失
        ce_loss_per_pixel = self.cross_entropy(output, target_ce) # [B, H, W]

        # 计算 Dice 损失
        # 注意：DiceLoss 目前不支持 reduction='none'，我们需要修改它或在外部处理
        # 让我们先看 DiceLoss 的实现
        dice_loss_val = self.dice(output, target_ce) # 这个返回的是一个标量

        # --- 修正 DiceLoss 以支持 per-pixel 计算 ---
        # 由于 DiceLoss 基于全局计算 (sum over batch, h, w)，很难直接拆分成 per-pixel
        # 最直接的方式是：对于需要 per-pixel 损失的场景（如 DCUS），我们暂时只使用 CE 损失进行调制
        # 或者，我们可以近似地将 DiceLoss 的标量值按像素平均分摊回去
        # 这里我们采用第一种方式，只调制 CE 部分，这是最常见且安全的做法
        # 因为 CE 损失对每个像素的贡献是明确的，而 Dice 是一个全局度量

        if reduction == 'none':
            # 只返回逐像素的 CE 损失，用于 DCUS 调制
            # 如果你想要包含 Dice 的影响，可以考虑一个近似：将 Dice_loss_val 除以有效像素数，然后加到 CE_loss 上
            # 但为简单起见，我们先返回 CE_loss_per_pixel
            # 为了更合理，我们可以将 Dice_loss 平摊到每个像素上
            # 但这会引入一些复杂性。一个更简单的做法是：
            # 1. 计算总的 CE 和 Dice
            # 2. 如果 reduction='none'，我们只返回 CE，因为 CE 是 per-pixel 的
            # 3. 如果 reduction='mean'/'sum'，我们返回 CE + Dice
            # 这里我们选择返回 CE_loss_per_pixel，因为它是 per-pixel 的，可以直接用于调制
            # 并且 CE_loss 本身已经包含了大部分像素级别的信息。
            return ce_loss_per_pixel # [B, H, W]

        # 如果不是 'none'，则按原始方式计算
        ce_loss_val = ce_loss_per_pixel.mean() if reduction == 'mean' else ce_loss_per_pixel.sum()
        total_loss = ce_loss_val + dice_loss_val
        return total_loss

if __name__ == "__main__":
    # 随机模拟2类输出（batch=2, channel=2, 256, 256）
    yhat = torch.randn(2, 2, 256, 256)
    # 随机模拟标签 [B, 1, H, W]，值为0或1
    target = torch.randint(0, 2, (2, 1, 256, 256))
    target_ce = target[:, 0, :, :].long()
    # 损失函数
    loss_fn = Softmax_CE_DiceLoss()
    loss = loss_fn(yhat, target_ce)
    print(loss.item())