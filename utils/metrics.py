import numpy as np
import torch
import torch.nn.functional as F

import torch
import numpy as np

def iou_score_softmax(output, target, threshold=0.5, smooth=1e-5):
    """
    output: logits [B, 2, H, W]
    target: [B, H, W] or [B, 1, H, W]
    """
    # 1. 预测类别
    if output.shape[1] == 2:
        # softmax得到概率, 取前景类（例：第1通道做为前景）
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)     # [B, H, W]，0/1
    else:
        raise ValueError('Output channel is not 2 (not softmax 2-class logits)')
    # 2. 处理 target
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0, :, :]
    target = target.long()

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    intersection = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    return (intersection + smooth) / (union + smooth)

def dice_coef_softmax(output, target, smooth=1e-5):
    """
    output: logits [B, 2, H, W]
    target: [B, H, W] or [B, 1, H, W]
    """
    if output.shape[1] == 2:
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).float() # [B, H, W]
    else:
        raise ValueError('Output channel is not 2')
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0, :, :]
    target = target.float()

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def precision_softmax(output, target, smooth=1e-5):
    if output.shape[1] == 2:
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).float()
    else:
        raise ValueError('Output channel is not 2')
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0, :, :]
    target = target.float()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (intersection + smooth) / (pred.sum() + smooth)

def recall_softmax(output, target, smooth=1e-5):
    if output.shape[1] == 2:
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).float()
    else:
        raise ValueError('Output channel is not 2')
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0, :, :]
    target = target.float()
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    intersection = (pred * target).sum()
    return (intersection + smooth) / (target.sum() + smooth)

def hausdorff_distance(mask_a,
                       mask_b,
                       percentile: int = 100,
                       spacing=None,
                       use_surface: bool = True) -> float:
    """
    计算二值掩码之间的对称 Hausdorff 距离（或 HD95）。
    
    参数：
        mask_a, mask_b : numpy.ndarray
            二值掩码（非零为前景）。形状需一致，支持 2D/3D。
        percentile : int, 默认 100
            百分位数。100 表示标准 HD；95 表示 HD95。
        spacing : None, float 或 序列
            体素间距/像素间距（如 (sy, sx) 或 (sz, sy, sx)），用于物理尺度距离。
        use_surface : bool, 默认 True
            若为 True 且可用 SciPy，则只用“表面”点来计算距离（更贴近日常评估且更快）。
    
    返回：
        float : 距离值（单位由 spacing 决定；未提供则为像素/体素单位）
    """
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"mask_a 与 mask_b 形状不一致: {a.shape} vs {b.shape}")

    # --- 可选：提取表面点 ---
    binary_erosion = None
    try:
        from scipy.ndimage import binary_erosion as _binary_erosion
        binary_erosion = _binary_erosion
    except Exception:
        pass  # 无 SciPy 时自动退化为使用所有前景点

    if use_surface and (binary_erosion is not None) and (a.ndim in (2, 3)):
        structure = np.ones((3,) * a.ndim, dtype=bool)
        a_surface = a ^ binary_erosion(a, structure=structure, border_value=0)
        b_surface = b ^ binary_erosion(b, structure=structure, border_value=0)
    else:
        a_surface = a
        b_surface = b

    # 取坐标点（行、列、(层)）
    pts_a = np.argwhere(a_surface)
    pts_b = np.argwhere(b_surface)

    # 边界情况
    if pts_a.size == 0 and pts_b.size == 0:
        return 0.0
    if pts_a.size == 0 or pts_b.size == 0:
        # 一个空、一个非空：通常定义为无穷大
        return float("inf")

    # 应用 spacing（像素到物理尺寸）
    if spacing is not None:
        sp = np.array(spacing, dtype=float)
        if sp.size == 1:
            sp = np.repeat(sp, pts_a.shape[1])
        if sp.size != pts_a.shape[1]:
            raise ValueError(f"spacing 维度应为 {pts_a.shape[1]}，实际为 {sp.size}")
        pts_a = pts_a * sp
        pts_b = pts_b * sp

    # --- 计算 P->Q 与 Q->P 的最近距离数组 ---
    def min_distances(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        # 优先用 KDTree 加速
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(Q)
            d, _ = tree.query(P, k=1)
            return d.astype(float)
        except Exception:
            # 退化为纯 numpy 计算（大图像可能较慢）
            diff = P[:, None, :] - Q[None, :, :]
            return np.sqrt((diff * diff).sum(axis=2)).min(axis=1)

    d_ab = min_distances(pts_a, pts_b)
    d_ba = min_distances(pts_b, pts_a)

    # 百分位 Hausdorff
    p = 100 if percentile is None else float(percentile)
    if p >= 100:
        return float(max(d_ab.max(initial=0.0), d_ba.max(initial=0.0)))
    else:
        return float(max(np.percentile(d_ab, p), np.percentile(d_ba, p)))


if __name__ == "__main__":
    # 简单测试
    mask1 = np.array([[0, 1, 1],
                      [0, 1, 0],
                      [0, 0, 0]])

    mask2 = np.array([[0, 0, 1],
                      [1, 1, 0],
                      [0, 0, 0]])

    hd = hausdorff_distance(mask1, mask2)
    print(f"Hausdorff Distance: {hd}")