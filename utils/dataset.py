import os
import numpy as np
import random
import torch
from torch.utils import data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
from torch.utils.data import DataLoader



class CustomDataset(data.Dataset):
    """
    Dataloader for custom segmentation tasks using Albumentations for synchronized augmentation.
    """
    def __init__(self, image_paths, gt_paths, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations == 'True'  # 转为布尔值更安全
        
        self.images = image_paths
        self.gts = gt_paths
        self.filter_files()
        self.size = len(self.images)

        # 定义加载函数（假设你已有，这里补充示例）
        self.rgb_loader = lambda x: Image.open(x).convert('RGB')
        self.binary_loader = lambda x: Image.open(x).convert('L')

        if self.augmentations:
            self.transform = A.Compose([
                A.Rotate(limit=90, p=0.5, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(num_holes_range=(3, 6),hole_height_range=(10, 20),hole_width_range=(10, 20),fill="random_uniform",),
                A.Resize(height=trainsize, width=trainsize),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(height=trainsize, width=trainsize),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def filter_files(self):
        # 确保 image 和 gt 长度一致（你可以保留原有逻辑）
        assert len(self.images) == len(self.gts), "Number of images and gts must be the same"
        # 可添加文件存在性检查等

    def __getitem__(self, index):
        image = np.array(self.rgb_loader(self.images[index]))  # HWC, uint8
        gt = np.array(self.binary_loader(self.gts[index]))     # HW, uint8 (0 or 255)

        # 确保 gt 是二值的（0 或 1）
        gt = (gt > 127).astype(np.uint8)  # 转为 0/1

        augmented = self.transform(image=image, mask=gt)
        image_tensor = augmented['image']   # [C, H, W], float32, normalized
        gt_tensor = augmented['mask']       # [H, W], float32 (0.0 or 1.0)

        # 如果你需要 gt 是 long 类型（如用于 CrossEntropyLoss），可加：
        # gt_tensor = gt_tensor.long()

        return image_tensor, gt_tensor

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


if __name__ == "__main__":
    # 读取清单文件
    train_imgs = pickle.load(open('data/train_val/img/train_val.data', 'rb'))
    train_masks = pickle.load(open('data/train_val/label/train_val.mask', 'rb'))

    # 创建数据集实例，直接传入预加载的路径列表
    train_ds = CustomDataset(
            image_paths=train_imgs, 
            gt_paths=train_masks,
            trainsize=512,
            augmentations='False'  # 根据需求设置为'True'或'False'
        )
    
    # 创建数据加载器
    dataloader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    
    # 测试数据加载
    for i, (img, mask) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Image shape: {img.shape}")   # torch.Size([batch_size, 3, 512, 512])
        print(f"Mask shape: {mask.shape}")   # torch.Size([batch_size, 1, 512, 512])
        print(f"Image dtype: {img.dtype}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Image min/max: {img.min():.3f}/{img.max():.3f}")
        print(f"Mask min/max: {mask.min():.3f}/{mask.max():.3f}")
        
        # 只测试前几个批次
        if i >= 2:
            break



