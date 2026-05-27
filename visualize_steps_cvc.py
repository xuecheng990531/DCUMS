import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.trans_new import TransUNet
from utils.dataset import CustomDataset
import pickle
from einops import rearrange

def visualize_attention_steps(model, image, mask, save_dir="visualizations_cvc"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        # image: [1, 3, 256, 256]
        out, vit_attn = model(image, return_attention=True)
        
        # Infer resolution h
        b_desc_sample = vit_attn[0]["boundary_descriptor"]
        h = int(b_desc_sample.shape[1]**0.5)

        # Original image for reference
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_ref = (img_np * std + mean).clip(0, 1)
        
        # Save input and mask for context
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_ref); plt.title("Input Image"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(mask[0].cpu().numpy(), cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
        plt.savefig(os.path.join(save_dir, "step0_input_gt.png"))
        plt.close()

        # Compare Block 0 and Last Block to show evolution
        for b_idx in [0, len(vit_attn)-1]:
            aux_b = vit_attn[b_idx]
            spatial_attn = aux_b["attention"][0, :, 1:, 1:].mean(dim=0)
            attn_received = spatial_attn.mean(dim=0).reshape(h, h).cpu().numpy()
            attn_resized = F.interpolate(torch.tensor(attn_received).unsqueeze(0).unsqueeze(0), 
                                       size=(256, 256), mode='bilinear').squeeze().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(img_ref); plt.imshow(attn_resized, cmap='jet', alpha=0.5)
            plt.title(f"Global Attention (Block {b_idx})"); plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"global_attention_block_{b_idx}.png"))
            plt.close()

        # Detailed 5-step analysis for Block 0 (initial gating)
        block_idx = 0
        aux = vit_attn[block_idx]
        
        # Step 0.5: Raw Feature Map & Token Tiling
        feat_2d = aux.get("feature_2d_raw")
        if feat_2d is not None:
            # feat_2d shape: [1, H, W, C] -> [16, 16] (assuming 16x16 tokens)
            feat_map = feat_2d[0].mean(dim=-1).cpu().numpy()
            
            # 1. 保存原始 Token 特征网格 (不插值，显示颗粒感)
            plt.figure(figsize=(8, 8))
            plt.imshow(feat_map, cmap='viridis')
            plt.title(f"Raw Token Feature Grid ({feat_map.shape[0]}x{feat_map.shape[1]})")
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, "step0_5_token_grid_raw.png"))
            plt.close()

            # 2. 将原始图像切分成 Token 对应的 Patch 小图
            token_dir = os.path.join(save_dir, "token_tiles")
            os.makedirs(token_dir, exist_ok=True)
            
            # img_ref is [256, 256, 3]
            grid_h, grid_w = feat_map.shape
            patch_size = 256 // grid_h # 通常是 16
            
            from PIL import Image
            
            # 遍历所有 Token 空间位置
            count = 0
            for r in range(grid_h):
                for c in range(grid_w):
                    # 提取该 Token 对应的图像 Patch
                    y1, y2 = r * patch_size, (r + 1) * patch_size
                    x1, x2 = c * patch_size, (c + 1) * patch_size
                    patch_img = img_ref[y1:y2, x1:x2]
                    
                    # 转换并放大 Patch 方便观察 (从 16x16 放大到 64x64)
                    patch_pil = Image.fromarray((patch_img * 255).astype(np.uint8))
                    patch_pil = patch_pil.resize((64, 64), Image.NEAREST)
                    
                    # 获取该位置的特征强度值
                    f_val = feat_map[r, c]
                    
                    # 保存文件名包含坐标和特征值
                    patch_pil.save(os.path.join(token_dir, f"token_r{r}_c{c}_val{f_val:.2f}.png"))
                    count += 1
            print(f"Exported {count} token tiles to {token_dir}")

        # Step 1: Semantic Attention Matrix
        sem_attn = aux["semantic_attn"][0, 0].cpu().numpy() # [T, T]
        plt.figure(figsize=(8, 8))
        plt.imshow(sem_attn, cmap='viridis')
        plt.title("Step 1: Semantic Attention Matrix (A_sem)")
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, "step1_semantic_attn_matrix.png"))
        plt.close()

        # Step 2: Boundary Descriptor (Overlay)
        b_desc = aux["boundary_descriptor"]
        h = int(b_desc.shape[1]**0.5) # Define h here
        if b_desc is not None:
            b_map = b_desc[0].reshape(h, h).cpu().numpy()
            b_map_resized = F.interpolate(torch.tensor(b_map).unsqueeze(0).unsqueeze(0), 
                                        size=(256, 256), mode='bilinear').squeeze().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(img_ref)
            plt.imshow(b_map_resized, cmap='hot', alpha=0.5)
            plt.title("Step 2: Boundary Descriptor Overlay (b_i)")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, "step2_boundary_overlay.png"))
            plt.close()
            
        # Step 3: Relevance Descriptor (Overlay)
        r_desc = aux["relevance_descriptor"]
        if r_desc is not None:
            r_map = r_desc[0].reshape(h, h).cpu().numpy()
            r_map_resized = F.interpolate(torch.tensor(r_map).unsqueeze(0).unsqueeze(0), 
                                        size=(256, 256), mode='bilinear').squeeze().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(img_ref)
            plt.imshow(r_map_resized, cmap='coolwarm', alpha=0.5)
            plt.title("Step 3: Relevance Descriptor Overlay (r_i)")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, "step3_relevance_overlay.png"))
            plt.close()

        # Step 4: Token-to-Token Constraints (B_ij and R_ij)
        b_map_ij = aux.get("boundary_map")
        if b_map_ij is not None:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            # b_map_ij is [B, T, T]
            plt.imshow(b_map_ij[0].cpu().numpy(), cmap='viridis')
            plt.title("Step 4a: Boundary Constraint Matrix (B_ij)")
            plt.colorbar()
            
            r_map_ij = aux.get("relevance_map")
            if r_map_ij is not None:
                plt.subplot(1, 2, 2)
                # r_map_ij is [B, T, T]
                plt.imshow(r_map_ij[0].cpu().numpy(), cmap='viridis')
                plt.title("Step 4b: Relevance Constraint Matrix (R_ij)")
                plt.colorbar()
            
            plt.savefig(os.path.join(save_dir, "step4_constraints_matrices.png"))
            plt.close()

        # Step 5: Final Attention Score (Gating Analysis)
        # Comparison between Raw Semantic and Gated Attention
        # CLS-to-Patch Raw Semantic
        raw_cls = aux["semantic_attn"][0, :, 0, 1:].mean(dim=0)
        raw_cls_2d = raw_cls.reshape(h, h).cpu().numpy()
        raw_cls_resized = F.interpolate(torch.tensor(raw_cls_2d).unsqueeze(0).unsqueeze(0), 
                                       size=(256, 256), mode='bilinear').squeeze().numpy()
        
        # CLS-to-Patch Final Attention
        cls_attn = aux["attention"][0, :, 0, 1:].mean(dim=0)
        attn_map_2d = cls_attn.reshape(h, h).cpu().numpy()
        attn_map_resized = F.interpolate(torch.tensor(attn_map_2d).unsqueeze(0).unsqueeze(0), 
                                       size=(256, 256), mode='bilinear').squeeze().numpy()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_ref); plt.imshow(raw_cls_resized, cmap='jet', alpha=0.5)
        plt.title("Step 5a: Raw Semantic CLS-Attn"); plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(img_ref); plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        plt.title("Step 5b: Gated CLS-Attn (Optimized)"); plt.axis('off')

        # Also visualize attention from a "central" or "lesion" patch if possible
        # We look at spatial patch-to-patch attention
        spatial_attn = aux["attention"][0, :, 1:, 1:].mean(dim=0) # [T, T]
        
        mask_down = F.interpolate(mask.float().unsqueeze(0), size=(h, h), mode='nearest').squeeze()
        lesion_indices = torch.where(mask_down.flatten() > 0.5)[0]
        if len(lesion_indices) > 0:
            target_idx = lesion_indices[len(lesion_indices)//2] # middle of lesion
            attn_from_lesion = spatial_attn[target_idx].reshape(h, h).cpu().numpy()
            attn_from_lesion_resized = F.interpolate(torch.tensor(attn_from_lesion).unsqueeze(0).unsqueeze(0), 
                                                   size=(256, 256), mode='bilinear').squeeze().numpy()
            plt.subplot(1, 3, 2)
            plt.imshow(img_ref)
            plt.imshow(attn_from_lesion_resized, cmap='jet', alpha=0.5)
            plt.title("Gated Attention from Lesion")
            plt.axis('off')
            
        # Prediction for context
        pred = torch.sigmoid(out)
        pred_np = (pred[0, 1] > 0.5).cpu().numpy() if pred.shape[1] > 1 else (pred[0, 0] > 0.5).cpu().numpy()
        plt.subplot(1, 3, 3)
        plt.imshow(img_ref)
        plt.imshow(pred_np, cmap='gray', alpha=0.3)
        plt.title("Model Prediction Overlay")
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, "step5_spatial_analysis.png"))
        plt.close()

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransUNet(img_dim=256, in_channels=3, out_channels=128, head_num=4, 
                      mlp_dim=512, block_num=4, patch_dim=16, class_num=2).to(DEVICE)
    
    weight_path = "/icislab/volume1/lxc/DCUMS/pth/DCUMS/keviar-seg/aaaaa.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"Weights loaded from {weight_path}")

    # Use CVC training data
    try:
        train_imgs = pickle.load(open('data/cvc/train_val/img/train_val.data', 'rb'))
        train_masks = pickle.load(open('data/cvc/train_val/label/train_val.mask', 'rb'))
        dataset = CustomDataset(image_paths=train_imgs, gt_paths=train_masks, trainsize=256, augmentations='False')
        
        # Pick a sample that is actually correctly predicted by the model from the training set
        found = False
        for idx in range(0, len(dataset), 10): # Check every 10th sample for variety
            image, mask = dataset[idx]
            if mask.sum() > 2000: # Ensure there is a substantial lesion
                img_input = image.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(img_input)
                    # For training set, we expect high IoU
                    pred = torch.softmax(out, dim=1)[:, 1] > 0.5
                    iou = (pred & (mask.to(DEVICE) > 0.5)).sum() / (pred | (mask.to(DEVICE) > 0.5)).sum()
                    
                if iou > 0.8: # High IoU for training sample
                    print(f"Using CVC Training sample index {idx} with IoU: {iou.item():.4f}")
                    visualize_attention_steps(model, img_input, mask.to(DEVICE).unsqueeze(0), save_dir="visualizations_train")
                    found = True
                    break
        
        if not found:
            print("No training sample with >0.8 IoU found, using first available significant lesion.")
            for idx in range(len(dataset)):
                image, mask = dataset[idx]
                if mask.sum() > 1000:
                    visualize_attention_steps(model, image.unsqueeze(0).to(DEVICE), mask.to(DEVICE).unsqueeze(0), save_dir="visualizations_train")
                    break
            
    except Exception as e:
        print(f"Error loading CVC training data: {e}")
        image = torch.randn(1, 3, 256, 256).to(DEVICE)
        mask = torch.zeros(1, 256, 256).to(DEVICE)
        visualize_attention_steps(model, image, mask)

    print("CVC training set visualizations generated in 'visualizations_train/' directory.")
