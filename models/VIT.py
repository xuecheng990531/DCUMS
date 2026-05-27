import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class BoundaryRelevanceMultiHeadAttention(nn.Module):
    """
    任务定制化 self-attention:
    A = semantic_attn * (1 + lb * boundary_map) * (1 + lr * relevance_map)

    - boundary_map: 基于 token 局部差分的结构/边界响应
    - relevance_map: 基于 token 自身响应的目标相关性
    """
    def __init__(self, embedding_dim, head_num, boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        assert embedding_dim % head_num == 0
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        self.scale = self.head_dim ** -0.5

        self.boundary_lambda = boundary_lambda
        self.relevance_lambda = relevance_lambda

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # relevance descriptor
        self.relevance_proj = nn.Linear(embedding_dim, 1, bias=True)

    def _infer_hw(self, token_num: int):
        """
        token_num 不包含 cls token
        这里默认 token 来自规则网格
        """
        h = int(token_num ** 0.5)
        w = h
        assert h * w == token_num, f"Token number {token_num} is not a square number."
        return h, w

    def _compute_boundary_descriptor(self, x_no_cls):
        """
        x_no_cls: [B, T, C]
        return: [B, T, 1]
        """
        B, T, C = x_no_cls.shape
        h, w = self._infer_hw(T)

        feat_2d = rearrange(x_no_cls, 'b (h w) c -> b h w c', h=h, w=w)

        # x方向差分（沿W）
        diff_x = feat_2d[:, :, 1:, :] - feat_2d[:, :, :-1, :]
        diff_x = torch.cat([diff_x, diff_x[:, :, -1:, :]], dim=2)   # 补最后一列

        # y方向差分（沿H）
        diff_y = feat_2d[:, 1:, :, :] - feat_2d[:, :-1, :, :]
        diff_y = torch.cat([diff_y, diff_y[:, -1:, :, :]], dim=1)   # 补最后一行

        grad = torch.sqrt(diff_x.pow(2) + diff_y.pow(2) + 1e-6)
        grad = grad.mean(dim=-1, keepdim=True)  # [B, H, W, 1]

        mean = grad.mean(dim=(1, 2), keepdim=True)
        std = grad.std(dim=(1, 2), keepdim=True) + 1e-6
        grad = (grad - mean) / std
        grad = torch.sigmoid(grad)

        b = rearrange(grad, 'b h w c -> b (h w) c')
        return b
    def _compute_relevance_descriptor(self, x_no_cls):
        """
        x_no_cls: [B, T, C]
        return: [B, T, 1]
        """
        r = torch.sigmoid(self.relevance_proj(x_no_cls))
        return r

    def forward(self, x, return_attention=False):
        """
        x: [B, T(+1), C]
        注意默认第0个token是cls token
        """
        B, T_all, C = x.shape

        qkv = self.qkv_layer(x)
        query, key, value = tuple(
            rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num)
        )  # [B, heads, T, head_dim]

        # 1) 原始语义注意力
        semantic_attn = torch.einsum("bhid,bhjd->bhij", query, key) * self.scale

        # 2) 为非cls token构造边界/相关性描述符
        if T_all > 1:
            x_no_cls = x[:, 1:, :]  # [B, T, C]
            h_feat, w_feat = self._infer_hw(T_all - 1)
            feat_2d_raw = rearrange(x_no_cls, 'b (h w) c -> b h w c', h=h_feat, w=w_feat)

            b = self._compute_boundary_descriptor(x_no_cls)   # [B, T, 1]
            r = self._compute_relevance_descriptor(x_no_cls)  # [B, T, 1]

            boundary_map = b @ b.transpose(-2, -1)            # [B, T, T]
            relevance_map = r @ r.transpose(-2, -1)           # [B, T, T]

            # 扩展到包含 cls token 的大小
            full_boundary = torch.ones(B, T_all, T_all, device=x.device, dtype=x.dtype)
            full_relevance = torch.ones(B, T_all, T_all, device=x.device, dtype=x.dtype)

            full_boundary[:, 1:, 1:] = 1.0 + self.boundary_lambda * boundary_map
            full_relevance[:, 1:, 1:] = 1.0 + self.relevance_lambda * relevance_map

            # 让 CLS token 也能感知到描述符的引导
            # CLS token 对 Patch 的注意力 (1, 0, 1:) 应该被 b 和 r 增强
            full_boundary[:, 0, 1:] = 1.0 + self.boundary_lambda * b.squeeze(-1)
            full_relevance[:, 0, 1:] = 1.0 + self.relevance_lambda * r.squeeze(-1)

            joint_gate = full_boundary * full_relevance       # [B, T_all, T_all]
            joint_gate = joint_gate.unsqueeze(1)              # [B, 1, T_all, T_all]
        else:
            b = None
            r = None
            boundary_map = None
            relevance_map = None
            joint_gate = torch.ones(B, 1, T_all, T_all, device=x.device, dtype=x.dtype)

        # 3) 联合门控后的注意力
        energy = semantic_attn * joint_gate
        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attention, value)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_attention(out)

        if return_attention:
            aux = {
                "semantic_attn": semantic_attn,
                "joint_gate": joint_gate,
                "attention": attention,
                "boundary_descriptor": b,
                "relevance_descriptor": r,
                "boundary_map": boundary_map,
                "relevance_map": relevance_map,
                "feature_2d_raw": feat_2d_raw if T_all > 1 else None
            }
            return out, aux
        return out


class BoundaryRelevanceCrossAttention(nn.Module):
    """
    对应原来的 MultiHeadAttention_tri
    用 tx 作为 query，x 作为 key/value
    同时在 x 的 token 上构造 boundary / relevance 约束
    """
    def __init__(self, embedding_dim, head_num, boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        assert embedding_dim % head_num == 0
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        self.scale = self.head_dim ** -0.5

        self.boundary_lambda = boundary_lambda
        self.relevance_lambda = relevance_lambda

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.tqkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.relevance_proj = nn.Linear(embedding_dim, 1, bias=True)

    def _infer_hw(self, token_num: int):
        h = int(token_num ** 0.5)
        w = h
        assert h * w == token_num
        return h, w

    def _compute_boundary_descriptor(self, x_no_cls):
        B, T, C = x_no_cls.shape
        h, w = self._infer_hw(T)

        feat_2d = rearrange(x_no_cls, 'b (h w) c -> b h w c', h=h, w=w)

        diff_x = feat_2d[:, :, 1:, :] - feat_2d[:, :, :-1, :]
        diff_x = F.pad(diff_x, (0, 0, 0, 0, 0, 1), mode='replicate')

        diff_y = feat_2d[:, 1:, :, :] - feat_2d[:, :-1, :, :]
        diff_y = F.pad(diff_y, (0, 0, 0, 1, 0, 0), mode='replicate')

        grad = torch.sqrt(diff_x.pow(2) + diff_y.pow(2) + 1e-6)
        grad = grad.mean(dim=-1, keepdim=True)

        mean = grad.mean(dim=(1, 2), keepdim=True)
        std = grad.std(dim=(1, 2), keepdim=True) + 1e-6
        grad = (grad - mean) / std
        grad = torch.sigmoid(grad)

        b = rearrange(grad, 'b h w c -> b (h w) c')
        return b

    def _compute_relevance_descriptor(self, x_no_cls):
        return torch.sigmoid(self.relevance_proj(x_no_cls))

    def forward(self, x, tx, return_attention=False):
        qkv = self.qkv_layer(x)
        tqkv = self.tqkv_layer(tx)

        query, key, value = tuple(
            rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num)
        )
        tquery, tkey, tvalue = tuple(
            rearrange(tqkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num)
        )

        semantic_attn = torch.einsum("bhid,bhjd->bhij", tquery, key) * self.scale

        B, T_all, _ = x.shape
        if T_all > 1:
            x_no_cls = x[:, 1:, :]
            h_feat, w_feat = self._infer_hw(T_all - 1)
            feat_2d_raw = rearrange(x_no_cls, 'b (h w) c -> b h w c', h=h_feat, w=w_feat)

            b = self._compute_boundary_descriptor(x_no_cls)
            r = self._compute_relevance_descriptor(x_no_cls)

            boundary_map = b @ b.transpose(-2, -1)
            relevance_map = r @ r.transpose(-2, -1)

            full_boundary = torch.ones(B, T_all, T_all, device=x.device, dtype=x.dtype)
            full_relevance = torch.ones(B, T_all, T_all, device=x.device, dtype=x.dtype)

            full_boundary[:, 1:, 1:] = 1.0 + self.boundary_lambda * boundary_map
            full_relevance[:, 1:, 1:] = 1.0 + self.relevance_lambda * relevance_map
            
            # CLS token gating
            full_boundary[:, 0, 1:] = 1.0 + self.boundary_lambda * b.squeeze(-1)
            full_relevance[:, 0, 1:] = 1.0 + self.relevance_lambda * r.squeeze(-1)

            joint_gate = (full_boundary * full_relevance).unsqueeze(1)
        else:
            b = None
            r = None
            boundary_map = None
            relevance_map = None
            joint_gate = torch.ones(B, 1, T_all, T_all, device=x.device, dtype=x.dtype)

        energy = semantic_attn * joint_gate
        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attention, value)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_attention(out)

        if return_attention:
            aux = {
                "semantic_attn": semantic_attn,
                "joint_gate": joint_gate,
                "attention": attention,
                "boundary_descriptor": b,
                "relevance_descriptor": r,
                "boundary_map": boundary_map,
                "relevance_map": relevance_map,
                "feature_2d_raw": feat_2d_raw if T_all > 1 else None
            }
            return out, aux
        return out


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.mlp_layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.multi_head_attention = BoundaryRelevanceMultiHeadAttention(
            embedding_dim, head_num,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, return_attention=False):
        if return_attention:
            _x, aux = self.multi_head_attention(x, return_attention=True)
            _x = self.dropout(_x)
            x = x + _x
            x = self.layer_norm1(x)

            _x = self.mlp(x)
            x = x + _x
            x = self.layer_norm2(x)
            return x, aux
        else:
            _x = self.multi_head_attention(x, return_attention=False)
            _x = self.dropout(_x)
            x = x + _x
            x = self.layer_norm1(x)

            _x = self.mlp(x)
            x = x + _x
            x = self.layer_norm2(x)
            return x


class TransformerEncoderBlock_tx(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.multi_head_attention = BoundaryRelevanceCrossAttention(
            embedding_dim, head_num,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, tx, return_attention=False):
        if return_attention:
            _x, aux = self.multi_head_attention(x, tx, return_attention=True)
            x = x + _x
            x = self.layer_norm1(x)

            _x = self.mlp(x)
            x = x + _x
            x = self.layer_norm2(x)
            return x, aux
        else:
            _x = self.multi_head_attention(x, tx, return_attention=False)
            x = x + _x
            x = self.layer_norm1(x)

            _x = self.mlp(x)
            x = x + _x
            x = self.layer_norm2(x)
            return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.layer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim, head_num, mlp_dim,
                boundary_lambda=boundary_lambda,
                relevance_lambda=relevance_lambda
            )
            for _ in range(block_num)
        ])

    def forward(self, x, return_attention=False):
        attn_list = []
        for layer_block in self.layer_blocks:
            if return_attention:
                x, aux = layer_block(x, return_attention=True)
                attn_list.append(aux)
            else:
                x = layer_block(x, return_attention=False)

        if return_attention:
            return x, attn_list
        return x


class TransformerEncoder_tx(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.layer_blocks = nn.ModuleList([
            TransformerEncoderBlock_tx(
                embedding_dim, head_num, mlp_dim,
                boundary_lambda=boundary_lambda,
                relevance_lambda=relevance_lambda
            )
            for _ in range(block_num)
        ])

    def forward(self, x, tx, return_attention=False):
        attn_list = []
        for layer_block in self.layer_blocks:
            if return_attention:
                x, aux = layer_block(x, tx, return_attention=True)
                attn_list.append(aux)
            else:
                x = layer_block(x, tx, return_attention=False)

        if return_attention:
            return x, attn_list
        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(
            embedding_dim, head_num, mlp_dim, block_num,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_attention=False):
        img_patches = rearrange(
            x,
            'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
            patch_x=self.patch_dim, patch_y=self.patch_dim
        )

        batch_size, tokens, _ = img_patches.shape
        project = self.projection(img_patches)

        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches = patches + self.embedding[:tokens + 1, :]
        patches = self.dropout(patches)

        if return_attention:
            x, attn_list = self.transformer(patches, return_attention=True)
        else:
            x = self.transformer(patches, return_attention=False)
            attn_list = None

        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        if return_attention:
            return x, attn_list
        return x


class ViT_tx(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1,
                 boundary_lambda=0.5, relevance_lambda=0.5):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.tprojection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.transformer = TransformerEncoder_tx(
            embedding_dim, head_num, mlp_dim, block_num,
            boundary_lambda=boundary_lambda,
            relevance_lambda=relevance_lambda
        )

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, trimap, return_attention=False):
        img_patches = rearrange(
            x,
            'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
            patch_x=self.patch_dim, patch_y=self.patch_dim
        )
        tri_patches = rearrange(
            trimap,
            'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
            patch_x=self.patch_dim, patch_y=self.patch_dim
        )

        batch_size, tokens, _ = img_patches.shape
        tbatch_size, ttokens, _ = tri_patches.shape

        project = self.projection(img_patches)
        tproject = self.tprojection(tri_patches)

        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)
        ttoken = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=tbatch_size)

        patches = torch.cat([token, project], dim=1)
        tpatches = torch.cat([ttoken, tproject], dim=1)

        patches = patches + self.embedding[:tokens + 1, :]
        tpatches = tpatches + self.embedding[:ttokens + 1, :]

        if return_attention:
            x, attn_list = self.transformer(patches, tpatches, return_attention=True)
        else:
            x = self.transformer(patches, tpatches, return_attention=False)
            attn_list = None

        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        if return_attention:
            return x, attn_list
        return x


if __name__ == '__main__':
    vit = ViT(
        img_dim=32,
        in_channels=3,
        patch_dim=16,
        embedding_dim=32,
        block_num=2,
        head_num=2,
        mlp_dim=128,
        classification=False,
        boundary_lambda=0.5,
        relevance_lambda=0.5
    ).cuda()

    feat = torch.rand(1, 3, 32, 32).cuda()
    out, attn_list = vit(feat, return_attention=True)
    print(out.shape)
    print(len(attn_list))
    print(attn_list[0]["attention"].shape)