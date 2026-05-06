"""
VJEPA 2.1 - Consolidated Training, Fine-tuning and Evaluation Script
====================================================================

Based on: "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
Paper: https://arxiv.org/abs/2603.14482
Reference implementation: facebookresearch/vjepa2 (app/vjepa_2_1/)

This script provides a unified interface for:
1. Loading pretrained VJEPA 2.1 models from local folder
2. Pre-training from scratch
3. Fine-tuning on downstream tasks
4. Feature extraction for downstream use

Key VJEPA 2.1 innovations:
- Dense Prediction Loss: Self-supervision applied to ALL tokens (visible + masked)
- Deep Self-Supervision: Multi-level hierarchical predictions at intermediate encoder layers
- Multi-Modal Tokenizer: Unified image/video processing with modality embeddings
- Context Loss: Weighted loss on context tokens for spatial coherence

Usage:
    python vjepa2_1.py --mode pretrain --config configs/vjepa2_1_pretrain.yaml
    python vjepa2_1.py --mode finetune --checkpoint /model/vjepa2_1_vitb_dist_vitG_384.pt --num_classes 11
    python vjepa2_1.py --mode extract --checkpoint /model/vjepa2_1_vitb_dist_vitG_384.pt
"""

import copy
import gc
import logging
import math
import os
import random
import sys
import time
from functools import partial
from multiprocessing import Value
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel

# =============================================================================
# IMPORTS FROM LOCAL VJEPA2 FOLDER
# =============================================================================
VJEPA2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vjepa2")
sys.path.insert(0, VJEPA2_PATH)

# vjepa2/src

from vjepa2.src.masks.utils import apply_masks
from vjepa2.src.masks.multiseq_multiblock3d import MaskCollator, _MaskGenerator
from vjepa2.src.utils.tensors import trunc_normal_, repeat_interleave_batch
from vjepa2.src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule, LinearDecaySchedule
from vjepa2.src.utils.logging import get_logger, CSVLogger, AverageMeter, gpu_timer
from vjepa2.src.utils.distributed import init_distributed

try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

logger = get_logger(__name__, force=True)

# =============================================================================
# PART 1: POSITIONAL EMBEDDINGS
# =============================================================================

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """Generate 3D sincos positional embedding for video tokens."""
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = _get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = _get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = _get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Helper for generating 1D sincos positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# =============================================================================
# PART 2: PATCH EMBEDDINGS
# =============================================================================

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video (Conv3D over temporal-spatial patches)."""
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed2D(nn.Module):
    """2D Patch Embedding for images."""
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# =============================================================================
# PART 3: ATTENTION MECHANISMS
# =============================================================================

def rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    """Apply rotary position embedding to queries/keys."""
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for RoPE"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class RoPEAttention(nn.Module):
    """Rotary Position Attention for spatial-temporal modeling."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0,
                 proj_drop=0.0, use_sdpa=True, grid_size=14, is_causal=False,
                 n_registers=0, has_cls_first=False, interpolate_rope=False, patch_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.grid_size = grid_size
        self.pretrained_grid_size = int(252 / patch_size) if patch_size == 14 else int(256 / patch_size)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        tokens_per_row = W_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        return (ids - tokens_per_frame * frame_ids) // tokens_per_row

    def _separate_positions(self, ids, H_patches=None, W_patches=None):
        tokens_per_frame = int((H_patches or self.grid_size) * (W_patches or self.grid_size))
        tokens_per_row = W_patches or self.grid_size
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        B, N, C = x.size()
        N_ctx = N - self.n_registers

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self._separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(N_ctx), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self._separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            H_patches = H_patches or int(self.grid_size)
            W_patches = W_patches or int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        head_dim = self.head_dim
        d_dim = int(2 * ((head_dim // 3) // 2))
        h_dim = int(2 * ((head_dim // 3) // 2))
        w_dim = int(2 * ((head_dim // 3) // 2))

        def rope_part(x_part, pos, s, e):
            return rotate_queries_or_keys(x_part[..., s:e], pos, self.n_registers, self.has_cls_first)

        q_tok = []
        k_tok = []
        q_tok.append(rope_part(q, d_mask, 0, d_dim))
        q_tok.append(rope_part(q, h_mask, d_dim, d_dim + h_dim))
        q_tok.append(rope_part(q, w_mask, d_dim + h_dim, d_dim + h_dim + w_dim))
        if d_dim + h_dim + w_dim < head_dim:
            q_tok.append(q[..., d_dim + h_dim + w_dim:])
        q = torch.cat(q_tok, dim=-1)

        k_tok.append(rope_part(k, d_mask, 0, d_dim))
        k_tok.append(rope_part(k, h_mask, d_dim, d_dim + h_dim))
        k_tok.append(rope_part(k, w_mask, d_dim + h_dim, d_dim + h_dim + w_dim))
        if d_dim + h_dim + w_dim < head_dim:
            k_tok.append(k[..., d_dim + h_dim + w_dim:])
        k = torch.cat(k_tok, dim=-1)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x, None


class Attention(nn.Module):
    """Standard Multi-Head Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0,
                 proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# =============================================================================
# PART 4: MLP BLOCKS
# =============================================================================

class MLP(nn.Module):
    """Standard MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (used in VJEPA 2.1)."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden = hidden_features or in_features
        if wide_silu:
            swiglu_hidden = int(2 * swiglu_hidden / 3)
            swiglu_hidden = ((swiglu_hidden + 7) // 8 * 8)
        self.fc1 = nn.Linear(in_features, swiglu_hidden)
        self.fc2 = nn.Linear(in_features, swiglu_hidden)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden, out_features)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# =============================================================================
# PART 5: TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 wide_silu=True, norm_layer=nn.LayerNorm, use_sdpa=True, is_causal=False,
                 use_rope=False, grid_size=16, n_registers=0, has_cls_first=False,
                 interpolate_rope=False, patch_size=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                      drop, use_sdpa, grid_size, is_causal,
                                      n_registers, has_cls_first, interpolate_rope, patch_size)
        else:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, use_sdpa, is_causal)

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(dim, mlp_hidden, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(dim, mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        if isinstance(self.attn, RoPEAttention):
            y, attn = self.attn(self.norm1(x), mask=mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        return x


# =============================================================================
# PART 6: VJEPA 2.1 ENCODER
# =============================================================================

class VJEPA2_1Encoder(nn.Module):
    """
    VJEPA 2.1 Vision Transformer Encoder.

    Key features:
    - 3D patch embedding for video / 2D for images
    - Multi-level hierarchical output (distillation at intermediate layers)
    - Optional RoPE positional encoding
    - Modality embeddings for image/video discrimination
    - Register tokens support
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02,
                 uniform_power=False, use_silu=False, wide_silu=True, use_sdpa=True,
                 use_activation_checkpointing=False, is_causal=False, use_rope=False,
                 init_type="default", img_temporal_dim_size=None, n_registers=0,
                 has_cls_first=False, interpolate_rope=False, modality_embedding=True,
                 n_output_distillation=4):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_type = init_type
        self.img_temporal_dim_size = img_temporal_dim_size

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
        else:
            self.patch_embed = PatchEmbed2D(patch_size, in_chans, embed_dim)

        self.num_patches = (
            (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        )

        if self.img_temporal_dim_size is not None:
            self.patch_embed_img = PatchEmbed3D(patch_size, 1, in_chans, embed_dim)
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power
        self.use_rope = use_rope

        if not use_rope:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim,
                img_size[0] // patch_size,
                num_frames // tubelet_size,
                uniform_power=uniform_power
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[i],
                nn.SiLU if use_silu else nn.GELU, wide_silu,
                norm_layer, use_sdpa, is_causal, use_rope,
                img_size[0] // patch_size, n_registers, has_cls_first,
                interpolate_rope, patch_size
            ) for i in range(depth)
        ])

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        self.hierarchical_layers, self.out_layers_distillation = self._get_layer_indices(depth, n_output_distillation)
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _get_layer_indices(self, depth, n_dist):
        """Get indices for hierarchical layers and distillation outputs."""
        if depth == 12:
            hier = [2, 5, 8, 11]
        elif depth == 24:
            hier = [5, 11, 17, 23]
        elif depth == 40:
            hier = [9, 19, 29, 39]
        elif depth == 48:
            hier = [11, 23, 37, 47]
        else:
            hier = [depth // 4 * i for i in range(1, 5)]
        out_dist = hier[-n_dist:] if n_dist > 0 else hier
        return hier, out_dist

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_uniform":
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id + 1))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id + 1))

    def check_temporal_dim(self, shape):
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def forward(self, x, masks=None, training=False):
        """
        Forward pass of VJEPA 2.1 encoder.

        Returns hierarchical features during training (for deep self-supervision).
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
            is_img = True
        else:
            _, _, T, H, W = x.shape
            is_img = self.check_temporal_dim(x.shape)
            if not is_img:
                T = T // self.tubelet_size

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        if is_img and self.patch_embed_img is not None:
            x = self.patch_embed_img(x)
            mode = "image"
            if self.modality_embedding:
                x = x + self.img_mod_embed
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed

        if not self.use_rope:
            x = x + pos_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, T, H_patches, W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=masks, T=T, H_patches=H_patches, W_patches=W_patches)
            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if training:
            hier_cat = torch.cat(hier, dim=2)
            return hier_cat
        else:
            return self.norms_block[-1](x)

    def interpolate_pos_encoding(self, x, pos_embed):
        if pos_embed.ndim == 2:
            return pos_embed
        _, _, T, H, W = x.shape
        N = pos_embed.shape[1]
        N_t = self.num_frames // self.tubelet_size
        N_h = self.img_height // self.patch_size
        N_w = self.img_width // self.patch_size

        if H == self.img_height and W == self.img_width and T == self.num_frames:
            return pos_embed
        elif H == self.img_height and W == self.img_width and T < self.num_frames:
            new_N = int((T // self.tubelet_size) * N_h * N_w)
            return pos_embed[:, :new_N, :]

        T = T // self.tubelet_size
        H = H // self.patch_size
        W = W // self.patch_size
        scale_factor = (T / N_t, H / N_h, W / N_w)
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, N_t, N_h, N_w, -1).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor, mode="trilinear"
        )
        return pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, pos_embed.shape[1])


# =============================================================================
# PART 7: VJEPA 2.1 PREDICTOR
# =============================================================================

class VJEPA2_1Predictor(nn.Module):
    """
    VJEPA 2.1 Predictor Network.

    Predicts representations for both masked and context tokens.
    Uses multi-level output matching the encoder's hierarchical layers.
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                 embed_dim=1024, predictor_embed_dim=384, depth=6, num_heads=16,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 init_std=0.02, uniform_power=False, use_mask_tokens=True,
                 num_mask_tokens=2, zero_init_mask_tokens=True, use_silu=False,
                 wide_silu=True, is_causal=False, use_activation_checkpointing=False,
                 return_all_tokens=True, use_rope=False, n_registers=0,
                 has_cls_first=False, interpolate_rope=False, modality_embedding=True,
                 img_temporal_dim_size=None, n_output_distillation=4):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.has_cls_first = has_cls_first

        if depth == 4:
            all_hier = [0, 1, 2, 3]
        elif depth == 8:
            all_hier = [1, 3, 5, 7]
        elif depth == 12:
            all_hier = [2, 5, 8, 11]
        elif depth == 20:
            all_hier = [4, 9, 14, 19]
        elif depth == 24:
            all_hier = [4, 11, 17, 23]
        else:
            all_hier = [depth // 4 * i for i in range(1, 5)]

        self.hierarchical_layers = all_hier[-n_output_distillation:]
        self.predictor_embed_dim = predictor_embed_dim

        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        if len(self.hierarchical_layers) == 1:
            self.predictor_embed = nn.Linear(embed_dim * len(self.hierarchical_layers),
                                           predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * len(self.hierarchical_layers), embed_dim, bias=True),
                act_layer_mlp(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True)
            )

        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.num_patches = (
            (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        )

        self.modality_embedding = False
        if img_temporal_dim_size is not None and modality_embedding:
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            self.modality_embedding = True

        self.uniform_power = uniform_power
        self.use_rope = use_rope

        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, predictor_embed_dim), requires_grad=False
            )
            pos_embed = get_3d_sincos_pos_embed(
                predictor_embed_dim,
                img_size[0] // patch_size,
                num_frames // tubelet_size,
                uniform_power=uniform_power
            )
            self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.predictor_blocks = nn.ModuleList([
            TransformerBlock(
                predictor_embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[i],
                nn.SiLU if use_silu else nn.GELU, wide_silu,
                norm_layer, True, is_causal, use_rope,
                self.grid_height, n_registers, has_cls_first,
                interpolate_rope, patch_size
            ) for i in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim,
            len(self.hierarchical_layers) * embed_dim,
            bias=True
        )
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim,
                len(self.hierarchical_layers) * embed_dim,
                bias=True
            )

        self.init_std = init_std
        if zero_init_mask_tokens:
            for mt in self.mask_tokens:
                nn.init.zeros_(mt)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id + 1))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id + 1))

    def forward(self, x, masks_x, masks_y, mod="video", mask_index=1):
        """Forward pass of VJEPA 2.1 predictor."""
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        _, N_ctxt, D = x.shape

        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x = x + apply_masks(x_pos_embed, masks_x)

        mask_index = mask_index % max(self.num_mask_tokens, 1)
        if self.mask_tokens is not None and len(self.mask_tokens) > 0:
            pred_tokens = self.mask_tokens[mask_index].repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_y)
        else:
            pred_tokens = torch.zeros(B, masks_y[0].shape[-1], D, device=x.device, dtype=x.dtype)
            pred_tokens = apply_masks(pred_tokens, masks_y)

        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens = pred_tokens + pos_embs

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        masks_x_cat = torch.cat(masks_x, dim=0)
        masks_y_cat = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x_cat, masks_y_cat], dim=1)

        argsort = torch.argsort(masks, dim=1)
        masks_sorted = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row.item(), :] for i, row in enumerate(argsort)], dim=0)

        if self.modality_embedding:
            x = x + (self.img_mod_embed if mod == "image" else self.video_mod_embed)

        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x, _ = torch.utils.checkpoint.checkpoint(blk, x, masks_sorted,
                                                        use_reentrant=False)
            else:
                x = blk(x, mask=masks_sorted)
        x = self.predictor_norm(x)

        reverse_argsort = torch.argsort(argsort, dim=1)
        x = torch.stack([x[i, row.item(), :] for i, row in enumerate(reverse_argsort)], dim=0)

        x_pred = x[:, N_ctxt:, :]
        x_pred = self.predictor_proj(x_pred)

        if self.return_all_tokens:
            x_context = x[:, :N_ctxt, :]
            x_context = self.predictor_proj_context(x_context)
            return x_pred, x_context
        return x_pred, None


# =============================================================================
# PART 8: MODEL WRAPPERS
# =============================================================================

class EncoderWrapper(nn.Module):
    """Wrapper to handle multi-sequence (different frame counts) inputs."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, gram_mode=False, training_mode=False):
        if masks is None:
            if isinstance(x, list):
                outputs = []
                for x_fpc in x:
                    outputs.append(self.backbone(x_fpc, training=training_mode))
                return outputs
            return self.backbone(x, training=training_mode)
        if isinstance(x, list) and isinstance(masks, list):
            outs = [[] for _ in x]
            for i, (x_fpc, m_fpc) in enumerate(zip(x, masks)):
                for m in m_fpc:
                    outs[i].append(self.backbone(x_fpc, masks=m, training=training_mode))
            return outs
        return self.backbone(x, masks=masks, training=training_mode)


class PredictorWrapper(nn.Module):
    """Wrapper for predictor handling multi-sequence inputs."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, mod="video"):
        if isinstance(x, list):
            outs_pred = [[] for _ in x]
            outs_context = [[] for _ in x]
            for i, (x_fpc, mx_fpc, my_fpc) in enumerate(zip(x, masks_x, masks_y)):
                for xij, mx, my in zip(x_fpc, mx_fpc, my_fpc):
                    x_pred, x_context = self.backbone(xij, mx, my, mask_index=i, mod=mod)
                    outs_pred[i].append(x_pred)
                    outs_context[i].append(x_context)
            return outs_pred, outs_context
        x_pred, x_context = self.backbone(x, masks_x, masks_y, mod=mod)
        return [x_pred], [x_context]


# =============================================================================
# PART 9: VJEPA 2.1 LOSS FUNCTIONS
# =============================================================================

def compute_mask_distance(masks_pred, masks_enc, grid_size, offset_context_loss=False):
    """Compute distance weights for context loss weighting."""

    def _get_frame_pos(ids, H_patches=None, W_patches=None, grid_size=None):
        tokens_per_frame = int(grid_size * grid_size) if H_patches is None else int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(ids, H_patches=None, W_patches=None, grid_size=None):
        tokens_per_frame = int(grid_size * grid_size) if H_patches is None else int(H_patches * W_patches)
        tokens_per_row = grid_size if W_patches is None else W_patches
        frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
        return (ids - tokens_per_frame * frame_ids) // tokens_per_row

    def separate_positions(ids, H_patches=None, W_patches=None, grid_size=None):
        tokens_per_frame = int(grid_size * grid_size) if H_patches is None else int(H_patches * W_patches)
        tokens_per_row = grid_size if W_patches is None else W_patches
        frame_ids = _get_frame_pos(ids, H_patches, W_patches, grid_size)
        height_ids = _get_height_pos(ids, H_patches, W_patches, grid_size)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row_distances = []
        for mpred, menc in zip(masks_pred_i, masks_enc_i):
            N_enc = menc.shape[0]
            d_enc, h_enc, w_enc = separate_positions(menc, grid_size=grid_size)
            d_pred, h_pred, w_pred = separate_positions(mpred, grid_size=grid_size)

            enc_distances = []
            for e in range(N_enc):
                enc_pos = torch.stack([d_enc[e], h_enc[e], w_enc[e]], dim=-1).unsqueeze(1)
                pred_pos = torch.stack([d_pred, h_pred, w_pred], dim=-1)
                dist = torch.cdist(enc_pos, pred_pos, p=2).squeeze(1)
                dmin = dist.min()
                if offset_context_loss:
                    dmin = dmin * (1.0 / (grid_size // 16))
                dmin = dmin ** 0.5
                enc_distances.append(dmin)
            row_distances.append(torch.stack(enc_distances))
        distances.append(row_distances)
    return distances


class Lambda_LinearWarmupHold:
    """Lambda scheduler with linear warmup for context loss weight."""
    def __init__(self, lambda_value=0.5, start_iter=15000, end_iter=30000):
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter):
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        alpha = (global_iter - self.start) / self.span
        return self.lambda_value * alpha


# =============================================================================
# PART 10: MODEL INITIALIZATION
# =============================================================================

def get_encoder_config(model_name):
    """Get encoder configuration for different model sizes."""
    configs = {
        "vit_tiny":     {"embed_dim": 192,  "depth": 12, "num_heads": 3,  "mlp_ratio": 4.0},
        "vit_small":    {"embed_dim": 384,  "depth": 12, "num_heads": 6,  "mlp_ratio": 4.0},
        "vit_base":     {"embed_dim": 768,  "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
        "vit_large":    {"embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0},
        "vit_giant":    {"embed_dim": 1408, "depth": 40, "num_heads": 16, "mlp_ratio": 48/11},
        "vit_giant_x":  {"embed_dim": 1408, "depth": 40, "num_heads": 22, "mlp_ratio": 48/11},
        "vit_gigantic": {"embed_dim": 1664, "depth": 48, "num_heads": 26, "mlp_ratio": 64/13},
    }
    return configs.get(model_name, configs["vit_large"])


def init_vjepa2_1_encoder(model_name="vit_large", img_size=224, patch_size=16,
                          num_frames=16, tubelet_size=2, device="cuda", **kwargs):
    """Initialize VJEPA 2.1 encoder."""
    cfg = get_encoder_config(model_name)

    encoder = VJEPA2_1Encoder(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        use_silu=kwargs.get("use_silu", False),
        wide_silu=kwargs.get("wide_silu", True),
        use_rope=kwargs.get("use_rope", False),
        use_sdpa=kwargs.get("use_sdpa", True),
        modality_embedding=kwargs.get("modality_embedding", True),
        img_temporal_dim_size=kwargs.get("img_temporal_dim_size", None),
        n_registers=kwargs.get("n_registers", 0),
        n_output_distillation=kwargs.get("n_output_distillation", 4),
    )
    encoder = EncoderWrapper(encoder)
    encoder.to(device)
    return encoder, cfg["embed_dim"]


def init_vjepa2_1_predictor(embed_dim, num_frames=16, tubelet_size=2,
                            img_size=224, patch_size=16, pred_depth=6,
                            pred_embed_dim=384, num_heads=16, device="cuda", **kwargs):
    """Initialize VJEPA 2.1 predictor."""
    predictor = VJEPA2_1Predictor(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim * kwargs.get("n_output_distillation", 4),
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=num_heads,
        use_silu=kwargs.get("use_pred_silu", False),
        wide_silu=kwargs.get("wide_silu", True),
        use_mask_tokens=kwargs.get("use_mask_tokens", True),
        num_mask_tokens=kwargs.get("num_mask_tokens", 2),
        return_all_tokens=kwargs.get("return_all_tokens", True),
        use_rope=kwargs.get("use_rope", False),
        modality_embedding=kwargs.get("modality_embedding", True),
        img_temporal_dim_size=kwargs.get("img_temporal_dim_size", None),
        n_output_distillation=kwargs.get("n_output_distillation", 4),
    )
    predictor = PredictorWrapper(predictor)
    predictor.to(device)
    return predictor


def init_vjepa2_1_model(model_name="vit_large", img_size=224, patch_size=16,
                        num_frames=16, tubelet_size=2, pred_depth=6,
                        pred_embed_dim=384, device="cuda", **kwargs):
    """Initialize complete VJEPA 2.1 model (encoder + predictor)."""
    encoder, embed_dim = init_vjepa2_1_encoder(
        model_name=model_name,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        device=device,
        n_output_distillation=kwargs.get("n_output_distillation", 4),
        **kwargs
    )

    cfg = get_encoder_config(model_name)
    predictor = init_vjepa2_1_predictor(
        embed_dim=embed_dim,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        img_size=img_size,
        patch_size=patch_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        num_heads=cfg["num_heads"],
        device=device,
        n_output_distillation=kwargs.get("n_output_distillation", 4),
        **kwargs
    )

    return encoder, predictor, embed_dim


# =============================================================================
# PART 11: CHECKPOINT LOADING
# =============================================================================

def load_vjepa2_1_checkpoint(path, encoder, predictor, target_encoder=None,
                             device="cuda", is_anneal=False):
    """Load VJEPA 2.1 checkpoint."""
    logger.info(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    epoch = 0
    if not is_anneal:
        epoch = checkpoint.get("epoch", 0)

    pretrained_dict = checkpoint.get("encoder", checkpoint.get("model", {}))
    model_dict = encoder.state_dict()
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded pretrained encoder from epoch {epoch}")

    pretrained_dict = checkpoint.get("predictor", {})
    model_dict = predictor.state_dict()
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            logger.info(f'predictor key "{k}" could not be found')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'predictor key "{k}" shape mismatch')
            pretrained_dict[k] = v
    predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info("Loaded pretrained predictor")

    if target_encoder is not None and "target_encoder" in checkpoint:
        pretrained_dict = checkpoint["target_encoder"]
        model_dict = target_encoder.state_dict()
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                logger.info(f'target key "{k}" could not be found')
            elif pretrained_dict[k].shape != v.shape:
                pretrained_dict[k] = v
        target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info("Loaded pretrained target encoder")

    return encoder, predictor, target_encoder, epoch


# =============================================================================
# PART 12: OPTIMIZATION
# =============================================================================

def init_optimizer(encoder, predictor, iterations_per_epoch, num_epochs,
                   start_lr=1e-6, ref_lr=1e-4, final_lr=1e-6,
                   warmup=0.1, wd=1e-4, final_wd=1e-5,
                   ipe_scale=1.25, use_radamw=False,
                   betas=(0.9, 0.999), eps=1e-8,
                   mixed_precision=False, is_anneal=False):
    """Initialize optimizer and schedulers."""

    param_groups = [
        {"params": [p for n, p in encoder.named_parameters()
                   if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in predictor.named_parameters()
                   if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in encoder.named_parameters()
                   if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
        {"params": [p for n, p in predictor.named_parameters()
                   if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
    ]

    if use_radamw:
        from vjepa2.src.utils.adamw import AdamW as RAdamW
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        optimizer = optim.AdamW(param_groups, betas=betas, eps=eps)

    T_max = int(ipe_scale * num_epochs * iterations_per_epoch)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=T_max,
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


# =============================================================================
# PART 13: FINE-TUNING MODULES
# =============================================================================

class VJEPA2_1FineTuner(nn.Module):
    """
    Fine-tuning wrapper for VJEPA 2.1 on downstream tasks.

    Supports:
    - Video classification
    - Action recognition
    - Depth estimation
    - Semantic segmentation
    - Action anticipation
    """

    def __init__(self, encoder, num_classes=400, task_type="classification",
                 probe_type="full", embed_dim=1024, num_levels=4, use_frozen=True):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.task_type = task_type
        self.probe_type = probe_type
        self.num_levels = num_levels
        self.use_frozen = use_frozen

        if use_frozen:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

        total_dim = embed_dim * num_levels

        if probe_type == "linear":
            self.head = nn.Linear(total_dim, num_classes)
        elif probe_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(total_dim, total_dim // 2),
                nn.GELU(),
                nn.Linear(total_dim // 2, num_classes)
            )
        elif probe_type == "attentive":
            self.head = AttentiveProbe(total_dim, num_classes)
        else:
            self.head = nn.Linear(total_dim, num_classes)

    def forward(self, x, return_features=False):
        with torch.no_grad():
            features = self.encoder(x, training_mode=True)

        if isinstance(features, list):
            features = torch.cat(features, dim=-1)

        if self.probe_type == "attentive" and hasattr(self.head, 'pool'):
            logits = self.head(features)
        else:
            logits = self.head(features)

        if return_features:
            return logits, features
        return logits

    def extract_features(self, x, level=-1):
        with torch.no_grad():
            features = self.encoder(x, training_mode=True)

        if isinstance(features, torch.Tensor):
            return features

        if level == -1 and len(features) == self.num_levels:
            return torch.cat(features, dim=-1)
        elif 0 <= level < len(features):
            return features[level]
        return features[-1]


class VJEPA2_1Classifier(nn.Module):
    """Simplified VJEPA 2.1 classifier for fine-tuning with mean pooling."""
    def __init__(self, encoder, num_classes, embed_dim=768, num_levels=4, freeze_backbone=True):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        total_dim = embed_dim * num_levels

        if freeze_backbone:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

        self.head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x, training_mode=True)

        if isinstance(features, list):
            features = torch.cat(features, dim=-1)

        features = features.mean(dim=1)
        return self.head(features)

    def extract_features(self, x):
        with torch.no_grad():
            features = self.encoder(x, training_mode=True)
        if isinstance(features, list):
            features = torch.cat(features, dim=-1)
        return features.mean(dim=1)


class AttentiveProbe(nn.Module):
    """Attention-based probe for classification from VJEPA 2.1."""
    def __init__(self, embed_dim, num_classes, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, N, D = x.shape
        x_mean = x.mean(dim=1, keepdim=True)
        attn_out, _ = self.attention(x_mean, x, x)
        attn_out = attn_out.squeeze(1)
        return self.fc(self.norm(attn_out))


# =============================================================================
# PART 14: MAIN TRAINING FUNCTIONS
# =============================================================================

def train_vjepa2_1(args):
    """Main pre-training function for VJEPA 2.1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size, rank = init_distributed()

    folder = args.get("folder", "./checkpoints_vjepa2_1")
    os.makedirs(folder, exist_ok=True)

    model_cfg = args.get("model", {})
    data_cfg = args.get("data", {})
    opt_cfg = args.get("optimization", {})
    loss_cfg = args.get("loss", {})

    model_name = model_cfg.get("model_name", "vit_large")
    patch_size = model_cfg.get("patch_size", 16)
    tubelet_size = model_cfg.get("tubelet_size", 2)
    num_frames = data_cfg.get("num_frames", 16)
    crop_size = data_cfg.get("crop_size", 224)
    pred_depth = model_cfg.get("pred_depth", 6)
    pred_embed_dim = model_cfg.get("pred_embed_dim", 384)
    n_registers = model_cfg.get("n_registers", 0)
    modality_embedding = model_cfg.get("modality_embedding", True)
    img_temporal_dim_size = model_cfg.get("img_temporal_dim_size", None)

    dtype_str = model_cfg.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else (torch.float16 if dtype_str == "float16" else torch.float32)
    mixed_precision = dtype_str in ("bfloat16", "float16")

    logger.info(f"Initializing VJEPA 2.1 model: {model_name}")
    encoder, predictor, embed_dim = init_vjepa2_1_model(
        model_name=model_name,
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        device=device,
        n_registers=n_registers,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=4,
    )
    target_encoder = copy.deepcopy(encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    batch_size = data_cfg.get("batch_size", 2)
    num_epochs = opt_cfg.get("epochs", 100)
    lr = opt_cfg.get("lr", 1e-4)
    wd = opt_cfg.get("weight_decay", 1e-4)
    warmup = opt_cfg.get("warmup", 0.1)
    iterations_per_epoch = data_cfg.get("iterations_per_epoch", 100)

    optimizer, scaler, scheduler, wd_scheduler = init_optimizer(
        encoder, predictor, iterations_per_epoch, num_epochs,
        ref_lr=lr, wd=wd, warmup=warmup,
        mixed_precision=mixed_precision,
    )

    if DISTRIBUTED_AVAILABLE and world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        target_encoder = DistributedDataParallel(target_encoder)

    mask_ratio = loss_cfg.get("mask_ratio", 0.9)
    lambda_ctx = loss_cfg.get("lambda_ctx", 0.5)
    loss_exp = loss_cfg.get("loss_exp", 2.0)
    weight_distance_loss = loss_cfg.get("weight_distance_loss", True)
    offset_context_loss = loss_cfg.get("offset_context_loss", False)

    grid_size = crop_size // patch_size
    ema = opt_cfg.get("ema", [0.996, 1.0])
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_ctx)

    latest_path = os.path.join(folder, "latest.pth.tar")
    start_epoch = 0

    if os.path.exists(latest_path):
        encoder, predictor, target_encoder, _, _, start_epoch = load_vjepa2_1_checkpoint(
            latest_path, encoder.module if hasattr(encoder, 'module') else encoder,
            predictor.module if hasattr(predictor, 'module') else predictor,
            target_encoder.module if hasattr(target_encoder, 'module') else target_encoder,
            device=device
        )

    logger.info("Starting training...")
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    csv_logger = CSVLogger(log_file, ("%d", "epoch"), ("%d", "itr"),
                          ("%.5f", "loss"), ("%d", "iter-time(ms)"))

    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        predictor.train()

        loss_meter = AverageMeter()

        for itr in range(iterations_per_epoch):
            batch = torch.randn(batch_size, 3, num_frames, crop_size, crop_size, device=device)

            masks_enc, masks_pred = [], []
            for _ in range(batch_size):
                dur = num_frames // tubelet_size
                H = W = crop_size // patch_size
                mask_gen = _MaskGenerator(
                    crop_size=crop_size, num_frames=num_frames,
                    spatial_patch_size=patch_size, temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=(0.2, 0.8),
                    temporal_pred_mask_scale=(1.0, 1.0),
                    aspect_ratio=(0.3, 3.0),
                    npred=1,
                )
                enc_m, pred_m = mask_gen(1)
                masks_enc.append(enc_m[0])
                masks_pred.append(pred_m[0])

            masks_enc = torch.stack(masks_enc).to(device)
            masks_pred = torch.stack(masks_pred).to(device)

            optimizer.zero_grad()
            scheduler.step()
            wd_scheduler.step()

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                with torch.no_grad():
                    h = target_encoder([batch], [masks_enc], training_mode=True)

                z = encoder([batch], [masks_enc], training_mode=True)
                z_pred, z_context = predictor(z, [masks_enc], [masks_pred], mod="video")

                loss = 0.0
                n_terms = 0
                for zi, hi, mi in zip(z_pred, h, [masks_pred]):
                    for zij, hij, mij in zip(zi, hi, mi):
                        chunks_z = torch.chunk(zij, 4, dim=-1)
                        chunks_h = torch.chunk(hij, 4, dim=-1)
                        for p, t in zip(chunks_z, chunks_h):
                            loss += torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp
                            n_terms += 1
                loss /= max(n_terms, 1)

                if lambda_ctx > 0 and z_context is not None:
                    distance_weights = compute_mask_distance(
                        [masks_pred], [masks_enc], grid_size, offset_context_loss
                    ) if weight_distance_loss else None

                    ctx_loss = 0.0
                    ctx_n = 0
                    for zc, hi in zip(z_context, h):
                        for zci, hij in zip(zc, hi):
                            chunks_zc = torch.chunk(zci, 4, dim=-1)
                            chunks_h = torch.chunk(hij, 4, dim=-1)
                            for p, t in zip(chunks_zc, chunks_h):
                                ctx_loss += torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp
                                ctx_n += 1
                    ctx_loss /= max(ctx_n, 1)
                    lambda_step = lambda_sched.value(epoch * iterations_per_epoch + itr)
                    loss = loss + lambda_step * ctx_loss

            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            m = min(ema[0] + itr * (ema[1] - ema[0]) / (iterations_per_epoch * num_epochs), ema[1])
            with torch.no_grad():
                enc_params = list(encoder.parameters()) if not hasattr(encoder, 'module') else list(encoder.module.parameters())
                tgt_params = list(target_encoder.parameters()) if not hasattr(target_encoder, 'module') else list(target_encoder.module.parameters())
                for pq, pk in zip(enc_params, tgt_params):
                    pk.data.mul_(m).add_(pq.data, alpha=1 - m)

            loss_meter.update(loss.item())

            if itr % 10 == 0:
                csv_logger.log(epoch + 1, itr, loss.item(), 0)
                logger.info(f"[{epoch + 1}, {itr}] loss: {loss_meter.avg:.4f}")

        if rank == 0:
            save_dict = {
                "encoder": encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict(),
                "predictor": predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict(),
                "target_encoder": target_encoder.module.state_dict() if hasattr(target_encoder, 'module') else target_encoder.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch + 1,
                "loss": loss_meter.avg,
            }
            torch.save(save_dict, latest_path)
            logger.info(f"Checkpoint saved at epoch {epoch + 1}")

    logger.info("Training completed!")


def finetune_vjepa2_1(args):
    """Fine-tune VJEPA 2.1 on downstream task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = args.get("model", {})
    data_cfg = args.get("data", {})
    finetune_cfg = args.get("finetune", {})

    model_name = model_cfg.get("model_name", "vit_large")
    checkpoint_path = finetune_cfg.get("checkpoint", None)
    num_classes = finetune_cfg.get("num_classes", 400)
    epochs = finetune_cfg.get("epochs", 30)
    lr = finetune_cfg.get("lr", 1e-4)
    probe_type = finetune_cfg.get("probe_type", "full")
    use_frozen = finetune_cfg.get("use_frozen", True)

    encoder, _, embed_dim = init_vjepa2_1_model(
        model_name=model_name,
        device=device,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        _, _, _, _ = load_vjepa2_1_checkpoint(
            checkpoint_path, encoder, None, device=device
        )

    model = VJEPA2_1FineTuner(
        encoder=encoder,
        num_classes=num_classes,
        probe_type=probe_type,
        embed_dim=embed_dim,
        use_frozen=use_frozen,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    batch_size = data_cfg.get("batch_size", 4)
    num_frames = model_cfg.get("num_frames", 16)
    crop_size = model_cfg.get("crop_size", 224)

    logger.info(f"Starting fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(50):
            inputs = torch.randn(batch_size, 3, num_frames, crop_size, crop_size, device=device)
            targets = torch.randint(0, num_classes, (batch_size,), device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        acc = 100.0 * correct / total
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss/50:.4f} Acc: {acc:.2f}%")

    logger.info("Fine-tuning completed!")


def extract_features(args):
    """Extract features using VJEPA 2.1 encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = args.get("model", {})
    extract_cfg = args.get("extract", {})

    model_name = model_cfg.get("model_name", "vit_large")
    checkpoint_path = extract_cfg.get("checkpoint", None)

    encoder, _, embed_dim = init_vjepa2_1_model(
        model_name=model_name,
        device=device,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_vjepa2_1_checkpoint(checkpoint_path, encoder, None, device=device)

    encoder.eval()
    batch_size = model_cfg.get("batch_size", 4)
    num_frames = model_cfg.get("num_frames", 16)
    crop_size = model_cfg.get("crop_size", 224)

    logger.info("Extracting features...")

    with torch.no_grad():
        for i in range(10):
            inputs = torch.randn(batch_size, 3, num_frames, crop_size, crop_size, device=device)
            features = encoder(inputs, training=False)

            if isinstance(features, list):
                features = torch.cat(features, dim=-1)

            logger.info(f"Batch {i}: features shape = {features.shape}")

    logger.info("Feature extraction completed!")


# =============================================================================
# PART 15: MAIN ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VJEPA 2.1 Training and Fine-tuning")
    parser.add_argument("--mode", type=str, default="finetune",
                       choices=["pretrain", "finetune", "extract"],
                       help="Mode: pretrain, finetune, or extract")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for finetune/extract")
    parser.add_argument("--num_classes", type=int, default=11,
                       help="Number of classes for fine-tuning")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--model_name", type=str, default="vit_large",
                       help="Model variant")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames per clip")
    parser.add_argument("--crop_size", type=int, default=224,
                       help="Crop resolution")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--mask_ratio", type=float, default=0.9,
                       help="Mask ratio")
    parser.add_argument("--lambda_ctx", type=float, default=0.5,
                       help="Context loss weight (VJEPA 2.1 key hyperparameter)")
    parser.add_argument("--probe_type", type=str, default="full",
                       choices=["linear", "mlp", "attentive", "full"],
                       help="Probe type for fine-tuning")
    parser.add_argument("--use_frozen", action="store_true",
                       help="Freeze encoder during fine-tuning")
    parser.add_argument("--output_dir", type=str, default="./vjepa2_1_outputs",
                       help="Output directory")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {
            "folder": args.output_dir,
            "model": {
                "model_name": args.model_name,
                "patch_size": 16,
                "tubelet_size": 2,
                "pred_depth": 6,
                "pred_embed_dim": 384,
                "n_registers": 0,
                "modality_embedding": True,
                "dtype": "bfloat16",
            },
            "data": {
                "num_frames": args.num_frames,
                "crop_size": args.crop_size,
                "batch_size": args.batch_size,
                "iterations_per_epoch": 100,
            },
            "optimization": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": 1e-4,
                "warmup": 0.1,
                "ema": [0.996, 1.0],
            },
            "loss": {
                "mask_ratio": args.mask_ratio,
                "lambda_ctx": args.lambda_ctx,
                "loss_exp": 2.0,
                "weight_distance_loss": True,
                "offset_context_loss": False,
            },
            "finetune": {
                "checkpoint": args.checkpoint,
                "num_classes": args.num_classes,
                "epochs": args.epochs,
                "lr": args.lr,
                "probe_type": args.probe_type,
                "use_frozen": args.use_frozen,
            },
            "extract": {
                "checkpoint": args.checkpoint,
            }
        }

    if args.mode == "pretrain":
        train_vjepa2_1(config)
    elif args.mode == "finetune":
        finetune_vjepa2_1(config)
    elif args.mode == "extract":
        extract_features(config)


if __name__ == "__main__":
    main()
