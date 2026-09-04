"""Token mixing replacements for QKV self-attention in SDFormerFlow.

Each mixer replaces Spiking_BN_WindowAttention3D with the same interface:
  Input:  [T, B_, H, W, C]  where H,W are within-window spatial dims
  Output: [T, B_, H, W, C]  same shape

All mixers are hardware-friendly: use only accumulate (AC) operations
with binary spike inputs, avoiding softmax and multiply-accumulate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TM0: Identity (no mixing, just projection) ────────────────────────

class IdentityTokenMixer(nn.Module):
    """Baseline: remove attention entirely, keep only projection.

    Tests whether QKV attention is truly redundant. If this works
    (reasonable accuracy), attention is confirmed unnecessary.
    """
    def __init__(self, dim, window_size, pretrained_window_size, num_heads,
                 version="swinv1", qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., norm=None, **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        spiking_kwargs['num_steps'] = window_size[0]
        self.norm_layer = norm

        self.proj = nn.Linear(dim, dim)
        from models.STSwinNet_SNN.Spiking_modules import SpikingNormLayer, Spiking_neuron
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.proj_bn = SpikingNormLayer(dim, window_size[0], norm,
                                            spiking_kwargs.get('v_th', 0.1))
        self.proj_sn = Spiking_neuron(**spiking_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        T, B_, H, W, C = x.shape
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.proj_sn(x)
        x = x.reshape(B_, H * W, C)
        x = self.proj_drop(x)
        # Return as tuple to match attention interface (attn_windows, attn_score)
        return x, None

    def flops(self, N):
        flops = 0
        flops += N * self.dim * self.dim  # proj linear
        return flops


# ── TM1: Conv-based token mixing ──────────────────────────────────────

class ConvTokenMixer(nn.Module):
    """ConvMixer-style: depthwise conv for spatial mixing + pointwise.

    Replaces QKV with:
      1. Reshape [T,B_,H,W,C] → [T*B_, C, H, W]
      2. Depthwise Conv (spatial mixing across tokens)
      3. Pointwise Conv (channel mixing)
      4. Reshape back

    Hardware: depthwise conv with binary inputs = addition-only.
    """

    def __init__(self, dim, window_size, pretrained_window_size, num_heads,
                 version="swinv1", qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., norm=None, kernel_size=3, **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        spiking_kwargs['num_steps'] = window_size[0]
        self.norm_layer = norm

        from models.STSwinNet_SNN.Spiking_modules import SpikingNormLayer, Spiking_neuron

        # Spatial mixing: depthwise conv over tokens (within window)
        self.spatial_mix = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=False
        )
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.spatial_bn = SpikingNormLayer(dim, window_size[0], norm,
                                               spiking_kwargs.get('v_th', 0.1))
        self.spatial_sn = Spiking_neuron(**spiking_kwargs)

        # Channel mixing: 1x1 conv (equivalent to Linear per token)
        self.channel_mix = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.channel_bn = SpikingNormLayer(dim, window_size[0], norm,
                                               spiking_kwargs.get('v_th', 0.1))
        self.channel_sn = Spiking_neuron(**spiking_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        T, B_, H, W, C = x.shape
        # x: [T, B_, H, W, C] → [T*B_, C, H, W]
        x = x.permute(0, 1, 4, 2, 3).reshape(T * B_, C, H, W)

        # Spatial mixing
        x = self.spatial_mix(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.spatial_bn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.spatial_sn(x)

        # Channel mixing
        x = self.channel_mix(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.channel_bn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.channel_sn(x)

        # Reshape back: [T*B_, C, H, W] → [T, B_, H, W, C]
        x = x.reshape(T, B_, C, H, W).permute(0, 1, 3, 4, 2)
        x = x.reshape(B_, H * W, C)
        x = self.proj_drop(x)
        return x, None

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 9  # depthwise conv (3x3)
        flops += N * self.dim * self.dim  # 1x1 conv
        return flops


# ── TM2: MLP-Mixer style token mixing ─────────────────────────────────

class MLPTokenMixer(nn.Module):
    """MLP-Mixer style: transpose → token-mixing MLP → transpose → channel-mixing MLP.

    Two MLPs:
      1. Token-mixing: applied across spatial dimension (N tokens)
      2. Channel-mixing: applied across channel dimension (C channels)

    Both use spiking neurons for binary activation → accumulation-only inference.
    Uses reduced hidden dim (dim // expansion) for efficiency.
    """

    def __init__(self, dim, window_size, pretrained_window_size, num_heads,
                 version="swinv1", qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., norm=None,
                 token_mlp_ratio=0.5, channel_mlp_ratio=2.0, **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        spiking_kwargs['num_steps'] = window_size[0]
        self.norm_layer = norm

        from models.STSwinNet_SNN.Spiking_modules import SpikingNormLayer, Spiking_neuron

        N = window_size[0] * window_size[1] * window_size[2]
        token_hidden = max(1, int(N * token_mlp_ratio))
        channel_hidden = max(1, int(dim * channel_mlp_ratio))

        # Token mixing: [B, C, N] → [B, token_hidden, N] → [B, C, N]
        self.token_fc1 = nn.Linear(N, token_hidden, bias=False)
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.token_bn1 = SpikingNormLayer(token_hidden, window_size[0], norm,
                                              spiking_kwargs.get('v_th', 0.1))
        self.token_sn1 = Spiking_neuron(**{**spiking_kwargs, 'num_steps': window_size[0]})
        self.token_fc2 = nn.Linear(token_hidden, N, bias=False)

        # Channel mixing: [B, N, C] → [B, N, ch_hidden] → [B, N, C]
        self.channel_fc1 = nn.Linear(dim, channel_hidden, bias=False)
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.channel_bn1 = SpikingNormLayer(channel_hidden, window_size[0], norm,
                                                spiking_kwargs.get('v_th', 0.1))
        self.channel_sn1 = Spiking_neuron(**{**spiking_kwargs, 'num_steps': window_size[0]})
        self.channel_fc2 = nn.Linear(channel_hidden, dim, bias=False)
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.channel_bn2 = SpikingNormLayer(dim, window_size[0], norm,
                                                spiking_kwargs.get('v_th', 0.1))
        self.channel_sn2 = Spiking_neuron(**{**spiking_kwargs, 'num_steps': window_size[0]})
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        T, B_, H, W, C = x.shape
        N = H * W

        # Reshape: [T, B_, H, W, C] → [T*B_, C, N]
        x = x.permute(0, 1, 4, 2, 3).reshape(T * B_, C, N)

        # Token mixing (across N dimension)
        x_t = x.transpose(-1, -2)  # [T*B_, N, C]
        x_t = self.token_fc1(x_t)  # [T*B_, N, token_hidden]
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x_t = self.token_bn1(x_t.permute(0, 2, 1)).permute(0, 2, 1)
        x_t = self.token_sn1(x_t)
        x_t = self.token_fc2(x_t)  # [T*B_, N, C]
        x = x + x_t  # residual connection for token mixing

        # Channel mixing (across C dimension)
        x = self.channel_fc1(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.channel_bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_sn1(x)
        x = self.channel_fc2(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.channel_bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.channel_sn2(x)

        # Reshape back: [T*B_, C, N] → [B_, N, C]
        x = x.reshape(T, B_, C, N).permute(0, 1, 3, 2).reshape(B_, N, C)
        x = self.proj_drop(x)
        return x, None

    def flops(self, N):
        token_hidden = max(1, int(N * 0.5))
        channel_hidden = max(1, int(self.dim * 2.0))
        flops = 0
        flops += N * self.dim * token_hidden  # token_fc1
        flops += N * token_hidden * self.dim  # token_fc2
        flops += N * self.dim * channel_hidden  # channel_fc1
        flops += N * channel_hidden * self.dim  # channel_fc2
        return flops


# ── TM3: Pooling-based token mixing ────────────────────────────────────

class PoolTokenMixer(nn.Module):
    """PoolFormer-style: simple pooling for token mixing.

    Even simpler than Conv/MLP mixing: avg_pool → skip connection.
    Minimal parameters, maximally hardware-friendly.
    """

    def __init__(self, dim, window_size, pretrained_window_size, num_heads,
                 version="swinv1", qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., norm=None, pool_size=3, **spiking_kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        spiking_kwargs['num_steps'] = window_size[0]
        self.norm_layer = norm

        from models.STSwinNet_SNN.Spiking_modules import SpikingNormLayer, Spiking_neuron

        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.proj = nn.Linear(dim, dim, bias=False)
        if norm in ["BN", "BNTT", "tdBN", "IN"]:
            self.proj_bn = SpikingNormLayer(dim, window_size[0], norm,
                                            spiking_kwargs.get('v_th', 0.1))
        self.proj_sn = Spiking_neuron(**spiking_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        T, B_, H, W, C = x.shape
        identity = x

        # Pooling-based spatial mixing
        x = x.permute(0, 1, 4, 2, 3).reshape(T * B_, C, H, W)
        x = self.pool(x)
        x = x.reshape(T, B_, C, H, W).permute(0, 1, 3, 4, 2)

        x = x - identity  # subtract identity to get "mixing residual"
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.proj_sn(x)
        x = x.reshape(B_, H * W, C)
        x = self.proj_drop(x)
        return x, None

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 9  # pooling (approx as 3x3 additions)
        flops += N * self.dim * self.dim  # proj
        return flops


# ── Mixer registry ─────────────────────────────────────────────────────

TOKEN_MIXER_REGISTRY = {
    "identity": IdentityTokenMixer,
    "conv": ConvTokenMixer,
    "mlp": MLPTokenMixer,
    "pool": PoolTokenMixer,
}


def install_token_mixers(mixer_type: str):
    """Monkey-patch Spiking_SwinTransformerBlock3D to use token mixers.

    Call BEFORE model construction. Replaces attn_module with the
    specified token mixer class.
    """
    import importlib
    from models.STSwinNet_SNN import Spiking_swin_transformer3D as sst

    if mixer_type not in TOKEN_MIXER_REGISTRY:
        raise ValueError(f"Unknown token mixer: {mixer_type}. "
                         f"Available: {list(TOKEN_MIXER_REGISTRY.keys())}")

    mixer_cls = TOKEN_MIXER_REGISTRY[mixer_type]
    sst.Spiking_SwinTransformerBlock3D.attn_module = mixer_cls
    # Also patch SDSA block if present
    if hasattr(sst, 'SDSA_SwinTransformerBlock3D'):
        sst.SDSA_SwinTransformerBlock3D.attn_module = mixer_cls

    return mixer_cls
