"""Monkey-patch: inject shiftmax into Spiking_BN_WindowAttention3D.forward.

Called from LOAD_MODEL_PATCH after model construction. Only patches
the swinv1 attention path (MS_SpikingformerFlowNet_en4 uses swinv1).
"""

import torch
import torch.nn as nn

from src.models.modules.sparse_ops.shiftmax import shiftmax

_ORIGINAL_FORWARDS = {}


def _make_patched_forward(original_forward):
    """Wrap the forward to add shiftmax after position bias."""

    def patched_forward(self, x, mask=None):
        # Call original (which does everything up to attn@v and proj)
        # We intercept AFTER position bias but BEFORE mask/@v
        # Use the same code path as original but insert shiftmax
        T, B_, H, W, C = x.shape
        q = self.linear_q(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        q = self.sn_q(q)
        k = self.linear_k(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        k = self.sn_k(k)
        v = self.linear_v(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            v = self.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        v = self.sn_v(v)
        q, k, v = (
            q.reshape(B_, self.num_heads, -1, C // self.num_heads),
            k.reshape(B_, self.num_heads, -1, C // self.num_heads),
            v.reshape(B_, self.num_heads, -1, C // self.num_heads),
        )
        N = q.shape[2]
        attn = self.vanilla_attn(q, k)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)
        ].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        # --- I6: Shiftmax ---
        attn = shiftmax(attn, dim=-1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(B_, self.num_heads, T, H, W, C // self.num_heads)
        x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C)
        x = self.proj(x)
        if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
            x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        x = self.proj_sn(x).reshape(B_, N, C)
        x = self.proj_drop(x)
        return x, attn

    return patched_forward


def install_shiftmax_patch(model: nn.Module, raw_config: dict | None) -> int:
    """Replace Spiking_BN_WindowAttention3D.forward with shiftmax version.

    Returns number of patched modules.
    """
    sc = (raw_config or {}).get("shiftmax", {})
    if not sc.get("enabled", False):
        return 0

    from models.STSwinNet_SNN.Spiking_swin_transformer3D import Spiking_BN_WindowAttention3D

    count = 0
    for module in model.modules():
        if isinstance(module, Spiking_BN_WindowAttention3D):
            if module not in _ORIGINAL_FORWARDS:
                _ORIGINAL_FORWARDS[id(module)] = module.forward
                module.forward = _make_patched_forward(module.forward).__get__(
                    module, Spiking_BN_WindowAttention3D
                )
            count += 1
    return count
