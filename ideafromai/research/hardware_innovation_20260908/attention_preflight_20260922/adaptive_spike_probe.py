"""Inference-only probes. Preserve the student's legacy Q/K tensor mapping.

Not a SAFformer reproduction: no DWConv, SGM, or new gate neuron. The fixed-k
arm isolates support selection, per existing logical head, with stable ties.
Existing Motion arithmetic still runs before the hook; timings are not savings.
"""
import torch


def legacy_qk(q, k, heads):
    t, b, h, w, c = q.shape
    d = c // heads
    q_orig = q.reshape(t, b, heads, -1, d)
    q_token = q_orig.permute(1, 2, 0, 3, 4).reshape(b, heads, -1, d)
    k_token = k.reshape(b, heads, -1, d)
    return q_token, k_token


def stable_select(scores, keep):
    """Reference sort, not a bucket implementation or RTL cycle model."""
    n = scores.shape[-1]
    if not 0 <= keep <= n:
        raise ValueError('keep outside row length')
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, order[..., :keep], True)
    return selected


def gate_probe(q, k, heads, mode, keep_fraction=0.5):
    q_token, k_token = legacy_qk(q, k, heads)
    counts = (q_token > 0).sum(-1)
    selected = torch.ones_like(counts, dtype=torch.bool)
    if mode == 'saf_fixed_half_probe':
        selected = stable_select(counts, int(counts.shape[-1] * keep_fraction))
    elif mode != 'q_presence_probe':
        raise ValueError(mode)
    # A stateless, unit-amplitude gate. K retains its actual {0, theta_K}
    # values. This is not the old (unused) sn2_q nor SAF's LIF gate.
    gate = selected & (counts > 0)
    output = k_token * gate.unsqueeze(-1)
    return output, counts, selected, gate


def mapping_statistics(gate, q, heads):
    t, b, h, w, c = q.shape
    d = c // heads
    expanded = gate.unsqueeze(-1).expand(b, heads, t*h*w, d)
    # Inverse of the ACTUAL k.reshape, rather than assuming head-contiguous
    # physical positions. Required before K values are inspected.
    raw_source_lanes = expanded.reshape(t, b, h, w, c)
    # Actual end-of-attention mapping into the output projection.
    proj_lanes = expanded.reshape(b, heads, t, h, w, d).permute(2, 0, 3, 4, 1, 5)
    proj_lanes = proj_lanes.reshape(t, b, h, w, c)
    return dict(
        postPSN_K_output_position_requirement=float(raw_source_lanes.any(-1).float().mean()),
        K_PSN_input_position_requirement_after_Tw_closure=float(
            raw_source_lanes.any(0, keepdim=True).expand_as(raw_source_lanes).any(-1).float().mean()),
        projection_position_requirement_after_gate=float(proj_lanes.any(-1).float().mean()),
        # K BN still uses every position: above counts do NOT license removal
        # of pre-BN K linear work in the current deployment function.
        legal_preBN_K_production_ratio_with_live_BN=1.0,
        scope='Current padded tensor coordinates; not true T10 groups or bank transactions')


class AttentionProbe:
    def __init__(self, attention):
        self.attention = attention
        self.mode = 'motion'
        self.rows = []
        self.cache = {}
        self.handles = [
            attention.sn_q.register_forward_hook(self.capture('q')),
            attention.sn_k.register_forward_hook(self.capture('k')),
            attention.attn_drop.register_forward_pre_hook(self.replace)]

    def capture(self, name):
        def hook(module, inputs, output):
            self.cache[name] = output
        return hook

    def replace(self, module, inputs):
        q, k = self.cache.pop('q'), self.cache.pop('k')
        heads = self.attention.num_heads
        q_token, k_token = legacy_qk(q, k, heads)
        if self.mode == 'motion':
            value = inputs[0]
            counts = (q_token > 0).sum(-1)
            selected = torch.ones_like(counts, dtype=torch.bool)
            gate = selected
        else:
            value, counts, selected, gate = gate_probe(q, k, heads, self.mode)
        row = dict(mode=self.mode, raw_Q_shape=list(q.shape), heads=heads,
            logical_Q_shape=list(q_token.shape),
            selected_logical_token_ratio=float(selected.float().mean()),
            nonzero_gate_ratio=float(gate.float().mean()),
            discarded_positive_count=int(((counts > 0) & ~selected).sum()),
            total_positive_count=int((counts > 0).sum()),
            projection_input_nonzero_ratio=float((value != 0).float().mean()),
            q_nonzero_ratio=float((q_token != 0).float().mean()),
            k_nonzero_ratio=float((k_token != 0).float().mean()),
            count_histogram=torch.bincount(counts.flatten(), minlength=q_token.shape[-1]+1).cpu().tolist())
        if self.mode != 'motion':
            row.update(mapping_statistics(gate, q, heads))
            row['carrier_values_preserved'] = bool(torch.all((value == 0) | (value == k_token)))
        self.rows.append(row)
        return (value,)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.cache.clear()
