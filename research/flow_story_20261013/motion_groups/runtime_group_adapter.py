"""CPU research selector: runtime coarse flow, matched exact slow work.

This is a zero-recovery identity-bypass diagnostic, not a trained DynamicSwin
fast/slow model or an ASIC-ready controller. All four policies and their common
reachable budget are computed online for a fair experiment, and counted.
"""
import contextlib
import hashlib
import math
import time

import torch

import coarse_head
import exact_budget_policy as budget
import group_policy as groups

LAYER = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp'
SHAPE = (10, 1, 30, 40, 384)
POLICIES = ('independent', 'motion', 'rotated90', 'horizontal_local')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class RuntimeGroupPolicy:
    """Only u and frozen head parameters enter decisions; no sample/GT input."""
    def __init__(self, head, policy, cap=900):
        require(policy in POLICIES and cap == 900, 'Fixed four-arm75% pilot required')
        require(isinstance(head, coarse_head.RuntimeCoarseFlowHead) and head.in_channels == 384,
                'Trained stage2 head required')
        require(not head.training and all(not p.requires_grad for p in head.parameters()), 'Freeze head first')
        self.head, self.policy, self.cap = head, policy, cap
        self.records = []

    def __call__(self, name, u):
        require(name == LAYER and tuple(u.shape) == SHAPE and u.device.type == 'cpu' and
                not torch.is_grad_enabled(), 'Only declared CPU stage2 complete-T10 inference supported')
        started = time.monotonic()
        descriptor = self.head.aggregate(u)
        flow = self.head.fc2(self.head.activation(self.head.fc1(descriptor)))
        require(tuple(flow.shape) == (1, 4, 5, 2) and bool(torch.isfinite(flow).all()), 'Invalid runtime flow')
        # Ordinary magnitude prior, given identically to all grouping choices.
        score = torch.linalg.vector_norm(flow, dim=-1)[0]
        values, scores = flow[0].tolist(), score.tolist()
        counts = [[min(9, 30-9*r) * min(9, 40-9*c) for c in range(5)] for r in range(4)]
        proposals = {p: groups.propose_pairs(scores, values, counts, counts, policy=p,
            association_mode='direction_one_hop', tile_size_pixels=(144., 144.), min_motion_pixels=1.)
            for p in POLICIES}
        common = budget.common_target(proposals, self.cap)
        target = common['target_cost']
        require(type(target) is int and 0 < target <= self.cap, 'No positive common work point')
        selected = budget.select_exact(proposals[self.policy], target)
        require(selected['feasible'] and selected['slow_service_cost'] == target,
                'Exact selector failed common reachable budget')
        keep = torch.tensor(selected['slow_mask'], dtype=torch.bool)
        require(tuple(keep.shape) == (4, 5), 'Mask geometry differs')
        self.records.append({'policy': self.policy, 'flow_source': 'frozen C384->16 GELU->2 on current pre-MLP u',
            'score_rule': 'L2 magnitude of runtime predicted flow, common to all arms',
            'input_shape_TBHWC': list(u.shape), 'input_sha256': tensor_sha(u),
            'descriptor_sha256': tensor_sha(descriptor), 'flow_dxdy_pixels': values, 'scores': scores,
            'points_per_tile': counts, 'nominal_slow_cap': self.cap, 'dense_spatial_points': 1200,
            'common_budget': common, 'proposal': proposals[self.policy], 'selection': selected,
            'controller_wall_seconds_CPU': time.monotonic()-started,
            'coarse_head_shape_cost': coarse_head.shape_operation_counts(SHAPE),
            'all_four_proposals_computed_for_cost_matching': True,
            'scheduler_kind': 'bounded exact DP CPU research reference; no ASIC cycle claim',
            'GT_or_final_flow_used': False, 'upstream_attention_already_executed': True})
        return keep


@contextlib.contextmanager
def replace_one_mlp(model, tile_module, mode, policy):
    """Keep external current-stat BN and all other native MLPs untouched."""
    require(all(not m.training for m in model.modules()), 'Whole model must be eval')
    parent_name, child = LAYER.rsplit('.', 1)
    parent = model.get_submodule(parent_name)
    original = getattr(parent, child)
    require(type(original).__name__ == 'MS_Spiking_Mlp' and getattr(parent, 'cnf', None) == 'ADD'
            and not getattr(parent, 'use_checkpoint', False), 'Native ADD/no-checkpoint parent required')
    wrapper = tile_module.SameShapeTileMLP(original, LAYER, mode, policy, audit_hashes=True)
    try:
        setattr(parent, child, wrapper)
        yield wrapper
    finally:
        setattr(parent, child, original)
