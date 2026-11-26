"""Explicit MLP-frozen / outside-current-stat context; old TileMLP is unchanged."""
import contextlib
from pathlib import Path
import types

import evaluate_breadth as base
import run_bn_scope as scope

OLD_TILE_SHA = '42729ab7617b325f929836af891c2dbdd739be01d2c715fa1269a9184be7f961'
SCOPE_SHA = '2b5c61e3abedcc02ba09adadb432dfc0ff2cd2fcd2b7b3e3a2475c63d7a11840'


@contextlib.contextmanager
def hybrid_tiled_mlp_model(model, mode, policy, old_tile, torch, selected_names=None,
                           expected_mlp_blocks=12, expected_total_bn=78):
    base.require(mode == 'actual_skip', 'This bounded hybrid runner measures actual_skip only')
    base.require(all(not m.training for m in model.modules()), 'Hybrid requires complete eval/dropout-off')
    all_bn, mlp_bn = scope.bn_inventory(model, torch, expected_mlp_blocks, expected_total_bn)
    for name, layer in all_bn.items():
        if name not in mlp_bn:
            base.require(not layer.track_running_stats and layer.running_mean is None and layer.running_var is None
                         and layer.num_batches_tracked is None, 'External BN must retain complete-domain current stats: '+name)
    targets = [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == 'MS_Spiking_Mlp']
    base.require(selected_names is None or set(selected_names) == {n for n, _ in targets}, 'Only all12 declared MLPs supported')
    # Crucially, the old guard is applied to the canceled producers themselves, not the whole net.
    for _, original in targets:
        old_tile.require_frozen(original)
    before = base.bn_signature(model, torch)
    replacements, wrappers = [], []
    try:
        for name, original in targets:
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            base.require(getattr(parent, 'cnf', None) == 'ADD' and not getattr(parent, 'use_checkpoint', False),
                         'Native complete residual ADD/no-checkpoint required')
            wrapped = old_tile.TileMLP(original, name, mode, policy)
            replacements.append((parent, child_name, original))
            setattr(parent, child_name, wrapped)
            wrappers.append(wrapped)
        yield wrappers
    finally:
        for parent, child_name, original in reversed(replacements):
            setattr(parent, child_name, original)
        base.require(base.bn_signature(model, torch) == before, 'Context mutated original BN configuration/buffers')


def make_module(prior_root):
    base.require(base.sha(scope.__file__) == SCOPE_SHA, 'BN-scope helper source changed')
    source = (Path(prior_root)/'deployment_contract/tile_mlp.py').resolve()
    base.require(base.sha(source) == OLD_TILE_SHA, 'Original TileMLP changed')
    original = base.import_file('hybrid_pinned_original_tile', source)
    import torch
    def context(model, mode, policy, selected_names=None):
        return hybrid_tiled_mlp_model(model, mode, policy, original, torch, selected_names)
    return types.SimpleNamespace(tiled_mlp_model=context,
                                 source_contract={'old_tile_sha256': OLD_TILE_SHA, 'scope_sha256': SCOPE_SHA,
                                                  'context_sha256': base.sha(__file__)})
