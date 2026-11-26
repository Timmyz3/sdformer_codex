"""R24 fixed H1 representative cover; retain frozen R16 V2 native-BN math.

Device-neutral inference primitive. No q, score, data, GT, fitting or selection.
The old module and its globals are never modified. Only a private function's
pack binding is replaced; both dense current-BN stages remain the old code.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

SHARED_SHA = 'ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec'
BASE_SHA = 'e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731'
_native_compact = None
_base_module = None
_base_path = None
expand = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack_two_slot(u, mask):
    """Complete spatial9 coarse -> nine H1 pairs; every real tail stays fine."""
    import torch
    require(tuple(u.shape[2:4]) == (30, 40) and tuple(mask.shape) == (4, 5)
            and mask.dtype == torch.bool, 'Fixed real30x40 and spatial9 mask')
    require(bool(mask[3, :].all()) and bool(mask[:, 4].all()), 'All eight real tail tiles must remain fine')
    cells, points = [], []
    for sy in range(0, 30, 9):
        for sx in range(0, 40, 9):
            ey, ex = min(sy+9, 30), min(sx+9, 40)
            if bool(mask[sy//9, sx//9]):
                rectangles = [(y, y+1, x, x+1) for y in range(sy, ey) for x in range(sx, ex)]
            else:
                require((ey-sy, ex-sx) == (9, 9), 'Only complete9x9 may be coarse')
                rectangles = []
                for y in range(sy, ey, 3):
                    for x in range(sx, ex, 3):
                        rectangles.extend(((y, y+1, x, x+3), (y+1, y+3, x, x+3)))
            for y0, y1, x0, x1 in rectangles:
                area = (y1-y0)*(x1-x0)
                points.append(u[:, :, y0:y1, x0:x1, :].sum((2, 3))/area)
                cells.append((y0, y1, x0, x1, area))
    packed = torch.stack(points, dim=2).unsqueeze(3).contiguous()
    areas = torch.tensor([c[4] for c in cells], dtype=u.dtype, device=u.device)
    require(sum(c[4] for c in cells) == 1200, 'Exact real spatial coverage')
    require(packed.shape[:2] == u.shape[:2] and packed.shape[-1] == u.shape[-1], 'Complete T/B/channel preservation')
    return packed, areas, cells


def configure(base_path):
    """Bind an explicit original primitive and its actual imported dependency."""
    global _native_compact, _base_module, _base_path, expand
    base_path = Path(base_path).resolve()
    shared_path = base_path.parent/'compact_current_mlp.py'
    require(sha(base_path) == BASE_SHA and sha(shared_path) == SHARED_SHA, 'Original V2/shared bytes')
    sys.path.insert(0, str(base_path.parent))
    spec = importlib.util.spec_from_file_location('_r24_original_native_dense_bn_v2', base_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    shared = sys.modules.get('compact_current_mlp')
    require(shared is not None and hasattr(shared, '__file__')
            and Path(shared.__file__).resolve() == shared_path.resolve()
            and sha(shared.__file__) == SHARED_SHA, 'Actually imported original shared math path/SHA')
    require(all(getattr(original, name) is getattr(shared, name) for name in ('validate', 'pack', 'expand')),
            'Original imported math symbol identity')
    private_globals = dict(original.compact_mlp.__globals__)
    private_globals['pack'] = pack_two_slot
    private = types.FunctionType(original.compact_mlp.__code__, private_globals,
                                 '_r24_fixed_H1_native_BN_compact', original.compact_mlp.__defaults__,
                                 original.compact_mlp.__closure__)
    require([key for key in private_globals if private_globals[key] is not original.compact_mlp.__globals__[key]] == ['pack'],
            'Only private pack binding may change')
    _native_compact, _base_module, _base_path, expand = private, original, base_path, shared.expand
    return {'base_primitive_sha256': BASE_SHA, 'shared_math_sha256': SHARED_SHA,
            'private_globals_overrides': ['pack'], 'original_globals_mutated': False,
            'BN_mode': 'NATIVE_DENSE_CURRENT'}


def compact_mlp(native, u, fine_mask):
    require(_native_compact is not None, 'Explicit frozen base must be configured')
    result, audit = _native_compact(native, u, fine_mask)
    audit.update(partition_rule='FIXED_H1_TOP1_BOTTOM2_FULL3X3_TAIL_TILES_FINE',
                 coarse_representatives_per_complete3x3=2,
                 base_primitive_sha256=BASE_SHA, shared_math_sha256=SHARED_SHA,
                 private_globals_overrides=['pack'], original_globals_mutated=False)
    return result, audit
