"""Metadata-only six-group construction; no family choice or model execution."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import group_precision_core as core

INVENTORY_SHA = 'f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'
require = core.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_from_inventory(path):
    require(sha(path) == INVENTORY_SHA, 'Exact frozen R27 hardware inventory bytes')
    obj = json.loads(Path(path).read_text())
    records = {}
    for row in obj['inventory']['affines']:
        name = row['weight'].removesuffix('.weight')
        if name in core.TARGETS:
            records[name] = {k: row[k] for k in ('T', 'H', 'W', 'K', 'Cout')}
    require(set(records) == set(core.TARGETS), 'All28 actual target geometries')
    return records


def tile_ticks(g):
    return 2 * g['T'] * 81 * math.ceil(g['K'] / 128) * math.ceil(g['Cout'] / 16)


def validate(contract):
    require(contract['schema'] == 'flow_e2_six_group_contract_v1', 'Explicit six-group contract')
    geometries = contract['geometries']
    require(set(geometries) == set(core.TARGETS), 'All28 geometry declarations')
    for name, g in geometries.items():
        require(set(g) == {'T', 'H', 'W', 'K', 'Cout'} and all(type(x) is int and x > 0 for x in g.values())
                and g['T'] == 10, 'Positive native fullT10 geometry')
    require(contract['inventory_sha256'] == INVENTORY_SHA, 'Bound source inventory')
    groups = contract['groups']
    require(len(groups) == 6 and [g['id'] for g in groups] == list(range(6)), 'Six ordered group IDs')
    seen = set()
    costs = []
    for group in groups:
        require(group['tiles'] and set(group['tiles']).issubset(core.TARGETS), 'Nonempty target scope')
        ticks = 0
        for name, coords in group['tiles'].items():
            g = geometries[name]
            require(isinstance(coords, list) and coords, 'Nonempty complete tile list')
            require(coords == sorted(coords), 'Canonical row-major coordinates')
            for cell in coords:
                require(isinstance(cell, list) and len(cell) == 2 and all(type(v) is int for v in cell),
                        'Integer tile coordinates')
                y, x = cell
                require(0 <= y < g['H']//9 and 0 <= x < g['W']//9, 'Complete native tile only')
                key = (name, y, x)
                require(key not in seen, 'No overlap within or across groups for an affine')
                seen.add(key)
                ticks += tile_ticks(g)
        require(group['cancelled_kernel_ticks'] == ticks, 'Explicit cancellation arithmetic')
        costs.append(ticks)
    require(len(set(costs)) == 1 and costs[0] > 0, 'Exactly equal positive kernel work for all groups')
    require(contract['single_cancelled_kernel_ticks'] == costs[0], 'Single cost')
    require(contract['pair_cancelled_kernel_ticks'] == 2*costs[0], 'Disjoint pair cost')
    return contract


def build(geometries, family_targets, *, source_state_sha256):
    """Return a candidate contract, never choose family or claim it is frozen.

    Within each selected affine, six contiguous row-major tile runs of equal
    size; leftover complete tiles and all true tails remain Q6/Q8 respectively.
    Affines with fewer than six complete tiles cannot supply six disjoint runs.
    """
    require(set(geometries) == set(core.TARGETS), 'Complete native geometry set')
    require(family_targets and len(set(family_targets)) == len(family_targets)
            and set(family_targets).issubset(core.TARGETS), 'Caller-selected unique target family')
    groups = [dict(id=i, tiles={}, cancelled_kernel_ticks=0) for i in range(6)]
    leftovers = {}
    omitted = []
    for name in family_targets:
        g = geometries[name]
        cells = [[y, x] for y in range(g['H']//9) for x in range(g['W']//9)]
        size = len(cells)//6
        if size == 0:
            omitted.append(name)
            leftovers[name] = cells
            continue
        for i, group in enumerate(groups):
            group['tiles'][name] = cells[i*size:(i+1)*size]
            group['cancelled_kernel_ticks'] += size*tile_ticks(g)
        leftovers[name] = cells[6*size:]
    require(groups[0]['tiles'], 'Chosen family cannot supply six nonempty disjoint groups')
    out = dict(
        schema='flow_e2_six_group_contract_v1', status='UNFROZEN_CALLER_REQUESTED_METADATA_CANDIDATE',
        inventory_sha256=INVENTORY_SHA, source_state_sha256=source_state_sha256,
        quantizer='same_q8_maxabs127_scale_tail_drop0_complete_drop2_selected_drop4',
        geometries=geometries, family_targets=list(family_targets),
        groups=groups, leftover_complete_tiles=leftovers,
        omitted_targets_fewer_than_six_tiles=omitted,
        single_cancelled_kernel_ticks=groups[0]['cancelled_kernel_ticks'],
        pair_cancelled_kernel_ticks=2*groups[0]['cancelled_kernel_ticks'],
        net_cycle_claim=False, GT_used=False,
        construction='six_contiguous_equal_row_major_tile_runs_per_target; leftover_Q6',
        causal_prefix_binding='REQUIRES_CALLER_FREEZE_BEFORE_LABEL_COLLECTION')
    return validate(out)


def actions():
    return [()] + [(i,) for i in range(6)] + list(itertools.combinations(range(6), 2))


def materialize(contract, action):
    import torch
    validate(contract)
    action = tuple(action)
    require(action in actions(), 'Only background, six singles or fifteen pairs')
    masks = {n: torch.zeros(g['H'], g['W'], dtype=torch.bool)
             for n, g in contract['geometries'].items()}
    for i in action:
        for name, coords in contract['groups'][i]['tiles'].items():
            for y, x in coords:
                masks[name][y*9:y*9+9, x*9:x*9+9] = True
    return masks, dict(
        action=list(action), selected_mask_sha256={n: core.digest(m) for n, m in masks.items()},
        selected_complete_tiles={n: int(m.sum())//81 for n, m in masks.items()},
        cancelled_kernel_ticks=len(action)*contract['single_cancelled_kernel_ticks'],
        net_cycle_claim=False)


class BoundCore:
    """Adapter to capture helpers; call capture_full with ste=False only."""
    def __init__(self, contract, action):
        self.contract = validate(contract)
        self.masks, self.mask_record = materialize(contract, action)
        self.mode = 'Q6' if not action else 'Q6_Q4'

    def __getattr__(self, key):
        return getattr(core, key)

    def installed(self, model, mode, *, ste=False, reference=False):
        require(mode == self.mode, 'Arm matches fixed action')
        return core.installed(model, mode, masks=self.masks, reference=reference, ste=ste,
                              geometries=self.contract['geometries'])
