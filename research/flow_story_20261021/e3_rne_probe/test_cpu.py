#!/usr/bin/env python3
"""Bounded CPU checks of E3 adapters; not full-model or GPU verification."""
import argparse
import ast
import copy
import json
import tempfile
from pathlib import Path
import torch
import probe_contract as pc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--selections', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    pc.require(not a.output.exists() and not torch.cuda.is_initialized(), 'Fresh CPU-only test')
    torch.set_num_threads(2); torch.manual_seed(20261021)
    core = pc.load_core(a.core)
    c = pc.shape_contract(core, a.inventory, a.selections)
    pc.validate(c, core, require_endpoint=False)
    checks = {}

    def rejects(fn):
        try:
            fn()
        except (ValueError, KeyError):
            return True
        return False

    checks['unbound_contract_cannot_execute'] = rejects(lambda: pc.validate(c, core))
    previous = {n: set() for n in core.CONVS}
    costs = []
    for action in range(-1, 6):
        masks, record = pc.materialize(c, action, core)
        pc.require(len(masks) == 16, 'All16 explicit mask groups')
        costs.append(record['added_core_ticks_vs_Q4'])
        for n in core.TARGETS:
            g = c['geometries'][n]; mask = masks[core.group_name(n)]
            tail, high, audit = core.row_masks(10, 1, g['H'], g['W'], 'cpu',
                'Q4' if action == -1 else 'M', None if action == -1 else mask)
            selected = 151 if action >= 0 and n in core.CONVS else 0
            pc.require(int((high & ~tail).sum()) == selected*81*10,
                       'Real E0 mask path counts only complete selected patch rows')
            if n in core.CONVS and action >= 0:
                cells = {tuple(x) for x in audit['selected_complete_tiles']}
                pc.require(len(cells) == 151 and not previous[n] & cells, 'Disjoint151 macro')
                previous[n] |= cells
        pc.require(all(v == 0 for n,v in record['selected_tiles'].items() if n not in core.CONVS),
                   'No MLP upgrade')
    pc.require(costs == [0]+[82192320]*6 and all(len(s) == 906 for s in previous.values()),
               'Fixed arithmetic and four complete leftovers per patch')
    checks['seven_action_geometry_domains_and_equal_core_work'] = True
    for value in (-2, 6, 0.0, (0,1)):
        pc.require(rejects(lambda value=value: pc.materialize(c, value, core)), 'Reject extra/pair actions')
    bad = copy.deepcopy(c); bad['groups'][0]['tiles'][core.CONVS[0]][0] = [26,0]
    pc.require(rejects(lambda: pc.validate(bad, core, require_endpoint=False)), 'Reject true-tail selection')
    checks['extra_actions_fragmentation_and_tail_rejection'] = True

    # Real frozen conv producer on six toy9x9 macros plus true tail (H10,W55).
    # Geometry is deliberately miniature, not claimed as full/native inference.
    source = torch.nn.Conv2d(2, 3, 3, padding=1, bias=False)
    x = torch.randint(0, 2, (10,1,2,10,55)).float()
    parity_counts = 0
    for action in range(6):
        mask = torch.zeros(10,55,dtype=torch.bool); mask[:9, action*9:(action+1)*9] = True
        actual = core.STEAffine(source, 'conv', 'M', selected=mask, ste=False, variant='rne4')
        dense = core.STEAffine(source, 'conv', 'M', selected=mask, ste=False, reference=True, variant='rne4')
        actual.capture_integer = dense.capture_integer = True
        with torch.no_grad():
            ya, yd = actual(x), dense(x)
        pc.require(torch.equal(actual.last_integer, dense.last_integer)
                   and torch.equal(ya.view(torch.uint8), yd.view(torch.uint8)), 'Actual/dense RNE conv bytes')
        w = actual.last_audit
        pc.require(w['restored_complete_rows'] == 810
                   and w['logical_row_plane_work'] == 8*w['high8_rows']+4*w['low4_rows'],
                   'Actual exclusive4/8 work')
        pc.require(all(k['bit'] >= 4 for k in w['kernel_calls'] if k['scope'] == 'low4'),
                   'No low-domain four unnecessary planes')
        parity_counts += 2
    checks['six_toy_conv_actions_real_plane_vs_dense_bytes'] = True
    # Background Q4 and all-false M are exact; no predictor input is needed.
    all_low = core.STEAffine(source,'conv','Q4',ste=False,variant='rne4')
    empty_m = core.STEAffine(source,'conv','M',selected=torch.zeros(10,55,dtype=torch.bool),
                             ste=False,variant='rne4')
    with torch.no_grad():
        pc.require(torch.equal(all_low(x),empty_m(x)), 'Background is the original E0 Q4 route')
    checks['background_empty_M_matches_frozen_Q4'] = True
    # Strict endpoint binding rejects incomplete results before looking for files.
    with tempfile.TemporaryDirectory(prefix='flow_e3_cpu_') as temp:
        path = Path(temp)/'incomplete.json'
        pc.write(path, dict(status='RUNNING'))
        pc.require(rejects(lambda: pc.endpoint_binding(path, Path(temp)/'absent.pt', c['target_ids'])),
                   'No running/partial endpoint accepted')
    checks['partial_endpoint_rejected'] = True
    for name in ('probe_contract.py','collect_probe.py','test_cpu.py'):
        ast.parse((Path(__file__).parent/name).read_text())
    pc.require(not torch.cuda.is_initialized(), 'Tests never initialize CUDA')
    checks['all_source_syntax_and_no_CUDA_initialization'] = True
    pc.write(a.output, dict(schema='flow_e3_rne_probe_cpu_v1', status='PASS', checks=checks,
        full_model_forwards=0, GPU_used=False, GT_arrays_read=False,
        toy_conv_module_forwards=parity_counts+2, expected_runtime_counts=dict(formal=112,engineering=6,total=118),
        core_sha256=pc.CORE_SHA, source_sha256={n:pc.sha(Path(__file__).parent/n)
            for n in ('probe_contract.py','collect_probe.py','test_cpu.py')},
        limitations='Micro producer/adapter checks; no actual model, endpoint checkpoint, remote data or GPU run'))
    print(json.dumps(dict(status='PASS',checks=len(checks),GPU_used=False)))


if __name__ == '__main__':
    main()
