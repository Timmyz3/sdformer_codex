#!/usr/bin/env python3
"""Logical storage obligations only; no new service model or model execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
SOURCE = RESEARCH/'flow_story_20261019/round27/hardware/UNIFIED_SERVICE_V1.json'
SOURCE_SHA = 'f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA
    data = json.loads(SOURCE.read_text())
    layers = []
    for a in data['inventory']['affines']:
        if a['scope'] not in ('MLP12', 'patch4'):
            continue
        weights = a['K']*a['Cout']
        assert weights % 2 == 0 and a['binary_input'] and not a['bias']
        layers.append(dict(
            name=a['name'], weight=a['weight'], scope=a['scope'], K=a['K'], Cout=a['Cout'],
            weights=weights, output_geometry=[a['T'], a['H'], a['W']],
            original_q8_bytes=weights, additional_packed_q4_bytes=weights//2,
            additional_int8_carrier_q4_bytes=weights,
            original_scale8_FP32_bytes=4*a['Cout'], additional_scale4_FP32_bytes=4*a['Cout']))
    assert len(layers) == 28 and len({x['weight'] for x in layers}) == 28
    assert sum(x['scope'] == 'MLP12' for x in layers) == 24
    totals = {k: sum(x[k] for x in layers) for k in (
        'weights', 'original_q8_bytes', 'additional_packed_q4_bytes',
        'additional_int8_carrier_q4_bytes', 'original_scale8_FP32_bytes',
        'additional_scale4_FP32_bytes')}
    totals['all28_output_channels'] = sum(x['Cout'] for x in layers)
    assert totals['weights'] == 17584128
    result = dict(
        schema='flow_r31_independent_Q4_Q8_logical_storage_v1',
        source_sha256=SOURCE_SHA, source_path=str(SOURCE), script_sha256=sha(Path(__file__)),
        status='DESIGN_OBLIGATIONS_NOT_EXECUTED', layers=layers, totals=totals,
        formulas=dict(weight_elements='K*Cout', extra_packed_Q4_bytes='K*Cout/2',
                      extra_unpacked_int8_Q4_bytes='K*Cout',
                      extra_scale4_bytes='4*Cout'),
        production_contract=[
            'Q8 scale and all eight Q8 weight planes remain required by true Q8 tails.',
            'Independent Q4 uses its own RNE scale/weights, not high four bits of Q8.',
            'q4<<4 and scale4/16 can use integer bits4..7 with the existing signed accumulator.',
            'High8 rows use eight Q8 planes; low4 rows use four distinct Q4 planes.',
            'Two weight slots must reserve explicit variant/stripe lifetimes; they cannot also provide free double buffering.',
            'Recount variant DMA bursts, staging, scale selection, local reads, canonical output scatter and BN pass ordering.'],
        limitations=[
            'Logical unique storage bytes only; no padding, burst rounding or repeated stripe transfers included.',
            'No real packed4 storage or runtime implementation has been established by this table.',
            'Same layer quotas do not imply the same traffic, service cost or net hardware saving.',
            'No R30 quota, frozen cost result, source, precision rule or formal execution is modified.'],
        model_forwards=0, GPU_used=False, GT_values_read=False)
    output = HERE/'LOGICAL_WEIGHT_STORAGE_V1.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output_sha256': sha(output), 'totals': totals}))


if __name__ == '__main__':
    main()
