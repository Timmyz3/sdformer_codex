#!/usr/bin/env python3
"""Reduce already audited actual final-W evidence; no new RTL/model execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT/'hardware/rtl_results/r30_bridge_v1/ACTUAL_FINAL_W_AUDIT_V1.json'
MANIFEST = ROOT/'actual_v1/final_vector_export_v1/FINAL_VECTOR_MANIFEST_V1.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a, m = json.loads(AUDIT.read_text()), json.loads(MANIFEST.read_text())
    assert a['status'] == 'PASS' and a['source_manifest_sha256'] == sha(MANIFEST)
    rows, serial_rows, padded_rows, calls = 0, 0, 0, 0
    bit_rows = [0]*8
    layers = []
    for name, v in m['work'].items():
        counts = [0]*8
        layer_pads = 0
        for c in v['kernel_calls']:
            assert c['kind'] == 'binary_weight_plane'
            bit = c['bit']
            assert c['scope'] == ('high8' if bit < 4 else 'all')
            assert c['coefficient'] == (-128 if bit == 7 else 2**bit)
            counts[bit] += c['rows']
            layer_pads += c['padding_rows']
        assert counts == [v['high8_rows']]*4 + [v['rows']]*4
        assert sum(counts) == v['logical_row_plane_work']
        assert layer_pads == v['explicit_padded_rows']
        assert v['low4_rows'] + v['high8_rows'] == v['rows']
        rows += v['rows']
        serial_rows += sum(counts)
        padded_rows += layer_pads
        calls += len(v['kernel_calls'])
        bit_rows = [x+y for x, y in zip(bit_rows, counts)]
        layers.append(dict(name=name, rows=v['rows'], high8_rows=v['high8_rows'],
                           low4_rows=v['low4_rows'], per_plane_actual_rows=counts,
                           explicit_padded_rows=layer_pads))
    assert len(layers) == 28 and rows == 4188000
    assert bit_rows == [1320600]*4 + [4188000]*4
    assert serial_rows == 22034400 and padded_rows == 288 and calls == 648
    cases = []
    for k in (384, 864):
        subset = [c for c in a['actual_cases'] if c['K'] == k]
        q4 = sum(c['mode'] == 'Q4' for c in subset)
        q8 = sum(c['mode'] == 'Q8' for c in subset)
        assert (q4, q8) == (4, 12)
        chunk = (k+127)//128
        cases.append(dict(K=k, outputs=16, actual_Q4_cases=q4, actual_Q8_cases=q8,
                          Q4_binary_accepted_beats_per_row=4*chunk,
                          Q8_binary_accepted_beats_per_row=8*chunk,
                          ordinary_int8_gateadd_beats_per_row=(k+15)//16,
                          actual_mixed_binary_beats=q4*4*chunk+q8*8*chunk,
                          counterfactual_same_cases_all_Q8_binary_beats=(q4+q8)*8*chunk))
    totals = dict(actual_case_count=32, actual_Q4_cases=8, actual_Q8_cases=24,
                  unique_cases=32, independent_captured_int32_values=512,
                  RTL_actual_int32_comparisons=32*16*4,
                  RTL_synthetic_int32_comparisons=110*16*4,
                  RTL_total_int32_comparisons=142*16*4,
                  actual_case_binary_accepted_beats=sum(c['actual_mixed_binary_beats'] for c in cases),
                  same_cases_all_Q8_binary_accepted_beats=sum(c['counterfactual_same_cases_all_Q8_binary_beats'] for c in cases),
                  full28_spatial_rows_retained=rows, full28_actual_row_planes=serial_rows,
                  full28_all_Q8_row_planes=rows*8, full28_cancelled_low_row_planes=rows*8-serial_rows,
                  full28_explicit_software_padding_rows=padded_rows,
                  full28_software_integer_kernel_calls=calls)
    result = dict(
        schema='flow_r30_actual_final_W_hardware_evidence_summary_v1', status='PASS_WITH_LOCAL_SCOPE',
        audit_sha256=sha(AUDIT), manifest_sha256=sha(MANIFEST), script_sha256=sha(Path(__file__)),
        totals=totals, actual_case_geometry=cases, actual_full28_per_plane_rows=layers,
        limitations=[
            'Case mix intentionally samples four categories, not their production frequency.',
            'Row-plane work is not a cycle metric or an FP32/native comparison.',
            'The additional forward retains all spatial rows, channels, T and all 78 current BN.',
            'Only actual sampled integer rows traverse RTL; scale, BN, SRAM and full graph do not.',
            'Formal M quality failure remains; correct local hardware does not make M deployable.',
            'Existing modeled ~11.08% arithmetic PE / ~9.84% operator-service reduction is conditional, not measured network speedup; future scorer/BN reorder/feed costs are unresolved.'])
    out = ROOT/'hardware/rtl_results/r30_bridge_v1/ACTUAL_FINAL_W_SUMMARY_V1.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'summary_sha256': sha(out), 'totals': totals}))


if __name__ == '__main__':
    main()
