"""Independent conservation identities, provenance and capacity16 regression."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

BASE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = BASE / 'c1_accounting_checks.json'
    assert not output.exists()
    raw = (BASE / 'c1_input.bin').read_bytes()
    header = struct.unpack('<8s8I2Q', raw[:56])
    assert header[:9] == (b'C1UNION1', 10, 15, 20, 768, 768, 3000, 6912, 16)
    masks = np.frombuffer(raw, dtype='<u2', offset=56+3000*96).reshape(3000,432)
    lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    support_count = int(lut[masks.view(np.uint8)].sum())
    nonempty_rows = int(np.any(masks != 0, axis=1).sum())
    assert support_count == 2455667 and nonempty_rows == 3000
    original = json.loads((BASE / 'c1_attempt_r1/result.json').read_text())['points']
    sensitivity = json.loads((BASE / 'c1_cache_attempt_r1/result.json').read_text())['points']
    key = lambda p: (p['mode'], p['value_memory_ports'])
    old = {key(p): p for p in original}
    regression = 0
    for p in sensitivity:
        if p['coefficient_cache_vectors'] == 16:
            assert {k:v for k,v in p.items() if k != 'coefficient_cache_vectors'} == old[key(p)]
            regression += 1
    assert regression == 10
    for p in original + sensitivity:
        c = p['counts']
        # Each original active source incidence is either issued directly,
        # inherited from an actual parent, or supplied by a common-vector reuse.
        assert c['coefficient_vector_reads'] + c['parent_inherited_coefficients'] + c['common_coefficient_reuse_with_M_rebuild'] == support_count*8
        assert c['coefficient_vector_reads'] == c['residual_direct_coefficient_issues'] + c['common_coefficient_issues']
        assert c['source_vector_adds'] + c['destination_vector_adds'] == (
            c['coefficient_vector_reads'] + c['parent_hits'] + c['common_scatter_consumers']
            - c['common_G_tap_builds'] - nonempty_rows*8)
        assert c['coefficient_dma_beats'] == 6*c['coefficient_vector_fills']
        assert c['output_vector_commits'] == 3000*8
        assert c['zero_output_markers'] == (3000-nonempty_rows)*8
        assert c['output_dma_beats'] == nonempty_rows*8*6+(3000-nonempty_rows)*8
        assert p['total_cycles'] == sum(p[n] for n in ['input_and_preparation_cycles','common_phase_cycles','residual_execution_cycles','frontend_wait_cycles','final_commit_cycles'])
        assert p['diagnostic_lane_values_checked'] == 24000
    receipts = []
    for folder in ['c1_attempt_r1','c1_cache_attempt_r1']:
        receipt = json.loads((BASE/folder/'receipt.json').read_text())
        for name, digest in receipt['input_sha256'].items():
            assert sha(BASE/name) == digest
            snapshot = BASE/folder/'source_snapshot'/name
            if snapshot.exists():
                assert sha(snapshot) == digest
        assert sha(BASE/folder/'result.json') == receipt['result_sha256']
        assert sha(BASE/folder/'model') == receipt['binary_sha256']
        receipts.append({'attempt':folder,'result_sha256':receipt['result_sha256']})
    record = {'status':'PASS_CONSERVATION_AND_PROVENANCE_NOT_RTL_PROOF',
              'original_lowered_support_ones':support_count,
              'whole_output_rows':nonempty_rows,
              'conservation_points_checked':len(original)+len(sensitivity),
              'capacity16_identical_all_fields_to_original_points':regression,
              'attempts':receipts,
              'boundary':'Algebraic service-count conservation and exact repeat of capacity16 do not prove resource schedule feasibility, ASIC timing, actual trained weights, or FP32 equivalence.'}
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
