"""C2 source-order finite shared-sum reference, an implementation variant."""
import sys
sys.dont_write_bytecode = True
from collections import Counter, OrderedDict
from pathlib import Path
import json
import numpy as np
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import screen_fc2_signatures as full


def simulate(sig, T, D, phi=None):
    cache, c = OrderedDict(), Counter()
    active_outputs = 0
    Y = np.zeros((T, phi.shape[1]), dtype=np.int64) if phi is not None else None

    def scatter(q, value, label):
        nonlocal active_outputs
        assert q
        c['destination_vector_adds'] += (q & active_outputs).bit_count()
        c['destination_value_accepts'] += q.bit_count()
        c[label+'_destination_accepts'] += q.bit_count()
        active_outputs |= q
        if Y is not None:
            for t in range(T):
                if q >> t & 1:
                    Y[t] += value

    def drain(q, value, eviction):
        c['partial_vector_reads_for_drain'] += 1
        c['partial_segments_drained'] += 1
        c['eviction_drains'] += eviction
        scatter(q, value, 'partial')

    for cid, raw in enumerate(sig):
        q = int(raw)
        c['source_signature_inspections'] += 1
        if q == 0:
            continue
        c['source_coefficient_vector_reads'] += 1
        value = phi[cid] if phi is not None else None
        if q.bit_count() == 1:
            scatter(q, value, 'single_time')
            c['single_time_source_bypasses'] += 1
            continue
        c['class_key_comparisons'] += len(cache)
        if q in cache:
            c['partial_update_adds'] += 1
            c['partial_vector_update_reads'] += 1
            c['partial_vector_update_writes'] += 1
            if phi is not None:
                cache[q] += value
            cache.move_to_end(q)
        else:
            if len(cache) == D:
                old_q, old_value = cache.popitem(last=False)
                drain(old_q, old_value, True)
            cache[q] = value.copy() if phi is not None else None
            c['partial_vector_assignments'] += 1
        c['peak_dictionary_vectors'] = max(c['peak_dictionary_vectors'], len(cache))
        assert len(cache) <= D
    while cache:
        q, value = cache.popitem(last=False)
        drain(q, value, False)
    c['binary_vector_adds'] = c['partial_update_adds'] + c['destination_vector_adds']
    c['output_vectors_ever_initialized'] = active_outputs.bit_count()
    c['original_destination_commits'] = T
    assert c['partial_vector_assignments'] == c['partial_segments_drained']
    if Y is not None:
        bits = ((np.asarray(sig)[None, :] >> np.arange(T)[:, None]) & 1).astype(np.int64)
        assert np.array_equal(Y, bits @ phi)
        c['diagnostic_lane_outputs_verified'] = T*phi.shape[1]
    return c


def main():
    out = BASE/'source_order_r1.json'
    assert not out.exists()
    plan_path = BASE/'source_order_plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = full.digest(plan_path)
    sealed_path = BASE/'fc2_signatures_r1.json'
    sealed = json.loads(sealed_path.read_text())
    directed = []
    for sig in ([0]*16, [3,5,6,7,9,10,3,1,3,2,5,4]*3,
                [3,1,3,2,3,1,3], [7,7,3,3,1,7]):
        phi = full.diagnostic_phi(len(sig))
        if len(sig) >= 2:
            phi[1] = -phi[0]
        for D in plan['capacities']:
            directed.append({'signatures':sig,'D':D,'counts':dict(simulate(sig,10,D,phi))})
    arrays = full.sources()
    results = []
    for row in sealed['layers']:
        spec, S = arrays[row['module']]
        P, C, T = row['P'], row['C'], row['T']
        sig = np.tensordot(1 << np.arange(T, dtype=np.int64), S.reshape(T,P,C), axes=(0,0))
        positions = set(row['diagnostics']['spatial_positions'])
        phi = full.diagnostic_phi(C)
        points = {}
        for D in plan['capacities']:
            total = Counter()
            for p in range(P):
                c = simulate(sig[p], T, D, phi if p in positions else None)
                for key,value in c.items():
                    if key.startswith('peak_'):
                        total[key] = max(total[key], value)
                    else:
                        total[key] += value
            assert total['source_coefficient_vector_reads'] == row['aggregate']['source_coefficient_vector_reads_per_output_slice_all_variants']
            assert total['original_destination_commits'] == P*T
            points[str(D)] = {
                'counts_per_output_slice_vector_or_per_output_channel_adds':dict(total),
                'adds_over_direct_FTP':total['binary_vector_adds']/row['aggregate']['direct_FTP_binary_adds_per_output_channel'],
                'adds_over_full_flat_groups':total['binary_vector_adds']/row['aggregate']['flat_total_binary_adds_per_output_channel'],
                'worst_reserved_working_vectors_including_two_pipeline_registers':D+2,
                'dictionary_key_and_valid_bits_excluding_LRU_and_wide_data':D*(T+1),
            }
            print(row['module'],'D',D,'adds/direct',points[str(D)]['adds_over_direct_FTP'],flush=True)
        results.append({'module':row['module'],'P':P,'C':C,'H':row['H'],'T':T,'points':points})
    assert full.digest(plan_path) == plan_sha
    report = {'date':'2026-09-07','status':'EXPLORATORY_SOURCE_ORDER_IMPLEMENTATION_REFERENCE',
              'plan':plan,'plan_sha256':plan_sha,'script_sha256':full.digest(Path(__file__)),
              'imported_fc2_script_sha256':full.digest(BASE/'screen_fc2_signatures.py'),
              'same_fc2_receipt_sha256':full.digest(sealed_path),'capture_sha256':full.EXPECTED,
              'directed_checks':directed,'layers':results,
              'limits':['Source ordering is preserved; simultaneous bank issue and dictionary hazard handling are not modeled.',
                        'Partial update reads/writes can be register-file activity; these are not claimed SRAM transactions.',
                        'Comparison against full flat grouping isolates segmentation cost; this does not claim a novel shared-sum algebra.',
                        'Signatures are available from captured fullT flags; producer ordering/BN buffering remains the original scope.'],
              'claim_boundary':{'RTL_speedup':False,'PPA':False,'new_AEE':False,'frozen_FP32_equivalence':False}}
    with out.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
