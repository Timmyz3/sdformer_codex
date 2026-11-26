#!/usr/bin/env python3
"""CPU-only final-W vector/source audit and existing RTL binary replay.

The 32 captured rows are not a replay of the complete SNN. The source forward
and its four head hashes remain runtime evidence bound to the audited final W.
This script independently reconstructs the two selected quantized matrices,
every selected signed dot, text/NPZ equivalence and exact RTL accepted-beat work.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPORTER_SHA = '1d418aa82b3d8ed49ae5166f972c20a184b725e4ee2ef7168a7f05b2e83f5aee'
AUDITOR_SHA = 'c31a3837c9408126a92bc418d507ba6372c53e07bb892578eb0db2f1628fd975'
MASK_SHA = '5a5018e92bfb632b56788dc20f7a3d7a41ce1b732e73f7e78245040fe2236d16'
GRAPH = 'flow_r30_joint_Q4_Q8_M_all12mlp_patch4_currentBN_v1'
TARGETS = (
    ('sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.fc1',
     'linear', (1536, 384), 30, 40),
    ('sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv1.0',
     'conv', (96, 96, 3, 3), 240, 320),
)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def child(folder, name):
    p = folder / name
    require(p.resolve().is_relative_to(folder.resolve()), 'Artifact stays inside source run')
    return p


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        v = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def quantize_numpy(weight):
    w = np.asarray(weight, dtype=np.float32).reshape(len(weight), -1)
    require(np.isfinite(w).all(), 'Finite original final weights')
    scale = np.max(np.abs(w), axis=1) / np.float32(127)
    divisor = np.where(scale == 0, np.float32(1), scale)
    q = np.clip(np.rint(w / divisor[:, None]), -127, 127).astype(np.int8)
    return q, scale


def expected_locations(contract, target, h, w):
    group = target.rsplit('.', 1)[0] if target.endswith('.fc1') else target
    g = contract['geometries']['full9'][group]
    require((g['H'], g['W']) == (h, w), 'Real full geometry')
    high = {tuple(x) for x in contract['evaluation_tiles'][group]}
    complete = [(r, c) for r in range(h // 9) for c in range(w // 9)]
    require(len(high) == g['Q8_complete_tiles'] and high <= set(complete),
            'Fixed restored complete quota')
    low = [x for x in complete if x not in high]
    require(high and low and h % 9 and w % 9, 'Four required distinct source categories')
    rh, ch = min(high)
    rl, cl = low[0]
    return [
        ('restored_complete_Q8', rh*9+4, ch*9+4, 'Q8'),
        ('low_complete_Q4', rl*9+4, cl*9+4, 'Q4'),
        ('right_tail_Q8', 0, w//9*9, 'Q8'),
        ('bottom_tail_Q8', h//9*9, 0, 'Q8'),
    ]


def inspect(a):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU-only audit')
    manifest_path = a.export_dir / 'FINAL_VECTOR_MANIFEST_V1.json'
    manifest, formal, audit, contract = map(read, (
        manifest_path, a.formal_result, a.actual_audit, a.mask_contract))
    freeze = read(a.bridge_freeze)
    require(freeze['schema'] == 'flow_r30_final_vector_bridge_freeze_v1',
            'Independent pre-result source freeze')
    for item in freeze['files']:
        require(sha(HERE.parents[1] / item['path']) == item['sha256'],
                'Frozen bridge source/binary bytes: ' + item['path'])
    require(sha(__file__) == freeze['auditor_sha256'], 'This exact independent auditor')
    require(sha(a.binary) == freeze['binary_sha256'], 'Exact compiled frozen RTL bridge')
    require(sha(HERE/'export_r30_final_vectors.py') == EXPORTER_SHA and
            sha(a.mask_contract) == MASK_SHA, 'Exporter and mask source bytes')
    require(manifest['schema'] == 'flow_r30_final_PE_vectors_v1' and
            manifest['status'] == 'FINAL_M_VECTOR_EXPORT_COMPLETE' and
            manifest['graph_id'] == formal['graph_id'] == audit['graph_id'] == GRAPH,
            'Complete one-forward export from exact graph')
    require(manifest['exporter_sha256'] == EXPORTER_SHA and
            manifest['formal_result_sha256'] == audit['source_result_sha256'] ==
            sha(a.formal_result) and manifest['actual_audit_sha256'] == sha(a.actual_audit) and
            audit['auditor_sha256'] == AUDITOR_SHA and audit['status'] == 'PASS' and
            audit['schema'] == 'flow_r30_independent_actual_execution_audit_v1',
            'Actual formal audit/source chain')
    require(audit['quality_analysis_allowed'] is True and
            audit['engineering_smoke'] is False and
            audit['counts'] == dict(optimizer_updates=512, quality_predictions=256,
                                    formal_model_forwards=1289), 'Completed formal scope')
    require(formal['schema'] == 'flow_r30_joint_recovery_v1' and
            formal['status'] == 'JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE' and
            formal['completed_optimizer_steps'] == 512 and
            formal['model_forwards'] == formal['expected_model_forwards'] == 1289 and
            formal['engineering_smoke'] is False, 'No smoke/incomplete formal substitution')
    require(manifest['model_forwards'] == manifest['expected_model_forwards'] == 1 and
            manifest['formal_model_forwards'] == 1289 and manifest['optimizer_updates'] == 0 and
            manifest['GT_values_read'] is False and manifest['quality_computed'] is False and
            manifest['diagnostic_quantization_calls'] == 2, 'Only one diagnostic forward')
    require(manifest['mask_contract_sha256'] == audit['source_mask_contract_sha256'] == MASK_SHA
            and manifest['sources'] == formal['sources'], 'Same exact mask and source package')
    weights = child(a.formal_result.parent, formal['weights_file'])
    require(sha(weights) == formal['weights_sha256'] == audit['checkpoint_sha256'] ==
            manifest['checkpoint_sha256'], 'Actual final checkpoint bytes')
    cp = torch.load(weights, map_location='cpu', weights_only=True)
    state = cp['model_state_dict']
    require(cp['schema'] == 'flow_r30_joint_checkpoint_v1' and cp['arm'] == 'joint' and
            cp['completed_optimizer_steps'] == 512 and len(state) == 477 and
            cp['sources'] == formal['sources'] and cp['recipe'] == formal['recipe'] and
            cp['joint_contract'] == formal['joint_contract'], 'Final state provenance')
    state_sha = state_digest(state)
    require(state_sha == cp['final_state_sha256'] == formal['final_state_sha256'] ==
            audit['final_state_sha256'] == manifest['source_state_sha256'] ==
            manifest['final_state_sha256'], 'Independent full 477-tensor state digest')
    parity = formal['final_parity']['M']
    require(manifest['mask_layout'] == parity['mask_layout'] and
            manifest['input_target_id'] == parity['input_target_id'] and
            manifest['input_evidence'] == parity['input_evidence'], 'Fixed formal parity input')
    source_record = parity['training_deployment']['actual_record']
    require(manifest['actual_BN_calls'] == source_record['actual_BN_calls'] and
            len(manifest['actual_BN_calls']) == 78 and
            set(manifest['actual_BN_calls'].values()) == {1}, 'Original 78 BN calls')
    require(set(manifest['actual_affine_calls']) == set(source_record['actual_target_calls']) and
            len(manifest['actual_affine_calls']) == 28 and
            all(len(v) == 1 for v in manifest['actual_affine_calls'].values()) and
            manifest['work'] == source_record['work'], 'Same actual 28 affine workloads')
    psn_names = {k[len('PSN::'):] for k in parity['training_deployment']['checks']
                 if k.startswith('PSN::')}
    require(set(manifest['actual_PSN_calls']) == psn_names and len(psn_names) == 105 and
            set(manifest['actual_PSN_calls'].values()) == {1}, 'All executed 105 PSN calls')
    head_file = child(a.formal_result.parent, parity['heads_file'])
    require(sha(head_file) == parity['heads_sha256'], 'Formal parity head archive bytes')
    with np.load(head_file, allow_pickle=False) as heads:
        require(set(heads.files) == {'head'+str(i) for i in range(4)}, 'Four formal heads')
        for key in heads.files:
            h = heads[key]
            rec = manifest['heads'][key]
            reference = parity['training_deployment']['checks'][key]
            require(h.dtype == np.float32 and h.shape == (1, 2, 480, 640) and
                    np.isfinite(h).all() and rec['bitwise_equal'] is True and
                    digest(h) == rec['actual_sha256'] == rec['formal_sha256'] ==
                    reference['actual_sha256'] == reference['reference_sha256'],
                    'New runtime head hash equals actual archived formal array: ' + key)
    paths = {}
    for kind, filekey, hashkey in (
            ('vectors', 'vectors_file', 'vectors_sha256'),
            ('expected', 'expected_file', 'expected_sha256'),
            ('payload', 'payload_file', 'payload_file_sha256')):
        paths[kind] = child(a.export_dir, manifest[filekey])
        require(sha(paths[kind]) == manifest[hashkey], 'Actual exported ' + kind + ' bytes')
    lines, elines = paths['vectors'].read_text().splitlines(), paths['expected'].read_text().splitlines()
    require(len(lines) == 97 and len(elines) == 33 and lines[0] == elines[0] == '32',
            'Exactly 32 text transactions')
    arrays, expected_cases, unique = {}, [], set()
    with np.load(paths['payload'], allow_pickle=False) as z:
        require(len(z.files) == 128 and len(manifest['cases']) == 32, '128 captured payloads')
        for target, kind, shape, h, w in TARGETS:
            weight = state[target+'.weight'].detach().cpu().numpy()
            require(weight.shape == shape and weight.dtype == np.float32, 'Exact real target weight shape')
            q, scale = quantize_numpy(weight)
            capture = manifest['capture_sources'][target]
            expected_input = [10, 1, h, w, shape[1]] if kind == 'linear' else [10, 1, shape[1], h, w]
            require(capture['kind'] == kind and capture['input_shape'] == expected_input and
                    capture['q8_shape'] == list(q.shape) and
                    capture['source_weight_sha256'] == digest(weight) and
                    capture['full_q8_sha256'] == digest(q) and
                    capture['full_scale_sha256'] == digest(scale) and
                    capture['diagnostic_quantization_calls'] == 1 and
                    capture['actual_work'] == manifest['work'][target], 'Actual selected producer binding')
            for t in (0, 9):
                for category, y, x, mode in expected_locations(contract, target, h, w):
                    for start in (0, shape[0]-16):
                        i = len(expected_cases)
                        record = manifest['cases'][i]
                        expected = dict(case=i, target=target, category=category,
                                        coordinate=[t, 0, y, x], row_index=t*h*w+y*w+x,
                                        output_start=start, outputs=16, K=q.shape[1], mode=mode)
                        require(all(record[k] == v for k, v in expected.items()), 'Fixed case identity/order')
                        identity = (target, t, y, x, start)
                        require(identity not in unique and start+16 <= shape[0], '32 unique valid output slices')
                        unique.add(identity)
                        p = {}
                        for suffix, dtype, ashape in (
                                ('x', np.int8, (q.shape[1],)), ('q8', np.int8, (16, q.shape[1])),
                                ('scale', np.float32, (16,)), ('actual_int32', np.int32, (16,))):
                            info = record['payloads'][suffix]
                            key = f'case{i:02d}__{suffix}'
                            require(info['key'] == key and key not in arrays, 'Unique exact payload key')
                            v = z[key]
                            require(v.dtype == dtype and v.shape == ashape and
                                    info['dtype'] == str(v.dtype) and info['shape'] == list(v.shape) and
                                    info['sha256'] == digest(v), 'Actual payload metadata/content')
                            arrays[key] = True
                            p[suffix] = v
                        require(np.isin(p['x'], (0, 1)).all() and
                                np.array_equal(p['q8'], q[start:start+16]) and
                                np.array_equal(p['scale'], scale[start:start+16]),
                                'Binary actual source and independently requantized final W')
                        effective = p['q8'].astype(np.int64)
                        if mode == 'Q4':
                            effective = (effective // 16) * 16
                        golden = effective @ p['x'].astype(np.int64)
                        require(np.array_equal(golden, p['actual_int32'].astype(np.int64)),
                                'Independent signed integer dot equals captured actual int32')
                        require(lines[1+i*3] == f'{q.shape[1]} {int(mode == "Q8")}' and
                                np.array_equal(np.fromstring(lines[2+i*3], sep=' ', dtype=np.int64), p['x']) and
                                np.array_equal(np.fromstring(lines[3+i*3], sep=' ', dtype=np.int64), p['q8'].ravel()) and
                                np.array_equal(np.fromstring(elines[i+1], sep=' ', dtype=np.int64), p['actual_int32']),
                                'Text consumed by RTL equals complete NPZ payload')
                        expected_cases.append(expected)
        require(set(z.files) == set(arrays), 'Every payload independently visited')
    del cp, state
    run = subprocess.run([str(a.binary.resolve()), str(paths['vectors'].resolve()),
                          str(paths['expected'].resolve())], capture_output=True, text=True, timeout=120)
    require(run.returncode == 0, 'Actual RTL bridge failed: ' + run.stderr[-2000:])
    rtl = json.loads(run.stdout)
    require(rtl['schema'] == 'flow_r30_captured_int32_PE_RTL_v1' and rtl['status'] == 'PASS' and
            rtl['case_count'] == 142 and rtl['captured_expected_values_checked'] == 512 and
            len(rtl['runs']) == 4 and rtl['malformed_protocol_rejections'] == 6, 'Complete bounded RTL replay')
    sizes = [(k, bits) for k in (1, 16, 96, 128, 129, 256, 384, 768, 864, 1536, 3072)
             for bits in (4, 8) for _ in range(5)]
    sizes += [(r['K'], int(r['mode'][1:])) for r in expected_cases]
    totals = dict(binary_popcount128=sum(((k+127)//128)*bits for k, bits in sizes),
                  ordinary_gateadd16=sum((k+15)//16 for k, _ in sizes))
    for r in rtl['runs']:
        require(r['accepted_beats'] == totals[r['engine']] and r['retired_rows'] == 142 and
                r['checked_int32_values'] == 142*16 and
                (r['stalls'] or r['cycles'] == r['accepted_beats']+1), 'Independent RTL beat conservation')
    return dict(schema='flow_r30_final_W_vector_RTL_independent_audit_v1', status='PASS',
                auditor_sha256=sha(__file__), source_freeze_sha256=sha(a.bridge_freeze),
                source_manifest_sha256=sha(manifest_path), source_result_sha256=sha(a.formal_result),
                source_actual_audit_sha256=sha(a.actual_audit), checkpoint_sha256=sha(weights),
                state_sha256=state_sha, mask_sha256=MASK_SHA,
                vectors_sha256=sha(paths['vectors']), expected_sha256=sha(paths['expected']),
                payload_sha256=sha(paths['payload']), binary_sha256=sha(a.binary),
                actual_case_count=32, unique_coordinate_output_cases=len(unique), actual_payloads=128,
                captured_int32_independent_dots=512, formal_archived_heads_checked=4,
                source_model_forwards=1, new_model_forwards=0, GPU_used=False, GT_values_read=False,
                actual_cases=expected_cases, RTL=rtl,
                limitations=[
                    'Original prehook rows are source-bound captures, not independent full-network recomputation.',
                    'Four new head hashes are runtime evidence matched against actual saved formal head arrays.',
                    'Only integer PE datapaths are replayed; scale, FP32 BN, SRAM and network schedule are outside RTL.',
                    'R26 V1 vectors remain separate historical evidence; this receipt uses final R30 W only.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir', 'formal-result', 'actual-audit', 'mask-contract', 'binary',
                 'bridge-freeze', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    require(not a.output.exists(), 'Fresh independent actual receipt')
    result = inspect(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'actual_case_count',
                                           'captured_int32_independent_dots', 'new_model_forwards')}))


if __name__ == '__main__':
    main()
