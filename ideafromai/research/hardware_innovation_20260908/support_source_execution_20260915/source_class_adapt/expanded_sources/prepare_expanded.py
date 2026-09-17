"""Python 3.12 / NumPy: fixed P0..31 from all 32 existing training records.

No training, FP32-gate replay, activity selection, or change to source parameters.
Only this directory is written. The existing ordinary and adapted static graphs
are retained byte-for-byte; the C++ fixture contains X and parameters, no gates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import io
import json
from pathlib import Path
import struct
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ADAPT = HERE.parent
SOURCE = ADAPT.parent
BASE = SOURCE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from prepare_sources_numpy312 import read_torch, quant, nearest


def read_bin(path):
    stream = io.BytesIO(path.read_bytes())
    stream.read(200 + 80 + 1536)
    for _ in range(2):
        for _ in range(2):
            n, = struct.unpack('<I', stream.read(4))
            stream.read(8*n)
        stream.read(24 + 96 + 96)
    split = stream.tell()
    prefix = path.read_bytes()[:split]
    count, = struct.unpack('<I', stream.read(4))
    cases = []
    for _ in range(count):
        real, length = struct.unpack('<II', stream.read(8))
        name = stream.read(length).decode()
        x = np.frombuffer(stream.read(960*4), dtype='<i4').reshape(96, 10).copy()
        cases.append((name, real, x))
    assert stream.read() == b''
    return prefix, cases


def static_prefix(A, tau, D, profiles, adapted=None):
    f = io.BytesIO()
    f.write(A.astype('<i2').tobytes())
    f.write(tau.astype('<i8').tobytes())
    f.write(D.astype('u1').tobytes())
    for order, q in enumerate(profiles):
        cl = q['class_nodes64'] if adapted is None else adapted[
            'class_nodes_natural' if order == 0 else 'class_nodes_entropy']
        for v in (q['code_nodes64'], cl):
            assert len(v) <= 2180
            f.write(struct.pack('<I', len(v)))
            f.write(v.astype('<u8').tobytes())
        roots = q['roots'].reshape(12).copy()
        canonical = q['response_canonical_code']
        if adapted is not None:
            roots[6:] = adapted['roots'][order]
            canonical = adapted['canonical']
        f.write(roots.astype('<u2').tobytes())
        f.write(canonical.astype('u1').tobytes())
        rank = np.full((6, 16), 15, dtype='u1')
        for g in range(6):
            rank[g, q[f'variables_g{g}']] = np.arange(len(q[f'variables_g{g}']), dtype='u1')
        f.write(rank.tobytes())
    return f.getvalue()


def write_bin(path, prefix, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(prefix)
        f.write(struct.pack('<I', len(cases)))
        for name, real, x in cases:
            encoded = name.encode()
            f.write(struct.pack('<II', real, len(encoded)))
            f.write(encoded)
            f.write(x.astype('<i4').tobytes())
    actual_prefix, actual_cases = read_bin(path)
    assert actual_prefix == prefix and len(actual_cases) == len(cases)
    for a, b in zip(actual_cases, cases):
        assert a[:2] == b[:2] and np.array_equal(a[2], b[2])


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Run with Python 3.12.')
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=BASE/'algorithm/support_training.tar.gz')
    ap.add_argument('--student-root', type=Path, default=BASE/'algorithm/support_training')
    args = ap.parse_args()
    with tarfile.open(args.archive, 'r:gz') as tar:
        cache = read_torch(io.BytesIO(tar.extractfile('support_training/teacher_cache.pt').read()))
    parameters = read_torch(args.student_root/'forced_code_parameters.pt')
    run = json.loads((args.student_root/'run.json').read_text())
    assert len(cache) == 32 and [r['file'] for r in cache] == run['train_files']
    A, bias, center, theta = [np.array(parameters[k], copy=True) for k in ('A', 'bias', 'center', 'theta')]
    Aq = quant(A, 12, 16).astype(np.int16)
    threshold_real = (theta.astype(np.float64) + center.astype(np.float64) - bias.astype(np.float64)).reshape(10)
    tau = quant(threshold_real, 28, 48)
    D = np.asarray(parameters['dictionary'], dtype=np.uint8)
    assert np.array_equal(D, np.load(args.student_root/'dictionary.npy'))
    cases, records, xs, raw_gates, projected_gates, indices = [], [], [], [], [], []
    prefix_abs_max = 0
    for frame, record in enumerate(cache):
        x = np.asarray(record['x'], dtype=np.float32)
        assert x.shape == (10, 512, 96)
        xq = quant(x[:, :32], 16, 24).astype(np.int32)
        accum = np.zeros((10, 32, 96), dtype=np.int64)
        for s in range(10):
            accum += Aq[:, s, None, None].astype(np.int64) * xq[s].astype(np.int64)[None]
            prefix_abs_max = max(prefix_abs_max, int(np.abs(accum).max()))
        assert np.array_equal(accum, np.einsum('ts,spc->tpc', Aq.astype(np.int64), xq.astype(np.int64)))
        gate = (accum >= tau[:, None, None]).astype(np.uint8)
        projected, code, _ = nearest(gate, D)
        xs.append(xq)
        raw_gates.append(gate)
        projected_gates.append(projected)
        indices.append(code.transpose(1, 0, 2))
        for position in range(32):
            cases.append((f'train{frame}_p{position}', 1, xq[:, position, :].T.copy()))
        records.append(dict(frame_index=frame, file=record['file'], sample_indices=[0, 31],
                            role='pair_selection' if frame == 0 else 'unselected_training_probe',
                            X_q16_min=int(xq.min()), X_q16_max=int(xq.max())))
        print('EXPORTED', frame, record['file'], flush=True)
    assert prefix_abs_max < (1 << 47)
    all_signed24_bound = int(np.abs(Aq.astype(np.int64)).sum(1).max()) * (1 << 23)
    assert all_signed24_bound < (1 << 47)
    diagnostic = [
        ('diagnostic_zero', 0, np.zeros((96, 10), dtype=np.int32)),
        ('diagnostic_positive_extreme', 0, np.full((96, 10), (1 << 23)-1, dtype=np.int32)),
        ('diagnostic_signed_extreme', 0, np.tile(np.array([-(1 << 23), (1 << 23)-1]*5, dtype=np.int32), (96, 1))),
    ]
    cases.extend(diagnostic)
    data = dict(X_q16=np.stack(xs), A_q12=Aq, threshold_q28=tau, D=D,
                A_fp32=A, bias_fp32=bias, center_fp32=center, theta_fp32=theta,
                threshold_real=threshold_real, raw_g_int=np.stack(raw_gates),
                projected_g_int=np.stack(projected_gates), code_index_int=np.stack(indices),
                frame_file=np.asarray([r['file'] for r in cache]), sample_index=np.arange(32, dtype=np.int32),
                frame_role=np.asarray([r['role'] for r in records]),
                case_name=np.asarray([c[0] for c in cases[:1024]]),
                reconstructed_spatial_index=np.rint(np.linspace(0, 19199, 512, dtype=np.float32)).astype(np.int64)[:32],
                spatial_index_status=np.array('reconstructed from original sampling rule, not coordinates stored in cache'),
                dut_input_fields=np.asarray(['X_q16', 'A_q12', 'threshold_q28', 'D']),
                oracle_fields=np.asarray(['raw_g_int', 'projected_g_int', 'code_index_int']),
                source_boundary=np.array('32 training-cache frames; fixed sample 0..31, Q12/Q16/Q28; no new AEE'))
    # The pre-existing 64 inputs and their integer oracles must remain identical.
    old_npz = np.load(SOURCE/'source_cases.npz')
    for key in ('X_q16', 'raw_g_int', 'projected_g_int', 'code_index_int', 'frame_file'):
        assert np.array_equal(data[key][:2], old_npz[key]), key
    for key in ('A_q12', 'threshold_q28', 'D', 'sample_index'):
        assert np.array_equal(data[key], old_npz[key]), key
    old_prefix, old_cases = read_bin(SOURCE/'source.bin')
    adapt_prefix, adapt_cases = read_bin(ADAPT/'source.bin')
    assert len(old_cases) == len(adapt_cases) == 67
    for i, original in enumerate(old_cases):
        other = adapt_cases[i]
        expanded = cases[i] if i < 64 else cases[1024+i-64]
        assert original[:2] == other[:2] == expanded[:2]
        assert np.array_equal(original[2], other[2]) and np.array_equal(original[2], expanded[2])
    profiles = [np.load(SOURCE/name) for name in ('prefix_tables.npz', 'prefix_tables_entropy.npz')]
    adapted = np.load(ADAPT/'adapt_tables.npz')
    regenerated_old_prefix = static_prefix(Aq, tau, D, profiles)
    regenerated_adapt_prefix = static_prefix(Aq, tau, D, profiles, adapted)
    assert regenerated_old_prefix == old_prefix
    assert regenerated_adapt_prefix == adapt_prefix
    np.savez_compressed(HERE/'source_cases.npz', **data)
    write_bin(HERE/'source.bin', regenerated_adapt_prefix, cases)
    write_bin(HERE/'old_class/source.bin', regenerated_old_prefix, cases)
    assert 'torch' not in sys.modules
    manifest = dict(python=sys.version.split()[0], numpy=np.__version__, torch_imported=False,
                    archive=str(args.archive), cache_member='support_training/teacher_cache.pt',
                    forced_parameters=str(args.student_root/'forced_code_parameters.pt'),
                    shape={k:list(data[k].shape) for k in ('X_q16', 'A_q12', 'threshold_q28', 'D', 'raw_g_int', 'code_index_int')},
                    real_cases=1024, diagnostic_cases=3, total_cases=len(cases),
                    calibration_frame_indices=[0], unselected_training_frame_indices=list(range(1, 32)),
                    selected_samples_per_frame=list(range(32)), no_activity_selection=True,
                    no_training=True, no_clipping=True, no_new_AEE=True, fp32_gate_replay=False,
                    first64_X_int32_words_equal=64*960, first64_X_input_bytes_equal=64*960*4,
                    first64_integer_oracles_equal=True, three_diagnostics_exactly_preserved=True,
                    original_static_prefix_bytes_equal=True, adapted_static_prefix_bytes_equal=True,
                    prefix_abs_max=prefix_abs_max, fixed_A_all_signed24_abs_bound=all_signed24_bound,
                    X_q16_min=int(data['X_q16'].min()), X_q16_max=int(data['X_q16'].max()),
                    raw_source_gold_in_DUT_binary=False,
                    profile_sources=['../../prefix_tables.npz', '../../prefix_tables_entropy.npz', '../adapt_tables.npz'],
                    output={'adapted':'source.bin', 'nearest_response':'old_class/source.bin', 'archive':'source_cases.npz'},
                    frames=records)
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PASS 1024 real + 3 diagnostic; original 64 X and integer oracles exact; both static prefixes exact', flush=True)


if __name__ == '__main__':
    main()
