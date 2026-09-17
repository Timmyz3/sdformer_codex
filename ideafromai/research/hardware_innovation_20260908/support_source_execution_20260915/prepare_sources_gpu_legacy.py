# Historical GPU replay used Python 3.10; retained for audit, not the supported current entry point.
"""Replay fixed forced-student source; no training or parameter mutation.

Teacher cache supplies X only. Student.source supplies the floating reference.
The fixed Q12/Q16/Q28 candidate has its own raw/projected gates and statistics.
Only first two training records, sampled-position indices 0:32, enter RTL cases.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import io
import json
import sys
import tarfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / 'algorithm'))
from train_exact_support_probe import Student


def nearest_numpy(gate, dictionary):
    shape = gate.shape
    grouped = gate.reshape(-1, 6, 16).astype(np.int16)
    d = dictionary.astype(np.int16)
    distance = (grouped.sum(-1)[..., None] + d.sum(-1)[None]
                - 2 * np.einsum('ngc,gkc->ngk', grouped, d))
    index = distance.argmin(-1)  # Lowest code index on a Hamming tie.
    selected = d[np.arange(6)[None], index].reshape(shape).astype(np.uint8)
    return selected, index.reshape(shape[0], shape[1], 6), distance.min(-1)


def round_admit(value, fractional, width, label):
    rounded = np.rint(value.astype(np.float64) * (2 ** fractional)).astype(np.int64)
    assert rounded.min() >= -(1 << (width - 1)), (label, int(rounded.min()))
    assert rounded.max() < (1 << (width - 1)), (label, int(rounded.max()))
    return rounded


def opportunity(gate, dictionary):
    grouped = gate.reshape(-1, 6, 16).astype(np.int16)
    _, _, distance = nearest_numpy(gate, dictionary)
    pop = grouped.sum(-1)
    exact = distance == 0
    useful = exact & (pop >= 2)
    return dict(groups=int(pop.size), active_bits=int(pop.sum()),
                exact_groups=int(exact.sum()), useful_exact_groups=int(useful.sum()),
                exact_saved_adds_unpriced=int(np.where(useful, pop - 1, 0).sum()),
                projection_hamming_bits=int(distance.sum()))


def cache_load(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.getmember('support_training/teacher_cache.pt')
        return torch.load(io.BytesIO(tar.extractfile(member).read()),
                          map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path,
                        default=BASE / 'algorithm/support_training.tar.gz')
    parser.add_argument('--student-root', type=Path,
                        default=BASE / 'algorithm/support_training')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True  # Original run's setting.
    torch.backends.cudnn.allow_tf32 = True
    cache = cache_load(args.archive)
    state = torch.load(args.student_root / 'forced_code_parameters.pt',
                       map_location='cpu', weights_only=False)
    run = json.loads((args.student_root / 'run.json').read_text())
    assert [r['file'] for r in cache] == run['train_files']
    assert len(cache) == 32
    assert all(tuple(r['x'].shape) == (10, 512, 96) for r in cache)
    assert set(r['file'] for r in cache).isdisjoint(run['validation_files'])
    A = state['A'].numpy().copy()
    bias = state['bias'].numpy().copy()
    center = state['center'].numpy().copy()
    theta = state['theta'].numpy().copy()
    D = state['dictionary'].numpy().astype(np.uint8)
    assert np.array_equal(D, np.load(args.student_root / 'dictionary.npy'))
    Aq = round_admit(A, 12, 16, 'A').astype(np.int16)
    # Fold stored FP32 constants in float64, then one ties-to-even Q28 rounding.
    threshold_real = (theta.astype(np.float64) + center.astype(np.float64)
                      - bias.astype(np.float64)).reshape(10)
    threshold = round_admit(threshold_real, 28, 48, 'threshold')
    student = SimpleNamespace(kind='forced_code', **{
        k: state[k].to(args.device) for k in ['A', 'bias', 'center', 'theta', 'dictionary']})
    cpu_student = SimpleNamespace(kind='forced_code', **{
        k: state[k].cpu() for k in ['A', 'bias', 'center', 'theta', 'dictionary']})

    per_frame = []
    selected = []
    observed = dict(x_min=np.inf, x_max=-np.inf, xq_min=2**63-1, xq_max=-2**63,
                    accumulation_min=2**63-1, accumulation_max=-2**63,
                    integer_prefix_abs_max=0, fp_margin_abs_min=np.inf)
    totals = dict(raw_gate_differences=0, projected_gate_differences=0,
                  projected_code_differences=0, teacher_gate_differences=0,
                  fp32_cpu_gate_differences=0, fp32_no_tf32_gate_differences=0,
                  fp32_cpu_projected_differences=0, fp32_no_tf32_projected_differences=0)
    margin_bins = {str(x): 0 for x in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}
    for fi, record in enumerate(cache):
        X = record['x'].numpy().copy()
        Xq = round_admit(X, 16, 24, 'X').astype(np.int32)
        with torch.no_grad():
            xx = record['x'].to(args.device)
            projected_t, raw_t, _ = Student.source(student, xx, differentiable=False)
            h = torch.addmm(student.bias, student.A, xx.reshape(10, -1)) - student.center
            margin = (h - student.theta).reshape_as(xx).cpu().numpy()
            raw = raw_t.cpu().numpy().astype(np.uint8)
            projected = projected_t.cpu().numpy().astype(np.uint8)
            cpu_projected, cpu_raw, _ = Student.source(cpu_student, record['x'], False)
            torch.backends.cuda.matmul.allow_tf32 = False
            strict_projected, strict_raw, _ = Student.source(student, xx, False)
            torch.backends.cuda.matmul.allow_tf32 = True
        np_projected, index, _ = nearest_numpy(raw, D)
        assert np.array_equal(projected, np_projected)
        # This explicit source-time loop is also the intended RTL accumulation order.
        accum = np.zeros((10, 512, 96), dtype=np.int64)
        prefix_abs = 0
        for s in range(10):
            accum += Aq[:, s, None, None].astype(np.int64) * Xq[s].astype(np.int64)[None]
            prefix_abs = max(prefix_abs, int(np.abs(accum).max()))
        assert prefix_abs < 2**47
        assert np.array_equal(accum, np.einsum('ts,spc->tpc', Aq.astype(np.int64),
                                             Xq.astype(np.int64)))
        integer_margin = accum - threshold[:, None, None]
        raw_i = (integer_margin >= 0).astype(np.uint8)
        projected_i, index_i, _ = nearest_numpy(raw_i, D)
        changed = raw != raw_i
        row = dict(index=fi, file=record['file'], gate_values=int(raw.size),
                   raw_gate_differences=int(changed.sum()),
                   projected_gate_differences=int((projected != projected_i).sum()),
                   projected_code_differences=int((index != index_i).sum()),
                   teacher_gate_differences=int((raw != record['gate'].numpy()).sum()),
                   fp32_cpu_gate_differences=int((raw != cpu_raw.numpy()).sum()),
                   fp32_no_tf32_gate_differences=int((raw != strict_raw.cpu().numpy()).sum()),
                   fp32_cpu_projected_differences=int((projected != cpu_projected.numpy()).sum()),
                   fp32_no_tf32_projected_differences=int((projected != strict_projected.cpu().numpy()).sum()),
                   changed_fp_margin_abs_max=float(np.abs(margin[changed]).max()) if changed.any() else 0.,
                   changed_fp_margin_abs_min=float(np.abs(margin[changed]).min()) if changed.any() else None,
                   raw_fp32=opportunity(raw, D), raw_integer=opportunity(raw_i, D),
                   projected_fp32=opportunity(projected, D),
                   projected_integer=opportunity(projected_i, D))
        for k in totals: totals[k] += row[k]
        for cutoff in margin_bins:
            margin_bins[cutoff] += int((np.abs(margin) <= float(cutoff)).sum())
        observed['x_min'] = min(observed['x_min'], float(X.min()))
        observed['x_max'] = max(observed['x_max'], float(X.max()))
        observed['xq_min'] = min(observed['xq_min'], int(Xq.min()))
        observed['xq_max'] = max(observed['xq_max'], int(Xq.max()))
        observed['accumulation_min'] = min(observed['accumulation_min'], int(accum.min()))
        observed['accumulation_max'] = max(observed['accumulation_max'], int(accum.max()))
        observed['integer_prefix_abs_max'] = max(observed['integer_prefix_abs_max'], prefix_abs)
        observed['fp_margin_abs_min'] = min(observed['fp_margin_abs_min'], float(np.abs(margin).min()))
        per_frame.append(row)
        if fi < 2:
            selected.append(dict(X_fp32=X[:, :32], X_q16=Xq[:, :32],
                raw_g_fp32=raw[:, :32], raw_g_int=raw_i[:, :32],
                projected_g_fp32=projected[:, :32], projected_g_int=projected_i[:, :32],
                fp32_margin=margin[:, :32], int_accum_q28=accum[:, :32],
                int_margin_q28=integer_margin[:, :32],
                code_index_fp32=index[:, :32].transpose(1, 0, 2).astype(np.uint8),
                code_index_int=index_i[:, :32].transpose(1, 0, 2).astype(np.uint8)))
        print('SOURCE', fi, record['file'], row['raw_gate_differences'],
              row['projected_gate_differences'], flush=True)

    fields = {k: np.stack([s[k] for s in selected]) for k in selected[0]}
    fields.update(A_fp32=A, A_q12=Aq, bias_fp32=bias, center_fp32=center,
        theta_fp32=theta, threshold_real=threshold_real, threshold_q28=threshold, D=D,
        frame_file=np.array([cache[i]['file'] for i in range(2)]),
        sample_index=np.arange(32, dtype=np.int32),
        # Archive omitted positions; reconstruct with the source capture's fixed rule.
        reconstructed_spatial_index=torch.linspace(0, 19199, 512,
            device=args.device).round().long().cpu().numpy()[:32],
        spatial_index_status=np.array('reconstructed from fixed 19200-position source shape; not stored in cache'),
        source_boundary=np.array('cached pre-source FP32 X from 32 training frames; forced student source parameters'),
        dut_input_fields=np.array(['X_q16','A_q12','threshold_q28','D']),
        oracle_fields=np.array(['X_fp32','raw_g_fp32','raw_g_int','projected_g_fp32',
            'projected_g_int','code_index_fp32','code_index_int','fp32_margin',
            'int_accum_q28','int_margin_q28']))
    np.savez_compressed(HERE / 'source_cases.npz', **fields)
    restored = np.load(HERE / 'source_cases.npz')
    for k, v in fields.items(): assert np.array_equal(restored[k], v), k
    aggregate_opportunity = {kind: {k: sum(r[kind][k] for r in per_frame)
        for k in per_frame[0][kind]} for kind in
        ['raw_fp32','raw_integer','projected_fp32','projected_integer']}
    report = dict(function='fixed forced student source + nearest Hamming code',
        source_archive=str(args.archive), cache_member='support_training/teacher_cache.pt',
        teacher_gate_is_not_used_as_student_input=True,
        no_training=True, no_parameter_mutation=True, no_AEE_claim=True,
        runtime=dict(torch=torch.__version__, device=args.device,
                     gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                     original_run_torch='2.2.2+cu121 on A800; this is a new replay environment',
                     allow_tf32=True),
        cohort=dict(frames=32, positions_per_frame=512, T=10, C=96,
                    gate_values=32*512*10*96, groups=32*512*10*6,
                    split='training cache, not held-out validation',
                    rtl_selection='first two cache records, sample indices 0:32'),
        quantization=dict(A='signed16 Q12 RNE', X='signed24 Q16 RNE',
                          threshold='signed48 Q28 RNE of float64(theta + center - bias)',
                          accumulator='signed48; no intermediate RNE, inclusive >=',
                          clipping=False,
                          static_prefix_abs_bound_any_signed24_X=int((np.abs(Aq.astype(np.int64)).sum(1)*(2**23)).max())),
        observed=observed, totals=totals, margin_abs_le_counts=margin_bins,
        opportunities=aggregate_opportunity, frames=per_frame)
    (HERE / 'source_statistics.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TOTAL', json.dumps(totals), flush=True)
    print('RANGES', json.dumps(observed), flush=True)


if __name__ == '__main__':
    main()
