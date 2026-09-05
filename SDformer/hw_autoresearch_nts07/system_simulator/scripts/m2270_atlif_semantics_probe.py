#!/usr/bin/env python3
"""Read-only checkpoint diagnosis plus small real-module CPU forward tests.

This is neither an AEE ablation nor full-network replay. 'binary' is a mode,
not proof that the numerical output alphabet is {0, 1}. No checkpoint edits.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HW = Path(__file__).resolve().parents[2]
ROOT = HW.parent
OVERLAY = ROOT / 'neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN'
CAPTURE = HW / 'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
CHECKPOINT = HW / 'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
CONFIG = HW / 'system_handoff/incoming/m2041_ep34_quant_binding_inputs/dsec_c12_alpha0125_ep29_resume5_20260830.yml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['model_state_dict']
    config = yaml.safe_load(CONFIG.read_text())
    activity = json.loads((CAPTURE / 'atlif_activity.json').read_text())
    live = {r['name']: r for r in activity}
    thresholds = {k[:-7]: float(v) for k, v in state.items() if k.endswith('.thresh')}
    spec = importlib.util.spec_from_file_location('m2270_neuron', OVERLAY / 'atlif_ternary_psn/atlif_ternary_psn.py')
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    generator = torch.Generator().manual_seed(2270)
    rows = []
    for name, theta in thresholds.items():
        row = dict(name=name, theta=theta, exact_one=theta == 1., invoked=name in live)
        if name in live:
            info = live[name]
            T = info['temporal_steps']
            model = implementation.ATLIFTernaryPSN(
                T, output_mode=info['output_mode'], threshold_mode=info['threshold_mode'],
                sparsity_eta=0., center_mode='zero').eval()
            model.load_state_dict({k[len(name)+1:]: v for k, v in state.items() if k.startswith(name+'.')})
            x = torch.randn(T, 257, generator=generator) * 3
            with torch.no_grad():
                observed = model(x)
                h = torch.addmm(model.bias, model.weight, x)
                support = h.ge(model.thresh)
                reference = support.float() * model.thresh
                mismatch = int(observed.ne(reference).sum())
                assert mismatch == 0
                assert bool(((observed == 0) | (observed == model.thresh)).all())
            w = model.weight.detach().double()
            row.update(
                T=T, output_mode=info['output_mode'], threshold_mode=info['threshold_mode'],
                deployment_dead_result=info['deployment_dead_result'],
                captured_calls=info['calls'], captured_active=info['active'],
                captured_elements=info['elements'], temporal_rank=int(torch.linalg.matrix_rank(w)),
                off_diagonal_l1_fraction=float((w.abs().sum()-w.diagonal().abs().sum())/w.abs().sum()),
                directed_elements=observed.numel(), directed_mismatches=mismatch,
                directed_nonzero_values=[float(v) for v in observed.unique() if v != 0],
                manual_growth_with_sp_zero=float(model.update_value))
        rows.append(row)
    live_rows = [r for r in rows if r['invoked']]
    visible = [r for r in live_rows if not r['deployment_dead_result']]
    # Prove that eta=0 does NOT disable the task-loss gradient of theta.
    theta = torch.tensor(1., requires_grad=True)
    h = torch.tensor([[0.5, 1., 1.5, 2.]], requires_grad=True)
    y, growth = implementation.OfficialATLIFSurrogate.apply(h, theta, 0.)
    y.sum().backward()
    assert growth == 0. and theta.grad != 0.
    atlif = config['atlif_ternary_psn']
    report = dict(
        checkpoint=str(CHECKPOINT), config=str(CONFIG), capture=str(CAPTURE),
        definition='h=A*x+b; output=theta*1[h>=theta]; no recurrent reset in this PSN forward',
        counts=dict(installed=len(rows), invoked=len(live_rows), result_consumed=len(visible),
                    installed_theta_exact_one=sum(r['exact_one'] for r in rows),
                    invoked_theta_exact_one=sum(r['exact_one'] for r in live_rows),
                    result_consumed_theta_exact_one=sum(r['exact_one'] for r in visible)),
        threshold_min=min(thresholds.values()), threshold_max=max(thresholds.values()),
        output_modes=dict(Counter(r['output_mode'] for r in activity)),
        manual_growth_zero_all_directed=all(r['manual_growth_with_sp_zero'] == 0 for r in live_rows),
        eta_zero_still_has_task_gradient=float(theta.grad),
        settings={k: atlif.get(k) for k in (
            'threshold_eta', 'activity_eta', 'target_rate', 'target_rate_eta',
            'threshold_init', 'threshold_freeze_after_step', 'freeze_threshold_grad_after_step',
            'threshold_lr_scale', 'trainable')},
        optimizer_threshold_lr=config['optimizer']['param_groups']['threshold_lr'],
        directed_mismatches=sum(r['directed_mismatches'] for r in live_rows),
        directed_elements=sum(r['directed_elements'] for r in live_rows),
        temporal_rank_hist=dict(Counter(f"T{r['T']}_rank{r['temporal_rank']}" for r in visible)),
        historical_capture_support_check_mismatches=sum(r['recomputed_reference_mismatch'] for r in activity),
        caveats=[
            'Capture output_mode copies module metadata; it is not an output value histogram.',
            'Historical reference check compared nonzero support on sampled inputs, not raw amplitudes.',
            'Directed CPU tests use random inputs, actual checkpoint parameters, and the actual neuron class.',
            'No full-network AEE or training ablation; theta near one alone does not establish task-level dispensability.',
            'threshold_freeze_after_step freezes manual growth; task-gradient freezing is separately opt-in.',
            'The inspected YAML is the ep29-to-ep34 resume configuration, not proof of all earlier training settings.',
            'A nonzero isolated surrogate gradient does not prove a nonzero task gradient through every actual network branch.',
            'Temporal full rank and off-diagonal coefficients show nontrivial mixing, not measured AEE benefit.',
            'Integer reassociation/threshold folding is not automatically FP32/TF32 bit-exact.'
        ], rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'caveats')}, indent=2))


if __name__ == '__main__':
    main()
