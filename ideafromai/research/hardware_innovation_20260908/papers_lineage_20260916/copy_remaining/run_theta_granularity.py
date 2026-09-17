"""Per-neuron / block AT-LIF θ vs layer-shared scalar.

Identity: o_h = θ_h H(m_h - θ_h), absorb θ_h into next W column, transmit binary s.
ep34 stores a scalar thresh (numel=1) per PSN. This probe freezes compile-time
θ maps from calibration U (no training) and measures 10-frame AEE + leftover extras.
"""
from __future__ import annotations

import json, os, sys, types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ALGO = HERE.parents[1] / 'algorithm'
REPO = Path('/home/zhumd/work/sdformer_codex/SDformer')
DATA = REPO / 'data/Datasets/DSEC/saved_flow_data'
INCOMING = REPO / 'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
sys.path[:0] = [str(HERE), str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
from run_bn_probe import build_model, input_frame  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402
from remaining_ops import (  # noqa: E402
    KEEP, block_pool_theta, keep_kill, lut_gemm_group_extra,
    psn_scrooge_extra, psn_scrooge_inspect_tax, rate_matched_theta,
    scale_theta_for_rate, word_stats,
)

PRED2 = 'sttmultires_unet.preds.2'
S0_FC1 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.fc1'
S0_SN1 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1.spiking_neuron'
S0_SN2 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2.spiking_neuron'
NB0 = 1.454602861


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def next_linear_name(sn_name):
    if sn_name.endswith('.sn1.spiking_neuron'):
        return sn_name.replace('.sn1.spiking_neuron', '.fc1')
    if sn_name.endswith('.sn2.spiking_neuron'):
        return sn_name.replace('.sn2.spiking_neuron', '.fc2')
    return None


def atlif_modules(mods):
    out = []
    for n, m in mods.items():
        if n.endswith('spiking_neuron') and hasattr(m, 'thresh') and hasattr(m, 'weight'):
            if tuple(m.weight.shape) == (getattr(m, 'T', 0), getattr(m, 'T', -1)):
                out.append((n, m))
    return out


def mix_U(sn, x_seq):
    flat = x_seq.flatten(1)
    h = torch.addmm(sn.bias, sn.weight, flat)
    C = x_seq.shape[-1]
    return h.view(h.shape[0], -1, C)


def install_theta_forward(sn, theta_h, emit_amplitude):
    """Binary s = H(U-θ_h). If emit_amplitude, out = θ_h s; else out = s (θ absorbed)."""
    th = theta_h.detach().float()

    def fwd(self, x_seq):
        h = mix_U(self, x_seq)
        t = th.to(device=h.device, dtype=h.dtype)
        s = (h >= t).to(h.dtype)
        out = s * t if emit_amplitude else s
        return out.reshape(x_seq.shape)

    sn.forward = types.MethodType(fwd, sn)


def restore_forwards(sn_orig):
    for sn, fn in sn_orig:
        sn.forward = fn


def absorb_into(linear, theta_h, theta0):
    th = theta_h.detach().to(device=linear.weight.device, dtype=linear.weight.dtype)
    scale = th / float(theta0)
    if linear.weight.shape[1] != scale.numel():
        return False
    linear.weight.data = linear.weight.data * scale.reshape(1, -1)
    return True


def main():
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
        checkpoint=INCOMING / 'checkpoint_epoch34.pth',
        data=DATA,
    )
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    atlif = atlif_modules(mods)
    inventory = []
    for n, m in atlif:
        inventory.append({
            'name': n,
            'T': int(m.T),
            'thresh_numel': int(m.thresh.numel()),
            'thresh': float(m.thresh.detach().float().mean()),
            'weight': list(m.weight.shape),
        })
    shared_all_scalar = all(x['thresh_numel'] == 1 for x in inventory)
    print('ATLIF', len(inventory), 'all_scalar', shared_all_scalar, flush=True)

    sn1 = mods[S0_SN1]
    sn2 = mods[S0_SN2]
    fc1 = mods[S0_FC1]
    theta0_sn1 = float(sn1.thresh.detach().float().mean())
    theta0_sn2 = float(sn2.thresh.detach().float().mean())
    A2 = sn2.weight.detach().float()
    Tmix = int(A2.shape[0])
    orig_fwd = [(m, m.forward) for _, m in atlif]
    orig_W = {n: mods[n].weight.detach().clone()
              for n in (S0_FC1, S0_FC1.replace('.fc1', '.fc2'))
              if n in mods}

    # ---- calib U on all 10 frames, sampled spatial ----
    cap = {'sn1': [], 'sn2': []}
    aee_base = []
    leftover = []
    bucket = {}

    def hook_fc1(m, inp):
        bucket['s'] = inp[0].detach()

    def hook_sn1(m, inp):
        bucket['sn1_in'] = inp[0].detach()

    def hook_sn2(m, inp):
        bucket['psn'] = inp[0].detach()

    def hook_p2(m, i, o):
        bucket['p2'] = o.detach()

    h0 = sn1.register_forward_pre_hook(hook_sn1)
    h1 = fc1.register_forward_pre_hook(hook_fc1)
    h2 = sn2.register_forward_pre_hook(hook_sn2)
    h3 = mods[PRED2].register_forward_hook(hook_p2)
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            s_in = bucket['s']
            y_psn = bucket['psn']
            C1 = s_in.shape[-1]
            H2 = y_psn.shape[-1]
            u1 = mix_U(sn1, bucket['sn1_in'])
            step = max(1, u1.shape[1] // 256)
            cap['sn1'].append(u1[:, ::step].float().cpu())
            u2 = mix_U(sn2, y_psn)
            cap['sn2'].append(u2[:, ::step].float().cpu())
            S = (s_in.reshape(-1, C1).abs() > 0).float().cpu()
            st = word_stats(S, fc1.weight.detach().float().cpu())
            lut, _, _ = lut_gemm_group_extra(S, group=4)
            ys = y_psn.reshape(y_psn.shape[0], -1, H2)[:, ::max(1, y_psn.reshape(y_psn.shape[0], -1, H2).shape[1] // 256)]
            tau = (sn2.thresh.detach().float().reshape(1, 1) - sn2.bias.detach().float().reshape(-1, 1))
            tau = tau.expand(Tmix, H2).contiguous()
            scr = psn_scrooge_extra(ys.float().cpu(), A2.cpu(), tau.cpu(), bound='l1_maxabs')
            leftover.append({'elem_nnz': st['elem_nnz'], 'lutg4': lut, 'scrooge': scr,
                             'scrooge_after_tax': scr - psn_scrooge_inspect_tax(Tmix, 'l1_maxabs')})
            p2 = F.interpolate(bucket['p2'].sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aee_base.append(aee(p2, label, mask))
            print('BASE', name[-12:], 'AEE', round(aee_base[-1], 4),
                  'nnz', round(st['elem_nnz'], 4), 'lutg4', round(lut, 3),
                  'scr', round(leftover[-1]['scrooge_after_tax'], 3), flush=True)
            del x, label, mask, p2, S
            torch.cuda.empty_cache()
    h0.remove(); h1.remove(); h2.remove(); h3.remove()

    U1 = torch.cat(cap['sn1'], dim=1)
    U2 = torch.cat(cap['sn2'], dim=1)
    th1, fire1 = rate_matched_theta(U1, theta0_sn1)
    th2, fire2 = rate_matched_theta(U2, theta0_sn2)
    print('fire_sn1_mean', float(fire1.mean()), 'fire_sn2_mean', float(fire2.mean()),
          'th1_std', float(th1.std()), 'th2_std', float(th2.std()), flush=True)

    def run_policy(tag, patches, leftover_on=True):
        """patches: list of (sn_name, theta_h, absorb_linear_name_or_None, emit_amp)."""
        for sn, fn in orig_fwd:
            sn.forward = fn
        for n, w in orig_W.items():
            mods[n].weight.data.copy_(w)
        for sn_name, theta_h, lin_name, emit_amp in patches:
            install_theta_forward(mods[sn_name], theta_h, emit_amp)
            if lin_name and lin_name in mods:
                t0 = float(mods[sn_name].thresh.detach().float().mean())
                absorb_into(mods[lin_name], theta_h, t0)
        aees = []
        extras = []
        hh0 = sn1.register_forward_pre_hook(hook_sn1)
        hh1 = fc1.register_forward_pre_hook(hook_fc1)
        hh2 = sn2.register_forward_pre_hook(hook_sn2)
        hh3 = mods[PRED2].register_forward_hook(hook_p2)
        with torch.no_grad():
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(DATA, name)
                model(x)
                p2 = F.interpolate(bucket['p2'].sum(0), size=(480, 640), mode='bilinear', align_corners=False)
                aees.append(aee(p2, label, mask))
                rec = {'AEE': aees[-1]}
                if leftover_on:
                    s_in = bucket['s']
                    C1 = s_in.shape[-1]
                    S = (s_in.reshape(-1, C1).abs() > 0).float().cpu()
                    lut, _, _ = lut_gemm_group_extra(S, group=4)
                    rec['elem_nnz'] = float(S.mean())
                    rec['lutg4'] = lut
                    y_psn = bucket['psn']
                    H2 = y_psn.shape[-1]
                    ys = y_psn.reshape(y_psn.shape[0], -1, H2)
                    step = max(1, ys.shape[1] // 256)
                    ys = ys[:, ::step].float().cpu()
                    # tau from the sn2 θ actually used
                    used = None
                    for sn_name, theta_h, _, _ in patches:
                        if sn_name == S0_SN2:
                            used = theta_h
                    if used is None:
                        used = sn2.thresh.detach().float().reshape(1)
                    used = used.detach().float().cpu().reshape(-1)
                    bias_t = sn2.bias.detach().float().cpu().reshape(-1, 1)
                    tau = used.reshape(1, -1) - bias_t
                    if tau.shape[1] == 1:
                        tau = tau.expand(Tmix, H2)
                    else:
                        tau = tau.expand(Tmix, tau.shape[1])
                    rec['scrooge_after_tax'] = psn_scrooge_extra(
                        ys, A2.cpu(), tau.contiguous(), bound='l1_maxabs'
                    ) - psn_scrooge_inspect_tax(Tmix, 'l1_maxabs')
                extras.append(rec)
                print(tag, name[-12:], round(aees[-1], 4),
                      'nnz', round(rec.get('elem_nnz', -1), 4),
                      'lut', round(rec.get('lutg4', -1), 3),
                      'scr', round(rec.get('scrooge_after_tax', -1), 3), flush=True)
                del x, label, mask, p2
                torch.cuda.empty_cache()
        hh0.remove(); hh1.remove(); hh2.remove(); hh3.remove()
        for sn, fn in orig_fwd:
            sn.forward = fn
        for n, w in orig_W.items():
            mods[n].weight.data.copy_(w)
        aee_m = float(np.mean(aees))
        out = {
            'policy': tag,
            'AEE_10frame': aee_m,
            'AEE_vs_NB0': aee_m < NB0,
            'AEE_vs_shared': aee_m - float(np.mean(aee_base)),
            'keep_kill_quality': keep_kill(1.0) if aee_m < NB0 else 'KILL',
        }
        if leftover_on and extras and 'lutg4' in extras[0]:
            out['elem_nnz'] = float(np.mean([e['elem_nnz'] for e in extras]))
            out['lutg4'] = float(np.mean([e['lutg4'] for e in extras]))
            out['scrooge_after_tax'] = float(np.mean([e['scrooge_after_tax'] for e in extras]))
            out['lutg4_keep_kill'] = keep_kill(out['lutg4'])
            out['scrooge_keep_kill'] = keep_kill(out['scrooge_after_tax'])
        print('POLICY', tag, 'AEE', round(aee_m, 4), 'dAEE', round(out['AEE_vs_shared'], 4), flush=True)
        return out

    fc2_name = S0_FC1.replace('.fc1', '.fc2')
    policies = []
    # S0 sn2 only, rate-matched, absorb into fc2 (official {0,θ_h})
    policies.append(run_policy(
        'S0.sn2_perh_rate_matched_absorb',
        [(S0_SN2, th2, fc2_name, True)]))
    policies.append(run_policy(
        'S0.sn2_block32_rate_matched_absorb',
        [(S0_SN2, block_pool_theta(th2, 32), fc2_name, True)]))
    policies.append(run_policy(
        'S0.sn2_block8_rate_matched_absorb',
        [(S0_SN2, block_pool_theta(th2, 8), fc2_name, True)]))
    # decision-only (amplitude stays 1, only compare changes)
    policies.append(run_policy(
        'S0.sn2_perh_rate_matched_binary_emit',
        [(S0_SN2, th2, None, False)]))
    # sn1 changes FC1 leftover S
    policies.append(run_policy(
        'S0.sn1_perh_rate_matched_absorb',
        [(S0_SN1, th1, S0_FC1, True)]))
    policies.append(run_policy(
        'S0.sn1_block32_rate_matched_absorb',
        [(S0_SN1, block_pool_theta(th1, 32), S0_FC1, True)]))
    th1_sp = scale_theta_for_rate(U1, 0.8 * fire1)
    policies.append(run_policy(
        'S0.sn1_perh_0p8rate_absorb',
        [(S0_SN1, th1_sp, S0_FC1, True)]))
    th2_sp = scale_theta_for_rate(U2, 0.8 * fire2)
    policies.append(run_policy(
        'S0.sn2_perh_0p8rate_absorb',
        [(S0_SN2, th2_sp, fc2_name, True)]))
    # compile-time dead channels via θ=∞ on lowest-rate 15% (threshold-as-mask)
    def kill_lowest_rate(theta, fire, frac=0.15, U=None):
        t = theta.clone()
        H = t.numel()
        nkill = max(1, int(round(frac * H)))
        idx = torch.argsort(fire)[:nkill]
        if U is None:
            t[idx] = t[idx] + 1.0e6
        else:
            umax = U.reshape(-1, H).max(0).values
            t[idx] = umax[idx] + 1.0
        return t
    policies.append(run_policy(
        'S0.sn1_kill_lowest15pct_via_theta',
        [(S0_SN1, kill_lowest_rate(th1, fire1, 0.15, U1), S0_FC1, True)]))
    policies.append(run_policy(
        'S0.sn2_kill_lowest15pct_via_theta',
        [(S0_SN2, kill_lowest_rate(th2, fire2, 0.15, U2), fc2_name, True)]))

    # all-net: every MLP sn1/sn2 rate-matched absorb (user question)
    all_patches = []
    # reuse S0 maps for other layers of same C when possible? calibrate only S0.
    # For other modules, scale their scalar θ by per-channel relative map from S0 sn2/sn1 of matching C.
    rel1 = th1 / max(theta0_sn1, 1e-6)
    rel2 = th2 / max(theta0_sn2, 1e-6)
    for n, m in atlif:
        t0 = float(m.thresh.detach().float().mean())
        lin = next_linear_name(n)
        # peek channel count from a dummy: use weight T,T only. Need C from a buffer — skip if not sn1/sn2 mlp
        if n.endswith('.sn1.spiking_neuron'):
            # cannot know C without forward; try 96 for stage0-like, else skip non-S0
            if '.layers.0.' in n:
                all_patches.append((n, th1 * (t0 / theta0_sn1), lin, True))
        elif n.endswith('.sn2.spiking_neuron') and '.layers.0.' in n:
            all_patches.append((n, th2 * (t0 / theta0_sn2), lin, True))
    policies.append(run_policy('stage0_all_mlp_sn_perh_rate_matched_absorb', all_patches))

    blk_patches = []
    for n, m in atlif:
        t0 = float(m.thresh.detach().float().mean())
        lin = next_linear_name(n)
        if n.endswith('.sn1.spiking_neuron') and '.layers.0.' in n:
            blk_patches.append((n, block_pool_theta(th1 * (t0 / theta0_sn1), 32), lin, True))
        elif n.endswith('.sn2.spiking_neuron') and '.layers.0.' in n:
            blk_patches.append((n, block_pool_theta(th2 * (t0 / theta0_sn2), 32), lin, True))
    policies.append(run_policy('stage0_all_mlp_sn_block32_rate_matched_absorb', blk_patches))

    base = {
        'AEE_10frame': float(np.mean(aee_base)),
        'elem_nnz': float(np.mean([x['elem_nnz'] for x in leftover])),
        'lutg4': float(np.mean([x['lutg4'] for x in leftover])),
        'scrooge_after_tax': float(np.mean([x['scrooge_after_tax'] for x in leftover])),
        'theta_shared_sn1': theta0_sn1,
        'theta_shared_sn2': theta0_sn2,
        'sn1_fire_mean': float(fire1.mean()),
        'sn2_fire_mean': float(fire2.mean()),
        'sn1_theta_std': float(th1.std()),
        'sn2_theta_std': float(th2.std()),
        'all_atlif_scalar': shared_all_scalar,
        'n_atlif': len(inventory),
    }
    idea = {
        'mechanism': 'compile_time_per_neuron_or_block_ATLIF_theta_absorb',
        'kind': 'new idea: θ granularity (per-h or tile-G) still absorb into next W; binary GeMM',
        'target_op': 'S0.b0.fc1 leftover S and S0.b0.psn tau',
        'fair_baseline': 'shared scalar θ as in ep34; leftover nnz(S)*Cout and PSN T×T',
        'note': 'BN already makes per-h effective tau; this varies the ATLIF θ itself',
    }
    # pick best quality-and-extra among policies that beat NB0
    ok = [p for p in policies if p.get('AEE_vs_NB0')]
    best = min(ok, key=lambda p: p['AEE_10frame']) if ok else min(policies, key=lambda p: p['AEE_10frame'])
    idea['best_policy'] = best['policy']
    idea['best_AEE'] = best['AEE_10frame']
    idea['best_dAEE_vs_shared'] = best['AEE_vs_shared']
    idea['lutg4'] = best.get('lutg4')
    idea['scrooge_after_tax'] = best.get('scrooge_after_tax')
    idea['keep_kill'] = keep_kill(best.get('lutg4') or 0) if best.get('lutg4', 0) >= KEEP or best.get('scrooge_after_tax', 0) >= KEEP else (
        'KEEP' if best['AEE_10frame'] + 0.02 < base['AEE_10frame'] and best['AEE_vs_NB0'] else 'KILL'
    )
    idea['keep_kill_note'] = 'KEEP leftover extra if lut/scrooge ≥15%; else KEEP only if AEE clearly better than shared (0.02) and <NB0'
    out = {
        'identity': 'AT-LIF {0,theta_h} absorb theta_h into next W column; spikes binary',
        'inventory': inventory,
        'baseline_shared_scalar': base,
        'policies': policies,
        'new_idea': idea,
        'KEEP_threshold': KEEP,
        'NB0': NB0,
        'frames': names,
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'theta_granularity.json').write_text(json.dumps(out, indent=2))
    print('BASELINE', base, flush=True)
    print('BEST', idea, flush=True)
    print('POLICIES', [(p['policy'], round(p['AEE_10frame'], 4), p.get('lutg4'), p.get('scrooge_after_tax')) for p in policies], flush=True)


if __name__ == '__main__':
    main()
