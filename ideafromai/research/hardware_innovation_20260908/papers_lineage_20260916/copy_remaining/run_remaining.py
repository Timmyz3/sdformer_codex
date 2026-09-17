"""Remaining-op copies vs frozen AAC word-adds and leftover PSN T×T MACs.

Skip-family is not recounted. BitWave KEEP bar is word-add leftover, not 7-serial.
CGNet KEEP is skipped/total all Cout; live BN forces KILL.
"""
from __future__ import annotations

import json, os, sys
from collections import defaultdict
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
    KEEP, TAX_RUNTIME, WBITS, emit_copy, k_keep_for_nnz_extra, keep_kill,
    lut_gemm_group_extra, prefix_nnz_extra, psn_lowrank_mac_extra, psn_scrooge_extra,
    psn_scrooge_inspect_tax, psn_zero_t_extra, sna_pea_extra, support_code_compute_extra,
    support_code_encode_tax, word_stats,
)

PRED2 = 'sttmultires_unet.preds.2'
NB0 = 1.454602861
S0_FC1 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.fc1'
S0_SN2 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2.spiking_neuron'
S0_BN1 = 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.bn1'

OPS = {
    'S0.b0.fc1': S0_FC1,
    'S0.b1.fc1': 'sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.fc1',
    'r1.conv1': 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.conv1.0',
    'stem': 'sttmultires_unet.encoders.swin3d.patch_embed.conv.conv.0',
}


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    return float(err[mask].sum()) / int(mask.sum())


def conv_im2col_sample(x, W, stride=16):
    w = W.detach().float().cpu()
    Cout, Cin, kh, kw = w.shape
    xs = x.detach()[:, 0, :, ::stride, ::stride].contiguous().cpu()
    unf = F.unfold(xs, kernel_size=(kh, kw), padding=kh // 2)
    S = (unf.abs() > 0).permute(0, 2, 1).reshape(-1, Cin * kh * kw)
    return S.float(), w.reshape(Cout, -1)


def psn_tau_from_module(sn, H, device, dtype):
    thresh = sn.thresh.detach().float().reshape(-1)
    bias = sn.bias.detach().float().reshape(-1, 1)
    center = sn.center.detach().float().reshape(-1, 1)
    if getattr(sn, 'center_mode', 'zero') == 'zero':
        center = torch.zeros_like(bias)
    # h = A@x + bias - center; spike if h >= thresh => A@x >= thresh - bias + center
    tau = (thresh.reshape(1, 1) - bias + center).reshape(-1, 1).expand(-1, H)
    return tau.to(device=device, dtype=dtype)


def reshape_tokens(x, last_dim):
    """(T, ..., C) or (T, P*C) -> (T, P, C)."""
    if x.dim() >= 3 and x.shape[-1] == last_dim:
        return x.reshape(x.shape[0], -1, last_dim)
    if x.dim() == 2 and x.shape[1] % last_dim == 0:
        return x.reshape(x.shape[0], -1, last_dim)
    if x.shape[-1] == last_dim:
        return x.reshape(-1, last_dim).unsqueeze(0)
    raise ValueError('reshape_tokens got %s last_dim=%s' % (tuple(x.shape), last_dim))


def run_aee_with_pre_hook(model, mods, names, hook_mod, pre_fn):
    h = hook_mod.register_forward_pre_hook(pre_fn)
    bucket = {}
    h2 = mods[PRED2].register_forward_hook(lambda m, i, o: bucket.__setitem__('p2', o.detach()))
    aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p2 = F.interpolate(bucket['p2'].sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aees.append(aee(p2, label, mask))
            print('AEE_PASS', getattr(pre_fn, 'tag', '?'), name[-12:], round(aees[-1], 4), flush=True)
            del x, label, mask, p2
    h.remove()
    h2.remove()
    return float(np.mean(aees))


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
    frame = {}
    handles = []

    bn1 = mods.get(S0_BN1)
    bn_mod = bn1.norm_layer if bn1 is not None and hasattr(bn1, 'norm_layer') else bn1
    bn_live = True
    if bn_mod is not None:
        bn_live = not bool(getattr(bn_mod, 'track_running_stats', False))
    sn2 = mods[S0_SN2]
    v_th = float(sn2.thresh.detach().float().mean().item())
    W0 = mods[S0_FC1].weight.detach().float().cpu()
    A_cpu = sn2.weight.detach().float().cpu()
    Tmix = int(A_cpu.shape[0])
    mass = W0.abs().sum(0)
    order_c = torch.argsort(mass, descending=True)
    order_h = torch.argsort(W0.pow(2).sum(1), descending=True)
    Hhid = W0.shape[0]
    acc = defaultdict(list)
    printed_shape = {'n': 0}

    def s0_fc1_pre(m, inp):
        x = inp[0].detach()
        cin = m.in_features
        if printed_shape['n'] == 0:
            print('SHAPE fc1_in', tuple(x.shape), 'cin', cin, flush=True)
        tpc = reshape_tokens(x, cin).cpu()
        del x
        S = (tpc.abs() > 0).float().reshape(-1, cin)
        st = word_stats(S, W0)
        extra_p85 = prefix_nnz_extra(S, order_c[:81])
        k15, extra15 = k_keep_for_nnz_extra(S, order_c, target=KEEP)
        st['extra_prefix85_nnz'] = extra_p85
        st['extra_prefix_nnz15'] = extra15
        st['k15'] = k15
        st['extra_zero_t_psn'] = psn_zero_t_extra(tpc)
        st['fc1_word_adds'] = float(st['baseline_aac_word_adds'])
        st['psn_lowrank4'] = psn_lowrank_mac_extra(Tmix, 4)
        n_nz = st['n_nz_rows']
        st['support_K256_compute'] = support_code_compute_extra(n_nz, 256)
        st['support_K256_tax'] = support_code_encode_tax(n_nz, 256, cin, st['baseline_aac_word_adds'])
        st['cout_drop15'] = 1.0 - int(0.85 * Hhid) / Hhid
        st['bn_live'] = bn_live
        lut_e, n_grp, nnz_l = lut_gemm_group_extra(S, group=4)
        st['extra_lut_gemm_g4'] = lut_e
        st['lut_gemm_groups'] = n_grp
        # SnaPEA on CPU sample; live BN still forces KILL later
        step = max(1, S.shape[0] // 256)
        y_th = torch.full((Hhid,), v_th)
        extra_sna, ns, _, _ = sna_pea_extra(S[::step].cpu(), W0, y_th, max_tokens=256)
        st['extra_snapea_oracle_bn'] = extra_sna
        st['snapea_tokens'] = ns
        frame['S0.b0.fc1'] = st
        del S, tpc

    def s0_b1_pre(m, inp):
        x = inp[0].detach()
        cin = m.in_features
        S = (x.reshape(-1, cin).abs() > 0).float().cpu()
        del x
        frame['S0.b1.fc1'] = word_stats(S, mods[OPS['S0.b1.fc1']].weight.detach().float().cpu())
        del S

    def conv_pre(key):
        def hook(m, inp, key=key):
            x = inp[0].detach()
            S, Ww = conv_im2col_sample(x, m.weight)
            del x
            frame[key] = word_stats(S, Ww)
        return hook

    def sn2_pre(m, inp):
        y = inp[0].detach()
        if printed_shape['n'] == 0:
            print('SHAPE sn2_in', tuple(y.shape), flush=True)
            printed_shape['n'] = 1
        y3 = reshape_tokens(y, Hhid)
        del y
        P = y3.shape[1]
        step = max(1, P // 256)
        ys = y3[:, ::step].contiguous().float().cpu()
        del y3
        tau = psn_tau_from_module(sn2, Hhid, ys.device, ys.dtype)
        extra = psn_scrooge_extra(ys, A_cpu, tau, bound='l1_maxabs')
        extra_oracle = psn_scrooge_extra(ys, A_cpu, tau, bound='tot_oracle')
        st = frame.get('S0.b0.fc1', {})
        st['extra_psn_scrooge'] = extra
        st['extra_psn_scrooge_tot_oracle'] = extra_oracle
        st['psn_tokens'] = int(ys.shape[1])
        st['psn_baseline_macs'] = float(P * Hhid * Tmix * Tmix)
        frame['S0.b0.fc1'] = st
        del ys

    def pred_hook(m, i, o):
        frame['p2'] = o.detach()

    handles.append(mods[S0_FC1].register_forward_pre_hook(s0_fc1_pre))
    handles.append(mods[OPS['S0.b1.fc1']].register_forward_pre_hook(s0_b1_pre))
    handles.append(mods[OPS['r1.conv1']].register_forward_pre_hook(conv_pre('r1.conv1')))
    handles.append(mods[OPS['stem']].register_forward_pre_hook(conv_pre('stem')))
    handles.append(mods[S0_SN2].register_forward_pre_hook(sn2_pre))
    handles.append(mods[PRED2].register_forward_hook(pred_hook))

    aees = []
    with torch.no_grad():
        for name in names:
            frame.clear()
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            for key in OPS:
                if key in frame:
                    acc[key].append(frame[key])
            p2 = F.interpolate(frame['p2'].sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aees.append(aee(p2, label, mask))
            last = acc['S0.b0.fc1'][-1]
            print('FRAME', name[-12:],
                  'bitwave_vs_word', round(last['extra_bitwave_vs_word_add_cycles'], 3),
                  'sna', round(last.get('extra_snapea_oracle_bn', 0), 3),
                  'p85', round(last['extra_prefix85_nnz'], 3),
                  'k15', last['k15'], round(last['extra_prefix_nnz15'], 3),
                  'scrooge', round(last.get('extra_psn_scrooge', 0), 3),
                  'lutg4', round(last.get('extra_lut_gemm_g4', 0), 3),
                  'zeroT', round(last['extra_zero_t_psn'], 3),
                  flush=True)
            del x, label, mask, p2, frame['p2']
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    def mean(op, field):
        return float(np.mean([r[field] for r in acc[op] if field in r]))

    psn_base = 'n_tokens*Cout*T*T PSN mix MACs after FC1 Y exists (skip-family not recounted)'
    copies = []
    copies.append(emit_copy('dual_side_Wzero', 'FireFly-S dual-side', 'S0.b0.fc1',
                            mean('S0.b0.fc1', 'extra_dual'), TAX_RUNTIME, True, 'vs word-adds'))
    copies.append(emit_copy(
        'BitWave_vs_word_add', 'BitWave/SparseCol but vs frozen word-add leftover',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_bitwave_vs_word_add_cycles'), TAX_RUNTIME, True,
        '1 cyc/bit vs 1 cyc/word; negative => bit-serial is more cycles. extra_bitwave_vs_7serial is NOT the KEEP bar'))
    copies.append(emit_copy('unique_row_merge', 'Prosperity identical-row merge on leftover',
                            'S0.b0.fc1', mean('S0.b0.fc1', 'extra_unique'), TAX_RUNTIME, True))
    copies.append(emit_copy(
        'CGNet_SnaPEA_oracleBN', 'CGNet/SnaPEA skipped/total all Cout, oracle BN y_th',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_snapea_oracle_bn'), TAX_RUNTIME, False,
        'KILL if bn_live (BN consumes full Y). extra=skipped/total not skipped/processed'))
    if acc['S0.b0.fc1'][0].get('bn_live', True):
        copies[-1]['keep_kill'] = 'KILL'
        copies[-1]['note'] += '; bn_live=True => A7 cannot drop tail; KEEP overridden to KILL'
    copies.append(emit_copy(
        'static_prefix85_nnz_fraction',
        'CGNet-static 85% L1 mass; extra=dropped nnz/total nnz, tax=0 compile-time',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_prefix85_nnz'), 0.0, False))
    c7 = emit_copy(
        'BitWave_vs_7serial_NOT_BAR', 'BitWave vs already-7serial PE (not word-add bar)',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_bitwave_vs_7serial'), TAX_RUNTIME, True,
        'informational; KEEP bar is word-add leftover; forced KILL so it cannot enter the idea-line stack')
    c7['keep_kill'] = 'KILL'
    copies.append(c7)
    copies.append(emit_copy('dual_side_r1', 'FireFly-S dual-side r1 leftover', 'r1.conv1',
                            mean('r1.conv1', 'extra_dual'), TAX_RUNTIME, True))
    copies.append(emit_copy('unique_r1', 'identical-row merge r1 leftover', 'r1.conv1',
                            mean('r1.conv1', 'extra_unique'), TAX_RUNTIME, True))
    copies.append(emit_copy(
        'PSN_skip_zero_Y_T', 'skip PSN mix columns when raw S-row is 0 (Y=0, FC1 bias=False)',
        'S0.b0.psn', mean('S0.b0.fc1', 'extra_zero_t_psn'), 0.0, True,
        'on leftover PSN after skip-family empty tokens', baseline=psn_base))
    copies.append(emit_copy(
        'Scrooge_PSN_l1_maxabs',
        'Scrooge DATE26 early-term on leftover PSN; rest=||A_rest||_1*max|Y| (no MAC on skipped Y)',
        'S0.b0.psn', mean('S0.b0.fc1', 'extra_psn_scrooge'),
        psn_scrooge_inspect_tax(Tmix, 'l1_maxabs'), True,
        'inspect tax=1/T for max|Y|; compile-time |A| order; live BN OK because Y already produced',
        baseline=psn_base))
    copies.append(emit_copy(
        'Scrooge_PSN_tot_oracle_NOT_BAR',
        'illegal tot=Σ|A||Y| certificate: tot is leftover T×T work, tax=1',
        'S0.b0.psn', mean('S0.b0.fc1', 'extra_psn_scrooge_tot_oracle'),
        psn_scrooge_inspect_tax(Tmix, 'tot_oracle'), True,
        'previous +85% KEEP was this tot-oracle with tax=0; extra after tax ≤0 → KILL',
        baseline=psn_base))
    c_lr = emit_copy(
        'PSN_lowrank4_SVD', 'ATLIF temporal_factor / da4ml-class rank-4 vs dense T×T',
        'S0.b0.psn', mean('S0.b0.fc1', 'psn_lowrank4'), 0.0, False,
        'compile-time extra 20% but A relF~0.35; not applied; forced KILL until AEE', baseline=psn_base)
    c_lr['keep_kill'] = 'KILL'
    copies.append(c_lr)
    k256_extra = mean('S0.b0.fc1', 'support_K256_compute')
    k256_tax = mean('S0.b0.fc1', 'support_K256_tax')
    c_k = emit_copy(
        'support_code_K256_LUT', 'LUT-DLA/LUT-NN support-code K=256 vs one matvec per nz row',
        'S0.b0.fc1', k256_extra, k256_tax, False,
        'compute extra vs word-adds; NOT lossless on ep34 (Phi AEE fail); forced KILL')
    c_k['keep_kill'] = 'KILL'
    copies.append(c_k)
    copies.append(emit_copy(
        'LUT_GEMM_G4_packed',
        'LUT-GEMM ICLR24 / Platinum: one table-add per nonempty 4-bit group vs nnz word-adds',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_lut_gemm_g4'), 0.0, True,
        'tax=0 because binary S is already packed bits; table lookup charged as 1 word-add; lossless'))
    copies.append(emit_copy(
        'LUT_GEMM_G4_runtime_inspect_tax',
        'same LUT-GEMM G=4 but charge 1/4 inspect tax if groups are not packed',
        'S0.b0.fc1', mean('S0.b0.fc1', 'extra_lut_gemm_g4'), 0.25, True,
        'sensitivity: if runtime bit inspect is 1/G, extra usually KILL'))
    copies.append(emit_copy(
        'static_Cout_drop15', 'compile-time drop 15% smallest-L2 FC1 output channels',
        'S0.b0.fc1', mean('S0.b0.fc1', 'cout_drop15'), 0.0, False,
        'extra is Cout fraction = word-add fraction; quality-changing'))

    k_star = int(round(np.mean([r.get('k15', 96) for r in acc['S0.b0.fc1']])))
    extra_star = mean('S0.b0.fc1', 'extra_prefix_nnz15')

    def cin_drop(m, inp, k=k_star):
        x = inp[0]
        cin = W0.shape[1]
        mask = torch.zeros(cin, device=x.device, dtype=x.dtype)
        mask[order_c[:k].to(x.device)] = 1
        if x.shape[-1] == cin:
            return (x * mask,)
        return None
    cin_drop.tag = 'CIN_K%d' % k_star

    def cout_drop(m, inp):
        return None

    aee_full = float(np.mean(aees))
    aee_cin = run_aee_with_pre_hook(model, mods, names, mods[S0_FC1], cin_drop)

    H = W0.shape[0]
    k_cout = int(0.85 * H)
    keep_h = order_h[:k_cout]

    def pre_cout(m, inp):
        return None

    def hook_cout_weight():
        orig = mods[S0_FC1].weight
        mask = torch.zeros(H, 1, device=orig.device, dtype=orig.dtype)
        mask[keep_h.to(orig.device)] = 1
        mods[S0_FC1].weight = torch.nn.Parameter(orig.detach() * mask, requires_grad=False)
        return orig

    orig_w = hook_cout_weight()
    aee_cout = []
    bucket2 = {}
    h2 = mods[PRED2].register_forward_hook(lambda m, i, o: bucket2.__setitem__('p2', o.detach()))
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p2 = F.interpolate(bucket2['p2'].sum(0), size=(480, 640), mode='bilinear', align_corners=False)
            aee_cout.append(aee(p2, label, mask))
            print('COUT15', name[-12:], round(aee_cout[-1], 4), flush=True)
            del x, label, mask, p2
    h2.remove()
    mods[S0_FC1].weight = orig_w
    aee_cout_m = float(np.mean(aee_cout))

    idea = {
        'mechanism': 'compile_time_channel_drop_to_15pct_nnz_on_S0_fc1',
        'kind': 'improvement of CGNet static prefix; extra is dropped AAC word-adds / remaining word-adds; tax=0',
        'target_op': 'S0.b0.fc1',
        'fair_baseline': 'nnz(S)*Cout word-adds',
        'k_keep': k_star,
        'Cin': 96,
        'extra_save': extra_star,
        'inspect_tax': 0.0,
        'extra_save_after_tax': extra_star,
        'keep_kill': keep_kill(extra_star),
        'lossless': False,
        'AEE_10frame': aee_cin,
        'AEE_full_ref': aee_full,
        'AEE_vs_NB0': aee_cin < NB0,
        'bn_live': bn_live,
        'v_th': v_th,
    }
    new_idea = {
        'mechanism': 'Scrooge_l1_maxabs_on_leftover_PSN_plus_optional_Cin_drop',
        'kind': 'new idea: Scrooge rest=||A_rest||_1*max|Y| on leftover PSN (not tot-oracle, not FC1 tail)',
        'target_op': 'S0.b0.psn',
        'fair_baseline': psn_base,
        'extra_save': mean('S0.b0.fc1', 'extra_psn_scrooge'),
        'inspect_tax': psn_scrooge_inspect_tax(Tmix, 'l1_maxabs'),
        'extra_save_after_tax': mean('S0.b0.fc1', 'extra_psn_scrooge') - psn_scrooge_inspect_tax(Tmix, 'l1_maxabs'),
        'keep_kill': keep_kill(mean('S0.b0.fc1', 'extra_psn_scrooge') - psn_scrooge_inspect_tax(Tmix, 'l1_maxabs')),
        'lossless': True,
        'AEE_10frame': aee_full,
        'AEE_vs_NB0': aee_full < NB0,
        'chain_note': 'FC1 word-adds unchanged; extra is leftover PSN MACs',
        'cout_drop15_AEE': aee_cout_m,
        'cout_drop15_extra': mean('S0.b0.fc1', 'cout_drop15'),
        'cout_drop15_keep_kill': keep_kill(mean('S0.b0.fc1', 'cout_drop15')) if aee_cout_m < NB0 else 'KILL',
    }
    new_idea['keep_kill'] = keep_kill(new_idea['extra_save_after_tax'])

    banned = {'BitWave_vs_7serial_NOT_BAR', 'LUT_GEMM_G4_runtime_inspect_tax', 'Scrooge_PSN_tot_oracle_NOT_BAR'}
    s0_keep = [c for c in copies
               if c['keep_kill'] == 'KEEP' and c.get('lossless', False)
               and c['mechanism'] not in banned]
    # quality-changing KEEP only if AEE vs NB0
    if idea['keep_kill'] == 'KEEP' and idea['AEE_vs_NB0']:
        s0_keep.append(idea)
    if (new_idea.get('cout_drop15_keep_kill') == 'KEEP'
            and aee_cout_m < NB0):
        s0_keep.append({
            'mechanism': 'static_Cout_drop15',
            'target_op': 'S0.b0.fc1',
            'extra_save_after_tax': mean('S0.b0.fc1', 'cout_drop15'),
        })
    stack = {
        'mechanism': 'stack_KEEP:' + ('+'.join(c['mechanism'] for c in s0_keep) or 'none'),
        'kind': 'stack of KEEP copies on leftover; max within same op, sum across fc1+psn chain',
        'target_op': 'S0.b0.fc1+psn',
        'fair_baseline': 'fc1 nnz(S)*Cout word-adds + psn n_tokens*Cout*T*T',
        'keepers': [c['mechanism'] for c in s0_keep],
        'independence_approx': False,
        'stack_rule': 'conservative max per op; no product; chain extra = weighted sum if both ops KEEP',
        'extra_save_after_tax': max((c['extra_save_after_tax'] for c in s0_keep), default=0.0),
    }
    fc1_keeps = [c['extra_save_after_tax'] for c in s0_keep if c.get('target_op') == 'S0.b0.fc1']
    psn_keeps = [c['extra_save_after_tax'] for c in s0_keep if 'psn' in str(c.get('target_op', ''))]
    fc1_e = max(fc1_keeps) if fc1_keeps else 0.0
    psn_e = max(psn_keeps) if psn_keeps else 0.0
    # chain weights from last frame means
    w_fc1 = mean('S0.b0.fc1', 'fc1_word_adds')
    w_psn = mean('S0.b0.fc1', 'psn_baseline_macs')
    chain = (fc1_e * w_fc1 + psn_e * w_psn) / max(w_fc1 + w_psn, 1.0)
    stack['fc1_extra'] = fc1_e
    stack['psn_extra'] = psn_e
    stack['chain_extra_after_tax'] = chain
    stack['keep_kill'] = keep_kill(max(stack['extra_save_after_tax'], chain))

    denom = {k: {f: mean(k, f) for f in (
        'elem_nnz', 'zero_row_frac', 'baseline_aac_word_adds', 'W_bit_density', 'extra_unique'
    ) if f in acc[k][0]} for k in acc}
    out = {
        'denominator': {
            'identity': 'AT-LIF {0,theta} absorb; binary GeMM; PED separate',
            'fair_baseline': 'nnz(S)*Cout word-adds after skip S==0; PSN leftover = n_tokens*Cout*T*T',
            'skip_family_not_recounted': True,
            'ops': denom,
            'bn_live_S0_fc1': bn_live,
            'sn2_thresh': v_th,
            'WBITS': WBITS,
        },
        'copies': copies,
        'stack': stack,
        'improvement': idea,
        'new_idea': new_idea,
        'AEE_full': aee_full,
        'KEEP_threshold': KEEP,
        'WBITS': WBITS,
        'frames': names,
    }
    dest = HERE / 'results'
    dest.mkdir(exist_ok=True)
    (dest / 'remaining_copies.json').write_text(json.dumps(out, indent=2))
    print('KEEP', [(c['mechanism'], c['keep_kill'], round(c['extra_save_after_tax'], 4)) for c in copies], flush=True)
    print('STACK', stack, flush=True)
    print('IDEA', idea['mechanism'], idea['keep_kill'], round(idea['extra_save_after_tax'], 4), 'AEE', idea['AEE_10frame'], flush=True)
    print('NEW', new_idea['mechanism'], new_idea['keep_kill'], round(new_idea['extra_save_after_tax'], 4), flush=True)


if __name__ == '__main__':
    main()
