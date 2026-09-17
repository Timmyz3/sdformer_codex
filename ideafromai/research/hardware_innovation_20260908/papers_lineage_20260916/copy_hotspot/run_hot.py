"""Target the expensive ops: patch r0 (~21.6% dense MAC).

Copy:
  1) Zhang DLSS onto r0 residual: if events look like the previous frame, reuse the
     previous residual_encoding output (skip r0.conv1+conv2).
  2) C-STEP early-window silence onto r0.conv2 *spatial tiles* (channel silence was 0):
     if a tile is zero in T0-T1, zero T2-T9 there.

Quality = preds.2 bilinear AEE on the 10 local GT frames. Skip fraction is a MAC proxy
for r0 only, not whole-net speedup.
"""
from __future__ import annotations

import json, os, sys, time
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
sys.path[:0] = [str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
from run_bn_probe import build_model, input_frame
from spikingjelly.activation_based import functional

R0 = 'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding'
CONV2 = R0 + '.resblocks.0.conv2.0'
PRED2 = 'sttmultires_unet.preds.2'
TILE = 8


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    n = int(mask.sum())
    return float(err[mask].sum()) / n, n


def events_binary(name):
    seq = name.rsplit('_', 1)[0]
    x = np.abs(np.load(DATA / 'event_tensors/10bins/left' / seq / name).astype(np.float32))
    return (x > 0).astype(np.float32)


def iou(a, b):
    inter = float(((a > 0) & (b > 0)).sum())
    union = float(((a > 0) | (b > 0)).sum())
    return inter / max(union, 1.0)


def upsample(flow):
    return F.interpolate(flow, size=(480, 640), mode='bilinear', align_corners=False)


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
        checkpoint=INCOMING / 'checkpoint_epoch34.pth',
        data=DATA,
    )
    t0 = time.time()
    model, *_ = build_model(args)
    mods = dict(model.named_modules())
    bucket = {}

    def save_pred(m, inp, output):
        bucket['p2'] = output.detach()

    def save_r0(m, inp, output):
        bucket['r0'] = output.detach()

    h_p2 = mods[PRED2].register_forward_hook(save_pred)
    h_r0 = mods[R0].register_forward_hook(save_r0)

    # ---- pass 1: baseline + capture r0 features and event IoU ----
    frames = []
    prev_bin = None
    r0_cache = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p2 = upsample(bucket.pop('p2').sum(0))
            r0 = bucket.pop('r0').contiguous()
            val, n = aee(p2, label, mask)
            binary = events_binary(name)
            rec = {'file': name, 'AEE_full': val, 'valid_pixels': n, 'r0_shape': list(r0.shape)}
            if prev_bin is not None:
                rec['IOU'] = iou(binary, prev_bin)
            frames.append(rec)
            r0_cache.append(r0.cpu())
            prev_bin = binary
            print('BASE', name, round(val, 4), rec.get('IOU'), flush=True)
            del x, label, mask, p2, r0

    h_r0.remove()

    # ---- pass 2: Zhang DLSS on r0 — reuse previous residual when IoU>=thr ----
    inject = {'use_prev': False, 'prev': None}

    def replace_r0(m, inp, output):
        if inject['use_prev']:
            return inject['prev'].to(output.device, output.dtype)
        return output

    h_rep = mods[R0].register_forward_hook(replace_r0)
    zhang = {}
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        aees = []
        skipped = 0
        with torch.no_grad():
            for i, name in enumerate(names):
                functional.reset_net(model)
                x, label, mask = input_frame(DATA, name)
                do = i > 0 and frames[i].get('IOU', 0) >= thr
                inject['use_prev'] = do
                inject['prev'] = r0_cache[i - 1] if do else None
                if do:
                    skipped += 1
                model(x)
                p2 = upsample(bucket.pop('p2').sum(0))
                val, _ = aee(p2, label, mask)
                aees.append(val)
                del x, label, mask, p2
        zhang[f'IOU>={thr}'] = {
            'AEE': float(np.mean(aees)),
            'skip_frames': skipped,
            'skip_frac': skipped / max(len(names) - 1, 1),
            'r0_mac_saved_proxy': 0.216 * skipped / len(names),
        }
        print('ZHANG', thr, zhang[f'IOU>={thr}'], flush=True)
    h_rep.remove()

    # ---- pass 3: C-STEP spatial-tile early silence on conv2 input ----
    stats = {'tiles': 0, 'silent_early': 0, 'late_nnz_if_silent': 0.0, 'late_nnz_if_active': 0.0,
             'n_silent': 0, 'n_active': 0}

    def conv2_gate(m, args):
        x = args[0]
        if x.ndim == 4:
            x = x.unsqueeze(1)
        # x: T,B,C,H,W
        T, B, C, H, W = x.shape
        assert H % TILE == 0 and W % TILE == 0
        xt = x.reshape(T, B, C, H // TILE, TILE, W // TILE, TILE)
        early = xt[:2].abs().sum(dim=(0, 1, 2, 4, 6))  # Hy, Wx
        silent = early == 0
        late = xt[2:].abs().sum(dim=(0, 1, 2, 4, 6))
        stats['tiles'] += int(silent.numel())
        stats['silent_early'] += int(silent.sum())
        if silent.any():
            stats['late_nnz_if_silent'] += float(late[silent].sum())
            stats['n_silent'] += int(silent.sum())
        if (~silent).any():
            stats['late_nnz_if_active'] += float(late[~silent].sum())
            stats['n_active'] += int((~silent).sum())
        if not silent.any():
            return
        mask = silent[None, None, None, :, None, :, None].to(x.dtype)
        xt = xt.clone()
        xt[2:] = xt[2:] * (1.0 - mask)
        return (xt.reshape(T, B, C, H, W),)

    h_c = mods[CONV2].register_forward_pre_hook(conv2_gate)
    cstep_aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p2 = upsample(bucket.pop('p2').sum(0))
            val, _ = aee(p2, label, mask)
            cstep_aees.append(val)
            print('CSTEP', name, round(val, 4), flush=True)
            del x, label, mask, p2
    h_c.remove()

    # ---- pass 4: pixel-level early silence (C-STEP adapted; 8x8 tiles were never silent) ----
    pix = {'pixels': 0, 'silent_early': 0}

    def conv2_pixel_gate(m, args):
        x = args[0]
        if x.ndim == 4:
            x = x.unsqueeze(1)
        T, B, C, H, W = x.shape
        early = x[:2].abs().sum(dim=(0, 1, 2))  # H,W
        silent = early == 0
        pix['pixels'] += int(silent.numel())
        pix['silent_early'] += int(silent.sum())
        if not silent.any():
            return
        x = x.clone()
        x[2:] = x[2:] * (~silent).to(x.dtype)[None, None, None]
        return (x,)

    h_px = mods[CONV2].register_forward_pre_hook(conv2_pixel_gate)
    pix_aees = []
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p2 = upsample(bucket.pop('p2').sum(0))
            val, _ = aee(p2, label, mask)
            pix_aees.append(val)
            print('PIXEL', name, round(val, 4), flush=True)
            del x, label, mask, p2
    h_px.remove()
    h_p2.remove()
    pix_frac = pix['silent_early'] / max(pix['pixels'], 1)

    silent_frac = stats['silent_early'] / max(stats['tiles'], 1)
    late_s = stats['late_nnz_if_silent'] / max(stats['n_silent'], 1)
    late_a = stats['late_nnz_if_active'] / max(stats['n_active'], 1)
    # later-T is 8/10 of conv2 if fully skipped on silent tiles
    conv2_saved = silent_frac * 0.8 * 0.1068

    summary = {
        'seconds': time.time() - t0,
        'gpu': torch.cuda.get_device_name(),
        'hotspots_from_profile': {
            'r0': '21.60% dense MAC (conv1+conv2 each 10.68%, nnz 4.4%/3.6%)',
            'swin_mlp': '26.24%',
            'swin_attn': '12.64%',
            'decoder_all': '8.08% (decoder2 alone 2.68%)',
        },
        'AEE_full': float(np.mean([f['AEE_full'] for f in frames])),
        'zhang_r0_reuse': zhang,
        'cstep_spatial_tile8': {
            'AEE': float(np.mean(cstep_aees)),
            'silent_early_tile_frac': silent_frac,
            'late_mass_silent_vs_active': [late_s, late_a],
            'conv2_mac_saved_proxy': conv2_saved,
            'note': 'zeros later T on tiles silent in T0-T1; live BN rest of net unchanged',
        },
        'cstep_pixel_early_silence': {
            'AEE': float(np.mean(pix_aees)),
            'silent_early_pixel_frac': pix_frac,
            'conv2_mac_saved_proxy': pix_frac * 0.8 * 0.1068,
            'note': 'later T zeroed at pixels with no spike in T0-T1',
        },
        'frames': frames,
        'NB0_ref': 1.454602861,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'frames'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
