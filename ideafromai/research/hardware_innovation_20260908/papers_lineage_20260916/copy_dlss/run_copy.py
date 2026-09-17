"""Copy Zhang CICC'26 DLSS onto SDformerFlow: skip decoder2 when inputs look similar.

Measures AEE on the 10 local GT frames. Does not claim RTL speedup.
Decoder2 skip uses current-frame preds.1 (shallower net), not previous flow.
"""
from __future__ import annotations

import csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ALGO = BASE / 'algorithm'
REPO = Path('/home/zhumd/work/sdformer_codex/SDformer')
DATA = REPO / 'data/Datasets/DSEC/saved_flow_data'
INCOMING = REPO / 'hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs'
sys.path[:0] = [str(ALGO), str(ALGO / 'nrv_cost_probe')]
os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'

from run_bn_probe import build_model, input_frame  # noqa: E402
from types import SimpleNamespace
from spikingjelly.activation_based import functional


def aee(pred, label, mask):
    err = (pred - label).square().sum(1).sqrt()
    n = int(mask.sum())
    return float(err[mask].sum()) / n, n


def occupancy(path):
    x = np.abs(np.load(path).astype(np.float32))
    mag = x / x.max() if x.max() > 0 else x
    binary = (x > 0).astype(np.float32)
    return mag, binary


def zhang_as(a, b):
    return 1.0 - float(np.mean(np.abs(a - b)))


def iou(a, b):
    inter = float(((a > 0) & (b > 0)).sum())
    union = float(((a > 0) | (b > 0)).sum())
    return inter / max(union, 1.0)


def upsample_flow(low, hw=(480, 640)):
    return F.interpolate(low, size=hw, mode='bilinear', align_corners=False)


def main():
    out = HERE / 'results'
    out.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in sorted((DATA / 'gt_tensors').glob('*.npy'))]
    args = SimpleNamespace(
        code_root=REPO,
        config=INCOMING / 'dsec_c12_alpha0125_ep29_resume5_20260830.yml',
        checkpoint=INCOMING / 'checkpoint_epoch34.pth',
        data=DATA,
    )
    t0 = time.time()
    model, cfg, installed, attention = build_model(args)
    modules = dict(model.named_modules())
    bucket = {}

    def hook(key):
        def fn(module, inputs, output):
            bucket[key] = output.detach()
        return fn

    h1 = modules['sttmultires_unet.preds.1'].register_forward_hook(hook('p1'))
    h2 = modules['sttmultires_unet.preds.2'].register_forward_hook(hook('p2'))

    rows = []
    prev_mag = prev_bin = None
    with torch.no_grad():
        for name in names:
            functional.reset_net(model)
            x, label, mask = input_frame(DATA, name)
            model(x)
            p1 = bucket.pop('p1').sum(0)  # [1,2,60,80]
            p2 = bucket.pop('p2').sum(0)  # [1,2,120,160]
            flow1 = upsample_flow(p1)
            flow2 = upsample_flow(p2)
            aee1, n = aee(flow1, label, mask)
            aee2, _ = aee(flow2, label, mask)
            ev = DATA / 'event_tensors/10bins/left' / name.rsplit('_', 1)[0] / name
            mag, binary = occupancy(ev)
            rec = {
                'file': name,
                'valid_pixels': n,
                'AEE_preds1': aee1,
                'AEE_preds2': aee2,
                'p1_hw': list(p1.shape[-2:]),
                'p2_hw': list(p2.shape[-2:]),
            }
            if prev_mag is not None:
                rec['AS_l1'] = zhang_as(mag, prev_mag)
                rec['AS_bin'] = zhang_as(binary, prev_bin)
                rec['IOU'] = iou(binary, prev_bin)
            # tile occupancy at 120x160 (preds.2 grid): 8x8 tiles
            occ = binary.max(axis=0)  # H,W 480x640
            occ_p2 = F.avg_pool2d(torch.from_numpy(occ)[None, None], 4).numpy()[0, 0]
            tiles = []
            for y in range(0, 120, 8):
                for x0 in range(0, 160, 8):
                    tiles.append(float(occ_p2[y:y + 8, x0:x0 + 8].mean()))
            rec['tile8_occ_mean'] = float(np.mean(tiles))
            rec['tile8_occ'] = tiles
            # quality oracle: mix P1-upsampled-to-P2-res with P2 by tile occupancy
            p1_at_p2 = upsample_flow(p1, (120, 160))
            mix_rows = {}
            for thr in (0.01, 0.02, 0.05, 0.10, 0.20):
                use_p1 = np.zeros((120, 160), np.float32)
                skipped = 0
                for i, y in enumerate(range(0, 120, 8)):
                    for j, x0 in enumerate(range(0, 160, 8)):
                        if tiles[i * 20 + j] < thr:
                            use_p1[y:y + 8, x0:x0 + 8] = 1
                            skipped += 1
                w = torch.from_numpy(use_p1)[None, None].to(p2.device)
                mixed = upsample_flow(p1_at_p2 * w + p2 * (1 - w))
                mix_aee, _ = aee(mixed, label, mask)
                mix_rows[str(thr)] = {'AEE': mix_aee, 'skip_tiles': skipped, 'skip_frac': skipped / 300.0}
            rec['tile_mix_oracle'] = mix_rows  # uses both P1 and P2; not a valid skip with live BN
            rows.append(rec)
            prev_mag, prev_bin = mag, binary
            print('FRAME', name, 'P1', round(aee1, 4), 'P2', round(aee2, 4), rec.get('IOU'), flush=True)
            del x, label, mask, p1, p2, flow1, flow2

    h1.remove()
    h2.remove()

    mean1 = float(np.mean([r['AEE_preds1'] for r in rows]))
    mean2 = float(np.mean([r['AEE_preds2'] for r in rows]))
    # consecutive skip: if similarity high, use current P1 else P2
    policies = {}
    for key, thrs in (('AS_l1', [0.90, 0.95, 0.97, 0.98, 0.99]),
                      ('AS_bin', [0.55, 0.60, 0.65, 0.70, 0.75]),
                      ('IOU', [0.50, 0.55, 0.60, 0.65, 0.70])):
        for thr in thrs:
            chosen = []
            skipped = 0
            for i, r in enumerate(rows):
                if i == 0 or key not in r:
                    chosen.append(r['AEE_preds2'])
                elif r[key] >= thr:
                    chosen.append(r['AEE_preds1'])
                    skipped += 1
                else:
                    chosen.append(r['AEE_preds2'])
            policies[f'{key}>={thr}'] = {
                'AEE': float(np.mean(chosen)),
                'skip_frames': skipped,
                'skip_frac': skipped / max(len(rows) - 1, 1),
            }

    summary = {
        'frames': names,
        'n': len(rows),
        'seconds': time.time() - t0,
        'gpu': torch.cuda.get_device_name(),
        'AEE_always_P1_skip_decoder2': mean1,
        'AEE_always_P2_full_decoder2': mean2,
        'NB0_diverse10_ref': 1.454602861,
        'NB0_valid825_ref': 1.445352536,
        'delta_P1_minus_P2': mean1 - mean2,
        'consecutive_policies': policies,
        'note': 'P1 skip is Zhang-style shallower net on CURRENT frame. Tile mix still computes P2 (live BN); quality oracle only.',
        'identity': 'AT-LIF absorb; no RTL; no PPA',
    }
    (out / 'frames.json').write_text(json.dumps(rows, indent=2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
