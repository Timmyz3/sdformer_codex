#!/usr/bin/env python3
"""A-line decoder0 probe (inference only, no training, no GT used for decisions
except the explicitly labelled valid-mask variant).
Model: SDformerFlow-v2 MS_SpikingformerFlowNet_en4, W271482 (native_crop9_fresh),
all78 current-stat BN ('P_paper'), FP32, full9 480x640, torch SNN backend.
Eval set: R27 dev64 snapshot (8 seq x 8 frames, training-domain dev set).
"""
import copy, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HOME = Path.home()
SRC = HOME/'work/sdformer_codex/SDformer/third_party/SDformerFlow'
CKPT = HOME/'work/sdformer_codex/research/flow_story_20261016/baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
CKPT_SHA = '1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
DATA = HOME/'work/sdformer_codex/research/flow_story_20261019/round27/dev64_source_snapshot/saved_flow_data'
R27_INIT = HOME/'work/sdformer_codex/research/flow_story_20261019/round27/actual_v1/formal_native_v1/predictions/initial'
OUT = Path(__file__).resolve().parent
THRESH_PRIMARY = [0.1, 0.25, 0.5]
THRESH_SUPPLEMENTARY = [0.75, 1.0, 1.5, 2.0, 3.0, 5.0]  # added after v1 to locate the quality-neutral edge
THRESH = THRESH_PRIMARY + THRESH_SUPPLEMENTARY
BLOCKS = [1, 8, 16, 32]
NEUTRAL_REL = 0.01   # pre-declared: quality-neutral = oracle pixel-pooled AEE <= +1.0% vs full model
GATE_REL = 0.03      # project gate (full-GT only; B/L not evaluated here)
DECISION_FRAC = 0.36

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()

def prepare_input(events, device):
    # identical to research_20261013/breadth/evaluate_breadth.py::prepare_input
    chunk = torch.from_numpy(np.ascontiguousarray(events, dtype=np.float32)).unsqueeze(0).to(device)
    chunk = torch.cat((torch.relu(chunk).unsqueeze(2), torch.relu(-chunk).unsqueeze(2)), dim=2)
    nz = chunk[chunk != 0]
    assert nz.numel() > 0
    low, high = nz.min(), nz.max()
    if low != high: chunk[chunk != 0] = (chunk[chunk != 0] - low) / (high - low)
    return chunk

def block_mean(x, w, b):
    """x,w: (H,W). returns block sum(x*w)/sum(w) and sum(w) per block."""
    H, W = x.shape
    s = (x*w).reshape(H//b, b, W//b, b).sum(axis=(1, 3))
    n = w.reshape(H//b, b, W//b, b).sum(axis=(1, 3))
    return s, n

def main():
    t0 = time.time()
    assert sha(CKPT) == CKPT_SHA, 'checkpoint SHA'
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    sys.path.insert(0, str(SRC)); os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    saved = torch.load(CKPT, map_location='cpu', weights_only=True)
    config = copy.deepcopy(saved['effective_config'])
    config['swin_transformer']['input_size'] = [480, 640]
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(saved['model_state_dict'], strict=True); del saved
    model.to(device); functional.set_step_mode(model, 'm'); configure_snn_backend(model, device, config, PSN)
    model.eval(); nbn = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            nbn += 1; m.track_running_stats = False; m.running_mean = m.running_var = m.num_batches_tracked = None
    model.requires_grad_(False)
    assert nbn == 78, nbn
    raw_shapes = {}
    def hook(mod, ins, out): raw_shapes['multires'] = [list(o.shape) for o in out]
    hs = []
    for k, pm in enumerate(model.sttmultires_unet.preds):
        hs.append(pm.register_forward_hook(lambda mod, ins, out, k=k: raw_shapes.__setitem__('pred%d' % k, list(out.shape))))

    names = sorted(os.listdir(DATA/'gt_tensors')); assert len(names) == 64
    acc = {'pix': {k: [0.0, 0] for k in range(4)}}
    frames = []
    # pooled counters
    pix_stats = {}
    blk = {(v, b, t): dict(unchanged=0, total=0, unchanged_area=0.0, total_area=0.0) for v in ('all', 'valid') for b in BLOCKS for t in THRESH}
    oracle = {(v, b, t): [0.0, 0, []] for v in ('all', 'valid') for b in BLOCKS for t in THRESH}
    r27_maxdiff = 0.0
    for i, name in enumerate(names):
        seq = name[:-len('_0000.npy')]
        ev = np.load(DATA/'event_tensors/10bins/left'/seq/name)
        gt = np.load(DATA/'gt_tensors'/name).astype(np.float64)
        valid = np.load(DATA/'mask_tensors'/name).astype(bool)
        chunk = prepare_input(ev, device)
        functional.reset_net(model)
        with torch.inference_mode():
            out = model(chunk)['flow']
        heads = [o[0].detach().float().cpu().numpy() for o in out]
        assert len(heads) == 4 and all(x.shape == (2, 480, 640) and np.isfinite(x).all() for x in heads)
        ref = np.load(R27_INIT/name.replace('.npy', '.npz'))['head3']
        dref = np.abs(heads[3].astype(np.float64) - ref); md = float(dref.max()); r27_maxdiff = max(r27_maxdiff, md)
        n = int(valid.sum())
        fr = dict(target_id=name[:-4], valid_pixels=n, r27_head3_maxabs=md, r27_head3_meanabs=float(dref.mean()), r27_head3_frac_gt_0p01=float((dref.max(0) > 0.01).mean()))
        for k in range(4):
            e = np.sqrt(((heads[k].astype(np.float64) - gt)**2).sum(0))
            s = float(e[valid].sum()); acc['pix'][k][0] += s; acc['pix'][k][1] += n
            fr['AEE_head%d' % k] = s/n
        f3 = heads[3].astype(np.float64); f0 = heads[0].astype(np.float64)
        dd = np.sqrt(((f3 - f0)**2).sum(0))
        e3 = np.sqrt(((f3 - gt)**2).sum(0)); e0 = np.sqrt(((f0 - gt)**2).sum(0))
        fr['mean_abs_d3_d0_all'] = float(dd.mean()); fr['mean_abs_d3_d0_valid'] = float(dd[valid].mean())
        for v in ('all', 'valid'):
            w = np.ones_like(dd) if v == 'all' else valid.astype(np.float64)
            for b in BLOCKS:
                s, cnt = block_mean(dd, w, b)
                has = cnt > 0
                mean = np.where(has, s/np.maximum(cnt, 1), np.inf)
                for t in THRESH:
                    keep0 = (mean < t) & has
                    r = blk[(v, b, t)]
                    r['unchanged'] += int(keep0.sum()); r['total'] += int(has.sum())
                    # area fraction over full image (all pixels in block)
                    r['unchanged_area'] += float(keep0.sum()*b*b); r['total_area'] += float(has.sum()*b*b)
                    km = np.kron(keep0, np.ones((b, b), dtype=bool)) if b > 1 else keep0
                    eo = np.where(km, e0, e3)
                    so = float(eo[valid].sum()); oracle[(v, b, t)][0] += so; oracle[(v, b, t)][1] += n
                    oracle[(v, b, t)][2].append(so/n)
        frames.append(fr)
        print(i, name, {k: round(fr['AEE_head%d' % k], 4) for k in range(4)}, 'r27diff', md, flush=True)
    [x.remove() for x in hs]
    full = acc['pix'][3][0]/acc['pix'][3][1]
    res = dict(schema='aline_decoder0_probe_v1', status='COMPLETE', created_local=time.strftime('%Y-%m-%d %H:%M:%S %z'),
               checkpoint=str(CKPT), checkpoint_sha256=CKPT_SHA, BN='all78 current-stat (P_paper)', precision='FP32 no TF32',
               geometry='full 480x640', eval_set='R27 dev64 snapshot (8 seq x 8 frames; training-domain dev; not official DSEC test)',
               gpu=torch.cuda.get_device_name(0), training_steps=0, model_forwards=len(names),
               raw_pred_shapes_before_upsample=dict(raw_shapes),
               upsampling='model native: sum over T then F.interpolate nearest to 480x640',
               r27_initial_head3_max_abs_diff=r27_maxdiff,
               known_reference={'R27 original native dev64 pixel-pooled AEE': 1.757176, 'frame-mean AEE': 1.707768,
                                'source': 'research/flow_story_20261019/ROUND27_MILESTONE.md'},
               AEE_pixel_pooled={'head%d' % k: acc['pix'][k][0]/acc['pix'][k][1] for k in range(4)},
               AEE_frame_mean={'head%d' % k: float(np.mean([f['AEE_head%d' % k] for f in frames])) for k in range(4)},
               valid_pixels=acc['pix'][3][1])
    d0 = res['AEE_pixel_pooled']['head0']
    res['decoder0_only'] = dict(AEE_pixel=d0, delta_abs=d0-full, delta_rel=(d0-full)/full,
                                AEE_frame_mean=res['AEE_frame_mean']['head0'],
                                delta_rel_frame_mean=(res['AEE_frame_mean']['head0']-res['AEE_frame_mean']['head3'])/res['AEE_frame_mean']['head3'])
    table = []
    for (v, b, t), r in blk.items():
        o = oracle[(v, b, t)]
        oa = o[0]/o[1]
        table.append(dict(threshold_role='primary' if t in THRESH_PRIMARY else 'supplementary', variant=v, block=b, threshold_px=t, unchanged_fraction=r['unchanged']/r['total'] if r['total'] else None,
                          unchanged_blocks=r['unchanged'], counted_blocks=r['total'],
                          oracle_AEE_pixel=oa, oracle_rel=(oa-full)/full, oracle_AEE_frame_mean=float(np.mean(o[2])),
                          quality_neutral_1pct=(oa-full)/full <= NEUTRAL_REL, within_3pct_gate=(oa-full)/full <= GATE_REL))
    res['fraction_table'] = table
    neutral = [r for r in table if r['quality_neutral_1pct'] and r['block'] > 1]
    neutral_primary = [r for r in neutral if r['threshold_role'] == 'primary']
    bestp = max(neutral_primary, key=lambda r: r['unchanged_fraction']) if neutral_primary else None
    best = max(neutral, key=lambda r: r['unchanged_fraction']) if neutral else None
    res['decision'] = dict(rule='unchanged fraction at quality-neutral threshold < 36% -> A line stops',
                           quality_neutral_definition='oracle pixel-pooled full-GT AEE <= +1.0% vs full model (pre-declared in script); B/L not evaluated',
                           best_neutral_block_entry_primary_thresholds=bestp, best_neutral_block_entry_all_thresholds=best,
                           verdict=None if best is None else ('A_STOP' if best['unchanged_fraction'] < DECISION_FRAC else 'A_NOT_STOPPED_BY_THIS_PROBE'))
    res['frames'] = frames
    res['elapsed_s'] = time.time()-t0
    res['peak_allocated_GB'] = torch.cuda.max_memory_allocated()/1e9
    (OUT/'RESULTS.json').write_text(json.dumps(res, indent=2, allow_nan=False)+'\n')
    print('DONE', json.dumps({k: res[k] for k in ('AEE_pixel_pooled', 'decoder0_only', 'decision')}, indent=1))

if __name__ == '__main__':
    main()
