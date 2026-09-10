"""Input-only spatial vs temporal-column production, a pre-training probe.

Masks are decided from existing sn1 amplitudes before calling conv1. Thresholds
come only from train32. Multiplying dense conv output is the GPU reference for
the hypothetical skipped production, not a measurement of GPU/RTL speedup.
Real dynamic BN and the complete T10 PSN follow the masked raw Y.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from run_patch_probe import probe, RES

BLOCK = 8
TARGET = RES+'1.conv1.0'


def scores(x):
    # Halo is read by the input-only predictor, including for discarded tiles.
    z = x.detach().abs().sum(2)
    return F.avg_pool2d(z, BLOCK+2, stride=BLOCK, padding=1,
                        count_include_pad=True)*(BLOCK+2)**2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    out = args.root/'algorithm/patch_probe'
    train = json.loads((args.root/'algorithm/direct_code_integer/run.json').read_text())['train']
    names = json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:10]
    samples = []
    class Captured(Exception):
        pass
    def before(m, inputs):
        samples.append(scores(inputs[0]).cpu())
        raise Captured()
    handle = modules[TARGET].register_forward_pre_hook(before)
    with torch.no_grad():
        for name in train:
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except Captured:
                pass
    handle.remove()
    s = torch.stack(samples)
    quantiles = (0.25, 0.5, 0.75)
    thresholds = {mode: {str(q): float(torch.quantile(
        (s.sum(1) if mode=='spatial' else s).reshape(-1), q)) for q in quantiles}
        for mode in ('spatial', 'temporal')}
    original = modules[TARGET].forward
    axis, threshold, frame_stats = 'parent', None, []
    def forward(x):
        sc = scores(x)
        if axis == 'parent':
            tile = torch.ones_like(sc, dtype=torch.bool)
        elif axis.startswith('spatial'):
            tile = (sc.sum(0, keepdim=True)>threshold).expand_as(sc)
        else:
            tile = sc>threshold
        t, b, c, h, w = x.shape
        produce = tile.repeat_interleave(BLOCK, -2).repeat_interleave(BLOCK, -1)
        # Decision above precedes the expensive conv; no full-Y/flow oracle.
        y = original(x)*produce[:, :, None].to(x.dtype)
        count = x.detach().ne(0).sum(2).float()
        conv_terms = F.conv2d(count, torch.ones(1,1,3,3,device=x.device), padding=1)
        live = conv_terms.ne(0)&produce
        required = F.max_pool2d(produce.float(), 3, stride=1, padding=1).bool()
        aligned_words = required.reshape(t,b,h,w//4,4).any(-1)
        actual_g = x.detach().ne(0)&required[:, :, None]
        nonempty_words = actual_g.reshape(t,b,c//16,16,h,w//4,4).any(3).any(-1)
        frame_stats.append(dict(
            produced_time_tile_fraction=float(tile.float().mean()),
            produced_space_tile_fraction=float(tile.any(0).float().mean()),
            remaining_conv1_active_terms=int((conv_terms*produce).sum())*96,
            parent_conv1_active_terms=int(conv_terms.sum())*96,
            surviving_PSN_input_columns=int(live.sum()),
            parent_nonzero_PSN_input_columns=int(conv_terms.ne(0).sum()),
            PSN_positions_not_proved_default=int(live.any(0).sum()),
            nonempty_C16_P4_source_words=int(nonempty_words.sum()),
            halo_C16_P4_geometric_source_words=int(aligned_words.sum())*(c//16),
            source_score_elements_examined=x.numel(),
            decision_count=tile.numel() if not axis.startswith('spatial') else tile[0].numel(),
            note='source-score cost, global threshold compare and dense sn1 remain; these are counts, not cycles'))
        return y
    modules[TARGET].forward = forward
    def consumer(m, inputs):
        g = inputs[0].detach().ne(0)
        h,w = g.shape[-2:]
        fanout = F.conv2d(torch.ones(1,1,h,w,device=g.device),
                         torch.ones(1,1,3,3,device=g.device),padding=1)[0,0]
        frame_stats[-1]['conv2_active_terms'] = int((g.sum((0,1,2))*fanout).sum())*96
        frame_stats[-1]['conv2_source_density'] = float(g.float().mean())
    hook = modules[RES+'1.conv2.0'].register_forward_pre_hook(consumer)
    summaries = {}
    with torch.no_grad():
        for name_axis in ['parent']+[f'{m}_{q}' for m in thresholds for q in thresholds[m]]:
            axis = name_axis
            if axis != 'parent':
                m,q = axis.split('_')
                threshold = thresholds[m][q]
            frame_stats = []
            rows = []
            started = time.monotonic()
            for name in names:
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480,640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask], dim=1)
                total,pixels = float(error.double().sum()),error.numel()
                rows.append(dict(file=name,valid_pixels=pixels,aee_sum=total,AEE=total/pixels))
            summaries[axis] = dict(**summarize(rows,False), threshold=threshold if axis!='parent' else None,
                wall_seconds=time.monotonic()-started,
                counts={k: sum(v[k] for v in frame_stats)/len(frame_stats) for k in frame_stats[0] if k!='note'})
            probe.save_json(out/'production_mask_valid10_summary.json', summaries)
            probe.save_json(out/(axis+'_production_frames.json'),
                            [dict(**row, **count) for row,count in zip(rows,frame_stats)])
            print('RESULT', axis, json.dumps(summaries[axis]), flush=True)
    hook.remove()
    probe.save_json(out/'production_mask_run.json',dict(train=train,valid=names,thresholds=thresholds,
        spatial_block=BLOCK, train_quantiles=list(quantiles), fitting='threshold calibration only, no gradient training',
        claim='exploratory input-only mask / AEE / operation counts, not a complete SBNet port or hardware speedup',
        BN='original full-domain dynamic BN retained; no active-only normalization'))


if __name__ == '__main__':
    main()
