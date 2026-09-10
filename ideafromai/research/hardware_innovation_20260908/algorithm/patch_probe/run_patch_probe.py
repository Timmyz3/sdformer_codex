"""Costly patch placement: zero-training sensitivity and actual support census.

This uses the saved integer-source/coarse-head student, with its inherited
whole-network numerical settings. Masks are FP weight ablations, not new W8
deployment or trained pruning. All BN, full T10 PSN, theta and shortcuts run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'nrv_cost_probe'))
import run_probe as probe

PATCH = 'sttmultires_unet.encoders.swin3d.patch_embed.'
RES = PATCH+'residual_encoding.resblocks.'
AXES = ['parent']+[f'r{b}_{mode}' for b in (0, 1)
                   for mode in ('drop_branch', 'hidden50', 'row2of4', 'group16')]


def support_stats(x):
    # Actual Conv input is [T,B,C,H,W]; no surrogate binary-valued assumption.
    x = x.detach()
    t, b, c, h, w = x.shape
    g = x.ne(0)
    spatial = g.any(2).reshape(t*b, 1, h, w)
    halo = F.max_pool2d(spatial.float(), 3, stride=1, padding=1).bool()
    all_t = g.any((0, 1, 2))[None, None]
    halo_all_t = F.max_pool2d(all_t.float(), 3, stride=1, padding=1).bool()
    # Valid (non-padding) fanout at every source location, including borders.
    fanout = F.conv2d(torch.ones(1, 1, h, w, device=x.device),
                     torch.ones(1, 1, 3, 3, device=x.device), padding=1)
    source_fanout = int((g.sum((0, 1, 2))*fanout[0, 0]).sum())
    nz = x[g]
    stats = dict(shape=list(x.shape), active_scalar=int(g.sum()), scalar_count=g.numel(),
        active_fraction=float(g.float().mean()),
        nonzero_amplitude_min=float(nz.min()) if nz.numel() else None,
        nonzero_amplitude_max=float(nz.max()) if nz.numel() else None,
        empty_pixel_each_t=float((~spatial).float().mean()),
        empty_window_each_t=float((~halo).float().mean()),
        empty_pixel_all_t=float((~all_t).float().mean()),
        empty_window_all_t=float((~halo_all_t).float().mean()),
        active_scalar_valid_kernel_fanout=source_fanout)
    words = (g.to(torch.int32)*(1 << torch.arange(t, device=x.device))[:, None, None, None, None]).sum(0)
    words = words.permute(1, 0, 2, 3).reshape(c, -1)
    hist = torch.zeros(c, 1 << t, device=x.device, dtype=torch.int32)
    hist.scatter_add_(1, words, torch.ones_like(words, dtype=torch.int32))
    mode = hist.argmax(1)
    match = words.eq(mode[:, None])
    stats.update(temporal_zero_fraction=float(words.eq(0).float().mean()),
                 modal_word_channel_mean=float(match.float().mean()),
                 modal_words=mode.tolist(),
                 modal_nonzero_channels=int(mode.ne(0).sum()),
                 modal_H8_same_pixel=float(match.reshape(c//8, 8, -1).all(1).float().mean()),
                 modal_P4_H8=float(match.reshape(c//8, 8, b*h, w//4, 4)
                                  .all(1).all(-1).float().mean()))
    # Direct C16/P4 source word occupancy, before im2col or any W mask.
    for width in (8, 16):
        live = g.reshape(t, b, c//width, width, h, w//4, 4).any(3).any(-1)
        stats[f'empty_P4_C{width}_each_t'] = float((~live).float().mean())
        stats[f'empty_P4_C{width}_all_t'] = float((~live.any(0)).float().mean())
    return stats


def apply_axis(modules, original, axis):
    for name, weight in original.items():
        modules[name].weight.copy_(weight)
    if axis == 'parent':
        return dict(kind='saved mother student; no patch modification')
    block, mode = axis.split('_', 1)
    prefix = RES+block[1:]+'.'
    c1, c2 = modules[prefix+'conv1.0'], modules[prefix+'conv2.0']
    w1, w2 = c1.weight, c2.weight
    if mode == 'drop_branch':
        # Real norm2(0), generally beta, and identity still consumed. This is
        # not a hard zero of the residual block or of the BN/shortcut output.
        w2.zero_()
        kept = 0
    elif mode == 'hidden50':
        score = w1.square().sum((1, 2, 3))*w2.square().sum((0, 2, 3))
        chosen = score.topk(len(score)//2).indices
        mask = torch.zeros_like(score)
        mask[chosen] = 1
        w1.mul_(mask[:, None, None, None])
        w2.mul_(mask[None, :, None, None])
        kept = int(mask.sum())
    elif mode == 'row2of4':
        chunks = w1.reshape(w1.shape[0], -1, 4)
        mask = torch.zeros_like(chunks)
        mask.scatter_(2, chunks.abs().topk(2, dim=2).indices, 1)
        w1.mul_(mask.reshape_as(w1))
        kept = w1.shape[0]
    elif mode == 'group16':
        flat = w1.reshape(w1.shape[0]//8, 8, -1, 16)
        score = flat.square().sum((1, 3))
        mask = torch.zeros_like(score)
        mask.scatter_(1, score.topk(score.shape[1]//2, dim=1).indices, 1)
        w1.mul_(mask[:, None, :, None].expand_as(flat).reshape_as(w1))
        kept = w1.shape[0]
    else:
        raise ValueError(mode)
    return dict(target=prefix, kind='zero-training FP-weight sensitivity ablation',
                retained_hidden=kept, w1_nonzero=int(w1.ne(0).sum()),
                w2_nonzero=int(w2.ne(0).sum()), w1_elements=w1.numel(), w2_elements=w2.numel(),
                consumer='real dynamic norm1/norm2, full PSN sn1/sn2, real shortcut')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--axes', nargs='+', default=AXES)
    p.add_argument('--limit', type=int, default=10)
    args = p.parse_args()
    system = probe.load_system(args)
    from run_bn_probe import input_frame, read_names
    from evaluate_stage2_deployment import CoarseReady, summarize
    from spikingjelly.activation_based import functional
    model, modules, _, current, sources, _, _, _ = system
    probe.install_sources(system, sources)
    current['count_codes'] = False
    out = args.root/'algorithm/patch_probe'
    out.mkdir(parents=True, exist_ok=True)
    names = (read_names(args.data, 'valid') if args.limit == 825 else
             json.loads((args.root/'algorithm/samples.json').read_text())['valid'][:args.limit])
    original = {RES+f'{b}.conv{j}.0': modules[RES+f'{b}.conv{j}.0'].weight.detach().clone()
                for b in (0, 1) for j in (1, 2)}
    metadata = {name: dict(type=type(m).__name__, parameters={k:list(v.shape) for k,v in m.named_parameters(recurse=False)})
                for name,m in modules.items() if name.startswith(RES)}
    probe.save_json(out/'modules.json', metadata)
    summaries, support = {}, []
    capture_enabled = False
    current_name = ''
    hooks = []
    for name in original:
        def capture(m, inputs, n=name):
            if capture_enabled:
                support.append(dict(file=current_name, module=n, **support_stats(inputs[0])))
        hooks.append(modules[name].register_forward_pre_hook(capture))
    probe.save_json(out/'run.json', dict(frames=names, axes=args.axes,
        parent='saved direct_code_integer/coarse-head student; no previous s2b3 pruning applied',
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
        training='none; placement sensitivity only',
        claim='AEE/support census, not RTL or cycle result'))
    with torch.no_grad():
        for axis in args.axes:
            info = apply_axis(modules, original, axis)
            rows, started = [], time.monotonic()
            capture_enabled = axis == 'parent' and args.limit <= 10
            for i, name in enumerate(names):
                current_name = name
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480,640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0,2,3,1)[mask]-label.permute(0,2,3,1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append(dict(file=name, valid_pixels=pixels, aee_sum=total, AEE=total/pixels))
                if (i+1)%50 == 0:
                    print('PROGRESS', axis, i+1, summarize(rows, False), flush=True)
                del x, label, mask, pred, error
            result = dict(**summarize(rows, args.limit == 825), **info,
                          wall_seconds=time.monotonic()-started)
            summaries[axis] = result
            suffix = 'valid825' if args.limit == 825 else 'valid'+str(args.limit)
            probe.save_json(out/(axis+'_'+suffix+'_frames.json'), rows)
            probe.save_json(out/(suffix+'_summary.json'), summaries)
            if capture_enabled:
                probe.save_json(out/'support_valid10.json', support)
            print('RESULT', axis, json.dumps(result), flush=True)
    for handle in hooks:
        handle.remove()


if __name__ == '__main__':
    main()
