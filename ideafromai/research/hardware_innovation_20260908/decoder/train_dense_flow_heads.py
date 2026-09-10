"""Small FP32 flow-only decoder controls; run explicitly on the free GPU.

The native ep34 teacher and the prefix through decoder3.sn stay frozen. Both
students consume its actual theta*g values, with all ten time steps retained.
There is no lifting, calibrated BN, integer S0, or 96-channel fine-resolution
feature tensor here. The 32/10 experiment is an adaptation probe, not valid825.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'algorithm'))
from run_bn_probe import (build_model, input_frame, read_names,
                          representative_train, save_json)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = 'sttmultires_unet.decoders.3.sn'
COARSE = 'sttmultires_unet.preds.2'
SOURCE_SHAPE = (10, 1, 194, 120, 160)


class SourceReady(Exception):
    """Stop before the old last deconvolution; the source hook has completed."""


class Capture:
    def __init__(self, model):
        modules = dict(model.named_modules())
        self.neuron = modules[SOURCE + '.spiking_neuron']
        self.source = self.coarse = None
        self.stop = False
        self.handles = [modules[SOURCE].register_forward_hook(self.source_hook),
                        modules[COARSE].register_forward_hook(self.coarse_hook)]

    def source_hook(self, module, inputs, output):
        self.source = output.detach()
        if self.stop:
            raise SourceReady()

    def coarse_hook(self, module, inputs, output):
        self.coarse = output.detach()

    def clear(self):
        self.source = self.coarse = None

    def close(self):
        for handle in self.handles:
            handle.remove()


class DenseFlowHead(nn.Module):
    def __init__(self, spatial, mean_flow):
        super().__init__()
        self.readout = nn.Conv2d(1940, 16, 1)
        self.spatial = nn.Conv2d(16, 16, 3, padding=1) if spatial else None
        self.output = nn.Conv2d(16, 32, 1)
        # PixelShuffle orders the sixteen subpixels of u before those of v.
        # Only training-teacher values enter this initialization.
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.as_tensor(mean_flow).repeat_interleave(16))

    def forward(self, source_tc):
        x = F.leaky_relu(self.readout(source_tc), negative_slope=0.125)
        if self.spatial is not None:
            x = F.leaky_relu(self.spatial(x), negative_slope=0.125)
        return F.pixel_shuffle(self.output(x), 4)


def source_tc(source):
    t, b, c, h, w = source.shape
    return source.permute(1, 0, 2, 3, 4).reshape(b, t*c, h, w)


def metric(name, pred, label, mask):
    # Index before arithmetic so invalid GT values cannot affect the result.
    error = torch.linalg.vector_norm(
        pred.permute(0, 2, 3, 1)[mask] - label.permute(0, 2, 3, 1)[mask], dim=1)
    count, total = error.numel(), float(error.double().sum())
    return {'file': name, 'valid_pixels': count, 'aee_sum': total, 'AEE': total/count}


def summarize(rows):
    return {'frames': len(rows),
            'AEE_frame_mean': float(np.mean([r['AEE'] for r in rows])),
            'AEE_pixel_mean': sum(r['aee_sum'] for r in rows) /
                              sum(r['valid_pixels'] for r in rows)}


def cache_path(args, name):
    return args.output / 'cache' / (Path(name).stem + '.npz')


def load_batch(args, names):
    sources, thetas, targets, labels, masks = [], [], [], [], []
    for name in names:
        with np.load(cache_path(args, name)) as d:
            bits = np.unpackbits(d['source_bits'], count=int(np.prod(SOURCE_SHAPE)),
                                 bitorder='little').reshape(1940, 120, 160)
            sources.append(bits)
            thetas.append(float(d['source_theta']))
            targets.append(d['teacher_flow'])
        labels.append(np.load(args.data / 'gt_tensors' / name))
        masks.append(np.load(args.data / 'mask_tensors' / name).reshape(480, 640))
    source = torch.from_numpy(np.stack(sources)).cuda().float()
    source.mul_(torch.tensor(thetas, device='cuda', dtype=torch.float32)[:, None, None, None])
    return (source, torch.from_numpy(np.stack(targets)).cuda().float(),
            torch.from_numpy(np.stack(labels)).cuda().float(),
            torch.from_numpy(np.stack(masks)).cuda().bool())


def cache_teacher(args, model, capture, train, valid, reset_net):
    rows = {'teacher_native': [], 'coarse_native': [], 'coarse_bilinear': []}
    frames, teacher_mean = [], np.zeros(2, dtype=np.float64)
    (args.output / 'cache').mkdir(parents=True, exist_ok=True)
    capture.stop = False
    with torch.no_grad():
        for name in train + valid:
            reset_net(model)
            capture.clear()
            x, label, mask = input_frame(args.data, name)
            flows = model(x)['flow']
            source = capture.source
            assert tuple(source.shape) == SOURCE_SHAPE
            assert capture.neuron.thresh.numel() == 1
            theta = capture.neuron.thresh.detach().float().reshape(())
            bits = source != 0
            residual = float((source - bits * theta).abs().max())
            assert residual == 0, 'The source is not exactly the captured scalar theta*g.'
            teacher = flows[-1].detach().float()
            is_train = name in train
            if is_train:
                teacher_mean += teacher.double().mean((0, 2, 3)).cpu().numpy()
            else:
                rows['teacher_native'].append(metric(name, teacher, label, mask))
                rows['coarse_native'].append(metric(name, flows[-2], label, mask))
                # Native flow values are already in full-image pixel units.
                # The original model uses nearest interpolation without *4.
                coarse = capture.coarse.sum(0)
                bilinear = F.interpolate(coarse, size=(480, 640), mode='bilinear', align_corners=False)
                rows['coarse_bilinear'].append(metric(name, bilinear, label, mask))
            nz = int(bits.sum())
            np.savez_compressed(cache_path(args, name),
                source_bits=np.packbits(bits.cpu().numpy().reshape(-1), bitorder='little'),
                source_shape=np.array(SOURCE_SHAPE, dtype=np.int32),
                source_theta=np.float32(theta.item()),
                teacher_flow=teacher[0].cpu().numpy())
            frames.append({'file': name, 'split': 'train' if is_train else 'valid',
                           'source_theta': float(theta), 'source_nonzero': nz,
                           'theta_g_max_residual': residual,
                           'active_readout_coefficient_terms': nz*16})
            print('CACHE', len(frames), name, 'nz', nz, flush=True)
            capture.clear()
            del source, bits, teacher, flows, x, label, mask
    save_json(args.output / 'cache_frames.json', frames)
    save_json(args.output / 'zero_train_frames.json', rows)
    summary = {mode: summarize(values) for mode, values in rows.items()}
    save_json(args.output / 'summary.json', summary)
    return teacher_mean/len(train), summary


def train_head(args, head, train, batches, mode):
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0)
    logs = []
    for step, indices in enumerate(batches, 1):
        names = [train[i] for i in indices]
        source, teacher, label, mask = load_batch(args, names)
        optimizer.zero_grad(set_to_none=True)
        pred = head(source)
        distillation = (pred - teacher).abs().mean()
        delta = pred.permute(0, 2, 3, 1)[mask] - label.permute(0, 2, 3, 1)[mask]
        supervised = (delta.square().sum(1) + 1e-6).sqrt().mean()
        loss = distillation + supervised
        loss.backward()
        optimizer.step()
        row = {'step': step, 'loss': float(loss.detach()),
               'teacher_L1': float(distillation.detach()),
               'valid_GT_smooth_AEE': float(supervised.detach()), 'files': names}
        logs.append(row)
        if step == 1 or step % 8 == 0:
            print('TRAIN', mode, json.dumps(row), flush=True)
        del source, teacher, label, mask, pred, delta, loss, distillation, supervised
    save_json(args.output / (mode + '_training.json'), logs)
    torch.save({'state_dict': head.state_dict(), 'mode': mode, 'updates': args.updates,
                'source_module': SOURCE, 'source_shape': SOURCE_SHAPE},
               args.output / (mode + '.pt'))


def evaluate_head(args, model, capture, head, valid, reset_net, mode):
    rows = []
    head.eval()
    capture.stop = True
    with torch.no_grad():
        for name in valid:
            reset_net(model)
            capture.clear()
            x, label, mask = input_frame(args.data, name)
            try:
                model(x)
            except SourceReady:
                pass
            else:
                raise RuntimeError('The intended last-decoder early stop was not reached.')
            pred = head(source_tc(capture.source))
            row = metric(name, pred, label, mask)
            with np.load(cache_path(args, name)) as d:
                teacher = torch.from_numpy(d['teacher_flow']).cuda()[None]
            row['teacher_L1'] = float((pred-teacher).abs().mean())
            rows.append(row)
            print('VALID', mode, json.dumps(row), flush=True)
            capture.clear()
            del x, label, mask, pred, teacher
    save_json(args.output / (mode + '_frames.json'), rows)
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'config', 'checkpoint', 'data', 'samples', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--train-frames', type=int, default=32, choices=range(1, 33))
    parser.add_argument('--max-valid', type=int, default=10, choices=range(1, 11))
    parser.add_argument('--updates', type=int, default=64, choices=(32, 64))
    parser.add_argument('--batch-size', type=int, default=4, choices=(1, 2, 4))
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = representative_train(read_names(args.data, 'train'), args.train_frames)
    official_valid = set(read_names(args.data, 'valid'))
    valid = json.loads(args.samples.read_text())['valid'][:args.max_valid]
    assert len(set(train)) == len(train) and not set(train) & official_valid
    assert set(valid) <= official_valid
    model, cfg, installed, attention = build_model(args)
    model.requires_grad_(False)
    model.eval()
    from spikingjelly.activation_based import functional
    capture = Capture(model)
    save_json(args.output / 'run.json', {
        'kind': 'new_FP32_dense_flow_only_heads_32train_10valid_probe',
        'checkpoint': str(args.checkpoint), 'config': str(args.config),
        'torch': torch.__version__, 'python': sys.version.split()[0],
        'train': train, 'valid': valid, 'updates_per_head': args.updates,
        'batch_size': args.batch_size, 'lr': args.lr,
        'teacher': 'native ep34; all BN use current-frame statistics; no integer S0',
        'installed_atlif': len(installed), 'attention_blocks': len(attention),
        'motion_alpha': cfg['bsa_attention']['binary_motion_xor_alpha'],
        'hardware_quant_enabled': cfg['bsa_attention']['hardware_quant_enabled'],
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'source_module': SOURCE, 'source_shape': SOURCE_SHAPE,
        'flow_units': 'full-image pixels; neither native nor student upsampling multiplies vectors',
        'loss': 'teacher full-flow mean L1 + valid-GT mean sqrt(EPE_squared+1e-6)',
        'selection': 'both fixed heads at the final update; no validation-based checkpoint selection',
        'initialization': 'seed 0 readout; zero output weights; training-teacher mean output bias',
        'limits': 'Exploratory 10 frames; limited-update failure does not disprove decoder adaptation.'})
    mean_flow, summaries = cache_teacher(args, model, capture, train, valid, functional.reset_net)
    rng = np.random.default_rng(0)
    order = []
    while len(order) < args.updates*args.batch_size:
        order.extend(rng.permutation(len(train)).tolist())
    batches = [order[i*args.batch_size:(i+1)*args.batch_size] for i in range(args.updates)]
    costs = {}
    for mode, spatial in (('point', False), ('spatial3x3', True)):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        head = DenseFlowHead(spatial, mean_flow).cuda().float()
        train_head(args, head, train, batches, mode)
        summaries[mode] = evaluate_head(args, model, capture, head, valid, functional.reset_net, mode)
        summaries[mode]['delta_AEE_frame_mean_vs_teacher'] = (
            summaries[mode]['AEE_frame_mean'] - summaries['teacher_native']['AEE_frame_mean'])
        costs[mode] = {
            'trainable_parameters': sum(p.numel() for p in head.parameters()),
            'dense_readout_coefficient_terms': 120*160*1940*16,
            'dense_optional_spatial_coefficient_terms': 120*160*16*16*9 if spatial else 0,
            'dense_output_coefficient_terms': 120*160*16*32,
            'dense_total_coefficient_terms': 120*160*(1940*16 + 16*32 + (16*16*9 if spatial else 0)),
            'source_packed_bytes': int(np.prod(SOURCE_SHAPE))//8,
            'source_theta_FP32_bytes': 4,
            'source_dense_FP32_bytes': int(np.prod(SOURCE_SHAPE))*4,
            'latent16_FP32_bytes': 120*160*16*4,
            'final_flow_FP32_bytes': 480*640*2*4,
            'parameter_FP32_bytes': sum(p.numel() for p in head.parameters())*4,
            'scope': 'head arithmetic/data sizes only, not cycles, peak RF, measured hardware speedup, or PPA; active readout terms are in cache_frames.json'}
        save_json(args.output / 'summary.json', summaries)
        save_json(args.output / 'costs.json', costs)
        del head
    capture.close()
    print('COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
