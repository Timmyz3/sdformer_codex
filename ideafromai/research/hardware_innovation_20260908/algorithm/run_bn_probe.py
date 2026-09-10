"""Train-split BN calibration competitor and ep34 hardware captures.

Supports a shared validation subset or the complete official valid825 split.
Original model code/parameters are read from explicit paths; outputs stay here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import types

os.environ['SDFORMER_USE_MLFLOW'] = '0'
os.environ['SDFORMER_MLFLOW_MODEL_LOGGING'] = '0'
sys.modules.setdefault('mlflow', types.ModuleType('mlflow'))

import numpy as np
import torch
import torchvision  # import before CuPy
import yaml

PREFIX = 'sttmultires_unet.encoders.swin3d.'
MLPS = [PREFIX + f'layers.{s}.swin_blocks.{b}.mlp.'
        for s, count in enumerate((2, 2, 6, 2)) for b in range(count)]
SHALLOW = [p+'bn1.norm_layer' for p in MLPS[:2]]
ALL_FC1 = [p+'bn1.norm_layer' for p in MLPS]
DECODER = 'sttmultires_unet.decoders.3.norm_layer.norm_layer'
CAP_MLPS = [PREFIX+f'layers.{s}.swin_blocks.0.mlp.' for s in (0, 3)]
CAP_PSN = {
    'head': PREFIX+'patch_embed.head.sn.spiking_neuron',
    'sn_k': PREFIX+'layers.0.swin_blocks.0.attn.sn_k.spiking_neuron',
}


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def read_names(data, split):
    with (data/'sequence_lists'/f'{split}_split_seq.csv').open() as f:
        return [row[0] for row in csv.reader(f) if row]


def representative_train(names, n):
    """Evenly distribute calibration across training sequences and time."""
    groups = {}
    for name in names:
        groups.setdefault(name.rsplit('_', 1)[0], []).append(name)
    sequences = sorted(groups)
    out = []
    rounds = (n+len(sequences)-1)//len(sequences)
    for r in range(rounds):
        for seq in sequences:
            xs = groups[seq]
            j = min(len(xs)-1, int((r+0.5)*len(xs)/rounds))
            out.append(xs[j])
    return out[:n]


def input_frame(data, name, targets=True):
    seq = name.rsplit('_', 1)[0]
    x = torch.from_numpy(np.load(data/'event_tensors'/'10bins'/'left'/seq/name)).cuda().float()[None]
    pos, neg = x.relu(), (-x).relu()
    x = torch.stack((pos, neg), dim=2)
    nz = x != 0
    if nz.any():
        lo, hi = x[nz].min(), x[nz].max()
        if hi != lo:
            x[nz] = (x[nz]-lo)/(hi-lo)
    if not targets:
        return x, None, None
    label = torch.from_numpy(np.load(data/'gt_tensors'/name)).cuda().float()[None]
    mask = torch.from_numpy(np.load(data/'mask_tensors'/name)).cuda().bool().reshape(1, 480, 640)
    return x, label, mask


def set_bn_mode(model, calibrated=None, chosen=()):
    chosen = set(chosen)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        module.eval()
        if name in chosen:
            stats = calibrated[name]
            module.track_running_stats = True
            module.running_mean = stats['mean'].to(module.weight.device, module.weight.dtype)
            module.running_var = stats['var'].to(module.weight.device, module.weight.dtype)
            module.num_batches_tracked = torch.tensor(0, device=module.weight.device)
        else:
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None


class Calibration:
    def __init__(self, model, targets):
        self.data = {}
        self.handles = []
        modules = dict(model.named_modules())
        for name in targets:
            self.handles.append(modules[name].register_forward_pre_hook(self.make_hook(name)))

    def make_hook(self, name):
        def hook(module, args):
            x = args[0].detach().float()
            axis = 2 if x.ndim == 5 else 1
            dims = tuple(i for i in range(x.ndim) if i != axis)
            var, mean = torch.var_mean(x, dim=dims, unbiased=False)
            mean, var = mean.double().cpu(), var.double().cpu()
            count = x.numel()//x.shape[axis]
            if name not in self.data:
                self.data[name] = {'count': 0, 'sum': torch.zeros_like(mean),
                                   'sum2': torch.zeros_like(mean), 'frames': 0}
            d = self.data[name]
            d['count'] += count
            d['sum'] += count*mean
            d['sum2'] += count*(var+mean.square())
            d['frames'] += 1
        return hook

    def finish(self):
        for h in self.handles:
            h.remove()
        result = {}
        for name, d in self.data.items():
            mean = d['sum']/d['count']
            var = (d['sum2']/d['count']-mean.square()).clamp_min(0)
            result[name] = {'mean': mean.float(), 'var': var.float(),
                            'count': d['count'], 'frames': d['frames']}
        return result


class HardwareCapture:
    def __init__(self, model, output):
        self.modules = dict(model.named_modules())
        self.output = Path(output)
        self.sample = None
        self.raw_y = {}
        self.handles = []
        for label, name in CAP_PSN.items():
            self.handles.append(self.modules[name].register_forward_hook(self.psn_hook(label, name)))
        for p in CAP_MLPS:
            self.handles.append(self.modules[p+'fc1'].register_forward_hook(self.fc_hook(p)))
            self.handles.append(self.modules[p+'sn2.spiking_neuron'].register_forward_hook(self.sn_hook(p)))

    def fc_hook(self, p):
        def hook(module, args, output):
            if self.sample is not None:
                self.raw_y[p] = output.detach().reshape(output.shape[0], -1, output.shape[-1])
        return hook

    def psn_hook(self, label, name):
        def hook(module, args, output):
            if self.sample is None:
                return
            x = args[0].detach().flatten(1)
            blocks = x.shape[1]//32
            starts = torch.linspace(0, blocks-1, min(128, blocks), device=x.device).round().long()*32
            indices = (starts[:, None]+torch.arange(32, device=x.device)[None]).flatten()
            path = self.output/self.sample/f'psn_{label}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x[:, indices].float().cpu().numpy(),
                output=output.detach().flatten(1)[:, indices].float().cpu().numpy(),
                indices=indices.cpu().numpy(), original_shape=np.array(args[0].shape),
                module=np.array(name), center_mode=np.array(module.center_mode),
                A=module.weight.detach().float().cpu().numpy(),
                bias=module.bias.detach().float().cpu().numpy(),
                center=module.center.detach().float().cpu().numpy(),
                theta=module.thresh.detach().float().cpu().numpy())
        return hook

    def sn_hook(self, p):
        def hook(module, args, output):
            if self.sample is None:
                return
            y = self.raw_y.pop(p)
            actual = output.detach().reshape_as(y).ne(0)
            bn = self.modules[p+'bn1.norm_layer']
            gamma, beta = bn.weight.detach(), bn.bias.detach()
            var, mean = torch.var_mean(y, dim=(0, 1), unbiased=False)
            stats = {'module': p, 'sample': self.sample, 'shape': list(y.shape),
                     'eps': bn.eps, 'theta': float(module.thresh),
                     'center_mode': module.center_mode,
                     'statistics': 'torch.var_mean on actual FC1 FP32 output; not cuDNN internal saved statistics',
                     'source_dtype': str(y.dtype), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'comparisons': {}}
            centered = module.center.detach() if module.center_mode != 'zero' else torch.zeros_like(module.bias)
            for dtype in (torch.float32, torch.float64):
                mismatches = 0
                min_margin = float('inf')
                for h in range(0, y.shape[2], 32):
                    yy = y[:, :, h:h+32].to(dtype)
                    v, m = torch.var_mean(yy, dim=(0, 1), unbiased=False)
                    a = module.weight.detach().to(dtype)
                    b = module.bias.detach().to(dtype).reshape(-1, 1)
                    c = centered.to(dtype).reshape(-1, 1)
                    g, be = gamma[h:h+32].to(dtype), beta[h:h+32].to(dtype)
                    row_sum = a.sum(1, keepdim=True)
                    u = (a @ yy.reshape(yy.shape[0], -1)).reshape_as(yy)
                    tau = m*row_sum+(v+bn.eps).sqrt()/g*(module.thresh.detach().to(dtype)+c-b-be*row_sum)
                    margin = (u-tau[:, None, :])*g.sign()[None, None, :]
                    pred = margin >= 0
                    mismatches += int((pred != actual[:, :, h:h+32]).sum())
                    min_margin = min(min_margin, float(margin.abs().min()))
                stats['comparisons'][str(dtype)] = {'support_mismatch': mismatches,
                                                   'min_abs_transformed_margin': min_margin}
            path = self.output/self.sample
            path.mkdir(parents=True, exist_ok=True)
            stage = p.split('.layers.')[1].split('.')[0]
            np.savez_compressed(path/f'bn_stage{stage}.npz', mean=mean.cpu().numpy(),
                var=var.cpu().numpy(), output_bits=np.packbits(actual.cpu().numpy(), axis=-1, bitorder='little'),
                output_shape=np.array(actual.shape), A=module.weight.detach().cpu().numpy(),
                bias=module.bias.detach().cpu().numpy(), center=centered.cpu().numpy(),
                theta=module.thresh.detach().cpu().numpy(), gamma=gamma.cpu().numpy(), beta=beta.cpu().numpy())
            save_json(path/f'bn_stage{stage}.json', stats)
            print('CAPTURE', self.sample, 'stage'+stage, stats['comparisons'], flush=True)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def build_model(args):
    sys.path[:0] = [str(args.code_root/'neuron_experiments/H9_bipolar_self_attention/overlay'),
                   str(args.code_root/'third_party/SDformerFlow')]
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.atlif_ternary_psn import install_atlif_ternary_psn
    from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat
    from spikingjelly.activation_based import functional

    cfg = YAMLParser.combine_entries(yaml.safe_load(args.config.read_text()))
    cfg['data']['path'] = str(args.data)
    cfg['swin_transformer']['input_size'] = [480, 640]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.allow_tf32 = bool(cfg['runtime']['allow_tf32'])
    torch.backends.cudnn.benchmark = bool(cfg['runtime']['cudnn_benchmark'])
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(cfg['model']), copy.deepcopy(cfg['swin_transformer'])).cuda()
    model.init_weights()
    installed = install_atlif_ternary_psn(model, cfg['atlif_ternary_psn'])
    attention = install_shiftmax_attention(model, cfg['bsa_attention'])
    register_shiftmax_pickle_compat()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    functional.set_step_mode(model, 'm')
    model.eval()
    set_bn_mode(model)
    print('MODEL_LOADED', len(installed), len(attention), flush=True)
    return model, cfg, installed, attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-frames', type=int, default=32)
    parser.add_argument('--max-valid', type=int, default=10)
    parser.add_argument('--full-valid', action='store_true')
    parser.add_argument('--load-calibration', type=Path)
    parser.add_argument('--scopes', nargs='+', default=['s0_fc1', 'all_fc1', 'decoder3'])
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, cfg, installed, attention = build_model(args)
    from spikingjelly.activation_based import functional
    sample_info = json.loads(args.samples.read_text())
    official_names = read_names(args.data, 'valid')
    valid = official_names if args.full_valid else sample_info['valid'][:args.max_valid]
    official_valid = set(official_names)
    assert set(valid).issubset(official_valid)
    train = representative_train(read_names(args.data, 'train'), args.calibration_frames)
    assert not set(train).intersection(official_valid)
    modules = dict(model.named_modules())
    targets = ALL_FC1+[DECODER]
    assert all(t in modules for t in targets)
    save_json(args.output/'run.json', {'checkpoint': str(args.checkpoint), 'config': str(args.config),
        'torch': torch.__version__, 'python': sys.version.split()[0], 'gpu': torch.cuda.get_device_name(),
        'installed_atlif': len(installed), 'attention_blocks': len(attention), 'valid': valid,
        'calibration_train': train, 'full_resolution': [480, 640], 'window': cfg['swin_transformer']['window_size'],
        'motion_alpha': cfg['bsa_attention']['binary_motion_xor_alpha'],
        'hardware_quant_enabled': cfg['bsa_attention']['hardware_quant_enabled'],
        'scopes': args.scopes, 'kind': 'train_calibration_official_valid825' if args.full_valid else 'exploratory_train_calibration_same_validation_subset',
        'loaded_calibration': str(args.load_calibration) if args.load_calibration else None,
        'tf32': torch.backends.cuda.matmul.allow_tf32})
    capture = HardwareCapture(model, args.output/'capture') if args.capture else None
    capture_names = set(sample_info.get('hardware_capture', valid[:2]))
    rows = []

    # Hardware traces may use a different cohort from the official validation
    # split. Capture those without counting them in the accuracy comparison.
    if capture:
        with torch.no_grad():
            for name in sorted(capture_names-set(valid)):
                capture.sample = name[:-4]
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name, targets=False)
                model(x)
                print('HARDWARE_CAPTURE_ONLY', name, flush=True)
                del x, label, mask
        capture.sample = None

    def evaluate(tag):
        summary = []
        with torch.no_grad():
            for i, name in enumerate(valid):
                if capture:
                    capture.sample = name[:-4] if tag == 'dynamic' and name in capture_names else None
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                start = time.monotonic()
                pred = model(x)['flow'][-1]
                error = (pred-label).square().sum(1).sqrt()
                count = int(mask.sum())
                numerator = float(error[mask].sum())
                row = {'mode': tag, 'file': name, 'valid_pixels': count,
                       'aee_sum': numerator, 'AEE': numerator/count,
                       'elapsed_s_including_hooks': time.monotonic()-start}
                rows.append(row)
                summary.append(row)
                if (i+1) % 25 == 0 or i+1 == len(valid):
                    save_json(args.output/'frames.json', rows)
                print('FRAME', json.dumps(row), flush=True)
                del pred, x, label, mask, error
        return {'frames': len(summary),
                'AEE_frame_mean': float(np.mean([r['AEE'] for r in summary])),
                'AEE_pixel_mean': sum(r['aee_sum'] for r in summary)/sum(r['valid_pixels'] for r in summary)}

    results = {'dynamic': evaluate('dynamic')}
    if capture:
        capture.close()
        capture = None
    save_json(args.output/'summary.json', results)
    if args.load_calibration:
        stats = torch.load(args.load_calibration, map_location='cpu', weights_only=False)
        print('CALIBRATION_LOADED', str(args.load_calibration), flush=True)
    else:
        print('CALIBRATION_START', len(train), flush=True)
        collector = Calibration(model, targets)
        with torch.no_grad():
            for i, name in enumerate(train):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name, targets=False)
                model(x)
                print('CALIBRATION', i+1, name, flush=True)
                del x, label, mask
        stats = collector.finish()
    torch.save(stats, args.output/'train_calibration.pt')
    scopes = {'s0_fc1': SHALLOW, 'all_fc1': ALL_FC1, 'decoder3': [DECODER]}
    for scope in args.scopes:
        set_bn_mode(model, stats, scopes[scope])
        results[scope] = evaluate(scope)
        results[scope]['delta_AEE_frame_mean'] = results[scope]['AEE_frame_mean']-results['dynamic']['AEE_frame_mean']
        save_json(args.output/'summary.json', results)
    print('COMPLETE', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
