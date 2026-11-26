#!/usr/bin/env python3
"""Execute the unmodified native MLP on CPU, with actual local tile omission.

Synthetic weights/inputs only: validates an execution contract, not optical-flow
accuracy, a trained checkpoint, GPU throughput, or cycle/energy savings.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import copy
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'SDformer/third_party/SDformerFlow'
sys.path[:0] = [str(HERE / 'deps'), str(SOURCE)]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp


def make_mlp(channels, dtype):
    torch.manual_seed(1701 + channels)
    model = MS_Spiking_Mlp(
        channels, hidden_features=4 * channels, norm_layer='BN', drop=0.0,
        num_steps=10, spike_norm='BN', neuron_type='psn', v_th=0.1,
        v_reset=None, surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True)
    functional.set_step_mode(model, 'm')
    with torch.no_grad():
        # Avoid a trivial all-zero synthetic network from PSN's default -1 bias.
        for sn in (model.sn1.spiking_neuron, model.sn2.spiking_neuron):
            sn.weight.normal_(0., 0.3)
            sn.bias.fill_(0.02)
        for bn in (model.bn1.norm_layer, model.bn2.norm_layer):
            bn.running_mean.uniform_(-0.1, 0.1)
            bn.running_var.uniform_(0.7, 1.3)
            bn.weight.uniform_(0.8, 1.2)
            bn.bias.uniform_(-0.1, 0.1)
    return model.to(device='cpu', dtype=dtype).eval()


def masks_for(h, w, tile):
    ny, nx = (h + tile - 1) // tile, (w + tile - 1) // tile
    masks = {}
    masks['all_on'] = [[True for _ in range(nx)] for _ in range(ny)]
    masks['all_off'] = [[False for _ in range(nx)] for _ in range(ny)]
    masks['checkerboard'] = [[(y + x) % 2 == 0 for x in range(nx)] for y in range(ny)]
    masks['last_edge_only'] = [[y == ny - 1 and x == nx - 1 for x in range(nx)] for y in range(ny)]
    masks['first_tile_only'] = [[y == 0 and x == 0 for x in range(nx)] for y in range(ny)]
    return masks


def actual_local_forward(model, x, decisions, tile):
    """Identity for rejected tiles; the native MLP is never called for them."""
    out = x.clone()
    count = {'fc1_rows': 0, 'fc2_rows': 0, 'mlp_calls': 0}
    handles = []
    for name in ('fc1', 'fc2'):
        def hook(module, inputs, outputs, label=name):
            count[label + '_rows'] += inputs[0].numel() // inputs[0].shape[-1]
        handles.append(getattr(model, name).register_forward_hook(hook))
    try:
        for gy, row in enumerate(decisions):
            for gx, keep in enumerate(row):
                if not keep:
                    continue
                ys, xs = slice(gy * tile, min((gy + 1) * tile, x.shape[2])), slice(gx * tile, min((gx + 1) * tile, x.shape[3]))
                sub = x[:, :, ys, xs, :].contiguous()
                out[:, :, ys, xs, :] += model(sub)
                count['mlp_calls'] += 1
    finally:
        for handle in handles:
            handle.remove()
    return out, count


def expanded_mask(x, decisions, tile):
    out = torch.zeros((1, 1, x.shape[2], x.shape[3], 1), dtype=x.dtype)
    for gy, row in enumerate(decisions):
        for gx, keep in enumerate(row):
            if keep:
                out[:, :, gy * tile:min((gy + 1) * tile, x.shape[2]), gx * tile:min((gx + 1) * tile, x.shape[3]), :] = 1
    return out


def no_running(model):
    out = copy.deepcopy(model).eval()
    for module in out.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None
    return out


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    records = []
    with torch.inference_mode():
        geometries = [(18, 21, 96, 9), (30, 32, 96, 15), (9, 11, 384, 9),
                      (120, 160, 96, 15), (30, 40, 384, 15),
                      (120, 160, 96, 9), (30, 40, 384, 9)]
        for h, w, c, tile in geometries:
            # Actual full-resolution stage0/stage2 geometry uses FP32; the
            # smaller contracts above also check float64 arithmetic.
            dtypes = (torch.float32,) if (h, w) in [(120,160),(30,40)] else (torch.float64, torch.float32)
            for dtype in dtypes:
                model = make_mlp(c, dtype)
                torch.manual_seed(4400 + h + c)
                x = torch.randn(10, 1, h, w, c, dtype=dtype)
                full = model(x)
                assert full.abs().max().item() > 0.01
                dense_rows = 10 * h * w
                local_results = []
                for label, decisions in masks_for(h, w, tile).items():
                    mask = expanded_mask(x, decisions, tile)
                    reference = x + full * mask
                    actual, counts = actual_local_forward(model, x, decisions, tile)
                    error = (reference - actual).abs().max().item()
                    expected_rows = 10 * int(mask.sum().item())
                    assert counts['fc1_rows'] == counts['fc2_rows'] == expected_rows
                    tol = 1e-11 if dtype == torch.float64 else 2e-5
                    passed = torch.allclose(actual, reference, atol=tol, rtol=tol)
                    assert passed, (h, w, c, str(dtype), label, error)
                    local_results.append({'mask': label, 'max_abs_error': error,
                        'absolute_and_relative_tolerance': tol, 'passed': passed,
                        'dense_fc_rows_per_layer': dense_rows, 'actual_calls': counts,
                        'cancelled_fc_rows_per_layer': dense_rows - expected_rows})
                # A concrete negative control using the exact same native modules.
                current = no_running(model)
                decisions = masks_for(h, w, tile)['checkerboard']
                cmask = expanded_mask(x, decisions, tile)
                ref_current = x + current(x) * cmask
                local_current, _ = actual_local_forward(current, x, decisions, tile)
                current_error = (ref_current - local_current).abs().max().item()
                assert current_error > 1e-4
                # Confirm this installed SpikingJelly BN reduces T,B,H,W jointly.
                bnx = torch.randn(10, 1, 4*c, h, w, dtype=dtype)
                bn = current.bn1.norm_layer
                bnactual = bn(bnx)
                axes = (0, 1, 3, 4)
                mu = bnx.mean(axes, keepdim=True)
                var = bnx.var(axes, correction=0, keepdim=True)
                manual = (bnx - mu) / torch.sqrt(var + bn.eps)
                manual = manual * bn.weight.view(1,1,-1,1,1) + bn.bias.view(1,1,-1,1,1)
                bnerror = (manual - bnactual).abs().max().item()
                assert torch.allclose(manual, bnactual, atol=2e-5, rtol=2e-5)
                records.append({'shape_TBHWC': list(x.shape), 'tile': tile, 'dtype': str(dtype),
                    'frozen_bn_cases': local_results,
                    'current_bn_negative_control_max_abs_error': current_error,
                    'native_bn_reduction_axes': ['T','B','H','W'],
                    'native_bn_vs_manual_max_abs_error': bnerror})
                print('PASS',h,w,c,tile,str(dtype),'frozen masks=5; current-stat mismatch=',current_error,flush=True)
    sources = [SOURCE / 'models/STSwinNet_SNN'/f for f in (
        'Spiking_swin_transformer3D.py', 'Spiking_submodules.py', 'Spiking_modules.py')]
    result = {'scope': 'Native PyTorch/SpikingJelly MLP execution on synthetic CPU tensors; no checkpoint or AEE',
        'python': platform.python_version(), 'packages': {k:version(k) for k in ['torch','spikingjelly','timm','numpy']},
        'einops_probe_dependency': '0.8.1, installed only in this probe directory',
        'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'threads': 2, 'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(),
        'trained_checkpoint_used': False, 'old_source_modified': False,
        'not_cycle_or_energy_measurement': True, 'elapsed_seconds': time.monotonic()-start,
        'frozen_mask_case_count': sum(len(r['frozen_bn_cases']) for r in records), 'records': records}
    assert result['cuda_initialized'] is False
    (HERE/'native_mlp_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote native_mlp_results.json',flush=True)


if __name__ == '__main__':
    main()
