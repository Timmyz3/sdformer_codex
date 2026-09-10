"""Four matched local recovery controls: frozen-mask factors, FP vs DeepShift-Q V.

Independent small implementation of the DeepShift-Q mathematical quantizer:
CVPR Workshops 2021 (MAI), not CVPR main conference. Author modules_q.py sets
weight_bits=5 => sign plus exponent p in [-15,0]; clamp nonzero magnitudes to
[2^-15,1], round log2 magnitude deterministically, identity rounding STE.
The author repository exposes no license file/API license as checked on
2026-09-09, so no author modules are vendored or imported here.

Only the V WEIGHT quantizer is transferred. Z remains the existing FP32
continuous reference; fixed-point activation quantization and physical shift
execution are not claimed. Both FP and shift controls train U/V for 256 steps,
freeze the exact saved spatial masks/connectivity/A/b/theta/prefix/gamma, use
the same full+mixed recovery loss, and set request-loss lambda to zero.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FACTOR = HERE.parent
PATCH = FACTOR.parent
sys.path.insert(0, str(FACTOR))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_factors import load_data, batch, evaluate
from demand_completion import calibrate, completion, measure, PREFIX, GAMMA

PRIMARY = {
    'paper': 'https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.html',
    'module': 'https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/modules_q.py',
    'ste': 'https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/ste.py',
    'math': 'https://raw.githubusercontent.com/mostafaelhoushi/DeepShift/master/pytorch/deepshift/utils.py',
    'read_sections': 'LinearShiftQ constructor/forward, RoundPowerOf2/ClampAbsFunction, get_shift_and_sign/round_power_of_2',
    'license_check': 'GitHub root contents have no LICENSE; /license API returned 404. Mathematical reimplementation, no copied module.',
}


class Pow2STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # log(0)=-inf and 2**(-inf)=0 preserve exact zeros in the forward.
        exponent = torch.round(torch.log(weight.abs())/math.log(2.0))
        return weight.sign()*torch.pow(2.0, exponent)

    @staticmethod
    def backward(ctx, upstream):
        return upstream


class FrozenMaskFactor(nn.Module):
    def __init__(self, arrays, shift):
        super().__init__()
        self.u = nn.Parameter(torch.tensor(arrays['u'], dtype=torch.float32))
        self.v_shadow = nn.Parameter(torch.tensor(arrays['v'], dtype=torch.float32))
        self.register_buffer('connectivity', torch.from_numpy(arrays['connectivity'].copy()).bool())
        self.register_buffer('fixed_masks', torch.from_numpy(arrays['masks'].copy()).float())
        self.register_buffer('logits', torch.from_numpy(arrays['logits'].copy()).float())
        self.structure = str(arrays['structure'].item())
        self.keep = int(arrays['active_tiles'])
        self.latent_tile = int(arrays['latent_tile'])
        self.shift = bool(shift)

    def masks(self, training=False):
        return self.fixed_masks

    @torch.no_grad()
    def project(self):
        self.v_shadow.mul_(self.connectivity)
        if self.shift:
            value = self.v_shadow
            value.copy_(value.sign()*value.abs().clamp(2**-15, 1.0))

    @property
    def v(self):
        if self.shift:
            return Pow2STE.apply(self.v_shadow)*self.connectivity
        return self.v_shadow*self.connectivity

    def forward(self, source, regions):
        mask = self.fixed_masks[regions].repeat_interleave(self.latent_tile, dim=1)
        z = (source@self.u)*mask[:, None, None, :]
        return z@self.v

    @torch.no_grad()
    def export(self, saved, moments):
        arrays = {k: v.copy() for k, v in saved.items()}
        arrays.update(u=self.u.cpu().numpy().copy(), v=self.v.cpu().numpy().copy(),
                      v_shadow=self.v_shadow.cpu().numpy().copy(),
                      masks=self.fixed_masks.cpu().numpy().astype(bool),
                      completion_mean=moments['mean'].cpu().numpy(),
                      completion_covariance=moments['covariance'].cpu().numpy(),
                      v_quantizer=np.array('DeepShift-Q weight-only: p=-15..0, sign, zero preserved' if self.shift else 'FP32'),
                      v_weight_bits=np.array(5 if self.shift else 32),
                      recovery_masks_frozen=np.array(True))
        v = arrays['v']
        if self.shift:
            nz = v != 0
            shift = np.zeros(v.shape, np.int8)
            shift[nz] = np.rint(np.log2(np.abs(v[nz]))).astype(np.int8)
            arrays.update(v_shift=shift, v_sign=np.sign(v).astype(np.int8), v_nonzero=nz)
        return arrays


def storage(arrays, shift):
    conn = arrays['connectivity'].astype(bool)
    v = arrays['v']
    alive = np.repeat(arrays['masks'].any(0), int(arrays['latent_tile']))
    nz = v != 0
    used = nz & alive[:, None]
    coefficient_count = int(nz.sum())
    live_count = int(used.sum())
    if shift:
        p = np.rint(np.log2(np.abs(v[nz]))).astype(int)
        if np.any((p < -15) | (p > 0)) or not np.array_equal(np.exp2(p), np.abs(v[nz])):
            raise ValueError('Exported V is outside the exact declared shift domain.')
        histogram = {str(e): int(np.count_nonzero(p == e)) for e in range(-15, 1)}
    else:
        histogram = None
    bits = 5 if shift else 32
    return dict(U_FP32_dense_bytes=int(arrays['u'].nbytes),
                U_FP32_after_global_inactive_latent_trim_bytes=int(arrays['u'][:, alive].nbytes),
                V_dense_slots=int(v.size), V_nonzero_coefficients=coefficient_count,
                V_global_live_nonzero_coefficients=live_count,
                V_payload_bytes_fixed_structure=math.ceil(coefficient_count*bits/8),
                V_payload_after_global_trim_bytes=math.ceil(live_count*bits/8),
                V_explicit_dense_nonzero_bitmap_bytes=math.ceil(v.size/8),
                mask_bits=int(arrays['masks'].size), mask_raw_bytes=math.ceil(arrays['masks'].size/8),
                forbidden_structure_nonzeros=int(np.count_nonzero(v[~conn])),
                allowed_exact_zeros=int(np.count_nonzero(conn & ~nz)),
                exponent_histogram=histogram,
                payload_note='5 bits encode 32 NONZERO values only; fixed connectivity omits structural zeros. Generic dense zero needs a separate marker/bitmap. U remains FP32; no whole-factor 5-bit claim.')


@torch.no_grad()
def export_check(model, arrays, item):
    mask = torch.from_numpy(np.repeat(arrays['masks'][item['regions'].numpy()], model.latent_tile, axis=1)).float()
    u, v = torch.from_numpy(arrays['u']), torch.from_numpy(arrays['v'])
    exported = ((item['source']@u)*mask[:, None, None])@v
    actual = model(item['source'], item['regions'])
    result = dict(exported_dense_factor_max_abs=float((exported-actual).abs().max()),
                  masks_equal_saved=True)
    if model.shift:
        z = ((item['source']@u)*mask[:, None, None])[0]
        p = torch.from_numpy(arrays['v_shift']).int()
        sign = torch.from_numpy(arrays['v_sign']).float()
        product = z[..., :, None]*v
        shifted = torch.ldexp(z[..., :, None], p)*sign
        result.update(real_Z_shift_product_mismatches=int((product != shifted).sum()),
                      real_Z_elements=int(z.numel()),
                      shift_note='Exact FP32 power-of-two product/ldexp check, not fixed-point Z or physical RTL.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PATCH/'joint_completion_20260909/full_capture4/capture')
    p.add_argument('--capture', type=Path, default=PATCH/'partial_completion/capture.pt')
    p.add_argument('--valid-source', type=Path, default=PATCH/'partial_completion/integer_valid10')
    p.add_argument('--operator', type=Path, default=PATCH/'partial_completion/shared_column_deployment_source.npz')
    p.add_argument('--temporal', default='common3', choices=['common3'])
    p.add_argument('--initial', type=Path, default=FACTOR/'demand_train16')
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(910)
    args.output.mkdir(parents=True, exist_ok=True)
    data, op, a, b, theta, yscale, mscale, rate, groups = load_data(args)
    constants = dict(a=a, b=b, theta=theta, bn_scale=torch.tensor(op['bn_scale']).float(),
                     bn_bias=torch.tensor(op['bn_bias']).float(), y_scale=yscale, margin_scale=mscale, rate=rate)
    source_theta = float(op['source_theta'])
    generator = torch.Generator().manual_seed(910)
    batches = torch.randint(len(data['train']['y']), (256, 8), generator=generator)
    result = dict(scope='ordinary DeepShift-Q V weight-only recovery; local train16/valid4, no AEE/GPU/RTL/PPA',
                  sources=PRIMARY, training=dict(steps=256, batch_native_P4=8, optimizer='Adam', lr=0.002,
                      seed=910, loss='normalized Y MSE + .125 balanced full BCE + .125 balanced mixed BCE',
                      request_lambda=0, trainable='U,V_shadow only',
                      frozen='exact saved masks/connectivity, A,b,theta_source/output,prefix,gamma',
                      moments='each student train16 at 0/64/128/192 and after256; detached',
                      schedule='matched local recovery, not a reproduction of author ImageNet training schedule'),
                  quantization=dict(weight_bits=5, exponents=list(range(-15,1)), signs=[-1,1],
                      zero='exact structural zeros are preserved, not raised to 2^-15',
                      clip='project nonzero abs(V_shadow) to [2^-15,1] before quantized forward',
                      rounding='round(log(abs(v))/log(2)), deterministic round-to-even',
                      backward='identity STE through rounding; input gradient uses quantized V',
                      activation='Z remains FP32 continuous; author activation Q16.16 is not enabled in this weight-only control'),
                  train=data['train']['files'], valid=data['valid']['files'], pairing={k:v['pairing'] for k,v in data.items()}, axes={})
    (args.output/'definition.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    started = time.monotonic()
    for structure in ('shared_compact', 'hybrid'):
        path = args.initial/(structure+'.npz')
        with np.load(path) as f:
            saved = {k:f[k].copy() for k in f.files}
        if not np.array_equal(saved['a'], a.numpy()) or not np.array_equal(saved['temporal_bias'], b.numpy()):
            raise ValueError('Saved student temporal parameters do not match the fixed teacher interface.')
        if list(saved['prefix']) != list(PREFIX) or float(saved['gamma']) != GAMMA:
            raise ValueError('Saved prefix/gamma differ from the common completion function.')
        for shift in (False, True):
            axis = structure+('_deepshift_q5' if shift else '_fp32')
            model = FrozenMaskFactor(saved, shift)
            model.project()
            moments = calibrate(model, data['train'], constants, 'cpu', source_theta)
            initial = evaluate(model, data['valid'], constants, 'cpu', source_theta)
            optimizer = torch.optim.Adam([model.u, model.v_shadow], lr=0.002)
            history = []
            for step, ids in enumerate(batches):
                if step % 64 == 0:
                    moments = calibrate(model, data['train'], constants, 'cpu', source_theta)
                model.train()
                item = batch(data['train'], ids, 'cpu', source_theta)
                state = completion(model, item, constants, moments, soft=True)
                y_loss = ((state['y']-item['y'])/yscale).square().mean()
                target = item['target'].ge(0)
                weights = torch.where(target, 0.5/rate, 0.5/(1-rate))
                temp = (0.25*mscale).clamp_min(.025)
                full_p = torch.sigmoid(state['full']/temp)
                pred_p = torch.sigmoid(state['predicted']/temp)
                mixed_p = state['accept_probability']*pred_p+(1-state['accept_probability'])*full_p
                full_loss = (F.binary_cross_entropy_with_logits(state['full']/temp, target.float(), reduction='none')*weights).mean()
                mixed_loss = (F.binary_cross_entropy(mixed_p.clamp(1e-6,1-1e-6), target.float(), reduction='none')*weights).mean()
                loss = y_loss+.125*full_loss+.125*mixed_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.u, model.v_shadow], 5.)
                optimizer.step()
                model.project()
                if step % 64 == 0 or step == 255:
                    row = dict(step=step+1, total=float(loss.detach()), normalized_y_mse=float(y_loss.detach()),
                               full_bce=float(full_loss.detach()), mixed_bce=float(mixed_loss.detach()))
                    history.append(row)
                    print(axis, json.dumps(row), flush=True)
            moments = calibrate(model, data['train'], constants, 'cpu', source_theta)
            train_full = evaluate(model, data['train'], constants, 'cpu', source_theta)
            valid_full = evaluate(model, data['valid'], constants, 'cpu', source_theta)
            valid_completion = measure(model, data['valid'], constants, moments, 'cpu', source_theta)
            arrays = model.export(saved, moments)
            if not np.array_equal(arrays['masks'], saved['masks']):
                raise ValueError('Frozen spatial masks changed.')
            if shift:
                valid_completion['V_operation_label'] = 'one signed power-of-two scaling plus accumulation per nonzero V term; continuous FP32 Z reference'
                for phase in ('prefix','tail','full'):
                    valid_completion[phase+'_v_shift_add_terms'] = valid_completion[phase+'_v_continuous_terms']
            else:
                valid_completion['V_operation_label'] = 'general continuous FP32 multiply-accumulate'
            check = export_check(model, arrays, batch(data['valid'], torch.arange(1), 'cpu', source_theta))
            np.savez_compressed(args.output/(axis+'.npz'), **arrays)
            result['axes'][axis] = dict(parent=str(path), initial_valid_full=initial, train_full=train_full,
                valid_full=valid_full, valid_completion=valid_completion, storage=storage(arrays,shift),
                export_check=check, frozen_masks_equal_parent=True, history=history)
            (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print('VALID', axis, json.dumps({k:valid_full[k] for k in ('normalized_y_mse','gate_error','false_negative_rate')}), flush=True)
    result['complete'] = True
    result['wall_seconds'] = time.monotonic()-started
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)


if __name__ == '__main__':
    main()
