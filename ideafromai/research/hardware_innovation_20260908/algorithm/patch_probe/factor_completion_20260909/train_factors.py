"""Fit one fixed CFMP-style factor/mask budget to saved real patch inputs.

Local train16/valid4 recovery, not author-code reproduction or network AEE.
All axes keep original post-Conv1 BN, full T10 A/bias/theta and the downstream
interface. The adapter below replaces only Conv1; real Conv2 is not replaced
by an isolated-tile loss. CUDA execution is explicitly launched by the root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from factor_reference import STRUCTURES, initialize, spatial_regions, storage_summary

HERE = Path(__file__).resolve().parent
PATCH = HERE.parent
PARTIAL = PATCH/'partial_completion'


def words_from_full_capture(data, groups):
    """Read only sampled indices from packed full T,C,H,W source bits."""
    times, channels, height, width = map(int, data['source_gate_shape'])
    if (times, channels, height, width) != (10,96,240,320):
        raise ValueError('Unexpected patch geometry in full source capture.')
    source = data['source_gate_bits'].reshape(-1)
    k = np.arange(864)
    c, kh, kw = k//9, (k//3)%3, k%3
    ys = groups[:, None, None]//(width//4)+kh[None, :, None]-1
    xs = (groups[:, None, None]%(width//4))*4+np.arange(4)[None, None, :]+kw[None, :, None]-1
    valid = (ys>=0)&(ys<height)&(xs>=0)&(xs<width)
    yy, xx = np.clip(ys,0,height-1), np.clip(xs,0,width-1)
    bitorder = str(data['bitorder'].item())
    words = np.zeros((len(groups),864,4), np.int16)
    for t in range(times):
        index = (((t*channels+c[None, :, None])*height+yy)*width+xx)
        bit = index%8 if bitorder=='little' else 7-index%8
        active = ((source[index//8] >> bit)&1)&valid
        words |= active.astype(np.int16)<<t
    return words


def load_source_rows(path):
    if path.is_dir():
        rows, groups = [], None
        for filename in sorted(path.rglob('sampled_source.npz')):
            with np.load(filename) as data:
                row = dict(file=str(data['frame_name'].item()), split='',
                    source_gate_words=torch.from_numpy(data['source_gate_words'].copy()),
                    theta_source=float(data['theta_source']))
                if 'Y' in data:
                    row['Y'] = torch.from_numpy(data['Y'].copy())
                    row['target_origin'] = 'same-forward sampled_source.npz Y'
                current = data['group_ids'].copy()
                if groups is not None and not np.array_equal(groups, current):
                    raise ValueError('Native group IDs differ within the new source capture.')
                groups = current
                rows.append(row)
        if groups is None:
            raise ValueError('No sampled_source.npz beneath '+str(path))
        known = {row['file'] for row in rows}
        for filename in sorted(path.rglob('gates.npz')):
            with np.load(filename) as data:
                name = str(data['frame_name'].item())
                if name in known:
                    continue
                words = words_from_full_capture(data, groups)
                rows.append(dict(file=name, split='valid', theta_source=float(data['theta_source']),
                    source_gate_words=torch.from_numpy(words),
                    source_origin='fresh complete source_gate_bits sampled at native P4'))
                known.add(name)
        return rows, groups
    if path.suffix == '.npz':
        with np.load(path) as data:
            key = 'source_gate_words' if 'source_gate_words' in data else 'words'
            splits = data['splits'] if 'splits' in data else data['split']
            files = data['files'] if 'files' in data else data['names']
            return [dict(file=str(name), split=str(split), source_gate_words=torch.from_numpy(word.copy()))
                    for name, split, word in zip(files, splits, data[key])], data['groups'].copy()
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data['samples'], np.asarray(data['groups'])


def load_data(args):
    cap = torch.load(args.capture, map_location='cpu', weights_only=False)
    source_rows, groups = load_source_rows(args.source)
    if not np.array_equal(groups, np.asarray(cap['groups'])):
        raise ValueError('Source groups must be the same native P4 groups as the membrane capture.')
    rows_by_file = {row['file']: row for row in source_rows}
    # Existing four validation source captures precede Conv1, so they can be
    # paired with old norm1 Y only after exact support/count agreement below.
    # They are not used to initialize or train any mask or factors.
    if args.valid_source is not None:
        for filename in sorted(args.valid_source.glob('capture_*.npz')):
            with np.load(filename) as data:
                name = str(data['file'].item())
                if name not in rows_by_file:
                    if not np.array_equal(groups, data['group_ids']):
                        raise ValueError('Validation native group IDs differ.')
                    rows_by_file[name] = dict(file=name, split='valid',
                        source_gate_words=torch.from_numpy(data['source_gate_words'].copy()))
                elif 'source_origin' in rows_by_file[name]:
                    actual = rows_by_file[name]['source_gate_words'].numpy()
                    if not np.array_equal(actual, data['source_gate_words']):
                        raise ValueError('Fresh validation source differs from complete old source words.')
                    rows_by_file[name]['full_source_words_equal_old_capture'] = True
    if cap['metadata']['center_mode'] not in ('zero', 'none'):
        raise ValueError('This explicit temporal adapter expects the captured zero-centered neuron.')
    with np.load(args.operator) as data:
        operator = {key: data[key].copy() for key in data.files}
    if args.temporal == 'native':
        a = cap['neuron_state']['weight'].float()
        b = cap['neuron_state']['bias'].float().reshape(10)
    elif args.temporal == 'common3':
        selected = json.loads((PARTIAL/'shared_column_result.json').read_text())['selected']['common3_diagonal_34']
        a, b = torch.tensor(selected['weight']).float(), torch.tensor(selected['bias']).float().reshape(10)
    else:
        selected = torch.load(PATCH/'dependency/parameters.pt', map_location='cpu', weights_only=False)['row34']
        a, b = selected['weight'].float(), selected['bias'].float().reshape(10)
    theta = float(cap['neuron_state']['thresh'])
    if abs(float(operator['source_theta'])-float(cap['metadata']['source_theta'])) > 1e-7:
        raise ValueError('Source theta differs between the paired capture and original operator.')
    if abs(float(operator['output_theta'])-theta) > 1e-7:
        raise ValueError('Output theta differs from the neuron capture.')
    splits = {}
    for split in ('train', 'valid'):
        rows = [r for r in cap['samples'] if r['split'] == split and r['file'] in rows_by_file]
        words, ys, files, pairing = [], [], [], []
        for row in rows:
            source = rows_by_file[row['file']]
            if source['split'] not in ('', split):
                raise ValueError('Source and Y split mismatch.')
            value = torch.as_tensor(source['source_gate_words']).short()
            if tuple(value.shape) != (64, 864, 4):
                raise ValueError('Expected native source words [G64,K864,P4].')
            if 'theta_source' in source and abs(source['theta_source']-float(operator['source_theta'])) > 1e-7:
                raise ValueError('New source amplitude differs from the paired operator.')
            old_or = row['source_words'].short()
            new_or = value[:, :, 0] | value[:, :, 1] | value[:, :, 2] | value[:, :, 3]
            if not torch.equal(new_or, old_or):
                raise ValueError('Source P4 OR differs from the prior paired capture for '+row['file'])
            new_count = torch.stack([((value.long() >> t) & 1).sum(1) for t in range(10)], -1)
            if not torch.equal(new_count.short(), row['source_active_terms'].short()):
                raise ValueError('Per-position source count differs for '+row['file'])
            words.append(value)
            # Saved Y is T,H,G,P. Both factors return G,T,P,H.
            target_y = source.get('Y', row['Y'])
            if tuple(target_y.shape) != (10,96,64,4):
                raise ValueError('Expected paired norm1 Y[T10,H96,G64,P4].')
            ys.append(target_y.permute(2, 0, 3, 1).contiguous().float())
            files.append(row['file'])
            pairing.append(dict(file=row['file'],
                Y_origin=source.get('target_origin','original real FP32 norm1 capture, unchanged source/operator'),
                fresh_full_source_equal_old=source.get('full_source_words_equal_old_capture'),
                same_forward_Y_difference_old=float((target_y-row['Y']).abs().max()) if 'Y' in source else None))
        if not rows:
            raise ValueError('No paired '+split+' frames; no silent validation substitution.')
        y = torch.stack(ys).flatten(0, 1)
        splits[split] = dict(words=torch.stack(words).flatten(0, 1), y=y,
            regions=torch.from_numpy(np.tile(spatial_regions(groups), len(rows))), files=files,
            teacher_margin=torch.einsum('ts,bsph->btph', a, y)+b[None, :, None, None]-theta,
            pairing=pairing)
    train = splits['train']
    scale = train['y'].std((0, 1, 2)).clamp_min(0.025)
    margin_scale = train['teacher_margin'].std((0, 1, 2)).clamp_min(0.025)
    positive_rate = train['teacher_margin'].ge(0).float().mean((0, 1, 2)).clamp(0.005, 0.995)
    return splits, operator, a, b, theta, scale, margin_scale, positive_rate, groups


class FactorModel(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.u = nn.Parameter(torch.from_numpy(initial['u']).float())
        self.v = nn.Parameter(torch.from_numpy(initial['v']).float())
        logits = initial['logits']/max(float(np.std(initial['logits'])), 1.)
        self.logits = nn.Parameter(torch.from_numpy(logits).float(),
                                   requires_grad=not initial['structure'].endswith('_compact'))
        self.register_buffer('connectivity', torch.from_numpy(initial['connectivity']))
        self.keep = int(initial['active_tiles'])
        self.latent_tile = int(initial['latent_tile'])
        self.structure = initial['structure']

    def masks(self, training=False):
        hard = torch.zeros_like(self.logits)
        # Stable original-index tie breaking, shared by every axis.
        ids = torch.argsort(self.logits, dim=1, descending=True, stable=True)[:, :self.keep]
        hard.scatter_(1, ids, 1)
        if training and self.logits.requires_grad:
            soft = self.logits.sigmoid()
            return hard + (soft-soft.detach())
        return hard

    def forward(self, source, regions):
        # Dense training/reference operations. The executable NumPy TC/TR
        # path separately checks that omitted latent columns need not exist.
        mask = self.masks(self.training)[regions].repeat_interleave(self.latent_tile, dim=1)
        z = (source @ self.u) * mask[:, None, None, :]
        return z @ (self.v*self.connectivity)

    def export(self):
        return dict(u=self.u.detach().cpu().numpy(),
            v=(self.v*self.connectivity).detach().cpu().numpy(),
            connectivity=self.connectivity.cpu().numpy(),
            masks=self.masks().detach().cpu().numpy().astype(bool),
            logits=self.logits.detach().cpu().numpy(), latent_tile=np.array(self.latent_tile),
            active_tiles=np.array(self.keep), structure=np.array(self.structure),
            region_rule=np.array('eight horizontal bands: min(y*8//H,7), shared over width and T10'))


def batch(data, ids, device, theta):
    words = data['words'][ids].to(device=device, dtype=torch.int32)
    # G,K,P,T -> G,T,P,K. Input remains theta*g, not arbitrary INT8.
    bits = ((words[..., None] >> torch.arange(10, device=device)) & 1)
    source = bits.permute(0, 3, 2, 1).float()*theta
    return dict(source=source, regions=data['regions'][ids].to(device),
                y=data['y'][ids].to(device), target=data['teacher_margin'][ids].to(device))


def forward_membrane(model, item, constants):
    raw = model(item['source'], item['regions'])
    y = raw*constants['bn_scale']+constants['bn_bias']
    margin = torch.einsum('ts,bsph->btph', constants['a'], y)+constants['b'][None, :, None, None]-constants['theta']
    return y, margin


@torch.no_grad()
def evaluate(model, data, constants, device, source_theta, chunk=16):
    model.eval()
    values = 0
    sums = dict(y_squared_error=0., normalized_y_squared_error=0., membrane_squared_error=0.,
                teacher_positive=0, false_positive=0, false_negative=0)
    for start in range(0, len(data['y']), chunk):
        ids = torch.arange(start, min(start+chunk, len(data['y'])))
        item = batch(data, ids, device, source_theta)
        y, margin = forward_membrane(model, item, constants)
        expected, gate = item['target'].ge(0), margin.ge(0)
        values += y.numel()
        sums['y_squared_error'] += float((y-item['y']).square().sum())
        sums['normalized_y_squared_error'] += float(((y-item['y'])/constants['y_scale']).square().sum())
        sums['membrane_squared_error'] += float((margin-item['target']).square().sum())
        sums['teacher_positive'] += int(expected.sum())
        sums['false_positive'] += int((gate & ~expected).sum())
        sums['false_negative'] += int((~gate & expected).sum())
    return dict(**sums, values=values, y_mse=sums['y_squared_error']/values,
        normalized_y_mse=sums['normalized_y_squared_error']/values,
        membrane_mse=sums['membrane_squared_error']/values,
        gate_error=(sums['false_positive']+sums['false_negative'])/values,
        false_negative_rate=sums['false_negative']/max(sums['teacher_positive'], 1),
        theta_output=constants['theta'], scope='sampled Conv1/BN/T10 gates, not flow AEE')


class FactorConv1(nn.Module):
    """Full-image dense functional adapter: replace only raw Conv1.

    This GPU-friendly adapter computes dense Z then applies its mask. It is
    a numerical network evaluator, not claimed to realize sparse speedup.
    Actual sparse TC/TR production is checked by factor_reference.py.
    """
    def __init__(self, parameters, input_channels=96, kernel=3, padding=1):
        super().__init__()
        self.register_buffer('u', torch.as_tensor(parameters['u']).float().T.reshape(-1, input_channels, kernel, kernel))
        self.register_buffer('v', torch.as_tensor(parameters['v']).float().T[:, :, None, None])
        mask = np.repeat(parameters['masks'], int(parameters['latent_tile']), axis=1)
        self.register_buffer('mask', torch.from_numpy(mask).float())
        self.padding = padding

    def forward(self, x):
        original = x.shape
        is_temporal = x.ndim == 5
        if is_temporal:
            x = x.flatten(0, 1)
        z = F.conv2d(x, self.u, padding=self.padding)
        height = z.shape[-2]
        region = torch.arange(height, device=x.device)*len(self.mask)//height
        z = z*self.mask[region].T[None, :, :, None]
        y = F.conv2d(z, self.v)
        if is_temporal:
            y = y.reshape(original[0], original[1], *y.shape[1:])
        return y


def install_conv1(module, parameter_file):
    """Returns (adapter, original_forward); caller retains the original BN/sn2."""
    if module.bias is not None:
        raise ValueError('The captured raw Conv1 has no bias; unexpected operator identity.')
    with np.load(parameter_file) as data:
        adapter = FactorConv1({k: data[k].copy() for k in data.files}).to(module.weight.device)
    old_forward = module.forward
    module.forward = adapter.forward
    return adapter, old_forward


def install_factor_stack(conv1, neuron, parameter_file):
    """Numerical network adapter, preserving real BN/Conv2/shortcut.

    The saved full A is the common parent for the factor comparison. A model
    freshly built from ep34 must install it, not silently retain native A.
    The returned originals let a caller restore its model between axes.
    """
    adapter, old_forward = install_conv1(conv1,parameter_file)
    original = dict(conv1_forward=old_forward,temporal_weight=neuron.weight.detach().clone(),
        temporal_bias=neuron.bias.detach().clone(),temporal_factor_rank=getattr(neuron,'temporal_factor_rank',0))
    with np.load(parameter_file) as values,torch.no_grad():
        if abs(float(neuron.thresh)-float(values['theta_output']))>1e-7:
            raise ValueError('Keep the captured output amplitude theta; this is not tau.')
        neuron.weight.copy_(torch.as_tensor(values['a']).to(neuron.weight))
        neuron.bias.copy_(torch.as_tensor(values['temporal_bias']).reshape_as(neuron.bias).to(neuron.bias))
        neuron.temporal_factor_rank=0
    return adapter,original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--valid-source', type=Path, default=PARTIAL/'integer_valid10')
    parser.add_argument('--capture', type=Path, default=PARTIAL/'capture.pt')
    parser.add_argument('--operator', type=Path, default=PARTIAL/'shared_column_deployment_source.npz')
    parser.add_argument('--temporal', choices=['native', 'row34', 'common3'], default='common3')
    parser.add_argument('--structures', nargs='+', choices=STRUCTURES, default=list(STRUCTURES))
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--batch-groups', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=HERE/'fit_train16')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(909)
    np.random.seed(909)
    # Inherited matmul/cudnn settings are recorded, not silently called exact.
    data, operator, a, b, theta, scale, margin_scale, rate, groups = load_data(args)
    source_theta = float(operator['source_theta'])
    constants = dict(a=a.to(args.device), b=b.to(args.device), theta=theta,
        bn_scale=torch.tensor(operator['bn_scale'], dtype=torch.float32, device=args.device),
        bn_bias=torch.tensor(operator['bn_bias'], dtype=torch.float32, device=args.device),
        y_scale=scale.to(args.device), margin_scale=margin_scale.to(args.device), rate=rate.to(args.device))
    weight = operator['weight'].reshape(96, 864).T
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(909)
    batches = torch.randint(len(data['train']['y']), (args.steps, args.batch_groups), generator=generator)
    result = dict(scope='fixed-R local factor/mask recovery on paired train16/valid4 native P4 capture',
        method_status='CFMP published TC/TR mask semantics + our explicitly specified initialization/STE/loss; not official training reproduction',
        training=dict(steps=args.steps, batch_native_P4=args.batch_groups, optimizer='Adam', lr=0.002, seed=909,
            fit_only='U/V and train-only spatial mask logits; fixed outer BN, A, temporal bias, theta',
            loss='normalized post-BN Y MSE + 0.25 train-class-balanced gate BCE',
            no_validation_selection=True, no_Conv2_surrogate_loss=True),
        mask=dict(regions=8, rule='horizontal y bands, y*8//240; shared across x and all T10',
            tile=[4,4], rank_pool=96, active_rank=48, expanded_mask_bytes=24,
            ordinary_controls='all-shared and H8-grouped compact rank48, no tiled mask'),
        teacher='captured original Conv1 Y followed by the same selected full A/bias/theta',
        temporal=args.temporal, theta_source=source_theta, theta_output=theta,
        numeric='new FP32 two-factor/BN-affine reference; not previous INT8 or frozen-FP equivalence',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        train=data['train']['files'], valid=data['valid']['files'], groups=groups.tolist(),
        pairing={split:data[split]['pairing'] for split in ('train','valid')},
        missing='no full-network AEE, Conv2 halo supervision, trained early-stop policy, finite-port timing, or PPA', axes={})
    started = time.monotonic()
    for structure in args.structures:
        initial = initialize(weight, structure)
        model = FactorModel(initial).to(args.device)
        before = evaluate(model, data['train'], constants, args.device, source_theta)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        history = []
        model.train()
        for step, ids in enumerate(batches):
            item = batch(data['train'], ids, args.device, source_theta)
            y, margin = forward_membrane(model, item, constants)
            fit = ((y-item['y'])/constants['y_scale']).square().mean()
            expected = item['target'].ge(0)
            weights = torch.where(expected, 0.5/constants['rate'], 0.5/(1-constants['rate']))
            logits = margin/(0.25*constants['margin_scale']).clamp_min(0.025)
            gate_loss = (F.binary_cross_entropy_with_logits(logits, expected.float(), reduction='none')*weights).mean()
            loss = fit+0.25*gate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            with torch.no_grad():
                model.v.mul_(model.connectivity)
            if step % 64 == 0 or step == args.steps-1:
                entry = dict(step=step+1, total=float(loss.detach()), normalized_y_mse=float(fit.detach()),
                             balanced_gate_bce=float(gate_loss.detach()))
                history.append(entry)
                print('TRAIN', structure, json.dumps(entry), flush=True)
        train = evaluate(model, data['train'], constants, args.device, source_theta)
        valid = evaluate(model, data['valid'], constants, args.device, source_theta)
        arrays = model.export()
        arrays.update(a=a.numpy(), temporal_bias=b.numpy(), theta_source=np.array(source_theta),
                      theta_output=np.array(theta), bn_scale=operator['bn_scale'], bn_bias=operator['bn_bias'])
        np.savez_compressed(args.output/(structure+'.npz'), **arrays)
        result['axes'][structure] = dict(initial_train=before, train=train, valid=valid,
            storage=storage_summary(initial), masks=arrays['masks'].astype(int).tolist(), history=history,
            actual_trained_v_nonzeros=int(np.count_nonzero(arrays['v'])),
            distinct_region_masks=int(np.unique(arrays['masks'], axis=0).shape[0]),
            constant_mask_control='If all region masks coincide, drop inactive factor columns statically; no runtime-mask advantage.')
        (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print('VALID', structure, json.dumps(valid), flush=True)
    result['complete'] = True
    result['wall_seconds'] = time.monotonic()-started
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('DONE', result['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
