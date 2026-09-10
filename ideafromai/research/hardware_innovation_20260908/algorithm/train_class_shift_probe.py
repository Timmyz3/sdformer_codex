"""Bounded-coefficient S2 class-state PSN: ordinary integers versus powers of2.

Use the existing K8 source and signed6 basis, keep all ten noncausal output
rows. Train AFTER basis folding, so the actual consumer coefficients obey the
format. Compare ordinary [-64,+64] integers (stored signed8) and zero/signed
powers1..64 (15 symbols). Both can use an ordinary CSD shift/add implementation.
Power-of-two quantizer with whole-function STE follows MFPSN; this is neither
its channelwise causal model nor a new arithmetic primitive.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, read_names, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag
from train_mfpsn_probe import ATanSpike


def quantize(x, kind):
    if kind == 'integer':
        return x.round().clamp(-64, 64)
    magnitude = torch.exp2(torch.log2(x.abs().clamp_min(.5)).round().clamp(0, 6))
    return torch.where(x.abs() < .5, 0, x.sign()*magnitude)


class BoundedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind):
        return quantize(x, kind)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def install_class_consumers(model, gpu, alg, current):
    modules = dict(model.named_modules())
    basis = np.load(alg/'stage2_temporal_codes/signed_basis.npz')
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    for prefix in S2:
        name, q = tag(prefix), gpu[prefix]
        words = book[name+'_words'].astype(np.int64)
        reverse = np.full(1024, -1, dtype=np.int64)
        reverse[words] = np.arange(8)
        q['map'] = torch.from_numpy(reverse[book[name+'_word_map'].astype(np.int64)]).cuda()
        q['E'] = torch.from_numpy(basis[name+'_coordinates_int8']).cuda().float()
        q['original_B'] = torch.from_numpy(basis[name+'_B_int32']).cuda().double()
        q['original_tau'] = q['tau'].clone()
        q['B'] = q['original_B']
        assert q['positive'].all() and q['variable'].all()

        def forward(self, x, p=prefix, compiled=q):
            if self.norm_layer in ('LN', 'GN'):
                x = self.norm(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
            source = self.drop1(self.sn1(x))
            shape = source.shape
            shifts = torch.arange(10, device=x.device, dtype=torch.int64).reshape(10, *([1]*(source.ndim-1)))
            raw = ((source != 0).long() << shifts).sum(0)
            codes = compiled['map'][raw].reshape(-1, compiled['w'].shape[1])
            if current.get('cache') is not None:
                current['cache'][p].append(codes.cpu().to(torch.uint8))
            if current.get('capture'):
                directory = current['capture']
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory/(f"v{current['index']:03d}_{Path(current['file']).stem}_{tag(p)}.npz"),
                                    codes=codes.cpu().numpy().astype(np.uint8), source_shape=np.array(shape),
                                    theta_source=np.float32(compiled['theta_source'].item()))
            partials = compiled['E'][codes].permute(0, 2, 1) @ compiled['w'].T
            u = torch.einsum('tr,prh->tph', compiled['B'], partials.double())
            hidden = (u >= compiled['tau'][:, None]).reshape(*shape[:-1], compiled['w'].shape[0]).float()*compiled['theta_output']
            out = self.fc2(self.drop2(hidden))
            if self.norm_layer in ('BN', 'BNTT', 'tdBN', 'IN'):
                out = self.bn2(out.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
            return out
        mlp = modules[prefix.rstrip('.')]
        mlp.forward = types.MethodType(forward, mlp)


def compile_consumer(weight, tau_reference, offset, sigma, source_bound, kind):
    coefficient = quantize(weight, kind).long()
    tau = (tau_reference-offset*sigma).ceil().long()
    bound = coefficient.abs().sum(1)[:, None]*source_bound[None]
    assert int(bound.max()) < 2**23-1, 'source and all intermediate sums must fit Acc24'
    tau = torch.maximum(torch.minimum(tau, bound+1), -bound-1)
    return {'B_int8': coefficient.cpu().to(torch.int8), 'tau_int32': tau.cpu().to(torch.int32),
            'accumulator_abs_bound_max': int(bound.max()),
            'kind': kind, 'coefficient_set': 'integers -64..64 stored signed8' if kind == 'integer' else 'zero or +/- 2^k, k=0..6; 15 symbols'}


def load_variant(gpu, records):
    for prefix, record in records.items():
        gpu[prefix]['B'] = record['B_int8'].cuda().double()
        gpu[prefix]['tau'] = record['tau_int32'].cuda().double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--evaluate', choices=('integer_trained', 'power2_trained'))
    parser.add_argument('--full-valid', action='store_true')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    out = alg/'stage2_class_shift'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train = old['train_codebook_frames']
    names = read_names(args.data, 'valid') if args.full_valid else json.loads((alg/'samples.json').read_text())['valid'][:10]
    model, cfg, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current = {'cache': None}
    install_class_consumers(model, gpu, alg, current)

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()
    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    variants, training = {}, {}
    if not args.evaluate:
        save_json(out/'run.json', {'train': train, 'steps_per_module_and_format': args.steps,
            'modules': S2, 'basis': 'fixed existing signed6; no joint basis learning',
            'changed': 'shared noncausal B after class folding and static t,h decision thresholds',
            'preserved': 'K8 source codebooks, WINT8, original continuous theta amplitudes and real FC2/BN2/shortcut',
            'formats': {'integer': '[-64,+64], signed8 storage', 'power2': '0 or signed powers1..64, 15 symbols'},
            'quantization': 'fixed positive dyadic row scale; whole-quantizer STE; no validation fitting',
            'loss': 'standardized teacher margin MSE + .1 actual FC2 contribution MSE, local distillation',
            'hardware_question': 'same Acc24 source adders and existing RF reused for consumer, ordinary CSD is a required comparator',
            'limits': 'not original ep34 equivalence, full task retraining, new quantization algebra, RTL or PPA'})
        current['cache'] = {p: [] for p in S2}
        with torch.no_grad():
            for i, name in enumerate(train):
                functional.reset_net(model)
                x, _, _ = input_frame(args.data, name, targets=False)
                try:
                    model(x)
                except CoarseReady:
                    current.pop('flow')
                del x
                print('TRAIN_CODES', i+1, name, flush=True)
        cached = current.pop('cache')
        for index, prefix in enumerate(S2):
            q = gpu[prefix]
            with torch.no_grad():
                positions = torch.linspace(0, cached[prefix][0].shape[0]-1, 256, device='cuda').round().long()
                all_codes = torch.stack(cached.pop(prefix)).cuda().long()[:, positions]
                membership = q['E'][all_codes].permute(0, 1, 3, 2)
                cache = (membership @ q['w'].T).to(torch.int16)
                del all_codes, membership
                row_scale = torch.exp2(torch.ceil(torch.log2(q['original_B'].abs().amax(1)/64)))
                original_weight = (q['original_B']/row_scale[:, None]).float()
                tau_reference = (q['original_tau']/row_scale[:, None]).float()
                first = torch.zeros_like(tau_reference); second = torch.zeros_like(first); count = 0
                for partial in cache:
                    u = torch.einsum('tr,prh->tph', original_weight, partial.float())
                    first += u.sum(1); second += u.square().sum(1); count += partial.shape[0]
                sigma = (second/count-(first/count).square()).clamp_min(1).sqrt()
                source_bound = params[prefix]['weight_int8'].long().abs().sum(1).cuda()
            w2 = modules[prefix+'fc2'].weight.detach()
            for kind in ('integer', 'power2'):
                weight = nn.Parameter(original_weight.clone())
                offset = nn.Parameter(torch.zeros_like(tau_reference))
                initial = compile_consumer(weight, tau_reference, offset, sigma, source_bound, kind)
                variants.setdefault(kind+'_rounded', {})[prefix] = initial
                optimizer = torch.optim.Adam([{'params': [weight], 'lr': .025}, {'params': [offset], 'lr': .003}])
                rows = []
                generator = torch.Generator(device='cuda').manual_seed(830+index)
                for step in range(args.steps):
                    pos = torch.randperm(cache.shape[1], device='cuda', generator=generator)[:128]
                    partial = cache[step%cache.shape[0], pos].float()
                    with torch.no_grad():
                        target = (torch.einsum('tr,prh->tph', original_weight, partial)-tau_reference[:, None])/sigma[:, None]
                        gates = (target >= 0).float()
                        consumer = F.linear(gates*q['theta_output'], w2)
                        denom = consumer.var(unbiased=False).clamp_min(.01)
                    quantized = BoundedSTE.apply(weight, kind)
                    margin = (torch.einsum('tr,prh->tph', quantized, partial)-tau_reference[:, None])/sigma[:, None]+offset[:, None]
                    spikes = ATanSpike.apply(margin)
                    membrane_loss = F.mse_loss(margin, target)
                    consumer_loss = F.mse_loss(F.linear(spikes*q['theta_output'], w2), consumer)/denom
                    loss = membrane_loss+.1*consumer_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        weight.clamp_(-64, 64)
                    if step%64 == 0 or step+1 == args.steps:
                        row = {'step': step+1, 'loss': float(loss), 'membrane': float(membrane_loss),
                               'consumer': float(consumer_loss), 'gate_disagreement': float((spikes != gates).float().mean()),
                               'teacher_activity': float(gates.mean()), 'student_activity': float(spikes.mean())}
                        rows.append(row)
                        print('FIT', tag(prefix), kind, json.dumps(row), flush=True)
                record = compile_consumer(weight, tau_reference, offset, sigma, source_bound, kind)
                variants.setdefault(kind+'_trained', {})[prefix] = record
                training[tag(prefix)+'_'+kind] = rows
                print('COMPILED', tag(prefix), kind, 'bound', record['accumulator_abs_bound_max'], flush=True)
            del cache
        torch.save(variants, out/'consumers.pt')
        save_json(out/'training.json', training)
        serial = {name: {tag(p): {key: value.tolist() if torch.is_tensor(value) else value for key, value in r.items()}
                         for p, r in records.items()} for name, records in variants.items()}
        save_json(out/'consumers.json', serial)
    else:
        variants = torch.load(out/'consumers.pt', map_location='cpu', weights_only=False)
        variants = {args.evaluate: variants[args.evaluate]}
    summaries = {}
    for variant, records in variants.items():
        load_variant(gpu, records)
        current['capture'] = out/(variant+'_capture') if not args.full_valid else None
        rows = []
        with torch.no_grad():
            for i, name in enumerate(names):
                current['file'], current['index'] = name, i
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, name)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), size=(480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                row = {'file': name, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels}
                rows.append(row)
                print('FRAME', variant, i+1, json.dumps(row), flush=True)
                if (i+1)%25 == 0 or i+1 == len(names):
                    stem = variant+('_valid825' if args.full_valid else '_valid10')
                    save_json(out/(stem+'_frames.json'), rows)
                    summaries[variant] = summarize(rows, len(rows) == len(names))
                    save_json(out/(stem+'_summary.json'), summaries[variant])
                del x, label, mask, pred, error
        print('COMPLETE', variant, json.dumps(summaries[variant]), flush=True)


if __name__ == '__main__':
    main()
