"""Train-calibrated signed12/signed8 direct-code source feasibility probe.

Reference: the six-source compiled FP32 bits3 student, with the existing
power2-trained class consumers. Calibration only observes its train32 inputs.
Validation compares FP32 and integer source encoders on the fixed diverse10.
No training, integer-source equivalence claim, RTL, or GPU work on import.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag
from evaluate_direct_code_stage2 import compiled_codes
from train_class_shift_probe import install_class_consumers, load_variant


XMIN, XMAX = -2048, 2047
WMIN, WMAX = -128, 127


def dyadic_step(required):
    exponent = math.ceil(math.log2(required)) if required > 0 else 0
    return math.ldexp(1.0, exponent), exponent


def compile_source(exported, minimum, maximum, theta):
    """Prove every implemented accumulator prefix over the full clipped domain."""
    sx, sx_exp = dyadic_step(max(maximum/XMAX, minimum/XMIN, 0.0))
    wf = exported['weight'].cpu().double()
    bias = exported['bias'].cpu().double()
    sw, sw_exp = [], []
    for row in wf:
        step, exponent = dyadic_step(max(float(row.max())/WMAX, float(row.min())/WMIN, 0.0))
        sw.append(step)
        sw_exp.append(exponent)
    sw = torch.tensor(sw, dtype=torch.float64)
    weight = (wf/sw[:, None]).round().clamp(WMIN, WMAX).long()
    lower_terms = torch.where(weight >= 0, weight*XMIN, weight*XMAX)
    upper_terms = torch.where(weight >= 0, weight*XMAX, weight*XMIN)
    dot_min, dot_max = lower_terms.sum(1), upper_terms.sum(1)
    raw_tau = [math.ceil(-float(bias[r])/(sx*float(sw[r]))) for r in range(weight.shape[0])]
    tau = torch.zeros(weight.shape[0], dtype=torch.int32)
    variable = torch.zeros(weight.shape[0], dtype=torch.bool)
    fixed = torch.zeros(weight.shape[0], dtype=torch.bool)
    prefix_min, prefix_max = [], []
    for r, threshold in enumerate(raw_tau):
        if threshold <= int(dot_min[r]):
            fixed[r] = True
        elif threshold <= int(dot_max[r]):
            variable[r] = True
            tau[r] = threshold
        if variable[r]:
            lo = torch.cat([torch.zeros(1, dtype=torch.int64), lower_terms[r].cumsum(0)])-int(tau[r])
            hi = torch.cat([torch.zeros(1, dtype=torch.int64), upper_terms[r].cumsum(0)])-int(tau[r])
        else:
            # Constant predicates have no stored threshold or MAC sequence.
            lo = hi = torch.zeros(weight.shape[1]+1, dtype=torch.int64)
        assert int(lo.min()) >= -(2**23) and int(hi.max()) <= 2**23-1, 'source sequence does not fit Acc24'
        prefix_min.append(lo)
        prefix_max.append(hi)
    record = {'weight_int8': weight.to(torch.int8), 'input_step': sx, 'input_step_exponent': sx_exp,
              'weight_steps': sw, 'weight_step_exponents': sw_exp,
              'threshold_int32': tau, 'unclamped_thresholds': raw_tau,
              'variable_rows': variable, 'constant_gates': fixed,
              'mapping': exported['mapping'].cpu().long(), 'theta_source': theta.detach().cpu(),
              'input_domain': [XMIN, XMAX], 'weight_domain': [WMIN, WMAX],
              'dot_min_int64': dot_min, 'dot_max_int64': dot_max,
              'prefix_min_int64': torch.stack(prefix_min), 'prefix_max_int64': torch.stack(prefix_max),
              'train_input_min': minimum, 'train_input_max': maximum,
              'arithmetic': 'round-even signed12 input, signed8 weights, ceil threshold, int64 reference; all variable prefixes fit Acc24',
              'scope': 'new quantized source model; continuous output theta is separate from decision tau'}
    return record


def integer_codes(x, record):
    scaled = x/record['input_step']
    rounded = scaled.round()
    clipped = (rounded < XMIN) | (rounded > XMAX)
    outside = (scaled < XMIN) | (scaled > XMAX)
    quantized = rounded.clamp(XMIN, XMAX).long()
    variable = record['variable_rows']
    weight = record['weight_int8'].long()*variable[:, None]
    margin = -record['threshold_int32'].long().expand(*x.shape[:-1], 3).clone()
    # CUDA elementwise int64 avoids relying on unsupported integer GEMM paths.
    for t in range(weight.shape[1]):
        margin += quantized[..., t, None]*weight[:, t]
    gates = torch.where(variable, margin >= 0, record['constant_gates'])
    address = (gates.long() << torch.arange(3, device=x.device)).sum(-1)
    return record['mapping'][address], clipped.sum(), outside.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    alg, out = args.root/'algorithm', args.root/'algorithm/direct_code_integer'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train = old['train_codebook_frames']
    valid = json.loads((alg/'samples.json').read_text())['valid'][:10]
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    old_params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, old_params)
    current = {'cache': None, 'phase': 'calibrate'}
    install_class_consumers(model, gpu, alg, current)
    consumers = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, consumers['power2_trained'])
    exported_all = torch.load(alg/'direct_code_stage2/producers.pt', map_location='cpu', weights_only=False)
    book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    reference, extrema, references, integer, counters = {}, {}, {}, {}, {}
    for prefix in S2:
        exported = exported_all[prefix]['bits3']
        reference[prefix] = {'kind': exported['kind'], 'weight': exported['weight'].cuda().float(),
                             'bias': exported['bias'].cuda().float(), 'mapping': exported['mapping'].cuda().long()}
        extrema[prefix] = {'minimum': torch.tensor(float('inf'), device='cuda'),
                           'maximum': torch.tensor(float('-inf'), device='cuda'), 'elements': 0}
        references[prefix] = []
        counters[prefix] = {'input_elements': 0, 'codes': 0,
                            'clipped': torch.zeros((), dtype=torch.int64, device='cuda'),
                            'outside': torch.zeros((), dtype=torch.int64, device='cuda'),
                            'same_input_disagreements': torch.zeros((), dtype=torch.int64, device='cuda'),
                            'end_to_end_disagreements': torch.zeros((), dtype=torch.int64, device='cuda')}
        dictionary = torch.from_numpy(book[tag(prefix)+'_dictionary']).cuda().float()

        def source(self, x, p=prefix, words=dictionary, q=gpu[prefix]):
            shape = x.shape
            vectors = x.reshape(shape[0], -1, q['w'].shape[1]).permute(1, 2, 0)
            fp_codes = compiled_codes(vectors, reference[p])
            if current['phase'] == 'calibrate':
                minimum, maximum = vectors.aminmax()
                extrema[p]['minimum'] = torch.minimum(extrema[p]['minimum'], minimum)
                extrema[p]['maximum'] = torch.maximum(extrema[p]['maximum'], maximum)
                extrema[p]['elements'] += vectors.numel()
                codes = fp_codes
            elif current['phase'] == 'float_reference':
                references[p].append(fp_codes.cpu().to(torch.uint8))
                codes = fp_codes
            else:
                codes, clipped, outside = integer_codes(vectors, integer[p])
                counts = counters[p]
                counts['input_elements'] += vectors.numel()
                counts['codes'] += codes.numel()
                counts['clipped'] += clipped
                counts['outside'] += outside
                counts['same_input_disagreements'] += (codes != fp_codes).sum()
                previous = references[p][current['index']].to(device=x.device)
                counts['end_to_end_disagreements'] += (codes != previous).sum()
            return (words[codes].permute(2, 0, 1)*q['theta_source']).reshape(shape)

        neuron = modules[prefix+'sn1.spiking_neuron']
        neuron.forward = types.MethodType(source, neuron)

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    save_json(out/'run.json', {'reference': 'six-source compiled FP32 bits3 student',
              'train': train, 'valid': valid, 'modules': S2,
              'calibration': 'actual inputs from current compiled FP32 student, train32 only; full spatial domain, no fitting',
              'input_quantizer': 'one positive dyadic step per module, signed12, round-even and clamp',
              'weight_quantizer': 'one positive dyadic step per decision row, signed8, round-even',
              'threshold': 'ceil(-folded_bias/(input_step*weight_step)); out-of-domain thresholds become constant predicates',
              'amplitudes': 'real theta_source and theta_output retained; decision tau is separate',
              'bounds': 'all prefixes initialized with minus tau checked against Acc24 over full clipped signed12 domain',
              'same_input_disagreement': 'integer and compiled FP32 decisions evaluated on the same integer-student upstream tensor',
              'end_to_end_disagreement': 'integer-student codes versus separate complete FP32-reference pass, including propagated input changes',
              'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
              'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
              'limits': 'new integer source model; not FP32 bit equivalence or hardware cycles; consumer and dynamic BN2 unchanged'})
    with torch.no_grad():
        for index, filename in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, filename, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            del x
            print('CALIBRATE', index+1, filename, flush=True)
    parameters, descriptions = {}, {}
    for prefix in S2:
        limits = extrema[prefix]
        record = compile_source(exported_all[prefix]['bits3'], float(limits['minimum']),
                                float(limits['maximum']), gpu[prefix]['theta_source'])
        parameters[prefix] = record
        integer[prefix] = {key: value.cuda() if torch.is_tensor(value) else value for key, value in record.items()}
        descriptions[tag(prefix)] = {
            key: value.tolist() if torch.is_tensor(value) else value for key, value in record.items()}
        descriptions[tag(prefix)]['calibration_input_elements'] = limits['elements']
    torch.save(parameters, out/'parameters.pt')
    save_json(out/'parameters.json', descriptions)
    summaries = {}
    for phase in ('float_reference', 'integer_student'):
        current['phase'] = phase
        rows = []
        with torch.no_grad():
            for index, filename in enumerate(valid):
                current['index'] = index
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]
                                                -label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                row = {'file': filename, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels}
                rows.append(row)
                print('FRAME', phase, index+1, json.dumps(row), flush=True)
                del x, label, mask, pred, error
        summaries[phase] = summarize(rows, True)
        save_json(out/(phase+'_valid10_frames.json'), rows)
        save_json(out/'summary.json', summaries)
        print('COMPLETE', phase, json.dumps(summaries[phase]), flush=True)
    source_statistics = {}
    for prefix, counts in counters.items():
        record = {key: int(value) for key, value in counts.items()}
        record['clipping_fraction'] = record['clipped']/record['input_elements']
        record['outside_representable_range_fraction'] = record['outside']/record['input_elements']
        record['same_input_code_disagreement_fraction'] = record['same_input_disagreements']/record['codes']
        record['end_to_end_code_disagreement_fraction'] = record['end_to_end_disagreements']/record['codes']
        source_statistics[tag(prefix)] = record
    save_json(out/'source_statistics.json', source_statistics)
    print('SOURCE_STATISTICS', json.dumps(source_statistics), flush=True)


if __name__ == '__main__':
    main()
