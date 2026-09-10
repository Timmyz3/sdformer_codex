"""Evaluate one compiled FP32 six-source variant without retraining.

Default: the fixed diverse ten validation frames, including actual code
captures. --full-valid: all official validation frames, without tensor dumps.
Folded FP32 affine decisions are evaluated directly; equivalence to the
training script's normalization followed by a dot product is not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

from run_bn_probe import ALL_FC1, build_model, input_frame, read_names, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize, tag
from train_class_shift_probe import install_class_consumers, load_variant


def compiled_codes(x, record):
    logits = F.linear(x, record['weight'], record['bias'])
    if record['kind'] == 'scores8':
        return logits.argmax(-1)
    if record['kind'] == 'scalar8':
        address = logits[..., 0].round().long().clamp(0, 7)
    else:
        address = ((logits >= 0).long() << torch.arange(3, device=x.device)).sum(-1)
    return record['mapping'][address]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', required=True,
                        choices=('teacher', 'rows3', 'bits3', 'scores8', 'scalar8', 'zero_code', 'identity'))
    parser.add_argument('--full-valid', action='store_true')
    args = parser.parse_args()
    alg = args.root/'algorithm'
    out = alg/'direct_code_stage2'/('compiled_'+args.variant)
    out.mkdir(parents=True, exist_ok=True)
    stem = 'valid825' if args.full_valid else 'valid10'
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    names = (read_names(args.data, 'valid') if args.full_valid
             else json.loads((alg/'samples.json').read_text())['valid'][:10])
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    modules = dict(model.named_modules())
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    gpu = install_saved_consumers(model, params)
    current = {'cache': None}
    install_class_consumers(model, gpu, alg, current)
    variants = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, variants['power2_trained'])

    needs_producer = args.variant in ('rows3', 'bits3', 'scores8', 'scalar8')
    if needs_producer:
        producers = torch.load(alg/'direct_code_stage2/producers.pt', map_location='cpu', weights_only=False)
        book = np.load(alg/'stage2_temporal_codes/codebooks.npz')
        for prefix in S2:
            exported = producers[prefix][args.variant]
            record = {'kind': exported['kind'], 'weight': exported['weight'].cuda().float(),
                      'bias': exported['bias'].cuda().float(),
                      'mapping': exported['mapping'].cuda().long()}
            dictionary = torch.from_numpy(book[tag(prefix)+'_dictionary']).cuda().float()

            def direct(self, x, compiled=record, words=dictionary, q=gpu[prefix]):
                shape = x.shape
                vectors = x.reshape(shape[0], -1, q['w'].shape[1]).permute(1, 2, 0)
                codes = compiled_codes(vectors, compiled)
                # Adapter to the real class consumer; its own capture records
                # the code actually consumed after this theta*g reconstruction.
                return (words[codes].permute(2, 0, 1)*q['theta_source']).reshape(shape)

            source = modules[prefix+'sn1.spiking_neuron']
            source.forward = types.MethodType(direct, source)
    elif args.variant == 'zero_code':
        for prefix in S2:
            # Actual code0 must be the all-silent word. The consumer still
            # computes (0 >= tau), FC2 and its current-frame BN2.
            assert int(gpu[prefix]['map'][0]) == 0

            def zero(self, x):
                return torch.zeros_like(x)

            source = modules[prefix+'sn1.spiking_neuron']
            source.forward = types.MethodType(zero, source)
    elif args.variant == 'identity':
        for prefix in S2:
            def skip(self, x):
                return torch.zeros_like(x)
            mlp = modules[prefix.rstrip('.')]
            mlp.forward = types.MethodType(skip, mlp)

    capture = not args.full_valid and args.variant != 'identity'
    current['capture'] = out/'capture' if capture else None

    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()

    modules['sttmultires_unet.preds.2'].register_forward_hook(stop)
    metadata = {'variant': args.variant, 'modules': S2, 'frames': names,
                'arithmetic': ('folded FP32 affine source decisions; no integer-source claim'
                               if needs_producer else 'original teacher sources' if args.variant == 'teacher'
                               else 'all-silent source code, actual fixed consumer retained' if args.variant == 'zero_code'
                               else 'entire six MLP branches removed; outer shortcut retained'),
                'teacher_consumer': 'fixed power2_trained K8 consumer in all six S2 blocks',
                'preserved': 'theta amplitudes, W/B/tau, real FC2/BN2/shortcut except explicitly removed identity branches',
                'numeric_boundary': 'compiled affine and normalized-then-dot FP32 can disagree near a decision boundary; this run evaluates the compiled model',
                'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
                'capture': ('actual class-consumer codes[P,C], logical3 bits stored uint8; source_shape and theta_source'
                            if capture else 'none: full validation does not save tensors' if args.full_valid
                            else 'none: identity bypass has no executed class consumer; no fabricated source codes'),
                'model_scope': 'new deployed FP32 source model; no fitting or validation-based selection in evaluator',
                'readout': 'preds.2 sum over time, bilinear to480x640, align_corners=False, no flow rescaling'}
    save_json(out/(stem+'_run.json'), metadata)
    rows = []
    with torch.no_grad():
        for index, filename in enumerate(names):
            current['file'], current['index'] = filename, index
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
            if not args.full_valid or (index+1) % 50 == 0 or index+1 == len(names):
                result = summarize(rows, len(rows) == len(names))
                result['variant'] = args.variant
                result['compiled_source'] = needs_producer
                result['captured_files'] = len(rows)*len(S2) if capture else 0
                save_json(out/(stem+'_frames.json'), rows)
                save_json(out/(stem+'_summary.json'), result)
                print('PROGRESS', args.variant, index+1, len(names), json.dumps(result), flush=True)
            del x, label, mask, pred, error
    print('COMPLETE', args.variant, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
