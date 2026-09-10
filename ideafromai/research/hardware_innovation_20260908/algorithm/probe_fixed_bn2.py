"""Cheap strong control for any proposed class-domain dynamic-BN hardware.

Calibrate BN2 on the existing train32 of the actual power2 class student;
evaluate fixed S2 BN2 and all12 BN2 on valid10. No weight training or RTL.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from run_bn_probe import ALL_FC1, MLPS, Calibration, build_model, input_frame, save_json, set_bn_mode
from evaluate_stage2_deployment import S2, CoarseReady, install_saved_consumers, summarize
from train_class_shift_probe import install_class_consumers, load_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    alg = args.root/'algorithm'
    out = alg/'fixed_bn2_probe'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((alg/'stage2_temporal_codes/run.json').read_text())
    args.code_root = args.root/'code/SDformer'
    args.config, args.checkpoint = Path(old['config']), Path(old['checkpoint'])
    args.data = args.root.parent/'sdformer_codex/SDformer/data/Datasets/DSEC/saved_flow_data'
    train, valid = old['train_codebook_frames'], json.loads((alg/'samples.json').read_text())['valid'][:10]
    model, _, _, _ = build_model(args)
    model.requires_grad_(False)
    from spikingjelly.activation_based import functional
    stats = torch.load(alg/'valid825_cal32/train_calibration.pt', map_location='cpu', weights_only=False)
    set_bn_mode(model, stats, ALL_FC1)
    params = torch.load(alg/'stage2_temporal_codes/integer_parameters.pt', map_location='cpu', weights_only=False)
    gpu = install_saved_consumers(model, params)
    current = {'cache': None}
    install_class_consumers(model, gpu, alg, current)
    variants = torch.load(alg/'stage2_class_shift/consumers.pt', map_location='cpu', weights_only=False)
    load_variant(gpu, variants['power2_trained'])
    def stop(module, inputs, output):
        current['flow'] = output.detach().sum(0)
        raise CoarseReady()
    dict(model.named_modules())['sttmultires_unet.preds.2'].register_forward_hook(stop)
    targets = [p+'bn2.norm_layer' for p in MLPS]
    collector = Calibration(model, targets)
    with torch.no_grad():
        for i, name in enumerate(train):
            functional.reset_net(model)
            x, _, _ = input_frame(args.data, name, targets=False)
            try:
                model(x)
            except CoarseReady:
                current.pop('flow')
            print('CALIBRATE', i+1, name, flush=True)
        bn2 = collector.finish()
    torch.save(bn2, out/'train_calibration.pt')
    stats.update(bn2)
    save_json(out/'run.json', {'train': train, 'valid': valid, 'targets': targets,
        'teacher': 'power2_trained K8; all12 fixed BN1; coarse preds2',
        'changed': 'BN2 constants only, trained-frame calibration, no gradients', 'limit': 'valid10 screening only'})
    summaries = {}
    for name, chosen in [('dynamic', []), ('S2_fixed', [p+'bn2.norm_layer' for p in S2]), ('all12_fixed', targets)]:
        set_bn_mode(model, stats, list(ALL_FC1)+chosen)
        rows = []
        with torch.no_grad():
            for i, filename in enumerate(valid):
                functional.reset_net(model)
                x, label, mask = input_frame(args.data, filename)
                try:
                    model(x)
                except CoarseReady:
                    pred = F.interpolate(current.pop('flow'), (480, 640), mode='bilinear', align_corners=False)
                error = torch.linalg.vector_norm(pred.permute(0, 2, 3, 1)[mask]-label.permute(0, 2, 3, 1)[mask], dim=1)
                total, pixels = float(error.double().sum()), error.numel()
                rows.append({'file': filename, 'valid_pixels': pixels, 'aee_sum': total, 'AEE': total/pixels})
                print('FRAME', name, i+1, rows[-1]['AEE'], flush=True)
        summaries[name] = summarize(rows, True)
        save_json(out/(name+'_frames.json'), rows)
        save_json(out/'summary.json', summaries)
        print('COMPLETE', name, summaries[name], flush=True)


if __name__ == '__main__':
    main()
