"""Drop the complete private latent tail of each saved recovered student.

No training or rank scan. Saved all-R U calls stay numerically compatible
with LatentPair; a normal compiler can remove all disconnected tail U/V.
Only the complete preview-only gate function is evaluated on local valid4.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from latent_stage import (HERE, FACTOR, PARTIAL, SHARED, load_data, batch,
    read_model, forward)
from postprocess_flow_recovery import arrays_from


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-files', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'flow_recovery64/preview_only')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data_args = argparse.Namespace(source=FACTOR.parent/'joint_completion_20260909/full_capture4/capture',
        capture=PARTIAL/'capture.pt', valid_source=PARTIAL/'integer_valid10',
        operator=PARTIAL/'shared_column_deployment_source.npz', temporal='common3')
    data, op, a, b, theta, yscale, mscale, rate, groups = load_data(data_args)
    constants = dict(a=a, b=b, theta=theta,
        bn_scale=torch.tensor(op['bn_scale'], dtype=torch.float32),
        bn_bias=torch.tensor(op['bn_bias'], dtype=torch.float32))
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False,
        scope='Fixed no-training ordinary shared32-only control for all six recovered students; local valid4 full gates, not network AEE',
        files=data['valid']['files'], native_groups_per_frame=len(groups),
        numeric='Original exported FP32/QDQ U and shared V unchanged. All-R U call retained in the adapter solely for numeric compatibility; no hardware cost attributed to disconnected tail.',
        axes={})
    with torch.no_grad():
        for filename in args.model_files:
            arrays = arrays_from(filename)
            shared_rank = int(arrays['shared_rank'])
            if shared_rank != SHARED:
                raise ValueError('This control keeps the existing shared32 prefix.')
            arrays['v'][shared_rank:] = 0
            arrays['connectivity'][shared_rank:] = False
            for key in ('v_shadow', 'v_shift', 'v_sign', 'v_nonzero'):
                if key in arrays:
                    arrays[key][shared_rank:] = 0
            arrays['completion_mean'].fill(0)
            arrays['completion_covariance'].fill(0)
            arrays.update(preview_only=np.array(True), effective_compiled_rank=np.array(shared_rank),
                completion_statistics_valid=np.array(True),
                completion_calibration_layout=np.array('exact zero tail contribution after V disconnection; no fitted statistics required'),
                preview_only_origin=np.array(str(filename)),
                preview_only_execution=np.array('full shared32-only theta*g function; no predictive acceptance needed; disconnected tail U/V may be deleted by ordinary compilation; stored R slots only preserve all-R adapter numeric call'),
                preview_only_extra_training_updates=np.array(0))
            output = args.output/filename.name
            np.savez_compressed(output, **arrays)
            model, original = read_model(output), read_model(filename)
            counts = dict(gates=0, teacher_positive=0, false_positive=0, false_negative=0,
                full_student_positive=0, dropped_tail_gate_differences=0,
                dropped_tail_false_negatives=0, dropped_tail_false_positives=0)
            snapshots = []
            for first in range(0, len(data['valid']['y']), 8):
                item = batch(data['valid'], torch.arange(first, min(first+8, len(data['valid']['y']))),
                    'cpu', float(arrays['theta_source']))
                gate = forward(model, item, constants)[-1].ge(0)
                old = forward(original, item, constants)[-1].ge(0)
                target = item['target'].ge(0)
                counts['gates'] += gate.numel()
                counts['teacher_positive'] += int(target.sum())
                counts['false_positive'] += int((gate & ~target).sum())
                counts['false_negative'] += int((~gate & target).sum())
                counts['full_student_positive'] += int(old.sum())
                counts['dropped_tail_gate_differences'] += int((gate != old).sum())
                counts['dropped_tail_false_negatives'] += int((~gate & old).sum())
                counts['dropped_tail_false_positives'] += int((gate & ~old).sum())
                snapshots.append(gate.numpy())
            counts.update(full_gate_error=(counts['false_positive']+counts['false_negative'])/counts['gates'],
                full_FN_rate=counts['false_negative']/max(counts['teacher_positive'], 1),
                dropped_tail_gate_difference_rate=counts['dropped_tail_gate_differences']/counts['gates'],
                dropped_tail_FN_rate=counts['dropped_tail_false_negatives']/max(counts['full_student_positive'], 1))
            np.savez_compressed(args.output/(filename.stem+'_gates.npz'),
                full=np.concatenate(snapshots), files=np.array(data['valid']['files']), group_ids=groups)
            result['axes'][filename.stem] = dict(parameters=str(output), valid=counts,
                original_R_slots=int(arrays['u'].shape[1]), effective_compiled_R=shared_rank,
                tail_V_nonzeros=int(np.count_nonzero(arrays['v'][shared_rank:])),
                tail_connectivity_bits=int(np.count_nonzero(arrays['connectivity'][shared_rank:])))
            (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print('PREVIEW_ONLY', filename.stem, json.dumps(counts), flush=True)
    result['complete'] = True
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
