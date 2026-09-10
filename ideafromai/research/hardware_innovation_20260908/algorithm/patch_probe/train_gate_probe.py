"""Compare fitted temporal controls against saved real train-only membranes.

This is a local diagnostic on 512 uniform positions per training frame, not
an AEE evaluation or a whole-layer equivalence check. Actual FP32 teacher
gates are retained; NumPy Float64 errors are reported explicitly.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'bn_state'))
from support_service_model import read_torch


def main():
    directory = HERE/'dependency'
    saved = read_torch(directory/'recovery_train_samples.pt')
    variants = json.loads((directory/'fit.json').read_text())['variants']
    variants['reordered_masked_k4'] = json.loads(
        (directory/'reordered_masked_k4.json').read_text())
    variants.update(json.loads((directory/'reassigned_groups.json').read_text())['variants'])
    theta = float(saved['theta'])
    result = dict(
        scope='train32, 512 uniformly sampled spatial positions/frame, all C96/T10',
        frames=saved['frames'], theta=theta,
        near_threshold_definition='abs(actual teacher membrane - theta) <= 0.1 * abs(theta)',
        arithmetic='NumPy Float64 on stored FP32 inputs and stored FP32 fitted coefficients; actual FP32 teacher gates used as reference',
        claim='train diagnostic only; no validation fitting, AEE, hardware timing, or end-to-end prior-method training',
        variants={},
    )
    for name, variant in variants.items():
        weight = np.asarray(variant['weight'], dtype=np.float64)
        bias = np.asarray(variant['bias'], dtype=np.float64)[:, None]
        total = positives = false_pos = false_neg = near_count = near_wrong = 0
        mse = 0.
        worst = 0.
        for sample in saved['samples']:
            x = sample['input'].reshape(10, -1).astype(np.float64)
            teacher_h = sample['membrane'].reshape(10, -1).astype(np.float64)
            teacher_gate = sample['gate'].reshape(10, -1)
            h = weight @ x+bias
            gate = h >= theta
            error = h-teacher_h
            mse += float(np.square(error).sum())
            worst = max(worst, float(np.abs(error).max()))
            near = np.abs(teacher_h-theta) <= 0.1*abs(theta)
            total += gate.size
            positives += int(teacher_gate.sum())
            false_pos += int((gate & ~teacher_gate).sum())
            false_neg += int((~gate & teacher_gate).sum())
            near_count += int(near.sum())
            near_wrong += int(((gate != teacher_gate) & near).sum())
        result['variants'][name] = dict(
            scalar_gates=total, teacher_active_gates=positives,
            teacher_activity=positives/total,
            student_activity=(positives-false_neg+false_pos)/total,
            false_positives=false_pos, false_negatives=false_neg,
            gate_error_fraction=(false_pos+false_neg)/total,
            false_negative_fraction_of_teacher_active=false_neg/positives,
            false_positive_fraction_of_teacher_inactive=false_pos/(total-positives),
            near_threshold_count=near_count,
            near_threshold_error_fraction=near_wrong/near_count,
            sampled_membrane_MSE=mse/total,
            max_abs_membrane_error=worst,
        )
    (directory/'train_gate_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for name, row in result['variants'].items():
        print(name, 'MSE', round(row['sampled_membrane_MSE'], 6),
              'gate_error', round(row['gate_error_fraction'], 6),
              'FN/active', round(row['false_negative_fraction_of_teacher_active'], 6),
              'near_error', round(row['near_threshold_error_fraction'], 6))


if __name__ == '__main__':
    main()
