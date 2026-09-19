"""Small CPU-only counterchecks; no inference, training, or torch dependency."""
import json
import math
import struct
from pathlib import Path

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

g, eps, lr = 5.31e-10, 1e-8, 1.25e-6
step = lr * g / (abs(g) + eps)
checks = {
    'fp32_at_one': {
        'next_above_gap': 2.0**-23, 'next_below_gap': 2.0**-24,
        'constant_grad_first_adam_step': step,
        'round_one_minus_step': f32(1.0-step),
        'round_one_plus_step': f32(1.0+step),
        'note': 'The cited median gradient is NOT below the half-ULP on its own.',
    },
    'BN_two_equal_size_batches': {
        'batches': [[0, 0], [10, 10]],
        'mean_batch_unbiased_var': 0.0,
        'pooled_population_var': 25.0,
        'pooled_unbiased_var': 100.0/3.0,
    },
    'H12_margin_kernel_convention': {
        'theta': 1, 'negative_scale': 4, 'input': -5,
        'positive_tmp': 0, 'negative_tmp': .75, 'ternary_code': -1,
        'chain_grad_theta': -.25, 'H12_grad_theta': -1.75,
        'chain_grad_input': .75/4, 'H12_grad_input': .75,
        'note': 'Chosen delta kernels p/theta and n/(r*theta); not a unique surrogate definition.',
    },
    'Q_functional_counterexample': {
        'mode': 'h60 support score; K equal nonzero for two tokens; head_dim=32; alpha0=.02',
        'before_gate': [1, 1],
        'after_flipping_one_Q_bit': [1, 2**(-1/32)],
        'note': 'Boolean score makes Q autograd zero while forward still depends on Q.',
    },
}
assert checks['fp32_at_one']['round_one_minus_step'] != 1.0
assert checks['fp32_at_one']['round_one_plus_step'] != 1.0
Path(__file__).with_suffix('.json').write_text(json.dumps(checks, indent=2)+'\n')
print(json.dumps(checks, indent=2))
