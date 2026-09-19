"""Meaningful forward/STE-graph/AMP promotion tests for the isolated candidates."""
from pathlib import Path
import json
import torch
from binary_surrogate import BinarySurrogate


def integral(u):
    return torch.where(u < -1, 0.0, torch.where(u < 0, .5 * (u + 1).square(),
           torch.where(u < 1, 1 - .5 * (1 - u).square(), 1.0)))


comparisons = 0
maximum_error = 0.0
for rule in ['margin', 'normalized']:
    for value in [.2, 1.0, 3.2]:
        for sign in [-1.0, 1.0]:
            x = torch.tensor([-.5, 0, .5, .9, 1, 1.1, 1.5, 2, 2.5], dtype=torch.float64) * value
            h, theta = x.clone().requires_grad_(), torch.tensor(value, dtype=torch.float64, requires_grad=True)
            y, update = BinarySurrogate.apply(h, theta, .008, rule)
            native_gate = (h >= theta).float()
            expected_update = (.008 * (1 - ((h - theta) / theta).abs()).clamp_min(0) * native_gate).sum(0).mean().item()
            assert torch.equal(y, native_gate * theta) and update == expected_update
            grads = torch.autograd.grad((y * sign).sum(), (h, theta))
            h2, th2 = x.clone().requires_grad_(), theta.detach().clone().requires_grad_()
            u = h2 / th2 - 1 if rule == 'normalized' else (h2 - th2) / th2.detach()
            f = integral(u)
            smooth_extension = th2 * (native_gate.detach() + f - f.detach())
            refs = torch.autograd.grad((smooth_extension * sign).sum(), (h2, th2))
            for actual, expected in zip(grads, refs):
                err = float((actual - expected).abs().max())
                maximum_error = max(maximum_error, err)
                assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12), (rule, actual, expected)
            comparisons += y.numel()

promotion_cases = []
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    for value in [1.0001, .9999, 1.0]:
        h = torch.tensor([.9995, 1.0, 1.0005], dtype=dtype, requires_grad=True)
        theta = torch.tensor(value, dtype=torch.float32, requires_grad=True)
        gate = h >= theta
        for rule in ['margin', 'normalized']:
            y, _ = BinarySurrogate.apply(h, theta, 0., rule)
            assert torch.equal(y, gate.float() * theta)
            gh, gt = torch.autograd.grad(y.sum(), (h, theta))
            assert torch.isfinite(gh).all() and torch.isfinite(gt)
        promotion_cases.append(dict(h_dtype=str(dtype), theta=value,
                                    native_vs_promoted_gate_differences=int((gate != (h.float() >= theta)).sum())))

for rule in ['margin', 'normalized']:
    observed = []
    for copies in [1, 8]:
        h = torch.tensor([.5, 1.3, 2.5], dtype=torch.float32).repeat(copies).requires_grad_()
        th = torch.tensor(1., requires_grad=True)
        y, _ = BinarySurrogate.apply(h, th, 0., rule)
        observed.append(float(torch.autograd.grad(y.mean(), th)[0]))
    assert abs(observed[0] - observed[1]) < 1e-7

result = dict(status='PASS', analytic_comparisons=comparisons, max_vjp_error=maximum_error,
              promotion_cases=promotion_cases, shared_mean_loss_replication_invariant=True,
              scope='CPU operator tests; no network AEE or long training')
Path(__file__).with_name('SURROGATE_TESTS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
