"""Isolated binary ATLIF candidates with an unchanged forward.

Rules are explicit surrogate choices, not derivatives of a hard step.
The forward comparison retains original PyTorch scalar-promotion behavior.
"""
import torch


class BinarySurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, theta, sparsity_eta, rule):
        gate = h >= theta
        out = gate.float()
        updates = (sparsity_eta * (1.0 - ((h - theta) / theta).abs()).clamp_min(0) * out).sum(0).mean().item()
        ctx.save_for_backward(h, theta, gate)
        ctx.rule = rule
        return out * theta, updates

    @staticmethod
    def backward(ctx, grad_output, ignored_updates):
        h, theta, gate = ctx.saved_tensors
        dtype = torch.float64 if torch.float64 in (h.dtype, theta.dtype, grad_output.dtype) else torch.float32
        x, th, a = h.to(dtype), theta.to(dtype), grad_output.to(dtype)
        p = (1.0 - ((x - th) / th).abs()).clamp_min(0)
        if ctx.rule == 'normalized':
            dtheta = gate.to(dtype) - (x / th) * p
        elif ctx.rule == 'margin':
            dtheta = gate.to(dtype) - p
        else:
            raise ValueError(ctx.rule)
        # Hold the inherited input-gradient path fixed for a theta-only trial.
        # Its kernel uses the saved operand dtypes, exactly like H9 official.
        native_p = (1.0 - ((h - theta) / theta).abs()).clamp_min(0)
        return (grad_output * native_p).to(h.dtype), (a * dtheta).sum().reshape_as(theta).to(theta.dtype), None, None


def install(model, rule):
    """Change only explicitly binary official ATLIF modules in this process."""
    count = 0
    for module in model.modules():
        if getattr(module, 'threshold_mode', None) != 'official_atlif':
            continue
        if module.output_mode != 'binary' or module.thresh.numel() != 1:
            raise ValueError('This candidate supports scalar-threshold binary ATLIF only')
        module.act = lambda h, theta, eta, rule=rule: BinarySurrogate.apply(h, theta, eta, rule)
        count += 1
    return count
