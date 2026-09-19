"""Process-local FP32 PSN arithmetic, for a separate mixed-precision trial.

This intentionally changes the AMP forward, so it must not be conflated with
the forward-preserving theta/Q backward candidates. Non-AMP FP32 inference is
unchanged. Keep convolution/linear consumers under their existing AMP context.
"""
import torch


def wrap(original):
    def forward(x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return original(x.float())
    return forward


def install(model):
    originals = []
    for module in model.modules():
        if getattr(module, 'threshold_mode', None) == 'official_atlif':
            assert module.output_mode == 'binary'
            originals.append((module, module.forward))
            module.forward = wrap(module.forward)
    return originals


def restore(originals):
    for module, forward in originals:
        module.forward = forward
