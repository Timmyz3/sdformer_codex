"""Train-side full-execution BN calibration for a separate deployment copy.

A single pass measures inputs in SOURCE graph P under its selected BN policy.
Installing all moments changes upstream activations in deployment graph D. This
is a P-to-D conversion, not fixed-point calibration or an AEE guarantee.
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ChannelMoments:
    max_chunk_elements: int = 262144
    count: int = 0
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    calls: int = 0
    chunks: int = 0
    max_chunk_elements_seen: int = 0

    @staticmethod
    def _merge(count, mean, m2, n, next_mean, next_m2):
        if not count:
            return n, next_mean, next_m2
        delta = next_mean - mean
        total = count + n
        return (total, mean + delta * (n / total),
                m2 + next_m2 + delta.square() * (count * n / total))

    def _views(self, x, channel_axis):
        # Split views BEFORE any reshape/contiguous copy or fp64 promotion.
        if x.numel() <= self.max_chunk_elements:
            yield x
            return
        candidates = [i for i, size in enumerate(x.shape)
                      if i != channel_axis and size > 1]
        if not candidates:
            raise ValueError('Chunk bound must cover at least one channel vector')
        axis = max(candidates, key=lambda i: x.shape[i])
        left = x.shape[axis] // 2
        yield from self._views(x.narrow(axis, 0, left), channel_axis)
        yield from self._views(x.narrow(axis, left, x.shape[axis] - left), channel_axis)

    def update(self, x, channel_axis):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError('Expected a floating-point BN input')
        if not isinstance(self.max_chunk_elements, int) or self.max_chunk_elements < 1:
            raise ValueError('max_chunk_elements must be a positive integer')
        if not -x.ndim <= channel_axis < x.ndim:
            raise ValueError('Invalid channel axis')
        channel_axis %= x.ndim
        channels = x.shape[channel_axis]
        if not x.numel():
            raise ValueError('Empty BN input is not a calibration sample')
        if channels > self.max_chunk_elements:
            raise ValueError('Chunk bound must cover at least one channel vector')
        if self.mean is not None and channels != self.mean.numel():
            raise ValueError('BN channel count changed during calibration')
        count, mean, m2, chunks, largest = 0, None, None, 0, 0
        for view in self._views(x.detach(), channel_axis):
            # Any layout copy, promotion and reduction temporary is chunk-sized.
            z = view.movedim(channel_axis, 0).reshape(channels, -1).to(dtype=torch.float64)
            if not torch.isfinite(z).all().item():
                raise ValueError('Non-finite BN calibration activation')
            variance, next_mean = torch.var_mean(z, dim=1, correction=0)
            n = z.shape[1]
            next_m2 = variance * n
            if not (torch.isfinite(next_mean).all() and torch.isfinite(next_m2).all()).item():
                raise ValueError('Non-finite BN calibration moments')
            count, mean, m2 = self._merge(count, mean, m2, n, next_mean.cpu(), next_m2.cpu())
            chunks += 1
            largest = max(largest, view.numel())
        count, mean, m2 = self._merge(self.count, self.mean, self.m2, count, mean, m2)
        if not (torch.isfinite(mean).all() and torch.isfinite(m2).all()).item():
            raise ValueError('Non-finite merged BN calibration moments')
        # Commit only after the whole BN call succeeds. Calls are not chunks.
        self.count, self.mean, self.m2 = count, mean, m2
        self.calls += 1
        self.chunks += chunks
        self.max_chunk_elements_seen = max(self.max_chunk_elements_seen, largest)

    def population_variance(self):
        if self.count < 1:
            raise ValueError('No moments collected')
        return self.m2 / self.count


class BNCalibrationCapture:
    """One authorized target per root forward, with B exactly 1.

    Supply the batch axis of the first positional root tensor (1 for native
    T,B,H,W,C; 0 for N,C,H,W). Select the source BN policy and reset any sample
    state before use. Call begin_sample inside the context before EACH root call.
    Every registered BN must execute in each call. IDs are checked against the
    supplied manifest sets, not inferred from tensor contents: the caller must
    bind each tensor to its ID and execute all branches. A hidden gate that still
    visits every BN cannot be detected by these hooks.
    """
    def __init__(self, model, allowed_training_ids, evaluation_ids, *,
                 sample_batch_axis, max_chunk_elements=262144):
        self.model = model
        self.allowed, self.evaluation = set(allowed_training_ids), set(evaluation_ids)
        if not self.allowed or self.allowed & self.evaluation:
            raise ValueError('Calibration membership empty or overlaps evaluation')
        if not isinstance(sample_batch_axis, int):
            raise ValueError('sample_batch_axis must explicitly identify the root batch axis')
        if not isinstance(max_chunk_elements, int) or max_chunk_elements < 1:
            raise ValueError('max_chunk_elements must be a positive integer')
        self.sample_batch_axis, self.max_chunk_elements = sample_batch_axis, max_chunk_elements
        self.seen, self.handles, self.moments, self.source_policy = [], [], {}, {}
        self.active_sample = None
        self.failed, self.failure_reason = False, None
        self._entered, self._closed, self._in_forward = False, False, False
        self._visited = set()

    def _fail(self, message):
        self.failed = True
        if self.failure_reason is None:
            self.failure_reason = message

    def _reject(self, message):
        self._fail(message)
        raise ValueError(message)

    def begin_sample(self, target_id):
        if not self._entered or self._closed or self.failed:
            self._reject('Calibration context is not active and healthy')
        if self.active_sample is not None or self._in_forward:
            self._reject('Previous target has not completed its root forward')
        if target_id not in self.allowed or target_id in self.evaluation:
            self._reject('Target is not in the permitted training calibration set')
        if target_id in self.seen:
            self._reject('Repeated target needs an explicit new capture')
        self.active_sample = target_id

    def _root_pre(self, module, inputs, kwargs):
        if self.failed or self.active_sample is None or self._in_forward:
            self._reject('Each root forward requires exactly one fresh begin_sample')
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            self._reject('First positional root input must be the target tensor')
        x = inputs[0]
        if not -x.ndim <= self.sample_batch_axis < x.ndim:
            self._reject('sample_batch_axis is outside the root input dimensions')
        if x.shape[self.sample_batch_axis] != 1:
            self._reject('One target ID requires root batch size B=1')
        self._visited = set()
        self._in_forward = True

    def _root_post(self, module, inputs, kwargs, output):
        # always_call=True also executes when forward/pre-hooks raise. A None
        # root result is unsupported; failed calls can never be installed.
        if output is None or not self._in_forward:
            self._fail('Root forward failed or returned no output')
        elif self._visited != set(self.moments):
            self._fail('Root forward did not execute every registered BN')
        if not self.failed:
            self.seen.append(self.active_sample)
        self.active_sample, self._in_forward = None, False

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        if self._entered or self._closed or self.failed:
            self._reject('A capture object can only be entered once')
        # Validate the ENTIRE source before registering any hooks.
        bn_modules = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        if not bn_modules:
            self._reject('No BN modules found')
        for name, module in bn_modules:
            if not isinstance(module, torch.nn.BatchNorm2d):
                self._reject('This v2 contract supports BatchNorm2d only: ' + name)
            if module.num_features > self.max_chunk_elements:
                self._reject('Chunk bound is smaller than BN channel count: ' + name)
            if module.training and module.track_running_stats:
                self._reject('Source BN would mutate running statistics; choose eval policy: ' + name)
        for name, module in bn_modules:
            self.moments[name] = ChannelMoments(max_chunk_elements=self.max_chunk_elements)
            self.source_policy[name] = {
                'training': module.training, 'track_running_stats': module.track_running_stats,
                'has_running_mean': module.running_mean is not None,
                'has_running_var': module.running_var is not None,
                'affine': module.affine, 'eps': module.eps,
                'step_mode': getattr(module, 'step_mode', None)}
        try:
            self.handles.append(self.model.register_forward_pre_hook(
                self._root_pre, prepend=True, with_kwargs=True))
            for name, module in bn_modules:
                def capture(mod, inputs, key=name):
                    try:
                        if self.failed or not self._in_forward or self.active_sample is None:
                            self._reject('BN execution must be inside an authorized root forward')
                        x = inputs[0]
                        if x.ndim == 5 and getattr(mod, 'step_mode', None) == 'm':
                            axis = 2  # NativeSJ T,B,C,H,W: reduce T/B/H/W.
                        elif x.ndim == 4 and getattr(mod, 'step_mode', 's') != 'm':
                            axis = 1  # Ordinary/single-step N,C,H,W.
                        else:
                            self._reject('Unsupported BN input contract: ' + str(tuple(x.shape)))
                        self.moments[key].update(x, axis)
                        self._visited.add(key)
                    except BaseException:
                        self._fail('BN capture failed: ' + key)
                        raise
                self.handles.append(module.register_forward_pre_hook(capture))
            self.handles.append(self.model.register_forward_hook(
                self._root_post, with_kwargs=True, always_call=True))
        except BaseException:
            self._fail('Hook registration failed')
            self._remove_hooks()
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._fail('Calibration context exited with an exception')
        if self.active_sample is not None or self._in_forward:
            self._fail('Calibration target did not complete its root forward')
        self._remove_hooks()
        self.active_sample, self._in_forward, self._closed = None, False, True
        return False

    def install(self, deployment_copy):
        if not self._closed or self.handles:
            raise RuntimeError('End collection before changing the deployment graph')
        if self.failed or not self.seen:
            raise ValueError('Cannot install incomplete/failed calibration: ' + str(self.failure_reason))
        source_objects = {id(module) for module in self.model.modules()}
        if any(id(module) in source_objects for module in deployment_copy.modules()):
            raise ValueError('Deployment must not be source or share source module objects')
        modules = {name: module for name, module in deployment_copy.named_modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        if set(modules) != set(self.moments):
            raise ValueError('Deployment BN names differ from the collected source graph')
        prepared, report = {}, {}
        # Validate and allocate every replacement before mutating ANY target.
        with torch.no_grad():
            for name, moments in self.moments.items():
                module, policy = modules[name], self.source_policy[name]
                if not isinstance(module, torch.nn.BatchNorm2d):
                    raise ValueError('Deployment BN type is unsupported: ' + name)
                if moments.count < 1 or module.num_features != moments.mean.numel():
                    raise ValueError('Missing moments or mismatched BN width: ' + name)
                if (module.affine != policy['affine'] or module.eps != policy['eps'] or
                        getattr(module, 'step_mode', None) != policy['step_mode']):
                    raise ValueError('Deployment BN semantics differ from source: ' + name)
                if module.affine:
                    device, dtype = module.weight.device, module.weight.dtype
                elif module.running_mean is not None:
                    device, dtype = module.running_mean.device, module.running_mean.dtype
                else:
                    raise ValueError('Explicit dtype/device needed for stateless non-affine BN: ' + name)
                mean = moments.mean.to(device=device, dtype=dtype).clone()
                variance = moments.population_variance().to(device=device, dtype=dtype).clone()
                if not (torch.isfinite(mean).all() and torch.isfinite(variance).all()).item():
                    raise ValueError('Non-finite installed BN statistics: ' + name)
                if (variance < 0).any().item():
                    raise ValueError('Negative installed BN variance: ' + name)
                prepared[name] = (mean, variance, torch.tensor(moments.calls, dtype=torch.long, device=device))
                report[name] = {
                    'values_per_channel': moments.count, 'calls': moments.calls,
                    'chunks': moments.chunks, 'max_chunk_elements_seen': moments.max_chunk_elements_seen,
                    'variance': 'pooled_population', 'eps': module.eps,
                    'source_bn_policy': dict(policy)}
        old_bn = {name: (module.running_mean, module.running_var, module.num_batches_tracked,
                         module.track_running_stats) for name, module in modules.items()}
        old_modes = [(module, module.training) for module in deployment_copy.modules()]
        try:
            for name, module in modules.items():
                module.running_mean, module.running_var, module.num_batches_tracked = prepared[name]
                module.track_running_stats = True
            deployment_copy.eval()
        except BaseException:
            for name, module in modules.items():
                old = old_bn[name]
                module._buffers['running_mean'], module._buffers['running_var'], module._buffers['num_batches_tracked'] = old[:3]
                module.track_running_stats = old[3]
            for module, mode in old_modes:
                module.training = mode
            raise
        return {
            'target_ids': list(self.seen), 'sample_batch_axis': self.sample_batch_axis,
            'root_batch_size': 1, 'max_chunk_elements': self.max_chunk_elements,
            'layers': report,
            'meaning': 'single-pass source-P moments installed in deployment D; full-execution D-P AEE required'}
