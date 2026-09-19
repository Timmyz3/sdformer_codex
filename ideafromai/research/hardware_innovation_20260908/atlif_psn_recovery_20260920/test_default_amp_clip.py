"""Numerical regression for H9's default clipping with AMP and accumulation.

CPU GradScaler exercises real scaling/unscaling without occupying the A800.
Also compile the complete patched upstream training source and legacy wrapper.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import types
import torch

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'SDformer').is_dir())
spec = importlib.util.spec_from_file_location('h9_entry', REPO / 'SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py')
h9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h9)
sys.path.insert(0, str(HERE.parent / 'neuron_repair_20260919'))
from train_binary_candidate import patch_training_source

SOURCE = '''def train(model, optimizer, scaler, batches, accumulation, clipping):
    config = {"loss": {"clip_grad": clipping}}
    num_acc_steps = accumulation
    train_dataloader = batches
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1):
        for sample, (x, y) in enumerate(train_dataloader):
            loss = (model(x) - y).square().mean() / num_acc_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
            if ((sample + 1) % num_acc_steps == 0) or (sample + 1 == len(train_dataloader)):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
'''


def fixture():
    m = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[0.75, -0.5]]))
    return m, torch.optim.SGD(m.parameters(), lr=0.1)


def check_manual_update_skip():
    # Execute the actual H9 scaler-step patch, with only the external
    # warmup/homeostatic callbacks stubbed. Scaling and optimizer are real.
    neuron = types.SimpleNamespace(update_value=0., act_value=0., threshold=1.)
    seen = []
    opt_module = types.ModuleType('models.STSwinNet_SNN.h28_optimizer')
    opt_module.apply_lr_warmup = lambda *a: None
    opt_module.lr_warmup_factor = lambda *a: None
    opt_module.freeze_threshold_gradients = lambda *a: 0
    neuron_module = types.ModuleType('models.STSwinNet_SNN.atlif_ternary_psn')

    def update(model, lr, config):
        seen.append(neuron.update_value)
        neuron.threshold += neuron.update_value
        neuron.update_value = 0.
        return {'manual_update': 1}

    neuron_module.threshold_update = update
    neuron_module.iter_atlif_ternary_psn = lambda model: [('', neuron)]
    stubs = {'models': types.ModuleType('models'),
             'models.STSwinNet_SNN': types.ModuleType('models.STSwinNet_SNN'),
             opt_module.__name__: opt_module, neuron_module.__name__: neuron_module}
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        source = 'def finish(model, optimizer, scaler, config, epoch, sample, train_dataloader):\n'
        source += textwrap.indent(textwrap.dedent(h9.SCALER_STEP_PATCH), '    ')
        source += '\n    return h8_update_stats\n'
        scope = {}
        exec(compile(source, '<actual H9 scaler step>', 'exec'), scope)
        model, optimizer = fixture()
        scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=1)
        rows = []
        for i, (overflow, pending) in enumerate([(False, 2.), (True, 3.), (False, 5.)]):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.tensor([[2., -1.]])).square().mean()
            scaler.scale(loss).backward()
            if overflow:
                model.weight.grad.fill_(float('inf'))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .7)
            neuron.update_value += pending
            neuron.act_value = loss
            weight_before = model.weight.detach().clone()
            scale_before = scaler.get_scale()
            stats = scope['finish'](model, optimizer, scaler, {}, 0, i, [0, 1, 2])
            if overflow:
                assert torch.equal(weight_before, model.weight)
                assert neuron.threshold == 3. and neuron.act_value == 0.
                assert stats == {'amp_skipped': 1}
            assert neuron.update_value == 0.
            rows.append(dict(overflow=overflow, scale_before=scale_before, scale_after=scaler.get_scale(),
                             threshold=neuron.threshold, pending=neuron.update_value))
        assert seen == [2., 5.] and neuron.threshold == 8.
        return dict(real_gradscaler=True, actual_h9_step_patch=True,
                    growth_success_allowed=True, overflow_pending_discarded=True, rows=rows)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main():
    tree_path = REPO / 'SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py'
    full = h9._patch_source(tree_path.read_text(), tree_path)
    compile(full, str(tree_path), 'exec')
    assert full.count('scaler.unscale_(optimizer)') == 1
    attrs = [n.func.attr for n in ast.walk(ast.parse(full)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert attrs.count('clip_grad_norm_') == 1
    for rule in ['official', 'normalized']:
        wrapped = patch_training_source(full, rule, theta_epsilon=1e-12, neuron_fp32=True)
        compile(wrapped, '<wrapped H9>', 'exec')
        assert wrapped.count('scaler.unscale_(optimizer)') == 1

    namespaces = []
    for source in [SOURCE, h9._patch_amp_clipping(SOURCE)]:
        scope = {'torch': torch}
        exec(compile(source, '<scalar regression>', 'exec'), scope)
        namespaces.append(scope['train'])
    batches = [(torch.tensor([[2., -1.], [0.5, 3.]]), torch.tensor([[v], [-v]]))
               for v in [2., -1., 0.5, 3., -2.]]
    results = []
    for accumulation, amp, clipping in [(1, True, .7), (2, True, .7), (3, True, .7),
                                        (2, False, .7), (2, True, None)]:
        reference, opt = fixture()
        for start in range(0, len(batches), accumulation):
            opt.zero_grad(set_to_none=True)
            sum((reference(x) - y).square().mean() / accumulation
                for x, y in batches[start:start + accumulation]).backward()
            if clipping is not None:
                torch.nn.utils.clip_grad_norm_(reference.parameters(), clipping)
            opt.step()
        errors = []
        for train in namespaces:
            model, optimizer = fixture()
            scaler = torch.amp.GradScaler('cpu', init_scale=65536.) if amp else None
            train(model, optimizer, scaler, batches, accumulation, clipping)
            errors.append(float((model.weight.detach() - reference.weight.detach()).abs().max()))
        assert errors[1] < 1e-7
        if clipping is not None and (amp or accumulation > 1):
            assert errors[0] > 1e-5
        results.append(dict(accumulation=accumulation, amp=amp, clipping=clipping,
                            old_max_error=errors[0], fixed_max_error=errors[1]))
    report = dict(full_training_source_compiles=True, wrapper_compatibility=True,
                  unscale_calls_in_default_and_wrapped=1, cases=results,
                  manual_update_skip=check_manual_update_skip(), torch=torch.__version__)
    HERE.joinpath('DEFAULT_AMP_CLIP_TESTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
