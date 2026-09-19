"""Small paired whole-network optimizer replay, no checkpoint publication."""
import gc
import json
import random
import time
import numpy as np
import torch
from binary_surrogate import install


def run(model, loss_function, dataset, config, device, transform, args, initial_buffers, scale):
    from spikingjelly.activation_based import functional
    from models.STSwinNet_SNN.h28_optimizer import build_optimizer
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN

    modules = {name: m for name, m in model.named_modules() if isinstance(m, ATLIFTernaryPSN)}
    original_acts = {name: m.act for name, m in modules.items()}
    original_state = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
    for name, v in initial_buffers.items():
        original_state[name] = v.detach().cpu().clone()
    batch_size = config['loader']['batch_size']
    starts = np.linspace(0, len(dataset) - batch_size, 2, dtype=int).tolist()
    inputs = []
    for k, start in enumerate(starts):
        random.seed(1900 + k); np.random.seed(1900 + k); torch.manual_seed(1900 + k)
        d = [dataset[i] for i in range(start, start + batch_size)]
        chunk, mask, label = [torch.stack([row[j] for row in d]).to(device) for j in range(3)]
        chunk, label, mask = chunk.float(), label.float(), mask.unsqueeze(1)
        if transform is not None:
            chunk, label, mask = transform((chunk, label, mask.float()))
        chunk = torch.stack((torch.relu(chunk), torch.relu(-chunk)), dim=2)
        active = chunk != 0
        lo, hi = chunk[active].min(), chunk[active].max()
        if lo != hi:
            chunk[active] = (chunk[active] - lo) / (hi - lo)
        inputs.append((chunk, label, mask))
    report = dict(scope='Paired short whole-network training replay, two fixed train batches; no AEE',
                  successful_updates_per_arm=args.pilot_steps, initial_loss_scale=scale,
                  input_gradient='unchanged H9 native kernel', eta=0, arms=[], complete=False)
    theta_initial = {name: float(m.thresh.detach()) for name, m in modules.items()}
    first_output = None
    for rule, order in [('official', 'old'), ('official', 'fixed'), ('margin', 'fixed'), ('normalized', 'fixed')]:
        functional.reset_net(model)
        model.load_state_dict(original_state, strict=True)
        for name, m in modules.items():
            m.act = original_acts[name]
            m.update_value = 0.0
            m.act_value = 0.0
        if rule != 'official':
            installed = install(model, rule)
            assert installed == len(modules)
        model.train()
        optimizer = build_optimizer(model, config)
        scaler = torch.cuda.amp.GradScaler(init_scale=scale)
        arm = dict(rule=rule, clip_order=order, rows=[], initial_forward_difference=None)
        report['arms'].append(arm)
        step = attempts = 0
        while step < args.pilot_steps:
            attempts += 1
            if attempts > args.pilot_steps + 12:
                raise RuntimeError('Repeated nonfinite updates in short replay')
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode'])
            torch.manual_seed(2900 + step); torch.cuda.manual_seed_all(2900 + step)
            chunk, label, mask = inputs[step % len(inputs)]
            started = time.monotonic()
            with torch.cuda.amp.autocast():
                out = model(chunk)['flow']
                loss = loss_function(out, label, mask, gamma=config['loss']['gamma'])
            if step == 0 and attempts == 1:
                current = out[-1].detach().cpu()
                if first_output is None:
                    first_output = current.clone()
                arm['initial_forward_difference'] = dict(
                    different_elements=int((current != first_output).sum()),
                    max_abs=float((current - first_output).abs().max()))
            theta_before = {name: float(m.thresh.detach()) for name, m in modules.items()}
            used_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            if order == 'fixed':
                scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config['loss']['clip_grad']))
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < used_scale
            after = {name: float(m.thresh.detach()) for name, m in modules.items()}
            row = dict(step=step, loss=float(loss.detach()), scale=used_scale,
                       preclip_norm=norm, skipped=overflow,
                       changed_this_step=sum(after[n] != theta_before[n] for n in after),
                       changed_from_start=sum(after[n] != theta_initial[n] for n in after),
                       max_abs_delta=max(abs(after[n] - theta_initial[n]) for n in after),
                       theta=after, seconds=time.monotonic() - started)
            arm['rows'].append(row)
            print('[neuron-pilot]', rule, order, {k: row[k] for k in row if k != 'theta'}, flush=True)
            (args.output / 'SHORT_REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')
            if not overflow:
                step += 1
            del out, loss
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler
        gc.collect()
    for name, m in modules.items():
        m.act = original_acts[name]
    model.load_state_dict(original_state, strict=True)
    report['complete'] = True
    (args.output / 'SHORT_REPLAY.json').write_text(json.dumps(report, indent=2) + '\n')
    print('[neuron-pilot] COMPLETE; no checkpoint saved', flush=True)
