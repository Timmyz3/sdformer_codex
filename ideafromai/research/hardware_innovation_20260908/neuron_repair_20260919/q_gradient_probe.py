"""Paired network backward: restore Q gradients without changing the forward."""
import json
import numpy as np
import torch
from binary_surrogate import install as install_theta
from q_support_proxy import install, restore
from recovery_probe import prepare


def run(model, optimizer, loss_function, dataset, config, device, transform, args):
    from spikingjelly.activation_based import functional
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
    modules = {n: m for n, m in model.named_modules() if isinstance(m, ATLIFTernaryPSN)}
    assert all(m.output_mode == 'binary' and m.sp == 0 for m in modules.values())
    assert install_theta(model, 'normalized') == len(modules)
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.train()
    report = dict(scope='Paired forward/backward only; no parameter update or AEE',
                  theta_rule='normalized_sum', checkpoint=str(args.checkpoint), batches=[])
    batch_size = config['loader']['batch_size']
    starts = np.linspace(0, len(dataset) - batch_size, args.batches, dtype=int)
    for b, start in enumerate(starts):
        x, gt, mask = prepare(dataset, range(start, start + batch_size), device, transform, 1900 + b)
        batch = dict(files=[str(dataset.files.iloc[start + k, 0]) for k in range(batch_size)], arms=[])
        for bridge in [False, True]:
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            functional.set_step_mode(model, config['data']['step_mode'])
            with torch.no_grad():
                for n, value in model.named_buffers():
                    value.copy_(buffers[n])
            torch.manual_seed(2900 + b); torch.cuda.manual_seed_all(2900 + b)
            if bridge:
                original, counters = install()
            with torch.cuda.amp.autocast():
                outputs = model(x)['flow']
                loss = loss_function(outputs, gt, mask, gamma=config['loss']['gamma'])
            if not bridge:
                reference = [v.detach().clone() for v in outputs]
            differences = [int((v.detach() != ref).sum()) for v, ref in zip(outputs, reference)]
            (loss * args.initial_scale).backward()
            gradients = {n: float(m.thresh.grad.detach()) / args.initial_scale
                         if m.thresh.grad is not None else None for n, m in modules.items()}
            row = dict(q_bridge=bridge, loss=float(loss.detach()),
                       forward_differences_per_head=differences,
                       score_calls=counters['calls'] if bridge else None,
                       nonzero_theta=sum(g not in (None, 0.) for g in gradients.values()),
                       q_nonzero=sum(g not in (None, 0.) for n, g in gradients.items() if '.sn_q.' in n),
                       q_firing={n: m.r for n, m in modules.items() if '.sn_q.' in n},
                       all_finite=all(g is None or __import__('math').isfinite(g) for g in gradients.values()),
                       theta_gradients=gradients)
            batch['arms'].append(row)
            print('[q-probe]', b, {k: v for k, v in row.items() if k not in {'theta_gradients', 'q_firing'}}, flush=True)
            if bridge:
                restore(original)
            del outputs, loss
        del reference, x, gt, mask
        report['batches'].append(batch)
        (args.output / 'Q_NETWORK.json').write_text(json.dumps(report, indent=2) + '\n')
    optimizer.zero_grad(set_to_none=True)
    report['complete'] = True
    (args.output / 'Q_NETWORK.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.paired_valid_checkpoint:
        from paired_validation import run as paired_validate
        paired_validate(model, config, device, args)
