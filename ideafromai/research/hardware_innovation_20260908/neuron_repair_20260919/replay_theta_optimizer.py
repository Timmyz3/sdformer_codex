"""One alternative explanation: inherited gradient, smaller theta-only epsilon."""
from pathlib import Path
import json
import torch

root = Path(__file__).resolve().parent
data = json.loads((root / 'amp_probe/DIAGNOSIS.json').read_text())
rows = []
for batch in data['batches']:
    arm = next(a for a in batch['arms'] if a['rule'] == 'official' and a['clip_order'] == 'unscale_then_clip')
    params = {name: torch.nn.Parameter(torch.tensor(v['before'], dtype=torch.float32))
              for name, v in arm['neurons'].items()}
    groups = []
    for name, p in params.items():
        v = arm['neurons'][name]
        groups.append(dict(params=[p], lr=v['lr'], eps=1e-12, weight_decay=0.0))
        if v['gradient_after_clip'] is not None:
            p.grad = torch.tensor(v['gradient_after_clip'], dtype=torch.float32)
    opt = torch.optim.AdamW(groups, foreach=False)
    opt.step()
    rows.append(dict(files=batch['files'], source_rule='official', clip='corrected',
                     original_epsilon=1e-8, experimental_theta_epsilon=1e-12,
                     original_changed=arm['changed'],
                     changed=sum(float(p.detach()) != arm['neurons'][n]['before'] for n, p in params.items()),
                     max_abs_delta=max(abs(float(p.detach()) - arm['neurons'][n]['before']) for n, p in params.items())))
result = dict(scope='Exact scalar AdamW first-step replay of observed gradients; not network training or AEE',
              rows=rows)
(root / 'EPSILON_COUNTERFACTUAL.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
