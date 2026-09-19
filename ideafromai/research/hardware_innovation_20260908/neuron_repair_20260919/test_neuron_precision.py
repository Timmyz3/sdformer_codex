"""Small PSN-linear/threshold precision example; no network or AEE claim."""
import json
from pathlib import Path
import torch
from neuron_precision import wrap


def main():
    theta = torch.tensor(1.00002)
    x = torch.tensor([[.99999, 1., 1.00001, 1.00003]])
    weight = torch.eye(1)
    observed = {}

    def forward(value):
        h = torch.mm(weight, value)
        observed['h_dtype'] = str(h.dtype)
        return (h >= theta).float() * theta

    precise = wrap(forward)
    reference = forward(x)
    assert torch.equal(precise(x), reference)
    rows = []
    for dtype in [torch.float16, torch.bfloat16]:
        with torch.autocast(device_type='cpu', dtype=dtype):
            ordinary = forward(x)
            ordinary_dtype = observed['h_dtype']
            fixed = precise(x)
            fixed_dtype = observed['h_dtype']
        assert torch.equal(fixed, reference)
        rows.append(dict(amp_dtype=str(dtype), ordinary_h_dtype=ordinary_dtype,
                         precise_h_dtype=fixed_dtype,
                         ordinary_differences=int((ordinary != reference).sum()),
                         precise_differences=int((fixed != reference).sum())))
    result = dict(scope='Constructed PSN precision example only; full-model AMP still affects preceding layers', rows=rows)
    Path('NEURON_PRECISION_TESTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
