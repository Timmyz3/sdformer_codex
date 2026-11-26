"""Functional checks for exact selection, amplitude, mapping and BN limits."""
import itertools
import json
from pathlib import Path
import torch
from adaptive_spike_probe import gate_probe, legacy_qk, mapping_statistics, stable_select


def main():
    cases = 0
    for n in range(1, 6):
        scores = torch.tensor(list(itertools.product(range(3), repeat=n)))
        for keep in range(n+1):
            actual = stable_select(scores, keep)
            for row, mask in zip(scores.tolist(), actual.tolist()):
                expected = sorted(range(n), key=lambda i: (-row[i], i))[:keep]
                assert mask == [i in expected for i in range(n)]
            cases += len(scores)
    torch.manual_seed(20260922)
    q = (torch.rand(2, 3, 3, 3, 8) < .2).float() * .875
    k = (torch.rand_like(q) < .3).float() * 1.125
    direct = gate_probe(q, k, 2, 'q_presence_probe')[0]
    assert torch.equal(direct, gate_probe(q, k, 2, 'saf_fixed_half_probe', 1.0)[0])
    assert torch.count_nonzero(gate_probe(q, k, 2, 'saf_fixed_half_probe', 0.0)[0]) == 0
    assert torch.count_nonzero(gate_probe(torch.zeros_like(q), k, 2, 'saf_fixed_half_probe')[0]) == 0
    for mode in ['q_presence_probe', 'saf_fixed_half_probe']:
        output, counts, selected, gate = gate_probe(q, k, 2, mode)
        assert bool(torch.all((output == 0) | (output == 1.125)))
        assert bool(torch.all(output[~gate] == 0))
        if mode.startswith('saf'):
            assert bool(torch.all(selected.sum(-1) == counts.shape[-1]//2))
        k_token = legacy_qk(q, k, 2)[1]
        # Independent source-coordinate reconstruction, retaining every lane.
        reference = torch.zeros_like(k)
        ids = torch.arange(k.numel()).reshape_as(k).reshape_as(k_token)
        reference.flatten()[ids[gate].flatten()] = k.flatten()[ids[gate].flatten()]
        assert torch.equal(output, reference.reshape_as(output))
        assert mapping_statistics(torch.zeros_like(gate), q, 2)['postPSN_K_output_position_requirement'] == 0
        assert mapping_statistics(torch.ones_like(gate), q, 2)['postPSN_K_output_position_requirement'] == 1
    q.requires_grad_(); k.requires_grad_()
    gate_probe(q, k, 2, 'saf_fixed_half_probe')[0].sum().backward()
    assert q.grad is None  # Expected: no STE or trainable guide in this probe.
    assert k.grad is not None and bool(torch.all(torch.isfinite(k.grad)))
    # A selected output's dynamic-BN result changes with an unselected input.
    def bn(v):
        return (v - v.mean()) / torch.sqrt(v.var(unbiased=False) + 1.e-5)
    assert not torch.isclose(bn(torch.tensor([1., 2., 3., 4.]))[0],
                             bn(torch.tensor([1., 2., 3., 40.]))[0])
    result = dict(status='PASS', stable_selection_cases=cases,
        checked=['stable ties, k=0/N', 'actual theta_K retained', 'hard zero mask',
                 'k=N equals A1, k=0 and zero Q', 'legacy lane mapping',
                 'expected absent Q selection gradient; finite K gradient',
                 'live BN prevents selected-only K'],
        scope='CPU inference reference; no SAF SGM training, AEE or RTL performance')
    Path(__file__).with_name('FUNCTIONAL_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
