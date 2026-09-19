"""Compare with the actual historical official-copy source, including BPTT."""
import ast
import json
from pathlib import Path
import torch
from pure_atlif import RecurrentATLIF


def main():
    repo = next(p for p in Path(__file__).resolve().parents if (p / 'SDformer').is_dir())
    source = repo / 'SDformer/neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py'
    tree = ast.parse(source.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in {'zif_backward', 'Surrogate', 'ATLIF'}]
    namespace = dict(torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source), 'exec'), namespace)
    torch.manual_seed(1909)
    cases = []
    for T in [2, 10]:
        for retention in [0.9, 1.0]:
            ref = namespace['ATLIF'](T, thresh=1.125, tau=retention)
            candidate = RecurrentATLIF(T, ref.thresh, namespace['Surrogate'].apply, retention)
            a = torch.randn(T, 2, 3, 5, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            out_ref, out = ref(a), candidate(b)
            grad = torch.randn_like(out)
            (out_ref * grad).sum().backward()
            (out * grad).sum().backward()
            assert torch.equal(out_ref, out)
            assert torch.equal(a.grad, b.grad)
            assert torch.equal(ref.thresh.grad, candidate.thresh.grad)
            with torch.no_grad():
                assert torch.equal(candidate(b), out)
                changed = b.detach().clone(); changed[T // 2:] += 100
                assert torch.equal(candidate(changed)[:T // 2], out[:T // 2])
            assert list(candidate.state_dict()) == ['thresh']
            cases.append(dict(T=T, retention=retention, forward_equal=True,
                              input_gradient_equal=True, theta_gradient_equal=True,
                              reset_between_calls=True, causal_on_supplied_axis=True))
    path = Path(__file__).with_name('RECURRENCE_TESTS.json')
    path.write_text(json.dumps(dict(reference=str(source.relative_to(repo)), cases=cases), indent=2) + '\n')
    print('PASS: four exact forward/BPTT comparisons; reset, causality, no PSN parameters')


if __name__ == '__main__':
    main()
