"""Check forward preservation and the explicit Q-only surrogate Jacobian."""
import json
import torch
from q_support_proxy import QSupportBridge


def main():
    torch.manual_seed(71)
    rows = []
    for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
        q = torch.randint(0, 2, (2, 3, 450, 32)).to(dtype).requires_grad_()
        k = torch.randint(0, 2, q.shape).to(dtype).requires_grad_()
        alpha, norm = .02, 1 / 32
        base = (((q > 0) & (k > 0)).to(dtype) + alpha * ((q == 0) & (k == 0)).to(dtype)).sum(-1, keepdim=True) * norm
        # Stand-in for the original score's live K branch. Its precise value is
        # irrelevant to verifying that the bridge leaves its gradient unchanged.
        base = base + .125 * k.sum(-1, keepdim=True)
        y = QSupportBridge.apply(base, q, k, alpha, norm)
        upstream = torch.randn_like(y)
        base_k = torch.autograd.grad(base, k, upstream, retain_graph=True)[0]
        dq, dk = torch.autograd.grad(y, (q, k), upstream)
        expected_q = torch.empty_like(q, dtype=torch.float64)
        active = k.detach() > 0
        expanded = upstream.double().expand_as(q)
        expected_q[active] = expanded[active] * norm
        expected_q[~active] = expanded[~active] * (-alpha) * norm
        assert torch.equal(y, base) and torch.equal(dk, base_k)
        assert torch.equal(dq[active], (upstream.expand_as(q)[active] * norm))
        torch.testing.assert_close(dq.double(), expected_q,
                                   rtol=3 * torch.finfo(dtype).eps,
                                   atol=3 * torch.finfo(dtype).tiny * torch.finfo(dtype).eps)
        rows.append(dict(dtype=str(dtype), forward_exact=True, original_k_gradient_exact=True,
                         q_gradient_matches_fp64_reference=True,
                         q_gradient_max_abs_error=float((dq.double() - expected_q).abs().max()),
                         active_k_coefficient_exactly_one=True,
                         nonzero_q_gradient=int((dq != 0).sum())))
    out = dict(scope='Synthetic Jacobian/forward tests, not network accuracy', rows=rows)
    __import__('pathlib').Path('Q_PROXY_TESTS.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
