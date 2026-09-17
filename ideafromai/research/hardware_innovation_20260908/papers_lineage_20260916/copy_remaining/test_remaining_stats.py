"""Drive shipped remaining_ops on tiny CPU tensors — not a reimplementation."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import remaining_ops as mod


def test_aac_baseline_all_zero_S():
    S = torch.zeros(8, 4)
    W = torch.randn(3, 4)
    st = mod.word_stats(S, W)
    assert st['nnz_S'] == 0
    assert st['baseline_aac_word_adds'] == 0


def test_word_add_baseline_is_nnz_times_Cout():
    S = torch.ones(16, 8)
    W = torch.randn(4, 8) * 0.01
    st = mod.word_stats(S, W)
    assert st['baseline_aac_word_adds'] == 16 * 8 * 4
    assert st['extra_bitwave_vs_word_add_cycles'] == 1.0 - st['bit_ops'] / st['baseline_aac_word_adds']
    # 7-serial is informational, not the KEEP bar
    assert abs(st['extra_bitwave_vs_7serial'] - (1.0 - st['bit_ops'] / (st['baseline_aac_word_adds'] * 7))) < 1e-9


def test_bitwave_vs_word_negative_when_bits_cost_more():
    torch.manual_seed(0)
    S = torch.ones(4, 4)
    W = torch.ones(2, 4)
    st = mod.word_stats(S, W)
    # dense q8 weights have many 1-bits; bit-serial does more cycles than 1-cycle word adds
    assert st['extra_bitwave_vs_word_add_cycles'] < 0.0
    assert mod.keep_kill(st['extra_bitwave_vs_word_add_cycles'] - st['inspect_tax_runtime']) == 'KILL'


def test_prefix_nnz_extra_is_dropped_nnz_fraction():
    S = torch.zeros(4, 4)
    S[:, 0] = 1
    S[:2, 1] = 1
    extra = mod.prefix_nnz_extra(S, keep_idx=torch.tensor([0]))
    # nnz=4+2=6, kept channel0 = 4, dropped=2 -> 2/6
    assert abs(extra - 2 / 6) < 1e-9


def test_k_keep_for_nnz_extra_max_k():
    S = torch.ones(10, 4)
    order = torch.arange(4)
    k, extra = mod.k_keep_for_nnz_extra(S, order, target=0.15)
    # uniform ones: extra = 1 - k/4; max k with 1-k/4 >= 0.15 is k=3 (extra=0.25)
    assert k == 3
    assert abs(extra - 0.25) < 1e-9


def test_sna_pea_is_skipped_over_total_all_Cout():
    S = torch.ones(4, 3)
    W = torch.zeros(2, 3)
    W[0] = torch.tensor([10.0, 0.1, 0.1])
    W[1] = torch.tensor([10.0, 0.1, 0.1])
    y_th = torch.tensor([1.0, 1.0])
    extra, ns, skipped, total = mod.sna_pea_extra(S, W, y_th, max_tokens=4)
    assert ns == 4
    assert total == 4 * 3 * 2  # all Cout
    assert 0.0 <= extra <= 1.0
    assert abs(extra - skipped / total) < 1e-9


def test_psn_scrooge_identity_easy_decide():
    T = 4
    A = torch.eye(T)
    Y = torch.zeros(T, 1, 1)
    Y[0, 0, 0] = 10.0
    tau = torch.ones(T, 1) * 0.5
    extra = mod.psn_scrooge_extra(Y, A, tau, bound='l1_maxabs')
    # each of 4 outputs decides after 1 term (rest ||A_rest||_1=0), skip 3/4
    assert abs(extra - 0.75) < 1e-6
    assert abs(mod.psn_scrooge_inspect_tax(T, 'l1_maxabs') - 0.25) < 1e-9
    assert mod.psn_scrooge_inspect_tax(T, 'tot_oracle') == 1.0


def test_psn_scrooge_tot_oracle_is_not_free_inspect():
    """Σ|A||Y| is leftover T×T work, not 1/T inspect. extra after tax ≤ 0."""
    T = 4
    A = torch.ones(T, T) * 0.1
    Y = torch.zeros(T, 2, 1)
    Y[0] = 100.0
    tau = torch.full((T, 1), 15.0)
    extra_oracle = mod.psn_scrooge_extra(Y, A, tau, bound='tot_oracle')
    extra_honest = mod.psn_scrooge_extra(Y, A, tau, bound='l1_maxabs')
    # tot at 0 MACs = 0.1*100 = 10 < 15 → oracle can skip-all
    assert extra_oracle > extra_honest + 0.2
    tax = mod.psn_scrooge_inspect_tax(T, 'tot_oracle')
    assert tax == 1.0
    assert extra_oracle - tax <= 0.0
    assert mod.keep_kill(extra_oracle - tax) == 'KILL'
    # 1/T would undercharge tot by factor T — must not be the tot_oracle tax
    assert tax != 1.0 / T


def test_psn_lowrank_and_zero_t():
    assert abs(mod.psn_lowrank_mac_extra(10, 4) - 0.20) < 1e-9
    assert mod.psn_lowrank_mac_extra(10, 5) == 0.0
    S = torch.zeros(4, 3, 2)
    S[0] = 1
    extra = mod.psn_zero_t_extra(S)
    assert abs(extra - 0.75) < 1e-6


def test_rate_matched_and_block_theta():
    torch.manual_seed(0)
    U = torch.randn(100, 8)
    shared = 0.0
    theta, fire = mod.rate_matched_theta(U, shared)
    assert theta.shape == (8,)
    fire2 = (U >= theta).float().mean(0)
    assert torch.allclose(fire, fire2, atol=0.05)
    blk = mod.block_pool_theta(theta, 4, how='mean')
    assert blk.shape == (8,)
    assert torch.allclose(blk[:4], blk[:4].mean().expand(4))
    assert torch.allclose(blk[4:], blk[4:].mean().expand(4))
    sparse = mod.scale_theta_for_rate(U, 0.5 * fire)
    fire_s = (U >= sparse).float().mean(0)
    assert float(fire_s.mean()) < float(fire.mean()) + 1e-6


def test_keep_kill_threshold():
    assert mod.keep_kill(0.15, 'x') == 'KEEP'
    assert mod.keep_kill(0.149, 'x') == 'KILL'


def test_lut_gemm_extra_is_vs_nnz_not_dense():
    S = torch.zeros(8, 8)
    S[:, 0] = 1  # one spike per row in group0
    extra, n_grp, nnz = mod.lut_gemm_group_extra(S, group=4)
    assert nnz == 8
    assert n_grp == 8  # one nonempty group per row
    assert abs(extra - 0.0) < 1e-9  # 1 group vs 1 nnz
    S2 = torch.ones(4, 8)
    extra2, n_grp2, nnz2 = mod.lut_gemm_group_extra(S2, group=4)
    assert nnz2 == 4 * 8
    assert n_grp2 == 4 * 2  # 2 groups of 4
    assert abs(extra2 - (1.0 - 8 / 32)) < 1e-9


def test_support_code_tax_is_vs_word_adds():
    extra = mod.support_code_compute_extra(100, 10)
    assert abs(extra - 0.9) < 1e-9
    tax = mod.support_code_encode_tax(100, 10, cin=96, baseline_word_adds=100 * 384)
    # 100*10*3 / (100*384) = 3000/38400
    assert abs(tax - 3000 / 38400) < 1e-9
    assert extra - tax > 0.15


if __name__ == '__main__':
    test_aac_baseline_all_zero_S()
    test_word_add_baseline_is_nnz_times_Cout()
    test_bitwave_vs_word_negative_when_bits_cost_more()
    test_prefix_nnz_extra_is_dropped_nnz_fraction()
    test_k_keep_for_nnz_extra_max_k()
    test_sna_pea_is_skipped_over_total_all_Cout()
    test_psn_scrooge_identity_easy_decide()
    test_psn_scrooge_tot_oracle_is_not_free_inspect()
    test_psn_lowrank_and_zero_t()
    test_rate_matched_and_block_theta()
    test_keep_kill_threshold()
    test_lut_gemm_extra_is_vs_nnz_not_dense()
    test_support_code_tax_is_vs_word_adds()
    print('PASS test_remaining_stats')
