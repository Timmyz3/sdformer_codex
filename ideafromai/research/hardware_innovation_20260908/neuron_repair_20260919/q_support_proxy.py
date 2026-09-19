"""Experimental Q-only backward bridge for the binary H67 score.

The original score is returned unchanged. Its existing K/motion backward is
retained. Only the previously disconnected Q support path receives the gradient
of q*k + alpha*(1-q)*(1-k), with a unit-slope support STE. This is a new training
rule, not the exact derivative of Boolean comparisons or a trained improvement.
"""
import torch


class QSupportBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_score, q_token, k_token, alpha, normalization):
        ctx.save_for_backward(k_token > 0)
        ctx.q_dtype = q_token.dtype
        ctx.alpha, ctx.normalization = alpha, normalization
        return original_score

    @staticmethod
    def backward(ctx, grad_score):
        (k_active,) = ctx.saved_tensors
        # Binary derivative is exactly 1 at active K and -alpha at silent K;
        # avoid (1+alpha)-alpha cancellation in low precision.
        coefficient = torch.where(k_active, grad_score.new_tensor(1.), grad_score.new_tensor(-ctx.alpha))
        dq = grad_score * coefficient * ctx.normalization
        return grad_score, dq.to(ctx.q_dtype), None, None, None


def install():
    from models.STSwinNet_SNN import bsa_attention as bsa
    original = bsa._ternary_alpha_xnor_token_scores
    counters = {'calls': 0}

    def bridged(q_orig, k_orig, cfg, beta=None):
        assert cfg.consensus_score_norm in {'head_dim', 'dim'}
        assert cfg.single_active_penalty == 0 and cfg.k_magnitude_alpha == 0
        score = original(q_orig, k_orig, cfg, beta)
        counters['calls'] += 1
        return QSupportBridge.apply(score, bsa._qkformer_token_q(q_orig), k_orig,
                                    float(cfg.alpha0), float(cfg.score_scale) / k_orig.shape[-1])

    bsa._ternary_alpha_xnor_token_scores = bridged
    return original, counters


def restore(original):
    from models.STSwinNet_SNN import bsa_attention as bsa
    bsa._ternary_alpha_xnor_token_scores = original
