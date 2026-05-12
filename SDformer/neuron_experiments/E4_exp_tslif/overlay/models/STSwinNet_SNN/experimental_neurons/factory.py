"""Factory for experiment-local SDFormerFlow neurons."""

from __future__ import annotations


EXPERIMENTAL_NEURONS = {
    "exp_sn",
    "exp_atlif",
    "exp_lmh",
    "exp_tslif",
    "exp_tsn",
    "fused_adaptive_psn",
    "fused_lmh_atlif",
    "fused_adaptive_tslif",
    "fused_lmh_tslif",
    "fused_signed_hybrid",
}


def is_experimental_neuron(neuron_type: str) -> bool:
    return neuron_type in EXPERIMENTAL_NEURONS


def build_experimental_neuron(
    neuron_type: str,
    num_steps: int,
    v_th: float = 1.0,
    v_reset=None,
    tau: float = 2.0,
    detach_reset: bool = True,
    surrogate_fun=None,
    **kwargs,
):
    if neuron_type == "exp_sn":
        from .single.sn import SNNode
        return SNNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau, detach_reset=detach_reset)
    if neuron_type == "exp_atlif":
        from .single.atlif import ATLIFNode
        return ATLIFNode(T=num_steps, v_threshold=v_th, tau=1.0 / tau)
    if neuron_type == "exp_lmh":
        from .single.lmh import LMHNode
        return LMHNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "exp_tslif":
        from .single.tslif import TSLIFNode
        return TSLIFNode(T=num_steps, v_threshold=v_th, v_reset=v_reset, detach_reset=detach_reset)
    if neuron_type == "exp_tsn":
        from .single.tsn import TSNNode
        return TSNNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau)
    if neuron_type == "fused_adaptive_psn":
        from .fused.adaptive_psn import AdaptivePSNNode
        return AdaptivePSNNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_lmh_atlif":
        from .fused.lmh_atlif import LMHATLIFNode
        return LMHATLIFNode(T=num_steps, v_threshold=v_th, tau=1.0 / tau)
    if neuron_type == "fused_adaptive_tslif":
        from .fused.adaptive_tslif import AdaptiveTSLIFNode
        return AdaptiveTSLIFNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_lmh_tslif":
        from .fused.lmh_tslif import LMHTSLIFNode
        return LMHTSLIFNode(T=num_steps, v_threshold=v_th)
    if neuron_type == "fused_signed_hybrid":
        from .fused.signed_hybrid import SignedHybridNode
        return SignedHybridNode(T=num_steps, v_threshold=v_th, decay=1.0 / tau)
    raise KeyError(f"unsupported experimental neuron: {neuron_type}")


def resolve_backend_neuron_type(neuron_type: str):
    return type(build_experimental_neuron(neuron_type=neuron_type, num_steps=1))
