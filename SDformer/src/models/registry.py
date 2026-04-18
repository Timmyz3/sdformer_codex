"""Simple registry helpers for models and pluggable modules."""

from __future__ import annotations

from typing import Any, Callable, Dict


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}
MODULE_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "attention": {},
    "external_inspirations": {},
    "spike_encoding": {},
    "normalization": {},
    "sparse_ops": {},
    "token_mixer": {},
    "spiking_neurons": {},
}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def register_module(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if kind not in MODULE_REGISTRY:
        raise KeyError(f"unsupported module kind: {kind}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODULE_REGISTRY[kind][name] = fn
        return fn

    return decorator


def build_model(cfg: Dict[str, Any]) -> Any:
    model_name = cfg["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        # Variants all build through the same adapter.
        model_name = "sdformer_baseline"
    return MODEL_REGISTRY[model_name](cfg)


def build_module(kind: str, name: str, **kwargs: Any) -> Any:
    if kind not in MODULE_REGISTRY:
        raise KeyError(f"unsupported module kind: {kind}")
    if name not in MODULE_REGISTRY[kind]:
        raise KeyError(f"unsupported {kind} module: {name}")
    return MODULE_REGISTRY[kind][name](**kwargs)


from src.models.sdformer.backbone import SDFormerFlowAdapter  # noqa: E402
from src.models.modules.attention.window_attention import (  # noqa: E402
    BaselineAttentionSpec,
    WindowSpikeAttentionSpec,
)
from src.models.modules.external_inspirations import (  # noqa: E402
    ActivityWindowScheduler,
    BlockSparseAttentionMasker,
    GraphImportanceTokenPruner,
    MotionGuidedTokenSelector,
    SimilarityTokenMerger,
    StructuredLatencyPruningController,
    TemporalAttentionReuse,
)
from src.models.modules.normalization.rmsnorm import RMSNorm  # noqa: E402
from src.models.modules.sparse_ops.head_pruning import HeadGroupPruner  # noqa: E402
from src.models.modules.sparse_ops.timestep_budget import TimestepBudgetPruner  # noqa: E402
from src.models.modules.sparse_ops.token_pruning import ActivityStats, StructuredTokenPruner  # noqa: E402
from src.models.modules.sparse_ops.window_pruning import WindowTopKPruner  # noqa: E402
from src.models.modules.spike_encoding.encoders import (  # noqa: E402
    LatencySpikeEncoder,
    TemporalContrastEncoder,
    VoxelSpikeEncoder,
)
from src.models.modules.token_mixer.identity import IdentityTokenMixer  # noqa: E402
from src.models.modules.token_mixer.temporal_shift import TemporalShiftTokenMixer  # noqa: E402


register_model("sdformer_baseline")(SDFormerFlowAdapter)
register_model("variant_a")(SDFormerFlowAdapter)
register_model("variant_b")(SDFormerFlowAdapter)
register_model("variant_c")(SDFormerFlowAdapter)
register_model("variant_modular")(SDFormerFlowAdapter)
