import os

import torch
from spikingjelly.activation_based import functional


def resolve_snn_backend(config):
    runtime = config.get("runtime", {})
    requested = os.getenv("SDFORMER_SNN_BACKEND", runtime.get("snn_backend", "cupy")).lower()
    if requested not in {"cupy", "torch", "auto"}:
        raise ValueError(f"Unsupported runtime.snn_backend: {requested}")

    if requested == "cupy":
        _require_cupy()
        return "cupy", "explicit config"

    if requested == "torch":
        return "torch", "explicit config"

    if torch.version.cuda is not None:
        try:
            _require_cupy()
            return "cupy", "CUDA runtime with CuPy available"
        except RuntimeError as exc:
            return "torch", f"CuPy unavailable ({exc.__cause__.__class__.__name__})"

    return "torch", "non-CUDA runtime detected"


def configure_snn_backend(model, device, config, neurontype):
    if device.type == "cpu":
        print("[runtime] CPU execution; skip explicit SNN backend selection")
        return None

    backend, reason = resolve_snn_backend(config)
    functional.set_backend(model, backend, neurontype)
    print(f"[runtime] SNN backend = {backend} ({reason})")
    return backend


def _require_cupy():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "CuPy backend requested but CuPy is unavailable. "
            "Set runtime.snn_backend to 'torch' or 'auto' for non-CUDA accelerators."
        ) from exc
