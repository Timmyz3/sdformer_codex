"""Layer-selective adapter for the frozen M284 near-match forward.

The base implementation and its numerical policy are reused byte-for-byte.
This adapter only restores the original production forward for disabled
bottleneck Conv operators.  The resulting per-layer enable bits are a strict
hardware/runtime subset of the all-four-layer M284 mechanism.
"""

import importlib.util
import hashlib
import os
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().with_name("near_match_residual_elision.py")
_EXPECTED_BASE_SHA256 = (
    "06ddeecc373e5f105f08109d7977c672efa70fe6c0210ab42accc0b44c201b7c")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(_BASE_PATH) != _EXPECTED_BASE_SHA256:
    raise RuntimeError("M306 frozen M284 base-module SHA drift")
_SPEC = importlib.util.spec_from_file_location("m306_frozen_m284_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("M306 cannot load the frozen M284 base module")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_SELECTIVE_STATE = "_m306_near_match_selective_state"


def _enabled_indices():
    raw = os.environ.get("M306_ENABLED_OPERATOR_INDICES", "")
    if not raw:
        raise RuntimeError("M306 enabled-operator policy was not supplied")
    values = tuple(int(value) for value in raw.split(",") if value != "")
    if (not values or tuple(sorted(set(values))) != values or
            any(value < 0 or value >= len(_BASE._OPERATORS) for value in values)):
        raise RuntimeError("M306 invalid enabled-operator index set")
    return values


def install_near_match_residual_elision(model, spec):
    if hasattr(model, _SELECTIVE_STATE):
        raise RuntimeError("M306 refuses a stale selective installation")
    enabled = _enabled_indices()
    modules = dict(model.named_modules())
    original_forwards = {}
    for name in _BASE._OPERATORS:
        module = modules.get(name)
        if module is None:
            raise RuntimeError("M306 missing operator " + name)
        original_forwards[name] = module.forward
    installed = _BASE.install_near_match_residual_elision(model, spec)
    if tuple(installed) != tuple(_BASE._OPERATORS):
        raise RuntimeError("M306 base operator order drift")
    disabled = []
    for index, name in enumerate(installed):
        if index not in enabled:
            modules[name].forward = original_forwards[name]
            disabled.append(index)
    setattr(model, _SELECTIVE_STATE, {
        "enabled_operator_indices": list(enabled),
        "disabled_operator_indices": disabled,
        "enabled_operator_names": [installed[index] for index in enabled],
        "disabled_operator_names": [installed[index] for index in disabled],
    })
    return installed


def near_match_residual_elision_summary(model):
    result = _BASE.near_match_residual_elision_summary(model)
    state = getattr(model, _SELECTIVE_STATE, None)
    if state is None:
        raise RuntimeError("M306 summary requested before selective installation")
    result = dict(result)
    result["selective_schema"] = "m306_near_match_layer_selective_runtime_v1"
    result.update(state)
    for index in state["disabled_operator_indices"]:
        name = _BASE._OPERATORS[index]
        if (int(result["calls"][name]) != 0 or
                int(result["partition_vectors"][name]) != 0 or
                int(result["snapped_partition_vectors"][name]) != 0 or
                int(result["exact_hit_snapped_partition_vectors"][name]) != 0 or
                int(result["positive_distance_snapped_partition_vectors"][name]) != 0):
            raise RuntimeError("M306 disabled operator executed modified forward")
    return result
