"""Hardware-weighted pattern-aware fine-tuning for the H67 bottleneck Conv set.

The forward path is not changed.  Hooks sample exact Conv3x3 input vectors and
add a differentiable proxy for the runtime cost

    min(popcount(x), 1 + min_p Hamming(x, p)).

The first branch is the existing bit-sparse zero fallback; the second is one
PWP vector plus signed correction vectors.  Pattern catalogs must be frozen
from training/calibration data before the candidate run.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


_STATE_ATTR = "_m71_pattern_paft_state"
_EXPECTED_SCHEMA = "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1"
_EXPECTED_ADMISSION_SCHEMA = "m77_pattern_paft_catalog_admission_contract_v1"
_EXPECTED_TRAIN_LIST_SHA256 = (
    "919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10")
_EXPECTED_VALID_LIST_SHA256 = (
    "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0")
_EXPECTED_CHECKPOINT_SHA256 = (
    "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158")
_EXPECTED_FORWARD_BASE_CONFIG_SHA256 = (
    "86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc")
_EXPECTED_OPERATORS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
_REVOKED_CATALOG_SHA256 = frozenset((
    "142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd",
))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_catalog(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_catalog(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("M71 PAFT catalog path is not a file")
    catalog_sha = _sha256(path)
    if catalog_sha in _REVOKED_CATALOG_SHA256:
        raise RuntimeError("M71 PAFT catalog SHA is permanently revoked")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != _EXPECTED_SCHEMA:
        raise RuntimeError("M71 PAFT requires the M77 train-only k-means schema")
    split = payload.get("split", {})
    if split.get("test_or_validation_data_used") is not False:
        raise RuntimeError("M71 PAFT refuses a catalog with validation/test leakage")
    expected_catalog_sha = str(cfg.get("catalog_sha256", ""))
    if len(expected_catalog_sha) != 64 or catalog_sha != expected_catalog_sha:
        raise RuntimeError("M71 PAFT catalog SHA pin absent or mismatched")
    if split.get("role") != "DSEC_TRAIN_ONLY_PAFT_CALIBRATION":
        raise RuntimeError("M71 PAFT catalog lacks the train-only role receipt")
    if split.get("train_catalog_eligible") is not True:
        raise RuntimeError("M71 PAFT catalog is not explicitly train eligible")
    contract_value = cfg.get("catalog_admission_contract")
    if not contract_value:
        raise RuntimeError("M71 PAFT enabled without external admission contract")
    contract_path = _resolve_catalog(str(contract_value))
    expected_contract_sha = str(cfg.get("catalog_admission_contract_sha256", ""))
    if (not contract_path.is_file() or len(expected_contract_sha) != 64 or
            _sha256(contract_path) != expected_contract_sha):
        raise RuntimeError("M71 PAFT admission-contract SHA absent or mismatched")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != _EXPECTED_ADMISSION_SCHEMA:
        raise RuntimeError("M71 PAFT admission-contract schema mismatch")
    if contract.get("unit_test_only") is not False:
        raise RuntimeError("M71 PAFT refuses a unit-test-only admission contract")
    if contract.get("train_only_admitted") is not True:
        raise RuntimeError("M71 PAFT admission contract does not admit train-only use")
    if contract.get("catalog_sha256") != catalog_sha:
        raise RuntimeError("M71 PAFT admission contract/catalog SHA mismatch")
    if contract.get("train_sequence_list_sha256") != _EXPECTED_TRAIN_LIST_SHA256:
        raise RuntimeError("M71 PAFT admission contract train-list identity mismatch")
    if contract.get("valid825_sequence_list_sha256") != _EXPECTED_VALID_LIST_SHA256:
        raise RuntimeError("M71 PAFT admission contract valid825 identity mismatch")
    if int(contract.get("train_valid825_key_overlap", -1)) != 0:
        raise RuntimeError("M71 PAFT admission contract does not prove zero overlap")
    if contract.get("checkpoint_sha256") != _EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("M71 PAFT admission contract checkpoint mismatch")
    if tuple(contract.get("operator_names", [])) != _EXPECTED_OPERATORS:
        raise RuntimeError("M71 PAFT admission contract operator mismatch")
    revoked = frozenset(contract.get("revoked_catalog_sha256", []))
    if not _REVOKED_CATALOG_SHA256.issubset(revoked):
        raise RuntimeError("M71 PAFT admission contract omits revoked catalog SHA")
    trace_sha = str(contract.get("train_trace_manifest_sha256", ""))
    if len(trace_sha) != 64:
        raise RuntimeError("M71 PAFT admission contract lacks train-trace SHA")
    identity = payload.get("identity", {})
    for key, expected in (
            ("train_sequence_list_sha256", _EXPECTED_TRAIN_LIST_SHA256),
            ("valid825_sequence_list_sha256", _EXPECTED_VALID_LIST_SHA256),
            ("checkpoint_sha256", _EXPECTED_CHECKPOINT_SHA256),
            ("train_trace_manifest_sha256", trace_sha)):
        if identity.get(key) != expected:
            raise RuntimeError("M71 PAFT catalog identity mismatch: " + key)
    payload["_validated_admission_contract"] = contract
    fmt = payload.get("format", {})
    if int(fmt.get("partition_bits", -1)) != 16:
        raise RuntimeError("M71 PAFT requires k16 catalog")
    if int(fmt.get("partitions_per_operator", -1)) != 432:
        raise RuntimeError("M71 PAFT catalog partition extent mismatch")
    if int(fmt.get("maximum_explicit_patterns_per_partition", -1)) != 16:
        raise RuntimeError("M71 PAFT requires q16 catalog")
    return payload


def _pattern_tensor(operator: Dict[str, Any]) -> torch.Tensor:
    partitions = operator.get("partitions", [])
    if len(partitions) != 432:
        raise RuntimeError("M71 PAFT operator partition extent mismatch")
    values: List[List[int]] = []
    for expected_partition, partition in enumerate(partitions):
        if int(partition.get("partition", -1)) != expected_partition:
            raise RuntimeError("M71 PAFT partition ordering mismatch")
        patterns = partition.get("patterns", [])
        if len(patterns) != 16:
            raise RuntimeError("M71 PAFT catalog must contain exactly q16 entries")
        row = [int(item["value_hex"], 16) for item in patterns]
        if any(value <= 0 or value >= (1 << 16) for value in row):
            raise RuntimeError("M71 PAFT explicit patterns must be nonzero uint16")
        if len(set(row)) != len(row):
            raise RuntimeError("M71 PAFT duplicate pattern within partition")
        values.append(row)
    packed = torch.tensor(values, dtype=torch.int64)
    shifts = torch.arange(16, dtype=torch.int64)
    return ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def _sample_conv3x3_vectors(x: torch.Tensor, maximum_vectors: int) -> torch.Tensor:
    if x.dim() == 5:
        # SpikingJelly multi-step layout: T, B, C, H, W.
        x4 = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    elif x.dim() == 4:
        x4 = x
    else:
        raise RuntimeError("M71 PAFT expected a 4D/5D Conv input")
    if x4.shape[1] != 768:
        raise RuntimeError("M71 PAFT expected 768 bottleneck input channels")
    n, _, height, width = x4.shape
    population = int(n * height * width)
    count = min(maximum_vectors, population)
    if count <= 0:
        raise RuntimeError("M71 PAFT empty activation population")
    # Integer stratification is deterministic and does not invoke a training
    # RNG.  It samples across time, batch and spatial position.
    flat = (torch.arange(count, device=x.device, dtype=torch.long) * population) // count
    sample_n = flat // (height * width)
    spatial = flat % (height * width)
    sample_y = spatial // width
    sample_x = spatial % width
    padded = F.pad(x4, (1, 1, 1, 1))
    taps = []
    for kernel_y in range(3):
        for kernel_x in range(3):
            taps.append(padded[sample_n, :, sample_y + kernel_y,
                               sample_x + kernel_x])
    # [P,C,9] -> I_KY_KX, matching the frozen accelerator trace.
    return torch.stack(taps, dim=2).reshape(count, 768 * 9)


def _hard_support_ste(vectors: torch.Tensor) -> torch.Tensor:
    """Return exact binary support in forward with an identity STE backward.

    Hardware work depends on whether a source event exists, not on its floating
    threshold amplitude.  The detached correction makes the forward value
    exactly ``vectors != 0`` while retaining a useful gradient to the source.
    """
    if not bool(torch.isfinite(vectors).all().item()):
        raise RuntimeError("M71 PAFT source contains NaN/Infinity")
    hard = (vectors != 0).to(dtype=vectors.dtype)
    # Keep the identity-STE correction parenthesized.  ``hard + vectors -
    # vectors.detach()`` can round after the first addition and produce a
    # forward value close to, but not exactly, one (notably in fp16).  The
    # subtraction of an identical finite tensor is exact zero in forward.
    return hard + (vectors - vectors.detach())


def _cost_proxy(vectors: torch.Tensor, patterns: torch.Tensor,
                partition_chunk: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    grouped = _hard_support_ste(vectors).reshape(vectors.shape[0], 432, 16)
    candidate_sum = grouped.new_zeros(())
    baseline_sum = grouped.new_zeros(())
    elements = 0
    for start in range(0, 432, partition_chunk):
        stop = min(start + partition_chunk, 432)
        activation = grouped[:, start:stop, :]
        catalog = patterns[start:stop].to(device=activation.device,
                                           dtype=activation.dtype)
        hamming = torch.abs(
            activation.unsqueeze(2) - catalog.unsqueeze(0)).sum(dim=-1)
        nearest = hamming.min(dim=2).values
        zero_fallback = activation.sum(dim=-1)
        candidate = torch.minimum(zero_fallback, nearest + 1.0)
        candidate_sum = candidate_sum + candidate.sum()
        baseline_sum = baseline_sum + zero_fallback.sum()
        elements += int(candidate.numel())
    return candidate_sum, baseline_sum, elements


def install_pattern_paft(model: torch.nn.Module,
                         config: Optional[Dict[str, Any]],
                         checkpoint_path: Optional[str] = None) -> List[str]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return []
    existing = getattr(model, _STATE_ATTR, None)
    if existing is not None:
        raise RuntimeError("M71 PAFT refuses a preexisting or stale PAFT state")
    catalog_value = cfg.get("catalog")
    if not catalog_value:
        raise RuntimeError("M71 PAFT enabled without catalog path")
    catalog_path = _resolve_catalog(str(catalog_value))
    catalog = _load_catalog(catalog_path, cfg)
    contract = catalog["_validated_admission_contract"]
    train_list_value = cfg.get("runtime_train_sequence_list")
    valid_list_value = cfg.get("runtime_valid825_sequence_list")
    trace_value = cfg.get("runtime_train_trace_manifest")
    if not train_list_value or not valid_list_value or not trace_value:
        raise RuntimeError("M71 PAFT runtime dataset/trace paths are incomplete")
    train_list_path = _resolve_catalog(str(train_list_value))
    valid_list_path = _resolve_catalog(str(valid_list_value))
    trace_path = _resolve_catalog(str(trace_value))
    if (not train_list_path.is_file() or
            _sha256(train_list_path) != _EXPECTED_TRAIN_LIST_SHA256):
        raise RuntimeError("M71 PAFT runtime train-list SHA mismatch")
    if (not valid_list_path.is_file() or
            _sha256(valid_list_path) != _EXPECTED_VALID_LIST_SHA256):
        raise RuntimeError("M71 PAFT runtime valid825-list SHA mismatch")
    expected_trace_sha = str(cfg.get("runtime_train_trace_manifest_sha256", ""))
    if (not trace_path.is_file() or len(expected_trace_sha) != 64 or
            _sha256(trace_path) != expected_trace_sha or
            contract.get("train_trace_manifest_sha256") != expected_trace_sha):
        raise RuntimeError("M71 PAFT runtime train-trace SHA mismatch")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("schema") != "m73_h67_ep35_train_calibration_packed_source_trace_v1":
        raise RuntimeError("M71 PAFT runtime train-trace schema mismatch")
    trace_split = trace.get("split_audit", {})
    if (trace_split.get("role") != "DSEC_TRAIN_ONLY_PAFT_CALIBRATION" or
            int(trace_split.get("full_train_valid825_key_overlap", -1)) != 0 or
            int(trace_split.get("selected_valid825_key_overlap", -1)) != 0):
        raise RuntimeError("M71 PAFT runtime train-trace split audit failed")
    trace_identity = trace.get("identity", {})
    if (trace_identity.get("train_sequence_list_sha256") !=
            _EXPECTED_TRAIN_LIST_SHA256 or
            trace_identity.get("valid825_sequence_list_sha256") !=
            _EXPECTED_VALID_LIST_SHA256 or
            trace_identity.get("checkpoint_sha256") !=
            _EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("M71 PAFT runtime train-trace identity mismatch")
    expected_forward = str(cfg.get(
        "expected_forward_base_config_sha256", ""))
    catalog_identity = catalog.get("identity", {})
    if (expected_forward != _EXPECTED_FORWARD_BASE_CONFIG_SHA256 or
            contract.get("forward_base_config_sha256") != expected_forward or
            catalog_identity.get("forward_base_config_sha256") !=
                expected_forward or
            trace_identity.get("config_sha256") != expected_forward or
            trace_identity.get("paft_forward_base_config_sha256") !=
                expected_forward):
        raise RuntimeError("M71 PAFT capture/training forward-config mismatch")
    if not checkpoint_path:
        raise RuntimeError("M71 PAFT enabled without checkpoint path for SHA pin")
    checkpoint = Path(str(checkpoint_path)).resolve()
    expected_checkpoint = str(cfg.get("expected_checkpoint_sha256", ""))
    if (not checkpoint.is_file() or
            expected_checkpoint != _EXPECTED_CHECKPOINT_SHA256 or
            _sha256(checkpoint) != expected_checkpoint):
        raise RuntimeError("M71 PAFT runtime checkpoint SHA mismatch")
    modules = dict(model.named_modules())
    expected_operators = list(cfg.get("expected_operator_names", []))
    catalog_operators = list(catalog.get("operators", []))
    catalog_names = [str(operator.get("operator"))
                     for operator in catalog_operators]
    if (catalog_names != expected_operators or
            tuple(expected_operators) != _EXPECTED_OPERATORS):
        raise RuntimeError("M71 PAFT operator-name/order pin mismatch")
    prepared = []
    for operator, name in zip(catalog_operators, catalog_names):
        module = modules.get(name)
        if module is None:
            raise RuntimeError("M71 PAFT missing model module: " + name)
        prepared.append((name, module, _pattern_tensor(operator)))
    maximum_vectors = int(cfg.get("sample_vectors_per_module", 64))
    partition_chunk = int(cfg.get("partition_chunk", 36))
    if maximum_vectors <= 0 or partition_chunk <= 0 or partition_chunk > 432:
        raise RuntimeError("M71 PAFT invalid sampling/chunk configuration")

    state: Dict[str, Any] = {
        "catalog_path": str(catalog_path),
        "operator_names": [],
        "patterns": {},
        "handles": [],
        "observations": [],
        "maximum_vectors": maximum_vectors,
        "partition_chunk": partition_chunk,
    }
    setattr(model, _STATE_ATTR, state)

    def clear_observations(_module: torch.nn.Module,
                           _inputs: Tuple[torch.Tensor, ...]) -> None:
        state["observations"] = []

    state["handles"].append(model.register_forward_pre_hook(clear_observations))

    for name, module, pattern in prepared:
        state["operator_names"].append(name)
        state["patterns"][name] = pattern

        def capture(_module: torch.nn.Module,
                    inputs: Tuple[torch.Tensor, ...],
                    operator_name: str = name) -> None:
            if not inputs:
                raise RuntimeError("M71 PAFT Conv hook received no input")
            vectors = _sample_conv3x3_vectors(inputs[0], maximum_vectors)
            candidate, baseline, elements = _cost_proxy(
                vectors, state["patterns"][operator_name], partition_chunk)
            state["observations"].append({
                "operator": operator_name,
                "candidate_sum": candidate,
                "baseline_sum": baseline,
                "elements": elements,
            })

        state["handles"].append(module.register_forward_pre_hook(capture))
    if state["operator_names"] != expected_operators:
        raise RuntimeError("M71 PAFT internal operator installation drift")
    return list(state["operator_names"])


def regularize_pattern_paft(model: torch.nn.Module,
                            config: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return None
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        raise RuntimeError("M71 PAFT regularizer called before installation")
    observations = state["observations"]
    if len(observations) != 4:
        raise RuntimeError("M71 PAFT did not observe all four bottleneck Conv inputs")
    candidate_sum = sum(item["candidate_sum"] for item in observations)
    elements = sum(int(item["elements"]) for item in observations)
    weight = float(cfg.get("regularization_weight", 0.0))
    if weight < 0.0:
        raise RuntimeError("M71 PAFT regularization weight must be nonnegative")
    # Eight output blocks consume each selected source pattern.  Division by
    # the sampled population leaves the user-facing weight stable as sampling
    # density changes while retaining the hardware fanout in the objective.
    return candidate_sum * (8.0 * weight / float(elements))


def pattern_paft_summary(model: torch.nn.Module) -> Dict[str, Any]:
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return {"installed": False}
    observations = state.get("observations", [])
    baseline = sum(float(item["baseline_sum"].detach().item())
                   for item in observations)
    candidate = sum(float(item["candidate_sum"].detach().item())
                    for item in observations)
    return {
        "installed": True,
        "catalog_path": state["catalog_path"],
        "operators": list(state["operator_names"]),
        "sample_vectors_per_module": state["maximum_vectors"],
        "sampled_baseline_vector_ops": baseline,
        "sampled_candidate_vector_ops": candidate,
        "sampled_proxy_speedup": baseline / candidate if candidate else None,
    }
