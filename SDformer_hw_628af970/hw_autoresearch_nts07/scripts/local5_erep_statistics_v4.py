#!/usr/bin/env python3
"""Exact, fail-closed statistics for the fixed Local5 EREP v4 G0 admission."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .local5_erep_archive_replay_v4 import validate_archive_files
    from .local5_erep_ledger_replay_v4 import validate_replayed_ledgers
else:
    from local5_erep_archive_replay_v4 import validate_archive_files
    from local5_erep_ledger_replay_v4 import validate_replayed_ledgers


SCHEMA = "local5_erep_g0_statistics_v4"
ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
SELECTION_PLAN = PROFILE_DIR / "joint_window_selection_plan.json"
FORMAL_MANIFEST = PROFILE_DIR / "ordered_term_manifest.json"
FORMAL_COHORT = PROFILE_DIR / "ordered_cohort.json"
FORMAL_GPU_AUDIT = PROFILE_DIR / "gpu_exclusivity_audit.json"
ADMISSION_DIR = ROOT / "results/local5_erep_g0_admission_v4_20260810"
ADMITTED_BUNDLE = ADMISSION_DIR / "admitted_rows.json"
ADMISSION_RECEIPT = ADMISSION_DIR / "admission_receipt.json"
HEAD_PHASE_LEDGER = ADMISSION_DIR / "head_phase_ledger.json"
WINDOW_SCHEDULE_LEDGER = ADMISSION_DIR / "window_schedule_ledger.json"
COMMAND_LEDGER = ADMISSION_DIR / "command_ledger.json"
RTL_TRACE_ARCHIVE = ADMISSION_DIR / "rtl_trace_archive.npz"
ACC32_MITER_ARCHIVE = ADMISSION_DIR / "acc32_miter_archive.npz"
RUNTIME_RECEIPT = ROOT / "contracts/local5_erep_g0_runtime_v4_20260810.json"

SELECTION_PLAN_SHA256 = "4e8732210a64cfcb553e7f4eee3657be70cc975a38839527e4792668d6deaf6b"
COHORT_SAMPLE_KEY_SHA256 = "32138614734d4ca9e14253ba9863f554ddb4d57552531e0ed153652d1acda125"
CANDIDATES = ("c0", "c1", "c2", "c3", "c4")
STAGE_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
STAGE_WEIGHTS = {0: 440, 1: 120, 2: 30, 3: 10}
ROW_FIELDS = frozenset(
    {
        "sample",
        "sample_key",
        "sequence_key",
        "stage",
        "block",
        "window",
        "weight",
        "command_ledger_sha256",
        "c4_source",
        *CANDIDATES,
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "selection_plan_sha256",
        "formal_manifest_sha256",
        "formal_payload_sha256",
        "formal_cohort_file_sha256",
        "formal_cohort_sample_key_sha256",
        "formal_identity_sha256",
        "formal_gpu_audit_sha256",
        "projection_contract_file_sha256",
        "projection_contract_payload_sha256",
        "head_phase_ledger_sha256",
        "window_schedule_ledger_sha256",
        "command_ledger_sha256",
        "rtl_trace_archive_sha256",
        "acc32_miter_archive_sha256",
        "admitted_bundle_sha256",
        "rows_canonical_sha256",
        "adapter_path",
        "adapter_sha256",
        "runtime_receipt_sha256",
    }
)
RUNTIME_FIELDS = frozenset(
    {
        "schema",
        "status",
        "python_executable",
        "python_version",
        "python_binary_sha256",
        "numpy_version",
        "numpy_init_path",
        "numpy_init_sha256",
        "numpy_multiarray_path",
        "numpy_multiarray_sha256",
        "pcg64_seed",
        "pcg64_first_draw_18",
        "bootstrap_trials",
        "percentile_method",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required frozen artifact is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _strict_uint(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a non-boolean Python integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _strict_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_runtime() -> dict[str, Any]:
    receipt = _load_json(RUNTIME_RECEIPT)
    if not isinstance(receipt, dict) or set(receipt) != RUNTIME_FIELDS:
        raise ValueError("runtime receipt has a non-frozen field set")
    if (
        receipt["schema"] != "local5_erep_g0_runtime_v4"
        or receipt["status"] != "FROZEN_LOCAL_RUNTIME"
        or receipt["python_executable"] != "/opt/conda/bin/python3.11"
        or receipt["python_version"] != "3.11.9"
        or receipt["numpy_version"] != "2.1.1"
        or receipt["pcg64_seed"] != 20260810
        or receipt["bootstrap_trials"] != 20_000
        or receipt["percentile_method"] != "exact_inverted_cdf"
    ):
        raise ValueError("runtime receipt values violate the frozen contract")
    bindings = (
        (Path(receipt["python_executable"]), "python_binary_sha256"),
        (Path(receipt["numpy_init_path"]), "numpy_init_sha256"),
        (Path(receipt["numpy_multiarray_path"]), "numpy_multiarray_sha256"),
    )
    for path, field in bindings:
        if not path.is_file() or _sha256(path) != receipt[field]:
            raise ValueError(f"runtime binding failed: {path}")
    if platform.python_version() != "3.11.9" or np.__version__ != "2.1.1":
        raise RuntimeError("the active interpreter is not the frozen G0 runtime")
    draw = np.random.Generator(np.random.PCG64(20260810)).integers(
        0, 18, size=18, dtype=np.int64
    ).tolist()
    if draw != receipt["pcg64_first_draw_18"]:
        raise RuntimeError("PCG64 golden vector does not match the frozen runtime")
    return receipt


def _verify_profile_artifacts(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load_json(SELECTION_PLAN)
    manifest = _load_json(FORMAL_MANIFEST)
    cohort = _load_json(FORMAL_COHORT)
    audit = _load_json(FORMAL_GPU_AUDIT)
    identity_path = PROFILE_DIR / "joint_head_run_identity.json"
    identity = _load_json(identity_path)

    if _sha256(SELECTION_PLAN) != SELECTION_PLAN_SHA256:
        raise ValueError("selection plan SHA is not the preregistered SHA")
    if receipt["selection_plan_sha256"] != SELECTION_PLAN_SHA256:
        raise ValueError("admission receipt does not bind the selection plan")
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or manifest.get("evidence_level") != "post_g0"
        or len(manifest.get("groups") or []) != 13_800
        or (manifest.get("qualification") or {}).get("qualified") is not True
        or manifest.get("cohort_sha256") != COHORT_SAMPLE_KEY_SHA256
        or receipt["formal_manifest_sha256"] != _sha256(FORMAL_MANIFEST)
        or receipt["formal_cohort_file_sha256"] != _sha256(FORMAL_COHORT)
        or receipt["formal_identity_sha256"] != _sha256(identity_path)
    ):
        raise ValueError("formal manifest/cohort/identity admission failed")
    payload = PROFILE_DIR / str(manifest.get("payload_file", ""))
    projection_json = PROFILE_DIR / str(manifest.get("projection_contract_file", ""))
    projection_npz = PROFILE_DIR / str(manifest.get("projection_contract_payload", ""))
    if (
        not payload.is_file()
        or receipt["formal_payload_sha256"] != _sha256(payload)
        or manifest.get("payload_sha256") != _sha256(payload)
        or not projection_json.is_file()
        or not projection_npz.is_file()
        or receipt["projection_contract_file_sha256"] != _sha256(projection_json)
        or receipt["projection_contract_payload_sha256"] != _sha256(projection_npz)
        or manifest.get("projection_contract_file_sha256") != _sha256(projection_json)
        or manifest.get("projection_contract_payload_sha256") != _sha256(projection_npz)
    ):
        raise ValueError("formal payload/projection binding failed")
    if (
        cohort.get("schema") != "ordered_trace_cohort_v2"
        or cohort.get("count") != 100
        or len(cohort.get("sample_keys") or []) != 100
        or len(cohort.get("sequence_keys") or []) != 100
        or len(set(cohort.get("sequence_keys") or [])) != 18
        or cohort.get("sample_key_sha256") != COHORT_SAMPLE_KEY_SHA256
        or receipt["formal_cohort_sample_key_sha256"] != COHORT_SAMPLE_KEY_SHA256
    ):
        raise ValueError("formal cohort identity failed")
    if (
        audit.get("schema") != "local5_joint_gpu_exclusivity_audit_v1"
        or audit.get("status") != "PASS"
        or audit.get("foreign_compute_pids") != []
        or receipt["formal_gpu_audit_sha256"] != _sha256(FORMAL_GPU_AUDIT)
        or audit.get("identity_sha256") != _sha256(identity_path)
        or audit.get("manifest_sha256") != _sha256(FORMAL_MANIFEST)
        or audit.get("payload_sha256") != _sha256(payload)
    ):
        raise ValueError("formal GPU exclusivity audit is not PASS-bound")
    if (
        identity.get("selection_plan_sha256") != SELECTION_PLAN_SHA256
        or identity.get("cohort_sha256") != COHORT_SAMPLE_KEY_SHA256
        or manifest.get("run_identity_file_sha256") != _sha256(identity_path)
    ):
        raise ValueError("formal run identity cross-binding failed")
    return plan, cohort, manifest


def _verify_archive_bindings(
    receipt: Mapping[str, Any],
    head_phase_ledger: Mapping[str, Any],
    *,
    admission_dir: Path = ADMISSION_DIR,
) -> tuple[Path, Path]:
    trace_archive = admission_dir / "rtl_trace_archive.npz"
    miter_archive = admission_dir / "acc32_miter_archive.npz"
    if (
        head_phase_ledger.get("rtl_trace_archive_file") != trace_archive.name
        or head_phase_ledger.get("acc32_miter_archive_file") != miter_archive.name
        or not trace_archive.is_file()
        or not miter_archive.is_file()
        or head_phase_ledger.get("rtl_trace_archive_sha256")
        != _sha256(trace_archive)
        or head_phase_ledger.get("acc32_miter_archive_sha256")
        != _sha256(miter_archive)
        or receipt.get("rtl_trace_archive_sha256") != _sha256(trace_archive)
        or receipt.get("acc32_miter_archive_sha256") != _sha256(miter_archive)
    ):
        raise ValueError("formal RTL trace/Acc32 miter archives are not SHA-bound")
    return trace_archive, miter_archive


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    plan_records: Sequence[Mapping[str, Any]],
    sample_keys: Sequence[str],
    sequence_keys_by_sample: Sequence[str],
    command_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != 1200 or len(plan_records) != 1200 or len(command_rows) != 1200:
        raise ValueError("rows, plan and command ledger must each contain exactly 1200 entries")
    if len(sample_keys) != 100 or len(sequence_keys_by_sample) != 100:
        raise ValueError("cohort must contain exactly 100 sample and sequence keys")

    normalized: list[dict[str, Any]] = []
    for index, (row, plan, command) in enumerate(
        zip(rows, plan_records, command_rows, strict=True)
    ):
        if not isinstance(row, Mapping) or set(row) != ROW_FIELDS:
            raise ValueError(f"row {index} has a non-frozen field set")
        if not isinstance(plan, Mapping) or not isinstance(command, Mapping):
            raise ValueError(f"row {index} plan/command entry must be an object")
        sample = _strict_uint(row["sample"], f"row {index} sample")
        stage = _strict_uint(row["stage"], f"row {index} stage")
        block = _strict_uint(row["block"], f"row {index} block")
        window = _strict_uint(row["window"], f"row {index} window")
        weight = _strict_uint(row["weight"], f"row {index} weight", positive=True)
        expected_coordinate = (
            index // 12,
            (0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3)[index % 12],
            (0, 1, 0, 1, 0, 1, 2, 3, 4, 5, 0, 1)[index % 12],
        )
        if (sample, stage, block) != expected_coordinate:
            raise ValueError(f"row {index} coordinate is not the frozen topology order")
        if set(plan) != {
            "sample", "stage", "block", "heads", "batch_windows", "window",
            "inclusion_probability", "analysis_weight"
        }:
            raise ValueError(f"plan row {index} has a non-frozen field set")
        if (
            type(plan["sample"]) is not int
            or type(plan["stage"]) is not int
            or type(plan["block"]) is not int
            or type(plan["window"]) is not int
            or (plan["sample"], plan["stage"], plan["block"]) != expected_coordinate
            or plan["window"] != window
            or plan["heads"] != (3, 6, 12, 24)[stage]
            or plan["batch_windows"] != STAGE_WEIGHTS[stage]
            or plan["analysis_weight"] != float(STAGE_WEIGHTS[stage])
            or plan["inclusion_probability"] != 1.0 / STAGE_WEIGHTS[stage]
            or weight != STAGE_WEIGHTS[stage]
        ):
            raise ValueError(f"row {index} does not match the frozen selection plan")
        if (
            row["sample_key"] != sample_keys[sample]
            or row["sequence_key"] != sequence_keys_by_sample[sample]
            or not isinstance(row["sample_key"], str)
            or not isinstance(row["sequence_key"], str)
            or not row["sample_key"]
            or not row["sequence_key"]
        ):
            raise ValueError(f"row {index} does not match the formal cohort")
        command_digest = _strict_sha(
            row["command_ledger_sha256"], f"row {index} command ledger digest"
        )
        if row["c4_source"] != "trace_derived_relaxed_oracle_5014_records_v4":
            raise ValueError(f"row {index} has an invalid C4 source")
        expected_command_fields = {
            "sample", "stage", "block", "window", "c0", "c1", "c2", "c3",
            "c4", "window_schedule_sha256", "command_ledger_sha256",
        }
        if set(command) != expected_command_fields:
            raise ValueError(f"command row {index} has a non-frozen field set")
        for field in ("sample", "stage", "block", "window", *CANDIDATES):
            if row[field] != command[field]:
                raise ValueError(f"row {index} differs from command ledger field {field}")
        window_digest = _strict_sha(
            command.get("window_schedule_sha256"),
            f"row {index} window schedule digest",
        )
        command_body = {
            field: command[field]
            for field in expected_command_fields
            if field != "command_ledger_sha256"
        }
        if (
            command.get("command_ledger_sha256") != _canonical_sha(command_body)
            or command.get("command_ledger_sha256") != command_digest
            or not window_digest
        ):
            raise ValueError(f"row {index} command ledger digest mismatch")

        item = dict(row)
        for candidate in CANDIDATES:
            item[candidate] = _strict_uint(
                row[candidate], f"row {index} {candidate}", positive=True
            )
        normalized.append(item)
    if len(set(sequence_keys_by_sample)) != 18:
        raise ValueError("the formal cohort must contain exactly 18 sequence clusters")
    return normalized


def _weighted_quantile(values: Sequence[int], weights: Sequence[int], q: Fraction) -> int:
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("weighted quantile vectors must be nonempty and aligned")
    ordered = sorted(zip(values, weights, strict=True), key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative * q.denominator >= total * q.numerator:
            return value
    return ordered[-1][0]


def _order_statistic(values: Sequence[Any], q: Fraction) -> Any:
    if not values:
        raise ValueError("order statistic requires nonempty values")
    ordered = sorted(values)
    numerator = q.numerator * len(ordered)
    index = (numerator + q.denominator - 1) // q.denominator - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _passes(value: Fraction | int, comparison: str, threshold: Fraction | int) -> bool:
    if comparison == ">=":
        return value >= threshold
    if comparison == ">":
        return value > threshold
    if comparison == "<=":
        return value <= threshold
    raise ValueError(f"unsupported exact comparison {comparison}")


def _evaluate_validated_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cluster_members: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cluster_members[str(row["sequence_key"])].append(index)
    cluster_names = sorted(cluster_members)
    if len(cluster_names) != 18:
        raise ValueError("the admitted rows must contain exactly 18 sequence clusters")
    cluster_totals = {
        name: {
            candidate: sum(
                int(rows[index]["weight"]) * int(rows[index][candidate])
                for index in cluster_members[name]
            )
            for candidate in CANDIDATES
        }
        for name in cluster_names
    }
    totals = {
        candidate: sum(cluster_totals[name][candidate] for name in cluster_names)
        for candidate in CANDIDATES
    }
    if any(total <= 0 for total in totals.values()):
        raise ValueError("every exact weighted candidate total must be positive")

    ratio_replicates = {
        f"c0_over_{candidate}": [] for candidate in CANDIDATES[1:]
    }
    ratio_replicates["synergy"] = []
    ratio_replicates["capacity_matched"] = []
    stage_deltas: list[list[int]] = []
    rng = np.random.Generator(np.random.PCG64(20260810))
    for _ in range(20_000):
        selected = rng.integers(0, 18, size=18, dtype=np.int64).tolist()
        selected_names = [cluster_names[index] for index in selected]
        replicate_totals = {
            candidate: sum(cluster_totals[name][candidate] for name in selected_names)
            for candidate in CANDIDATES
        }
        for candidate in CANDIDATES[1:]:
            ratio_replicates[f"c0_over_{candidate}"].append(
                Fraction(replicate_totals["c0"], replicate_totals[candidate])
            )
        ratio_replicates["synergy"].append(
            Fraction(
                min(replicate_totals["c1"], replicate_totals["c2"]),
                replicate_totals["c3"],
            )
        )
        ratio_replicates["capacity_matched"].append(
            Fraction(replicate_totals["c4"], replicate_totals["c3"])
        )
        expanded = [
            row_index
            for name in selected_names
            for row_index in cluster_members[name]
        ]
        deltas: list[int] = []
        for stage in range(4):
            stage_indices = [index for index in expanded if rows[index]["stage"] == stage]
            stage_weights = [int(rows[index]["weight"]) for index in stage_indices]
            c0_p95 = _weighted_quantile(
                [int(rows[index]["c0"]) for index in stage_indices],
                stage_weights,
                Fraction(19, 20),
            )
            c3_p95 = _weighted_quantile(
                [int(rows[index]["c3"]) for index in stage_indices],
                stage_weights,
                Fraction(19, 20),
            )
            deltas.append(c3_p95 - c0_p95)
        stage_deltas.append(deltas)

    def interval(values: Sequence[Fraction]) -> dict[str, Any]:
        return {
            "method": "exact_inverted_cdf",
            "confidence": "95/100",
            "lower": _fraction_record(_order_statistic(values, Fraction(1, 40))),
            "upper": _fraction_record(_order_statistic(values, Fraction(39, 40))),
        }

    ratios: dict[str, Any] = {}
    for candidate in CANDIDATES[1:]:
        key = f"c0_over_{candidate}"
        ratios[key] = {
            "estimate": _fraction_record(Fraction(totals["c0"], totals[candidate])),
            "bootstrap_ci95": interval(ratio_replicates[key]),
        }
    synergy_value = Fraction(min(totals["c1"], totals["c2"]), totals["c3"])
    capacity_value = Fraction(totals["c4"], totals["c3"])
    synergy = {
        "estimate": _fraction_record(synergy_value),
        "bootstrap_ci95": interval(ratio_replicates["synergy"]),
    }
    capacity = {
        "estimate": _fraction_record(capacity_value),
        "bootstrap_ci95": interval(ratio_replicates["capacity_matched"]),
    }
    stage_rows = []
    for stage in range(4):
        indices = [index for index, row in enumerate(rows) if row["stage"] == stage]
        weights = [int(rows[index]["weight"]) for index in indices]
        c0_p95 = _weighted_quantile(
            [int(rows[index]["c0"]) for index in indices], weights, Fraction(19, 20)
        )
        c3_p95 = _weighted_quantile(
            [int(rows[index]["c3"]) for index in indices], weights, Fraction(19, 20)
        )
        upper = _order_statistic(
            [replicate[stage] for replicate in stage_deltas], Fraction(79, 80)
        )
        stage_rows.append(
            {
                "stage": stage,
                "c0_p95": c0_p95,
                "c3_p95": c3_p95,
                "delta_c3_minus_c0": c3_p95 - c0_p95,
                "bootstrap_upper_bound": upper,
                "one_sided_confidence": "79/80",
                "passed": _passes(upper, "<=", 0),
            }
        )

    primary = Fraction(totals["c0"], totals["c3"])
    primary_lower = _order_statistic(
        ratio_replicates["c0_over_c3"], Fraction(1, 40)
    )
    synergy_lower = _order_statistic(ratio_replicates["synergy"], Fraction(1, 40))
    capacity_lower = _order_statistic(
        ratio_replicates["capacity_matched"], Fraction(1, 40)
    )
    gates = [
        {"name": "primary_speedup_c0_over_c3", "passed": _passes(primary, ">=", Fraction(5, 4))},
        {"name": "primary_bootstrap_ci95_lower", "passed": _passes(primary_lower, ">", 1)},
        {"name": "all_stage_p95_delta_upper_bounds", "passed": all(row["passed"] for row in stage_rows)},
        {"name": "synergy_speedup", "passed": _passes(synergy_value, ">=", Fraction(21, 20))},
        {"name": "synergy_bootstrap_ci95_lower", "passed": _passes(synergy_lower, ">", 1)},
        {"name": "capacity_matched_speedup", "passed": _passes(capacity_value, ">=", Fraction(21, 20))},
        {"name": "capacity_matched_bootstrap_ci95_lower", "passed": _passes(capacity_lower, ">", 1)},
    ]
    return {
        "schema": SCHEMA,
        "rows": 1200,
        "sequence_clusters": 18,
        "weighted_totals_exact_integer": totals,
        "determinism": {
            "bootstrap_method": "paired_sequence_cluster_complete_expansion",
            "bit_generator": "numpy.PCG64",
            "seed": 20260810,
            "trials": 20_000,
            "percentile_method": "exact_inverted_cdf",
            "decision_arithmetic": "Python integer and fractions.Fraction",
        },
        "ratios": ratios,
        "synergy": synergy,
        "capacity_matched": capacity,
        "stage_p95": stage_rows,
        "g0_gates": gates,
        "g0_passed": all(gate["passed"] for gate in gates),
    }


def evaluate_formal_g0() -> dict[str, Any]:
    runtime = _verify_runtime()
    receipt = _load_json(ADMISSION_RECEIPT)
    bundle = _load_json(ADMITTED_BUNDLE)
    head_phase_ledger = _load_json(HEAD_PHASE_LEDGER)
    window_schedule_ledger = _load_json(WINDOW_SCHEDULE_LEDGER)
    command_ledger = _load_json(COMMAND_LEDGER)
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("admission receipt has a non-frozen field set")
    if (
        receipt["schema"] != "local5_erep_g0_admission_receipt_v4"
        or receipt["status"] != "PASS"
        or receipt["admitted_bundle_sha256"] != _sha256(ADMITTED_BUNDLE)
        or receipt["head_phase_ledger_sha256"] != _sha256(HEAD_PHASE_LEDGER)
        or receipt["window_schedule_ledger_sha256"] != _sha256(WINDOW_SCHEDULE_LEDGER)
        or receipt["command_ledger_sha256"] != _sha256(COMMAND_LEDGER)
        or receipt["runtime_receipt_sha256"] != _sha256(RUNTIME_RECEIPT)
    ):
        raise ValueError("admission receipt is not PASS-bound to frozen artifacts")
    adapter = ROOT / str(receipt["adapter_path"])
    if not adapter.is_file() or receipt["adapter_sha256"] != _sha256(adapter):
        raise ValueError("formal adapter source binding failed")
    plan, cohort, _ = _verify_profile_artifacts(receipt)
    if not isinstance(bundle, dict) or set(bundle) != {"schema", "rows"} or (
        bundle["schema"] != "local5_erep_g0_admitted_rows_v4"
    ):
        raise ValueError("admitted bundle schema is invalid")
    if (
        not isinstance(head_phase_ledger, Mapping)
        or head_phase_ledger.get("formal_manifest_sha256")
        != receipt["formal_manifest_sha256"]
        or head_phase_ledger.get("projection_contract_sha256")
        != receipt["projection_contract_file_sha256"]
    ):
        raise ValueError("head phase ledger is not bound to admitted formal inputs")
    trace_archive, miter_archive = _verify_archive_bindings(
        receipt, head_phase_ledger
    )
    if trace_archive != RTL_TRACE_ARCHIVE or miter_archive != ACC32_MITER_ARCHIVE:
        raise AssertionError("formal archive helper returned non-frozen paths")
    archive_replay = validate_archive_files(
        trace_archive,
        miter_archive,
        head_phase_ledger,
        formal=True,
    )
    replayed_commands = validate_replayed_ledgers(
        head_phase_ledger,
        window_schedule_ledger,
        command_ledger,
        formal=True,
        plan_records=plan["records"],
    )
    rows = _validate_rows(
        bundle["rows"],
        plan["records"],
        cohort["sample_keys"],
        cohort["sequence_keys"],
        replayed_commands,
    )
    if receipt["rows_canonical_sha256"] != _canonical_sha(rows):
        raise ValueError("admitted row canonical digest mismatch")
    report = _evaluate_validated_rows(rows)
    report["runtime_receipt_sha256"] = _sha256(RUNTIME_RECEIPT)
    report["admission_receipt_sha256"] = _sha256(ADMISSION_RECEIPT)
    report["rows_canonical_sha256"] = _canonical_sha(rows)
    report["archive_content_replay"] = archive_replay
    report["runtime"] = {
        "python": runtime["python_version"],
        "numpy": runtime["numpy_version"],
    }
    return report


def main() -> int:
    print(json.dumps(evaluate_formal_g0(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
