#!/usr/bin/env python3
"""Build the final Motion checkpoint selection/rebind binder.

This program is deliberately read-only with respect to the training run.  It
does not evaluate a model, start a GPU process, copy a checkpoint, or launch a
hardware replay.  It may run only after the predeclared five standard-valid825
profiles have completed.  Selection is the minimum exact AEE, with epoch as a
deterministic tie break.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


EPOCHS = (9, 14, 19, 24, 29)
EXPECTED_CONFIG_SHA256 = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
EXPECTED_RUN_DIR = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx"
)
EXPECTED_CONFIG = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
)
EXPECTED_RANKING = EXPECTED_RUN_DIR / "profile_ranking_valid825.md"


class BinderError(RuntimeError):
    """A fail-closed input or provenance failure."""


@dataclass(frozen=True)
class RunPolicy:
    run_dir: Path
    config: Path
    ranking: Path
    config_sha256: str
    epochs: tuple[int, ...] = EPOCHS
    ranking_mode: str = "aee"
    atlif_modules: int = 105
    attention_modules: int = 12


PRODUCTION_POLICY = RunPolicy(
    run_dir=EXPECTED_RUN_DIR,
    config=EXPECTED_CONFIG,
    ranking=EXPECTED_RANKING,
    config_sha256=EXPECTED_CONFIG_SHA256,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BinderError(message)


def _no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BinderError(f"non-finite JSON constant: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    regular_file(path, f"JSON input {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise BinderError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise BinderError(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(mode), f"not a regular file for {label}: {path}")
    require(not path.is_symlink(), f"symlink forbidden for {label}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_identity(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
        == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev),
        f"{label} changed while hashing: {path}",
    )
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def decimal_metric(metrics: dict[str, Any], key: str, epoch: int) -> str:
    require(key in metrics, f"epoch{epoch} missing metric {key}")
    try:
        value = Decimal(str(metrics[key]))
    except (InvalidOperation, ValueError) as exc:
        raise BinderError(f"epoch{epoch} invalid metric {key}") from exc
    require(value.is_finite(), f"epoch{epoch} non-finite metric {key}")
    return str(value)


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BinderError(f"invalid numeric field {label}") from exc
    require(math.isfinite(result), f"non-finite numeric field {label}")
    return result


def validate_profile(
    profile_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    policy: RunPolicy,
) -> dict[str, Any]:
    profile = strict_json(profile_path)
    require(profile.get("samples") == 825, f"epoch{epoch} samples must equal 825")

    identity = profile.get("artifact_identity")
    require(isinstance(identity, dict), f"epoch{epoch} missing artifact_identity")
    expected_identity = {
        "config_path": config["absolute_path"],
        "config_sha256": config["sha256"],
        "checkpoint_path": checkpoint["absolute_path"],
        "checkpoint_size": checkpoint["size_bytes"],
        "checkpoint_mtime_ns": checkpoint["mtime_ns"],
        "checkpoint_sha256": checkpoint["sha256"],
    }
    require(identity == expected_identity, f"epoch{epoch} artifact identity mismatch")

    audit = profile.get("checkpoint_load_audit")
    require(isinstance(audit, dict), f"epoch{epoch} missing checkpoint_load_audit")
    require(audit.get("checkpoint") == checkpoint["absolute_path"],
            f"epoch{epoch} checkpoint load path mismatch")
    for key in (
        "missing_count",
        "unexpected_count",
        "overlay_missing_count",
        "overlay_unexpected_count",
    ):
        require(audit.get(key) == 0, f"epoch{epoch} {key} must equal zero")
    require(audit.get("checkpoint_overlay_keys") == 210,
            f"epoch{epoch} checkpoint overlay count must equal 210")
    require(audit.get("model_overlay_keys") == 210,
            f"epoch{epoch} model overlay count must equal 210")

    counts = profile.get("module_counts")
    require(
        counts == {
            "ATLIFTernaryPSN": policy.atlif_modules,
            "ShiftmaxAttention": policy.attention_modules,
        },
        f"epoch{epoch} module counts must be exactly "
        f"{policy.atlif_modules} ATLIF and {policy.attention_modules} attention",
    )

    metrics = profile.get("metrics")
    require(isinstance(metrics, dict), f"epoch{epoch} missing metrics")
    metric_keys = (
        "AEE",
        "AAE",
        "AAE_Benchmark",
        "AEE_PE1",
        "AEE_PE2",
        "AEE_PE3",
        "AEE_outliers",
        "DSEC_Fl",
    )
    metric_row = {key: decimal_metric(metrics, key, epoch) for key in metric_keys}

    total_spikes = profile.get("total_spikes")
    require(isinstance(total_spikes, int) and not isinstance(total_spikes, bool)
            and total_spikes > 0, f"epoch{epoch} total_spikes must be a positive integer")
    firing = finite_float(profile.get("global_firing_rate"), f"epoch{epoch} firing")
    dense = finite_float(profile.get("dense_flops"), f"epoch{epoch} dense_flops")
    effective = finite_float(profile.get("effective_flops"), f"epoch{epoch} effective_flops")
    energy = finite_float(profile.get("energy_uj"), f"epoch{epoch} energy_uj")
    require(0.0 <= firing <= 1.0, f"epoch{epoch} firing outside [0,1]")
    require(dense > 0.0 and 0.0 <= effective <= dense,
            f"epoch{epoch} invalid dense/effective FLOPs")
    require(energy > 0.0, f"epoch{epoch} spike energy proxy must be positive")

    return {
        "epoch": epoch,
        "checkpoint": checkpoint,
        "profile": {
            **stable_identity(profile_path, f"epoch{epoch} spike profile"),
            "samples": 825,
            "artifact_identity_exact": True,
            "load_missing_count": 0,
            "load_unexpected_count": 0,
            "overlay_missing_count": 0,
            "overlay_unexpected_count": 0,
            "module_counts": counts,
        },
        "accuracy_metrics": metric_row,
        "activity": {
            "total_spikes": total_spikes,
            "global_firing_rate": firing,
            "dense_flops": dense,
            "effective_flops": effective,
            "effective_sparsity": 1.0 - effective / dense,
            "spike_energy_proxy_uj": energy,
            "energy_scope": "spike_activity_proxy_not_hardware_energy",
        },
    }


def validate_ranking(path: Path, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    regular_file(path, "valid825 ranking")
    text = path.read_text(encoding="utf-8")
    require(f"Ranking mode: `{mode}`." in text, f"ranking_mode must be {mode}")
    ranking_rows: list[tuple[int, int]] = []
    pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            ranking_rows.append((int(match.group(1)), int(match.group(2))))
    expected = sorted(
        rows,
        key=lambda row: (Decimal(row["accuracy_metrics"]["AEE"]), row["epoch"]),
    )
    expected_pairs = [(rank, row["epoch"]) for rank, row in enumerate(expected, 1)]
    require(ranking_rows == expected_pairs, "ranking rows do not match exact AEE order")
    return {**stable_identity(path, "valid825 ranking"), "ranking_mode": mode,
            "ordered_epochs": [row["epoch"] for row in expected]}


def rebind_targets() -> list[dict[str, Any]]:
    return [
        {"id": "E0", "target": "final checkpoint and deployment identity",
         "state_after_selection": "BOUND_BY_THIS_RECEIPT",
         "next_gate": "independent hammer must verify selection/config/profile/checkpoint identity"},
        {"id": "E1", "target": "standard plus dyadic/quantized/hardware-order valid825",
         "state_after_selection": "STANDARD_VALID825_BOUND__DEPLOYMENT_NUMERICS_INVALIDATED",
         "next_gate": "run dyadic/quantized and RTL-exact accuracy without retuning valid825"},
        {"id": "E2", "target": "unified ordered full-network capture",
         "state_after_selection": "INVALIDATED_RECAPTURE_REQUIRED",
         "next_gate": "single selected-checkpoint load; fixed C1 and decoder cohorts"},
        {"id": "E3", "target": "C1 four-Conv ledger and official baseline replay",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "selected-checkpoint 51.84M source-row same-ledger replay"},
        {"id": "E4", "target": "decoder D0-D3 payload, numeric miter and address cycles",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "decoder-complete selected-checkpoint payload and address replay"},
        {"id": "E5", "target": "ATLIF/FC/patch/BN activity and traffic",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "derive activity only from sealed E2 capture"},
        {"id": "E6", "target": "attention/RQTB NPZ, exact miter and Amdahl",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "selected-checkpoint Q/K capture and Fixed-RQTB replay"},
        {"id": "E7", "target": "real-trace SAIF/PTPX and decoder-complete Table A",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "E2-E6 complete, then real-trace VCS/SAIF/PTPX and same-resource join"},
        {"id": "E8", "target": "weight/range/compression re-admission",
         "state_after_selection": "INVALIDATED_REPLAY_REQUIRED",
         "next_gate": "selected-checkpoint export, overflow proof, encoding miter and fit"},
    ]


def build(policy: RunPolicy) -> dict[str, Any]:
    require(policy.ranking_mode == "aee", "only ranking_mode=aee is authorized")
    require(tuple(policy.epochs) == EPOCHS, "epoch population must be exactly 9/14/19/24/29")
    config = stable_identity(policy.config, "deployment config")
    require(config["sha256"] == policy.config_sha256, "deployment config SHA mismatch")
    require(config["absolute_path"] == str(policy.config.resolve()), "config path mismatch")

    rows: list[dict[str, Any]] = []
    for epoch in policy.epochs:
        checkpoint_path = policy.run_dir / f"checkpoint_epoch{epoch}.pth"
        checkpoint = stable_identity(checkpoint_path, f"epoch{epoch} checkpoint")
        profile_path = policy.run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
        rows.append(validate_profile(profile_path, checkpoint, config, epoch, policy))

    ranking = validate_ranking(policy.ranking, rows, policy.ranking_mode)
    selected = min(
        rows,
        key=lambda row: (Decimal(row["accuracy_metrics"]["AEE"]), row["epoch"]),
    )
    return {
        "schema": "m1163_motion_final_checkpoint_selection_rebind_binder_r1_v1",
        "status": "READY_FINAL_CHECKPOINT_SELECTION__HARDWARE_REBIND_NOT_AUTHORIZED",
        "run": {
            "absolute_path": str(policy.run_dir.resolve()),
            "label": "date_two_contribution_full30_20260826/c12_binary_motion_ttx",
            "predeclared_epochs": list(policy.epochs),
        },
        "selection_rule": {
            "ranking_mode": "aee",
            "primary": "minimum exact standard-valid825 AEE",
            "tie_break": "lowest epoch",
            "all_five_profiles_required": True,
            "valid825_reuse_for_retuning_forbidden": True,
        },
        "configuration": config,
        "ranking": ranking,
        "five_checkpoint_metric_table": rows,
        "selected": {
            "epoch": selected["epoch"],
            "checkpoint": selected["checkpoint"],
            "accuracy_metrics": selected["accuracy_metrics"],
            "activity": selected["activity"],
            "profile_sha256": selected["profile"]["sha256"],
        },
        "e0_e8_invalidation_and_rebind_targets": rebind_targets(),
        "claim_boundary": {
            "final_selection_bound": True,
            "standard_valid825_bound": True,
            "independent_hammer_required_before_hardware_rebind": True,
            "hardware_rebind_authorized": False,
            "hardware_replay_complete": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
            "checkpoint_copied": False,
            "gpu_started_by_binder": False,
            "eda_started_by_binder": False,
        },
    }


def write_receipt(output_dir: Path, result: dict[str, Any]) -> None:
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp_name:
        tmp = Path(tmp_name)
        selection = tmp / "final_checkpoint_selection.json"
        selection.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        csv_path = tmp / "five_checkpoint_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "epoch", "checkpoint_sha256", "checkpoint_size_bytes", "checkpoint_mtime_ns",
                "profile_sha256", "samples", "AEE", "AAE", "AAE_Benchmark", "AEE_PE1",
                "AEE_PE2", "AEE_PE3", "AEE_outliers", "DSEC_Fl", "total_spikes",
                "global_firing_rate", "dense_flops", "effective_flops", "effective_sparsity",
                "spike_energy_proxy_uj",
            ])
            for row in result["five_checkpoint_metric_table"]:
                checkpoint, profile = row["checkpoint"], row["profile"]
                metrics, activity = row["accuracy_metrics"], row["activity"]
                writer.writerow([
                    row["epoch"], checkpoint["sha256"], checkpoint["size_bytes"],
                    checkpoint["mtime_ns"], profile["sha256"], profile["samples"],
                    *[metrics[key] for key in (
                        "AEE", "AAE", "AAE_Benchmark", "AEE_PE1", "AEE_PE2",
                        "AEE_PE3", "AEE_outliers", "DSEC_Fl")],
                    activity["total_spikes"], activity["global_firing_rate"],
                    activity["dense_flops"], activity["effective_flops"],
                    activity["effective_sparsity"], activity["spike_energy_proxy_uj"],
                ])

        targets = tmp / "e0_e8_rebind_targets.json"
        targets.write_text(json.dumps(
            result["e0_e8_invalidation_and_rebind_targets"], indent=2, sort_keys=True
        ) + "\n", encoding="utf-8")
        complete = tmp / "RUN_COMPLETE.txt"
        complete.write_text(
            "PASS_M1163_FINAL_CHECKPOINT_SELECTED__INDEPENDENT_HAMMER_REQUIRED__"
            "NO_HARDWARE_REBIND_AUTHORITY\n",
            encoding="utf-8",
        )
        payloads = sorted((selection, csv_path, targets, complete), key=lambda path: path.name)
        manifest = tmp / "SHA256SUMS"
        manifest.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in payloads),
                            encoding="utf-8")
        seal = tmp / "SHA256SUMS.seal.sha256"
        seal.write_text(f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
        os.replace(tmp, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--ranking-mode", choices=("aee",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.run_dir == PRODUCTION_POLICY.run_dir, "production run path mismatch")
    require(args.config == PRODUCTION_POLICY.config, "production config path mismatch")
    require(args.ranking == PRODUCTION_POLICY.ranking, "production ranking path mismatch")
    require(args.ranking_mode == "aee", "production ranking_mode must be aee")
    result = build(PRODUCTION_POLICY)
    write_receipt(args.output_dir, result)
    print(
        "PASS_M1163_FINAL_CHECKPOINT_SELECTED__INDEPENDENT_HAMMER_REQUIRED__"
        "NO_HARDWARE_REBIND_AUTHORITY"
    )
    print(f"selected_epoch={result['selected']['epoch']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
