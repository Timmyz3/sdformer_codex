#!/usr/bin/env python3
"""M1166 r2 final-checkpoint binder with exact typed-zero load audits.

This is a source-only hardening wrapper around the sealed M1163 r1 builder.
It pins that dependency by SHA256, rejects JSON booleans/floats/strings for all
four zero-valued load-audit counters, and preserves every other r1 gate.  It
does not evaluate a model, start or interrupt a GPU process, copy a checkpoint,
capture a profile, launch hardware replay, or invoke EDA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
R1_SOURCE = HERE / "build_m1163_motion_final_checkpoint_selection_rebind_binder.py"
R1_SOURCE_SHA256 = "50d22cb0f7d656c79eeb99894cb85c975441f16fd46d7df55c37ff34976aaf32"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(R1_SOURCE) != R1_SOURCE_SHA256:
    raise RuntimeError("sealed M1163 r1 source identity drift")
_SPEC = importlib.util.spec_from_file_location("m1163_binder_sealed_r1", R1_SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load sealed M1163 r1 source")
R1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = R1
_SPEC.loader.exec_module(R1)
_R1_VALIDATE_PROFILE = R1.validate_profile
_R1_VALIDATE_RANKING = R1.validate_ranking


LOAD_AUDIT_ZERO_KEYS = (
    "missing_count",
    "unexpected_count",
    "overlay_missing_count",
    "overlay_unexpected_count",
)
EPOCH_DIRECTORY = re.compile(r"epoch([0-9]+)")
RANKING_MODE_LINE = re.compile(r"^Ranking mode: `([^`]+)`\.$")


def exact_nonbool_int_zero(value: Any, label: str) -> None:
    R1.require(type(value) is int and value == 0, f"{label} must be exact non-bool int zero")


def validate_profile_r2(
    profile_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    policy: Any,
) -> dict[str, Any]:
    profile = R1.strict_json(profile_path)
    audit = profile.get("checkpoint_load_audit")
    R1.require(isinstance(audit, dict), f"epoch{epoch} missing checkpoint_load_audit")
    for key in LOAD_AUDIT_ZERO_KEYS:
        exact_nonbool_int_zero(audit.get(key), f"epoch{epoch} {key}")
    return _R1_VALIDATE_PROFILE(profile_path, checkpoint, config, epoch, policy)


def exact_profile_population(policy: Any) -> None:
    standard = policy.run_dir / "standard_valid825"
    try:
        mode = standard.lstat().st_mode
    except FileNotFoundError as exc:
        raise R1.BinderError(f"missing standard_valid825 directory: {standard}") from exc
    R1.require(stat.S_ISDIR(mode) and not standard.is_symlink(),
               f"standard_valid825 must be a non-symlink directory: {standard}")
    observed: set[int] = set()
    for entry in standard.iterdir():
        match = EPOCH_DIRECTORY.fullmatch(entry.name)
        if match is None:
            continue
        R1.require(entry.is_dir() and not entry.is_symlink(),
                   f"epoch profile entry must be a non-symlink directory: {entry}")
        observed.add(int(match.group(1)))
    R1.require(observed == set(policy.epochs),
               f"standard_valid825 epoch directory population mismatch: {sorted(observed)}")


def validate_ranking_r2(path: Path, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    R1.regular_file(path, "valid825 ranking")
    declarations: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RANKING_MODE_LINE.fullmatch(line)
        if match is not None:
            declarations.append(match.group(1))
    R1.require(declarations == ["aee"],
               f"ranking must contain exactly one anchored aee declaration: {declarations}")
    R1.require(mode == "aee", "ranking validation mode must be aee")
    return _R1_VALIDATE_RANKING(path, rows, mode)


def build(policy: Any) -> dict[str, Any]:
    exact_profile_population(policy)
    original_profile = R1.validate_profile
    original_ranking = R1.validate_ranking
    R1.validate_profile = validate_profile_r2
    R1.validate_ranking = validate_ranking_r2
    try:
        result = R1.build(policy)
    finally:
        R1.validate_profile = original_profile
        R1.validate_ranking = original_ranking
    result["schema"] = "m1166_motion_final_checkpoint_selection_rebind_binder_r2_v1"
    result["status"] = (
        "READY_FINAL_CHECKPOINT_SELECTION_R2_TYPED_ZERO__"
        "HARDWARE_REBIND_NOT_AUTHORIZED"
    )
    result["source_hardening"] = {
        "revision": "r2",
        "sealed_r1_dependency_sha256": R1_SOURCE_SHA256,
        "typed_zero_rule": "type(value) is int and value == 0",
        "protected_fields": list(LOAD_AUDIT_ZERO_KEYS),
        "json_false_rejected": True,
        "json_true_rejected": True,
        "json_string_zero_rejected": True,
        "json_float_zero_rejected": True,
        "exact_epoch_profile_directory_population": list(policy.epochs),
        "exact_anchored_ranking_mode_declarations": ["aee"],
    }
    return result


def write_receipt(output_dir: Path, result: dict[str, Any]) -> None:
    R1.require(not output_dir.exists(), f"output directory already exists: {output_dir}")
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
            "PASS_M1166_FINAL_CHECKPOINT_SELECTED_R2_TYPED_ZERO__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
            encoding="utf-8",
        )
        payloads = sorted((selection, csv_path, targets, complete), key=lambda path: path.name)
        manifest = tmp / "SHA256SUMS"
        manifest.write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in payloads),
                            encoding="utf-8")
        seal = tmp / "SHA256SUMS.seal.sha256"
        seal.write_text(f"{_sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")
        os.replace(tmp, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--ranking-mode", choices=("aee",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    R1.require(args.run_dir == R1.PRODUCTION_POLICY.run_dir, "production run path mismatch")
    R1.require(args.config == R1.PRODUCTION_POLICY.config, "production config path mismatch")
    R1.require(args.ranking == R1.PRODUCTION_POLICY.ranking, "production ranking path mismatch")
    R1.require(args.ranking_mode == "aee", "production ranking_mode must be aee")
    result = build(R1.PRODUCTION_POLICY)
    write_receipt(args.output_dir, result)
    print(
        "PASS_M1166_FINAL_CHECKPOINT_SELECTED_R2_TYPED_ZERO__"
        "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"
    )
    print(f"selected_epoch={result['selected']['epoch']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
