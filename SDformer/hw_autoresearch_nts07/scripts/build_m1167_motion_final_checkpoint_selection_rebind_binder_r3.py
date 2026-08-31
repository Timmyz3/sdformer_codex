#!/usr/bin/env python3
"""M1167 r3 checkpoint binder with canonical epoch-directory names.

This additive source-only hardening layer pins sealed M1166 r2 by SHA256 and
requires the raw entry-name set under standard_valid825 to be exactly the five
canonical names epoch9/epoch14/epoch19/epoch24/epoch29.  It therefore rejects
numeric aliases such as epoch09 before any integer parsing or selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
R2_SOURCE = HERE / "build_m1166_motion_final_checkpoint_selection_rebind_binder_r2.py"
R2_SOURCE_SHA256 = "2171da4909fc1844c1323ca5138ccc1232fdad61d3b00446709a144461d7472c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(R2_SOURCE) != R2_SOURCE_SHA256:
    raise RuntimeError("sealed M1166 r2 source identity drift")
_SPEC = importlib.util.spec_from_file_location("m1166_binder_sealed_r2", R2_SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load sealed M1166 r2 source")
R2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = R2
_SPEC.loader.exec_module(R2)
R1 = R2.R1
_R2_VALIDATE_PROFILE = R2.validate_profile_r2


CANONICAL_EPOCH_ENTRY_NAMES = frozenset(f"epoch{epoch}" for epoch in R1.EPOCHS)


def exact_nonbool_int(value: Any, expected: int, label: str) -> None:
    R1.require(type(value) is int and value == expected,
               f"{label} must be exact non-bool int {expected}")


def positive_nonbool_int(value: Any, label: str) -> None:
    R1.require(type(value) is int and value > 0,
               f"{label} must be an exact positive non-bool int")


def validate_profile_r3(
    profile_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    policy: Any,
) -> dict[str, Any]:
    profile = R1.strict_json(profile_path)
    exact_nonbool_int(profile.get("samples"), 825, f"epoch{epoch} samples")

    identity = profile.get("artifact_identity")
    R1.require(isinstance(identity, dict), f"epoch{epoch} missing artifact_identity")
    identity_keys = {
        "config_path", "config_sha256", "checkpoint_path", "checkpoint_size",
        "checkpoint_mtime_ns", "checkpoint_sha256",
    }
    R1.require(set(identity) == identity_keys, f"epoch{epoch} artifact identity keys mismatch")
    for key in ("config_path", "config_sha256", "checkpoint_path", "checkpoint_sha256"):
        R1.require(type(identity.get(key)) is str and bool(identity[key]),
                   f"epoch{epoch} artifact {key} must be a nonempty string")
    positive_nonbool_int(identity.get("checkpoint_size"),
                         f"epoch{epoch} artifact checkpoint_size")
    positive_nonbool_int(identity.get("checkpoint_mtime_ns"),
                         f"epoch{epoch} artifact checkpoint_mtime_ns")

    audit = profile.get("checkpoint_load_audit")
    R1.require(isinstance(audit, dict), f"epoch{epoch} missing checkpoint_load_audit")
    exact_nonbool_int(audit.get("checkpoint_overlay_keys"), 210,
                      f"epoch{epoch} checkpoint_overlay_keys")
    exact_nonbool_int(audit.get("model_overlay_keys"), 210,
                      f"epoch{epoch} model_overlay_keys")

    counts = profile.get("module_counts")
    R1.require(isinstance(counts, dict) and set(counts) == {
        "ATLIFTernaryPSN", "ShiftmaxAttention"
    }, f"epoch{epoch} module count keys mismatch")
    exact_nonbool_int(counts.get("ATLIFTernaryPSN"), policy.atlif_modules,
                      f"epoch{epoch} ATLIFTernaryPSN count")
    exact_nonbool_int(counts.get("ShiftmaxAttention"), policy.attention_modules,
                      f"epoch{epoch} ShiftmaxAttention count")
    return _R2_VALIDATE_PROFILE(profile_path, checkpoint, config, epoch, policy)


def exact_profile_population_r3(policy: Any) -> None:
    standard = policy.run_dir / "standard_valid825"
    try:
        mode = standard.lstat().st_mode
    except FileNotFoundError as exc:
        raise R1.BinderError(f"missing standard_valid825 directory: {standard}") from exc
    R1.require(stat.S_ISDIR(mode) and not standard.is_symlink(),
               f"standard_valid825 must be a non-symlink directory: {standard}")
    entries = list(standard.iterdir())
    observed_names = {entry.name for entry in entries}
    R1.require(
        observed_names == CANONICAL_EPOCH_ENTRY_NAMES and len(entries) == len(CANONICAL_EPOCH_ENTRY_NAMES),
        "standard_valid825 raw entry-name population mismatch: " + repr(sorted(observed_names)),
    )
    for entry in entries:
        entry_mode = entry.lstat().st_mode
        R1.require(stat.S_ISDIR(entry_mode) and not entry.is_symlink(),
                   f"canonical epoch entry must be a non-symlink directory: {entry}")


def build(policy: Any) -> dict[str, Any]:
    original_population = R2.exact_profile_population
    original_profile = R2.validate_profile_r2
    R2.exact_profile_population = exact_profile_population_r3
    R2.validate_profile_r2 = validate_profile_r3
    try:
        result = R2.build(policy)
    finally:
        R2.exact_profile_population = original_population
        R2.validate_profile_r2 = original_profile
    result["schema"] = "m1167_motion_final_checkpoint_selection_rebind_binder_r3_v1"
    result["status"] = (
        "READY_FINAL_CHECKPOINT_SELECTION_R3_CANONICAL_EPOCH_NAMES__"
        "HARDWARE_REBIND_NOT_AUTHORIZED"
    )
    result["source_hardening"]["revision"] = "r3"
    result["source_hardening"]["sealed_r2_dependency_sha256"] = R2_SOURCE_SHA256
    result["source_hardening"]["canonical_epoch_entry_names"] = sorted(
        CANONICAL_EPOCH_ENTRY_NAMES, key=lambda value: int(value[5:])
    )
    result["source_hardening"]["raw_entry_name_set_must_be_exact"] = True
    result["source_hardening"]["schema_exact_nonbool_int_fields"] = {
        "samples": 825,
        "checkpoint_overlay_keys": 210,
        "model_overlay_keys": 210,
        "ATLIFTernaryPSN": policy.atlif_modules,
        "ShiftmaxAttention": policy.attention_modules,
        "checkpoint_size": "positive",
        "checkpoint_mtime_ns": "positive",
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
            "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
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
    policy = R1.PRODUCTION_POLICY
    R1.require(args.run_dir == policy.run_dir, "production run path mismatch")
    R1.require(args.config == policy.config, "production config path mismatch")
    R1.require(args.ranking == policy.ranking, "production ranking path mismatch")
    R1.require(args.ranking_mode == "aee", "production ranking_mode must be aee")
    result = build(policy)
    write_receipt(args.output_dir, result)
    print(
        "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
        "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"
    )
    print(f"selected_epoch={result['selected']['epoch']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
