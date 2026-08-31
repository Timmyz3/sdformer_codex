#!/usr/bin/env python3
"""M1228 lazy cross-run Motion checkpoint selector and hardware rebind binder.

This source deliberately models four candidates across two run directories and
two configuration identities.  Importing it performs no filesystem access.
Only an explicit future invocation of ``build(PRODUCTION_POLICY)`` or ``main``
reads candidate artifacts.  The program never imports the model, runs valid825,
uses a GPU, contacts a remote host, copies a checkpoint, or launches EDA.

Admission requires all four predeclared candidates to exist and to carry exact
825-sample profiles, zero typed load-audit counters, 105 ATLIF and 12 attention
modules, and exact profile-to-checkpoint/config path/SHA/size/mtime bindings.
Selection is minimum exact AEE with the lower epoch as the sole tie break.
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
import stat
import tempfile
from typing import Any


REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
OLD_RUN = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx"
)
OLD_CONFIG = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
)
OLD_CONFIG_SHA256 = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
OLD_EP29_CHECKPOINT_SHA256 = (
    "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a"
)

NEW_RUN = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_c12_alpha0125_ep29_resume5_20260830"
)
NEW_CONFIG = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.yml"
)
NEW_CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
NEW_MANIFEST = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.json"
)
NEW_EVALUATION_EPOCHS = (30, 32, 34)

LOAD_AUDIT_ZERO_KEYS = (
    "missing_count",
    "unexpected_count",
    "overlay_missing_count",
    "overlay_unexpected_count",
)
METRIC_KEYS = (
    "AEE",
    "AAE",
    "AAE_Benchmark",
    "AEE_PE1",
    "AEE_PE2",
    "AEE_PE3",
    "AEE_outliers",
    "DSEC_Fl",
)
ARTIFACT_IDENTITY_KEYS = {
    "config_path",
    "config_sha256",
    "checkpoint_path",
    "checkpoint_size",
    "checkpoint_mtime_ns",
    "checkpoint_sha256",
}


class BinderError(RuntimeError):
    """A fail-closed input, identity, or provenance error."""


@dataclass(frozen=True)
class CandidatePolicy:
    candidate_id: str
    run_dir: Path
    config: Path
    config_sha256: str
    epoch: int
    expected_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class CrossRunPolicy:
    candidates: tuple[CandidatePolicy, ...]
    new_run_manifest: Path
    new_evaluation_epochs: tuple[int, ...]
    ranking_mode: str = "aee"
    atlif_modules: int = 105
    attention_modules: int = 12


PRODUCTION_POLICY = CrossRunPolicy(
    candidates=(
        CandidatePolicy(
            "legacy_ep29", OLD_RUN, OLD_CONFIG, OLD_CONFIG_SHA256, 29,
            OLD_EP29_CHECKPOINT_SHA256,
        ),
        CandidatePolicy("resume_ep30", NEW_RUN, NEW_CONFIG, NEW_CONFIG_SHA256, 30),
        CandidatePolicy("resume_ep32", NEW_RUN, NEW_CONFIG, NEW_CONFIG_SHA256, 32),
        CandidatePolicy("resume_ep34", NEW_RUN, NEW_CONFIG, NEW_CONFIG_SHA256, 34),
    ),
    new_run_manifest=NEW_MANIFEST,
    new_evaluation_epochs=NEW_EVALUATION_EPOCHS,
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


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise BinderError(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(mode), f"not a regular file for {label}: {path}")
    require(not path.is_symlink(), f"symlink forbidden for {label}: {path}")


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
    require(after.st_size > 0 and after.st_mtime_ns > 0, f"invalid identity for {label}")
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def exact_int(value: Any, expected: int, label: str) -> None:
    require(type(value) is int and value == expected,
            f"{label} must be exact non-bool int {expected}")


def decimal_metric(metrics: dict[str, Any], key: str, epoch: int) -> str:
    require(key in metrics, f"epoch{epoch} missing metric {key}")
    require(type(metrics[key]) in (int, float, str), f"epoch{epoch} invalid metric type {key}")
    try:
        value = Decimal(str(metrics[key]))
    except (InvalidOperation, ValueError) as exc:
        raise BinderError(f"epoch{epoch} invalid metric {key}") from exc
    require(value.is_finite(), f"epoch{epoch} non-finite metric {key}")
    return str(value)


def finite_float(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    require(math.isfinite(result), f"non-finite numeric field {label}")
    return result


def validate_policy(policy: CrossRunPolicy) -> None:
    require(policy.ranking_mode == "aee", "only ranking_mode=aee is permitted")
    expected_ids = ("legacy_ep29", "resume_ep30", "resume_ep32", "resume_ep34")
    require(tuple(candidate.candidate_id for candidate in policy.candidates) == expected_ids,
            "candidate id/order population must be exact")
    require(tuple(candidate.epoch for candidate in policy.candidates) == (29, 30, 32, 34),
            "candidate epoch population must be exactly 29/30/32/34")
    require(len({candidate.candidate_id for candidate in policy.candidates}) == 4,
            "candidate ids must be unique")
    require(len({candidate.epoch for candidate in policy.candidates}) == 4,
            "candidate epochs must be unique")
    legacy, *resume = policy.candidates
    require(all(candidate.run_dir == legacy.run_dir for candidate in (legacy,)),
            "legacy run policy malformed")
    require(all(candidate.run_dir == resume[0].run_dir for candidate in resume),
            "all resume candidates must share the new run directory")
    require(legacy.run_dir != resume[0].run_dir, "old and new run directories must differ")
    require(all(candidate.config == resume[0].config for candidate in resume),
            "all resume candidates must share the new config")
    require(legacy.config != resume[0].config, "old and new configs must differ")
    require(tuple(policy.new_evaluation_epochs) == NEW_EVALUATION_EPOCHS,
            "new evaluation epochs must be exactly 30/32/34")
    require(policy.atlif_modules == 105 and policy.attention_modules == 12,
            "topology policy must be exactly 105 ATLIF and 12 attention")


def validate_new_manifest(policy: CrossRunPolicy) -> dict[str, Any]:
    value = strict_json(policy.new_run_manifest)
    observed = value.get("evaluation_epochs")
    require(isinstance(observed, list), "new manifest evaluation_epochs must be a list")
    require(all(type(epoch) is int for epoch in observed),
            "new manifest evaluation_epochs must contain exact non-bool integers")
    require(tuple(observed) == tuple(policy.new_evaluation_epochs),
            "new manifest evaluation_epochs must be exactly [30,32,34]")
    return stable_identity(policy.new_run_manifest, "new run manifest")


def expected_profile_identity(
    checkpoint: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config_path": config["absolute_path"],
        "config_sha256": config["sha256"],
        "checkpoint_path": checkpoint["absolute_path"],
        "checkpoint_size": checkpoint["size_bytes"],
        "checkpoint_mtime_ns": checkpoint["mtime_ns"],
        "checkpoint_sha256": checkpoint["sha256"],
    }


def validate_profile(
    candidate: CandidatePolicy,
    profile_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    policy: CrossRunPolicy,
) -> dict[str, Any]:
    epoch = candidate.epoch
    profile = strict_json(profile_path)
    exact_int(profile.get("samples"), 825, f"epoch{epoch} samples")

    identity = profile.get("artifact_identity")
    require(isinstance(identity, dict), f"epoch{epoch} missing artifact_identity")
    require(set(identity) == ARTIFACT_IDENTITY_KEYS,
            f"epoch{epoch} artifact identity keys mismatch")
    require(identity == expected_profile_identity(checkpoint, config),
            f"epoch{epoch} artifact identity mismatch")

    audit = profile.get("checkpoint_load_audit")
    require(isinstance(audit, dict), f"epoch{epoch} missing checkpoint_load_audit")
    require(audit.get("checkpoint") == checkpoint["absolute_path"],
            f"epoch{epoch} checkpoint load path mismatch")
    for key in LOAD_AUDIT_ZERO_KEYS:
        exact_int(audit.get(key), 0, f"epoch{epoch} {key}")

    counts = profile.get("module_counts")
    require(isinstance(counts, dict) and set(counts) == {
        "ATLIFTernaryPSN", "ShiftmaxAttention"
    }, f"epoch{epoch} module count keys mismatch")
    exact_int(counts.get("ATLIFTernaryPSN"), policy.atlif_modules,
              f"epoch{epoch} ATLIFTernaryPSN count")
    exact_int(counts.get("ShiftmaxAttention"), policy.attention_modules,
              f"epoch{epoch} ShiftmaxAttention count")

    metrics = profile.get("metrics")
    require(isinstance(metrics, dict), f"epoch{epoch} missing metrics")
    metric_row = {key: decimal_metric(metrics, key, epoch) for key in METRIC_KEYS}

    total_spikes = profile.get("total_spikes")
    require(type(total_spikes) is int and total_spikes > 0,
            f"epoch{epoch} total_spikes must be positive exact integer")
    firing = finite_float(profile.get("global_firing_rate"), f"epoch{epoch} firing")
    dense = finite_float(profile.get("dense_flops"), f"epoch{epoch} dense_flops")
    effective = finite_float(profile.get("effective_flops"), f"epoch{epoch} effective_flops")
    energy = finite_float(profile.get("energy_uj"), f"epoch{epoch} energy_uj")
    require(0.0 <= firing <= 1.0, f"epoch{epoch} firing outside [0,1]")
    require(dense > 0.0 and 0.0 <= effective <= dense,
            f"epoch{epoch} invalid dense/effective FLOPs")
    require(energy > 0.0, f"epoch{epoch} energy proxy must be positive")

    return {
        "candidate_id": candidate.candidate_id,
        "epoch": epoch,
        "run_directory": str(candidate.run_dir.resolve()),
        "checkpoint": checkpoint,
        "configuration": config,
        "profile": {
            **stable_identity(profile_path, f"epoch{epoch} spike profile"),
            "samples": 825,
            "artifact_identity_exact": True,
            "load_audit_exact_zero": True,
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


def activation_rebind_targets() -> list[dict[str, Any]]:
    common = {
        "dependency": "selected checkpoint SHA/size/mtime + selected config SHA/size/mtime",
        "reuse_rule": (
            "reuse only when an independently sealed artifact binds the exact selected "
            "checkpoint and config identities; otherwise invalidate and regenerate"
        ),
    }
    rows = [
        ("E0", "final checkpoint/config/profile selection identity",
         "BOUND_BY_BINDER_AFTER_INDEPENDENT_RESULT_HAMMER"),
        ("E1", "standard plus dyadic/quantized/hardware-order valid825",
         "STANDARD_PROFILE_BOUND__DEPLOYMENT_NUMERICS_IDENTITY_CONDITIONAL"),
        ("E2", "unified ordered full-network activation capture",
         "ACTIVATION_IDENTITY_CONDITIONAL_RECAPTURE"),
        ("E3", "C1 Conv ledger and official-baseline replay",
         "ACTIVATION_AND_WEIGHT_IDENTITY_CONDITIONAL_REPLAY"),
        ("E4", "decoder D0-D3 payload, numeric miter and address cycles",
         "ACTIVATION_AND_WEIGHT_IDENTITY_CONDITIONAL_REPLAY"),
        ("E5", "ATLIF/FC/patch/BN activity, traffic and range",
         "ACTIVATION_IDENTITY_CONDITIONAL_REPLAY"),
        ("E6", "attention/RQTB Q/K/gate capture, miter and Amdahl",
         "ACTIVATION_IDENTITY_CONDITIONAL_REPLAY"),
        ("E7", "real-trace VCS/SAIF/PTPX and decoder-complete system table",
         "TRANSITIVE_E2_E6_IDENTITY_CONDITIONAL_REPLAY"),
        ("E8", "weight export, numeric range, compression and macro-fit admission",
         "CHECKPOINT_AND_CONFIG_IDENTITY_CONDITIONAL_REBIND"),
    ]
    return [
        {"id": identifier, "target": target, "state_after_selection": state, **common}
        for identifier, target, state in rows
    ]


def build(policy: CrossRunPolicy) -> dict[str, Any]:
    validate_policy(policy)
    manifest = validate_new_manifest(policy)

    config_cache: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in policy.candidates:
        if candidate.config not in config_cache:
            identity = stable_identity(candidate.config, f"{candidate.candidate_id} config")
            require(identity["sha256"] == candidate.config_sha256,
                    f"{candidate.candidate_id} config SHA mismatch")
            config_cache[candidate.config] = identity
        config = config_cache[candidate.config]
        checkpoint_path = candidate.run_dir / f"checkpoint_epoch{candidate.epoch}.pth"
        checkpoint = stable_identity(checkpoint_path, f"{candidate.candidate_id} checkpoint")
        if candidate.expected_checkpoint_sha256 is not None:
            require(checkpoint["sha256"] == candidate.expected_checkpoint_sha256,
                    f"{candidate.candidate_id} checkpoint SHA mismatch")
        profile_path = (
            candidate.run_dir / "standard_valid825" /
            f"epoch{candidate.epoch}" / "spike_profile.json"
        )
        rows.append(validate_profile(candidate, profile_path, checkpoint, config, policy))

    selected = min(
        rows,
        key=lambda row: (Decimal(row["accuracy_metrics"]["AEE"]), row["epoch"]),
    )
    targets = activation_rebind_targets()
    return {
        "schema": "m1228_motion_cross_run_final_checkpoint_rebind_binder_source_r1_v1",
        "status": (
            "READY_CROSS_RUN_SELECTION__INDEPENDENT_RESULT_HAMMER_REQUIRED__"
            "HARDWARE_REBIND_NOT_AUTHORIZED"
        ),
        "new_run_manifest": manifest,
        "candidate_population": rows,
        "selection_rule": {
            "candidate_ids": [candidate.candidate_id for candidate in policy.candidates],
            "epochs": [candidate.epoch for candidate in policy.candidates],
            "primary": "minimum exact standard-valid825 AEE",
            "tie_break": "lowest epoch",
            "all_four_candidates_required": True,
            "cross_run": True,
            "cross_config": True,
            "valid825_reuse_for_retuning_forbidden": True,
        },
        "selected": {
            "candidate_id": selected["candidate_id"],
            "epoch": selected["epoch"],
            "run_directory": selected["run_directory"],
            "checkpoint": selected["checkpoint"],
            "configuration": selected["configuration"],
            "profile": selected["profile"],
            "accuracy_metrics": selected["accuracy_metrics"],
            "activity": selected["activity"],
        },
        "e0_e8_activation_dependent_invalidation_and_rebind_targets": targets,
        "claim_boundary": {
            "source_only_authoring_package": False,
            "selection_bound_after_execution": True,
            "independent_result_hammer_required": True,
            "hardware_rebind_authorized": False,
            "hardware_replay_complete": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
            "checkpoint_copied": False,
            "gpu_started_by_binder": False,
            "remote_access_by_binder": False,
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

        selected_identity = tmp / "selected_checkpoint_and_config.json"
        selected_identity.write_text(json.dumps({
            "schema": "m1228_selected_checkpoint_and_config_r1_v1",
            "candidate_id": result["selected"]["candidate_id"],
            "epoch": result["selected"]["epoch"],
            "run_directory": result["selected"]["run_directory"],
            "checkpoint": result["selected"]["checkpoint"],
            "configuration": result["selected"]["configuration"],
            "profile": result["selected"]["profile"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        csv_path = tmp / "four_checkpoint_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "candidate_id", "epoch", "run_directory", "config_sha256",
                "config_size_bytes", "config_mtime_ns", "checkpoint_sha256",
                "checkpoint_size_bytes", "checkpoint_mtime_ns", "profile_sha256",
                "profile_size_bytes", "profile_mtime_ns", "samples", *METRIC_KEYS,
                "total_spikes", "global_firing_rate", "dense_flops", "effective_flops",
                "effective_sparsity", "spike_energy_proxy_uj",
            ])
            for row in result["candidate_population"]:
                config, checkpoint, profile = (
                    row["configuration"], row["checkpoint"], row["profile"]
                )
                writer.writerow([
                    row["candidate_id"], row["epoch"], row["run_directory"],
                    config["sha256"], config["size_bytes"], config["mtime_ns"],
                    checkpoint["sha256"], checkpoint["size_bytes"], checkpoint["mtime_ns"],
                    profile["sha256"], profile["size_bytes"], profile["mtime_ns"],
                    profile["samples"],
                    *[row["accuracy_metrics"][key] for key in METRIC_KEYS],
                    row["activity"]["total_spikes"], row["activity"]["global_firing_rate"],
                    row["activity"]["dense_flops"], row["activity"]["effective_flops"],
                    row["activity"]["effective_sparsity"],
                    row["activity"]["spike_energy_proxy_uj"],
                ])

        targets = tmp / "e0_e8_activation_rebind_targets.json"
        targets.write_text(json.dumps(
            result["e0_e8_activation_dependent_invalidation_and_rebind_targets"],
            indent=2,
            sort_keys=True,
        ) + "\n", encoding="utf-8")

        complete = tmp / "RUN_COMPLETE.txt"
        complete.write_text(
            "PASS_M1228_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
            encoding="utf-8",
        )
        payloads = sorted(
            (selection, selected_identity, csv_path, targets, complete),
            key=lambda path: path.name,
        )
        manifest_path = tmp / "SHA256SUMS"
        manifest_path.write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in payloads),
            encoding="utf-8",
        )
        seal = tmp / "SHA256SUMS.seal.sha256"
        seal.write_text(f"{sha256(manifest_path)}  SHA256SUMS\n", encoding="utf-8")
        os.replace(tmp, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-mode", choices=("aee",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.ranking_mode == "aee", "production ranking mode must be aee")
    result = build(PRODUCTION_POLICY)
    write_receipt(args.output_dir, result)
    print(
        "PASS_M1228_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
        "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"
    )
    print(f"selected_candidate={result['selected']['candidate_id']}")
    print(f"selected_epoch={result['selected']['epoch']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
