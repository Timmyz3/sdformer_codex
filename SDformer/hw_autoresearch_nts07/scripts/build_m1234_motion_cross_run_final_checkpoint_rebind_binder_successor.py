#!/usr/bin/env python3
"""M1234 additive M1228 successor with immutable profile-byte binding.

Import is inert.  An explicit build lazily loads the sealed M1228 substrate,
retains its exact-four/two-run/two-config policy and E0--E8 semantics, but
parses and hashes every profile from one immutable byte snapshot.  All optical
flow error metrics must be finite and nonnegative.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PREDECESSOR = Path(__file__).with_name(
    "build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py")
PREDECESSOR_SHA256 = "9b2b43b4d36ed64741cbb39db0d9f5d75eb7bec09b00f4e496f3d52ce3ae5efe"
REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
OLD_RUN = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx")
NEW_RUN = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_c12_alpha0125_ep29_resume5_20260830")
OLD_CONFIG = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml")
NEW_CONFIG = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
NEW_MANIFEST = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.json")
OLD_CONFIG_SHA256 = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
NEW_CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
OLD_EP29_CHECKPOINT_SHA256 = (
    "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a")
NEW_EVALUATION_EPOCHS = (30, 32, 34)
LOAD_AUDIT_ZERO_KEYS = (
    "missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count")
ERROR_METRIC_KEYS = (
    "AEE", "AAE", "AAE_Benchmark", "AEE_PE1", "AEE_PE2", "AEE_PE3",
    "AEE_outliers", "DSEC_Fl")
ARTIFACT_IDENTITY_KEYS = {
    "config_path", "config_sha256", "checkpoint_path", "checkpoint_size",
    "checkpoint_mtime_ns", "checkpoint_sha256"}


class BinderError(RuntimeError):
    pass


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
        CandidatePolicy("legacy_ep29", OLD_RUN, OLD_CONFIG, OLD_CONFIG_SHA256, 29,
                        OLD_EP29_CHECKPOINT_SHA256),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predecessor() -> Any:
    require(PREDECESSOR.is_file() and not PREDECESSOR.is_symlink(),
            "sealed M1228 predecessor must be a regular non-symlink file")
    require(sha256(PREDECESSOR) == PREDECESSOR_SHA256, "M1228 predecessor SHA drift")
    spec = importlib.util.spec_from_file_location("m1234_sealed_m1228", str(PREDECESSOR))
    require(spec is not None and spec.loader is not None, "cannot import M1228 predecessor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BinderError("non-finite JSON constant: " + value)


def _identity_tuple(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)


def immutable_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and parse one read; reject link/type/identity drift around that read."""
    path = Path(path)
    try:
        path_before = path.lstat()
    except FileNotFoundError as exc:
        raise BinderError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(path_before.st_mode) and not path.is_symlink(),
            "{} must be a regular non-symlink file: {}".format(label, path))
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise BinderError("cannot securely open {}: {}".format(label, path)) from exc
    try:
        fd_before = os.fstat(descriptor)
        require(stat.S_ISREG(fd_before.st_mode), label + " descriptor is not regular")
        require(_identity_tuple(path_before) == _identity_tuple(fd_before),
                label + " path/descriptor identity mismatch before read")
        blocks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as exc:
        raise BinderError(label + " disappeared during immutable read") from exc
    require(not path.is_symlink() and stat.S_ISREG(path_after.st_mode),
            label + " became symlink/non-regular during read")
    identity = _identity_tuple(fd_before)
    require(identity == _identity_tuple(fd_after) == _identity_tuple(path_after),
            label + " changed during immutable read")
    require(len(payload) == fd_after.st_size and len(payload) > 0,
            label + " byte population mismatch")
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_no_duplicate,
                           parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BinderError("invalid JSON {}: {}".format(path, exc)) from exc
    require(isinstance(value, dict), label + " JSON root must be an object")
    return value, {
        "absolute_path": str(path.resolve()),
        "size_bytes": fd_after.st_size,
        "mtime_ns": fd_after.st_mtime_ns,
        "device": fd_after.st_dev,
        "inode": fd_after.st_ino,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "immutable_single_read": True,
        "hash_and_parse_same_bytes": True,
    }


def exact_int(value: Any, expected: int, label: str) -> None:
    require(type(value) is int and value == expected,
            "{} must be exact non-bool int {}".format(label, expected))


def nonnegative_decimal_metric(metrics: dict[str, Any], key: str, epoch: int) -> str:
    require(key in metrics, "epoch{} missing metric {}".format(epoch, key))
    require(type(metrics[key]) in (int, float, str),
            "epoch{} invalid metric type {}".format(epoch, key))
    try:
        value = Decimal(str(metrics[key]))
    except Exception as exc:
        raise BinderError("epoch{} invalid metric {}".format(epoch, key)) from exc
    require(value.is_finite(), "epoch{} non-finite metric {}".format(epoch, key))
    require(value >= 0, "epoch{} negative error metric {}".format(epoch, key))
    return str(value)


def finite_float(value: Any, label: str) -> float:
    require(type(value) in (int, float), label + " must be a JSON number")
    result = float(value)
    require(math.isfinite(result), "non-finite numeric field " + label)
    return result


def validate_policy(policy: CrossRunPolicy) -> None:
    require(policy.ranking_mode == "aee", "only ranking_mode=aee is permitted")
    require(tuple(row.candidate_id for row in policy.candidates) ==
            ("legacy_ep29", "resume_ep30", "resume_ep32", "resume_ep34"),
            "candidate id/order population must be exact")
    require(tuple(row.epoch for row in policy.candidates) == (29, 30, 32, 34),
            "candidate epochs must be exactly 29/30/32/34")
    legacy, *resume = policy.candidates
    require(all(row.run_dir == resume[0].run_dir for row in resume) and
            legacy.run_dir != resume[0].run_dir, "two-run topology mismatch")
    require(all(row.config == resume[0].config for row in resume) and
            legacy.config != resume[0].config, "two-config topology mismatch")
    require(tuple(policy.new_evaluation_epochs) == NEW_EVALUATION_EPOCHS,
            "new evaluation epochs must be 30/32/34")
    require(policy.atlif_modules == 105 and policy.attention_modules == 12,
            "module topology must be 105/12")


def validate_profile(candidate: CandidatePolicy, profile_path: Path,
                     checkpoint: dict[str, Any], config: dict[str, Any],
                     policy: CrossRunPolicy) -> dict[str, Any]:
    profile, profile_identity = immutable_json_snapshot(
        profile_path, "epoch{} spike profile".format(candidate.epoch))
    epoch = candidate.epoch
    exact_int(profile.get("samples"), 825, "epoch{} samples".format(epoch))
    identity = profile.get("artifact_identity")
    require(isinstance(identity, dict) and set(identity) == ARTIFACT_IDENTITY_KEYS,
            "epoch{} artifact identity keys mismatch".format(epoch))
    expected_identity = {
        "config_path": config["absolute_path"], "config_sha256": config["sha256"],
        "checkpoint_path": checkpoint["absolute_path"],
        "checkpoint_size": checkpoint["size_bytes"],
        "checkpoint_mtime_ns": checkpoint["mtime_ns"],
        "checkpoint_sha256": checkpoint["sha256"],
    }
    require(identity == expected_identity,
            "epoch{} artifact identity mismatch".format(epoch))
    audit = profile.get("checkpoint_load_audit")
    require(isinstance(audit, dict) and audit.get("checkpoint") == checkpoint["absolute_path"],
            "epoch{} checkpoint load audit/path mismatch".format(epoch))
    for key in LOAD_AUDIT_ZERO_KEYS:
        exact_int(audit.get(key), 0, "epoch{} {}".format(epoch, key))
    counts = profile.get("module_counts")
    require(isinstance(counts, dict) and set(counts) == {
        "ATLIFTernaryPSN", "ShiftmaxAttention"},
        "epoch{} module count keys mismatch".format(epoch))
    exact_int(counts.get("ATLIFTernaryPSN"), policy.atlif_modules,
              "epoch{} ATLIFTernaryPSN".format(epoch))
    exact_int(counts.get("ShiftmaxAttention"), policy.attention_modules,
              "epoch{} ShiftmaxAttention".format(epoch))
    metrics = profile.get("metrics")
    require(isinstance(metrics, dict), "epoch{} missing metrics".format(epoch))
    metric_row = {key: nonnegative_decimal_metric(metrics, key, epoch)
                  for key in ERROR_METRIC_KEYS}
    total_spikes = profile.get("total_spikes")
    require(type(total_spikes) is int and total_spikes > 0,
            "epoch{} total_spikes must be positive int".format(epoch))
    firing = finite_float(profile.get("global_firing_rate"), "epoch{} firing".format(epoch))
    dense = finite_float(profile.get("dense_flops"), "epoch{} dense".format(epoch))
    effective = finite_float(profile.get("effective_flops"), "epoch{} effective".format(epoch))
    energy = finite_float(profile.get("energy_uj"), "epoch{} energy".format(epoch))
    require(0 <= firing <= 1, "epoch{} firing outside [0,1]".format(epoch))
    require(dense > 0 and 0 <= effective <= dense,
            "epoch{} invalid dense/effective FLOPs".format(epoch))
    require(energy > 0, "epoch{} activity energy proxy must be positive".format(epoch))
    return {
        "candidate_id": candidate.candidate_id, "epoch": epoch,
        "run_directory": str(candidate.run_dir.resolve()),
        "checkpoint": checkpoint, "configuration": config,
        "profile": dict(profile_identity, samples=825, artifact_identity_exact=True,
                        load_audit_exact_zero=True, module_counts=counts),
        "accuracy_metrics": metric_row,
        "activity": {
            "total_spikes": total_spikes, "global_firing_rate": firing,
            "dense_flops": dense, "effective_flops": effective,
            "effective_sparsity": 1.0 - effective / dense,
            "spike_energy_proxy_uj": energy,
            "energy_scope": "spike_activity_proxy_not_hardware_energy",
        },
    }


def build(policy: CrossRunPolicy) -> dict[str, Any]:
    validate_policy(policy)
    r1 = load_predecessor()
    manifest = r1.validate_new_manifest(policy)
    config_cache: dict[Path, dict[str, Any]] = {}
    rows = []
    for candidate in policy.candidates:
        if candidate.config not in config_cache:
            config = r1.stable_identity(candidate.config, candidate.candidate_id + " config")
            require(config["sha256"] == candidate.config_sha256,
                    candidate.candidate_id + " config SHA mismatch")
            config_cache[candidate.config] = config
        config = config_cache[candidate.config]
        checkpoint_path = candidate.run_dir / "checkpoint_epoch{}.pth".format(candidate.epoch)
        checkpoint = r1.stable_identity(checkpoint_path, candidate.candidate_id + " checkpoint")
        if candidate.expected_checkpoint_sha256 is not None:
            require(checkpoint["sha256"] == candidate.expected_checkpoint_sha256,
                    candidate.candidate_id + " checkpoint SHA mismatch")
        profile_path = candidate.run_dir / "standard_valid825" / (
            "epoch{}".format(candidate.epoch)) / "spike_profile.json"
        rows.append(validate_profile(candidate, profile_path, checkpoint, config, policy))
    selected = min(rows, key=lambda row: (
        Decimal(row["accuracy_metrics"]["AEE"]), row["epoch"]))
    return {
        "schema": "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1",
        "status": (
            "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"),
        "new_run_manifest": manifest, "candidate_population": rows,
        "selection_rule": {
            "candidate_ids": [row.candidate_id for row in policy.candidates],
            "epochs": [row.epoch for row in policy.candidates],
            "primary": "minimum finite nonnegative standard-valid825 AEE",
            "tie_break": "lowest epoch", "all_four_candidates_required": True,
            "cross_run": True, "cross_config": True,
            "profile_hash_and_parse_same_immutable_bytes": True,
        },
        "selected": {
            "candidate_id": selected["candidate_id"], "epoch": selected["epoch"],
            "run_directory": selected["run_directory"],
            "checkpoint": selected["checkpoint"], "configuration": selected["configuration"],
            "profile": selected["profile"], "accuracy_metrics": selected["accuracy_metrics"],
            "activity": selected["activity"],
        },
        "e0_e8_activation_dependent_invalidation_and_rebind_targets":
            r1.activation_rebind_targets(),
        "claim_boundary": {
            "selection_bound_after_execution": True,
            "fresh_result_hammer_required": True, "hardware_rebind_authorized": False,
            "hardware_replay_complete": False, "hardware_speedup": False,
            "system_speedup": False, "power_or_energy": False,
            "checkpoint_copied": False, "gpu_started_by_binder": False,
            "remote_access_by_binder": False, "eda_started_by_binder": False,
        },
    }


def write_receipt(output_dir: Path, result: dict[str, Any]) -> None:
    require(not output_dir.exists() and not output_dir.is_symlink(),
            "fresh non-symlink output namespace required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + output_dir.name + ".",
                                     dir=output_dir.parent) as name:
        root = Path(name)
        (root / "final_checkpoint_selection.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "selected_checkpoint_and_config.json").write_text(json.dumps({
            "schema": "m1234_selected_checkpoint_and_config_r1_v1",
            **{key: result["selected"][key] for key in (
                "candidate_id", "epoch", "run_directory", "checkpoint",
                "configuration", "profile")}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (root / "four_checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["candidate_id", "epoch", "config_sha256", "checkpoint_sha256",
                             "profile_sha256", "samples", *ERROR_METRIC_KEYS])
            for row in result["candidate_population"]:
                writer.writerow([row["candidate_id"], row["epoch"],
                                 row["configuration"]["sha256"], row["checkpoint"]["sha256"],
                                 row["profile"]["sha256"], row["profile"]["samples"],
                                 *[row["accuracy_metrics"][key] for key in ERROR_METRIC_KEYS]])
        (root / "e0_e8_activation_rebind_targets.json").write_text(json.dumps(
            result["e0_e8_activation_dependent_invalidation_and_rebind_targets"],
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "RUN_COMPLETE.txt").write_text(
            "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
            "FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n", encoding="utf-8")
        members = sorted(path for path in root.iterdir() if path.is_file())
        manifest = root / "SHA256SUMS"
        manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                    for path in members), encoding="utf-8")
        (root / "SHA256SUMS.seal.sha256").write_text(
            "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
        os.replace(str(root), str(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-mode", choices=("aee",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(PRODUCTION_POLICY)
    write_receipt(args.output_dir, result)
    print("PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__FRESH_RESULT_HAMMER_REQUIRED")
    print("selected_candidate=" + result["selected"]["candidate_id"])
    print("selected_epoch=" + str(result["selected"]["epoch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
