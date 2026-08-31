#!/usr/bin/env python3
"""M1241 additive M1234 successor with descriptor-rooted path identity.

Import is inert.  Explicit build keeps the exact M1234 selection and output
interface while closing two path-identity gaps: every path component is opened
without following symlinks, profile JSON is parsed before the final pathname
identity comparison, and the two run roots must have distinct device/inode
identities.  No mutable pathname is resolved after its final identity check.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


PREDECESSOR = Path(__file__).with_name(
    "build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py")
PREDECESSOR_SHA256 = "570ff4a6762a2ec9822a6161fb2f666becd6706a26586fe137f81b16fb188d0b"


class BinderError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenDirectory:
    absolute_path: str
    physical_identity: tuple[int, int, int]


@dataclass(frozen=True)
class FrozenFile:
    public_identity: dict[str, Any]
    physical_identity: tuple[int, int, int]
    pathname_identity: tuple[int, int, int, int, int]
    json_value: dict[str, Any] | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BinderError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predecessor() -> Any:
    require(PREDECESSOR.is_file() and not PREDECESSOR.is_symlink(),
            "sealed M1234 predecessor must be a regular non-symlink file")
    require(_sha256(PREDECESSOR) == PREDECESSOR_SHA256,
            "M1234 predecessor SHA drift")
    spec = importlib.util.spec_from_file_location("m1241_sealed_m1234", str(PREDECESSOR))
    require(spec is not None and spec.loader is not None,
            "cannot import M1234 predecessor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _lexical_absolute(path: Path) -> str:
    text = os.fspath(path)
    require(os.path.isabs(text), "all binder paths must be absolute: " + text)
    normalized = os.path.normpath(text)
    require(normalized == text, "path must already be normalized: " + text)
    return normalized


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode,
            value.st_size, value.st_mtime_ns)


def _physical(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _open_chain(absolute: str, final_directory: bool) -> tuple[int, dict[str, tuple[int, int, int]]]:
    """Open every component with O_NOFOLLOW and return final fd plus prefix ids."""
    flags_dir = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags_dir |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags_dir |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags_dir |= os.O_NOFOLLOW
    current = os.open("/", flags_dir)
    prefixes = {"/": _physical(os.fstat(current))}
    parts = [part for part in Path(absolute).parts if part != "/"]
    require(parts, "filesystem root is not a valid binder target")
    prefix = ""
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            flags = os.O_RDONLY
            if not last or final_directory:
                if hasattr(os, "O_DIRECTORY"):
                    flags |= os.O_DIRECTORY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                successor = os.open(part, flags, dir_fd=current)
            except OSError as exc:
                raise BinderError("cannot no-follow open path component {} in {}".format(
                    part, absolute)) from exc
            os.close(current)
            current = successor
            prefix += "/" + part
            observed = os.fstat(current)
            if not last or final_directory:
                require(stat.S_ISDIR(observed.st_mode),
                        "non-directory path component: " + prefix)
            prefixes[prefix] = _physical(observed)
        return current, prefixes
    except Exception:
        os.close(current)
        raise


def freeze_directory(path: Path, label: str) -> FrozenDirectory:
    absolute = _lexical_absolute(path)
    try:
        path_before = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise BinderError("missing {}: {}".format(label, absolute)) from exc
    require(stat.S_ISDIR(path_before.st_mode) and not stat.S_ISLNK(path_before.st_mode),
            label + " must be a non-symlink directory")
    descriptor, _ = _open_chain(absolute, final_directory=True)
    try:
        fd_identity = os.fstat(descriptor)
        try:
            path_after = os.lstat(absolute)
        except FileNotFoundError as exc:
            raise BinderError(label + " disappeared during directory freeze") from exc
        require(_identity(path_before) == _identity(fd_identity) == _identity(path_after),
                label + " path/descriptor identity mismatch")
        frozen = FrozenDirectory(absolute, _physical(fd_identity))
    finally:
        os.close(descriptor)
    return frozen


def _no_duplicate(r2: Any, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BinderError("non-finite JSON constant: " + value)


def freeze_file(path: Path, label: str, *, parse_json: bool,
                containment: FrozenDirectory | None = None,
                profile_identity: bool = False) -> FrozenFile:
    """Read/hash one descriptor, parse if requested, then perform final lstat."""
    absolute = _lexical_absolute(path)
    try:
        path_before = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise BinderError("missing {}: {}".format(label, absolute)) from exc
    require(stat.S_ISREG(path_before.st_mode) and not stat.S_ISLNK(path_before.st_mode),
            label + " must be a regular non-symlink file")
    descriptor, prefixes = _open_chain(absolute, final_directory=False)
    try:
        fd_before = os.fstat(descriptor)
        require(stat.S_ISREG(fd_before.st_mode), label + " descriptor is not regular")
        require(_identity(path_before) == _identity(fd_before),
                label + " path/descriptor mismatch before read")
        if containment is not None:
            require(containment.absolute_path in prefixes,
                    label + " is not descriptor-contained by its run root")
            require(prefixes[containment.absolute_path] == containment.physical_identity,
                    label + " run-root descriptor identity mismatch")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            if parse_json:
                blocks.append(block)
        fd_after = os.fstat(descriptor)
        require(_identity(fd_before) == _identity(fd_after),
                label + " descriptor changed during read")
        value = None
        if parse_json:
            payload = b"".join(blocks)
            require(len(payload) == fd_after.st_size and len(payload) > 0,
                    label + " byte population mismatch")
            try:
                value = json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=lambda pairs: _no_duplicate(None, pairs),
                    parse_constant=_reject_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise BinderError("invalid JSON {}: {}".format(absolute, exc)) from exc
            require(isinstance(value, dict), label + " JSON root must be an object")
        # This is deliberately after JSON parse/validation of the byte syntax.
        # No pathname operation is permitted after this equality before return.
        try:
            path_final = os.lstat(absolute)
        except FileNotFoundError as exc:
            raise BinderError(label + " disappeared before publication") from exc
        require(_identity(fd_after) == _identity(path_final),
                label + " changed before frozen identity publication")
        public = {
            "absolute_path": absolute,
            "size_bytes": fd_after.st_size,
            "mtime_ns": fd_after.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
        if profile_identity:
            public.update({
                "device": fd_after.st_dev,
                "inode": fd_after.st_ino,
                "immutable_single_read": True,
                "hash_and_parse_same_bytes": True,
                "post_parse_path_identity_frozen": True,
                "descriptor_rooted_no_symlink_components": True,
            })
        frozen = FrozenFile(public, _physical(fd_after), _identity(fd_after), value)
    finally:
        os.close(descriptor)
    return frozen


def confirm_frozen_path(file: FrozenFile, label: str) -> None:
    """Final post-semantic-validation pathname check; never resolve the path."""
    try:
        observed = os.lstat(file.public_identity["absolute_path"])
    except FileNotFoundError as exc:
        raise BinderError(label + " disappeared after semantic validation") from exc
    require(_identity(observed) == file.pathname_identity,
            label + " changed during semantic validation")


def validate_profile(r2: Any, candidate: Any, profile_file: FrozenFile,
                     checkpoint: FrozenFile, config: FrozenFile,
                     policy: Any) -> dict[str, Any]:
    profile = profile_file.json_value
    require(isinstance(profile, dict), "profile JSON missing")
    epoch = candidate.epoch
    r2.exact_int(profile.get("samples"), 825, "epoch{} samples".format(epoch))
    identity = profile.get("artifact_identity")
    require(isinstance(identity, dict) and set(identity) == r2.ARTIFACT_IDENTITY_KEYS,
            "epoch{} artifact identity keys mismatch".format(epoch))
    expected_identity = {
        "config_path": config.public_identity["absolute_path"],
        "config_sha256": config.public_identity["sha256"],
        "checkpoint_path": checkpoint.public_identity["absolute_path"],
        "checkpoint_size": checkpoint.public_identity["size_bytes"],
        "checkpoint_mtime_ns": checkpoint.public_identity["mtime_ns"],
        "checkpoint_sha256": checkpoint.public_identity["sha256"],
    }
    require(identity == expected_identity,
            "epoch{} artifact identity mismatch".format(epoch))
    audit = profile.get("checkpoint_load_audit")
    require(isinstance(audit, dict) and
            audit.get("checkpoint") == checkpoint.public_identity["absolute_path"],
            "epoch{} checkpoint load audit/path mismatch".format(epoch))
    for key in r2.LOAD_AUDIT_ZERO_KEYS:
        r2.exact_int(audit.get(key), 0, "epoch{} {}".format(epoch, key))
    counts = profile.get("module_counts")
    require(isinstance(counts, dict) and set(counts) == {
        "ATLIFTernaryPSN", "ShiftmaxAttention"},
        "epoch{} module count keys mismatch".format(epoch))
    r2.exact_int(counts.get("ATLIFTernaryPSN"), policy.atlif_modules,
                 "epoch{} ATLIFTernaryPSN".format(epoch))
    r2.exact_int(counts.get("ShiftmaxAttention"), policy.attention_modules,
                 "epoch{} ShiftmaxAttention".format(epoch))
    metrics = profile.get("metrics")
    require(isinstance(metrics, dict), "epoch{} missing metrics".format(epoch))
    metric_row = {key: r2.nonnegative_decimal_metric(metrics, key, epoch)
                  for key in r2.ERROR_METRIC_KEYS}
    total_spikes = profile.get("total_spikes")
    require(type(total_spikes) is int and total_spikes > 0,
            "epoch{} total_spikes must be positive int".format(epoch))
    firing = r2.finite_float(profile.get("global_firing_rate"),
                             "epoch{} firing".format(epoch))
    dense = r2.finite_float(profile.get("dense_flops"),
                            "epoch{} dense".format(epoch))
    effective = r2.finite_float(profile.get("effective_flops"),
                                "epoch{} effective".format(epoch))
    energy = r2.finite_float(profile.get("energy_uj"),
                             "epoch{} energy".format(epoch))
    require(0 <= firing <= 1, "epoch{} firing outside [0,1]".format(epoch))
    require(dense > 0 and 0 <= effective <= dense,
            "epoch{} invalid dense/effective FLOPs".format(epoch))
    require(energy > 0, "epoch{} activity energy proxy must be positive".format(epoch))
    profile_public = dict(profile_file.public_identity)
    profile_public.update(samples=825, artifact_identity_exact=True,
                          load_audit_exact_zero=True, module_counts=counts)
    return {
        "candidate_id": candidate.candidate_id,
        "epoch": epoch,
        "run_directory": _lexical_absolute(candidate.run_dir),
        "checkpoint": checkpoint.public_identity,
        "configuration": config.public_identity,
        "profile": profile_public,
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


def build(policy: Any) -> dict[str, Any]:
    r2 = load_predecessor()
    try:
        r2.validate_policy(policy)
    except r2.BinderError as exc:
        raise BinderError(str(exc)) from exc

    roots_by_path: dict[str, FrozenDirectory] = {}
    for candidate in policy.candidates:
        absolute = _lexical_absolute(candidate.run_dir)
        if absolute not in roots_by_path:
            roots_by_path[absolute] = freeze_directory(candidate.run_dir,
                                                       candidate.candidate_id + " run root")
    require(len(roots_by_path) == 2, "candidate topology must name exactly two run roots")
    require(len({row.physical_identity for row in roots_by_path.values()}) == 2,
            "old and resume run roots must be physically distinct")

    manifest_file = freeze_file(policy.new_run_manifest, "new run manifest",
                                parse_json=True)
    manifest = manifest_file.json_value
    require(isinstance(manifest, dict) and isinstance(manifest.get("evaluation_epochs"), list),
            "new manifest evaluation_epochs must be a list")
    require(all(type(epoch) is int for epoch in manifest["evaluation_epochs"]),
            "new manifest evaluation epochs must be typed integers")
    require(tuple(manifest["evaluation_epochs"]) == tuple(policy.new_evaluation_epochs),
            "new manifest evaluation epochs mismatch")

    config_cache: dict[str, FrozenFile] = {}
    rows = []
    for candidate in policy.candidates:
        config_path = _lexical_absolute(candidate.config)
        if config_path not in config_cache:
            config_cache[config_path] = freeze_file(
                candidate.config, candidate.candidate_id + " config", parse_json=False)
        config = config_cache[config_path]
        require(config.public_identity["sha256"] == candidate.config_sha256,
                candidate.candidate_id + " config SHA mismatch")
        root = roots_by_path[_lexical_absolute(candidate.run_dir)]
        checkpoint_path = candidate.run_dir / (
            "checkpoint_epoch{}.pth".format(candidate.epoch))
        checkpoint = freeze_file(
            checkpoint_path, candidate.candidate_id + " checkpoint",
            parse_json=False, containment=root)
        if candidate.expected_checkpoint_sha256 is not None:
            require(checkpoint.public_identity["sha256"] ==
                    candidate.expected_checkpoint_sha256,
                    candidate.candidate_id + " checkpoint SHA mismatch")
        profile_path = candidate.run_dir / "standard_valid825" / (
            "epoch{}".format(candidate.epoch)) / "spike_profile.json"
        profile = freeze_file(
            profile_path, "epoch{} spike profile".format(candidate.epoch),
            parse_json=True, containment=root, profile_identity=True)
        try:
            row = validate_profile(r2, candidate, profile, checkpoint, config, policy)
        except r2.BinderError as exc:
            raise BinderError(str(exc)) from exc
        # This is the final pathname operation for the profile.  The row below
        # is assembled solely from the already-frozen descriptor identity.
        confirm_frozen_path(profile, "epoch{} spike profile".format(candidate.epoch))
        rows.append(row)

    require(len(config_cache) == 2, "candidate topology must name exactly two configs")
    require(len({row.physical_identity for row in config_cache.values()}) == 2,
            "old and resume configurations must be physically distinct")
    require(len({row.public_identity["sha256"] for row in config_cache.values()}) == 2,
            "old and resume configuration SHA identities must be distinct")

    selected = min(rows, key=lambda row: (
        Decimal(row["accuracy_metrics"]["AEE"]), row["epoch"]))
    m1228 = r2.load_predecessor()
    return {
        # Keep the M1234 consumer interface exactly stable.
        "schema": "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1",
        "status": (
            "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"),
        "new_run_manifest": manifest_file.public_identity,
        "candidate_population": rows,
        "selection_rule": {
            "candidate_ids": [row.candidate_id for row in policy.candidates],
            "epochs": [row.epoch for row in policy.candidates],
            "primary": "minimum finite nonnegative standard-valid825 AEE",
            "tie_break": "lowest epoch",
            "all_four_candidates_required": True,
            "cross_run": True,
            "cross_config": True,
            "profile_hash_and_parse_same_immutable_bytes": True,
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
        "e0_e8_activation_dependent_invalidation_and_rebind_targets":
            m1228.activation_rebind_targets(),
        "claim_boundary": {
            "selection_bound_after_execution": True,
            "fresh_result_hammer_required": True,
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
    r2 = load_predecessor()
    try:
        r2.write_receipt(output_dir, result)
    except r2.BinderError as exc:
        raise BinderError(str(exc)) from exc


def main() -> int:
    r2 = load_predecessor()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-mode", choices=("aee",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(r2.PRODUCTION_POLICY)
    write_receipt(args.output_dir, result)
    print("PASS_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R3_SECURITY_SUCCESSOR__"
          "FRESH_RESULT_HAMMER_REQUIRED")
    print("selected_candidate=" + result["selected"]["candidate_id"])
    print("selected_epoch=" + str(result["selected"]["epoch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
