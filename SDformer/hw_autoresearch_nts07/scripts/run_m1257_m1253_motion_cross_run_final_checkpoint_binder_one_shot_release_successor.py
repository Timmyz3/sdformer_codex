#!/opt/conda/envs/sdformerflow/bin/python
"""M1257 source-only successor closing the M1255 binder findings.

The predecessor provides descriptor-rooted snapshots, sealed execution memfds,
and one-shot mechanics.  This successor additionally binds full st_mode, closes
every published schema, and pins the exact E0-E8 invalidation map.  Import is
read-only and inert; production execution remains a future, separately hammered
one-shot action after all four strict-valid825 inputs exist.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Sequence


BASE_SOURCE = Path(__file__).with_name(
    "run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py")
BASE_SOURCE_SHA256 = "491b4b9bfe2d268b184d538ca99b8811f962a39acc8a7947a627735f63f1fd30"


def _load_base() -> ModuleType:
    payload = BASE_SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BASE_SOURCE_SHA256:
        raise RuntimeError("M1253 predecessor source SHA drift")
    module = ModuleType("m1257_sealed_source_predecessor")
    module.__file__ = str(BASE_SOURCE)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(payload, str(BASE_SOURCE), "exec"), module.__dict__)
    return module


B = _load_base()

M1253_PINS = {
    Path("hw_autoresearch_nts07/scripts/run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"):
        BASE_SOURCE_SHA256,
    Path("hw_autoresearch_nts07/tests/test_run_m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor.py"):
        "55ffe2a8b8df9c452ada84b8dd06c6abaf1ba576a5d92a99dfc4d2ea1e0c0d0f",
    Path("hw_autoresearch_nts07/contracts/m1253_m1248_motion_cross_run_final_checkpoint_binder_one_shot_release_successor_source_contract_r1_20260830.json"):
        "06c951085df50bf1776e84e2cadcd43a347070ac181bcf337d9b1c23e342262b",
}
M1255_REL = Path(
    "hw_autoresearch_nts07/reviews/m1255_m1253_production_binder_release_successor_independent_hammer_r1_20260830")
M1255_MANIFEST_SHA256 = "5c214ab9b838efe74956165038da60bec4e512012f234a5a72fc2dcb2841aeee"
M1255_OUTER_SHA256 = "841b3ab4edb0b5eb41e360e5ccad9c4144b8a0ee7339c02f2cbf5e3679308795"
M1255_SCHEMA = "m1255_m1253_production_binder_release_successor_independent_hammer_r1_v1"
M1255_STATUS = "BLOCK_M1255_M1253_RELEASE__MODE_AND_RECEIPT_CLOSURE_GAPS__SUCCESSOR_REQUIRED"

OUTPUT_REL = Path(
    "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830")
ATTEMPT_REL = Path(
    "hw_autoresearch_nts07/results/.m1257_motion_cross_run_final_checkpoint_selection_r5_attempt_consumed")
LOG_REL = Path(
    "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830.launch.log")
CHILD_TOKEN = (
    "PASS_M1257_SEALED_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
    "FRESH_RESULT_HAMMER_REQUIRED")

IDENTITY_KEYS = frozenset({
    "absolute_path", "sha256", "size_bytes", "mtime_ns", "device", "inode", "mode",
})
PROFILE_EXTRA_KEYS = frozenset({
    "immutable_single_read", "hash_and_parse_same_bytes",
    "post_parse_path_identity_frozen", "descriptor_rooted_no_symlink_components",
    "samples", "artifact_identity_exact", "load_audit_exact_zero", "module_counts",
})
ACTIVITY_KEYS = frozenset({
    "total_spikes", "global_firing_rate", "dense_flops", "effective_flops",
    "effective_sparsity", "spike_energy_proxy_uj", "energy_scope",
})
RESULT_KEYS = frozenset({
    "schema", "status", "new_run_manifest", "candidate_population",
    "selection_rule", "selected",
    "e0_e8_activation_dependent_invalidation_and_rebind_targets", "claim_boundary",
})


def exact_rebind_targets() -> list[dict[str, Any]]:
    common = {
        "dependency": "selected checkpoint SHA/size/mtime + selected config SHA/size/mtime",
        "reuse_rule": (
            "reuse only when an independently sealed artifact binds the exact selected "
            "checkpoint and config identities; otherwise invalidate and regenerate"),
    }
    rows = (
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
    )
    return [
        {"id": identifier, "target": target, "state_after_selection": state, **common}
        for identifier, target, state in rows
    ]


SEALED_LAUNCHER = r'''
import os,sys,types
from pathlib import Path
def load(fd, name, filename):
    os.lseek(fd, 0, os.SEEK_SET)
    blocks=[]
    while True:
        block=os.read(fd, 1<<20)
        if not block: break
        blocks.append(block)
    module=types.ModuleType(name)
    module.__file__=filename
    module.__package__=""
    sys.modules[name]=module
    exec(compile(b"".join(blocks), filename, "exec"), module.__dict__)
    return module
m1241=load(int(sys.argv[1]), "m1257_sealed_m1241", "build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py")
m1234=load(int(sys.argv[2]), "m1257_sealed_m1234", "build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py")
m1228=load(int(sys.argv[3]), "m1257_sealed_m1228", "build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py")
m1234.load_predecessor=lambda: m1228
m1241.load_predecessor=lambda: m1234
original_freeze=m1241.freeze_file
def enriched_freeze(*args, **kwargs):
    frozen=original_freeze(*args, **kwargs)
    frozen.public_identity["device"]=frozen.pathname_identity[0]
    frozen.public_identity["inode"]=frozen.pathname_identity[1]
    frozen.public_identity["mode"]=frozen.pathname_identity[2]
    return frozen
m1241.freeze_file=enriched_freeze
result=m1241.build(m1234.PRODUCTION_POLICY)
m1241.write_receipt(Path(sys.argv[4]), result)
print("PASS_M1257_SEALED_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__FRESH_RESULT_HAMMER_REQUIRED")
print("selected_candidate="+result["selected"]["candidate_id"])
print("selected_epoch="+str(result["selected"]["epoch"]))
'''


@dataclass(frozen=True)
class Policy:
    base: Any
    successor_review_rel: Path
    successor_manifest_sha256: str
    successor_outer_sha256: str


_authority = dict(B.M1248_PINS)
_authority.update(M1253_PINS)
_base_policy = B.Policy(
    B.REPO, B.INTERPRETER, B.PYTHON_VERSION, _authority,
    dict(B.EXECUTION_PINS), dict(B.M1241_AUX_PINS),
    B.M1251_REL, B.M1251_MANIFEST_SHA256, B.M1251_OUTER_SHA256,
    B.DOCS359_REL, B.DOCS359_SHA256, B.PRODUCTION_POLICY.candidates,
    B.NEW_MANIFEST_REL, OUTPUT_REL, ATTEMPT_REL, LOG_REL,
)
PRODUCTION_POLICY = Policy(
    _base_policy, M1255_REL, M1255_MANIFEST_SHA256, M1255_OUTER_SHA256)


def _strict_any(payload: bytes, label: str) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            B.require(key not in value, "duplicate JSON key in {}: {}".format(label, key))
            value[key] = item
        return value
    def constant(value):
        raise B.ReleaseError("non-finite JSON constant in {}: {}".format(label, value))
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise B.ReleaseError("invalid {} JSON".format(label)) from exc


def _verify_m1255(policy: Policy) -> None:
    root = policy.base.repo / policy.successor_review_rel
    B.directory(root, "M1255 review")
    manifest, manifest_payload = B.snapshot_file(root / B.MANIFEST, "M1255 manifest")
    outer, outer_payload = B.snapshot_file(root / B.OUTER, "M1255 outer")
    B.require(manifest.sha256 == policy.successor_manifest_sha256, "M1255 manifest SHA drift")
    B.require(outer.sha256 == policy.successor_outer_sha256, "M1255 outer SHA drift")
    B.require(outer_payload.decode("utf-8").split() ==
              [policy.successor_manifest_sha256, B.MANIFEST], "M1255 outer mismatch")
    review_payload = None
    seen = set()
    for line in manifest_payload.decode("utf-8").splitlines():
        fields = line.split(None, 1)
        B.require(len(fields) == 2 and len(fields[0]) == 64, "invalid M1255 manifest row")
        name = fields[1].lstrip("*")
        B.require(Path(name).name == name and name not in seen, "invalid M1255 member")
        observed, payload = B.snapshot_file(root / name, "M1255 member " + name)
        B.require(observed.sha256 == fields[0], "M1255 member drift: " + name)
        seen.add(name)
        if name == "review.json":
            review_payload = payload
    B.require(review_payload is not None, "M1255 review.json missing")
    review = B.strict_json_payload(review_payload, "M1255 review")
    B.require(review.get("schema") == M1255_SCHEMA and review.get("status") == M1255_STATUS,
              "M1255 schema/status mismatch")
    authority = review.get("authority")
    B.require(isinstance(authority, dict) and
              authority.get("production_execution_authorized_now") is False and
              authority.get("future_execution_authorized_by_M1255") is False and
              authority.get("release_successor_authoring_required") is True and
              authority.get("fresh_different_author_successor_hammer_required") is True,
              "M1255 successor authority mismatch")


def prepare(policy: Policy, executable_path: Path, version: str, cwd: Path):
    prepared = B.prepare(policy.base, executable_path, version, cwd)
    try:
        _verify_m1255(policy)
        prepared.command = [
            str(policy.base.interpreter), "-I", "-B", "-c", SEALED_LAUNCHER,
            *(str(descriptor) for descriptor in prepared.source_fds),
            str(policy.base.repo / policy.base.output_rel),
        ]
        return prepared
    except Exception:
        prepared.close()
        raise


def _snapshot_identity(snapshot: Any) -> dict[str, Any]:
    return {
        "absolute_path": snapshot.absolute_path, "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes, "mtime_ns": snapshot.mtime_ns,
        "device": snapshot.device, "inode": snapshot.inode, "mode": snapshot.mode,
    }


def _exact_identity(observed: Any, expected: Any, label: str) -> None:
    wanted = _snapshot_identity(expected)
    B.require(isinstance(observed, dict) and set(observed) == IDENTITY_KEYS,
              label + " exact identity key mismatch")
    B.require(observed == wanted, label + " exact identity mismatch")


def _exact_profile(observed: Any, expected: Any, label: str) -> None:
    B.require(isinstance(observed, dict) and
              set(observed) == IDENTITY_KEYS | PROFILE_EXTRA_KEYS,
              label + " exact profile key mismatch")
    for key, value in _snapshot_identity(expected).items():
        B.require(observed.get(key) == value, label + " " + key + " mismatch")
    for key in ("immutable_single_read", "hash_and_parse_same_bytes",
                "post_parse_path_identity_frozen", "descriptor_rooted_no_symlink_components",
                "artifact_identity_exact", "load_audit_exact_zero"):
        B.require(observed.get(key) is True, label + " " + key + " mismatch")
    B.require(type(observed.get("samples")) is int and observed["samples"] == 825,
              label + " samples mismatch")
    B.require(observed.get("module_counts") == {
        "ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
        label + " module count mismatch")


def _metric(value: Any, label: str) -> Decimal:
    B.require(type(value) is str, label + " metric must be canonical string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise B.ReleaseError(label + " invalid decimal") from exc
    B.require(result.is_finite() and result >= 0, label + " metric invalid")
    return result


def _exact_activity(value: Any, label: str) -> None:
    B.require(isinstance(value, dict) and set(value) == ACTIVITY_KEYS,
              label + " exact activity key mismatch")
    B.require(type(value["total_spikes"]) is int and value["total_spikes"] > 0,
              label + " total_spikes mismatch")
    for key in ("global_firing_rate", "dense_flops", "effective_flops",
                "effective_sparsity", "spike_energy_proxy_uj"):
        B.require(type(value[key]) is float and math.isfinite(value[key]),
                  label + " " + key + " type/value mismatch")
    B.require(0 <= value["global_firing_rate"] <= 1 and value["dense_flops"] > 0 and
              0 <= value["effective_flops"] <= value["dense_flops"] and
              0 <= value["effective_sparsity"] <= 1 and value["spike_energy_proxy_uj"] > 0,
              label + " activity range mismatch")
    B.require(value["energy_scope"] == "spike_activity_proxy_not_hardware_energy",
              label + " energy scope mismatch")


def _verify_rows(result: dict[str, Any], prepared: Any) -> list[dict[str, Any]]:
    rows = result.get("candidate_population")
    B.require(isinstance(rows, list) and len(rows) == 4, "candidate population mismatch")
    row_keys = {
        "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
        "profile", "accuracy_metrics", "activity",
    }
    for row, candidate in zip(rows, prepared.policy.candidates):
        B.require(isinstance(row, dict) and set(row) == row_keys,
                  "candidate exact row keys mismatch")
        B.require((row["candidate_id"], row["epoch"]) ==
                  (candidate.candidate_id, candidate.epoch), "candidate pair/order mismatch")
        B.require(type(row["candidate_id"]) is str and type(row["epoch"]) is int,
                  "candidate pair types mismatch")
        B.require(row["run_directory"] == str(prepared.policy.repo / candidate.run_rel),
                  "candidate run directory mismatch")
        _exact_identity(row["checkpoint"],
                        prepared.snapshots[candidate.candidate_id + ":checkpoint"],
                        candidate.candidate_id + " checkpoint")
        _exact_identity(row["configuration"],
                        prepared.snapshots["config:" + candidate.config_key],
                        candidate.candidate_id + " configuration")
        _exact_profile(row["profile"],
                       prepared.snapshots[candidate.candidate_id + ":profile"],
                       candidate.candidate_id + " profile")
        metrics = row["accuracy_metrics"]
        B.require(isinstance(metrics, dict) and set(metrics) == set(B.ERROR_METRIC_KEYS),
                  "accuracy metric exact keys mismatch")
        for key in B.ERROR_METRIC_KEYS:
            _metric(metrics[key], candidate.candidate_id + " " + key)
        _exact_activity(row["activity"], candidate.candidate_id)
    return rows


def verify_receipt(output: Path, prepared: Any) -> dict[str, Any]:
    B.directory(output, "M1257 selection receipt")
    observed, payloads = set(), {}
    for member in output.iterdir():
        snapshot, payload = B.snapshot_file(member, "receipt member " + member.name)
        B.require(stat.S_ISREG(snapshot.mode), "receipt member must be regular")
        observed.add(member.name)
        payloads[member.name] = payload
    B.require(observed == B.RESULT_PAYLOADS | {B.MANIFEST, B.OUTER},
              "receipt member population mismatch")
    B.require(payloads[B.OUTER].decode("utf-8").split() ==
              [B.sha256_bytes(payloads[B.MANIFEST]), B.MANIFEST], "outer seal mismatch")
    rows_by_name = {}
    for line in payloads[B.MANIFEST].decode("utf-8").splitlines():
        fields = line.split(None, 1)
        B.require(len(fields) == 2 and len(fields[0]) == 64, "invalid manifest row")
        name = fields[1].lstrip("*")
        B.require(name in B.RESULT_PAYLOADS and Path(name).name == name and
                  name not in rows_by_name, "invalid receipt member")
        B.require(B.sha256_bytes(payloads[name]) == fields[0], "payload SHA drift: " + name)
        rows_by_name[name] = fields[0]
    B.require(set(rows_by_name) == B.RESULT_PAYLOADS, "manifest population mismatch")
    B.require(payloads["RUN_COMPLETE.txt"].decode("utf-8") == B.RUN_COMPLETE,
              "terminal mismatch")

    result = B.strict_json_payload(payloads["final_checkpoint_selection.json"], "selection")
    B.require(set(result) == RESULT_KEYS, "selection exact root key mismatch")
    B.require(result["schema"] == B.RESULT_SCHEMA and result["status"] == B.RESULT_STATUS,
              "selection schema/status mismatch")
    B.require(result["claim_boundary"] == dict(B.EXACT_CLAIM_BOUNDARY),
              "exact claim boundary mismatch")
    _exact_identity(result["new_run_manifest"], prepared.snapshots["manifest"], "manifest")
    candidate_rows = _verify_rows(result, prepared)
    rule = {
        "candidate_ids": list(B.EXACT_PAIRS), "epochs": list(B.EXACT_PAIRS.values()),
        "primary": "minimum finite nonnegative standard-valid825 AEE",
        "tie_break": "lowest epoch", "all_four_candidates_required": True,
        "cross_run": True, "cross_config": True,
        "profile_hash_and_parse_same_immutable_bytes": True,
    }
    B.require(result["selection_rule"] == rule, "exact selection rule mismatch")
    winner = min(candidate_rows, key=lambda row: (
        _metric(row["accuracy_metrics"]["AEE"], "AEE"), row["epoch"]))
    selected = {key: winner[key] for key in (
        "candidate_id", "epoch", "run_directory", "checkpoint", "configuration",
        "profile", "accuracy_metrics", "activity")}
    B.require(result["selected"] == selected, "selected projection mismatch")

    selected_file = B.strict_json_payload(
        payloads["selected_checkpoint_and_config.json"], "selected sidecar")
    B.require(selected_file == {
        "schema": "m1234_selected_checkpoint_and_config_r1_v1",
        **{key: selected[key] for key in (
            "candidate_id", "epoch", "run_directory", "checkpoint",
            "configuration", "profile")}}, "selected sidecar mismatch")

    targets = exact_rebind_targets()
    B.require(result["e0_e8_activation_dependent_invalidation_and_rebind_targets"] == targets,
              "result E0-E8 exact map mismatch")
    sidecar_targets = _strict_any(payloads["e0_e8_activation_rebind_targets.json"],
                                  "E0-E8 sidecar")
    B.require(sidecar_targets == targets, "sidecar E0-E8 exact map mismatch")

    csv_rows = list(csv.DictReader(io.StringIO(
        payloads["four_checkpoint_metrics.csv"].decode("utf-8"), newline="")))
    B.require(len(csv_rows) == 4, "metrics CSV row population mismatch")
    header = ["candidate_id", "epoch", "config_sha256", "checkpoint_sha256",
              "profile_sha256", "samples", *B.ERROR_METRIC_KEYS]
    B.require(list(csv_rows[0]) == header, "metrics CSV header mismatch")
    for csv_row, row in zip(csv_rows, candidate_rows):
        B.require(csv_row["candidate_id"] == row["candidate_id"] and
                  csv_row["epoch"] == str(row["epoch"]) and
                  csv_row["config_sha256"] == row["configuration"]["sha256"] and
                  csv_row["checkpoint_sha256"] == row["checkpoint"]["sha256"] and
                  csv_row["profile_sha256"] == row["profile"]["sha256"] and
                  csv_row["samples"] == "825", "metrics CSV identity mismatch")
        for key in B.ERROR_METRIC_KEYS:
            B.require(csv_row[key] == row["accuracy_metrics"][key],
                      "metrics CSV value mismatch: " + key)
    return result


def consume_attempt(prepared: Any) -> None:
    attempt = prepared.policy.repo / prepared.policy.attempt_rel
    descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        population = {key: _snapshot_identity(prepared.snapshots[key])
                      for key in sorted(prepared.snapshots)}
        body = (
            "M1257_PRODUCTION_BINDER_ATTEMPT_CONSUMED_BEFORE_SEALED_CHILD\n"
            "automatic_retry=false\ninput_snapshot_sha256={}\ncommand_sha256={}\n".format(
                B.sha256_bytes(json.dumps(population, sort_keys=True,
                                          separators=(",", ":")).encode()),
                B.sha256_bytes("\0".join(prepared.command).encode())))
        os.write(descriptor, body.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_log(prepared: Any, completed: subprocess.CompletedProcess[str]) -> None:
    log = prepared.policy.repo / prepared.policy.log_rel
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        body = (
            "M1257_SEALED_CHILD_LOG\nreturncode={}\nstdout_sha256={}\nstderr_sha256={}\n".format(
                completed.returncode, B.sha256_bytes(completed.stdout.encode()),
                B.sha256_bytes(completed.stderr.encode())))
        os.write(descriptor, body.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


Runner = Callable[[Sequence[str], Path, tuple[int, ...]], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path, pass_fds: tuple[int, ...]):
    return subprocess.run(list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, pass_fds=pass_fds,
                          env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})


def execute_once(policy: Policy, executable_path: Path, version: str, cwd: Path,
                 runner: Runner = default_runner):
    prepared = prepare(policy, executable_path, version, cwd)
    try:
        consume_attempt(prepared)
        completed = runner(prepared.command, policy.base.repo, prepared.source_fds)
        publish_log(prepared, completed)
        B.require(completed.returncode == 0,
                  "single sealed M1241 child failed after attempt; no retry authorized")
        B.require(completed.stdout.count(CHILD_TOKEN) == 1,
                  "sealed child terminal stdout mismatch")
        verify_receipt(policy.base.repo / policy.base.output_rel, prepared)
        return completed
    finally:
        prepared.close()


def main() -> int:
    B.require(len(sys.argv) == 1, "production M1257 release accepts zero arguments")
    completed = execute_once(PRODUCTION_POLICY, Path(sys.executable),
                             platform.python_version(), Path.cwd())
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1257_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
