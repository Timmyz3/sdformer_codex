#!/opt/conda/envs/sdformerflow/bin/python
"""M1248 one-shot production wrapper for the hammered M1241 binder.

Import is inert.  A future zero-argument invocation on the A800 host performs
the complete read-only artifact/authority preflight, atomically consumes one
attempt, and starts exactly one pinned M1241 child.  The child only reads and
hashes the four completed candidates and writes a small double-sealed selection
receipt.  This wrapper never retries and never starts GPU, training, capture,
or EDA work.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Callable, Sequence


REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
HW = REPO / "hw_autoresearch_nts07"
INTERPRETER = Path("/opt/conda/envs/sdformerflow/bin/python")
PYTHON_VERSION = "3.10.20"

M1241_REL = Path(
    "hw_autoresearch_nts07/scripts/"
    "build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py")
M1241_TEST_REL = Path(
    "hw_autoresearch_nts07/tests/"
    "test_build_m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor.py")
M1241_CONTRACT_REL = Path(
    "hw_autoresearch_nts07/contracts/"
    "m1241_motion_cross_run_final_checkpoint_rebind_binder_r3_successor_source_contract_r1_20260830.json")
M1241_PINS = {
    M1241_REL: "10e97e31362064e63eea153ea087fc0e04379b172516bbe6061a6a249bce5f9b",
    M1241_TEST_REL: "c941406ee5e1de6d680d035f5496d5dddf2ea062fbd2c2c3f9399327948559b7",
    M1241_CONTRACT_REL: "9493529d5399f1557d236cdab714fa4addb21666a9916db6ea3427f64924b02a",
}

M1245_REL = Path(
    "hw_autoresearch_nts07/reviews/"
    "m1245_m1241_cross_run_binder_r3_successor_source_hammer_r1_20260830")
M1245_MANIFEST_SHA256 = "7c91fe0f2b7b3ff9e71c55b0ff2207205675e75932b8e65fd0ae80b22072a372"
M1245_OUTER_SHA256 = "0580883f80107cbe94150c92839e424cf9aab590f8d3c79240dae7358732184c"
M1245_REVIEW_SCHEMA = "m1245_m1241_cross_run_binder_r3_successor_source_hammer_r1_v1"
M1245_REVIEW_STATUS = "PASS_M1245_M1241_SOURCE_HAMMER__RELEASE_AUTHORING_ALLOWED"

DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

OLD_RUN_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx")
NEW_RUN_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "dsec_c12_alpha0125_ep29_resume5_20260830")
OLD_CONFIG_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml")
NEW_CONFIG_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
NEW_MANIFEST_REL = Path(
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_c12_alpha0125_ep29_resume5_20260830.json")

OUTPUT_REL = Path(
    "hw_autoresearch_nts07/results/"
    "m1248_motion_cross_run_final_checkpoint_selection_r3_20260830")
ATTEMPT_REL = Path(
    "hw_autoresearch_nts07/results/"
    ".m1248_motion_cross_run_final_checkpoint_selection_r3_attempt_consumed")
LOG_REL = Path(
    "hw_autoresearch_nts07/results/"
    "m1248_motion_cross_run_final_checkpoint_selection_r3_20260830.launch.log")

RESULT_PAYLOADS = frozenset({
    "RUN_COMPLETE.txt",
    "e0_e8_activation_rebind_targets.json",
    "final_checkpoint_selection.json",
    "four_checkpoint_metrics.csv",
    "selected_checkpoint_and_config.json",
})
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"
RESULT_SCHEMA = "m1234_motion_cross_run_final_checkpoint_rebind_binder_r2_v1"
RESULT_STATUS = (
    "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__"
    "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY")
CHILD_TOKEN = (
    "PASS_M1241_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R3_SECURITY_SUCCESSOR__"
    "FRESH_RESULT_HAMMER_REQUIRED")
RUN_COMPLETE = (
    "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED__"
    "FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n")


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode),
            "{} must be a non-symlink regular file: {}".format(label, path))


def directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode),
            "{} must be a non-symlink directory: {}".format(label, path))


def executable(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise ReleaseError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode),
            label + " must be a regular file or symlink")
    try:
        target = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ReleaseError("broken or cyclic " + label) from exc
    require(stat.S_ISREG(target.stat().st_mode) and os.access(target, os.X_OK),
            label + " target must be an executable regular file")


def verify_double_sealed_review(directory_path: Path, manifest_sha: str,
                                outer_sha: str) -> dict[str, object]:
    directory(directory_path, "M1245 review")
    manifest = directory_path / MANIFEST
    outer = directory_path / OUTER
    regular(manifest, "M1245 manifest")
    regular(outer, "M1245 outer seal")
    require(sha256(manifest) == manifest_sha, "M1245 manifest SHA drift")
    require(sha256(outer) == outer_sha, "M1245 outer seal SHA drift")
    fields = outer.read_text(encoding="utf-8").split()
    require(fields == [manifest_sha, MANIFEST], "M1245 outer seal content mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        require(len(parts) == 2 and len(parts[0]) == 64 and
                all(char in "0123456789abcdef" for char in parts[0]),
                "invalid M1245 manifest row")
        name = parts[1].lstrip("*")
        require(Path(name).name == name and name not in rows,
                "invalid or duplicate M1245 member")
        member = directory_path / name
        regular(member, "M1245 member " + name)
        require(sha256(member) == parts[0], "M1245 member SHA drift: " + name)
        rows[name] = parts[0]
    require("review.json" in rows, "M1245 review.json missing from manifest")
    review = json.loads((directory_path / "review.json").read_text(encoding="utf-8"))
    require(review.get("schema") == M1245_REVIEW_SCHEMA, "M1245 schema mismatch")
    require(review.get("status") == M1245_REVIEW_STATUS, "M1245 status mismatch")
    authority = review.get("authority")
    require(isinstance(authority, dict) and
            authority.get("release_authoring_allowed") is True and
            authority.get("production_binder_execution_allowed_by_this_review") is False and
            authority.get("hardware_rebind_authorized") is False and
            authority.get("result_hammer_still_required") is True,
            "M1245 authority boundary mismatch")
    return review


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    run_rel: Path
    config_rel: Path
    epoch: int


@dataclass(frozen=True)
class Policy:
    repo: Path
    interpreter: Path
    python_version: str
    m1241_pins: dict[Path, str]
    m1245_rel: Path
    m1245_manifest_sha256: str
    m1245_outer_sha256: str
    docs_rel: Path
    docs_sha256: str
    candidates: tuple[CandidateInput, ...]
    new_manifest_rel: Path
    output_rel: Path
    attempt_rel: Path
    log_rel: Path


PRODUCTION_POLICY = Policy(
    repo=REPO,
    interpreter=INTERPRETER,
    python_version=PYTHON_VERSION,
    m1241_pins=M1241_PINS,
    m1245_rel=M1245_REL,
    m1245_manifest_sha256=M1245_MANIFEST_SHA256,
    m1245_outer_sha256=M1245_OUTER_SHA256,
    docs_rel=DOCS359_REL,
    docs_sha256=DOCS359_SHA256,
    candidates=(
        CandidateInput("legacy_ep29", OLD_RUN_REL, OLD_CONFIG_REL, 29),
        CandidateInput("resume_ep30", NEW_RUN_REL, NEW_CONFIG_REL, 30),
        CandidateInput("resume_ep32", NEW_RUN_REL, NEW_CONFIG_REL, 32),
        CandidateInput("resume_ep34", NEW_RUN_REL, NEW_CONFIG_REL, 34),
    ),
    new_manifest_rel=NEW_MANIFEST_REL,
    output_rel=OUTPUT_REL,
    attempt_rel=ATTEMPT_REL,
    log_rel=LOG_REL,
)


def artifact_files(policy: Policy) -> tuple[Path, ...]:
    values = [policy.repo / policy.new_manifest_rel]
    values.extend(policy.repo / relative for relative in {
        row.config_rel for row in policy.candidates})
    for row in policy.candidates:
        run = policy.repo / row.run_rel
        values.append(run / "checkpoint_epoch{}.pth".format(row.epoch))
        values.append(run / "standard_valid825" /
                      "epoch{}".format(row.epoch) / "spike_profile.json")
    return tuple(values)


def _fresh(path: Path, label: str) -> None:
    require(not path.exists() and not path.is_symlink(),
            "fresh {} namespace required: {}".format(label, path))


def preflight(policy: Policy, executable_path: Path, version: str,
              cwd: Path) -> list[str]:
    require(executable_path == policy.interpreter, "interpreter path mismatch")
    require(version == policy.python_version, "interpreter version mismatch")
    require(cwd == policy.repo, "repository cwd mismatch")
    directory(policy.repo, "repository")
    executable(policy.interpreter, "production interpreter")

    for relative, expected in policy.m1241_pins.items():
        path = policy.repo / relative
        regular(path, "pinned M1241 input " + str(relative))
        require(sha256(path) == expected, "M1241 input SHA drift: " + str(relative))
    verify_double_sealed_review(
        policy.repo / policy.m1245_rel,
        policy.m1245_manifest_sha256,
        policy.m1245_outer_sha256)
    docs = policy.repo / policy.docs_rel
    regular(docs, "protected docs/359")
    require(sha256(docs) == policy.docs_sha256, "protected docs/359 SHA drift")

    run_paths = {policy.repo / row.run_rel for row in policy.candidates}
    require(len(run_paths) == 2, "exactly two run paths required")
    for run in run_paths:
        directory(run, "candidate run")
    expected_ids = ("legacy_ep29", "resume_ep30", "resume_ep32", "resume_ep34")
    expected_epochs = (29, 30, 32, 34)
    require(tuple(row.candidate_id for row in policy.candidates) == expected_ids and
            tuple(row.epoch for row in policy.candidates) == expected_epochs,
            "exact four candidate topology mismatch")
    for path in artifact_files(policy):
        regular(path, "required completed candidate artifact")

    output = policy.repo / policy.output_rel
    attempt = policy.repo / policy.attempt_rel
    log = policy.repo / policy.log_rel
    require(len({str(output), str(attempt), str(log)}) == 3,
            "output/attempt/log namespaces must be distinct")
    require(output.parent == attempt.parent == log.parent,
            "output/attempt/log must share the pinned results parent")
    directory(output.parent, "results parent")
    _fresh(output, "output")
    _fresh(attempt, "attempt")
    _fresh(log, "log")

    child = policy.repo / M1241_REL
    return [str(policy.interpreter), str(child), "--ranking-mode", "aee",
            "--output-dir", str(output)]


def consume_attempt(policy: Policy, command: Sequence[str]) -> None:
    attempt = policy.repo / policy.attempt_rel
    descriptor = os.open(attempt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        body = (
            "M1248_M1241_PRODUCTION_BINDER_ATTEMPT_CONSUMED_BEFORE_CHILD\n"
            "automatic_retry=false\n"
            "output={}\nlog={}\ncommand_sha256={}\n".format(
                policy.repo / policy.output_rel,
                policy.repo / policy.log_rel,
                hashlib.sha256(os.fsencode(chr(0).join(command))).hexdigest()))
        os.write(descriptor, body.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_log(policy: Policy, command: Sequence[str],
                completed: subprocess.CompletedProcess[str]) -> None:
    log = policy.repo / policy.log_rel
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        body = (
            "M1248_ONE_SHOT_CHILD_LOG\nreturncode={}\ncommand_sha256={}\n"
            "stdout_sha256={}\nstderr_sha256={}\n".format(
                completed.returncode,
                hashlib.sha256(os.fsencode(chr(0).join(command))).hexdigest(),
                hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()))
        os.write(descriptor, body.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_selection_receipt(output: Path) -> dict[str, object]:
    directory(output, "M1241 selection receipt")
    observed = set()
    for member in output.iterdir():
        regular(member, "selection receipt member " + member.name)
        observed.add(member.name)
    require(observed == RESULT_PAYLOADS | {MANIFEST, OUTER},
            "selection receipt member population mismatch")
    manifest = output / MANIFEST
    outer = output / OUTER
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), MANIFEST],
            "selection receipt outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and
                all(char in "0123456789abcdef" for char in fields[0]),
                "invalid selection manifest row")
        name = fields[1].lstrip("*")
        require(name in RESULT_PAYLOADS and Path(name).name == name and name not in rows,
                "invalid or duplicate selection member")
        rows[name] = fields[0]
    require(set(rows) == RESULT_PAYLOADS, "selection manifest population mismatch")
    for name, expected in rows.items():
        require(sha256(output / name) == expected, "selection payload SHA drift: " + name)
    require((output / "RUN_COMPLETE.txt").read_text(encoding="utf-8") == RUN_COMPLETE,
            "selection terminal token mismatch")
    result = json.loads((output / "final_checkpoint_selection.json").read_text(
        encoding="utf-8"))
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == RESULT_STATUS,
            "selection schema/status mismatch")
    boundary = result.get("claim_boundary")
    require(isinstance(boundary, dict) and
            boundary.get("fresh_result_hammer_required") is True and
            boundary.get("hardware_rebind_authorized") is False and
            boundary.get("hardware_speedup") is False and
            boundary.get("system_speedup") is False,
            "selection claim boundary mismatch")
    selected = result.get("selected")
    require(isinstance(selected, dict) and
            selected.get("candidate_id") in {
                "legacy_ep29", "resume_ep30", "resume_ep32", "resume_ep34"} and
            selected.get("epoch") in {29, 30, 32, 34},
            "selected candidate/epoch mismatch")
    return result


Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})


def execute_once(policy: Policy, executable_path: Path, version: str, cwd: Path,
                 runner: Runner = default_runner) -> subprocess.CompletedProcess[str]:
    command = preflight(policy, executable_path, version, cwd)
    consume_attempt(policy, command)
    completed = runner(command, policy.repo)
    publish_log(policy, command, completed)
    require(completed.returncode == 0,
            "single M1241 child failed after attempt consumption; no retry authorized")
    require(completed.stdout.count(CHILD_TOKEN) == 1,
            "single M1241 child terminal stdout mismatch")
    verify_selection_receipt(policy.repo / policy.output_rel)
    return completed


def main() -> int:
    require(len(sys.argv) == 1, "production M1248 release accepts zero arguments")
    completed = execute_once(
        PRODUCTION_POLICY, Path(sys.executable), platform.python_version(), Path.cwd())
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1248_M1241_ONE_SHOT_SELECTION_RECEIPT__FRESH_RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
