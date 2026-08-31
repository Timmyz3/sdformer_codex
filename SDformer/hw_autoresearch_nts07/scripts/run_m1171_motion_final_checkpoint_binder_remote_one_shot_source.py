#!/opt/conda/envs/sdformerflow/bin/python
"""M1171 one-shot remote launcher for the sealed M1167 checkpoint binder.

This program is intended to run *on* the A800 host from the repository root.
It performs a read-only fail-closed preflight, atomically consumes one attempt,
and invokes exactly one pinned M1167 child.  It never retries.  The M1167
child reads/hashes completed validation artifacts and publishes its own sealed
small receipt; it does not run evaluation, copy a checkpoint, or start EDA.

Author and independent review must exercise the injectable functions with
temporary fixtures only.  They must not call ``main``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
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

R1_REL = Path("hw_autoresearch_nts07/scripts/build_m1163_motion_final_checkpoint_selection_rebind_binder.py")
R2_REL = Path("hw_autoresearch_nts07/scripts/build_m1166_motion_final_checkpoint_selection_rebind_binder_r2.py")
R3_REL = Path("hw_autoresearch_nts07/scripts/build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py")
DOCS359_REL = Path("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md")
SOURCE_SHA256 = {
    R1_REL: "50d22cb0f7d656c79eeb99894cb85c975441f16fd46d7df55c37ff34976aaf32",
    R2_REL: "2171da4909fc1844c1323ca5138ccc1232fdad61d3b00446709a144461d7472c",
    R3_REL: "7ea88b861ad54f6029f2631766a7da21b3626054217d36c27c4509293ce35d89",
}
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RUN_DIR = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx"
)
CONFIG = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
)
CONFIG_SHA256 = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
RANKING = RUN_DIR / "profile_ranking_valid825.md"
EPOCHS = (9, 14, 19, 24, 29)
OUTPUT = HW / "results/m1171_motion_final_checkpoint_selection_rebind_binder_r4_20260830"
ATTEMPT = HW / "results/.m1171_motion_final_checkpoint_selection_rebind_binder_r4_attempt_consumed"

PAYLOAD_MEMBERS = frozenset({
    "RUN_COMPLETE.txt",
    "e0_e8_rebind_targets.json",
    "final_checkpoint_selection.json",
    "five_checkpoint_metrics.csv",
})
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class LaunchError(RuntimeError):
    """Fail-closed launcher error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


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
        raise LaunchError(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),
            f"{label} must be a non-symlink regular file: {path}")


def directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise LaunchError(f"missing {label}: {path}") from exc
    require(stat.S_ISDIR(value.st_mode) and not path.is_symlink(),
            f"{label} must be a non-symlink directory: {path}")


def executable_path(path: Path, label: str) -> None:
    """Accept a conda python symlink while pinning the invoked path itself."""
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise LaunchError(f"missing {label}: {path}") from exc
    require(stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode),
            f"{label} must be a regular file or symlink: {path}")
    try:
        target = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise LaunchError(f"broken or cyclic {label}: {path}") from exc
    require(stat.S_ISREG(target.stat().st_mode) and os.access(target, os.X_OK),
            f"{label} target must be an executable regular file: {target}")


@dataclass(frozen=True)
class Policy:
    repo: Path
    interpreter: Path
    python_version: str
    source_sha256: dict[Path, str]
    docs_rel: Path
    docs_sha256: str
    run_dir: Path
    config: Path
    config_sha256: str
    ranking: Path
    epochs: tuple[int, ...]
    output: Path
    attempt: Path


PRODUCTION_POLICY = Policy(
    repo=REPO,
    interpreter=INTERPRETER,
    python_version=PYTHON_VERSION,
    source_sha256=SOURCE_SHA256,
    docs_rel=DOCS359_REL,
    docs_sha256=DOCS359_SHA256,
    run_dir=RUN_DIR,
    config=CONFIG,
    config_sha256=CONFIG_SHA256,
    ranking=RANKING,
    epochs=EPOCHS,
    output=OUTPUT,
    attempt=ATTEMPT,
)


def preflight(policy: Policy, executable: Path, version: str, cwd: Path) -> list[str]:
    require(executable == policy.interpreter,
            f"interpreter path mismatch: {executable} != {policy.interpreter}")
    require(version == policy.python_version,
            f"interpreter version mismatch: {version} != {policy.python_version}")
    require(cwd == policy.repo, f"repository cwd mismatch: {cwd} != {policy.repo}")
    directory(policy.repo, "repository")
    executable_path(policy.interpreter, "remote interpreter")

    for relative, expected in policy.source_sha256.items():
        candidate = policy.repo / relative
        regular(candidate, f"sealed source {relative}")
        require(sha256(candidate) == expected, f"sealed source SHA drift: {relative}")
    docs = policy.repo / policy.docs_rel
    regular(docs, "protected docs/359")
    require(sha256(docs) == policy.docs_sha256, "protected docs/359 SHA drift")

    directory(policy.run_dir, "remote training run")
    regular(policy.config, "remote configuration")
    require(sha256(policy.config) == policy.config_sha256, "configuration SHA drift")
    regular(policy.ranking, "standard-valid825 ranking")
    standard = policy.run_dir / "standard_valid825"
    directory(standard, "standard_valid825")
    expected_names = {f"epoch{epoch}" for epoch in policy.epochs}
    entries = list(standard.iterdir())
    require(len(entries) == len(expected_names) and {item.name for item in entries} == expected_names,
            "standard_valid825 canonical epoch population mismatch")
    for epoch in policy.epochs:
        epoch_dir = standard / f"epoch{epoch}"
        directory(epoch_dir, f"epoch{epoch} profile directory")
        regular(epoch_dir / "spike_profile.json", f"epoch{epoch} spike profile")

    require(not policy.output.exists() and not policy.output.is_symlink(),
            f"fresh output namespace required: {policy.output}")
    require(not policy.attempt.exists() and not policy.attempt.is_symlink(),
            f"attempt already consumed: {policy.attempt}")
    directory(policy.output.parent, "remote results parent")

    r3 = policy.repo / R3_REL
    return [
        str(policy.interpreter), str(r3),
        "--run-dir", str(policy.run_dir),
        "--config", str(policy.config),
        "--ranking", str(policy.ranking),
        "--ranking-mode", "aee",
        "--output-dir", str(policy.output),
    ]


def consume_attempt(policy: Policy, command: Sequence[str]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(policy.attempt, flags, 0o400)
    try:
        payload = (
            "M1171_REMOTE_BINDER_ATTEMPT_CONSUMED_BEFORE_CHILD\n"
            f"interpreter={policy.interpreter}\n"
            f"python_version={policy.python_version}\n"
            f"output={policy.output}\n"
            "automatic_retry=false\n"
            f"command_sha256={hashlib.sha256(os.fsencode(chr(0).join(command))).hexdigest()}\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_sealed_output(output: Path) -> None:
    directory(output, "M1171 binder output")
    observed: set[str] = set()
    for member in output.iterdir():
        regular(member, f"M1171 output member {member.name}")
        observed.add(member.name)
    require(observed == PAYLOAD_MEMBERS | {MANIFEST, OUTER},
            f"sealed output member set mismatch: {sorted(observed)}")
    manifest = output / MANIFEST
    outer = output / OUTER
    outer_fields = outer.read_text(encoding="utf-8").split()
    require(outer_fields == [sha256(manifest), MANIFEST], "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and
                all(char in "0123456789abcdef" for char in fields[0]),
                "invalid manifest row")
        name = fields[1].lstrip("*")
        require(name in PAYLOAD_MEMBERS and name not in rows and Path(name).name == name,
                "invalid or duplicate manifest member")
        rows[name] = fields[0]
    require(set(rows) == PAYLOAD_MEMBERS, "manifest payload population mismatch")
    for name, expected in rows.items():
        require(sha256(output / name) == expected, f"payload SHA mismatch: {name}")
    require((output / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
            "binder terminal token mismatch")


Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def execute_once(
    policy: Policy,
    executable: Path,
    version: str,
    cwd: Path,
    runner: Runner = default_runner,
) -> subprocess.CompletedProcess[str]:
    command = preflight(policy, executable, version, cwd)
    consume_attempt(policy, command)
    completed = runner(command, policy.repo)
    require(completed.returncode == 0,
            "single binder child failed after attempt consumption; no retry authorized")
    require(completed.stdout.count(
        "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
        "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY"
    ) == 1, "single binder child terminal stdout mismatch")
    verify_sealed_output(policy.output)
    return completed


def main() -> int:
    require(len(sys.argv) == 1, "production M1171 launcher accepts zero arguments")
    completed = execute_once(
        PRODUCTION_POLICY,
        Path(sys.executable),
        platform.python_version(),
        Path.cwd(),
    )
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    print("PASS_M1171_REMOTE_ONE_SHOT_LAUNCH_AND_SEALED_M1167_RESULT__RESULT_HAMMER_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
