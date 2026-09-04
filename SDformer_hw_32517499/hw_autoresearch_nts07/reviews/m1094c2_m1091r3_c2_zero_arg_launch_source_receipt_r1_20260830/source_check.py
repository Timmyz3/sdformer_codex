#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static-only M1094C2 launcher audit; never imports or executes launch code."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat


HW = Path(__file__).resolve().parents[2]
LAUNCHER = HW / "dc_handoff/scripts/run_m1091r3_m1090r3_c2_observation_authorized_launch_r1.py"
LAUNCHER_SHA = "64eb690f557c8aa61461034f714a8eefe7e7176aa85c700e3f3290f2b902f56a"
ENGINE = HW / "dc_handoff/scripts/m1091r3_m1090r3_c2_observation_authorized_engine_r1.py"
ENGINE_SHA = "41b7899083152f8099acac759109a8eb22c381cb6a17506ae85e6666656daf04"
CONTRACT = HW / "contracts/m1094c2_m1091r3_c2_zero_arg_launch_source_contract_r1_20260830.json"
CONTRACT_SHA = "171d6007b18f64ebebb0d353d0e7d2d655a37ef66f74cdf6b60b8ea2cdbe5551"
CONTRACT_SIDE_SHA = "fc06fa6af72a05819c0cf6bec9c28a4a71d1d6c00ca51323c9955a5bc2d38e04"
LAUNCH_RECEIPT = HW / "contracts/m1091r3_m1090r3_c2_observation_authorized_launch_receipt_r1_20260830.json"
LAUNCH_RECEIPT_SHA = "538e6cfb9323a06040f229fdde0a20e3f3d5acfc49b383bc2442f3939afbb10d"
LAUNCH_RECEIPT_SIDE_SHA = "c9621244aa5fb0ab32044e68c19f3744219648a77a2191cd5eb119cf75cc2f96"
SOURCE_RECEIPT = HW / "reviews/m1090r3_m1091r3_c2_observation_fixed_history_source_receipt_r1_20260830"
SOURCE_RECEIPT_OUTER = "8bc6f725ef0ec7055441afafa2c0bd5c5ba54620c4354feaf2a6763fbabedd9e"
M1093R2 = HW / "reviews/m1093r2_m1090r3_m1091r3_c2_observation_engine_hammer_r1_20260830"
M1093R2_OUTER = "d6fa5ecb89342188586fb179d9dcaa1018078b4f3db6c609f6f1fd1b0559f9cc"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ATTEMPT = HW / "results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830"


checks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " sha")


def double(path: Path, file_sha: str, side_sha: str, label: str) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, file_sha, label + " file")
    regular(side, side_sha, label + " side")
    require(side.read_text(encoding="utf-8").split() ==
            [file_sha, path.relative_to(HW).as_posix()], label + " side content")
    require(outer.is_file() and not outer.is_symlink(), label + " outer regular")
    require(outer.read_text(encoding="utf-8").split() ==
            [side_sha, side.relative_to(HW).as_posix()], label + " outer content")
    return sha(outer)


def flat(directory: Path, expected_outer: str, status: str, label: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), label + " directory")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(outer, expected_outer, label + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [sha(manifest), "SHA256SUMS"], label + " outer content")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        regular(directory / relative.lstrip("*"), expected, label + " member " + relative)
    review = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    require(review["status"] == status, label + " status")


regular(LAUNCHER, LAUNCHER_SHA, "launcher")
regular(ENGINE, ENGINE_SHA, "engine")
regular(DOCS359, DOCS359_SHA, "docs359")
contract_outer = double(CONTRACT, CONTRACT_SHA, CONTRACT_SIDE_SHA, "contract")
launch_receipt_outer = double(
    LAUNCH_RECEIPT, LAUNCH_RECEIPT_SHA, LAUNCH_RECEIPT_SIDE_SHA, "launch receipt"
)
flat(
    SOURCE_RECEIPT,
    SOURCE_RECEIPT_OUTER,
    "PASS_M1090R3_M1091R3_FIXED_HISTORY_SOURCE_ONLY__M1093R2_REQUIRED__NO_EDA",
    "source receipt",
)
flat(
    M1093R2,
    M1093R2_OUTER,
    "PASS_M1093R2_M1090R3_M1091R3_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA",
    "M1093r2",
)

source = LAUNCHER.read_text(encoding="utf-8")
tree = ast.parse(source)
require('ENGINE_SHA256 = "' + ENGINE_SHA + '"' in source, "engine literal pin")
require('M1093R2_OUTER_SHA256 = "' + M1093R2_OUTER + '"' in source,
        "M1093r2 literal pin")
require("len(sys.argv) == 1" in source, "zero argv gate")
require("os.getenv" not in source and "os.environ[" not in source and
        "os.environ.get" not in source, "no caller environment reads")
require("expected_hash" not in source.lower(), "no caller expected-hash channel")
run_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
require(len(run_calls) == 1, "one subprocess run call")
run_text = ast.unparse(run_calls[0])
require("[str(PYTHON), '-I', str(ENGINE), '--authorized-launch']" in run_text,
        "exact child argv")
require("env=clean_child_environment()" in run_text, "constant child environment")
clean_function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "clean_child_environment"
)
clean_text = ast.unparse(clean_function)
require("os.environ" not in clean_text and "getenv" not in clean_text,
        "environment constructor caller-blind")
require(all(token in source for token in
            ("27030@ic.ismd-nemo", "/opt/synopsys/Synopsys.dat",
             "/usr/bin:/bin", "C.UTF-8", "/tmp")),
        "environment constants present")

receipt = json.loads(LAUNCH_RECEIPT.read_text(encoding="utf-8"))
require(receipt["status"] == "M1091R3_LAUNCH_SOURCE_FROZEN__M1096R2_REQUIRED__NO_EDA",
        "engine-compatible receipt status")
require(receipt["launcher_sha256"] == LAUNCHER_SHA, "receipt launcher pin")
require(receipt["engine_sha256"] == ENGINE_SHA, "receipt engine pin")
require(receipt["m1093r2_outer_seal_file_sha256"] == M1093R2_OUTER,
        "receipt M1093r2 pin")
require(receipt["launch_now"] is False and receipt["attempt_now"] is False and
        receipt["dc_now"] is False and receipt["mapped_vcs_now"] is False,
        "receipt no-launch boundary")
require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), "attempt absent")
require(not RESULT.exists() and not RESULT.is_symlink(), "result absent")
require(not any((HW / "results").glob(
            ".m1091r3_m1090r3_c2_observation_dc_mapped_vcs_work.*")), "work absent")
require(not any((HW / "results").glob(
            "m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
        "quarantine absent")

print(json.dumps({
    "status": "PASS_M1094C2_STATIC_SOURCE_CHECK__NO_LAUNCH_NO_EDA",
    "checks_passed": len(checks),
    "launcher_sha256": LAUNCHER_SHA,
    "engine_sha256": ENGINE_SHA,
    "contract_outer_seal_file_sha256": contract_outer,
    "launch_receipt_outer_seal_file_sha256": launch_receipt_outer,
    "m1093r2_outer_seal_file_sha256": M1093R2_OUTER,
    "attempt_absent": True,
    "result_absent": True,
    "dc_commands": 0,
    "vcs_commands": 0,
}, indent=2, sort_keys=True))
