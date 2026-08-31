#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Supersede the incomplete M1087 GO after a contract/population cross-check."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / "system_simulator/scripts/run_m1086_c1_zero_work_exact_1rw_source.py"
CONTRACT = HW / "contracts/m1086_c1_zero_work_exact_1rw_source_contract_r1_20260830.json"
M1087 = HW / "reviews/m1087_m1086_c1_zero_work_exact_1rw_source_hammer_r1_20260830"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED_SOURCE = "3925c97de922393786b4aa8ae6ca6b4942489e3cf10485f5d1b6cd423e797a51"
EXPECTED_CONTRACT = "cd4a315d0f153925acee893fd24d9d2b227d45ef9e40e2534f76e35c8abfebe8"
EXPECTED_M1087_OUTER = "dfe945edfce00b3a8d2995279daa432bcefae3feda001ee265c5ff63f3219a55"
EXPECTED_DOCS = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def req(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_flat(directory: Path, expected_outer: str) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    req(directory.is_dir() and not directory.is_symlink(), "M1087 directory drift")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        member = directory / name.lstrip("*")
        req(member.is_file() and not member.is_symlink() and sha(member) == expected,
            "M1087 member drift")
    req(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"] and
        sha(outer) == expected_outer, "M1087 outer drift")


def seal(directory: Path) -> tuple[str, str]:
    members = sorted(path for path in directory.rglob("*")
                     if path.is_file() and path.name not in
                     {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members),
        encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(manifest), sha(outer)


def main() -> None:
    req(sha(SOURCE) == EXPECTED_SOURCE and sha(CONTRACT) == EXPECTED_CONTRACT and
        sha(DOCS) == EXPECTED_DOCS, "frozen identity drift")
    verify_flat(M1087, EXPECTED_M1087_OUTER)
    spec = importlib.util.spec_from_file_location("m1087r2_target_m1086", SOURCE)
    req(spec is not None and spec.loader is not None, "cannot load source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tasks = module.M1072.TASKS
    designs = len(module.DESIGNS)
    values = tasks * designs
    declared_values = contract["production_interfaces"]["expected_values_checked"]
    declared_text = contract["production_interfaces"]["work_domain_preflight"]
    req((tasks, designs, values) == (812160, 3, 2436480),
        "unexpected frozen geometry")
    req(declared_values == 812160 and "270720 tasks x 3 designs" in declared_text,
        "expected M1086 contract discrepancy absent")
    req("values_checked':M1072.TASKS*3" in SOURCE.read_text(encoding="utf-8"),
        "source no longer exposes the conflicting actual population")
    review = {
        "schema": "m1087r2_m1087_m1086_population_supersession_review_v1",
        "status": "STOP_M1087R2_M1086_CONTRACT_POPULATION_MISMATCH",
        "verdict": "SUPERSEDE_AND_REVOKE_M1087_GO__NO_RELEASE__ADDITIVE_CONTRACT_REPAIR_REQUIRED",
        "score": 62,
        "p0_count": 1,
        "receipt_blind": True,
        "identity": {
            "m1086_source_sha256": EXPECTED_SOURCE,
            "m1086_contract_sha256": EXPECTED_CONTRACT,
            "superseded_m1087_outer_seal_file_sha256": EXPECTED_M1087_OUTER,
            "docs359_sha256": EXPECTED_DOCS,
        },
        "finding": {
            "severity": "P0",
            "id": "M1087R2-P0-01",
            "title": "M1086 sealed contract understates exhaustive production population by 3x",
            "frozen_tasks": tasks,
            "frozen_designs": designs,
            "actual_task_design_values": values,
            "contract_declared_tasks": 270720,
            "contract_declared_task_design_values": declared_values,
            "understatement_factor": values / declared_values,
            "source_loop_population_is_correct": True,
            "source_return_values_checked": "M1072.TASKS*3",
            "impact": "A successful production preflight would return 2436480, contradicting the sealed contract's expected 812160. A runner cannot bind both identities fail-closed.",
        },
        "supersession": {
            "m1087_go_usable": False,
            "m1087_release_created": False,
            "reason": "The first hammer checked frozen task count but failed to cross-check it against both contract population fields.",
        },
        "repair": {
            "allowed": True,
            "scope": "new additive M1086 contract/author receipt namespace only; keep source/tests/M1056/M1072/M1074 frozen",
            "required_values": {
                "tasks": 812160,
                "designs": 3,
                "task_design_values": 2436480,
            },
            "new_independent_hammer_required": True,
        },
        "execution": {
            "exhaustive_preflight_executed": False,
            "full_replay_executed": False,
            "attempt_consumed": False,
            "eda_gpu_remote_used": False,
        },
        "claim_boundary": {
            "one_shot_wrapper_authoring_authorized": False,
            "full_replay_execution_authorized": False,
            "matched_cycles_admitted": False,
            "speedup_admitted": False,
            "rtl_cycles": False,
            "paper_ppa_ready": False,
        },
    }
    checks = {
        "status": "STOP",
        "frozen_tasks": tasks,
        "designs": designs,
        "actual_task_design_values": values,
        "contract_expected_values_checked": declared_values,
        "contract_text": declared_text,
        "ratio": values / declared_values,
        "preflight_executed": False,
        "full_replay_executed": False,
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n",
                                      encoding="utf-8")
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(checks, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (HERE / "review.md").write_text(
        "# M1087r2 population supersession\n\n"
        "**STOP：M1087 GO 已撤回，不得生成 release。**\n\n"
        "冻结几何是 812,160 tasks × 3 designs = 2,436,480 task-design work values；"
        "M1086 封存 contract 却写成 270,720 tasks × 3 且 expected_values_checked=812,160。"
        "source 实际 loop 是对的，但生产 preflight 返回必与 contract 冲突。"
        "只允许新 namespace 的 additive contract/receipt 修复，然后重做独立 hammer。\n",
        encoding="utf-8")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "STOP_M1087R2_M1086_CONTRACT_POPULATION_MISMATCH\n", encoding="utf-8")
    manifest, outer = seal(HERE)
    print("M1087R2_REVIEW_SHA=" + sha(HERE / "review.json"))
    print("M1087R2_MANIFEST_SHA=" + manifest)
    print("M1087R2_OUTER_SHA=" + outer)


if __name__ == "__main__":
    main()
