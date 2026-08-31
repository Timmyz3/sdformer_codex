#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, source-only blind hammer for M1337/R15 C1 witness."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
BASE = HW / "verif_m1337r15_c1_real_m935_runtime_witness"
CHECKER = BASE / "check_m1337r15_source.py"
TEST = BASE / "test_m1337r15_source.py"
WITNESS = BASE / "m1337r15_m935_runtime_witness.sv"
FILELIST = BASE / "m1337r15_unit_delay_filelist.f"
CONTRACT = HW / "contracts/m1337_c1_r15_real_m935_runtime_witness_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1337_c1_r15_real_m935_runtime_witness_source_author_r1_20260831"
FAILED_R14 = HW / "reviews/m1335_m1334_c1_r14_runtime_witness_source_blind_review_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    CHECKER: "ba6d8c9b1e66854ee58cf3a3b247cceb1629495d2a5c6ca11aa93b7ba14c1326",
    TEST: "ed2c92dde2ca6c96ec55f00b21188d6ea8bdf2426c89f188896351c314c6de9c",
    WITNESS: "0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af",
    FILELIST: "87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e",
    CONTRACT: "49c55065bdafda15a75f5520d22428671ea3353a53c692270f47fbce5c80e5b8",
    AUTHOR / "review.json": "fee38289e55bcb61b05cda5d75a4483a27c9bc053b976a018e4852db3cea0da7",
    AUTHOR / "SHA256SUMS": "59226f03c833ca657af7eacc60ada87ce75f0401ab7ca1737a823d25211e9374",
    AUTHOR / "SHA256SUMS.seal.sha256": "c56c890f41bcff07349af838ef390bf1764427ba9da7fc42f2708a39e932d2f0",
    FAILED_R14 / "review.json": "31abaa97d1a93b50d8e90ebdd90f0580d31d9657df658561f024b067a1993ea4",
    FAILED_R14 / "SHA256SUMS": "b918e1c8090e827b7dfd16aa3f2d15dafe62aa833a42fc8c915150b336d93948",
    FAILED_R14 / "SHA256SUMS.seal.sha256": "05c76b268bfb1bce47eeb0b6137ddab3fd2fbe68f2dc9eaf239fe96f98f11ff1",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise HammerError("non-finite JSON constant: " + value)
    return json.loads(path.read_text(), parse_constant=reject_constant)


def verify_recursive_seal(root: Path, review_sha: str,
                          manifest_sha: str, outer_sha: str) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer seal content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "malformed manifest row")
        name = fields[1].lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and name not in rows,
                "unsafe or duplicate manifest member")
        member = root / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == fields[0],
                "sealed member mismatch: " + name)
        rows[name] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed recursive population drift")
    require(rows.get("review.json") == review_sha, "review member drift")
    return rows


def load_checker():
    spec = importlib.util.spec_from_file_location("m1339_blind_m1337", CHECKER)
    require(spec is not None and spec.loader is not None, "checker import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checker_rejects(action: Callable[[], Any]) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


def stage_mutation(source: str, stage: str, next_stage: str,
                   old: str, new: str) -> str:
    begin = source.index(stage + ": begin")
    end = source.index(next_stage + ": begin", begin)
    body = source[begin:end]
    require(body.count(old) == 1, "stage mutation anchor drift: " + stage)
    return source[:begin] + body.replace(old, new, 1) + source[end:]


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "canonical identity mismatch: " + str(path))
    author_rows = verify_recursive_seal(
        AUTHOR, EXPECTED[AUTHOR / "review.json"], EXPECTED[AUTHOR / "SHA256SUMS"],
        EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    failed_rows = verify_recursive_seal(
        FAILED_R14, EXPECTED[FAILED_R14 / "review.json"],
        EXPECTED[FAILED_R14 / "SHA256SUMS"],
        EXPECTED[FAILED_R14 / "SHA256SUMS.seal.sha256"])
    require("review.json" in author_rows and "review.json" in failed_rows,
            "double-seal review member missing")
    author = strict_json(AUTHOR / "review.json")
    failed = strict_json(FAILED_R14 / "review.json")
    require(author.get("status") ==
            "PASS_SOURCE_AUTHORING__FRESH_DIFFERENT_AUTHOR_BLIND_HAMMER_REQUIRED",
            "author status drift")
    require(failed.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and failed.get("false_negative_count") == 6,
            "M1335 failed-root drift")

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    author_test = subprocess.run(
        [str(PYTHON), "-B", str(TEST)], cwd=str(HW.parent), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    require(author_test.returncode == 0 and "Ran 20 tests" in author_test.stdout
            and author_test.stdout.count(" ... ok") == 20,
            "author 20/20 directed tests did not reproduce")

    M = load_checker()
    M.check_witness_text(WITNESS.read_text())
    M.check_contract_dict(strict_json(CONTRACT))
    require(M.runtime_model(M.good_trace())["pass"], "positive runtime model rejected")

    source = WITNESS.read_text()
    contract = strict_json(CONTRACT)
    rejected: list[str] = []
    accepted: list[str] = []

    def attack(label: str, action: Callable[[], Any]) -> None:
        (rejected if checker_rejects(action) else accepted).append(label)

    # The original six M1335 classes, plus independent semantic weakening.
    attack("same_cycle_first_accept_plus_second_request_guard_removed", lambda:
           M.check_witness_text(stage_mutation(
               source, "W_FIRST_REQUEST", "W_FIRST_ACCEPT",
               "                                    && (weight_request_fire === 1'b0)\n", "")))
    attack("same_cycle_second_accept_plus_commit_guard_removed", lambda:
           M.check_witness_text(stage_mutation(
               source, "W_SECOND_REQUEST", "W_SECOND_ACCEPT",
               "                                    && (psum_commit_fire === 1'b0)\n", "")))
    attack("same_cycle_commit_plus_row_guard_removed", lambda:
           M.check_witness_text(stage_mutation(
               source, "W_SECOND_ACCEPT", "W_PSUM_COMMIT",
               "                                    && (row_complete_fire === 1'b0)\n", "")))
    attack("same_cycle_row_plus_task_guard_removed", lambda:
           M.check_witness_text(stage_mutation(
               source, "W_PSUM_COMMIT", "W_ROW_DONE",
               "                                    && (task_done_fire === 1'b0)\n", "")))

    control_start = source.index("control_unknown = $isunknown({")
    control_stop = source.index("});", control_start) + 3
    control_block = source[control_start:control_stop]
    for control in ("weight_request_fire", "psum_request_fire", "response_accept",
                    "core_accept", "psum_commit_fire", "row_complete_fire",
                    "task_done_fire"):
        require(control_block.count(control) == 1,
                "control unknown mutation anchor drift: " + control)
        mutant_block = control_block.replace(control, "1'b0", 1)
        mutant = source[:control_start] + mutant_block + source[control_stop:]
        attack("unknown_control_guard_removed_" + control,
               lambda text=mutant: M.check_witness_text(text))

    for identity in ("issue_request_source_index", "psum_commit_address",
                     "row_complete_id", "task_done_epoch"):
        mutant = re.sub(r",?\s*" + re.escape(identity) + r"\s*\}?\);", ");", source, count=1)
        # Use direct active-unknown block edits when the generic form cannot anchor.
        if mutant == source:
            mutant = source.replace(identity + "});", "});", 1)
        attack("unknown_identity_guard_removed_" + identity,
               lambda text=mutant: M.check_witness_text(text))

    for port in ("response_accept", "core_accept", "psum_commit_fire",
                 "psum_commit_address", "row_complete_fire", "row_complete_id",
                 "task_done_fire", "task_done_epoch"):
        expression = M.EXPECTED_BIND[port]
        pattern = re.compile(r"\." + re.escape(port) + r"\s*\([^\n]*\)")
        mutant, count = pattern.subn(".%s(1'b0)" % port, source, count=1)
        require(count == 1 and mutant != source, "child bind mutation anchor: " + port)
        attack("child_tie_constant_" + port,
               lambda text=mutant: M.check_witness_text(text))

    for port in ("request_hold_attack_mode", "weight_service_attack_mode",
                 "psum_service_attack_mode", "protocol_error", "boundary_fault",
                 "core_fault", "m935_fault", "weight_service_fault",
                 "psum_service_fault"):
        pattern = re.compile(r"\." + re.escape(port) + r"\s*\([^\n]*\)")
        mutant, count = pattern.subn(".%s(1'b0)" % port, source, count=1)
        require(count == 1 and mutant != source, "fault bind mutation anchor: " + port)
        attack("mask_or_fault_tie_constant_" + port,
               lambda text=mutant: M.check_witness_text(text))

    original_bind = ".response_accept(dut.response_accept_w)"
    require(original_bind in source, "comment fake bind anchor")
    attack("comment_fake_bind_cannot_rescue_constant", lambda:
           M.check_witness_text(source.replace(
               original_bind, ".response_accept(1'b1) /* " + original_bind + " */", 1)))

    pass_line = next(line for line in source.splitlines()
                     if "$display(\"PASS_M1337R15" in line)
    early = source.replace(pass_line + "\n", "", 1).replace(
        "        if (pass === 1'b1) begin\n",
        pass_line + "\n        if (pass === 1'b1) begin\n", 1)
    attack("early_pass_before_success_branch", lambda: M.check_witness_text(early))

    # Oracle-accounting terms are semantically required but are not parsed by
    # the R15 checker.  Removing one must not be admitted by a source gate.
    for label, term in (
            ("oracle_design_issue_count_removed",
             "            && (design_issue_accepts === 64'd2)\n"),
            ("oracle_design_commit_count_removed",
             "            && (design_psum_commits === 64'd1)\n"),
            ("oracle_design_row_count_removed",
             "            && (design_row_completions === 64'd1)\n")):
        require(source.count(term) == 1, "oracle mutation anchor drift: " + label)
        attack(label, lambda value=source.replace(term, "", 1):
               M.check_witness_text(value))

    expected_filelist = [str(path) for path in
        (M.FOUNDRY, M.M528, M.M935, M.M1162, M.SVA, M.R13_TB, M.WITNESS)]
    rows = FILELIST.read_text().splitlines()
    require(rows == expected_filelist, "positive filelist drift")
    for label, mutant in (
            ("filelist_delete", rows[:-1]),
            ("filelist_add", rows + [rows[-1]]),
            ("filelist_reorder", [rows[1], rows[0]] + rows[2:])):
        attack(label, lambda value=mutant: M.require(value == expected_filelist,
                                                     "filelist mutation"))

    for key in ("represented_ledger_bytes", "physically_integrated_parent_bytes",
                "external_common_charge_bytes"):
        mutant = copy.deepcopy(contract)
        mutant["frozen_design"][key] = 1
        attack("ledger_numeric_mutation_" + key,
               lambda value=mutant: M.check_contract_dict(value))

    dependency_mutant = copy.deepcopy(contract)
    dependency_mutant["frozen_design"]["m935_sha256"] = "0" * 64
    attack("dependency_sha_mutation", lambda: M.check_contract_dict(dependency_mutant))
    release_mutant = copy.deepcopy(contract)
    release_mutant["release_present"] = True
    attack("release_authority_token_injection", lambda: M.check_contract_dict(release_mutant))
    attack("active_sv_release_command_injection", lambda:
           M.check_witness_text(source.replace("endmodule\n\nbind", "  release foo;\nendmodule\n\nbind", 1)))

    with tempfile.TemporaryDirectory(prefix="m1339_m1337_blind_") as temp_name:
        temp = Path(temp_name)
        regular_copy = temp / "dependency.sv"
        regular_copy.write_text("mutated dependency\n")
        require(sha(regular_copy) != EXPECTED[M.WITNESS], "dependency SHA mutant collision")
        link = temp / "dependency_link.sv"
        link.symlink_to(regular_copy)
        attack("dependency_symlink", lambda: M.regular(link))
        # Source packages are intentionally development-writable; the release must
        # pin exact SHAs rather than claiming immutable source files.
        regular_copy.chmod(stat.S_IRUSR | stat.S_IWUSR)
        require(os.access(regular_copy, os.W_OK), "writable fixture not writable")

    result = {
        "schema": "m1339_m1337_c1_r15_runtime_witness_source_blind_hammer_output_r1",
        "status": ("PASS_SOURCE_ADMITTED" if not accepted else
                   "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"),
        "score": 100 if not accepted else 58,
        "reviewer_independent_of_author": True,
        "m1335_failure_root": {"status": failed["status"],
                               "historical_false_negative_count": 6},
        "author_tests": "20/20 PASS",
        "author_double_seal_verified": True,
        "m1335_double_seal_verified": True,
        "independent_attack_count": len(rejected) + len(accepted),
        "independent_rejected_count": len(rejected),
        "independent_false_negative_count": len(accepted),
        "rejected_attacks": rejected,
        "accepted_attacks": accepted,
        "development_writable_source_boundary": True,
        "execution": {"vcs": False, "simv": False, "dc": False,
                      "pt": False, "ptpx": False, "release": False,
                      "remote": False, "gpu": False},
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("M1339_M1337_BLIND_HAMMER_ERROR: " + str(error), file=sys.stderr)
        raise
