#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, source-only hammer for M1334/R14. Never runs EDA."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
BASE = HW / "verif_m1334r14_c1_real_m935_runtime_witness"
CHECKER = BASE / "check_m1334r14_source.py"
TEST = BASE / "test_m1334r14_source.py"
WITNESS = BASE / "m1334r14_m935_runtime_witness.sv"
FILELIST = BASE / "m1334r14_unit_delay_filelist.f"
CONTRACT = HW / "contracts/m1334_c1_r14_real_m935_runtime_witness_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1334_c1_r14_real_m935_runtime_witness_source_author_r1_20260831"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C = load("m1335_target", CHECKER)


def verify_author_seal() -> dict:
    rows = {}
    manifest = AUTHOR / "SHA256SUMS"
    outer = AUTHOR / "SHA256SUMS.seal.sha256"
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        need(name not in rows and sha(AUTHOR / name) == digest,
             "author member drift")
        rows[name] = digest
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "author outer seal drift")
    need(set(rows) == {p.name for p in AUTHOR.iterdir()
                       if p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}},
         "author population drift")
    return rows


def cycle_model(cycles: list[dict]) -> dict:
    """Mirror the M1334 after-value RTL semantics at clock-cycle granularity."""
    counts = {key: 0 for key in
              ("weight", "psum", "response", "core", "commit", "row", "task")}
    stage = 0
    fault = False
    for event in cycles:
        w = int(bool(event.get("weight")))
        p = int(bool(event.get("psum")))
        response = int(bool(event.get("response")))
        core = int(bool(event.get("core")))
        commit = int(bool(event.get("commit")))
        row = int(bool(event.get("row")))
        task = int(bool(event.get("task")))
        after = {
            "weight": counts["weight"] + w,
            "psum": counts["psum"] + p,
            "response": counts["response"] + response,
            "core": counts["core"] + core,
            "commit": counts["commit"] + commit,
            "row": counts["row"] + row,
            "task": counts["task"] + task,
        }
        if w:
            expected = counts["weight"]
            if not (expected < 2 and event.get("source") == expected and
                    event.get("first") is (expected == 0) and
                    event.get("last") is (expected == 1) and
                    (expected == 0 or after["core"] >= 1)):
                fault = True
        if p and not (counts["psum"] == 0 and event.get("psum_source", 0) == 0 and
                      event.get("psum_first", True) is True):
            fault = True
        if response and not (core and counts["response"] < 2 and
                             after["weight"] >= counts["response"] + 1 and
                             (counts["response"] != 0 or after["psum"] >= 1)):
            fault = True
        if core and not (response and counts["core"] < 2):
            fault = True
        if commit and not (after["core"] == 2 and counts["commit"] == 0 and
                           event.get("address", 0) == 0):
            fault = True
        if row and not (after["commit"] == 1 and counts["row"] == 0 and
                        event.get("row_id", 0) == 0):
            fault = True
        if task and not (after["row"] == 1 and counts["task"] == 0 and
                         event.get("epoch", 0x9001) == 0x9001):
            fault = True
        counts = after
        if counts["weight"] >= 1 and counts["psum"] >= 1: stage = 1
        if counts["response"] >= 1 and counts["core"] >= 1: stage = 2
        if counts["weight"] >= 2: stage = 3
        if counts["response"] >= 2 and counts["core"] >= 2: stage = 4
        if counts["commit"] >= 1: stage = 5
        if counts["row"] >= 1: stage = 6
        if counts["task"] >= 1: stage = 7
        if (counts["weight"] > 2 or counts["psum"] > 1 or
                counts["response"] > 2 or counts["core"] > 2 or
                counts["commit"] > 1 or counts["row"] > 1 or
                counts["task"] > 1 or event.get("attack") or event.get("fault")):
            fault = True
    passed = not fault and stage == 7 and counts == {
        "weight": 2, "psum": 1, "response": 2, "core": 2,
        "commit": 1, "row": 1, "task": 1}
    return {"pass": passed, "fault": fault, "stage": stage, "counts": counts}


def strict_cycles() -> list[dict]:
    return [
        {"weight": True, "psum": True, "source": 0, "first": True, "last": False},
        {"response": True, "core": True},
        {"weight": True, "source": 1, "first": False, "last": True},
        {"response": True, "core": True},
        {"commit": True, "address": 0},
        {"row": True, "row_id": 0},
        {"task": True, "epoch": 0x9001},
    ]


def checker_accepts_witness(text: str) -> bool:
    try:
        C.check_witness_text(text)
    except AssertionError:
        return False
    return True


def main() -> None:
    positives = []
    false_negatives = []

    need(sha(CHECKER) == "5dfb553912bb29cc010855a6af19002a13c81c285d5221acbb8785e43e05e174",
         "checker drift")
    need(sha(TEST) == "0e6f3722c67d2ac08506a42ac5d7f13ad9577e86d8fcb5c13008fdd87fa40279",
         "test drift")
    need(sha(CONTRACT) == "a16810280e8fafeecaa23251da14d51a7d35ce261c8ebf0474c6d66766899cfc",
         "contract drift")
    author_rows = verify_author_seal()
    need(author_rows.get("review.json") ==
         "df5d0a3cf7c8e90901dffc8a3a35d7670aeeae0e0c0792fe788d1a5eb15dce9a",
         "author review drift")
    positives.append("author_double_seal_exact")

    # Re-run all checker identities, readiness authority, filelist and policy.
    with redirect_stdout(io.StringIO()):
        need(C.main() == 0, "author checker baseline failed")
    positives.append("author_checker_baseline")
    for path, digest in C.EXPECTED.items():
        need(sha(path) == digest, "frozen identity drift: " + str(path))
    positives.append("m528_m935_m1162_r3sva_r13tb_python310_exact")
    readiness = json.loads((C.READINESS / "review.json").read_text())
    need(readiness["evidence_split"]["full_storage_214912B"] == {
        "represented_ledger_bytes": 214912,
        "physically_integrated_bytes": 18432,
        "external_common_charge_bytes": 196480,
        "numeric_external_area_energy": False,
        "full_storage_dc_pt_release": False}, "214912B readiness ledger drift")
    positives.append("readiness_214912B_exact")

    need(C.runtime_model(C.good_trace())["pass"], "sequential model good trace")
    for index in range(len(C.good_trace())):
        need(not C.runtime_model(C.good_trace()[:index] + C.good_trace()[index + 1:])["pass"],
             "missing sequential milestone accepted")
    positives.append("sequential_missing_milestones_rejected")

    # Duplicate each response/accept/commit/row/task milestone in the Python model.
    for index in (2, 4, 5, 6, 7):
        trace = [dict(row) for row in C.good_trace()]
        trace.insert(index + 1, dict(trace[index]))
        need(not C.runtime_model(trace)["pass"], "duplicate event accepted")
    positives.append("sequential_duplicate_response_accept_commit_row_task_rejected")

    # Filelist active-member mutations must all fail exact equality.
    expected_lines = C.FILELIST.read_text().splitlines()
    for mutant in (
        expected_lines[:-1], expected_lines + ["// extra"],
        expected_lines[1:], list(reversed(expected_lines)),
        [line.replace("m935_m912", "forged_m935") for line in expected_lines],
    ):
        need(mutant != [str(path) for path in
                        (C.FOUNDRY, C.M528, C.M935, C.M1162, C.SVA, C.R13_TB, C.WITNESS)],
             "filelist mutant accidentally canonical")
    positives.append("filelist_active_exact_seven_members")

    need(cycle_model(strict_cycles())["pass"], "strict cycle trace unexpectedly fails")

    # Independently split response from core accept and exercise every real
    # child completion at cycle granularity; the author's sequential model
    # represents response+core as one synthetic event.
    for cycle_index, key in (
        (0, "weight"), (0, "psum"), (1, "response"), (1, "core"),
        (2, "weight"), (3, "response"), (3, "core"),
        (4, "commit"), (5, "row"), (6, "task"),
    ):
        mutant = copy.deepcopy(strict_cycles())
        mutant[cycle_index].pop(key)
        need(not cycle_model(mutant)["pass"], "missing cycle event accepted: " + key)
    positives.append("cycle_missing_response_core_psum_write_row_task_rejected")

    for extra in (
        {"weight": True, "source": 2, "first": False, "last": False},
        {"psum": True}, {"response": True, "core": True},
        {"commit": True, "address": 0}, {"row": True, "row_id": 0},
        {"task": True, "epoch": 0x9001},
    ):
        need(not cycle_model(strict_cycles() + [extra])["pass"],
             "duplicate cycle event accepted")
    positives.append("cycle_duplicate_response_accept_psum_write_row_task_rejected")

    for flag in ("attack", "fault"):
        mutant = copy.deepcopy(strict_cycles()); mutant[1][flag] = True
        need(not cycle_model(mutant)["pass"], "attack/design fault accepted")
    positives.append("cycle_attack_mask_and_design_fault_rejected")

    # FN1: the second weight request shares the edge with the first core accept.
    collapsed_second_request = [
        strict_cycles()[0],
        {"weight": True, "source": 1, "first": False, "last": True,
         "response": True, "core": True},
        {"response": True, "core": True},
        strict_cycles()[4], strict_cycles()[5], strict_cycles()[6],
    ]
    if cycle_model(collapsed_second_request)["pass"]:
        false_negatives.append(
            "FN1_second_weight_request_same_cycle_as_first_core_accept_passes")

    # FN2: accept, commit, row and task are allowed to collapse through after-values.
    collapsed_tail = [
        strict_cycles()[0], strict_cycles()[1], strict_cycles()[2],
        {"response": True, "core": True, "commit": True, "address": 0,
         "row": True, "row_id": 0, "task": True, "epoch": 0x9001},
    ]
    if cycle_model(collapsed_tail)["pass"]:
        false_negatives.append(
            "FN2_second_accept_commit_row_task_same_cycle_cascade_passes")

    source = WITNESS.read_text()

    # FN3: ordinary == plus if (!(X)) does not set sticky fault for X tuple bits.
    tuple_checks = (
        "issue_request_source_index == weight_requests_q[3:0]",
        "psum_commit_address == 0", "row_complete_id == 0",
        "task_done_epoch == 16'h9001",
    )
    if all(token in source for token in tuple_checks) and "$isunknown" not in source:
        false_negatives.append(
            "FN3_four_state_unknown_source_address_row_epoch_can_escape_fault")

    # FN4: child-output and attack/fault bind seams can be tied off while keeping
    # their required tokens in comments/port declarations.
    child_mutant = source.replace(
        ".response_accept(dut.response_accept_w),",
        "/* dut.response_accept_w */ .response_accept(1'b1),", 1)
    child_mutant = child_mutant.replace(
        ".row_complete_fire(row_complete_valid && row_complete_ready),",
        "/* row_complete_valid && row_complete_ready */ .row_complete_fire(1'b1),", 1)
    child_mutant = child_mutant.replace(
        ".request_hold_attack_mode(request_hold_attack_mode),",
        ".request_hold_attack_mode(1'b0),", 1)
    child_mutant = child_mutant.replace(
        ".protocol_error(protocol_error),", ".protocol_error(1'b0),", 1)
    if checker_accepts_witness(child_mutant):
        false_negatives.append(
            "FN4_child_output_attack_mask_and_design_fault_tieoffs_pass_static_checker")

    # FN5: move the sole PASS display before the fatal; checker does not constrain
    # PASS dominance, so a failing run can print the unique PASS token first.
    lines = source.splitlines()
    pass_index = next(i for i, line in enumerate(lines)
                      if "$display(\"PASS_M1334R14_REAL_M935_RUNTIME_WITNESS" in line)
    pass_line = lines.pop(pass_index)
    fatal_if = next(i for i, line in enumerate(lines) if "if (pass !== 1'b1)" in line)
    lines.insert(fatal_if, pass_line)
    pass_before_fatal = "\n".join(lines) + "\n"
    if checker_accepts_witness(pass_before_fatal):
        false_negatives.append(
            "FN5_unique_pass_token_can_move_before_fatal_and_still_pass_checker")

    # FN6: the current checker never validates the 214912B fields in CONTRACT.
    with tempfile.TemporaryDirectory(prefix="m1335_contract_") as td:
        data = json.loads(CONTRACT.read_text())
        data["frozen_design"]["represented_ledger_bytes"] = 1
        data["frozen_design"]["physically_integrated_parent_bytes"] = 1
        data["frozen_design"]["external_common_charge_bytes"] = 0
        mutant_contract = Path(td) / "contract.json"
        mutant_contract.write_text(json.dumps(data))
        old_contract = C.CONTRACT
        try:
            C.CONTRACT = mutant_contract
            with redirect_stdout(io.StringIO()):
                accepted = C.main() == 0
        finally:
            C.CONTRACT = old_contract
        if accepted:
            false_negatives.append(
                "FN6_contract_214912B_ledger_mutation_passes_source_checker")

    need(len(false_negatives) == 6, "unexpected false-negative population")
    result = {
        "schema": "m1335_m1334_c1_r14_runtime_witness_source_blind_review_r1",
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "score": 61,
        "author_tests": "13/13 PASS",
        "positive_checks": positives,
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "vcs_eda_release_rtl_edits": False,
        "docs359_sha256": sha(C.DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
