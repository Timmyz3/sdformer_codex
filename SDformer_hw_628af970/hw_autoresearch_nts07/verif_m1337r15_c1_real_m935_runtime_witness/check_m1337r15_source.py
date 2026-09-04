#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed source gate for the additive M1337/R15 C1 witness."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
HW = HERE.parent
WITNESS = HERE / "m1337r15_m935_runtime_witness.sv"
FILELIST = HERE / "m1337r15_unit_delay_filelist.f"
CHECKER = Path(__file__).resolve()
TEST = HERE / "test_m1337r15_source.py"
CONTRACT = HW / "contracts/m1337_c1_r15_real_m935_runtime_witness_source_contract_r1_20260831.json"
R14_FAILED = HW / "reviews/m1335_m1334_c1_r14_runtime_witness_source_blind_review_r1_20260831"
M528 = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1162 = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
R13_TB = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

FROZEN_PATH_SHA = {
    M528: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1162: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    R13_TB: "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
R14_FAILED_SEAL = {
    "review": "31abaa97d1a93b50d8e90ebdd90f0580d31d9657df658561f024b067a1993ea4",
    "manifest": "b918e1c8090e827b7dfd16aa3f2d15dafe62aa833a42fc8c915150b336d93948",
    "outer_file": "05c76b268bfb1bce47eeb0b6137ddab3fd2fbe68f2dc9eaf239fe96f98f11ff1",
}
EXPECTED_FROZEN_DESIGN = {
    "m528_sha256": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "m935_sha256": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "m1162_sha256": "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    "r3_sva_sha256": "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    "r13_tb_sha256": "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    "foundry_unit_delay_model_sha256": "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    "represented_ledger_bytes": 214912,
    "physically_integrated_parent_bytes": 18432,
    "external_common_charge_bytes": 196480,
    "full_214912B_physically_integrated": False,
}

EXPECTED_BIND = {
    "clk_core": "clk_core",
    "reset_n": "reset_n",
    "issue_request_valid": "issue_request_valid",
    "issue_request_first": "issue_request_first",
    "issue_request_last": "issue_request_last",
    "issue_request_source_valid": "issue_request_source_valid",
    "issue_request_source_index": "issue_request_source_index",
    "weight_request_fire": "weight_req_valid&&weight_req_ready",
    "psum_request_fire": "psum_req_valid&&psum_req_ready",
    "response_accept": "dut.response_accept_w",
    "core_accept": "dut.core_issue_data_valid&&dut.core_issue_data_ready",
    "psum_commit_fire": "psum_write_valid&&psum_write_ready",
    "psum_commit_address": "psum_write_address",
    "row_complete_fire": "row_complete_valid&&row_complete_ready",
    "row_complete_id": "row_complete_id",
    "task_done_fire": "task_done_valid",
    "task_done_epoch": "task_done_epoch",
    "request_hold_attack_mode": "request_hold_attack_mode",
    "weight_service_attack_mode": "weight_service_attack_mode",
    "psum_service_attack_mode": "psum_service_attack_mode",
    "protocol_error": "protocol_error",
    "boundary_fault": "dut.boundary_fault_q",
    "core_fault": "dut.core_protocol_error",
    "m935_fault": "dut.u_frozen_m935.fault_q",
    "weight_service_fault": "weight_service_fault",
    "psum_service_fault": "psum_service_fault",
    "design_issue_accepts": "count_issue_accepts",
    "design_psum_commits": "count_psum_commits",
    "design_row_completions": "count_row_completions",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def verify_dir(root: Path, expected: dict[str, str]) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    for base, dirs, files in os.walk(root, followlinks=False):
        parent = Path(base)
        for name in dirs + files:
            require(not (parent / name).is_symlink(), "sealed tree symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    require(sha(manifest) == expected["manifest"], "manifest identity drift")
    require(sha(outer) == expected["outer_file"], "outer identity drift")
    require(outer.read_text().split() == [expected["manifest"], "SHA256SUMS"],
            "outer content drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(name not in rows and not Path(name).is_absolute()
                and ".." not in Path(name).parts, "unsafe manifest member")
        regular(root / name)
        require(sha(root / name) == digest, "sealed member drift: " + name)
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed recursive population drift")
    require(rows.get("review.json") == expected["review"], "review member drift")
    return rows


def strip_sv_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def norm_expr(text: str) -> str:
    return re.sub(r"\s+", "", text)


def parse_active_bind(text: str) -> dict[str, str]:
    code = strip_sv_comments(text)
    match = re.search(
        r"\bbind\s+tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13\s+"
        r"m1337r15_m935_runtime_witness\s+u_m1337r15_runtime_witness\s*"
        r"\((.*?)\)\s*;", code, flags=re.S)
    require(match is not None, "exact active R15 bind missing")
    rows = re.findall(r"\.(\w+)\s*\(([^()]*)\)", match.group(1), flags=re.S)
    require(len(rows) == len(EXPECTED_BIND), "bind connection cardinality drift")
    connections = {name: norm_expr(expr) for name, expr in rows}
    require(len(connections) == len(rows), "duplicate bind port")
    require(connections == EXPECTED_BIND, "active bind expression drift")
    require(all(not re.search(r"(?:^|[^\w])(?:1'b[01xXzZ]|\d+)(?:$|[^\w])", expr)
                for expr in connections.values()), "constant-tied active bind")
    return connections


def check_final_oracle(code: str) -> None:
    require(code.count("PASS_M1337R15_REAL_M935_RUNTIME_WITNESS") == 1,
            "PASS token cardinality")
    require(code.count("$fatal(") == 1, "fatal cardinality")
    start = code.find("final begin : witness_final_oracle")
    stop = code.find("end\nendmodule", start)
    require(start >= 0 and stop > start, "final oracle missing")
    oracle = code[start:stop]
    operand = oracle.find("M1337R15_WITNESS_OPERANDS")
    branch = re.search(
        r"if\s*\(\s*pass\s*===\s*1'b1\s*\)\s*begin\s*"
        r"\$display\(\s*\"PASS_M1337R15_REAL_M935_RUNTIME_WITNESS[^\"]*\"\s*\)\s*;\s*"
        r"end\s*else\s*begin\s*\$fatal\(", oracle, flags=re.S)
    require(operand >= 0 and branch is not None and operand < branch.start(),
            "PASS is not terminal-success dominated after operand dump")
    require("$finish" not in code, "early finish authority forbidden")


def check_witness_text(text: str) -> None:
    code = strip_sv_comments(text)
    require(code.count("module m1337r15_m935_runtime_witness") == 1,
            "witness module cardinality")
    require(len(re.findall(r"\balways_ff\b", code)) == 1
            and len(re.findall(r"\bfinal\s+begin\b", code)) == 1
            and not re.search(r"\binitial\s+begin\b", code), "runtime structure drift")
    require(not re.search(r"\b(?:force|release)\b", code), "force/release seam")
    require("_after" not in code, "same-edge cumulative frontier forbidden")
    require("case (stage_q)" in code, "registered stage case missing")
    for stage in ("W_RESET", "W_FIRST_REQUEST", "W_FIRST_ACCEPT",
                  "W_SECOND_REQUEST", "W_SECOND_ACCEPT", "W_PSUM_COMMIT",
                  "W_ROW_DONE", "W_TASK_DONE"):
        require(code.count(stage + ": begin") == 1,
                "registered stage branch missing: " + stage)
    for identity in ("issue_request_source_index", "psum_commit_address",
                     "row_complete_id", "task_done_epoch"):
        require(re.search(r"\$isunknown\s*\([^;]*\b" + identity + r"\b[^;]*\)",
                          code, flags=re.S) is not None,
                "explicit unknown rejection missing: " + identity)
    for exact in ("issue_request_source_index === 4'd0",
                  "issue_request_source_index === 4'd1",
                  "psum_commit_address === 6'd0",
                  "row_complete_id === 6'd0",
                  "task_done_epoch === 16'h9001"):
        require(exact in code, "case equality missing: " + exact)
    parse_active_bind(text)
    check_final_oracle(code)


CONTROL_KEYS = ("weight", "psum", "response", "core", "commit", "row", "task")


def cycle(**kwargs: Any) -> dict[str, Any]:
    row = {key: False for key in CONTROL_KEYS}
    row.update({"attack": False, "fault": False})
    row.update(kwargs)
    return row


def good_trace() -> list[dict[str, Any]]:
    return [
        cycle(weight=True, psum=True, request_valid=True, source_valid=True,
              source=0, first=True, last=False),
        cycle(response=True, core=True),
        cycle(weight=True, request_valid=True, source_valid=True,
              source=1, first=False, last=True),
        cycle(response=True, core=True),
        cycle(commit=True, address=0),
        cycle(row=True, row_id=0),
        cycle(task=True, epoch=0x9001),
    ]


def runtime_model(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stage = 0
    fault = False
    counts = {key: 0 for key in ("weight", "psum", "response", "core",
                                  "commit", "row", "task")}
    for raw in events:
        event = cycle(**raw)
        controls = [event[key] for key in CONTROL_KEYS]
        if any(value is None for value in controls + [event["attack"], event["fault"]]):
            fault = True
            continue
        if event["attack"] or event["fault"]:
            fault = True
        active = any(bool(event[key]) for key in CONTROL_KEYS)
        if not active:
            continue

        if (event["weight"] or event["psum"]) and any(
                event.get(key) is None for key in
                ("request_valid", "source_valid", "source", "first", "last")):
            fault = True
            continue
        if event["commit"] and event.get("address") is None:
            fault = True
            continue
        if event["row"] and event.get("row_id") is None:
            fault = True
            continue
        if event["task"] and event.get("epoch") is None:
            fault = True
            continue

        exact = False
        if stage == 0:
            exact = (event["weight"] and event["psum"]
                     and not any(event[key] for key in
                                 ("response", "core", "commit", "row", "task"))
                     and event.get("request_valid") is True
                     and event.get("source_valid") is True
                     and event.get("source") == 0
                     and event.get("first") is True
                     and event.get("last") is False)
            if exact:
                counts["weight"] += 1; counts["psum"] += 1; stage = 1
        elif stage == 1:
            exact = (event["response"] and event["core"]
                     and not any(event[key] for key in
                                 ("weight", "psum", "commit", "row", "task")))
            if exact:
                counts["response"] += 1; counts["core"] += 1; stage = 2
        elif stage == 2:
            exact = (event["weight"] and not any(event[key] for key in
                     ("psum", "response", "core", "commit", "row", "task"))
                     and event.get("request_valid") is True
                     and event.get("source_valid") is True
                     and event.get("source") == 1
                     and event.get("first") is False
                     and event.get("last") is True)
            if exact:
                counts["weight"] += 1; stage = 3
        elif stage == 3:
            exact = (event["response"] and event["core"]
                     and not any(event[key] for key in
                                 ("weight", "psum", "commit", "row", "task")))
            if exact:
                counts["response"] += 1; counts["core"] += 1; stage = 4
        elif stage == 4:
            exact = (event["commit"] and not any(event[key] for key in
                     ("weight", "psum", "response", "core", "row", "task"))
                     and event.get("address") == 0)
            if exact:
                counts["commit"] += 1; stage = 5
        elif stage == 5:
            exact = (event["row"] and not any(event[key] for key in
                     ("weight", "psum", "response", "core", "commit", "task"))
                     and event.get("row_id") == 0)
            if exact:
                counts["row"] += 1; stage = 6
        elif stage == 6:
            exact = (event["task"] and not any(event[key] for key in
                     ("weight", "psum", "response", "core", "commit", "row"))
                     and event.get("epoch") == 0x9001)
            if exact:
                counts["task"] += 1; stage = 7
        else:
            exact = False
        if not exact:
            fault = True
    expected = {"weight": 2, "psum": 1, "response": 2, "core": 2,
                "commit": 1, "row": 1, "task": 1}
    return {"pass": not fault and stage == 7 and counts == expected,
            "fault": fault, "stage": stage, "counts": counts}


def check_contract_dict(contract: dict[str, Any]) -> None:
    require(contract.get("schema") ==
            "m1337_c1_r15_real_m935_runtime_witness_source_contract_r1_v1",
            "contract schema drift")
    require(contract.get("status") ==
            "SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA",
            "contract status drift")
    require(contract.get("r14_failed_authority") == {
        "path": "hw_autoresearch_nts07/reviews/m1335_m1334_c1_r14_runtime_witness_source_blind_review_r1_20260831",
        "review_sha256": R14_FAILED_SEAL["review"],
        "manifest_sha256": R14_FAILED_SEAL["manifest"],
        "outer_file_sha256": R14_FAILED_SEAL["outer_file"],
        "status": "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "false_negative_count": 6,
    }, "R14 failed authority drift")
    frozen = contract.get("frozen_design")
    require(frozen == EXPECTED_FROZEN_DESIGN, "complete frozen_design drift")
    for key in ("represented_ledger_bytes", "physically_integrated_parent_bytes",
                "external_common_charge_bytes"):
        require(type(frozen[key]) is int, "ledger value is not parsed integer: " + key)
    require(frozen["represented_ledger_bytes"] == 214912
            and frozen["physically_integrated_parent_bytes"] == 18432
            and frozen["external_common_charge_bytes"] == 196480
            and frozen["represented_ledger_bytes"] ==
            frozen["physically_integrated_parent_bytes"]
            + frozen["external_common_charge_bytes"], "214912-byte ledger equation fails")
    source = contract.get("new_source", {})
    expected_source = {
        "witness_path": WITNESS.relative_to(HW.parent).as_posix(),
        "witness_sha256": sha(WITNESS),
        "filelist_path": FILELIST.relative_to(HW.parent).as_posix(),
        "filelist_sha256": sha(FILELIST),
        "checker_path": CHECKER.relative_to(HW.parent).as_posix(),
        "checker_sha256": sha(CHECKER),
        "test_path": TEST.relative_to(HW.parent).as_posix(),
        "test_sha256": sha(TEST),
        "python_path": str(PYTHON),
        "python_sha256": sha(PYTHON),
    }
    require(source == expected_source, "source identity dictionary drift")
    require(contract.get("launch_authorized") is False
            and contract.get("release_present") is False, "launch/release boundary drift")
    boundary = contract.get("claim_boundary", {})
    require(boundary.get("source_only") is True and all(boundary.get(key) is False
            for key in ("functional_vcs", "timing_verified", "cycles_measured",
                        "speedup", "ppa", "power", "energy", "system_speedup",
                        "headline")), "claim boundary drift")


def main() -> int:
    for path, digest in FROZEN_PATH_SHA.items():
        regular(path); require(sha(path) == digest, "frozen identity drift: " + str(path))
    verify_dir(R14_FAILED, R14_FAILED_SEAL)
    failed = json.loads((R14_FAILED / "review.json").read_text())
    require(failed["status"] == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and failed["false_negative_count"] == 6
            and failed["authorization"]["additive_witness_successor"] is True,
            "R14 failure verdict drift")
    for path in (WITNESS, FILELIST, CHECKER, TEST, CONTRACT):
        regular(path)
    check_witness_text(WITNESS.read_text())
    expected_filelist = [str(path) for path in
        (FOUNDRY, M528, M935, M1162, SVA, R13_TB, WITNESS)]
    require(FILELIST.read_text().splitlines() == expected_filelist,
            "exact seven-member filelist drift")
    require(runtime_model(good_trace())["pass"], "good registered-stage trace fails")
    check_contract_dict(json.loads(CONTRACT.read_text()))
    require(not list(HW.glob("contracts/m1337*c1*r15*release*.json")),
            "R15 release unexpectedly exists")
    print(json.dumps({
        "status": "PASS_M1337R15_SOURCE_ONLY__NO_VCS_NO_EDA",
        "directed_registered_stage_model": True,
        "active_bind_exact": True,
        "ledger_equation": "214912=18432+196480",
        "launch_authorized": False,
        "vcs_runs": 0,
        "eda_runs": 0,
        "docs359_sha256": sha(DOCS359),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
