#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static/source-only gate for the M1334/R14 runtime-witness package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable


HERE = Path(__file__).resolve().parent
HW = HERE.parent
ROOT = HW.parent
WITNESS = HERE / "m1334r14_m935_runtime_witness.sv"
FILELIST = HERE / "m1334r14_unit_delay_filelist.f"
TEST = HERE / "test_m1334r14_source.py"
CONTRACT = HW / "contracts/m1334_c1_r14_real_m935_runtime_witness_source_contract_r1_20260831.json"
READINESS_SOURCE = HW / "system_simulator/scripts/audit_m1333_c1_closure_release_readiness.py"
READINESS = HW / "reviews/m1333_c1_closure_release_readiness_readonly_audit_r1_20260831"
R13_TB = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
R13_CONTRACT = HW / "contracts/m1270_c1_r13_real_m935_integrated_protocol_source_contract_r1_20260830.json"
R13_REVIEW = HW / "reviews/m1273_m1272_m1270_c1_r13_checker_final_independent_hammer_r1_20260830"
M528 = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M1162 = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    WITNESS: "9707cb72123ebf9b941497f1a7815a2d57cc62f4172957682242586ec7739295",
    FILELIST: "3ea94a7c2e5402f36f924b9029c6aecada041edd2a705d2ebd21c5523209dfc6",
    READINESS_SOURCE: "5f2b0df6fbf8e5e61fd38d616a4324b12d23df32d6e46a039666983fbe1de97b",
    R13_TB: "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    R13_CONTRACT: "f17a02226b4d8a391d6cbb5830e16f7e0716b7a9f1e342457add79e0438e15ee",
    M528: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    M1162: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
READINESS_SEAL = {
    "manifest": "6a0c1442d8d9e61279f108be1f1f177483a66b2049925b76d406ee467bce4138",
    "outer_file": "92878e75b557a87e5d43fd3bbaa41363181393647f6c76261f4278b6098c13d7",
    "review": "d61a4acfc6e81c414513382934b69974849444ec7c8d7942d00dbff84899a877",
}
R13_REVIEW_SEAL = {
    "manifest": "f3dd833c356272893ba9c52c8993f7df66793bed91a75992be063c32de584f71",
    "outer_file": "848c7eb0b1e266fda8eaaaef18272c6093641ed7687cadf81d3cdf5a9e9bf2d4",
    "review": "caf61dd7de32f546e0c0e681b020c8717e8a4aca536ab17df62854483dc4749a",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def regular(path: Path) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "not regular: " + str(path))


def verify_dir(root: Path, expected: dict[str, str]) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    for base, dirs, files in os.walk(root, followlinks=False):
        parent = Path(base)
        for name in dirs + files:
            require(not (parent / name).is_symlink(), "sealed tree symlink")
    manifest = root / "SHA256SUMS"; outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    require(sha(manifest) == expected["manifest"] and
            sha(outer) == expected["outer_file"], "seal identity drift")
    require(outer.read_text().split() == [expected["manifest"], "SHA256SUMS"],
            "outer content drift")
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(name not in rows and not Path(name).is_absolute() and
                ".." not in Path(name).parts, "manifest member unsafe")
        regular(root / name); require(sha(root / name) == digest, "member drift")
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "recursive population drift")
    require(rows.get("review.json") == expected["review"], "review member drift")
    return rows


def runtime_model(events: Iterable[dict]) -> dict:
    counts = {key: 0 for key in ("weight", "psum", "response", "core",
                                  "commit", "row", "task")}
    fault = False
    stage = 0
    for event in events:
        if event.get("attack") or event.get("fault"):
            fault = True
        kind = event["kind"]
        if kind == "weight":
            expected = counts["weight"]
            if not (expected < 2 and event.get("source") == expected and
                    event.get("first") is (expected == 0) and
                    event.get("last") is (expected == 1) and
                    (expected == 0 or counts["core"] >= 1)):
                fault = True
            counts["weight"] += 1
        elif kind == "psum":
            if not (counts["psum"] == 0 and event.get("first") is True and
                    event.get("source") == 0):
                fault = True
            counts["psum"] += 1
        elif kind == "accept":
            if not (counts["response"] < 2 and counts["weight"] >= counts["response"] + 1 and
                    (counts["response"] != 0 or counts["psum"] >= 1)):
                fault = True
            counts["response"] += 1; counts["core"] += 1
        elif kind == "commit":
            if not (counts["core"] == 2 and counts["commit"] == 0 and
                    event.get("address") == 0):
                fault = True
            counts["commit"] += 1
        elif kind == "row":
            if not (counts["commit"] == 1 and counts["row"] == 0 and
                    event.get("row_id") == 0):
                fault = True
            counts["row"] += 1
        elif kind == "task":
            if not (counts["row"] == 1 and counts["task"] == 0 and
                    event.get("epoch") == 0x9001):
                fault = True
            counts["task"] += 1
        else:
            fault = True
        if counts["weight"] >= 1 and counts["psum"] >= 1: stage = max(stage, 1)
        if counts["core"] >= 1: stage = max(stage, 2)
        if counts["weight"] >= 2: stage = max(stage, 3)
        if counts["core"] >= 2: stage = max(stage, 4)
        if counts["commit"] >= 1: stage = max(stage, 5)
        if counts["row"] >= 1: stage = max(stage, 6)
        if counts["task"] >= 1: stage = max(stage, 7)
        if (counts["weight"] > 2 or counts["psum"] > 1 or counts["core"] > 2 or
                counts["commit"] > 1 or counts["row"] > 1 or counts["task"] > 1):
            fault = True
    passed = not fault and stage == 7 and counts == {
        "weight": 2, "psum": 1, "response": 2, "core": 2,
        "commit": 1, "row": 1, "task": 1}
    return {"pass": passed, "fault": fault, "stage": stage, "counts": counts}


def good_trace() -> list[dict]:
    return [
        {"kind": "weight", "source": 0, "first": True, "last": False},
        {"kind": "psum", "source": 0, "first": True},
        {"kind": "accept"},
        {"kind": "weight", "source": 1, "first": False, "last": True},
        {"kind": "accept"},
        {"kind": "commit", "address": 0},
        {"kind": "row", "row_id": 0},
        {"kind": "task", "epoch": 0x9001},
    ]


def check_witness_text(text: str) -> None:
    require(text.count("module m1334r14_m935_runtime_witness") == 1,
            "witness module cardinality")
    require(text.count("bind tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13") == 1,
            "real R13 bind cardinality")
    require(len(re.findall(r"\balways_ff\b", text)) == 1 and
            len(re.findall(r"\bfinal\s+begin\b", text)) == 1 and
            not re.search(r"\binitial\s+begin\b", text), "runtime structure drift")
    require(not re.search(r"\b(?:force|release)\b", re.sub(r"//.*", "", text)),
            "force/release seam")
    require(text.count("PASS_M1334R14_REAL_M935_RUNTIME_WITNESS") == 1 and
            text.count("$fatal(") == 1 and
            text.index("M1334R14_WITNESS_OPERANDS") < text.index("$fatal("),
            "final operand/fatal/pass structure drift")
    for token in ("dut.response_accept_w", "dut.core_issue_data_valid && dut.core_issue_data_ready",
                  "psum_write_valid && psum_write_ready",
                  "row_complete_valid && row_complete_ready", "task_done_valid",
                  "design_issue_accepts", "design_psum_commits",
                  "design_row_completions", "attack_mask_active", "any_design_fault"):
        require(token in text, "missing runtime binding: " + token)
    for stage in ("W_FIRST_REQUEST", "W_FIRST_ACCEPT", "W_SECOND_REQUEST",
                  "W_SECOND_ACCEPT", "W_PSUM_COMMIT", "W_ROW_DONE", "W_TASK_DONE"):
        require(text.count(stage) >= 2, "missing monotonic stage " + stage)


def main() -> int:
    for path, digest in EXPECTED.items():
        regular(path); require(sha(path) == digest, "identity drift: " + str(path))
    verify_dir(READINESS, READINESS_SEAL)
    verify_dir(R13_REVIEW, R13_REVIEW_SEAL)
    readiness = json.loads((READINESS / "review.json").read_text())
    require(readiness["status"] == "NO_GO_DIRECT_C1_VCS_DC_PT__NO_UNCONSUMED_ADMITTED_RELEASE" and
            readiness["unique_next_source"]["name"] ==
            "additive R14 real-M935 runtime-witness wrapper VCS source package" and
            readiness["unique_next_source"]["does_not_authorize_execution"] is True,
            "M1333 readiness authority drift")
    r13 = json.loads((R13_REVIEW / "review.json").read_text())
    require(r13["status"] == "SOURCE_NO_GO__NO_RELEASE_NO_VCS__CHECKER_EXPANSION_STOPPED" and
            r13["vcs_authorized"] is False, "R13 NO-GO boundary drift")
    check_witness_text(WITNESS.read_text())
    expected_filelist = [str(path) for path in
        (FOUNDRY, M528, M935, M1162, SVA, R13_TB, WITNESS)]
    require(FILELIST.read_text().splitlines() == expected_filelist, "filelist drift")
    require(runtime_model(good_trace())["pass"], "good model trace fails")
    contract = json.loads(CONTRACT.read_text())
    require(contract["status"] ==
            "SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_EDA" and
            contract["launch_authorized"] is False and
            all(contract["claim_boundary"][key] is False for key in
                ("functional_vcs", "timing_verified", "cycles_measured", "speedup",
                 "ppa", "power", "energy", "system_speedup", "headline")),
            "contract boundary drift")
    releases = list(HW.glob("contracts/m1334*c1*r14*release*.json"))
    require(not releases, "R14 release unexpectedly exists")
    print(json.dumps({"status": "PASS_M1334R14_SOURCE_ONLY__NO_VCS_NO_EDA",
                      "runtime_model_good": True, "launch_authorized": False,
                      "vcs_runs": 0, "eda_runs": 0,
                      "docs359_sha256": sha(DOCS359)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
