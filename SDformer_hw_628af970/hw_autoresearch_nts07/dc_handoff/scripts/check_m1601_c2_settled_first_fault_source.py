#!/usr/bin/env python3
"""Fail-closed source check for the M1601 one-timeprecision settle repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat


HERE = Path(__file__).resolve()
HW = HERE.parents[2]
OLD_TB = HW / "dc_handoff/tb/tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv"
NEW_TB = HW / "dc_handoff/tb/tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv"
OLD_FL = HW / "dc_handoff/filelists/date_m1578_c2_rtl_vs_mapped_k8_case0_first_fault_source.f"
NEW_FL = HW / "dc_handoff/filelists/date_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault_source.f"
CONTRACT = HW / "contracts/m1601_c2_settled_first_fault_source_contract_r1_20260901.json"
M1594 = HW / "reviews/m1594_m1593_c2_first_fault_independent_cone_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PINS = {
    OLD_TB: "4a2ef4c40037274aadd936db8dbe38258aa39fa14a7e0322741f92acd958c435",
    OLD_FL: "09166d29aedc0a03266f9726ec006ac96efdd396c5290edb423ae303ad2548f1",
    M1594 / "review.json": "97370ae3eeae00ad79e3647b6ec34df3e114c88b807d773a69b250b0cfac324e",
    M1594 / "SHA256SUMS": "b629c1eef0489e96840e2444f2ef252ee2d261ccc47159de17fed9180dd8ae53",
    M1594 / "SHA256SUMS.seal.sha256": "1785184a2a61bba0f43c6727f93225919adc7bda2f728d646ff3e6f3b62951f5",
    NEW_TB: "3e8a9254fd9104aeeb4d3f05077a9f2b8ae33a9617d3236447108a5b666ba8e4",
    NEW_FL: "b6e384a3b7de9541a66af0302722c9ae9ca12b50e5e57a1ac764bf1576a39a53",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), "nonregular: " + str(path))
    require(sha256(path) == expected, "SHA drift: " + str(path))


def strict_json(path: Path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def normalized_new_tb(text: str) -> str:
    text = text.replace("tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault",
                        "tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault")
    text = text.replace("M1601_TRACE", "M1578_TRACE")
    text = text.replace("M1601_FIRST_STOP", "M1578_FIRST_STOP")
    text = text.replace("M1601 diagnostic watchdog", "M1578 diagnostic watchdog")
    text = text.replace("M1601 absolute watchdog", "M1578 absolute watchdog")
    settle = re.compile(
        r"\n            // Gate-level sequential cells and their zero-delay combinational\n"
        r"            // fanout update in later event regions than this posedge callback\.\n"
        r"            // One timeprecision step preserves the cycle while removing the\n"
        r"            // active-region race diagnosed by the sealed M1594 review\.\n"
        r"            #1ps;\n")
    text, count = settle.subn("\n", text)
    require(count == 1, "exactly one authorized settle block required")
    return text


def check():
    for path, digest in PINS.items():
        regular(path, digest)
    review = strict_json(M1594 / "review.json")
    require(review["decision"]["minimum_repair"] ==
            "add exactly one 1ps/one-timeprecision post-posedge settle before trace and all stop decisions",
            "M1594 repair authority drift")
    old_text = OLD_TB.read_text(encoding="utf-8")
    new_text = NEW_TB.read_text(encoding="utf-8")
    require(normalized_new_tb(new_text) == old_text, "TB differs beyond naming and settle")
    require(new_text.count("#1ps;") == 1, "settle count")
    delay = new_text.index("#1ps;")
    require(new_text.rfind("cycle_ordinal = cycle_ordinal + 1;", 0, delay) >= 0 and
            delay < new_text.index("trace_edge();", delay) and
            delay < new_text.index("if (difference_now", delay),
            "settle placement")
    old_rows = OLD_FL.read_text(encoding="utf-8").splitlines()
    new_rows = NEW_FL.read_text(encoding="utf-8").splitlines()
    require(old_rows[:-1] == new_rows[:-1] and len(old_rows) == len(new_rows) == 16,
            "filelist prefix drift")
    require(old_rows[-1].endswith("tb_m1578_c2_rtl_vs_mapped_k8_case0_first_fault.sv") and
            new_rows[-1].endswith("tb_m1601_c2_rtl_vs_mapped_k8_case0_settled_first_fault.sv"),
            "filelist TB-only delta")
    contract = strict_json(CONTRACT)
    require(contract["schema"] == "m1601_c2_settled_first_fault_source_contract_r1_v1" and
            contract["future_execution"]["authorized_now"] is False and
            contract["claim_boundary"]["paper_citable"] is False,
            "contract escalation")
    return {"schema": "m1601_c2_settled_first_fault_source_check_r1_v1",
            "status": "PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED",
            "filelist_entries": 16, "settle_count": 1,
            "vcs_compiles": 0, "simv_runs": 0, "claim": False}


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True, allow_nan=False))
