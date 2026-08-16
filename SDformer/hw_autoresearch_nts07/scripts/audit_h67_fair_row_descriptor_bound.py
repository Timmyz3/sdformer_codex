#!/usr/bin/env python3
"""Audit row-level RTL descriptor counts against independently recomputed equality."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_h67_empty_active_equal import load_rows, motionxor_q7


ROW_RE = re.compile(
    r"^FAIR_ROW row=(?P<row>\d+) active=(?P<active>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) "
    r"fslots=(?P<fslots>\d+) rslots=(?P<rslots>\d+) equal=(?P<equal>\d+)$",
    re.MULTILINE,
)
SUM_RE = re.compile(
    r"^FAIR_SUM rows=(?P<rows>\d+) skip=(?P<skip>\d+) "
    r"fixed=(?P<fixed>\d+) rqtb=(?P<rqtb>\d+) shared=(?P<shared>\d+) "
    r"fpairs=(?P<fpairs>\d+) fslots=(?P<fslots>\d+) fequal=(?P<fequal>\d+) "
    r"rpairs=(?P<rpairs>\d+) rslots=(?P<rslots>\d+) requal=(?P<requal>\d+)$",
    re.MULTILINE,
)

LOCKED = {
    "rows": 138,
    "fixed": 112589,
    "rqtb": 94891,
    "fpairs": 31050,
    "fslots": 62100,
    "fequal": 28001,
    "rpairs": 31050,
    "rslots": 34099,
    "requal": 28001,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(vectors_path: Path, log_path: Path) -> dict[str, Any]:
    rows = load_rows(vectors_path)
    text = log_path.read_text(encoding="utf-8")
    if "ERROR:" in text or "FATAL:" in text:
        raise ValueError("fair log contains ERROR/FATAL")
    matches = list(ROW_RE.finditer(text))
    if len(rows) != 138 or len(matches) != len(rows):
        raise ValueError(f"row count vectors={len(rows)} log={len(matches)}")
    if [int(match["row"]) for match in matches] != list(range(138)):
        raise ValueError("FAIR_ROW IDs must be unique, complete, and ordered")
    sum_match = SUM_RE.search(text)
    if sum_match is None:
        raise ValueError("missing FAIR_SUM")
    parsed_sum = {key: int(sum_match[key]) for key in LOCKED}
    if parsed_sum != LOCKED:
        raise ValueError(f"FAIR_SUM drift: {parsed_sum}")
    if "PASS tb_h67_laws_fair_lfsr_threeway_2s" not in text:
        raise ValueError("missing terminal PASS")

    mismatches: list[dict[str, int]] = []
    row_receipts: list[dict[str, int]] = []
    for row_index, ((qs, ks), match) in enumerate(zip(rows, matches, strict=True)):
        equal_model = 0
        for pair in range(225):
            score0 = motionxor_q7(qs[pair], ks[pair], ks[pair + 225])
            score1 = motionxor_q7(qs[pair + 225], ks[pair + 225], ks[pair])
            equal_model += int(score0 == score1)
        equal_rtl = int(match["equal"])
        fixed_slots = int(match["fslots"])
        rqtb_slots = int(match["rslots"])
        expected_slots = 450 - equal_model
        receipt = {
            "row": row_index,
            "equal_model": equal_model,
            "equal_rtl": equal_rtl,
            "fixed_slots_rtl": fixed_slots,
            "rqtb_slots_rtl": rqtb_slots,
            "d_min": expected_slots,
        }
        row_receipts.append(receipt)
        if (
            equal_rtl != equal_model
            or fixed_slots != 450
            or rqtb_slots != expected_slots
            or rqtb_slots + equal_rtl != fixed_slots
        ):
            mismatches.append(receipt)

    return {
        "schema": "h67_fair_row_descriptor_bound_audit_v1",
        "status": "PASS" if not mismatches else "FAIL",
        "evidence": "[rtl]+[independent integer score replay]",
        "scope": (
            "ep35 sample0/window0 all12, 138 T450 head-row under frozen fair LFSR; "
            "not multisample RTL"
        ),
        "contract": "for every row: fixed_slots=2P=450 and rqtb_slots=D_min=2P-E",
        "rows": len(rows),
        "row_contract_pass": len(rows) - len(mismatches),
        "mismatches": mismatches,
        "locked_sum": parsed_sum,
        "row_receipt_sha256": hashlib.sha256(
            json.dumps(row_receipts, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
        "inputs": {
            "vectors": str(vectors_path.resolve()),
            "vectors_sha256": sha256_file(vectors_path),
            "fair_log": str(log_path.resolve()),
            "fair_log_sha256": sha256_file(log_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--fair-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.vectors, args.fair_log)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    (args.output_dir / "report.md").write_text(
        "# Motion 公平 RTL 逐行 descriptor 下界审计\n\n"
        f"- 裁决：`{report['status']}`；\n"
        f"- 范围：{report['scope']}；\n"
        f"- 逐行：{report['row_contract_pass']}/{report['rows']} 满足 "
        "`rslots=450-equal`，equal 同独立整数 score 重算；\n"
        f"- 汇总：Fixed/RQTB `{report['locked_sum']['fixed']}/"
        f"{report['locked_sum']['rqtb']}`，slot "
        f"`{report['locked_sum']['fslots']}/{report['locked_sum']['rslots']}`，"
        f"equal `{report['locked_sum']['requal']}`。\n\n"
        "这是 138 行实际 RTL descriptor 账本，不是多样本证据；不改 `docs/359`。\n",
        encoding="utf-8",
    )
    print(
        f"{report['status']} H67 fair row descriptor bound "
        f"rows={report['row_contract_pass']}/{report['rows']}"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
