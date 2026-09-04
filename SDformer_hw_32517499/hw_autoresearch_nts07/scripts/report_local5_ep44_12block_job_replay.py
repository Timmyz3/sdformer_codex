#!/usr/bin/env python3
"""Seal the bounded ep44 12-block tagged-job RTL replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
PASS_RE = re.compile(
    r"PASS Local5 ep44 12-block tagged jobs seed=(?P<seed>\d+) "
    r"cycles=(?P<cycles>\d+) jobs=(?P<jobs>\d+) "
    r"token=(?P<token>\d+) weight=(?P<weight>\d+) "
    r"result=(?P<result>\d+) result_stall=(?P<result_stall>\d+) "
    r"token_stall=(?P<token_stall>\d+) weight_stall=(?P<weight_stall>\d+)"
)
BLOCK_RE = re.compile(
    r"BLOCK ordinal=(?P<ordinal>\d+) stage=(?P<stage>\d+) "
    r"block=(?P<block>\d+) group=(?P<group>\d+) empty=(?P<empty>\d+) "
    r"cycles=(?P<cycles>\d+) results=(?P<results>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> tuple[dict[str, int], list[dict[str, int]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?:\bERROR:|\bFATAL:|%Error|mismatch)", text, re.I):
        raise ValueError(f"failure marker in {path}")
    pass_rows = list(PASS_RE.finditer(text))
    block_rows = list(BLOCK_RE.finditer(text))
    if len(pass_rows) != 1 or len(block_rows) != 12:
        raise ValueError(f"incomplete PASS/BLOCK ledger in {path}")
    summary = {
        key: int(value) for key, value in pass_rows[0].groupdict().items()
    }
    blocks = [
        {key: int(value) for key, value in row.groupdict().items()}
        for row in block_rows
    ]
    return summary, blocks


def validate_plan(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema") != "local5_ep44_12block_job_plan_v1"
        or plan.get("status") != "PASS"
        or plan.get("jobs") != 12
        or plan.get("nonempty_jobs") != 10
        or len(plan.get("rows", [])) != 12
    ):
        raise ValueError("12-block plan contract failed")
    for artifact in (plan.get("artifacts") or {}).values():
        path = plan_path.parent / artifact["file"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"plan artifact drift: {path}")
    vector_manifest = Path(plan["source_vector_manifest"])
    if (
        not vector_manifest.is_file()
        or sha256(vector_manifest) != plan["source_vector_manifest_sha256"]
    ):
        raise ValueError("source vector manifest drift")
    vector_contract = json.loads(vector_manifest.read_text(encoding="utf-8"))
    for artifact in (vector_contract.get("artifacts") or {}).values():
        path = vector_manifest.parent / artifact["file"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"source vector artifact drift: {path}")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = args.result_dir.resolve()
    plan_path = args.plan.resolve()

    plan = validate_plan(plan_path)
    logs = {
        "iverilog": result_dir / "iverilog_seed_23133.log",
        "verilator_sva": result_dir / "verilator_seed_23133.log",
    }
    parsed = {name: parse_log(path) for name, path in logs.items()}
    expected_summary = {
        "seed": 23133,
        "jobs": 12,
        "token": 5400,
        "weight": 768,
        "result": 10800,
    }
    checks = {
        f"{simulator}_{field}": summary[field] == value
        for simulator, (summary, _) in parsed.items()
        for field, value in expected_summary.items()
    }
    checks["simulator_summary_match"] = (
        parsed["iverilog"][0] == parsed["verilator_sva"][0]
    )
    checks["simulator_block_ledgers_match"] = (
        parsed["iverilog"][1] == parsed["verilator_sva"][1]
    )
    plan_rows = plan["rows"]
    for simulator, (_, blocks) in parsed.items():
        checks[f"{simulator}_block_identity"] = all(
            row["ordinal"] == ordinal
            and row["stage"] == int(plan_rows[ordinal]["stage"])
            and row["block"] == int(plan_rows[ordinal]["block"])
            and row["group"] == int(plan_rows[ordinal]["group_index"])
            and row["empty"] == int(plan_rows[ordinal]["empty"])
            and row["results"] == 900
            for ordinal, row in enumerate(blocks)
        )
        summary = parsed[simulator][0]
        checks[f"{simulator}_backpressure_hit"] = all(
            summary[field] > 0
            for field in ("result_stall", "token_stall", "weight_stall")
        )
    docs359 = root / "docs/359_DATE终局冻结_20260813.md"
    checks["docs359_frozen"] = (
        docs359.is_file() and sha256(docs359) == DOCS359_SHA256
    )
    checks["yosys_structural"] = (
        "Found and reported 0 problems."
        in (result_dir / "yosys.log").read_text(encoding="utf-8", errors="replace")
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("12-block replay checks failed: " + ",".join(failed))

    source_manifest = result_dir / "source_sha256.txt"
    report = {
        "schema": "local5_ep44_12block_tagged_job_rtl_v1",
        "status": "PASS",
        "evidence": "[rtl]+[profile-qualified-trace]+[real-checkpoint-int8]",
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "scope": (
            "one ep44 OUT_DIM=2 tagged production job per Local5 block, "
            "raw Q/K through score/relation/projection to Acc32 readback"
        ),
        "checks": checks,
        "selection": plan["selection"],
        "jobs": plan["jobs"],
        "nonempty_jobs": plan["nonempty_jobs"],
        "simulators": {
            name: {"summary": summary, "blocks": blocks}
            for name, (summary, blocks) in parsed.items()
        },
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "claim_boundary": [
            "coverage-seeking correctness set; cycle totals are not performance evidence",
            "12 independent real groups covering all blocks, not one sample/window trajectory",
            "not same-window cross-head accumulation",
            "not the 1320-window encoder schedule",
            "pre-bias/pre-BN/pre-requant/pre-residual OUT_DIM=2",
            "weight service checks requested tags/indices and returns the selected group's sealed pair; it does not independently prove a global tile-to-weight address transform",
            "no foundry PPA and no docs/359 replacement",
        ],
        "docs359_sha256": sha256(docs359),
    }
    report_path = result_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sealed_files = {
        path.name: sha256(path)
        for path in sorted(result_dir.iterdir())
        if path.is_file() and path.name != "complete.json"
    }
    complete = {
        "schema": "local5_ep44_12block_tagged_job_complete_v1",
        "status": "SEALED",
        "report_sha256": sha256(report_path),
        "files": sealed_files,
    }
    (result_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "SEALED", "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
