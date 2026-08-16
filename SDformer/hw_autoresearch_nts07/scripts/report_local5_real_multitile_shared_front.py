#!/usr/bin/env python3
"""Report the matched real-checkpoint Local5 multi-tile memo ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PASS_RE = re.compile(
    r"^PASS Local5 multi-tile memo=(?P<memo>[01]).*?"
    r"cycles=(?P<cycles>\d+) token=(?P<token>\d+).*?"
    r"hits=(?P<hits>\d+) fallback=(?P<fallback>\d+) "
    r"replay_records=(?P<replay>\d+).*?"
    r"partial=(?P<partial>\d+) final=(?P<final>\d+).*?"
    r"weight_cycles=(?P<weight_cycles>\d+) "
    r"frontend_cycles=(?P<frontend_cycles>\d+) "
    r"readout_cycles=(?P<readout_cycles>\d+) "
    r"release_cycles=(?P<release_cycles>\d+) "
    r"rmw_cycles=(?P<rmw_cycles>\d+) "
    r"drain_cycles=(?P<drain_cycles>\d+) "
    r"scheduler_cycles=(?P<scheduler_cycles>\d+).*?$",
    re.MULTILINE,
)
BAD_RE = re.compile(r"\b(?:ERROR|FATAL):|mismatch", re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_terminal(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    if BAD_RE.search(text):
        raise ValueError(f"bad marker in {path}")
    matches = list(PASS_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one terminal PASS in {path}")
    return {key: int(value) for key, value in matches[0].groupdict().items()}


def load_side(directory: Path, memo: int) -> dict[str, object]:
    rows = {
        simulator: parse_terminal(directory / f"{simulator}.log")
        for simulator in ("icarus", "verilator")
    }
    if rows["icarus"] != rows["verilator"]:
        raise ValueError(f"cross-simulator ledger mismatch in {directory}")
    row = rows["icarus"]
    if row["memo"] != memo:
        raise ValueError(f"unexpected memo mode in {directory}")

    merge = json.loads((directory / "merge_report.json").read_text(encoding="utf-8"))
    if (
        merge.get("status") != "PASS_INTEGRATED_CROSS_HEAD_CANARY_NOT_G0"
        or merge.get("formal_g0") != "DENY"
        or bool(merge.get("use_relation_memo")) != bool(memo)
        or merge.get("scalar_count") != 43_200
        or any(item.get("mismatch_count") != 0 for item in merge.get("simulators", []))
    ):
        raise ValueError(f"merge contract failed in {directory}")

    return {
        "directory": str(directory),
        "ledger": row,
        "vector_result_mode": merge.get("vector_result_mode"),
        "identity": merge.get("identity"),
        "merge_report_sha256": sha256(directory / "merge_report.json"),
        "source_sha256_file_sha256": sha256(directory / "source_sha256.txt"),
        "expected_sha256": sha256(
            directory / "software_expected" / "software_expected.npz"
        ),
        "input_sha256": sha256(directory / "vectors" / "combined_head_inputs.txt"),
        "weight_sha256": sha256(directory / "vectors" / "projection_weights.txt"),
        "actual_acc32_sha256": merge["simulators"][0]["actual_acc32_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--memo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_side(args.baseline, memo=0)
    memo = load_side(args.memo, memo=1)
    identity_keys = (
        "source_sha256_file_sha256",
        "expected_sha256",
        "input_sha256",
        "weight_sha256",
        "actual_acc32_sha256",
        "vector_result_mode",
        "identity",
    )
    if any(baseline[key] != memo[key] for key in identity_keys):
        raise ValueError("baseline/memo source, workload, weight, or result identity mismatch")

    b = baseline["ledger"]
    m = memo["ledger"]
    unchanged = (
        "partial",
        "final",
        "weight_cycles",
        "readout_cycles",
        "release_cycles",
        "rmw_cycles",
        "drain_cycles",
        "scheduler_cycles",
    )
    if any(b[key] != m[key] for key in unchanged):
        raise ValueError("memo changed a non-front-end ledger")
    if m["hits"] + m["fallback"] != 6 or m["token"] != (3 + m["fallback"]) * 450:
        raise ValueError("memo hit/fallback/token conservation failed")

    report = {
        "schema": "local5_real_multitile_shared_front_ablation_v1",
        "status": "PASS_RTL_NEGATIVE_PERFORMANCE_NOT_G0",
        "evidence": "[rtl]+[软件整数金参考]",
        "scope": (
            f"sample{memo['identity']['sample']}/stage{memo['identity']['stage']}/"
            f"block{memo['identity']['block']}/window{memo['identity']['window']}, "
            f"{memo['identity']['heads']} input heads x "
            f"{memo['identity']['heads']} output tiles, "
            f"OUT_DIM={memo['identity']['out_dim']}, "
            + (
                "vector child-result transfer"
                if memo["vector_result_mode"]
                else "scalar child-result transfer"
            )
        ),
        "baseline": baseline,
        "memo": memo,
        "comparison": {
            "speedup": b["cycles"] / m["cycles"],
            "cycle_reduction_fraction": 1.0 - m["cycles"] / b["cycles"],
            "token_reduction_fraction": 1.0 - m["token"] / b["token"],
            "frontend_cycle_reduction_fraction": (
                1.0 - m["frontend_cycles"] / b["frontend_cycles"]
            ),
            "frontend_cycles_saved": b["frontend_cycles"] - m["frontend_cycles"],
            "total_cycles_saved": b["cycles"] - m["cycles"],
        },
        "decision": "NO_GO_AS_STANDALONE_DATE_CONTRIBUTION_KEEP_AS_COMPLETENESS_EVIDENCE",
        "claim_boundary": [
            "The memo path is a true one-build/multi-output-tile relation replay, not naive front-end replay.",
            "Only one of three heads is resident on this real window; four later jobs fall back exactly.",
            "Cycles are component RTL cycles, not encoder speedup, energy, DC, or ASIC PPA.",
            "The result must not replace any frozen docs/359 number.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["comparison"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
