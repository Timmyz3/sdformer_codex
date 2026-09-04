#!/usr/bin/env python3
"""Generate the DP-TME port contract for the paper 15x15 spatial window."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_dptme_port_contract import analyze, write_markdown


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = analyze(positions=15 * 15)
    result["geometry"]["window"] = [15, 15]
    result["geometry"]["time_planes"] = 2
    result["evidence_scope"] = "fullres_w15_architecture_lower_bound_not_measured_full_encoder_latency"
    output = ROOT / "results/dptme_fullres_w15_port_contract.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, output.with_suffix(".md"))
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
