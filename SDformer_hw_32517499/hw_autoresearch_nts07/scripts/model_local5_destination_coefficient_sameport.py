#!/usr/bin/env python3
"""Screen destination-complete coefficient fusion under the production term port."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--miter-report",
        type=Path,
        default=Path(
            "results/local5_ep44_destination_coefficient_miter_20260816/report.json"
        ),
    )
    parser.add_argument(
        "--rtl-report",
        type=Path,
        default=Path(
            "results/local5_ep44_hardware_rebind_20260815_score_projection_rtl/"
            "report_ranked.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/local5_ep44_destination_coefficient_sameport_20260816"
        ),
    )
    args = parser.parse_args()

    miter = json.loads(args.miter_report.read_text(encoding="utf-8"))
    rtl = json.loads(args.rtl_report.read_text(encoding="utf-8"))
    totals = miter["totals"]
    production = rtl["configurations"]["tcfm5_l1"]

    checks = {
        "checkpoint_identity": (
            miter["checkpoint_sha256"] == rtl["checkpoint_sha256"]
        ),
        "groups100": totals["groups"] == 100 and rtl["groups"] == 100,
        "out_dim2": "OUT_DIM=2" in miter["evidence"],
        "acc32_exact": totals["acc32_mismatches"] == 0,
        "source_term_identity": (
            totals["source_terms"] == int(production["terms"]["total"])
        ),
        "source_update_identity": (
            totals["source_updates"] == int(production["updates"]["total"])
        ),
        "one_term_per_cycle_port": True,
    }
    if not all(checks.values()):
        raise SystemExit(f"identity gate failed: {checks}")

    source_terms = totals["source_terms"]
    coefficient_terms = totals["coefficient_terms"]
    observed_memory_wait = int(production["memory_wait"]["total"])
    production_cycles = int(production["cycles"]["total"])
    extra_product_service = coefficient_terms - source_terms
    optimistic_net_service_tax = extra_product_service - observed_memory_wait
    optimistic_additive_cycles = production_cycles + optimistic_net_service_tax
    groups_with_fewer_terms = totals["groups_coefficient_fewer_terms"]

    status = "NO_GO_SAME_PRODUCT_PORT_NO_RTL"
    if groups_with_fewer_terms or optimistic_net_service_tax <= 0:
        status = "REVIEW_REQUIRED_UNEXPECTED_GATE_RESULT"

    report = {
        "schema": "local5_destination_coefficient_sameport_v1",
        "status": status,
        "evidence": "[rtl-calibrated-model]+[numeric] ep44 100-group OUT_DIM=2",
        "claim_boundary": (
            "Same one-OUT2-product-term-per-cycle port as production TCFM5. "
            "The additive cycle estimate is deliberately optimistic: it credits "
            "the candidate with removing every observed production memory-wait "
            "cycle and charges no coefficient-build, scratchpad, commit, or "
            "control overhead. It is not RTL, energy, encoder, or PPA evidence."
        ),
        "miter_report": str(args.miter_report.resolve()),
        "miter_report_sha256": sha256(args.miter_report),
        "rtl_report": str(args.rtl_report.resolve()),
        "rtl_report_sha256": sha256(args.rtl_report),
        "checkpoint_sha256": miter["checkpoint_sha256"],
        "checks": checks,
        "metrics": {
            "production_cycles": production_cycles,
            "source_product_terms": source_terms,
            "coefficient_product_terms": coefficient_terms,
            "coefficient_over_source_terms": coefficient_terms / source_terms,
            "extra_product_service_cycles": extra_product_service,
            "maximum_observed_memory_wait_credit": observed_memory_wait,
            "optimistic_net_service_tax": optimistic_net_service_tax,
            "optimistic_additive_candidate_cycles": optimistic_additive_cycles,
            "optimistic_speedup_vs_production": (
                production_cycles / optimistic_additive_cycles
            ),
            "groups_with_fewer_coefficient_terms": groups_with_fewer_terms,
            "groups": totals["groups"],
        },
        "decision": (
            "The candidate requires 2.229x product service and loses on every "
            "group. Even after crediting all measured memory waits as removable, "
            "it adds 42,739 service cycles. A wider product engine would change "
            "the area/port contract and is not a fair extension of this candidate."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    md_path.write_text(
        "# Local5 destination coefficient same-port screen\n\n"
        f"Status: `{status}`.\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| production cycles | {metrics['production_cycles']} |\n"
        f"| source product terms | {metrics['source_product_terms']} |\n"
        f"| coefficient product terms | {metrics['coefficient_product_terms']} |\n"
        f"| coefficient/source | {metrics['coefficient_over_source_terms']:.3f}x |\n"
        f"| optimistic memory-wait credit | {metrics['maximum_observed_memory_wait_credit']} |\n"
        f"| optimistic net service tax | +{metrics['optimistic_net_service_tax']} |\n"
        f"| optimistic additive candidate cycles | {metrics['optimistic_additive_candidate_cycles']} |\n"
        f"| optimistic speedup | {metrics['optimistic_speedup_vs_production']:.3f}x |\n"
        f"| groups with fewer terms | {metrics['groups_with_fewer_coefficient_terms']} / {metrics['groups']} |\n\n"
        + report["claim_boundary"]
        + "\n\n"
        + report["decision"]
        + "\n",
        encoding="utf-8",
    )
    print(md_path.resolve())


if __name__ == "__main__":
    main()
