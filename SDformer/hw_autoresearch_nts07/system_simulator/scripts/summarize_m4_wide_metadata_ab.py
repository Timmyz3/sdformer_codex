#!/usr/bin/env python3
"""Close the VCS A/B evidence for bank-coherent M4 temporal metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evidence(run_dir: Path) -> None:
    ledger = run_dir / "evidence.sha256"
    if not ledger.is_file():
        raise ValueError(f"missing evidence ledger: {ledger}")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed evidence line: {line}")
        path = Path(match.group(2))
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file() or sha256(path) != match.group(1):
            raise ValueError(f"evidence SHA mismatch: {path}")


def parse_run(run_dir: Path, require_perf: bool) -> dict[str, Any]:
    verify_evidence(run_dir)
    simulation = (run_dir / "simulation.log").read_text(
        encoding="utf-8", errors="replace"
    )
    assertions = (run_dir / "assertion_report.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    passed = re.search(
        r"PASS_M4_STATEFUL_REAL sequences=80 batches=800 descriptors=12880 "
        r"outputs=9360 local_outputs=7677 motion_outputs=1683 "
        r"request_beats=(\d+) bank_reads=(\d+) request_stalls=(\d+) "
        r"output_stalls=(\d+) rmw_backpressure=(\d+)",
        simulation,
    )
    stream = (
        "PASS_M4_STATEFUL_STREAMING sequences=80 batches=800 outputs=9360 "
        "fifo_writes=9360 fifo_reads=9360"
    )
    if passed is None or stream not in simulation:
        raise ValueError(f"run lacks full streaming admission: {run_dir}")
    if re.search(r"Fatal:|Assertion failed|failed at", simulation + assertions):
        raise ValueError(f"run contains a VCS failure: {run_dir}")
    perf = re.search(
        r"PASS_M4_STATEFUL_STREAMING_PERF sequences=80 batches=800 "
        r"outputs=9360 cycles=(\d+)", simulation,
    )
    if require_perf != (perf is not None):
        raise ValueError(f"run performance mode mismatch: {run_dir}")
    cover_matches = [
        int(value) for value in re.findall(
            r"cp_rmw_backpressure,\s+\d+ attempts,\s+(\d+) match",
            assertions,
        )
    ]
    return {
        "run_dir": str(run_dir.resolve()),
        "evidence_ledger_sha256": sha256(run_dir / "evidence.sha256"),
        "request_beats": int(passed.group(1)),
        "bank_reads": int(passed.group(2)),
        "request_stalls": int(passed.group(3)),
        "output_stalls": int(passed.group(4)),
        "rmw_backpressure_cycles": int(passed.group(5)),
        "streaming_cycles": int(perf.group(1)) if perf else None,
        "rmw_backpressure_cover_max": max(cover_matches, default=0),
    }


def summarize(
    shared_perf: dict[str, Any],
    legacy_perf: dict[str, Any],
    shared_random: dict[str, Any],
    legacy_random: dict[str, Any],
    bit_audit: dict[str, Any],
) -> dict[str, Any]:
    for field in ("request_beats", "bank_reads", "streaming_cycles"):
        if shared_perf[field] != legacy_perf[field]:
            raise ValueError(f"wide metadata A/B changed {field}")
    if shared_perf["request_stalls"] != 0 or shared_perf["output_stalls"] != 0:
        raise ValueError("shared performance run is not always-ready")
    if legacy_perf["request_stalls"] != 0 or legacy_perf["output_stalls"] != 0:
        raise ValueError("legacy performance run is not always-ready")
    random_fields = (
        "request_beats", "bank_reads", "request_stalls", "output_stalls",
        "rmw_backpressure_cycles",
    )
    for field in random_fields:
        if shared_random[field] != legacy_random[field]:
            raise ValueError(f"random wide metadata A/B changed {field}")
    for label, random_run in (
        ("shared", shared_random), ("legacy", legacy_random),
    ):
        if (random_run["request_stalls"] <= 0 or
                random_run["output_stalls"] <= 0 or
                random_run["rmw_backpressure_cycles"] <= 0 or
                random_run["rmw_backpressure_cover_max"] <= 0):
            raise ValueError(f"{label} random run missed required backpressure")
    if bit_audit.get("status") != "PASS_M4_WIDE_METADATA_BIT_AUDIT_PRE_DC":
        raise ValueError("metadata bit audit is not admitted")
    return {
        "bounded_workload_functional_cycle_match": True,
        "always_ready_cycles": shared_perf["streaming_cycles"],
        "request_beats": shared_perf["request_beats"],
        "bank_reads": shared_perf["bank_reads"],
        "shared_perf": shared_perf,
        "legacy_perf": legacy_perf,
        "shared_random": shared_random,
        "legacy_random": legacy_random,
        "metadata_bits": {
            "legacy": bit_audit["legacy_per_bank_metadata_bits"],
            "shared": bit_audit["shared_wide_metadata_bits"],
            "reduction": bit_audit["metadata_bit_reduction"],
            "reduction_fraction": bit_audit["metadata_reduction_fraction"],
            "persistent_destination_data_plus_metadata_reduction_fraction":
                bit_audit[
                    "persistent_destination_data_plus_metadata_reduction_fraction"
                ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-perf", type=Path, required=True)
    parser.add_argument("--legacy-perf", type=Path, required=True)
    parser.add_argument("--shared-random", type=Path, required=True)
    parser.add_argument("--legacy-random", type=Path, required=True)
    parser.add_argument("--bit-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bit_audit = json.loads(args.bit_audit.read_text(encoding="utf-8"))
    result = {
        "schema": "m4_wide_metadata_vcs_ab_v1",
        "status": "PASS_M4_WIDE_METADATA_VCS_AB_PRE_DC",
        "claim_boundary": (
            "Cycle-identical Synopsys VCS A/B on the bounded checkpoint-bitmap "
            "Motion-enriched workload, plus exact RTL-state bit counts. Acc32 "
            "uses deterministic synthetic INT8 weights. No standard-cell or "
            "SRAM macro area, energy, full-population, or full-system speedup "
            "is claimed."
        ),
        **summarize(
            parse_run(args.shared_perf, True),
            parse_run(args.legacy_perf, True),
            parse_run(args.shared_random, False),
            parse_run(args.legacy_random, False),
            bit_audit,
        ),
        "bit_audit": str(args.bit_audit.resolve()),
        "bit_audit_sha256": sha256(args.bit_audit),
        "summarizer_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS M4 wide metadata VCS A/B -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
