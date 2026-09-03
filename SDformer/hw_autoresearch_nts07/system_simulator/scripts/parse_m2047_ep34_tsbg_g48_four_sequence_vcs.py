#!/usr/bin/env python3
"""Fail-closed parser for the M2047 four-sequence TSBG VCS run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
FIXTURE_JSON = HW / "tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.json"
FIXTURE_MEMH = HW / "tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.memh"
TB = HW / "tb_m2018/tb_m2046_ep34_tsbg_g48_four_sequence_cycle.sv"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER = HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m2046_ep34_tsbg_g48_four_sequence_cycle_vcs.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    0: {"sample_id": 0, "rows": 149, "issues": 1278, "products": 29472,
        "bundles_base": 1788, "bundles_tsbg": 576,
        "scalar_base": 14304, "scalar_tsbg": 4608},
    1: {"sample_id": 10, "rows": 159, "issues": 1410, "products": 31680,
        "bundles_base": 1908, "bundles_tsbg": 564,
        "scalar_base": 15264, "scalar_tsbg": 4512},
    2: {"sample_id": 20, "rows": 174, "issues": 1668, "products": 42240,
        "bundles_base": 2088, "bundles_tsbg": 576,
        "scalar_base": 16704, "scalar_tsbg": 4608},
    3: {"sample_id": 30, "rows": 153, "issues": 1296, "products": 28416,
        "bundles_base": 1836, "bundles_tsbg": 576,
        "scalar_base": 14688, "scalar_tsbg": 4608},
}

PASS_RE = re.compile(
    r"^PASS_M2046_EP34_TSBG_G48_CYCLE "
    r"sample_slot=(?P<sample_slot>\d+) sample_id=(?P<sample_id>\d+) "
    r"layer=(?P<layer>\d+) rows=(?P<rows>\d+) issues=(?P<issues>\d+) "
    r"products=(?P<products>\d+) commits=(?P<commits>\d+) "
    r"base_cycles=(?P<base_cycles>\d+) tsbg_cycles=(?P<tsbg_cycles>\d+) "
    r"bundles_base=(?P<bundles_base>\d+) bundles_tsbg=(?P<bundles_tsbg>\d+) "
    r"scalar_base=(?P<scalar_base>\d+) scalar_tsbg=(?P<scalar_tsbg>\d+) "
    r"stale=(?P<stale>\d+) retired_replay=(?P<retired_replay>\d+) "
    r"replay_accept=(?P<replay_accept>\d+) reset=(?P<reset>\d+) "
    r"recovery=(?P<recovery>\d+) system_speedup=(?P<system_speedup>\w+)$",
    re.MULTILINE,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_compile(path: Path) -> None:
    text = path.read_text(errors="replace")
    assert text.count("Chronologic VCS (TM)") == 1
    assert text.count("Version V-2023.12-SP1_Full64") == 1
    assert text.count("Top Level Modules:\n       tb_m2046_ep34_tsbg_g48_four_sequence_cycle") == 1
    assert text.count("7 modules and 0 UDP read.") == 1
    assert text.count("All of 7 modules done") == 1
    parsed = re.findall(r"^Parsing design file '([^']+)'$", text, re.MULTILINE)
    assert parsed == FILELIST.read_text().splitlines()
    for pattern in (
        r"Error-", r"Warning-\[SVAA-RNF\]", r"Warning-\[KUAI\]",
        r"Ignoring.*global_finish_maxfail",
        r"global_finish_maxfail.*(?:ignored|unknown)",
    ):
        assert not re.search(pattern, text, re.IGNORECASE), pattern


def parse_sim(path: Path, expected_slot: int) -> dict[str, int | float]:
    text = path.read_text(errors="replace")
    matches = list(PASS_RE.finditer(text))
    assert len(matches) == 1
    row = {key: int(value) if value.isdigit() else value
           for key, value in matches[0].groupdict().items()}
    assert row["sample_slot"] == expected_slot
    expected = EXPECTED[expected_slot]
    for key, value in expected.items():
        assert row[key] == value, (expected_slot, key, row[key], value)
    assert row["layer"] == 28 and row["commits"] == 24
    assert row["stale"] == 1 and row["retired_replay"] == 1
    assert row["replay_accept"] == 0 and row["reset"] == 2
    assert row["recovery"] == 1 and row["system_speedup"] == "false"
    assert row["base_cycles"] > row["tsbg_cycles"] > 0
    for pattern in (
        r"Fatal:", r"\$(?:error|fatal)", r"Assertion[^\n]*failed",
        r"Error-\[SVA", r"whole-test watchdog expired", r"LOAD_TIMEOUT",
    ):
        assert not re.search(pattern, text, re.IGNORECASE), (path, pattern)
    assert re.search(r"sva_tsbg\.cp_bridge_negative, \d+ attempts, [1-9]\d* match", text)
    assert re.search(r"sva_tsbg\.cp_stale_attack, \d+ attempts, [1-9]\d* match", text)
    assert re.search(
        r"sva_tsbg\.cp_reset_recovery_minimum_one_cycle, \d+ attempts, [1-9]\d* match",
        text,
    )
    row["speedup"] = row["base_cycles"] / row["tsbg_cycles"]
    row["sim_log_sha256"] = sha(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parse_compile(args.compile_log)
    fixture = json.loads(FIXTURE_JSON.read_text())
    assert fixture["selection_uses_performance"] is False
    assert fixture["geometry"] == {
        "contexts": 4, "groups": 48, "samples": 4, "sources": 16}
    assert fixture["fixture_sha256"] == sha(FIXTURE_MEMH)
    rows = [parse_sim(args.sim_dir / f"sim_slot{slot}.log", slot)
            for slot in range(4)]

    base_cycles = sum(int(row["base_cycles"]) for row in rows)
    tsbg_cycles = sum(int(row["tsbg_cycles"]) for row in rows)
    bundles_base = sum(int(row["bundles_base"]) for row in rows)
    bundles_tsbg = sum(int(row["bundles_tsbg"]) for row in rows)
    scalar_base = sum(int(row["scalar_base"]) for row in rows)
    scalar_tsbg = sum(int(row["scalar_tsbg"]) for row in rows)
    result = {
        "schema": "m2047_ep34_tsbg_g48_four_sequence_vcs_result_r1_v1",
        "status": "RAW_PASS_PENDING_INDEPENDENT_REVIEW",
        "selection": fixture["selection_rule"],
        "selection_uses_performance": False,
        "workload_scope": {
            "checkpoint": "motion_ep34_live93",
            "sequences": 4,
            "samples": [0, 10, 20, 30],
            "layer": 28,
            "tokens_per_sample": 4,
            "source_groups": 48,
            "real_activity_and_sign_descriptors": True,
            "real_weights": False,
            "weight_values": "deterministic directed INT8 values; schedule is weight-value independent",
        },
        "axes": {
            "baseline": "ordinary-LRU4 schedule_mode=0",
            "candidate": "TSBG-B4 schedule_mode=1",
            "same_parametric_rtl": True,
            "same_external_ports": True,
            "same_cache_capacity": True,
            "same_backpressure_schedule": True,
        },
        "rows": rows,
        "aggregate": {
            "base_cycles": base_cycles,
            "tsbg_cycles": tsbg_cycles,
            "weighted_cycle_speedup": base_cycles / tsbg_cycles,
            "geomean_slot_speedup": math.prod(float(row["speedup"]) for row in rows) ** 0.25,
            "min_slot_speedup": min(float(row["speedup"]) for row in rows),
            "max_slot_speedup": max(float(row["speedup"]) for row in rows),
            "bundles_base": bundles_base,
            "bundles_tsbg": bundles_tsbg,
            "bundle_reduction_fraction": 1.0 - bundles_tsbg / bundles_base,
            "scalar_requests_base": scalar_base,
            "scalar_requests_tsbg": scalar_tsbg,
            "scalar_request_reduction_fraction": 1.0 - scalar_tsbg / scalar_base,
        },
        "identity": {
            "compile_log_sha256": sha(args.compile_log),
            "fixture_json_sha256": sha(FIXTURE_JSON),
            "fixture_memh_sha256": sha(FIXTURE_MEMH),
            "testbench_sha256": sha(TB),
            "rtl_sha256": sha(RTL),
            "adapter_sha256": sha(ADAPTER),
            "m803_sha256": sha(M803),
            "sva_sha256": sha(SVA),
            "filelist_sha256": sha(FILELIST),
            "docs359_sha256": sha(DOCS359),
        },
        "claim_boundary": {
            "directed_real_descriptor_component_cycle_result": True,
            "full_fc1": False,
            "full_capture": False,
            "real_weights": False,
            "same_area": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_admitted": False,
        },
    }
    assert result["identity"]["docs359_sha256"] == (
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
