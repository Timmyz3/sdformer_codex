#!/opt/anaconda3/bin/python
"""Fail-closed parser for the M2052 ep34 full-40-sample TSBG VCS replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
FIXTURE_JSON = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
FIXTURE_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
STATS_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh"
TB = HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv"
RTL = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
ADAPTER = HW / "rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m2051_ep34_tsbg_full40_cycle_vcs.f"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PASS_RE = re.compile(
    r"^PASS_M2051_EP34_TSBG_FULL40_CYCLE "
    r"workload_slot=(?P<workload_slot>\d+) sample_id=(?P<sample_id>\d+) "
    r"layer=(?P<layer_id>\d+) is_fc2=(?P<is_fc2>\d+) "
    r"token_start=(?P<token_start>\d+) source_groups=(?P<source_groups>\d+) "
    r"physical_groups=(?P<physical_groups>\d+) rows=(?P<live_rows>\d+) "
    r"issues=(?P<issues>\d+) products=(?P<products>\d+) "
    r"commits=(?P<commits>\d+) base_cycles=(?P<base_cycles>\d+) "
    r"tsbg_cycles=(?P<tsbg_cycles>\d+) bundles_base=(?P<bundles_base>\d+) "
    r"bundles_tsbg=(?P<bundles_tsbg>\d+) scalar_base=(?P<scalar_base>\d+) "
    r"scalar_tsbg=(?P<scalar_tsbg>\d+) stale=(?P<stale>\d+) "
    r"retired_replay=(?P<retired_replay>\d+) replay_accept=(?P<replay_accept>\d+) "
    r"reset=(?P<reset>\d+) recovery=(?P<recovery>\d+) "
    r"real_weights=(?P<real_weights>\w+) system_speedup=(?P<system_speedup>\w+)$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_compile(path: Path) -> None:
    text = path.read_text(errors="replace")
    assert text.count("Chronologic VCS (TM)") == 1
    assert text.count("Version V-2023.12-SP1_Full64") == 1
    assert text.count(
        "Top Level Modules:\n       tb_m2051_ep34_tsbg_full40_cycle"
    ) == 1
    assert text.count("7 modules and 0 UDP read.") == 1
    assert text.count("All of 7 modules done") == 1
    assert re.findall(r"^Parsing design file '([^']+)'$", text, re.MULTILINE) == \
        FILELIST.read_text().splitlines()
    for pattern in (
        r"Error-", r"Warning-\[SVAA-RNF\]", r"Warning-\[KUAI\]",
        r"Ignoring.*global_finish_maxfail",
        r"global_finish_maxfail.*(?:ignored|unknown)",
    ):
        assert not re.search(pattern, text, re.IGNORECASE), pattern


def parse_sim(path: Path, fixture_row: dict) -> dict:
    text = path.read_text(errors="replace")
    matches = list(PASS_RE.finditer(text))
    assert len(matches) == 1, path
    row = {key: int(value) if value.isdigit() else value
           for key, value in matches[0].groupdict().items()}
    direct = (
        "workload_slot", "sample_id", "layer_id", "is_fc2", "token_start",
        "source_groups", "live_rows", "issues", "products",
    )
    for key in direct:
        expected_key = "slot" if key == "workload_slot" else key
        assert row[key] == fixture_row[expected_key], (path, key)
    assert row["physical_groups"] == 48 and row["commits"] == 24
    assert row["bundles_base"] == fixture_row["base_misses"] * 12
    assert row["bundles_tsbg"] == fixture_row["tsbg_misses"] * 12
    assert row["scalar_base"] == row["bundles_base"] * 8
    assert row["scalar_tsbg"] == row["bundles_tsbg"] * 8
    assert row["stale"] == 1 and row["replay_accept"] == 0
    assert row["reset"] == 2 and row["recovery"] == 1
    assert row["retired_replay"] == int(row["live_rows"] != 0)
    assert row["real_weights"] == "false" and row["system_speedup"] == "false"
    assert row["base_cycles"] > 0 and row["tsbg_cycles"] > 0
    if row["live_rows"] == 0:
        assert row["base_cycles"] == row["tsbg_cycles"]
        assert text.count("M2048_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE") == 1
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
    row.update({
        "sequence": fixture_row["sequence"],
        "target": fixture_row["target"],
        "token_role": fixture_row["token_role"],
        "speedup": row["base_cycles"] / row["tsbg_cycles"],
        "sim_log_sha256": sha256(path),
    })
    return row


def summarize(rows: list[dict]) -> dict:
    base = sum(row["base_cycles"] for row in rows)
    tsbg = sum(row["tsbg_cycles"] for row in rows)
    bundles_base = sum(row["bundles_base"] for row in rows)
    bundles_tsbg = sum(row["bundles_tsbg"] for row in rows)
    nonempty = [row for row in rows if row["live_rows"] != 0]
    return {
        "workloads": len(rows),
        "nonempty_workloads": len(nonempty),
        "empty_workloads": len(rows) - len(nonempty),
        "base_cycles": base,
        "tsbg_cycles": tsbg,
        "weighted_cycle_speedup": base / tsbg,
        "time_reduction_fraction": 1.0 - tsbg / base,
        "geomean_workload_speedup": math.prod(row["speedup"] for row in rows) **
                                     (1.0 / len(rows)),
        "median_workload_speedup": median(row["speedup"] for row in rows),
        "min_workload_speedup": min(row["speedup"] for row in rows),
        "max_workload_speedup": max(row["speedup"] for row in rows),
        "min_nonempty_workload_speedup": min(row["speedup"] for row in nonempty),
        "bundles_base": bundles_base,
        "bundles_tsbg": bundles_tsbg,
        "bundle_reduction_fraction": 1.0 - bundles_tsbg / bundles_base,
        "scalar_requests_base": bundles_base * 8,
        "scalar_requests_tsbg": bundles_tsbg * 8,
        "scalar_request_reduction_fraction": 1.0 - bundles_tsbg / bundles_base,
    }


def breakdown(rows: list[dict], field: str) -> dict:
    values = {}
    for value in sorted({row[field] for row in rows}, key=str):
        values[str(value)] = summarize([row for row in rows if row[field] == value])
    return values


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
        "contexts": 4, "fc1_layers": 12, "fc2_layers": 4,
        "physical_source_groups": 48, "quartets_per_layer_sample": 3,
        "samples": 40, "sequences": 4, "sources_per_group": 16,
        "supported_layers": 16, "workloads": 1920,
    }
    assert fixture["fixture_sha256"] == sha256(FIXTURE_MEMH)
    assert fixture["stats_sha256"] == sha256(STATS_MEMH)
    assert len(fixture["rows"]) == 1920
    rows = [parse_sim(args.sim_dir / f"sim_slot{slot}.log", fixture["rows"][slot])
            for slot in range(1920)]
    result = {
        "schema": "m2052_ep34_tsbg_full40_vcs_result_r1_v1",
        "status": "RAW_PASS_PENDING_INDEPENDENT_REVIEW",
        "selection": fixture["selection_rule"],
        "selection_uses_performance": False,
        "workload_scope": {
            "checkpoint": "motion_ep34_live93",
            "sequences": 4,
            "samples": list(range(40)),
            "supported_fc_layers": 16,
            "fc1_layers": 12,
            "fc2_layers": 4,
            "token_regions": ["first", "middle", "last"],
            "workloads": 1920,
            "contexts_per_workload": 4,
            "physical_source_groups": 48,
            "unsupported_fc2_layer_ids_over_g48":
                fixture["unsupported_fc2_layer_ids_over_g48"],
            "real_activity_and_sign_descriptors": True,
            "real_weights": False,
            "weight_values": (
                "deterministic directed INT8 values; scheduling and cycle count "
                "are weight-value independent"
            ),
        },
        "axes": {
            "baseline": "ordinary-LRU4 schedule_mode=0",
            "candidate": "TSBG-B4 schedule_mode=1",
            "same_parametric_rtl": True,
            "same_physical_g48_engine": True,
            "smaller_layers_zero_padded_to_g48": True,
            "same_external_ports": True,
            "same_cache_capacity": True,
            "same_backpressure_schedule": True,
            "identical_descriptor_preload_cycles_per_workload": 383,
            "descriptor_preload_excluded_from_execute_cycles": True,
        },
        "rows": rows,
        "aggregate": summarize(rows),
        "breakdown": {
            "target": breakdown(rows, "target"),
            "layer_id": breakdown(rows, "layer_id"),
            "sequence": breakdown(rows, "sequence"),
            "token_role": breakdown(rows, "token_role"),
            "source_groups": breakdown(rows, "source_groups"),
        },
        "identity": {
            "compile_log_sha256": sha256(args.compile_log),
            "fixture_json_sha256": sha256(FIXTURE_JSON),
            "fixture_memh_sha256": sha256(FIXTURE_MEMH),
            "stats_memh_sha256": sha256(STATS_MEMH),
            "testbench_sha256": sha256(TB),
            "rtl_sha256": sha256(RTL),
            "adapter_sha256": sha256(ADAPTER),
            "m803_sha256": sha256(M803),
            "sva_sha256": sha256(SVA),
            "filelist_sha256": sha256(FILELIST),
            "docs359_sha256": sha256(DOCS359),
        },
        "claim_boundary": {
            "directed_real_descriptor_component_cycle_distribution": True,
            "all_fc1_layers_supported": True,
            "all_fc2_layers_supported": False,
            "full_fc_population": False,
            "real_weights": False,
            "same_area": False,
            "energy": False,
            "system_speedup": False,
            "headline": False,
            "paper_admitted": False,
        },
    }
    assert result["aggregate"]["weighted_cycle_speedup"] >= 1.15
    assert result["identity"]["docs359_sha256"] == (
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
