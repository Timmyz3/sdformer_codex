#!/usr/bin/env python3
import hashlib
import json
import math
import pathlib
import re


REVIEW = pathlib.Path(__file__).resolve().parent
HW = REVIEW.parent.parent


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL M126 independent audit: " + message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


audit = json.loads(read(REVIEW / "m126_block_phased_k4_forwarding_accumulator_independent_audit.json"))
contract = json.loads(read(HW / "contracts/m126_block_phased_k4_forwarding_accumulator_vcs_contract_r1_20260824.json"))
m122 = json.loads(read(HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json"))

frozen = {
    "contract_sha256": HW / "contracts/m126_block_phased_k4_forwarding_accumulator_vcs_contract_r1_20260824.json",
    "m125_rtl_sha256": HW / "rtl_m125/m125_block_phased_k4_row_fold.sv",
    "m123_core_rtl_sha256": HW / "rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv",
    "m123_adapter_rtl_sha256": HW / "rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv",
    "m126_rtl_sha256": HW / "rtl_m126/m126_block_phased_k4_forwarding_accumulator_island.sv",
    "m126_sva_sha256": HW / "verif_m126/m126_block_phased_k4_forwarding_accumulator_island_assertions.sv",
    "production_tb_sha256": HW / "tb_m126/tb_m126_block_phased_k4_forwarding_accumulator_island.sv",
    "filelist_sha256": HW / "dc_handoff/filelists/date_m126_block_phased_k4_forwarding_accumulator_directed_vcs.f",
    "m122_result_sha256": HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json",
    "m125_review_manifest_sha256": HW / "reviews/m125_block_phased_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256",
    "m123_review_manifest_sha256": HW / "reviews/m123_w384_signed19_forwarding_accumulator_independent_hammer_r1_20260824/manifest.sha256",
    "docs_359_sha256": HW / "docs/359_DATE终局冻结_20260813.md",
}
for key, path in frozen.items():
    require(sha256(path) == audit["frozen_input_identity"][key], key + " drift")

for name, value in contract["frozen_sources"].items():
    mapping = {
        "m125_rtl_sha256": "m125_rtl_sha256",
        "m123_core_rtl_sha256": "m123_core_rtl_sha256",
        "m123_adapter_rtl_sha256": "m123_adapter_rtl_sha256",
        "m126_rtl_sha256": "m126_rtl_sha256",
        "m126_sva_sha256": "m126_sva_sha256",
        "m126_testbench_sha256": "production_tb_sha256",
        "m126_filelist_sha256": "filelist_sha256",
    }
    require(value == audit["frozen_input_identity"][mapping[name]], "contract source " + name)
for flag in ("foundry_weight_macro", "foundry_accumulator_macro", "macro_inclusive_ppa",
             "physical_speedup", "system_speedup", "headline"):
    require(contract["admission"][flag] is False, "unsafe production admission " + flag)

sealed = REVIEW / "sealed_vcs_rerun"
independent = REVIEW / "independent_vcs"
boundary = REVIEW / "boundary_vcs"
for directory in (sealed, independent, boundary):
    require(read(directory / "compile.rc").strip() == "0", str(directory) + " compile rc")
    require(read(directory / "sim.rc").strip() == "0", str(directory) + " sim rc")
    bad = re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout",
                    read(directory / "sim.raw.log"), re.I | re.M)
    require(bad is None, str(directory) + " failure marker")

sealed_log = read(sealed / "sim.raw.log")
require(contract["expected_pass_line"] in sealed_log, "sealed exact pass line")
sealed_assert = read(sealed / "assert.report")
for name, count in audit["production_exact_sha_vcs_rerun"]["covers"].items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match", sealed_assert) is not None,
            "sealed cover " + name)

independent_log = read(independent / "sim.raw.log")
match = re.search(r"^PASS M126 independent hammer (.+)$", independent_log, re.M)
require(match is not None, "independent pass line")
tokens = dict(item.split("=", 1) for item in match.group(1).split() if "=" in item)
expected = {
    "positive_rows": "12", "positive_folds": "18",
    "positive_accumulator_accepts": "18", "selected_sources": "65",
    "lane_writes": "18", "write_lane_checks": "1728",
    "forwarding_pairs": "7", "full_k4": "14", "tails": "4",
    "same_row_replays": "2", "block_transition_checks": "1",
    "plus512": "1", "minus512": "1", "commits": "3072",
    "commit_lane_checks": "294912", "commit_stalls": "1088",
    "reset_high_cycles": "9", "reset_handshake_violations": "0",
    "reset_pending_internal_write_visible": "1",
    "reset_edge_suppressed_writes": "1",
    "source_update_write_commit_conservation": "true",
    "reset_isolation": "true", "physical_speedup": "false",
    "system_speedup": "false",
}
for key, value in expected.items():
    require(tokens.get(key) == value, "independent token " + key)
independent_assert = read(independent / "assert.report")
for name, count in audit["independent_adversarial_vcs"]["covers"].items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match", independent_assert) is not None,
            "independent cover " + name)

boundary_log = read(boundary / "sim.raw.log")
match = re.search(r"^PASS M126 overflow identity hammer (.+)$", boundary_log, re.M)
require(match is not None, "boundary pass line")
tokens = dict(item.split("=", 1) for item in match.group(1).split() if "=" in item)
expected = {
    "identity_row_accepts": "1", "identity_fold_accepts": "0",
    "identity_writes": "0", "out_of_range_row_fail_closed": "true",
    "overflow_row_accepts": "512", "overflow_fold_accepts": "512",
    "overflow_writes": "511", "last_valid_lane0": "261632",
    "overflow_fail_closed": "true", "overflow_retry": "false",
    "physical_speedup": "false", "system_speedup": "false",
}
for key, value in expected.items():
    require(tokens.get(key) == value, "boundary token " + key)

traffic = audit["claim_audit"]["directed_traffic_ratio"]
require(math.fabs(float(traffic["source_contributions"]) / traffic["fold_updates"]
                  - traffic["ratio"]) < 1e-15, "directed traffic ratio")
k1 = [item for item in m122["fold_dse"] if item["fold_sources_per_update"] == 1][0]
k4 = [item for item in m122["fold_dse"] if item["fold_sources_per_update"] == 4][0]
base = k4["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"]
k1_cycles = k1["dual_timeline_recurrence"]["candidate_cycles"]
k4_cycles = k4["dual_timeline_recurrence"]["candidate_cycles"]
require(math.fabs(float(base) / k4_cycles - 3.1725369008459166) < 1e-15,
        "M122 fixed8 projection")
require(math.fabs(float(k1_cycles) / k4_cycles - 1.2512657845537327) < 1e-15,
        "M122 incremental projection")
require(audit["admission"]["reset_external_isolation"] is True, "reset repair admission")
require(audit["admission"]["arbitrary_overflow_exact_once"] is False, "overflow boundary")
require(audit["admission"]["invalid_row_rejected_before_accept"] is False, "identity boundary")
require(audit["admission"]["physical_performance"] is False, "physical claim")
require(audit["admission"]["system_performance"] is False, "system claim")
require(audit["score"] == {"total": 92, "out_of": 100, "p0": 0, "p1": 0, "p2": 4},
        "score drift")

print("PASS M126 independent machine audit score=92 p0=0 p1=0 p2=4 reset_repair=true directed_functional=true physical_speedup=false system_speedup=false")
