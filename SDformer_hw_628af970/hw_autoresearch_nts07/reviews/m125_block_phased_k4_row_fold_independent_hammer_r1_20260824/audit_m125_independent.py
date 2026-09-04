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
        raise SystemExit("FAIL M125 independent audit: " + message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


audit_path = REVIEW / "m125_block_phased_k4_row_fold_independent_audit.json"
audit = json.loads(read(audit_path))
contract = json.loads(read(HW / "contracts/m125_block_phased_k4_row_fold_vcs_contract_r1_20260824.json"))
m122 = json.loads(read(HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json"))

frozen = {
    "contract_sha256": HW / "contracts/m125_block_phased_k4_row_fold_vcs_contract_r1_20260824.json",
    "rtl_sha256": HW / "rtl_m125/m125_block_phased_k4_row_fold.sv",
    "sva_sha256": HW / "verif_m125/m125_block_phased_k4_row_fold_assertions.sv",
    "production_tb_sha256": HW / "tb_m125/tb_m125_block_phased_k4_row_fold.sv",
    "filelist_sha256": HW / "dc_handoff/filelists/date_m125_block_phased_k4_row_fold_directed_vcs.f",
    "m122_correction_sha256": HW / "contracts/m122_r1_row_fold_admission_and_width_correction_r1_20260824.json",
    "m122_result_sha256": HW / "results/m122_w384_row_synchronous_source_fold_dse_r1_20260824/m122_w384_row_synchronous_source_fold_dse.json",
    "m122_review_manifest_sha256": HW / "reviews/m122_w384_row_synchronous_source_fold_independent_hammer_r1_20260824/manifest.sha256",
    "docs_359_sha256": HW / "docs/359_DATE终局冻结_20260813.md",
}
for key, path in frozen.items():
    require(sha256(path) == audit["frozen_input_identity"][key], key + " drift")

require(contract["frozen_sources"]["rtl_sha256"] == audit["frozen_input_identity"]["rtl_sha256"], "contract RTL identity")
require(contract["architecture"]["logical_weight_cache_bytes"] == 1536, "contract cache bytes")
require(contract["architecture"]["logical_read_bits_per_update"] == 3072, "contract logical reads")
require(contract["architecture"]["fold_bits"] == 11, "contract fold width")
for flag in ("m123_accumulator_integrated", "foundry_weight_macro", "physical_speedup", "system_speedup", "headline"):
    require(contract["admission"][flag] is False, "unsafe contract admission " + flag)

sealed = REVIEW / "sealed_vcs_rerun"
independent = REVIEW / "independent_vcs"
for directory in (sealed, independent):
    require(read(directory / "compile.rc").strip() == "0", str(directory) + " compile rc")
    require(read(directory / "sim.rc").strip() == "0", str(directory) + " sim rc")
    bad = re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout",
                    read(directory / "sim.raw.log") + "\n" + read(directory / "assert.report"),
                    re.I | re.M)
    require(bad is None, str(directory) + " failure marker")

sealed_log = read(sealed / "sim.raw.log")
require(contract["expected_pass_line"] in sealed_log, "sealed exact pass line")
sealed_assert = read(sealed / "assert.report")
for name, count in audit["production_exact_sha_vcs_rerun"]["covers"].items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match", sealed_assert) is not None,
            "sealed cover " + name)

independent_log = read(independent / "sim.raw.log")
match = re.search(r"^PASS M125 independent hammer (.+)$", independent_log, re.M)
require(match is not None, "independent pass line")
tokens = dict(item.split("=", 1) for item in match.group(1).split() if "=" in item)
expected_tokens = {
    "rows": "9", "updates": "12", "sources": "40", "lanes": "1152",
    "stalls": "28", "full_k4": "8", "tails": "4", "same_row_replays": "1",
    "cache_identity_attacks": "1", "block_identity_attacks": "1",
    "fill_sequence_attacks": "1", "block_transition_checks": "1",
    "plus512": "1", "minus512": "1", "reset_fill_phantom": "1",
    "reset_row_phantom": "1", "reset_update_visible": "1",
    "reset_update_phantom": "1", "reset_quiescence": "false",
    "canonical_lowest4": "true", "source_conservation": "true",
    "physical_speedup": "false", "system_speedup": "false",
}
for key, value in expected_tokens.items():
    require(tokens.get(key) == value, "independent token " + key)

independent_assert = read(independent / "assert.report")
for name, count in audit["independent_adversarial_vcs"]["sva_covers"].items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match", independent_assert) is not None,
            "independent cover " + name)

require(16 * 96 * 8 // 8 == audit["claim_audit"]["logical_cache"]["bytes"], "cache equation")
require(4 * 96 * 8 == audit["claim_audit"]["logical_reads"]["bits_per_update"], "read equation")
k1 = [item for item in m122["fold_dse"] if item["fold_sources_per_update"] == 1][0]
k4 = [item for item in m122["fold_dse"] if item["fold_sources_per_update"] == 4][0]
baseline = k4["dual_timeline_recurrence"]["fair_fixed8_baseline_cycles"]
k1_cycles = k1["dual_timeline_recurrence"]["candidate_cycles"]
k4_cycles = k4["dual_timeline_recurrence"]["candidate_cycles"]
require(baseline == 1114863448 and k1_cycles == 439708199 and k4_cycles == 351410711,
        "projection cycle provenance")
require(math.fabs(float(baseline) / k4_cycles - 3.1725369008459166) < 1e-15,
        "3.1725 projection arithmetic")
require(math.fabs(float(k1_cycles) / k4_cycles - 1.2512657845537327) < 1e-15,
        "1.2513 incremental arithmetic")
require(audit["claim_audit"]["cycle_projection"]["physical_speedup"] is False,
        "physical claim boundary")
require(audit["claim_audit"]["cycle_projection"]["system_speedup"] is False,
        "system claim boundary")
require(audit["standalone_admission"]["reset_free_functional"] is True,
        "reset-free admission")
require(audit["standalone_admission"]["reset_safe"] is False,
        "reset finding")
require(audit["score"] == {"total": 87, "out_of": 100, "p0": 0, "p1": 1, "p2": 3},
        "score drift")

print("PASS M125 independent machine audit score=87 p0=0 p1=1 p2=3 reset_free_admission=true reset_safe=false physical_speedup=false system_speedup=false")
