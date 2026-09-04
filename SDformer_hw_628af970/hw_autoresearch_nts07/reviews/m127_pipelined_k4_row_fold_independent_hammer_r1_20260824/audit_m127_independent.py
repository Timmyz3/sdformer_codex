#!/usr/bin/env python3
"""Fail-closed machine audit for the M127 independent hammer review."""

import hashlib
import json
import re
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m127_independent_machine_audit.json"

EXPECTED = {
    "contract": ("contracts/m127_pipelined_k4_row_fold_vcs_contract_r1_20260824.json",
                 "2640b4ba5545cffcd0dd55dce002f4cb3d18222a2379c4f41170888a1a0bc293"),
    "m125_rtl": ("rtl_m125/m125_block_phased_k4_row_fold.sv",
                 "cc343bd514777a215ef5e00cf64f8bf00cea700a1d066bdccd5a16feedcc3d30"),
    "m127_rtl": ("rtl_m127/m127_block_phased_pipelined_k4_row_fold.sv",
                 "5c0c779e8ab463b6589804736bc4d83e77e28cd626a8a117c50caf4a7ea15a5c"),
    "sva": ("verif_m127/m127_block_phased_pipelined_k4_row_fold_assertions.sv",
            "f825e7f2ff7f6617d6cd42c81e620e39675164e430dcf528e1e0c7c1986209bb"),
    "production_tb": ("tb_m127/tb_m127_block_phased_pipelined_k4_row_fold.sv",
                      "abb4462609bf8fe719b7eddde077670fff7a2257632144b794935ae4b26d07a6"),
    "production_filelist": ("dc_handoff/filelists/date_m127_pipelined_k4_row_fold_differential_vcs.f",
                            "10b1b4c156f68f3442b576b156aca5b57c29ca83bb1fdc2f07dbabff5961de63"),
    "production_runner": ("dc_handoff/scripts/run_vcs_m127_pipelined_k4_row_fold.sh",
                          "36043ea63b23ecbfad15adb64e9314dc132a3f2aa90d18d105213b968652a255"),
    "m125_review": ("reviews/m125_block_phased_k4_row_fold_independent_hammer_r1_20260824/manifest.sha256",
                    "ce917784a653cc9b865bb595a59faaa3b10b228c7760abceb1bb87935a99296e"),
    "docs359": ("docs/359_DATE终局冻结_20260813.md",
                "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
    "review_tb": ("reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/tb_m127_independent_hammer.sv",
                  "92809a53c86a9b78dafdd459f34b6a2ba28c75663a40f86e5631e72d0017ba52"),
    "review_filelist": ("reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/m127_independent.f",
                        "338a4fa5e04062e44ca844cf79858ef7ac32cb9e2cd26ebabf38f430e69c61c7"),
    "review_runner": ("reviews/m127_pipelined_k4_row_fold_independent_hammer_r1_20260824/run_vcs_m127_independent_hammer.sh",
                      "239af4ee86d92bfb4c457e6210f30dbad2ded61dbd1283ab18b385e14aca053f"),
}


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL M127 independent audit: " + message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def strict_json(path):
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result

    return json.loads(read(path), object_pairs_hook=pairs_hook,
                      parse_constant=lambda value: require(False, value))


def tokens_from_pass(log, prefix):
    match = re.search(r"^" + re.escape(prefix) + r" (.+)$", log, re.M)
    require(match is not None, "missing pass line " + prefix)
    return dict(item.split("=", 1) for item in match.group(1).split()
                if "=" in item)


def verify_sha_list(path):
    count = 0
    for line in read(path).splitlines():
        digest, label = line.split(None, 1)
        label = label.strip()
        target = Path(label)
        require(target.is_absolute(), "expected absolute VCS output label")
        require(target.is_file(), "missing VCS output " + label)
        require(sha256(target) == digest, "VCS output digest " + label)
        count += 1
    return count


observed = {}
for name, (label, expected) in EXPECTED.items():
    actual = sha256(HW / label)
    require(actual == expected, name + " SHA drift")
    observed[name] = actual

contract = strict_json(HW / EXPECTED["contract"][0])
review = strict_json(REVIEW / "m127_pipelined_k4_row_fold_independent_hammer_review.json")
require(review["score"] == {
    "overall": 91,
    "exact_sha_commercial_vcs": 20,
    "accepted_cycle_and_numeric_equivalence": 30,
    "pipeline_backpressure_reset_and_boundaries": 15,
    "throughput_claim_precision": 13,
    "storage_physical_and_paper_claim_discipline": 13,
}, "score drift")
require(review["severity_counts"] == {"P0": 0, "P1": 1, "P2": 4},
        "severity drift")
for flag in ("dc_frequency_improvement", "macro_inclusive_ppa",
             "physical_speedup", "system_speedup", "headline"):
    require(contract["admission"][flag] is False,
            "unsafe production contract admission " + flag)

for directory in (REVIEW / "sealed_vcs_replay", REVIEW / "independent_vcs"):
    require(read(directory / "compile.rc").strip() == "0",
            str(directory) + " compile rc")
    require(read(directory / "sim.rc").strip() == "0",
            str(directory) + " sim rc")
    combined = read(directory / "sim.raw.log") + "\n" + read(directory / "assert.report")
    require(re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout",
                      combined, re.I | re.M) is None,
            str(directory) + " failure marker")
    require("Version V-2023.12-SP1_Full64" in read(directory / "compile.raw.log"),
            str(directory) + " VCS version")

sealed_log = read(REVIEW / "sealed_vcs_replay/sim.raw.log")
require(contract["expected_pass_line"] in sealed_log,
        "exact production pass line")
sealed_covers = {
    "cp_four_ii1_groups": 1,
    "cp_full_k4": 126,
    "cp_tail_k1": 18,
    "cp_update_stall_release": 35,
    "cp_empty_row": 2,
    "cp_reset_requests_quiesced": 2,
}
sealed_assert = read(REVIEW / "sealed_vcs_replay/assert.report")
for name, count in sealed_covers.items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match",
                      sealed_assert) is not None, "sealed cover " + name)

independent_log = read(REVIEW / "independent_vcs/sim.raw.log")
tokens = tokens_from_pass(independent_log, "PASS M127 independent hammer")
expected_tokens = {
    "rows": "81", "updates": "168", "sources": "572",
    "lanes": "16128", "canonical": "168",
    "stall_cycles": "4642", "max_stall_burst": "63",
    "tail_k1": "18", "tail_k2": "14", "tail_k3": "18",
    "tail_k4": "30", "intra_row_ii1_pairs": "9",
    "four_group_ii1_rows": "1", "inter_row_single_k4_updates": "4",
    "inter_row_single_k4_min_interval": "2",
    "inter_row_single_k4_max_interval": "2",
    "plus512": "1", "minus512": "1", "cycle_exact_checks": "5129",
    "first_group_checks": "15", "first_group_absolute_latency_cycles": "1",
    "first_group_additional_cycles_vs_m125": "0",
    "cache_transition_checks": "1", "cache_attacks": "1",
    "block_attacks": "1", "fill_sequence_attacks": "1",
    "reset_isolation_checks": "2", "pipeline_stall_flush_checks": "1",
    "pair_sum_payload_bits": "1920",
    "full_elastic_stage_bits_at_least": "1950",
    "canonical_groups": "true", "valid_numeric_equivalence": "true",
    "reset_isolation": "true", "intra_row_update_ii1": "true",
    "cross_row_single_group_ii1": "false",
    "dc_frequency_improvement": "false", "physical_speedup": "false",
    "system_speedup": "false", "headline": "false",
}
for key, value in expected_tokens.items():
    require(tokens.get(key) == value, "independent token " + key)

independent_covers = {
    "cp_four_ii1_groups": 1,
    "cp_full_k4": 118,
    "cp_tail_k1": 18,
    "cp_update_stall_release": 144,
    "cp_empty_row": 1,
    "cp_reset_requests_quiesced": 4,
}
independent_assert = read(REVIEW / "independent_vcs/assert.report")
for name, count in independent_covers.items():
    require(re.search(re.escape(name) + r", .* " + str(count) + r" match",
                      independent_assert) is not None,
            "independent cover " + name)

require(verify_sha_list(REVIEW / "vcs_output.sha256") == 6,
        "VCS output manifest count")
rtl = read(HW / EXPECTED["m127_rtl"][0])
require("logic signed [9:0] pipe_pair_sum_q [0:1][0:LANES-1];" in rtl,
        "pair-sum declaration drift")
require("&& !pipe_valid_q && resident_block_valid_q" in rtl,
        "row-ready pipeline exclusion drift")
pair_payload = 2 * 96 * 10
metadata = 1 + 3 + 9 + 16 + 1
require(pair_payload == 1920 and pair_payload + metadata == 1950,
        "pipeline storage arithmetic")
require(contract["architecture"]["pair_sum_pipeline_storage_bits"] == 1920,
        "contract pair payload")
require(contract["architecture"]["first_group_extra_cycles_vs_m125"] == 0,
        "contract first-group differential latency")

machine = {
    "schema": "m127_independent_machine_audit_v1",
    "status": "PASS_FUNCTIONAL_EQUIVALENCE_THROUGHPUT_SCOPE_FINDING_CONFIRMED",
    "score": {"overall": 91, "P0": 0, "P1": 1, "P2": 4},
    "frozen_sha256": observed,
    "production_exact_sha_vcs": {
        "compile_rc": 0, "sim_rc": 0,
        "pass_line_exact": True, "covers": sealed_covers,
    },
    "independent_vcs": {
        "compile_rc": 0, "sim_rc": 0,
        "pass_tokens": tokens, "covers": independent_covers,
        "accepted_cycle_and_numeric_equivalence": True,
    },
    "throughput_counterexample": {
        "intra_row_four_group_update_ii": 1,
        "cross_row_single_group_update_ii": 2,
        "unqualified_ii1_safe": False,
    },
    "pipeline_storage": {
        "pair_sum_payload_bits": pair_payload,
        "elastic_metadata_bits": metadata,
        "elastic_stage_bits_at_least": pair_payload + metadata,
    },
    "claim_boundary": {
        "first_group_absolute_latency_cycles": 1,
        "first_group_additional_cycles_vs_m125": 0,
        "frequency": False, "physical_speedup": False,
        "macro_ppa": False, "system_speedup": False, "headline": False,
    },
}
OUTPUT.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print("PASS M127 independent machine audit score=91 p0=0 p1=1 p2=4 "
      "intra_row_ii=1 cross_row_single_group_ii=2 physical_speedup=false")
