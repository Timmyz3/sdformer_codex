#!/usr/bin/env python3
"""M911: H67 FC2 equal-bandwidth K8/K1x8 SRAM-latency premodel.

This is an additive CPU-only evidence adapter.  It does not claim that the
M803 or K1x8 RTL was replayed cycle by cycle on all 120 records.  It binds the
payload-checked M218 lane-aligned service schedule to the fair eight-bank
K8/K1x8 memory boundary at fixed L=1/2/4 and reports the deliberately
conservative memory-service envelope in which both consume the same critical
per-bank schedule.  Controller differences remain represented only by the
separate frozen directed VCS receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_CONTRACT = (
    "a956a18ffdb8c7d5d611c4b897a6c1c0ad27fa86da6965241dd406c1ad50a001"
)
EXPECTED_DOCS359 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
EXPECTED_WORK = {
    "records": 120,
    "tokens": 5580000,
    "events": 143894510,
    "raw96_beats": 36480000,
    "nonzero96_descriptors": 18869376,
    "k8_group_commands": 73380812,
}
EXPECTED_LATENCY_CYCLES = {
    1: 504300928,
    2: 508016984,
    4: 515449096,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator, denominator):
    require(float(denominator) > 0.0, "zero ratio denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "float": float(numerator) / float(denominator),
    }


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    contract = args.contract.resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite output directory")
    require(sha256(contract) == EXPECTED_CONTRACT, "M911 contract drift")
    contract_data = load_json(contract)
    require(
        contract_data["status"] == "AUTHORIZED_CPU_ONLY_ADDITIVE_PREMODEL",
        "M911 contract is not active",
    )

    input_audit = {}
    loaded = {}
    for name, entry in contract_data["frozen_inputs"].items():
        path = root / entry["path"]
        require(path.is_file() and not path.is_symlink(),
                "missing or symlink frozen input: {}".format(name))
        observed = sha256(path)
        require(observed == entry["sha256"],
                "frozen input drift: {}".format(name))
        input_audit[name] = {
            "path": entry["path"],
            "sha256": observed,
            "regular_nonsymlink": True,
        }
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    require(input_audit["docs359"]["sha256"] == EXPECTED_DOCS359,
            "docs359 drift")
    review_entry = contract_data["frozen_inputs"]["m903_m803_dc_result_review"]
    review_seal = root / (
        "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_"
        "20260829/SHA256SUMS.seal.sha256"
    )
    require(sha256(review_seal) == review_entry["outer_seal_file_sha256"],
            "M903 outer seal drift")

    m218 = loaded["m218_h67_service_premodel"]
    m216 = loaded["m216_h67_frontend_replay"]
    m903 = loaded["m903_m803_dc_result_review"]
    require(m218["status"] ==
            "PASS_FROZEN_H67_TAGGED_SLICE_SERVICE_PREMODEL_GO",
            "M218 premodel not admitted")
    require(m216["status"] ==
            "PASS_EXACT_FROZEN_H67_SCOPE_MATCHED_K1_K8_FRONTEND_REPLAY",
            "M216 replay not admitted")
    require(m903["status"] ==
            "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED",
            "M903 result not admitted")

    for key, expected in EXPECTED_WORK.items():
        require(int(m218["aggregate_work"][key]) == expected,
                "M218 work identity drift: {}".format(key))
    require(int(m218["aggregate_work"]["k8_group_commands"]) * 6 ==
            440284872, "six-slice request identity drift")
    for key in ("records", "tokens", "events", "raw96_beats",
                "nonzero96_descriptors"):
        require(int(m216["aggregate"][key]) == EXPECTED_WORK[key],
                "M216/M218 identity mismatch: {}".format(key))

    k8_area = float(m903["dc_evidence"]["axes"]["k8"]["area_um2"])
    k1x8_area = float(m903["dc_evidence"]["axes"]["k1x8"]["area_um2"])
    require(k8_area == 131086.241193, "M903 K8 area drift")
    require(k1x8_area == 585479.153645, "M903 K1x8 area drift")
    area_ratio = k1x8_area / k8_area
    area_saving_percent = (1.0 - k8_area / k1x8_area) * 100.0

    points = {}
    l1_cycles = EXPECTED_LATENCY_CYCLES[1]
    for latency in (1, 2, 4):
        name = "L{}_O8_II1".format(latency)
        producer = m218["points"][name]
        k8_cycles = int(producer["service_k8_cycles"])
        require(k8_cycles == EXPECTED_LATENCY_CYCLES[latency],
                "M218 latency point drift: {}".format(name))
        require(int(producer["k8_context_hazard_cycles"]) == 0,
                "unexpected K8 context hazard: {}".format(name))
        # The fair memory-only envelope exposes the same eight scalar banks,
        # one request/bank/cycle, O8 and II1 to both architectures.  K1x8's
        # critical lane is the same max-per-bank schedule used to form every
        # M216 SOURCE_CAP=8 group.  Therefore this model intentionally assigns
        # the same service span; any controller/join delta is left unclaimed.
        k1x8_cycles = k8_cycles
        points[str(latency)] = {
            "latency_cycles": latency,
            "outstanding_per_bank": 8,
            "initiation_interval_cycles": 1,
            "k8_memory_service_cycles": k8_cycles,
            "k1x8_memory_service_cycles": k1x8_cycles,
            "equal_bandwidth_cycle_speedup_k8_vs_k1x8": ratio(
                k1x8_cycles, k8_cycles),
            "k8_throughput_retention_vs_l1": ratio(l1_cycles, k8_cycles),
            "k1x8_throughput_retention_vs_l1": ratio(
                l1_cycles, k1x8_cycles),
            "equal_bandwidth_throughput_per_logic_area_ratio_k8_vs_k1x8":
                area_ratio,
            "controller_delta_modeled": False,
        }

    directed = m903["fair_equal_bandwidth_metrics"]
    require(directed["aggregate_sum_cycles"] == {"k1x8": 1945, "k8": 1913},
            "M903 directed cycle drift")
    require(abs(float(directed[
        "aggregate_equal_bandwidth_cycle_speedup_k8_vs_k1x8"])
        - 1.0167276529012024) < 1e-15, "M903 directed ratio drift")

    stage_rows = {}
    for stage in ("0", "1", "2", "3"):
        row = m218["per_stage"][stage]
        stage_rows[stage] = {
            "records": int(row["records"]),
            "tokens": int(row["tokens"]),
            "events": int(row["events"]),
            "output_blocks": int(row["output_blocks"]),
            "k8_group_commands": int(row["k8_group_commands"]),
            "pinned_l1_o8_ii1_k8_service_cycles": int(
                row["oracle_service"]["k8_cycles"]),
            "pinned_l4_o8_ii1_k8_service_cycles": int(
                row["primary_service"]["k8_cycles"]),
            "per_stage_k1x8_cycle_claim": False,
        }

    result = {
        "schema": "m911_m803_c2_h67_equalbw_fixed_latency_premodel_v1",
        "status": "PASS_H67_EQUAL_BANDWIDTH_FIXED_LATENCY_COMPONENT_PREMODEL",
        "milestone": "M911",
        "date": "2026-08-29",
        "identity": {
            "analyzer_start_sha256": sha256(Path(__file__).resolve()),
            "contract_sha256": EXPECTED_CONTRACT,
            "docs359_sha256": EXPECTED_DOCS359,
            "inputs": input_audit,
            "m903_outer_seal_file_sha256": sha256(review_seal),
        },
        "population": {
            "checkpoint": "H67 ep35 frozen M216/M218 identity",
            "operator": "mlp.fc2",
            **EXPECTED_WORK,
            "six_slice_weight_requests": 440284872,
            "payload_verification":
                "inherited from pinned M218: all 120 payload SHA/size/popcount checked",
        },
        "fair_resource_boundary": {
            "comparison": "typed signed K8 versus equal-bandwidth K1x8",
            "physical_weight_banks_each": 8,
            "bank_word_bits": 128,
            "accept_rate_each": "one request per bank per cycle",
            "outstanding_per_bank": 8,
            "initiation_interval_cycles": 1,
            "fixed_latency_cycles": [1, 2, 4],
            "external_stalls": False,
            "area_source":
                "M903 Synopsys DC 3ns ideal-clock ZeroWireload logic-only",
            "k8_logic_area_um2": k8_area,
            "k1x8_logic_area_um2": k1x8_area,
            "k8_logic_area_saving_vs_k1x8_percent": area_saving_percent,
        },
        "latency_points": points,
        "per_stage_pinned_sensitivity": stage_rows,
        "directed_vcs_crosscheck_separate_scope": {
            "source":
                "M903 frozen five-case directed M803/K1x8 component VCS",
            "latency_of_scalar_memory_model": 4,
            "k8_cycles": 1913,
            "k1x8_cycles": 1945,
            "k8_speedup_vs_k1x8": 1.0167276529012024,
            "throughput_per_logic_area_ratio": 4.541077997893274,
            "not_extrapolated_to_120_record_points": True,
        },
        "direct_full_trace_replay_audit": {
            "available": False,
            "raw_h67_payload_available": True,
            "exact_blockers": [
                "No frozen ordered per-token header/raw4/end transaction stream exists for direct M803 and K1x8 endpoint consumption.",
                "No cycle-calibrated CPU recurrence models the complete M803 adapter, M218 service, eight scalar fixed-latency banks, K1x8 eight services, result join, and token-done fences on that stream.",
                "The existing M218 producer materializes exact work and fixed-latency service spans but not M803-vs-K1x8 controller/join deltas on every token."
            ],
            "minimum_next_capture": {
                "gpu_or_checkpoint_needed": False,
                "source": "reuse the existing frozen M216 manifest and bitpacks",
                "required_local_artifact":
                    "one SHA-sealed raw4 stream containing header, ordered raw beats, raw_last and expected token boundary for all 120 FC2 records",
                "required_driver":
                    "a bounded file-driven M803/K1x8 replay or a VCS-calibrated Python recurrence with configurable scalar-bank LATENCY=1/2/4",
                "command_template_not_yet_executable":
                    "python3 system_simulator/scripts/pack_m911_h67_fc2_raw4_stream.py --manifest <frozen_manifest> --payload-root <frozen_payload_root> --output <new_nonexistent_stream_dir>"
            }
        },
        "interpretation": {
            "cycle":
                "The fair eight-bank memory-service envelope is 1.000x at L1/L2/L4; latency degrades both equally and only 2.16% from L1 to L4.",
            "physical_efficiency":
                "At equal premodel service, the admitted logic-only areas imply 4.466x throughput per logic area and 77.61% logic-area saving for shared-state K8 versus replicated K1x8.",
            "novelty_role":
                "support C2 shared-state efficiency; do not sell equal-bandwidth cycle acceleration."
        },
        "claim_boundary": {
            "exact_h67_work_identity": True,
            "fixed_latency_memory_service_premodel": True,
            "component_supporting_evidence": True,
            "direct_cycle_accurate_m803_full_trace": False,
            "rtl_measured_full_trace": False,
            "complete_fc2": False,
            "complete_ffn": False,
            "physical_sram_macro": False,
            "power": False,
            "energy": False,
            "ppa": False,
            "system_speedup": False,
            "headline": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "m911_m803_c2_h67_equalbw_fixed_latency_premodel_r1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    require(sha256(Path(__file__).resolve()) ==
            result["identity"]["analyzer_start_sha256"],
            "analyzer mutated during execution")
    print(json.dumps({
        "status": result["status"],
        "latency_points": {key: value["k8_memory_service_cycles"]
                           for key, value in points.items()},
        "equalbw_cycle_speedup": 1.0,
        "throughput_per_logic_area_ratio": area_ratio,
        "direct_full_trace_replay": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
