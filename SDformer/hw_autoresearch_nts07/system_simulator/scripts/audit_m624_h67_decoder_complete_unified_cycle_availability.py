#!/usr/bin/env python3
"""M624 fail-closed availability audit for a decoder-complete H67 simulator.

This script never fabricates cycles.  It verifies the frozen evidence already on
disk, profiles the exact population and reports either READY_TO_SIMULATE or the
minimum missing capture/transfer/software gates.  It performs no GPU, EDA or
remote work.
"""

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "m510": ("hw_autoresearch_nts07/results/m510_h67_convtranspose_coverage_gap_audit_r2_20260827/m510_h67_convtranspose_coverage_gap_audit.json", "20c45030e9171ba241076f0bded3ec762db2eb60a2be20f7b596172d63d0b681"),
    "m511_contract": ("hw_autoresearch_nts07/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json", "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e"),
    "m520_registry": ("hw_autoresearch_nts07/system_simulator/config/m520_h67_paper_metric_registry_v1_20260827.json", "9d5878c5317a2734dfb2685d9f824f693d807ded351b053b014df063a08fefd7"),
    "m522_mapper_dc": ("hw_autoresearch_nts07/dc_handoff/runs/m522_m514_c2d_logic_only_dc_3p000ns_r4_20260827/m522_m514_c2d_logic_only_dc_receipt_r4.json", "f86d016906839e1c1d9ba31fbc9fc392d36497e2e27e3190cd78b8196bec2484"),
    "m523_bundler_vcs": ("hw_autoresearch_nts07/results/m523_c2d_k8_polyphase_tap_bundler_vcs_r2_20260827/m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_v2.json", "aeb99262e85962ba45d77d83c80ba47c13eb66588f9241b6b9a39271f8dfaf7b"),
    "m527_ladder": ("hw_autoresearch_nts07/contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json", "83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b"),
    "m590_source_contract": ("hw_autoresearch_nts07/contracts/m590_m559_pbr4_pre_rtl_cpu_runner_source_contract_r3_20260828.json", "0b70d57ca54169ed7cd0661f6ca7902e9db5e853b3b0d2d89406ad99699b08a3"),
    "m596_m590_review": ("hw_autoresearch_nts07/reviews/m596_m590_m559_pbr4_pre_rtl_cpu_runner_static_hammer_r1_20260828/review.json", "e5587e895fa399f2107aaa57d5e51c0088ac29776a244288d6c43d35b87a0ae9"),
    "ordered_trace": ("hw_autoresearch_nts07/results/h67_ep35_full_network_ordered_trace_s10_20260821/execution_trace.csv", "ad8d1f286c0936ce7cf42324068cfd074aeef3cf77af62890e0598b663b91bfd"),
    "dual_line_trace": ("hw_autoresearch_nts07/results/h67_ep35_full_network_ordered_trace_s10_20260821/dual_line_operator_trace.csv", "2390dc3ee5f093a2c760cd53d7b9587f874767b78073da8b99f3a88b5079bd1c"),
    "m51_manifest": ("hw_autoresearch_nts07/system_handoff/incoming/m51_capture_bundle_r2_20260823/manifest.json", "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"),
    "m51_validation": ("hw_autoresearch_nts07/results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/m51_h67_ep35_binary_input_trace_gpu_payload_validation_receipt_r1.json", "d37e26a9e3206229746eb21209603376a4c07c3aa69f7500d0b960f64c580c32"),
    "m216_fc2_admission": ("hw_autoresearch_nts07/results/m216_scope_matched_sparse_frontend_admission_r1_20260825/m216_scope_matched_sparse_frontend_admission_r1.json", "8059a908cb47534995d928bfc95893da93e979d82a333cbc5199ab8f53a34894"),
    "m518_fixed_vcs": ("hw_autoresearch_nts07/results/m518_matched_fixed_t10_atlif_vcs_r11_exact_20260827/m518_matched_fixed_t10_atlif_author_vcs_receipt_r11.json", "3ee16477d160d333fc72df234b2b1243330066d11a6e512a4753b0113afa9df9"),
    "m519_k1_k1x8_vcs": ("hw_autoresearch_nts07/results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827/m519_fc2_registered_release_vcs_receipt_r2.json", "7228d99fc3384fc2ee77e6fddbd1ca7e0df88870847c8a1c3525583df66627a8"),
    "m528_c1_result": ("hw_autoresearch_nts07/results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json", "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1"),
    "m528_c1_review": ("hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827/review.json", "4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e"),
    "m22_partial_traffic": ("hw_autoresearch_nts07/results/m22_ordered_system_transactions_s10_r2_binaryclosed2_20260822/m22_summary.json", "c4aa0dd1eb5f452294454c2978fc536cdd530bf7df7b9c38d32d416b2f5ed2df"),
    "m23_memory_envelope": ("hw_autoresearch_nts07/results/m23_physical_memory_r5_rowclosed_20260822/m23_summary.json", "40684db2497f08988539738360cfd419ee67541a27795af04a36084c68925893"),
    "docs359": ("hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md", "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"),
}

OPTIONAL_RUNTIME = {
    "m511_decoder_inputs": "hw_autoresearch_nts07/system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827",
    "m511_decoder_verification": "hw_autoresearch_nts07/results/m511_h67_ep35_convtranspose_payload_verify_r1_20260827",
    "m578_decoder_weights": "hw_autoresearch_nts07/system_handoff/outgoing/m578_h67_ep35_decoder_signed_int8_weights_r2_20260828",
    "m590_decoder_cpu_result": "hw_autoresearch_nts07/results/m590_m559_pbr4_pre_rtl_cpu_r6_20260828",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_markdown(path, result):
    matrix = result["configuration_availability"]
    lines = [
        "# M624 H67 ep35 decoder-complete unified cycle simulator availability",
        "",
        "## Verdict",
        "",
        "`{}`. No decoder-complete cycles, traffic, stall, Fixed numerator, speedup or headline were generated.".format(result["status"]),
        "",
        "The frozen ordered trace is useful but partial: {} rows = {} operator + {} ATLIF + {} attention rows across {} samples, and it contains zero ConvTranspose rows. M51 declares {} binary records but only {} payloads are locally present; {} payloads ({} bytes) are missing.".format(
            result["ordered_trace"]["rows"], result["ordered_trace"]["operator_rows"],
            result["ordered_trace"]["atlif_rows"], result["ordered_trace"]["attention_rows"],
            result["ordered_trace"]["samples"], result["m51_local_payload"]["manifest_records"],
            result["m51_local_payload"]["present_records"], result["m51_local_payload"]["missing_records"],
            result["m51_local_payload"]["missing_bytes"]),
        "",
        "## Configuration matrix",
        "",
        "| Row | Current executable path | Blocking gate | Unified metrics |",
        "|---|---|---|---|",
    ]
    for row in matrix:
        lines.append("| {} | {} | {} | null |".format(
            row["configuration_id"], "<br>".join(row["current_paths"]),
            "<br>".join(row["blockers"])))
    lines.extend([
        "",
        "## Minimum data handoff",
        "",
        "| Item | Action | Exact population |",
        "|---|---|---|",
    ])
    for item in result["minimum_capture_or_transfer_request"]:
        lines.append("| {} | {} | {} |".format(item["id"], item["action"], item["population"]))
    lines.extend([
        "",
        "M511 is marked pending a superseding local launch review; this audit does not authorize or run it. M590 r6 remains unusable because M596 found P0=3/P1=2 and forbids execution.",
        "",
        "All M510 decoder ranges remain analytic projections only. M522/M523 prove mapper/bundler support, not decoder cycles. M22/M23 remain partial logical/envelope ledgers, not executable total cycles.",
        "",
        "`docs/359` remains `{}`.".format(result["docs359_sha256"]),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[3]
    identities = {}
    for key, (relative, expected) in EXPECTED.items():
        path = repo / relative
        if not path.is_file():
            raise SystemExit("missing required input: " + relative)
        actual = sha256(path)
        if actual != expected:
            raise SystemExit("required input SHA drift: {} {}".format(relative, actual))
        identities[key] = {"path": relative, "sha256": actual, "bytes": path.stat().st_size}

    contract_path = Path(args.contract).resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != "m624_h67_decoder_complete_unified_cycle_availability_contract_v1":
        raise SystemExit("contract schema mismatch")

    ordered = csv_rows(repo / EXPECTED["ordered_trace"][0])
    dual = csv_rows(repo / EXPECTED["dual_line_trace"][0])
    kinds = Counter(row["kind"] for row in ordered)
    operators = Counter(row["operator"] for row in ordered if row["kind"] == "operator")
    samples = sorted(set(int(row["sample_id"]) for row in ordered))
    op_names = set(row["name"] for row in ordered if row["kind"] == "operator")
    decoder_rows = [row for row in ordered if row.get("operator") == "ConvTranspose2d"]

    m51_path = repo / EXPECTED["m51_manifest"][0]
    m51 = load_json(m51_path)
    m51_root = m51_path.parent
    present = []
    missing = []
    present_hash_mismatches = []
    for row in m51["records"]:
        payload = m51_root / row["relative_path"]
        if payload.is_file():
            present.append(row)
            if payload.stat().st_size != int(row["packed_bytes"]) or sha256(payload) != row["file_sha256"]:
                present_hash_mismatches.append(row["relative_path"])
        else:
            missing.append(row)
    if present_hash_mismatches:
        raise SystemExit("present M51 payload identity mismatch")

    m510 = load_json(repo / EXPECTED["m510"][0])
    m520 = load_json(repo / EXPECTED["m520_registry"][0])
    m527 = load_json(repo / EXPECTED["m527_ladder"][0])
    m596 = load_json(repo / EXPECTED["m596_m590_review"][0])
    optional = {key: {"path": rel, "present": (repo / rel).is_dir()}
                for key, rel in OPTIONAL_RUNTIME.items()}

    decoder_ready = (optional["m511_decoder_inputs"]["present"] and
                     optional["m511_decoder_verification"]["present"] and
                     optional["m578_decoder_weights"]["present"])
    m590_safe = bool(m596.get("decision", {}).get("formal_cpu_execution_allowed", False))
    m51_complete = len(missing) == 0
    registry = m527["configuration_registry"]
    registry_ready = bool(registry["admission_gate"]["current_value"])
    numerator_ready = bool(m527["fixed_throughput_numerators"]["admission_gate"]["current_value"])
    complete_order = len(decoder_rows) == 40
    global_ready = all((decoder_ready, m590_safe, m51_complete, registry_ready,
                        numerator_ready, complete_order))

    common = [
        "10-sample ordered trace: shapes/order for 79 Conv2d/Linear modules plus ATLIF/attention",
        "M22 logical traffic and M23 bank-port envelope are partial inventory only",
    ]
    configurations = [
        {
            "configuration_id": "B0_Dense96_Fixed_T10",
            "status": "BLOCKED_PARTIAL_EXECUTABLE_COMPONENTS",
            "current_paths": common + ["M518 Fixed-T10 directed VCS component"],
            "blockers": ["zero ConvTranspose rows/bitpacks", "no complete operator-scope/fixed numerator", "no common compute+memory completion schedule"],
        },
        {
            "configuration_id": "B1_PTB_like_structured_K1x8",
            "status": "BLOCKED_DEFINITION_ONLY",
            "current_paths": ["M527 project-defined PTB-like semantics only"],
            "blockers": ["no executable configuration manifest", "no per-group full-population scan ledger", "no charged dense fallback/decoder schedule"],
        },
        {
            "configuration_id": "B2_exact_bit_sparse_K1",
            "status": "BLOCKED_COMPONENT_AND_PAYLOAD_GAPS",
            "current_paths": ["M216 FC2 aggregate K1 CPU component", "M519 directed K1 VCS component", "M51 exact-binary manifest"],
            "blockers": ["{} of 310 M51 payloads locally absent".format(len(missing)), "decoder bitpacks absent", "no all-operator K1/common-memory schedule"],
        },
        {
            "configuration_id": "B3_exact_bit_sparse_K1x8",
            "status": "BLOCKED_COMPONENT_AND_PAYLOAD_GAPS",
            "current_paths": ["M519 directed K1x8 VCS component", "M51 exact-binary manifest"],
            "blockers": ["{} of 310 M51 payloads locally absent".format(len(missing)), "decoder bitpacks absent", "no all-operator replicated-state/control/resource manifest"],
        },
        {
            "configuration_id": "Ours_C1_C2_C3_exact",
            "status": "BLOCKED_NO_COEXECUTABLE_CONFIGURATION",
            "current_paths": ["C1 M528 exact CPU same-ledger four-bottleneck-Conv candidate", "C2 M216 FC2 + M522 decoder mapper + M523 tap bundler", "C3 M518 Fixed-T10 directed VCS component"],
            "blockers": ["components are disjoint and cannot be summed/multiplied", "M590 r6 failed static review P0=3/P1=2", "decoder inputs/weights/result absent", "no non-overlap shared SRAM/DRAM schedule"],
        },
    ]

    missing_by_operator = Counter(row["operator"] for row in missing)
    request = [
        {
            "id": "R1_M511_DECODER_INPUTS",
            "action": "Run the superseding independently reviewed local M511 capture once; then run the sealed payload verifier.",
            "population": "40 records = 10 samples x 4 ConvTranspose2d; 87,030,000 packed bytes",
            "status": "PENDING_NEW_LOCAL_LAUNCH_REVIEW__NOT_AUTHORIZED_BY_M624",
        },
        {
            "id": "R2_M578_DECODER_WEIGHTS",
            "action": "Export and seal four checkpoint-bound signed-INT8 COUT_CIN_KY_KX tensors; no synthetic weights.",
            "population": "4 tensors; 7,140,096 int8 bytes; shapes 384x1536x3x3, 192x770x3x3, 96x386x3x3, 96x194x3x3",
            "status": "ABSENT",
        },
        {
            "id": "R3_M51_MISSING_PAYLOAD_TRANSFER",
            "action": "Transfer only manifest-listed missing members and verify each existing SHA; do not recapture or regenerate.",
            "population": "{} records; {} bytes; operator counts {}".format(len(missing), sum(int(row["packed_bytes"]) for row in missing), dict(missing_by_operator)),
            "status": "ABSENT_LOCALLY__SOURCE_GPU_VALIDATION_ALREADY_EXISTS",
        },
        {
            "id": "R4_DECODER_ORDER_EXTENSION",
            "action": "Capture/seal global execution ordinal for each decoder call or emit a new complete ordered trace; module-local order alone is insufficient for a unified schedule.",
            "population": "40 metadata rows = 10 samples x 4 decoder calls",
            "status": "ABSENT",
        },
        {
            "id": "R5_OPERATOR_SCOPE_AND_FIXED_NUMERATOR",
            "action": "Create a complete operator-scope manifest and M527 fixed-numerator receipt, explicitly charging normalization/state/update/control/fallback work.",
            "population": "one frozen 10-sample population; included/excluded partition must be exhaustive",
            "status": "ABSENT__LOCAL_AUTHORING_AFTER_R1_R4",
        },
        {
            "id": "R6_SAFE_UNIFIED_CPU_SOURCE",
            "action": "Repair/supersede M590 r6 or implement a new common scheduler; require fresh static hammer before any production CPU run.",
            "population": "B0/B1/B2/B3/Ours in one 96-lane, 240-KiB, 192-byte-per-3ns-cycle resource schema",
            "status": "M590_R6_FORBIDDEN_BY_M596",
        },
    ]

    null_metrics = {
        "per_operator_sample_cycles": None,
        "per_operator_sample_sram_read_bytes": None,
        "per_operator_sample_sram_write_bytes": None,
        "per_operator_sample_dram_read_bytes": None,
        "per_operator_sample_dram_write_bytes": None,
        "per_operator_sample_stall_breakdown": None,
        "per_sample_total_cycles": None,
        "overall_total_cycles": None,
        "fixed_dense_equivalent_ops_numerator": None,
        "speedups": None,
    }
    result = {
        "schema": "m624_h67_decoder_complete_unified_cycle_availability_v1",
        "date": "2026-08-28",
        "status": "READY_TO_SIMULATE" if global_ready else "FAIL_CLOSED_INPUT_AND_EXECUTABLE_SCHEMA_GAPS__NO_CYCLE_RESULT",
        "identity": {
            "contract": {"path": str(contract_path.relative_to(repo)), "sha256": sha256(contract_path)},
            "inputs": identities,
        },
        "ordered_trace": {
            "rows": len(ordered), "samples": len(samples), "sample_ids": samples,
            "operator_rows": kinds["operator"], "atlif_rows": kinds["atlif"],
            "attention_rows": kinds["attention"], "dual_line_rows": len(dual),
            "operator_modules": len(op_names), "operator_type_rows": dict(operators),
            "convtranspose_rows": len(decoder_rows),
            "scope_complete": False,
            "reason": "Profiler hook universe excludes four ConvTranspose2d and does not prove exhaustive normalization/state/control operator coverage.",
        },
        "m51_local_payload": {
            "manifest_records": len(m51["records"]),
            "present_records": len(present), "missing_records": len(missing),
            "present_bytes": sum(int(row["packed_bytes"]) for row in present),
            "missing_bytes": sum(int(row["packed_bytes"]) for row in missing),
            "present_hash_or_size_mismatches": len(present_hash_mismatches),
            "present_modules": len(set(row["name"] for row in present)),
            "missing_modules": len(set(row["name"] for row in missing)),
            "missing_operator_counts": dict(missing_by_operator),
        },
        "decoder": {
            "m510_analytic_only": True,
            "m510_corrected_envelope_range": [m510["analytic_bounds"]["corrected_envelope_lower"], m510["analytic_bounds"]["corrected_envelope_upper"]],
            "m510_decoder_share_range": [m510["analytic_bounds"]["decoder_share_lower"], m510["analytic_bounds"]["decoder_share_upper"]],
            "m510_projection_promoted_to_result": False,
            "expected_records": 40, "expected_packed_bytes": 87030000,
            "optional_runtime_artifacts": optional,
            "m522_mapper_support_only": True,
            "m523_bundler_support_only": True,
        },
        "configuration_availability": configurations,
        "global_gates": {
            "decoder_inputs_weights_verified": decoder_ready,
            "complete_m51_payload_local": m51_complete,
            "complete_ordered_trace_with_40_decoder_rows": complete_order,
            "safe_decoder_cpu_source": m590_safe,
            "m527_configuration_registry_ready": registry_ready,
            "m527_fixed_numerator_ready": numerator_ready,
            "m520_system_speedup_generated": bool(m520["system_speedup_generated"]),
            "all_ready": global_ready,
        },
        "metrics": null_metrics,
        "minimum_capture_or_transfer_request": request,
        "claim_boundary": {
            "availability_audit": True,
            "cpu_exact_simulator_executed": False,
            "analysis_projection_used_as_result": False,
            "decoder_complete": False,
            "full_network": False,
            "cycles": False,
            "traffic_result": False,
            "stall_result": False,
            "fixed_numerator": False,
            "speedup": False,
            "energy": False,
            "ppa": False,
            "headline": False,
        },
        "execution_receipt": {"cpu_audit_runs": 1, "cpu_simulator_runs": 0, "gpu_runs": 0, "eda_runs": 0, "remote_runs": 0},
        "docs359_sha256": identities["docs359"]["sha256"],
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=False)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_md, result)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
