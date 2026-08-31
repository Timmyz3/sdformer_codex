#!/usr/bin/env python3
"""Deterministic, read-only recomputation for the M711 fresh hammer."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SOURCES = {
    "m709_readme": (
        "hw_autoresearch_nts07/reviews/"
        "m709_first_principles_hardware_innovation_audit_r1_20260828/README.md",
        "e61c01d256e6bf07329407603d17f2d4f36c88e00149908de2c49e109352da71",
    ),
    "m709_review": (
        "hw_autoresearch_nts07/reviews/"
        "m709_first_principles_hardware_innovation_audit_r1_20260828/review.json",
        "6ccdd6eedf1764211bbfdce8b6455a589343107eb830ab08c178eb0aeff6eac6",
    ),
    "m709_manifest": (
        "hw_autoresearch_nts07/reviews/"
        "m709_first_principles_hardware_innovation_audit_r1_20260828/SHA256SUMS",
        "9f417950f6764ef57082a68ffd3e28c6688956f174d07b898f1cc34087d1a931",
    ),
    "m709_outer_seal": (
        "hw_autoresearch_nts07/reviews/"
        "m709_first_principles_hardware_innovation_audit_r1_20260828/SHA256SUMS.seal.sha256",
        "351d2af227f73027c0781e8a930a8176733a4af9cab0a6970d84d5b129690a76",
    ),
    "m596_review": (
        "hw_autoresearch_nts07/reviews/"
        "m596_m590_m559_pbr4_pre_rtl_cpu_runner_static_hammer_r1_20260828/review.json",
        "e5587e895fa399f2107aaa57d5e51c0088ac29776a244288d6c43d35b87a0ae9",
    ),
    "m523_review": (
        "hw_autoresearch_nts07/reviews/"
        "m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_blind_hammer_r1_20260827/"
        "m523_c2d_k8_polyphase_tap_bundler_vcs_receipt_blind_hammer_r1_20260827.json",
        "fa7aab5a182c3e74999cf3d9fdbd69249f18fe6fcc29c6f36d9fa9e4a0d2f515",
    ),
    "m528_review": (
        "hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827/review.json",
        "4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e",
    ),
    "m519_review": (
        "hw_autoresearch_nts07/reviews/"
        "m519_registered_release_vcs_receipt_hammer_r2_20260827/"
        "m519_registered_release_vcs_receipt_hammer_verdict_r2.json",
        "be73fa85d5d0f3556974526578d79f53a05d1bda3bf5a4d4eb7740843d4d3480",
    ),
    "m518_review": (
        "hw_autoresearch_nts07/reviews/"
        "m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_r1_20260827/"
        "m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_r1.json",
        "513c5d916859b0f48b9ffeced6853ad89a8ace5ea6a9b264baf05d1ed1966665",
    ),
    "m480_review": (
        "hw_autoresearch_nts07/reviews/m480_independent_hammer_r1_20260826/"
        "m480_independent_hammer_receipt_r1.json",
        "0a8a7db2d017f66735516ff051dbef94c5e0ba32a766cee682b8a029121a10b9",
    ),
    "m502_review": (
        "hw_autoresearch_nts07/reviews/m502_bn_bittight_preflight_r1_20260827/"
        "m502_bn_bittight_preflight_r1.json",
        "36dce2bc1e8de957587634173ad1225d809caff97e5ffd7162b052865e4c8fcb",
    ),
    "m481_result": (
        "hw_autoresearch_nts07/results/"
        "m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826/"
        "payload/m481_fc1_fullwidth_resource_matched_context_factorized_dse_r2.json",
        "2a7a1c917cb2f9aa1adb61092c7619de8d9b495aab5550f1fa41291188006578",
    ),
    "m534_screen": (
        "hw_autoresearch_nts07/reviews/m534_next_rtl_candidate_screen_r4_20260827/"
        "m534_next_rtl_candidate_screen_r4.json",
        "a1594d8c92778269a4223bb900e5b73e7068b4db81c9a21e5a07a44297b4074b",
    ),
    "m552_contract": (
        "hw_autoresearch_nts07/contracts/"
        "m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r3_20260827.json",
        "16119c935cd4357da477fee7b0416dcbb38a3c467a7d95c9e8b3b7487f5ebb57",
    ),
    "m559_contract": (
        "hw_autoresearch_nts07/contracts/"
        "m559_m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r4_20260828.json",
        "6a8a76f8d71188a115a44e9f0a6f0af2be897973d5c8eaa16d62b4e1fffbd56c",
    ),
    "m707_lower_bound": (
        "hw_autoresearch_nts07/reviews/"
        "m707_h67_first_principles_lower_bound_and_idea_screen_r1_20260828/"
        "lower_bound_ledger.json",
        "c61badf4f88670e3e96996c8dbf7bca4b4ea17c5b5037247e842d9c457ff0ebb",
    ),
    "docs359": (
        "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md",
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str):
    return json.loads((ROOT / SOURCES[name][0]).read_text(encoding="utf-8"))


def verify_m709_double_seal() -> bool:
    directory = ROOT / Path(SOURCES["m709_manifest"][0]).parent
    manifest = directory / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, member = line.split(maxsplit=1)
        if sha256(directory / member) != expected:
            return False
    seal_expected, seal_member = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8"
    ).split()
    return seal_member == "SHA256SUMS" and sha256(manifest) == seal_expected


def main() -> None:
    source_checks = {}
    for name, (relative, expected) in SOURCES.items():
        actual = sha256(ROOT / relative)
        source_checks[name] = actual == expected
        if actual != expected:
            raise SystemExit(f"SOURCE_SHA_MISMATCH {name}: {actual} != {expected}")

    m528 = load("m528_review")
    m519 = load("m519_review")
    m518 = load("m518_review")
    m596 = load("m596_review")
    m480 = load("m480_review")
    m481 = load("m481_result")
    lower = load("m707_lower_bound")

    cycles = m528["validated_metrics"]["aggregate_cycles"]
    c1 = {
        "candidate_cycles": cycles["m505_dead_write_only_1rw_cycles"],
        "recomputed_vs_strong_zero": cycles["m468_strong_zero_cycles"]
        / cycles["m505_dead_write_only_1rw_cycles"],
        "recomputed_vs_same_bit": cycles["m473_same_coordinate_bit_cycles"]
        / cycles["m505_dead_write_only_1rw_cycles"],
        "scratch_bytes": m528["validated_metrics"]["capacity"]
        ["m505_dead_write_only_macro_rounded_bytes"],
    }

    active_rows = [r for r in m519["cycle_rows_read_from_logs"] if r["events"]]
    k1_over_k8 = [r["registered_release_k1_cycles"] / r["registered_release_k8_cycles"] for r in active_rows]
    k1x8_over_k8 = [r["registered_release_k1x8_cycles"] / r["registered_release_k8_cycles"] for r in active_rows]
    c2 = {
        "m709_claimed_k1_over_k8_range": [4.89, 6.32],
        "cited_m519_k1_over_k8_range": [min(k1_over_k8), max(k1_over_k8)],
        "cited_m519_k1x8_over_k8_range": [min(k1x8_over_k8), max(k1x8_over_k8)],
        "claimed_range_reproduces": [4.89, 6.32] == [min(k1_over_k8), max(k1_over_k8)],
        "m519_p1_count": m519["p1_count"],
    }

    fixed_issue = m518["cycle_anchors"]["issue_cycles_per_tile"]
    fixed_n1 = m518["cycle_anchors"]["clean_cycles_N1"]
    fixed_n4 = m518["cycle_anchors"]["clean_cycles_N4"]
    fixed_nonissue_n1 = fixed_n1 - fixed_issue
    fixed_nonissue_n4 = fixed_n4 - 4 * fixed_issue
    logical_table_bits = 2 * 32 * 10 * 11
    physical_unique_bytes = 2 * 32 * 10 * 16 // 8
    active_replicated_bytes = 16 * physical_unique_bytes
    acc24_bytes = 160 * 24 // 8
    tda = {
        "logical_table_bits": logical_table_bits,
        "logical_table_bytes": logical_table_bits // 8,
        "physical_unique_table_bytes_16bit_slot": physical_unique_bytes,
        "active_16lane_replicated_table_bytes": active_replicated_bytes,
        "active_acc24_bytes": acc24_bytes,
        "active_table_plus_acc_bytes": active_replicated_bytes + acc24_bytes,
        "headroom_to_m709_24k_gate_bytes": 24576 - active_replicated_bytes - acc24_bytes,
        "all_45_unique_unreplicated_bytes": 45 * physical_unique_bytes,
        "all_45_16lane_replicated_bytes": 45 * active_replicated_bytes,
        "fixed_issue_cycles": fixed_issue,
        "fixed_clean_cycles_n1_n4": [fixed_n1, fixed_n4],
        "nonissue_cycles_n1_n4": [fixed_nonissue_n1, fixed_nonissue_n4],
        "issue_only_ratio_at_m709_gate_10": fixed_issue / 10,
        "full_service_upper_ratio_n1_at_issue10": fixed_n1 / (fixed_nonissue_n1 + 10),
        "full_service_upper_ratio_n4_at_issue10": fixed_n4 / (fixed_nonissue_n4 + 4 * 10),
        "full_service_upper_ratio_n1_at_ideal_issue8": fixed_n1 / (fixed_nonissue_n1 + 8),
        "full_service_upper_ratio_n4_at_ideal_issue8": fixed_n4 / (fixed_nonissue_n4 + 4 * 8),
    }

    fc1_best_projected = min(
        point["scope_corrected_projection"]["all_fc1_projected_cycles"]
        for point in m481["points"]
    )
    fc2 = next(op for op in lower["operators"] if op["operator"] == "fc2")
    fc2_old_cycles = int(fc2["work_share"].split()[0])
    fc2_optimistic = fc2_old_cycles / fc2["legal_denominators"]["single_k1_diagnostic"]["speedup"]
    optimistic_producer_cycles = fc1_best_projected + fc2_optimistic
    raw_write_read_bytes = m480["recomputed"]["reference_q24_bw64_overlap"]["fused_traffic_bytes"]
    speedup_upper = {}
    for bandwidth in (32, 64, 128):
        raw_cycles = raw_write_read_bytes / bandwidth
        speedup_upper[str(bandwidth)] = {
            "raw_write_read_cycles": raw_cycles,
            "optimistic_serial_speedup_upper": (optimistic_producer_cycles + raw_cycles)
            / (2 * optimistic_producer_cycles),
        }
    source_payload = int(
        next(op for op in lower["operators"] if op["operator"] == "fc1")
        ["source_lower_bound"].split("payload is ")[1].split(" bytes")[0]
    )
    q24_peak = m480["recomputed"]["peak_raw_retention_bytes"]["24"]
    rs_bn = {
        "raw_q24_write_read_bytes": raw_write_read_bytes,
        "optimistic_fc1_projected_cycles_unadmitted": fc1_best_projected,
        "optimistic_fc2_cycles_using_unfair_single_k1_ratio": fc2_optimistic,
        "optimistic_total_recompute_producer_cycles": optimistic_producer_cycles,
        "serial_speedup_upper_ignoring_moments_barrier_and_extra_state": speedup_upper,
        "m709_cycle_gate": 1.15,
        "exact_fc1_bitpacked_source_payload_bytes": source_payload,
        "q24_peak_raw_retention_bytes": q24_peak,
        "q24_raw_to_source_payload_ratio_before_descriptors": q24_peak / source_payload,
        "m709_peak_reduction_gate": 8.0,
    }

    decoder = next(op for op in lower["operators"] if op["operator"] == "decoder_convtranspose")
    density = decoder["legal_denominators"]["current_status"]["multisequence_density"]
    pidp = {
        "m596_status": m596["status"],
        "m596_p0_p1_p2": [m596["p0_count"], m596["p1_count"], m596["p2_count"]],
        "decoder_weighted_density": density,
        "dense_pull_probe_to_active_scatter_work_ratio_ignoring_boundaries": 1.0 / density,
        "m523_is_descriptor_only": load("m523_review")["claim_boundary"]["descriptor_only"],
        "m523_decoder_cycle_speedup": load("m523_review")["claim_boundary"]["decoder_cycle_speedup"],
    }

    out = {
        "status": "PASS_RECOMPUTE_M711",
        "all_source_sha_match": all(source_checks.values()),
        "m709_double_seal_pass": verify_m709_double_seal(),
        "docs359_sha256": sha256(ROOT / SOURCES["docs359"][0]),
        "c1": c1,
        "c2": c2,
        "pidp": pidp,
        "tda": tda,
        "rs_bn": rs_bn,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
