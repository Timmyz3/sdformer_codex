#!/usr/bin/env python3
"""Read-only P0 audit of M1156 ordering and the already sealed M712 PIDP ledger."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, getcontext
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

getcontext().prec = 40
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
OUT = HERE / "protocol_order_and_m712_reconciliation.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1156_SOURCE = HW / "system_simulator/scripts/analyze_m1156hc_decoder_hot_psum_cross_layer_dse.py"
MAPPER = HW / "system_simulator/scripts/map_m672_decoder_convtranspose_polyphase_workload_r3.py"
MAPPER_R2 = HW / "system_simulator/scripts/map_m670_decoder_convtranspose_polyphase_workload_r2.py"
M514_CONTRACT = HW / "contracts/m514_c2d_directed_vcs_contract_r1_20260827.json"
M514_RTL = HW / "rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv"
M523_CONTRACT = HW / "contracts/m523_c2d_k8_polyphase_tap_bundler_vcs_contract_draft_r3_20260827.json"
M523_RTL = HW / "rtl_m523/m523_c2d_k8_polyphase_tap_bundler.sv"
M712 = HW / "results/m712_pidp_decoder_exact_cpu_fastkill_r1_20260828"
M718 = HW / "reviews/m718_m712_pidp_decoder_fresh_result_hammer_r1_20260828"

EXPECTED = {
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1156_SOURCE: "3816aadf9770ffe59a807a8f25ad1968d8e93db5ec312cd69ee4426443e0f491",
    MAPPER: "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
    MAPPER_R2: "875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b",
    M514_CONTRACT: "60e4fe5921a374f399bef82fd1902718428bb8f9d6f3d86dc5d03bda7953ab5b",
    M514_RTL: "90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a",
    M523_CONTRACT: "6dac33f9fe035c0ed1c14ddd7dbc7d9ebfabcdec279cf027ce07cf0774baa415",
    M523_RTL: "ad6def7cd81e5f3cd1570ef23fd062da19ee8b2a35498d6deca1c010522a0920",
}
M712_MANIFEST = "00f042b35b64f242b5c4a19ee24fb36f9b5a8a31999d714919c61c70b727330e"
M712_OUTER = "f15c6d45e41e81b623982deda94c5b52f7213417f639f53d9116c457aca49806"
M718_MANIFEST = "45d4d9acece5ff6945eeb4241df98dd838279e213eb3cce08716f3574c4459b7"
M718_OUTER = "c7fb4716baba1e6d8993eaf349a3dea7c2c09b891e927738ac323423dbb4e6bf"
M1156_TERMS = {"D0": 29_622_568, "D1": 30_338_394,
               "D2": 30_328_495, "D3": 96_760_057}
M1156_UPDATES = {"D0": 4_465_036, "D1": 4_647_272,
                 "D2": 5_087_981, "D3": 17_288_869}
M1156_BASE = {"D0": 17_863_747, "D1": 18_592_651,
              "D2": 20_355_467, "D3": 69_162_219}
M1156_CAND = {"D0": 9_025_999, "D1": 9_486_475,
              "D2": 10_559_856, "D3": 36_113_672}
GEOMETRY = {
    "D0": (1536, 15, 20, 384),
    "D1": (770, 30, 40, 192),
    "D2": (386, 60, 80, 96),
    "D3": (194, 120, 160, 96),
}
checks = 0


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON token: " + token)))


def verify_file(path: Path, expected: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def verify_sealed_directory(path: Path, manifest_sha: str, outer_sha: str) -> None:
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    verify_file(manifest, manifest_sha)
    verify_file(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [manifest_sha, "SHA256SUMS"],
            "outer seal mismatch: " + str(path))
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                "manifest row")
        name = fields[1].lstrip("*")
        pure = PurePosixPath(name)
        require(not pure.is_absolute() and ".." not in pure.parts and name not in listed,
                "manifest member")
        listed[name] = fields[0]
    actual = set()
    for member in path.rglob("*"):
        require(not member.is_symlink(), "symlink in sealed directory")
        if member.is_file() and member.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(member.relative_to(path).as_posix())
    require(actual == set(listed), "sealed member set")
    for name, digest in listed.items():
        verify_file(path.joinpath(*PurePosixPath(name).parts), digest)


def ratio(numerator: int, denominator: int) -> str:
    require(denominator > 0, "zero ratio denominator")
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def main() -> int:
    for path, digest in EXPECTED.items():
        verify_file(path, digest)
    verify_sealed_directory(M712, M712_MANIFEST, M712_OUTER)
    verify_sealed_directory(M718, M718_MANIFEST, M718_OUTER)

    mapper_text = MAPPER.read_text(encoding="utf-8")
    mapper_r2_text = MAPPER_R2.read_text(encoding="utf-8")
    m1156_text = M1156_SOURCE.read_text(encoding="utf-8")
    m514_text = M514_RTL.read_text(encoding="utf-8")
    m523_text = M523_RTL.read_text(encoding="utf-8")
    m523_contract = strict_json(M523_CONTRACT)
    require("import map_m670_decoder_convtranspose_polyphase_workload_r2 as R2" in mapper_text and
            "for tile in R2.iter_polyphase_tiles" in mapper_text and
            "for bank in phases:" in mapper_r2_text and
            "for m_start in range(0, plan[\"m\"], tile_m):" in mapper_r2_text and
            "destination_y" in mapper_r2_text and "source_flat_index" in mapper_r2_text,
            "mapper is not destination-major matrix pull")
    require("for tile in mapper.iter_polyphase_tiles" in m1156_text and
            "for local_m, (dy, dx)" in m1156_text and
            m1156_text.index("for tile in mapper.iter_polyphase_tiles") <
            m1156_text.index("for local_m, (dy, dx)"),
            "M1156 destination-major loop binding")
    require("consumes one binary ATLIF source event" in m514_text and
            "removing inter-event bubbles without reordering taps" in m514_text,
            "M514 source-event no-reorder semantics")
    require("Each source event is accepted atomically" in m523_text and
            "event_accept" in m523_text and "fifo_destination_y" in m523_text and
            m523_contract["exact_functional_contract"]["atomic_event_accept"] is True and
            m523_contract["objective"].find("atomically expands each source event") >= 0,
            "M523 source-event atomic expansion semantics")
    require(m523_contract["performance_admission_gate"]["current_m218_c2_integration"] is False and
            m523_contract["evaluation_boundary"]["decoder_cycle_speedup"] is False,
            "M523 performance boundary")

    report712 = strict_json(M712 / "report.json")
    review718 = strict_json(M718 / "review.json")
    require(report712["status"] == "KILL_NO_RTL" and
            review718["decision"]["full_pidp"] == "KILL_NO_RTL" and
            review718["decision"]["allow_rtl_vcs_eda_or_headline"] is False,
            "M712/M718 decision drift")
    aggregate = defaultdict(lambda: defaultdict(int))
    selected_rows = 0
    with (M712 / "rows.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sequence"] != "interlaken_01_a" or row["sequence_sample_id"] != 0:
                continue
            layer = row["module"]
            selected_rows += 1
            fields = aggregate[layer]
            fields["contributors"] += int(row["contributors"])
            fields["a1_cycles"] += int(row["a1_cycles"]["A1-OSG"]["total"])
            fields["pidp_cycles"] += int(row["pidp_cycles"]["total"])
            fields["source_stream"] += int(row["pidp_cycles"]["source_stream"])
            fields["bitmap_probe"] += int(row["pidp_cycles"]["bitmap_probe"])
            fields["groups"] += int(row["optimistic_k8_groups"])
            fields["group_service"] += int(row["pidp_cycles"]["group_service_optimistic"])
            fields["weight_refill"] += int(row["pidp_cycles"]["weight_refill"])
            fields["dense_commit"] += int(row["pidp_cycles"]["dense_output_commit"])
            fields["weight_refs"] += int(row["weight_cache"]["pidp_references"])
            fields["weight_misses"] += int(row["weight_cache"]["pidp_misses"])
            fields["active_weight_identities"] = max(
                fields["active_weight_identities"],
                int(row["weight_cache"]["active_tile_identities"]))
            fields["weight_cache_entries"] = int(row["weight_cache"]["cache_entries"])
    require(selected_rows == 40 and set(aggregate) == set(GEOMETRY), "M712 sample0 population")

    layers = []
    scan_words_total = 0
    for layer, (cin, hin, win, cout) in GEOMETRY.items():
        fields = aggregate[layer]
        require(fields["contributors"] == M1156_TERMS[layer], layer + " contributor bridge")
        scan_words = 10 * hin * win * sum(math.ceil(taps * cin / 96)
                                         for taps in (4, 2, 2, 1))
        scan_words_total += scan_words
        layers.append({
            "layer": layer,
            "same_contributor_multiset_count": fields["contributors"],
            "m1156_destination_major_updates": M1156_UPDATES[layer],
            "m1156_free_order_baseline_cycles": M1156_BASE[layer],
            "m1156_free_order_candidate_cycles": M1156_CAND[layer],
            "m1156_free_order_local_speedup": ratio(M1156_BASE[layer], M1156_CAND[layer]),
            "inverse_scan_96b_words_analytic": scan_words,
            "inverse_scan_96b_bytes_analytic": scan_words * 12,
            "m712_a1_osg_cycles": fields["a1_cycles"],
            "m712_pidp_cycles": fields["pidp_cycles"],
            "m712_a1_over_pidp": ratio(fields["a1_cycles"], fields["pidp_cycles"]),
            "m712_charged_128b_source_stream_cycles": fields["source_stream"],
            "m712_charged_128b_bitmap_probe_cycles": fields["bitmap_probe"],
            "m712_optimistic_groups": fields["groups"],
            "m712_group_service_cycles_10_per_group": fields["group_service"],
            "m712_weight_refill_cycles": fields["weight_refill"],
            "m712_dense_commit_cycles": fields["dense_commit"],
            "m712_weight_references": fields["weight_refs"],
            "m712_weight_misses": fields["weight_misses"],
            "m712_active_weight_identities": fields["active_weight_identities"],
            "m712_weight_cache_entries": fields["weight_cache_entries"],
        })

    m1156_base_sum = sum(M1156_BASE.values())
    m1156_cand_sum = sum(M1156_CAND.values())
    m712_a1_sum = sum(row["m712_a1_osg_cycles"] for row in layers)
    m712_pidp_sum = sum(row["m712_pidp_cycles"] for row in layers)
    selective_sum = sum(row["m712_pidp_cycles"] if row["layer"] == "D3"
                        else row["m712_a1_osg_cycles"] for row in layers)
    require((m1156_base_sum, m1156_cand_sum, ratio(m1156_base_sum, m1156_cand_sum)) ==
            (125_974_084, 65_186_002, "1.932532754501"), "M1156 aggregate")
    require(scan_words_total == 7_488_000, "96-bit inverse scan count")
    require({row["layer"]: row["m712_active_weight_identities"] for row in layers} ==
            {"D0": 384, "D1": 98, "D2": 25, "D3": 13} and
            all(row["m712_weight_cache_entries"] == 16 for row in layers),
            "M712 weight fit identity")
    require(review718["fairness_sensitivities_not_admission"]
            ["joint_a1_128bit_ingress_and_pidp_15cycle_group"]
            ["a1_over_selective"] == "1.214175731477", "M718 joint fairness sensitivity")

    grouped_descriptors = sum(M1156_UPDATES.values())
    grouped_descriptor_bytes = grouped_descriptors * 16
    fixed_bytes = 243_200
    cache_bytes = 290
    slack = 245_760 - fixed_bytes - cache_bytes
    dense_keys_per_timestep = sum((2 * hin) * (2 * win) * math.ceil(cout / 96)
                                  for cin, hin, win, cout in GEOMETRY.values())
    bitmap_bytes_per_all_layer_timestep = math.ceil(dense_keys_per_timestep / 8)
    count16_bytes_per_all_layer_timestep = dense_keys_per_timestep * 2
    require((grouped_descriptors, grouped_descriptor_bytes, slack) ==
            (31_489_158, 503_826_528, 2_270), "reorder lower-bound arithmetic")

    result = {
        "schema": "m1157hc_protocol_order_and_m712_reconciliation_v1",
        "status": "DOWNGRADE_M1156_TO_DESTINATION_MAJOR_FREE_REORDER_UPPER_BOUND__FULL_PIDP_KILL__D3_STATIC_WEIGHT_FIT_CPU_ONLY_REMAINS",
        "checks": checks,
        "ordering_verdict": {
            "m1105_m1156": "destination-major matrix-pull",
            "m514_m523": "source-event-major atomic tap expansion without reorder",
            "direct_executable_bridge_present": False,
            "one_entry_hit_rate_admitted_for_m523_stream": False,
            "reason": "The same destination key is contiguous only after an unmodeled global pull/reorder; M523 emits 4/6/9 destinations per source event and explicitly lacks M218/C2 integration."},
        "layers": layers,
        "reconciliation": {
            "same_contributor_counts_all_four": True,
            "m1156_free_order_ratio": ratio(m1156_base_sum, m1156_cand_sum),
            "m712_charged_full_pidp_ratio_all_four_sample0": ratio(m712_a1_sum, m712_pidp_sum),
            "m712_static_D3_only_selective_ratio_all_four_sample0": ratio(m712_a1_sum, selective_sum),
            "m718_three_sequence_headline_static_selective_ratio": "1.474346419118",
            "m718_joint_fairness_sensitivity_not_admission": "1.214175731477",
            "full_pidp_decision": "KILL_NO_RTL",
            "only_live_branch": "D3 static 13-of-16 weight-fit path under a new CPU contract"},
        "inverse_scan_lower_bound": {
            "analytic_96b_words": scan_words_total,
            "analytic_bytes": scan_words_total * 12,
            "zero_detect_and_K8_compaction_free": False,
            "cycle_result_admitted": False,
            "why_not_new_run": "M712 already charges deterministic destination pull, bitmap probes, optimistic K8 groups, dense commit and a fully associative weight cache; D0-D2 lose from weight refill even in that candidate-favorable coordinate."},
        "global_reorder_lower_bound": {
            "grouped_update_descriptors": grouped_descriptors,
            "bytes_one_materialized_16B_pass": grouped_descriptor_bytes,
            "bytes_write_plus_read": 2 * grouped_descriptor_bytes,
            "dense_destination_output_block_keys_per_timestep_all_layers": dense_keys_per_timestep,
            "one_bit_bitmap_bytes_per_all_layer_timestep": bitmap_bytes_per_all_layer_timestep,
            "uint16_count_or_offset_bytes_per_all_layer_timestep": count16_bytes_per_all_layer_timestep,
            "remaining_240KiB_slack_bytes_after_m1156_cache": slack,
            "catalog_fits_remaining_slack": False},
        "authorization": {
            "run_accumulator_rtl": False,
            "bridge_rtl": False,
            "vcs": False,
            "eda": False,
            "new_D3_static_weight_fit_bridge_inclusive_CPU_contract_only": True},
        "claim_boundary": {
            "m1156_ratio_is_upper_bound_only": True,
            "decoder_complete": False,
            "system_speedup": False,
            "headline": False,
            "paper_ppa_ready": False},
        "identity": {
            "docs359_sha256": EXPECTED[DOCS359],
            "m712_result_outer_file_sha256": M712_OUTER,
            "m718_review_outer_file_sha256": M718_OUTER,
            "m523_contract_sha256": EXPECTED[M523_CONTRACT],
            "m523_rtl_sha256": EXPECTED[M523_RTL]},
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    temporary = OUT.with_name(OUT.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, OUT)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
