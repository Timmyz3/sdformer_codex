#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1119C read-only proof that M1116C metadata cannot yet be physicalized."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
M1116C = HW / "reviews/m1116c_c1_full_storage_dc_ready_closure_readonly_audit_r1_20260830"
M1116C_ID = ("acaa72f8d6766611f234abef1ae7d824f3bab2b80eba4bc7a5bf8f8315e31ea4",
             "679992d84eee986fca063c40cf8b5721874d7189ee12bd3aacba5cdc0018446b",
             "eb8ab19c4c02df0b53315c09889695a33fdcba35eb1cd5b64e3079de89e63df7")
DRAFT = HW / "contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_dc_source_contract_DRAFT_r0_20260830.json"
DRAFT_ID = ("5176d0e297bc29739b2185708272cef842e065e160fb5460646d270dd34dceb7",
            "b09624bcfc1652c73c148678d6baa25bee897ccd7ee9ed22fa2c5c914a394348",
            "56802405b8096776ecb802623a0a757a8a698725ef59afb65f30add973385757")
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = ("475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
            "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
            "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5")
M528 = HW / "results/m528_h67_single_port_same_ledger_recompute_r4_20260827"
M528_ID = ("4556a3383507e81ad9883f59bb55bb3d4fd08e7ec03977b215108b5ce4565073",
           "02abbf7f9209d9a41d803c9942bfb43550be0d40945e3c094c1e457bda0db053")
M528_REVIEW = HW / "reviews/m528_r4_result_hammer_r1_20260827"
M528_REVIEW_ID = ("4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e",
                  "678a0541702b9804691a5700a55fb4dc8c07f524ee5b6176800196371ebe3b56",
                  "ec442c74ca4dee305178e863a97e976940e0f5d6b98a0ad57e52cd298c01653e")
M1102 = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M1102_RESULT = M1102 / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_ID = ("a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
            "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12",
            "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f")
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M935_SHA = "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
OUT = HERE / "gap_analysis.json"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, expected):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift: " + str(path))


def no_duplicate(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise RuntimeError("nonfinite JSON: " + value)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate,
                      parse_constant=reject_constant)


def verify_flat(root, identity):
    review_sha, manifest_sha, outer_sha = identity
    review, manifest, outer = root / "review.json", root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    regular(review, review_sha); regular(manifest, manifest_sha); regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n", "outer content")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1); require(len(fields) == 2, "manifest grammar")
        rel = fields[1][2:] if fields[1].startswith("./") else fields[1]
        require(rel and rel not in listed and ".." not in Path(rel).parts, "manifest member")
        listed.add(rel); regular(root / rel, fields[0])
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} and
              not path.relative_to(root).as_posix().startswith("__pycache__/")}
    require(listed == actual, "flat coverage: " + str(root))
    return load(review)


def verify_double(path, identity):
    file_sha, side_sha, outer_sha = identity
    side, outer = Path(str(path) + ".sha256"), Path(str(path) + ".sha256.seal.sha256")
    regular(path, file_sha); regular(side, side_sha); regular(outer, outer_sha)
    require(file_sha in side.read_text(encoding="utf-8") and
            side_sha in outer.read_text(encoding="utf-8"), "double seal content")
    return load(path)


def verify_result_root():
    manifest, outer = M528 / "SHA256SUMS", M528 / "SHA256SUMS.seal.sha256"
    regular(manifest, M528_ID[0]); regular(outer, M528_ID[1])
    require(outer.read_text(encoding="utf-8") == M528_ID[0] + "  SHA256SUMS\n", "M528 outer")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1); require(len(fields) == 2, "M528 manifest grammar")
        rel = fields[1][2:] if fields[1].startswith("./") else fields[1]
        require(rel not in listed and ".." not in Path(rel).parts, "M528 manifest member")
        listed.add(rel); regular(M528 / rel, fields[0])
    actual = {path.relative_to(M528).as_posix() for path in M528.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(listed == actual, "M528 coverage")
    return load(M528 / "m528_h67_single_port_same_ledger_recompute_result_r1.json")


def verify_m1102():
    regular(M1102_RESULT, M1102_ID[0])
    seal = M1102 / ".m1102_atomic_seal"
    regular(seal / "SHA256SUMS", M1102_ID[1]); regular(seal / "SHA256SUMS.seal.sha256", M1102_ID[2])
    require((seal / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            M1102_ID[1] + "  SHA256SUMS\n", "M1102 outer")
    return load(M1102_RESULT)


m1116c = verify_flat(M1116C, M1116C_ID)
draft = verify_double(DRAFT, DRAFT_ID)
m1000 = verify_flat(M1000, M1000_ID)
m528_review = verify_flat(M528_REVIEW, M528_REVIEW_ID)
m528 = verify_result_root()
m1102 = verify_m1102()
regular(M935, M935_SHA); regular(DOCS359, DOCS359_SHA)

require(m1116c["status"] == "STOP_M1116C_CURRENT_C1_SOURCE_NOT_FULL_STORAGE_DC_READY__GO_SOURCE_AUTHORING",
        "M1116C status")
storage = m1116c["storage_reconciliation"]
require((storage["budget_bytes"], storage["frozen_obligation_bytes"],
         storage["known_parent_psum_weight_macro_count"],
         storage["known_parent_psum_weight_macro_bytes"],
         storage["unresolved_metadata_and_reserve_bytes"]) ==
        (245760, 214912, 93, 190464, 24448), "M1116C storage")
require(draft["frozen_capacity_authority"]["unresolved_physical_mapping_bytes"] == 24448 and
        draft["current_source_readiness"]["exact_mapping_for_all_capacity_categories_exists"] is False and
        draft["authorization"]["dc_now"] is False, "draft boundary")
require(m1000["reconciliation"]["active_bitmap_proxy_bytes_mapping_status"].startswith("ambiguous") and
        m1000["reconciliation"]["fifo_control_reserve_status"].startswith(
            "16384-byte analytical reserve is not an instantiated memory"), "M1000 gap")
require(m528_review["status"].startswith("PASS") and m528["claim_boundary"]["rtl"] is False,
        "M528 boundary")

parent_other = m1102["raw_cpu_model"]["capacity"]["parent_plus_other"]
breakdown = {"active_bitmap_bytes": 1152, "descriptor_pingpong_bytes": 2304,
             "fifo_control_reserve_bytes": 16384, "parent_liveness_class_bytes": 1152,
             "psum_valid_sidecar_bytes": 1152, "source_mask_pingpong_bytes": 2304}
require(all(parent_other[key] == value for key, value in breakdown.items()) and
        sum(breakdown.values()) == 24448 and parent_other["parent_scratch_bytes"] == 18432,
        "M1102 metadata breakdown")

m528_capacity = m528["capacity"]["m505_dead_write_only_1rw"]
require(m528_capacity["logical_items"]["fifo_control_reserve"] == 16384 and
        m528_capacity["macro_rounded_items"]["fifo_control_reserve"] == 16384 and
        "at least 288 B" in m528_capacity["capacity_obligation_map"]
        ["one_cycle_parent_response_and_scheduler_queues"], "M528 FIFO lower-bound only")

source = M935.read_text(encoding="utf-8")
require("existing two 1152-bit parent-response slots remain the only wide queue" in source and
        re.search(r"logic \[1151:0\] slot0_data_q, slot1_data_q;", source) is not None,
        "M935 response slots")
explicit_response_payload_bytes = (2 * 1152) // 8
require(explicit_response_payload_bytes == 288, "response payload arithmetic")

snapshot = {"budget_bytes": 245760, "obligation_bytes": 214912,
    "known_macro_count": 93, "known_macro_bytes": 190464,
    "metadata_reserve_bytes": 24448, "fifo_reserve_bytes": 16384,
    "fifo_explicit_payload_lower_bound_bytes": 288,
    "fifo_residual_without_sealed_assignment_bytes": 16096,
    "active_bitmap_one_to_one_mapping": False,
    "fifo_full_semantics_defined": False, "metadata_port_lifetime_graph_defined": False,
    "full_storage_ready": False, "dc_authorized": False,
    "new_storage_rtl_authored": False}


def validate_stop(value):
    require(set(value) == set(snapshot), "snapshot keys")
    require(value == snapshot, "STOP snapshot drift")
    require(value["known_macro_bytes"] + value["metadata_reserve_bytes"] ==
            value["obligation_bytes"], "storage sum")
    require(value["fifo_reserve_bytes"] - value["fifo_explicit_payload_lower_bound_bytes"] ==
            value["fifo_residual_without_sealed_assignment_bytes"], "FIFO gap sum")
    return True


require(validate_stop(snapshot), "canonical STOP")
mutations = {}
for name, key, replacement in (
        ("erase_gap", "metadata_reserve_bytes", 0),
        ("claim_full_storage", "full_storage_ready", True),
        ("authorize_dc", "dc_authorized", True),
        ("pretend_fifo_complete", "fifo_full_semantics_defined", True),
        ("pretend_active_mapping", "active_bitmap_one_to_one_mapping", True),
        ("pretend_port_graph", "metadata_port_lifetime_graph_defined", True),
        ("claim_rtl_authored", "new_storage_rtl_authored", True),
        ("inflate_fifo_payload", "fifo_explicit_payload_lower_bound_bytes", 16384),
        ("invent_macro_count", "known_macro_count", 105),
        ("change_obligation", "obligation_bytes", 214911)):
    forged = copy.deepcopy(snapshot); forged[key] = replacement
    try:
        validate_stop(forged)
    except RuntimeError:
        mutations[name] = True
    else:
        mutations[name] = False
require(all(mutations.values()), "mutation escaped")

forbidden_new = [
    HW / "rtl_m1116c_c1_full_storage",
    HW / "dc_handoff/filelists/date_m1116c_m935_full_storage_dc.f",
    HW / "dc_handoff/scripts/run_dc_m1116c_m935_full_storage.tcl",
]
require(not any(path.exists() or path.is_symlink() for path in forbidden_new),
        "unproven storage source exists")

output = {"schema": "m1119c_m1116c_c1_full_storage_metadata_gap_analysis_v1",
    "status": "STOP_M1119C_M1116C_METADATA_SEMANTICS_UNPROVEN__NO_STORAGE_RTL_AUTHORED",
    "score": 100,
    "identity": {"m1116c_outer": M1116C_ID[2], "draft_outer": DRAFT_ID[2],
        "m1000_outer": M1000_ID[2], "m528_outer": M528_ID[1],
        "m528_hammer_outer": M528_REVIEW_ID[2], "m1102_outer": M1102_ID[2],
        "m935_sha256": M935_SHA, "docs359_sha256": DOCS359_SHA},
    "storage": snapshot,
    "metadata_breakdown_bytes": breakdown,
    "first_principles_gap": {
        "fifo_reserve_is_analytical_not_instantiated": True,
        "only_explicit_wide_queue_payload_bytes": 288,
        "fifo_residual_without_sealed_assignment_bytes": 16096,
        "small_proxy_sizes_not_whole_2048B_foundry_macros": [1152, 2304],
        "width_depth_ports_clock_reset_read_write_lifetime_missing": True,
        "dummy_or_tied_off_mapping_would_violate_m1116c": True},
    "mutations": {"rejected": sum(mutations.values()), "total": len(mutations),
                  "cases": mutations},
    "execution": {"rtl_authored": False, "wrapper_authored": False,
                  "filelist_authored": False, "tcl_authored": False,
                  "vcs": False, "dc": False, "eda": False, "gpu": False,
                  "remote": False},
    "minimum_unblock": [
        "Freeze a semantic inventory for all 16384 FIFO/control reserve bytes, including depth, width, payload/control fields, read/write concurrency, clock/reset and live consumers.",
        "Freeze one-to-one owners and simultaneous-access/lifetime constraints for active bitmap, descriptor ping-pong, liveness, psum-valid and source-mask proxies.",
        "Choose foundry macro, implemented standard-cell state, or identical external common charge for every byte without double counting; independently hammer that mapping before RTL authoring."],
    "authorization": {"next_semantic_mapping_source_only": True,
                      "storage_rtl_now": False, "eda_now": False}}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
               encoding="utf-8")
print(output["status"])
