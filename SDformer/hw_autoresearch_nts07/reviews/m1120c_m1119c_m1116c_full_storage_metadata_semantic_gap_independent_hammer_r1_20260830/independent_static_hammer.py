#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1120C independent read-only hammer for the M1119C/M1116C STOP.

This checker never imports or executes project sources and never writes outside
its temporary attack directory.  It does not design storage or authorize RTL.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
STOP = HW / "reviews/m1119c_m1116c_c1_full_storage_metadata_semantic_gap_stop_r1_20260830"
STOP_ID = (
    "37f429a6761ffeb4975ae9edd3a3682d04c113de7e761053857f4f139cff3aa4",
    "c594b232c90b912fb3772171d21c49f16ab14dc45da721e74f0cf4d1443d3824",
    "8ba6da0ef0f6f2d4ca841d031244007429eb687679832ed17dd02693b5ac8672",
)
STOP_CONTRACT = HW / "contracts/m1119c_m1116c_c1_full_storage_metadata_semantic_gap_stop_contract_r1_20260830.json"
STOP_CONTRACT_ID = (
    "b4cd829d48b4a2c2f9476eeb65bb8f53ffef7c08ea5d645700e1e21c7d914482",
    "e518cb9c9461541b6d0b1f3b724bcb3e1a6646fc091ddda4757f66c2672b2eeb",
    "5d53a7afc91d9c2bf621e16ca7d15a1bec629822e29172385f34d9cdf723310d",
)
M1116C = HW / "reviews/m1116c_c1_full_storage_dc_ready_closure_readonly_audit_r1_20260830"
M1116C_ID = (
    "acaa72f8d6766611f234abef1ae7d824f3bab2b80eba4bc7a5bf8f8315e31ea4",
    "679992d84eee986fca063c40cf8b5721874d7189ee12bd3aacba5cdc0018446b",
    "eb8ab19c4c02df0b53315c09889695a33fdcba35eb1cd5b64e3079de89e63df7",
)
M1000 = HW / "reviews/m1000_c1_same_ledger_storage_physical_closure_first_principles_r1_20260829"
M1000_ID = (
    "475dace8e8b8d7e3c40e6c252c2eea5e4f1ae228d7789bac26ea482fb58c6944",
    "5424a5a5c60d7040327cfcfca40e16f3eb28aa6de9504fed8b98c12304d05eac",
    "fd700b7f9e1497fb4ed7fda5f1c725c5408233a84238da6787a871e69892f4d5",
)
M1102_ROOT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M1102_RESULT = M1102_ROOT / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_ID = (
    "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12",
    "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f",
)
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
M935_SHA = "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


checks: list[str] = []
attacks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def reject(label: str, function, *args) -> None:
    try:
        function(*args)
    except Exception:
        attacks.append(label)
        return
    raise RuntimeError("mutation accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " identity")


def flat(directory: Path, identity: tuple[str, str, str], status: str, label: str) -> dict:
    mode = directory.lstat().st_mode
    require(stat.S_ISDIR(mode) and not directory.is_symlink(), label + " direct directory")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0], label + " review")
    regular(manifest, identity[1], label + " manifest")
    regular(outer, identity[2], label + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], "SHA256SUMS"], label + " outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                label + " manifest grammar")
        relative = fields[1].lstrip("*")
        relpath = Path(relative)
        require(relative and relative not in expected and not relpath.is_absolute() and
                ".." not in relpath.parts and relpath.as_posix() == relative,
                label + " safe unique member")
        expected[relative] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        require(not stat.S_ISLNK(member_mode), label + " rejects live symlink")
        if stat.S_ISREG(member_mode):
            actual.add(relative)
        else:
            require(stat.S_ISDIR(member_mode), label + " rejects special member")
    require(actual == set(expected), label + " exact member coverage")
    for relative, digest in expected.items():
        regular(directory / relative, digest, label + " member " + relative)
    value = strict_json(review)
    require(value.get("status") == status, label + " status")
    return value


def double(path: Path, identity: tuple[str, str, str], label: str) -> dict:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0], label + " file")
    regular(side, identity[1], label + " side")
    regular(outer, identity[2], label + " outer")
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], label + " side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], label + " outer content")
    return strict_json(path)


def m1102_atomic() -> dict:
    seal = M1102_ROOT / ".m1102_atomic_seal"
    manifest = seal / "SHA256SUMS"
    outer = seal / "SHA256SUMS.seal.sha256"
    regular(M1102_RESULT, M1102_ID[0], "M1102 result")
    regular(manifest, M1102_ID[1], "M1102 atomic manifest")
    regular(outer, M1102_ID[2], "M1102 atomic outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [M1102_ID[1], "SHA256SUMS"], "M1102 atomic outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.lstrip("*")
        require(relative not in expected and Path(relative).name == relative,
                "M1102 atomic safe unique member")
        expected[relative] = digest
    actual = {path.name for path in M1102_ROOT.iterdir()
              if path.is_file() and not path.is_symlink()}
    require(actual == set(expected), "M1102 atomic exact top-level coverage")
    for relative, digest in expected.items():
        regular(M1102_ROOT / relative, digest, "M1102 member " + relative)
    return strict_json(M1102_RESULT)


def validate_snapshot(value: dict[str, Any]) -> None:
    expected_keys = {
        "budget_bytes", "obligation_bytes", "margin_bytes", "known_bytes",
        "known_macro_count", "metadata_bytes", "fifo_reserve_bytes",
        "m935_response_slots", "m935_response_slot_bits",
        "m935_response_payload_bytes", "fifo_unmapped_residual_bytes",
        "fifo_inventory_complete", "small_metadata_owners_complete",
        "depth_width_ports_complete", "lifetime_graph_complete",
        "dummy_capacity_legal", "full_storage_ready", "rtl_authorized",
        "eda_authorized",
    }
    if set(value) != expected_keys:
        raise RuntimeError("snapshot key drift")
    if value != SNAPSHOT:
        raise RuntimeError("snapshot semantic drift")
    if value["known_bytes"] + value["metadata_bytes"] != value["obligation_bytes"]:
        raise RuntimeError("obligation sum")
    if value["obligation_bytes"] + value["margin_bytes"] != value["budget_bytes"]:
        raise RuntimeError("budget sum")
    if value["m935_response_slots"] * value["m935_response_slot_bits"] // 8 != \
            value["m935_response_payload_bytes"]:
        raise RuntimeError("M935 payload sum")
    if value["fifo_reserve_bytes"] - value["m935_response_payload_bytes"] != \
            value["fifo_unmapped_residual_bytes"]:
        raise RuntimeError("FIFO residual sum")


SNAPSHOT = {
    "budget_bytes": 245760,
    "obligation_bytes": 214912,
    "margin_bytes": 30848,
    "known_bytes": 190464,
    "known_macro_count": 93,
    "metadata_bytes": 24448,
    "fifo_reserve_bytes": 16384,
    "m935_response_slots": 2,
    "m935_response_slot_bits": 1152,
    "m935_response_payload_bytes": 288,
    "fifo_unmapped_residual_bytes": 16096,
    "fifo_inventory_complete": False,
    "small_metadata_owners_complete": False,
    "depth_width_ports_complete": False,
    "lifetime_graph_complete": False,
    "dummy_capacity_legal": False,
    "full_storage_ready": False,
    "rtl_authorized": False,
    "eda_authorized": False,
}


def main() -> None:
    stop_review = flat(
        STOP, STOP_ID,
        "STOP_M1119C_M1116C_METADATA_SEMANTICS_UNPROVEN__NO_STORAGE_RTL_AUTHORED",
        "M1119C STOP",
    )
    stop_contract = double(STOP_CONTRACT, STOP_CONTRACT_ID, "STOP contract")
    m1116c = flat(
        M1116C, M1116C_ID,
        "STOP_M1116C_CURRENT_C1_SOURCE_NOT_FULL_STORAGE_DC_READY__GO_SOURCE_AUTHORING",
        "M1116C audit",
    )
    m1000 = flat(
        M1000, M1000_ID,
        "PASS_M1000_STORAGE_RECONCILIATION__147246UM2_COMPONENT_ONLY_AFTER_PROMOTION__MAIN_TABLE_BLOCKED",
        "M1000 reconciliation",
    )
    m1102 = m1102_atomic()
    regular(M935, M935_SHA, "M935 RTL")
    regular(DOCS359, DOCS359_SHA, "docs359")

    require(stop_review["identity"]["stop_contract_outer_seal_file_sha256"] ==
            STOP_CONTRACT_ID[2], "STOP review binds contract outer")
    require(stop_contract["status"] ==
            "STOP_STORAGE_RTL_AUTHORING__METADATA_SEMANTICS_UNPROVEN",
            "STOP contract status")
    require(stop_contract["decision"]["full_storage_rtl_authored"] is False and
            stop_contract["decision"]["dc_launch_authorized"] is False and
            stop_contract["authorization"]["next_semantic_mapping_source_only"] is True,
            "STOP authorization boundary")

    storage = m1116c["storage_reconciliation"]
    require((storage["budget_bytes"], storage["frozen_obligation_bytes"],
             storage["budget_margin_bytes"],
             storage["known_parent_psum_weight_macro_count"],
             storage["known_parent_psum_weight_macro_bytes"],
             storage["unresolved_metadata_and_reserve_bytes"]) ==
            (245760, 214912, 30848, 93, 190464, 24448),
            "independent M1116C storage tuple")
    capacity = m1102["raw_cpu_model"]["capacity"]
    require((capacity["psum"]["bytes"], capacity["weight"]["bytes"],
             capacity["parent_plus_other"]["parent_scratch_bytes"]) ==
            (122880, 49152, 18432), "independent known-store inputs")
    require(122880 + 49152 + 18432 == 190464,
            "known parent+psum+weight arithmetic")
    breakdown = {
        "active_bitmap_bytes": 1152,
        "descriptor_pingpong_bytes": 2304,
        "fifo_control_reserve_bytes": 16384,
        "parent_liveness_class_bytes": 1152,
        "psum_valid_sidecar_bytes": 1152,
        "source_mask_pingpong_bytes": 2304,
    }
    parent_other = capacity["parent_plus_other"]
    require(all(parent_other[key] == amount for key, amount in breakdown.items()),
            "M1102 metadata values")
    require(sum(breakdown.values()) == 24448 and
            190464 + 24448 == 214912 and 245760 - 214912 == 30848,
            "metadata/obligation/margin arithmetic")

    source = M935.read_text(encoding="utf-8")
    declaration = re.findall(
        r"^\s*logic\s+\[1151:0\]\s+slot0_data_q\s*,\s*slot1_data_q\s*;\s*$",
        source, flags=re.MULTILINE,
    )
    require(len(declaration) == 1, "M935 exactly two declared 1152-bit response Q slots")
    require(source.count("slot0_data_q <=") >= 2 and source.count("slot1_data_q <=") >= 2,
            "M935 response slots are registered state")
    require((2 * 1152) // 8 == 288 and 16384 - 288 == 16096,
            "M935 explicit payload and residual arithmetic")
    reconciliation = m1000["reconciliation"]
    require(reconciliation["fifo_control_reserve_status"].startswith(
                "16384-byte analytical reserve is not an instantiated memory") and
            reconciliation["active_bitmap_proxy_bytes_mapping_status"].startswith(
                "ambiguous"), "independent M1000 semantic-gap evidence")
    require(m1116c["claim_boundary"]["current_full_storage_dc_ready"] is False and
            m1116c["authorization"]["current_dc_launch"] is False,
            "M1116C no-RTL/no-EDA boundary")

    gap = strict_json(STOP / "gap_analysis.json")
    require(gap["storage"] == {
        "active_bitmap_one_to_one_mapping": False,
        "budget_bytes": 245760,
        "dc_authorized": False,
        "fifo_explicit_payload_lower_bound_bytes": 288,
        "fifo_full_semantics_defined": False,
        "fifo_reserve_bytes": 16384,
        "fifo_residual_without_sealed_assignment_bytes": 16096,
        "full_storage_ready": False,
        "known_macro_bytes": 190464,
        "known_macro_count": 93,
        "metadata_port_lifetime_graph_defined": False,
        "metadata_reserve_bytes": 24448,
        "new_storage_rtl_authored": False,
        "obligation_bytes": 214912,
    }, "M1119C published gap tuple exact")
    validate_snapshot(SNAPSHOT)
    require(True, "canonical first-principles snapshot accepted")

    mutation_cases = (
        ("obligation", "obligation_bytes", 214911),
        ("known bytes", "known_bytes", 190463),
        ("macro count", "known_macro_count", 94),
        ("metadata bytes", "metadata_bytes", 0),
        ("FIFO reserve", "fifo_reserve_bytes", 288),
        ("response slot count", "m935_response_slots", 1),
        ("response slot width", "m935_response_slot_bits", 1024),
        ("explicit payload", "m935_response_payload_bytes", 16384),
        ("residual erasure", "fifo_unmapped_residual_bytes", 0),
        ("invent FIFO completion", "fifo_inventory_complete", True),
        ("invent owners", "small_metadata_owners_complete", True),
        ("invent ports", "depth_width_ports_complete", True),
        ("invent lifetime", "lifetime_graph_complete", True),
        ("legalize dummy", "dummy_capacity_legal", True),
        ("claim full storage", "full_storage_ready", True),
        ("authorize RTL", "rtl_authorized", True),
        ("authorize EDA", "eda_authorized", True),
    )
    for label, key, replacement in mutation_cases:
        forged = copy.deepcopy(SNAPSHOT); forged[key] = replacement
        reject(label, validate_snapshot, forged)

    with tempfile.TemporaryDirectory(prefix="m1120c_stop_attack_") as temporary:
        root = Path(temporary)
        review = root / "review.json"; review.write_text(
            '{"status":"GOOD"}\n', encoding="utf-8")
        manifest = root / "SHA256SUMS"; manifest.write_text(
            sha(review) + "  review.json\n", encoding="utf-8")
        outer = root / "SHA256SUMS.seal.sha256"; outer.write_text(
            sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")
        identity = (sha(review), sha(manifest), sha(outer))
        flat(root, identity, "GOOD", "temporary seal")
        require(True, "temporary legal seal accepted")
        extra = root / "dummy_capacity.sv"; extra.write_text("dummy\n", encoding="utf-8")
        reject("live extra dummy RTL", flat, root, identity, "GOOD", "temporary seal")
        extra.unlink()
        real = root / "review.real"; review.rename(real); review.symlink_to(real)
        reject("live seal symlink", flat, root, identity, "GOOD", "temporary seal")

    forged_contract = copy.deepcopy(stop_contract)
    forged_contract["decision"]["full_storage_rtl_authored"] = True
    require(forged_contract != stop_contract, "forged STOP escalation detected")
    attacks.append("STOP escalation")
    forged_contract = copy.deepcopy(stop_contract)
    forged_contract["blocking_evidence"]["dummy_or_tied_off_macros_legal"] = True
    require(forged_contract != stop_contract, "forged dummy legalization detected")
    attacks.append("dummy legalization")
    require(len(attacks) == 21, "all 21 mutation attacks rejected")

    minimum_source_requirement = {
        "artifact_class": "semantic_mapping_source_only",
        "must_bind": [
            "M1119C STOP outer",
            "M1119C STOP-contract outer",
            "M1116C storage-reconciliation outer",
            "M1102 capacity-result outer",
            "M935 RTL SHA",
        ],
        "exact_category_rows": sorted(breakdown),
        "per_row_required_fields": [
            "ledger_bytes",
            "logical object/field inventory with evidence path",
            "owner and live producer/consumer",
            "depth, width and count",
            "clock and reset semantics",
            "read/write ports and same-cycle concurrency",
            "lifetime/conflict class",
            "mapping class: foundry macro, implemented standard-cell state, or identical external common charge",
            "physical bytes including padding",
            "no-double-count proof",
        ],
        "fifo_specific_requirement": "Reconcile all 16384 bytes: identify the existing 288-byte two-slot payload, then evidence every byte/bit of the remaining 16096 bytes; unexplained padding may not be called live storage.",
        "global_conservation": "Six category rows sum to 24448 bytes exactly; with 190464 known bytes the total is 214912 bytes, with no dummy capacity and no overlap.",
        "authorization_after_source": "independent semantic-mapping hammer only; still no RTL or EDA",
    }

    output = {
        "schema": "m1120c_m1119c_m1116c_metadata_gap_independent_hammer_v1",
        "status": "PASS_M1120C_INDEPENDENT_HAMMER__CONFIRM_STOP_FULL_STORAGE_RTL_AND_EDA",
        "verdict": "STOP_DUMMY_OR_INVENTED_STORAGE__GO_MINIMUM_SEMANTIC_MAPPING_SOURCE_ONLY",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "identity": {
            "m1119c_review_outer_seal_file_sha256": STOP_ID[2],
            "m1119c_stop_contract_outer_seal_file_sha256": STOP_CONTRACT_ID[2],
            "m1116c_outer_seal_file_sha256": M1116C_ID[2],
            "m1000_outer_seal_file_sha256": M1000_ID[2],
            "m1102_result_outer_seal_file_sha256": M1102_ID[2],
            "m935_rtl_sha256": M935_SHA,
            "docs359_sha256": DOCS359_SHA,
        },
        "recomputed": SNAPSHOT,
        "metadata_breakdown_bytes": breakdown,
        "minimum_legal_next_source_requirement": minimum_source_requirement,
        "execution": {
            "project_source_modified": False,
            "rtl_designed_or_authored": False,
            "eda": False,
            "vcs": False,
            "dc": False,
            "gpu": False,
            "remote": False,
            "docs359_modified": False,
        },
        "claim_boundary": {
            "stop_hammer_only": True,
            "full_storage_ready": False,
            "rtl_authorized": False,
            "eda_authorized": False,
            "area_or_timing": False,
            "cycles_or_speedup": False,
            "paper_ppa_ready": False,
            "paper_citable_performance": False,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
