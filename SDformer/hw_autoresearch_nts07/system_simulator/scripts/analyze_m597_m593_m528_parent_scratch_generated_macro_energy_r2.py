#!/usr/bin/env python3
"""M597 r2 bounded generated-macro component model for M528 parent scratch.

This source prices only nine generated 128x128-bit 1RW parent-scratch macros.
M504 all-write traffic comes from the sealed M504 physical single-port result:
same-address RAW forwarding is listed but is not charged as a macro read.  The
M528 dead-write-only row is priced separately.  Every energy value is per
frozen sampled inference, not per camera frame, C1, network, chip, or silicon.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_REL = "hw_autoresearch_nts07/contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json"
CONTRACT_PATH = REPO_ROOT / CONTRACT_REL
CONTRACT_SHA256 = "90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf"
CONTRACT_ID = "m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828"
CONTRACT_KEYS = {
    "contract_id",
    "date",
    "status",
    "objective",
    "analyzer_binding",
    "frozen_inputs",
    "frozen_model",
    "frozen_counts",
    "repair_lineage",
    "mandatory_static_hammer_checks",
    "claim_boundary",
    "forbidden_actions",
}

DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
WORD_BYTES = 144
OUTPUT_BLOCK_BANKS = 8
MACRO_COUNT = 9
VOLTAGE_V = 0.9
CLOCK_PERIOD_NS = 3.0
SAMPLE_COUNT = 10

EXPECTED_INPUTS = {
    "m504_result": {
        "path": "hw_autoresearch_nts07/results/m504_h67_single_port_parent_scratch_r3_20260827/m504_h67_single_port_parent_scratch_result_r3.json",
        "sha256": "a0d2234a3a660df42bb87be04d42085c6c19025e55bdc35a1d61b9c48a54634b",
        "directory": "hw_autoresearch_nts07/results/m504_h67_single_port_parent_scratch_r3_20260827",
        "manifest_sha256": "f682a43c35847fa1fd2d9234bff9f225943ed582db7c65bb3590fb634b51212c",
        "outer_seal_file_sha256": "87f3af91debc5dff7fa8510bd8bf91abc57884b996452d8157d7ab51c369568c",
    },
    "m504_result_hammer": {
        "path": "hw_autoresearch_nts07/reviews/m504_r3_result_hammer_r1_20260827/m504_r3_result_hammer_r1_20260827.json",
        "sha256": "ac3a961a41a4c1b6511275c9c98fcdf5669f9c0ed98399f2afcd2ded075389a1",
        "directory": "hw_autoresearch_nts07/reviews/m504_r3_result_hammer_r1_20260827",
        "manifest_sha256": "766305f189ffe95e03ac54d1bc1a79e8f199aa5532901034d4e38d0877908545",
        "outer_seal_file_sha256": "4b13077464eb96e21091663a0bd4598af7340c29d1920c5e4a2561075fb70f4d",
    },
    "m528_result": {
        "path": "hw_autoresearch_nts07/results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json",
        "sha256": "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
        "directory": "hw_autoresearch_nts07/results/m528_h67_single_port_same_ledger_recompute_r4_20260827",
        "manifest_sha256": "4556a3383507e81ad9883f59bb55bb3d4fd08e7ec03977b215108b5ce4565073",
        "outer_seal_file_sha256": "02abbf7f9209d9a41d803c9942bfb43550be0d40945e3c094c1e457bda0db053",
    },
    "m528_result_hammer": {
        "path": "hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827/review.json",
        "sha256": "4f70610dcb5c0778fd7874b8f70239f9139c5f98732ae439ab246129ede53d6e",
        "directory": "hw_autoresearch_nts07/reviews/m528_r4_result_hammer_r1_20260827",
        "manifest_sha256": "678a0541702b9804691a5700a55fb4dc8c07f524ee5b6176800196371ebe3b56",
        "outer_seal_file_sha256": "ec442c74ca4dee305178e863a97e976940e0f5d6b98a0ad57e52cd298c01653e",
    },
    "generated_macro_mapping": {
        "path": "hw_autoresearch_nts07/reviews/tsmc28_sram_macro_audit_r1_20260827/tsmc28_sram_mapping_r1.json",
        "sha256": "68017fb51773713dd7dbee9463ec60d1dcdac9dea6e56588463e7f4ded96be4d",
        "directory": "hw_autoresearch_nts07/reviews/tsmc28_sram_macro_audit_r1_20260827",
        "manifest_sha256": "34be39b31afc57b0f22775590a7977c3b42f5277c52e8062c8b1b3bc0d648321",
        "outer_seal_file_sha256": "7832fea23f44038be1528c1480bfeed705c7c9705d1e727d367d678ae9720df4",
    },
    "m595_failed_review": {
        "path": "hw_autoresearch_nts07/reviews/m595_m593_parent_scratch_energy_source_static_hammer_r1_20260828/review.json",
        "sha256": "b8db95dbe045025fb815c2a6513cf258b519faa334446f5c3b4ccb8d2e23f875",
        "directory": "hw_autoresearch_nts07/reviews/m595_m593_parent_scratch_energy_source_static_hammer_r1_20260828",
        "manifest_sha256": "200c8d1ac338ff2746e540e7243514dbb6b704ed18a2c5c2620bf9c363c674da",
        "outer_seal_file_sha256": "921fef583a8dd4a3e3b19e4ca97059fa53504e049d7edd59c9d0ac0703ac071f",
    },
    "docs359": {
        "path": "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md",
        "sha256": DOCS359_SHA256,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: %s" % key)
        result[key] = value
    return result


def reject_json_constant(token: str) -> None:
    raise ValueError("non-standard JSON numeric token: %s" % token)


def assert_finite(value: Any, label: str = "$") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite JSON number at %s" % label)
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, "%s.%s" % (label, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, "%s[%d]" % (label, index))


def load_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), "missing JSON file: %s" % path)
    require(not path.is_symlink(), "refuse symlink JSON file: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    require(isinstance(value, dict), "top-level JSON must be object: %s" % path)
    assert_finite(value)
    return value


def exact_repo_path(relative: str) -> Path:
    path = REPO_ROOT / relative
    require(path.resolve() == (REPO_ROOT / relative).resolve(), "path resolution drift: %s" % relative)
    require(not path.is_symlink(), "refuse symlink frozen input: %s" % path)
    return path


def verify_manifest(directory: Path, expected_manifest_sha: str, expected_outer_file_sha: str) -> Dict[str, str]:
    require(directory.is_dir(), "missing sealed directory: %s" % directory)
    require(not directory.is_symlink(), "refuse symlink sealed directory: %s" % directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file(), "missing double seal: %s" % directory)
    require(not manifest.is_symlink() and not outer.is_symlink(), "refuse symlink seal: %s" % directory)
    actual_manifest_sha = sha256_file(manifest)
    require(actual_manifest_sha == expected_manifest_sha, "manifest SHA drift: %s" % directory)
    require(sha256_file(outer) == expected_outer_file_sha, "outer seal file SHA drift: %s" % directory)
    outer_tokens = outer.read_text(encoding="utf-8").strip().split()
    require(
        len(outer_tokens) == 2
        and outer_tokens[0] == actual_manifest_sha
        and outer_tokens[1] == "SHA256SUMS",
        "outer seal content drift: %s" % directory,
    )
    entries: Dict[str, str] = {}
    root_real = str(directory.resolve())
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split(maxsplit=1)
        require(len(tokens) == 2, "malformed manifest row: %s" % directory)
        digest, name = tokens
        name = name.lstrip("*")
        require(name not in entries, "duplicate manifest member: %s" % name)
        require(not os.path.isabs(name), "absolute manifest member: %s" % name)
        member = directory / name
        require(os.path.commonpath([root_real, str(member.resolve())]) == root_real, "manifest traversal: %s" % name)
        require(member.is_file() and not member.is_symlink(), "missing/symlink manifest member: %s" % member)
        require(sha256_file(member) == digest, "manifest member SHA drift: %s" % member)
        entries[name] = digest
    return entries


def validate_contract(path: Path) -> Dict[str, Any]:
    require(path.resolve() == CONTRACT_PATH.resolve(), "source contract path drift")
    require(not path.is_symlink(), "refuse symlink source contract")
    require(sha256_file(path) == CONTRACT_SHA256, "source contract SHA drift")
    contract = load_json(path)
    require(set(contract.keys()) == CONTRACT_KEYS, "source contract top-level key-set drift")
    require(contract.get("contract_id") == CONTRACT_ID, "source contract id drift")
    require(
        contract.get("status")
        == "SOURCE_ONLY_R2__AWAITING_FRESH_INDEPENDENT_STATIC_HAMMER__FORMAL_RUN_FORBIDDEN",
        "source contract status drift",
    )
    binding = contract.get("analyzer_binding", {})
    require(binding.get("contract_path") == CONTRACT_REL, "source contract self-path drift")
    require(binding.get("caller_supplied_input_paths_or_expected_sha_allowed") is False, "caller SHA boundary drift")
    require(contract.get("frozen_inputs") == EXPECTED_INPUTS, "source contract frozen-input map drift")
    require(contract.get("frozen_model", {}).get("sample_unit") == "one frozen sampled inference window; not asserted to be a camera frame", "sample unit drift")
    require(contract.get("claim_boundary", {}).get("per_frozen_sampled_inference") is True, "sample claim missing")
    require(contract.get("claim_boundary", {}).get("sample_is_camera_frame") is False, "frame claim drift")
    return contract


def verify_frozen_inputs(contract: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]]]:
    observed: Dict[str, Dict[str, str]] = {}
    parsed: Dict[str, Dict[str, Any]] = {}
    for name in sorted(EXPECTED_INPUTS.keys()):
        entry = contract["frozen_inputs"][name]
        path = exact_repo_path(entry["path"])
        require(path.is_file(), "missing frozen input: %s" % path)
        actual_sha = sha256_file(path)
        require(actual_sha == entry["sha256"], "%s SHA drift" % name)
        receipt = {
            "path": entry["path"],
            "sha256": actual_sha,
        }
        if "directory" in entry:
            directory = exact_repo_path(entry["directory"])
            members = verify_manifest(
                directory,
                entry["manifest_sha256"],
                entry["outer_seal_file_sha256"],
            )
            relative_member = os.path.relpath(str(path), str(directory)).replace(os.sep, "/")
            require(relative_member in members or ("./" + relative_member) in members, "%s absent from sealed manifest" % name)
            manifest_value = members.get(relative_member, members.get("./" + relative_member))
            require(manifest_value == actual_sha, "%s manifest member digest drift" % name)
            receipt.update({
                "directory": entry["directory"],
                "manifest_sha256": entry["manifest_sha256"],
                "outer_seal_file_sha256": entry["outer_seal_file_sha256"],
                "double_seal_pass": "true",
            })
        observed[name] = receipt
    for name in (
        "m504_result",
        "m504_result_hammer",
        "m528_result",
        "m528_result_hammer",
        "generated_macro_mapping",
        "m595_failed_review",
    ):
        parsed[name] = load_json(exact_repo_path(EXPECTED_INPUTS[name]["path"]))
    return observed, parsed


def unique_design(rows: List[Dict[str, Any]], design: str) -> Dict[str, Any]:
    matches = [row for row in rows if row.get("design") == design]
    require(len(matches) == 1, "expected one traffic row %s; got %d" % (design, len(matches)))
    return matches[0]


def component_row(
    design: str,
    cycle_source: str,
    traffic_source: str,
    cycles_s10: int,
    macro_reads_per_output_block: int,
    raw_forwards_per_output_block: int,
    macro_writes_per_output_block: int,
    parent_edges_per_output_block: int,
    active_rows_per_output_block: int,
    read_energy_pj: float,
    write_energy_pj: float,
    leakage_power_mw: float,
) -> Dict[str, Any]:
    require(macro_reads_per_output_block + raw_forwards_per_output_block == parent_edges_per_output_block, "%s read/forward conservation mismatch" % design)
    require(macro_writes_per_output_block <= active_rows_per_output_block, "%s writes exceed active rows" % design)
    read_accesses_s10 = macro_reads_per_output_block * OUTPUT_BLOCK_BANKS
    write_accesses_s10 = macro_writes_per_output_block * OUTPUT_BLOCK_BANKS
    read_bytes_s10 = read_accesses_s10 * WORD_BYTES
    write_bytes_s10 = write_accesses_s10 * WORD_BYTES
    dynamic_mj = (
        read_accesses_s10 * read_energy_pj
        + write_accesses_s10 * write_energy_pj
    ) / SAMPLE_COUNT / 1.0e9
    latency_ms = cycles_s10 * CLOCK_PERIOD_NS / SAMPLE_COUNT / 1.0e6
    leakage_mj = leakage_power_mw * latency_ms / 1000.0
    return {
        "design": design,
        "cycle_source": cycle_source,
        "traffic_source": traffic_source,
        "cycles_s10": cycles_s10,
        "latency_ms_per_frozen_sampled_inference_at_3ns": latency_ms,
        "macro_reads_per_output_block": macro_reads_per_output_block,
        "raw_forwards_per_output_block": raw_forwards_per_output_block,
        "macro_writes_per_output_block": macro_writes_per_output_block,
        "parent_edges_per_output_block": parent_edges_per_output_block,
        "active_rows_per_output_block": active_rows_per_output_block,
        "output_block_banks": OUTPUT_BLOCK_BANKS,
        "logical_word_bytes": WORD_BYTES,
        "read_accesses_s10": read_accesses_s10,
        "write_accesses_s10": write_accesses_s10,
        "read_bytes_s10": read_bytes_s10,
        "write_bytes_s10": write_bytes_s10,
        "raw_forward_macro_read_energy_charged": False,
        "read_plus_forward_equals_parent_edges": True,
        "writes_do_not_exceed_active_rows": True,
        "dynamic_energy_mj_per_frozen_sampled_inference": dynamic_mj,
        "leakage_energy_mj_per_frozen_sampled_inference": leakage_mj,
        "modeled_parent_scratch_energy_mj_per_frozen_sampled_inference": dynamic_mj + leakage_mj,
    }


def build_result(contract: Dict[str, Any], identity: Dict[str, Dict[str, str]], parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    m504 = parsed["m504_result"]
    m504_hammer = parsed["m504_result_hammer"]
    m528 = parsed["m528_result"]
    m528_hammer = parsed["m528_result_hammer"]
    macro = parsed["generated_macro_mapping"]
    m595 = parsed["m595_failed_review"]

    require(m504.get("schema") == "m504_h67_single_port_parent_scratch_result_v3", "M504 result schema drift")
    require(m504.get("status") == "PASS_M504_SINGLE_PORT_FASTKILL", "M504 result status drift")
    require(m504.get("claim_boundary", {}).get("energy") is False, "M504 logical traffic already claims energy")
    require(m504_hammer.get("schema") == "m504_r3_result_hammer_v1", "M504 hammer schema drift")
    require(m504_hammer.get("status") == "PASS_NO_GO_M504_RTL", "M504 hammer status drift")
    require(int(m504_hammer.get("score")) == 98, "M504 hammer score drift")
    require(m504_hammer.get("identity_audit", {}).get("result_manifest_pass") is True, "M504 result manifest not admitted")
    require(m504_hammer.get("identity_audit", {}).get("result_outer_seal_pass") is True, "M504 result outer seal not admitted")
    require(m504_hammer.get("claim_boundary", {}).get("m504_exact_cpu_cycle_audit") is True, "M504 exact CPU audit missing")
    require(m504_hammer.get("claim_boundary", {}).get("m504_performance_admitted") is False, "M504 performance boundary drift")

    require(m528.get("schema") == "m528_h67_single_port_same_ledger_recompute_result_v1", "M528 result schema drift")
    require(m528.get("claim_boundary", {}).get("exact_cpu_cycle_recompute") is True, "M528 exact CPU scope missing")
    require(m528.get("claim_boundary", {}).get("energy") is False, "M528 logical traffic already claims energy")
    require(m528_hammer.get("score_100") == 99, "M528 hammer score drift")
    require(m528_hammer.get("claim_boundary", {}).get("admitted_exact_cpu_cycle_candidate") is True, "M528 candidate admission missing")
    require(m528_hammer.get("claim_boundary", {}).get("traffic_is_logical_bytes_not_energy") is True, "M528 byte boundary missing")

    require(m595.get("schema") == "m595_m593_parent_scratch_generated_macro_energy_source_static_hammer_v1", "M595 lineage schema drift")
    require(m595.get("status") == "FAIL_SOURCE_STATIC__NO_EXACT_RUNNER_OR_LAUNCH_ADMISSION__REPAIR_REQUIRED", "M595 lineage status drift")
    require((m595.get("p0_count"), m595.get("p1_count"), m595.get("p2_count")) == (1, 1, 1), "M595 finding count drift")

    aggregate = m504["aggregate_one_output_block"]
    m504_reads = int(aggregate["deadline_macro_reads"])
    m504_forwards = int(aggregate["deadline_forwarded_reads"])
    m504_writes = int(aggregate["deadline_macro_writes"])
    parent_edges = int(aggregate["parent_edges"])
    active_rows = int(aggregate["active_rows"])
    m504_cycles = int(m504["cycle_comparison"]["deadline_lookahead_single_port_cycles"])
    require(m504_reads == 16490761, "M504 macro-read anchor drift")
    require(m504_forwards == 1714628, "M504 RAW-forward anchor drift")
    require(m504_writes == 27305568, "M504 macro-write anchor drift")
    require(parent_edges == 18205389, "M504 parent-edge anchor drift")
    require(active_rows == 27305568, "M504 active-row anchor drift")
    require(m504_cycles == 456016645, "M504 cycle anchor drift")
    require(m504_reads + m504_forwards == parent_edges, "M504 read/forward conservation mismatch")
    require(m504_writes == active_rows, "M504 all-write conservation mismatch")
    hammer_port = m504_hammer["port_tax_decomposition"]
    require(int(hammer_port["macro_reads"]) == m504_reads, "M504 hammer macro-read drift")
    require(int(hammer_port["forwarded_parent_edges"]) == m504_forwards, "M504 hammer forward drift")

    traffic_rows = m528.get("traffic", {}).get("rows", [])
    require(isinstance(traffic_rows, list), "M528 traffic rows missing")
    dead_traffic = unique_design(traffic_rows, "m505_dead_write_only_1rw")
    dead_read_bytes = int(dead_traffic["parent_scratch_read_bytes"])
    dead_write_bytes = int(dead_traffic["parent_scratch_write_bytes"])
    require(dead_read_bytes % (OUTPUT_BLOCK_BANKS * WORD_BYTES) == 0, "dead read bytes not bank/word aligned")
    require(dead_write_bytes % (OUTPUT_BLOCK_BANKS * WORD_BYTES) == 0, "dead write bytes not bank/word aligned")
    dead_reads = dead_read_bytes // (OUTPUT_BLOCK_BANKS * WORD_BYTES)
    dead_writes = dead_write_bytes // (OUTPUT_BLOCK_BANKS * WORD_BYTES)
    dead_cycles = int(m528["aggregate_cycles"]["m505_dead_write_only_1rw_cycles"])
    require(dead_reads == 16490761, "dead-only macro-read anchor drift")
    require(dead_writes == 9947701, "dead-only macro-write anchor drift")
    require(dead_cycles == 435293339, "dead-only cycle anchor drift")
    require(int(m528["aggregate_cycles"]["m504_all_write_1rw_cycles"]) == m504_cycles, "M504/M528 cycle cross-check drift")
    require(dead_reads == m504_reads, "dead-write suppression changed physical macro-read count")
    m528_conservation = m528_hammer["validated_metrics"]["conservation"]
    require(int(m528_conservation["dead_only_macro_reads"]) == dead_reads, "M528 hammer dead read drift")
    require(int(m528_conservation["dead_only_forwards"]) == m504_forwards, "M528 hammer forward drift")
    require(int(m528_conservation["dead_only_writes"]) == dead_writes, "M528 hammer dead write drift")
    dead_elisions = int(m528_conservation["dead_write_elisions"])
    require(dead_writes + dead_elisions == active_rows, "dead-write conservation mismatch")

    require(macro.get("schema") == "tsmc28_sram_macro_mapping_audit_v1", "macro-map schema drift")
    require(macro.get("status") == "PARTIAL_FAIL_CLOSED", "macro-map fail-closed status drift")
    require(macro.get("docs359_sha256") == DOCS359_SHA256, "macro-map docs359 drift")
    inventory = macro["generated_view_inventory"]
    slow = inventory["slow"]
    require(inventory.get("cell") == "TS1N28HPCPHVTB128X128M4S", "macro cell drift")
    require(inventory.get("logical_shape") == "128x128b 1RW SP", "macro shape drift")
    require(inventory.get("checksum_verification") == "13/13 OK on 2026-08-27", "macro checksum status drift")
    require(inventory.get("integration_status") == "AVAILABLE_BUT_NOT_IN_CURRENT_REPO_TREE", "macro integration boundary drift")
    require(slow.get("corner") == "ssg0p9v125c", "macro slow corner drift")
    require(float(slow["area_um2"]) == 8758.3606, "macro area drift")
    require(float(slow["cycle_ns"]) == 0.616, "macro cycle drift")
    require(float(slow["access_ns"]) == 0.4679, "macro access drift")
    require(float(slow["readc_uA_per_MHz"]) == 11.6754, "macro read-current drift")
    require(float(slow["writec_uA_per_MHz"]) == 11.1923, "macro write-current drift")
    require(float(slow["leakage_uA"]) == 66.6783, "macro leakage drift")
    generated = m528["capacity"]["m505_dead_write_only_1rw"]["generated_parent_scratch"]
    require(generated["organization"] == "9 x 128x128-bit 1RW SP; lower 64 rows used", "M528 macro organization drift")
    require(int(generated["physical_capacity_bytes"]) == 18432, "M528 physical macro capacity drift")
    require(int(generated["logical_payload_bytes"]) == 9216, "M528 logical payload drift")
    require(float(generated["area_um2"]) == MACRO_COUNT * float(slow["area_um2"]), "nine-macro area mismatch")

    read_energy_pj = MACRO_COUNT * float(slow["readc_uA_per_MHz"]) * VOLTAGE_V
    write_energy_pj = MACRO_COUNT * float(slow["writec_uA_per_MHz"]) * VOLTAGE_V
    leakage_power_mw = MACRO_COUNT * float(slow["leakage_uA"]) * VOLTAGE_V / 1000.0

    all_write = component_row(
        "m504_all_write_1rw_parent_scratch",
        "sealed M504 result: cycle_comparison.deadline_lookahead_single_port_cycles",
        "sealed M504 result: aggregate_one_output_block.deadline_macro_reads/deadline_forwarded_reads/deadline_macro_writes",
        m504_cycles,
        m504_reads,
        m504_forwards,
        m504_writes,
        parent_edges,
        active_rows,
        read_energy_pj,
        write_energy_pj,
        leakage_power_mw,
    )
    dead_only = component_row(
        "m528_dead_write_only_1rw_parent_scratch",
        "sealed M528 result: aggregate_cycles.m505_dead_write_only_1rw_cycles",
        "sealed M528 result: traffic row m505_dead_write_only_1rw; forward split cross-checked by sealed M528 hammer",
        dead_cycles,
        dead_reads,
        m504_forwards,
        dead_writes,
        parent_edges,
        active_rows,
        read_energy_pj,
        write_energy_pj,
        leakage_power_mw,
    )
    require(all_write["read_bytes_s10"] == 18997356672, "M504 physical read-byte drift")
    require(all_write["write_bytes_s10"] == 31456014336, "M504 physical write-byte drift")
    require(dead_only["read_bytes_s10"] == dead_read_bytes, "dead physical read-byte mismatch")
    require(dead_only["write_bytes_s10"] == dead_write_bytes, "dead physical write-byte mismatch")

    baseline_energy = all_write["modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"]
    candidate_energy = dead_only["modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"]
    cycle_speedup = float(m504_cycles) / float(dead_cycles)
    reduction_fraction = 1.0 - candidate_energy / baseline_energy
    saved_mj = baseline_energy - candidate_energy
    diagnostic = contract["repair_lineage"]["m595_corrected_diagnostic_reference_only"]
    require(math.isclose(reduction_fraction * 100.0, float(diagnostic["component_energy_reduction_percent"]), rel_tol=0.0, abs_tol=1e-12), "independent reduction does not match M595 diagnostic")
    require(math.isclose(saved_mj, float(diagnostic["component_energy_saved_mj_per_frozen_sampled_inference"]), rel_tol=0.0, abs_tol=1e-12), "independent saved-energy diagnostic drift")
    require(math.isclose(cycle_speedup, float(m528["aggregate_cycles"]["m504_to_dead_write_speedup"]), rel_tol=0.0, abs_tol=1e-15), "cycle-ablation ratio drift")

    result_identity = {
        "source_contract": {
            "path": CONTRACT_REL,
            "sha256": CONTRACT_SHA256,
            "exact_key_set_pass": True,
        },
        "frozen_inputs": identity,
    }
    return {
        "schema": "m597_m593_m528_parent_scratch_generated_macro_energy_result_v2",
        "date": "2026-08-28",
        "status": "PASS_BOUNDED_GENERATED_MACRO_COMPONENT_MODEL__PENDING_FRESH_INDEPENDENT_RESULT_HAMMER",
        "identity": result_identity,
        "scope": {
            "checkpoint": "H67 ep35",
            "sequence_count": 1,
            "frozen_sampled_inference_count": SAMPLE_COUNT,
            "sample_is_camera_frame": False,
            "operators": "four bottleneck Conv3x3 only",
            "component": "nine generated 128x128-bit 1RW parent-scratch macros only",
            "corner": "ssg0p9v125c at 0.9 V",
            "clock_period_ns": CLOCK_PERIOD_NS,
            "leakage_assumption": "all nine macros remain powered for the complete modeled four-Conv sample schedule; no power gating credited",
        },
        "macro": {
            "cell": inventory["cell"],
            "count": MACRO_COUNT,
            "area_um2": MACRO_COUNT * float(slow["area_um2"]),
            "cycle_ns": float(slow["cycle_ns"]),
            "access_ns": float(slow["access_ns"]),
            "full_1152b_read_energy_pj_per_physical_macro_access": read_energy_pj,
            "full_1152b_write_energy_pj_per_physical_macro_access": write_energy_pj,
            "leakage_power_mw": leakage_power_mw,
            "model_note": "generated-macro slow-corner datasheet current; all nine 128-bit slices activated per physical 1152-bit access",
        },
        "rows": [all_write, dead_only],
        "conservation": {
            "m504_macro_reads_plus_raw_forwards_equal_parent_edges": m504_reads + m504_forwards == parent_edges,
            "m504_macro_writes_equal_active_rows": m504_writes == active_rows,
            "m504_and_dead_only_macro_reads_equal": m504_reads == dead_reads,
            "dead_macro_writes_plus_dead_elisions_equal_active_rows": dead_writes + dead_elisions == active_rows,
            "raw_forwards_charged_as_macro_reads": False,
            "all_byte_counts_are_accesses_times_144": True,
            "all_equalities_pass": True,
        },
        "ablation": {
            "dead_write_only_cycle_speedup_vs_m504_all_write": cycle_speedup,
            "dead_write_only_parent_scratch_component_energy_reduction_fraction": reduction_fraction,
            "dead_write_only_parent_scratch_component_energy_reduction_percent": reduction_fraction * 100.0,
            "dead_write_only_parent_scratch_component_energy_saved_mj_per_frozen_sampled_inference": saved_mj,
            "label": "generated-macro datasheet component ablation on ten frozen sampled inferences; pending independent result hammer",
        },
        "claim_boundary": {
            "allowed_label_after_independent_result_hammer": "generated-macro datasheet component model for M528 parent scratch",
            "component_energy_model": True,
            "sealed_trace_physical_macro_access_counts": True,
            "per_frozen_sampled_inference": True,
            "sample_is_camera_frame": False,
            "rtl_integrated_macro_ppa": False,
            "interconnect_or_clock_tree_energy": False,
            "logic_energy": False,
            "other_sram_energy": False,
            "dram_energy": False,
            "c1_total_energy": False,
            "full_network_energy": False,
            "energy_per_system_frame": False,
            "system_energy": False,
            "silicon_measurement": False,
            "system_speedup": False,
            "date_headline": False,
            "result_hammer_pending": True,
        },
    }


def write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def publish_result(output_dir: Path, result: Dict[str, Any]) -> None:
    require(not output_dir.exists(), "refuse to overwrite output directory: %s" % output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (".%s.m597_staging_%d" % (output_dir.name, os.getpid()))
    require(not staging.exists(), "staging directory already exists: %s" % staging)
    staging.mkdir()
    json_name = "m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json"
    csv_name = "m597_parent_scratch_energy_rows_r2.csv"
    complete_name = "RUN_COMPLETE.txt"
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_exclusive(staging / json_name, json_text)
    rows = result["rows"]
    with (staging / csv_name).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    write_exclusive(staging / complete_name, "PASS_M597_R2_ANALYZER_OUTPUT_PENDING_INDEPENDENT_RESULT_HAMMER\n")
    member_names = [complete_name, csv_name, json_name]
    manifest_lines = ["%s  %s" % (sha256_file(staging / name), name) for name in member_names]
    manifest_text = "\n".join(manifest_lines) + "\n"
    write_exclusive(staging / "SHA256SUMS", manifest_text)
    manifest_sha = sha256_file(staging / "SHA256SUMS")
    write_exclusive(staging / "SHA256SUMS.seal.sha256", "%s  SHA256SUMS\n" % manifest_sha)
    os.rename(str(staging), str(output_dir))


def self_test(contract: Dict[str, Any]) -> None:
    require(contract["contract_id"] == CONTRACT_ID, "self-test contract drift")
    with tempfile.TemporaryDirectory(prefix="m597_r2_selftest_") as temp_name:
        temp = Path(temp_name)
        cases = {
            "duplicate": "{\"x\":1,\"x\":2}",
            "nan": "{\"x\":NaN}",
            "infinity": "{\"x\":Infinity}",
            "list": "[1,2]",
        }
        for name, text in cases.items():
            path = temp / (name + ".json")
            path.write_text(text, encoding="utf-8")
            rejected = False
            try:
                load_json(path)
            except ValueError:
                rejected = True
            require(rejected, "strict JSON self-test accepted %s" % name)
    read_pj = MACRO_COUNT * 11.6754 * VOLTAGE_V
    write_pj = MACRO_COUNT * 11.1923 * VOLTAGE_V
    leakage_mw = MACRO_COUNT * 66.6783 * VOLTAGE_V / 1000.0
    baseline = component_row(
        "selftest_all_write",
        "literal",
        "literal",
        456016645,
        16490761,
        1714628,
        27305568,
        18205389,
        27305568,
        read_pj,
        write_pj,
        leakage_mw,
    )
    candidate = component_row(
        "selftest_dead_only",
        "literal",
        "literal",
        435293339,
        16490761,
        1714628,
        9947701,
        18205389,
        27305568,
        read_pj,
        write_pj,
        leakage_mw,
    )
    b = baseline["modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"]
    c = candidate["modeled_parent_scratch_energy_mj_per_frozen_sampled_inference"]
    require(math.isclose((1.0 - c / b) * 100.0, 38.228307918921945, rel_tol=0.0, abs_tol=1e-12), "self-test energy reduction drift")
    require(math.isclose(b - c, 1.2622562286593053, rel_tol=0.0, abs_tol=1e-12), "self-test saved energy drift")
    require(baseline["read_bytes_s10"] == candidate["read_bytes_s10"] == 18997356672, "self-test RAW-forward traffic drift")
    print("PASS_M597_R2_STATIC_SELF_TEST__NO_BUSINESS_INPUT_OR_RESULT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    contract = validate_contract(args.source_contract)
    if args.self_test:
        require(args.output_dir is None, "self-test must not name output directory")
        self_test(contract)
        return
    require(args.output_dir is not None, "production mode requires --output-dir")
    identity, parsed = verify_frozen_inputs(contract)
    result = build_result(contract, identity, parsed)
    publish_result(args.output_dir, result)


if __name__ == "__main__":
    main()
