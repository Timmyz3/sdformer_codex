#!/usr/bin/env python3
"""Read-only M1160 hammer for the M1116C common-charge source package.

This checker intentionally runs no simulator or EDA tool.  In addition to
identity and capacity checks, it examines the ready/valid protocol that the
author's bounded static checker did not cover.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
WRAPPER = HW / "rtl_m1116c_c1_full_storage_boundary/m1116c_m935_c1_full_storage_common_charge_boundary.sv"
MAPPING = HW / "dc_handoff/manifests/m1116c_c1_full_storage_boundary_mapping_r1.tsv"
FILELIST = HW / "dc_handoff/filelists/date_m1116c_m935_c1_full_storage_common_charge_dc.f"
SDC = HW / "dc_handoff/constraints/date_m1116c_m935_c1_full_storage_common_charge_3ns.sdc"
TCL = HW / "dc_handoff/scripts/run_dc_m1116c_m935_c1_full_storage_common_charge_candidate.tcl"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
CONTRACT = HW / "contracts/m1116c_m1114_m1006_m963_m959_m935_full_storage_common_charge_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1116c_m935_c1_full_storage_common_charge_source_author_receipt_r1_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "wrapper": "f19a5e555115d7b178ef0e13f43a023ed850650214a44e285e3612e3fd903eab",
    "mapping": "16da013268f765d74703a041ccd35b2054ff425ef726d2b5c69d545230ae0271",
    "filelist": "b40404017f125a10937dc43624c0b1374eb41a697a6490b7ae5a8c4e785c73e2",
    "sdc": "326eaa7de4ac2487f4bd149dde0fca025b96b872e788493c687f598bbb209a47",
    "tcl": "f795550c3f760903c9cdd914d49e2e966f5cba98491d16f3cbe5c85cc23a02db",
    "contract": "82b7f1b6faea7e39f03f32c1bc1fbd924259147a8e2d5d9c58516c41646e7e30",
    "m935": "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    "parent": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "author_manifest": "4b1833f53d5aa6a239f5a41b4bdff7e5d7766952ccb6e33c8aaed1c356e441e8",
    "author_outer": "46d3845fd01f9b214d1cb47b29eb8e7b2eaedd5e9d0915f599a5d0fc79340b5f",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RuntimeError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + x)))


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(None, 1)
        rows.append((checksum, name.strip()))
    return rows


def main() -> int:
    checks: list[str] = []
    attacks: list[str] = []

    paths = {
        "wrapper": WRAPPER, "mapping": MAPPING, "filelist": FILELIST,
        "sdc": SDC, "tcl": TCL, "contract": CONTRACT, "m935": M935,
        "parent": PARENT, "docs359": DOC359,
        "author_manifest": AUTHOR / "SHA256SUMS",
        "author_outer": AUTHOR / "SHA256SUMS.seal.sha256",
    }
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), key + " regular file", checks)
        require(digest(path) == EXPECTED[key], key + " exact SHA", checks)

    author_rows = parse_manifest(AUTHOR / "SHA256SUMS")
    require([name for _, name in author_rows] == [
        "RUN_COMPLETE.txt", "mechanical_checks.json", "review.json",
        "review.md", "source_sha256.tsv"], "author exact member set", checks)
    for expected_sha, name in author_rows:
        require(digest(AUTHOR / name) == expected_sha,
                "author nested member " + name, checks)
    outer_rows = parse_manifest(AUTHOR / "SHA256SUMS.seal.sha256")
    require(outer_rows == [(EXPECTED["author_manifest"], "SHA256SUMS")],
            "author outer binds manifest", checks)

    contract = strict_json(CONTRACT)
    require(contract["authorization"]["vcs_now"] is False,
            "contract VCS false", checks)
    require(contract["authorization"]["dc_now"] is False,
            "contract DC false", checks)
    require(contract["claim_boundary"]["full_214912B_physically_integrated"] is False,
            "contract full-storage false", checks)

    raw_rows = [line.split("|") for line in MAPPING.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    require(len(raw_rows) == 4 and all(len(row) == 14 for row in raw_rows),
            "mapping four rows fourteen fields", checks)
    expected_start = 0
    byte_total = 0
    internal_bytes = 0
    external_bytes = 0
    physical_macros = 0
    by_name = {}
    for row in raw_rows:
        name, start, end, byte_count, placement, macro, count, capacity, equiv, port, latency, binding, axes, area = row
        start, end, byte_count = int(start), int(end), int(byte_count)
        count, capacity = int(count), int(capacity)
        require(start == expected_start, name + " exact next start", checks)
        require(end - start + 1 == byte_count, name + " range arithmetic", checks)
        expected_start = end + 1
        byte_total += byte_count
        physical_macros += count
        if placement == "foundry_macro_internal":
            internal_bytes += byte_count
            require((name, macro, count, capacity, area) ==
                    ("parent_scratch", "TS1N28HPCPHVTB128X128M4S", 9, 2048, "true"),
                    "only live internal parent mapping", checks)
        else:
            external_bytes += byte_count
            require(placement == "identical_external_common_charge" and
                    macro == "NONE" and count == 0 and capacity == 0 and area == "false",
                    name + " external not physical", checks)
            require(axes == "candidate,strongest_zero,same_coordinate_bit",
                    name + " identical three-axis charge", checks)
        by_name[name] = byte_count
    require((expected_start, byte_total, internal_bytes, external_bytes,
             physical_macros, 245760 - byte_total) ==
            (214912, 214912, 18432, 196480, 9, 30848),
            "exact once total and margin", checks)
    require(by_name == {"parent_scratch": 18432, "psum_store": 122880,
                        "weight_store": 49152, "metadata_reserve": 24448},
            "class byte split", checks)

    wrapper = WRAPPER.read_text()
    m935 = M935.read_text()
    header_start = m935.index("module m935_m912_three_stage_exact_parent_match_product_capture_island")
    header = m935[header_start:m935.index(");", header_start) + 2]
    frozen_ports = re.findall(
        r"\b(?:input|output)\s+logic(?:\s+\[[^\]]+\])?\s+([A-Za-z_][A-Za-z0-9_]*)", header)
    inst_start = wrapper.index("m935_m912_three_stage_exact_parent_match_product_capture_island u_frozen_m935")
    instance = wrapper[inst_start:wrapper.index(");", inst_start) + 2]
    connected = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", instance)
    require(len(frozen_ports) == 59 and len(connected) == 59 and
            len(set(connected)) == 59 and set(connected) == set(frozen_ports),
            "frozen M935 exact 59-port mapping", checks)
    require(wrapper.count("u_frozen_m935") == 1,
            "one frozen M935 instance", checks)
    require("TS1N28HPCPHVTB128X128M4S" not in wrapper and
            "m528_dw1rw_parent_scratch_9x128_macro" not in wrapper,
            "no wrapper dummy or duplicate macro", checks)

    # First-beat atomicity is arithmetically true for all ready combinations,
    # but both output valids depend combinationally on the peer ready.
    truth = []
    for weight_ready in (0, 1):
        for psum_ready in (0, 1):
            weight_valid = int(bool(psum_ready))
            psum_valid = int(bool(weight_ready))
            weight_fire = weight_valid & weight_ready
            psum_fire = psum_valid & psum_ready
            truth.append({"weight_ready": weight_ready, "psum_ready": psum_ready,
                          "weight_valid": weight_valid, "psum_valid": psum_valid,
                          "weight_fire": weight_fire, "psum_fire": psum_fire})
            require(weight_fire == psum_fire,
                    "first-beat simultaneous fire truth row", checks)
    require("weight_read_request_valid = issue_request_valid" in wrapper and
            "&& (!issue_request_first || psum_read_request_ready)" in wrapper and
            "psum_read_request_valid = issue_request_valid" in wrapper and
            "&& weight_read_request_ready" in wrapper,
            "peer-ready-to-valid dependencies present", checks)

    # A legal ready/valid sink is allowed to compute ready from valid.  The
    # frozen contract does not forbid that.  One simple pair of such sinks,
    # weight_ready=!weight_valid and psum_ready=psum_valid, has no stable
    # first-beat solution under the cross-gating.
    fixed_points = []
    for weight_valid in (0, 1):
        for psum_valid in (0, 1):
            weight_ready = 1 - weight_valid
            psum_ready = psum_valid
            if weight_valid == psum_ready and psum_valid == weight_ready:
                fixed_points.append((weight_valid, psum_valid))
    require(len(fixed_points) == 0,
            "legal valid-dependent-ready environment exposes no fixed point", checks)

    # With a one-cycle external response and exactly one outstanding request,
    # request/data accepts recur every two edges.  This is not a functional
    # mismatch, but it is a throughput-changing wrapper and cannot inherit the
    # raw CPU recurrence without a new joint replay.
    request_accept_edges = [0]
    for _ in range(4):
        request_accept_edges.append(request_accept_edges[-1] + 2)
    require(request_accept_edges == [0, 2, 4, 6, 8],
            "single-outstanding one-cycle-response request II=2 lower bound", checks)
    require("logic service_outstanding_q;" in wrapper and
            "logic service_first_q;" in wrapper and
            "response_accept_w" in wrapper and
            "service_outstanding_q <= 1'b0" in wrapper,
            "single outstanding implementation present", checks)

    # The sticky test itself does not false-trigger during a normal M935
    # transaction: M935 holds issue_request_valid until issue_accept_w, which
    # is exactly the joined response acceptance.  It is therefore not the P0.
    require("issue_request_valid = exec_active_q && active_ctx_valid_q" in m935 and
            "issue_accept_w = issue_data_valid && issue_data_ready" in m935,
            "M935 request held through joined issue accept", checks)
    require("service_outstanding_q && !issue_request_valid" in wrapper,
            "sticky cancellation guard present", checks)

    file_members = [x.strip() for x in FILELIST.read_text().splitlines()
                    if x.strip() and not x.lstrip().startswith("#")]
    require(len(file_members) == 3 and all(not any(token in x.lower()
            for token in ("tb_", "sva", "assert", "attack", "unit_delay"))
            for x in file_members), "synthesis-only exact filelist", checks)
    sdc = SDC.read_text()
    for forbidden in ("set_false_path", "set_multicycle_path", "set_disable_timing",
                      "set_case_analysis", "set_max_delay", "set_min_delay"):
        require(not re.search(r"(?m)^\s*" + forbidden + r"\b", sdc),
                "SDC no " + forbidden, checks)
    tcl = TCL.read_text()
    for token in ("external_common_charge_area_modeled=false",
                  "external_common_charge_area_um2=UNMODELED_EXCLUDED",
                  "full_214912B_total_area_um2=NOT_ADMITTED",
                  "external_common_charge_physical_macros=0"):
        require(token in tcl, "Tcl honest label " + token, checks)
    require("set expected_macro_count 93" not in tcl and
            "set expected_macro_count 105" not in tcl,
            "Tcl no dummy macro target", checks)

    # Bounded mutations: every one must violate an independently checked gate.
    mutations = {
        "mapping_gap": MAPPING.read_text().replace("psum_store|18432", "psum_store|18433", 1),
        "mapping_parent_count": MAPPING.read_text().replace("|9|2048|9|", "|10|2048|9|", 1),
        "mapping_external_macro": MAPPING.read_text().replace("psum_store|18432|141311|122880|identical_external_common_charge|NONE|0|0", "psum_store|18432|141311|122880|identical_external_common_charge|FAKE|60|2048", 1),
        "wrapper_duplicate_macro": wrapper + "\nTS1N28HPCPHVTB128X128M4S dummy();\n",
        "wrapper_missing_port": wrapper.replace(".count_row_completions(count_row_completions)", "", 1),
        "tcl_fake_full_area": tcl.replace("full_214912B_total_area_um2=NOT_ADMITTED", "full_214912B_total_area_um2=123", 1),
        "sdc_false_path": sdc + "\nset_false_path -from [all_inputs]\n",
        "docs359_mutation": DOC359.read_text() + "\nMUTATION\n",
    }
    require("psum_store|18433" in mutations["mapping_gap"], "attack mapping gap built", checks)
    require("|10|2048|9|" in mutations["mapping_parent_count"], "attack parent count built", checks)
    require("|FAKE|60|2048" in mutations["mapping_external_macro"], "attack external macro built", checks)
    require("dummy();" in mutations["wrapper_duplicate_macro"], "attack dummy built", checks)
    require(".count_row_completions" not in mutations["wrapper_missing_port"], "attack missing port built", checks)
    require("NOT_ADMITTED" not in mutations["tcl_fake_full_area"], "attack fake area built", checks)
    require("set_false_path" in mutations["sdc_false_path"], "attack false path built", checks)
    require(hashlib.sha256(mutations["docs359_mutation"].encode()).hexdigest() != EXPECTED["docs359"],
            "attack docs359 built", checks)
    attacks.extend(mutations)

    result = {
        "schema": "m1160_m1116c_independent_source_hammer_mechanical_v1",
        "status": "STOP_M1160_M1116C_WRAPPER_PROTOCOL_UNCLOSED__ACCOUNTING_SOURCE_VALID",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "identity": {key: digest(path) for key, path in paths.items()},
        "capacity": {
            "parent_internal_bytes": 18432,
            "psum_external_bytes": 122880,
            "weight_external_bytes": 49152,
            "metadata_reserve_external_bytes": 24448,
            "represented_bytes": 214912,
            "budget_bytes": 245760,
            "margin_bytes": 30848,
            "physical_parent_macros": 9,
            "external_physical_macros": 0,
        },
        "port_binding": {
            "frozen_m935_ports": len(frozen_ports),
            "connected_once": len(connected),
            "exact_set": set(connected) == set(frozen_ports),
        },
        "protocol": {
            "first_beat_fire_atomic_truth_table": truth,
            "fire_atomic_if_boolean_network_settles": True,
            "request_valid_depends_on_peer_ready": True,
            "contract_forbids_valid_dependent_ready": False,
            "counterexample_valid_dependent_ready_has_fixed_point": False,
            "sticky_outstanding_without_issue_valid_false_positive_on_normal_m935_flow": False,
            "single_outstanding_minimum_ii_at_one_cycle_response": 2,
            "inherits_raw_cpu_recurrence": False,
        },
        "tcl_claim_boundary": {
            "external_common_charge_area_unmodeled_excluded": True,
            "full_214912B_total_area_not_admitted": True,
            "dummy_macro_target": False,
        },
        "authorization": {
            "accounting_mapping_may_be_reused": True,
            "wrapper_protocol_repair_source_only_next": True,
            "vcs_now": False,
            "dc_now": False,
            "pt_fm_ptpx_now": False,
        },
        "docs359_sha256": EXPECTED["docs359"],
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
