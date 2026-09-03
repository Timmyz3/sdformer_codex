#!/usr/bin/python3.12
"""UNSEALED M2063 source-only parser for a new mapped-energy attempt.

M2063 combines race-free reset release with explicitly disclosed deterministic
binary-zero register initialization for zero-delay four-state X-pessimism.  It
is not an M2061 retry,
silicon power-on model, gate-delay repair, or mapped-equivalence result.  Source
admission fails until a sealed contract and independent source hammer exist.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
CONTRACT = HW / "contracts/m2063_m2056_m2018_tsbg_init0_mapped_energy_source_contract_r1_20260903.json"
TOP = "tb_m2063_m2018_tsbg_matched_mapped_energy"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2063_m2018_tsbg_matched_mapped_energy.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2056_REVIEW = HW / "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903"
M2058_FAILURE_REVIEW = HW / "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903"
M2061_FAILURE_REVIEW = HW / "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903"
M2066_SOURCE_REVIEW = HW / "reviews/m2066_m2063_m2056_tsbg_init0_mapped_energy_runner_source_hammer_r1_20260903"
RUNNER = HW / "dc_handoff/scripts/run_m2063_m2056_m2018_tsbg_init0_mapped_energy_one_shot.py"

AXIS_ORDER = ("ordinary_lru4", "tsbg_b4")
AXES = {
    "ordinary_lru4": {
        "filelist": HW / "dc_handoff/filelists/iscas_m2063_m2018_tsbg_ordinary_mapped_energy.f",
        "ucli": HW / "dc_handoff/scripts/m2063_m2018_tsbg_ordinary_mapped_energy.ucli.tcl",
        "netlist": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
        "sdc": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc",
        "design": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE0",
        "scope": TOP + ".core.dut_base.g_mapped.mapped_implementation",
        "cycles": 20292,
        "scalar_weight_reads": 14304,
        "end_marker": "M2063_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge global_slot=42 measurement_cycles=20292",
    },
    "tsbg_b4": {
        "filelist": HW / "dc_handoff/filelists/iscas_m2063_m2018_tsbg_tsbg_mapped_energy.f",
        "ucli": HW / "dc_handoff/scripts/m2063_m2018_tsbg_tsbg_mapped_energy.ucli.tcl",
        "netlist": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v",
        "sdc": HW / "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc",
        "design": "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_SCHEDULE_MODE1",
        "scope": TOP + ".core.dut_tsbg.g_mapped.mapped_implementation",
        "cycles": 7569,
        "scalar_weight_reads": 4608,
        "end_marker": "M2063_SAIF_WINDOW_END axis=tsbg_b4 sampling=settled_negedge global_slot=42 measurement_cycles=7569",
    },
}

BEGIN_MARKER = ("M2063_SAIF_WINDOW_BEGIN sampling=settled_negedge "
                "global_slot=42 m2047_anchor_slot=0 sample=0 layer=28 "
                "is_fc2=0 token_start=0 source_groups=48 preload_cycles=383")
PASS_PREFIX = "PASS_M2063_EP34_TSBG_FULL40_CYCLE_M2051_EQUIVALENT "
EXPECTED_PASS = {
    "m2051_equivalent": "true", "reset_release": "negedge",
    "deterministic_init": "zero", "workload_slot": "42",
    "sample_id": "0", "layer": "28",
    "is_fc2": "0", "token_start": "0", "source_groups": "48",
    "physical_groups": "48", "rows": "149", "issues": "1278",
    "products": "29472", "commits": "24", "base_cycles": "20292",
    "tsbg_cycles": "7569", "bundles_base": "1788",
    "bundles_tsbg": "576", "scalar_base": "14304",
    "scalar_tsbg": "4608", "stale": "1", "retired_replay": "1",
    "replay_accept": "0", "reset": "2", "recovery": "1",
    "real_weights": "false", "system_speedup": "false",
}

SOURCE_SHA256 = {
    "dc_handoff/filelists/iscas_m2063_m2018_tsbg_ordinary_mapped_energy.f": "9dd309e38fb3506126195400e5e7d18bb7b857643ec266647f6993f343473246",
    "dc_handoff/filelists/iscas_m2063_m2018_tsbg_tsbg_mapped_energy.f": "7dc12eea46672f4e9f94e91681e2dbf07ae5efdf4b81896ca7f5f7881530edde",
    "dc_handoff/scripts/m2063_m2018_tsbg_ordinary_mapped_energy.ucli.tcl": "bf8089bf65505d50207c78613aca551996bef3aae55647a9a969f6eee8ffbe57",
    "dc_handoff/scripts/m2063_m2018_tsbg_tsbg_mapped_energy.ucli.tcl": "c3357b2c474063179727101675169e094eccb0ea104ad11e7dff17976bce72b8",
    "dc_handoff/scripts/run_ptpx_m2063_m2018_tsbg_matched_mapped_energy.tcl": "d20b5df8b66166ea91ae82be9b7ef4ba9b7c491e78b2a633a13f69f73e57e91e",
    "tb_m2018/tb_m2063_ep34_tsbg_full40_cycle.sv": "d297c4419075035516e21e084f97c061ae4d82d1fbea563cae8978044bca5246",
    "tb_m2018/tb_m2063_m2018_tsbg_matched_mapped_energy.sv": "9ac794347ccb40b2f56c7b1dfc87737d1a9da80d36554a3b15a7c524f63b9619",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v": "f5847f355329a52511ab044ef458284a19ae424ac778418a4bc4778bb2d3a2b0",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/ordinary_lru4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc": "46b4bd73ace0cfb67f7794321f641ebfabfc0cabd542776ed586d65438970838",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.v": "739eb76dcb732ec0c66b75392c768cbe36027ecc5d458bd4b088f8488f67c9af",
    "dc_handoff/runs/m2029_m2018_c2_tsbg_b4_divfree_matched_two_axis_logic_only_dc_r1_20260902/tsbg_b4/netlist/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend_mapped.sdc": "c7b894cee479badcca22977b29d6ba69a20ca85d9b20e402c9c46ad92ed16d70",
    "rtl_m2018/m2056_m2018_matched_mapped_axis_adapter.sv": "5c84f5f8c61b7f48f3560b54a34b3a1df669421a16a15255fe206db9239a7fcd",
    "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv": "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv": "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
    "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv": "64805bdedb7c80d5c6141bc36e59ef61234507b40942e69ccbf4a30ac2383436",
    "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh": "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh": "70810fdf3ac4ba2d281d750995810f08561addb50871550aa83343a2a04a6dca",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/sim_slot42.log": "5e2e0e72c119815901449737e1f1440275cf0e922b74d123060119fd52c6806f",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/result.json": "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/SHA256SUMS": "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971",
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/SHA256SUMS.seal.sha256": "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/RUN_COMPLETE.txt": "78aaa244c365a8e171996fa079375a0d14a4fc1b14348f45821cd4a5e027367d",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/input_manifest.sha256": "0416ad6fe5b2ff1cbc3f900bc80e330a2df356e6b1a22a1947f54c0c7e1d5bf4",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/review.json": "9d2c98fbe80c4eaebcce60109ec9d795cbb10d78127605be3c5b65e4ca0bd76f",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/review.md": "ca5e2ac44161219a2d66988e335f4f9ed6142dd08566fb36173157871dd6b9b4",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/SHA256SUMS": "bc308ec9dfd27afd6e00b231c54461c9628eec5678768c9613db33425d7ea9c2",
    "reviews/m2056_m2054_m2018_tsbg_matched_mapped_energy_successor_source_hammer_r1_20260903/SHA256SUMS.seal.sha256": "2f5a193dbd79728a1b3dd75e7889accaedfd1bd750610f69a731dcc9a66b2c86",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/RUN_COMPLETE.txt": "13cface773803a8d9827367ffb7ad4b68e3235ab1509ea2d17bf4b39ba0bb165",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/input_manifest.sha256": "8547756720def4ee9c4b17b747e6e3dec441a4a1eceb95771b5a5b79ff54e86f",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/review.json": "05b714994e5222929cc6cd7829c92806a77c18fa63ae68cde1d7e3e6f35fdbda",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/review.md": "a5e38c697601692bc47ac822a7a1d8e24c81ea30bbc9b05170f6b46a7ab7c6c5",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/SHA256SUMS": "f7bbdbfea903814071a5c05cdcd6fcbbe5303f7511fc72753f095f7cd25da93b",
    "reviews/m2058_m2056_tsbg_matched_mapped_energy_failure_hammer_r1_20260903/SHA256SUMS.seal.sha256": "a938562e1f5399dc08313e13c1c1e4b2383b7b969eeb74d2bf5d92e0e37ce495",
    "results/.m2058_m2056_tsbg_matched_mapped_energy_attempt_consumed/attempt.json": "c33e9f86b640f6eab7f637973a23c6741da259df9650f37a3620b9a20ef84c2e",
    "results/.m2058_m2056_tsbg_matched_mapped_energy_attempt_consumed/SHA256SUMS": "06cba5221dd54556ee50cde3f43fc2b2ffee096ec386ceac4699cbdb50bffc37",
    "results/.m2058_m2056_tsbg_matched_mapped_energy_attempt_consumed/SHA256SUMS.seal.sha256": "c72f5f9cb9a9bfe2837733ef78a33f14bad713892fc40b8f7898976cea05cff7",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/RUN_COMPLETE.txt": "1b4933546469d07224222094eaf35b5cce778d9c7978aad7f7eb4157337ef395",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/input_manifest.sha256": "e29ab3449191e359a9946311b160ec4692d3da42012fb3bb6d89c6615aecf5a8",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/review.json": "f9c96c89b3c6e8f81dedd9291b90213ce065eb8493e8ad61348ca0a5d8e69f2b",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/review.md": "3b6c48935979cbf8b2d0f1c0d24928403f11de046a4cbe0f9da55b1e1bda247b",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/SHA256SUMS": "59ef2322ee737dbbb459a650ea64d739f7b8c208df9b268000596f9d3b992f5b",
    "reviews/m2061_m2056_tsbg_settled_mapped_energy_failure_hammer_r1_20260903/SHA256SUMS.seal.sha256": "5e6df2b4ffd1834997c7112b2b40818a6304814a0fb618fba3cfca87107599e4",
    "results/.m2061_m2056_tsbg_settled_mapped_energy_attempt_consumed/attempt.json": "3eb994c1772e38cbae644e660a95fd90df0cab68499f3c9ac766bb5fec20ca60",
    "results/.m2061_m2056_tsbg_settled_mapped_energy_attempt_consumed/SHA256SUMS": "c0232c3aa31c8fe05c6cbfd6b0c9e395567921bc7dc376aa011a4e71c2b74286",
    "results/.m2061_m2056_tsbg_settled_mapped_energy_attempt_consumed/SHA256SUMS.seal.sha256": "37eed57a9f829e4b73e9c2cf032fc686576d42bcb66629bb69f8d9bcba91abde",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

POWER_FIELDS = ("Net Switching Power", "Cell Internal Power",
                "Cell Leakage Power", "Total Power")
CLAIM_BOUNDARY = {
    "workload": "single_pre_registered_ep34_G48_component_workload",
    "selection_uses_performance": False,
    "weights": "deterministic_directed_INT8_not_checkpoint_weights",
    "logic_scope": "mapped_standard_cells_only",
    "external_weight_sram_included": False,
    "power_corner": "TT_0p9V_25C",
    "power_mode": "averaged_prelayout",
    "clock_network": "ideal_no_cts",
    "wireload": "ZeroWireload",
    "macro_count": 0,
    "mapped_simulation": "zero_delay_functional_no_SDF",
    "unit_delay_fix_claimed": False,
    "deterministic_register_initialization": "VCS_compile_random_runtime_zero",
    "silicon_power_on_claimed": False,
    "reset_release": "negedge_all_three_sites",
    "reset_phase_only_fix": False,
    "checker_protocol": "settled_negedge_unconditional_qualifiers_valid_gated_sidebands_per_signal_diagnostics",
    "system_speedup": False,
    "paper_ppa_ready": False,
}


class Failure(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "missing/symlink " + str(path))
    need(sha(path) == digest, "identity drift " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_double_sealed_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    need(outer.is_file() and not outer.is_symlink(), "outer seal absent")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content " + str(root))
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts,
             "unsafe manifest member")
        name = rel.as_posix()
        need(name not in mapping, "duplicate manifest member")
        exact(root / rel, fields[0])
        mapping[name] = fields[0]
    actual = set()
    for member in root.rglob("*"):
        need(not member.is_symlink(), "symlink in sealed review/result")
        if member.is_file() and member.name not in {
                "SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(member.relative_to(root).as_posix())
    need(actual == set(mapping), "non-exhaustive directory seal")
    return mapping


def validate_sources():
    need(SOURCE_SHA256, "M2063_DRAFT_NO_FROZEN_SOURCE_INVENTORY")
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(),
         "M2063_DRAFT_NO_SEALED_CONTRACT")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract outer seal")
    for rel, digest in SOURCE_SHA256.items():
        exact(HW / rel, digest)
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2063_m2056_m2018_tsbg_init0_mapped_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__INDEPENDENT_M2066_REVIEW_REQUIRED_BEFORE_ONE_EXECUTION__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary")
    need(contract.get("execution_budget") == {
        "license_preflight_lmstat": 1, "vcs_compiles": 2,
        "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2,
        "p1_serial": True, "automatic_retry": False,
        "reuse_prior_simv_saif_ptpx": False}, "execution budget")
    source_rows = contract.get("m2063_frozen_sources", [])
    need(type(source_rows) is list and len(source_rows) == len(SOURCE_SHA256),
         "contract source inventory cardinality")
    need(all(type(row) is dict and set(row) == {"path", "sha256"}
             for row in source_rows), "contract source row schema")
    inventory = {row["path"]: row["sha256"] for row in source_rows}
    need(len(inventory) == len(source_rows), "contract duplicate source path")
    need(inventory == SOURCE_SHA256, "contract frozen-source inventory")
    for review_dir, status_prefix in (
            (M2056_REVIEW, "PASS_M2056_"),
            (M2058_FAILURE_REVIEW, "PASS_M2058_"),
            (M2061_FAILURE_REVIEW, "PASS_M2061_")):
        sealed = verify_double_sealed_directory(review_dir)
        need(sealed.get("review.json") == sha(review_dir / "review.json"),
             "review json not sealed")
        review = strict_json(review_dir / "review.json")
        need(review.get("status", "").startswith(status_prefix),
             "review status " + str(review_dir))
    failure_review = strict_json(M2058_FAILURE_REVIEW / "review.json")
    need(failure_review.get("claim_boundary", {}).get("m2058_no_retry") is True,
         "M2058 no-retry boundary")
    m2061_failure = strict_json(M2061_FAILURE_REVIEW / "review.json")
    need(m2061_failure.get("claim_boundary", {}).get("m2061_no_retry") is True
         and m2061_failure.get("claim_boundary", {}).get(
             "m2063_execution_authorized") is False,
         "M2061 no-retry/source-only successor boundary")
    m2066_sealed = verify_double_sealed_directory(M2066_SOURCE_REVIEW)
    need(m2066_sealed.get("review.json") ==
         sha(M2066_SOURCE_REVIEW / "review.json"),
         "M2066 source review json not sealed")
    m2066_review = strict_json(M2066_SOURCE_REVIEW / "review.json")
    identity = m2066_review.get("reviewed_draft_identity", {})
    need(m2066_review.get("status", "").startswith("PASS_M2066_")
         and identity.get("contract_sha256") == sha(CONTRACT)
         and identity.get("runner_sha256") == sha(RUNNER)
         and identity.get("parser_sha256") == sha(Path(__file__).resolve()),
         "M2066 review is stale for corrected compile/runtime initreg sources")
    for axis in AXIS_ORDER:
        cfg = AXES[axis]
        lines = cfg["filelist"].read_text().splitlines()
        need(sum("_mapped.v" in row for row in lines) == 1,
             "mapped netlist cardinality " + axis)
        need(str(cfg["netlist"]) in lines, "mapped netlist mismatch " + axis)
        need(any(row.endswith("/tb_m2063_ep34_tsbg_full40_cycle.sv")
                 for row in lines)
             and any(row.endswith(
                 "/tb_m2063_m2018_tsbg_matched_mapped_energy.sv")
                 for row in lines), "M2063 TB identity " + axis)
        need(not any("/tb_m2051_" in row or "/tb_m2061_" in row
                     for row in lines), "frozen/predecessor TB reuse " + axis)
        ucli = cfg["ucli"].read_text()
        need(ucli.count("\nrun\n") + int(ucli.startswith("run\n")) == 3,
             "UCLI run count " + axis)
        need(ucli.count("power -enable") == 1
             and ucli.count("power -disable") == 1
             and ucli.count("power -report") == 1
             and ucli.count(cfg["scope"]) == 2,
             "UCLI scope/commands " + axis)
    tb = (HW / "tb_m2018/tb_m2063_m2018_tsbg_matched_mapped_energy.sv").read_text()
    need(tb.count("$stop;") == 2, "two-stop TB")
    need("always @(negedge core.clk_core)" in tb and "#0.01;" in tb,
         "settled-negedge monitor")
    need("require_known" in tb and "$isunknown({" not in tb,
         "per-signal diagnostics/no grouped unknown check")
    base_tb = (HW / "tb_m2018/tb_m2063_ep34_tsbg_full40_cycle.sv").read_text()
    need(re.search(r"(?m)^\s*force\b", base_tb) is None
         and re.search(r"(?m)^\s*release\b", base_tb) is None,
         "M2063 force seam forbidden")
    need(base_tb.count("@(negedge clk_core);\n        rst_core = 0;") == 3,
         "all three reset releases must be negedge")
    need(base_tb.count(
        "PASS_M2063_EP34_TSBG_FULL40_CYCLE_M2051_EQUIVALENT") == 1,
         "M2051-equivalent PASS identity")
    return {"status": "PASS_M2063_STATIC_SOURCE_IDENTITY",
            "frozen_sources": len(SOURCE_SHA256), "axes": list(AXIS_ORDER)}


def parse_command_log(path, axis):
    text = Path(path).read_text(errors="strict")
    need(not re.search(r"(?im)^\s*(?:Error-\[|Fatal:)", text),
         "compile fatal/error " + axis)
    rows = re.findall(r"^M2063_COMPILE_COMMAND_JSON=(.+)$", text,
                      flags=re.MULTILINE)
    need(len(rows) == 1, "compile command record " + axis)
    command = json.loads(rows[0])
    cfg = AXES[axis]
    need(command.count("-top") == 1 and command[command.index("-top") + 1] == TOP,
         "compile top " + axis)
    need(command.count("-f") == 1
         and command[command.index("-f") + 1] == str(cfg["filelist"]),
         "compile filelist " + axis)
    need(command.count("+vcs+initreg+random") == 1
         and not any(item.startswith("+vcs+initreg+")
                     and item != "+vcs+initreg+random" for item in command),
         "compile initreg-random enable command " + axis)
    need("+define+UNIT_DELAY" not in command,
         "zero-delay command unexpectedly defines UNIT_DELAY " + axis)
    return {"axis": axis, "log_sha256": sha(path), "command": command}


def parse_runtime(path, axis):
    cfg = AXES[axis]
    text = Path(path).read_text(errors="strict")
    rows = re.findall(r"^M2063_SIM_COMMAND_JSON=(.+)$", text,
                      flags=re.MULTILINE)
    need(len(rows) == 1, "runtime command record " + axis)
    command = json.loads(rows[0])
    need(command.count("+vcs+initreg+0") == 1
         and not any(item.startswith("+vcs+initreg+")
                     and item != "+vcs+initreg+0" for item in command),
         "runtime initreg-zero command " + axis)
    need([item for item in command[1:] if item.startswith("+")] ==
         ["+vcs+initreg+0", "+WORKLOAD_SLOT=42"],
         "runtime plusarg surface " + axis)
    need(not re.search(r"(?im)(?:Fatal:|\$fatal|Assertion failed|M2063 mapped X/Z)",
                       text), "runtime fatal/XZ " + axis)
    need(text.count(BEGIN_MARKER) == 1, "first stop marker " + axis)
    need(text.count(cfg["end_marker"]) == 1, "second stop marker " + axis)
    pass_lines = [row for row in text.splitlines() if row.startswith(PASS_PREFIX)]
    need(len(pass_lines) == 1, "final M2051 PASS count " + axis)
    begin_position = text.index(BEGIN_MARKER)
    end_position = text.index(cfg["end_marker"])
    pass_position = text.index(pass_lines[0])
    need(begin_position < end_position < pass_position,
         "two-stop/final-PASS ordering " + axis)
    fields = {}
    for token in pass_lines[0][len(PASS_PREFIX):].split():
        need(token.count("=") == 1, "PASS token syntax")
        key, value = token.split("=", 1)
        need(key not in fields, "duplicate PASS field " + key)
        fields[key] = value
    need(fields == EXPECTED_PASS,
         "M2063 M2051-equivalent PASS identity/ledger drift " + axis)
    return {"axis": axis, "log_sha256": sha(path), "stop_markers": 2,
            "final_m2051_equivalent_passes": 1, "cycles": cfg["cycles"],
            "scalar_weight_reads": cfg["scalar_weight_reads"]}


def parse_saif(path, axis):
    cfg = AXES[axis]
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         "SAIF regular/nonempty " + axis)
    header = ""
    tx_count = 0
    positive_tc = 0
    mapped_scope = False
    with path.open("r", errors="strict") as handle:
        for index, line in enumerate(handle):
            if index < 256:
                header += line
            mapped_scope = mapped_scope or "mapped_implementation" in line
            for value in re.findall(r"\(TX\s+([0-9.eE+-]+)\)", line):
                tx_count += 1
                need(float(value) == 0.0, "nonzero SAIF TX " + axis)
            positive_tc += sum(float(value) > 0.0 for value in
                               re.findall(r"\(TC\s+([0-9.eE+-]+)\)", line))
    timescale = re.findall(r"\(TIMESCALE\s+([0-9.eE+-]+)\s+([a-zA-Z]+)\)", header)
    duration = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", header)
    need(len(timescale) == 1 and len(duration) == 1, "SAIF header " + axis)
    units = {"s": 1.0e9, "ms": 1.0e6, "us": 1.0e3,
             "ns": 1.0, "ps": 1.0e-3, "fs": 1.0e-6}
    need(timescale[0][1] in units, "SAIF unit " + axis)
    duration_ns = (float(duration[0]) * float(timescale[0][0])
                   * units[timescale[0][1]])
    need(abs(duration_ns - cfg["cycles"] * 3.0) <= 1.0e-6,
         "SAIF duration " + axis)
    need(tx_count > 0 and positive_tc > 0 and mapped_scope,
         "SAIF activity/scope " + axis)
    return {"axis": axis, "saif_sha256": sha(path),
            "duration_ns": duration_ns, "tx_entries": tx_count,
            "nonzero_tx_entries": 0, "positive_tc_entries": positive_tc}


def parse_power_report(path):
    text = Path(path).read_text(errors="strict")
    need("Report : Averaged Power" in text and "-unit mW" in text,
         "averaged mW report")
    values = {}
    for field in POWER_FIELDS:
        hits = re.findall(re.escape(field) + r"\s*=\s*([0-9.eE+-]+)", text)
        need(len(hits) == 1, "unique power field " + field)
        value = float(hits[0])
        need(math.isfinite(value) and value >= 0.0, "power value " + field)
        values[field] = value
    need(values["Total Power"] > 0.0, "positive total power")
    subtotal = sum(values[field] for field in POWER_FIELDS[:3])
    need(abs(subtotal - values["Total Power"])
         <= max(1.0e-6, values["Total Power"] * 1.0e-4), "power subtotal")
    return {"switching_mw": values[POWER_FIELDS[0]],
            "internal_mw": values[POWER_FIELDS[1]],
            "leakage_mw": values[POWER_FIELDS[2]],
            "total_mw": values[POWER_FIELDS[3]]}


def parse_ptpx(root, axis):
    cfg = AXES[axis]
    root = Path(root)
    log_text = (root / "ptpx.log").read_text(errors="strict")
    need(not re.search(r"(?im)^\s*(?:Error:|Fatal:)", log_text),
         "PTPX fatal/error " + axis)
    marker = (root / "PTPX_INTERNAL_COMPLETE.txt").read_text(errors="strict")
    need(marker.count(
        "PASS_M2063_M2018_TSBG_SETTLED_MAPPED_PTPX_PENDING_RESULT_HAMMER") == 1,
        "PTPX marker " + axis)
    need(marker.count("axis=" + axis) == 1
         and marker.count("measurement_cycles=" + str(cfg["cycles"])) == 1,
         "PTPX marker identity " + axis)
    annotation = (root / "reports/saif_annotation_summary.rpt").read_text(
        errors="strict")
    total_net = re.findall(r"Total number of nets = ([0-9]+)", annotation)
    net = re.findall(r"Number of annotated nets = ([0-9]+) \(([0-9.]+)%\)", annotation)
    total_leaf = re.findall(r"Total number of leaf cells = ([0-9]+)", annotation)
    leaf = re.findall(r"Number of fully annotated leaf cells = ([0-9]+) \(([0-9.]+)%\)", annotation)
    need(len(total_net) == len(net) == len(total_leaf) == len(leaf) == 1,
         "annotation parse " + axis)
    need(int(total_net[0]) > 0 and int(net[0][0]) == int(total_net[0])
         and float(net[0][1]) == 100.0 and int(total_leaf[0]) > 0
         and int(leaf[0][0]) == int(total_leaf[0])
         and float(leaf[0][1]) == 100.0, "annotation coverage " + axis)
    boundary = {}
    for row in (root / "reports/scope_and_boundary.rpt").read_text().splitlines():
        need(row.count("=") == 1, "boundary syntax")
        key, value = row.split("=", 1)
        need(key not in boundary, "boundary duplicate")
        boundary[key] = value
    expected = {
        "milestone": "M2063", "axis": axis, "design": cfg["design"],
        "sampling": "settled_negedge_valid_gated_sideband_checker",
        "mapped_simulation": "zero_delay_functional_no_SDF",
        "unit_delay_fix_claimed": "false",
        "deterministic_register_initialization": "VCS_compile_random_runtime_zero",
        "silicon_power_on_claimed": "false",
        "reset_release": "negedge_all_three_sites",
        "reset_phase_only_fix": "false",
        "window_alignment": "first_settled_execute_negedge_to_settled_completion_negedge",
        "first_half_cycle_transition_excluded": "true",
        "analysis": "averaged_prelayout_standard_cell_power",
        "power_corner": "tt0p9v25c", "clock_period_ns": "3.0",
        "measurement_cycles": str(cfg["cycles"]),
        "descriptor_preload_cycles_excluded": "383",
        "workload": "ep34_full40_global_slot42_sample0_layer28_fc1_token0_g48",
        "saif_scope": cfg["scope"], "clock_network": "ideal_no_cts",
        "wireload": "ZeroWireload", "spef": "false", "macro_count": "0",
        "external_weight_sram_excluded": "true",
    }
    for key, value in expected.items():
        need(boundary.get(key) == value, "boundary field " + key + " " + axis)
    need(abs(float(boundary.get("measurement_duration_ns", "nan"))
             - cfg["cycles"] * 3.0) <= 1.0e-6,
         "boundary duration " + axis)
    power = parse_power_report(root / "reports/power.rpt")
    duration_ns = cfg["cycles"] * 3.0
    return {"axis": axis, "ptpx_log_sha256": sha(root / "ptpx.log"),
            "annotation": {"nets": int(total_net[0]), "net_percent": 100.0,
                           "leaf_cells": int(total_leaf[0]), "leaf_percent": 100.0},
            "power": power,
            "execute_energy_pj": {
                "switching": power["switching_mw"] * duration_ns,
                "internal": power["internal_mw"] * duration_ns,
                "leakage": power["leakage_mw"] * duration_ns,
                "total": power["total_mw"] * duration_ns}}


def parse_candidate(candidate, compile_dir):
    candidate = Path(candidate)
    compile_dir = Path(compile_dir)
    axes = {}
    for axis in AXIS_ORDER:
        root = candidate / axis
        axes[axis] = {
            "compile": parse_command_log(compile_dir / (axis + ".compile.log"), axis),
            "runtime": parse_runtime(root / "mapped_sim.log", axis),
            "saif": parse_saif(root / "mapped_execute.saif", axis),
            "ptpx": parse_ptpx(root / "ptpx", axis),
        }
    ordinary_energy = axes["ordinary_lru4"]["ptpx"]["execute_energy_pj"]["total"]
    tsbg_energy = axes["tsbg_b4"]["ptpx"]["execute_energy_pj"]["total"]
    need(ordinary_energy > 0.0 and tsbg_energy > 0.0, "positive energy")
    return {
        "schema": "m2063_m2056_tsbg_init0_mapped_energy_candidate_receipt_r1_v1",
        "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_budget_observed": {
            "license_preflight_lmstat": 1, "vcs_compiles": 2,
            "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2,
            "p1_serial": True, "automatic_retry": False},
        "predecessor": {"m2058_failed_no_retry": True,
                        "m2058_results_reused": False,
                        "m2061_failed_no_retry": True,
                        "m2061_results_reused": False},
        "workload": {"global_slot": 42, "m2047_anchor_slot": 0,
                     "sample_id": 0, "layer_id": 28, "operator": "FC1",
                     "token_start": 0, "source_groups": 48,
                     "descriptor_preload_cycles_excluded": 383,
                     "real_activity_masks": True,
                     "real_checkpoint_weights": False},
        "axes": axes,
        "measured_logic_only_comparison": {
            "cycle_speedup_ordinary_over_tsbg": 20292 / 7569,
            "cycle_reduction_fraction": 1.0 - 7569 / 20292,
            "logic_execute_energy_ratio_ordinary_over_tsbg": ordinary_energy / tsbg_energy,
            "logic_execute_energy_reduction_fraction": 1.0 - tsbg_energy / ordinary_energy},
        "external_weight_sram_symbolic_only": {
            "ordinary_scalar_128b_reads": 14304,
            "tsbg_scalar_128b_reads": 4608,
            "read_reduction_fraction": 1.0 - 4608 / 14304,
            "formula_with_Eread_128b_pJ": {
                "ordinary": "14304 * Eread_128b_pJ",
                "tsbg": "4608 * Eread_128b_pJ"},
            "formula_with_Eread_bit_pJ": {
                "ordinary": "14304 * 128 * Eread_bit_pJ",
                "tsbg": "4608 * 128 * Eread_bit_pJ"},
            "numeric_macro_energy_reported": False,
            "logic_plus_sram_total_numeric_reported": False}}


def validate_sealed_result(root):
    mapping = verify_double_sealed_directory(root)
    need("receipt.json" in mapping, "sealed receipt absent")
    receipt = strict_json(Path(root) / "receipt.json")
    need(receipt.get("status") ==
         "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
         "sealed receipt status")
    need(receipt.get("claim_boundary") == CLAIM_BOUNDARY,
         "sealed receipt boundary")
    return {"status": "PASS_M2063_SEALED_RESULT_STRUCTURE",
            "members": len(mapping), "receipt_sha256": sha(Path(root) / "receipt.json")}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--static", action="store_true")
    group.add_argument("--sealed-result")
    args = parser.parse_args()
    result = validate_sources() if args.static else validate_sealed_result(
        Path(args.sealed_result))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
