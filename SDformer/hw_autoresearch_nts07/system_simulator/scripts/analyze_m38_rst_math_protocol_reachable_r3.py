#!/usr/bin/env python3
"""M38-r3 math, strict protocol, and finite reachable-state reference."""

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import re
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r3_20260822.json"

CONTRACT_TOP_KEYS = {
    "schema", "identity", "supersedes", "claim_boundary", "inputs",
    "independent_review_admissions", "frozen_architecture",
    "canonical_configuration_frame", "offer_schemas", "reachable_state_model",
    "theory_rules",
}
INPUT_KEYS = {
    "m29_config_generator", "m31_vcs_contract", "m31_vcs_receipt",
    "m31_review_validator", "m31_r4_snapshot_ledger",
    "m31_r4_snapshot_admission",
    "m37_math_contract", "m37_math_result", "m37_vcs_contract",
    "m37_vcs_receipt", "m37_r8_frozen_rtl", "m37_r8_snapshot_provenance",
    "m37_r8_snapshot_manifest", "m37_review_validator",
}
EXPECTED_CLAIM = (
    "M38-r3 executable Python3.6 reference for 768 scalar q8-by-ternary pairs, "
    "constructive coverage of every integer rank sum from -384 through 384, exact "
    "Q24 saturation/threshold semantics, strict canonical CRC32C fragment loading, "
    "exact typed offer validation with state-atomic rejection, a complete finite "
    "reachable-state safety graph, directed drain liveness, and conditional kernel "
    "scheduling only. Recursive M31-r4/M37-r8 identity is admitted only through "
    "both hash-bound independent VCS-only review artifacts; those artifacts admit "
    "no DC/STA/Formality/PPA/system claims. Integrated RTL, VCS of "
    "integrated RTL, DC/STA/Formality, PPA, power, energy, memory, trained coverage, "
    "Local/Motion system cycles, speedup, and headline claims remain unadmitted."
)

GOLDEN_CONFIG = {
    "right_factor_q8": [
        -128, -127, -64, -33, -17, -9, -5, -3, -2, -1,
        0, 1, 2, 3, 5, 7, 9, 11, 13, 17,
        23, 31, 47, 63, 79, 95, 111, 120, 126, 127,
    ],
    "left_ternary_code": [0, 1, 2] * 10,
    "bias_q24": [
        -8388608, -4000000, -123456, -1, 0,
        1, 123456, 4000000, 8388606, 8388607,
    ],
    "threshold_q24": -7654321,
    "stage1_requant_shift_u5": 13,
    "generation_u16": 48879,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def exact_keys(payload, expected, label):
    require(isinstance(payload, dict), "{} must be object".format(label))
    require(set(payload) == set(expected), "{} population drift".format(label))


def integer(value, minimum, maximum, label):
    require(isinstance(value, int) and not isinstance(value, bool),
            "{} type violation".format(label))
    require(minimum <= value <= maximum, "{} range violation".format(label))
    return value


def fraction_json(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest(path):
    rows = []
    seen = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed manifest line {}".format(line_no))
        digest, raw = match.groups()
        require(raw not in seen, "duplicate manifest member")
        seen.add(raw)
        rows.append((digest, raw))
    require(rows, "empty manifest")
    return rows


def verify_manifest(path, expected_sha, expected_count, base, substitutions=None):
    path = Path(path)
    require(path.is_file(), "manifest missing: {}".format(path))
    require(sha256(path) == expected_sha, "manifest hash drift: {}".format(path.name))
    rows = read_manifest(path)
    require(len(rows) == expected_count, "manifest member population drift")
    verified = {}
    substitutions = substitutions or {}
    for digest, raw in rows:
        item = Path(raw)
        target = substitutions.get(raw)
        if target is None:
            target = item.resolve() if item.is_absolute() else (Path(base) / item).resolve()
        target = Path(target)
        require(target.is_file(), "manifest member missing: {}".format(raw))
        require(sha256(target) == digest, "manifest member hash drift: {}".format(raw))
        verified[raw] = digest
    return verified


def receipt_pair(pair, label):
    require(isinstance(pair, list) and len(pair) == 2,
            "{} identity shape drift".format(label))
    path = resolve(pair[0])
    require(path.is_file(), "{} missing".format(label))
    require(sha256(path) == pair[1], "{} hash drift".format(label))
    return path


def property_counts(path):
    text = Path(path).read_text(encoding="utf-8")
    return (len(re.findall(r"\bassert\s+property\b", text)),
            len(re.findall(r"\bcover\s+property\b", text)))


def vcs_cover_matches(text):
    return [int(value) for value in re.findall(r"\d+ attempts, (\d+) match", text)]


def load_contract(path=DEFAULT_CONTRACT):
    path = Path(path)
    contract = load_json(path)
    exact_keys(contract, CONTRACT_TOP_KEYS, "M38-r3 contract")
    require(contract["schema"] == "m38_rst_math_input_contract_v3",
            "M38-r3 contract schema drift")
    require(contract["identity"] ==
            "M31_r4_M37_r8_M38_RST_math_strict_protocol_and_reachable_state_reference_only",
            "M38-r3 contract identity drift")
    require(contract["claim_boundary"] == EXPECTED_CLAIM,
            "M38-r3 claim boundary drift")
    require(contract["supersedes"] == {
        "contract": [
            "hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r2_20260822.json",
            "6aedab8129034c490b9914592b1815878a2395a1b3c9964c7174009fbf28f5dc"],
        "result": [
            "hw_autoresearch_nts07/results/m38_rst_math_crc_and_cycle_r2_20260822/m38_rst_math_crc_and_cycle.json",
            "6065f9662bd864eb3162d080e36f9f4b83881f665b62b3193cdf8410e9bab095"],
        "state": "NO_GO_STALE_SUPERSEDED_DO_NOT_CITE",
        "reasons": [
            "r2 cannot rebuild after live M31/M37 source drift",
            "r2 overclaimed all legal rank triples rather than proving scalar coverage plus constructive rank-sum coverage",
            "r2 generalized safety and liveness from a 578-case credit projection that admitted reserved values outside the reachable reconstruction-phase relation",
            "r2 offer validation did not precede every FIFO pop and state installation"],
    }, "M38-r3 supersedes drift")
    for pair_name in ("contract", "result"):
        receipt_pair(contract["supersedes"][pair_name], "M38-r2 {}".format(pair_name))
    exact_keys(contract["inputs"], INPUT_KEYS, "M38-r3 inputs")
    payloads, hashes = {}, {}
    for name, spec in sorted(contract["inputs"].items()):
        exact_keys(spec, {"path", "sha256"}, "M38-r3 input {}".format(name))
        source = resolve(spec["path"])
        require(source.is_file(), "M38-r3 input missing: {}".format(name))
        actual = sha256(source)
        require(actual == spec["sha256"], "M38-r3 input hash drift: {}".format(name))
        text = source.read_text(encoding="utf-8")
        payloads[name] = json.loads(text) if source.suffix == ".json" else text
        hashes[name] = actual
    return contract, payloads, hashes


def validate_frozen_contract(contract, payloads):
    require(contract["frozen_architecture"] == {
        "temporal_rows": 10, "rank": 3, "lanes": 16, "signed_input_bits": 8,
        "stage1_accumulator_bits": 24, "stage1_intermediate_bits": 8,
        "bias_bits": 24, "threshold_bits": 24,
        "shared_signed_int8_multiplier_lanes": 96, "rows_per_phase": 2,
        "phases_per_tile": 5, "result_beats_per_tile": 5,
        "result_fifo_entries": 16, "result_fifo_atomic_credit_per_t10_tile": 5,
        "intermediate_elastic_slots_target": 1, "intermediate_slot_bits": 384,
        "ternary_codes": {"0": 0, "1": 1, "2": -1, "3": "illegal"},
        "t10_factorized_modules_expected_from_m29_interface": 45,
        "t2_dense_fallback_modules_expected_from_m29_interface": 60,
    }, "M38-r3 architecture drift")
    generator = payloads["m29_config_generator"]
    for marker in ('"m29_expected_t10_factorized_modules": 45',
                   '"m29_expected_t2_dense_fallback_modules": 60',
                   '"temporal_factor_rank": 3'):
        require(marker in generator, "M29 interface marker drift")
    frame = contract["canonical_configuration_frame"]
    require(frame["fragment_exact_keys"] == ["data_u64", "index", "valid_bits"]
            and frame["fragment_types"] == {
                "data_u64": "integer_not_boolean",
                "index": "integer_not_boolean",
                "valid_bits": "integer_not_boolean"}
            and frame["fragment_ranges"] == {
                "data_u64": [0, 18446744073709551615], "index": [0, 9],
                "valid_bits_by_index": "64_for_0_through_8_and_48_for_9"},
            "M38-r3 fragment schema drift")
    require(frame["golden_config_sha256"] == canonical_sha256(GOLDEN_CONFIG),
            "M38-r3 golden config drift")
    offer = contract["offer_schemas"]
    require(offer["invalid_offer_atomicity"] ==
            "validate_every_offer_before_fifo_pop_stage1_accept_slot_install_or_any_counter_history_mutation",
            "M38-r3 offer atomicity contract drift")
    reachable = contract["reachable_state_model"]
    require(reachable["reserved_domain"] == [0, 5]
            and reachable["fifo_occupancy_domain"] == [0, 16]
            and reachable["bfs_scope"].startswith("complete_fixpoint"),
            "M38-r3 reachable-state contract drift")
    theory = contract["theory_rules"]
    require(theory == {
        "conditional_t10_steady_ii_serialized": 10,
        "conditional_t10_steady_ii_parallel": 5,
        "conditional_t10_steady_throughput_limit": {"numerator": 2, "denominator": 1},
        "integrated_parallel_cycles_for_n_tiles": "5 + 5*N",
        "serialized_cycles_for_n_tiles": "10*N",
        "finite_n_ratio": "10*N/(5+5*N)",
        "finite_n_regression_values": [1, 2, 3, 32, 100],
        "eventual_sink_stall_regression_cycles": [0, 90, 500],
        "configuration_load_cycles_included": False,
        "result_backpressure_included_in_theory": False,
        "system_speedup_admitted": False, "area_admitted": False,
        "energy_admitted": False,
    }, "M38-r3 theory drift")


def validate_m31(top_contract, payloads, hashes):
    contract = payloads["m31_vcs_contract"]
    receipt = payloads["m31_vcs_receipt"]
    require(contract["contract"] == "m31_unified_t10_t2_vcs_contract_r2",
            "M31-r4 contract schema drift")
    require(receipt["schema"] == "m31_output_receipt_v4"
            and receipt["status"] ==
            "PASS_UNIFIED_T10_T2_STATIC_PHASE_EXACT_FIXED_POINT_SINGLE_SOURCE_MUL96_VCS_NO_DC_FORMALITY_PPA_OR_SYSTEM_CLAIM",
            "M31-r4 receipt identity drift")
    require(receipt["contract"]["sha256"] == hashes["m31_vcs_contract"],
            "M31-r4 recursive contract drift")
    exact_keys(receipt["files"], {
        "multiplier_pool_rtl", "unified_core_rtl", "assertions", "testbench",
        "filelist", "run_script"}, "M31-r4 source receipt")
    snapshot_ledger = resolve(
        top_contract["inputs"]["m31_r4_snapshot_ledger"]["path"])
    snapshot_root = snapshot_ledger.parent / "m31_r4_vcs_inputs_c094849e_20260822"
    require(snapshot_root.is_dir()
            and (snapshot_root.stat().st_mode & 0o777) == 0o555,
            "M31-r4 snapshot directory identity/mode drift")
    snapshot_members = verify_manifest(
        snapshot_ledger, hashes["m31_r4_snapshot_ledger"], 10,
        snapshot_ledger.parent)
    require((snapshot_ledger.stat().st_mode & 0o777) == 0o444,
            "M31-r4 snapshot ledger mode drift")
    snapshot_admission = payloads["m31_r4_snapshot_admission"]
    require(snapshot_admission == {
        "all_live_inputs_rehashed_before_copy": True,
        "all_snapshot_inputs_rehashed_after_copy": True,
        "claim_boundary": {
            "permitted": "immutable byte snapshot of the exact six M31-r4 VCS PASS inputs",
            "forbidden": "new VCS execution, DC/Formality/PPA/power/system claims, or live-source replacement"},
        "core_rtl_sha256":
        "c094849e88c0d9fc3a390d0cf6fc9adf10ff4dc31d77e265e425e5cf71b5ef15",
        "input_count": 6,
        "manifest_sha256":
        "efdc366f86198519f2b58fbd9e155e4aed2f2a8f0785225c6defd719ae8b3093",
        "schema": "m31_r4_vcs_six_input_snapshot_v1",
        "sealer_sha256":
        "ffad1c9f08118928c5f47d35c1910fd8ffc982dda8c7810d0b6710599fb64327",
        "status": "PASS_EXACT_FROZEN_SIX_INPUTS_READ_ONLY"},
        "M31-r4 snapshot admission drift")
    source_hashes = set()
    source_paths = {}
    for name, pair in receipt["files"].items():
        require(isinstance(pair, list) and len(pair) == 2
                and pair[0].startswith("hw_autoresearch_nts07/"),
                "M31-r4 historical source receipt anchor drift")
        relative = pair[0][len("hw_autoresearch_nts07/"):]
        source_paths[name] = snapshot_root / "inputs/hw_root" / relative
        require(source_paths[name].is_file()
                and sha256(source_paths[name]) == pair[1]
                and (source_paths[name].stat().st_mode & 0o777) == 0o444,
                "M31-r4 frozen source drift: {}".format(name))
        source_hashes.add(pair[1])
    snapshot_prefix = snapshot_root.name
    expected_snapshot_members = {
        "{}/input_sha256.txt".format(snapshot_prefix):
        snapshot_admission["manifest_sha256"],
        "{}/snapshot_admission.json".format(snapshot_prefix):
        hashes["m31_r4_snapshot_admission"],
        "{}/source_map.tsv".format(snapshot_prefix):
        "ecb01cef2d186c990628418f191ca0f198adf7ad137091d12f2acd7b5ab0156b",
        "{}/tools/seal_m31_r4_vcs_inputs.py".format(snapshot_prefix):
        snapshot_admission["sealer_sha256"],
    }
    for name, pair in receipt["files"].items():
        relative = pair[0][len("hw_autoresearch_nts07/"):]
        expected_snapshot_members[
            "{}/inputs/hw_root/{}".format(snapshot_prefix, relative)] = pair[1]
    require(snapshot_members == expected_snapshot_members,
            "M31-r4 snapshot ledger population drift")
    for raw in snapshot_members:
        require(((snapshot_ledger.parent / raw).stat().st_mode & 0o777) == 0o444,
                "M31-r4 snapshot member mode drift")
    require(property_counts(source_paths["assertions"]) == (24, 4),
            "M31-r4 SVA population drift")
    run = receipt["vcs_run"]
    run_dir = Path(run["directory"])
    substitutions = {}
    for name, pair in receipt["files"].items():
        raw = pair[0][len("hw_autoresearch_nts07/"):]
        substitutions[raw] = source_paths[name]
    frozen_input_manifest = snapshot_root / "input_sha256.txt"
    require(sha256(frozen_input_manifest) == run["input_sha256_manifest"],
            "M31-r4 frozen input manifest drift")
    inputs = verify_manifest(frozen_input_manifest,
                             run["input_sha256_manifest"], 6, HW_ROOT,
                             substitutions)
    verify_manifest(run_dir / "input_sha256.txt",
                    run["input_sha256_manifest"], 6, HW_ROOT,
                    substitutions)
    outputs = verify_manifest(run_dir / "output_sha256.txt",
                              run["output_sha256_manifest"], 2, run_dir)
    require(set(inputs.values()) == source_hashes, "M31-r4 input manifest population drift")
    require(outputs.get(str(run_dir / "compile.log")) == run["compile_log"]
            and outputs.get(str(run_dir / "sim.log")) == run["sim_log"],
            "M31-r4 output manifest drift")
    compile_text = (run_dir / "compile.log").read_text(encoding="utf-8", errors="replace")
    sim_text = (run_dir / "sim.log").read_text(encoding="utf-8", errors="replace")
    require("Chronologic VCS" in compile_text and "M31_SVA_BOUND=1" in sim_text
            and "SIMULATOR=Synopsys VCS" in sim_text and "ASSERTIONS=enabled" in sim_text
            and "M31_PASS modes=T10_T2N_T2S_T10" in sim_text,
            "M31-r4 VCS marker drift")
    require(vcs_cover_matches(sim_text) == [32, 1, 26, 85],
            "M31-r4 cover drift")
    require(receipt["observed"]["conditional_t10_no_stall_accept_ii"] == 10
            and receipt["observed"]["fifo_full_simultaneous_pop_push_cycles"] == 8
            and receipt["source_revision"]["dynamic_phase_indexed_t10_arrays"] == 0,
            "M31-r4 observed receipt drift")
    return {
        "receipt_schema": receipt["schema"], "receipt_status": receipt["status"],
        "receipt_sha256": hashes["m31_vcs_receipt"],
        "run_basename": run_dir.name, "source_files": 6,
        "input_manifest_sha256": run["input_sha256_manifest"],
        "output_manifest_sha256": run["output_sha256_manifest"],
        "frozen_input_snapshot_ledger_sha256":
        hashes["m31_r4_snapshot_ledger"],
        "frozen_input_snapshot_admission_sha256":
        hashes["m31_r4_snapshot_admission"],
        "frozen_input_snapshot_members": len(snapshot_members),
        "assert_properties": 24, "cover_matches": [32, 1, 26, 85],
        "independent_review_required": True,
    }


def validate_m37(top_contract, payloads, hashes):
    contract = payloads["m37_vcs_contract"]
    receipt = payloads["m37_vcs_receipt"]
    require(contract["contract"] == "m37_csd_reconstruct_t10_vcs_contract_r3",
            "M37-r8 contract schema drift")
    require(receipt["schema"] == "m37_output_receipt_v3"
            and receipt["status"] ==
            "PASS_R8_STANDALONE_T10_CSD_RECONSTRUCTION_SHIFT_ADD_CONTROL_INDEX_VCS_PENDING_INDEPENDENT_REVIEW_NO_DC_OR_SYSTEM_CLAIM",
            "M37-r8 receipt identity drift")
    require(receipt["contract"]["sha256"] == hashes["m37_vcs_contract"],
            "M37-r8 recursive contract drift")
    require(receipt["math_anchor"]["contract"][1] == hashes["m37_math_contract"]
            and receipt["math_anchor"]["result"][1] == hashes["m37_math_result"],
            "M37-r8 math anchor drift")
    exact_keys(receipt["files"], {"rtl", "assertions", "testbench", "filelist", "runner"},
               "M37-r8 source receipt")
    source_paths = {}
    for name, pair in receipt["files"].items():
        if name == "rtl":
            require(isinstance(pair, list) and len(pair) == 2
                    and pair[1] == hashes["m37_r8_frozen_rtl"],
                    "M37-r8 historical RTL receipt anchor drift")
            source_paths[name] = resolve(
                top_contract["inputs"]["m37_r8_frozen_rtl"]["path"])
        else:
            source_paths[name] = receipt_pair(pair, "M37-r8 {}".format(name))
    require(property_counts(source_paths["assertions"]) == (21, 8),
            "M37-r8 SVA population drift")
    run = receipt["vcs_run"]
    run_dir = Path(run["directory"])
    verify_manifest(
        run_dir / "input_sha256.txt", run["input_sha256_manifest"], 8, HW_ROOT,
        {"rtl_m37/qfit_atlif_csd_reconstruct_t10.sv": source_paths["rtl"]})
    outputs = verify_manifest(run_dir / "output_sha256.txt",
                              run["output_sha256_manifest"], 5, run_dir)
    verify_manifest(run_dir / "run_local_seal.sha256", run["run_local_seal"], 3, run_dir)
    for filename, key in (("compile.log", "compile_log"), ("sim.log", "sim_log"),
                          ("vectors.txt", "vectors"),
                          ("rtl_multiplier_intent_audit.txt", "rtl_multiplier_intent_audit"),
                          ("runner_status.txt", "runner_status")):
        require(outputs.get(str(run_dir / filename)) == run[key],
                "M37-r8 output anchor drift: {}".format(filename))
    sim_text = (run_dir / "sim.log").read_text(encoding="utf-8", errors="replace")
    require("M37_SVA_BOUND=1" in sim_text and "SIMULATOR=Synopsys VCS" in sim_text
            and "ASSERTIONS=enabled" in sim_text and "M37_PASS total_tiles=245" in sim_text
            and "no_data_multiplier=1" in sim_text,
            "M37-r8 VCS marker drift")
    require(vcs_cover_matches(sim_text) == [220, 1271, 249, 117, 245, 571, 133, 210],
            "M37-r8 cover drift")
    require(receipt["r8_delta"]["new_rtl_sha256"] == receipt["files"]["rtl"][1]
            and receipt["observed"]["rank3_runtime_control_index_multiply_matches"] == 0,
            "M37-r8 shift-add delta drift")
    snapshot_manifest = resolve(
        top_contract["inputs"]["m37_r8_snapshot_manifest"]["path"])
    snapshot_dir = snapshot_manifest.parent
    snapshot_members = verify_manifest(
        snapshot_manifest, hashes["m37_r8_snapshot_manifest"], 2, snapshot_dir)
    require(snapshot_members == {
        "qfit_atlif_csd_reconstruct_t10.sv": hashes["m37_r8_frozen_rtl"],
        "README.provenance.txt": hashes["m37_r8_snapshot_provenance"]},
        "M37-r8 snapshot ledger population drift")
    require((source_paths["rtl"].stat().st_mode & 0o777) == 0o444,
            "M37-r8 frozen snapshot mode drift")
    provenance = payloads["m37_r8_snapshot_provenance"]
    require("exact historical M37-r8 RTL source" in provenance
            and "does not close r8 Formality" in provenance
            and "SHA256 {}".format(hashes["m37_r8_frozen_rtl"]) in provenance,
            "M37-r8 snapshot provenance boundary drift")
    return {
        "receipt_schema": receipt["schema"], "receipt_status": receipt["status"],
        "receipt_sha256": hashes["m37_vcs_receipt"],
        "run_basename": run_dir.name, "source_files": 5,
        "input_manifest_sha256": run["input_sha256_manifest"],
        "output_manifest_sha256": run["output_sha256_manifest"],
        "frozen_rtl_snapshot_sha256": hashes["m37_r8_frozen_rtl"],
        "snapshot_manifest_sha256": hashes["m37_r8_snapshot_manifest"],
        "snapshot_provenance_sha256": hashes["m37_r8_snapshot_provenance"],
        "assert_properties": 21,
        "cover_matches": [220, 1271, 249, 117, 245, 571, 133, 210],
        "independent_review_required": True,
    }


def load_python_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "review validator import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_review_admissions(contract, payloads, hashes):
    specs = contract["independent_review_admissions"]
    exact_keys(specs, {"m31_r4", "m37_r8"}, "M38-r3 review admissions")
    audits = {}
    all_bound = True
    for name, spec in sorted(specs.items()):
        exact_keys(spec, {"state", "path", "sha256", "expected_schema",
                          "expected_status", "validator_path", "validator_sha256",
                          "required_for_recursive_admission"},
                   "M38-r3 {} review".format(name))
        require(spec["required_for_recursive_admission"] is True,
                "review requirement opened")
        if spec["state"] == "PENDING_NOT_BOUND":
            require(spec["path"] is None and spec["sha256"] is None
                    and spec["expected_schema"] is None
                    and spec["expected_status"] is None
                    and spec["validator_path"] is None
                    and spec["validator_sha256"] is None,
                    "pending review artifact must be null")
            audits[name] = {"state": "PENDING_NOT_BOUND", "admitted": False}
            all_bound = False
            continue
        require(spec["state"] == "BOUND_PASS", "unknown review admission state")
        path = resolve(spec["path"])
        require(path.is_file() and sha256(path) == spec["sha256"],
                "review admission identity drift")
        payload = load_json(path)
        require(payload.get("schema") == spec["expected_schema"]
                and payload.get("status") == spec["expected_status"],
                "review admission schema/status drift")
        validator_path = resolve(spec["validator_path"])
        require(validator_path.is_file() and sha256(validator_path) ==
                spec["validator_sha256"], "review validator identity drift")
        if name == "m31_r4":
            require(hashes["m31_review_validator"] == spec["validator_sha256"],
                    "M31 review validator contract input drift")
            require(payload.get("identity", {}).get("receipt_sha256") ==
                    hashes["m31_vcs_receipt"],
                    "M31 review admission receipt anchor drift")
            require(payload.get("identity", {}).get("validator_sha256") ==
                    spec["validator_sha256"],
                    "M31 review admission validator anchor drift")
            require(payload.get("identity", {}).get("contract_sha256") ==
                    hashes["m31_vcs_contract"], "M31 review contract anchor drift")
            require(payload.get("admission") == {
                "current_r4_vcs_source_admitted": True,
                "dc_sta_admitted": False, "formality_admitted": False,
                "headline_admitted": False,
                "phase_fault_recovery_admitted": False,
                "ppa_power_energy_admitted": False,
                "system_speedup_admitted": False},
                "M31 review admission boundary drift")
            require(payload.get("manifest_audit") == {
                "all_live_content_rehashed": True, "input_count": 6,
                "output_count": 2}, "M31 review manifest audit drift")
            require(payload.get("log_audit", {}).get("assert_property_count") == 24
                    and payload.get("log_audit", {}).get("cover_property_count") == 4
                    and payload.get("log_audit", {}).get("failure_signature_count") == 0,
                    "M31 review log audit drift")
        elif name == "m37_r8":
            require(hashes["m37_review_validator"] == spec["validator_sha256"],
                    "M37 review validator contract input drift")
            exact_keys(payload, {
                "schema", "status", "date", "review", "anchors", "vcs",
                "observed", "independent_source_reaudit", "admitted", "validator",
                "historical_manifest_resolution",
                "review_required_for_any_scope_extension"},
                "M37 review admission")
            anchors = payload.get("anchors", {})
            receipt = payloads["m37_vcs_receipt"]
            run = receipt["vcs_run"]
            require(anchors.get("receipt") == [
                contract["inputs"]["m37_vcs_receipt"]["path"],
                hashes["m37_vcs_receipt"]],
                "M37 review admission receipt anchor drift")
            require(anchors.get("contract") == [
                contract["inputs"]["m37_vcs_contract"]["path"],
                hashes["m37_vcs_contract"]],
                "M37 review admission contract anchor drift")
            require(anchors.get("rtl") == [
                contract["inputs"]["m37_r8_frozen_rtl"]["path"],
                hashes["m37_r8_frozen_rtl"]]
                    and anchors["rtl"][1] == receipt["files"]["rtl"][1],
                    "M37 review admission RTL anchor drift")
            require(anchors.get("snapshot_provenance") == [
                contract["inputs"]["m37_r8_snapshot_provenance"]["path"],
                hashes["m37_r8_snapshot_provenance"]]
                    and anchors.get("snapshot_ledger") == [
                contract["inputs"]["m37_r8_snapshot_manifest"]["path"],
                hashes["m37_r8_snapshot_manifest"], 2],
                    "M37 review snapshot ledger/provenance anchor drift")
            require(anchors.get("input_manifest", [None, None, None])[1:] ==
                    [run["input_sha256_manifest"], 8]
                    and anchors.get("output_manifest", [None, None, None])[1:] ==
                    [run["output_sha256_manifest"], 5]
                    and anchors.get("run_local_seal", [None, None, None])[1:] ==
                    [run["run_local_seal"], 3],
                    "M37 review manifest anchor drift")
            require(payload.get("review") == {
                "independent_of_r8_implementation": True,
                "score_0_to_100": 94, "p0": 0, "p1": 1, "p2": 3,
                "go": "STANDALONE_R8_VCS_AND_EXACT_SHA_BOUND_SOURCE_INTENT_ONLY",
                "nogo": "DC_STA_FORMALITY_PPA_POWER_ENERGY_SYSTEM_HEADLINE"},
                "M37 review decision drift")
            require(payload.get("admitted") == {
                "standalone_r8_vcs_functional": True,
                "exact_sha_bound_source_intent": True,
                "physical_zero_multiplier": False,
                "dc": False, "sta": False, "formality": False,
                "ppa": False, "power": False, "energy": False,
                "system": False, "headline": False},
                "M37 review admission boundary drift")
            require(payload.get("vcs") == {
                "tool": "Synopsys VCS V-2023.12-SP1_Full64",
                "compile_log_sha256": run["compile_log"],
                "sim_log_sha256": run["sim_log"],
                "vectors_sha256": run["vectors"],
                "original_runner_source_intent_audit_sha256":
                run["rtl_multiplier_intent_audit"],
                "runner_status_sha256": run["runner_status"],
                "compile_warning_error_fatal_signatures": 0,
                "assertion_failure_signatures": 0,
                "uncovered_sva_cover_properties": 0},
                "M37 review VCS audit drift")
            require(payload.get("observed") == {
                "seed": "0x4d370203", "tiles": 245, "nominal_tiles": 96,
                "signed_inputs": 256, "input_coefficient_pairs": 65536,
                "product_miters": 117600, "bit_miters": 39200,
                "arithmetic_issues": 1225,
                "illegal_accept_reject": [210, 210],
                "illegal_classes": [30, 30, 30, 30, 30, 30, 30],
                "sva_cover_matches": [220, 1271, 249, 117, 245, 571, 133, 210]},
                "M37 review observed audit drift")
            source = payload.get("independent_source_reaudit", {})
            require(source.get("auditor") == [
                "hw_autoresearch_nts07/dc_handoff/scripts/audit_m37_r8_source_intent.py",
                "6fcf221ac018e38283723b687852e1809941aabdbbfa031dd812da14113cc856"]
                    and source.get("comments_and_strings_removed") is True
                    and source.get("canonical_star_tokens") == 44
                    and source.get("data_multiplication_tokens") == 0
                    and source.get("runtime_non_power_of_two_control_multiplication_tokens") == 0
                    and source.get("rank3_shift_add_matches") == 1
                    and source.get("forged_comment_signature_real_multiply_rejected") is True
                    and source.get("forged_hidden_data_a_times_b_rejected") is True
                    and source.get("forged_selected_row_times_space_rank_rejected") is True
                    and source.get("dut_constant_uses_integer_multiplier_used_as_structure_proof") is False,
                    "M37 independent source reaudit drift")
            require(payload.get("validator") == spec["validator_path"]
                    and payload.get("review_required_for_any_scope_extension") is True,
                    "M37 review validator declaration drift")
            require(payload.get("historical_manifest_resolution") == {
                "input_manifest_original_target":
                "rtl_m37/qfit_atlif_csd_reconstruct_t10.sv",
                "live_r9_may_not_be_used_to_validate_r8": True,
                "reason": "live_rtl_advanced_to_r9_after_r8_review",
                "required_snapshot_mode_octal": "0444",
                "required_snapshot_sha256": hashes["m37_r8_frozen_rtl"],
                "snapshot_ledger_sha256": hashes["m37_r8_snapshot_manifest"],
                "snapshot_provenance_sha256": hashes["m37_r8_snapshot_provenance"],
                "validator_substitution": "anchors.rtl immutable r8 snapshot"},
                "M37 historical manifest resolution drift")
            require(sha256(resolve(source["auditor"][0])) == source["auditor"][1],
                    "M37 independent source auditor identity drift")
            validator = load_python_module(validator_path, "m38_m37_review_validator")
            try:
                validator.validate_payload(payload)
                validator.validate_external(payload, HW_ROOT)
            except Exception as error:
                raise ValueError("M37 review validator failed: {}".format(error))
        else:
            raise ValueError("unknown review admission name")
        audits[name] = {"state": "BOUND_PASS", "admitted": True,
                        "path": spec["path"], "sha256": spec["sha256"],
                        "validator_path": spec["validator_path"],
                        "validator_sha256": spec["validator_sha256"],
                        "schema": spec["expected_schema"],
                        "status": spec["expected_status"]}
    return audits, all_bound


def ternary_product(value, code):
    value = integer(value, -128, 127, "q8 input")
    code = integer(code, 0, 3, "ternary code")
    require(code != 3, "illegal ternary code")
    coefficient = {0: 0, 1: 1, 2: -1}[code]
    return value * coefficient


def constructive_rank3_decomposition(total):
    total = integer(total, -384, 384, "rank3 sum")
    remaining = total
    terms = []
    for _ in range(3):
        term = max(-128, min(128, remaining))
        remaining -= term
        if term == 128:
            terms.append({"q8": -128, "ternary_code": 2, "product": 128})
        else:
            terms.append({"q8": term, "ternary_code": 1, "product": term})
    require(remaining == 0, "constructive rank3 decomposition failed")
    require(sum(ternary_product(row["q8"], row["ternary_code"]) for row in terms) == total,
            "constructive rank3 witness mismatch")
    return terms


def saturate_q24(value):
    return max(-(1 << 23), min((1 << 23) - 1, int(value)))


def build_math_audit():
    scalar_rows = []
    for value in range(-128, 128):
        for code in (0, 1, 2):
            product = ternary_product(value, code)
            scalar_rows.append([value, code, product])
    require(len(scalar_rows) == 768, "scalar ternary population drift")
    products = [row[2] for row in scalar_rows]
    witnesses = []
    for total in range(-384, 385):
        terms = constructive_rank3_decomposition(total)
        witnesses.append({"sum": total, "terms": terms})
    require(len(witnesses) == 769, "rank sum population drift")
    require([row["sum"] for row in witnesses] == list(range(-384, 385)),
            "rank sum coverage gap")
    q24_cases = []
    for rank_sum, bias, threshold in (
            (-384, -(1 << 23), -(1 << 23)),
            (384, (1 << 23) - 1, (1 << 23) - 1),
            (0, 12345, 12345), (0, 12344, 12345),
            (128, (1 << 23) - 64, (1 << 23) - 1),
            (-128, -(1 << 23) + 64, -(1 << 23))):
        raw = bias + rank_sum
        saturated = saturate_q24(raw)
        q24_cases.append({"rank_sum": rank_sum, "bias": bias, "raw": raw,
                          "saturated": saturated, "threshold": threshold,
                          "event": int(saturated >= threshold)})
    return ({
        "statement": "ALL_768_Q8_BY_LEGAL_TERNARY_SCALAR_PAIRS",
        "pairs_checked": 768, "product_range": [min(products), max(products)],
        "minimum_signed_product_bits": 9,
        "negative_minimum_negation_witness": {"q8": -128, "code": 2, "result": 128},
        "rows_sha256": canonical_sha256(scalar_rows), "rows_stored_inline": False,
    }, {
        "statement":
        "EVERY_INTEGER_RANK_SUM_MINUS384_THROUGH_384_HAS_A_MACHINE_VERIFIED_CONSTRUCTIVE_THREE_TERM_DECOMPOSITION",
        "all_legal_rank_triples_exhaustively_checked": False,
        "constructive_rank_sum_values_checked": 769,
        "rank3_sum_range": [-384, 384], "minimum_signed_rank3_sum_bits": 10,
        "constructive_witnesses_sha256": canonical_sha256(witnesses),
        "constructive_witnesses_stored_inline": False,
        "boundary_witnesses": {
            "minus384": witnesses[0]["terms"], "zero": witnesses[384]["terms"],
            "plus384": witnesses[-1]["terms"]},
        "mathematical_minimum_bias_plus_rank_sum_bits": 25,
        "implemented_pre_saturation_bits_target": 26,
        "q24_saturation_threshold_cases": q24_cases,
        "threshold_equality_event": 1, "threshold_just_below_event": 0,
    })


def put_bits(bits, value, width):
    value = int(value) & ((1 << width) - 1)
    bits.extend((value >> index) & 1 for index in range(width))


def bits_to_bytes(bits):
    require(len(bits) % 8 == 0, "bit vector is not byte aligned")
    return bytes(sum(bits[offset + bit] << bit for bit in range(8))
                 for offset in range(0, len(bits), 8))


def crc32c(data):
    crc = 0xFFFFFFFF
    for value in bytearray(data):
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF


def validate_configuration(config):
    exact_keys(config, {"right_factor_q8", "left_ternary_code", "bias_q24",
                        "threshold_q24", "stage1_requant_shift_u5", "generation_u16"},
               "M38 configuration")
    require(isinstance(config["right_factor_q8"], list)
            and len(config["right_factor_q8"]) == 30, "right-factor population drift")
    require(isinstance(config["left_ternary_code"], list)
            and len(config["left_ternary_code"]) == 30, "ternary population drift")
    require(isinstance(config["bias_q24"], list) and len(config["bias_q24"]) == 10,
            "bias population drift")
    for value in config["right_factor_q8"]:
        integer(value, -128, 127, "right-factor q8")
    for code in config["left_ternary_code"]:
        integer(code, 0, 2, "ternary code")
    for value in config["bias_q24"]:
        integer(value, -(1 << 23), (1 << 23) - 1, "bias q24")
    integer(config["threshold_q24"], -(1 << 23), (1 << 23) - 1, "threshold q24")
    integer(config["stage1_requant_shift_u5"], 0, 23, "stage1 shift")
    integer(config["generation_u16"], 0, 0xFFFF, "generation")


def pack_protected_payload(config):
    validate_configuration(config)
    bits = []
    for value in config["right_factor_q8"]:
        put_bits(bits, value, 8)
    for value in config["left_ternary_code"]:
        put_bits(bits, value, 2)
    for value in config["bias_q24"]:
        put_bits(bits, value, 24)
    put_bits(bits, config["threshold_q24"], 24)
    put_bits(bits, config["stage1_requant_shift_u5"], 5)
    require(len(bits) == 569, "arithmetic payload width drift")
    put_bits(bits, config["generation_u16"], 16)
    require(len(bits) == 585, "protected logical width drift")
    bits.extend([0] * 7)
    payload = bits_to_bytes(bits)
    require(len(payload) == 74, "protected payload byte width drift")
    return payload


def pack_configuration_frame(config):
    payload = pack_protected_payload(config)
    return payload + crc32c(payload).to_bytes(4, byteorder="little")


class BitReader(object):
    def __init__(self, payload):
        self.bits = [(value >> bit) & 1 for value in bytearray(payload) for bit in range(8)]
        self.offset = 0

    def take(self, width, signed=False):
        value = sum(self.bits[self.offset + bit] << bit for bit in range(width))
        self.offset += width
        if signed and value & (1 << (width - 1)):
            value -= 1 << width
        return value


def decode_configuration_frame(frame):
    require(isinstance(frame, (bytes, bytearray)), "serialized frame type violation")
    frame = bytes(frame)
    require(len(frame) == 78, "serialized frame must be exactly 624 bits")
    payload = frame[:74]
    require(int.from_bytes(frame[74:], byteorder="little") == crc32c(payload),
            "configuration CRC mismatch")
    reader = BitReader(payload)
    config = {
        "right_factor_q8": [reader.take(8, signed=True) for _ in range(30)],
        "left_ternary_code": [reader.take(2) for _ in range(30)],
        "bias_q24": [reader.take(24, signed=True) for _ in range(10)],
        "threshold_q24": reader.take(24, signed=True),
        "stage1_requant_shift_u5": reader.take(5),
        "generation_u16": reader.take(16),
    }
    require(not any(reader.take(1) for _ in range(7)),
            "configuration zero padding is nonzero")
    validate_configuration(config)
    return config


def make_fragments(frame):
    require(isinstance(frame, (bytes, bytearray)) and len(frame) == 78,
            "fragment source frame width/type drift")
    padded = bytes(frame) + bytes(2)
    return [{"index": index,
             "data_u64": int.from_bytes(padded[index * 8:(index + 1) * 8],
                                        byteorder="little"),
             "valid_bits": 48 if index == 9 else 64}
            for index in range(10)]


def validate_fragment(fragment):
    exact_keys(fragment, {"index", "data_u64", "valid_bits"},
               "configuration fragment")
    index = integer(fragment["index"], 0, 9, "configuration fragment index")
    integer(fragment["data_u64"], 0, (1 << 64) - 1, "configuration fragment data")
    expected = 48 if index == 9 else 64
    valid = integer(fragment["valid_bits"], 0, 64, "configuration fragment valid bits")
    require(valid == expected, "configuration fragment valid-bit violation")
    if index == 9:
        require(fragment["data_u64"] >> 48 == 0,
                "last configuration fragment high bits are nonzero")
    return index


def generation_is_newer(candidate, active):
    candidate = integer(candidate, 0, 0xFFFF, "candidate generation")
    if active is None:
        return True
    active = integer(active, 0, 0xFFFF, "active generation")
    delta = (candidate - active) & 0xFFFF
    return 1 <= delta <= 0x7FFF


class StrictFragmentLoader(object):
    def __init__(self, active_config=None):
        if active_config is not None:
            validate_configuration(active_config)
        self.active_config = copy.deepcopy(active_config)
        self.next_index = 0
        self.shadow = bytearray()
        self.failed = False

    def _fail(self, message):
        self.failed = True
        self.next_index = 0
        self.shadow = bytearray()
        raise ValueError(message)

    def accept(self, fragment, datapath_drained=False):
        try:
            index = validate_fragment(fragment)
        except ValueError as error:
            self._fail(str(error))
        require(isinstance(datapath_drained, bool), "datapath_drained type violation")
        if self.failed:
            if index != 0:
                self._fail("failed load restart requires fragment zero")
            self.failed = False
        if index == 0 and self.next_index != 0:
            self._fail("duplicate or premature nonzero fragment zero")
        if index != self.next_index:
            self._fail("configuration fragment order or duplicate violation")
        self.shadow.extend(fragment["data_u64"].to_bytes(8, byteorder="little"))
        self.next_index += 1
        if index != 9:
            return False
        frame = bytes(self.shadow[:78])
        try:
            candidate = decode_configuration_frame(frame)
            active_generation = (None if self.active_config is None
                                 else self.active_config["generation_u16"])
            require(generation_is_newer(candidate["generation_u16"], active_generation),
                    "configuration generation is stale or ambiguous")
            require(datapath_drained, "configuration activation requires drained datapath")
        except ValueError as error:
            self._fail(str(error))
        self.active_config = candidate
        self.next_index = 0
        self.shadow = bytearray()
        return True

    def finalize_incomplete(self):
        if self.next_index != 0:
            self._fail("incomplete configuration frame")


def frame_with_generation(config, generation):
    changed = copy.deepcopy(config)
    changed["generation_u16"] = generation
    return pack_configuration_frame(changed)


def build_protocol_audit(contract):
    frame_contract = contract["canonical_configuration_frame"]
    payload = pack_protected_payload(GOLDEN_CONFIG)
    frame = pack_configuration_frame(GOLDEN_CONFIG)
    require(crc32c(b"123456789") == 0xE3069283, "CRC32C standard check drift")
    require(crc32c(payload) == int(frame_contract["golden_crc32c"], 16),
            "golden CRC drift")
    require(frame.hex() == frame_contract["golden_serialized_frame_hex"],
            "golden frame drift")
    require(decode_configuration_frame(frame) == GOLDEN_CONFIG,
            "golden round-trip drift")
    fragments = make_fragments(frame)
    loader = StrictFragmentLoader()
    for item in fragments:
        activated = loader.accept(item, datapath_drained=True)
    require(activated and loader.active_config == GOLDEN_CONFIG,
            "golden fragment activation drift")
    rejected = []

    def expect(name, items, active=None, drained=True):
        instance = StrictFragmentLoader(active)
        try:
            for item in items:
                instance.accept(item, datapath_drained=drained)
        except ValueError:
            require(instance.active_config == active,
                    "loader failure changed active context")
            rejected.append(name)
            return instance
        raise ValueError("negative loader case accepted: {}".format(name))

    expect("wrong_valid_nonlast", [dict(fragments[0], valid_bits=63)])
    expect("wrong_valid_last", fragments[:9] + [dict(fragments[9], valid_bits=64)])
    expect("extra_key", [dict(fragments[0], forged=1)])
    expect("boolean_index", [dict(fragments[0], index=False)])
    expect("out_of_range_index", [dict(fragments[0], index=10)])
    expect("out_of_order", [fragments[1]])
    expect("nonzero_duplicate", [fragments[0], dict(fragments[0])])
    expect("nonzero_unused_high_bits", fragments[:9] +
           [dict(fragments[9], data_u64=fragments[9]["data_u64"] | (1 << 63))])
    corrupted = bytearray(frame); corrupted[20] ^= 1
    expect("bad_crc", make_fragments(corrupted), active=copy.deepcopy(GOLDEN_CONFIG))
    pad_payload = bytearray(payload); pad_payload[73] |= 1 << 1
    pad_frame = bytes(pad_payload) + crc32c(pad_payload).to_bytes(4, byteorder="little")
    expect("crc_correct_nonzero_pad", make_fragments(pad_frame))
    bad_code = bytearray(frame)
    bit_offset = 240 + 7 * 2
    bad_code[bit_offset // 8] |= 3 << (bit_offset % 8)
    bad_payload = bytes(bad_code[:74])
    bad_code[74:] = crc32c(bad_payload).to_bytes(4, byteorder="little")
    expect("illegal_ternary", make_fragments(bad_code))
    expect("equal_generation", fragments, active=copy.deepcopy(GOLDEN_CONFIG))
    active = copy.deepcopy(GOLDEN_CONFIG); active["generation_u16"] = 0
    expect("ambiguous_delta_0x8000", make_fragments(frame_with_generation(
        GOLDEN_CONFIG, 0x8000)), active=active)
    active = copy.deepcopy(GOLDEN_CONFIG); active["generation_u16"] = 0x8001
    expect("stale_delta_0x8001", make_fragments(frame_with_generation(
        GOLDEN_CONFIG, 0x0002)), active=active)
    expect("undrained_activation", fragments, drained=False)
    incomplete = StrictFragmentLoader()
    for item in fragments[:4]:
        incomplete.accept(item, datapath_drained=True)
    try:
        incomplete.finalize_incomplete()
    except ValueError:
        rejected.append("incomplete_frame")
    else:
        raise ValueError("incomplete frame accepted")

    recovery = StrictFragmentLoader()
    try:
        recovery.accept(fragments[1], datapath_drained=True)
    except ValueError:
        pass
    else:
        raise ValueError("recovery witness did not first fail")
    for item in fragments:
        recovered = recovery.accept(item, datapath_drained=True)
    require(recovered and recovery.active_config == GOLDEN_CONFIG,
            "failed loader did not recover from fragment zero")
    require(generation_is_newer(0x7FFF, 0), "delta 0x7fff rejected")
    require(not generation_is_newer(0x8000, 0), "delta 0x8000 accepted")
    require(not generation_is_newer(7, 7), "equal generation accepted")
    require(generation_is_newer(1, 0xFFFE), "forward wrap rejected")
    return {
        "crc32c_ascii_123456789": "0xE3069283",
        "golden_config_sha256": canonical_sha256(GOLDEN_CONFIG),
        "golden_serialized_frame_sha256": hashlib.sha256(frame).hexdigest(),
        "golden_crc32c": "0x{:08X}".format(crc32c(payload)),
        "protected_payload_bytes": 74, "serialized_frame_bytes": 78,
        "fragment_count": 10, "fragment_exact_keys_types_ranges_enforced": True,
        "negative_cases_rejected": rejected,
        "failure_then_fragment0_full_recovery": True,
        "active_context_unchanged_on_every_failure": True,
        "generation_boundaries": {
            "delta_0x7fff": True, "delta_0x8000": False,
            "equal": False, "forward_wrap_fffe_to_0001": True},
    }


def validate_t10_offer(offer, context_mode, context_generation):
    exact_keys(offer, {"tag", "generation"}, "T10 offer")
    integer(offer["tag"], 0, 0x7FFFFFFF, "T10 offer tag")
    integer(offer["generation"], 0, 0xFFFF, "T10 offer generation")
    require(context_mode == "T10", "T10 offer outside T10 context")
    require(offer["generation"] == context_generation, "T10 offer generation mismatch")


def validate_other_offer(offer, context_mode, context_generation):
    exact_keys(offer, {"writer", "mode", "tag", "beat", "generation"},
               "other writer offer")
    require(isinstance(offer["writer"], str) and offer["writer"] == "OTHER",
            "other writer enum/type violation")
    require(isinstance(offer["mode"], str) and offer["mode"] in ("T10", "T2"),
            "other writer mode type/enum violation")
    integer(offer["tag"], 0, 0x7FFFFFFF, "other writer tag")
    integer(offer["beat"], 0, 4, "other writer beat")
    integer(offer["generation"], 0, 0xFFFF, "other writer generation")
    require(offer["mode"] == context_mode, "other writer context mismatch")
    require(offer["generation"] == context_generation,
            "other writer generation mismatch")


class IntegratedCycleModel(object):
    """Finite-control executable reference, not an RTL or PPA model."""

    def __init__(self, fifo_depth=16):
        self.fifo_depth = integer(fifo_depth, 1, 1024, "FIFO depth")
        self.fifo = []
        self.reserved = 0
        self.slot = None
        self.pending = None
        self.stage1 = None
        self.reconstruction = None
        self.context_mode = None
        self.context_generation = None
        self.cycle = 0
        self.history = []
        self.done_tags = []
        self.maximum_fifo_occupancy = 0
        self.maximum_occupancy_plus_reserved = 0

    def canonical_snapshot(self):
        return copy.deepcopy({
            "fifo_depth": self.fifo_depth, "fifo": self.fifo,
            "reserved": self.reserved, "slot": self.slot,
            "pending": self.pending, "stage1": self.stage1,
            "reconstruction": self.reconstruction,
            "context_mode": self.context_mode,
            "context_generation": self.context_generation,
            "cycle": self.cycle, "history": self.history,
            "done_tags": self.done_tags,
            "maximum_fifo_occupancy": self.maximum_fifo_occupancy,
            "maximum_occupancy_plus_reserved": self.maximum_occupancy_plus_reserved,
        })

    def drained(self):
        return (self.stage1 is None and self.pending is None and self.slot is None
                and self.reconstruction is None and self.reserved == 0 and not self.fifo)

    def switch_context(self, mode, generation):
        require(isinstance(mode, str) and mode in ("T10", "T2"),
                "context mode type/enum violation")
        generation = integer(generation, 0, 0xFFFF, "context generation")
        require(self.drained(), "context switch requires complete drain")
        self.context_mode = mode
        self.context_generation = generation

    def seed_fifo(self, count):
        count = integer(count, 0, self.fifo_depth, "FIFO seed")
        require(not self.fifo and self.reserved == 0, "FIFO seed requires empty state")
        require(self.context_mode in ("T10", "T2"), "FIFO seed requires context")
        self.fifo = [{"writer": "seed", "mode": self.context_mode, "tag": tag,
                      "beat": 0, "generation": self.context_generation}
                     for tag in range(count)]
        self._check_invariant()

    def _check_invariant(self):
        occupancy = len(self.fifo)
        require(0 <= self.reserved <= 5, "reservation outside 0..5")
        require(occupancy + self.reserved <= self.fifo_depth,
                "FIFO occupancy plus reservation overflow")
        if self.reconstruction is None:
            require(self.reserved == 0, "orphan reservation")
        else:
            phase = integer(self.reconstruction["phase"], 0, 4,
                            "reconstruction phase")
            require(self.reserved == 5 - phase,
                    "reconstruction phase/reservation relation drift")
            require(self.slot is not None
                    and self.slot["tag"] == self.reconstruction["tile"]["tag"],
                    "reconstruction slot ownership drift")
        if self.pending is not None:
            require(self.slot is not None, "pending without occupied slot")
        self.maximum_fifo_occupancy = max(self.maximum_fifo_occupancy, occupancy)
        self.maximum_occupancy_plus_reserved = max(
            self.maximum_occupancy_plus_reserved, occupancy + self.reserved)

    def step(self, sink_ready=False, t10_offer=None, other_writer_offer=None):
        # All external validation is deliberately before every mutation, including pop.
        require(isinstance(sink_ready, bool), "sink_ready type violation")
        if t10_offer is not None:
            validate_t10_offer(t10_offer, self.context_mode, self.context_generation)
        if other_writer_offer is not None:
            validate_other_offer(other_writer_offer, self.context_mode,
                                 self.context_generation)

        event = {"cycle": self.cycle, "fifo_occupancy_before": len(self.fifo),
                 "reserved_before": self.reserved, "stage1_accept": False,
                 "reconstruction_launch": False, "m38_push": False,
                 "other_writer_push": False, "other_writer_denied": False,
                 "slot_pop": False, "slot_push": False,
                 "pending_materialize": False, "done": False}
        fifo_was_full = len(self.fifo) == self.fifo_depth
        popped = self.fifo.pop(0) if sink_ready and self.fifo else None
        event["fifo_pop"] = popped

        accepted = None
        if t10_offer is not None and self.stage1 is None and self.pending is None:
            accepted = {"tile": dict(t10_offer), "phase": 0}
            self.stage1 = accepted
            event["stage1_accept"] = True

        recon_existed = self.reconstruction is not None
        if recon_existed:
            phase = self.reconstruction["phase"]
            tile = self.reconstruction["tile"]
            require(self.reserved > 0 and len(self.fifo) < self.fifo_depth,
                    "reserved M38 beat cannot commit")
            self.fifo.append({"writer": "M38", "mode": "T10", "tag": tile["tag"],
                              "beat": phase, "generation": tile["generation"]})
            self.reserved -= 1
            event.update({"m38_push": True, "m38_push_tag": tile["tag"],
                          "m38_push_beat": phase})
            if other_writer_offer is not None:
                event["other_writer_denied"] = True
            if phase == 4:
                event.update({"done": True, "done_tag": tile["tag"],
                              "slot_pop": True,
                              "slot_old_read_tag": self.slot["tag"]})
                self.done_tags.append(tile["tag"])
                self.reconstruction = None
            else:
                self.reconstruction["phase"] += 1
        elif other_writer_offer is not None:
            if len(self.fifo) + self.reserved + 1 <= self.fifo_depth:
                self.fifo.append(dict(other_writer_offer))
                event["other_writer_push"] = True
            else:
                event["other_writer_denied"] = True

        stage1_finish = None
        if self.stage1 is not None:
            if self.stage1["phase"] == 4:
                stage1_finish = dict(self.stage1["tile"])
                self.stage1 = None
                event["stage1_finish"] = True
            else:
                self.stage1["phase"] += 1

        if event["slot_pop"]:
            self.slot = None
        if self.slot is None:
            if self.pending is not None:
                self.slot = self.pending
                self.pending = None
                event.update({"slot_push": True, "pending_materialize": True,
                              "slot_new_write_tag": self.slot["tag"]})
            elif stage1_finish is not None:
                self.slot = stage1_finish
                stage1_finish = None
                event.update({"slot_push": True,
                              "slot_new_write_tag": self.slot["tag"]})
        if stage1_finish is not None:
            require(self.pending is None, "completed-pending overflow")
            self.pending = stage1_finish
            event["stage1_completed_pending"] = True

        if self.reconstruction is None and self.slot is not None:
            if len(self.fifo) + self.reserved + 5 <= self.fifo_depth:
                self.reconstruction = {"tile": dict(self.slot), "phase": 0}
                self.reserved = 5
                event["reconstruction_launch"] = True

        event["full_old_read_new_write"] = bool(
            fifo_was_full and popped is not None
            and (event["m38_push"] or event["other_writer_push"]))
        event["fifo_occupancy_after"] = len(self.fifo)
        event["reserved_after"] = self.reserved
        require(int(event["m38_push"]) + int(event["other_writer_push"]) <= 1,
                "single-writer invariant violated")
        self._check_invariant()
        self.history.append(event)
        self.cycle += 1
        return event


CompactState = collections.namedtuple(
    "CompactState",
    "mode drain stage1 slot pending reconstruction reserved occupancy")


def compact_drained(state):
    return (state.stage1 == -1 and not state.slot and not state.pending
            and state.reconstruction == -1 and state.reserved == 0
            and state.occupancy == 0)


def check_compact(state):
    require(state.mode in ("T10", "T2"), "compact context drift")
    require(isinstance(state.drain, bool), "compact drain type drift")
    require(state.stage1 in (-1, 1, 2, 3, 4), "compact stage1 phase drift")
    require(state.reconstruction in (-1, 0, 1, 2, 3, 4),
            "compact reconstruction phase drift")
    require(0 <= state.reserved <= 5 and 0 <= state.occupancy <= 16,
            "compact credit range drift")
    require(state.occupancy + state.reserved <= 16, "compact credit overflow")
    if state.reconstruction == -1:
        require(state.reserved == 0, "compact orphan reservation")
    else:
        require(state.reserved == 5 - state.reconstruction,
                "compact phase/reservation relation drift")
        require(state.slot, "compact reconstruction without slot")
    if state.pending:
        require(state.slot, "compact pending without slot")
    if state.mode == "T2":
        require(state.stage1 == -1 and not state.slot and not state.pending
                and state.reconstruction == -1 and state.reserved == 0,
                "T2 contains T10 pipeline state")


def compact_transition(state, sink=False, t10=False, other=False,
                       request_drain=False, switch=False):
    check_compact(state)
    for value in (sink, t10, other, request_drain, switch):
        require(isinstance(value, bool), "compact action type drift")
    drain = state.drain or request_drain
    if t10 and state.mode != "T10":
        return None, None
    if switch:
        if not drain or not compact_drained(state) or sink or t10 or other:
            return None, None
        target = CompactState("T2" if state.mode == "T10" else "T10", False,
                              -1, False, False, -1, 0, 0)
        return target, {"pushes": 0, "switched": True}
    occupancy = state.occupancy - (1 if sink and state.occupancy else 0)
    stage1, slot, pending = state.stage1, state.slot, state.pending
    reconstruction, reserved = state.reconstruction, state.reserved
    accepted = bool(t10 and state.mode == "T10" and not drain
                    and stage1 == -1 and not pending)
    if accepted:
        stage1 = 1
    pushes = 0
    slot_pop = False
    if reconstruction != -1:
        occupancy += 1
        reserved -= 1
        pushes += 1
        if reconstruction == 4:
            reconstruction = -1
            slot_pop = True
        else:
            reconstruction += 1
    elif other and not drain and occupancy + reserved + 1 <= 16:
        occupancy += 1
        pushes += 1
    stage1_finish = False
    if state.stage1 != -1:
        if state.stage1 == 4:
            stage1 = -1
            stage1_finish = True
        else:
            stage1 = state.stage1 + 1
    if slot_pop:
        slot = False
    if not slot:
        if pending:
            slot, pending = True, False
        elif stage1_finish:
            slot, stage1_finish = True, False
    if stage1_finish:
        require(not pending, "compact pending overflow")
        pending = True
    if reconstruction == -1 and slot and occupancy + reserved + 5 <= 16:
        reconstruction, reserved = 0, 5
    target = CompactState(state.mode, drain, stage1, slot, pending,
                          reconstruction, reserved, occupancy)
    check_compact(target)
    require(pushes <= 1, "compact single-writer violation")
    return target, {"pushes": pushes, "switched": False}


def directed_drain_steps(state, limit=128):
    current = state
    for steps in range(limit + 1):
        if compact_drained(current):
            return steps
        current, _ = compact_transition(current, sink=True, request_drain=True)
        require(current is not None, "directed drain transition missing")
    raise ValueError("reachable state failed directed drain")


def audit_reachable_state_bfs():
    initial = [CompactState(mode, False, -1, False, False, -1, 0, 0)
               for mode in ("T10", "T2")]
    queue = collections.deque(initial)
    visited = set(initial)
    transitions = 0
    maximum_credit = 0
    maximum_drain_steps = 0
    reserved_values = set()
    stage1_values = set()
    recon_values = set()
    modes = set()
    drain_values = set()
    while queue:
        state = queue.popleft()
        check_compact(state)
        maximum_credit = max(maximum_credit, state.occupancy + state.reserved)
        reserved_values.add(state.reserved)
        stage1_values.add(state.stage1)
        recon_values.add(state.reconstruction)
        modes.add(state.mode)
        drain_values.add(state.drain)
        maximum_drain_steps = max(maximum_drain_steps, directed_drain_steps(state))
        for sink in (False, True):
            for t10 in (False, True):
                for other in (False, True):
                    for request_drain in (False, True):
                        for switch in (False, True):
                            target, meta = compact_transition(
                                state, sink, t10, other, request_drain, switch)
                            if target is None:
                                continue
                            transitions += 1
                            require(meta["pushes"] <= 1, "BFS single writer failure")
                            if target not in visited:
                                visited.add(target)
                                queue.append(target)
    require(maximum_credit == 16, "BFS did not cover full credit boundary")
    require(reserved_values == set(range(6)), "BFS reserved domain coverage drift")
    require(modes == {"T10", "T2"} and drain_values == {False, True},
            "BFS context/drain coverage drift")
    require(stage1_values == {-1, 1, 2, 3, 4}, "BFS stage1 phase coverage drift")
    require(recon_values == {-1, 0, 1, 2, 3, 4},
            "BFS reconstruction phase coverage drift")
    return {
        "graph_scope": "COMPLETE_FIXPOINT_FINITE_ABSTRACT_REACHABLE_STATE_GRAPH",
        "reachable_states": len(visited), "transitions_checked": transitions,
        "maximum_occupancy_plus_reserved": maximum_credit,
        "reserved_values_reached": sorted(reserved_values),
        "stage1_phases_reached": ["idle" if value == -1 else value
                                  for value in sorted(stage1_values)],
        "reconstruction_phases_reached": ["idle" if value == -1 else value
                                           for value in sorted(recon_values)],
        "context_modes_reached": sorted(modes),
        "context_drain_states_reached": sorted(drain_values),
        "reservation_relation_holds": True, "single_writer_holds": True,
        "no_overflow_holds": True,
        "all_reachable_states_have_directed_drain_path": True,
        "maximum_directed_drain_steps": maximum_drain_steps,
        "general_fairness_liveness_admitted": False,
    }


def run_no_stall_tiles(tile_count):
    tile_count = integer(tile_count, 1, 100000, "tile count")
    model = IntegratedCycleModel()
    model.switch_context("T10", 7)
    next_tile = 0
    accepts, dones = [], []
    for _ in range(20 + 8 * tile_count):
        offer = ({"tag": next_tile, "generation": 7}
                 if next_tile < tile_count else None)
        event = model.step(sink_ready=True, t10_offer=offer)
        if event["stage1_accept"]:
            accepts.append(event["cycle"]); next_tile += 1
        if event["done"]:
            dones.append(event["cycle"])
        if len(dones) == tile_count:
            break
    require(len(dones) == tile_count, "no-stall model did not finish")
    require(accepts == list(range(0, 5 * tile_count, 5)), "stage1 II5 drift")
    require(dones == list(range(9, 9 + 5 * tile_count, 5)), "done II5 drift")
    require(dones[-1] + 1 == 5 + 5 * tile_count, "finite-N equation drift")
    return model, accepts, dones


def run_eventual_sink(tile_count, stalled_cycles):
    model = IntegratedCycleModel()
    model.switch_context("T10", 11)
    next_tile = 0
    for cycle in range(10000):
        offer = ({"tag": next_tile, "generation": 11}
                 if next_tile < tile_count else None)
        event = model.step(sink_ready=(cycle >= stalled_cycles), t10_offer=offer)
        if event["stage1_accept"]:
            next_tile += 1
        if len(model.done_tags) == tile_count and model.drained():
            return model, cycle + 1
    raise ValueError("eventual sink regression timeout")


def run_pending_trace():
    model = IntegratedCycleModel()
    model.switch_context("T10", 9)
    model.seed_fifo(12)
    next_tag = 0
    materialize = None
    for cycle in range(100):
        offer = ({"tag": next_tag, "generation": 9} if next_tag < 2 else None)
        event = model.step(sink_ready=(cycle >= 10), t10_offer=offer)
        if event["stage1_accept"]:
            next_tag += 1
        if event["pending_materialize"]:
            materialize = event
        if len(model.done_tags) == 2 and model.pending is None and model.slot is None:
            break
    require(materialize is not None, "pending never materialized")
    require(materialize["slot_old_read_tag"] == 0
            and materialize["slot_new_write_tag"] == 1,
            "pending old-read/new-write drift")
    require(model.done_tags == [0, 1], "pending done order drift")
    return model, materialize


def run_full_pop_push():
    model = IntegratedCycleModel()
    model.switch_context("T2", 21)
    model.seed_fifo(16)
    event = model.step(sink_ready=True, other_writer_offer={
        "writer": "OTHER", "mode": "T2", "tag": 100, "beat": 0,
        "generation": 21})
    require(event["full_old_read_new_write"] and len(model.fifo) == 16,
            "full FIFO pop/push drift")
    return event, model


def run_writer_conflict():
    model = IntegratedCycleModel()
    model.switch_context("T10", 23)
    model.seed_fifo(12)
    model.slot = {"tag": 0, "generation": 23}
    first = model.step(sink_ready=True)
    require(first["reconstruction_launch"] and model.reserved == 5,
            "reservation-five launch drift")
    conflict = model.step(sink_ready=False, other_writer_offer={
        "writer": "OTHER", "mode": "T10", "tag": 99, "beat": 0,
        "generation": 23})
    require(conflict["m38_push"] and conflict["other_writer_denied"],
            "M38 writer priority drift")
    for _ in range(4):
        model.step()
    require(len(model.fifo) == 16 and model.reserved == 0,
            "reserved completion drift")
    return conflict, model


def run_t10_t2_t10():
    model = IntegratedCycleModel()
    model.switch_context("T10", 30)
    accepted = False
    rejections = 0
    for cycle in range(50):
        offer = None if accepted else {"tag": 1, "generation": 30}
        event = model.step(sink_ready=(cycle >= 10), t10_offer=offer)
        accepted = accepted or event["stage1_accept"]
        if cycle == 1:
            try:
                model.switch_context("T2", 31)
            except ValueError:
                rejections += 1
        if model.done_tags == [1] and model.drained():
            break
    model.switch_context("T2", 31)
    model.step(other_writer_offer={"writer": "OTHER", "mode": "T2", "tag": 2,
                                   "beat": 0, "generation": 31})
    try:
        model.switch_context("T10", 32)
    except ValueError:
        rejections += 1
    while model.fifo:
        model.step(sink_ready=True)
    model.switch_context("T10", 32)
    require(rejections == 2 and model.context_mode == "T10",
            "T10/T2/T10 drain sequence drift")
    return {"mode_sequence": ["T10", "T2", "T10"],
            "undrained_switch_rejections": rejections}


def run_invalid_offer_atomicity():
    cases = []
    invalids = [
        ("sink_ready_integer", {"sink_ready": 1}),
        ("t10_missing_key", {"sink_ready": True, "t10_offer": {"tag": 1}}),
        ("t10_extra_key", {"sink_ready": True,
                           "t10_offer": {"tag": 1, "generation": 7, "x": 0}}),
        ("t10_boolean_tag", {"sink_ready": True,
                             "t10_offer": {"tag": True, "generation": 7}}),
        ("t10_generation_range", {"sink_ready": True,
                                  "t10_offer": {"tag": 1, "generation": 65536}}),
        ("t10_generation_mismatch", {"sink_ready": True,
                                     "t10_offer": {"tag": 1, "generation": 8}}),
        ("other_missing_key", {"sink_ready": True,
                               "other_writer_offer": {"writer": "OTHER"}}),
        ("other_extra_key", {"sink_ready": True, "other_writer_offer": {
            "writer": "OTHER", "mode": "T10", "tag": 1, "beat": 0,
            "generation": 7, "x": 0}}),
        ("other_boolean_beat", {"sink_ready": True, "other_writer_offer": {
            "writer": "OTHER", "mode": "T10", "tag": 1, "beat": False,
            "generation": 7}}),
        ("other_writer_enum", {"sink_ready": True, "other_writer_offer": {
            "writer": "T2", "mode": "T10", "tag": 1, "beat": 0,
            "generation": 7}}),
        ("other_context_mismatch", {"sink_ready": True, "other_writer_offer": {
            "writer": "OTHER", "mode": "T2", "tag": 1, "beat": 0,
            "generation": 7}}),
    ]
    for name, kwargs in invalids:
        model = IntegratedCycleModel()
        model.switch_context("T10", 7)
        model.seed_fifo(3)
        before = model.canonical_snapshot()
        try:
            model.step(**kwargs)
        except ValueError:
            pass
        else:
            raise ValueError("invalid offer accepted: {}".format(name))
        require(model.canonical_snapshot() == before,
                "invalid offer changed state: {}".format(name))
        cases.append(name)
    return {"negative_cases": cases, "state_atomic_rejections": len(cases),
            "all_snapshots_exactly_equal_before_after": True}


def build_cycle_audit(contract):
    finite = []
    for count in contract["theory_rules"]["finite_n_regression_values"]:
        model, accepts, dones = run_no_stall_tiles(count)
        finite.append({"tiles": count, "accept_cycles_sha256": canonical_sha256(accepts),
                       "done_cycles_sha256": canonical_sha256(dones),
                       "parallel_commit_cycles": dones[-1] + 1,
                       "serialized_cycles": 10 * count,
                       "exact_ratio": fraction_json(Fraction(10 * count, dones[-1] + 1)),
                       "maximum_occupancy_plus_reserved":
                       model.maximum_occupancy_plus_reserved})
    stalls = []
    for stalled in contract["theory_rules"]["eventual_sink_stall_regression_cycles"]:
        model, cycles = run_eventual_sink(40, stalled)
        stalls.append({"stalled_cycles": stalled, "tiles": 40,
                       "completion_and_drain_cycles": cycles,
                       "maximum_fifo_occupancy": model.maximum_fifo_occupancy,
                       "maximum_occupancy_plus_reserved":
                       model.maximum_occupancy_plus_reserved})
    pending, pending_event = run_pending_trace()
    full_event, full_model = run_full_pop_push()
    conflict_event, conflict_model = run_writer_conflict()
    return {
        "finite_n_regressions": finite, "eventual_sink_regressions": stalls,
        "pending_old_read_new_write": {
            "cycle": pending_event["cycle"],
            "old_slot_tag": pending_event["slot_old_read_tag"],
            "new_slot_tag": pending_event["slot_new_write_tag"],
            "done_tags": pending.done_tags},
        "full_fifo_pop_push": {
            "old_head_returned_new_tail_written": full_event["full_old_read_new_write"],
            "final_occupancy": len(full_model.fifo)},
        "writer_conflict": {
            "m38_push": conflict_event["m38_push"],
            "other_writer_denied": conflict_event["other_writer_denied"],
            "final_occupancy": len(conflict_model.fifo),
            "final_reserved": conflict_model.reserved},
        "T10_T2_T10": run_t10_t2_t10(),
        "beat4_commit_done_same_cycle": True,
    }


def configuration_ledger():
    common = 30 * 8 + 10 * 24 + 24 + 5
    return {
        "common_payload_bits_excluding_left_factor": common,
        "m31_serialized_parameter_payload_bits": common + 30 * 8,
        "m37_csd4_parameter_payload_bits": common + 30 * 8 + 30 * 4 + 30 * 4 + 30 * 4 * 3,
        "m38_rst_arithmetic_payload_bits": common + 30 * 2,
        "m38_rst_logical_context_bits": common + 30 * 2 + 16 + 32,
        "m38_rst_serialized_context_bits": 624,
        "parameter_load_cycles_included_in_throughput": False,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes = load_contract(contract_path)
    validate_frozen_contract(contract, payloads)
    m31 = validate_m31(contract, payloads, hashes)
    m37 = validate_m37(contract, payloads, hashes)
    reviews, reviews_bound = validate_review_admissions(contract, payloads, hashes)
    scalar, rank3 = build_math_audit()
    protocol = build_protocol_audit(contract)
    offer_atomicity = run_invalid_offer_atomicity()
    reachable = audit_reachable_state_bfs()
    cycle = build_cycle_audit(contract)
    status = ("PASS_M38_R3_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY"
              if reviews_bound else
              "BLOCKED_PENDING_INDEPENDENT_REVIEW_ADMISSIONS_MATH_PROTOCOL_REACHABLE_STATE_ONLY")
    return {
        "schema": "m38_rst_math_protocol_reachable_state_audit_v3",
        "status": status,
        "identity": {
            "contract": "hw_autoresearch_nts07/contracts/{}".format(
                Path(contract_path).name),
            "contract_sha256": sha256(contract_path),
            "analyzer":
            "hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r3.py",
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
        },
        "supersedes": contract["supersedes"],
        "recursive_anchor_audit": {"m31_r4": m31, "m37_r8": m37},
        "independent_review_admission_audit": reviews,
        "scalar_ternary_audit": scalar,
        "rank3_q24_threshold_audit": rank3,
        "configuration_bit_ledger": configuration_ledger(),
        "canonical_crc_and_fragment_protocol_audit": protocol,
        "offer_validation_atomicity_audit": offer_atomicity,
        "finite_reachable_state_audit": reachable,
        "abstract_cycle_regression_audit": cycle,
        "conditional_theory": {
            "serialized_steady_ii": 10, "parallel_steady_ii": 5,
            "steady_t10_kernel_throughput_limit": {"numerator": 2, "denominator": 1},
            "finite_n_ratio": "10*N/(5+5*N)",
            "system_speedup_admitted": False,
        },
        "admission": {
            "m31_r4_current_receipt_and_run_verified": True,
            "m37_r8_current_receipt_and_run_verified": True,
            "both_independent_review_admissions_bound": reviews_bound,
            "recursive_anchor_identity_admitted": reviews_bound,
            "q8_times_ternary_scalar_math_admitted": True,
            "constructive_every_integer_rank_sum_admitted": True,
            "all_legal_rank_triples_exhaustively_checked": False,
            "canonical_crc32c_strict_fragment_reference_admitted": True,
            "invalid_offer_state_atomicity_admitted": True,
            "finite_abstract_reachable_state_safety_admitted": True,
            "directed_drain_liveness_admitted": True,
            "general_fairness_or_hardware_liveness_admitted": False,
            "integrated_rtl_admitted": False, "integrated_rtl_vcs_admitted": False,
            "dc_sta_formality_admitted": False, "area_power_energy_admitted": False,
            "memory_and_system_cycles_admitted": False,
            "system_speedup_admitted": False, "headline_admitted": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }


def write_output(path, payload):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite existing M38-r3 output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract)
    write_output(args.output, result)
    print(json.dumps({"status": result["status"],
                      "output": str(args.output.resolve()),
                      "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
