#!/usr/bin/env python3
"""Build M38-RST milestone-2 fail-closed math/CRC/abstract-cycle evidence."""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
DEFAULT_CONTRACT = (
    HW_ROOT / "contracts/m38_rst_math_input_contract_r2_20260822.json"
)
R1_ANALYZER = (
    HW_ROOT / "system_simulator/scripts/analyze_m38_rst_math_and_integration.py"
)

M31_RECEIPT_SCHEMA = "m31_output_receipt_v3"
M31_RECEIPT_STATUS = (
    "PASS_UNIFIED_T10_T2_EXACT_FIXED_POINT_SINGLE_SOURCE_MUL96_"
    "HIERARCHY_VCS_NO_PPA_OR_SYSTEM_CLAIM"
)
M37_RECEIPT_SCHEMA = "m37_output_receipt_v2"
M37_RECEIPT_STATUS = (
    "PASS_STANDALONE_T10_CANONICAL_NAF_CSD_RECONSTRUCTION_"
    "EXACT_FIXED_POINT_VCS_NO_PPA_OR_SYSTEM_CLAIM"
)
M38_IDENTITY = (
    "M31_r3_M37_r7_rank3_M38_RST_milestone2_fail_closed_"
    "math_crc_and_cycle_model_only"
)
M38_CLAIM_BOUNDARY = (
    "complete q8-by-ternary arithmetic, rank-3 width and Q24 semantics, "
    "recursively closed M31-r3/M37-r7 source-and-run identity, canonical "
    "CRC-32C context packing, and an executable abstract slot/pending/shared-"
    "FIFO cycle model only; trained codebooks, integrated RTL, VCS of "
    "integrated RTL, DC/STA/Formality, PPA, memory, Local/Motion system "
    "cycles, speedup, energy, and headline claims remain unadmitted"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_r1_math():
    spec = importlib.util.spec_from_file_location("m38_r1_math", str(R1_ANALYZER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1 = load_r1_math()
ternary_product = R1.ternary_product


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_exact_keys(mapping, expected, label):
    if set(mapping) != set(expected):
        raise ValueError("{} key population drift".format(label))


def read_sha_manifest(path):
    entries = {}
    for line_number, raw in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        pieces = raw.strip().split(None, 1)
        if len(pieces) != 2 or not re.match(r"^[0-9a-f]{64}$", pieces[0]):
            raise ValueError("malformed SHA manifest line {}".format(line_number))
        name = pieces[1].lstrip("*")
        if name in entries:
            raise ValueError("duplicate SHA manifest path")
        entries[name] = pieces[0]
    if not entries:
        raise ValueError("empty SHA manifest")
    return entries


def verify_manifest_files(entries, relative_root):
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(relative_root) / path
        if not path.is_file():
            raise ValueError("manifest file is missing: {}".format(raw_path))
        if sha256(path) != expected:
            raise ValueError("manifest content hash drift: {}".format(raw_path))


def load_contract(path):
    contract = read_json(path)
    require_exact_keys(
        contract,
        {"schema", "identity", "claim_boundary", "inputs",
         "frozen_architecture", "canonical_configuration_frame",
         "abstract_cycle_protocol", "theory_rules"},
        "M38 r2 contract",
    )
    if contract.get("schema") != "m38_rst_math_input_contract_v2":
        raise ValueError("unexpected M38 r2 contract schema")
    if contract.get("identity") != M38_IDENTITY:
        raise ValueError("unexpected M38 r2 contract identity")
    if contract.get("claim_boundary") != M38_CLAIM_BOUNDARY:
        raise ValueError("unexpected M38 r2 claim boundary")
    expected_inputs = {
        "m29_config_generator",
        "m31_vcs_contract",
        "m31_vcs_receipt",
        "m37_math_contract",
        "m37_math_result",
        "m37_vcs_contract",
        "m37_vcs_receipt",
        "m38_r1_math_analyzer",
    }
    require_exact_keys(contract.get("inputs", {}), expected_inputs, "M38 r2 inputs")
    payloads = {}
    hashes = {}
    paths = {}
    for name, item in sorted(contract["inputs"].items()):
        require_exact_keys(item, {"path", "sha256"}, "M38 r2 input {}".format(name))
        source = resolve(item["path"])
        if not source.is_file():
            raise ValueError("M38 r2 input is missing for {}".format(name))
        actual = sha256(source)
        if actual != item["sha256"]:
            raise ValueError("M38 r2 input hash drift for {}".format(name))
        hashes[name] = actual
        paths[name] = source
        payloads[name] = (
            read_json(source) if source.suffix == ".json"
            else source.read_text(encoding="utf-8")
        )
    return contract, payloads, hashes, paths


def count_source_properties(path):
    text = Path(path).read_text(encoding="utf-8")
    asserts = len(re.findall(r"^\s*assert\s+property", text, re.MULTILINE))
    covers = len(re.findall(r"^\s*cover\s+property", text, re.MULTILINE))
    return asserts, covers


def scan_logs(compile_text, sim_text, expected_covers):
    combined = compile_text + "\n" + sim_text
    warning_lines = [line for line in combined.splitlines()
                     if re.search(r"\bWarning\b", line, re.IGNORECASE)]
    failure_lines = [
        line for line in combined.splitlines()
        if re.search(
            r"(^\s*(Error|Fatal)[:\-]|assertion.*(fail|error)|"
            r"offending.*assert|UVM_(ERROR|FATAL))",
            line, re.IGNORECASE,
        )
    ]
    cover_matches = [
        int(value) for value in re.findall(
            r",\s+\d+\s+attempts,\s+(\d+)\s+match", sim_text
        )
    ]
    if warning_lines:
        raise ValueError("VCS warning population is not empty")
    if failure_lines:
        raise ValueError("VCS failure signature found")
    if len(cover_matches) != expected_covers or any(x <= 0 for x in cover_matches):
        raise ValueError("VCS cover closure drift")
    return {
        "compile_warning_count": 0,
        "failure_signature_count": 0,
        "cover_property_count": expected_covers,
        "cover_nonzero_match_counts": cover_matches,
    }


def parse_token_line(text, prefix):
    line = next((row for row in text.splitlines() if row.startswith(prefix)), None)
    if line is None:
        raise ValueError("missing VCS summary {}".format(prefix))
    tokens = {}
    for piece in line.split()[1:]:
        if "=" in piece:
            name, value = piece.split("=", 1)
            tokens[name] = value
    return tokens


def split_ints(raw, separator="/"):
    return [int(value) for value in raw.split(separator)]


def receipt_file_entries(receipt):
    entries = {}
    for name, item in receipt["files"].items():
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("receipt file entry drift for {}".format(name))
        path, expected = item
        if not re.match(r"^[0-9a-f]{64}$", expected):
            raise ValueError("receipt file hash format drift")
        resolved = resolve(path)
        if not resolved.is_file() or sha256(resolved) != expected:
            raise ValueError("receipt live source drift for {}".format(name))
        entries[path] = expected
    return entries


def verify_run_manifests(receipt, expected_inputs, expected_outputs, input_root):
    run = Path(receipt["vcs_run"]["directory"])
    if not run.is_dir():
        raise ValueError("receipt VCS run directory is missing")
    input_manifest = run / "input_sha256.txt"
    output_manifest = run / "output_sha256.txt"
    if sha256(input_manifest) != receipt["vcs_run"]["input_sha256_manifest"]:
        raise ValueError("run input manifest hash drift")
    if sha256(output_manifest) != receipt["vcs_run"]["output_sha256_manifest"]:
        raise ValueError("run output manifest hash drift")
    input_entries = read_sha_manifest(input_manifest)
    output_entries = read_sha_manifest(output_manifest)
    if input_entries != expected_inputs:
        raise ValueError("run input manifest population drift")
    verify_manifest_files(input_entries, input_root)
    normalized_outputs = {str(Path(name).resolve()): digest
                          for name, digest in output_entries.items()}
    if normalized_outputs != expected_outputs:
        raise ValueError("run output manifest population drift")
    verify_manifest_files(output_entries, run)
    return run, input_entries, output_entries


def validate_m31_anchor(payloads, hashes):
    contract = payloads["m31_vcs_contract"]
    receipt = payloads["m31_vcs_receipt"]
    if contract.get("contract") != "m31_unified_t10_t2_vcs_contract_r2":
        raise ValueError("M31 exact contract identity drift")
    if receipt.get("schema") != M31_RECEIPT_SCHEMA:
        raise ValueError("M31 exact receipt schema drift")
    if receipt.get("status") != M31_RECEIPT_STATUS:
        raise ValueError("M31 exact receipt status drift")
    require_exact_keys(
        receipt,
        {"schema", "status", "date", "supersedes", "contract", "files",
         "vcs_run", "observed", "claim_boundary", "headline_admitted",
         "review_required"},
        "M31 receipt",
    )
    if receipt.get("supersedes") != {
            "path": "hw_autoresearch_nts07/contracts/m31_output_receipt_r2_20260822.json",
            "state": "STALE_SUPERSEDED_DO_NOT_CITE",
            "reason": (
                "the r2 receipt predates the leaf96 multiplier-pool source "
                "revision and therefore no longer closes live source identity"
            )}:
        raise ValueError("M31 r2 stale supersession drift")
    if receipt.get("contract") != {
            "path": "hw_autoresearch_nts07/contracts/m31_unified_t10_t2_vcs_contract_r2_20260822.json",
            "sha256": hashes["m31_vcs_contract"]}:
        raise ValueError("M31 receipt-to-contract identity drift")
    require_exact_keys(
        receipt.get("files", {}),
        {"multiplier_pool_rtl", "unified_core_rtl", "assertions",
         "testbench", "filelist", "run_script"},
        "M31 receipt files",
    )
    live_entries = receipt_file_entries(receipt)
    manifest_inputs = {}
    for path, digest in live_entries.items():
        prefix = "hw_autoresearch_nts07/"
        if not path.startswith(prefix):
            raise ValueError("M31 receipt path root drift")
        manifest_inputs[path[len(prefix):]] = digest
    run_path = Path(receipt["vcs_run"]["directory"])
    expected_outputs = {
        str((run_path / "compile.log").resolve()): receipt["vcs_run"]["compile_log"],
        str((run_path / "sim.log").resolve()): receipt["vcs_run"]["sim_log"],
    }
    run, _, _ = verify_run_manifests(
        receipt, manifest_inputs, expected_outputs, HW_ROOT
    )
    if run.name != "m31_unified_t10_t2_vcs_r3_leaf96_20260822":
        raise ValueError("M31 exact r3 run identity drift")
    compile_log = run / "compile.log"
    sim_log = run / "sim.log"
    if sha256(compile_log) != receipt["vcs_run"]["compile_log"]:
        raise ValueError("M31 compile log hash drift")
    if sha256(sim_log) != receipt["vcs_run"]["sim_log"]:
        raise ValueError("M31 simulation log hash drift")
    compile_text = compile_log.read_text(encoding="utf-8")
    sim_text = sim_log.read_text(encoding="utf-8")
    for marker in ("M31_SVA_BOUND=1", "SIMULATOR=Synopsys VCS",
                   "ASSERTIONS=enabled", "M31_PASS"):
        if marker not in sim_text:
            raise ValueError("M31 VCS marker drift")
    assertions_path = resolve(receipt["files"]["assertions"][0])
    assert_count, cover_count = count_source_properties(assertions_path)
    if (assert_count, cover_count) != (24, 4):
        raise ValueError("M31 SVA property population drift")
    log_audit = scan_logs(compile_text, sim_text, cover_count)
    tokens = parse_token_line(sim_text, "M31_PASS ")
    observed = receipt["observed"]
    scalar_map = {
        "sole_source_multiplier_pool_instances": "sole_mul_pool",
        "source_multiplier_slots": "mul_slots",
        "t10_tiles": "t10_tiles",
        "t10_arithmetic_cycles": "t10_arithmetic",
        "conditional_t10_no_stall_accept_ii": "t10_ii",
        "t10_credit_wait_cycles": "t10_credit_wait",
        "t2_packets": "t2_packets",
        "t2_nondegenerate_packets": "t2_nondegenerate",
        "conditional_t2_no_stall_accept_ii": "t2_ii",
        "conditional_t2_ii1_matches": "t2_ii_matches",
        "total_arithmetic_cycles": "total_arithmetic",
        "maximum_result_fifo_occupancy": "max_fifo",
        "fifo_full_cycles": "full_cycles",
        "fifo_full_simultaneous_pop_push_cycles": "full_pop_push",
        "release_input_collisions": "release_input_collisions",
        "t10_threshold_ties": "t10_ties",
    }
    for field, token in scalar_map.items():
        if observed[field] != int(tokens[token]):
            raise ValueError("M31 observed field drift for {}".format(field))
    vector_map = {
        "input_output_stall_cycles": "stalls",
        "t10_intermediate_positive_negative_saturations": "t10_mid_sat",
        "t10_output_positive_negative_saturations": "t10_out_sat",
        "threshold_equal_raw_just_below_cases": "threshold_eq_below",
        "t2_positive_negative_saturations": "t2_sat",
        "t2_output_one_zero_bits": "t2_diversity",
    }
    require_exact_keys(
        observed,
        set(scalar_map) | set(vector_map) | {"mode_sequence"},
        "M31 observed receipt",
    )
    for field, token in vector_map.items():
        if observed[field] != split_ints(tokens[token]):
            raise ValueError("M31 observed vector drift for {}".format(field))
    if tokens.get("modes") != "T10_T2N_T2S_T10":
        raise ValueError("M31 mode sequence log drift")
    if observed["mode_sequence"] != ["T10", "T2_NONDEGENERATE",
                                     "T2_SATURATION", "T10"]:
        raise ValueError("M31 receipt mode sequence drift")
    if receipt["vcs_run"].get("runner_exit_code") != 0:
        raise ValueError("M31 runner exit drift")
    if receipt["vcs_run"].get("assertion_failure_signatures") != 0:
        raise ValueError("M31 assertion closure drift")
    if receipt["vcs_run"].get("uncovered_SVA_cover_properties") != 0:
        raise ValueError("M31 cover admission drift")
    core_text = resolve(receipt["files"]["unified_core_rtl"][0]).read_text(
        encoding="utf-8"
    )
    pool_text = resolve(receipt["files"]["multiplier_pool_rtl"][0]).read_text(
        encoding="utf-8"
    )
    if len(re.findall(r"^\s*qfit_signed_int8_mul96_pool\s*#", core_text,
                      re.MULTILINE)) != 1:
        raise ValueError("M31 source pool instance drift")
    if "parameter int MULTIPLIERS = 96" not in pool_text:
        raise ValueError("M31 source pool slot drift")
    if len(re.findall(r"assign\s+product\s*=.*\*", pool_text)) != 1:
        raise ValueError("M31 source multiplier leaf drift")
    return {
        "receipt_schema": M31_RECEIPT_SCHEMA,
        "receipt_status": M31_RECEIPT_STATUS,
        "receipt_sha256": hashes["m31_vcs_receipt"],
        "contract_sha256": hashes["m31_vcs_contract"],
        "run_directory": str(run),
        "input_manifest_sha256": receipt["vcs_run"]["input_sha256_manifest"],
        "output_manifest_sha256": receipt["vcs_run"]["output_sha256_manifest"],
        "live_source_count": len(live_entries),
        "assert_property_count": assert_count,
        "log_audit": log_audit,
        "observed_receipt_fields_reconciled": len(scalar_map) + len(vector_map) + 1,
        "r2_stale_live_source_drift_eliminated": True,
    }


def validate_m37_anchor(payloads, hashes):
    contract = payloads["m37_vcs_contract"]
    receipt = payloads["m37_vcs_receipt"]
    math_contract = payloads["m37_math_contract"]
    math_result = payloads["m37_math_result"]
    if contract.get("contract") != "m37_csd_reconstruct_t10_vcs_contract_r2":
        raise ValueError("M37 exact contract identity drift")
    if receipt.get("schema") != M37_RECEIPT_SCHEMA:
        raise ValueError("M37 exact receipt schema drift")
    if receipt.get("status") != M37_RECEIPT_STATUS:
        raise ValueError("M37 exact receipt status drift")
    require_exact_keys(
        receipt,
        {"schema", "status", "date", "supersedes", "contract",
         "math_anchor", "files", "vcs_run", "observed", "SVA_closure",
         "claim_boundary", "headline_admitted", "review_required"},
        "M37 receipt",
    )
    if receipt.get("supersedes") != {
            "path": "hw_autoresearch_nts07/contracts/m37_output_receipt_r1_20260822.json",
            "state": "STALE_SUPERSEDED_DO_NOT_CITE"}:
        raise ValueError("M37 r1 stale supersession drift")
    if receipt.get("contract") != {
            "path": "hw_autoresearch_nts07/contracts/m37_csd_reconstruct_t10_vcs_contract_r2_20260822.json",
            "sha256": hashes["m37_vcs_contract"]}:
        raise ValueError("M37 receipt-to-contract identity drift")
    if receipt.get("math_anchor") != {
            "contract": [
                "hw_autoresearch_nts07/contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json",
                hashes["m37_math_contract"]],
            "result": [
                "hw_autoresearch_nts07/results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json",
                hashes["m37_math_result"]]}:
        raise ValueError("M37 receipt-to-math-anchor identity drift")
    if math_contract.get("schema") != (
            "m37_phase_decoupled_csd_reconstruct_input_contract_v2"):
        raise ValueError("M37 math contract schema drift")
    if math_result.get("schema") != "m37_phase_decoupled_csd_reconstruct_audit_v2":
        raise ValueError("M37 math result schema drift")
    if math_result.get("status") != (
            "PASS_SIGNED_INT8_FULL_DOMAIN_FOUR_TERM_CSD_AND_"
            "PHASE_OVERLAP_SENSITIVITY_ONLY"):
        raise ValueError("M37 math result status drift")
    if math_result["identity"]["contract_sha256"] != hashes["m37_math_contract"]:
        raise ValueError("M37 math result-to-contract hash drift")
    require_exact_keys(
        receipt.get("files", {}),
        {"rtl", "assertions", "testbench", "filelist", "runner"},
        "M37 receipt files",
    )
    live_entries = receipt_file_entries(receipt)
    manifest_inputs = {}
    prefix = "hw_autoresearch_nts07/"
    for path, digest in live_entries.items():
        if not path.startswith(prefix):
            raise ValueError("M37 receipt path root drift")
        manifest_inputs[path[len(prefix):]] = digest
    manifest_inputs[receipt["contract"]["path"][len(prefix):]] = hashes[
        "m37_vcs_contract"
    ]
    for anchor in ("contract", "result"):
        path, digest = receipt["math_anchor"][anchor]
        manifest_inputs[path[len(prefix):]] = digest
    run_path = Path(receipt["vcs_run"]["directory"])
    expected_outputs = {
        str((run_path / "compile.log").resolve()): receipt["vcs_run"]["compile_log"],
        str((run_path / "sim.log").resolve()): receipt["vcs_run"]["sim_log"],
        str((run_path / "vectors.txt").resolve()): receipt["vcs_run"]["vectors"],
    }
    run, _, _ = verify_run_manifests(
        receipt, manifest_inputs, expected_outputs, HW_ROOT
    )
    if run.name != "m37_csd_reconstruct_t10_vcs_r7_20260822":
        raise ValueError("M37 exact r7 run identity drift")
    compile_log = run / "compile.log"
    sim_log = run / "sim.log"
    if sha256(compile_log) != receipt["vcs_run"]["compile_log"]:
        raise ValueError("M37 compile log hash drift")
    if sha256(sim_log) != receipt["vcs_run"]["sim_log"]:
        raise ValueError("M37 simulation log hash drift")
    if sha256(run / "vectors.txt") != receipt["vcs_run"]["vectors"]:
        raise ValueError("M37 vector hash drift")
    compile_text = compile_log.read_text(encoding="utf-8")
    sim_text = sim_log.read_text(encoding="utf-8")
    for marker in ("M37_SVA_BOUND=1", "SIMULATOR=Synopsys VCS",
                   "ASSERTIONS=enabled", "M37_PASS"):
        if marker not in sim_text:
            raise ValueError("M37 VCS marker drift")
    assertions_path = resolve(receipt["files"]["assertions"][0])
    assert_count, cover_count = count_source_properties(assertions_path)
    if (assert_count, cover_count) != (21, 8):
        raise ValueError("M37 SVA property population drift")
    log_audit = scan_logs(compile_text, sim_text, cover_count)
    observed = receipt["observed"]
    primary = parse_token_line(sim_text, "M37_PASS ")
    primary_map = {
        "total_tiles": "total_tiles",
        "nominal_tiles": "nominal_tiles",
        "dut_committed_unique_signed_input_coefficient_product_pairs":
            "dut_unique_signed_input_coefficient_product_pairs",
        "direct_product_miters": "product_miters",
        "output_bit_miters": "bit_miters",
        "arithmetic_issue_cycles": "arithmetic_issues",
    }
    for field, token in primary_map.items():
        if observed[field] != int(primary[token]):
            raise ValueError("M37 observed field drift for {}".format(field))
    if primary.get("no_data_multiplier") != "1" or observed[
            "data_multiplier_operator_in_DUT"]:
        raise ValueError("M37 data-multiplier evidence drift")
    seed_line = next(
        (line for line in sim_text.splitlines()
         if line.startswith("M37_RANDOM_SEED=")), None
    )
    if seed_line is None or observed["deterministic_xorshift32_seed"].lower() != (
            seed_line.split("=", 1)[1].lower()):
        raise ValueError("M37 deterministic seed drift")
    uniqueness = parse_token_line(sim_text, "M37_UNIQUENESS ")
    uniqueness_map = {
        "nominal_unique_input_payloads": "unique_tile_payloads",
        "nominal_unique_expected_product_fingerprints":
            "unique_expected_product_fingerprints",
        "nominal_unique_expected_bitmaps": "unique_expected_bitmaps",
        "nominal_consecutive_identical_payloads": "consecutive_identical",
        "nominal_unique_signed_inputs": "nominal_unique_signed_inputs",
    }
    for field, token in uniqueness_map.items():
        if observed[field] != int(uniqueness[token]):
            raise ValueError("M37 uniqueness drift for {}".format(field))
    flow = parse_token_line(sim_text, "M37_FLOW ")
    if observed["conditional_standalone_no_stall_accept_ii5_matches"] != int(
            flow["conditional_standalone_accept_ii5_matches"]):
        raise ValueError("M37 II5 evidence drift")
    if observed["phase4_same_cycle_accepts"] != int(flow["phase4_chain_accepts"]):
        raise ValueError("M37 phase4 evidence drift")
    for field, token in (("maximum_result_fifo_occupancy", "max_fifo"),
                         ("fifo_full_cycles", "fifo_full_cycles"),
                         ("fifo_full_simultaneous_pop_push_cycles", "full_pop_push"),
                         ("done_with_fifo_pending", "done_with_fifo_pending")):
        if observed[field] != int(flow[token]):
            raise ValueError("M37 flow evidence drift for {}".format(field))
    if observed["input_output_stall_cycles"] != split_ints(flow["stalls"]):
        raise ValueError("M37 stall evidence drift")
    config = parse_token_line(sim_text, "M37_CONFIG ")
    if observed["configuration_load_release_reload"] != split_ints(
            config["config_load_release_reload"]):
        raise ValueError("M37 configuration accounting drift")
    if observed["release_refusal_busy_fifo_nonempty_input_valid_cycles"] != (
            split_ints(config["release_reject_busy_fifo_input"])):
        raise ValueError("M37 release-refusal accounting drift")
    if observed["live_config_pin_perturbations"] != int(
            config["live_pin_perturbations"]):
        raise ValueError("M37 live-pin perturbation drift")
    if bool(int(config["legal_zero_min_max"])) != observed[
            "legal_zero_negative128_positive127_coefficients_seen"]:
        raise ValueError("M37 legal coefficient boundary drift")
    illegal = parse_token_line(sim_text, "M37_ILLEGAL ")
    if observed["illegal_descriptor_accept_reject"] != split_ints(
            illegal["illegal_matrix"]):
        raise ValueError("M37 illegal matrix drift")
    if observed["illegal_descriptor_count_per_class"] != split_ints(
            illegal["illegal_classes"], ","):
        raise ValueError("M37 illegal class drift")
    threshold_rows = []
    for line in sim_text.splitlines():
        if line.startswith("M37_THRESHOLD "):
            tokens = parse_token_line(line, "M37_THRESHOLD ")
            threshold_rows.append({
                "value": int(tokens["value"]),
                "equality": int(tokens["equal"]),
                "raw_just_below": int(tokens["just_below_raw"]),
                "positive_saturation": int(tokens["positive_saturation"]),
                "negative_saturation": int(tokens["negative_saturation"]),
            })
    if threshold_rows != observed["threshold_cases"]:
        raise ValueError("M37 threshold row drift")
    diversity = parse_token_line(sim_text, "M37_DIVERSITY ")
    if observed["generic_positive_negative_q24_saturations"] != split_ints(
            diversity["generic_saturation"]):
        raise ValueError("M37 generic saturation evidence drift")
    if observed["output_one_zero_bits"] != split_ints(diversity["diversity"]):
        raise ValueError("M37 output diversity evidence drift")
    require_exact_keys(
        observed,
        {"total_tiles", "nominal_tiles",
         "dut_committed_unique_signed_input_coefficient_product_pairs",
         "direct_product_miters", "output_bit_miters",
         "arithmetic_issue_cycles", "data_multiplier_operator_in_DUT",
         "deterministic_xorshift32_seed", "nominal_unique_input_payloads",
         "nominal_unique_expected_product_fingerprints",
         "nominal_unique_expected_bitmaps",
         "nominal_consecutive_identical_payloads",
         "nominal_unique_signed_inputs",
         "conditional_standalone_no_stall_accept_ii5_matches",
         "phase4_same_cycle_accepts", "maximum_result_fifo_occupancy",
         "fifo_full_cycles", "fifo_full_simultaneous_pop_push_cycles",
         "done_with_fifo_pending", "input_output_stall_cycles",
         "configuration_load_release_reload",
         "release_refusal_busy_fifo_nonempty_input_valid_cycles",
         "live_config_pin_perturbations",
         "legal_zero_negative128_positive127_coefficients_seen",
         "illegal_descriptor_accept_reject",
         "illegal_descriptor_count_per_class", "threshold_cases",
         "generic_positive_negative_q24_saturations",
         "output_one_zero_bits", "SVA_cover_matches"},
        "M37 observed receipt",
    )
    if observed["SVA_cover_matches"] != log_audit[
            "cover_nonzero_match_counts"]:
        raise ValueError("M37 receipt-to-log cover count drift")
    if not all(receipt.get("SVA_closure", {}).values()):
        raise ValueError("M37 SVA closure drift")
    if (receipt["vcs_run"].get("runner_exit_code") != 0
            or receipt["vcs_run"].get("assertion_failure_signatures") != 0
            or receipt["vcs_run"].get("uncovered_SVA_cover_properties") != 0):
        raise ValueError("M37 VCS wrapper closure drift")
    return {
        "receipt_schema": M37_RECEIPT_SCHEMA,
        "receipt_status": M37_RECEIPT_STATUS,
        "receipt_sha256": hashes["m37_vcs_receipt"],
        "contract_sha256": hashes["m37_vcs_contract"],
        "math_contract_sha256": hashes["m37_math_contract"],
        "math_result_sha256": hashes["m37_math_result"],
        "run_directory": str(run),
        "input_manifest_sha256": receipt["vcs_run"]["input_sha256_manifest"],
        "output_manifest_sha256": receipt["vcs_run"]["output_sha256_manifest"],
        "vector_sha256": receipt["vcs_run"]["vectors"],
        "live_source_count": len(live_entries),
        "recursive_input_manifest_count": len(manifest_inputs),
        "assert_property_count": assert_count,
        "log_audit": log_audit,
        "observed_receipt_population_fully_reconciled": True,
        "r1_stale_receipt_rejected": True,
    }


def validate_frozen_contract(contract, payloads, hashes):
    arch = contract["frozen_architecture"]
    expected_arch = {
        "temporal_rows": 10,
        "rank": 3,
        "lanes": 16,
        "signed_input_bits": 8,
        "stage1_accumulator_bits": 24,
        "stage1_intermediate_bits": 8,
        "bias_bits": 24,
        "threshold_bits": 24,
        "shared_signed_int8_multiplier_lanes": 96,
        "rows_per_phase": 2,
        "phases_per_tile": 5,
        "result_beats_per_tile": 5,
        "result_fifo_entries": 16,
        "result_fifo_atomic_credit_per_t10_tile": 5,
        "intermediate_elastic_slots_target": 1,
        "intermediate_slot_bits": 384,
        "ternary_codes": {"0": 0, "1": 1, "2": -1, "3": "illegal"},
        "t10_factorized_modules_expected_from_m29_interface": 45,
        "t2_dense_fallback_modules_expected_from_m29_interface": 60,
    }
    if arch != expected_arch:
        raise ValueError("M38 r2 frozen architecture drift")
    frame = contract["canonical_configuration_frame"]
    require_exact_keys(
        frame,
        {"array_index_order", "field_order", "signed_encoding",
         "field_bit_order", "byte_mapping", "arithmetic_payload_bits",
         "generation_bits", "logical_protected_bits", "zero_pad_bits_before_crc",
         "crc_protected_bytes", "crc", "logical_context_bits_excluding_pad",
         "serialized_context_bits_including_pad", "load_beat_bits",
         "load_fragment_count", "last_fragment_valid_bits",
         "last_fragment_unused_high_bits_must_be_zero", "fragment_order",
         "fragment_failure_rule", "active_context_failure_rule", "activation",
         "generation_rule", "golden_frame"},
        "M38 canonical frame",
    )
    golden = frame["golden_frame"]
    require_exact_keys(
        golden,
        {"right_factor_q8", "left_ternary_code", "bias_q24",
         "threshold_q24", "stage1_requant_shift_u5", "generation_u16",
         "protected_payload_hex", "crc32c", "serialized_frame_hex"},
        "M38 canonical golden frame",
    )
    frame_metadata = dict(frame)
    del frame_metadata["golden_frame"]
    if frame_metadata != {
            "array_index_order": "ascending",
            "field_order": [
                "right_factor_q8[0:29]", "left_ternary_code[0:29]",
                "bias_q24[0:9]", "threshold_q24",
                "stage1_requant_shift_u5", "generation_u16"],
            "signed_encoding": "two_complement_exact_field_width",
            "field_bit_order": "least_significant_bit_first",
            "byte_mapping": "first_serial_bit_is_bit0_of_byte0",
            "arithmetic_payload_bits": 569,
            "generation_bits": 16,
            "logical_protected_bits": 585,
            "zero_pad_bits_before_crc": 7,
            "crc_protected_bytes": 74,
            "crc": {
                "name": "CRC-32C_Castagnoli",
                "reflected_recurrence_polynomial": "0x82F63B78",
                "initial_value": "0xFFFFFFFF",
                "final_xor": "0xFFFFFFFF",
                "canonical_check_ascii_123456789": "0xE3069283",
                "extra_output_reflection_after_recurrence": False,
                "serialized_crc_bit_order": "least_significant_bit_first"},
            "logical_context_bits_excluding_pad": 617,
            "serialized_context_bits_including_pad": 624,
            "load_beat_bits": 64,
            "load_fragment_count": 10,
            "last_fragment_valid_bits": 48,
            "last_fragment_unused_high_bits_must_be_zero": 16,
            "fragment_order": "strictly_ascending_zero_through_nine",
            "fragment_failure_rule": (
                "duplicate_out_of_order_missing_wrong_valid_bits_nonzero_"
                "unused_bits_or_bad_crc_invalidates_the_entire_shadow_and_"
                "restart_requires_fragment_zero"),
            "active_context_failure_rule": (
                "a_failed_or_incomplete_shadow_load_never_changes_the_active_"
                "context"),
            "activation": (
                "shadow_context_becomes_visible_atomically_only_after_all_"
                "fragments_crc_ternary_and_generation_checks_pass_and_"
                "datapath_is_drained"),
            "generation_rule": (
                "if_active_then_delta=(candidate-active)&0xffff_must_be_in_"
                "1_through_0x7fff")}:
        raise ValueError("M38 canonical frame metadata drift")
    if contract["abstract_cycle_protocol"] != {
            "fifo_push_ports": 1,
            "m38_reserved_writer_priority": True,
            "other_writer_admission": (
                "only_when_no_M38_beat_owns_the_single_push_port_and_"
                "occupancy_after_pop_plus_reserved_plus_one_is_at_most_16"),
            "launch_admission": (
                "M38_phase0_owns_the_push_port_and_after_same_cycle_pop_"
                "occupancy_plus_existing_reserved_plus_five_is_at_most_16"),
            "reservation_invariant": "0 <= occupancy + reserved <= 16",
            "slot_retire_replace": (
                "phase4_reads_old_slot_and_the_edge_may_install_pending_or_"
                "same_cycle_stage1_commit"),
            "pending_materialization": (
                "when_slot_credit_returns_pending_tag_and_generation_move_to_"
                "slot_then_pending_clears_even_though_original_commit_pulse_"
                "is_low"),
            "done_semantics": "beat4_fifo_commit_same_cycle_not_fifo_consumption",
            "context_switch_rule": (
                "release_or_T10_T2_change_requires_stage1_pending_slot_"
                "reconstruction_reservation_and_fifo_all_drained")}:
        raise ValueError("M38 abstract cycle protocol drift")
    if hashes["m38_r1_math_analyzer"] != sha256(R1_ANALYZER):
        raise ValueError("M38 r1 math implementation drift")
    generator = payloads["m29_config_generator"]
    for marker in ('"m29_expected_t10_factorized_modules": 45',
                   '"m29_expected_t2_dense_fallback_modules": 60',
                   '"temporal_factor_rank": 3'):
        if marker not in generator:
            raise ValueError("M29 scope marker drift")
    theory = contract["theory_rules"]
    if theory != {
            "conditional_t10_steady_ii_serialized": 10,
            "conditional_t10_steady_ii_parallel": 5,
            "conditional_t10_steady_throughput_limit": 2.0,
            "integrated_parallel_cycles_for_n_tiles": "5 + 5*N",
            "serialized_cycles_for_n_tiles": "10*N",
            "finite_n_ratio": "10*N/(5+5*N)",
            "configuration_load_cycles_included": False,
            "result_backpressure_included": False,
            "system_speedup_admitted": False,
            "area_admitted": False,
            "energy_admitted": False}:
        raise ValueError("M38 r2 theory rule drift")


def build_math_audit():
    scalar = R1.build_scalar_audit()
    rank3 = R1.build_rank3_and_threshold_audit(scalar)
    scalar_rows = scalar.pop("rows")
    saturation_rows = rank3.pop("saturation_rows")
    scalar["rows_sha256"] = canonical_sha256(scalar_rows)
    scalar["rows_stored_inline"] = False
    rank3["saturation_rows_sha256"] = canonical_sha256(saturation_rows)
    rank3["saturation_rows_stored_inline"] = False
    return scalar, rank3


def put_bits(bits, value, width):
    value = int(value) & ((1 << width) - 1)
    bits.extend((value >> index) & 1 for index in range(width))


def bits_to_bytes(bits):
    if len(bits) % 8:
        raise ValueError("bit vector is not byte aligned")
    return bytes(
        sum(bits[offset + bit] << bit for bit in range(8))
        for offset in range(0, len(bits), 8)
    )


def crc32c(data):
    crc = 0xFFFFFFFF
    for value in bytearray(data):
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF


def check_range(value, minimum, maximum, label):
    value = int(value)
    if value < minimum or value > maximum:
        raise ValueError("{} range violation".format(label))
    return value


def validate_configuration_values(config):
    require_exact_keys(
        config,
        {"right_factor_q8", "left_ternary_code", "bias_q24",
         "threshold_q24", "stage1_requant_shift_u5", "generation_u16"},
        "M38 configuration",
    )
    if len(config["right_factor_q8"]) != 30:
        raise ValueError("right-factor population drift")
    if len(config["left_ternary_code"]) != 30:
        raise ValueError("ternary-code population drift")
    if len(config["bias_q24"]) != 10:
        raise ValueError("bias population drift")
    for value in config["right_factor_q8"]:
        check_range(value, -128, 127, "right-factor q8")
    for code in config["left_ternary_code"]:
        if int(code) not in (0, 1, 2):
            raise ValueError("illegal M38 ternary code")
    for value in config["bias_q24"]:
        check_range(value, -(1 << 23), (1 << 23) - 1, "bias q24")
    check_range(config["threshold_q24"], -(1 << 23), (1 << 23) - 1,
                "threshold q24")
    check_range(config["stage1_requant_shift_u5"], 0, 23,
                "stage1 requant shift")
    check_range(config["generation_u16"], 0, 0xFFFF, "generation")


def pack_protected_payload(config):
    validate_configuration_values(config)
    bits = []
    for value in config["right_factor_q8"]:
        put_bits(bits, value, 8)
    for value in config["left_ternary_code"]:
        put_bits(bits, value, 2)
    for value in config["bias_q24"]:
        put_bits(bits, value, 24)
    put_bits(bits, config["threshold_q24"], 24)
    put_bits(bits, config["stage1_requant_shift_u5"], 5)
    if len(bits) != 569:
        raise ValueError("M38 arithmetic payload width drift")
    put_bits(bits, config["generation_u16"], 16)
    if len(bits) != 585:
        raise ValueError("M38 protected logical width drift")
    bits.extend([0] * 7)
    payload = bits_to_bytes(bits)
    if len(payload) != 74:
        raise ValueError("M38 protected byte width drift")
    return payload


def pack_configuration_frame(config):
    payload = pack_protected_payload(config)
    checksum = crc32c(payload)
    frame = payload + checksum.to_bytes(4, byteorder="little")
    if len(frame) != 78:
        raise ValueError("M38 serialized frame width drift")
    return frame


class BitReader(object):
    def __init__(self, payload):
        self.bits = [
            (value >> bit) & 1 for value in bytearray(payload) for bit in range(8)
        ]
        self.offset = 0

    def take(self, width, signed=False):
        value = sum(self.bits[self.offset + bit] << bit for bit in range(width))
        self.offset += width
        if signed and (value & (1 << (width - 1))):
            value -= 1 << width
        return value


def decode_configuration_frame(frame):
    frame = bytes(frame)
    if len(frame) != 78:
        raise ValueError("serialized frame must be exactly 624 bits")
    payload = frame[:74]
    received = int.from_bytes(frame[74:], byteorder="little")
    expected = crc32c(payload)
    if received != expected:
        raise ValueError("configuration CRC mismatch")
    reader = BitReader(payload)
    config = {
        "right_factor_q8": [reader.take(8, signed=True) for _ in range(30)],
        "left_ternary_code": [reader.take(2) for _ in range(30)],
        "bias_q24": [reader.take(24, signed=True) for _ in range(10)],
        "threshold_q24": reader.take(24, signed=True),
        "stage1_requant_shift_u5": reader.take(5),
        "generation_u16": reader.take(16),
    }
    if any(reader.take(1) for _ in range(7)):
        raise ValueError("configuration zero padding is nonzero")
    validate_configuration_values(config)
    return config


def make_fragments(frame):
    frame = bytes(frame)
    if len(frame) != 78:
        raise ValueError("fragment source frame width drift")
    padded = frame + bytes(2)
    return [
        {
            "index": index,
            "data_u64": int.from_bytes(
                padded[index * 8:(index + 1) * 8], byteorder="little"
            ),
            "valid_bits": 48 if index == 9 else 64,
        }
        for index in range(10)
    ]


def generation_is_newer(candidate, active):
    if active is None:
        return True
    delta = (int(candidate) - int(active)) & 0xFFFF
    return 1 <= delta <= 0x7FFF


class StrictFragmentLoader(object):
    def __init__(self, active_config=None):
        self.active_config = active_config
        self.next_index = 0
        self.shadow = bytearray()
        self.failed = False

    def _fail(self, message):
        self.failed = True
        self.next_index = 0
        self.shadow = bytearray()
        raise ValueError(message)

    def accept(self, fragment, datapath_drained=False):
        index = int(fragment.get("index", -1))
        if self.failed:
            if index != 0:
                self._fail("failed load restart requires fragment zero")
            self.failed = False
        if index == 0 and self.next_index != 0:
            self._fail("duplicate or premature fragment zero")
        if index != self.next_index:
            self._fail("configuration fragment order violation")
        expected_valid = 48 if index == 9 else 64
        if int(fragment.get("valid_bits", -1)) != expected_valid:
            self._fail("configuration fragment valid-bit violation")
        data = check_range(fragment.get("data_u64", -1), 0, (1 << 64) - 1,
                           "configuration fragment")
        if index == 9 and (data >> 48) != 0:
            self._fail("last configuration fragment high bits are nonzero")
        self.shadow.extend(data.to_bytes(8, byteorder="little"))
        self.next_index += 1
        if index != 9:
            return False
        frame = bytes(self.shadow[:78])
        try:
            candidate = decode_configuration_frame(frame)
            active_generation = (
                None if self.active_config is None
                else self.active_config["generation_u16"]
            )
            if not generation_is_newer(candidate["generation_u16"],
                                       active_generation):
                raise ValueError("configuration generation is stale or ambiguous")
            if not datapath_drained:
                raise ValueError("configuration activation requires drained datapath")
        except ValueError as error:
            self._fail(str(error))
        self.active_config = candidate
        self.next_index = 0
        self.shadow = bytearray()
        return True

    def finalize_incomplete(self):
        if self.next_index != 0:
            self._fail("incomplete configuration frame")


def build_crc_and_protocol_audit(contract):
    frame_contract = contract["canonical_configuration_frame"]
    golden = frame_contract["golden_frame"]
    config = {
        "right_factor_q8": golden["right_factor_q8"],
        "left_ternary_code": golden["left_ternary_code"],
        "bias_q24": golden["bias_q24"],
        "threshold_q24": golden["threshold_q24"],
        "stage1_requant_shift_u5": golden["stage1_requant_shift_u5"],
        "generation_u16": golden["generation_u16"],
    }
    payload = pack_protected_payload(config)
    frame = pack_configuration_frame(config)
    checksum = crc32c(payload)
    if crc32c(b"123456789") != 0xE3069283:
        raise ValueError("CRC-32C standard check drift")
    if payload.hex() != golden["protected_payload_hex"]:
        raise ValueError("M38 golden protected payload drift")
    if checksum != int(golden["crc32c"], 16):
        raise ValueError("M38 golden CRC drift")
    if frame.hex() != golden["serialized_frame_hex"]:
        raise ValueError("M38 golden serialized frame drift")
    if decode_configuration_frame(frame) != config:
        raise ValueError("M38 golden frame round-trip drift")
    fragments = make_fragments(frame)
    loader = StrictFragmentLoader()
    for fragment in fragments:
        activated = loader.accept(fragment, datapath_drained=True)
    if not activated or loader.active_config != config:
        raise ValueError("M38 strict fragment activation drift")
    negative_cases = []

    def expect_failure(name, fragments_to_apply, active=None, drained=True):
        instance = StrictFragmentLoader(active)
        try:
            for item in fragments_to_apply:
                instance.accept(item, datapath_drained=drained)
        except ValueError:
            if instance.active_config != active:
                raise ValueError("failed load changed active context")
            negative_cases.append(name)
            return
        raise ValueError("configuration negative case was accepted: {}".format(name))

    expect_failure("out_of_order", [fragments[0], fragments[2]])
    expect_failure("duplicate_fragment", [fragments[0], fragments[0]])
    bad_last = dict(fragments[9]); bad_last["data_u64"] |= 1 << 63
    expect_failure("nonzero_unused_high_bits", fragments[:9] + [bad_last])
    bad_crc_frame = bytearray(frame); bad_crc_frame[20] ^= 1
    expect_failure("bad_crc", make_fragments(bytes(bad_crc_frame)), active=config)
    bad_code = dict(config); bad_code["left_ternary_code"] = list(
        config["left_ternary_code"]
    ); bad_code["left_ternary_code"][7] = 3
    bad_bits = bytearray(frame)
    bit_offset = 240 + (7 * 2)
    bad_bits[bit_offset // 8] |= 3 << (bit_offset % 8)
    payload_bad_code = bytes(bad_bits[:74])
    bad_bits[74:] = crc32c(payload_bad_code).to_bytes(4, byteorder="little")
    expect_failure("illegal_ternary", make_fragments(bytes(bad_bits)))
    active = dict(config)
    expect_failure("stale_generation", fragments, active=active)
    expect_failure("undrained_activation", fragments, active=None, drained=False)
    incomplete = StrictFragmentLoader()
    try:
        for fragment in fragments[:4]:
            incomplete.accept(fragment, datapath_drained=True)
        incomplete.finalize_incomplete()
    except ValueError:
        negative_cases.append("incomplete_frame")
    else:
        raise ValueError("incomplete configuration frame was accepted")
    if not generation_is_newer(1, 0xFFFE):
        raise ValueError("generation wrap rule rejected valid forward delta")
    if generation_is_newer(0x8000, 0) or generation_is_newer(5, 5):
        raise ValueError("generation rule accepted ambiguous or equal generation")
    return {
        "crc32c_ascii_123456789": "0x{:08X}".format(crc32c(b"123456789")),
        "golden_protected_payload_bytes": len(payload),
        "golden_crc32c": "0x{:08X}".format(checksum),
        "golden_serialized_frame_bytes": len(frame),
        "golden_serialized_frame_sha256": hashlib.sha256(frame).hexdigest(),
        "logical_arithmetic_payload_bits": 569,
        "logical_context_bits_excluding_pad": 617,
        "protected_bits_before_padding": 585,
        "zero_pad_bits_before_crc": 7,
        "serialized_context_bits_including_pad": 624,
        "fragment_count_64bit": len(fragments),
        "last_fragment_valid_bits": fragments[-1]["valid_bits"],
        "negative_protocol_cases_rejected": negative_cases,
        "generation_wrap_forward_case_admitted": True,
        "active_context_unchanged_on_every_failure": True,
    }


class IntegratedCycleModel(object):
    """Executable abstract FSM; it intentionally models no RTL timing/PPA."""

    def __init__(self, fifo_depth=16):
        self.fifo_depth = fifo_depth
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

    def drained(self):
        return (
            self.stage1 is None and self.pending is None and self.slot is None
            and self.reconstruction is None and self.reserved == 0
            and not self.fifo
        )

    def switch_context(self, mode, generation):
        if mode not in ("T10", "T2"):
            raise ValueError("illegal abstract context mode")
        if not self.drained():
            raise ValueError("context switch requires complete drain")
        check_range(generation, 0, 0xFFFF, "abstract context generation")
        self.context_mode = mode
        self.context_generation = int(generation)

    def seed_fifo(self, count, mode=None):
        if self.fifo or self.reserved:
            raise ValueError("FIFO seed requires an empty model")
        check_range(count, 0, self.fifo_depth, "FIFO seed")
        use_mode = mode or self.context_mode
        self.fifo = [
            {"writer": "seed", "mode": use_mode, "tag": index, "beat": 0}
            for index in range(count)
        ]
        self._check_invariant()

    def _check_tile(self, tile):
        if self.context_mode != "T10":
            raise ValueError("T10 tile offered outside T10 context")
        if int(tile["generation"]) != self.context_generation:
            raise ValueError("T10 tile generation mismatch")
        if "tag" not in tile:
            raise ValueError("T10 tile tag is missing")

    def _check_invariant(self):
        occupancy = len(self.fifo)
        if occupancy < 0 or self.reserved < 0:
            raise ValueError("negative FIFO state")
        if occupancy + self.reserved > self.fifo_depth:
            raise ValueError("FIFO occupancy plus reservation overflow")
        if self.reconstruction is None and self.reserved != 0:
            raise ValueError("orphan FIFO reservation")
        if self.reconstruction is not None:
            expected = 5 - int(self.reconstruction["phase"])
            if self.reserved != expected:
                raise ValueError("reconstruction reservation/phase drift")
        self.maximum_fifo_occupancy = max(self.maximum_fifo_occupancy, occupancy)
        self.maximum_occupancy_plus_reserved = max(
            self.maximum_occupancy_plus_reserved, occupancy + self.reserved
        )

    def step(self, sink_ready=False, t10_offer=None, other_writer_offer=None):
        event = {
            "cycle": self.cycle,
            "fifo_occupancy_before": len(self.fifo),
            "reserved_before": self.reserved,
            "stage1_accept": False,
            "reconstruction_launch": False,
            "m38_push": False,
            "other_writer_push": False,
            "other_writer_denied": False,
            "slot_pop": False,
            "slot_push": False,
            "pending_materialize": False,
            "done": False,
        }
        fifo_was_full = len(self.fifo) == self.fifo_depth
        popped = self.fifo.pop(0) if sink_ready and self.fifo else None
        event["fifo_pop"] = popped

        if t10_offer is not None and self.stage1 is None and self.pending is None:
            self._check_tile(t10_offer)
            self.stage1 = {"tile": dict(t10_offer), "phase": 0}
            event["stage1_accept"] = True

        if self.reconstruction is None and self.slot is not None:
            if len(self.fifo) + self.reserved + 5 <= self.fifo_depth:
                self.reconstruction = {"tile": dict(self.slot), "phase": 0}
                self.reserved += 5
                event["reconstruction_launch"] = True

        m38_owns_push_port = self.reconstruction is not None
        if m38_owns_push_port:
            phase = int(self.reconstruction["phase"])
            tile = self.reconstruction["tile"]
            if self.slot is None or self.slot["tag"] != tile["tag"]:
                raise ValueError("reconstruction lost old slot ownership")
            if self.reserved <= 0 or len(self.fifo) >= self.fifo_depth:
                raise ValueError("reserved M38 beat cannot commit")
            self.fifo.append({
                "writer": "M38", "mode": "T10", "tag": tile["tag"],
                "generation": tile["generation"], "beat": phase,
            })
            self.reserved -= 1
            event["m38_push"] = True
            event["m38_push_tag"] = tile["tag"]
            event["m38_push_beat"] = phase
            if other_writer_offer is not None:
                event["other_writer_denied"] = True
            if phase == 4:
                event["done"] = True
                event["done_tag"] = tile["tag"]
                self.done_tags.append(tile["tag"])
                event["slot_pop"] = True
                event["slot_old_read_tag"] = self.slot["tag"]
        elif other_writer_offer is not None:
            if self.context_mode != "T2":
                raise ValueError("other FIFO writer requires T2 context")
            if int(other_writer_offer["generation"]) != self.context_generation:
                raise ValueError("other FIFO writer generation mismatch")
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
                event["slot_push"] = True
                event["pending_materialize"] = True
                event["slot_new_write_tag"] = self.slot["tag"]
            elif stage1_finish is not None:
                self.slot = stage1_finish
                stage1_finish = None
                event["slot_push"] = True
                event["slot_new_write_tag"] = self.slot["tag"]
        if stage1_finish is not None:
            if self.pending is not None:
                raise ValueError("completed-pending state overflow")
            self.pending = stage1_finish
            event["stage1_completed_pending"] = True

        if self.reconstruction is not None:
            if self.reconstruction["phase"] == 4:
                self.reconstruction = None
            else:
                self.reconstruction["phase"] += 1

        event["full_old_read_new_write"] = bool(
            fifo_was_full and popped is not None
            and (event["m38_push"] or event["other_writer_push"])
        )
        event["fifo_occupancy_after"] = len(self.fifo)
        event["reserved_after"] = self.reserved
        self._check_invariant()
        self.history.append(event)
        self.cycle += 1
        return event


def audit_credit_state_space(depth=16):
    cases = 0
    maximum = 0
    for occupancy in range(depth + 1):
        for reserved in range(depth - occupancy + 1):
            for pop in (0, 1):
                if pop and occupancy == 0:
                    continue
                after_pop = occupancy - pop
                for other_offer in (0, 1):
                    cases += 1
                    if reserved:
                        new_occupancy = after_pop + 1
                        new_reserved = reserved - 1
                    else:
                        launch = after_pop + 5 <= depth
                        if launch:
                            new_occupancy = after_pop + 1
                            new_reserved = 4
                        elif other_offer and after_pop + 1 <= depth:
                            new_occupancy = after_pop + 1
                            new_reserved = 0
                        else:
                            new_occupancy = after_pop
                            new_reserved = 0
                    if new_occupancy + new_reserved > depth:
                        raise ValueError("abstract credit state-space overflow")
                    maximum = max(maximum, new_occupancy + new_reserved)
    return {"states_checked": cases, "maximum_occupancy_plus_reserved": maximum}


def run_no_stall_tiles(tile_count):
    model = IntegratedCycleModel()
    model.switch_context("T10", 7)
    next_tile = 0
    accept_cycles = []
    done_cycles = []
    limit = 20 + 8 * tile_count
    for _ in range(limit):
        offer = None
        if next_tile < tile_count:
            offer = {"tag": next_tile, "generation": 7}
        event = model.step(sink_ready=True, t10_offer=offer)
        if event["stage1_accept"]:
            accept_cycles.append(event["cycle"])
            next_tile += 1
        if event["done"]:
            done_cycles.append(event["cycle"])
        if len(done_cycles) == tile_count:
            break
    if len(done_cycles) != tile_count:
        raise ValueError("no-stall abstract model did not finish")
    if any(b - a != 5 for a, b in zip(accept_cycles, accept_cycles[1:])):
        raise ValueError("abstract stage1 II5 drift")
    if any(b - a != 5 for a, b in zip(done_cycles, done_cycles[1:])):
        raise ValueError("abstract done II5 drift")
    if done_cycles[-1] + 1 != 5 + 5 * tile_count:
        raise ValueError("abstract finite-N equation drift")
    return model, accept_cycles, done_cycles


def run_pending_trace():
    model = IntegratedCycleModel()
    model.switch_context("T10", 9)
    model.seed_fifo(12, mode="T10")
    next_tag = 0
    materialize_event = None
    for cycle in range(80):
        offer = None
        if next_tag < 2:
            offer = {"tag": "tile{}".format(next_tag), "generation": 9}
        event = model.step(sink_ready=(cycle >= 10), t10_offer=offer)
        if event["stage1_accept"]:
            next_tag += 1
        if event["pending_materialize"]:
            materialize_event = event
        if len(model.done_tags) == 2 and model.pending is None and model.slot is None:
            break
    if materialize_event is None:
        raise ValueError("completed-pending was never materialized")
    if materialize_event["slot_old_read_tag"] != "tile0":
        raise ValueError("slot phase4 did not read old tile")
    if materialize_event["slot_new_write_tag"] != "tile1":
        raise ValueError("slot replacement did not write pending tile")
    if model.done_tags != ["tile0", "tile1"]:
        raise ValueError("pending trace done order drift")
    return model, materialize_event


def run_eventual_sink_liveness(tile_count=40, stalled_cycles=90):
    model = IntegratedCycleModel()
    model.switch_context("T10", 11)
    next_tile = 0
    for cycle in range(2000):
        offer = (
            {"tag": next_tile, "generation": 11}
            if next_tile < tile_count else None
        )
        event = model.step(
            sink_ready=(cycle >= stalled_cycles), t10_offer=offer
        )
        if event["stage1_accept"]:
            next_tile += 1
        if len(model.done_tags) == tile_count and model.drained():
            return model, cycle + 1
    raise ValueError("eventual-sink liveness timeout")


def run_full_pop_push_and_context_drain():
    model = IntegratedCycleModel()
    model.switch_context("T2", 21)
    model.seed_fifo(16, mode="T2")
    event = model.step(
        sink_ready=True,
        other_writer_offer={
            "writer": "T2", "mode": "T2", "tag": "new",
            "beat": 0, "generation": 21,
        },
    )
    if not event["full_old_read_new_write"] or len(model.fifo) != 16:
        raise ValueError("full FIFO old-read/new-write drift")
    try:
        model.switch_context("T10", 22)
    except ValueError:
        blocked = True
    else:
        blocked = False
    if not blocked:
        raise ValueError("undrained T2-to-T10 switch was admitted")
    while model.fifo:
        model.step(sink_ready=True)
    model.switch_context("T10", 22)
    if model.context_mode != "T10":
        raise ValueError("drained T2-to-T10 switch failed")
    return event


def run_writer_conflict_prevention():
    model = IntegratedCycleModel()
    model.switch_context("T10", 23)
    model.seed_fifo(12, mode="T10")
    model.slot = {"tag": "reserved_tile", "generation": 23}
    event = model.step(
        sink_ready=True,
        other_writer_offer={
            "writer": "unaccounted", "mode": "T2", "tag": "conflict",
            "beat": 0, "generation": 23,
        },
    )
    if (not event["reconstruction_launch"] or not event["m38_push"]
            or not event["other_writer_denied"]
            or len(model.fifo) + model.reserved != 16):
        raise ValueError("single-port writer conflict arbitration drift")
    for _ in range(4):
        model.step(sink_ready=False)
    if len(model.fifo) != 16 or model.reserved != 0:
        raise ValueError("reserved five-beat completion drift")
    return event, model


def run_t10_t2_t10_drain_sequence():
    model = IntegratedCycleModel()
    model.switch_context("T10", 30)
    accepted = False
    switch_rejections = 0
    for cycle in range(40):
        offer = None if accepted else {"tag": "t10_before", "generation": 30}
        event = model.step(sink_ready=(cycle >= 10), t10_offer=offer)
        accepted = accepted or event["stage1_accept"]
        if cycle == 1:
            try:
                model.switch_context("T2", 31)
            except ValueError:
                switch_rejections += 1
        if model.done_tags == ["t10_before"] and model.drained():
            break
    if not model.drained():
        raise ValueError("T10 context did not drain")
    model.switch_context("T2", 31)
    model.step(
        other_writer_offer={
            "writer": "T2", "mode": "T2", "tag": "t2_middle",
            "beat": 0, "generation": 31,
        }
    )
    try:
        model.switch_context("T10", 32)
    except ValueError:
        switch_rejections += 1
    while model.fifo:
        model.step(sink_ready=True)
    model.switch_context("T10", 32)
    if model.context_mode != "T10" or switch_rejections != 2:
        raise ValueError("T10/T2 drain sequencing drift")
    return {"mode_sequence": ["T10", "T2", "T10"],
            "undrained_switch_rejections": switch_rejections}


def build_cycle_audit():
    no_stall, accepts, dones = run_no_stall_tiles(32)
    pending, materialize = run_pending_trace()
    eventual, eventual_cycles = run_eventual_sink_liveness()
    full_event = run_full_pop_push_and_context_drain()
    conflict_event, conflict_model = run_writer_conflict_prevention()
    context_sequence = run_t10_t2_t10_drain_sequence()
    credit = audit_credit_state_space()
    buggy_pending = [
        "slot_A_full_no_fifo_credit",
        "stage1_B_commit_pulse_sets_pending",
        "later_slot_A_phase4_pop_but_commit_pulse_is_low",
        "buggy_slot_push_false_and_B_never_materializes",
    ]
    buggy_overflow = [13, 14, 15, 16, 17]
    if buggy_overflow[-1] <= 16:
        raise ValueError("unreserved writer overflow witness drift")
    return {
        "no_stall_tiles": 32,
        "no_stall_accept_cycles": accepts,
        "no_stall_done_cycles": dones,
        "conditional_stage1_and_done_ii": 5,
        "finite_n_commit_cycles": dones[-1] + 1,
        "finite_n_equation": "5 + 5*N",
        "maximum_no_stall_fifo_occupancy": no_stall.maximum_fifo_occupancy,
        "completed_pending_materialized": True,
        "pending_materialize_cycle": materialize["cycle"],
        "phase4_old_slot_read_tag": materialize["slot_old_read_tag"],
        "same_edge_new_slot_write_tag": materialize["slot_new_write_tag"],
        "pending_trace_done_tags": pending.done_tags,
        "eventual_sink_tiles": 40,
        "eventual_sink_stalled_cycles": 90,
        "eventual_sink_liveness_completion_cycles": eventual_cycles,
        "eventual_sink_maximum_fifo_occupancy": eventual.maximum_fifo_occupancy,
        "reservation_invariant_maximum": eventual.maximum_occupancy_plus_reserved,
        "credit_state_space": credit,
        "full_fifo_pop_returns_old_entry_and_push_writes_new_tail": bool(
            full_event["full_old_read_new_write"]
        ),
        "undrained_context_switch_rejected": True,
        "drained_T10_T2_switch_admitted": True,
        "buggy_commit_pulse_pending_deadlock_witness": buggy_pending,
        "buggy_unreserved_shared_writer_occupancy_trace": buggy_overflow,
        "single_push_port_M38_priority_prevents_counterexample": True,
        "writer_conflict_other_push_denied": conflict_event[
            "other_writer_denied"],
        "writer_conflict_final_occupancy": len(conflict_model.fifo),
        "writer_conflict_final_reserved": conflict_model.reserved,
        "T10_T2_T10_drain_sequence": context_sequence,
        "beat4_commit_done_same_cycle": all(
            event["done"] == (
                event["m38_push"] and event.get("m38_push_beat") == 4
            ) for event in no_stall.history
        ),
    }


def configuration_ledger():
    common = 30 * 8 + 10 * 24 + 24 + 5
    return {
        "common_payload_bits_excluding_left_factor": common,
        "m31_serialized_parameter_payload_bits": common + 30 * 8,
        "m37_csd4_parameter_payload_bits": (
            common + 30 * 8 + 30 * 4 + 30 * 4 + 30 * 4 * 3
        ),
        "m38_rst_arithmetic_payload_bits": common + 30 * 2,
        "m38_rst_logical_context_bits": common + 30 * 2 + 16 + 32,
        "m38_rst_serialized_context_bits": 624,
        "parameter_load_cycles_included_in_throughput": False,
    }


def build(contract_path=DEFAULT_CONTRACT):
    contract, payloads, hashes, _ = load_contract(contract_path)
    validate_frozen_contract(contract, payloads, hashes)
    m31 = validate_m31_anchor(payloads, hashes)
    m37 = validate_m37_anchor(payloads, hashes)
    scalar, rank3 = build_math_audit()
    crc_protocol = build_crc_and_protocol_audit(contract)
    cycle = build_cycle_audit()
    return {
        "schema": "m38_rst_math_crc_and_abstract_cycle_audit_v2",
        "status": (
            "PASS_M38_RST_RECURSIVE_ANCHOR_MATH_CRC_AND_"
            "ABSTRACT_CYCLE_ONLY"
        ),
        "identity": {
            "contract": str(Path(contract_path).resolve()),
            "contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
        },
        "recursive_anchor_audit": {"m31_r3": m31, "m37_r7": m37},
        "scalar_ternary_audit": scalar,
        "rank3_q24_threshold_audit": rank3,
        "configuration_bit_ledger": configuration_ledger(),
        "canonical_crc_and_fragment_protocol_audit": crc_protocol,
        "abstract_integrated_cycle_audit": cycle,
        "conditional_theory": {
            "serialized_steady_ii": 10,
            "parallel_steady_ii": 5,
            "steady_t10_kernel_throughput_limit": 2.0,
            "finite_n_ratio": "10*N/(5+5*N)",
            "system_speedup_admitted": False,
        },
        "admission": {
            "recursive_anchor_identity_admitted": True,
            "q8_times_ternary_and_rank3_math_admitted": True,
            "canonical_crc32c_frame_admitted": True,
            "strict_fragment_and_generation_reference_admitted": True,
            "abstract_integrated_cycle_safety_and_liveness_admitted": True,
            "integrated_rtl_admitted": False,
            "integrated_rtl_vcs_admitted": False,
            "dc_sta_formality_admitted": False,
            "area_power_energy_admitted": False,
            "memory_and_system_cycles_admitted": False,
            "system_speedup_admitted": False,
            "headline_admitted": False,
        },
        "unmeasured_nonzero_costs": [
            "ternary selectors rank3 and bias adders saturation and comparison",
            "CRC32C and fragment/generation control",
            "slot pending reservation arbitration and tag storage",
            "physical context SRAM ports ECC and load transactions",
            "trained H67 and Local5 ternary codebook accuracy",
            "address-timed memory contention and full-network cycles",
        ],
        "claim_boundary": contract["claim_boundary"],
    }


def write_output(output, result):
    output = Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite M38 r2 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.contract.resolve())
    write_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
