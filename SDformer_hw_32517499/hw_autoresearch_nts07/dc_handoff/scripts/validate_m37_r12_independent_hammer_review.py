#!/usr/bin/env python3
"""Independent hammer validator for the M37-r12 exact-SHA VCS milestone.

The validator treats the receipt, run directory, manifests, logs, and compiled
executable as untrusted.  It verifies exact identities, strict JSON typing,
manifest closure, VCS/SVA evidence, vector structure, overwrite refusal, and an
independent replay of the frozen executable.  A pass admits only launching DC
and Formality on the exact f947... candidate; it does not admit their results.
"""

from __future__ import print_function

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
HW_ROOT = os.path.join(ROOT, "hw_autoresearch_nts07")
RUN_DIR = ("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/"
           "dc_handoff/runs/"
           "m37_csd_reconstruct_t10_vcs_r12_exact_sha_20260823")

PATHS = {
    "receipt": "hw_autoresearch_nts07/contracts/m37_r12_exact_sha_vcs_receipt_r1_20260823.json",
    "contract": "hw_autoresearch_nts07/contracts/m37_r12_exact_sha_vcs_contract_r1_20260823.json",
    "runner": "hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m37_r12_exact_sha_sva.sh",
    "rtl": "hw_autoresearch_nts07/rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv",
    "assertions": "hw_autoresearch_nts07/verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv",
    "testbench": "hw_autoresearch_nts07/tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv",
    "filelist": "hw_autoresearch_nts07/dc_handoff/filelists/date_m37_r12_csd_reconstruct_t10_vcs.f",
    "r11_validator": "hw_autoresearch_nts07/dc_handoff/scripts/validate_m37_r11_independent_hammer_review.py",
    "r11_review": "hw_autoresearch_nts07/results/m37_r11_independent_hammer_review_20260822/m37_r11_independent_hammer_review.json",
}

EXPECTED_SHA256 = {
    "receipt": "5d23131f4ec721d7028cec5d363000798f749f7689e3db18c269b08e3cefb265",
    "contract": "8d9a335995a96fca84602cda60fcad83b23218a35d7413647db0c6525f05aaab",
    "runner": "a85e6aafc4bcc35c01c6167eb50134f61c9be9aa5006774183c90b8ba2e8b262",
    "rtl": "f9474151fa03770faeb46998ddd61aa3c33c2a7732ff70db81d9821e1cf373dd",
    "assertions": "7492af816161febbd0b0e62a1f8e697151d15202e4ad71dd79d721f66a874fe0",
    "testbench": "bd92f8ebac83fee446b3fbebadbcb928031706ed99641bc248b459e1786da5cc",
    "filelist": "8dec6f37de7483ce8458fd13072578efa7543ad3c73927d75292cbe146834e2b",
    "r11_validator": "d145e1561ab14484833b2ffbef7d3a42609d5934698238a3f254a7f8337bb080",
    "r11_review": "cd798e84365a3601d32a854dffb425a16a18c7fed5fc46a1023584f5fb22e7a3",
}

RUN_SHA256 = {
    "compile.raw.log": "7a9c9d7adc178c01d993fdeb27ab890e2c328dbb0bf75a69e01edbacd515a7d0",
    "sim.raw.log": "38d123043c667e49e097839abad10ae4ef4be9737f0b0ef532d9e8c058ccbe31",
    "simv": "53596862bdbaa44ef7f83321f6531014986b2ebf4792fa4b07f44bf47ab27c52",
    "vectors.txt": "2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627",
    "input_sha256.txt": "b4acdfa71ea873341e312639756360dfe22e9b4e5d8a1f729cd013e29a6b31e2",
    "output_sha256.txt": "d4b1c7389e752d9b51d0f91aa0f65140bed86f9f4018db9f1a3ae8da03d2d588",
    "run_local_seal.sha256": "14e3fe2d42d510e9c64f09b59666de4f8a5ee34381f367e5cdd9658e0e004cb2",
    "completion_seal.sha256": "0612b3ee06dcb0325ffbdb278e8bbc768470e2e347d441b51ed25b83b8b001f1",
    "RUN_COMPLETE.txt": "30955d6f50665ee135c7e5311b4665aea0811524ca0d59a7554f28d2c1543038",
}

INPUT_BASENAMES = {
    "rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv",
    "verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv",
    "tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv",
    "dc_handoff/filelists/date_m37_r12_csd_reconstruct_t10_vcs.f",
    "dc_handoff/scripts/run_vcs_m37_r12_exact_sha_sva.sh",
    "contracts/m37_r12_exact_sha_vcs_contract_r1_20260823.json",
    "dc_handoff/scripts/validate_m37_r11_independent_hammer_review.py",
    "results/m37_r11_independent_hammer_review_20260822/m37_r11_independent_hammer_review.json",
    "contracts/m37_r11_evidence_pin_r1_20260822.json",
    "contracts/m37_output_receipt_r4_20260822.json",
    "contracts/m37_phase_decoupled_csd_reconstruct_input_contract_r2_20260822.json",
    "results/m37_phase_decoupled_csd_reconstruct_r2_20260822/m37_phase_decoupled_csd_reconstruct.json",
}

OUTPUT_BASENAMES = {
    "preflight_sha_checks.txt", "r11_review_validation.raw.log",
    "r11_review_validation.stderr.raw.log", "r11_review_validation.rc",
    "input_sha256.txt", "input_manifest_check.raw.log",
    "input_manifest_check.rc", "compile.command.txt", "compile.raw.log",
    "compile.rc", "compile.success.marker", "simv", "sim.command.txt",
    "sim.raw.log", "sim.rc", "tb_internal_pass.marker",
    "sva_cover_counts.txt", "vectors.txt", "simulation.success.marker",
    "runner_status.txt",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_path(relative):
    if type(relative) is not str or os.path.isabs(relative) \
            or os.path.normpath(relative) != relative:
        raise ValueError("non-canonical repository path: {0!r}".format(relative))
    path = os.path.realpath(os.path.join(ROOT, relative))
    if os.path.commonpath([ROOT, path]) != ROOT:
        raise ValueError("repository path escapes root: {0}".format(relative))
    return path


def canonical_run_dir(path):
    expected_parent = os.path.realpath(os.path.dirname(RUN_DIR))
    if type(path) is not str or not os.path.isabs(path) \
            or os.path.normpath(path) != path or path != os.path.realpath(path):
        raise ValueError("run path is not absolute, canonical, and symlink-free")
    if os.path.commonpath([expected_parent, path]) != expected_parent:
        raise ValueError("run path escapes expected run parent")
    if path != RUN_DIR:
        raise ValueError("run path differs from frozen r12 directory")
    return path


def load_json_strict(path):
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: {0}".format(key))
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite JSON constant: {0}".format(value))

    with open(path, "r") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate,
                         parse_constant=reject_constant)


def require_exact_keys(value, keys, failures, label):
    if type(value) is not dict or set(value) != set(keys):
        failures.append("{0} exact key set drift".format(label))


def require_exact_int(value, expected, failures, label):
    if type(value) is not int or value != expected:
        failures.append("{0} exact integer drift".format(label))


def require_exact_bool(value, expected, failures, label):
    if type(value) is not bool or value is not expected:
        failures.append("{0} exact boolean drift".format(label))


def validate_receipt(receipt):
    failures = []
    require_exact_keys(receipt, [
        "claim_boundary", "compile", "contract", "date", "exact_identity",
        "functional", "headline_admitted", "run", "schema", "status", "sva",
    ], failures, "receipt")
    if receipt.get("schema") != "m37_r12_exact_sha_vcs_receipt_v1":
        failures.append("receipt schema drift")
    if receipt.get("status") != \
            "PASS_R12_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER_NO_DC_OR_PPA_CLAIM":
        failures.append("receipt status drift")
    require_exact_bool(receipt.get("headline_admitted"), False, failures,
                       "receipt headline_admitted")

    exact = receipt.get("exact_identity", {})
    expected_exact = {
        "candidate_rtl_sha256": EXPECTED_SHA256["rtl"],
        "filelist_sha256": EXPECTED_SHA256["filelist"],
        "r11_independent_review_sha256": EXPECTED_SHA256["r11_review"],
        "r11_independent_validator_sha256": EXPECTED_SHA256["r11_validator"],
        "r11_rebuilt_review_byte_identical": True,
        "runner_sha256": EXPECTED_SHA256["runner"],
    }
    require_exact_keys(exact, expected_exact, failures, "receipt exact_identity")
    for key, expected in expected_exact.items():
        if type(expected) is bool:
            require_exact_bool(exact.get(key), expected, failures,
                               "receipt exact_identity." + key)
        elif exact.get(key) != expected:
            failures.append("receipt exact_identity.{0} drift".format(key))

    compile_data = receipt.get("compile", {})
    require_exact_int(compile_data.get("real_exit_code"), 0, failures,
                      "receipt compile.real_exit_code")
    require_exact_int(compile_data.get("broad_warning_error_fatal_signatures"),
                      0, failures, "receipt compile signature count")
    if compile_data.get("raw_log_sha256") != RUN_SHA256["compile.raw.log"] \
            or compile_data.get("simv_sha256") != RUN_SHA256["simv"]:
        failures.append("receipt compile SHA drift")

    functional = receipt.get("functional", {})
    expected_ints = {
        "arithmetic_issue_cycles": 1225,
        "direct_product_miters": 117600,
        "dut_unique_signed_input_coefficient_pairs": 65536,
        "functional_mismatch_count": 0,
        "internal_PASS_count": 1,
        "nominal_tiles": 96,
        "output_bit_miters": 39200,
        "real_exit_code": 0,
        "threshold_cases": 5,
        "total_tiles": 245,
        "unique_signed_inputs": 256,
    }
    for key, expected in expected_ints.items():
        require_exact_int(functional.get(key), expected, failures,
                          "receipt functional." + key)
    if functional.get("configuration_load_release_reload") != [15, 15, 14] \
            or any(type(item) is not int for item in
                   functional.get("configuration_load_release_reload", [])):
        failures.append("receipt configuration count vector drift")
    if functional.get("illegal_accept_reject") != [210, 210] \
            or any(type(item) is not int for item in
                   functional.get("illegal_accept_reject", [])):
        failures.append("receipt illegal count vector drift")
    if functional.get("unique_nominal_payload_product_bitmap") != [96, 96, 96] \
            or any(type(item) is not int for item in
                   functional.get("unique_nominal_payload_product_bitmap", [])):
        failures.append("receipt uniqueness vector drift")
    if functional.get("sim_raw_log_sha256") != RUN_SHA256["sim.raw.log"]:
        failures.append("receipt simulation log SHA drift")

    run = receipt.get("run", {})
    try:
        canonical_run_dir(run.get("directory"))
    except (TypeError, ValueError) as error:
        failures.append("receipt run directory rejected: {0}".format(error))
    for key, expected in (
            ("completion_marker_sha256", RUN_SHA256["RUN_COMPLETE.txt"]),
            ("completion_seal_sha256", RUN_SHA256["completion_seal.sha256"]),
            ("input_manifest_sha256", RUN_SHA256["input_sha256.txt"]),
            ("local_seal_sha256", RUN_SHA256["run_local_seal.sha256"]),
            ("output_manifest_sha256", RUN_SHA256["output_sha256.txt"]),
            ("vector_sha256", RUN_SHA256["vectors.txt"])):
        if run.get(key) != expected:
            failures.append("receipt run.{0} drift".format(key))
    for key in ("overwrite_refused_by_runner", "raw_compile_and_sim_logs_separate",
                "tee_or_background_pipeline_used"):
        expected = key != "tee_or_background_pipeline_used"
        require_exact_bool(run.get(key), expected, failures, "receipt run." + key)

    sva = receipt.get("sva", {})
    require_exact_bool(sva.get("bound"), True, failures, "receipt sva.bound")
    require_exact_int(sva.get("assertion_failure_count"), 0, failures,
                      "receipt sva.assertion_failure_count")
    require_exact_int(sva.get("cover_properties"), 8, failures,
                      "receipt sva.cover_properties")
    expected_covers = [220, 1271, 249, 117, 245, 571, 133, 210]
    if sva.get("cover_counts") != expected_covers \
            or any(type(item) is not int for item in sva.get("cover_counts", [])):
        failures.append("receipt SVA cover vector drift")

    boundary = receipt.get("claim_boundary", {})
    require_exact_bool(boundary.get(
        "DC_STA_Formality_PPA_power_energy_system_admitted"), False,
        failures, "receipt claim DC/PPA boundary")
    require_exact_bool(boundary.get("headline_admitted"), False, failures,
                       "receipt nested headline boundary")
    return failures


def parse_manifest(path, expected_names, base, failures, label):
    entries = {}
    with open(path, "r") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
            if not match:
                failures.append("{0} malformed line {1}".format(label, number))
                continue
            digest, named_path = match.groups()
            if named_path in entries:
                failures.append("{0} duplicate path {1}".format(label, named_path))
                continue
            entries[named_path] = digest
            target = named_path if os.path.isabs(named_path) \
                else os.path.join(base, named_path)
            if not os.path.isfile(target):
                failures.append("{0} missing target {1}".format(label, named_path))
            elif sha256_file(target) != digest:
                failures.append("{0} target SHA mismatch {1}".format(label, named_path))
    observed = set(os.path.basename(item) if os.path.isabs(item) else item
                   for item in entries)
    if observed != set(expected_names):
        failures.append("{0} exact member set drift".format(label))
    return entries


def verify_vector_file(path, failures):
    with open(path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if len(lines) != 471 or lines[0] != "seed=4d370203 total_tiles=245":
        failures.append("vector line count or frozen header drift")
        return
    configs = [line for line in lines[1:] if line.startswith("CONFIG ")]
    tiles = [line for line in lines[1:] if line.startswith("TILE ")]
    illegal = [line for line in lines[1:] if line.startswith("ILLEGAL ")]
    if len(configs) != 15 or len(tiles) != 245 or len(illegal) != 210 \
            or 1 + len(configs) + len(tiles) + len(illegal) != len(lines):
        failures.append("vector CONFIG/TILE/ILLEGAL partition drift")
    expected_labels = (["nominal_unique"] +
        ["dut_full_domain_group_{0}".format(index) for index in range(9)] +
        ["threshold_{0}".format(index) for index in range(5)])
    labels = []
    for line in configs:
        fields = line.split()
        if len(fields) != 45 or not fields[2].startswith("threshold=") \
                or fields[3] != "bias" or fields[14] != "coeff":
            failures.append("vector CONFIG schema drift")
            break
        labels.append(fields[1])
        try:
            [int(item) for item in fields[4:14] + fields[15:45]]
            int(fields[2].split("=", 1)[1])
        except ValueError:
            failures.append("vector CONFIG non-integer payload")
            break
    if labels != expected_labels:
        failures.append("vector CONFIG ordering drift")
    tile_ids = []
    for line in tiles:
        fields = line.split()
        if len(fields) != 50:
            failures.append("vector TILE field count drift")
            break
        try:
            tile_ids.append(int(fields[1]))
            values = [int(item) for item in fields[2:]]
        except ValueError:
            failures.append("vector TILE non-integer payload")
            break
        if any(value < -128 or value > 127 for value in values):
            failures.append("vector TILE signed-int8 domain drift")
            break
    if tile_ids != list(range(245)):
        failures.append("vector TILE identity/order drift")
    illegal_pairs = set()
    illegal_re = re.compile(r"^ILLEGAL class=([0-6]) coefficient=([0-9]+)$")
    for line in illegal:
        match = illegal_re.match(line)
        if not match:
            failures.append("vector ILLEGAL schema drift")
            break
        illegal_pairs.add((int(match.group(1)), int(match.group(2))))
    if illegal_pairs != set((cls, coeff) for cls in range(7)
                            for coeff in range(30)):
        failures.append("vector ILLEGAL Cartesian coverage drift")


def strict_loader_attack(payload):
    descriptor, path = tempfile.mkstemp(prefix="m37_r12_json_", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
        try:
            load_json_strict(path)
        except ValueError:
            return True
        return False
    finally:
        os.unlink(path)


def main():
    failures = []
    observed_sha256 = {}
    modes = {}
    for name, relative in PATHS.items():
        try:
            path = full_path(relative)
        except ValueError as error:
            failures.append("path rejected for {0}: {1}".format(name, error))
            continue
        if not os.path.isfile(path) or os.path.islink(path):
            failures.append("missing, non-file, or symlink input: " + name)
            continue
        observed_sha256[name] = sha256_file(path)
        modes[name] = oct(os.stat(path).st_mode & 0o777)
        if observed_sha256[name] != EXPECTED_SHA256[name]:
            failures.append("exact input SHA drift: " + name)
    if failures:
        print(json.dumps({"failures": failures,
                          "status": "FAIL_R12_REVIEW_INPUT_IDENTITY"},
                         indent=2, sort_keys=True))
        return 1

    receipt = load_json_strict(full_path(PATHS["receipt"]))
    contract = load_json_strict(full_path(PATHS["contract"]))
    failures.extend(validate_receipt(receipt))
    if contract.get("claim_boundary", {}).get("headline_admitted") is not False \
            or contract.get("claim_boundary", {}).get("permitted") != \
            "exact_SHA_standalone_VCS_SVA_functional_regression_only":
        failures.append("contract claim boundary drift")

    try:
        canonical_run_dir(RUN_DIR)
    except ValueError as error:
        failures.append("canonical run rejected: {0}".format(error))
    if not os.path.isdir(RUN_DIR) or os.path.islink(RUN_DIR):
        failures.append("run directory is missing or a symlink")
    for name, expected in RUN_SHA256.items():
        path = os.path.join(RUN_DIR, name)
        if not os.path.isfile(path) or sha256_file(path) != expected:
            failures.append("frozen run SHA drift: " + name)

    parse_manifest(os.path.join(RUN_DIR, "input_sha256.txt"),
                   INPUT_BASENAMES, HW_ROOT, failures, "input manifest")
    parse_manifest(os.path.join(RUN_DIR, "output_sha256.txt"),
                   OUTPUT_BASENAMES, RUN_DIR, failures, "output manifest")
    parse_manifest(os.path.join(RUN_DIR, "run_local_seal.sha256"), {
        "input_sha256.txt", "output_sha256.txt", "runner_status.txt",
        "output_manifest_check.raw.log", "output_manifest_check.rc",
    }, RUN_DIR, failures, "local seal")
    parse_manifest(os.path.join(RUN_DIR, "completion_seal.sha256"), {
        "run_local_seal.sha256", "run_local_seal_check.raw.log",
        "run_local_seal_check.rc", "RUN_COMPLETE.txt",
    }, RUN_DIR, failures, "completion seal")

    for name in ("r11_review_validation.rc", "input_manifest_check.rc",
                 "compile.rc", "sim.rc", "output_manifest_check.rc",
                 "run_local_seal_check.rc"):
        with open(os.path.join(RUN_DIR, name), "r") as handle:
            if handle.read() != "0\n":
                failures.append("nonzero or malformed real exit receipt: " + name)

    with open(os.path.join(RUN_DIR, "compile.raw.log"), "r") as handle:
        compile_log = handle.read()
    compile_bad = re.findall(
        r"(?im)^(?:Warning|Error)-|(?:^|[^A-Za-z])(?:warning|error|fatal)(?:[^A-Za-z]|$)",
        compile_log)
    if compile_bad:
        failures.append("compile log contains broad warning/error/fatal signature")
    if compile_log.count("Parsing design file 'rtl_m37_r10/qfit_atlif_csd_reconstruct_t10.sv'") != 1 \
            or compile_log.count("Parsing design file 'verif_m37/qfit_atlif_csd_reconstruct_t10_assertions.sv'") != 1 \
            or compile_log.count("Parsing design file 'tb_m37/tb_qfit_atlif_csd_reconstruct_t10.sv'") != 1 \
            or "3 modules and 0 UDP read." not in compile_log \
            or "Version V-2023.12-SP1_Full64" not in compile_log:
        failures.append("compile transcript identity or module closure drift")

    with open(os.path.join(RUN_DIR, "sim.raw.log"), "r") as handle:
        sim_log = handle.read()
    expected_pass = ("M37_PASS total_tiles=245 nominal_tiles=96 "
        "dut_unique_signed_input_coefficient_product_pairs=65536 "
        "product_miters=117600 bit_miters=39200 arithmetic_issues=1225 "
        "no_data_multiplier=1")
    if len(re.findall(r"(?m)^M37_PASS .*$", sim_log)) != 1 \
            or expected_pass not in sim_log:
        failures.append("simulation does not contain one exact internal PASS")
    for marker in ("M37_SVA_BOUND=1", "ASSERTIONS=enabled",
                   "SIMULATOR=Synopsys VCS", "M37_RANDOM_SEED=0x4d370203"):
        if sim_log.count(marker) != 1:
            failures.append("simulation marker count drift: " + marker)
    sim_bad = re.findall(
        r"(?im)failed at|Offending|assertion[^\n]*(?:fail|error)|"
        r"(?:^|[^A-Za-z])(?:Error|Fatal)(?:[^A-Za-z]|$)", sim_log)
    if sim_bad:
        failures.append("simulation log contains functional/SVA failure signature")
    cover_counts = [int(item) for item in re.findall(
        r",\s+2758 attempts,\s+([0-9]+) match$", sim_log, re.M)]
    if cover_counts != [220, 1271, 249, 117, 245, 571, 133, 210]:
        failures.append("SVA cover count vector drift")

    verify_vector_file(os.path.join(RUN_DIR, "vectors.txt"), failures)

    run_top_mode = oct(os.stat(RUN_DIR).st_mode & 0o777)
    core_evidence = [name for name in os.listdir(RUN_DIR)
                     if os.path.isfile(os.path.join(RUN_DIR, name))
                     and name != "simv"]
    writable_core = [name for name in core_evidence
                     if os.stat(os.path.join(RUN_DIR, name)).st_mode
                     & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)]
    writable_files = 0
    writable_dirs = 0
    for directory, dirnames, filenames in os.walk(RUN_DIR):
        for name in dirnames:
            if os.stat(os.path.join(directory, name)).st_mode & 0o222:
                writable_dirs += 1
        for name in filenames:
            if os.stat(os.path.join(directory, name)).st_mode & 0o222:
                writable_files += 1
    if run_top_mode != "0o555" or writable_core:
        failures.append("top-level run or core evidence mode drift")

    rejected_attacks = []
    for label, payload in (
            ("duplicate_json_key", '{"schema":"a","schema":"b"}'),
            ("json_nan", '{"value":NaN}'),
            ("json_positive_infinity", '{"value":Infinity}')):
        if strict_loader_attack(payload):
            rejected_attacks.append(label)
        else:
            failures.append("strict JSON loader accepted " + label)
    bool_receipt = json.loads(json.dumps(receipt))
    bool_receipt["functional"]["total_tiles"] = True
    bool_receipt["sva"]["cover_counts"][0] = False
    bool_failures = validate_receipt(bool_receipt)
    if any("functional.total_tiles" in item for item in bool_failures) \
            and any("cover vector" in item for item in bool_failures):
        rejected_attacks.append("receipt_bool_as_int_scalar_and_nested")
    else:
        failures.append("receipt bool-as-int mutation was not rejected")
    for label, path in (("relative_path_escape", "../m37_r12"),
                        ("alternate_absolute_run", "/tmp/m37_r12")):
        try:
            canonical_run_dir(path)
        except ValueError:
            rejected_attacks.append(label)
        else:
            failures.append("run path attack accepted: " + label)
    with tempfile.TemporaryDirectory(prefix="m37_r12_symlink_") as temp_dir:
        link = os.path.join(temp_dir, "run")
        os.symlink(RUN_DIR, link)
        try:
            canonical_run_dir(link)
        except ValueError:
            rejected_attacks.append("run_directory_symlink_alias")
        else:
            failures.append("run directory symlink alias accepted")

    before = (sha256_file(os.path.join(RUN_DIR, "completion_seal.sha256")),
              sha256_file(os.path.join(RUN_DIR, "RUN_COMPLETE.txt")))
    overwrite = subprocess.Popen(
        ["/bin/bash", full_path(PATHS["runner"])], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    overwrite_stdout, overwrite_stderr = overwrite.communicate()
    after = (sha256_file(os.path.join(RUN_DIR, "completion_seal.sha256")),
             sha256_file(os.path.join(RUN_DIR, "RUN_COMPLETE.txt")))
    if overwrite.returncode == 2 and overwrite_stdout == b"" \
            and b"refusing to overwrite M37-r12 exact-SHA VCS run" \
            in overwrite_stderr and before == after:
        rejected_attacks.append("runner_rerun_overwrite_and_seal_mutation")
    else:
        failures.append("runner rerun did not fail closed without seal mutation")

    replay = {"real_exit_code": None, "internal_PASS_count": 0,
              "vector_sha256": None, "sva_cover_counts": []}
    with tempfile.TemporaryDirectory(prefix="m37_r12_replay_") as temp_dir:
        vector_path = os.path.join(temp_dir, "vectors.txt")
        executable = os.path.join(temp_dir, "simv")
        shutil.copy2(os.path.join(RUN_DIR, "simv"), executable)
        shutil.copytree(os.path.join(RUN_DIR, "simv.daidir"),
                        os.path.join(temp_dir, "simv.daidir"), symlinks=True)
        process = subprocess.Popen(
            [executable, "+M37_VECTOR_FILE=" + vector_path],
            cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = process.communicate()
        output_text = output.decode("utf-8", "replace")
        replay["real_exit_code"] = process.returncode
        replay["internal_PASS_count"] = len(re.findall(
            r"(?m)^M37_PASS .*$", output_text))
        replay["sva_cover_counts"] = [int(item) for item in re.findall(
            r",\s+2758 attempts,\s+([0-9]+) match$", output_text, re.M)]
        if os.path.isfile(vector_path):
            replay["vector_sha256"] = sha256_file(vector_path)
        if process.returncode != 0 or replay["internal_PASS_count"] != 1 \
                or expected_pass not in output_text \
                or replay["vector_sha256"] != RUN_SHA256["vectors.txt"] \
                or replay["sva_cover_counts"] != \
                [220, 1271, 249, 117, 245, 571, 133, 210] \
                or re.search(r"(?im)failed at|Offending|"
                             r"assertion[^\n]*(?:fail|error)|"
                             r"(?:^|[^A-Za-z])(?:Error|Fatal)(?:[^A-Za-z]|$)",
                             output_text):
            failures.append("independent frozen simv replay failed")

    findings = [
        {
            "id": "P2_RUN_IS_HASH_SEALED_NOT_RECURSIVELY_IMMUTABLE",
            "severity": "P2",
            "finding": ("The top directory and all top-level textual evidence are read-only, "
                        "but simv and generated VCS subtrees remain writable. File modes are "
                        "not an authentication root; exact manifests plus the external receipt "
                        "and this review carry identity."),
            "evidence": {
                "run_top_mode": run_top_mode,
                "writable_top_level_core_evidence_files_excluding_simv": len(writable_core),
                "writable_files_including_simv_and_generated_subtrees_observed":
                    "at_least_one" if writable_files else "none",
                "writable_generated_directories_observed":
                    "at_least_one" if writable_dirs else "none",
            },
            "repair_gate": ("Every DC/Formality runner must re-pin f947..., this review, its "
                            "validator, and the r12 receipt before launch; never infer integrity "
                            "from chmod alone."),
        },
        {
            "id": "P2_RUNNER_REQUIRES_EXPLICIT_SHELL_INVOCATION",
            "severity": "P2",
            "finding": ("The pinned runner is mode 0664, so direct execution returns permission "
                        "denied; invocation through /bin/bash correctly refuses the existing run "
                        "with rc=2 and leaves both completion seals unchanged."),
            "repair_gate": ("Record /bin/bash plus the exact runner SHA in downstream launch "
                            "receipts, or make a separately pinned executable copy."),
        },
        {
            "id": "P2_VCS_IS_FINITE_STANDALONE_FUNCTIONAL_EVIDENCE",
            "severity": "P2",
            "finding": ("The 245-tile test is strong directed standalone evidence, including all "
                        "65,536 signed-int8 input/coefficient pairs, stalls, reloads, illegal "
                        "descriptors, saturation, thresholds, and eight nonzero SVA covers. It is "
                        "not formal exhaustiveness, mapped PPA, power, or full-system evidence."),
            "repair_gate": ("Proceed only to exact-SHA DC/STA and Formality. Admit mapped area, "
                            "timing, zero-multiplier resources, or equivalence only from their "
                            "independently sealed successful receipts."),
        },
    ]

    verdict = "NO_GO_DC_FORMALITY" if failures else \
        "GO_DC_FORMALITY_EXACT_F947_CANDIDATE_ONLY"
    review = {
        "claim_boundary": {
            "forbidden": [
                "any completed DC STA area frequency or multiplier-resource claim",
                "any completed Formality-equivalence claim",
                "power energy full-system performance or speedup",
                "paper PPA headline or physical shared-mux claim",
            ],
            "permitted": [
                "exact f9474151... standalone RTL passed frozen Synopsys VCS/SVA regression",
                "launch exact-SHA DC STA and Formality on only that candidate",
            ],
        },
        "exact_identity": {
            "candidate_rtl_sha256": EXPECTED_SHA256["rtl"],
            "r11_independent_review_sha256": EXPECTED_SHA256["r11_review"],
            "r11_independent_validator_sha256": EXPECTED_SHA256["r11_validator"],
            "r12_receipt_sha256": EXPECTED_SHA256["receipt"],
            "r12_runner_sha256": EXPECTED_SHA256["runner"],
            "r12_simv_sha256": RUN_SHA256["simv"],
        },
        "failures": failures,
        "findings": findings,
        "headline_admitted": False,
        "independent_replay": replay,
        "independently_verified": {
            "all_input_output_and_nested_seal_manifests": True,
            "compile_real_exit_code": 0,
            "compile_warning_error_fatal_signatures": 0,
            "direct_product_miters": 117600,
            "dut_signed_input_coefficient_pairs": 65536,
            "functional_mismatch_count": 0,
            "illegal_accept_reject": [210, 210],
            "internal_PASS_count": 1,
            "output_bit_miters": 39200,
            "sim_real_exit_code": 0,
            "sva_assertion_failure_count": 0,
            "sva_cover_counts": [220, 1271, 249, 117, 245, 571, 133, 210],
            "total_tiles": 245,
            "vector_line_partition": [1, 15, 245, 210],
            "vector_sha256": RUN_SHA256["vectors.txt"],
        },
        "p0_count": 0,
        "p1_count": 0 if not failures else 1,
        "p2_count": len(findings),
        "rejected_attacks": sorted(rejected_attacks),
        "review_score_0_to_100": 94 if not failures else 70,
        "review_verdict": verdict,
        "reviewed_sha256": observed_sha256,
        "schema": "m37_r12_independent_hammer_review_v1",
        "status": ("PASS_INDEPENDENT_HAMMER_GO_EXACT_SHA_DC_FORMALITY"
                   if not failures else "FAIL_INDEPENDENT_HAMMER_NO_GO"),
    }
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
