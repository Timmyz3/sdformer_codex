#!/usr/bin/env python3
"""Fail-closed static checker for the M1856 diagnostic-only source package."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import sys


HW = Path(__file__).resolve().parents[2]
PATHS = {
    "runner": HW / "dc_handoff/scripts/run_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_one_shot.py",
    "tb": HW / "dc_handoff/tb/tb_m1856_c2_k8_case0_mapped_fault_xz_diagnostic.sv",
    "filelist": HW / "dc_handoff/filelists/date_m1856_c2_k8_case0_mapped_fault_xz_diagnostic.f",
    "checker": Path(__file__).resolve(),
    "test": HW / "system_simulator/tests/test_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source.py",
    "contract": HW / "contracts/m1856_m1854_m1845_c2_k8_case0_mapped_fault_xz_diagnostic_source_contract_r1_20260902.json",
}
CONTRACT = PATHS["contract"]
MAPPED = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902/k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v"
MAPPED_SHA256 = "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792"
FILELIST = PATHS["filelist"]
FILELIST_SHA256 = "3d879d295c5a45e763001c0403f43a16091f2a2172b2dcaf2b2987f098262afe"
TOP = "tb_m1856_c2_k8_case0_mapped_fault_xz_diagnostic"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1854_REVIEW_SHA256 = "9ef9f00091b145e438a03c9039123af198f28ef16f6d3180169361ca6470d0a6"
M1854_MANIFEST_SHA256 = "49176d9165cbfe449f243fe4f76b2e2ae3af1e388398d8556f843e75dbfd10b8"
M1854_OUTER_SHA256 = "28089b517fbe5a7f052dfef98b031cfc9887f8f288d610f917cd3234acc2c1f4"

CLAIMS = {
    "diagnostic_source_only": True,
    "m1845_retry": False,
    "mapped_functionality": False,
    "production_functionality": False,
    "power": False,
    "energy": False,
    "performance": False,
    "speedup": False,
    "system_speedup": False,
    "paper_citable": False,
    "headline": False,
}


class CheckFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise CheckFailure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           CheckFailure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise CheckFailure("JSON root")
    return value


def source_map():
    values = {}
    for name, path in PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise CheckFailure("missing source " + str(path))
        values[name] = path.read_text()
    return values


def require(text, token, where):
    if token not in text:
        raise CheckFailure(where + " missing token " + token)


def reject(text, token, where):
    if token in text:
        raise CheckFailure(where + " forbidden token " + token)


def check_runner(text):
    required = (
        "M1856_EXPECTED_M1857_REVIEW_SHA256",
        "M1856_EXPECTED_M1858_RELEASE_SHA256",
        '"vcs_compiles": 1, "simv_runs": 1',
        '"ucli_runs": 0, "saif_files": 0, "ptpx_runs": 0',
        '"all_other_eda_runs": 0, "automatic_retry": False',
        'run(["./simv", "-lca", "+M979_CASE=0"]',
        "validate_diagnostic_log(WORK / \"diagnostic.log\")",
        "M1856_DIAGNOSTIC_ATTEMPT_CONSUMED",
        "M1856_DIAGNOSTIC_LOCALIZATION_COMPLETE_DO_NOT_CITE_AS_PRODUCTION",
        "M1856_DIAGNOSTIC_FAILED_OR_INCOMPLETE_DO_NOT_RETRY",
        "publish_no_replace(STAGE, RESULT)",
    )
    for token in required:
        require(text, token, "runner")
    for token in (
            "+M979_UCLI_SAIF", '"-ucli"', "/opt/synopsys/prime", "ptpx_runs\"] = 1",
            "saif_files\"] = 1", "automatic_retry\": True", "lmutil",
            "reuse_prior_simv", "M1845_EXPECTED_RUNNER_SHA256"):
        reject(text, token, "runner")
    if text.count("run(compile_command(), WORK") != 1:
        raise CheckFailure("runner compile call cardinality")
    if text.count('run(["./simv", "-lca", "+M979_CASE=0"]') != 1:
        raise CheckFailure("runner sim call cardinality")
    if text.index("verify_authority()") > text.index("ATTEMPT.mkdir()"):
        raise CheckFailure("authority must precede attempt")
    if text.index("ATTEMPT.mkdir()") > text.index("run(compile_command(), WORK"):
        raise CheckFailure("attempt must precede compile")


def check_tb(text):
    required = (
        "module " + TOP,
        "tb_m979_c2_three_axis_mapped_gate_case_saif core();",
        "core.g_memory[bank].memory.endpoint_protocol_fault_q",
        "core.dut.implementation.g_k8_implementation_memory_adapter_stale_q",
        "core.dut.implementation.g_k8_implementation_memory_adapter_fault_q",
        "core.dut.implementation.g_k8_implementation_core_g_k8_service_fault_q",
        "core.dut.implementation.g_k8_implementation_core_adapter_fault_q",
        "(value === 1'b0) || (value === 1'b1)",
        'print_and_localize("posedge")',
        'print_and_localize("negedge")',
        "#1ps;",
        "M1856_BIT name=protocol_error",
        "M1856_BIT name=numeric_overflow",
        "M1856_BIT name=stale_response_seen",
        "M1856_BIT name=endpoint_fault[%0d]",
        "M1856_FIRST_NONBINARY",
        "$finish;",
    )
    for token in required:
        require(text, token, "tb")
    for token in ("M979_UCLI_SAIF", "$display(\"PASS", "$assertoff",
                  "force ", "release ", "SVA_RUNTIME_ENABLED"):
        reject(text, token, "tb")
    if text.count("M1856_FIRST_NONBINARY") != 4:
        raise CheckFailure("TB first-nonbinary class cardinality")
    if len(re.findall(r"if \(!is_binary\(endpoint_fault\[bank\]\)\)", text)) != 1:
        raise CheckFailure("TB endpoint localization check")
    if re.search(r"if\s*\(\s*!is_binary\(mapped_", text):
        raise CheckFailure("internal XMR must not decide localization")


def check_filelist(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 6:
        raise CheckFailure("filelist row count")
    if rows[0] != "+define+M1831_AXIS_K8":
        raise CheckFailure("filelist K8 identity")
    if any("M1831_AXIS_K1X8" in row or "SVA_RUNTIME_ENABLED" in row for row in rows):
        raise CheckFailure("filelist forbidden define")
    expected_suffixes = (
        "tcbn28hpcplusbwp35p140.v",
        "m1334_c2_production_activity_reset_safe_memory_model.sv",
        "tb_m1831_c2_fresh_mapped_gate_case_core.sv",
        "tb_m1856_c2_k8_case0_mapped_fault_xz_diagnostic.sv",
    )
    for suffix in expected_suffixes:
        if sum(row.endswith(suffix) for row in rows) != 1:
            raise CheckFailure("filelist exact member " + suffix)
    for token in ("m1831_c2_registered_public_fault_production_assertions.sv",
                  "tb_m1831_c2_fresh_mapped_production_energy.sv", ".ucli"):
        reject(text, token, "filelist")


def check_contract(text, texts):
    value = strict_json_text(text)
    if (value.get("schema") !=
            "m1856_m1854_m1845_c2_k8_case0_mapped_fault_xz_diagnostic_source_contract_r1_v1"
            or value.get("status") !=
            "SOURCE_ONLY_M1856_C2_K8_CASE0_MAPPED_FAULT_XZ_DIAGNOSTIC__NO_EDA_NO_LICENSE_NO_ATTEMPT"):
        raise CheckFailure("contract identity/status")
    if value.get("authorization_now") != {
            "license_queries": 0, "attempts_created": 0,
            "vcs_compiles": 0, "simv_runs": 0, "ucli_runs": 0,
            "saif_files": 0, "ptpx_runs": 0, "all_other_eda_runs": 0,
            "results_created": 0, "releases_created": 0}:
        raise CheckFailure("contract authoring authorization")
    if value.get("future_execution_budget") != {
            "vcs_compiles_exact": 1, "simv_runs_exact": 1,
            "case": 0, "axis": "K8", "ucli_runs": 0,
            "saif_files": 0, "ptpx_runs": 0, "all_other_eda_runs": 0,
            "automatic_retry": False, "reuse_m1845_simv": False}:
        raise CheckFailure("contract future budget")
    if value.get("claim_boundary") != CLAIMS:
        raise CheckFailure("contract claim boundary")
    upstream = value.get("upstream_failure_authority", {})
    expected_upstream = {
        "m1854_review_sha256": M1854_REVIEW_SHA256,
        "m1854_manifest_sha256": M1854_MANIFEST_SHA256,
        "m1854_outer_seal_file_sha256": M1854_OUTER_SHA256,
        "m1845_attempt_consumed": True,
        "m1845_automatic_retry": False,
        "direct_unknown_boundary": "{protocol_error,numeric_overflow,stale_response_seen,endpoint_fault[7:0]}",
    }
    if upstream != expected_upstream:
        raise CheckFailure("contract M1854/M1845 boundary")
    exact_identity = value.get("exact_diagnostic_identity", {})
    if (exact_identity.get("mapped_netlist_sha256") != MAPPED_SHA256
            or exact_identity.get("filelist_sha256") != FILELIST_SHA256
            or exact_identity.get("top") != TOP
            or exact_identity.get("m979_case") != 0
            or exact_identity.get("axis") != "K8"):
        raise CheckFailure("contract exact diagnostic identity")
    source_files = value.get("source_files")
    if type(source_files) is not dict or len(source_files) != 5:
        raise CheckFailure("contract source inventory")
    for name in ("runner", "tb", "filelist", "checker", "test"):
        rel = PATHS[name].relative_to(HW).as_posix()
        if source_files.get(rel) != hashlib.sha256(texts[name].encode()).hexdigest():
            raise CheckFailure("contract source hash " + name)
    if value.get("docs359_sha256") != DOCS359_SHA256:
        raise CheckFailure("docs359 identity")
    future = value.get("future_authority", {})
    if (future.get("source_review") !=
            "reviews/m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_source_hammer_r1_20260902"
            or future.get("launch_release") !=
            "contracts/m1858_m1857_m1856_c2_k8_case0_mapped_fault_xz_diagnostic_launch_release_r1_20260902.json"
            or future.get("source_review_and_release_present_now") is not False):
        raise CheckFailure("contract future authority")


def validate_diagnostic_log(path):
    text = Path(path).read_text(errors="strict")
    matches = re.findall(
        r"M1856_FIRST_NONBINARY time_ps=(\d+) edge=(posedge|negedge) "
        r"name=(protocol_error|numeric_overflow|stale_response_seen|endpoint_fault\[[0-7]\]) value=([xz])",
        text, flags=re.IGNORECASE)
    if len(matches) != 1:
        raise CheckFailure("diagnostic first-nonbinary token cardinality")
    if "+M979_UCLI_SAIF" in text or "M1856_DIAGNOSTIC_PASS" in text:
        raise CheckFailure("diagnostic log claim contamination")
    if text.count("M1856_SAMPLE") < 1 or text.count("M1856_AUX") < 1:
        raise CheckFailure("diagnostic samples absent")
    return {"time_ps": int(matches[0][0]), "edge": matches[0][1],
            "name": matches[0][2], "value": matches[0][3].lower(),
            "diagnostic_only": True}


def check(overrides=None):
    texts = source_map()
    if overrides:
        texts.update(overrides)
    check_runner(texts["runner"])
    check_tb(texts["tb"])
    check_filelist(texts["filelist"])
    check_contract(texts["contract"], texts)
    if sha(MAPPED) != MAPPED_SHA256:
        raise CheckFailure("mapped netlist drift")
    return {"status": "PASS_M1856_DIAGNOSTIC_SOURCE_STATIC",
            "eda_or_license_run": False, "launch_authorized": False,
            "future_vcs_compiles": 1, "future_simv_runs": 1,
            "future_ucli_saif_ptpx": 0, "paper_claim": False}


def validate_sources():
    return check()


if __name__ == "__main__":
    try:
        print(json.dumps(check(), sort_keys=True))
    except Exception as error:
        print("FAIL_M1856_DIAGNOSTIC_SOURCE_STATIC: " + str(error), file=sys.stderr)
        raise
