#!/usr/bin/env python3
"""Fail-closed static/runtime checker for additive M1798; never launches EDA."""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
BASE_CHECKER = HW / "system_simulator/scripts/check_m1790_c3_m1454_fixed_t10_mapped_energy_source.py"
BASE_SPEC = importlib.util.spec_from_file_location("m1790_checker_for_m1798",
                                                   str(BASE_CHECKER))
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("M1790 predecessor checker unavailable")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

M1790_CONTRACT = HW / "contracts/m1790_c3_m1454_fixed_t10_mapped_energy_source_contract_r1_20260902.json"
M1790_RUNNER = HW / "dc_handoff/scripts/run_m1790_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
M1790_PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1790_c3_m1454_fixed_t10_mapped_energy.tcl"
M1791 = HW / "reviews/m1791_m1790_c3_m1454_fixed_t10_mapped_energy_source_hammer_r1_20260902"
DOC359 = BASE.DOC359
NET = BASE.NET
SDC = BASE.SDC
CELL_V = BASE.CELL_V
TT_DB = BASE.TT_DB

TB_BASE = BASE.TB
TB_TAG = HW / "dc_handoff/tb/tb_m1798_c3_m1454_fixed_t10_mapped_energy_tag_scoreboard.sv"
FILELIST = HW / "dc_handoff/filelists/iscas_m1798_c3_m1454_fixed_t10_mapped_energy.f"
UCLI = HW / "dc_handoff/scripts/m1798_c3_m1454_fixed_t10_mapped_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1798_c3_m1454_fixed_t10_mapped_energy.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m1798_c3_m1454_fixed_t10_mapped_energy_one_shot.py"
CHECKER = Path(__file__).resolve()
TEST = HW / "system_simulator/tests/test_m1798_c3_m1454_fixed_t10_mapped_energy_source.py"
CONTRACT = HW / "contracts/m1798_m1791_c3_m1454_fixed_t10_mapped_energy_source_contract_r1_20260902.json"

TOP = BASE.TOP
SAIF_SCOPE = BASE.SAIF_SCOPE
CLAIMS = dict(BASE.CLAIMS)

FIXED = {
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    NET: "7c01af42322b8feed904df2862aac6e21cbe165b988f1b248f2e94d23f23a7a7",
    SDC: "bb3697e833cb987e4a85ab2a62b4f40946a8c3d6b7eaba08504570f5a862f23f",
    CELL_V: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
    BASE_CHECKER: "921dedb65d56ba7a22717cd4fb4a8ae371843e4e8422391d293ddf34373191c4",
    TB_BASE: "4b4d2c4fb0f96a2644c3f5f1d03afc1efa0c4a3e8262157970d76c8f61d2c142",
    M1790_RUNNER: "809d72f14513fb1db36934cbf4566aaa439becfed4dc2768869eb6a538572c3c",
    M1790_PT_TCL: "6ab4e05a3b00cdd8ac7ba34b168b602afe9f830f154706a70b62cee2b2d1cb23",
    M1790_CONTRACT: "1fd4c303347d6141dff9d68f18ef625d71623230ec88e173455b58b2483f7c14",
    Path(str(M1790_CONTRACT) + ".sha256"): "84759b88e084587287f7016c0b98095e05c88d7142ba2b0c06f17dc5b3bffe8c",
    Path(str(M1790_CONTRACT) + ".sha256.seal.sha256"): "d33ebf2222caadb08007828aa1f409352021410927b75c88b38c3dc26131a878",
    M1791 / "review.json": "a4c0a3c517b24b06ac9227aa4ebd5346daa4d8e8ae86a95213017431f06675cb",
    M1791 / "SHA256SUMS": "aa805d815b56405967c62f85bb105cfb955c02ee2c1dc82f08c182dbbb71718b",
    M1791 / "SHA256SUMS.seal.sha256": "ebae12121e98e832ccdd351a460ba36895354c44113412bfaf21d9daa9f08d23",
}

RELEASE_BINDING_TOKENS = (
    "M1798_EXPECTED_RUNNER_SHA256",
    "M1798_EXPECTED_SOURCE_CONTRACT_SHA256",
    "M1798_EXPECTED_M1799_MANIFEST_SHA256",
    "M1798_EXPECTED_M1799_OUTER_FILE_SHA256",
    "M1798_EXPECTED_M1799_REVIEW_SHA256",
    "M1798_EXPECTED_M1800_RELEASE_SHA256",
    "M1798_EXPECTED_M1800_SIDECAR_SHA256",
    "M1798_EXPECTED_M1800_OUTER_FILE_SHA256",
    "m1800_m1799_m1798_c3_m1454_fixed_t10_mapped_energy_launch_release_r1_v1",
    "AUTHORIZE_ONE_FRESH_M1798_C3_MAPPED_ENERGY_CAMPAIGN",
    "PASS_M1799_M1798_C3_MAPPED_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FRESH_M1798_CAMPAIGN",
    "release.get(\"identity\") != expected_identity",
    "release.get(\"prelaunch_claim_boundary\") != CHECK.CLAIMS",
    "release.get(\"measurement_boundary\") != RELEASE_BOUNDARY",
    "release.get(\"attempt_uniqueness\") != ATTEMPT_UNIQUENESS",
    "verify_review_member(M1799)",
    "verify_file_double_seal(M1800",
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal content")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        need((root / rel).is_file() and not (root / rel).is_symlink()
             and sha(root / rel) == fields[0], "manifest drift " + name)
        listed.add(name)


def active_lines(text):
    return [raw.split("#", 1)[0].strip() for raw in text.splitlines()
            if raw.split("#", 1)[0].strip()]


def strip_sv_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def validate_semantics(texts):
    tag = texts[TB_TAG]
    active_tag = strip_sv_comments(tag).lower()
    need("force " not in active_tag and "$root" not in active_tag
         and "dut." not in active_tag, "tag monitor hierarchy/state bypass")
    for forbidden in ("+notimingcheck", "+no_notifier", "+nospecify",
                      "+initreg", "deposit(", "vpi_handle_by_name"):
        need(forbidden not in (tag + texts[RUNNER]).lower(),
             "forbidden gate bypass " + forbidden)
    for token in (
            "EXPECTED_TOTAL_TAGS = 9", "EXPECTED_MEASURED_TAGS = 8",
            "expected_tile_done_tag [0:15]",
            "sampled_raw_tag !== directed_tag(expected_write)",
            "sampled_tile_done_tag !==",
            "expected_tile_done_tag[expected_read]",
            "expected_read >= expected_write",
            "expected_write != EXPECTED_TOTAL_TAGS",
            "raw_stall_cycles == 0", "result_stall_cycles == 0",
            "M1798_TILE_DONE_TAG_CHECK total=%0d warmup=1 measured=%0d",
            "PASS_M1798_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD",
            "bind tb_m1790_c3_m1454_fixed_t10_mapped_energy"):
        need(token in tag, "tag scoreboard omits " + token)

    expected_filelist = [str(CELL_V), str(NET), str(TB_BASE), str(TB_TAG)]
    need(active_lines(texts[FILELIST]) == expected_filelist,
         "filelist/order drift")
    need(active_lines(texts[UCLI]) == [
        "power -gate_level all mda sv", "power " + SAIF_SCOPE, "run",
        "power -enable", "run", "power -disable",
        "power -report $::env(M1798_SAIF_FILE) 1e-9 " + SAIF_SCOPE,
        "quit"], "UCLI scope/order drift")

    pt = texts[PT_TCL]
    for token in ("M1798_TT_LIB_DB", "M1798_MAPPED_NETLIST",
                  "M1798_MAPPED_SDC", "M1798_GATE_SAIF",
                  "M1798_OUTPUT_DIR", "M1798_SAIF_INSTANCE",
                  "M1798_MEASUREMENT_CYCLES", "M1798_SAIF_DURATION_NS",
                  str(M1790_PT_TCL)):
        need(token in pt, "PTPX wrapper omits " + token)

    runner = texts[RUNNER]
    for token in (
            "results/.m1798_c3_mapped_energy_attempt_consumed",
            "date_dual_synopsys_same_uid_eda_queue.lock", "collision_gate()",
            "automatic_retry\": False", "reuse_prior_simv_saif_ptpx\": False",
            "+define+UNIT_DELAY", "CHECK.validate_runtime(sim_log)",
            "CHECK.validate_saif(saif", "CHECK.component_power(",
            "vcs_compiles\": 1", "simv_runs\": 1", "saif_files\": 1",
            "ptpx_runs\": 1", "publish_no_replace(STAGE, RESULT)"):
        need(token in runner, "runner omits " + token)
    for token in RELEASE_BINDING_TOKENS:
        need(token in runner, "release binding omits " + token)
    need(runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1
         and runner.count("state[\"saif_files\"] += 1") == 1
         and runner.count("state[\"ptpx_runs\"] += 1") == 1,
         "runner execution budget drift")


def validate_sources():
    BASE.validate_sources()
    for path, digest in FIXED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift " + str(path))
    verify_seal(M1791)
    predecessor = strict_json(M1791 / "review.json")
    need(predecessor.get("status") ==
         "FAIL_CLOSED_M1791_M1790_C3_MAPPED_ENERGY_SOURCE_HAMMER__P1_2__NO_EDA"
         and predecessor.get("severity_counts") == {"p0": 0, "p1": 2, "p2": 0},
         "M1791 predecessor finding drift")

    source_paths = (TB_TAG, FILELIST, UCLI, PT_TCL, RUNNER, CHECKER, TEST)
    for path in source_paths:
        need(path.is_file() and not path.is_symlink(), "source absent " + str(path))
    texts = dict((path, path.read_text()) for path in source_paths)
    validate_semantics(texts)
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m1798_m1791_c3_m1454_fixed_t10_mapped_energy_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M1791_P1_FIXED__M1799_REVIEW_AND_M1800_RELEASE_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("claim_boundary") == CLAIMS, "source claim promotion")
    need(contract.get("execution_budget") == dict(
        vcs_compiles=1, simv_runs=1, saif_files=1, ptpx_runs=1,
        automatic_retry=False, reuse_prior_simv_saif_ptpx=False),
        "contract budget")
    need(contract.get("launch_governance", {}).get("exact_release_required") == "M1800"
         and contract.get("launch_governance", {}).get(
             "different_author_source_hammer_required") == "M1799"
         and contract.get("launch_governance", {}).get("double_seal") is True,
         "contract governance")
    mapping = dict((row.get("path"), row.get("sha256"))
                   for row in contract.get("source_files", []))
    need(len(mapping) == len(source_paths), "source inventory cardinality")
    for path in source_paths:
        need(mapping.get(str(path.relative_to(HW))) == sha(path),
             "source inventory drift " + str(path))
    return {"status": "PASS_M1798_SOURCE_STATIC", "source_files": len(source_paths),
            "predecessor_source_static": "PASS", "checks": 1}


def validate_runtime(path):
    result = dict(BASE.validate_runtime(path))
    text = Path(path).read_text(errors="strict")
    need(text.count("PASS_M1798_C3_ORDERED_TILE_DONE_TAG_SCOREBOARD") == 1,
         "M1798 tag PASS count")
    tag = re.findall(
        r"M1798_TILE_DONE_TAG_CHECK total=([0-9]+) warmup=([0-9]+) measured=([0-9]+) mismatches=([0-9]+) raw_stall=([0-9]+) result_stall=([0-9]+)",
        text)
    need(len(tag) == 1 and tag[0][0:4] == ("9", "1", "8", "0")
         and int(tag[0][4]) > 0 and int(tag[0][5]) > 0,
         "M1798 ordered tile-done tag checker")
    result["status"] = "PASS_M1798_PUBLIC_RUNTIME_WITH_ORDERED_TILE_DONE_TAG"
    result["tile_done_tags_checked"] = 9
    result["measured_tile_done_tags_checked"] = 8
    result["tile_done_tag_mismatches"] = 0
    return result


def validate_saif(path, cycles):
    result = dict(BASE.validate_saif(path, cycles))
    result["status"] = "PASS_M1798_DUT_ONLY_SAIF"
    return result


def component_power(path, cycles):
    result = dict(BASE.component_power(path, cycles))
    result["status"] = "PASS_M1798_COMPONENT_METRIC_PENDING_RESULT_HAMMER"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    need(args.static, "only --static is allowed")
    print(json.dumps(validate_sources(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
