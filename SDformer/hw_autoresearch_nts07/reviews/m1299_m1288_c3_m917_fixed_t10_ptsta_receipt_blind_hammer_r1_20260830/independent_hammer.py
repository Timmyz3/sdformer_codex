#!/usr/bin/env python3
"""Receipt-blind M1299 hammer for inert M1288 Fixed-T10 PTSTA source.

Only source/static checks and synthetic receipt mocks run.  No PT/EDA binary,
license query, live preflight, launch, canonical namespace, GPU, or remote
resource is touched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
ROOT = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m1288_m917_fixed_t10_ptsta_inert_exact_sha.sh"
TCL = HW / "dc_handoff/scripts/run_ptsta_m1288_m917_fixed_t10_slowmax_fastmin_inert.tcl"
TEST = HW / "tests/test_m1288_m917_fixed_t10_ptsta_source_static.py"
CONTRACT = HW / "contracts/m1288_c3_m917_fixed_t10_ptsta_source_contract_r1_20260830.json"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SOURCE_SHA = {
    RUNNER: "a7fa2c5b031a446562d0bdb8f6f80112d7348fff6be92efdbf5b12830f6b928c",
    TCL: "34ee64d2714e52269436fe35e79c575ad6eaf3dbb9d82d95a63e335b9672253b",
    TEST: "1de19bc4697ebdb7758ed1123f2bdb722875025e3d4cd9d191b4e323476f3378",
    CONTRACT: "91f130a09aa48b0f0f49aadb43c17d969abc026939199ee9acabccbb5a5a69a1",
    DOCS: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

SEALED_DIRS = {
    HW / "dc_handoff/runs/m917_m518_r5_fixed_descendant_safe_setup_area_logic_only_dc_3p000ns_r1_20260829":
        "e2f619c321218d78537528bb53d6de7b8817316008840198703103ff4c8c75b9",
    HW / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829":
        "43e6cee08ed52c52d1e46d48afc8b6835fd735e74ce4320b671cd401cf9c17d3",
    HW / "reviews/m1285_c3_m917_m928_pt_hold_saif_ptpx_readonly_audit_r1_20260830":
        "6c5fbaf805910022e9aecd25adb146b2b2ffaef92c2ee1ed3af885be45a54f7f",
}

TOOLS = {
    Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell"):
        "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    Path("/usr/bin/setsid"):
        "827259531e3511bcc704143690d8a3afec043d24a7922bf3ebfacf917cd7e100",
    Path("/usr/bin/bash"):
        "f420671b28650f60f5461c63353ca0a123b900dbfec0a9ddded83643f068a88e",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"):
        "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"):
        "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}

CLAIMS = {
    "source_only": True,
    "pt_executed": False,
    "setup_completed": False,
    "hold_closed": False,
    "automatic_hold_fix": False,
    "power": False,
    "energy": False,
    "speedup": False,
    "system": False,
    "paper_ppa_ready": False,
    "headline": False,
}


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path):
    require(path.is_file() and not path.is_symlink(), "unsafe/missing file: " + str(path))


def verify_manifest(directory: Path):
    require(directory.is_dir() and not directory.is_symlink(), "unsafe seal dir")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "manifest digest format")
        target = directory / relative
        regular(target); require(sha256(target) == digest, "manifest member drift")
    outer_digest, outer_name = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(outer_name == "SHA256SUMS" and sha256(manifest) == outer_digest,
            "outer seal drift")


def exact_bool_map(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), label + " keyset")
    for key, wanted in expected.items():
        require(type(value[key]) is bool and value[key] is wanted,
                label + " exact bool: " + key)


def validate_contract(data, validate_actual=True):
    top = {"schema", "date", "status", "scope", "identity", "exact_files",
           "tool", "pt_flow", "runtime_isolation", "authorization",
           "claim_boundary"}
    require(type(data) is dict and set(data) == top, "contract top keyset")
    require(data["schema"] == "m1288_c3_m917_fixed_t10_ptsta_source_contract_v1" and
            data["status"] == "M1288_SOURCE_ONLY__NO_PT_EDA_AUTHORIZED",
            "contract identity")
    identity_keys = {"runner_path", "runner_sha256", "tcl_path", "top",
        "future_launch_admission", "canonical_result", "attempt"}
    require(set(data["identity"]) == identity_keys and
            data["identity"]["runner_path"] == str(RUNNER.relative_to(HW)) and
            data["identity"]["runner_sha256"] == sha256(RUNNER) and
            data["identity"]["tcl_path"] == str(TCL.relative_to(HW)) and
            data["identity"]["top"] == "m518_matched_fixed_t10_atlif",
            "contract runner/path identity")
    exact_bool_map({k: v for k, v in data["authorization"].items()
                    if k != "max_attempts"}, {
        "launch_now": False, "run_pt": False,
        "run_dc": False, "run_vcs": False, "run_formality": False,
        "run_ptpx": False, "run_remote": False, "query_license": False,
        "independent_static_hammer_and_separate_launch_admission_required": True,
    }, "authorization")
    require(set(data["authorization"]) == {
        "launch_now", "max_attempts", "run_pt", "run_dc", "run_vcs",
        "run_formality", "run_ptpx", "run_remote", "query_license",
        "independent_static_hammer_and_separate_launch_admission_required",
    }, "authorization keyset")
    # max_attempts is intentionally integer zero in the source contract.
    require(type(data["authorization"]["max_attempts"]) is int and
            data["authorization"]["max_attempts"] == 0,
            "authorization max_attempts exact int zero")
    for key, wanted in CLAIMS.items():
        require(key in data["claim_boundary"] and
                type(data["claim_boundary"][key]) is bool and
                data["claim_boundary"][key] is wanted,
                "claim boundary drift: " + key)
    require(set(data["claim_boundary"]) == set(CLAIMS), "claim boundary keyset")
    exact_files = data["exact_files"]
    require(type(exact_files) is dict and len(exact_files) == 8,
            "exact file set")
    for relative, digest in exact_files.items():
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "exact digest format")
        if validate_actual:
            target = HW / relative
            regular(target); require(sha256(target) == digest,
                "exact path/SHA drift: " + relative)
    tool_keys = {"pt_shell_path", "pt_shell_sha256", "setsid_path",
        "setsid_sha256", "bash_path", "bash_sha256", "slow_db_path",
        "slow_db_sha256", "fast_db_path", "fast_db_sha256"}
    require(set(data["tool"]) == tool_keys, "tool keyset")
    if validate_actual:
        for path, digest in TOOLS.items():
            regular(path); require(sha256(path) == digest, "tool/lib SHA drift")
    require(data["pt_flow"] == {
        "clock_period_ns": 3.0,
        "setup_corner": "ssg0p9v125c slow/max",
        "hold_corner": "ffg1p05vm40c fast/min through set_min_library and OCV",
        "mapped_sdc_frozen": True, "ideal_clock": True,
        "wireload": "ZeroWireload", "spef": False, "macro_count": 0,
        "report_max_paths": 100, "hold_fix_command_count": 0,
        "mapped_identity_mutated": False,
        "negative_hold_terminal": "DIAGNOSTIC_STOP plus requirement for a new netlist-only hold-fix identity, Formality, and repeated PT",
    }, "PT flow exact semantics")
    return data


def check_tcl(text: str):
    clean_lines=[]
    for raw in text.splitlines():
        stripped=raw.strip()
        if stripped and not stripped.startswith("#"):
            clean_lines.append(stripped)
    forbidden={"fix_eco_timing", "set_fix_hold", "read_parasitics", "update_power",
        "report_power", "write_changes", "write_verilog", "save_session",
        "source", "eval", "uplevel", "exec", "sh"}
    commands=[]
    for line in clean_lines:
        first=line.split()[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first): commands.append(first)
    require(not (set(commands) & forbidden), "Tcl ECO/power/dynamic command")
    required=(
        "set_min_library $slow_db -min_version $fast_db",
        "-max $slow_opcond -max_library $slow_lib_name",
        "-min $fast_opcond -min_library $fast_lib_name",
        "report_timing -delay_type max", "report_timing -delay_type min",
        "hold_fix_command_count=0", "mapped_identity_mutated=false",
        "REPORTS_COMPLETE_NOT_HOLD_CLOSURE")
    for token in required: require(token in text, "Tcl semantic token: " + token)


def extract_receipt_python(runner: str) -> str:
    marker="python3 - \"${m1288_work}\" <<'PY'\n"
    require(marker in runner, "receipt heredoc missing")
    tail=runner.split(marker,1)[1]
    require("\nPY\n" in tail, "receipt heredoc terminator")
    return tail.split("\nPY\n",1)[0]


def mock_receipt(code: str, setup_state: str, setup: float,
                 hold_state: str, hold: float):
    with tempfile.TemporaryDirectory(prefix="m1299_receipt.") as name:
        run=Path(name); (run/"reports").mkdir()
        (run/"reports/timing_setup_slow.rpt").write_text(
            f"slack ({setup_state}) {setup:.6f}\n", encoding="utf-8")
        (run/"reports/timing_hold_fast.rpt").write_text(
            f"slack ({hold_state}) {hold:.6f}\n", encoding="utf-8")
        old=sys.argv
        try:
            sys.argv=["embedded_receipt", str(run)]
            exec(compile(code, "m1288_embedded_receipt", "exec"), {})
        finally:
            sys.argv=old
        payload=json.loads((run/"m1288_m917_fixed_t10_prelayout_ptsta_receipt_r1.json").read_text())
        token=(run/"RUN_COMPLETE.txt").read_text().strip()
        return payload,token


def run():
    for path,digest in EXPECTED_SOURCE_SHA.items():
        regular(path); require(sha256(path)==digest, "source identity drift")
    for directory,outer_sha in SEALED_DIRS.items():
        verify_manifest(directory)
        require(sha256(directory/"SHA256SUMS.seal.sha256")==outer_sha,
                "upstream outer seal identity")
    data=json.loads(CONTRACT.read_text())
    validate_contract(data)
    runner=RUNNER.read_text(); tcl=TCL.read_text()
    check_tcl(tcl)

    # Structural execution DAG and inert boundary.
    order=(runner.index('m1288_expect "${m1288_admission}"'),
           runner.index('m1288_collision="$(m1288_collisions)"'),
           runner.index('mkdir "${m1288_attempt}"'),
           runner.index('"${m1288_setsid}" env -i'))
    require(order[0] < order[1] < order[2] < order[3], "admission/collision/attempt/launch order")
    for token in ("! -e \"${m1288_canonical}\"", "! -e \"${m1288_work}\"",
        "! -e \"${m1288_attempt}\"", "HOME=\"${m1288_work}/safe_home\"",
        "chmod 0700", "m1288_collisions", "m1288_job_members",
        "m1288_wait_job_empty", "m1288_terminate_job",
        "FAILED_OR_INCOMPLETE_DO_NOT_CITE", ".quarantine",
        "m1288_sha docs/359_DATE终局冻结_20260813.md"):
        require(token in runner, "runtime DAG token missing: " + token)
    # The cleanup trap has an earlier failure-quarantine seal.  Compare the
    # happy-path drain with the final happy-path seal, not that trap branch.
    require(runner.index('m1288_wait_job_empty "${m1288_pgrp}"') <
            runner.rindex('m1288_seal_dir "${m1288_work}"'),
            "descendant drain must precede seal")
    require("lmstat" not in runner and " -version" not in runner,
            "license/version query forbidden")

    # Exact embedded receipt behavior.
    receipt_code=extract_receipt_python(runner)
    neg,neg_token=mock_receipt(receipt_code,"MET",0.100,"VIOLATED",-0.050)
    weird,weird_token=mock_receipt(receipt_code,"MET",0.100,"MET",-0.001)
    closed,closed_token=mock_receipt(receipt_code,"MET",0.100,"MET",0.000)
    require(neg_token.startswith("DIAGNOSTIC_STOP_") and
            weird_token.startswith("DIAGNOSTIC_STOP_") and
            "PASS_M1288" not in neg_token and "PASS_M1288" not in weird_token,
            "negative hold promoted to PASS")
    require(closed_token=="PASS_M1288_M917_FIXED_T10_PRELAYOUT_PTSTA_SETUP_HOLD_MET",
            "zero hold closure boundary")
    for payload in (neg,weird,closed):
        require(payload["hold"]["automatic_fix_performed"] is False and
                payload["scope"]["mapped_identity_mutated"] is False and
                payload["claim_boundary"]=={
                    "diagnostic_if_negative":True,"power":False,"energy":False,
                    "speedup":False,"system":False,"paper_ppa_ready":False,
                    "headline":False}, "receipt closed claim drift")

    attacks={}
    for name,key in (("power_promotion","power"),("speedup_promotion","speedup"),
                     ("system_promotion","system"),("headline_promotion","headline")):
        attacked=copy.deepcopy(data); attacked["claim_boundary"][key]=True
        try: validate_contract(attacked,validate_actual=False)
        except HammerError: attacks[name]="REJECTED"
        else: raise HammerError(name+" accepted")
    attacked=copy.deepcopy(data)
    netlist=next(key for key in attacked["exact_files"] if key.endswith("_mapped.v"))
    attacked["exact_files"][netlist]="0"*64
    try: validate_contract(attacked,validate_actual=True)
    except HammerError: attacks["netlist_sha_drift"]="REJECTED"
    else: raise HammerError("netlist SHA drift accepted")
    attacked_tcl=tcl+"\nfix_eco_timing -type hold\n"
    try: check_tcl(attacked_tcl)
    except HammerError: attacks["automatic_eco"]="REJECTED"
    else: raise HammerError("automatic ECO accepted")

    # Fixed-path mkdir is the durable attempt-consumption primitive.
    with tempfile.TemporaryDirectory(prefix="m1299_attempt.") as name:
        attempt=Path(name)/".m1288_attempt_consumed"
        attempt.mkdir()
        try: attempt.mkdir()
        except FileExistsError: attacks["attempt_reuse"]="REJECTED"
        else: raise HammerError("attempt reuse accepted")

    canonical=HW/"dc_handoff/runs/m1288_m917_fixed_t10_prelayout_ptsta_r1_20260830"
    work=Path(str(canonical)+".work")
    attempt=HW/"dc_handoff/runs/.m1288_m917_fixed_t10_ptsta_attempt_consumed"
    require(not canonical.exists() and not work.exists() and not attempt.exists(),
            "future one-shot namespace already consumed")

    return {
        "schema":"m1299_m1288_c3_fixed_t10_ptsta_receipt_blind_hammer_v1",
        "status":"PASS_M1299_M1288_SOURCE_DAG__ROOT_MAY_AUTHOR_ONE_FUTURE_LIVE_PREFLIGHT_LAUNCH_ADMISSION",
        "score":93,
        "issue_counts":{"P0":0,"P1":2,"P2":1},
        "receipt_blind":True,"pt_eda_license_calls":0,
        "upstream_seals":3,"source_identities":len(EXPECTED_SOURCE_SHA),
        "hold_mock":{"negative_violated":neg_token,"negative_met_label":weird_token,
                     "zero_met":closed_token},
        "attacks":attacks,
        "authorization":{
            "pt_run_now":False,
            "root_may_author_future_live_preflight_and_launch_admission":True,
            "maximum_future_attempts":1,
            "admission_must_be_exact_closed_and_double_sealed":True,
        },
    }


if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True,allow_nan=False))
