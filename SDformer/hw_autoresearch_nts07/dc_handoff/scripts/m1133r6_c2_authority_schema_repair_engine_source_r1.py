#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1133r6 additive authority-schema repair for the frozen M1129r5 engine.

No RTL, TB, filelist, selector, reset-provenance, or execution mechanism is
changed.  M1121 is checked directly here and by the future launch receipt;
the engine-hammer review is intentionally not asked for a nonexistent M1121
identity field.  This source remains non-executable until M1134r6, M1135r6,
and M1136r6 independently seal the future authority chain.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ENGINE = Path(__file__).resolve()
HW = ENGINE.parent.parent.parent
BASE_ENGINE = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
BASE_ENGINE_SHA256 = "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b"
CONTRACT = HW / "contracts/m1133r6_c2_authority_schema_repair_engine_source_contract_r1_20260830.json"
CONTRACT_SHA256 = "4dc16ffccb3c4a145f69f565500d67407ca821304ee838f93659918055a3ac8a"
CONTRACT_OUTER_SHA256 = "82b6d6a6568fc8fc95f1a1b7b6bf05690e06e064a143de41eadfa0e76ac9b849"
LAUNCHER = HW / "dc_handoff/scripts/run_m1133r6_c2_authority_schema_repair_authorized_launch_r1.py"
LAUNCH_RECEIPT = HW / "contracts/m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_20260830.json"
M1134R6 = HW / "reviews/m1134r6_m1133r6_c2_authority_schema_engine_hammer_r1_20260830"
M1136R6 = HW / "reviews/m1136r6_m1133r6_c2_final_launch_hammer_r1_20260830"
M1132R5_STOP = HW / "reviews/m1132r5_m1129r5_c2_dc_selector_launch_hammer_r1_20260830"
M1132R5_STOP_OUTER_SHA256 = "bc073b90787189710986381b74c18b9a3afbe4ccd2f7969e85b596d3df1adf48"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
M1121_OUTER_SHA256 = "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RTL_SHA256 = "86df0f7be383e6ba8ee17c1e27fc25fd18eb6fecc01329c41a976cd836004dd0"
TB_SHA256 = "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b"
FILELIST_SHA256 = "1ac2715245cce259f3dcba37cbeecac0e9a2ab9b16a60463f6a53f668ff9e106"

RESULT = HW / "results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830"
ATTEMPT = HW / "results/.m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed"
WORK = HW / f"results/.m1133r6_c2_authority_schema_repair_dc_mapped_vcs_work.{os.getpid()}"
FAILURE = HW / f"results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
WORK_GLOB = ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_work.*"
FAILURE_GLOB = "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
LOCK = Path("/tmp/m1133r6_c2_authority_schema_repair_eda.lock")

R5_ATTEMPT = HW / "results/.m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R5_RESULT = HW / "results/m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R5_WORK_GLOB = ".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*"
R5_FAILURE_GLOB = "m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
R5_LOCK = Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha(BASE_ENGINE) != BASE_ENGINE_SHA256:
    raise RuntimeError("frozen M1129r5 base-engine identity drift")
_spec = importlib.util.spec_from_file_location("m1129r5_frozen_base", BASE_ENGINE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen M1129r5 base engine")
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)

GateFailure = BASE.GateFailure
fail = BASE.fail


def configure_base() -> None:
    """Bind the frozen execution implementation to fresh r6 namespaces."""
    BASE.ENGINE = ENGINE
    BASE.CONTRACT = CONTRACT
    BASE.LAUNCHER = LAUNCHER
    BASE.LAUNCH_RECEIPT = LAUNCH_RECEIPT
    BASE.M1130R5 = M1134R6
    BASE.M1132R5 = M1136R6
    BASE.RESULT = RESULT
    BASE.ATTEMPT = ATTEMPT
    BASE.WORK = WORK
    BASE.FAILURE = FAILURE
    BASE.WORK_GLOB = WORK_GLOB
    BASE.FAILURE_GLOB = FAILURE_GLOB
    BASE.LOCK = LOCK
    BASE.CONTRACT_SHA256 = CONTRACT_SHA256
    BASE.CONTRACT_OUTER_SHA256 = CONTRACT_OUTER_SHA256
    BASE.verify_future_authority = verify_future_authority
    BASE.static_gate = static_gate
    BASE.flow = flow
    BASE.quarantine = quarantine


def verify_future_authority() -> dict:
    """Verify the acyclic future r6 chain with an exact receipt schema."""
    if not LAUNCHER.exists() or not LAUNCH_RECEIPT.exists():
        fail("future r6 launcher/receipt absent: source stage cannot execute")
    receipt_outer = BASE.verify_double_self_consistent(LAUNCH_RECEIPT)
    receipt = BASE.load(LAUNCH_RECEIPT)
    expected_keys = {
        "schema", "status", "launcher_sha256", "engine_sha256",
        "engine_contract_sha256", "engine_contract_outer_seal_file_sha256",
        "engine_author_receipt_outer_seal_file_sha256",
        "m1121_outer_seal_file_sha256", "m1132r5_stop_outer_seal_file_sha256",
        "m1134r6_outer_seal_file_sha256", "arguments",
        "caller_selected_authority_allowed", "caller_environment_forwarded",
        "m1136r6_required", "launch_now", "attempt_now", "dc_now",
        "mapped_vcs_now", "maximum_attempts", "automatic_retry",
        "paper_citable",
    }
    if set(receipt) != expected_keys:
        fail("future r6 launch receipt exact-key drift")
    if receipt["status"] != "M1133R6_LAUNCH_SOURCE_FROZEN__M1136R6_REQUIRED__NO_EDA":
        fail("future r6 launch receipt status")
    if (receipt["schema"] != "m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_v1" or
            receipt["engine_sha256"] != _sha(ENGINE) or
            receipt["engine_contract_sha256"] != CONTRACT_SHA256 or
            receipt["engine_contract_outer_seal_file_sha256"] != CONTRACT_OUTER_SHA256 or
            receipt["m1121_outer_seal_file_sha256"] != M1121_OUTER_SHA256 or
            receipt["m1132r5_stop_outer_seal_file_sha256"] != M1132R5_STOP_OUTER_SHA256 or
            receipt["arguments"] != 0 or
            receipt["caller_selected_authority_allowed"] is not False or
            receipt["caller_environment_forwarded"] is not False or
            receipt["m1136r6_required"] is not True or
            receipt["launch_now"] is not False or receipt["attempt_now"] is not False or
            receipt["dc_now"] is not False or receipt["mapped_vcs_now"] is not False or
            receipt["maximum_attempts"] != 1 or receipt["automatic_retry"] is not False or
            receipt["paper_citable"] is not False):
        fail("future r6 launch receipt authority/boundary drift")

    BASE.verify_exact_flat(M1134R6, receipt["m1134r6_outer_seal_file_sha256"])
    engine_review = BASE.load(M1134R6 / "review.json")
    if (engine_review["status"] !=
            "PASS_M1134R6_M1133R6_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA"):
        fail("M1134r6 has no GO")
    engine_identity = engine_review["identity"]
    # Deliberately only the three fields owned by the engine hammer.  M1121 is
    # not in this schema and must never be fetched from engine_identity.
    if (set(engine_identity) != {"engine_sha256", "contract_sha256",
                                "author_receipt_outer_seal_file_sha256"} or
            engine_identity["engine_sha256"] != _sha(ENGINE) or
            engine_identity["contract_sha256"] != CONTRACT_SHA256 or
            receipt["engine_author_receipt_outer_seal_file_sha256"] !=
                engine_identity["author_receipt_outer_seal_file_sha256"]):
        fail("M1134r6 exact engine authority drift")

    launch_outer = BASE.verify_flat_self_consistent(M1136R6)
    launch_review = BASE.load(M1136R6 / "review.json")
    if launch_review["status"] != "PASS_M1136R6_M1133R6_FINAL_LAUNCH_HAMMER__GO_ONE_ATTEMPT":
        fail("M1136r6 has no GO")
    identity = launch_review["identity"]
    if (set(identity) != {"launch_receipt_outer_seal_file_sha256", "launcher_sha256",
                          "engine_sha256", "engine_contract_outer_seal_file_sha256",
                          "engine_author_receipt_outer_seal_file_sha256",
                          "m1121_outer_seal_file_sha256",
                          "m1132r5_stop_outer_seal_file_sha256",
                          "m1134r6_outer_seal_file_sha256"} or
            identity["launch_receipt_outer_seal_file_sha256"] != receipt_outer or
            identity["launcher_sha256"] != receipt["launcher_sha256"] or
            identity["engine_sha256"] != _sha(ENGINE) or
            identity["engine_contract_outer_seal_file_sha256"] != CONTRACT_OUTER_SHA256 or
            identity["engine_author_receipt_outer_seal_file_sha256"] !=
                receipt["engine_author_receipt_outer_seal_file_sha256"] or
            identity["m1121_outer_seal_file_sha256"] != M1121_OUTER_SHA256 or
            identity["m1132r5_stop_outer_seal_file_sha256"] != M1132R5_STOP_OUTER_SHA256 or
            identity["m1134r6_outer_seal_file_sha256"] !=
                receipt["m1134r6_outer_seal_file_sha256"]):
        fail("M1136r6 exact launcher/receipt/engine authority drift")
    BASE.verify_parent_launcher(receipt)
    result = dict(receipt)
    result["m1136r6_outer_seal_file_sha256"] = launch_outer
    return result


def _namespace_absent(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def static_gate() -> dict:
    """Exact source/authority gate; it performs no EDA or external mutation."""
    if sys.argv[1:] != ["--authorized-launch"]:
        fail("fixed argv required")
    if Path(sys.executable) != BASE.PYTHON:
        fail("unpinned Python")
    BASE.verify_regular(ENGINE, _sha(ENGINE))
    BASE.verify_regular(BASE_ENGINE, BASE_ENGINE_SHA256)
    BASE.verify_double(CONTRACT, CONTRACT_SHA256, CONTRACT_OUTER_SHA256)
    contract = BASE.load(CONTRACT)
    if (contract["status"] !=
            "M1133R6_AUTHORITY_SCHEMA_REPAIR_SOURCE_ONLY__DIFFERENT_AUTHOR_ENGINE_HAMMER_REQUIRED__NO_EDA" or
            contract["launch_now"] is not False or contract["max_attempts_now"] != 0):
        fail("M1133r6 source boundary drift")
    frozen = contract["frozen_real_module_identity"]
    if (frozen["design"] != BASE.DESIGN or frozen["tb_top"] != BASE.TB_TOP or
            frozen["rtl_sha256"] != RTL_SHA256 or frozen["tb_sha256"] != TB_SHA256 or
            frozen["filelist_sha256"] != FILELIST_SHA256 or
            frozen["rtl_tb_filelist_modified"] is not False or
            frozen["real_module_identity_reused_exactly"] is not True):
        fail("frozen r5 real-module contract drift")
    repair = contract["authority_schema_repair"]
    if (repair["forbidden_lookup"] !=
            "engine_hammer_review.identity.m1121_outer_seal_file_sha256" or
            repair["engine_hammer_required_identity_keys"] != [
                "engine_sha256", "contract_sha256",
                "author_receipt_outer_seal_file_sha256"] or
            repair["missing_extra_wrong_future_keys_fail_closed"] is not True or
            repair["complete_future_authority_fixture_must_reach_static_gate_return"] is not True):
        fail("authority-schema repair contract drift")
    if (contract["m1121_authority"]["outer_seal_file_sha256"] != M1121_OUTER_SHA256 or
            contract["m1121_authority"]["engine_static_gate_exact_flat_verification_required"] is not True or
            contract["m1121_authority"]["future_launch_receipt_exact_value_required"] is not True or
            contract["m1121_authority"]["engine_hammer_review_identity_lookup_required"] is not False or
            contract["m1132r5_stop_authority"]["outer_seal_file_sha256"] != M1132R5_STOP_OUTER_SHA256 or
            contract["m1132r5_stop_authority"]["r5_launch_withdrawn"] is not True or
            contract["m1132r5_stop_authority"]["r5_attempt_created"] is not False):
        fail("M1121/M1132 authority contract drift")

    BASE.verify_regular(BASE.DOCS359, DOCS359_SHA256)
    BASE.verify_regular(BASE.RTL, RTL_SHA256)
    BASE.verify_regular(BASE.TB, TB_SHA256)
    BASE.verify_regular(BASE.FILELIST, FILELIST_SHA256)
    lexical = BASE.lexical_real_name_and_diff_gate()
    BASE.verify_exact_flat(M1121, M1121_OUTER_SHA256)
    if BASE.load(M1121 / "review.json")["status"] != (
            "PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY"):
        fail("M1121 exact authority drift")
    BASE.verify_exact_flat(M1132R5_STOP, M1132R5_STOP_OUTER_SHA256)
    stop = BASE.load(M1132R5_STOP / "review.json")
    if (stop["status"] !=
            "FAIL_M1132R5_M1129R5_POSTSEAL_FUTURE_AUTHORITY__ADDITIVE_R6_REQUIRED__NO_LAUNCH" or
            stop["execution"]["r5_namespace_created"] is not False or
            stop["authorization"]["r5_command_withdrawn"] is not True):
        fail("M1132r5 STOP authority drift")
    if (not _namespace_absent(R5_ATTEMPT) or not _namespace_absent(R5_RESULT) or
            not _namespace_absent(R5_LOCK) or
            any((HW / "results").glob(R5_WORK_GLOB)) or
            any((HW / "results").glob(R5_FAILURE_GLOB))):
        fail("stopped r5 namespace must remain absent")
    if (not _namespace_absent(ATTEMPT) or not _namespace_absent(RESULT) or
            not _namespace_absent(WORK) or not _namespace_absent(LOCK) or
            any((HW / "results").glob(WORK_GLOB)) or
            any((HW / "results").glob(FAILURE_GLOB))):
        fail("fresh r6 namespace collision")
    authority = verify_future_authority()
    authority["lexical_identity"] = lexical
    authority["m1121_exact_static_authority"] = True
    authority["m1132r5_stop_exact_static_authority"] = True
    return authority


phase = "SOURCE_PREFLIGHT"
attempted = False
complete = False


def flow() -> None:
    """Run the frozen r5 execution mechanics only after the full r6 gate."""
    global phase, attempted, complete
    configure_base()
    authority = static_gate()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("exclusive lock busy")
        BASE.collision_gate(); BASE.resource_gate(); BASE.license_gate()
        phase = "ATTEMPT_CONSUME_AFTER_M1134R6_M1136R6"
        ATTEMPT.mkdir(); attempted = True
        BASE.write_json(ATTEMPT / "attempt.json", {
            "status": "M1133R6_ATTEMPT_CONSUMED_AFTER_M1134R6_M1136R6",
            "engine_sha256": _sha(ENGINE), "contract_sha256": CONTRACT_SHA256,
            "launcher_sha256": authority["launcher_sha256"],
            "m1121_outer_seal_file_sha256": M1121_OUTER_SHA256,
            "m1132r5_stop_outer_seal_file_sha256": M1132R5_STOP_OUTER_SHA256,
            "m1134r6_outer_seal_file_sha256": authority["m1134r6_outer_seal_file_sha256"],
            "m1136r6_outer_seal_file_sha256": authority["m1136r6_outer_seal_file_sha256"],
            "dc_attempts": 1, "mapped_cases": 1, "random_initialization": False,
        }); BASE.seal(ATTEMPT)
        WORK.mkdir()
        env = os.environ.copy()
        env.update({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                    "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"})
        phase = "FRESH_DC_SELECTOR_M1133R6"
        dc = WORK / "dc"; dc.mkdir()
        dc_env = env.copy()
        dc_env.update({
            "DESIGN_NAME": BASE.DESIGN, "HW_ROOT": str(HW),
            "RTL_FILELIST": str(BASE.FILELIST), "LIB_DB": str(BASE.SLOW),
            "MIN_LIB_DB": str(BASE.FAST), "SDC_FILE": str(BASE.SDC),
            "OUTPUT_DIR": str(dc), "ELAB_PARAMETERS": "",
            "OPERATING_CONDITION": "ssg0p9v125c",
        })
        rc = BASE.run_dc_with_selector_capture(
            dc / "dc.log", 21600, dc_env, dc / "dc_selector_runtime_identity.json")
        if rc or not (dc / "TCL_PASS_TERMINAL.txt").is_file():
            fail("fresh DC failed")
        netlist = dc / f"netlist/{BASE.DESIGN}_mapped.v"
        if not netlist.is_file() or not netlist.stat().st_size:
            fail("mapped netlist absent")
        phase = "MAPPED_RESET_PROVENANCE_337"
        reset_census = BASE.structural_reset_gate(netlist)
        phase = "FRESH_MAPPED_VCS_CASE0_SHORT_128"
        mapped = WORK / "mapped_vcs"; mapped.mkdir(); simv = mapped / "simv"
        rc = BASE.run([
            str(BASE.VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            f"-Mdir={mapped / 'csrc'}", str(BASE.CELL), str(netlist),
            str(BASE.MEMORY), str(BASE.TB), "-top", BASE.TB_TOP, "-o", str(simv),
        ], mapped / "compile.log", 1800, env)
        if rc or not simv.is_file():
            fail("mapped compile failed")
        rc = BASE.run([str(simv), "-no_save"], mapped / "case0.log", 300, env)
        text = (mapped / "case0.log").read_text(encoding="utf-8", errors="replace")
        token = "PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 unknown_bitmap=000000 diagnostic_only=1"
        if rc or token not in text or "M1112_FIRST_X" in text:
            fail("mapped short window failed")
        BASE.write_json(WORK / "receipt.json", {
            "status": "PASS_M1133R6_DC_SELECTOR_RESET_PROVENANCE_MAPPED_SHORT_WINDOW",
            "dc_selector_runtime_identity_sha256": _sha(dc / "dc_selector_runtime_identity.json"),
            "mapped_netlist_sha256": _sha(netlist), "reset_provenance": reset_census,
            "stage_lines": len(BASE.re.findall(r"^M1112_STAGE", text, BASE.re.M)),
            "window_cycles": 128, "unknown_bitmap": "000000",
            "diagnostic_only": True, "paper_citable": False,
        })
        (WORK / "RUN_COMPLETE.txt").write_text(
            "PASS_M1133R6_DC_SELECTOR_RESET_PROVENANCE_MAPPED_SHORT_WINDOW\n",
            encoding="utf-8")
        BASE.seal(WORK); os.rename(WORK, RESULT); complete = True


def quarantine(message: str) -> None:
    if attempted and not complete:
        WORK.mkdir(parents=True, exist_ok=True)
        BASE.write_json(WORK / "failure.json", {
            "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE", "phase": phase,
            "message": message, "m1133r6_retry": False,
            "m1129r5_retry": False, "m1122r4_retry": False,
            "m1112r3_retry": False,
        })
        BASE.seal(WORK); os.rename(WORK, FAILURE)


def main() -> int:
    configure_base()
    for caught_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(caught_signal, lambda signum, _frame: (_ for _ in ()).throw(
            GateFailure("signal " + str(signum))))
    try:
        flow()
    except (GateFailure, OSError, subprocess.TimeoutExpired, KeyError,
            ValueError, json.JSONDecodeError) as exc:
        quarantine(str(exc))
        print("M1133r6 failure: " + str(exc), file=sys.stderr)
        return 3
    print("PASS_M1133R6_DC_SELECTOR_RESET_PROVENANCE_MAPPED_SHORT_WINDOW")
    return 0


configure_base()

if __name__ == "__main__":
    raise SystemExit(main())
