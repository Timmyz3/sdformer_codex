#!/opt/anaconda3/bin/python3
"""One-shot runner for M2172's balanced-scope native-SAIF repair.

SOURCE ONLY until a double-sealed M2173 independent hammer authorizes one
fresh M2174 identity.  No EDA, license query, GPU work, or production artifact
is created by source authoring or --static.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
RUNNER = Path(__file__).resolve()
HELPER_PATH = HW / "dc_handoff/scripts/run_m2160_m2018_ordinary_native_saif_report_reset_preflight_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
BASE_PARSER = HW / "system_simulator/scripts/parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py"
CONTRACT = HW / "contracts/m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2173_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_hammer_r1_20260904"
M2161 = HW / "reviews/m2161_m2160_m2018_ordinary_native_saif_report_reset_preflight_source_hammer_r1_20260904"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2160_m2018_ordinary_native_saif_report_reset_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
TB = HW / "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"
TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RESULT = HW / "results/m2174_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2174_m2172_ordinary_native_saif_balanced_scope_preflight_attempt_consumed"
LOCK = HW / "results/.m2174_m2172_ordinary_native_saif_balanced_scope_preflight_launch_lock"
COUNTS = {
    "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
    "raw_saif_files_written": 2, "diagnostic_saif_files_written": 1,
    "admitted_measurement_saif_files": 1, "admitted_saif_files": 1,
    "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0,
}
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
VCS_SHA256 = "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LMUTIL_SHA256 = "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07"
LICENSE_SERVER = "27030@ic.ismd-nemo"


def load_helper():
    spec = importlib.util.spec_from_file_location("m2160_runner_frozen_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("M2160 helper import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()
Failure = H.Failure
need = H.need
sha256 = H.sha256
strict_json = H.strict_json
write_json = H.write_json
verify_seal = H.verify_seal
seal_dir = H.seal_dir
seal_file = H.seal_file
clean_env = H.clean_env
run = H.run
validate_regular_tool = H.validate_regular_tool
validate_timing_surface = H.validate_timing_surface


def audit_m2161_rejection() -> dict[str, object]:
    members = verify_seal(M2161)
    review = strict_json(M2161 / "review.json")
    need(review.get("status") ==
         "FAIL_M2161_M2160_SOURCE_HAMMER__RESET_WARNING_AND_SAIF_SCOPE_GATES_FAIL_OPEN__M2162_NOT_AUTHORIZED",
         "M2161 rejection status")
    need(review.get("severity_counts") == {"p0": 2, "p1": 0, "p2": 0},
         "M2161 rejection severity")
    need(review.get("authorization", {}).get("m2162_authorized") is False,
         "M2161 M2162 authority")
    return {
        "review_sha256": sha256(M2161 / "review.json"),
        "manifest_sha256": sha256(M2161 / "SHA256SUMS"),
        "outer_sha256": sha256(M2161 / "SHA256SUMS.seal.sha256"),
        "member_count": len(members), "m2162_authorized": False,
        "allowed_now": "FRESH_SOURCE_AUTHORING_ONLY",
    }


def topology_audit() -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("m2172_parser_for_runner", PARSER)
    need(spec is not None and spec.loader is not None, "parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_single_axis_source(TB.read_text(), FILELIST.read_text())


def source_validation(require_review: bool) -> dict:
    need(sha256(DOC359) == DOC359_SHA256, "docs359 identity")
    validate_regular_tool(VCS, VCS_SHA256, "VCS")
    validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2173_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "ordinary_only": True,
        "single_frontend": True, "reuse_old_artifacts": False},
        "execution budget")
    need(contract.get("m2161_rejection") == audit_m2161_rejection(),
         "M2161 rejection fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    need(topology_audit() == contract.get("single_axis_topology"),
         "topology fingerprint")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2173"),
             "M2173 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2173 severity")
        need(review.get("score_over_100", 0) >= 95, "M2173 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "ordinary_only": True,
            "single_frontend": True, "reuse_old_artifacts": False},
            "M2173 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha256(RUNNER),
             "M2173 runner identity")
        need(identity.get("parser_sha256") == sha256(PARSER),
             "M2173 parser identity")
        need(identity.get("contract_sha256") == sha256(CONTRACT),
             "M2173 contract identity")
    return contract


def no_same_uid_eda() -> None:
    own_pid = os.getpid()
    offenders: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if re.search(r"(^|/|\s)(vcs|simv|dc_shell|pt_shell|icc2_shell)(\s|$)", cmdline):
            offenders.append(f"{entry.name}:{cmdline[:240]}")
    need(not offenders, f"same-UID EDA active: {offenders}")


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2174_ATTEMPT_CONSUMED", "budget": COUNTS,
        "automatic_retry": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2174_m2172_work.", dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2174_m2172_stage.", dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    commands: dict[str, object] = {}
    try:
        counts["license_queries"] += 1
        license_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        commands["license_preflight"] = license_command
        run(license_command, work, clean_env({}), 60, stage / "license_preflight.log")

        build = work / "vcs_build"
        build.mkdir()
        sources = [(REPO / line.strip()).resolve()
                   for line in FILELIST.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        need(len(sources) == 4 and sources[-1] == TB.resolve(),
             "VCS source count/order")
        need(all(path.is_file() and not path.is_symlink() for path in sources),
             "VCS source absent/symlink")
        resolved = build / "sources.absolute.f"
        resolved.write_text("\n".join(map(str, sources)) + "\n")
        compile_command = [
            str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            "+vcs+initreg+random", "-debug_access+r", "-lca", "+vcs+lic+wait",
            f"-Mdir={build / 'csrc'}", "-f", str(resolved), "-top",
            "tb_m2160_m2018_ordinary_native_saif_report_reset_preflight",
            "-o", str(build / "simv"),
        ]
        commands["timing_surface"] = validate_timing_surface(
            compile_command, [FILELIST, *sources])
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        run(compile_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent),
                       "VCS_ARCH_OVERRIDE": "linux"}),
            21600, stage / "vcs_compile.log")
        need((build / "simv").is_file() and not (build / "simv").is_symlink(),
             "simv absent/symlink")

        prehistory_saif = stage / "rtl_prehistory.saif"
        measurement_saif = stage / "rtl_measurement.saif"
        sim_command = ["./simv", "-lca", "+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                       "+M2160_AXIS_ORDINARY", "-no_save", "-ucli", "-i", str(UCLI)]
        need([item for item in sim_command if item.startswith("+")] ==
             ["+vcs+initreg+0", "+WORKLOAD_SLOT=42", "+M2160_AXIS_ORDINARY"],
             "runtime plusarg surface")
        commands["simv"] = sim_command
        counts["simv_runs"] += 1
        run(sim_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent),
                       "VCS_ARCH_OVERRIDE": "linux",
                       "M2160_PREHISTORY_SAIF_FILE": str(prehistory_saif),
                       "M2160_MEASUREMENT_SAIF_FILE": str(measurement_saif)}),
            21600, stage / "rtl_sim.log")
        for path in (prehistory_saif, measurement_saif):
            need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                 f"raw SAIF absent/empty/symlink: {path.name}")
            seal_file(path)
            counts["raw_saif_files_written"] += 1
        counts["diagnostic_saif_files_written"] += 1
        for command, timeout, log in (
            ([sys.executable, str(PARSER), "runtime", "--path",
              str(stage / "rtl_sim.log")], 60, "runtime_parse.log"),
            ([sys.executable, str(PARSER), "saif", "--path", str(prehistory_saif),
              "--role", "diagnostic_prehistory"], 180, "prehistory_saif_parse.log"),
            ([sys.executable, str(PARSER), "saif", "--path", str(measurement_saif),
              "--role", "measurement"], 180, "saif_parse.log"),
        ):
            run(command, REPO, clean_env({}), timeout, stage / log)
        counts["admitted_measurement_saif_files"] += 1
        counts["admitted_saif_files"] += 1
        need(counts == COUNTS, f"execution counts: {counts}")
        write_json(stage / "execution_commands.json", commands)
        run([sys.executable, str(PARSER), "final", "--root", str(stage),
             "--output", str(stage / "result.json")],
            REPO, clean_env({}), 240, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_counts": counts,
            "identity": {
                "runner_sha256": sha256(RUNNER), "parser_sha256": sha256(PARSER),
                "contract_sha256": sha256(CONTRACT),
                "m2173_review_sha256": sha256(REVIEW / "review.json"),
                "m2161_rejection_sha256": sha256(M2161 / "review.json"),
                "docs359_sha256": sha256(DOC359),
                "vcs_sha256": sha256(VCS), "lmutil_sha256": sha256(LMUTIL),
            },
            "independent_result_hammer_required": True,
        })
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2174_M2172_BALANCED_SCOPE_NATIVE_SAIF_PREFLIGHT_PENDING_M2175_RESULT_HAMMER\n")
        seal_dir(stage)
        os.rename(stage, RESULT)
        shutil.rmtree(work)
        LOCK.rmdir()
        return 0
    except BaseException as exc:
        (stage / "FAILED_DO_NOT_CITE.txt").write_text(
            f"status=FAILED_DO_NOT_CITE\nexception={type(exc).__name__}: {exc}\n"
            "automatic_retry=false\n")
        write_json(stage / "execution_counts.json", counts)
        write_json(stage / "execution_commands.json", commands)
        seal_dir(stage)
        quarantine = Path(str(RESULT) + f".failed.{os.getpid()}.quarantine")
        if not quarantine.exists():
            os.rename(stage, quarantine)
        if LOCK.exists():
            LOCK.rmdir()
        if work.exists():
            shutil.rmtree(work)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    if args.static:
        contract = source_validation(require_review=False)
        print(json.dumps({
            "status": "PASS_M2172_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "single_axis_topology": topology_audit(),
            "m2161_rejection": audit_m2161_rejection(),
            "tools": {"vcs": validate_regular_tool(VCS, VCS_SHA256, "VCS"),
                      "lmutil": validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")},
        }, indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2172_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
