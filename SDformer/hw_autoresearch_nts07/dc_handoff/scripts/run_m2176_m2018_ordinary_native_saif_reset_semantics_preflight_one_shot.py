#!/opt/anaconda3/bin/python3
"""One-shot M2176 reset-semantic successor; source-only until M2177 PASS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
RUNNER = Path(__file__).resolve()
BASE_RUNNER_PATH = HW / "dc_handoff/scripts/run_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"
CONTRACT = HW / "contracts/m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2177_m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_hammer_r1_20260904"
M2173 = HW / "reviews/m2173_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_hammer_r1_20260904"
TEST = HW / "tests/test_m2176_ordinary_native_saif_reset_semantics_preflight.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m2178_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2178_m2176_ordinary_native_saif_reset_semantics_preflight_attempt_consumed"
LOCK = HW / "results/.m2178_m2176_ordinary_native_saif_reset_semantics_preflight_launch_lock"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def load_base():
    spec = importlib.util.spec_from_file_location("m2172_runner_frozen_for_m2176", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("M2172 runner import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()
Failure = B.Failure
need = B.need
sha256 = B.sha256
strict_json = B.strict_json
write_json = B.write_json
verify_seal = B.verify_seal
seal_dir = B.seal_dir
seal_file = B.seal_file
clean_env = B.clean_env
run = B.run
validate_regular_tool = B.validate_regular_tool
validate_timing_surface = B.validate_timing_surface
no_same_uid_eda = B.no_same_uid_eda
FILELIST = B.FILELIST
UCLI = B.UCLI
TB = B.TB
VCS = B.VCS
VCS_SHA256 = B.VCS_SHA256
LMUTIL = B.LMUTIL
LMUTIL_SHA256 = B.LMUTIL_SHA256
LICENSE_SERVER = B.LICENSE_SERVER
COUNTS = dict(B.COUNTS)


def audit_m2173_rejection() -> dict[str, object]:
    members = verify_seal(M2173)
    review = strict_json(M2173 / "review.json")
    need(review.get("status") ==
         "FAIL_M2173_M2172_SOURCE_HAMMER__RESET_SYNONYM_GATE_INCOMPLETE__M2174_NOT_AUTHORIZED",
         "M2173 rejection status")
    need(review.get("severity_counts") == {"p0": 1, "p1": 0, "p2": 0},
         "M2173 rejection severity")
    need(review.get("authorization", {}).get("m2174_authorized") is False,
         "M2173 M2174 authority")
    return {
        "review_sha256": sha256(M2173 / "review.json"),
        "manifest_sha256": sha256(M2173 / "SHA256SUMS"),
        "outer_sha256": sha256(M2173 / "SHA256SUMS.seal.sha256"),
        "member_count": len(members), "m2174_authorized": False,
        "allowed_now": "FRESH_M2176_SOURCE_AUTHORING_ONLY",
    }


def topology_audit() -> dict[str, object]:
    spec = importlib.util.spec_from_file_location("m2176_parser_for_runner", PARSER)
    need(spec is not None and spec.loader is not None, "parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_single_axis_source(TB.read_text(), FILELIST.read_text())


def source_validation(require_review: bool) -> dict:
    need(sha256(DOC359) == DOC359_SHA256, "docs359 identity")
    validate_regular_tool(VCS, VCS_SHA256, "VCS")
    validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name], "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") == "SOURCE_ONLY__M2177_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "ordinary_only": True,
        "single_frontend": True, "reuse_old_artifacts": False}, "execution budget")
    need(contract.get("m2173_rejection") == audit_m2173_rejection(),
         "M2173 rejection fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    need(topology_audit() == contract.get("single_axis_topology"), "topology fingerprint")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(review.get("status") ==
             "PASS_M2177_M2176_SOURCE_HAMMER__M2178_ONE_SHOT_AUTHORIZED",
             "M2177 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2177 severity")
        need(review.get("score_over_100", 0) >= 95, "M2177 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "ordinary_only": True,
            "single_frontend": True, "reuse_old_artifacts": False},
            "M2177 authorization")
        identity = review.get("identity", {})
        for key, path in (("runner_sha256", RUNNER), ("parser_sha256", PARSER),
                          ("contract_sha256", CONTRACT)):
            need(identity.get(key) == sha256(path), f"M2177 {key}")
    return contract


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2178_ATTEMPT_CONSUMED", "budget": COUNTS,
        "automatic_retry": False, "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2178_m2176_work.", dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2178_m2176_stage.", dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    commands: dict[str, object] = {}
    try:
        counts["license_queries"] += 1
        license_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        commands["license_preflight"] = license_command
        run(license_command, work, clean_env({}), 60, stage / "license_preflight.log")
        build = work / "vcs_build"
        build.mkdir()
        sources = [(REPO / line.strip()).resolve() for line in FILELIST.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        need(len(sources) == 4 and sources[-1] == TB.resolve(), "VCS source count/order")
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
        commands["timing_surface"] = validate_timing_surface(compile_command, [FILELIST, *sources])
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        run(compile_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux"}),
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
        run(sim_command, build, clean_env({
            "VCS_HOME": str(VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux",
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
            ([sys.executable, str(PARSER), "runtime", "--path", str(stage / "rtl_sim.log")], 60, "runtime_parse.log"),
            ([sys.executable, str(PARSER), "saif", "--path", str(prehistory_saif), "--role", "diagnostic_prehistory"], 180, "prehistory_saif_parse.log"),
            ([sys.executable, str(PARSER), "saif", "--path", str(measurement_saif), "--role", "measurement"], 180, "saif_parse.log"),
        ):
            run(command, REPO, clean_env({}), timeout, stage / log)
        counts["admitted_measurement_saif_files"] += 1
        counts["admitted_saif_files"] += 1
        need(counts == COUNTS, f"execution counts: {counts}")
        write_json(stage / "execution_commands.json", commands)
        run([sys.executable, str(PARSER), "final", "--root", str(stage),
             "--output", str(stage / "result.json")], REPO, clean_env({}), 240,
            stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_counts": counts,
            "identity": {
                "runner_sha256": sha256(RUNNER), "parser_sha256": sha256(PARSER),
                "contract_sha256": sha256(CONTRACT),
                "m2177_review_sha256": sha256(REVIEW / "review.json"),
                "m2173_rejection_sha256": sha256(M2173 / "review.json"),
                "docs359_sha256": sha256(DOC359),
                "vcs_sha256": sha256(VCS), "lmutil_sha256": sha256(LMUTIL),
            },
            "independent_result_hammer_required": True,
        })
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2178_M2176_RESET_SEMANTICS_NATIVE_SAIF_PREFLIGHT_PENDING_M2179_RESULT_HAMMER\n")
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
            "status": "PASS_M2176_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "single_axis_topology": topology_audit(),
            "m2173_rejection": audit_m2173_rejection(),
            "tools": {"vcs": validate_regular_tool(VCS, VCS_SHA256, "VCS"),
                      "lmutil": validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")},
        }, indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2176_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
