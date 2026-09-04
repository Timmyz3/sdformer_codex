#!/opt/anaconda3/bin/python3
"""One-shot M2137 successor to the consumed/failed M2127 diagnostic.

This additive runner reuses the exact M2125 RTL/TB/UCLI/parser data plane and
changes only the pre-launch timing-contamination validation and fresh campaign
identities.  Arbitrary path operands may contain ``SDformer``.  Actual SDF
options, UNIT_DELAY defines, or active-source timing annotations remain fatal.

SOURCE ONLY until an exhaustive double-sealed M2138 independent review.  A
future M2139 is exactly one license query, one shared VCS compile, two serial
simv runs, and two DUT-only SAIFs.  It has no DC/PT/ICC2/GPU/retry path.
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
M2125_RUNNER = HW / "dc_handoff/scripts/run_m2125_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
CONTRACT = HW / "contracts/m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2138_m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904"
M2128 = HW / "reviews/m2128_m2127_m2125_tsbg_rtl_saif_window_diagnostic_failure_hammer_r1_20260904"
M2126 = HW / "reviews/m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
TEST = HW / "tests/test_m2137_tsbg_rtl_saif_option_aware_timing_surface.py"

RESULT = HW / "results/m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
ATTEMPT = HW / "results/.m2139_m2137_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
LOCK = HW / "results/.m2139_m2137_tsbg_rtl_saif_window_diagnostic_launch_lock"
COUNTS = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
          "saif_files": 2, "dc_runs": 0, "ptpx_runs": 0}

M2125_RUNNER_SHA256 = "6021c4a9b4297e5527f09006f21dd3a06d98b2a7ad76ffc55ca259029e658815"
M2126_MANIFEST_SHA256 = "db8f8bd83ddc6a483baff88bd1460e8b829b51757ec524421399a45d84235bdc"
M2126_OUTER_SHA256 = "d3313574bf92184c6029d078dfa8010e733c0936519f76e790add24e8f6a87f7"
M2128_MANIFEST_SHA256 = "5ecbb1bd4fc6bf1d3851566c259b837aeaa3c94d3a1bce2631a735c28b20ae4c"
M2128_OUTER_SHA256 = "9a2ad99b8dfaaccb121ec391fa0c2540d7aa8c88e6ce8b6384776474edaf524e"
M2128_REVIEW_SHA256 = "e43f1e38b8c11b522a9d35041260d8398dfbe07b5aa3db1e312a26952ee63928"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def load_m2125():
    need(M2125_RUNNER.is_file() and not M2125_RUNNER.is_symlink(),
         "M2125 runner absent/symlink")
    spec = importlib.util.spec_from_file_location("m2137_frozen_m2125", M2125_RUNNER)
    need(spec is not None and spec.loader is not None, "M2125 import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M2125 = load_m2125()


def sha256(path: Path) -> str:
    return M2125.sha256(path)


def strict_json(path: Path) -> dict:
    return M2125.strict_json(path)


def write_json(path: Path, value: object) -> None:
    M2125.write_json(path, value)


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict[str, str]:
    return M2125.verify_seal(root, expected_manifest, expected_outer)


def seal_dir(root: Path) -> None:
    M2125.seal_dir(root)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    return M2125.clean_env(extra)


def is_explicit_sdf_option(token: str) -> bool:
    lowered = token.lower()
    return lowered.startswith("-sdf") or lowered.startswith("+sdf")


def defines_unit_delay(token: str) -> bool:
    if not token.lower().startswith("+define+"):
        return False
    definitions = token[len("+define+"):].split("+")
    return any(item.split("=", 1)[0].upper() == "UNIT_DELAY"
               for item in definitions)


def validate_timing_surface(command: list[str], active_inputs: list[Path]) -> dict[str, object]:
    """Reject controls/content, never harmless substrings in pathname operands."""
    explicit_sdf = [token for token in command if is_explicit_sdf_option(token)]
    unit_delay_defines = [token for token in command if defines_unit_delay(token)]
    need(not explicit_sdf, f"explicit SDF option: {explicit_sdf}")
    need(not unit_delay_defines, f"explicit UNIT_DELAY define: {unit_delay_defines}")
    for path in active_inputs:
        need(path.is_file() and not path.is_symlink(), f"active input: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        need(not re.search(r"\$sdf_annotate\b", text, flags=re.IGNORECASE),
             f"source-level SDF annotation: {path}")
        need("UNIT_DELAY" not in text, f"source-level UNIT_DELAY: {path}")
    return {
        "explicit_sdf_options": 0,
        "explicit_unit_delay_defines": 0,
        "active_input_count": len(active_inputs),
        "path_operands_may_contain_sdf_substring": True,
    }


def audit_m2127_disposition() -> dict[str, object]:
    verify_seal(M2126, M2126_MANIFEST_SHA256, M2126_OUTER_SHA256)
    verify_seal(M2128, M2128_MANIFEST_SHA256, M2128_OUTER_SHA256)
    review_path = M2128 / "review.json"
    need(sha256(review_path) == M2128_REVIEW_SHA256, "M2128 review SHA")
    review = strict_json(review_path)
    need(review.get("status") ==
         "PASS_M2128_M2127_FAILURE_HAMMER__CONSUMED_NO_RETRY__FRESH_SOURCE_ONLY",
         "M2128 status")
    disposition = review.get("m2127_disposition", {})
    need(disposition.get("attempt_consumed") is True
         and disposition.get("automatic_retry") is False
         and disposition.get("retry_authorized") is False
         and disposition.get("paper_citable") is False, "M2127 disposition")
    successor = review.get("only_allowed_successor", {})
    need(successor.get("authorization_now") == "SOURCE_AUTHORING_ONLY"
         and successor.get("m2125_edit_or_retry_allowed") is False,
         "M2128 successor boundary")
    return {
        "m2128_review_sha256": sha256(review_path),
        "m2128_outer_sha256": sha256(M2128 / "SHA256SUMS.seal.sha256"),
        "m2127_attempt_consumed": True,
        "m2127_retry_authorized": False,
        "m2127_paper_citable": False,
    }


def source_validation(require_review: bool) -> dict:
    need(sha256(M2125_RUNNER) == M2125_RUNNER_SHA256, "M2125 runner SHA")
    # This freezes all 15 M2125 sources, tools, M2119/M2120 evidence and docs359.
    old_contract = M2125.source_validation(require_review=False)
    disposition = audit_m2127_disposition()
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2138_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "p1_serial": True,
        "reuse_old_artifacts": False}, "execution budget")
    need(contract.get("m2127_disposition") == disposition,
         "M2127 disposition fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    need(old_contract.get("source_inventory") is not None, "M2125 inventory")
    need(sha256(DOC359) == DOC359_SHA256, "docs359 identity")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2138"),
             "M2138 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2138 severities")
        need(review.get("score_over_100", 0) >= 95, "M2138 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "p1_serial": True,
            "reuse_old_artifacts": False}, "M2138 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha256(RUNNER), "review runner")
        need(identity.get("contract_sha256") == sha256(CONTRACT), "review contract")
    return contract


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    M2125.no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2139_ATTEMPT_CONSUMED",
        "budget": COUNTS,
        "automatic_retry": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2139_m2137_work.",
                                 dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2139_m2137_stage.",
                                  dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    commands: dict[str, object] = {}
    try:
        counts["license_queries"] += 1
        license_command = [str(M2125.LMUTIL), "lmstat", "-a", "-c",
                           M2125.LICENSE_SERVER]
        commands["license_preflight"] = license_command
        M2125.run(license_command, work, clean_env({}), 60,
                  stage / "license_preflight.log")

        build = work / "vcs_build"
        build.mkdir()
        resolved_filelist = build / "sources.absolute.f"
        sources: list[Path] = []
        for line in M2125.FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                source = (REPO / stripped).resolve()
                need(source.is_file() and not source.is_symlink(),
                     f"VCS source absent: {source}")
                sources.append(source)
        need(len(sources) == 6, "VCS source count")
        resolved_filelist.write_text("\n".join(map(str, sources)) + "\n")
        compile_command = [
            str(M2125.VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "+vcs+initreg+random",
            "-debug_access+r", "-lca", "+vcs+lic+wait",
            f"-Mdir={build / 'csrc'}", "-f", str(resolved_filelist),
            "-top", "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic",
            "-o", str(build / "simv"),
        ]
        need(compile_command.count("+vcs+initreg+random") == 1
             and not any(item.startswith("+vcs+initreg+")
                         and item != "+vcs+initreg+random"
                         for item in compile_command), "compile initreg surface")
        commands["timing_surface"] = validate_timing_surface(
            compile_command, [M2125.FILELIST, *sources])
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        M2125.run(compile_command, build,
                  clean_env({"VCS_HOME": str(M2125.VCS.parent.parent),
                             "VCS_ARCH_OVERRIDE": "linux"}),
                  21600, stage / "vcs_compile.log")
        need((build / "simv").is_file() and not (build / "simv").is_symlink(),
             "simv absent/symlink")

        commands["simv"] = {}
        for axis, cfg in M2125.AXES.items():
            axis_root = stage / axis
            axis_root.mkdir()
            saif = axis_root / "rtl_execute.saif"
            sim_command = [
                "./simv", "-lca", "+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                cfg["plusarg"], "-no_save", "-ucli", "-i",
                str(M2125.UCLI[axis]),
            ]
            plusargs = [item for item in sim_command if item.startswith("+")]
            need(plusargs == ["+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                              cfg["plusarg"]], "runtime plusarg surface")
            commands["simv"][axis] = sim_command
            counts["simv_runs"] += 1
            M2125.run(sim_command, build,
                clean_env({"VCS_HOME": str(M2125.VCS.parent.parent),
                           "VCS_ARCH_OVERRIDE": "linux",
                           "M2125_RTL_SAIF_FILE": str(saif)}),
                21600, axis_root / "rtl_sim.log")
            M2125.run([sys.executable, str(M2125.PARSER), "runtime",
                       "--axis", axis, "--path",
                       str(axis_root / "rtl_sim.log")],
                      REPO, clean_env({}), 60,
                      axis_root / "runtime_parse.log")
            M2125.run([sys.executable, str(M2125.PARSER), "saif",
                       "--axis", axis, "--path", str(saif)],
                      REPO, clean_env({}), 120,
                      axis_root / "saif_parse.log")
            counts["saif_files"] += 1

        need(counts == COUNTS, f"execution counts: {counts}")
        write_json(stage / "execution_commands.json", commands)
        M2125.run([sys.executable, str(M2125.PARSER), "final", "--root",
                   str(stage), "--output", str(stage / "result.json")],
                  REPO, clean_env({}), 180, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result["schema"] = "m2137_m2018_tsbg_rtl_saif_window_diagnostic_result_r1_v1"
        result["status"] = (
            "PASS_RAW_M2139_M2137_RTL_SAIF_DIAGNOSTIC_PENDING_M2140_RESULT_HAMMER")
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        result["execution_counts"] = counts
        result["identity"] = {
            "runner_sha256": sha256(RUNNER),
            "contract_sha256": sha256(CONTRACT),
            "m2138_review_sha256": sha256(REVIEW / "review.json"),
            "m2125_runner_sha256": sha256(M2125_RUNNER),
            "m2128_review_sha256": sha256(M2128 / "review.json"),
            "parser_sha256": sha256(M2125.PARSER),
            "docs359_sha256": sha256(DOC359),
            "vcs_sha256": sha256(M2125.VCS),
            "lmutil_sha256": sha256(M2125.LMUTIL),
        }
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2139_M2137_RTL_SAIF_DIAGNOSTIC_PENDING_M2140_RESULT_HAMMER\n")
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
            "status": "PASS_M2137_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "m2127_disposition": audit_m2127_disposition(),
            "option_aware_timing_surface": True,
        }, indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Failure, M2125.Failure) as exc:
        print(f"M2137_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
