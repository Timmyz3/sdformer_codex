#!/opt/anaconda3/bin/python3
"""One-shot ordinary native-SAIF acquisition preflight authorized by M2140.

SOURCE ONLY until an exhaustive, double-sealed M2143 independent source
hammer authorizes one future M2144 attempt.  M2144 may consume one license
query, one VCS compile, one ordinary simulation, and at most one raw/admitted
SAIF.  It has no TSBG simulation, DC, PT/PTPX, ICC2, GPU, or retry path.
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
PARSER = HW / "system_simulator/scripts/parse_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.py"
CONTRACT = HW / "contracts/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2143_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.ucli.tcl"
TB = HW / "tb_m2018/tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.sv"
TEST = HW / "tests/test_m2142_tsbg_ordinary_late_enable_saif_preflight.py"

M2125_RUNNER = HW / "dc_handoff/scripts/run_m2125_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py"
M2125_RUNNER_SHA256 = "6021c4a9b4297e5527f09006f21dd3a06d98b2a7ad76ffc55ca259029e658815"
M2140 = HW / "reviews/m2140_m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_failure_hammer_r1_20260904"
M2140_REVIEW_SHA256 = "5612255e46cb6d8017c84049aa1ebb2202f04cb1fe5ca181a3d974425bfb6ff8"
M2140_MANIFEST_SHA256 = "8f315963e4aede2ef2135cb2c766841b87db090dac4c381eb2a8677865ec99d2"
M2140_OUTER_SHA256 = "f690041cfb31564ea8d714480aa44ed8d496812dd60507f87e83ac030be1762e"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = HW / "results/m2144_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2144_m2142_tsbg_ordinary_late_enable_saif_preflight_attempt_consumed"
LOCK = HW / "results/.m2144_m2142_tsbg_ordinary_late_enable_saif_preflight_launch_lock"
COUNTS = {
    "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
    "raw_saif_files_written": 1, "admitted_saif_files": 1,
    "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0,
}

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
VCS_SHA256 = "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LMUTIL_SHA256 = "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07"
LICENSE_SERVER = "27030@ic.ismd-nemo"


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def load_m2125():
    need(M2125_RUNNER.is_file() and not M2125_RUNNER.is_symlink(),
         "M2125 runner absent/symlink")
    need(_raw_sha256(M2125_RUNNER) == M2125_RUNNER_SHA256,
         "M2125 runner identity")
    spec = importlib.util.spec_from_file_location("m2142_frozen_m2125", M2125_RUNNER)
    need(spec is not None and spec.loader is not None, "M2125 import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def validate_regular_tool(path: Path, digest: str, name: str) -> dict[str, str]:
    need(path.is_file() and not path.is_symlink(), f"{name} absent/symlink")
    need(sha256(path) == digest, f"{name} digest drift")
    return {"path": str(path), "sha256": digest, "type": "regular_file"}


def validate_timing_surface(command: list[str], active_inputs: list[Path]) -> dict[str, object]:
    explicit_sdf = [token for token in command
                    if token.lower().startswith(("-sdf", "+sdf"))]
    unit_delay = []
    for token in command:
        if token.lower().startswith("+define+"):
            definitions = token[len("+define+"):].split("+")
            if any(item.split("=", 1)[0].upper() == "UNIT_DELAY"
                   for item in definitions):
                unit_delay.append(token)
    need(not explicit_sdf, f"explicit SDF option: {explicit_sdf}")
    need(not unit_delay, f"explicit UNIT_DELAY define: {unit_delay}")
    for path in active_inputs:
        need(path.is_file() and not path.is_symlink(), f"active input: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        need(not re.search(r"\$sdf_annotate\b", text, flags=re.IGNORECASE),
             f"source-level SDF annotation: {path}")
        need("UNIT_DELAY" not in text, f"source-level UNIT_DELAY: {path}")
    return {
        "explicit_sdf_options": 0, "explicit_unit_delay_defines": 0,
        "active_input_count": len(active_inputs),
        "path_operands_may_contain_sdf_substring": True,
    }


def audit_m2140() -> dict[str, object]:
    members = verify_seal(M2140, M2140_MANIFEST_SHA256, M2140_OUTER_SHA256)
    review_path = M2140 / "review.json"
    need(sha256(review_path) == M2140_REVIEW_SHA256, "M2140 review identity")
    review = strict_json(review_path)
    need(review.get("status") ==
         "PASS_M2140_M2139_FAILURE_HAMMER__CONSUMED_NO_RETRY__LATE_ENABLE_OBSERVER_GAP__SOURCE_ONLY",
         "M2140 status")
    need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M2140 severity counts")
    disposition = review.get("m2139_disposition", {})
    need(disposition.get("attempt_consumed") is True
         and disposition.get("retry_authorized") is False
         and disposition.get("paper_citable") is False
         and disposition.get("tsbg_axis_executed") is False,
         "M2139 disposition")
    successor = review.get("only_allowed_successor", {})
    need(successor.get("authorization_now") == "SOURCE_AUTHORING_ONLY"
         and successor.get("direct_vcs_or_eda_execution_authorized_now") is False
         and successor.get("m2139_retry_allowed") is False
         and "ordinary-axis native-SAIF acquisition preflight" in
             successor.get("scope", ""), "M2140 successor boundary")
    return {
        "review_sha256": sha256(review_path),
        "manifest_sha256": sha256(M2140 / "SHA256SUMS"),
        "outer_sha256": sha256(M2140 / "SHA256SUMS.seal.sha256"),
        "member_count": len(members),
        "m2139_attempt_consumed": True,
        "m2139_retry_authorized": False,
        "m2139_paper_citable": False,
        "tsbg_axis_executed": False,
        "only_safe_next": "ordinary_late_enable_native_saif_preflight",
    }


def source_validation(require_review: bool) -> dict:
    # Retain all frozen M2125 RTL/workload/tool identities and independently
    # bind the newer M2140 causal disposition.
    old_contract = M2125.source_validation(require_review=False)
    need(isinstance(old_contract.get("source_inventory"), dict),
         "M2125 source inventory")
    validate_regular_tool(VCS, VCS_SHA256, "VCS")
    validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")
    disposition = audit_m2140()
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2143_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "ordinary_only": True,
        "reuse_old_artifacts": False}, "execution budget")
    need(contract.get("m2140_disposition") == disposition,
         "M2140 disposition fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    need(sha256(DOC359) == DOC359_SHA256, "docs359 identity")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2143"),
             "M2143 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2143 severity counts")
        need(review.get("score_over_100", 0) >= 95, "M2143 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "ordinary_only": True,
            "reuse_old_artifacts": False}, "M2143 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha256(RUNNER),
             "review runner")
        need(identity.get("contract_sha256") == sha256(CONTRACT),
             "review contract")
    return contract


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    M2125.no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2144_ATTEMPT_CONSUMED", "budget": COUNTS,
        "automatic_retry": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2144_m2142_work.",
                                 dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2144_m2142_stage.",
                                  dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    commands: dict[str, object] = {}
    try:
        counts["license_queries"] += 1
        license_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        commands["license_preflight"] = license_command
        M2125.run(license_command, work, clean_env({}), 60,
                  stage / "license_preflight.log")

        build = work / "vcs_build"
        build.mkdir()
        resolved_filelist = build / "sources.absolute.f"
        sources: list[Path] = []
        for line in FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                source = (REPO / stripped).resolve()
                need(source.is_file() and not source.is_symlink(),
                     f"VCS source absent: {source}")
                sources.append(source)
        need(len(sources) == 6 and sources[-1] == TB.resolve(),
             "VCS source count/order")
        resolved_filelist.write_text("\n".join(map(str, sources)) + "\n")
        compile_command = [
            str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "+vcs+initreg+random",
            "-debug_access+r", "-lca", "+vcs+lic+wait",
            f"-Mdir={build / 'csrc'}", "-f", str(resolved_filelist),
            "-top", "tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight",
            "-o", str(build / "simv"),
        ]
        need(compile_command.count("+vcs+initreg+random") == 1
             and not any(item.startswith("+vcs+initreg+")
                         and item != "+vcs+initreg+random"
                         for item in compile_command), "compile initreg surface")
        commands["timing_surface"] = validate_timing_surface(
            compile_command, [FILELIST, *sources])
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        M2125.run(compile_command, build,
                  clean_env({"VCS_HOME": str(VCS.parent.parent),
                             "VCS_ARCH_OVERRIDE": "linux"}),
                  21600, stage / "vcs_compile.log")
        need((build / "simv").is_file() and not (build / "simv").is_symlink(),
             "simv absent/symlink")

        saif = stage / "rtl_execute.saif"
        sim_command = [
            "./simv", "-lca", "+vcs+initreg+0", "+WORKLOAD_SLOT=42",
            "+M2142_AXIS_ORDINARY", "-no_save", "-ucli", "-i", str(UCLI),
        ]
        plusargs = [item for item in sim_command if item.startswith("+")]
        need(plusargs == ["+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                          "+M2142_AXIS_ORDINARY"], "runtime plusarg surface")
        commands["simv"] = sim_command
        counts["simv_runs"] += 1
        M2125.run(sim_command, build,
                  clean_env({"VCS_HOME": str(VCS.parent.parent),
                             "VCS_ARCH_OVERRIDE": "linux",
                             "M2142_RTL_SAIF_FILE": str(saif)}),
                  21600, stage / "rtl_sim.log")
        need(saif.is_file() and not saif.is_symlink() and saif.stat().st_size > 0,
             "raw SAIF absent/empty/symlink")
        counts["raw_saif_files_written"] += 1
        M2125.run([sys.executable, str(PARSER), "runtime", "--path",
                   str(stage / "rtl_sim.log")], REPO, clean_env({}), 60,
                  stage / "runtime_parse.log")
        M2125.run([sys.executable, str(PARSER), "saif", "--path", str(saif)],
                  REPO, clean_env({}), 120, stage / "saif_parse.log")
        counts["admitted_saif_files"] += 1
        need(counts == COUNTS, f"execution counts: {counts}")
        write_json(stage / "execution_commands.json", commands)
        M2125.run([sys.executable, str(PARSER), "final", "--root", str(stage),
                   "--output", str(stage / "result.json")],
                  REPO, clean_env({}), 180, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        result["execution_counts"] = counts
        result["identity"] = {
            "runner_sha256": sha256(RUNNER),
            "parser_sha256": sha256(PARSER),
            "contract_sha256": sha256(CONTRACT),
            "m2143_review_sha256": sha256(REVIEW / "review.json"),
            "m2140_review_sha256": M2140_REVIEW_SHA256,
            "docs359_sha256": sha256(DOC359),
            "vcs_sha256": sha256(VCS), "lmutil_sha256": sha256(LMUTIL),
        }
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2144_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT_PENDING_M2145_RESULT_HAMMER\n")
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
            "status": "PASS_M2142_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "m2140_disposition": audit_m2140(),
            "tools": {
                "vcs": validate_regular_tool(VCS, VCS_SHA256, "VCS"),
                "lmutil": validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil"),
            },
        }, indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2142_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
