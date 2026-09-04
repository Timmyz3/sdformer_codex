#!/opt/anaconda3/bin/python3
"""One-shot runner for M2160's report-before-reset native-SAIF preflight.

SOURCE ONLY until a double-sealed M2161 independent source hammer authorizes
one fresh M2162 identity.  M2162 permits one license query, one VCS compile,
one ordinary simv, one diagnostic prehistory SAIF, and one candidate
measurement SAIF.  It has no second-axis,
DC, PT/PTPX, ICC2, GPU, old-artifact reuse, or retry path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
REPO = HW.parent
RUNNER = Path(__file__).resolve()
PARSER = HW / "system_simulator/scripts/parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py"
CONTRACT = HW / "contracts/m2160_m2018_ordinary_native_saif_report_reset_preflight_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2161_m2160_m2018_ordinary_native_saif_report_reset_preflight_source_hammer_r1_20260904"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2160_m2018_ordinary_native_saif_report_reset_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
TB = HW / "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"
TEST = HW / "tests/test_m2160_ordinary_native_saif_report_reset_preflight.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2152 = HW / "reviews/m2152_m2151_m2149_ordinary_single_axis_native_saif_preflight_failure_hammer_r1_20260904"

M2152_REVIEW_SHA256 = "dcd39864072078f0fa2d3a25151d76e84c14fe0e3fea0bcf2f44201f17f1b4da"
M2152_MANIFEST_SHA256 = "741fa6b7e3027ab78e127aa3281e365f4c57834a7e521cd9bc779f5abd29cf25"
M2152_OUTER_SHA256 = "022683cc8db3c4ca21b786382383c21237f06c7665bb34e01df4e604d4bb9a64"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

RESULT = HW / "results/m2162_m2160_m2018_ordinary_native_saif_report_reset_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2162_m2160_ordinary_native_saif_report_reset_preflight_attempt_consumed"
LOCK = HW / "results/.m2162_m2160_ordinary_native_saif_report_reset_preflight_launch_lock"
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


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f"missing/symlink JSON: {path}")
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict[str, str]:
    need(root.is_dir() and not root.is_symlink(), f"sealed root: {root}")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), f"missing seal: {root}")
    if expected_manifest is not None:
        need(sha256(manifest) == expected_manifest, "manifest identity")
    if expected_outer is not None:
        need(sha256(outer) == expected_outer, "outer-seal identity")
    outer_fields = outer.read_text().split()
    need(outer_fields == [sha256(manifest), "SHA256SUMS"], "outer seal")
    members: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, f"malformed manifest row: {line}")
        digest, rel = fields
        path = root / rel
        need(path.is_file() and not path.is_symlink(), f"manifest member: {rel}")
        need(sha256(path) == digest, f"manifest digest: {rel}")
        members[rel] = digest
    actual = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(sorted(members) == actual, "non-exhaustive sealed directory")
    return members


def seal_dir(root: Path) -> None:
    members = sorted(path for path in root.rglob("*") if path.is_file()
                     and path.name not in
                     {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(path)}  {path.relative_to(root)}\n" for path in members))
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n")


def seal_file(path: Path) -> dict[str, str]:
    need(path.is_file() and not path.is_symlink(), f"raw file: {path}")
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{sha256(path)}  {path.name}\n")
    outer = Path(str(sidecar) + ".seal.sha256")
    outer.write_text(f"{sha256(sidecar)}  {sidecar.name}\n")
    return {"sha256": sha256(path), "sidecar_sha256": sha256(sidecar),
            "outer_sha256": sha256(outer)}


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    allowed = {
        "PATH", "LD_LIBRARY_PATH", "LM_LICENSE_FILE", "SNPSLMD_LICENSE_FILE",
        "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "SHELL",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(extra)
    return env


def run(command: list[str], cwd: Path, env: dict[str, str], timeout: int,
        log: Path) -> None:
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, timeout=timeout,
                                text=True, check=False)
    need(result.returncode == 0,
         f"command failed rc={result.returncode}: {command[0]}")


def validate_regular_tool(path: Path, digest: str, name: str) -> dict[str, str]:
    need(path.is_file() and not path.is_symlink(), f"{name} absent/symlink")
    need(sha256(path) == digest, f"{name} digest drift")
    return {"path": str(path), "sha256": digest, "type": "regular_file"}


def audit_m2152() -> dict[str, object]:
    members = verify_seal(M2152, M2152_MANIFEST_SHA256, M2152_OUTER_SHA256)
    review_path = M2152 / "review.json"
    need(sha256(review_path) == M2152_REVIEW_SHA256, "M2152 review identity")
    review = strict_json(review_path)
    need(review.get("status") ==
         "PASS_M2152_M2151_FAILURE_HAMMER__CONSUMED_NO_RETRY__RESET_IGNORED_AND_HEADER_ONLY_SAIF__SOURCE_ONLY",
         "M2152 failure-hammer status")
    need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
         "M2152 severity")
    disposition = review.get("m2151_disposition", {})
    need(disposition.get("attempt_consumed") is True
         and disposition.get("automatic_retry") is False
         and disposition.get("retry_authorized") is False
         and disposition.get("canonical_result_exists") is False,
         "M2151 consumed-failure disposition")
    successor = review.get("only_allowed_successor", {})
    need(successor.get("authorization_now") == "SOURCE_AUTHORING_ONLY"
         and successor.get("direct_vcs_or_eda_execution_authorized_now") is False
         and successor.get("m2151_retry_allowed") is False,
         "M2152 successor authority")
    chain = successor.get("proposed_fresh_identity_chain", {})
    need(chain == {"source": "M2160", "independent_source_hammer": "M2161",
                   "one_shot_ordinary_preflight": "M2162",
                   "independent_result_hammer": "M2163"},
         "M2152 successor identity chain")
    return {
        "review_sha256": sha256(review_path),
        "manifest_sha256": sha256(M2152 / "SHA256SUMS"),
        "outer_sha256": sha256(M2152 / "SHA256SUMS.seal.sha256"),
        "member_count": len(members),
        "m2151_attempt_consumed": True,
        "m2151_retry_authorized": False,
        "m2151_canonical_result_exists": False,
        "successor_authority": "SOURCE_AUTHORING_ONLY",
    }


def topology_audit() -> dict[str, object]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("m2160_parser", PARSER)
    need(spec is not None and spec.loader is not None, "parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_single_axis_source(TB.read_text(), FILELIST.read_text())


def source_validation(require_review: bool) -> dict:
    audit_m2152()
    validate_regular_tool(VCS, VCS_SHA256, "VCS")
    validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")
    need(sha256(DOC359) == DOC359_SHA256, "docs359 identity")
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2160_m2018_ordinary_native_saif_report_reset_preflight_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2161_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "ordinary_only": True,
        "single_frontend": True, "reuse_old_artifacts": False},
        "execution budget")
    need(contract.get("m2152_disposition") == audit_m2152(),
         "M2152 disposition fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    topology = topology_audit()
    need(topology == contract.get("single_axis_topology"), "topology fingerprint")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2161"),
             "M2161 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2161 severity")
        need(review.get("score_over_100", 0) >= 95, "M2161 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "ordinary_only": True,
            "single_frontend": True, "reuse_old_artifacts": False},
            "M2161 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha256(RUNNER),
             "M2161 runner identity")
        need(identity.get("contract_sha256") == sha256(CONTRACT),
             "M2161 contract identity")
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
        if re.search(r"(^|/|\s)(vcs|simv|dc_shell|pt_shell|icc2_shell)(\s|$)",
                     cmdline):
            offenders.append(f"{entry.name}:{cmdline[:240]}")
    need(not offenders, f"same-UID EDA active: {offenders}")


def validate_timing_surface(command: list[str], active_inputs: list[Path]) -> dict[str, object]:
    need(not any(token.lower().startswith(("-sdf", "+sdf")) for token in command),
         "explicit SDF option")
    need(not any(token.lower().startswith("+define+") and
                 "UNIT_DELAY" in token.upper() for token in command),
         "explicit UNIT_DELAY define")
    for path in active_inputs:
        text = path.read_text(encoding="utf-8", errors="replace")
        need(not re.search(r"\$sdf_annotate\b", text, re.IGNORECASE),
             f"source-level SDF: {path}")
        need("UNIT_DELAY" not in text, f"source-level UNIT_DELAY: {path}")
    return {"explicit_sdf_options": 0, "explicit_unit_delay_defines": 0,
            "active_input_count": len(active_inputs)}


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2162_ATTEMPT_CONSUMED", "budget": COUNTS,
        "automatic_retry": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2162_m2160_work.",
                                 dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2162_m2160_stage.",
                                  dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    commands: dict[str, object] = {}
    try:
        counts["license_queries"] += 1
        license_command = [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER]
        commands["license_preflight"] = license_command
        run(license_command, work, clean_env({}), 60,
            stage / "license_preflight.log")

        build = work / "vcs_build"
        build.mkdir()
        sources: list[Path] = []
        for line in FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                source = (REPO / stripped).resolve()
                need(source.is_file() and not source.is_symlink(),
                     f"VCS source absent: {source}")
                sources.append(source)
        need(len(sources) == 4 and sources[-1] == TB.resolve(),
             "VCS source count/order")
        resolved = build / "sources.absolute.f"
        resolved.write_text("\n".join(map(str, sources)) + "\n")
        compile_command = [
            str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "+vcs+initreg+random",
            "-debug_access+r", "-lca", "+vcs+lic+wait",
            f"-Mdir={build / 'csrc'}", "-f", str(resolved),
            "-top", "tb_m2160_m2018_ordinary_native_saif_report_reset_preflight",
            "-o", str(build / "simv"),
        ]
        need(compile_command.count("+vcs+initreg+random") == 1,
             "compile initreg surface")
        commands["timing_surface"] = validate_timing_surface(
            compile_command, [FILELIST, *sources])
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        run(compile_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent),
                       "VCS_ARCH_OVERRIDE": "linux"}),
            21600, stage / "vcs_compile.log")
        need((build / "simv").is_file()
             and not (build / "simv").is_symlink(), "simv absent/symlink")

        prehistory_saif = stage / "rtl_prehistory.saif"
        measurement_saif = stage / "rtl_measurement.saif"
        sim_command = [
            "./simv", "-lca", "+vcs+initreg+0", "+WORKLOAD_SLOT=42",
            "+M2160_AXIS_ORDINARY", "-no_save", "-ucli", "-i", str(UCLI),
        ]
        plusargs = [item for item in sim_command if item.startswith("+")]
        need(plusargs == ["+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                          "+M2160_AXIS_ORDINARY"], "runtime plusarg surface")
        commands["simv"] = sim_command
        counts["simv_runs"] += 1
        run(sim_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent),
                       "VCS_ARCH_OVERRIDE": "linux",
                       "M2160_PREHISTORY_SAIF_FILE": str(prehistory_saif),
                       "M2160_MEASUREMENT_SAIF_FILE": str(measurement_saif)}),
            21600, stage / "rtl_sim.log")
        for path in (prehistory_saif, measurement_saif):
            need(path.is_file() and not path.is_symlink()
                 and path.stat().st_size > 0,
                 f"raw SAIF absent/empty/symlink: {path.name}")
            seal_file(path)
            counts["raw_saif_files_written"] += 1
        counts["diagnostic_saif_files_written"] += 1
        run([sys.executable, str(PARSER), "runtime", "--path",
             str(stage / "rtl_sim.log")], REPO, clean_env({}), 60,
            stage / "runtime_parse.log")
        run([sys.executable, str(PARSER), "saif", "--path",
             str(prehistory_saif), "--role", "diagnostic_prehistory"],
            REPO, clean_env({}), 120, stage / "prehistory_saif_parse.log")
        run([sys.executable, str(PARSER), "saif", "--path",
             str(measurement_saif), "--role", "measurement"],
            REPO, clean_env({}), 120, stage / "saif_parse.log")
        counts["admitted_measurement_saif_files"] += 1
        counts["admitted_saif_files"] += 1
        need(counts == COUNTS, f"execution counts: {counts}")
        write_json(stage / "execution_commands.json", commands)
        run([sys.executable, str(PARSER), "final", "--root", str(stage),
             "--output", str(stage / "result.json")],
            REPO, clean_env({}), 180, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        result["execution_counts"] = counts
        result["identity"] = {
            "runner_sha256": sha256(RUNNER),
            "parser_sha256": sha256(PARSER),
            "contract_sha256": sha256(CONTRACT),
            "m2161_review_sha256": sha256(REVIEW / "review.json"),
            "m2152_review_sha256": M2152_REVIEW_SHA256,
            "docs359_sha256": sha256(DOC359),
            "vcs_sha256": sha256(VCS), "lmutil_sha256": sha256(LMUTIL),
        }
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2162_M2160_REPORT_RESET_NATIVE_SAIF_PREFLIGHT_PENDING_M2163_RESULT_HAMMER\n")
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
            "status": "PASS_M2160_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "single_axis_topology": topology_audit(),
            "m2152_disposition": audit_m2152(),
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
        print(f"M2160_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
