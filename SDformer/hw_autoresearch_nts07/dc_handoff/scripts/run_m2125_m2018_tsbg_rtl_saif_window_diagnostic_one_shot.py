#!/opt/anaconda3/bin/python3
"""One-shot M2125 VCS-only RTL-SAIF window diagnostic.

SOURCE ONLY until an exhaustive double-sealed M2126 independent source review
authorizes one M2127 attempt: one license query, one shared VCS compile, two
strictly serial simv runs and two fresh DUT-only SAIF files.  No DC, PT, PTPX,
SDF, mapped netlist, reuse, or automatic retry exists in this runner.  M2119
is a permanently consumed/failed predecessor and is only audited read-only.
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
PARSER = HW / "system_simulator/scripts/parse_m2125_m2018_tsbg_rtl_saif_window_diagnostic.py"
CONTRACT = HW / "contracts/m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2125_m2018_tsbg_rtl_saif_window_diagnostic_vcs.f"
UCLI = {
    "ordinary_lru4": HW / "dc_handoff/scripts/m2125_m2018_tsbg_ordinary_rtl_saif_window_diagnostic.ucli.tcl",
    "tsbg_b4": HW / "dc_handoff/scripts/m2125_m2018_tsbg_tsbg_rtl_saif_window_diagnostic.ucli.tcl",
}
AXES = {
    "ordinary_lru4": {"plusarg": "+M2125_AXIS_ORDINARY", "cycles": 20292,
                      "reads": 14304},
    "tsbg_b4": {"plusarg": "+M2125_AXIS_TSBG", "cycles": 7569,
                "reads": 4608},
}

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
VCS_SHA256 = "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LMUTIL_SHA256 = "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07"
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = "/opt/synopsys/Synopsys.dat"

PREDECESSOR = HW / "results/m2119_m2117_m2018_tsbg_rtl_saifmap_power_r1_20260904.failed.1771297.quarantine"
PREDECESSOR_MANIFEST_SHA256 = "e9892b47d4ad6c847d8fdb92c724bf0c6a98d7906f291abfd219563816d8a778"
PREDECESSOR_OUTER_SHA256 = "6cda58321724db9ae3682a50abf7fbc03c2d2aecf7d56574740d908357ceafe2"
PREDECESSOR_SAIF_SHA256 = "662f148e68371a7e39df3dedabb10897c1014dc7a2237f49038fdf1689f0733a"
M2120_REVIEW = HW / "reviews/m2120_m2119_m2117_tsbg_saifmap_power_failure_hammer_r1_20260904"
M2120_MANIFEST_SHA256 = "51e85bb9393b89b1e694b390e0574204470b32236b1541e639b4d201a9bc5114"
M2120_OUTER_SHA256 = "4c683dcb5250346e2bd6b9445ad4506b9cf0c486b2c8457c1f6d934590bdefbc"
M2120_REVIEW_JSON_SHA256 = "9457fc09d62198618d54669347aa645811c82e797e0ee1f0340940e6af881dec"

RESULT = HW / "results/m2127_m2125_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
ATTEMPT = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
LOCK = HW / "results/.m2127_m2125_tsbg_rtl_saif_window_diagnostic_launch_lock"
COUNTS = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
          "saif_files": 2, "dc_runs": 0, "ptpx_runs": 0}


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
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure(f"nonfinite JSON: {token}")))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, expected_manifest: str | None = None,
                expected_outer: str | None = None) -> dict[str, str]:
    need(root.is_dir() and not root.is_symlink(), f"sealed dir: {root}")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), "manifest absent")
    need(outer.is_file() and not outer.is_symlink(), "outer seal absent")
    if expected_manifest is not None:
        need(sha256(manifest) == expected_manifest, "manifest identity drift")
    if expected_outer is not None:
        need(sha256(outer) == expected_outer, "outer identity drift")
    need(outer.read_text().split() == [sha256(manifest), "SHA256SUMS"],
         "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe manifest")
        path = root / rel
        need(path.is_file() and not path.is_symlink(), f"manifest member: {rel}")
        need(sha256(path) == fields[0], f"member digest: {rel}")
        need(rel.as_posix() not in rows, f"duplicate manifest member: {rel}")
        rows[rel.as_posix()] = fields[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == set(rows), "non-exhaustive sealed directory")
    return rows


def seal_dir(root: Path) -> None:
    members = sorted(p for p in root.rglob("*") if p.is_file()
                     and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    with (root / "SHA256SUMS").open("w") as handle:
        for path in members:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(root / 'SHA256SUMS')}  SHA256SUMS\n")
    verify_seal(root)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
             "TMPDIR": "/tmp", "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": LICENSE_FILE}
    value.update(extra)
    return value


def audit_predecessor() -> dict[str, object]:
    members = verify_seal(PREDECESSOR, PREDECESSOR_MANIFEST_SHA256,
                          PREDECESSOR_OUTER_SHA256)
    expected_members = {
        "FAILED_DO_NOT_CITE.txt", "execution_counts.json",
        "license_preflight.log", "ordinary_lru4/rtl_execute.saif",
        "ordinary_lru4/rtl_sim.log", "ordinary_lru4/saif_parse.log",
        "vcs_compile.log",
    }
    need(set(members) == expected_members, "predecessor member set drift")
    failure = (PREDECESSOR / "FAILED_DO_NOT_CITE.txt").read_text()
    need("status=FAILED_DO_NOT_CITE" in failure
         and "command failed (2)" in failure
         and "automatic_retry=false" in failure,
         "predecessor failure boundary drift")
    counts = strict_json(PREDECESSOR / "execution_counts.json")
    need(counts == {"dc_runs": 0, "license_queries": 1, "ptpx_runs": 0,
                    "saif_files": 0, "simv_runs": 1, "vcs_compiles": 1},
         "predecessor execution count drift")
    saif = PREDECESSOR / "ordinary_lru4/rtl_execute.saif"
    need(sha256(saif) == PREDECESSOR_SAIF_SHA256, "predecessor SAIF drift")
    text = saif.read_text(errors="replace")
    duration = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", text)
    records = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
    need(duration == ["60877.50"] and len(records) == 93971,
         "predecessor duration/record fingerprint drift")
    need(sum(float(row[2]) > 0.0 for row in records) == 58277,
         "predecessor TX fingerprint drift")
    verify_seal(M2120_REVIEW, M2120_MANIFEST_SHA256, M2120_OUTER_SHA256)
    review_path = M2120_REVIEW / "review.json"
    need(sha256(review_path) == M2120_REVIEW_JSON_SHA256,
         "M2120 review identity drift")
    review = strict_json(review_path)
    need(review.get("status") ==
         "PASS_M2120_M2119_FAILURE_HAMMER__M2119_CONSUMED_NO_POWER__M2125_SOURCE_AUTHORING_ONLY_ALLOWED",
         "M2120 status drift")
    need(review.get("only_allowed_successor", {}).get("source_identity") == "M2125"
         and review.get("only_allowed_successor", {}).get(
             "direct_vcs_execution_authorized_now") is False,
         "M2120 successor boundary drift")
    return {
        "path": str(PREDECESSOR.relative_to(REPO)),
        "manifest_sha256": sha256(PREDECESSOR / "SHA256SUMS"),
        "outer_sha256": sha256(PREDECESSOR / "SHA256SUMS.seal.sha256"),
        "member_count": len(members),
        "ordinary_saif_duration_ns": 60877.5,
        "ordinary_saif_record_count": 93971,
        "ordinary_saif_tx_nonzero_record_count": 58277,
        "m2120_review_json_sha256": M2120_REVIEW_JSON_SHA256,
        "m2120_review_outer_sha256": M2120_OUTER_SHA256,
        "permanently_failed_do_not_retry": True,
    }


def validate_regular_tool(path: Path, digest: str, name: str) -> dict[str, str]:
    need(path.is_file() and not path.is_symlink(), f"{name} absent/symlink")
    need(sha256(path) == digest, f"{name} digest drift")
    return {"path": str(path), "sha256": digest, "type": "regular_file"}


def source_validation(require_review: bool) -> dict:
    validate_regular_tool(VCS, VCS_SHA256, "VCS")
    validate_regular_tool(LMUTIL, LMUTIL_SHA256, "lmutil")
    predecessor = audit_predecessor()
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha256(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2126_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "p1_serial": True,
        "reuse_old_artifacts": False}, "execution budget")
    need(contract.get("predecessor_fingerprint") == predecessor,
         "predecessor contract fingerprint")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha256(path) == digest,
             f"source identity: {rel}")
    need(sha256(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs/359 identity")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2126"),
             "M2126 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2126 severities")
        need(review.get("score_over_100", 0) >= 95, "M2126 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "p1_serial": True,
            "reuse_old_artifacts": False}, "M2126 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha256(RUNNER), "review runner")
        need(identity.get("contract_sha256") == sha256(CONTRACT), "review contract")
    return contract


def no_same_uid_eda() -> None:
    blocked = {"vcs", "simv", "snps_shell", "dc_shell", "common_shell_exec",
               "common_shell_exe", "pt_shell", "icc2_shell", "lmstat"}
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            names = {Path(value.decode(errors="replace")).name for value in
                     (proc / "cmdline").read_bytes().split(b"\0") if value}
            comm = (proc / "comm").read_text().strip()
        except (OSError, PermissionError):
            continue
        if comm in blocked or names & blocked:
            hits.append((proc.name, comm, sorted(names & blocked)))
    need(not hits, f"same-UID EDA collision: {hits}")


def run(command: list[str], cwd: Path, env: dict[str, str], timeout: int,
        log: Path) -> None:
    with log.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   check=False)
    need(completed.returncode == 0,
         f"command failed ({completed.returncode}): {log}")


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2127_ATTEMPT_CONSUMED",
        "budget": COUNTS,
        "automatic_retry": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2127_m2125_work.",
                                 dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2127_m2125_stage.",
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
        resolved_filelist = build / "sources.absolute.f"
        sources = []
        for line in FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                source = (REPO / stripped).resolve()
                need(source.is_file() and not source.is_symlink(),
                     f"VCS source absent: {source}")
                sources.append(str(source))
        need(len(sources) == 6, "VCS source count")
        resolved_filelist.write_text("\n".join(sources) + "\n")
        compile_command = [
            str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", "+vcs+initreg+random",
            "-debug_access+r", "-lca", "+vcs+lic+wait",
            f"-Mdir={build / 'csrc'}", "-f", str(resolved_filelist),
            "-top", "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic",
            "-o", str(build / "simv"),
        ]
        need(compile_command.count("+vcs+initreg+random") == 1
             and not any(item.startswith("+vcs+initreg+")
                         and item != "+vcs+initreg+random"
                         for item in compile_command),
             "compile initreg surface")
        need(not any("UNIT_DELAY" in item or "sdf" in item.lower()
                     for item in compile_command), "timing contamination")
        commands["vcs_compile"] = compile_command
        counts["vcs_compiles"] += 1
        run(compile_command, build,
            clean_env({"VCS_HOME": str(VCS.parent.parent),
                       "VCS_ARCH_OVERRIDE": "linux"}),
            21600, stage / "vcs_compile.log")
        need((build / "simv").is_file() and not (build / "simv").is_symlink(),
             "simv absent/symlink")

        commands["simv"] = {}
        for axis, cfg in AXES.items():
            axis_root = stage / axis
            axis_root.mkdir()
            saif = axis_root / "rtl_execute.saif"
            sim_command = [
                "./simv", "-lca", "+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                cfg["plusarg"], "-no_save", "-ucli", "-i", str(UCLI[axis]),
            ]
            plusargs = [item for item in sim_command if item.startswith("+")]
            need(plusargs == ["+vcs+initreg+0", "+WORKLOAD_SLOT=42",
                              cfg["plusarg"]], "runtime plusarg surface")
            commands["simv"][axis] = sim_command
            counts["simv_runs"] += 1
            run(sim_command, build,
                clean_env({"VCS_HOME": str(VCS.parent.parent),
                           "VCS_ARCH_OVERRIDE": "linux",
                           "M2125_RTL_SAIF_FILE": str(saif)}),
                21600, axis_root / "rtl_sim.log")
            run([sys.executable, str(PARSER), "runtime", "--axis", axis,
                 "--path", str(axis_root / "rtl_sim.log")],
                REPO, clean_env({}), 60, axis_root / "runtime_parse.log")
            run([sys.executable, str(PARSER), "saif", "--axis", axis,
                 "--path", str(saif)], REPO, clean_env({}), 120,
                axis_root / "saif_parse.log")
            counts["saif_files"] += 1

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
            "m2126_review_sha256": sha256(REVIEW / "review.json"),
            "docs359_sha256": sha256(DOC359),
            "vcs_sha256": sha256(VCS),
            "lmutil_sha256": sha256(LMUTIL),
            "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
            "predecessor_outer_sha256": PREDECESSOR_OUTER_SHA256,
            "m2120_review_json_sha256": M2120_REVIEW_JSON_SHA256,
            "m2120_review_outer_sha256": M2120_OUTER_SHA256,
        }
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2127_M2125_RTL_SAIF_DIAGNOSTIC_PENDING_M2128_RESULT_HAMMER\n")
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
            "status": "PASS_M2125_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "predecessor": audit_predecessor(),
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
        print(f"M2125_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
