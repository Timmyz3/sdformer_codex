#!/opt/anaconda3/bin/python3
"""One-shot M2113 matched RTL-SAIF -> DC saif_map -> PT-PX campaign.

Additive correction of the preflight-only M2105 source identity: use the real
non-symlink DC/PT executables and guard the actual snps_shell process name.
M2105 sources and its unconsumed M2107 predecessor identity remain immutable.

SOURCE ONLY.  With --static this performs no EDA.  Production refuses to run
until an exhaustive, double-sealed M2114 independent review authorizes exactly
one license query, one VCS compile, two simv runs, two DC runs, and two PT runs.
There is no automatic retry and no reuse of earlier SAIF/netlists/maps/power.
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
PARSER = HW / "system_simulator/scripts/parse_m2113_m2018_tsbg_rtl_saifmap_power.py"
CONTRACT = HW / "contracts/m2113_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
REVIEW = HW / "reviews/m2114_m2113_m2018_tsbg_rtl_saifmap_power_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
VCS_FILELIST = HW / "dc_handoff/filelists/tcasii_m2113_m2018_tsbg_rtl_saif_vcs.f"
DC_FILELIST = HW / "dc_handoff/filelists/tcasii_m2113_m2018_tsbg_saifmap_dc.f"
TB = HW / "tb_m2018/tb_m2113_m2018_tsbg_rtl_saifmap_power.sv"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m2113_m2018_tsbg_saifmap_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2113_m2018_tsbg_rtl_saifmap_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
UCLI = {
    "ordinary_lru4": HW / "dc_handoff/scripts/m2113_m2018_tsbg_ordinary_rtl_saif.ucli.tcl",
    "tsbg_b4": HW / "dc_handoff/scripts/m2113_m2018_tsbg_tsbg_rtl_saif.ucli.tcl",
}
AXES = {
    "ordinary_lru4": {"mode": 0, "cycles": 20292, "reads": 14304,
                      "plusarg": "+M2113_AXIS_ORDINARY"},
    "tsbg_b4": {"mode": 1, "cycles": 7569, "reads": 4608,
                "plusarg": "+M2113_AXIS_TSBG"},
}

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = "/opt/synopsys/Synopsys.dat"
SLOW_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")

RESULT = HW / "results/m2115_m2113_m2018_tsbg_rtl_saifmap_power_r1_20260904"
ATTEMPT = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_attempt_consumed"
LOCK = HW / "results/.m2115_m2113_tsbg_rtl_saifmap_power_launch_lock"
COUNTS = {"license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
          "dc_runs": 2, "ptpx_runs": 2, "saif_files": 2}


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha(path: Path) -> str:
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


def verify_seal(root: Path) -> dict[str, str]:
    need(root.is_dir() and not root.is_symlink(), f"review dir: {root}")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer review seal")
    rows = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe manifest")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == fields[0],
             f"manifest member: {rel}")
        rows[rel.as_posix()] = fields[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == set(rows), "non-exhaustive review seal")
    return rows


def seal_dir(root: Path) -> None:
    members = sorted(p for p in root.rglob("*") if p.is_file()
                     and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    with (root / "SHA256SUMS").open("w") as handle:
        for path in members:
            handle.write(f"{sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n")
    verify_seal(root)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
             "TMPDIR": "/tmp", "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": LICENSE_FILE}
    value.update(extra)
    return value


def source_validation(require_review: bool) -> dict:
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract outer seal")
    contract = strict_json(CONTRACT)
    need(contract.get("schema") ==
         "m2113_m2018_tsbg_rtl_saifmap_power_source_contract_r1_v1",
         "contract schema")
    need(contract.get("status") ==
         "SOURCE_ONLY__M2114_INDEPENDENT_REVIEW_REQUIRED__NO_EDA", "status")
    need(contract.get("execution_budget") == {
        **COUNTS, "automatic_retry": False, "p1_serial": True,
        "reuse_old_artifacts": False}, "execution budget")
    inventory = contract.get("source_inventory")
    need(isinstance(inventory, dict) and inventory, "source inventory")
    for rel, digest in inventory.items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"source identity: {rel}")
    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs/359 identity")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(str(review.get("status", "")).startswith("PASS_M2114"),
             "M2114 status")
        need(review.get("severity_counts") == {"p0": 0, "p1": 0, "p2": 0},
             "M2114 severities")
        need(review.get("score_over_100", 0) >= 95, "M2114 score")
        need(review.get("authorization") == {
            **COUNTS, "automatic_retry": False, "p1_serial": True,
            "reuse_old_artifacts": False}, "M2114 authorization")
        identity = review.get("identity", {})
        need(identity.get("runner_sha256") == sha(RUNNER), "review runner")
        need(identity.get("contract_sha256") == sha(CONTRACT), "review contract")
    return contract


def no_same_uid_eda() -> None:
    blocked = {"vcs", "simv", "snps_shell", "dc_shell", "pt_shell", "lmstat"}
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            names = {Path(v.decode(errors="replace")).name for v in
                     (proc / "cmdline").read_bytes().split(b"\0") if v}
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
    need(completed.returncode == 0, f"command failed ({completed.returncode}): {log}")


def validate_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    errors = [line for line in text.splitlines()
              if re.match(r"^(Error:|Fatal:)", line)]
    bootstrap = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
    errors = [line for line in errors if line != bootstrap]
    need(not errors, f"EDA log errors: {errors[:4]}")


def production() -> int:
    contract = source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "attempt/result/lock exists")
    for tool in (VCS, DC, PT, LMUTIL, SLOW_DB, FAST_DB, TT_DB):
        need(tool.is_file() and not tool.is_symlink(), f"tool/input absent: {tool}")
    no_same_uid_eda()
    LOCK.mkdir()
    ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {
        "status": "M2115_ATTEMPT_CONSUMED", "budget": COUNTS,
        "automatic_retry": False, "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2115_m2113_work.", dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2115_m2113_stage.", dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    try:
        counts["license_queries"] += 1
        run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER], work,
            clean_env({}), 60, stage / "license_preflight.log")

        build = work / "vcs_build"
        build.mkdir()
        resolved_filelist = build / "sources.absolute.f"
        source_lines = []
        for line in VCS_FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                source = (REPO / stripped).resolve()
                need(source.is_file() and not source.is_symlink(),
                     f"VCS source absent: {source}")
                source_lines.append(str(source))
        need(len(source_lines) == 6, "VCS filelist source count")
        resolved_filelist.write_text("\n".join(source_lines) + "\n")
        compile_log = stage / "vcs_compile.log"
        counts["vcs_compiles"] += 1
        run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
             "-debug_access+r", "-lca", "+vcs+lic+wait",
             f"-Mdir={build / 'csrc'}",
             "-f", str(resolved_filelist), "-top",
             "tb_m2113_m2018_tsbg_rtl_saifmap_power", "-o",
             str(build / "simv")],
            build, clean_env({"VCS_HOME": str(VCS.parent.parent),
                              "VCS_ARCH_OVERRIDE": "linux"}), 21600, compile_log)
        need((build / "simv").is_file(), "simv absent")
        validate_log(compile_log)

        for axis, cfg in AXES.items():
            axis_root = stage / axis
            dc_root, pt_root = axis_root / "dc", axis_root / "ptpx"
            axis_root.mkdir(); dc_root.mkdir(); pt_root.mkdir()
            saif = axis_root / "rtl_execute.saif"
            sim_log = axis_root / "rtl_sim.log"
            counts["simv_runs"] += 1
            run(["./simv", "-lca", "+WORKLOAD_SLOT=42", cfg["plusarg"],
                 "-no_save", "-ucli", "-i", str(UCLI[axis])], build,
                clean_env({"VCS_HOME": str(VCS.parent.parent),
                           "VCS_ARCH_OVERRIDE": "linux",
                           "M2113_RTL_SAIF_FILE": str(saif)}), 21600, sim_log)
            log_text = sim_log.read_text(errors="replace")
            need("M2113_RTL_SAIF_WINDOW_BEGIN" in log_text, "SAIF begin marker")
            need(f"M2113_RTL_SAIF_WINDOW_END axis={axis} " in log_text,
                 "SAIF end marker")
            need(not re.search(r"Fatal:|^Error:|Assertion failed", log_text,
                               flags=re.MULTILINE), "RTL sim failure token")
            run([sys.executable, str(PARSER), "saif", "--axis", axis,
                 "--path", str(saif)], REPO, clean_env({}), 60,
                axis_root / "saif_parse.log")
            counts["saif_files"] += 1

            counts["dc_runs"] += 1
            dc_log = dc_root / "dc.log"
            run([str(DC), "-f", str(DC_TCL)], REPO, clean_env({
                "M2113_SCHEDULE_MODE": str(cfg["mode"]),
                "M2113_HW_ROOT": str(HW), "M2113_RTL_FILELIST": str(DC_FILELIST),
                "M2113_LIB_DB": str(SLOW_DB), "M2113_MIN_LIB_DB": str(FAST_DB),
                "M2113_SDC_FILE": str(SDC), "M2113_OUTPUT_DIR": str(dc_root),
                "M2113_OPERATING_CONDITION": "ssg0p9v125c",
            }), 21600, dc_log)
            validate_log(dc_log)
            need((dc_root / "TCL_INTERNAL_COMPLETE.txt").is_file(), "DC terminal")
            run([sys.executable, str(PARSER), "maps", "--default",
                 str(dc_root / "netlist/m2018_axis.ptpx_map.default.tcl"),
                 "--essential",
                 str(dc_root / "netlist/m2018_axis.ptpx_map.essential.tcl"),
                 "--output", str(axis_root / "map_classification.json")],
                REPO, clean_env({}), 60, axis_root / "map_parse.log")
            identity = (dc_root / "reports/identity.rpt").read_text()
            match = re.findall(r"^design=(\S+)$", identity, flags=re.MULTILINE)
            need(len(match) == 1, "DC design identity")

            counts["ptpx_runs"] += 1
            pt_log = pt_root / "ptpx.log"
            run([str(PT), "-f", str(PT_TCL)], REPO, clean_env({
                "M2113_AXIS": axis, "M2113_DESIGN_NAME": match[0],
                "M2113_TT_LIB_DB": str(TT_DB),
                "M2113_MAPPED_NETLIST": str(dc_root / "netlist/m2018_axis_mapped.v"),
                "M2113_MAPPED_SDC": str(dc_root / "netlist/m2018_axis_mapped.sdc"),
                "M2113_DEFAULT_MAP": str(dc_root / "netlist/m2018_axis.ptpx_map.default.tcl"),
                "M2113_ESSENTIAL_MAP": str(dc_root / "netlist/m2018_axis.ptpx_map.essential.tcl"),
                "M2113_RTL_SAIF": str(saif), "M2113_OUTPUT_DIR": str(pt_root),
            }), 21600, pt_log)
            validate_log(pt_log)

        need(counts == COUNTS, f"execution counts: {counts}")
        run([sys.executable, str(PARSER), "final", "--root", str(stage),
             "--output", str(stage / "result.json")], REPO, clean_env({}),
            120, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        result["execution_counts"] = counts
        result["identity"] = {
            "runner_sha256": sha(RUNNER), "parser_sha256": sha(PARSER),
            "contract_sha256": sha(CONTRACT),
            "m2114_review_sha256": sha(REVIEW / "review.json"),
            "docs359_sha256": sha(DOC359),
        }
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2115_M2113_MATCHED_RTL_SAIFMAP_PTPX_PENDING_INDEPENDENT_RESULT_HAMMER\n")
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
        seal_dir(stage)
        quarantine = Path(str(RESULT) + f".failed.{os.getpid()}.quarantine")
        if not quarantine.exists():
            os.rename(stage, quarantine)
        if LOCK.exists():
            LOCK.rmdir()
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    args = ap.parse_args()
    if args.static:
        contract = source_validation(require_review=False)
        print(json.dumps({"status": "PASS_M2113_STATIC_RUNNER",
                          "source_count": len(contract["source_inventory"]),
                          "execution_budget": contract["execution_budget"]},
                         indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print(f"M2113_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
