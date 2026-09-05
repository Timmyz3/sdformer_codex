#!/opt/anaconda3/bin/python3
"""M2233 complete local parser-import closure for the matched TSBG power campaign.

No tool is run with --static. Production requires a double-sealed M2234
release and consumes one fresh M2235 identity. M2219 and M2227 are neither reused nor consumed. M2204 supplies methodology
only; the complete M2217 -> M2172 -> M2160 and M2217 -> M2117 local import
closure is pinned before any parser import or tool run.
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
PARSER = HW / "system_simulator/scripts/parse_m2217_ep34_tsbg_matched_power.py"
SELECTOR = HW / "system_simulator/scripts/select_m2217_ep34_tsbg_matched_power_windows.py"
SELECTION = HW / "tb_m2018/fixtures/m2217_ep34_tsbg_matched_power_windows.json"
CONTRACT = HW / "contracts/m2233_ep34_tsbg_matched_power_source_repair_contract_r1_20260905.json"
REVIEW = HW / "reviews/m2234_m2233_ep34_tsbg_matched_power_source_repair_hammer_r1_20260905"
M2204 = HW / "reviews/m2204_m2203_m2201_ordinary_native_saif_subtick_quantized_preflight_result_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
STRUCT_HELPER = HW / (
    "system_simulator/scripts/"
    "parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py")
POWER_HELPER = HW / (
    "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py")
BASE_HELPER = HW / (
    "system_simulator/scripts/"
    "parse_m2160_m2018_ordinary_native_saif_report_reset_preflight.py")
VCS_FILELIST = HW / "dc_handoff/filelists/tcasii_m2217_m2018_single_dut_native_saif_vcs.f"
DC_FILELIST = HW / "dc_handoff/filelists/tcasii_m2217_m2018_matched_power_dc.f"
UCLI = HW / "dc_handoff/scripts/m2217_m2018_single_dut_native_saif.ucli.tcl"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m2217_m2018_matched_power_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2217_m2018_matched_power_window.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
DC = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
SLOW_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = "/opt/synopsys/Synopsys.dat"
RESULT = HW / "results/m2235_m2233_ep34_tsbg_matched_power_repair_r1_20260905"
ATTEMPT = HW / "results/.m2235_m2233_ep34_tsbg_matched_power_repair_attempt_consumed"
LOCK = HW / "results/.m2235_m2233_ep34_tsbg_matched_power_repair_launch_lock"
AXES = {"ordinary_lru4": 0, "tsbg_b4": 1}
STRATA = ("low", "median", "high")
COUNTS = {"license_queries": 1, "vcs_compiles": 2, "simv_runs": 6,
          "diagnostic_saif_files": 6, "measurement_saif_files": 6,
          "dc_runs": 2, "ptpx_runs": 6}
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DC_TARGET_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"
STRUCT_HELPER_SHA = "42fd87d6991c46366e80db1d08c20ec5e0d463f3bca8c6050673093d04f3bfe2"
POWER_HELPER_SHA = "2787e8858799577db8f87297d2d1c1c16ccf0a3933b00f9a039071e968ea3547"
BASE_HELPER_SHA = "381fbaac6c75aed86aa1dd12aad41dffeb8348c7a875e95f1c162256df6ba22b"


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value: raise Failure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key: " + key); out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seal_file(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(side) + ".seal.sha256")
    side.write_text(f"{sha(path)}  {path.name}\n")
    outer.write_text(f"{sha(side)}  {side.name}\n")


def verify_seal(root: Path) -> dict[str, str]:
    need(root.is_dir() and not root.is_symlink(), "sealed directory")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); rel = Path(name.lstrip("*"))
        path = root / rel
        need(not rel.is_absolute() and ".." not in rel.parts and path.is_file()
             and not path.is_symlink() and sha(path) == digest, "seal member")
        rows[rel.as_posix()] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == set(rows), "non-exhaustive seal")
    return rows


def seal_dir(root: Path) -> None:
    members = sorted(path for path in root.rglob("*") if path.is_file()
                     and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (root / "SHA256SUMS").write_text("".join(
        f"{sha(path)}  {path.relative_to(root).as_posix()}\n" for path in members))
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n")
    verify_seal(root)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
           "TMPDIR": "/tmp", "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
           "LM_LICENSE_FILE": LICENSE_FILE}
    env.update(extra); return env


def validate_dc_launcher() -> None:
    need(DC.is_symlink() and os.readlink(DC) == "snps_shell", "DC launcher link")
    need(DC.resolve(strict=True) == DC_TARGET and DC_TARGET.is_file()
         and not DC_TARGET.is_symlink() and sha(DC_TARGET) == DC_TARGET_SHA,
         "DC launcher target")


def source_validation(require_review: bool) -> dict:
    need(sha(DOC359) == DOC_SHA, "docs359 identity")
    need(sha(STRUCT_HELPER) == STRUCT_HELPER_SHA, "M2172 helper identity")
    need(sha(POWER_HELPER) == POWER_HELPER_SHA, "M2117 helper identity")
    need(sha(BASE_HELPER) == BASE_HELPER_SHA, "M2160 helper identity")
    validate_dc_launcher()
    need(CONTRACT.is_file() and not CONTRACT.is_symlink(), "contract absent")
    side, outer = Path(str(CONTRACT) + ".sha256"), Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(side.read_text().split() == [sha(CONTRACT), CONTRACT.name], "contract sidecar")
    need(outer.read_text().split() == [sha(side), side.name], "contract outer")
    contract = strict_json(CONTRACT)
    need(contract["schema"] == "m2233_ep34_tsbg_matched_power_source_repair_contract_r1_v1"
         and contract["status"] == "SOURCE_ONLY_REPAIR__M2234_REVIEW_REQUIRED__NO_EDA",
         "contract status")
    need(contract["execution_budget"] == {**COUNTS, "automatic_retry": False,
         "p1_serial": True, "reuse_m2203_raw": False}, "execution budget")
    for rel, digest in contract["source_inventory"].items():
        path = REPO / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "source identity: " + rel)
    verify_seal(M2204)
    need(strict_json(M2204 / "review.json")["status"].startswith("PASS_M2204"),
         "M2204 methodology authority")
    if require_review:
        verify_seal(REVIEW)
        review = strict_json(REVIEW / "review.json")
        need(review["status"] == "PASS_M2234_M2233_MATCHED_POWER_SOURCE_REPAIR_RELEASE"
             and review.get("score_over_100", 0) >= 95
             and review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
             and review["authorization"] == {**COUNTS, "automatic_retry": False,
                 "p1_serial": True, "reuse_m2203_raw": False}, "M2234 release")
        need(review["identity"]["runner_sha256"] == sha(RUNNER)
             and review["identity"]["contract_sha256"] == sha(CONTRACT),
             "M2234 binding")
        need(review["identity"]["m2172_helper_sha256"] == sha(STRUCT_HELPER)
             and review["identity"]["m2117_helper_sha256"] == sha(POWER_HELPER)
             and review["identity"]["m2160_helper_sha256"] == sha(BASE_HELPER),
             "M2234 helper binding")
    return contract


def no_same_uid_eda() -> None:
    blocked = {"vcs", "simv", "snps_shell", "dc_shell", "common_shell_exec",
               "common_shell_exe", "pt_shell", "lmstat"}
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            names = {Path(raw.decode(errors="replace")).name
                     for raw in (proc / "cmdline").read_bytes().split(b"\0") if raw}
            comm = (proc / "comm").read_text().strip()
        except OSError: continue
        if comm in blocked or names & blocked: hits.append((proc.name, comm))
    need(not hits, "same-UID EDA collision: " + repr(hits))


def run(command: list[str], cwd: Path, env: dict[str, str], timeout: int,
        log: Path) -> None:
    with log.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    need(completed.returncode == 0, "command failed: " + str(log))


def validate_log(path: Path) -> None:
    text = path.read_text(errors="replace")
    errors = [line for line in text.splitlines()
              if re.match(r"^(Error:|Fatal:)", line)
              and "Error during sourcing of" not in line]
    need(not errors, "tool log errors: " + repr(errors[:3]))


def selections() -> dict[str, dict]:
    data = strict_json(SELECTION)
    rows = {row["stratum"]: row for row in data["selections"]}
    need(tuple(rows) == STRATA and data["population"]["rows"] == 2880,
         "selection manifest")
    return rows


def production() -> int:
    source_validation(require_review=True)
    need(not RESULT.exists() and not ATTEMPT.exists() and not LOCK.exists(),
         "M2235 namespace consumed")
    for path in (VCS, PT, LMUTIL, SLOW_DB, FAST_DB, TT_DB):
        need(path.is_file() and not path.is_symlink(), "tool/input: " + str(path))
    no_same_uid_eda(); LOCK.mkdir(); ATTEMPT.mkdir()
    write_json(ATTEMPT / "attempt.json", {"status": "M2235_ATTEMPT_CONSUMED",
        "automatic_retry": False, "budget": COUNTS,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    seal_dir(ATTEMPT)
    work = Path(tempfile.mkdtemp(prefix=".m2235_m2233_work.", dir=HW / "results"))
    stage = Path(tempfile.mkdtemp(prefix=".m2235_m2233_stage.", dir=HW / "results"))
    counts = {key: 0 for key in COUNTS}
    try:
        counts["license_queries"] += 1
        run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER], work,
            clean_env({}), 60, stage / "license_preflight.log")
        source_paths = []
        for line in VCS_FILELIST.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                path = (REPO / stripped).resolve()
                need(path.is_file() and not path.is_symlink(), "VCS source")
                source_paths.append(path)
        need(len(source_paths) == 5, "VCS source count")
        selected = selections()
        for axis, mode in AXES.items():
            build = work / ("vcs_" + axis); build.mkdir()
            resolved = build / "sources.absolute.f"
            resolved.write_text("\n".join(map(str, source_paths)) + "\n")
            counts["vcs_compiles"] += 1
            run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
                 "+vcs+initreg+random", f"+define+M2217_SCHEDULE_MODE={mode}",
                 "-debug_access+r", "-assert", "svaext", "-lca", "+vcs+lic+wait",
                 f"-Mdir={build / 'csrc'}", "-f", str(resolved), "-top",
                 "tb_m2217_m2018_tsbg_matched_native_saif_power", "-o",
                 str(build / "simv")], build,
                clean_env({"VCS_HOME": str(VCS.parent.parent),
                           "VCS_ARCH_OVERRIDE": "linux"}), 21600,
                stage / f"vcs_compile_{axis}.log")
            validate_log(stage / f"vcs_compile_{axis}.log")
            need((build / "simv").is_file(), "simv absent")
            for stratum in STRATA:
                point = stage / axis / stratum
                point.mkdir(parents=True)
                prehistory = point / "rtl_prehistory.saif"
                measurement = point / "rtl_measurement.saif"
                counts["simv_runs"] += 1
                run(["./simv", "-lca", "+vcs+initreg+0",
                     f"+M2217_STRATUM={stratum}", "-no_save", "-ucli", "-i", str(UCLI)],
                    build, clean_env({"VCS_HOME": str(VCS.parent.parent),
                        "VCS_ARCH_OVERRIDE": "linux",
                        "M2217_PREHISTORY_SAIF_FILE": str(prehistory),
                        "M2217_MEASUREMENT_SAIF_FILE": str(measurement)}),
                    21600, point / "rtl_sim.log")
                validate_log(point / "rtl_sim.log")
                for path, counter in ((prehistory, "diagnostic_saif_files"),
                                      (measurement, "measurement_saif_files")):
                    need(path.is_file() and path.stat().st_size > 0, "SAIF absent")
                    seal_file(path); counts[counter] += 1
                for role, path in (("diagnostic_prehistory", prehistory),
                                   ("measurement", measurement)):
                    run([sys.executable, str(PARSER), "saif", "--axis", axis,
                         "--stratum", stratum, "--role", role, "--path", str(path)],
                        REPO, clean_env({}), 120, point / f"parse_{role}.log")

        for axis, mode in AXES.items():
            dc_root = stage / axis / "dc"; dc_root.mkdir()
            counts["dc_runs"] += 1
            run([str(DC), "-f", str(DC_TCL)], REPO, clean_env({
                "M2217_SCHEDULE_MODE": str(mode), "M2217_HW_ROOT": str(HW),
                "M2217_RTL_FILELIST": str(DC_FILELIST), "M2217_LIB_DB": str(SLOW_DB),
                "M2217_MIN_LIB_DB": str(FAST_DB), "M2217_SDC_FILE": str(SDC),
                "M2217_OUTPUT_DIR": str(dc_root),
                "M2217_OPERATING_CONDITION": "ssg0p9v125c"}), 21600,
                dc_root / "dc.log")
            validate_log(dc_root / "dc.log")
            identity = (dc_root / "reports/identity.rpt").read_text()
            design = re.findall(r"^design=(\S+)$", identity, re.MULTILINE)
            need(len(design) == 1, "DC design identity")
            for stratum in STRATA:
                cfg = selected[stratum]["ordinary" if mode == 0 else "tsbg"]
                point = stage / axis / stratum
                pt_root = point / "ptpx"; pt_root.mkdir()
                counts["ptpx_runs"] += 1
                run([str(PT), "-f", str(PT_TCL)], REPO, clean_env({
                    "M2217_AXIS": axis, "M2217_STRATUM": stratum,
                    "M2217_DESIGN_NAME": design[0], "M2217_TT_LIB_DB": str(TT_DB),
                    "M2217_MAPPED_NETLIST": str(dc_root / "netlist/m2018_axis_mapped.v"),
                    "M2217_MAPPED_SDC": str(dc_root / "netlist/m2018_axis_mapped.sdc"),
                    "M2217_DEFAULT_MAP": str(dc_root / "netlist/m2018_axis.ptpx_map.default.tcl"),
                    "M2217_ESSENTIAL_MAP": str(dc_root / "netlist/m2018_axis.ptpx_map.essential.tcl"),
                    "M2217_RTL_SAIF": str(point / "rtl_measurement.saif"),
                    "M2217_OUTPUT_DIR": str(pt_root),
                    "M2217_MEASUREMENT_CYCLES": str(cfg["cycles"]),
                    "M2217_ACCEPTED_BANK_REQUESTS": str(cfg["accepted_bank_requests"])}),
                    21600, pt_root / "ptpx.log")
                validate_log(pt_root / "ptpx.log")
        need(counts == COUNTS, "execution count mismatch")
        run([sys.executable, str(PARSER), "final", "--root", str(stage),
             "--output", str(stage / "result.json")], REPO, clean_env({}),
            300, stage / "final_parse.log")
        result = strict_json(stage / "result.json")
        result["schema"] = (
            "m2235_m2233_ep34_tsbg_matched_power_repair_result_r1_v1")
        result["status"] = (
            "PASS_RAW_M2235_PENDING_M2236_INDEPENDENT_RESULT_HAMMER")
        result["execution_counts"] = counts
        result["identity"] = {"runner_sha256": sha(RUNNER),
            "parser_sha256": sha(PARSER), "contract_sha256": sha(CONTRACT),
            "selection_sha256": sha(SELECTION),
            "m2234_review_sha256": sha(REVIEW / "review.json"),
            "m2172_helper_sha256": sha(STRUCT_HELPER),
            "m2117_helper_sha256": sha(POWER_HELPER),
            "m2160_helper_sha256": sha(BASE_HELPER),
            "docs359_sha256": sha(DOC359)}
        result["implementation_corners"] = {
            "dc_max_corner": "SSG0P9V125C",
            "dc_min_corner": "FFG1P05VM40C",
            "ptpx_corner": "TT0P9V25C",
            "dc_to_ptpx_is_mixed_corner": True,
            "sram_dynamic_model": "FOUNDRY_QRT_TT1V85C_DEEP_SEGMENT_CONSERVATIVE",
            "sram_leakage_model": "FOUNDRY_GENERATED_128X128_HVT_SSG0P9V125C_AREA_SCALED_PROXY"}
        result["aggregate"]["scope"] = (
            "FIXED_THREE_WINDOW_WEIGHTED_INDEX__NOT_POPULATION_MEAN")
        comparison = result["aggregate"]["comparison"]
        comparison["fixed_three_window_index_weights"] = comparison.pop(
            "fixed_population_tercile_weights")
        result["claim_boundary"].update({
            "aggregate_is_three_representative_window_weighted_index": True,
            "aggregate_is_2880_workload_population_mean": False,
            "aggregate_is_frame_energy": False,
            "selection_tiebreak_maximizes_ordinary_requests": True})
        result["independent_result_hammer_required"] = True
        write_json(stage / "result.json", result)
        (stage / "RUN_COMPLETE.txt").write_text(
            "PASS_M2235_RAW_PENDING_M2236_INDEPENDENT_RESULT_HAMMER\n")
        seal_dir(stage); os.rename(stage, RESULT); shutil.rmtree(work); LOCK.rmdir()
        return 0
    except BaseException as exc:
        (stage / "FAILED_DO_NOT_CITE.txt").write_text(
            f"status=FAILED_DO_NOT_CITE\nexception={type(exc).__name__}: {exc}\nautomatic_retry=false\n")
        write_json(stage / "execution_counts.json", counts); seal_dir(stage)
        quarantine = Path(str(RESULT) + f".failed.{os.getpid()}.quarantine")
        if not quarantine.exists(): os.rename(stage, quarantine)
        if LOCK.exists(): LOCK.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    if args.static:
        contract = source_validation(require_review=False)
        print(json.dumps({"status": "PASS_M2233_STATIC_RUNNER",
            "source_count": len(contract["source_inventory"]),
            "execution_budget": contract["execution_budget"],
            "selection_rows": selections()}, indent=2, sort_keys=True))
        return 0
    return production()


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Failure as exc:
        print("M2233_FAIL_CLOSED: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
