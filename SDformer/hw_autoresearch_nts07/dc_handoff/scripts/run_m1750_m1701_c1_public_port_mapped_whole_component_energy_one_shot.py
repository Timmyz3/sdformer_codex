#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot M1701 public-port mapped VCS->SAIF->PTPX component campaign.

The file is source-only until an independent M1751 source review, an M1752
release, the sealed M1745 fail disposition, and the canonical zero-EDA M1740
timing result created under M1743 authority are exact-SHA pinned.
It executes one mapped compile, one directed 64-row simulation/SAIF window and
one PTPX run.  No retry or alternate workload is permitted.
"""
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1750_m1701_c1_public_port_mapped_whole_component_energy_source_contract_r1_20260901.json"
M1745_FAIL = HW / "reviews/m1745_m1739_m1701_c1_public_port_mapped_production_energy_source_hammer_r1_20260901"
M1751 = HW / "reviews/m1751_m1750_m1701_c1_public_port_mapped_whole_component_energy_source_hammer_r1_20260901"
M1752 = HW / "contracts/m1752_m1751_m1750_m1701_c1_public_port_mapped_whole_component_energy_launch_release_r1_20260901.json"
M1743 = HW / "contracts/m1743_m1742_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_release_r1_20260901.json"
TIMING_RESULT = HW / "dc_handoff/runs/m1740_c1_readonly_formality_pt_salvage_r1_20260901"
CHECKER = HW / "system_simulator/scripts/check_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1750_checker", CHECKER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1750 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

M1701 = CHECK.M1701
NET = CHECK.NET
SDC = CHECK.SDC
FILELIST = CHECK.FILELIST
UCLI = CHECK.UCLI
PT_TCL = CHECK.PT_TCL
STD_TT = CHECK.STD_TT
STD_SS = CHECK.STD_SS
MACRO_DB = CHECK.MACRO_DB
TOP = CHECK.TOP
SAIF_SCOPE = CHECK.SAIF_SCOPE
DESIGN = CHECK.DESIGN

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1750_c1_public_port_mapped_component_energy_attempt_consumed"
RESULT = HW / "results/m1750_c1_public_port_mapped_component_energy_r1_20260901"
FAILURE = HW / "results/m1750_c1_public_port_mapped_component_energy_r1_20260901.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1750_c1_public_port_mapped_component_energy_r1_20260901.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1750_c1_public_port_mapped_component_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1750_c1_public_port_mapped_component_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1750_c1_public_port_mapped_component_energy_failure_stage." + str(os.getpid()))
LOCK = Path("/tmp/m1750_c1_public_port_mapped_component_energy.lock")
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

COUNTS = {"vcs_compiles": 1, "simv_runs": 1, "saif_files": 1,
          "ptpx_runs": 1}
SOURCE_CLAIMS = CHECK.CLAIMS
M1743_SHA = "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921"
M1743_SUM_SHA = "7d481d605bffd1386b8926e709424a2c78b3f78eff340caf1727dbe7ec84cfe1"
M1743_OUTER_SHA = "7a52c2e7692b62857dfe1d2b1bd9e2825372a0fc839822abf086d4837bbcf112"
M1745_FAIL_REVIEW_SHA = "44fca21fde5163ae39f249f5a485c5f2d4953910d8ff76e911aff6a543373359"
M1745_FAIL_MANIFEST_SHA = "c5b1f83b618ab8aadff16dc9e2a8f6498a852c66559d7a55171f93831bf3595a"
M1745_FAIL_OUTER_SHA = "f81c8c0166da2d2e6ce7a99aa469bad9d800193edab865c33fe64ab6753c0404"
TIMING_RECEIPT_SHA = "0b3ee22f9369a38eb83f674a4f1eb73fac39757ee85a3e1aeebe032bd0c76a1e"
TIMING_MANIFEST_SHA = "d3f2e14a6f6c0600abce2f5af2479402d41986736e3d9c32c6044e4225f64c75"
TIMING_OUTER_SHA = "6f2b17f7016665cd663b9694a1ccbd29fa551ecf75ba29aa52c4bb56c5769b38"


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    if (not path.is_file() or path.is_symlink() or sha(path) != digest):
        raise Failure("identity drift " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid " + str(root))
    exact(root / "SHA256SUMS", manifest_sha)
    exact(root / "SHA256SUMS.seal.sha256", outer_sha)
    if (root / "SHA256SUMS.seal.sha256").read_text().split() != [
            manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content")
    listed = set()
    for row in (root / "SHA256SUMS").read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in listed:
            raise Failure("unsafe manifest")
        exact(root / rel, fields[0])
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift " + str(root))


def verify_file_seal(path, payload_sha, sum_sha, outer_sha):
    sum_path = Path(str(path) + ".sha256")
    outer_path = Path(str(path) + ".sha256.seal.sha256")
    exact(path, payload_sha)
    exact(sum_path, sum_sha)
    exact(outer_path, outer_sha)
    if sum_path.read_text() != payload_sha + "  " + path.name + "\n":
        raise Failure("file digest sidecar content drift " + str(path))
    if outer_path.read_text() != sum_sha + "  " + sum_path.name + "\n":
        raise Failure("file outer sidecar content drift " + str(path))


def seal_dir(root):
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def authority_pin(name):
    value = os.environ.get(name, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Failure("exact authority absent " + name)
    return value


def verify_authority():
    exact(RUNNER, authority_pin("M1750_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1750_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_seal(M1745_FAIL, M1745_FAIL_MANIFEST_SHA, M1745_FAIL_OUTER_SHA)
    exact(M1745_FAIL / "review.json", M1745_FAIL_REVIEW_SHA)
    verify_seal(M1751, authority_pin("M1750_EXPECTED_M1751_MANIFEST_SHA256"),
                authority_pin("M1750_EXPECTED_M1751_OUTER_FILE_SHA256"))
    exact(M1751 / "review.json",
          authority_pin("M1750_EXPECTED_M1751_REVIEW_SHA256"))
    exact(M1752, authority_pin("M1750_EXPECTED_M1752_RELEASE_SHA256"))
    verify_file_seal(M1743, M1743_SHA, M1743_SUM_SHA, M1743_OUTER_SHA)
    verify_seal(TIMING_RESULT, TIMING_MANIFEST_SHA, TIMING_OUTER_SHA)
    exact(TIMING_RESULT / "receipt.json", TIMING_RECEIPT_SHA)
    failed_review = strict_json(M1745_FAIL / "review.json")
    review = strict_json(M1751 / "review.json")
    release = strict_json(M1752)
    timing_release = strict_json(M1743)
    timing = strict_json(TIMING_RESULT / "receipt.json")
    if (failed_review.get("status") !=
            "FAIL_M1745_P0_DO_NOT_AUTHORIZE_M1746__ADDITIVE_PTPX_MACRO_POWER_REPAIR_REQUIRED"
            or failed_review.get("m1746_authorized") is not False
            or review.get("status") !=
            "PASS_M1751_M1750_C1_WHOLE_COMPONENT_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_CAMPAIGN"
            or release.get("status") !=
            "AUTHORIZE_ONE_M1750_C1_PUBLIC_PORT_MAPPED_WHOLE_COMPONENT_ENERGY_CAMPAIGN"
            or timing_release.get("status") !=
            "AUTHORIZE_ONE_M1740_C1_READONLY_FORMALITY_PT_SALVAGE_CANONICALIZATION"
            or timing.get("schema") !=
            "m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_receipt_r1_v1"
            or timing.get("status") !=
            "PASS_CANONICAL_C1_FORMALITY_AND_INDEPENDENT_PT_PRELAYOUT"):
        raise Failure("review/release/timing authority semantic drift")
    formal = timing.get("formality", {})
    prime = timing.get("prime_time", {})
    scope = timing.get("scope", {})
    if ({key: formal.get(key) for key in (
            "source", "verification_succeeded", "passing_compare_points",
            "failing", "aborted", "unverified", "unmatched",
            "macro_instances_per_side")} !=
            {"source": "sealed_M1722_failure_subproof",
             "verification_succeeded": True,
             "passing_compare_points": 16549, "failing": 0,
             "aborted": 0, "unverified": 0, "unmatched": 0,
             "macro_instances_per_side": 9}
            or prime != {"setup_wns_ns": "0.027871",
                         "setup_tns_ns": "0.0",
                         "setup_violating_paths": "0",
                         "hold_wns_ns": "0.001827",
                         "hold_tns_ns": "0.0",
                         "hold_violating_paths": "0",
                         "macro_count": "9", "clock_period_ns": "3.000",
                         "setup_uncertainty_ns": "0.200",
                         "hold_uncertainty_ns": "0.050"}
            or scope != {"prelayout": True, "ideal_clock": True,
                         "wireload": "ZeroWireload", "parasitics": False,
                         "macro_count": 9, "power_or_energy": False}):
        raise Failure("canonical timing receipt content drift")
    if timing.get("claim_boundary") != {
            "cycle_speedup": False, "dc": False, "energy": False,
            "formality": True, "headline": False, "independent_pt": True,
            "paper_citable": True, "paper_ppa_ready": False,
            "power": False, "system_speedup": False}:
        raise Failure("canonical timing claim boundary drift")


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue " + str(path))


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "pt_shell",
               "fm_shell", "icc2_shell", "common_shell_exec"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception:
            break
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in ancestry:
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            comm = (item / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked:
            hits.append((item.name, comm))
    if hits:
        raise Failure("same-UID EDA collision " + repr(hits))


def resource_gate():
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "SwapFree",
                                    "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 24 * 1024 * 1024:
        raise Failure("MemAvailable below 24 GiB")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("SwapFree below 8 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit headroom below 24 GiB")


def clean_env(extra):
    # Preserve the real HOME to allow Synopsys startup files to resolve; do not
    # synthesize or repurpose it.  Proxy variables are intentionally omitted.
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "HOME": os.environ["HOME"],
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def run(command, cwd, env, timeout, output):
    with output.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure " + command[0])


def main():
    if len(sys.argv) != 1:
        raise Failure("M1750 accepts no arguments")
    state = {"phase": "SOURCE", "attempt": False, "complete": False}
    state.update(dict((key, 0) for key in COUNTS))
    lock_handle = LOCK.open("a+")
    queue_handle = SHARED_QUEUE.open("a+")
    try:
        verify_authority()
        CHECK.validate_sources()
        namespaces_fresh()
        collision_gate()
        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate()
        resource_gate()
        namespaces_fresh()
        if not LICENSE_FILE.is_file() or LICENSE_FILE.is_symlink():
            raise Failure("license file invalid")
        state["phase"] = "LICENSE_PREFLIGHT"
        probe = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                               env=clean_env({}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=False)
        if probe.returncode != 0:
            raise Failure("license preflight")

        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1750_ATTEMPT_CONSUMED", "budget": COUNTS,
            "workload": "ep34_density_conditioned_directed_component_activity",
            "residual_and_psum_data": "synthetic",
            "automatic_retry": False})
        seal_dir(ATTEMPT)
        WORK.mkdir()
        build = WORK / "build"
        candidate = WORK / "candidate"
        build.mkdir()
        candidate.mkdir()

        state["phase"] = "VCS_COMPILE"
        state["vcs_compiles"] += 1
        run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
             "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc",
             "-f", str(FILELIST), "-top", TOP, "-o", "simv"],
            build, clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                              "VCS_ARCH_OVERRIDE": "linux"}), 3600,
            build / "compile.log")
        if not (build / "simv").is_file():
            raise Failure("simv absent")

        state["phase"] = "MAPPED_SIM_SAIF"
        state["simv_runs"] += 1
        saif = candidate / "m1750_c1_directed_component.saif"
        sim_log = candidate / "mapped_sim.log"
        run(["./simv", "-lca", "+M1739_UCLI_SAIF", "-no_save", "-ucli",
             "-i", str(UCLI)], build,
            clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                       "VCS_ARCH_OVERRIDE": "linux",
                       "M1739_SAIF_FILE": str(saif)}), 3600, sim_log)
        runtime = CHECK.validate_runtime(sim_log)
        if not saif.is_file() or saif.is_symlink() or saif.stat().st_size == 0:
            raise Failure("SAIF absent")
        saif_check = CHECK.validate_saif(
            saif, runtime["measurement_cycles"])
        state["saif_files"] += 1

        state["phase"] = "PTPX"
        state["ptpx_runs"] += 1
        pt_dir = candidate / "ptpx"
        pt_dir.mkdir()
        run([str(PT), "-f", str(PT_TCL)], HW,
            clean_env({"M1750_STD_TT_DB": str(STD_TT),
                       "M1750_STD_SS_DB": str(STD_SS),
                       "M1750_MACRO_SLOW_DB": str(MACRO_DB),
                       "M1750_MAPPED_NETLIST": str(NET),
                       "M1750_MAPPED_SDC": str(SDC),
                       "M1750_GATE_SAIF": str(saif),
                       "M1750_OUTPUT_DIR": str(pt_dir),
                       "M1750_SAIF_INSTANCE": SAIF_SCOPE,
                       "M1750_MEASUREMENT_CYCLES": str(runtime["measurement_cycles"]),
                       "M1750_SAIF_DURATION_NS": str(runtime["measurement_cycles"] * 3)}),
            3600, pt_dir / "ptpx.log")
        marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
        if (not marker.is_file() or marker.read_text().count(
                "PASS_M1750_C1_M1701_PUBLIC_PORT_MAPPED_WHOLE_COMPONENT_PTPX_TOOL_COMPLETE") != 1):
            raise Failure("PTPX marker absent")
        metrics = CHECK.whole_component_power(
            pt_dir / "reports/ptpx_whole_mapped_c1_including_9macro_liberty.rpt",
            runtime["measurement_cycles"], runtime["macro_reads"],
            runtime["macro_writes"])
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")

        STAGE.mkdir()
        shutil.copytree(candidate, STAGE / "candidate")
        shutil.copy2(build / "compile.log", STAGE / "compile.log")
        write_json(STAGE / "runtime.json", runtime)
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1750_c1_directed_whole_mapped_component_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "one_shot": {**COUNTS, "automatic_retry": False},
            "workload": "ep34_density_conditioned_directed_component_activity",
            "residual_and_psum_data": "synthetic",
            "power_corner": {
                "classification": "mixed_corner_component_estimate",
                "standard_cells": "TT 0.9V 25C",
                "parent_sram_macro_liberty": "SSG 0.9V 125C"},
            "ptpx_plus_datasheet_sram_combined": False,
            "saif_check": saif_check,
            "support_tiers": {"ep34_active_only_p25": 1,
                              "ep34_active_only_p50": 2,
                              "ep34_active_only_p75": 4},
            "public_port_only": True, "new_rtl_wrapper": False,
            "claim_boundary": metrics["claim_boundary"]})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1750_C1_PUBLIC_PORT_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1750_C1_PUBLIC_PORT_MAPPED_COMPONENT_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "counts": dict((key, state[key]) for key in COUNTS),
                    "automatic_retry": False, "canonical_result": False})
                seal_dir(FAIL_STAGE)
                publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
            if WORK.is_dir() and not PRIVATE.exists():
                try:
                    publish_no_replace(WORK, PRIVATE)
                except BaseException:
                    pass
        raise
    finally:
        lock_handle.close()
        queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
