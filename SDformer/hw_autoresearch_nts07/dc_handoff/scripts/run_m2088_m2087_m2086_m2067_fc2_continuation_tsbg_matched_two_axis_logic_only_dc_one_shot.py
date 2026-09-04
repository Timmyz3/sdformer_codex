#!/opt/anaconda3/bin/python3.12
"""M2088: one owner-safe matched two-axis DC run for the M2067 FC2 island.

The ordinary and TSBG points differ only in SCHEDULE_MODE.  Both elaborate
the complete G96/G192 continuation wrapper, use the same public ports, TSMC28
libraries, 3 ns constraint, and logic-only pre-CTS flow.  Hold is diagnostic.
Execution remains blocked until the fresh M2067 R9 VCS result is independently
accepted by M2085 and this source is independently accepted by M2087.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / (
    "contracts/m2086_m2067_fc2_continuation_tsbg_matched_dc_source_"
    "contract_r1_20260904.json")
FILELIST = HW / (
    "dc_handoff/filelists/iscas_m2086_m2067_fc2_continuation_tsbg_"
    "matched_two_axis_logic_only_dc.f")
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
RTL_M803 = HW / (
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv")
RTL_M2018 = HW / (
    "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv")
RTL_M2067 = HW / "rtl_m2067/m2067_fc2_exact_continuation_wrapper.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2085 = HW / (
    "reviews/m2085_m2067_ep34_fc2_exact_continuation_vcs_r9_result_"
    "hammer_r1_20260904")
M2087 = HW / (
    "reviews/m2087_m2086_m2067_fc2_continuation_tsbg_matched_dc_source_"
    "hammer_r1_20260904")
R9_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r9_codex_ownerfix_"
    "20260904")

PYTHON = Path("/opt/anaconda3/bin/python3.12")
DC_SHELL = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
SLOW_DB = Path(
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
    "TSMCHOME/digital/Front_End/timing_power_noise/NLDM/"
    "tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST_DB = Path(
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/"
    "TSMCHOME/digital/Front_End/timing_power_noise/NLDM/"
    "tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
DESIGN = "m2067_fc2_exact_continuation_wrapper"
TOOL_SHA256 = {
    PYTHON: "873a1168d6d2a7d1b406b85c2a1ea986a6f086041069ab1ee3f70b9217f10161",
    DC_SHELL: "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    SLOW_DB: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    FAST_DB: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}
SOURCE_SHA256 = {
    FILELIST: "f5f661eb98e011c9e5f9922bf298eb91083e014869e714fdf1c1d8971d1b490d",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    RTL_M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    RTL_M2018: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    RTL_M2067: "755027453b9fc91264f44918cc16e31b278cf70e1b13821666ca2be602022c92",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

RUN_ROOT = HW / "dc_handoff/runs"
ATTEMPT = RUN_ROOT / ".m2088_m2067_fc2_continuation_tsbg_dc_attempt_consumed"
RESULT = RUN_ROOT / (
    "m2088_m2067_fc2_continuation_tsbg_matched_two_axis_logic_only_dc_"
    "r1_20260904")
FAILURE = RUN_ROOT / (
    "m2088_m2067_fc2_continuation_tsbg_matched_two_axis_logic_only_dc_"
    "r1_20260904.failed_or_incomplete.quarantine")
WORK = RUN_ROOT / (
    ".m2088_m2067_fc2_continuation_tsbg_dc_work." + str(os.getpid()))
FAIL_STAGE = RUN_ROOT / (
    ".m2088_m2067_fc2_continuation_tsbg_dc_failure." + str(os.getpid()))
OWNER_LOCK = Path("/tmp/m2088_m2067_fc2_continuation_tsbg_dc.lock")
EDA_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

ATTEMPT_OWNED = False
OWNER_NONCE = ""
RESULT_PUBLISHED = False
LMSTAT_OUTPUT = b""
RUN_STATE = {
    "phase": "PRE_AUTHORITY", "license_preflight_lmstat": 0,
    "dc_shell_runs": 0, "completed_axes": [], "current_axis": None,
}


class Failure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256(path) != digest:
        raise Failure("identity drift " + str(path))


def authority(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Failure("authority pin absent " + name)
    return value


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root " + str(path))
    return value


def sealed_directory(root: Path, manifest_pin: str, outer_pin: str) -> dict:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_pin)
    exact(outer, outer_pin)
    if outer.read_text().split() != [sha256(manifest), "SHA256SUMS"]:
        raise Failure("outer seal content " + str(root))
    mapping = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        relative = Path(fields[1].lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise Failure("unsafe manifest member")
        exact(root / relative, fields[0])
        if relative.as_posix() in mapping:
            raise Failure("duplicate manifest member")
        mapping[relative.as_posix()] = fields[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {
                  "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if any(p.is_symlink() for p in root.rglob("*")) or actual != set(mapping):
        raise Failure("non-exhaustive or linked seal " + str(root))
    return mapping


def verify_review(root: Path, prefix: str, status_prefix: str) -> dict:
    mapping = sealed_directory(
        root, authority(prefix + "_MANIFEST_SHA256"),
        authority(prefix + "_OUTER_FILE_SHA256"))
    review_path = root / "review.json"
    exact(review_path, authority(prefix + "_REVIEW_SHA256"))
    if mapping.get("review.json") != sha256(review_path):
        raise Failure("review is not sealed " + str(root))
    review = strict_json(review_path)
    if not review.get("status", "").startswith(status_prefix):
        raise Failure("review status " + str(root))
    return review


def verify_authority() -> tuple[dict, dict]:
    exact(RUNNER, authority("M2088_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority("M2088_EXPECTED_CONTRACT_SHA256"))
    for path, digest in {**TOOL_SHA256, **SOURCE_SHA256}.items():
        exact(path, digest)
    result_review = verify_review(M2085, "M2088_EXPECTED_M2085", "PASS_M2085_")
    source_review = verify_review(M2087, "M2088_EXPECTED_M2087", "PASS_M2087_")
    r9_mapping = sealed_directory(
        R9_RESULT, authority("M2088_EXPECTED_R9_RESULT_MANIFEST_SHA256"),
        authority("M2088_EXPECTED_R9_RESULT_OUTER_FILE_SHA256"))
    identity = source_review.get("reviewed_source_identity", {})
    if (identity.get("runner_sha256") != sha256(RUNNER)
            or identity.get("contract_sha256") != sha256(CONTRACT)
            or identity.get("filelist_sha256") != sha256(FILELIST)
            or source_review.get("authorization", {}).get("execute_once") is not True
            or source_review.get("authorization", {}).get("automatic_retry") is not False):
        raise Failure("M2087 source authorization")
    result_identity = result_review.get("reviewed_result_identity", {})
    if (result_review.get("authorization", {}).get("m2088_two_axis_dc") is not True
            or result_review.get("observed", {}).get("workloads") != 960
            or result_identity.get("result_json_sha256")
                != sha256(R9_RESULT / "result.json")
            or result_identity.get("manifest_sha256")
                != sha256(R9_RESULT / "SHA256SUMS")
            or result_identity.get("outer_file_sha256")
                != sha256(R9_RESULT / "SHA256SUMS.seal.sha256")
            or r9_mapping.get("result.json") != sha256(R9_RESULT / "result.json")):
        raise Failure("M2085 result authorization")
    return result_review, source_review


def fresh_namespaces() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, WORK, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    for pattern in (".m2088_m2067_fc2_continuation_tsbg_dc_work.*",
                    ".m2088_m2067_fc2_continuation_tsbg_dc_failure.*"):
        if next(RUN_ROOT.glob(pattern), None) is not None:
            raise Failure("private namespace residue " + pattern)


def resource_gate() -> None:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":") in {
                "MemAvailable", "CommitLimit", "Committed_AS"}:
            values[fields[0].rstrip(":")] = int(fields[1])
    if values.get("MemAvailable", 0) < 64 * 1024 * 1024:
        raise Failure("MemAvailable below 64 GiB gate")
    if (values.get("CommitLimit", 0) - values.get("Committed_AS", 0)
            < 32 * 1024 * 1024):
        raise Failure("commit headroom below 32 GiB gate")


def collision_gate() -> None:
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (OSError, ValueError):
            break
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in ancestry:
            continue
        try:
            if item.stat().st_uid == os.getuid():
                command = (item / "comm").read_text().strip()
                if command in blocked:
                    hits.append((item.name, command))
        except OSError:
            pass
    if hits:
        raise Failure("same-UID EDA collision " + repr(hits))


def clean_env(axis: str, mode: int, output: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "TMPDIR": "/tmp", "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
        "LM_LICENSE_FILE": str(LICENSE_FILE), "DESIGN_NAME": DESIGN,
        "HW_ROOT": str(HW), "RTL_FILELIST": str(FILELIST),
        "LIB_DB": str(SLOW_DB), "MIN_LIB_DB": str(FAST_DB),
        "SDC_FILE": str(SDC), "OUTPUT_DIR": str(output),
        "OPERATING_CONDITION": "ssg0p9v125c", "CLOCK_PERIOD_NS": "3.000",
        "ELAB_PARAMETERS": "SCHEDULE_MODE=" + str(mode),
        "M2088_AXIS": axis,
    }


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal_dir(root: Path) -> None:
    members = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise Failure("symlink before seal")
    manifest = root / "SHA256SUMS"
    with manifest.open("x") as stream:
        for path in members:
            stream.write(sha256(path) + "  "
                         + path.relative_to(root).as_posix() + "\n")
    with (root / "SHA256SUMS.seal.sha256").open("x") as stream:
        stream.write(sha256(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source: Path, destination: Path) -> None:
    if os.path.lexists(str(destination)):
        raise Failure("publish destination exists")
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.renameat2(-100, os.fsencode(source), -100,
                        os.fsencode(destination), 1)
    if rc != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def attempt_owned_by_this_process() -> bool:
    if not ATTEMPT_OWNED or not OWNER_NONCE:
        return False
    try:
        owner = strict_json(ATTEMPT / "owner.json")
    except Exception:
        return False
    return (owner.get("pid") == os.getpid()
            and owner.get("nonce") == OWNER_NONCE
            and owner.get("runner_sha256") == sha256(RUNNER))


def publish_failure(exc: BaseException) -> None:
    if RESULT_PUBLISHED or not attempt_owned_by_this_process():
        return
    if os.path.lexists(str(RESULT)) or os.path.lexists(str(FAILURE)):
        return
    FAIL_STAGE.mkdir()
    state = dict(RUN_STATE)
    state.update({
        "status": "FAILED_DO_NOT_CITE_NO_RETRY",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__, "error": str(exc),
        "runner_sha256": sha256(RUNNER), "contract_sha256": sha256(CONTRACT),
        "owner_nonce": OWNER_NONCE, "automatic_retry": False,
    })
    write_json(FAIL_STAGE / "failure.json", state)
    if LMSTAT_OUTPUT:
        with (FAIL_STAGE / "lmstat.log").open("xb") as stream:
            stream.write(LMSTAT_OUTPUT)
    if WORK.is_dir() and not WORK.is_symlink():
        shutil.move(str(WORK), str(FAIL_STAGE / "work"))
    seal_dir(FAIL_STAGE)
    publish_no_replace(FAIL_STAGE, FAILURE)


def minimum_slack(path: Path) -> float:
    values = re.findall(
        r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)",
        path.read_text(errors="replace"))
    if not values:
        raise Failure("missing slack " + str(path))
    return min(float(value) for value in values)


def validate_dc_log(path: Path) -> None:
    lines = path.read_text(errors="replace").splitlines()
    expected = (
        "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/"
        "auxx/gui/dv/.synopsys_dv.tcl")
    hits = [index for index, line in enumerate(lines) if line == expected]
    if len(hits) != 1:
        raise Failure("unexpected DC error cardinality " + str(path))
    start = hits[0]
    end = start + 15
    if (start < 1 or start > 63 or end + 1 >= len(lines)
            or lines[start - 1] != "Initializing..."
            or not lines[end + 1].startswith("Current time:")):
        raise Failure("bootstrap block placement " + str(path))
    block = "\n".join(lines[start:end + 1]) + "\n"
    if hashlib.sha256(block.encode()).hexdigest() != (
            "3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1"):
        raise Failure("bootstrap block identity " + str(path))
    filtered = lines[:start] + lines[end + 1:]
    if any(re.search(r"(?i)(error:|fatal:)", line) for line in filtered):
        raise Failure("non-whitelisted DC error " + str(path))
    if any(re.search(r"\((TIM-209|OPT-150)\)", line)
           for line in filtered):
        raise Failure("loop diagnostic after whitelist " + str(path))


def parse_axis(root: Path, mode: int) -> dict:
    required = [
        root / "TCL_PASS_TERMINAL.txt", root / "reports/area.rpt",
        root / "reports/qor.rpt", root / "reports/timing_setup.rpt",
        root / "reports/timing_hold_diagnostic.rpt",
        root / "reports/precompile_loop_gate.rpt",
        root / "reports/flow_contract.rpt",
        root / "reports/compile_receipt.rpt",
        root / "reports/constraint_max_capacitance.rpt",
        root / "reports/constraint_max_transition.rpt",
        root / "reports/constraint_max_fanout.rpt",
        root / "reports/port_count.txt",
        root / "netlist" / f"{DESIGN}_mapped.v",
        root / "netlist" / f"{DESIGN}_mapped.sdc",
        root / "netlist" / f"{DESIGN}.ddc",
        root / "netlist" / f"{DESIGN}.svf",
    ]
    if any(not p.is_file() or p.is_symlink() or p.stat().st_size == 0
           for p in required):
        raise Failure("missing axis artifact " + str(root))
    validate_dc_log(root / "dc.log")
    if (root / "reports/precompile_loop_gate.rpt").read_text().splitlines()[:2] != [
            "TIM-209=0", "OPT-150=0"]:
        raise Failure("precompile loop gate " + str(root))
    flow = (root / "reports/flow_contract.rpt").read_text()
    compile_receipt = (root / "reports/compile_receipt.rpt").read_text()
    for token in ("compile_ultra_count=1", "incremental_compile_count=0",
                  "hold_optimization_count=0"):
        if token not in compile_receipt and token not in flow:
            raise Failure("compile-flow receipt " + token + " " + str(root))
    for name in ("capacitance", "transition", "fanout"):
        report = root / f"reports/constraint_max_{name}.rpt"
        if "This design has no violated constraints." not in report.read_text(
                errors="replace"):
            raise Failure("electrical constraint violation " + str(report))
    area_match = re.search(
        r"Total cell area:\s*([0-9.]+)",
        (root / "reports/area.rpt").read_text(errors="replace"))
    if area_match is None:
        raise Failure("missing area " + str(root))
    return {
        "schedule_mode": mode,
        "area_um2": float(area_match.group(1)),
        "setup_wns_ns": minimum_slack(root / "reports/timing_setup.rpt"),
        "hold_diagnostic_wns_ns": minimum_slack(
            root / "reports/timing_hold_diagnostic.rpt"),
        "public_port_count": int((root / "reports/port_count.txt").read_text()),
        "mapped_netlist_sha256": sha256(
            root / "netlist" / f"{DESIGN}_mapped.v"),
        "dc_log_sha256": sha256(root / "dc.log"),
    }


def main() -> int:
    global ATTEMPT_OWNED, OWNER_NONCE, RESULT_PUBLISHED, LMSTAT_OUTPUT
    if len(sys.argv) != 1:
        raise Failure("runner accepts no arguments")
    verify_authority()
    OWNER_LOCK.touch(exist_ok=True)
    EDA_QUEUE.touch(exist_ok=True)
    with OWNER_LOCK.open("r+") as owner_lock, EDA_QUEUE.open("r+") as queue_lock:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        result_review, source_review = verify_authority()
        fresh_namespaces()
        collision_gate()
        resource_gate()
        OWNER_NONCE = os.urandom(16).hex()
        ATTEMPT.mkdir()
        ATTEMPT_OWNED = True
        write_json(ATTEMPT / "owner.json", {
            "schema": "m2088_attempt_owner_v1", "pid": os.getpid(),
            "nonce": OWNER_NONCE, "runner_sha256": sha256(RUNNER),
        })
        write_json(ATTEMPT / "attempt.json", {
            "schema": "m2088_m2067_fc2_continuation_tsbg_dc_attempt_v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256(RUNNER),
            "contract_sha256": sha256(CONTRACT),
            "dc_shell_runs_budget": 2, "automatic_retry": False,
        })
        seal_dir(ATTEMPT)

        RUN_STATE["phase"] = "LICENSE_PREFLIGHT"
        RUN_STATE["license_preflight_lmstat"] = 1
        completed = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER], cwd=HW,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                 "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
                 "LM_LICENSE_FILE": str(LICENSE_FILE)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=120, check=False)
        LMSTAT_OUTPUT = completed.stdout
        if completed.returncode != 0 or b"Users of Design-Compiler" not in LMSTAT_OUTPUT:
            raise Failure("single license preflight failed")

        WORK.mkdir()
        with (WORK / "lmstat.log").open("xb") as stream:
            stream.write(LMSTAT_OUTPUT)
        axes = {}
        for axis, mode in (("ordinary", 0), ("tsbg_b4", 1)):
            verify_authority()
            collision_gate()
            axis_root = WORK / axis
            axis_root.mkdir()
            log = axis_root / "dc.log"
            RUN_STATE.update({"phase": "DC_AXIS", "current_axis": axis,
                              "dc_shell_runs": mode + 1})
            with log.open("xb") as stream:
                completed = subprocess.run(
                    [str(DC_SHELL), "-f", str(TCL)], cwd=HW,
                    env=clean_env(axis, mode, axis_root), stdout=stream,
                    stderr=subprocess.STDOUT, timeout=21600, check=False)
            if completed.returncode != 0:
                raise Failure("dc_shell failure " + axis)
            axes[axis] = parse_axis(axis_root, mode)
            RUN_STATE["completed_axes"].append(axis)

        verify_authority()
        base = axes["ordinary"]
        candidate = axes["tsbg_b4"]
        cycle_ratio = float(
            result_review["observed"]["rtl_cycle_ratio_observed"])
        area_ratio = candidate["area_um2"] / base["area_um2"]
        throughput_per_area_ratio = cycle_ratio / area_ratio
        comparison = {
            "ordinary_over_tsbg_rtl_cycle_ratio": cycle_ratio,
            "tsbg_over_ordinary_logic_area_ratio": area_ratio,
            "tsbg_logic_area_overhead_fraction": area_ratio - 1.0,
            "tsbg_over_ordinary_throughput_per_logic_area_ratio":
                throughput_per_area_ratio,
            "both_setup_met": base["setup_wns_ns"] >= 0
                and candidate["setup_wns_ns"] >= 0,
            "public_ports_equal": base["public_port_count"]
                == candidate["public_port_count"],
        }
        comparison["candidate_gate"] = {
            "both_setup_met": comparison["both_setup_met"],
            "public_ports_equal": comparison["public_ports_equal"],
            "logic_area_tax_at_most_2pct": area_ratio <= 1.02,
            "throughput_per_logic_area_at_least_1p15x":
                throughput_per_area_ratio >= 1.15,
        }
        result = {
            "schema": "m2088_m2067_fc2_continuation_tsbg_matched_dc_result_v1",
            "status": "PENDING_INDEPENDENT_RESULT_HAMMER_DO_NOT_CITE",
            "axes": axes, "comparison": comparison,
            "execution": {"license_queries": 1, "dc_shell_runs": 2,
                          "automatic_retry": False},
            "source_and_authority_identity": {
                "runner_sha256": sha256(RUNNER),
                "contract_sha256": sha256(CONTRACT),
                "filelist_sha256": sha256(FILELIST),
                "m2085_review_sha256": sha256(M2085 / "review.json"),
                "m2087_review_sha256": sha256(M2087 / "review.json"),
                "r9_result_json_sha256": sha256(R9_RESULT / "result.json"),
            },
            "claim_boundary": {
                "same_public_ports_library_clock_constraints": True,
                "logic_only_pre_macro": True, "ideal_clock": True,
                "wireload": "ZeroWireload", "macro_count": 0,
                "hold_diagnostic_not_closed": True, "power": False,
                "energy": False, "full_fc_wall_time": False,
                "system_speedup": False, "paper_ppa_ready": False,
                "paper_admitted": False,
            },
        }
        write_json(WORK / "result.json", result)
        write_json(WORK / "source_review_snapshot.json", {
            "m2087_status": source_review["status"],
            "m2087_review_sha256": sha256(M2087 / "review.json"),
        })
        seal_dir(WORK)
        publish_no_replace(WORK, RESULT)
        RESULT_PUBLISHED = True
    print("PASS_M2088_COMPLETE_PENDING_INDEPENDENT_RESULT_HAMMER")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        publish_failure(exc)
        raise
    raise SystemExit(code)
