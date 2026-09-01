#!/usr/bin/python3.12
"""One-shot M1777 equal-bandwidth C2 mapped energy campaign.

Only the fair K8 and K1x8 axes are executed.  The single-K1 mapped DC point
remains a disclosed diagnostic and the sealed M1753 K1 X/Z failure remains
bound evidence; neither is converted into an energy coordinate.  All ten
checked DUT-only SAIF files must exist before the first whole-component PTPX
launch.  This source is inert until exact M1778 review and M1779 release pins
are supplied.
"""
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CHECKER = HW / "system_simulator/scripts/check_m1777_m1776_c2_two_axis_equal_bandwidth_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1777_m1776_c2_two_axis_equal_bandwidth_energy_source.py"
CONTRACT = HW / "contracts/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_contract_r1_20260902.json"
SOURCE_SPEC = HW / "contracts/m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_spec_r1_20260902.json"
M1776 = HW / "reviews/m1776_m1770_m1753_c2_k1_mapped_fault_failure_diagnosis_r1_20260902"
M1778 = HW / "reviews/m1778_m1777_c2_two_axis_equal_bandwidth_energy_source_hammer_r1_20260902"
M1779 = HW / "contracts/m1779_m1778_m1777_c2_two_axis_equal_bandwidth_energy_launch_release_r1_20260902.json"

BASE = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
NET_REL = "netlist/" + DESIGN + "_mapped.v"
SDC_REL = "netlist/" + DESIGN + "_mapped.sdc"
TOP = "tb_m1684_c2_m1609_fresh_mapped_production_energy"
SAIF_SCOPE = TOP + ".core.dut"
FILELISTS = {
    "k8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k8_fresh_mapped_production_energy.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1684_c2_m1609_k1x8_fresh_mapped_production_energy.f",
}
UCLI = HW / "dc_handoff/scripts/m1684_c2_m1609_fresh_mapped_production_energy.ucli.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m1684_c2_m1609_fresh_mapped_production_energy_tt0p9v25c.tcl"
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
SS_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
PYTHON312 = Path("/usr/bin/python3.12")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LOCK = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")

AXES = ("k8", "k1x8")
CASES = tuple(range(5))
CYCLES = {"k8": [51, 131, 486, 1231, 14],
          "k1x8": [53, 133, 499, 1246, 14]}
EVENTS = [20, 41, 90, 110, 0]
AREAS_UM2 = {"k8": 130476.905184, "k1x8": 585534.971643}
COUNTS = {"vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))

ATTEMPT = HW / "results/.m1777_c2_two_axis_equal_bandwidth_energy_attempt_consumed"
RESULT = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902"
FAILURE = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1777_c2_two_axis_equal_bandwidth_energy_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1777_c2_two_axis_equal_bandwidth_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1777_c2_two_axis_equal_bandwidth_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1777_c2_two_axis_equal_bandwidth_energy_failure_stage." + str(os.getpid()))

EXPECTED_PROXY = {
    "HTTP_PROXY": "http://127.0.0.1:7897",
    "HTTPS_PROXY": "http://127.0.0.1:7897",
    "ALL_PROXY": "http://127.0.0.1:7897",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "http_proxy": "http://127.0.0.1:7897",
    "https_proxy": "http://127.0.0.1:7897",
}
PROXY_KEYS = tuple(EXPECTED_PROXY)


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode) or sha(path) != digest):
        raise Failure("identity drift: " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key: " + key)
            value[key] = item
        return value
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text() != manifest_sha + "  SHA256SUMS\n":
        raise Failure("outer seal content drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in listed:
            raise Failure("unsafe/duplicate manifest member")
        exact(root / rel, fields[0])
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift: " + str(root))


def verify_file_seal(path, payload_sha, sum_sha, outer_sha):
    path = Path(path)
    sum_path = Path(str(path) + ".sha256")
    outer_path = Path(str(path) + ".sha256.seal.sha256")
    exact(path, payload_sha)
    exact(sum_path, sum_sha)
    exact(outer_path, outer_sha)
    if sum_path.read_text() != payload_sha + "  " + path.name + "\n":
        raise Failure("file seal content drift")
    if outer_path.read_text() != sum_sha + "  " + sum_path.name + "\n":
        raise Failure("file outer seal content drift")


def seal_dir(root):
    rows = []
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = Path(root) / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows))
    (Path(root) / "SHA256SUMS.seal.sha256").write_text(
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
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")


def authority_pin(name):
    value = os.environ.get(name, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Failure("exact authority absent: " + name)
    return value


def verify_interpreter():
    if (Path(sys.executable) != PYTHON312
            or Path(sys.executable).resolve() != PYTHON312
            or platform.python_implementation() != "CPython"
            or platform.python_version() != "3.12.13"
            or tuple(sys.version_info[:3]) != (3, 12, 13)):
        raise Failure("interpreter path/version drift")
    exact(PYTHON312, "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814")
    if RUNNER.read_bytes().splitlines()[0] != b"#!/usr/bin/python3.12":
        raise Failure("runner shebang drift")


def source_mapping(contract):
    rows = contract.get("execution_files")
    if not isinstance(rows, list):
        raise Failure("execution inventory absent")
    mapping = {}
    for row in rows:
        if (type(row) is not dict or set(row) != {"path", "sha256"}
                or row["path"] in mapping
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise Failure("execution inventory malformed")
        mapping[row["path"]] = row["sha256"]
    return mapping


def verify_execution_sources():
    contract = strict_json(CONTRACT)
    if (contract.get("schema") !=
            "m1777_m1776_c2_two_axis_equal_bandwidth_energy_source_contract_r1_v1"
            or contract.get("status") !=
            "SOURCE_ONLY__M1778_REVIEW_AND_M1779_RELEASE_REQUIRED__NO_EDA"
            or contract.get("claim_boundary") != CLAIMS):
        raise Failure("M1777 source contract semantics")
    mapping = source_mapping(contract)
    for rel, digest in mapping.items():
        path = HW / rel
        exact(path, digest)
    required = {RUNNER, CHECKER, TEST, UCLI, PT_TCL,
                FILELISTS["k8"], FILELISTS["k1x8"]}
    if not set(path.relative_to(HW).as_posix() for path in required).issubset(mapping):
        raise Failure("execution source inventory incomplete")
    for axis in AXES:
        mapped = contract["mapped_axes"][axis]
        exact(HW / mapped["netlist"]["path"], mapped["netlist"]["sha256"])
        exact(HW / mapped["sdc"]["path"], mapped["sdc"]["sha256"])
    return contract


def verify_authority():
    pins = {name: authority_pin(name) for name in (
        "M1777_EXPECTED_RUNNER_SHA256",
        "M1777_EXPECTED_SOURCE_CONTRACT_SHA256",
        "M1777_EXPECTED_M1778_REVIEW_SHA256",
        "M1777_EXPECTED_M1778_MANIFEST_SHA256",
        "M1777_EXPECTED_M1778_OUTER_FILE_SHA256",
        "M1777_EXPECTED_M1779_RELEASE_SHA256")}
    exact(RUNNER, pins["M1777_EXPECTED_RUNNER_SHA256"])
    exact(CONTRACT, pins["M1777_EXPECTED_SOURCE_CONTRACT_SHA256"])
    contract = verify_execution_sources()
    verify_seal(M1778, pins["M1777_EXPECTED_M1778_MANIFEST_SHA256"],
                pins["M1777_EXPECTED_M1778_OUTER_FILE_SHA256"])
    exact(M1778 / "review.json", pins["M1777_EXPECTED_M1778_REVIEW_SHA256"])
    exact(M1779, pins["M1777_EXPECTED_M1779_RELEASE_SHA256"])
    sum_path = Path(str(M1779) + ".sha256")
    outer_path = Path(str(M1779) + ".sha256.seal.sha256")
    if (sum_path.read_text() != sha(M1779) + "  " + M1779.name + "\n"
            or outer_path.read_text() != sha(sum_path) + "  " + sum_path.name + "\n"):
        raise Failure("M1779 double seal drift")
    review = strict_json(M1778 / "review.json")
    release = strict_json(M1779)
    if (review.get("status") !=
            "PASS_M1778_M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_ATTEMPT"
            or release.get("status") !=
            "AUTHORIZE_ONE_M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_ATTEMPT"
            or release.get("authorization") != {
                "future_m1777_attempts": 1, "automatic_retry": False,
                "vcs_compiles": 2, "simv_runs": 10,
                "saif_files": 10, "ptpx_runs": 10}
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "m1778_review_sha256": sha(M1778 / "review.json")}
            or release.get("claim_boundary") != CLAIMS):
        raise Failure("M1778/M1779 authority semantics")
    return contract


def verify_predecessors_and_inputs():
    verify_file_seal(SOURCE_SPEC,
        "cf167945554df6c2b3f1f647995613987d1dc7beb3bdfc1fe08633d175ad2b37",
        "9c133c415114892acb86093ce630359d2b2e6eeb6de1bf987fbc21ac1e6bf0df",
        "b92b25d74887b7ba8e992bb8ab5e15a08e0a5d4cdcb1104eae0cdf391c5e3d8d")
    verify_seal(M1776,
        "72ef5e9727d6b4a845b61c3dca46b96639d2531afcca340d62f99435dbcdc6ab",
        "acdf0d9c60100971639b45215fc0b4bd9cf1ba49d437e3e61e970621f22be580")
    exact(M1776 / "receipt.json",
          "9b671c96cf8f745199e8a8828e0e2b9e03c376afb4e2df6242f09845c75672a7")
    verify_seal(BASE,
        "22388b70b68f4b038a464446704bdc37fb9f51d536fc12b656b0e51045f5efac",
        "f41253a98d74e7b5087c39f49ddbade856ac825f1286c0c73ccf18bdbc6cd4a2")
    receipt = strict_json(M1776 / "receipt.json")
    if (receipt.get("first_principles_decision", {}).get("paper_primary_energy_comparison") !=
            "k8_vs_equal_bandwidth_k1x8"
            or receipt.get("first_principles_decision", {}).get("k1_energy_in_m1777") is not False
            or receipt.get("retained_k1_dc_diagnostic", {}).get("energy") is not False):
        raise Failure("M1776 two-axis decision drift")
    fixed = {
        HW / "docs/359_DATE终局冻结_20260813.md":
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
        TT_DB: "d8975a427b9f5f6b6667ee5dbc7ff33eac15ab480a871d756af48cd9afa18070",
        SS_DB: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
        VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
        PT: "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
        LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    }
    for path, digest in fixed.items():
        exact(path, digest)
    verify_execution_sources()


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1777_c2_two_axis_equal_bandwidth_energy_work.*",
                    ".m1777_c2_two_axis_equal_bandwidth_energy_stage.*",
                    ".m1777_c2_two_axis_equal_bandwidth_energy_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("private namespace residue: " + pattern)


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
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
            argv = [Path(part.decode(errors="replace")).name
                    for part in (item / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(argv):
            hits.append((item.name, comm, argv[:4]))
    if hits:
        raise Failure("same-UID EDA collision: " + repr(hits))


def resource_gate():
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "SwapFree", "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 24 * 1024 * 1024:
        raise Failure("MemAvailable below 24 GiB")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024:
        raise Failure("SwapFree below 8 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit headroom below 24 GiB")
    if shutil.disk_usage(HW / "results").free < 16 * 1024 * 1024 * 1024:
        raise Failure("result disk free below 16 GiB")


def capture_compile_proxy():
    captured = {name: os.environ.get(name, "") for name in PROXY_KEYS}
    if captured != EXPECTED_PROXY:
        raise Failure("compile-only proxy identity drift")
    try:
        connection = socket.create_connection(("127.0.0.1", 7897), timeout=2.0)
        connection.close()
    except OSError as error:
        raise Failure("VCS compile proxy preflight failed") from error
    return captured


def clean_env(extra, proxy=None):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    if proxy is not None:
        if proxy != EXPECTED_PROXY:
            raise Failure("non-exact proxy rejected")
        value.update(proxy)
    return value


def run(command, cwd, env, timeout, output):
    verify_execution_sources()
    if Path(command[0]).name in {"vcs", "pt_shell"}:
        collision_gate()
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure: " + " ".join(command[:2]))


def identity():
    return {"runner_sha256": sha(RUNNER),
            "source_contract_sha256": sha(CONTRACT),
            "m1778_review_sha256": sha(M1778 / "review.json"),
            "m1779_release_sha256": sha(M1779),
            "m1776_receipt_sha256": sha(M1776 / "receipt.json"),
            "m1661_manifest_sha256": sha(BASE / "SHA256SUMS")}


def load_checker():
    spec = importlib.util.spec_from_file_location("m1777_runtime_checker", str(CHECKER))
    if spec is None or spec.loader is None:
        raise Failure("M1777 checker import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 1:
        raise Failure("M1777 accepts no arguments")
    verify_interpreter()
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        verify_predecessors_and_inputs()
        namespaces_fresh()
        state["phase"] = "QUEUE_WAIT"
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        collision_gate()
        resource_gate()
        verify_execution_sources()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        license_check = subprocess.run(
            [str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=clean_env({}), timeout=60, check=False)
        if license_check.returncode != 0:
            raise Failure("license preflight failed")
        state["phase"] = "PROXY_PREFLIGHT"
        proxy = capture_compile_proxy()
        verify_execution_sources()
        namespaces_fresh()

        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_ATTEMPT_CONSUMED",
            "identity": identity(), "budget": COUNTS, "axes": list(AXES),
            "cases": list(CASES), "accepted_sources_per_axis": sum(EVENTS),
            "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "k1_energy": "NOT_MEASURED__M1753_K1_MAPPED_XZ_FAILURE_DISCLOSED",
            "automatic_retry": False})
        seal_dir(ATTEMPT)
        WORK.mkdir()
        (WORK / "build").mkdir()
        (WORK / "candidate").mkdir()
        checker = load_checker()

        for axis in AXES:
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis
            axis_dir.mkdir()
            state["vcs_compiles"] += 1
            run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
                 "-assert", "svaext", "-debug_access+r", "-lca", "+vcs+lic+wait",
                 "-Mdir=csrc", "-f", str(FILELISTS[axis]), "-top", TOP, "-o", "simv"],
                cwd=axis_dir,
                env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                               "VCS_ARCH_OVERRIDE": "linux"}, proxy),
                timeout=7200, output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file():
                raise Failure("fresh simv absent: " + axis)
            for case_id in CASES:
                state["phase"] = "SIM_" + axis + "_" + str(case_id)
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                log = candidate / (axis + "_case" + str(case_id) + ".log")
                assertion = candidate / (axis + "_case" + str(case_id) + ".assert.report")
                run(["./simv", "-lca", "+M979_UCLI_SAIF", "+M979_CASE=" + str(case_id),
                     "-no_save", "-assert", "report=" + str(assertion),
                     "-ucli", "-i", str(UCLI)], cwd=axis_dir,
                    env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux",
                                   "M1684_SAIF_FILE": str(saif)}),
                    timeout=1800, output=log)
                check = candidate / (axis + "_case" + str(case_id) + ".saif_check.json")
                run([str(PYTHON312), "-I", str(CHECKER), "--mode", "saif",
                     "--axis", axis, "--case", str(case_id),
                     "--cycles", str(CYCLES[axis][case_id]),
                     "--saif", str(saif), "--log", str(log)],
                    cwd=HW, env=clean_env({}), timeout=240, output=check)
                checked = strict_json(check)
                if (checked.get("status") != "PASS_M1777_DIRECTED_COMPONENT_DUT_ONLY_SAIF"
                        or checked.get("accepted_sources") != EVENTS[case_id]):
                    raise Failure("mapped directed SAIF check failed")
                state["saif_files"] += 1

        # This gate intentionally precedes every PTPX launch.
        if any(state[key] != COUNTS[key]
               for key in ("vcs_compiles", "simv_runs", "saif_files")):
            raise Failure("all ten checked SAIF coordinates required before any PTPX")

        rows = []
        for axis in AXES:
            for case_id in CASES:
                state["phase"] = "PTPX_" + axis + "_" + str(case_id)
                state["ptpx_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                pt_dir = candidate / (axis + "_case" + str(case_id) + ".ptpx")
                pt_dir.mkdir()
                run([str(PT), "-f", str(PT_TCL)], cwd=HW,
                    env=clean_env({"DESIGN_NAME": DESIGN, "TT_LIB_DB": str(TT_DB),
                        "SDC_LIB_DB": str(SS_DB),
                        "MAPPED_NETLIST": str(BASE / axis / NET_REL),
                        "MAPPED_SDC": str(BASE / axis / SDC_REL),
                        "GATE_SAIF_FILE": str(saif), "OUTPUT_DIR": str(pt_dir),
                        "SAIF_INSTANCE": SAIF_SCOPE,
                        "SAIF_DURATION_NS": str(CYCLES[axis][case_id] * 3),
                        "MEASUREMENT_CYCLES": str(CYCLES[axis][case_id]),
                        "ACCEPTED_SOURCES": str(EVENTS[case_id]),
                        "AXIS": axis, "CASE_ID": str(case_id),
                        "FAULT_BINARY_CLEAN": "true",
                        "REGISTERED_FAULT_PUBLIC_ZERO": "true"}),
                    timeout=5400, output=pt_dir / "ptpx.log")
                marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
                if (not marker.is_file() or
                        "PASS_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_PTPX" not in marker.read_text()):
                    raise Failure("PTPX completion marker absent")
                power_check = pt_dir / "power_check.json"
                run([str(PYTHON312), "-I", str(CHECKER), "--mode", "power",
                     "--power-report", str(pt_dir / "reports/ptpx_power.rpt")],
                    cwd=HW, env=clean_env({}), timeout=120, output=power_check)
                power = strict_json(power_check)
                rows.append({"axis": axis, "case": case_id,
                             "cycles": CYCLES[axis][case_id],
                             "accepted_sources": EVENTS[case_id], **power})
        if any(state[key] != COUNTS[key] for key in COUNTS):
            raise Failure("execution count drift")

        metrics = checker.aggregate_metrics(rows)
        metrics.update({
            "schema": "m1777_c2_two_axis_equal_bandwidth_energy_metrics_r1_v1",
            "status": "CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "scope": "whole_mapped_C2_component_logic_only_premacro_TT0p9V25C",
            "clock_period_ns": 3.0, "case_rows": rows,
            "external_exclusions": ["weight_sram", "testbench_memory_model",
                "io_phy", "clock_tree", "postlayout_parasitics"]})
        STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in AXES:
            shutil.copy2(WORK / "build" / axis / "compile.log",
                         STAGE / (axis + ".compile.log"))
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1777_c2_two_axis_equal_bandwidth_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity(), "one_shot": {"attempt_consumed": True,
                **COUNTS, "automatic_retry": False},
            "axes": list(AXES), "cases_per_axis": 5,
            "accepted_sources_per_axis": sum(EVENTS),
            "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
            "k1_energy": "NOT_MEASURED__M1753_K1_MAPPED_XZ_FAILURE_DISCLOSED",
            "k1_dc_role": "DIAGNOSTIC_ONLY",
            "same_workloads_and_clock": True,
            "whole_component_report_power": True,
            "equal_bandwidth_joint_disclosure_required": {
                "cycle_speedup_k8_vs_k1x8": 1945.0 / 1913.0,
                "throughput_per_mm2_k8_vs_k1x8":
                    (1945.0 * AREAS_UM2["k1x8"]) /
                    (1913.0 * AREAS_UM2["k8"]),
                "must_be_same_table_and_sentence": True},
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1777_C2_TWO_AXIS_EQUAL_BANDWIDTH_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "counts": dict((key, state[key]) for key in COUNTS),
                    "fault_localization_required_if_xz": ["protocol_error",
                        "numeric_overflow", "stale_response_seen", "endpoint_fault[7:0]"],
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
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


if __name__ == "__main__":
    raise SystemExit(main())
