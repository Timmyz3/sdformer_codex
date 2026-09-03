#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""One-shot M1845 superseding fresh-mapped C2 production-energy campaign.

M1811/M1830 are immutable canonical identities; M1831/M1833 are the frozen
failed predecessor chain. Execution remains inert until a different-author
M1848 PASS review and an exact double-sealed M1849 launch release are supplied.
"""
from datetime import datetime, timezone
import ctypes
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
CHECKER = HW / "system_simulator/scripts/check_m1845_c2_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1845_checker", str(CHECKER))
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("M1845 checker unavailable")
CHECK = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(CHECK)
CONTRACT = CHECK.CONTRACT
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")

# Only the recursively sealed M1811 canonical is admissible.  No partial
# .m1811 work directory, predecessor netlist, SAIF, or PTPX result is reusable.
M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1831_FAILED_CONTRACT = HW / "contracts/m1831_m1830_m1811_c2_fresh_mapped_production_energy_source_contract_r1_20260902.json"
M1833_FAILED_REVIEW = HW / "reviews/m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_r1_20260902"
M1848_SOURCE_REVIEW = HW / "reviews/m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_r1_20260902"
M1849_RELEASE = HW / "contracts/m1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_20260902.json"
M1849_RELEASE_SIDECAR = Path(str(M1849_RELEASE) + ".sha256")
M1849_RELEASE_OUTER = Path(str(M1849_RELEASE) + ".sha256.seal.sha256")

M1811_MANIFEST_SHA256 = "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"
M1811_OUTER_SHA256 = "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b"
M1811_RECEIPT_SHA256 = "3bec6bb629d81a756b5eb9bb4570b04fc1de17a21c0a6143bca6b6c886945e6d"
M1830_MANIFEST_SHA256 = "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06"
M1830_OUTER_SHA256 = "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d"
M1830_REVIEW_SHA256 = "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"
M1831_CONTRACT_SHA256 = "79052bcd5da95aa0ed441f5efbd403d83605b7ae6dd14f1a4c9a3cbaf746fdc3"
M1831_CONTRACT_SIDECAR_SHA256 = "875631f366ce10b6602f7fec1319c921628312a882bb0ca186164bc4b7333d13"
M1831_CONTRACT_OUTER_SHA256 = "61bce3bc6d2dd5155890d00461aafc76b5c649a54806c50f049bcf695a9b4f26"
M1833_REVIEW_SHA256 = "877254bf4de7b25255226e85c535637d2711413a82fc0feb20f2692aac074918"
M1833_MANIFEST_SHA256 = "dc48c4afb7943d7237ceef5cea6aaae3b73620634acd8ed0df98dfd5adcc5884"
M1833_OUTER_SHA256 = "24e66afa58e38cf6680244f970826f2a295005a0da4b6f16f1d71b078bdb1b08"

DESIGN_BASE = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
AXIS_ORDER = ("k8", "k1x8")
CASES = tuple(range(5))
AXES = {
    "k8": {"derived_top": "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE0",
           "cycles": [51, 131, 486, 1231, 14],
           "filelist": CHECK.FILELISTS["k8"],
           "netlist_sha256": "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
           "sdc_sha256": "af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018"},
    "k1x8": {"derived_top": "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_ARCH_MODE1",
             "cycles": [53, 133, 499, 1246, 14],
             "filelist": CHECK.FILELISTS["k1x8"],
             "netlist_sha256": "8698d227f3408b6e40c03bfe9282de458b0ba5cba4e22ec5f0c9bfd4ff16fc1b",
             "sdc_sha256": "1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a"},
}
EVENTS = [20, 41, 90, 110, 0]
TOP = CHECK.TOP
SAIF_SCOPE = CHECK.SAIF_SCOPE
UCLI = CHECK.UCLI
PT_TCL = CHECK.PT_TCL
CELL_TT_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140tt0p9v25c.db")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

COUNTS = {"license_queries": 1, "vcs_compiles": 2, "simv_runs": 10,
          "saif_files": 10, "ptpx_runs": 10}
ATTEMPT = HW / "results/.m1845_c2_fresh_mapped_production_energy_attempt_consumed"
RESULT = HW / "results/m1845_c2_fresh_mapped_production_energy_r1_20260902"
FAILURE = HW / "results/m1845_c2_fresh_mapped_production_energy_r1_20260902.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1845_c2_fresh_mapped_production_energy_r1_20260902.private_build.unsealed_do_not_cite"
WORK = HW / ("results/.m1845_c2_fresh_mapped_production_energy_work." + str(os.getpid()))
STAGE = HW / ("results/.m1845_c2_fresh_mapped_production_energy_stage." + str(os.getpid()))
FAIL_STAGE = HW / ("results/.m1845_c2_fresh_mapped_production_energy_failure_stage." + str(os.getpid()))
QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
LOCAL_LOCK = Path("/tmp/m1845_c2_fresh_mapped_production_energy.lock")


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
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Failure("identity drift " + str(path))


def authority_pin(name):
    value = os.environ.get(name, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Failure("authority pin absent " + name)
    return value


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict: raise Failure("JSON root")
    return value


def verify_directory_seal(root, manifest_sha, outer_sha):
    root = Path(root); manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha); exact(outer, outer_sha)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise Failure("outer seal")
    mapping = {}
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2: raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*")); name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in mapping:
            raise Failure("unsafe manifest")
        exact(root / rel, fields[0]); mapping[name] = fields[0]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if (path.is_file() and path.name not in
                {"SHA256SUMS", "SHA256SUMS.seal.sha256"}):
            actual.add(path.relative_to(root).as_posix())
    if actual != set(mapping):
        raise Failure("manifest not exhaustive")
    return mapping


def verify_file_double_seal(path, file_sha, sidecar_sha, outer_sha):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(path, file_sha); exact(sidecar, sidecar_sha); exact(outer, outer_sha)
    if sidecar.read_text().split() != [sha(path), Path(path).name]:
        raise Failure("sidecar content")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]:
        raise Failure("outer content")


def canonical_inputs():
    mapped = {}
    for axis in AXIS_ORDER:
        root = M1811 / axis / "netlist"
        mapped[axis] = {"netlist": root / (DESIGN_BASE + "_mapped.v"),
                        "sdc": root / (DESIGN_BASE + "_mapped.sdc")}
    return mapped


def verify_authority_and_canonical():
    # M1849 caller pins freeze the executable successor. M1811/M1830 and the
    # failed M1831/M1833 chain are immutable members of the M1845 contract.
    exact(RUNNER, authority_pin("M1845_EXPECTED_RUNNER_SHA256"))
    exact(CONTRACT, authority_pin("M1845_EXPECTED_SOURCE_CONTRACT_SHA256"))
    if (not CONTRACT_SIDECAR.is_file() or CONTRACT_SIDECAR.is_symlink()
            or not CONTRACT_OUTER.is_file() or CONTRACT_OUTER.is_symlink()
            or CONTRACT_SIDECAR.read_text().split() !=
                [sha(CONTRACT), CONTRACT.name]
            or CONTRACT_OUTER.read_text().split() !=
                [sha(CONTRACT_SIDECAR), CONTRACT_SIDECAR.name]):
        raise Failure("M1845 source contract double seal")
    m1811_map = verify_directory_seal(
        M1811, M1811_MANIFEST_SHA256, M1811_OUTER_SHA256)
    exact(M1811 / "receipt.json", M1811_RECEIPT_SHA256)
    m1830_map = verify_directory_seal(
        M1830, M1830_MANIFEST_SHA256, M1830_OUTER_SHA256)
    exact(M1830 / "review.json", M1830_REVIEW_SHA256)
    if m1811_map.get("receipt.json") != sha(M1811 / "receipt.json"):
        raise Failure("M1811 receipt not sealed")
    if m1830_map.get("review.json") != sha(M1830 / "review.json"):
        raise Failure("M1830 review not sealed")
    review = strict_json(M1830 / "review.json")
    if (review.get("status") !=
            "PASS_M1830_M1811_C2_REGISTERED_FAULT_MATCHED_TWO_AXIS_DC_RESULT_HAMMER__P0_0_P1_0_P2_0__SETUP_AREA_ADMITTED"
            or review.get("severity_counts") != {"p0": 0, "p1": 0, "p2": 0}):
        raise Failure("M1830 not admitted")
    verify_file_double_seal(
        M1831_FAILED_CONTRACT, M1831_CONTRACT_SHA256,
        M1831_CONTRACT_SIDECAR_SHA256, M1831_CONTRACT_OUTER_SHA256)
    failed_map = verify_directory_seal(
        M1833_FAILED_REVIEW, M1833_MANIFEST_SHA256, M1833_OUTER_SHA256)
    exact(M1833_FAILED_REVIEW / "review.json", M1833_REVIEW_SHA256)
    if failed_map.get("review.json") != M1833_REVIEW_SHA256:
        raise Failure("M1833 failed review not sealed")
    failed_review = strict_json(M1833_FAILED_REVIEW / "review.json")
    if (failed_review.get("schema") !=
            "m1833_m1831_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1"
            or failed_review.get("status") !=
            "FAIL_CLOSED_M1833_M1831_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_3_P2_0__NO_EDA_NO_LICENSE_NO_RELEASE"
            or failed_review.get("severity_counts") !=
            {"p0": 0, "p1": 3, "p2": 0}):
        raise Failure("M1833 failure chain drift")
    mapped = canonical_inputs()
    for axis in AXIS_ORDER:
        exact(mapped[axis]["netlist"], AXES[axis]["netlist_sha256"])
        exact(mapped[axis]["sdc"], AXES[axis]["sdc_sha256"])
        modules = re.findall(r"(?m)^\s*module\s+([^\s(]+)",
                             mapped[axis]["netlist"].read_text(errors="strict"))
        if modules.count(AXES[axis]["derived_top"]) != 1:
            raise Failure("derived top absent/duplicate " + axis)
    review_map = verify_directory_seal(
        M1848_SOURCE_REVIEW,
        authority_pin("M1845_EXPECTED_M1848_MANIFEST_SHA256"),
        authority_pin("M1845_EXPECTED_M1848_OUTER_FILE_SHA256"))
    exact(M1848_SOURCE_REVIEW / "review.json",
          authority_pin("M1845_EXPECTED_M1848_REVIEW_SHA256"))
    if review_map.get("review.json") != sha(M1848_SOURCE_REVIEW / "review.json"):
        raise Failure("M1848 source review not sealed")
    source_review = strict_json(M1848_SOURCE_REVIEW / "review.json")
    if (source_review.get("schema") !=
            "m1848_m1845_c2_fresh_mapped_production_energy_source_hammer_review_r1_v1"
            or source_review.get("status") !=
            "PASS_M1848_M1845_C2_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__P0_0_P1_0_P2_0__AUTHORIZED_FOR_M1849_RELEASE"
            or source_review.get("severity_counts") !=
            {"p0": 0, "p1": 0, "p2": 0}
            or source_review.get("reviewer_identity") ==
            "/root/m1804_c2_two_vcs_result_hammer"
            or source_review.get("authorization") != {
                "create_m1849_release": True,
                "run_m1845_vcs": False,
                "run_simv": False,
                "run_ptpx": False,
                "query_license": False,
                "create_attempt": False,
                "create_result": False,
                "automatic_retry": False}):
        raise Failure("M1848 source review schema/status/severity/authorization")
    verify_file_double_seal(
        M1849_RELEASE, authority_pin("M1845_EXPECTED_M1849_RELEASE_SHA256"),
        authority_pin("M1845_EXPECTED_M1849_SIDECAR_SHA256"),
        authority_pin("M1845_EXPECTED_M1849_OUTER_FILE_SHA256"))
    release = strict_json(M1849_RELEASE)
    if (release.get("schema") !=
            "m1849_m1848_m1845_c2_fresh_mapped_production_energy_launch_release_r1_v1"
            or release.get("status") !=
            "AUTHORIZE_ONE_M1845_C2_FRESH_MAPPED_ENERGY_CAMPAIGN"):
        raise Failure("M1849 release status")
    expected_release_identity = {
        "runner_sha256": sha(RUNNER),
        "source_contract_sha256": sha(CONTRACT),
        "source_contract_sidecar_sha256": sha(CONTRACT_SIDECAR),
        "source_contract_outer_file_sha256": sha(CONTRACT_OUTER),
        "source_review_json_sha256": sha(M1848_SOURCE_REVIEW / "review.json"),
        "source_review_manifest_sha256": sha(M1848_SOURCE_REVIEW / "SHA256SUMS"),
        "source_review_outer_file_sha256": sha(
            M1848_SOURCE_REVIEW / "SHA256SUMS.seal.sha256"),
        "m1811_receipt_sha256": M1811_RECEIPT_SHA256,
        "m1811_manifest_sha256": M1811_MANIFEST_SHA256,
        "m1811_outer_file_sha256": M1811_OUTER_SHA256,
        "m1830_review_sha256": M1830_REVIEW_SHA256,
        "m1830_manifest_sha256": M1830_MANIFEST_SHA256,
        "m1830_outer_file_sha256": M1830_OUTER_SHA256,
        "m1831_failed_contract_sha256": M1831_CONTRACT_SHA256,
        "m1831_failed_contract_sidecar_sha256": M1831_CONTRACT_SIDECAR_SHA256,
        "m1831_failed_contract_outer_file_sha256": M1831_CONTRACT_OUTER_SHA256,
        "m1833_failed_review_sha256": M1833_REVIEW_SHA256,
        "m1833_failed_manifest_sha256": M1833_MANIFEST_SHA256,
        "m1833_failed_outer_file_sha256": M1833_OUTER_SHA256,
        "k8_mapped_netlist_sha256": AXES["k8"]["netlist_sha256"],
        "k8_mapped_sdc_sha256": AXES["k8"]["sdc_sha256"],
        "k1x8_mapped_netlist_sha256": AXES["k1x8"]["netlist_sha256"],
        "k1x8_mapped_sdc_sha256": AXES["k1x8"]["sdc_sha256"],
    }
    if release.get("identity") != expected_release_identity:
        raise Failure("M1849 transitive identity")
    if release.get("prelaunch_claim_boundary") != CHECK.CLAIMS:
        raise Failure("M1849 prelaunch claim boundary")
    if release.get("fresh_execution_budget") != dict(
            COUNTS, automatic_retry=False, reuse_prior_simv=False):
        raise Failure("M1849 release budget")
    if release.get("authorization") != {
            "launch_m1845_once": True,
            "automatic_retry": False,
            "publish_only_after_all_gates": True,
            "independent_result_hammer_required": True}:
        raise Failure("M1849 authorization")
    return mapped


def namespaces_fresh():
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(str(path)): raise Failure("namespace residue " + str(path))


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit(): continue
        try:
            if item.stat().st_uid != os.getuid(): continue
            comm = (item / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked: hits.append((item.name, comm))
    if hits: raise Failure("same-UID EDA collision " + repr(hits))


def resource_gate():
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "SwapFree",
                                    "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 24 * 1024 * 1024: raise Failure("memory")
    if values.get("SwapFree", 0) < 8 * 1024 * 1024: raise Failure("swap")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit")
    if shutil.disk_usage(HW / "results").free < 20 * 1024 * 1024 * 1024:
        raise Failure("disk")


def clean_env(extra):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER,
             "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra); return value


def run(command, cwd, env, timeout, output, record_command=False):
    CHECK.validate_sources(); collision_gate()
    with Path(output).open("wb") as stream:
        if record_command:
            stream.write(("M1845_COMMAND_JSON=" + json.dumps(
                list(command), separators=(",", ":"), ensure_ascii=True)
                + "\n").encode("ascii"))
            stream.flush()
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   check=False)
    if completed.returncode != 0: raise Failure("tool failure " + Path(command[0]).name)


def compile_command(mapped, axis):
    return [str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            "-assert", "svaext", "-debug_access+r", "-lca", "+vcs+lic+wait",
            "-Mdir=csrc", str(mapped[axis]["netlist"]), "-f",
            str(AXES[axis]["filelist"]), "-top", TOP, "-o", "simv"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_dir(root):
    rows = []
    for path in Path(root).rglob("*"):
        if path.is_symlink(): raise Failure("symlink")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort(); manifest = Path(root) / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n" for name, digest in rows))
    (Path(root) / "SHA256SUMS.seal.sha256").write_text(sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error), str(destination))


def main():
    if len(sys.argv) != 1: raise Failure("M1845 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
             "saif_files": 0, "ptpx_runs": 0}
    queue_handle = QUEUE.open("a+"); local_handle = LOCAL_LOCK.open("a+")
    try:
        CHECK.validate_sources()
        mapped = verify_authority_and_canonical()
        namespaces_fresh()
        fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX)
        fcntl.flock(local_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate(); resource_gate(); namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"; state["license_queries"] += 1
        probe = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                               env=clean_env({}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60, check=False)
        if probe.returncode != 0: raise Failure("license preflight")
        ATTEMPT.mkdir(); state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {"status": "M1845_ATTEMPT_CONSUMED",
            "budget": COUNTS, "automatic_retry": False, "reuse_prior_simv": False})
        seal_dir(ATTEMPT)
        WORK.mkdir(); (WORK / "build").mkdir(); (WORK / "candidate").mkdir()
        builds = {}
        for axis in AXIS_ORDER:
            state["phase"] = "COMPILE_" + axis; state["vcs_compiles"] += 1
            build = WORK / "build" / axis; build.mkdir(); builds[axis] = build
            command = compile_command(mapped, axis)
            run(command,
                build, clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                  "VCS_ARCH_OVERRIDE": "linux"}), 7200,
                build / "compile.log", record_command=True)
            CHECK.validate_compile_log(build / "compile.log", axis, command)
            if not (build / "simv").is_file(): raise Failure("simv absent " + axis)

        saif_rows = []
        for axis in AXIS_ORDER:
            for case_id in CASES:
                state["phase"] = "SIM_" + axis + "_" + str(case_id)
                state["simv_runs"] += 1
                candidate = WORK / "candidate"
                saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                log = candidate / (axis + "_case" + str(case_id) + ".log")
                assertion = candidate / (axis + "_case" + str(case_id) + ".assert.report")
                run(["./simv", "-lca", "+M979_UCLI_SAIF", "+M979_CASE=" + str(case_id),
                     "-no_save", "-assert", "report=" + str(assertion),
                     "-ucli", "-i", str(UCLI)], builds[axis],
                    clean_env({"M1831_SAIF_FILE": str(saif)}), 1800, log)
                checked = CHECK.validate_saif(saif, axis, case_id,
                                              AXES[axis]["cycles"][case_id])
                checked["runtime"] = CHECK.validate_runtime_log(log, axis, case_id)
                state["saif_files"] += 1; saif_rows.append(checked)

        # all ten mapped SAIF coordinates required before PTPX
        if state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or state["saif_files"] != 10:
            raise Failure("mapped SAIF completeness gate")

        power_rows = []
        for axis in AXIS_ORDER:
            for case_id in CASES:
                state["phase"] = "PTPX_" + axis + "_" + str(case_id)
                state["ptpx_runs"] += 1
                candidate = WORK / "candidate"
                pt_dir = candidate / (axis + "_case" + str(case_id) + ".ptpx")
                pt_dir.mkdir(); saif = candidate / (axis + "_case" + str(case_id) + ".saif")
                cycles = AXES[axis]["cycles"][case_id]
                run([str(PT), "-f", str(PT_TCL)], HW, clean_env({
                    "M1831_DESIGN_NAME": AXES[axis]["derived_top"],
                    "M1831_TT_LIB_DB": str(CELL_TT_DB),
                    "M1831_MAPPED_NETLIST": str(mapped[axis]["netlist"]),
                    "M1831_MAPPED_SDC": str(mapped[axis]["sdc"]),
                    "M1831_GATE_SAIF": str(saif),
                    "M1831_OUTPUT_DIR": str(pt_dir),
                    "M1831_SAIF_INSTANCE": SAIF_SCOPE,
                    "M1831_MEASUREMENT_CYCLES": str(cycles),
                    "M1831_ACCEPTED_SOURCES": str(EVENTS[case_id]),
                    "M1831_AXIS": axis, "M1831_CASE_ID": str(case_id)}),
                    5400, pt_dir / "ptpx.log")
                marker = pt_dir / "PTPX_INTERNAL_COMPLETE.txt"
                if not marker.is_file() or "PASS_M1831_C2_FRESH_MAPPED_PRODUCTION_PTPX_PENDING_RESULT_HAMMER" not in marker.read_text():
                    raise Failure("PTPX marker")
                power = CHECK.parse_power_report(pt_dir / "reports/power.rpt")
                power_rows.append({"axis": axis, "case": case_id,
                                   "cycles": cycles,
                                   "accepted_sources": EVENTS[case_id], **power})
        if any(state[key] != value for key, value in COUNTS.items()):
            raise Failure("execution count drift")
        metrics = CHECK.aggregate_metrics(power_rows)
        STAGE.mkdir(); shutil.copytree(WORK / "candidate", STAGE / "candidate")
        compile_evidence = STAGE / "compile_logs"; compile_evidence.mkdir()
        compile_rows = []
        for axis in AXIS_ORDER:
            source_log = builds[axis] / "compile.log"
            sealed_log = compile_evidence / (axis + ".compile.log")
            shutil.copy2(str(source_log), str(sealed_log))
            compile_rows.append(CHECK.validate_compile_log(
                sealed_log, axis, compile_command(mapped, axis)))
        write_json(STAGE / "compile_log_rows.json", compile_rows)
        write_json(STAGE / "saif_rows.json", saif_rows)
        write_json(STAGE / "metrics.json", metrics)
        write_json(STAGE / "receipt.json", {
            "schema": "m1845_c2_fresh_mapped_energy_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "one_shot": dict(COUNTS, automatic_retry=False, reuse_prior_simv=False),
            "identity": {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "source_review_json_sha256": sha(M1848_SOURCE_REVIEW / "review.json"),
                "launch_release_sha256": sha(M1849_RELEASE),
                "m1811_receipt_sha256": M1811_RECEIPT_SHA256,
                "m1811_manifest_sha256": M1811_MANIFEST_SHA256,
                "m1811_outer_file_sha256": M1811_OUTER_SHA256,
                "m1830_review_sha256": M1830_REVIEW_SHA256,
                "m1830_manifest_sha256": M1830_MANIFEST_SHA256,
                "m1830_outer_file_sha256": M1830_OUTER_SHA256,
                "m1831_failed_contract_sha256": M1831_CONTRACT_SHA256,
                "m1833_failed_review_sha256": M1833_REVIEW_SHA256,
                "k8_mapped_netlist_sha256": AXES["k8"]["netlist_sha256"],
                "k8_mapped_sdc_sha256": AXES["k8"]["sdc_sha256"],
                "k1x8_mapped_netlist_sha256": AXES["k1x8"]["netlist_sha256"],
                "k1x8_mapped_sdc_sha256": AXES["k1x8"]["sdc_sha256"]},
            "derived_tops": dict((axis, AXES[axis]["derived_top"]) for axis in AXIS_ORDER),
            "compile_logs": compile_rows,
            "metrics": metrics, "claim_boundary": dict(CHECK.CLAIMS)})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1845_C2_FRESH_MAPPED_ENERGY_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        CHECK.validate_sealed_result_stage(STAGE)
        publish_no_replace(WORK, PRIVATE)
        publish_no_replace(STAGE, RESULT); state["complete"] = True
        return 0
    except BaseException as error:
        if state["attempt"] and not state["complete"]:
            try:
                FAIL_STAGE.mkdir(); write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_RETRY",
                    "phase": state["phase"], "error": type(error).__name__,
                    "counts": dict((key, state[key]) for key in COUNTS),
                    "automatic_retry": False})
                seal_dir(FAIL_STAGE); publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException: pass
            if WORK.is_dir() and not PRIVATE.exists():
                try: publish_no_replace(WORK, PRIVATE)
                except BaseException: pass
        raise
    finally:
        local_handle.close(); queue_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
