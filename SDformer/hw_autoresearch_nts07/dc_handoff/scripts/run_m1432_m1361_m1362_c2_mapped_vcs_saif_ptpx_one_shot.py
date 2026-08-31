#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Exact one-shot M1432 C2 mapped VCS -> SAIF -> PTPX executor.

This file is inert unless invoked after a fresh M1440 final hammer supplies all
eight exact external SHA pins.  Source authoring and tests never invoke it.
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
import signal
import stat
import subprocess
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
AUTHORITY = HW / "contracts/m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_authority_r1_20260831.json"
M1361_CHECKER = HW / "verif_m1361_c2_activity_final_launch_exact/static_check_m1361_c2_activity_final_launch_exact_source.py"
M1361_TEST = HW / "verif_m1361_c2_activity_final_launch_exact/test_m1361_c2_activity_final_launch_exact_source.py"
M1361_CONTRACT = HW / "contracts/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_contract_r1_20260831.json"
M1361_AUTHOR = HW / "reviews/m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_author_r1_20260831"
M1362 = HW / "reviews/m1362_m1361_c2_mapped_activity_vcs_saif_final_launch_exact_source_blind_hammer_r1_20260831"
M1440 = HW / "reviews/m1440_m1432_m1361_m1362_c2_mapped_vcs_saif_ptpx_final_launch_hammer_r1_20260831"

SOURCE_CHECKER = HW / "system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
CELL_MODEL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
RESET_MEMORY_MODEL = HW / "dc_handoff/tb/m1334_c2_production_activity_reset_safe_memory_model.sv"
CASE_TB = HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv"
ASSERTIONS = HW / "dc_handoff/tb/m1334_c2_production_activity_assertions.sv"
MAPPED_TB = HW / "dc_handoff/tb/tb_m1334_c2_headline_mapped_production_activity.sv"
FILELIST = {
    "k8": HW / "dc_handoff/filelists/date_m1334_c2_k8_mapped_production_activity.f",
    "k1x8": HW / "dc_handoff/filelists/date_m1334_c2_k1x8_mapped_production_activity.f",
}
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
PTPX_TCL = HW / "dc_handoff/scripts/run_ptpx.tcl"
M872 = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
DESIGN = "m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24"
TB_TOP = "tb_m1334_c2_headline_mapped_production_activity"
SAIF_INSTANCE = TB_TOP + ".core.dut"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
LIB_DB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
LICENSE_SERVER = "27030@ic.ismd-nemo"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")

ATTEMPT = HW / "results/.m1432_c2_mapped_vcs_saif_ptpx_attempt_consumed"
RESULT = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831"
FAILURE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.failed_or_incomplete.quarantine"
PRIVATE = HW / "results/m1432_c2_mapped_vcs_saif_ptpx_r1_20260831.private_build.unsealed_do_not_cite"
WORK = HW / f"results/.m1432_c2_mapped_vcs_saif_ptpx_work.{os.getpid()}"
STAGE = HW / f"results/.m1432_c2_mapped_vcs_saif_ptpx_result_stage.{os.getpid()}"
FAIL_STAGE = HW / f"results/.m1432_c2_mapped_vcs_saif_ptpx_failure_stage.{os.getpid()}"
LOCK = Path("/tmp/m1432_c2_mapped_vcs_saif_ptpx.lock")

CYCLES = {"k8": [51, 131, 486, 1231, 14],
          "k1x8": [53, 133, 499, 1246, 14]}
EVENTS = [20, 41, 90, 110, 0]
STATIC_SHA = {
    "m1361_checker": "13a98be09ec5e00d5f6ec7f07e53f27bc2d66c5d72d11b778c19e5a511422745",
    "m1361_test": "2938595d4192528e05b1aea22201f4086f35a5789756348e2d9034f35afdc8dd",
    "m1361_contract": "fb2e5f83a4befef0252a030402c2e18f8babc336e326d30f7d91d90969c00c9a",
    "m1361_review": "d4369a78849b7f3f7411cc1c21365e17450275b01ed906468c368781b140126c",
    "m1361_manifest": "e00f9cfc6222c92ecd7f6b7e0ca7d0f1c46204634f208cdac3545e707e4edaaa",
    "m1361_outer": "634258227ac5143d820fa696ed8cb572f8c622d7b4ad8e3c0db404a0b2adbdaf",
    "m1362_review": "dafe39f181c85c1b08c7aaaaee29039005ec6a6b55386f2a2755aabca3f441b5",
    "m1362_manifest": "b546b35fbed2b0a8966b66ee34c22f0f72c93db00e5248c9808c0eda40360dd5",
    "m1362_outer": "32dae68fe7bdca213619ca19e2361799873e91b87f5e1b75e2402201bc71e4bb",
    "source_checker": "c9326ff934239e8773e9f991e6bf0be94bba9c9c602be199433c22d1cd4c9da9",
    "cell_model": "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
    "reset_memory_model": "f9b0d87dd3b951a24b79545555c09b32bbce695e85cc71df2948e5065981c7c3",
    "case_tb": "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    "assertions": "86be3fa541bf65afa6ada99aa3e2bd494ed689594fece18cfea135b91420c32a",
    "mapped_tb": "eacc165bad9eb3ef6c38e87f6f0de8cafd75e167f0ef02d340647634540982ca",
    "filelist_k8": "9030ca8f6e42a21546332f25009e08033e6a6740f5d95fd8c5a36f190ac00e6d",
    "filelist_k1x8": "cca8a9b0bfe0c32d85f554994ab61c2b78dba425e6dee194fe9f1557b54998e9",
    "ucli": "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
    "ptpx_tcl": "879398c8b8708589d42346af10d4825afac19c7c0622601685d1ea3f72245368",
    "vcs": "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    "pt": "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    "lmutil": "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "lib_db": "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "k8_netlist": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    "k8_sdc": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
    "k1x8_netlist": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
    "k1x8_sdc": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
}
ENV_SHA = (
    "M1432_EXPECTED_RUNNER_SHA256", "M1432_EXPECTED_AUTHORITY_SHA256",
    "M1432_EXPECTED_M1440_REVIEW_SHA256", "M1432_EXPECTED_M1440_MANIFEST_SHA256",
    "M1432_EXPECTED_M1440_OUTER_FILE_SHA256",
)
IDENTITY_KEYS = (
    "m1361_checker_sha256", "m1361_test_sha256", "m1361_contract_sha256",
    "m1361_contract_digest_file_sha256", "m1361_contract_outer_file_sha256",
    "m1361_author_review_sha256", "m1361_author_manifest_sha256",
    "m1361_author_outer_file_sha256", "m1362_review_sha256",
    "m1362_manifest_sha256", "m1362_outer_file_sha256",
    "m1432_authority_sha256", "m1440_review_sha256", "m1440_manifest_sha256",
    "m1440_outer_file_sha256", "mapped_runner_sha256", "ptpx_tcl_sha256",
)
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power", "energy",
    "performance", "system_speedup", "paper_ppa_ready", "headline")}


class Failure(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(path: Path, digest: str) -> None:
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Failure("identity drift: " + str(path))


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output: raise Failure("duplicate JSON key: " + key)
            output[key] = value
        return output
    exact_type = path.is_file() and not path.is_symlink()
    if not exact_type: raise Failure("JSON absent/nonregular: " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON: " + token)))
    if type(value) is not dict: raise Failure("JSON root is not object")
    return value


def verify_dir(root: Path, review_sha: str, manifest_sha: str, outer_sha: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink(): raise Failure("sealed directory invalid")
    exact(root / "review.json", review_sha); exact(root / "SHA256SUMS", manifest_sha)
    exact(root / "SHA256SUMS.seal.sha256", outer_sha)
    if (root / "SHA256SUMS.seal.sha256").read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content drift")
    listed = set()
    for row in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts: raise Failure("manifest row")
        exact(root / rel, digest); listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if actual != listed: raise Failure("sealed directory population drift")
    return strict_json(root / "review.json")


def seal_dir(root: Path) -> None:
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink(): raise Failure("symlink in result")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            if not stat.S_ISREG(path.stat().st_mode): raise Failure("nonregular result")
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort(); manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n")


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True); renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error), str(destination))


def collision_gate() -> None:
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "icc2_shell", "common_shell_exec",
               "common_shell_exe"}
    ancestry = set(); pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try: pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception: break
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) in ancestry: continue
        try:
            if item.stat().st_uid != os.getuid(): continue
            comm = (item / "comm").read_text().strip()
            argv = [Path(part.decode(errors="replace")).name
                    for part in (item / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(argv): hits.append((item.name, comm, argv[:4]))
    if hits: raise Failure("same-UID EDA collision: " + repr(hits))


def namespaces_fresh() -> None:
    for path in (ATTEMPT, RESULT, FAILURE, PRIVATE, WORK, STAGE, FAIL_STAGE):
        if os.path.lexists(path): raise Failure("namespace residue: " + str(path))
    for pattern in (".m1432_c2_mapped_vcs_saif_ptpx_work.*",
                    ".m1432_c2_mapped_vcs_saif_ptpx_result_stage.*",
                    ".m1432_c2_mapped_vcs_saif_ptpx_failure_stage.*"):
        if next((HW / "results").glob(pattern), None) is not None:
            raise Failure("stale private namespace: " + pattern)


def resource_gate() -> None:
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    minimum = 16 * 1024 * 1024
    if values.get("MemAvailable", 0) < minimum or \
            values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < minimum:
        raise Failure("resource preflight below 16 GiB")


def identity() -> dict[str, str]:
    return {
        "m1361_checker_sha256": STATIC_SHA["m1361_checker"],
        "m1361_test_sha256": STATIC_SHA["m1361_test"],
        "m1361_contract_sha256": STATIC_SHA["m1361_contract"],
        "m1361_contract_digest_file_sha256": sha(Path(str(M1361_CONTRACT) + ".sha256")),
        "m1361_contract_outer_file_sha256": sha(Path(str(M1361_CONTRACT) + ".sha256.seal.sha256")),
        "m1361_author_review_sha256": STATIC_SHA["m1361_review"],
        "m1361_author_manifest_sha256": STATIC_SHA["m1361_manifest"],
        "m1361_author_outer_file_sha256": STATIC_SHA["m1361_outer"],
        "m1362_review_sha256": STATIC_SHA["m1362_review"],
        "m1362_manifest_sha256": STATIC_SHA["m1362_manifest"],
        "m1362_outer_file_sha256": STATIC_SHA["m1362_outer"],
        "m1432_authority_sha256": sha(AUTHORITY),
        "m1440_review_sha256": sha(M1440 / "review.json"),
        "m1440_manifest_sha256": sha(M1440 / "SHA256SUMS"),
        "m1440_outer_file_sha256": sha(M1440 / "SHA256SUMS.seal.sha256"),
        "mapped_runner_sha256": sha(RUNNER),
        "ptpx_tcl_sha256": STATIC_SHA["ptpx_tcl"],
    }


def verify_authority() -> None:
    expected = {name: os.environ.get(name, "") for name in ENV_SHA}
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in expected.values()):
        raise Failure("required exact SHA environment absent")
    exact(RUNNER, expected["M1432_EXPECTED_RUNNER_SHA256"])
    exact(AUTHORITY, expected["M1432_EXPECTED_AUTHORITY_SHA256"])
    authority_sum = Path(str(AUTHORITY) + ".sha256")
    authority_outer = Path(str(authority_sum) + ".seal.sha256")
    if authority_sum.read_text() != f"{sha(AUTHORITY)}  {AUTHORITY.name}\n" or \
            authority_outer.read_text() != f"{sha(authority_sum)}  {authority_sum.name}\n":
        raise Failure("M1432 authority recursive sidecar drift")
    exact(M1361_CHECKER, STATIC_SHA["m1361_checker"]); exact(M1361_TEST, STATIC_SHA["m1361_test"])
    exact(M1361_CONTRACT, STATIC_SHA["m1361_contract"])
    verify_dir(M1361_AUTHOR, STATIC_SHA["m1361_review"], STATIC_SHA["m1361_manifest"],
               STATIC_SHA["m1361_outer"])
    blind = verify_dir(M1362, STATIC_SHA["m1362_review"], STATIC_SHA["m1362_manifest"],
                       STATIC_SHA["m1362_outer"])
    final = verify_dir(M1440, expected["M1432_EXPECTED_M1440_REVIEW_SHA256"],
                       expected["M1432_EXPECTED_M1440_MANIFEST_SHA256"],
                       expected["M1432_EXPECTED_M1440_OUTER_FILE_SHA256"])
    authority = strict_json(AUTHORITY)
    if authority.get("status") != "AUTHORIZE_AT_MOST_ONE_C2_MAPPED_VCS_SAIF_PTPX_ATTEMPT__FRESH_M1440_REQUIRED":
        raise Failure("authority status drift")
    if authority.get("identity", {}).get("mapped_activity_runner_sha256") != sha(RUNNER):
        raise Failure("authority runner binding drift")
    if blind.get("status") != "PASS_M1361_EXACT_SOURCE__FINAL_LAUNCH_AUTHORITY_AUTHORING_ONLY":
        raise Failure("M1362 status drift")
    if final.get("status") != "PASS_M1440_AUTHORIZE_ONE_M1432_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH" or \
            final.get("authorization") != {"launch": True, "campaigns": 1,
                                            "automatic_retry": False}:
        raise Failure("M1440 final authorization drift")
    required_bindings = {"runner_sha256": sha(RUNNER), "authority_sha256": sha(AUTHORITY),
                         "m1362_review_sha256": STATIC_SHA["m1362_review"],
                         "m1362_manifest_sha256": STATIC_SHA["m1362_manifest"],
                         "m1362_outer_file_sha256": STATIC_SHA["m1362_outer"]}
    if final.get("bindings") != required_bindings or final.get("claim_boundary") != CLAIMS:
        raise Failure("M1440 bindings/claims drift")
    for path, digest in ((SOURCE_CHECKER, STATIC_SHA["source_checker"]),
                         (CELL_MODEL, STATIC_SHA["cell_model"]),
                         (RESET_MEMORY_MODEL, STATIC_SHA["reset_memory_model"]),
                         (CASE_TB, STATIC_SHA["case_tb"]),
                         (ASSERTIONS, STATIC_SHA["assertions"]),
                         (MAPPED_TB, STATIC_SHA["mapped_tb"]),
                         (FILELIST["k8"], STATIC_SHA["filelist_k8"]),
                         (FILELIST["k1x8"], STATIC_SHA["filelist_k1x8"]),
                         (UCLI, STATIC_SHA["ucli"]), (PTPX_TCL, STATIC_SHA["ptpx_tcl"]),
                         (VCS, STATIC_SHA["vcs"]), (PT, STATIC_SHA["pt"]),
                         (LMUTIL, STATIC_SHA["lmutil"]), (PYTHON, STATIC_SHA["python"]),
                         (LIB_DB, STATIC_SHA["lib_db"]), (DOCS359, STATIC_SHA["docs359"])):
        exact(path, digest)
    for axis in ("k8", "k1x8"):
        netlist = M872 / axis / "netlist" / f"{DESIGN}_mapped.v"
        sdc = M872 / axis / "netlist" / f"{DESIGN}_mapped.sdc"
        exact(netlist, STATIC_SHA[axis + "_netlist"]); exact(sdc, STATIC_SHA[axis + "_sdc"])
    if list(identity()) != list(IDENTITY_KEYS): raise Failure("receipt identity order drift")


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER, "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra); return value


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
        output: Path) -> None:
    with output.open("wb") as stream:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if completed.returncode != 0: raise Failure("tool failure: " + " ".join(command[:2]))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) != 1: raise Failure("M1432 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False, "complete": False,
             "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0, "ptpx_runs": 0}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority(); state["identity"] = identity(); namespaces_fresh()
        # Both exact same-UID collision gates occur before lmstat or any EDA tool.
        collision_gate()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate(); resource_gate(); namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        if not LICENSE_FILE.is_file() or LICENSE_FILE.is_symlink(): raise Failure("license file invalid")
        license_log = subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", LICENSE_SERVER],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     timeout=60, check=False)
        if license_log.returncode != 0: raise Failure("license preflight failed")
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir(); state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1432_ATTEMPT_CONSUMED", "identity": identity(),
            "campaigns": 1, "automatic_retry": False,
            "budget": {"vcs_compiles": 2, "simv_runs": 10,
                       "saif_files": 10, "ptpx_runs": 10}})
        seal_dir(ATTEMPT)
        WORK.mkdir(); (WORK / "build").mkdir(); (WORK / "candidate").mkdir()
        for axis in ("k8", "k1x8"):
            state["phase"] = "COMPILE_" + axis
            axis_dir = WORK / "build" / axis; axis_dir.mkdir()
            state["vcs_compiles"] += 1
            run([str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
                 "-assert", "svaext", "+vcs+lic+wait", "-Mdir=csrc", "-f",
                 str(FILELIST[axis]), "-top", TB_TOP, "-o", "simv"], cwd=axis_dir,
                env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                               "VCS_ARCH_OVERRIDE": "linux"}), timeout=1800,
                output=axis_dir / "compile.log")
            if not (axis_dir / "simv").is_file(): raise Failure("simv absent: " + axis)
            for case in range(5):
                state["phase"] = f"SIM_{axis}_{case}"; state["simv_runs"] += 1
                candidate = WORK / "candidate"; saif = candidate / f"{axis}_case{case}.saif"
                log = candidate / f"{axis}_case{case}.log"
                report = candidate / f"{axis}_case{case}.assert.report"
                run(["./simv", "+M979_UCLI_SAIF", f"+M979_CASE={case}", "-no_save",
                     "-assert", f"report={report}", "-ucli", "-i", str(UCLI)], cwd=axis_dir,
                    env=clean_env({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                                   "VCS_ARCH_OVERRIDE": "linux", "M1334_SAIF_FILE": str(saif)}),
                    timeout=600, output=log)
                display = "K8" if axis == "k8" else "K1x8"; cycles = CYCLES[axis][case]
                expected = (f"PASS M979 mapped replay axis={display} case={case} "
                            f"events={EVENTS[case]} cycles={cycles} saif_duration_ns={cycles*3} "
                            "numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 "
                            "accepted_unknowns=0 protocol_errors=0")
                if expected not in log.read_text(errors="replace"): raise Failure("PASS mismatch")
                check = candidate / f"{axis}_case{case}.saif_check.json"
                run([str(PYTHON), "-I", str(SOURCE_CHECKER), "--saif", str(saif),
                     "--axis", axis, "--case", str(case), "--cycles", str(cycles)],
                    cwd=HW, env=clean_env({}), timeout=120, output=check)
                state["saif_files"] += 1

        # No PTPX may start until every mapped replay and all ten SAIF gates pass.
        if state["vcs_compiles"] != 2 or state["simv_runs"] != 10 or \
                state["saif_files"] != 10:
            raise Failure("mapped VCS/SAIF campaign incomplete before PTPX")
        for axis in ("k8", "k1x8"):
            for case in range(5):
                candidate = WORK / "candidate"
                saif = candidate / f"{axis}_case{case}.saif"
                state["phase"] = f"PTPX_{axis}_{case}"; state["ptpx_runs"] += 1
                pt_dir = candidate / f"{axis}_case{case}.ptpx"; pt_dir.mkdir()
                netlist = M872 / axis / "netlist" / f"{DESIGN}_mapped.v"
                sdc = M872 / axis / "netlist" / f"{DESIGN}_mapped.sdc"
                run([str(PT), "-f", str(PTPX_TCL)], cwd=HW,
                    env=clean_env({"DESIGN_NAME": DESIGN, "LIB_DB": str(LIB_DB),
                                   "MAPPED_NETLIST": str(netlist), "MAPPED_SDC": str(sdc),
                                   "SAIF_FILE": str(saif), "OUTPUT_DIR": str(pt_dir),
                                   "OPERATING_CONDITION": "ssg0p9v125c",
                                   "CORNER_ROLE": "slow_prelayout_power",
                                   "SAIF_INSTANCE": SAIF_INSTANCE}), timeout=1800,
                    output=pt_dir / "ptpx.log")
                required = ("ptpx_check_power.rpt", "ptpx_power.rpt",
                            "ptpx_power_hierarchy.rpt", "ptpx_switching_summary.rpt")
                if any(not (pt_dir / "reports" / name).is_file() or
                           (pt_dir / "reports" / name).stat().st_size == 0 for name in required):
                    raise Failure("PTPX report absent")
        exact_counts = {"vcs_compiles": 2, "simv_runs": 10,
                        "saif_files": 10, "ptpx_runs": 10}
        if any(state[key] != value for key, value in exact_counts.items()):
            raise Failure("execution count drift")
        state["phase"] = "SUCCESS_STAGE"; STAGE.mkdir()
        shutil.copytree(WORK / "candidate", STAGE / "candidate")
        for axis in ("k8", "k1x8"):
            shutil.copy2(WORK / "build" / axis / "compile.log", STAGE / f"{axis}.compile.log")
        write_json(STAGE / "m1432_receipt.json", {
            "schema": "m1432_c2_mapped_vcs_saif_ptpx_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity(),
            "one_shot": {"attempt_consumed": True, **exact_counts, "automatic_retry": False},
            "axes": ["k8", "k1x8"], "cases_per_axis": 5,
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1432_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE); publish_no_replace(WORK, PRIVATE); publish_no_replace(STAGE, RESULT)
        state["complete"] = True
        print("PASS_M1432_C2_MAPPED_VCS_SAIF_PTPX_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"]:
            try:
                FAIL_STAGE.mkdir(exist_ok=False)
                write_json(FAIL_STAGE / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE", "phase": state["phase"],
                    "error": type(error).__name__, "attempt_consumed": state["attempt"],
                    "identity": state.get("identity"),
                    "counts": {key: state[key] for key in (
                        "vcs_compiles", "simv_runs", "saif_files", "ptpx_runs")},
                    "automatic_retry": False, "canonical_result": False,
                    "partial_axis_citable": False})
                seal_dir(FAIL_STAGE); publish_no_replace(FAIL_STAGE, FAILURE)
            except BaseException:
                pass
            if WORK.is_dir() and not PRIVATE.exists():
                try: publish_no_replace(WORK, PRIVATE)
                except BaseException: pass
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
