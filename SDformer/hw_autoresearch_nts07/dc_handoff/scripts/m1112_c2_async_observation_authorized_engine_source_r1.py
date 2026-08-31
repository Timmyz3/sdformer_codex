#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1112 fixed-identity async-observation diagnostic engine source.

This file is intentionally non-executable at the source stage.  A future
zero-argument launcher, double-sealed launch receipt, independent engine
hammer, and independent launch hammer must all exist before one fresh attempt
can be consumed.  No caller-selected hash or path is accepted.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path


DESIGN = "m1112_c2_k1_async_observation_shadow_wrapper"
TB_TOP = "tb_m1112_c2_k1_async_observation_shadow_case0_short"
DC_ROOT = Path(__file__).resolve().parent.parent
HW = DC_ROOT.parent
ENGINE = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1112_c2_async_observation_shadow_source_contract_r1_20260830.json"
LAUNCHER = HW / "dc_handoff/scripts/run_m1112_c2_async_observation_authorized_launch_r1.py"
LAUNCH_RECEIPT = HW / "contracts/m1112_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
M1113 = HW / "reviews/m1113_m1112_c2_async_observation_engine_hammer_r1_20260830"
M1114 = HW / "reviews/m1114_m1112_c2_async_observation_launch_hammer_r1_20260830"
M1109 = HW / "reviews/m1109_m1091r3_c2_observation_mapped_x_failure_audit_r1_20260830"
OLD_ATTEMPT = HW / "results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed"
OLD_FAILURE = HW / "results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.3507131.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FILELIST = HW / "dc_handoff/filelists/date_m1112_c2_k1_async_observation_shadow_logic_only_dc.f"
TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DC_SHELL = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
SLOW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
RESULT = HW / "results/m1112_c2_async_observation_dc_mapped_vcs_r1_20260830"
ATTEMPT = HW / "results/.m1112_c2_async_observation_dc_mapped_vcs_attempt_consumed"
WORK = HW / f"results/.m1112_c2_async_observation_dc_mapped_vcs_work.{os.getpid()}"
FAILURE = HW / f"results/m1112_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
LOCK = Path("/tmp/m1112_c2_async_observation_eda.lock")

CONTRACT_SHA256 = "016290ad92593f6d43989a9b57576657340d481ebc6f72d7e82c8081740f3a08"
CONTRACT_OUTER_SHA256 = "ed757521f3b8b3a06b12c02e8bc9b3278da5855bd3ce5f1b90cbd99a219a9781"
M1109_OUTER_SHA256 = "5c7a1f667c6c800f84a0e8219ddf58574412090812cda5d8bdaf36265f43d52d"
OLD_ATTEMPT_OUTER_SHA256 = "615b231d1a1d96b99222198c1ccff546f7ca9749370159104b664d589f1f3972"
OLD_FAILURE_OUTER_SHA256 = "805948588c90b66e68cf32186322d22d875809330b9f4edad705f9993a9cdaee"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SHADOW_REGISTER_BITS = 337

EXTERNAL_SHA256 = {
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DC_TARGET: "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    SLOW: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    FAST: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
}

phase = "SOURCE_PREFLIGHT"
attempted = False
complete = False


class GateFailure(RuntimeError):
    pass


def fail(message: str) -> None:
    raise GateFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_regular(path: Path, expected: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        fail(f"missing regular file: {path}")
    if not stat.S_ISREG(mode) or path.is_symlink():
        fail(f"non-regular or direct symlink rejected: {path}")
    if sha(path) != expected:
        fail(f"identity drift: {path}")


def verify_double(path: Path, expected: str, expected_outer: str) -> None:
    verify_regular(path, expected)
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    if side.read_text(encoding="utf-8").split() != [expected, path.relative_to(HW).as_posix()]:
        fail(f"sidecar drift: {path}")
    verify_regular(outer, expected_outer)
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail(f"outer drift: {path}")


def verify_flat(directory: Path, expected_outer: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        fail(f"sealed directory absent/symlink: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        verify_regular(directory / name.lstrip("*"), digest)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail(f"outer content drift: {directory}")


def verify_historical_quarantine() -> None:
    """The sole symlink exception is restricted to the sealed old failure."""
    if not OLD_FAILURE.is_dir() or OLD_FAILURE.is_symlink():
        fail("historical quarantine absent/symlink")
    root = OLD_FAILURE.resolve(strict=True)
    manifest = OLD_FAILURE / "SHA256SUMS"
    outer = OLD_FAILURE / "SHA256SUMS.seal.sha256"
    verify_regular(outer, OLD_FAILURE_OUTER_SHA256)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("historical quarantine outer drift")
    symlinks = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        relative = Path(name.lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            fail("historical manifest path escape")
        member = OLD_FAILURE / relative
        mode = member.lstat().st_mode
        if stat.S_ISLNK(mode):
            symlinks += 1
            resolved = member.resolve(strict=True)
            if resolved != root and root not in resolved.parents:
                fail("historical symlink escapes quarantine")
            if not stat.S_ISREG(resolved.lstat().st_mode) or resolved.is_symlink():
                fail("historical symlink target is not regular")
        elif not stat.S_ISREG(mode):
            fail("historical manifest member is not regular/symlink")
        if sha(member) != digest:
            fail("historical followed-byte digest drift")
    if symlinks != 1:
        fail("historical quarantine symlink census drift")


def verify_self_consistent_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    expected, relative = side.read_text(encoding="utf-8").split()
    if relative != path.relative_to(HW).as_posix():
        fail("future receipt path drift")
    verify_regular(path, expected)
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail("future receipt outer drift")
    return sha(outer)


def verify_parent_launcher(receipt: dict) -> None:
    verify_regular(LAUNCHER, receipt["launcher_sha256"])
    raw = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    argv = [item.decode(errors="strict") for item in raw if item]
    if len(argv) != 2 or Path(argv[0]) != PYTHON or Path(argv[1]).resolve() != LAUNCHER:
        fail("fixed launcher must be invoked with zero arguments under pinned Python")


def verify_future_launch_chain() -> dict:
    if not LAUNCHER.exists() or not LAUNCH_RECEIPT.exists():
        fail("future fixed launcher/receipt absent: source stage cannot execute")
    receipt_outer = verify_self_consistent_double(LAUNCH_RECEIPT)
    receipt = load(LAUNCH_RECEIPT)
    if receipt["status"] != "M1112_LAUNCH_SOURCE_FROZEN__M1114_REQUIRED__NO_EDA":
        fail("future launch receipt boundary")
    if receipt["engine_sha256"] != sha(ENGINE):
        fail("future launch receipt engine drift")
    verify_flat(M1113, receipt["m1113_outer_seal_file_sha256"])
    if load(M1113 / "review.json")["status"] != "PASS_M1113_M1112_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA":
        fail("M1113 has no GO")
    m1114_outer = sha(M1114 / "SHA256SUMS.seal.sha256")
    verify_flat(M1114, m1114_outer)
    review = load(M1114 / "review.json")
    if review["status"] != "PASS_M1114_M1112_LAUNCH_HAMMER__GO_ONE_ATTEMPT":
        fail("M1114 has no GO")
    identity = review["identity"]
    if identity["engine_sha256"] != sha(ENGINE):
        fail("M1114 engine drift")
    if identity["launcher_sha256"] != receipt["launcher_sha256"]:
        fail("M1114 launcher drift")
    if identity["launch_receipt_outer_seal_file_sha256"] != receipt_outer:
        fail("M1114 launch receipt drift")
    verify_parent_launcher(receipt)
    receipt["m1114_outer_seal_file_sha256"] = m1114_outer
    return receipt


def verify_dc_shell() -> None:
    if not stat.S_ISLNK(DC_SHELL.lstat().st_mode) or os.readlink(DC_SHELL) != "snps_shell":
        fail("dc_shell symlink drift")
    if DC_SHELL.resolve(strict=True) != DC_TARGET:
        fail("dc_shell target drift")
    verify_regular(DC_TARGET, EXTERNAL_SHA256[DC_TARGET])


def static_gate() -> dict:
    if sys.argv[1:] != ["--authorized-launch"]:
        fail("fixed argv required")
    if Path(sys.executable) != PYTHON:
        fail("unpinned Python")
    verify_regular(ENGINE, sha(ENGINE))
    verify_double(CONTRACT, CONTRACT_SHA256, CONTRACT_OUTER_SHA256)
    contract = load(CONTRACT)
    if contract["status"] != "M1112_ASYNC_OBSERVATION_SHADOW_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA":
        fail("contract boundary")
    if contract["launch_now"] is not False or contract["max_attempts_now"] != 0:
        fail("source stage accidentally authorizes an attempt")
    verify_regular(DOCS359, DOCS359_SHA256)
    for relative, expected in contract["source_sha256"].items():
        verify_regular(HW / relative, expected)
    for relative, expected in contract["frozen_filelist_member_sha256"].items():
        verify_regular(HW / relative, expected)
    for path, expected in EXTERNAL_SHA256.items():
        if path != DC_TARGET:
            verify_regular(path, expected)
    verify_dc_shell()
    verify_flat(M1109, M1109_OUTER_SHA256)
    if load(M1109 / "review.json")["status"] != "PASS_M1109_FAILURE_AUDIT__M1091R3_DO_NOT_RETRY__NEW_ASYNC_OBSERVATION_SHADOW_REPAIR_ONLY":
        fail("M1109 authority drift")
    verify_flat(OLD_ATTEMPT, OLD_ATTEMPT_OUTER_SHA256)
    verify_historical_quarantine()
    if any(path.exists() or path.is_symlink() for path in (ATTEMPT, RESULT, WORK)):
        fail("new M1112 namespace is not fresh")
    return verify_future_launch_chain()


def structural_reset_gate(netlist: Path) -> dict:
    """Require every mapped shadow register instance to have an async pin.

    TSMC28 async cells expose CDN/CN/SDN/SN.  The checker deliberately rejects
    D/CP-only flops even when synthesis recreated reset behavior in data logic.
    """
    text = netlist.read_text(encoding="utf-8", errors="strict")
    statements = [item + ";" for item in text.split(";")]
    shadow = [item for item in statements if re.search(r"shadow_\w+_q_reg", item)]
    if len(shadow) != SHADOW_REGISTER_BITS:
        fail(f"mapped shadow register census {len(shadow)} != {SHADOW_REGISTER_BITS}")
    missing_async = []
    cell_types = set()
    for statement in shadow:
        match = re.search(r"^\s*(\w+)\s+\S*shadow_", statement, re.S)
        cell_type = match.group(1) if match else "UNPARSED"
        cell_types.add(cell_type)
        if not re.search(r"\.(?:CDN|CN|SDN|SN)\s*\(", statement):
            missing_async.append(statement[:160])
    if missing_async:
        fail(f"mapped shadow bits without explicit async reset/set pin: {len(missing_async)}")
    return {"shadow_register_bits": len(shadow), "resettable_cell_types": sorted(cell_types)}


def resource_gate() -> None:
    info = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemAvailable", "CommitLimit", "Committed_AS"}:
            info[key] = int(raw.split()[0])
    if info["MemAvailable"] < 8 * 1024 * 1024 or info["CommitLimit"] - info["Committed_AS"] < 8 * 1024 * 1024:
        fail("resource gate")


def collision_gate() -> None:
    uid = str(os.getuid())
    for name in ("vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell", "pt_shell", "simv"):
        if subprocess.run(["/usr/bin/pgrep", "-u", uid, "-x", name], stdout=subprocess.DEVNULL, check=False).returncode == 0:
            fail("EDA collision " + name)


def license_gate() -> None:
    route = os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")
    if not route:
        fail("license route absent")
    if subprocess.run([str(LMUTIL), "lmstat", "-a", "-c", route], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False).returncode:
        fail("license gate")


def run(argv: list[str], log: Path, timeout: int, env: dict[str, str]) -> int:
    with log.open("w", encoding="utf-8") as output:
        return subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, env=env, check=False).returncode


def seal(directory: Path) -> None:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")


def flow() -> None:
    global phase, attempted, complete
    authority = static_gate()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("exclusive lock busy")
        collision_gate(); resource_gate(); license_gate()
        phase = "ATTEMPT_CONSUME_AFTER_M1113_M1114"
        ATTEMPT.mkdir(); attempted = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1112_ATTEMPT_CONSUMED_AFTER_M1113_M1114",
            "engine_sha256": sha(ENGINE),
            "contract_sha256": CONTRACT_SHA256,
            "launcher_sha256": authority["launcher_sha256"],
            "m1113_outer_seal_file_sha256": authority["m1113_outer_seal_file_sha256"],
            "m1114_outer_seal_file_sha256": authority["m1114_outer_seal_file_sha256"],
            "dc_attempts": 1, "mapped_cases": 1, "random_initialization": False,
        })
        seal(ATTEMPT)
        WORK.mkdir()
        env = os.environ.copy()
        env.update({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"})
        phase = "FRESH_DC_M1112_ASYNC_OBSERVATION_TOP"
        dc = WORK / "dc"; dc.mkdir()
        dc_env = env.copy()
        dc_env.update({
            "DESIGN_NAME": DESIGN, "HW_ROOT": str(HW), "RTL_FILELIST": str(FILELIST),
            "LIB_DB": str(SLOW), "MIN_LIB_DB": str(FAST), "SDC_FILE": str(SDC),
            "OUTPUT_DIR": str(dc), "ELAB_PARAMETERS": "", "OPERATING_CONDITION": "ssg0p9v125c",
        })
        rc = run([str(DC_SHELL), "-f", str(DC_TCL)], dc / "dc.log", 21600, dc_env)
        (dc / "dc.rc").write_text(str(rc) + "\n", encoding="utf-8")
        if rc or not (dc / "TCL_PASS_TERMINAL.txt").is_file():
            fail("fresh DC failed")
        netlist = dc / f"netlist/{DESIGN}_mapped.v"
        if not netlist.is_file() or not netlist.stat().st_size:
            fail("mapped netlist absent")
        phase = "MAPPED_RESETTABLE_SHADOW_STRUCTURE"
        reset_census = structural_reset_gate(netlist)
        phase = "FRESH_MAPPED_VCS_CASE0_SHORT_128"
        mapped = WORK / "mapped_vcs"; mapped.mkdir()
        simv = mapped / "simv"
        rc = run([
            str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            f"-Mdir={mapped / 'csrc'}", str(CELL), str(netlist), str(MEMORY),
            str(TB), "-top", TB_TOP, "-o", str(simv),
        ], mapped / "compile.log", 1800, env)
        if rc or not simv.is_file():
            fail("mapped compile failed")
        rc = run([str(simv), "-no_save"], mapped / "case0.log", 300, env)
        text = (mapped / "case0.log").read_text(encoding="utf-8", errors="replace")
        token = "PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 unknown_bitmap=000000 diagnostic_only=1"
        if rc or token not in text or "M1112_FIRST_X" in text:
            fail("mapped short window failed; inspect sealed quarantine")
        write_json(WORK / "receipt.json", {
            "status": "PASS_M1112_FRESH_DC_RESETTABLE_SHADOW_MAPPED_SHORT_WINDOW",
            "mapped_netlist_sha256": sha(netlist),
            "stage_lines": len(re.findall(r"^M1112_STAGE", text, re.M)),
            "resettable_shadow_structure": reset_census,
            "window_cycles": 128, "unknown_bitmap": "000000",
            "diagnostic_only": True, "paper_citable": False,
        })
        (WORK / "RUN_COMPLETE.txt").write_text("PASS_M1112_FRESH_DC_RESETTABLE_SHADOW_MAPPED_SHORT_WINDOW\n", encoding="utf-8")
        seal(WORK); os.rename(WORK, RESULT); complete = True


def quarantine(message: str) -> None:
    if attempted and not complete:
        WORK.mkdir(parents=True, exist_ok=True)
        write_json(WORK / "failure.json", {
            "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE", "phase": phase,
            "message": message, "m1091r3_retry": False,
        })
        seal(WORK); os.rename(WORK, FAILURE)


def handler(signum: int, _frame: object) -> None:
    raise GateFailure("signal " + str(signum))


for caught_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(caught_signal, handler)

try:
    flow()
except (GateFailure, OSError, subprocess.TimeoutExpired, KeyError, ValueError, json.JSONDecodeError) as exc:
    quarantine(str(exc))
    print("M1112 failure: " + str(exc), file=sys.stderr)
    raise SystemExit(3)
print("PASS_M1112_FRESH_DC_RESETTABLE_SHADOW_MAPPED_SHORT_WINDOW")
