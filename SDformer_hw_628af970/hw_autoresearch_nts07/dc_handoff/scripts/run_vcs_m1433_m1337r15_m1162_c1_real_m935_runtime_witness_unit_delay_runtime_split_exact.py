#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fail-closed one-shot M1433 C1/R16 functional VCS runner.

The source-only author suite is deliberately not executable from this runner.
Only the separately exact-pinned runtime-present suite is used at launch.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone


if len(sys.argv) != 1:
    raise SystemExit("M1433: no arguments accepted")

SCRIPT_DIR = Path(__file__).resolve().parent
HW = SCRIPT_DIR.parents[1]
RUNNER = Path(__file__).resolve()
FILELIST = HW / "verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_unit_delay_filelist.f"
WITNESS = HW / "verif_m1337r15_c1_real_m935_runtime_witness/m1337r15_m935_runtime_witness.sv"
TB = HW / "verif_m1270r13_c1_real_m935_protocol/tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13.sv"
PARENT = HW / "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv"
M935 = HW / "rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv"
WRAPPER = HW / "rtl_m1162_c1_common_charge_protocol/m1162_m935_c1_common_charge_protocol_boundary.sv"
SVA = HW / "verif_m1168r3_c1_common_charge_protocol/m1168r3_m1162_common_charge_protocol_assertions_r3.sv"
R16_CHECKER = HW / "verif_m1345r16_c1_real_m935_runtime_witness/check_m1345r16_source.py"
R16_TESTS = HW / "verif_m1345r16_c1_real_m935_runtime_witness/test_m1345r16_source.py"
R16_CONTRACT = HW / "contracts/m1345_c1_r16_real_m935_runtime_witness_source_contract_r1_20260831.json"
R16_AUTHOR = HW / "reviews/m1345_c1_r16_real_m935_runtime_witness_source_author_r1_20260831"
R16_HAMMER = HW / "reviews/m1352_m1345_c1_r16_runtime_witness_source_blind_hammer_r1_20260831"
M1354_AUTHOR = HW / "reviews/m1354_c1_r16_real_m935_runtime_witness_vcs_release_source_author_r1_20260831"
M1355_FAIL = HW / "reviews/m1355_m1354_c1_r16_real_m935_runtime_witness_vcs_release_blind_hammer_r1_20260831"
M1363_RUNNER = HW / "dc_handoff/scripts/run_vcs_m1363_m1337r15_m1162_c1_real_m935_runtime_witness_unit_delay_exact_sha.sh"
M1363_CHECKER = HW / "verif_m1363_c1_r16_vcs_release_exact/check_m1363_c1_r16_vcs_release_exact_source.py"
M1363_TESTS = HW / "verif_m1363_c1_r16_vcs_release_exact/test_m1363_c1_r16_vcs_release_exact_source.py"
M1363_CONTRACT = HW / "contracts/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_contract_r1_20260831.json"
M1363_AUTHOR = HW / "reviews/m1363_c1_r16_real_m935_runtime_witness_vcs_release_exact_source_author_r1_20260831"
M1364_FAIL = HW / "reviews/m1364_m1363_c1_r16_real_m935_runtime_witness_vcs_release_source_blind_hammer_r1_20260831"
SOURCE_CHECKER = HW / "verif_m1433_c1_r16_vcs_runtime_split/check_m1433_c1_r16_vcs_runtime_split_source.py"
SOURCE_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_split_source.py"
RUNTIME_TESTS = HW / "verif_m1433_c1_r16_vcs_runtime_split/test_m1433_c1_r16_vcs_runtime_present.py"
SOURCE_CONTRACT = HW / "contracts/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_contract_r1_20260831.json"
AUTHOR_DIR = HW / "reviews/m1433_c1_r16_real_m935_runtime_witness_vcs_runtime_split_source_author_r1_20260831"
SOURCE_HAMMER = HW / "reviews/m1441_m1433_c1_r16_runtime_split_source_blind_hammer_r1_20260831"
LAUNCH_RELEASE = HW / "contracts/m1442_m1441_m1433_c1_r16_runtime_split_vcs_launch_release_r1_20260831.json"
FINAL_HAMMER = HW / "reviews/m1443_m1442_m1433_c1_r16_runtime_split_vcs_final_launch_hammer_r1_20260831"
FOUNDRY = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1433_c1_r16_runtime_split_vcs_attempt_consumed"
RESULT = HW / "results/m1433_c1_r16_real_m935_runtime_witness_unit_delay_vcs_r1_20260831"
QUARANTINE = Path(str(RESULT) + ".failed_or_incomplete.quarantine")
WORK = HW / f"results/.m1433_c1_r16_runtime_split_vcs_work.{os.getpid()}"
ATTEMPT_STAGE = HW / f"results/.m1433_c1_r16_runtime_split_vcs_attempt_stage.{os.getpid()}"
FAILURE_STAGE = HW / f"results/.m1433_c1_r16_runtime_split_vcs_failure_stage.{os.getpid()}"
TOP = "tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13"
R13_PASS = "PASS_M1270R13_REAL_M935_INTEGRATED_PROTOCOL_SOURCE_CANDIDATE real_m935=true parent_issue_override=0 child_issue_override=0 first_beats=1 nonfirst_beats=1 weight_requests=2 psum_requests=1 response_join_hold_cycles=2 ii_ge_2=true row_completions=1 task_completions=1 boundary_fault=0 core_fault=0 m935_fault=0 every_oracle_operands=true zero_sva_failures_required=true functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false system_speedup=false headline=false"
R15_PASS = "PASS_M1337R15_REAL_M935_RUNTIME_WITNESS wrapper_functional_candidate=true strict_registered_stages=true unknown_fail_closed=true structural_bind=true ledger_bytes=214912 functional_vcs=false timing_verified=false cycles_measured=false speedup=false ppa=false energy=false headline=false"
COMPILE_TIMEOUT_SECONDS = 1200
SIM_TIMEOUT_SECONDS = 1800
MIN_HEADROOM_KIB = 16777216
CLAIMS = {"source_only": True, "functional_vcs": False, "timing_verified": False,
          "cycles_measured": False, "speedup": False, "ppa": False,
          "power": False, "energy": False, "system_speedup": False, "headline": False}

EXACT = {
    FILELIST: "87a8b5e7500808a8afbd4339668aae3a44db2de7924a948020e2c7bffce4289e",
    WITNESS: "0ec7179e36f9af09e3020f76a5a927298d877b3cc20c6ac9ab4686bf465d18af",
    TB: "b749c7d635dc5b65669320aec7b7edb40cd5e2a5d781a9e474e3d28cbb054263",
    PARENT: "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    M935: "e834b52401db67043cc3941f5593395a48879113b33247677156fdd417600ae8",
    WRAPPER: "639de97196898432696b96b204105059a8888bbc07e48d70576a73c26fd95595",
    SVA: "c07fc94a293be19c4c6f4d2126c6eb38e71f70dc12138af30cf4a770af772472",
    R16_CHECKER: "b570eeb7a49bb042de2abca2f6739df09ab1895f208103dbe4dfdac2e340cea4",
    R16_TESTS: "5427063ef93e89cd7059b6e48422626a71fd0913427f9614da65faf9fca29929",
    R16_CONTRACT: "c9749b4a7f9e3e6f8b38cbaf4735b036d7753f79a407e208d28f09aecd375f33",
    M1363_RUNNER: "ac473072accc6d48ec15c1e541d3fd7caad64638a2942655766dad14a1879de3",
    M1363_CHECKER: "f45ccb9e1844058106296a2565a1377d79ddb084f31724994a3147bfc4a48246",
    M1363_TESTS: "0f14a471e6f22d6ae8e6733af538f9cf3878598672643dee8231f990358bc704",
    M1363_CONTRACT: "c869fc5cc79bc711eb6cd139db832f95f8f80157ec202699c06db5b86371f13c",
    SOURCE_CHECKER: "0e7976f11a01588c00f55af83c224e148296f5e5fe6d8c85371e64dc1dfff1d5",
    SOURCE_TESTS: "9a2bd010aa5b0b97cd8923848940faac7dc1ce74caa9c7e1174bbf8257970e85",
    RUNTIME_TESTS: "b3b9d130749eb4a8a79148072350b76aeeb59520f85718e0663df62f40731ad4",
    FOUNDRY: "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
SEALED = {
    R16_AUTHOR: ("a5b136fce2bc3c5b5a5920b1e88cff092b1228b49a7ff6fd9959ff95e06772e5", "bd875634a0be33cb5dc2f0600734fa90e014ade961658c3d1f480ce40425a616", "c9700d4411dd087b12494e4aaf2f5fde0de52f7e30b7397573b205371837e99f"),
    R16_HAMMER: ("74969404ea26e5a522c205328c05a3527fca6daeefb74f6fb103cacb990e94ea", "d703fb23ff2a7726049f58d09e7d304d0e4e8adcaa781f34856115dcb4de40e6", "29c6bf6de6a7ed91dc523dfc3360d7731c324a24cd3548a0fe3a346018e37ec7"),
    M1354_AUTHOR: ("378ce7f6e8b0ae20f98c94d197c2fad1dcd7e1082fa269320041480319daddae", "799616b204bb88333193baad0188aac846cdca9a0493c19476f31ca1f7f866f2", "862b93fa2e781f48e4c1a59cc63262fe6541787e32171f28109ce6fd3eb0cbb6"),
    M1355_FAIL: ("7c06c50e2087e2794957508cf042d6931d73cb22ce3a3cada5628a2d55ae4c8d", "9709d1c21ce13df3b84efa19d4dfa47d2116fa661327f18d0666b17d924ec5f8", "8b7aea4d1bc0764c1e9137196e2fc0ea3b86cee27baf6ab459c2c717bd201105"),
    M1363_AUTHOR: ("26b419c7160ce29150cfcde8e82a43421aa1a1a16b9821c3a6e97111a2a146c6", "78c1db8a27fbda68a4efcb452262f506f48eaa06f14479b5283b446a28053f9a", "6ddf2def2c9475fbc70280f8bf4c39c383e74cf743de74e09d7242dcf0297c10"),
    M1364_FAIL: ("5e2e1497f0bee80e9067ec7ea5e699828cf1a27e6d76e3cf573f2907c8df701f", "6d4996d47095e9e7b727ecd6acc8015bb2561ab73d019cf04288765f4e69e36e", "b87b2efa20786e9c431015861ad7947c4e7444144e3bd5f9051b88addc9b37eb"),
}

ENV_PINS = (
    "M1433_EXPECTED_RUNNER_SHA256", "M1433_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
    "M1433_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256", "M1433_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
    "M1433_EXPECTED_LAUNCH_RELEASE_SHA256", "M1433_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
    "M1433_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256", "M1433_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")

COMPILE_COMMAND = [str(VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
                   "-assert", "svaext", "+define+UNIT_DELAY", "+vcs+lic+wait",
                   "-f", str(FILELIST), "-top", TOP, "-o", "simv"]
SIM_COMMAND = ["./simv", "-no_save"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink() or sha(path) != digest:
        raise RuntimeError("identity drift: " + str(path))


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise RuntimeError("duplicate JSON key")
            value[key] = item
        return value
    exact_mode = path.lstat().st_mode
    if not stat.S_ISREG(exact_mode) or path.is_symlink(): raise RuntimeError("JSON not regular")
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(RuntimeError(token)))


def verify_file_sidecar(path: Path) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    for item in (path, sidecar, outer):
        if not item.is_file() or item.is_symlink(): raise RuntimeError("sidecar absent")
    if sidecar.read_text().split() != [sha(path), path.name]: raise RuntimeError("sidecar mismatch")
    if outer.read_text().split() != [sha(sidecar), sidecar.name]: raise RuntimeError("outer mismatch")


def verify_recursive_seal(root: Path, pins=None) -> dict:
    if not root.is_dir() or root.is_symlink(): raise RuntimeError("sealed directory invalid")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    if pins is not None:
        if (sha(root / "review.json"), sha(manifest), sha(outer)) != pins:
            raise RuntimeError("sealed exact pin drift")
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise RuntimeError("outer seal drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or name in listed or rel.is_absolute()
                or ".." in rel.parts): raise RuntimeError("manifest row invalid")
        member = root / rel
        exact(member, digest); listed.add(name)
    actual = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        if any((base_path / name).is_symlink() for name in dirs + files):
            raise RuntimeError("sealed symlink")
        for name in files:
            rel = (base_path / name).relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: actual.add(rel)
    if listed != actual: raise RuntimeError("sealed membership drift")
    return strict_json(root / "review.json")


def seal_dir(root: Path) -> None:
    rows = []
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        if any((base_path / name).is_symlink() for name in dirs + files):
            raise RuntimeError("cannot seal symlink")
        for name in files:
            path = base_path / name; rel = path.relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                if not stat.S_ISREG(path.lstat().st_mode): raise RuntimeError("nonregular result")
                rows.append((rel, sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest}  {name}\n" for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n")
    verify_recursive_seal(root)


def publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error), str(destination))


def collision_gate() -> None:
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t", "pt_shell",
               "fm_shell", "icc2_shell", "common_shell_exec", "common_shell_exe"}
    ancestry = set(); pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try: pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except Exception: break
    hits = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in ancestry: continue
        try:
            if path.stat().st_uid != os.getuid(): continue
            comm = (path / "comm").read_text().strip()
            argv = {Path(item.decode(errors="replace")).name
                    for item in (path / "cmdline").read_bytes().split(b"\0") if item}
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if comm in blocked or blocked.intersection(argv): hits.append((path.name, comm))
    if hits: raise RuntimeError("same-UID EDA collision: " + repr(hits))


def resource_gate() -> None:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {"MemAvailable:", "CommitLimit:", "Committed_AS:"}:
            values[fields[0]] = int(fields[1])
    if (values.get("MemAvailable:", 0) < MIN_HEADROOM_KIB or
            values.get("CommitLimit:", 0) - values.get("Committed_AS:", 0) < MIN_HEADROOM_KIB):
        raise RuntimeError("resource preflight below 16 GiB")


def namespace_gate() -> None:
    for path in (ATTEMPT, RESULT, QUARANTINE, WORK, ATTEMPT_STAGE, FAILURE_STAGE):
        if os.path.lexists(path): raise RuntimeError("namespace residue: " + str(path))
    for pattern in (".m1433_c1_r16_runtime_split_vcs_work.*",
                    ".m1433_c1_r16_runtime_split_vcs_attempt_stage.*",
                    ".m1433_c1_r16_runtime_split_vcs_failure_stage.*"):
        if list((HW / "results").glob(pattern)): raise RuntimeError("stale stage: " + pattern)


def run_python_gate(path: Path, mode: str) -> None:
    completed = subprocess.run([str(PYTHON), "-I", str(path), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               timeout=120, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"python gate failed {path.name}: {completed.stderr}")


def run_tool(command, log: Path, timeout_seconds: int, environment: dict[str, str]) -> int:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, env=environment, cwd=WORK)
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try: output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); output, _ = process.communicate()
        log.write_bytes(output or b"")
        raise RuntimeError("tool timeout")
    output = output or b""; log.write_bytes(output); sys.stdout.buffer.write(output); sys.stdout.flush()
    return process.returncode


def main() -> int:
    os.umask(0o077)
    for name in ENV_PINS:
        if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get(name, "")):
            raise RuntimeError("external digest absent/invalid: " + name)
    exact(RUNNER, os.environ["M1433_EXPECTED_RUNNER_SHA256"])
    for path, digest in EXACT.items(): exact(path, digest)
    expected_filelist = [str(path) for path in (FOUNDRY, PARENT, M935, WRAPPER, SVA, TB, WITNESS)]
    if FILELIST.read_text().splitlines() != expected_filelist: raise RuntimeError("filelist/order drift")
    for root, pins in SEALED.items(): verify_recursive_seal(root, pins)
    verify_file_sidecar(SOURCE_CONTRACT)
    author = verify_recursive_seal(AUTHOR_DIR)
    source_hammer = verify_recursive_seal(SOURCE_HAMMER)
    exact(SOURCE_HAMMER / "review.json", os.environ["M1433_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256"])
    exact(SOURCE_HAMMER / "SHA256SUMS", os.environ["M1433_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256"])
    exact(SOURCE_HAMMER / "SHA256SUMS.seal.sha256", os.environ["M1433_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256"])
    verify_file_sidecar(LAUNCH_RELEASE)
    exact(LAUNCH_RELEASE, os.environ["M1433_EXPECTED_LAUNCH_RELEASE_SHA256"])
    final_hammer = verify_recursive_seal(FINAL_HAMMER)
    exact(FINAL_HAMMER / "review.json", os.environ["M1433_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"])
    exact(FINAL_HAMMER / "SHA256SUMS", os.environ["M1433_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"])
    exact(FINAL_HAMMER / "SHA256SUMS.seal.sha256", os.environ["M1433_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"])
    contract = strict_json(SOURCE_CONTRACT); release = strict_json(LAUNCH_RELEASE)
    if contract["status"] != "M1433_C1_R16_RUNTIME_SPLIT_SOURCE_READY__FRESH_M1441_REQUIRED__NO_LAUNCH": raise RuntimeError("contract status")
    bindings = {"runner_sha256": sha(RUNNER), "source_checker_sha256": sha(SOURCE_CHECKER),
                "source_tests_sha256": sha(SOURCE_TESTS), "runtime_tests_sha256": sha(RUNTIME_TESTS),
                "source_contract_sha256": sha(SOURCE_CONTRACT)}
    if any(author["bindings"].get(key) != value for key, value in bindings.items()): raise RuntimeError("author binding")
    if any(source_hammer["bindings"].get(key) != value for key, value in bindings.items()): raise RuntimeError("source hammer binding")
    if release.get("status") != "AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_UNIT_DELAY_VCS_ATTEMPT": raise RuntimeError("release status")
    if release.get("identity", {}).get("source_hammer_review_sha256") != sha(SOURCE_HAMMER / "review.json"): raise RuntimeError("release source hammer")
    if final_hammer.get("status") != "PASS_M1443_AUTHORIZE_ONE_M1433_C1_R16_RUNTIME_SPLIT_VCS_LAUNCH": raise RuntimeError("final status")
    if final_hammer.get("bindings", {}).get("launch_release_sha256") != sha(LAUNCH_RELEASE): raise RuntimeError("final release binding")
    authorization = {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0, "automatic_retry": False}
    if release.get("authorization") != authorization or final_hammer.get("authorization") != authorization: raise RuntimeError("authorization")
    if any(item.get("claim_boundary") != CLAIMS for item in (contract, author, source_hammer, release, final_hammer)): raise RuntimeError("claim boundary")
    run_python_gate(SOURCE_CHECKER, "runtime_present")
    run_python_gate(RUNTIME_TESTS, "runtime_present")
    namespace_gate()

    phase = 'RESOURCE_PREFLIGHT'
    collision_gate()
    resource_gate()
    collision_gate()
    phase = 'ATTEMPT_CONSUME'
    complete = False; failure_armed = True; compile_count = 0; sim_count = 0
    try:
        def interrupted(signum, _frame):
            raise RuntimeError("interrupted by signal " + str(signum))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        ATTEMPT_STAGE.mkdir()
        (ATTEMPT_STAGE / "attempt.json").write_text(json.dumps({
            "status": "M1433_ATTEMPT_CONSUMED", "runner_sha256": sha(RUNNER),
            "source_contract_sha256": sha(SOURCE_CONTRACT),
            "source_hammer_review_sha256": sha(SOURCE_HAMMER / "review.json"),
            "launch_release_sha256": sha(LAUNCH_RELEASE),
            "final_hammer_review_sha256": sha(FINAL_HAMMER / "review.json"),
            "automatic_retry": False, "maximum_vcs_compiles": 1, "maximum_simv_runs": 1,
        }, indent=2, sort_keys=True) + "\n")
        seal_dir(ATTEMPT_STAGE); publish_no_replace(ATTEMPT_STAGE, ATTEMPT)
        WORK.mkdir()
        environment = dict(os.environ)
        environment.update({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1",
                            "VCS_ARCH_OVERRIDE": "linux",
                            "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
                            "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat"})
        phase = 'COMPILE'
        compile_count = 1
        if run_tool(COMPILE_COMMAND, WORK / "compile.log", COMPILE_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("compile failed")
        simv = WORK / "simv"
        if not simv.is_file() or not os.access(simv, os.X_OK): raise RuntimeError("simv absent")
        phase = 'SIMULATE'
        sim_count = 1
        if run_tool(SIM_COMMAND, WORK / "sim.log", SIM_TIMEOUT_SECONDS, environment) != 0:
            raise RuntimeError("simulation failed")
        log = (WORK / "sim.log").read_text(errors="replace")
        if log.splitlines().count(R13_PASS) != 1 or log.splitlines().count(R15_PASS) != 1:
            raise RuntimeError("pass token cardinality")
        patterns = (
            r"^PHASE_M1270R13_REAL_M935_INTEGRATED_ENTER$",
            r"^PHASE_M1270R13_REAL_M935_INTEGRATED_COMPLETE$",
            r"^M1337R15_WITNESS_OPERANDS pass=1 ",
            r"^COVERAGE_M1270R13_REAL_M935 first_beats=1 nonfirst_beats=1 join_hold_cycles=2 issue_accepts=2 psum_reads=1 row_completions=1 task_completions=1 response_cycle_gap=[2-9][0-9]* oracle_records=[8-9][0-9]* parent_issue_override=0 child_issue_override=0$",
        )
        if any(len(re.findall(pattern, log, re.MULTILINE)) != 1 for pattern in patterns):
            raise RuntimeError("coverage cardinality")
        if re.search(r"(^|[^A-Za-z0-9_])(Error|Fatal|Assertion|\$error|\$fatal)([^A-Za-z0-9_]|$)", log, re.IGNORECASE):
            raise RuntimeError("error/fatal/assertion line")
        receipt = {
            "schema": "m1433_c1_r16_runtime_split_unit_delay_vcs_receipt_r1_v1",
            "status": "PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": {"runner_sha256": sha(RUNNER),
                         "source_contract_sha256": sha(SOURCE_CONTRACT),
                         "release_sha256": sha(LAUNCH_RELEASE),
                         "final_hammer_review_sha256": sha(FINAL_HAMMER / "review.json")},
            "runtime_split": {"source_tests_invoked": False, "runtime_tests_invoked": True},
            "macro_model": "foundry_UNIT_DELAY_functional",
            "one_shot": {"attempt_consumed": True, "vcs_compiles": 1, "simv_runs": 1,
                         "automatic_retry": False, "compile_timeout_seconds": 1200,
                         "sim_timeout_seconds": 1800},
            "claim_boundary": {**CLAIMS, "source_only": False, "functional_vcs": True},
        }
        (WORK / "m1433_c1_r16_runtime_split_unit_delay_vcs_receipt_r1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (WORK / "RUN_COMPLETE.txt").write_text("PASS_FUNCTIONAL_VCS_REAL_M935_RUNTIME_WITNESS\n")
        phase = 'SUCCESS_PUBLISH'
        seal_dir(WORK); publish_no_replace(WORK, RESULT); complete = True
        print("PASS M1433 C1/R16 functional VCS result=" + str(RESULT))
        return 0
    except BaseException as exc:
        if failure_armed and not complete:
            FAILURE_STAGE.mkdir()
            if ATTEMPT_STAGE.is_dir() and not ATTEMPT_STAGE.is_symlink():
                os.rename(ATTEMPT_STAGE, FAILURE_STAGE / "private_attempt_stage")
            if WORK.is_dir() and not WORK.is_symlink():
                private = FAILURE_STAGE / "private_build"; os.rename(WORK, private)
            (FAILURE_STAGE / "RUN_FAILED_OR_INCOMPLETE.json").write_text(json.dumps({
                "status": "FAILED_OR_INCOMPLETE", "phase": phase,
                "exception": type(exc).__name__ + ": " + str(exc),
                "compile_count": compile_count, "sim_count": sim_count,
                "automatic_retry": False, "functional_vcs": False,
                "timing_verified": False, "cycles_measured": False, "speedup": False,
                "ppa": False, "power": False, "energy": False,
                "system_speedup": False, "headline": False,
            }, indent=2, sort_keys=True) + "\n")
            seal_dir(FAILURE_STAGE)
            publish_no_replace(FAILURE_STAGE, QUARANTINE)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
