#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1068 one-shot: fresh M1058 ARCH_MODE0 DC, then five mapped VCS cases.

This is release source only until an independently sealed M1069 hammer grants
one attempt. It never runs SAIF or PTPX.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

DESIGN = "m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24"
ANCHORS = [259, 737, 3153, 7569, 14]
EXPECTED_VCS_HOME = Path("/opt/synopsys/vcs/V-2023.12-SP1")
EXPECTED_DC_HOME = Path("/opt/synopsys/syn/V-2023.12-SP3")
DC_ROOT = Path(__file__).resolve().parent.parent
HW_ROOT = DC_ROOT.parent
RUNNER = Path(__file__).resolve()
CONTRACT = HW_ROOT / "contracts/m1068_c2_k1_reset_hygiene_dc_mapped_vcs_source_contract_r1_20260830.json"
RELEASE = HW_ROOT / "contracts/m1068_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_release_r1_20260830.json"
M1069 = HW_ROOT / "reviews/m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830"
M1059 = HW_ROOT / "reviews/m1059_m1058_c2_k1_reset_hygiene_source_release_hammer_r1_20260830"
M1058_RESULT = HW_ROOT / "results/m1058_c2_k1_reset_hygiene_rtl_vcs_r1_20260830"
M1058_CONTRACT = HW_ROOT / "contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json"
M1058_CANDIDATE = HW_ROOT / "contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json"
M1050 = HW_ROOT / "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829"
M1046 = HW_ROOT / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine"
DC_TCL = HW_ROOT / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
DC_FILELIST = HW_ROOT / "dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_logic_only_dc.f"
SDC = HW_ROOT / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
MAPPED_TB = HW_ROOT / "dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv"
MEMORY_MODEL = HW_ROOT / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
SLOW_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST_LIB = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
CELL_MODEL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
DC_SHELL = EXPECTED_DC_HOME / "bin/dc_shell"
DC_ACTUAL = EXPECTED_DC_HOME / "linux64/syn/bin/common_shell_exec"
VCS = EXPECTED_VCS_HOME / "bin/vcs"
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW_ROOT / "results/m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830"
ATTEMPT = HW_ROOT / "results/.m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed"
WORK = HW_ROOT / f"results/.m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_work.{os.getpid()}"
FAILURE = HW_ROOT / f"results/m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
LOCK = Path("/tmp/m1068_m1058_c2_k1_reset_hygiene_eda.lock")

phase = "SOURCE_PREFLIGHT"
attempt_consumed = False
complete = False


class GateFailure(RuntimeError):
    pass


def fail(message: str) -> None:
    raise GateFailure(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expect_sha(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        fail(f"identity drift: {path}")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def verify_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        if rel.startswith("*"):
            rel = rel[1:]
        target = directory / rel
        if not target.is_file() or sha(target) != digest:
            fail(f"manifest mismatch: {target}")
    outer_line = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split()
    if len(outer_line) < 2 or outer_line[1] != "SHA256SUMS":
        fail(f"malformed inner seal: {directory}")
    if sha(manifest) != outer_line[0]:
        fail(f"inner seal mismatch: {directory}")


def verify_seal(directory: Path, expected_outer: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        fail(f"sealed directory absent: {directory}")
    verify_manifest(directory)
    if sha(directory / "SHA256SUMS.seal.sha256") != expected_outer:
        fail(f"outer seal mismatch: {directory}")


def verify_sidecar(path: Path, expected_outer: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for item in (path, side, outer):
        if not item.is_file() or item.is_symlink():
            fail(f"sidecar set absent: {path}")
    digest, name = side.read_text(encoding="utf-8").strip().split()[:2]
    if name != path.name or sha(path) != digest:
        fail(f"primary sidecar mismatch: {path}")
    digest2, name2 = outer.read_text(encoding="utf-8").strip().split()[:2]
    if name2 != side.name or sha(side) != digest2:
        fail(f"outer sidecar content mismatch: {path}")
    if sha(outer) != expected_outer:
        fail(f"outer sidecar identity mismatch: {path}")


def seal_dir(directory: Path) -> None:
    members = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    )
    lines = [f"{sha(p)}  {p.relative_to(directory).as_posix()}" for p in members]
    manifest = directory / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")


def check_collisions() -> None:
    uid = os.getuid()
    for proc in ("vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t",
                 "fm_shell", "pt_shell"):
        hit = subprocess.run(
            ["/usr/bin/pgrep", "-u", str(uid), "-x", proc],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
        if hit.returncode == 0:
            fail(f"same-UID EDA collision: {proc}")
    listing = subprocess.run(
        ["/usr/bin/pgrep", "-u", str(uid), "-a", "simv"],
        text=True, capture_output=True, check=False)
    if listing.returncode == 0:
        fail("same-UID simv collision")


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemAvailable", "CommitLimit", "Committed_AS"}:
            values[key] = int(raw.strip().split()[0])
    return values


def resource_gate(minimum_kib: int) -> None:
    info = meminfo()
    if info["MemAvailable"] < minimum_kib:
        fail("insufficient MemAvailable")
    if info["CommitLimit"] - info["Committed_AS"] < minimum_kib:
        fail("insufficient commit headroom")


def license_gate() -> None:
    route = os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get(
        "LM_LICENSE_FILE")
    if not route:
        fail("nonempty Synopsys license route required")
    probe = subprocess.run(
        [str(LMUTIL), "lmstat", "-a", "-c", route],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=60, check=False)
    if probe.returncode != 0:
        fail("Synopsys license status failed")


def run_logged(argv: list[str], log: Path, timeout: int,
               env: dict[str, str] | None = None) -> int:
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(
            argv, stdout=out, stderr=subprocess.STDOUT, env=env,
            timeout=timeout, check=False)
    return result.returncode


def atomic_publish(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        fail(f"publish destination already exists: {destination}")
    os.rename(source, destination)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def signal_handler(signum: int, _frame: object) -> None:
    raise GateFailure(f"signal {signum}")


def static_identity_gate() -> None:
    expect_sha(DOCS359, "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
    identities = {
        DC_SHELL: "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
        DC_ACTUAL: "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
        VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
        LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
        Path(sys.executable): "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
        SLOW_LIB: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
        FAST_LIB: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
        CELL_MODEL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
        MEMORY_MODEL: "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
        DC_TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
        SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
        DC_FILELIST: "4cfd47438de45a66a601433ee07a2493c7296b1dea8669f9c7826898364e7192",
        MAPPED_TB: "fdbb1ccc5be4af11d263a6581f84ec823f9fd8077b07fa8b12a88a32a056ae0f",
    }
    for path, digest in identities.items():
        expect_sha(path, digest)

    source_identities = {
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv": "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
        "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
        "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
        "rtl_m218/m218_fc2_tagged_slice_service_island.sv": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
        "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv": "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
        "rtl_m519/m519_fc2_k1_registered_release_service_island.sv": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
        "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
        "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
        "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
        "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
        "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv": "7876d8603ef4fc6e326287aecc7a4c9a9a66cab5400bcaa3b24f498518ff9d9d",
        "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv": "a3d6628f28c6c277e9feda143f3cf9e1365eaad5648ab358f75e82bbd9187768",
        "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv": "f9f7319fd2495dc4a67ec20ecf6f34ef8884c88f9bf49f8f2a74e7ee88e3e0f8",
        "rtl_m1058/m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24.sv": "33e2fa8427eff64bae3bde2c11bf7e6a3a15969aff076cb0ab7b96431227a565",
    }
    for rel, digest in source_identities.items():
        expect_sha(HW_ROOT / rel, digest)

    verify_sidecar(M1058_CONTRACT,
                   "1d06a6bdda5b15e404c758e5571498d026cb23e586fc7ba1d929f1c064518b44")
    verify_sidecar(M1058_CANDIDATE,
                   "12c131029fc6f049e2f2a58082dcb6e4f72c4056a9bc68cde9006d585b2c7f82")
    verify_seal(M1058_RESULT,
                "f22a55c33fadf74749060546e877fc10f892649aa31f3fa0da2d3fd164b70787")
    verify_seal(M1050,
                "bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb")
    verify_seal(M1046,
                "cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705")
    verify_seal(M1059,
                "c22d41a87f82f939487637155b35d11496234850631b5894d159ff41e41fb4b3")

    if load_json(M1058_CONTRACT)["status"] != \
            "PASS_SOURCE_RTL_VCS__MAPPED_FIX_NOT_ADMITTED__REQUIRES_M1059":
        fail("M1058 source status mismatch")
    if load_json(M1058_CANDIDATE)["launch_now"] is not False:
        fail("M1058 candidate launch boundary mismatch")
    if load_json(M1050 / "review.json")["status"] != \
            "PASS_M1050_M1046_WATCHDOG_FAILURE_AUDIT__M1046_DO_NOT_RETRY":
        fail("M1050 status mismatch")
    m1046_failure = load_json(M1046 / "failure.json")
    if (m1046_failure["status"], m1046_failure["phase"]) != \
            ("FAILED_OR_INCOMPLETE", "RUN_k1_CASE0"):
        fail("M1046 failure identity mismatch")
    m1059_review = load_json(M1059 / "review.json")
    if m1059_review["status"] != \
            "PASS_M1059_M1058_C2_K1_RESET_HYGIENE_SOURCE_RELEASE_HAMMER":
        fail("M1059 status mismatch")
    if m1059_review["verdict"] != \
            "GO_EXACT_NON_LAUNCHING_CANDIDATE_FOR_M1068_RELEASE_AUTHORING":
        fail("M1059 verdict mismatch")


def release_chain_gate() -> None:
    expected_runner = os.environ.get("M1068_EXPECTED_RUNNER_SHA256")
    expected_m1069 = os.environ.get("M1068_EXPECTED_M1069_OUTER_SHA256")
    if not expected_runner or sha(RUNNER) != expected_runner:
        fail("caller must pin exact M1068 runner SHA")
    if not expected_m1069:
        fail("caller must pin independent M1069 outer seal")
    release = load_json(RELEASE)
    verify_sidecar(CONTRACT, release["identity"]["contract_outer_seal_sha256"])
    verify_seal(M1069, expected_m1069)
    hammer = load_json(M1069 / "review.json")
    verify_sidecar(RELEASE, hammer["identity"]["release_outer_seal_sha256"])
    if release["status"] != \
            "PASS_M1068_C2_K1_RESET_HYGIENE_DC_MAPPED_VCS_ONE_SHOT_RELEASE_SOURCE":
        fail("M1068 release status mismatch")
    if release["launch_now"] is not False:
        fail("M1068 release must remain non-launching")
    if release["runner_sha256"] != sha(RUNNER):
        fail("release runner identity mismatch")
    if release["source_contract_sha256"] != sha(CONTRACT):
        fail("release contract identity mismatch")
    if hammer["status"] != \
            "PASS_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER":
        fail("M1069 release hammer status mismatch")
    if hammer["authorization"]["one_m1068_dc_then_mapped_vcs_attempt"] is not True:
        fail("M1069 did not authorize the one-shot")
    forbidden = "+vcs+" + "".join(chr(x) for x in (105, 110, 105, 116, 114, 101, 103))
    if forbidden in RUNNER.read_text(encoding="utf-8"):
        fail("forbidden production initialization option present")


def run_flow() -> None:
    global phase, attempt_consumed, complete
    static_identity_gate()
    release_chain_gate()
    if RESULT.exists() or ATTEMPT.exists() or WORK.exists():
        fail("result/attempt/work namespace collision")

    clean_env = os.environ.copy()
    clean_env["VCS_HOME"] = str(EXPECTED_VCS_HOME)
    clean_env["PATH"] = f"{EXPECTED_VCS_HOME}/bin:/usr/bin:/bin"

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateFailure("M1068 global EDA flock busy") from exc

        check_collisions()
        resource_gate(16 * 1024 * 1024)
        license_gate()
        check_collisions()

        phase = "ATTEMPT_ATOMIC_CONSUME"
        ATTEMPT.mkdir()
        attempt_consumed = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1068_ATTEMPT_CONSUMED",
            "runner_sha256": sha(RUNNER),
            "contract_sha256": sha(CONTRACT),
            "release_sha256": sha(RELEASE),
            "m1069_outer_seal_sha256":
                os.environ["M1068_EXPECTED_M1069_OUTER_SHA256"],
            "dc_attempts_authorized": 1,
            "mapped_vcs_cases_required": 5,
            "saif_authorized": False,
            "ptpx_authorized": False,
        })
        seal_dir(ATTEMPT)
        WORK.mkdir()

        phase = "FRESH_DC_ARCH_MODE0"
        check_collisions()
        resource_gate(16 * 1024 * 1024)
        license_gate()
        dc_dir = WORK / "dc"
        dc_dir.mkdir()
        dc_env = clean_env.copy()
        dc_env.update({
            "DESIGN_NAME": DESIGN,
            "HW_ROOT": str(HW_ROOT),
            "RTL_FILELIST": str(DC_FILELIST),
            "LIB_DB": str(SLOW_LIB),
            "MIN_LIB_DB": str(FAST_LIB),
            "SDC_FILE": str(SDC),
            "OUTPUT_DIR": str(dc_dir),
            "ELAB_PARAMETERS": "ARCH_MODE=0",
            "OPERATING_CONDITION": "ssg0p9v125c",
        })
        dc_rc = run_logged(
            [str(DC_SHELL), "-f", str(DC_TCL)],
            dc_dir / "dc.log", timeout=21600, env=dc_env)
        (dc_dir / "dc.rc").write_text(f"{dc_rc}\n", encoding="utf-8")
        if dc_rc != 0:
            fail(f"fresh DC returned {dc_rc}")
        if not (dc_dir / "TCL_PASS_TERMINAL.txt").is_file():
            fail("DC terminal PASS absent")
        if (dc_dir / "TCL_EXPLICIT_FAILURE.txt").exists():
            fail("DC explicit failure sentinel present")
        loop_gate = (dc_dir / "reports/precompile_loop_gate.rpt").read_text()
        for token in ("TIM-209=0", "OPT-150=0",
                      "status=PASS_PRECOMPILE_LOOP_GATE"):
            if token not in loop_gate:
                fail(f"DC precompile gate missing {token}")
        errors = [
            line.rstrip("\n")
            for line in (dc_dir / "dc.log").read_text(
                encoding="utf-8", errors="replace").splitlines(True)
            if line.startswith("Error:") or line.startswith("Fatal:")
        ]
        allowed = (
            "Error: Error during sourcing of "
            "/opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
        )
        if errors not in ([], [allowed]):
            fail(f"unexpected DC error/fatal lines: {errors!r}")

        netlist = dc_dir / f"netlist/{DESIGN}_mapped.v"
        if not netlist.is_file() or netlist.stat().st_size == 0:
            fail("fresh mapped netlist absent")
        (dc_dir / "netlist_sha256.txt").write_text(
            f"{sha(netlist)}  {netlist.name}\n", encoding="utf-8")

        phase = "FRESH_MAPPED_VCS_COMPILE"
        check_collisions()
        resource_gate(8 * 1024 * 1024)
        license_gate()
        vcs_dir = WORK / "mapped_vcs"
        vcs_dir.mkdir()
        simv = vcs_dir / "simv"
        vcs_rc = run_logged([
            str(VCS), "-full64", "-sverilog", "+v2k",
            "-timescale=1ns/1ps", f"-Mdir={vcs_dir / 'csrc'}",
            str(CELL_MODEL), str(netlist), str(MEMORY_MODEL), str(MAPPED_TB),
            "-top", "tb_m1058_c2_k1_reset_hygiene_mapped_gate_case",
            "-o", str(simv),
        ], vcs_dir / "compile.log", timeout=1800, env=clean_env)
        (vcs_dir / "compile.rc").write_text(f"{vcs_rc}\n", encoding="utf-8")
        if vcs_rc != 0 or not simv.is_file() or not os.access(simv, os.X_OK):
            fail("fresh mapped VCS compile failed")
        compile_text = (vcs_dir / "compile.log").read_text(
            encoding="utf-8", errors="replace")
        if re.search(r"Error-\[|^Error|^Fatal|Fatal:", compile_text, re.I | re.M):
            fail("mapped VCS compile log contains error/fatal")

        for case_id, anchor in enumerate(ANCHORS):
            phase = f"FRESH_MAPPED_VCS_CASE{case_id}"
            log = vcs_dir / f"case{case_id}.log"
            sim_rc = run_logged(
                [str(simv), f"+M979_CASE={case_id}", "-no_save"],
                log, timeout=900, env=clean_env)
            (vcs_dir / f"case{case_id}.rc").write_text(
                f"{sim_rc}\n", encoding="utf-8")
            if sim_rc != 0:
                fail(f"mapped case {case_id} returned {sim_rc}")
            text = log.read_text(encoding="utf-8", errors="replace")
            pattern = (
                rf"^PASS M979 mapped replay axis=K1 case={case_id} "
                rf".*cycles={anchor} .*numeric_mismatches=0 "
                rf"tuple_mismatches=0 weight_mismatches=0 "
                rf"accepted_unknowns=0 protocol_errors=0$"
            )
            if not re.search(pattern, text, re.M):
                fail(f"mapped case {case_id} anchor/mismatch gate failed")
            if re.search(r"watchdog|failed at|Offending|^Error|^Fatal|Fatal:",
                         text, re.I | re.M):
                fail(f"mapped case {case_id} contains failure token")

        phase = "METRICS_AND_RECEIPT"
        area_text = (dc_dir / "reports/area.rpt").read_text(
            encoding="utf-8", errors="replace")
        timing_text = (dc_dir / "reports/timing_setup.rpt").read_text(
            encoding="utf-8", errors="replace")
        area_match = re.search(
            r"Total cell area:\s*([-+0-9.eE]+)", area_text)
        slacks = [
            float(value) for value in re.findall(
                r"slack\s+\((?:MET|VIOLATED)\)\s+([-+0-9.eE]+)",
                timing_text)
        ]
        if not area_match or not slacks:
            fail("area or setup WNS parse failure")
        write_json(WORK / "m1068_dc_mapped_vcs_receipt_r1.json", {
            "status": "PASS_M1068_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS",
            "design": DESIGN,
            "arch_mode": 0,
            "clock_period_ns": 3.0,
            "cell_area_um2": float(area_match.group(1)),
            "setup_wns_ns": min(slacks),
            "setup_timing_met": min(slacks) >= 0.0,
            "mapped_netlist_sha256": sha(netlist),
            "mapped_cases": 5,
            "anchors": ANCHORS,
            "numeric_mismatches": 0,
            "tuple_mismatches": 0,
            "weight_mismatches": 0,
            "accepted_unknowns": 0,
            "protocol_errors": 0,
            "random_register_initialization_used": False,
            "saif_files": 0,
            "ptpx_runs": 0,
            "paper_ppa_ready": False,
            "system_speedup_admitted": False,
        })
        (WORK / "RUN_COMPLETE.txt").write_text(
            "PASS_M1068_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS\n",
            encoding="utf-8")
        phase = "SUCCESS_SEAL"
        seal_dir(WORK)
        phase = "SUCCESS_ATOMIC_PUBLISH"
        atomic_publish(WORK, RESULT)
        complete = True


def quarantine_failure(message: str, return_code: int) -> None:
    if not attempt_consumed or complete:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / "failure.json", {
        "status": "FAILED_OR_INCOMPLETE",
        "phase": phase,
        "return_code": return_code,
        "message": message,
    })
    seal_dir(WORK)
    atomic_publish(WORK, FAILURE)


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, signal_handler)

try:
    run_flow()
except (GateFailure, subprocess.TimeoutExpired, OSError, ValueError,
        KeyError, json.JSONDecodeError) as exc:
    quarantine_failure(str(exc), 3)
    print(f"M1068 one-shot failure: {exc}", file=sys.stderr)
    raise SystemExit(3)

print("PASS_M1068_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS")

