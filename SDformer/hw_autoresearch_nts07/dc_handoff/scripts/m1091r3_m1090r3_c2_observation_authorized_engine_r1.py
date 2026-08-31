#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fixed-identity M1091r3 C2 diagnostic engine; source-only until M1093r2/M1096r2.

The engine accepts exactly ``--authorized-launch``.  It has no caller-selected
expected hashes.  Before it can consume an attempt it requires a fixed-path
launcher, a fixed-path double-sealed launch receipt, and independent M1093r2 and
M1096r2 seals.  Those launch artifacts intentionally do not exist at this source
stage, so direct execution fails before any namespace or EDA mutation.
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


DESIGN = "m1090r3_c2_k1_observation_wrapper"
DC_ROOT = Path(__file__).resolve().parent.parent
HW = DC_ROOT.parent
ENGINE = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1090r3_c2_k1_observation_fixed_history_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1090r3_c2_k1_observation_fixed_history_release_r1_20260830.json"
LAUNCHER = HW / "dc_handoff/scripts/run_m1091r3_m1090r3_c2_observation_authorized_launch_r1.py"
LAUNCH_RECEIPT = HW / "contracts/m1091r3_m1090r3_c2_observation_authorized_launch_receipt_r1_20260830.json"
M1093 = HW / "reviews/m1093r2_m1090r3_m1091r3_c2_observation_engine_hammer_r1_20260830"
M1096 = HW / "reviews/m1096r2_m1091r3_c2_observation_launch_hammer_r1_20260830"
M1092 = HW / "reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830"
M1093_STOP = HW / "reviews/m1093_m1090r2_m1091r2_c2_observation_engine_hammer_r1_20260830"
M1088 = HW / "reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830"
M1080_ATTEMPT = HW / "results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed"
M1080_FAILURE = HW / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"
OLD_M1091_ATTEMPT = HW / "results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed"
OLD_M1091R2_ATTEMPT = HW / "results/.m1091r2_m1090r2_c2_observation_dc_mapped_vcs_attempt_consumed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FILELIST = HW / "dc_handoff/filelists/date_m1090r3_c2_k1_observation_logic_only_dc.f"
TB = HW / "dc_handoff/tb/tb_m1090r3_c2_k1_observation_mapped_case0_short.sv"
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
RESULT = HW / "results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830"
ATTEMPT = HW / "results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed"
WORK = HW / f"results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_work.{os.getpid()}"
FAILURE = HW / f"results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
LOCK = Path("/tmp/m1091r3_m1090r3_c2_observation_eda.lock")

CONTRACT_SHA256 = "bdb443003de0e26b7dcb6e29838eec8e024e843f90ce033aac2203330287e808"
CONTRACT_OUTER_SHA256 = "d2e5d49d9e5cc11f1927ad75bc621b59f341c56352c58242b7f2dbd84db82c0d"
RELEASE_SHA256 = "15f40b39b3f96b06978b9d9966c9bfeedfcbff7c018651101c1d926f8f7df954"
RELEASE_OUTER_SHA256 = "fc6bb48800c7d595203aee21ccba140753f38fd04bc28f93425a9dd74dc9c853"
M1092_OUTER_SHA256 = "f55dc0afde8d350d1ff028c30e511eb15b2670f3ad1ee2f5643759406ca8ccb4"
M1093_STOP_OUTER_SHA256 = "8188a86aa07856217223d6d939f7b3cd8c84ee3b10d7bacc62dee777d8e2e2ac"
M1088_OUTER_SHA256 = "fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f"
M1080_ATTEMPT_OUTER_SHA256 = "21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c"
M1080_FAILURE_OUTER_SHA256 = "2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

EXTERNAL_SHA256 = {
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DC_TARGET: "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    SLOW: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    FAST: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    CELL: "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
}

SOURCE_SHA256 = {
    HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    HW / "dc_handoff/filelists/date_m1090r3_c2_k1_observation_logic_only_dc.f": "e27aadb338cf8d1c4506234c027c671508642de3551e531bf5422f0ae52ba752",
    HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl": "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    HW / "dc_handoff/tb/tb_m1090r3_c2_k1_observation_mapped_case0_short.sv": "fe6c00be2cc2747c37b37bfc556b1686563ee2108be9c27f1c507d4f120d13cd",
    HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_8bank_raw4_acc24.sv": "f9f7319fd2495dc4a67ec20ecf6f34ef8884c88f9bf49f8f2a74e7ee88e3e0f8",
    HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv": "7876d8603ef4fc6e326287aecc7a4c9a9a66cab5400bcaa3b24f498518ff9d9d",
    HW / "rtl_m1058/m1058_fc2_reset_hygiene_channel_split_registered_release_matched_8bank_raw4_acc24.sv": "33e2fa8427eff64bae3bde2c11bf7e6a3a15969aff076cb0ab7b96431227a565",
    HW / "rtl_m1058/m1058_fc2_reset_hygiene_registered_release_standalone_raw4_acc24.sv": "a3d6628f28c6c277e9feda143f3cf9e1365eaad5648ab358f75e82bbd9187768",
    HW / "rtl_m1090r3/m1090r3_c2_k1_observation_wrapper.sv": "da82f6f176848b8cfdfb3c2baa1f5ba178eb0bac4059cba020a2e69e6732a37d",
    HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv": "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    HW / "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv": "8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0",
    HW / "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv": "529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267",
    HW / "rtl_m218/m218_fc2_tagged_slice_service_island.sv": "f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1",
    HW / "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv": "44f7df331af66ba62fadf5e336b9c0c00d00f809e215aa8e091e9de011c5627e",
    HW / "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv": "6ea038ef935b1144d5424634e75446301270362c259341a8e7e7117523b25815",
    HW / "rtl_m519/m519_fc2_k1_registered_release_service_island.sv": "3811998fc48d31e6519ecc6c6cfb8f5d38db6fc6dd070e09d73a5f70b7579871",
    HW / "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv": "11080d39c06672cebb64988e931c41e1d4c04134a312aeb8e250d01f0ac576ff",
    HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv": "010fe9e6786db1d3bbcad7759bda17a783ce5cfe15cae02c5b4c9ebf96e9950b",
    HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv": "2588f890213d29aab6829dff679719c0f9ce4762c17bb061d1869b27a2f1d50e",
    HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv": "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
}

phase = "SOURCE_PREFLIGHT"
attempted = False
complete = False


class GateFailure(RuntimeError):
    pass


def fail(message: str) -> None:
    raise GateFailure(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def verify_dc_shell() -> None:
    try:
        mode = DC_SHELL.lstat().st_mode
    except FileNotFoundError:
        fail("dc_shell missing")
    if not stat.S_ISLNK(mode) or os.readlink(DC_SHELL) != "snps_shell":
        fail("dc_shell must be the exact pinned snps_shell symlink")
    if DC_SHELL.resolve(strict=True) != DC_TARGET:
        fail("dc_shell resolved target drift")
    verify_regular(DC_TARGET, EXTERNAL_SHA256[DC_TARGET])


def verify_double(path: Path, expected_sha: str, expected_outer: str) -> None:
    verify_regular(path, expected_sha)
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(side, sha(side) if side.exists() and not side.is_symlink() else "")
    if side.read_text(encoding="utf-8").split() != [expected_sha, path.relative_to(HW).as_posix()]:
        fail(f"sidecar drift: {path}")
    verify_regular(outer, expected_outer)
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail(f"outer content drift: {path}")


def verify_double_self_consistent(path: Path) -> str:
    """Verify a future receipt structurally; its exact outer is pinned by launcher."""
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    if not side.exists() or side.is_symlink() or not outer.exists() or outer.is_symlink():
        fail("launch receipt sidecar/outer absent or symlink")
    expected_sha, expected_name = side.read_text(encoding="utf-8").split()
    if expected_name != path.relative_to(HW).as_posix():
        fail("launch receipt sidecar path drift")
    verify_regular(path, expected_sha)
    if not stat.S_ISREG(side.lstat().st_mode) or not stat.S_ISREG(outer.lstat().st_mode):
        fail("launch receipt seal is not regular")
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail("launch receipt outer content drift")
    return sha(outer)


def verify_flat(directory: Path, expected_outer: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        fail(f"sealed directory absent/symlink: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if not stat.S_ISREG(manifest.lstat().st_mode) or manifest.is_symlink():
        fail("manifest non-regular")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        member = directory / name.lstrip("*")
        verify_regular(member, digest)
    verify_regular(outer, expected_outer)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("flat outer content drift")


def verify_frozen_history_flat(directory: Path, expected_outer: str) -> int:
    """Verify immutable tool output using its original followed-byte manifest.

    This exception is deliberately limited to the exact frozen M1080
    quarantine.  A manifest-listed symlink is accepted only if it resolves to
    a regular file inside the sealed directory and the bytes reached through
    the link match the manifest digest.  Live inputs continue to use
    ``verify_regular`` and can never enter this function.
    """
    if directory != M1080_FAILURE or not directory.is_dir() or directory.is_symlink():
        fail("historical evidence validator is restricted to exact M1080 quarantine")
    root = directory.resolve(strict=True)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if not stat.S_ISREG(manifest.lstat().st_mode) or manifest.is_symlink():
        fail("historical manifest non-regular")
    verify_regular(outer, expected_outer)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("historical outer content drift")
    symlinks = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        relative = Path(name.lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            fail("historical manifest path escape")
        member = directory / relative
        try:
            mode = member.lstat().st_mode
        except FileNotFoundError:
            fail("historical manifest member absent")
        if stat.S_ISLNK(mode):
            symlinks += 1
            resolved = member.resolve(strict=True)
            if resolved != root and root not in resolved.parents:
                fail("historical symlink escapes sealed directory")
            if not stat.S_ISREG(resolved.lstat().st_mode) or resolved.is_symlink():
                fail("historical symlink target is not regular")
        elif not stat.S_ISREG(mode):
            fail("historical manifest member is neither regular nor symlink")
        if sha(member) != digest:
            fail("historical followed-byte digest drift")
    return symlinks


def verify_flat_self_consistent(directory: Path) -> str:
    if not directory.is_dir() or directory.is_symlink():
        fail(f"sealed directory absent/symlink: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if not manifest.exists() or not outer.exists() or manifest.is_symlink() or outer.is_symlink():
        fail("self-consistent seal members absent or symlink")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        verify_regular(directory / name.lstrip("*"), digest)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("self-consistent outer content drift")
    return sha(outer)


def seal(directory: Path) -> None:
    members = sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")


def verify_parent_launcher(receipt: dict) -> None:
    verify_regular(LAUNCHER, receipt["launcher_sha256"])
    raw = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    argv = [item.decode(errors="strict") for item in raw if item]
    if len(argv) < 2 or Path(argv[0]) != PYTHON or Path(argv[1]).resolve() != LAUNCHER:
        fail("engine must be called by the fixed launcher under the pinned Python")


def verify_launch_authority() -> dict:
    if not LAUNCHER.exists() or not LAUNCH_RECEIPT.exists():
        fail("fixed launch wrapper/receipt absent: source stage is not executable")
    receipt = load(LAUNCH_RECEIPT)
    receipt_outer = verify_double_self_consistent(LAUNCH_RECEIPT)
    if receipt["status"] != "M1091R3_LAUNCH_SOURCE_FROZEN__M1096R2_REQUIRED__NO_EDA":
        fail("launch source receipt boundary")
    if receipt["engine_sha256"] != sha(ENGINE):
        fail("launch receipt engine drift")
    verify_regular(LAUNCHER, receipt["launcher_sha256"])
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    if f'M1093R2_OUTER_SHA256 = "{receipt["m1093r2_outer_seal_file_sha256"]}"' not in launcher_text:
        fail("launcher does not hard-code receipt M1093 outer")
    if f'ENGINE_SHA256 = "{sha(ENGINE)}"' not in launcher_text:
        fail("launcher does not hard-code engine identity")
    verify_flat(M1093, receipt["m1093r2_outer_seal_file_sha256"])
    if load(M1093 / "review.json")["status"] != "PASS_M1093R2_M1090R3_M1091R3_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA":
        fail("M1093 engine hammer has no GO")
    m1096r2_outer = verify_flat_self_consistent(M1096)
    m1096r2_review = load(M1096 / "review.json")
    if m1096r2_review["status"] != "PASS_M1096R2_M1091R3_AUTHORIZED_LAUNCH_HAMMER__GO_ONE_ATTEMPT":
        fail("M1096 launch hammer has no GO")
    if m1096r2_review["identity"]["engine_sha256"] != sha(ENGINE):
        fail("M1096 engine pin drift")
    if m1096r2_review["identity"]["launcher_sha256"] != receipt["launcher_sha256"]:
        fail("M1096 launcher pin drift")
    if m1096r2_review["identity"]["launch_receipt_outer_seal_file_sha256"] != receipt_outer:
        fail("M1096 launch receipt pin drift")
    if m1096r2_review["identity"]["m1093r2_outer_seal_file_sha256"] != receipt["m1093r2_outer_seal_file_sha256"]:
        fail("M1096 M1093 pin drift")
    verify_parent_launcher(receipt)
    receipt["m1096r2_outer_seal_file_sha256"] = m1096r2_outer
    return receipt


def static_gate() -> dict:
    if sys.argv[1:] != ["--authorized-launch"]:
        fail("fixed argv required")
    if Path(sys.executable) != PYTHON:
        fail("unpinned Python invocation")
    verify_regular(PYTHON, EXTERNAL_SHA256[PYTHON])
    verify_regular(ENGINE, sha(ENGINE))
    verify_double(CONTRACT, CONTRACT_SHA256, CONTRACT_OUTER_SHA256)
    verify_double(RELEASE, RELEASE_SHA256, RELEASE_OUTER_SHA256)
    contract = load(CONTRACT)
    release = load(RELEASE)
    if contract["status"] != "M1090R3_FIXED_HISTORY_SOURCE_ONLY__M1093R2_ENGINE_HAMMER_REQUIRED__NO_EDA" or contract["launch_now"] is not False:
        fail("contract boundary")
    if release["status"] != "M1090R3_FIXED_HISTORY_RELEASE_FROZEN__M1093R2_ENGINE_HAMMER_REQUIRED__NO_EDA" or release["launch_now"] is not False:
        fail("release boundary")
    if release["contract_sha256"] != CONTRACT_SHA256 or release["contract_outer_seal_file_sha256"] != CONTRACT_OUTER_SHA256:
        fail("release-to-contract binding")
    if sha(DOCS359) != DOCS359_SHA256:
        fail("docs359 drift")
    for path, expected in SOURCE_SHA256.items():
        verify_regular(path, expected)
    for path, expected in EXTERNAL_SHA256.items():
        if path != DC_TARGET:
            verify_regular(path, expected)
    verify_dc_shell()
    verify_flat(M1092, M1092_OUTER_SHA256)
    if load(M1092 / "review.json")["status"] != "STOP_M1092_M1090_M1091_SELF_SIGNED_CALLER_AUTHORITY__NO_M1091_ATTEMPT":
        fail("M1092 STOP drift")
    verify_flat(M1093_STOP, M1093_STOP_OUTER_SHA256)
    if load(M1093_STOP / "review.json")["status"] != "STOP_M1093_M1091R2_ENGINE_REJECTS_FROZEN_M1080_QUARANTINE_SYMLINK__NO_EDA_NO_ATTEMPT":
        fail("M1093 STOP drift")
    verify_flat(M1088, M1088_OUTER_SHA256)
    if load(M1088 / "review.json")["status"] != "PASS_M1088_M1080_FAILURE_AUDIT__M1080_DO_NOT_RETRY":
        fail("M1080 retry boundary")
    verify_flat(M1080_ATTEMPT, M1080_ATTEMPT_OUTER_SHA256)
    if verify_frozen_history_flat(M1080_FAILURE, M1080_FAILURE_OUTER_SHA256) != 1:
        fail("historical M1080 quarantine symlink count drift")
    if OLD_M1091_ATTEMPT.exists() or OLD_M1091_ATTEMPT.is_symlink():
        fail("old M1091 attempt must remain absent; old runner is DO_NOT_RUN")
    if OLD_M1091R2_ATTEMPT.exists() or OLD_M1091R2_ATTEMPT.is_symlink():
        fail("old M1091r2 attempt must remain absent; stopped engine is DO_NOT_RUN")
    return verify_launch_authority()


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


def flow() -> None:
    global phase, attempted, complete
    launch_receipt = static_gate()
    if any(path.exists() or path.is_symlink() for path in (RESULT, ATTEMPT, WORK)):
        fail("fresh namespace collision")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("exclusive lock busy")
        collision_gate()
        resource_gate()
        license_gate()
        phase = "ATTEMPT_CONSUME_AFTER_FIXED_LAUNCH_AUTHORITY"
        ATTEMPT.mkdir()
        attempted = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1091R3_ATTEMPT_CONSUMED_AFTER_M1093R2_M1096R2",
            "engine_sha256": sha(ENGINE),
            "contract_sha256": CONTRACT_SHA256,
            "release_sha256": RELEASE_SHA256,
            "launcher_sha256": launch_receipt["launcher_sha256"],
            "m1093r2_outer_seal_file_sha256": launch_receipt["m1093r2_outer_seal_file_sha256"],
            "m1096r2_outer_seal_file_sha256": launch_receipt["m1096r2_outer_seal_file_sha256"],
            "dc_attempts": 1,
            "mapped_cases": 1,
            "activity_files": 0,
            "random_initialization": False,
        })
        seal(ATTEMPT)
        WORK.mkdir()
        env = os.environ.copy()
        env.update({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"})
        phase = "FRESH_DC_M1090R3_OBSERVATION_TOP"
        dc = WORK / "dc"
        dc.mkdir()
        dc_env = env.copy()
        dc_env.update({
            "DESIGN_NAME": DESIGN,
            "HW_ROOT": str(HW),
            "RTL_FILELIST": str(FILELIST),
            "LIB_DB": str(SLOW),
            "MIN_LIB_DB": str(FAST),
            "SDC_FILE": str(SDC),
            "OUTPUT_DIR": str(dc),
            "ELAB_PARAMETERS": "",
            "OPERATING_CONDITION": "ssg0p9v125c",
        })
        rc = run([str(DC_SHELL), "-f", str(DC_TCL)], dc / "dc.log", 21600, dc_env)
        (dc / "dc.rc").write_text(str(rc) + "\n", encoding="utf-8")
        if rc or not (dc / "TCL_PASS_TERMINAL.txt").is_file():
            fail("fresh DC failed")
        netlist = dc / f"netlist/{DESIGN}_mapped.v"
        if not netlist.is_file() or not netlist.stat().st_size:
            fail("mapped netlist absent")
        phase = "FRESH_MAPPED_VCS_CASE0_COMPILE"
        mapped = WORK / "mapped_vcs"
        mapped.mkdir()
        simv = mapped / "simv"
        rc = run([
            str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            f"-Mdir={mapped / 'csrc'}", str(CELL), str(netlist), str(MEMORY),
            str(TB), "-top", "tb_m1090r3_c2_k1_observation_mapped_case0_short",
            "-o", str(simv),
        ], mapped / "compile.log", 1800, env)
        (mapped / "compile.rc").write_text(str(rc) + "\n", encoding="utf-8")
        if rc or not simv.is_file():
            fail("mapped compile failed")
        phase = "FRESH_MAPPED_VCS_CASE0_SHORT_128"
        rc = run([str(simv), "-no_save"], mapped / "case0.log", 300, env)
        (mapped / "case0.rc").write_text(str(rc) + "\n", encoding="utf-8")
        text = (mapped / "case0.log").read_text(errors="replace")
        token = "PASS_M1090R3_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 no_unknown=1 diagnostic_only=1"
        if rc or token not in text:
            fail("short observation window found X/stall; inspect quarantine case0.log")
        write_json(WORK / "receipt.json", {
            "status": "PASS_M1091R3_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW",
            "mapped_netlist_sha256": sha(netlist),
            "stage_lines": len(re.findall(r"^M1090R3_STAGE", text, re.M)),
            "window_cycles": 128,
            "unknowns": 0,
            "diagnostic_only": True,
            "paper_citable": False,
        })
        (WORK / "RUN_COMPLETE.txt").write_text("PASS_M1091R3_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW\n", encoding="utf-8")
        seal(WORK)
        os.rename(WORK, RESULT)
        complete = True


def quarantine(message: str) -> None:
    if attempted and not complete:
        WORK.mkdir(parents=True, exist_ok=True)
        write_json(WORK / "failure.json", {
            "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE",
            "phase": phase,
            "message": message,
            "m1080_retry": False,
        })
        seal(WORK)
        os.rename(WORK, FAILURE)


def handler(signum: int, _frame: object) -> None:
    raise GateFailure("signal " + str(signum))


for caught_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(caught_signal, handler)

try:
    flow()
except (GateFailure, OSError, subprocess.TimeoutExpired, KeyError, ValueError, json.JSONDecodeError) as exc:
    quarantine(str(exc))
    print("M1091r2 failure: " + str(exc), file=sys.stderr)
    raise SystemExit(3)
print("PASS_M1091R3_FRESH_DC_MAPPED_OBSERVATION_SHORT_WINDOW")
