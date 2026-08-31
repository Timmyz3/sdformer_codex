#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1112r3 fixed-identity C2 async-observation diagnostic engine source.

Source-only until a different-author engine hammer, a fixed zero-argument
launcher receipt, and a second launch hammer exist.  No caller-selected path,
hash, reset name, cell name, manifest member, or environment is accepted.
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


DESIGN = "m1112r2_c2_k1_async_observation_shadow_wrapper"
TB_TOP = "tb_m1112r2_c2_k1_async_observation_shadow_case0_short"
DC_ROOT = Path(__file__).resolve().parent.parent
HW = DC_ROOT.parent
ENGINE = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
LAUNCHER = HW / "dc_handoff/scripts/run_m1112r3_c2_async_observation_authorized_launch_r1.py"
LAUNCH_RECEIPT = HW / "contracts/m1112r3_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
M1117R3 = HW / "reviews/m1117r3_m1112r3_c2_async_observation_engine_hammer_r1_20260830"
M1118R3 = HW / "reviews/m1118r3_m1112r3_c2_async_observation_launch_hammer_r1_20260830"
M1116 = HW / "reviews/m1116_m1112r2_c2_launch_chain_circularity_audit_r1_20260830"
M1109 = HW / "reviews/m1109_m1091r3_c2_observation_mapped_x_failure_audit_r1_20260830"
M1113_STOP = HW / "reviews/m1113_m1112_c2_async_observation_engine_hammer_r1_20260830"
M1088 = HW / "reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830"
M1080_ATTEMPT = HW / "results/.m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed"
M1080_FAILURE = HW / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine"
OLD_M1112_ATTEMPT = HW / "results/.m1112_c2_async_observation_dc_mapped_vcs_attempt_consumed"
OLD_M1112R2_ATTEMPT = HW / "results/.m1112r2_c2_async_observation_dc_mapped_vcs_attempt_consumed"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FILELIST = HW / "dc_handoff/filelists/date_m1112r2_c2_k1_async_observation_shadow_logic_only_dc.f"
TB = HW / "dc_handoff/tb/tb_m1112r2_c2_k1_async_observation_shadow_case0_short.sv"
MEMORY = HW / "tb_m349/m349_fc2_scalar_bank_memory_model.sv"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
SLOW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
FAST = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
CELL = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v")
RESULT = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"
ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
WORK = HW / f"results/.m1112r3_c2_async_observation_dc_mapped_vcs_work.{os.getpid()}"
FAILURE = HW / f"results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.{os.getpid()}.quarantine"
LOCK = Path("/tmp/m1112r3_c2_async_observation_eda.lock")

CONTRACT_SHA256 = "92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef"
CONTRACT_OUTER_SHA256 = "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf"
M1113_STOP_OUTER_SHA256 = "ee665be8def8c598669566467a6d1e59dc021a3b0743e2faf43122ed0991da64"
M1116_OUTER_SHA256 = "3aacd467ac2a7d3fcd58d82e0977a53541b317abc1d88cf71e34ee0b95651f94"
M1109_OUTER_SHA256 = "5c7a1f667c6c800f84a0e8219ddf58574412090812cda5d8bdaf36265f43d52d"
M1088_OUTER_SHA256 = "fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f"
M1080_ATTEMPT_OUTER_SHA256 = "21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c"
M1080_FAILURE_OUTER_SHA256 = "2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SHADOW_REGISTER_BITS = 337
CANONICAL_RESET = "rst_core"

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
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                fail("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            GateFailure("nonfinite JSON: " + token)))


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


def safe_manifest_names(manifest: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            fail(f"malformed manifest line: {manifest}")
        name = fields[1].lstrip("*")
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            fail(f"unsafe manifest member: {name}")
        if name in entries:
            fail(f"duplicate manifest member: {name}")
        entries[name] = fields[0]
    return entries


def verify_double(path: Path, expected: str, expected_outer: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, expected)
    verify_regular(side, sha(side) if side.exists() and not side.is_symlink() else "")
    verify_regular(outer, expected_outer)
    if side.read_text(encoding="utf-8").split() != [expected, path.relative_to(HW).as_posix()]:
        fail(f"sidecar content drift: {path}")
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail(f"outer content drift: {path}")


def verify_double_self_consistent(path: Path) -> str:
    """Verify future receipt metadata without creating an identity cycle."""
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    if not side.exists() or side.is_symlink() or not outer.exists() or outer.is_symlink():
        fail("future receipt seal metadata absent/symlink")
    verify_regular(side, sha(side))
    verify_regular(outer, sha(outer))
    fields = side.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != path.relative_to(HW).as_posix():
        fail("future receipt sidecar content")
    verify_regular(path, fields[0])
    if outer.read_text(encoding="utf-8").split() != [sha(side), side.relative_to(HW).as_posix()]:
        fail("future receipt outer content")
    return sha(outer)


def verify_exact_flat(directory: Path, expected_outer: str) -> None:
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        fail(f"sealed directory absent: {directory}")
    if not stat.S_ISDIR(mode) or directory.is_symlink():
        fail(f"sealed directory non-directory/symlink: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, sha(manifest) if manifest.exists() and not manifest.is_symlink() else "")
    verify_regular(outer, expected_outer)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail(f"sealed outer content drift: {directory}")
    expected = safe_manifest_names(manifest)
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        if stat.S_ISLNK(member_mode):
            fail(f"live sealed symlink rejected: {relative}")
        if stat.S_ISREG(member_mode):
            actual.add(relative)
        elif not stat.S_ISDIR(member_mode):
            fail(f"live sealed special member rejected: {relative}")
    if actual != set(expected):
        fail(f"sealed exact member mismatch missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)


def verify_flat_self_consistent(directory: Path) -> str:
    """Discover a future hammer outer without making it a receipt dependency."""
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        fail(f"future sealed directory absent: {directory}")
    if not stat.S_ISDIR(mode) or directory.is_symlink():
        fail(f"future sealed directory non-directory/symlink: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, sha(manifest) if manifest.exists() and not manifest.is_symlink() else "")
    verify_regular(outer, sha(outer) if outer.exists() and not outer.is_symlink() else "")
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("future sealed outer content drift")
    expected = safe_manifest_names(manifest)
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        if stat.S_ISLNK(member_mode):
            fail("future sealed symlink rejected: " + relative)
        if stat.S_ISREG(member_mode):
            actual.add(relative)
        elif not stat.S_ISDIR(member_mode):
            fail("future sealed special member rejected: " + relative)
    if actual != set(expected):
        fail("future sealed exact member mismatch")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    return sha(outer)


def verify_historical_m1080() -> None:
    """Only M1080 may use its original manifest followed-byte symlink rule."""
    directory = M1080_FAILURE
    if directory != HW / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine":
        fail("historical exception path drift")
    if not stat.S_ISDIR(directory.lstat().st_mode) or directory.is_symlink():
        fail("historical directory absent/symlink")
    root = directory.resolve(strict=True)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(manifest, sha(manifest) if manifest.exists() and not manifest.is_symlink() else "")
    verify_regular(outer, M1080_FAILURE_OUTER_SHA256)
    if outer.read_text(encoding="utf-8").split() != [sha(manifest), "SHA256SUMS"]:
        fail("historical outer drift")
    expected = safe_manifest_names(manifest)
    actual: set[str] = set()
    symlinks = 0
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            actual.add(relative)
        elif not stat.S_ISDIR(mode):
            fail("historical special member")
    if actual != set(expected):
        fail("historical exact member mismatch")
    for name, digest in expected.items():
        member = directory / name
        mode = member.lstat().st_mode
        if stat.S_ISLNK(mode):
            symlinks += 1
            resolved = member.resolve(strict=True)
            if resolved != root and root not in resolved.parents:
                fail("historical symlink path escape")
            if not stat.S_ISREG(resolved.lstat().st_mode) or resolved.is_symlink():
                fail("historical symlink target not regular")
        elif not stat.S_ISREG(mode):
            fail("historical manifest member type")
        if sha(member) != digest:
            fail("historical followed-byte digest drift")
    if symlinks != 1:
        fail("historical symlink census drift")


def parse_instances(text: str) -> list[tuple[str, str, dict[str, str]]]:
    instances: list[tuple[str, str, dict[str, str]]] = []
    pattern = re.compile(r"(?ms)^\s*(\w+)\s+(\\?[^\s(]+)\s*\((.*?)\)\s*;")
    for match in pattern.finditer(text):
        pins: dict[str, str] = {}
        for pin, net in re.findall(r"\.(\w+)\s*\(\s*([^()\s,]+)\s*\)", match.group(3)):
            if pin in pins:
                fail(f"duplicate mapped pin {match.group(2)}.{pin}")
            pins[pin] = net
        instances.append((match.group(1), match.group(2), pins))
    return instances


def is_allowed_inverter(cell: str, pins: dict[str, str]) -> bool:
    allowed = bool(re.fullmatch(r"(?:INV(?:D(?:0P7|1P5|0|1|2|3|4|6|8|9|12|15|16|18|20|21|24|32))|CKND(?:0|1|2|3|4|6|8|12|16|20|24))BWP35P140", cell))
    return allowed and set(pins) == {"I", "ZN"}


def structural_reset_gate_text(text: str) -> dict:
    """Trace every shadow async pin to canonical reset with exact polarity.

    RTL reset is active-high and drives Q to zero.  TSMC28 CDN/CN clear pins
    are active-low and therefore must be driven by exactly one allowed inverter
    whose sole input is rst_core.  Active-low set pins are never a valid reset-
    to-zero source.  Constants, direct wrong-polarity reset, unrelated nets,
    multi-level logic, buffers, gates, and reconvergent cones are rejected.
    """
    instances = parse_instances(text)
    drivers: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    for cell, name, pins in instances:
        for output_pin in ("ZN", "Z", "Q", "QN"):
            if output_pin in pins:
                drivers.setdefault(pins[output_pin], []).append((cell, name, pins))
    shadow = [(cell, name, pins) for cell, name, pins in instances if re.search(r"shadow_\w+_q_reg", name)]
    if len(shadow) != SHADOW_REGISTER_BITS:
        fail(f"mapped shadow register census {len(shadow)} != {SHADOW_REGISTER_BITS}")
    accepted_reset_nets: set[str] = set()
    cell_types: set[str] = set()
    for cell, name, pins in shadow:
        cell_types.add(cell)
        clear_pins = [pin for pin in ("CDN", "CN") if pin in pins]
        set_pins = [pin for pin in ("SDN", "SN") if pin in pins]
        if len(clear_pins) != 1:
            fail(f"{name}: exactly one active-low clear pin required")
        clear_net = pins[clear_pins[0]]
        if clear_net in {CANONICAL_RESET, "1'b0", "1'b1", "1'h0", "1'h1", "1", "0"}:
            fail(f"{name}: wrong-polarity direct/constant clear source {clear_net}")
        source = drivers.get(clear_net, [])
        if len(source) != 1:
            fail(f"{name}: clear source must have exactly one driver")
        inv_cell, inv_name, inv_pins = source[0]
        if not is_allowed_inverter(inv_cell, inv_pins):
            fail(f"{name}: clear source is not an allowed single-input inverter")
        if inv_pins["I"] != CANONICAL_RESET:
            fail(f"{name}: inverter input is not canonical rst_core")
        if inv_pins["ZN"] != clear_net:
            fail(f"{name}: inverter output mismatch")
        accepted_reset_nets.add(clear_net)
        for set_pin in set_pins:
            if pins[set_pin] not in {"1'b1", "1'h1", "1"}:
                fail(f"{name}: async set must be inactive high")
    return {
        "shadow_register_bits": len(shadow),
        "resettable_cell_types": sorted(cell_types),
        "canonical_reset": CANONICAL_RESET,
        "inversion_depth": 1,
        "active_low_clear_nets": sorted(accepted_reset_nets),
    }


def structural_reset_gate(netlist: Path) -> dict:
    mode = netlist.lstat().st_mode
    if not stat.S_ISREG(mode) or netlist.is_symlink():
        fail("mapped netlist non-regular/symlink")
    return structural_reset_gate_text(netlist.read_text(encoding="utf-8", errors="strict"))


def verify_dc_executable() -> None:
    """Invoke the pinned regular target directly; no live tool symlink."""
    verify_regular(DC_TARGET, EXTERNAL_SHA256[DC_TARGET])


def verify_parent_launcher(receipt: dict) -> None:
    verify_regular(LAUNCHER, receipt["launcher_sha256"])
    raw = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    argv = [item.decode(errors="strict") for item in raw if item]
    if len(argv) != 2 or Path(argv[0]) != PYTHON or Path(argv[1]).resolve() != LAUNCHER:
        fail("zero-argument fixed launcher parent required")


def verify_future_authority() -> dict:
    if not LAUNCHER.exists() or not LAUNCH_RECEIPT.exists():
        fail("future launcher/receipt absent: source stage cannot execute")
    receipt_outer = verify_double_self_consistent(LAUNCH_RECEIPT)
    receipt = load(LAUNCH_RECEIPT)
    expected_receipt_keys = {
        "schema", "status", "launcher_sha256", "engine_sha256",
        "engine_contract_sha256", "engine_contract_outer_seal_file_sha256",
        "engine_author_receipt_outer_seal_file_sha256",
        "m1116_outer_seal_file_sha256", "m1117r3_outer_seal_file_sha256",
        "arguments", "caller_selected_authority_allowed",
        "caller_environment_forwarded", "m1118r3_required", "launch_now",
        "attempt_now", "dc_now", "mapped_vcs_now", "maximum_attempts",
        "automatic_retry", "paper_citable",
    }
    if set(receipt) != expected_receipt_keys:
        fail("future launch receipt exact-key drift")
    if receipt["status"] != "M1112R3_LAUNCH_SOURCE_FROZEN__M1118R3_REQUIRED__NO_EDA":
        fail("future launch receipt status")
    if (receipt["schema"] != "m1112r3_c2_authorized_launch_receipt_r1_v1" or
            receipt["engine_sha256"] != sha(ENGINE) or
            receipt["engine_contract_sha256"] != CONTRACT_SHA256 or
            receipt["engine_contract_outer_seal_file_sha256"] != CONTRACT_OUTER_SHA256 or
            receipt["m1116_outer_seal_file_sha256"] != M1116_OUTER_SHA256 or
            receipt["arguments"] != 0 or receipt["caller_selected_authority_allowed"] is not False or
            receipt["caller_environment_forwarded"] is not False or
            receipt["m1118r3_required"] is not True or receipt["launch_now"] is not False or
            receipt["attempt_now"] is not False or receipt["dc_now"] is not False or
            receipt["mapped_vcs_now"] is not False or receipt["maximum_attempts"] != 1 or
            receipt["automatic_retry"] is not False or receipt["paper_citable"] is not False or
            "m1118r3_outer_seal_file_sha256" in receipt):
        fail("future launch receipt authority/boundary drift")
    verify_exact_flat(M1117R3, receipt["m1117r3_outer_seal_file_sha256"])
    engine_review = load(M1117R3 / "review.json")
    if (engine_review["status"] !=
            "PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA" or
            engine_review["identity"]["engine_sha256"] != sha(ENGINE) or
            engine_review["identity"]["contract_sha256"] != CONTRACT_SHA256 or
            engine_review["identity"]["m1116_outer_seal_file_sha256"] != M1116_OUTER_SHA256 or
            receipt["engine_author_receipt_outer_seal_file_sha256"] !=
                engine_review["identity"]["author_receipt_outer_seal_file_sha256"]):
        fail("M1117r3 has no GO")
    launch_outer = verify_flat_self_consistent(M1118R3)
    launch_review = load(M1118R3 / "review.json")
    if launch_review["status"] != "PASS_M1118R3_M1112R3_LAUNCH_HAMMER__GO_ONE_ATTEMPT":
        fail("M1118r3 has no GO")
    identity = launch_review["identity"]
    if (identity["launch_receipt_outer_seal_file_sha256"] != receipt_outer or
            identity["launcher_sha256"] != receipt["launcher_sha256"] or
            identity["engine_sha256"] != sha(ENGINE) or
            identity["engine_contract_outer_seal_file_sha256"] != CONTRACT_OUTER_SHA256 or
            identity["engine_author_receipt_outer_seal_file_sha256"] !=
                receipt["engine_author_receipt_outer_seal_file_sha256"] or
            identity["m1116_outer_seal_file_sha256"] != M1116_OUTER_SHA256 or
            identity["m1117r3_outer_seal_file_sha256"] !=
                receipt["m1117r3_outer_seal_file_sha256"]):
        fail("M1118r3 exact launcher/receipt/engine authority drift")
    verify_parent_launcher(receipt)
    result = dict(receipt)
    result["m1118r3_outer_seal_file_sha256"] = launch_outer
    return result


def static_gate() -> dict:
    if sys.argv[1:] != ["--authorized-launch"]:
        fail("fixed argv required")
    if Path(sys.executable) != PYTHON:
        fail("unpinned Python")
    verify_regular(ENGINE, sha(ENGINE))
    verify_double(CONTRACT, CONTRACT_SHA256, CONTRACT_OUTER_SHA256)
    contract = load(CONTRACT)
    if contract["status"] != "M1112R3_RESET_PROVENANCE_AND_LIVE_SEAL_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED__NO_EDA":
        fail("contract status")
    if contract["launch_now"] is not False or contract["max_attempts_now"] != 0:
        fail("source attempt boundary")
    if (contract["m1116_circularity_authority"]["outer_seal_file_sha256"] != M1116_OUTER_SHA256 or
            contract["m1109_failure_authority"]["outer_seal_file_sha256"] != M1109_OUTER_SHA256 or
            contract["m1113_stop_authority"]["outer_seal_file_sha256"] != M1113_STOP_OUTER_SHA256 or
            contract["m1112r2_frozen_authority"]["engine_sha256"] !=
                "cd4f3eb4d9c659b14fca143651b2e5a4c0d3147335469b9ec22063b1113980c4" or
            contract["m1112r2_frozen_authority"]["m1114r2_outer_seal_file_sha256"] !=
                "15e1f136aa4d892a965a005f97a5845d81a144634f86c9db432bfdf4bec884a9" or
            contract["m1112r2_frozen_authority"]["launcher_go_withdrawn_by_m1116"] is not True or
            contract["future_chain"]["launch_receipt_contains_future_m1118r3_outer"] is not False or
            contract["future_chain"]["m1118r3_outer_discovery"] !=
                "verify_flat_self_consistent at authorized execution" or
            contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is not False or
            contract["frozen_stopped_namespaces"]["m1091r3_attempt_reused"] is not False or
            contract["frozen_stopped_namespaces"]["m1112r2_attempt_reused"] is not False or
            contract["frozen_stopped_namespaces"]["m1112r3_maximum_attempts_after_all_hammers"] != 1 or
            contract["frozen_stopped_namespaces"]["automatic_retry"] is not False or
            contract["frozen_stopped_namespaces"]["post_attempt_failure_quarantine_required"] is not True):
        fail("M1112r3 launch-chain/namespace contract drift")
    verify_regular(DOCS359, DOCS359_SHA256)
    for relative, expected in contract["source_sha256"].items():
        verify_regular(HW / relative, expected)
    for relative, expected in contract["frozen_filelist_member_sha256"].items():
        verify_regular(HW / relative, expected)
    for path, expected in EXTERNAL_SHA256.items():
        if path != DC_TARGET:
            verify_regular(path, expected)
    verify_dc_executable()
    verify_exact_flat(M1116, M1116_OUTER_SHA256)
    if (load(M1116 / "review.json")["status"] !=
            "STOP_M1116_M1112R2_FUTURE_LAUNCH_HASH_CYCLE__ADDITIVE_R3_REQUIRED"):
        fail("M1116 STOP drift")
    verify_exact_flat(M1109, M1109_OUTER_SHA256)
    if (load(M1109 / "review.json")["status"] !=
            "PASS_M1109_FAILURE_AUDIT__M1091R3_DO_NOT_RETRY__NEW_ASYNC_OBSERVATION_SHADOW_REPAIR_ONLY"):
        fail("M1109 authority drift")
    verify_exact_flat(M1113_STOP, M1113_STOP_OUTER_SHA256)
    if load(M1113_STOP / "review.json")["status"] != "FAIL_M1113_ENGINE_HAMMER__SOURCE_REPAIR_REQUIRED__NO_LAUNCHER_NO_EDA":
        fail("M1113 STOP drift")
    verify_exact_flat(M1088, M1088_OUTER_SHA256)
    verify_exact_flat(M1080_ATTEMPT, M1080_ATTEMPT_OUTER_SHA256)
    verify_historical_m1080()
    if (OLD_M1112_ATTEMPT.exists() or OLD_M1112_ATTEMPT.is_symlink() or
            OLD_M1112R2_ATTEMPT.exists() or OLD_M1112R2_ATTEMPT.is_symlink()):
        fail("old stopped M1112/M1112r2 attempt must remain absent")
    if any(path.exists() or path.is_symlink() for path in (ATTEMPT, RESULT, WORK)):
        fail("fresh M1112r3 namespace collision")
    return verify_future_authority()


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
        phase = "ATTEMPT_CONSUME_AFTER_M1117R3_M1118R3"
        ATTEMPT.mkdir(); attempted = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1112R3_ATTEMPT_CONSUMED_AFTER_M1117R3_M1118R3",
            "engine_sha256": sha(ENGINE), "contract_sha256": CONTRACT_SHA256,
            "launcher_sha256": authority["launcher_sha256"],
            "m1117r3_outer_seal_file_sha256": authority["m1117r3_outer_seal_file_sha256"],
            "m1118r3_outer_seal_file_sha256": authority["m1118r3_outer_seal_file_sha256"],
            "dc_attempts": 1, "mapped_cases": 1, "random_initialization": False,
        }); seal(ATTEMPT)
        WORK.mkdir()
        env = os.environ.copy()
        env.update({"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "PATH": "/opt/synopsys/vcs/V-2023.12-SP1/bin:/usr/bin:/bin"})
        phase = "FRESH_DC_M1112R3"
        dc = WORK / "dc"; dc.mkdir()
        dc_env = env.copy()
        dc_env.update({
            "DESIGN_NAME": DESIGN, "HW_ROOT": str(HW), "RTL_FILELIST": str(FILELIST),
            "LIB_DB": str(SLOW), "MIN_LIB_DB": str(FAST), "SDC_FILE": str(SDC),
            "OUTPUT_DIR": str(dc), "ELAB_PARAMETERS": "", "OPERATING_CONDITION": "ssg0p9v125c",
        })
        rc = run([str(DC_TARGET), "-f", str(DC_TCL)], dc / "dc.log", 21600, dc_env)
        if rc or not (dc / "TCL_PASS_TERMINAL.txt").is_file():
            fail("fresh DC failed")
        netlist = dc / f"netlist/{DESIGN}_mapped.v"
        if not netlist.is_file() or not netlist.stat().st_size:
            fail("mapped netlist absent")
        phase = "MAPPED_RESET_PROVENANCE_337"
        reset_census = structural_reset_gate(netlist)
        phase = "FRESH_MAPPED_VCS_CASE0_SHORT_128"
        mapped = WORK / "mapped_vcs"; mapped.mkdir(); simv = mapped / "simv"
        rc = run([
            str(VCS), "-full64", "-sverilog", "+v2k", "-timescale=1ns/1ps",
            f"-Mdir={mapped / 'csrc'}", str(CELL), str(netlist), str(MEMORY), str(TB),
            "-top", TB_TOP, "-o", str(simv),
        ], mapped / "compile.log", 1800, env)
        if rc or not simv.is_file():
            fail("mapped compile failed")
        rc = run([str(simv), "-no_save"], mapped / "case0.log", 300, env)
        text = (mapped / "case0.log").read_text(encoding="utf-8", errors="replace")
        token = "PASS_M1112_ASYNC_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 unknown_bitmap=000000 diagnostic_only=1"
        if rc or token not in text or "M1112_FIRST_X" in text:
            fail("mapped short window failed")
        write_json(WORK / "receipt.json", {
            "status": "PASS_M1112R3_RESET_PROVENANCE_MAPPED_SHORT_WINDOW",
            "mapped_netlist_sha256": sha(netlist), "reset_provenance": reset_census,
            "stage_lines": len(re.findall(r"^M1112_STAGE", text, re.M)),
            "window_cycles": 128, "unknown_bitmap": "000000",
            "diagnostic_only": True, "paper_citable": False,
        })
        (WORK / "RUN_COMPLETE.txt").write_text("PASS_M1112R3_RESET_PROVENANCE_MAPPED_SHORT_WINDOW\n", encoding="utf-8")
        seal(WORK); os.rename(WORK, RESULT); complete = True


def quarantine(message: str) -> None:
    if attempted and not complete:
        WORK.mkdir(parents=True, exist_ok=True)
        write_json(WORK / "failure.json", {
            "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE", "phase": phase,
            "message": message, "m1112_retry": False,
        })
        seal(WORK); os.rename(WORK, FAILURE)


def main() -> int:
    for caught_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(caught_signal, lambda signum, _frame: (_ for _ in ()).throw(GateFailure("signal " + str(signum))))
    try:
        flow()
    except (GateFailure, OSError, subprocess.TimeoutExpired, KeyError, ValueError, json.JSONDecodeError) as exc:
        quarantine(str(exc))
        print("M1112r3 failure: " + str(exc), file=sys.stderr)
        return 3
    print("PASS_M1112R3_RESET_PROVENANCE_MAPPED_SHORT_WINDOW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
