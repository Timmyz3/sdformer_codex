#!/usr/bin/env python3
"""One-shot C1 M1701 salvage Formality plus independent PrimeTime campaign.

Source-only until an independent M1723 review and a separately sealed M1724
release are exact-SHA pinned by the caller.  The runner never invokes DC and
never mutates or promotes the M1701 quarantine.
"""
from __future__ import print_function

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1722_m1701_c1_salvage_formality_pt_source_contract_r1_20260901.json"
TEST = HW / "system_simulator/tests/test_m1722_m1701_c1_salvage_formality_pt_source.py"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m1722_c1_m1665_to_m1701_gate_to_gate.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1722_c1_m1701_slowmax_fastmin.tcl"
M1723 = HW / "reviews/m1723_m1722_m1701_c1_salvage_formality_pt_source_hammer_r1_20260901"
M1724 = HW / "contracts/m1724_m1723_m1722_m1701_c1_salvage_formality_pt_launch_release_r1_20260901.json"

TOP = "m935_m912_three_stage_exact_parent_match_product_capture_island"
M1701 = HW / "dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901.failed_or_incomplete.2502881.quarantine"
M1665 = HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901/original_quarantine"
M1714 = HW / "reviews/m1714_m1701_c1_tool_entity_repair_quarantine_readonly_salvage_audit_r1_20260901"
M1701_NETLIST = M1701 / ("netlist/" + TOP + "_m1695_fastmin_hold_closed_mapped.v")
M1701_SDC = M1701 / ("netlist/" + TOP + "_m1695_fastmin_hold_closed_mapped.sdc")
M1701_DDC = M1701 / ("netlist/" + TOP + "_m1695_fastmin_hold_closed.ddc")
M1701_SVF = M1701 / ("netlist/" + TOP + "_m1695_fastmin_hold_closed.svf")
M1665_NETLIST = M1665 / ("netlist/" + TOP + "_m1630_residual_hold_closed_mapped.v")
M1665_SDC = M1665 / ("netlist/" + TOP + "_m1630_residual_hold_closed_mapped.sdc")
M1665_DDC = M1665 / ("netlist/" + TOP + "_m1630_residual_hold_closed.ddc")
M1665_SVF = M1665 / ("netlist/" + TOP + "_m1630_residual_hold_closed.svf")

FM = Path("/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell")
PT = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LICENSE_SERVER = "27030@ic.ismd-nemo"
STD_SLOW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
STD_FAST = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
MACRO_ROOT = Path("/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821")
MACRO_SLOW = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.db"
MACRO_FAST = MACRO_ROOT / "ts1n28hpcphvtb128x128m4s_180a_ffg1p05vm40c.db"

RESULT = HW / "dc_handoff/runs/m1722_m1701_c1_salvage_formality_pt_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1722_m1701_c1_salvage_formality_pt_attempt_consumed"
FAILURE = HW / "dc_handoff/runs/m1722_m1701_c1_salvage_formality_pt_r1_20260901.failed_or_incomplete.quarantine"
WORK = HW / ("dc_handoff/runs/.m1722_m1701_c1_salvage_formality_pt_work." + str(os.getpid()))
STAGE = HW / ("dc_handoff/runs/.m1722_m1701_c1_salvage_formality_pt_stage." + str(os.getpid()))
LOCK = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")
AREA_CEILING = 168188.4885824
GUI_ALLOW = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
CLAIMS = dict((key, False) for key in (
    "formality", "independent_pt", "dc", "power", "energy",
    "cycle_speedup", "system_speedup", "paper_ppa_ready", "paper_citable",
    "headline"))

FIXED_SHA = {
    "m1701_manifest": "f132ca694a747e2da51708fb03f2ba6c84360606b4d38d2cc2e97998f9f3a022",
    "m1701_outer": "a65f2901b4ab4339a94bb032b9412b652a77afc50d1c72b403c8bd44d15f55a6",
    "m1701_netlist": "d990bb416370fd07a1c241849e2fa494b94a179b47687a1a3ff2b1ab92c255e8",
    "m1701_sdc": "04cb67affcfd629cd9540d789110107888d9ae956168dae37c34aa44c15e2d62",
    "m1701_ddc": "7966c00c5f456a0e6d5da29cfccd837a0dccdcf2ad72daf96cf74f0b7de3b0db",
    "m1701_svf": "fc0d3534a5e8f2ce38ba4eb765eaea8ddf2a4f27e6291af670c56568010910c8",
    "m1665_manifest": "e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843",
    "m1665_outer": "c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44",
    "m1665_netlist": "842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee",
    "m1665_sdc": "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
    "m1665_ddc": "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
    "m1665_svf": "7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274",
    "m1714_review": "16da57cc633f9731b5bf641a79a9a64a224a060984df41162e24ca62ac3b55ac",
    "m1714_manifest": "acf21d2e06385c67974ece1e77efe783136fd7883b3e52a9412e057b1f6a9aae",
    "m1714_outer": "a3824a44279c9add18f94394133e20395954dcb6e22c0ffd6b2599ceeb8d475d",
    "fm": "aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b",
    "pt": "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    "lmutil": "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    "license": "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490",
    "std_slow": "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    "std_fast": "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    "macro_slow": "cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf",
    "macro_fast": "8c163161060d8d4415837da4ad65bbd83c99eb64872df76f5e0adc0b18cedb5f",
    "macro_manifest": "c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


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
        raise Failure("JSON root is not object")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid: " + str(root))
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    exact(manifest, manifest_sha)
    exact(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise Failure("outer seal content drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise Failure("unsafe/duplicate manifest member")
        exact(root / rel, digest)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift: " + str(root))


def verify_file_seal(path):
    path = Path(path)
    digest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact(path, sha(path))
    if (not digest.is_file() or digest.is_symlink()
            or digest.read_text() != sha(path) + "  " + path.name + "\n"
            or not outer.is_file() or outer.is_symlink()
            or outer.read_text() != sha(digest) + "  " + digest.name + "\n"):
        raise Failure("file double seal drift: " + str(path))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_dir(root):
    root = Path(root)
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n" for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def verify_contract_sources(contract):
    if (contract.get("schema") != "m1722_m1701_c1_salvage_formality_pt_source_contract_r1_v1"
            or contract.get("status") != "SOURCE_ONLY__M1723_REVIEW_AND_M1724_RELEASE_REQUIRED__NO_EDA"
            or contract.get("claim_boundary") != CLAIMS):
        raise Failure("source contract semantic drift")
    rows = contract.get("source_files")
    if not isinstance(rows, list):
        raise Failure("source inventory absent")
    mapping = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or row["path"] in mapping:
            raise Failure("source inventory malformed")
        mapping[row["path"]] = row["sha256"]
        exact(HW / row["path"], row["sha256"])
    expected = {path.relative_to(HW).as_posix() for path in (RUNNER, FM_TCL, PT_TCL, TEST)}
    if set(mapping) != expected:
        raise Failure("source inventory incomplete")


def verify_authority():
    names = ("M1722_EXPECTED_RUNNER_SHA256", "M1722_EXPECTED_SOURCE_CONTRACT_SHA256",
             "M1722_EXPECTED_M1723_REVIEW_SHA256", "M1722_EXPECTED_M1723_MANIFEST_SHA256",
             "M1722_EXPECTED_M1723_OUTER_FILE_SHA256", "M1722_EXPECTED_M1724_RELEASE_SHA256")
    pins = dict((name, os.environ.get(name, "")) for name in names)
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values()):
        raise Failure("M1723/M1724 exact SHA authority absent")
    exact(RUNNER, pins["M1722_EXPECTED_RUNNER_SHA256"])
    exact(CONTRACT, pins["M1722_EXPECTED_SOURCE_CONTRACT_SHA256"])
    contract = strict_json(CONTRACT)
    verify_contract_sources(contract)
    verify_seal(M1723, pins["M1722_EXPECTED_M1723_MANIFEST_SHA256"],
                pins["M1722_EXPECTED_M1723_OUTER_FILE_SHA256"])
    exact(M1723 / "review.json", pins["M1722_EXPECTED_M1723_REVIEW_SHA256"])
    review = strict_json(M1723 / "review.json")
    verify_file_seal(M1724)
    exact(M1724, pins["M1722_EXPECTED_M1724_RELEASE_SHA256"])
    release = strict_json(M1724)
    if review.get("status") != "PASS_M1723_M1722_M1701_C1_SALVAGE_FORMALITY_PT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT":
        raise Failure("M1723 status drift")
    expected_auth = {"future_m1722_attempts": 1, "automatic_retry": False,
                     "formality_runs": 1, "pt_runs": 1, "dc_runs": 0}
    if (release.get("status") != "AUTHORIZE_ONE_M1722_M1701_C1_SALVAGE_FORMALITY_PT_ATTEMPT"
            or release.get("authorization") != expected_auth
            or release.get("identity") != {
                "runner_sha256": sha(RUNNER),
                "source_contract_sha256": sha(CONTRACT),
                "m1723_review_sha256": sha(M1723 / "review.json")}
            or release.get("claim_boundary") != CLAIMS):
        raise Failure("M1724 authority semantic drift")


def verify_inputs():
    verify_seal(M1701, FIXED_SHA["m1701_manifest"], FIXED_SHA["m1701_outer"])
    verify_seal(M1665, FIXED_SHA["m1665_manifest"], FIXED_SHA["m1665_outer"])
    verify_seal(M1714, FIXED_SHA["m1714_manifest"], FIXED_SHA["m1714_outer"])
    exact(M1714 / "review.json", FIXED_SHA["m1714_review"])
    if strict_json(M1714 / "review.json").get("status") != "PASS_SALVAGE_CANDIDATE_ONLY":
        raise Failure("M1714 salvage admission drift")
    for path, key in ((M1701_NETLIST, "m1701_netlist"), (M1701_SDC, "m1701_sdc"),
                      (M1701_DDC, "m1701_ddc"), (M1701_SVF, "m1701_svf"),
                      (M1665_NETLIST, "m1665_netlist"), (M1665_SDC, "m1665_sdc"),
                      (M1665_DDC, "m1665_ddc"), (M1665_SVF, "m1665_svf"),
                      (FM, "fm"), (PT, "pt"), (LMUTIL, "lmutil"),
                      (LICENSE_FILE, "license"), (STD_SLOW, "std_slow"),
                      (STD_FAST, "std_fast"), (MACRO_SLOW, "macro_slow"),
                      (MACRO_FAST, "macro_fast"),
                      (MACRO_ROOT / "SHA256SUMS", "macro_manifest"),
                      (HW / "docs/359_DATE终局冻结_20260813.md", "docs359")):
        exact(path, FIXED_SHA[key])
    sdc = M1701_SDC.read_text()
    if (len(re.findall(r"^\s*create_clock .* -period 3(?:\.0+)?(?:\s|$)", sdc, re.M)) != 1
            or len(re.findall(r"^\s*set_clock_uncertainty -setup 0\.2(?:\s|$)", sdc, re.M)) != 1
            or len(re.findall(r"^\s*set_clock_uncertainty -hold 0\.05(?:\s|$)", sdc, re.M)) != 1
            or re.search(r"^\s*(set_false_path|set_multicycle_path|set_min_delay|set_max_delay|set_disable_timing|set_case_analysis)(?:\s|$)", sdc, re.M)):
        raise Failure("M1701 3ns/0.2/0.05 constraint identity")
    area = (M1701 / "reports/area_posthold.rpt").read_text()
    match = re.search(r"Total cell area:\s*([0-9.]+)", area)
    if not match or not math.isfinite(float(match.group(1))) or float(match.group(1)) > AREA_CEILING:
        raise Failure("M1701 area ceiling")
    if M1701_NETLIST.read_text(errors="replace").count("TS1N28HPCPHVTB128X128M4S") != 9:
        raise Failure("M1701 mapped macro population")
    return float(match.group(1))


def namespaces_fresh():
    for path in (RESULT, ATTEMPT, FAILURE, WORK, STAGE):
        if os.path.lexists(path):
            raise Failure("namespace residue: " + str(path))
    for pattern in (".m1722_m1701_c1_salvage_formality_pt_work.*",
                    ".m1722_m1701_c1_salvage_formality_pt_stage.*"):
        if next((HW / "dc_handoff/runs").glob(pattern), None) is not None:
            raise Failure("stale private namespace: " + pattern)


def parent_pid(pid):
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        return int(text[text.rfind(")") + 1:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def owned_or_ancestor(pid, runner_pid=None):
    runner_pid = os.getpid() if runner_pid is None else runner_pid
    ancestry, cursor = set(), runner_pid
    while cursor > 1 and cursor not in ancestry:
        ancestry.add(cursor)
        cursor = parent_pid(cursor)
        if cursor is None:
            break
    if pid in ancestry:
        return True
    seen, cursor = set(), pid
    while cursor > 1 and cursor not in seen:
        if cursor == runner_pid:
            return True
        seen.add(cursor)
        cursor = parent_pid(cursor)
        if cursor is None:
            break
    return False


def collision_gate():
    blocked = {"vcs", "vcs1", "vlogan", "simv", "dc_shell", "dc_shell-t",
               "pt_shell", "fm_shell", "fm_shell_exec", "icc2_shell",
               "common_shell_exec", "common_shell_exe"}
    hits = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or owned_or_ancestor(int(item.name)):
            continue
        try:
            if item.stat().st_uid != os.getuid():
                continue
            comm = (item / "comm").read_text().strip()
            argv = [Path(part.decode(errors="replace")).name
                    for part in (item / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if comm in blocked or blocked.intersection(argv):
            hits.append((item.name, comm, argv[:4]))
    if hits:
        raise Failure("same-UID EDA collision: " + repr(hits))


def resource_gate():
    values = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        fields = row.replace(":", "").split()
        if fields and fields[0] in {"MemAvailable", "CommitLimit", "Committed_AS"}:
            values[fields[0]] = int(fields[1])
    if values.get("MemAvailable", 0) < 16 * 1024 * 1024:
        raise Failure("MemAvailable below 16 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 24 * 1024 * 1024:
        raise Failure("commit headroom below 24 GiB")
    if shutil.disk_usage(HW / "dc_handoff/runs").free < 4 * 1024 * 1024 * 1024:
        raise Failure("run disk free below 4 GiB")


def clean_env(extra):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SNPSLMD_LICENSE_FILE": LICENSE_SERVER, "LM_LICENSE_FILE": str(LICENSE_FILE)}
    value.update(extra)
    return value


def scan_tool_log(path):
    allowed = 0
    fatal = re.compile(
        r"(?:error|fatal)\s*:|LINK-[0-9]+|\bunresolved\b|"
        r"\bloop\b|\bunable\s+to\s+resolve\b|\((?:TIM-209|OPT-150)\)",
        re.I)
    for number, line in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        # splitlines() removes only the line terminator.  No other prefix,
        # suffix, or whitespace normalization is permitted for the sole
        # environment-owned diagnostic exception.
        if line == GUI_ALLOW:
            allowed += 1
            continue
        if fatal.search(line):
            raise Failure("non-allowlisted fatal log line %d: %s" % (number, line))
    if allowed > 1:
        raise Failure("GUI allowlist cardinality")
    return allowed


def run_tool(command, cwd, env, timeout, output):
    collision_gate()
    with Path(output).open("wb") as stream:
        completed = subprocess.run(command, cwd=str(cwd), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise Failure("tool failure: " + str(command[0]))
    return scan_tool_log(output)


def read_machine(path):
    value = {}
    for row in Path(path).read_text().splitlines():
        if "=" in row:
            key, item = row.split("=", 1)
            if key in value:
                raise Failure("duplicate machine key")
            value[key] = item
    required = {"setup_wns_ns", "setup_tns_ns", "setup_violating_paths",
                "hold_wns_ns", "hold_tns_ns", "hold_violating_paths",
                "macro_count", "clock_period_ns", "setup_uncertainty_ns",
                "hold_uncertainty_ns"}
    if set(value) != required:
        raise Failure("machine summary keyset")
    numeric = dict((key, float(value[key])) for key in required
                   if key not in {"setup_violating_paths", "hold_violating_paths", "macro_count"})
    if (numeric["setup_wns_ns"] < 0 or numeric["hold_wns_ns"] < 0
            or numeric["setup_tns_ns"] != 0 or numeric["hold_tns_ns"] != 0
            or int(value["setup_violating_paths"]) != 0
            or int(value["hold_violating_paths"]) != 0
            or int(value["macro_count"]) != 9
            or numeric["clock_period_ns"] != 3.0
            or numeric["setup_uncertainty_ns"] != 0.2
            or numeric["hold_uncertainty_ns"] != 0.05):
        raise Failure("independent PT gate")
    return value


def main():
    if len(sys.argv) != 1:
        raise Failure("M1722 accepts no arguments")
    state = {"phase": "SOURCE_CHAIN", "attempt": False,
             "formality_runs": 0, "pt_runs": 0, "complete": False}
    lock_handle = LOCK.open("a+")
    try:
        verify_authority()
        area = verify_inputs()
        namespaces_fresh()
        collision_gate()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        collision_gate()
        resource_gate()
        namespaces_fresh()
        state["phase"] = "LICENSE_PREFLIGHT"
        for feature in ("Formality", "PrimeTime"):
            completed = subprocess.run([str(LMUTIL), "lmstat", "-c", LICENSE_SERVER, "-f", feature],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=60, check=False, env=clean_env({}))
            if completed.returncode != 0:
                raise Failure(feature + " license preflight failed")
            match = re.search(
                rb"Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use",
                completed.stdout, re.S)
            if not match or int(match.group(1)) <= int(match.group(2)):
                raise Failure(feature + " license unavailable")
        state["phase"] = "ATTEMPT_CONSUME"
        ATTEMPT.mkdir()
        state["attempt"] = True
        write_json(ATTEMPT / "attempt.json", {
            "status": "M1722_C1_SALVAGE_FORMALITY_PT_ATTEMPT_CONSUMED",
            "formality_runs": 1, "pt_runs": 1, "dc_runs": 0,
            "automatic_retry": False})
        seal_dir(ATTEMPT)
        (WORK / "formality/reports").mkdir(parents=True)
        (WORK / "formality/work").mkdir()
        (WORK / "ptsta/reports").mkdir(parents=True)
        (WORK / "ptsta/work").mkdir()

        common = {"M1722_STD_SLOW_DB": str(STD_SLOW),
                  "M1722_STD_FAST_DB": str(STD_FAST),
                  "M1722_MACRO_SLOW_DB": str(MACRO_SLOW),
                  "M1722_MACRO_FAST_DB": str(MACRO_FAST),
                  "M1722_M1665_REFERENCE_NETLIST": str(M1665_NETLIST),
                  "M1722_M1701_IMPLEMENTATION_NETLIST": str(M1701_NETLIST),
                  "M1722_M1701_IMPLEMENTATION_SDC": str(M1701_SDC)}
        state["phase"] = "FORMALITY"
        state["formality_runs"] += 1
        fm_dir = WORK / "formality"
        fm_gui = run_tool([str(FM), "-f", str(FM_TCL)], fm_dir / "work",
                          clean_env(dict(common, M1722_FM_OUTPUT_DIR=str(fm_dir))),
                          10800, fm_dir / "formality.raw.log")
        if ((fm_dir / "FORMALITY_INTERNAL_COMPLETE.txt").read_text().splitlines()[0]
                != "M1722_C1_M1665_TO_M1701_GATE_FORMALITY_INTERNAL_COMPLETE=PASS"):
            raise Failure("Formality marker")
        reports = fm_dir / "reports"
        status = (reports / "formality_status.rpt").read_text(errors="replace")
        if ("Verification SUCCEEDED" not in status
                or re.search(r"[1-9][0-9]* Passing compare points", status) is None
                or "No unmatched points" not in (reports / "formality_unmatched.rpt").read_text(errors="replace")
                or "No failing compare points" not in (reports / "formality_failing.rpt").read_text(errors="replace")
                or "No aborted compare points" not in (reports / "formality_aborted.rpt").read_text(errors="replace")
                or "No unverified compare points" not in (reports / "formality_unverified.rpt").read_text(errors="replace")):
            raise Failure("Formality proof gate")

        state["phase"] = "PRIMETIME"
        state["pt_runs"] += 1
        pt_dir = WORK / "ptsta"
        pt_gui = run_tool([str(PT), "-f", str(PT_TCL)], pt_dir / "work",
                          clean_env(dict(common, M1722_PT_OUTPUT_DIR=str(pt_dir))),
                          7200, pt_dir / "pt.raw.log")
        if ((pt_dir / "PTSTA_INTERNAL_COMPLETE.txt").read_text().splitlines()[0]
                != "M1722_C1_M1701_PRELAYOUT_PTSTA_INTERNAL_COMPLETE=PASS"):
            raise Failure("PT marker")
        machine = read_machine(pt_dir / "reports/timing_summary_machine.txt")
        for report in ("check_timing.rpt", "analysis_coverage.rpt", "global_timing.rpt",
                       "timing_setup_slow.rpt", "timing_hold_fast.rpt",
                       "constraint_violators.rpt", "clock.rpt", "exceptions.rpt",
                       "design.rpt", "wire_load.rpt", "libraries.rpt", "runtime_scope.rpt"):
            if not (pt_dir / "reports" / report).is_file():
                raise Failure("PT report absent: " + report)
        if ("slack (VIOLATED)" in (pt_dir / "reports/timing_setup_slow.rpt").read_text()
                or "slack (VIOLATED)" in (pt_dir / "reports/timing_hold_fast.rpt").read_text()
                or "slack (VIOLATED)" in (pt_dir / "reports/constraint_violators.rpt").read_text()):
            raise Failure("PT violation report")
        if state["formality_runs"] != 1 or state["pt_runs"] != 1:
            raise Failure("tool budget drift")

        shutil.rmtree(WORK / "formality/work")
        shutil.rmtree(WORK / "ptsta/work")
        STAGE.mkdir()
        shutil.copytree(WORK / "formality", STAGE / "formality")
        shutil.copytree(WORK / "ptsta", STAGE / "ptsta")
        write_json(STAGE / "receipt.json", {
            "schema": "m1722_m1701_c1_salvage_formality_pt_candidate_receipt_r1_v1",
            "status": "PASS_CANDIDATE_PENDING_INDEPENDENT_RESULT_HAMMER",
            "tool_runs": {"formality": 1, "pt": 1, "dc": 0},
            "formality": {"m1665_reference_to_m1701_implementation": True,
                           "failing": 0, "aborted": 0, "unverified": 0,
                           "unmatched": 0},
            "prime_time": machine,
            "physical": {"dc_cell_area_um2": area,
                         "area_ceiling_um2": AREA_CEILING, "macro_count": 9},
            "log_allowlist": {"exact_gui_line": GUI_ALLOW,
                              "formality_occurrences": fm_gui,
                              "pt_occurrences": pt_gui,
                              "all_other_fatal_rejected": True},
            "m1701_quarantine_modified_or_promoted": False,
            "automatic_retry": False,
            "claim_boundary": CLAIMS})
        (STAGE / "RUN_COMPLETE.txt").write_text(
            "PASS_M1722_C1_SALVAGE_FORMALITY_PT_CANDIDATE_PENDING_RESULT_HAMMER\n")
        seal_dir(STAGE)
        publish_no_replace(STAGE, RESULT)
        shutil.rmtree(WORK)
        state["complete"] = True
        print("PASS_M1722_C1_SALVAGE_FORMALITY_PT_CANDIDATE_PENDING_RESULT_HAMMER")
        return 0
    except BaseException as error:
        if not state["complete"] and WORK.is_dir():
            try:
                for rel in ("formality/work", "ptsta/work"):
                    path = WORK / rel
                    if path.is_dir():
                        shutil.rmtree(path)
                write_json(WORK / "failure.json", {
                    "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
                    "phase": state["phase"], "error": type(error).__name__,
                    "attempt_consumed": state["attempt"],
                    "formality_runs": state["formality_runs"],
                    "pt_runs": state["pt_runs"], "automatic_retry": False})
                seal_dir(WORK)
                publish_no_replace(WORK, FAILURE)
            except BaseException:
                pass
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
