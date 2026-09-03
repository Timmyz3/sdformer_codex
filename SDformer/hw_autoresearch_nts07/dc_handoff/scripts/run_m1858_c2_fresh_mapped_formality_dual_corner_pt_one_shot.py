#!/usr/bin/env python3
"""Fail-closed M1858 two-axis fresh-mapped Formality and PrimeTime runner.

M1858 source authoring executes nothing.  A future, exactly pinned M1859
different-author source review and double-sealed M1860 release may authorize
one attempt.  That attempt runs exactly one Formality and one independent
dual-corner PrimeTime process for each of K8 and matched K1x8.  Negative setup
or hold slack is reported faithfully and never repaired, excepted, or hidden.
"""
from __future__ import print_function

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile


HW = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1858_m1857_m1850_failure_c2_fresh_mapped_formality_dual_corner_pt_source_contract_r1_20260902.json"
AUTHOR_DIR = HW / "reviews/m1858_m1857_m1850_failure_c2_fresh_mapped_formality_dual_corner_pt_source_author_receipt_r1_20260902"
AUTHOR_RECEIPT = AUTHOR_DIR / "author_receipt.json"
M1857_DIR = HW / "reviews/m1857_m1850_c2_formality_pt_failure_hammer_r1_20260902"
M1857_REVIEW = M1857_DIR / "review.json"
M1857_REVIEW_SHA = "90f68f526c17052a65433adcf5a3f79d91a938f8c290b1b239e5a117ce062a26"
M1857_MANIFEST_SHA = "f4b022970901ce9c1fd55f21157fcab091c5a31178bea865fd47ae6e3b8000ce"
M1857_OUTER_SHA = "7a7827bbccf416804d9dcf6b9e450f3674bc000ac793db2f013c2a2176eedcbd"
M1859_DIR = HW / "reviews/m1859_m1858_c2_fresh_mapped_formality_dual_corner_pt_source_hammer_r1_20260902"
M1859_REVIEW = M1859_DIR / "review.json"
M1860_RELEASE = HW / "contracts/m1860_m1859_m1858_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_20260902.json"

M1811 = HW / "dc_handoff/runs/m1811_m1810_m1809_c2_registered_fault_matched_two_axis_dc_r1_20260902"
M1830 = HW / "reviews/m1830_m1811_c2_registered_fault_matched_two_axis_dc_result_hammer_r1_20260902"
M1811_MANIFEST_SHA = "695050260d54ca9b9d6f7b74d03021dd59afd642168981a13df0438e9fe12066"
M1811_OUTER_SHA = "04aa6bea4a06a8be3c441ddb984c68a046810a137fd2eca096adf513af0d324b"
M1830_REVIEW_SHA = "79e1885fad8ddac4ec0a6eee4d9034657761e778da384093fae5ab937f98f99b"
M1830_MANIFEST_SHA = "d0ef8172f33378e9b025aab18043da19335fd9f00d1cd8d240bfb620997c0d06"
M1830_OUTER_SHA = "0b9dc1915096db8df6702e3ab5027d267fb99a3178bc2288a8b5625e611e343d"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REFERENCE_TOP = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
REFERENCE_FILELIST = HW / "dc_handoff/filelists/iscas_m1809_c2_registered_fault_matched_k8_k1x8_logic_only_dc.f"
REFERENCE_FILELIST_SHA = "1dc9703bafb12ed35dda1dc9b7248881145d600c06129b00b34b7308eaeaf661"
M1811_INPUT_FILELIST = M1811 / "input_filelist.f"
FM_TCL = HW / "dc_handoff/scripts/run_formality_m1858_m1809_c2_fresh_mapped_two_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptsta_m1858_m1809_c2_fresh_mapped_dual_corner.tcl"

DESIGN = "m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24"
AXIS_ORDER = ("K8", "K1X8")
AXES = {
    "K8": {
        "arch_mode": 0,
        "elab_parameters": "ARCH_MODE=0",
        "implementation_top": DESIGN + "_ARCH_MODE0",
        "mapped_v": M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v",
        "mapped_v_sha": "63605469818c36574ce9719130877610e79cf0c3b7317c0e69848539afa6b792",
        "mapped_sdc": M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.sdc",
        "mapped_sdc_sha": "af2fbde96a5046053aed137facc4fd2741b3f517eb678710c81eef9f7ed49018",
        "svf": M1811 / "k8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24.svf",
        "svf_sha": "b5fe89b8c44e6edd9aa4e1a06e9d13234148f2dbd2b7b00cb8014bd838b65543",
    },
    "K1X8": {
        "arch_mode": 1,
        "elab_parameters": "ARCH_MODE=1",
        "implementation_top": DESIGN + "_ARCH_MODE1",
        "mapped_v": M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.v",
        "mapped_v_sha": "8698d227f3408b6e40c03bfe9282de458b0ba5cba4e22ec5f0c9bfd4ff16fc1b",
        "mapped_sdc": M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24_mapped.sdc",
        "mapped_sdc_sha": "1631f7d0cc3d0257439dea5f9ed2a2fc004556dc0f8f5657152a7d3f5f3e6c0a",
        "svf": M1811 / "k1x8/netlist/m1809_c2_registered_fault_matched_k8_k1x8_raw4_acc24.svf",
        "svf_sha": "bcb2f9f974be2ee8d4927d41d99b4e06abac77635e8449e61d61163c6b05d2dc",
    },
}
EXPECTED_FMR_ELAB_147_SITES = {
    ("queue_bitmap_q", "/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor/queue_transition",
     "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv", 197),
    ("queue_beat_index_q", "/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor/queue_transition",
     "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv", 199),
    ("queue_window_last_q", "/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor/queue_transition",
     "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv", 201),
    ("fm_N390", "/m218_fc2_tagged_slice_service_island/protocol_analysis",
     "rtl_m218/m218_fc2_tagged_slice_service_island.sv", 266),
    ("fm_N3940", "/m218_fc2_tagged_slice_service_island/service_state",
     "rtl_m218/m218_fc2_tagged_slice_service_island.sv", 632),
    ("fm_N394e", "/m218_fc2_tagged_slice_service_island/service_state",
     "rtl_m218/m218_fc2_tagged_slice_service_island.sv", 633),
    ("fm_N296c5", "/m218_fc2_tagged_slice_service_island",
     "rtl_m218/m218_fc2_tagged_slice_service_island.sv", 338),
    ("fm_N296c8", "/m218_fc2_tagged_slice_service_island",
     "rtl_m218/m218_fc2_tagged_slice_service_island.sv", 339),
}

FM_SHELL = Path("/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell")
PT_SHELL = Path("/opt/synopsys/prime/W-2024.09-SP3/bin/pt_shell")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
STD_SLOW = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db")
STD_FAST = Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db")
TOOL_SHA = {
    FM_SHELL: "aceb24fb490927bf292dba8ce6a783fbad1dd648bb7e41710fc750b2dafed53b",
    PT_SHELL: "afdcfa7071f86d229ed3b4481f67adefd1c51bb0288b7e7d370213a43b70c9ef",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    LICENSE_FILE: "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490",
    STD_SLOW: "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    STD_FAST: "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
}

RESULT = HW / "dc_handoff/runs/m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_r1_20260902"
ATTEMPT = HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_attempt_consumed"
WORK = HW / ("dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_work." + str(os.getpid()))
LAUNCH_LOCK = HW / "dc_handoff/runs/.m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_launch_lock"
SHARED_QUEUE = Path("/tmp/date_dual_synopsys_same_uid_eda_queue.lock")


class M1858Error(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_regular(path, expected):
    path = Path(path)
    if (re.fullmatch(r"[0-9a-f]{64}", expected or "") is None
            or not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.lstat().st_mode)
            or sha256(path) != expected):
        raise M1858Error("exact regular identity mismatch: " + str(path))


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise M1858Error("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          M1858Error("nonfinite JSON token: " + token)))


def verify_sealed_directory(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise M1858Error("sealed directory absent/invalid: " + str(root))
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    exact_regular(manifest, manifest_sha)
    exact_regular(outer, outer_sha)
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise M1858Error("outer seal semantic mismatch: " + str(root))
    listed = set()
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise M1858Error("manifest syntax: " + str(root))
        name = fields[1].lstrip("*")
        rel = Path(name)
        if name in listed or rel.is_absolute() or ".." in rel.parts:
            raise M1858Error("manifest path unsafe/duplicate: " + name)
        exact_regular(root / rel, fields[0])
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise M1858Error("symlink in sealed directory: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if listed != actual:
        raise M1858Error("sealed directory population drift: " + str(root))


def verify_file_double_seal(path, expected_sha):
    path = Path(path)
    exact_regular(path, expected_sha)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    exact_regular(sidecar, sha256(sidecar))
    exact_regular(outer, sha256(outer))
    if sidecar.read_text() != expected_sha + "  " + path.name + "\n":
        raise M1858Error("file sidecar mismatch: " + str(path))
    expected_outer = sha256(sidecar) + "  " + sidecar.name + "\n"
    if outer.read_text() != expected_outer:
        raise M1858Error("file outer seal mismatch: " + str(path))


def seal_directory(root):
    root = Path(root)
    members = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise M1858Error("refuse to seal symlink: " + str(path))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            members.append(path)
    members.sort(key=lambda p: p.relative_to(root).as_posix())
    manifest_text = "".join(
        sha256(path) + "  " + path.relative_to(root).as_posix() + "\n"
        for path in members)
    (root / "SHA256SUMS").write_text(manifest_text)
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(root / "SHA256SUMS") + "  SHA256SUMS\n")
    verify_sealed_directory(
        root, sha256(root / "SHA256SUMS"),
        sha256(root / "SHA256SUMS.seal.sha256"))


def atomic_publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise M1858Error("renameat2 unavailable; fail closed")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    rc = function(-100, os.fsencode(str(source)), -100,
                  os.fsencode(str(destination)), 1)
    if rc != 0:
        error = ctypes.get_errno()
        raise M1858Error("atomic no-replace publish failed: " + os.strerror(error))


def require_env(name):
    value = os.environ.get(name, "")
    if not value:
        raise M1858Error("caller exact pin absent: " + name)
    return value


def verify_live_rtl_identity(review):
    """Bind fresh reference elaboration to M1830's exact live RTL identity."""
    source_identity = review.get("source_identity")
    if type(source_identity) is not dict:
        raise M1858Error("M1830 source_identity absent/nonobject")
    sources = source_identity.get("sources")
    if (type(sources) is not dict or len(sources) != 13
            or source_identity.get("filelist_rows") != 13
            or source_identity.get("unique_filelist_rows") != 13
            or source_identity.get("all_current_source_shas_match_runner_pins") is not True):
        raise M1858Error("M1830 live RTL identity cardinality/authority drift")
    rows = [line.strip() for line in REFERENCE_FILELIST.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if rows != list(sources.keys()) or len(rows) != 13 or len(set(rows)) != 13:
        raise M1858Error("live RTL filelist set/order differs from M1830 source identity")
    exact_regular(M1811_INPUT_FILELIST, REFERENCE_FILELIST_SHA)
    if M1811_INPUT_FILELIST.read_bytes() != REFERENCE_FILELIST.read_bytes():
        raise M1858Error("canonical M1811 input_filelist.f differs byte-for-byte")
    for rel in rows:
        if (Path(rel).is_absolute() or ".." in Path(rel).parts
                or re.fullmatch(r"[0-9a-f]{64}", sources.get(rel, "")) is None):
            raise M1858Error("unsafe or malformed live RTL identity: " + rel)
        exact_regular(HW / rel, sources[rel])
    return {rel: sources[rel] for rel in rows}


def verify_authority():
    exact_regular(DOCS359, DOCS359_SHA)
    exact_regular(REFERENCE_FILELIST, REFERENCE_FILELIST_SHA)
    rows = [line.strip() for line in REFERENCE_FILELIST.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 13 or len(set(rows)) != 13:
        raise M1858Error("reference filelist must remain 13 unique rows")

    verify_sealed_directory(M1811, M1811_MANIFEST_SHA, M1811_OUTER_SHA)
    verify_sealed_directory(M1830, M1830_MANIFEST_SHA, M1830_OUTER_SHA)
    exact_regular(M1830 / "review.json", M1830_REVIEW_SHA)
    review = strict_json(M1830 / "review.json")
    if (not str(review.get("status", "")).startswith("PASS_M1830")
            or review.get("p0_count") != 0 or review.get("p1_count") != 0
            or review.get("p2_count") != 0):
        raise M1858Error("M1830 authority not severity-zero PASS")
    live_rtl_identity = verify_live_rtl_identity(review)

    for axis in AXIS_ORDER:
        row = AXES[axis]
        exact_regular(row["mapped_v"], row["mapped_v_sha"])
        exact_regular(row["mapped_sdc"], row["mapped_sdc_sha"])
        exact_regular(row["svf"], row["svf_sha"])
    paths = [AXES[a][k].resolve() for a in AXIS_ORDER
             for k in ("mapped_v", "mapped_sdc", "svf")]
    if len(set(paths)) != 6:
        raise M1858Error("axis artifacts shared or crossed")

    for path, digest in TOOL_SHA.items():
        exact_regular(path, digest)
    runner_sha = require_env("M1858_EXPECTED_RUNNER_SHA256")
    contract_sha = require_env("M1858_EXPECTED_SOURCE_CONTRACT_SHA256")
    exact_regular(RUNNER, runner_sha)
    exact_regular(CONTRACT, contract_sha)
    verify_file_double_seal(CONTRACT, contract_sha)
    verify_sealed_directory(AUTHOR_DIR, sha256(AUTHOR_DIR / "SHA256SUMS"),
                            sha256(AUTHOR_DIR / "SHA256SUMS.seal.sha256"))

    verify_sealed_directory(M1857_DIR, M1857_MANIFEST_SHA, M1857_OUTER_SHA)
    exact_regular(M1857_REVIEW, M1857_REVIEW_SHA)
    failure_review = strict_json(M1857_REVIEW)
    if (failure_review.get("status") !=
            "PASS_M1857_INDEPENDENT_FAILURE_AUDIT__M1850_FORMALITY_PT_FAIL_CLOSED__P0_0_P1_1_P2_0__NO_RETRY__NO_EQUIVALENCE_OR_PT"
            or failure_review.get("audit_status") != "PASS"
            or failure_review.get("production_admission") != "FAIL_CLOSED"
            or failure_review.get("severity_counts") != {
                "p0": 0, "p1": 1, "p2": 0}):
        raise M1858Error("M1857 M1850-failure audit semantic drift")

    review_sha = require_env("M1858_EXPECTED_M1859_SOURCE_REVIEW_SHA256")
    review_manifest = require_env("M1858_EXPECTED_M1859_SOURCE_REVIEW_MANIFEST_SHA256")
    review_outer = require_env("M1858_EXPECTED_M1859_SOURCE_REVIEW_OUTER_SHA256")
    verify_sealed_directory(M1859_DIR, review_manifest, review_outer)
    exact_regular(M1859_REVIEW, review_sha)
    hammer = strict_json(M1859_REVIEW)
    expected_hammer_status = (
        "PASS_M1859_M1858_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT")
    if (hammer.get("status") != expected_hammer_status
            or hammer.get("p0_count") != 0 or hammer.get("p1_count") != 0
            or hammer.get("p2_count") != 0
            or hammer.get("authorization") != {
                "future_m1858_attempts": 1,
                "formality_runs": 2,
                "pt_runs": 2,
                "all_other_eda_runs": 0,
                "automatic_retry": False,
            }):
        raise M1858Error("M1859 source-review semantic drift")

    release_sha = require_env("M1858_EXPECTED_M1860_LAUNCH_RELEASE_SHA256")
    verify_file_double_seal(M1860_RELEASE, release_sha)
    release = strict_json(M1860_RELEASE)
    expected_identity = {
        "runner_sha256": runner_sha,
        "source_contract_sha256": contract_sha,
        "author_receipt_sha256": sha256(AUTHOR_RECEIPT),
        "author_manifest_sha256": sha256(AUTHOR_DIR / "SHA256SUMS"),
        "author_outer_seal_file_sha256": sha256(AUTHOR_DIR / "SHA256SUMS.seal.sha256"),
        "m1857_failure_review_sha256": M1857_REVIEW_SHA,
        "m1857_failure_review_manifest_sha256": M1857_MANIFEST_SHA,
        "m1857_failure_review_outer_seal_file_sha256": M1857_OUTER_SHA,
        "m1859_source_review_sha256": review_sha,
        "m1859_source_review_manifest_sha256": review_manifest,
        "m1859_source_review_outer_seal_file_sha256": review_outer,
        "m1811_manifest_sha256": M1811_MANIFEST_SHA,
        "m1811_outer_seal_file_sha256": M1811_OUTER_SHA,
        "m1830_review_sha256": M1830_REVIEW_SHA,
        "m1830_manifest_sha256": M1830_MANIFEST_SHA,
        "m1830_outer_seal_file_sha256": M1830_OUTER_SHA,
    }
    expected_budget = {
        "max_attempts": 1,
        "formality_runs": 2,
        "pt_runs": 2,
        "dc_runs": 0,
        "vcs_runs": 0,
        "ptpx_runs": 0,
        "automatic_retry": False,
    }
    if (release.get("schema") != "m1860_m1859_m1858_c2_fresh_mapped_formality_dual_corner_pt_launch_release_r1_v1"
            or release.get("status") != "AUTHORIZE_ONE_M1858_C2_FRESH_MAPPED_FORMALITY_DUAL_CORNER_PT_ATTEMPT"
            or release.get("identity") != expected_identity
            or release.get("authorization") != expected_budget):
        raise M1858Error("M1860 release semantic drift")
    return release_sha, live_rtl_identity


def same_uid_eda():
    names = {"dc_shell", "dc_shell-t", "fm_shell", "fm_shell_exec",
             "pt_shell", "vcs", "vcs1", "vlogan", "simv",
             "common_shell_exec", "common_shell_exe"}
    hits = []
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            state = (entry / "stat").read_text().split(")", 1)[1].split()[0]
            comm = (entry / "comm").read_text().strip()
            args = {Path(x.decode(errors="replace")).name
                    for x in (entry / "cmdline").read_bytes().split(b"\0") if x}
        except (OSError, ValueError):
            continue
        if state != "Z" and (comm in names or names.intersection(args)):
            hits.append(entry.name + ":" + comm)
    return sorted(hits)


def resource_gate():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0].rstrip(":")] = int(fields[1])
    if values.get("MemAvailable", 0) < 16777216:
        raise M1858Error("MemAvailable below 16 GiB")
    if values.get("CommitLimit", 0) - values.get("Committed_AS", 0) < 50331648:
        raise M1858Error("commit headroom below 48 GiB")
    if shutil.disk_usage(str(HW)).free < 4 * 1024 ** 3:
        raise M1858Error("disk headroom below 4 GiB")


def license_gate():
    for feature in ("Formality", "PrimeTime"):
        result = subprocess.run(
            [str(LMUTIL), "lmstat", "-c", "27030@ic.ismd-nemo", "-f", feature],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, check=False)
        if result.returncode != 0:
            raise M1858Error(feature + " license query failed")
        match = re.search(
            r"Total of\s+(\d+)\s+licenses? issued;\s+Total of\s+(\d+)\s+licenses? in use",
            result.stdout, re.S)
        if not match or int(match.group(1)) <= int(match.group(2)):
            raise M1858Error("no available " + feature + " license")


def tool_environment(axis, output_dir, tool_home):
    row = AXES[axis]
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "HOME": str(tool_home),
        "TMPDIR": str(tool_home),
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo",
        "LM_LICENSE_FILE": str(LICENSE_FILE),
        "M1858_AXIS": axis,
        "M1858_HW_ROOT": str(HW),
        "M1858_REFERENCE_FILELIST": str(REFERENCE_FILELIST),
        "M1858_REFERENCE_TOP": REFERENCE_TOP,
        "M1858_REF_ELAB_PARAMETERS": row["elab_parameters"],
        "M1858_STD_SLOW_DB": str(STD_SLOW),
        "M1858_STD_FAST_DB": str(STD_FAST),
        "M1858_IMPLEMENTATION_NETLIST": str(row["mapped_v"]),
        "M1858_IMPLEMENTATION_SDC": str(row["mapped_sdc"]),
        "M1858_IMPLEMENTATION_SVF": str(row["svf"]),
        "M1858_IMPLEMENTATION_TOP": row["implementation_top"],
        "M1858_FORMALITY_OUTPUT_DIR": str(output_dir),
        "M1858_PT_OUTPUT_DIR": str(output_dir),
    }


def run_tool(executable, script, axis, output_dir, log_name):
    output_dir.mkdir(parents=True, exist_ok=False)
    tool_home = Path(tempfile.mkdtemp(prefix="m1858_" + axis.lower() + "_", dir="/tmp"))
    try:
        environment = tool_environment(axis, output_dir, tool_home)
        with (output_dir / log_name).open("w") as log:
            completed = subprocess.run(
                [str(executable), "-f", str(script)], cwd=str(tool_home),
                env=environment, stdout=log, stderr=subprocess.STDOUT,
                check=False)
        (output_dir / ("formality.rc" if executable == FM_SHELL else "pt.rc")).write_text(
            str(completed.returncode) + "\n")
        if completed.returncode != 0:
            raise M1858Error(axis + " " + executable.name + " exited nonzero")
        log_text = (output_dir / log_name).read_text(errors="replace")
        if re.search(r"^(?:Error|Fatal):", log_text, re.M):
            raise M1858Error(axis + " " + executable.name + " reported Error/Fatal")
    finally:
        shutil.rmtree(str(tool_home), ignore_errors=True)


def verify_formality(axis, directory):
    marker = directory / "FORMALITY_INTERNAL_COMPLETE.txt"
    reports = directory / "reports"
    log = directory / "formality.log"
    required = [marker, log, reports / "formality_status.rpt",
                reports / "formality_unmatched.rpt",
                reports / "formality_failing.rpt",
                reports / "formality_aborted.rpt",
                reports / "formality_unverified.rpt",
                reports / "formality_black_boxes.rpt"]
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size == 0
           for path in required):
        raise M1858Error(axis + " Formality report set incomplete")
    if marker.read_text().count("M1858_C2_FRESH_MAPPED_FORMALITY_INTERNAL_COMPLETE=PASS") != 1:
        raise M1858Error(axis + " Formality terminal missing/nonunique")
    marker_text = marker.read_text()
    for token in ("axis=" + axis,
                  "reference_top=" + REFERENCE_TOP,
                  "reference_elab_parameters=" + AXES[axis]["elab_parameters"],
                  "implementation_top=" + AXES[axis]["implementation_top"]):
        if marker_text.count(token) != 1:
            raise M1858Error(axis + " valid design-pair marker absent/nonunique: " + token)
    log_text = log.read_text(errors="replace")
    warning_pattern = re.compile(
        r"Signal: ([^ ]+) Block: ([^ ]+) File: "
        + re.escape(str(HW))
        + r"/([^ ]+) Line: ([0-9]+)\).*\(FMR_ELAB-147\)")
    warning_sites = {(signal, block, rel, int(line))
                     for signal, block, rel, line in warning_pattern.findall(log_text)}
    if (warning_sites != EXPECTED_FMR_ELAB_147_SITES
            or len(warning_pattern.findall(log_text)) != 8):
        raise M1858Error(axis + " FMR_ELAB-147 warning-site identity/count drift")
    status = (reports / "formality_status.rpt").read_text(errors="replace")
    for token in ("Reference      : r:/WORK/" + REFERENCE_TOP,
                  "Implementation : i:/WORK/" + AXES[axis]["implementation_top"],
                  "Verification SUCCEEDED"):
        if status.count(token) != 1:
            raise M1858Error(axis + " Formality valid design pair/proof token drift")
    if not re.search(r"[1-9][0-9]*\s+Passing compare points", status):
        raise M1858Error(axis + " Formality did not prove equivalence")
    failing_row = re.search(
        r"(?m)^Failing \(not equivalent\)\s+(?:[0-9]+\s+){7}([0-9]+)\s*$",
        status)
    if failing_row is None or int(failing_row.group(1)) != 0:
        raise M1858Error(axis + " Formality failing compare-point total nonzero/absent")
    for name, forbidden in (("formality_unmatched.rpt", "unmatched"),
                            ("formality_failing.rpt", "failing"),
                            ("formality_aborted.rpt", "aborted"),
                            ("formality_unverified.rpt", "unverified")):
        text = (reports / name).read_text(errors="replace")
        if re.search(r"[1-9][0-9]*\s+" + forbidden, text, re.I):
            raise M1858Error(axis + " nonzero " + forbidden + " points")
    black_boxes = (reports / "formality_black_boxes.rpt").read_text(
        errors="replace")
    if "Reference and implementation designs are not set" in black_boxes:
        raise M1858Error(axis + " Formality black-box report lacks valid design pair")
    if re.search(r"(?m)^\s*(?:u|e|\*)\s+\S+[\s\S]{0,180}?Instances\s*:\s*[1-9][0-9]*", black_boxes):
        raise M1858Error(axis + " unresolved/empty/unlinked black box nonzero")
    match = re.search(r"([1-9][0-9]*)\s+Passing compare points", status)
    return int(match.group(1))


def parse_machine_summary(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split("=", 1)
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    for key in ("setup_wns_ns", "hold_wns_ns"):
        if key not in values:
            raise M1858Error("PT machine summary missing " + key)
        value = float(values[key])
        if not math.isfinite(value):
            raise M1858Error("PT nonfinite " + key)
        values[key] = value
    return values


def parse_pt_semantics(reports):
    check_text = (reports / "check_timing.rpt").read_text(errors="replace")
    if check_text.count("check_timing succeeded.") != 1:
        raise M1858Error("PT check_timing success token absent/nonunique")
    warning_count = len(re.findall(r"^Warning:", check_text, re.M))
    unconstrained_count = 0
    for pattern in (
            r"There are\s+([0-9]+)\s+unconstrained endpoints",
            r"([0-9]+)\s+endpoints?\s+(?:is|are)\s+not constrained"):
        unconstrained_count += sum(int(value) for value in re.findall(
            pattern, check_text, re.I))

    coverage_text = (reports / "analysis_coverage.rpt").read_text(
        errors="replace")
    coverage = {}
    pattern = re.compile(
        r"^\s*(setup|hold|All Checks)\s+(\d+)\s+(\d+)\s+\([^\n]*?\)\s+"
        r"(\d+)\s+\([^\n]*?\)\s+(\d+)\s+\([^\n]*?\)\s*$", re.M)
    for name, total, met, violated, untested in pattern.findall(coverage_text):
        row = {"total": int(total), "met": int(met),
               "violated": int(violated), "untested": int(untested)}
        if row["total"] != row["met"] + row["violated"] + row["untested"]:
            raise M1858Error("PT analysis coverage conservation failure: " + name)
        coverage[name] = row
    if set(coverage) != {"setup", "hold", "All Checks"}:
        raise M1858Error("PT analysis coverage semantic rows missing/duplicate")

    constraint_values = {}
    for line in (reports / "constraint_semantics_machine.txt").read_text().splitlines():
        fields = line.split("=", 1)
        if len(fields) == 2:
            constraint_values[fields[0]] = fields[1]
    for key in ("setup_violating_paths", "hold_violating_paths"):
        if re.fullmatch(r"\d+", constraint_values.get(key, "")) is None:
            raise M1858Error("PT constraint semantic count absent/malformed: " + key)
        constraint_values[key] = int(constraint_values[key])
    constraint_report = (reports / "constraint_violators.rpt").read_text(
        errors="replace")
    raw_constraint_violation_marker_count = len(re.findall(
        r"\bVIOLATED\b", constraint_report, re.I))
    if ((constraint_values["setup_violating_paths"] > 0
         or constraint_values["hold_violating_paths"] > 0)
            and raw_constraint_violation_marker_count == 0):
        raise M1858Error("PT constraint report hides machine-counted violations")

    exceptions = (reports / "exceptions.rpt").read_text(errors="replace")
    exception_tokens = re.findall(
        r"\b(false_path|multicycle_path|min_delay|max_delay|disable_timing|case_analysis)\b",
        exceptions, re.I)
    design = (reports / "design.rpt").read_text(errors="replace")
    wire_load = (reports / "wire_load.rpt").read_text(errors="replace")
    runtime = (reports / "runtime_scope.rpt").read_text(errors="replace")
    for token in ("analysis_type=on_chip_variation", "parasitics=none_prelayout",
                  "timing_exceptions_added=false"):
        if runtime.count(token) != 1:
            raise M1858Error("PT runtime scope semantic drift: " + token)
    if DESIGN not in design or DESIGN not in wire_load:
        raise M1858Error("PT design/wire-load top identity absent")
    if "ZeroWireload" not in wire_load:
        raise M1858Error("PT ZeroWireload identity absent")
    return {
        "check_timing_succeeded": True,
        "check_timing_warning_count": warning_count,
        "unconstrained_endpoint_count": unconstrained_count,
        "coverage": coverage,
        "setup_constraint_violation_count": constraint_values["setup_violating_paths"],
        "hold_constraint_violation_count": constraint_values["hold_violating_paths"],
        "raw_constraint_violation_marker_count": raw_constraint_violation_marker_count,
        "forbidden_exception_token_count": len(exception_tokens),
        "constraint_clean": (constraint_values["setup_violating_paths"] == 0
                             and constraint_values["hold_violating_paths"] == 0
                             and raw_constraint_violation_marker_count == 0),
        "coverage_clean": (coverage["setup"]["violated"] == 0
                           and coverage["setup"]["untested"] == 0
                           and coverage["hold"]["violated"] == 0
                           and coverage["hold"]["untested"] == 0),
    }


def verify_pt(axis, directory):
    marker = directory / "PTSTA_INTERNAL_COMPLETE.txt"
    reports = directory / "reports"
    required = [marker, reports / "check_timing.rpt",
                reports / "analysis_coverage.rpt",
                reports / "global_timing.rpt",
                reports / "timing_setup_slow.rpt",
                reports / "timing_hold_fast.rpt",
                reports / "constraint_violators.rpt",
                reports / "constraint_semantics_machine.txt",
                reports / "clock.rpt", reports / "libraries.rpt",
                reports / "exceptions.rpt", reports / "design.rpt",
                reports / "wire_load.rpt",
                reports / "runtime_scope.rpt",
                reports / "timing_summary_machine.txt"]
    if any(not path.is_file() or path.is_symlink() or path.stat().st_size == 0
           for path in required):
        raise M1858Error(axis + " PT report set incomplete")
    if marker.read_text().count("M1858_C2_FRESH_MAPPED_DUAL_CORNER_PT_INTERNAL_COMPLETE=PASS") != 1:
        raise M1858Error(axis + " PT terminal missing/nonunique")
    summary = parse_machine_summary(reports / "timing_summary_machine.txt")
    summary.update(parse_pt_semantics(reports))
    summary["setup_closed"] = summary["setup_wns_ns"] >= 0.0
    summary["hold_closed"] = summary["hold_wns_ns"] >= 0.0
    return summary


def write_attempt(release_sha):
    ATTEMPT.mkdir()
    payload = {
        "schema": "m1858_c2_fresh_mapped_formality_dual_corner_pt_attempt_r1_v1",
        "status": "M1858_ATTEMPT_CONSUMED_BEFORE_FIRST_EDA",
        "axes": list(AXIS_ORDER),
        "formality_runs": 2,
        "pt_runs": 2,
        "automatic_retry": False,
        "release_sha256": release_sha,
    }
    (ATTEMPT / "attempt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    seal_directory(ATTEMPT)


def publish_failure(error):
    if not WORK.exists():
        return
    (WORK / "RUN_FAILED_OR_INCOMPLETE.txt").write_text(
        "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
        "error=" + str(error).replace("\n", " ") + "\n"
        "retry=false\n")
    seal_directory(WORK)
    failure = Path(str(RESULT) + ".failed_or_incomplete." + str(os.getpid()) + ".quarantine")
    atomic_publish_no_replace(WORK, failure)


def execute():
    if any(path.exists() for path in (RESULT, ATTEMPT, WORK, LAUNCH_LOCK)):
        raise M1858Error("M1858 namespace not fresh")
    release_sha, live_rtl_identity = verify_authority()
    resource_gate()

    SHARED_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with SHARED_QUEUE.open("a+") as queue_handle:
        try:
            fcntl.flock(queue_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise M1858Error("shared EDA queue busy")
            raise
        if same_uid_eda():
            raise M1858Error("same-UID EDA collision: " + ",".join(same_uid_eda()))
        resource_gate()
        license_gate()
        if same_uid_eda():
            raise M1858Error("same-UID EDA appeared after license gate")
        if any(path.exists() for path in (RESULT, ATTEMPT, WORK, LAUNCH_LOCK)):
            raise M1858Error("M1858 namespace changed during preflight")
        LAUNCH_LOCK.mkdir()
        try:
            WORK.mkdir()
            # Revalidate every live RTL byte and exact filelist order immediately
            # before consuming the unique attempt, after all resource/license gates.
            current_release_sha, current_live_rtl_identity = verify_authority()
            if (current_release_sha != release_sha
                    or current_live_rtl_identity != live_rtl_identity):
                raise M1858Error("sealed launch/live RTL identity changed during preflight")
            write_attempt(release_sha)
            inputs = {
                "runner_sha256": sha256(RUNNER),
                "source_contract_sha256": sha256(CONTRACT),
                "m1811_manifest_sha256": M1811_MANIFEST_SHA,
                "m1830_review_sha256": M1830_REVIEW_SHA,
                "m1860_release_sha256": release_sha,
                "formality_tcl_sha256": sha256(FM_TCL),
                "pt_tcl_sha256": sha256(PT_TCL),
                "live_rtl_source_identity": live_rtl_identity,
                "reference_filelist_order": list(live_rtl_identity.keys()),
            }
            for axis in AXIS_ORDER:
                row = AXES[axis]
                inputs[axis.lower() + "_mapped_v_sha256"] = row["mapped_v_sha"]
                inputs[axis.lower() + "_mapped_sdc_sha256"] = row["mapped_sdc_sha"]
                inputs[axis.lower() + "_svf_sha256"] = row["svf_sha"]
            (WORK / "input_identity.json").write_text(
                json.dumps(inputs, indent=2, sort_keys=True) + "\n")

            metrics = {}
            for axis in AXIS_ORDER:
                axis_dir = WORK / axis.lower()
                fm_dir = axis_dir / "formality"
                pt_dir = axis_dir / "pt"
                run_tool(FM_SHELL, FM_TCL, axis, fm_dir, "formality.log")
                passing = verify_formality(axis, fm_dir)
                run_tool(PT_SHELL, PT_TCL, axis, pt_dir, "pt.log")
                timing = verify_pt(axis, pt_dir)
                metrics[axis] = {
                    "arch_mode": AXES[axis]["arch_mode"],
                    "implementation_top": AXES[axis]["implementation_top"],
                    "formality_passing_compare_points": passing,
                    "formality_equivalent_candidate": True,
                    "setup_wns_ns": timing["setup_wns_ns"],
                    "hold_wns_ns": timing["hold_wns_ns"],
                    "setup_closed_candidate": timing["setup_closed"],
                    "hold_closed_candidate": timing["hold_closed"],
                    "pt_semantics": {
                        key: timing[key] for key in (
                            "check_timing_succeeded", "check_timing_warning_count",
                            "unconstrained_endpoint_count", "coverage",
                            "setup_constraint_violation_count",
                            "hold_constraint_violation_count",
                            "raw_constraint_violation_marker_count",
                            "forbidden_exception_token_count", "constraint_clean",
                            "coverage_clean")},
                }

            receipt = {
                "schema": "m1858_m1811_c2_fresh_mapped_formality_dual_corner_pt_receipt_r1_v1",
                "status": "PASS_RAW_M1858_C2_TWO_AXIS_FORMALITY_AND_DUAL_CORNER_PT_PENDING_INDEPENDENT_RESULT_REVIEW",
                "execution": {
                    "axes": list(AXIS_ORDER),
                    "formality_runs": 2,
                    "pt_runs": 2,
                    "dc_runs": 0,
                    "vcs_runs": 0,
                    "ptpx_runs": 0,
                    "automatic_retry": False,
                },
                "axes": metrics,
                "timing_policy": {
                    "setup_corner": "slow-max ssg0p9v125c",
                    "hold_corner": "fast-min ffg1p05vm40c",
                    "ocv": True,
                    "negative_slack_reported_not_hidden": True,
                    "timing_exceptions_added": False,
                    "pt_eco": False,
                    "hold_failure_blocks_result_publication": False,
                },
                "claim_boundary": {
                    "pending_independent_result_review": True,
                    "formality": False,
                    "prime_time": False,
                    "hold_closed": False,
                    "power": False,
                    "energy": False,
                    "cycle_speedup": False,
                    "system_speedup": False,
                    "paper_ppa_ready": False,
                    "paper_citable": False,
                    "headline": False,
                },
            }
            (WORK / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            (WORK / "RUN_COMPLETE.txt").write_text(
                "status=PASS_RAW_M1858_C2_TWO_AXIS_FORMALITY_AND_DUAL_CORNER_PT_PENDING_INDEPENDENT_RESULT_REVIEW\n"
                "formality_runs=2\npt_runs=2\nretry=false\n"
                "negative_hold_reported_not_hidden=true\n"
                "power=false\nenergy=false\ncycle_speedup=false\n"
                "system_speedup=false\npaper_ppa_ready=false\n")
            seal_directory(WORK)
            atomic_publish_no_replace(WORK, RESULT)
        except BaseException as error:
            publish_failure(error)
            raise
        finally:
            try:
                LAUNCH_LOCK.rmdir()
            except OSError:
                pass
            fcntl.flock(queue_handle.fileno(), fcntl.LOCK_UN)
    print("PASS_RAW_M1858_C2_TWO_AXIS_FORMALITY_AND_DUAL_CORNER_PT_PENDING_INDEPENDENT_RESULT_REVIEW")


def _signal_handler(signum, frame):
    del frame
    raise M1858Error("interrupted by signal " + str(signum))


def main():
    if len(sys.argv) != 1:
        raise M1858Error("M1858 accepts no arguments")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _signal_handler)
    execute()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print("ERROR: M1858 " + str(error), file=sys.stderr)
        raise SystemExit(1)
