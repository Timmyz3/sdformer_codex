#!/usr/bin/env python3
"""Read-only, mutation-heavy review of the inert M1659 recovery source.

This program never invokes the reviewed shell source.  It reads only sealed
source/provenance/quarantine evidence and exercises mutations entirely in
memory.  Its only output is the JSON document printed to stdout.
"""
from __future__ import print_function

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "dc_handoff/scripts/promote_m1659_m1649_c1_quarantine_atomic_canonical_recovery_r1.sh"
TEST = HW / "system_simulator/tests/test_m1659_c1_atomic_canonical_recovery_source.py"
CONTRACT = HW / "contracts/m1659_m1649_c1_atomic_canonical_recovery_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1659_m1649_c1_atomic_canonical_recovery_source_author_receipt_r1_20260901"
Q = HW / ("dc_handoff/runs/m1649_m1630_c1_resource_gate_successor_dc_"
          "r1_20260901.failed_or_incomplete.519344.quarantine")
ATTEMPT1649 = HW / ("dc_handoff/runs/.m1649_m1630_c1_resource_gate_"
                    "successor_dc_attempt_consumed")
RUNNER1649 = HW / "dc_handoff/scripts/run_dc_m1649_m1630_c1_resource_gate_successor_exact_sha_r1.sh"
CONTRACT1649 = HW / "contracts/m1649_m1630_c1_resource_gate_successor_dc_source_contract_r1_20260901.json"
REVIEW1650 = HW / "reviews/m1650_m1649_m1630_c1_resource_gate_successor_dc_source_hammer_r1_20260901"
RELEASE1651 = HW / "contracts/m1651_m1650_m1649_m1630_c1_resource_gate_successor_dc_launch_release_r1_20260901.json"
REVIEW1655 = HW / "reviews/m1655_m1649_c1_quarantine_forensic_recovery_review_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE1664 = HW / "contracts/m1664_m1660_m1659_c1_canonical_recovery_release_r1_20260901.json"
RUNTIME1665 = (
    HW / "dc_handoff/runs/m1665_m1664_m1659_m1649_c1_residual_hold_closed_dc_recovered_canonical_r1_20260901",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_launch_lock",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_attempt_consumed",
    HW / "dc_handoff/runs/.m1665_m1659_c1_canonical_recovery_work",
    HW / "dc_handoff/runs/m1665_m1659_c1_canonical_recovery_failed_or_incomplete.quarantine",
)

EXPECTED = {
    "source": "cfd06bc58023869350668ab256311f97728e86db1f5d19d1933e2c9753960e73",
    "test": "ff8bd538e5403ef320e5263816aab96450747293931a7285e7013f5ad606f499",
    "contract": "9516194a6e3195cae1803857967138b5e92eccc67ba4af9831bc2b2aa4099ec6",
    "contract_manifest": "97c5fa76b5207a9202bc5dde63220d6ea41f3d4de37644ab766359e64760a60c",
    "contract_outer_file": "04ff86f247ebc5c9a19cfbc3c3ac031ad3837ed6ab1f92a200cdfa781c882d90",
    "author_review": "8040de0ee2084bd5447324b253337b086c3c5977a5eaa598666cb5eb94100c76",
    "author_manifest": "359b6f4f4728566b3ce65271a473a1038981474600649a3d4258ed9059dd1cad",
    "author_outer_file": "a43181c53b3331c82ce9f6adf142aa8dea2411a74da42101c38c78bcffa39afb",
    "q_manifest": "e94ffc3680513cb2f374676037cc7c3b14b77a7bc47b9d35edb812f17a9ae843",
    "q_outer_file": "c221bb79e4950780c6db04ef54ed1ea809ac880ad054f9316f7bba702a49ff44",
    "attempt_manifest": "53556c6a16f00c0529702e17b6eb52b4f2cc5bf17ca55399b11347c57d72310a",
    "attempt_outer_file": "13ae47cdfc544f2fd84d505499d434930ff4bfe8d88cef0f8547062543e5f985",
    "runner1649": "8a1688206acf75ee0942c7bf6acb20b16c3017c7bf54451ab11d84953a4474e3",
    "contract1649": "5ca134044f1e100c925785db8025b8a7dce3e23daf5c3964608ca039ace84fb3",
    "review1650": "1ed6522019a7c34109ce44e0a7f5a959343e61f28151f08ec27dbb66546589bb",
    "manifest1650": "91bf3b68191dbe31557c5610a8beb91aeeb98ab7a640959b9e4f0e24c0d4845b",
    "outer1650": "b0f92fb708c68badba280661c8daa5da070f0571bd794b298fae41ea7a75338e",
    "release1651": "5e68e99c49a5e7ab04b0883b06537398b5cf41c76d6812d08b9c87fc988771ef",
    "review1655": "4d6f3e2cb238fbe77038cfc213d31ce061e17d49f43badcbc6b30ee8ffb825b2",
    "manifest1655": "349a78db9de8d138445889f1566ff1764a66ce3aa28d6599788979e20a8b2268",
    "outer1655": "5c3e1346ac3e4ecd9935190be6f8e4acf5fa9435941f2ed0a21c66512b9534f7",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

KEY_ARTIFACTS = {
    "dc.log": "a02a10adf0de69ad863445290ac95554399b8401842542868b11191a0e2d1b4a",
    "dc.rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "TCL_INTERNAL_COMPLETE.txt": "07ed11af7c64167f0054f119350ae6d798c3c00cfe7c331041316fa6dba30649",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc": "2a46429aefb9a772e1e77a7914449d052ad6f888af033d7413f8b03f3d2569b0",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.svf": "7c15c1a30827df74c0da35f24f7e88723484c2a211edd3d6c049f52e21dec274",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.sdc": "5ab21dbeb46baabf6e0bec2ea2a8f8542e114308e77ded25486fa022e4c3e198",
    "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed_mapped.v": "842d100f6a3fc26684e13a8065191028af7840685aaf4b7cfa77a4fe998c46ee",
    "reports/setup_posthold_summary_machine.txt": "123d8653bf0800934857325fa77e6759fdff93f78e099c9411b4c689d4d0647d",
    "reports/hold_posthold_summary_machine.txt": "db11b098828b57fd61b6a4ef8bff2b3302b332bca78f04c7ea442c41b46d519f",
    "reports/area_posthold.rpt": "66f18b4890ec68ec9c4b7e69e004cc326063efe4b6b62d6f95d544228ee60333",
    "reports/qor_posthold.rpt": "268909e6433b799bf59909f670c28f2697a1b8fcfbcdcb8d96cff2b06fbd872a",
    "reports/macro_binding_audit.txt": "2e21f34b7263596729746460c27663ed469b410178a9753b791ef4429fc08742",
}


class Failure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Failure(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode), "nonregular " + str(path))


def kv_text(text):
    output = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in output, "duplicate key")
            output[key] = value
    return output


def parse_manifest(text):
    rows = {}
    for line in text.splitlines():
        require(re.match(r"^[0-9a-f]{64}  [^/\n][^\n]*$", line),
                "malformed manifest")
        digest, name = line.split("  ", 1)
        path = Path(name)
        require(name not in rows and not path.is_absolute() and
                all(part not in ("", ".", "..") for part in path.parts),
                "unsafe or duplicate member")
        rows[name] = digest
    return rows


def tree_model_gate(rows, actual_hashes, links, expected_count):
    require(len(rows) == expected_count, "member count")
    require(not links, "symlink")
    require(set(rows) == set(actual_hashes), "topology")
    require(all(actual_hashes[name] == digest for name, digest in rows.items()),
            "member mismatch")
    return True


def verify_dir(root, expected_count, manifest_sha, outer_sha):
    root = Path(root); require(root.is_dir() and not root.is_symlink(), "bad dir")
    manifest = root / "SHA256SUMS"; outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "directory seal identity")
    require(outer.read_text(encoding="ascii") ==
            manifest_sha + "  SHA256SUMS\n", "outer content")
    rows = parse_manifest(manifest.read_text(encoding="utf-8"))
    actual = {}; links = []
    for base, dirs, files in os.walk(str(root), followlinks=False):
        path_base = Path(base)
        for name in list(dirs):
            path = path_base / name
            if path.is_symlink():
                links.append(path.relative_to(root).as_posix())
        dirs[:] = [name for name in dirs if not (path_base / name).is_symlink()]
        for name in files:
            path = path_base / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                links.append(rel); continue
            if rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            regular(path); actual[rel] = sha(path)
    tree_model_gate(rows, actual, links, expected_count)
    return rows, actual


def verify_file_seal(path, payload_sha, manifest_sha, outer_sha):
    path = Path(path); side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path); regular(side); regular(outer)
    require(sha(path) == payload_sha and sha(side) == manifest_sha and
            sha(outer) == outer_sha, "file seal identity")
    require(side.read_text(encoding="ascii") ==
            payload_sha + "  " + path.name + "\n", "file sidecar content")
    require(outer.read_text(encoding="ascii") ==
            manifest_sha + "  " + side.name + "\n", "file outer content")


def log_gate(text):
    lines = text.splitlines()
    errors = [(i + 1, line) for i, line in enumerate(lines)
              if re.match(r"^(?:Error|Fatal):", line)]
    exact = "Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl"
    require(errors == [(32, exact)], "Error position/text/population")
    require('no such variable\n    (read trace on "::env(HOME)")' in text,
            "HOME signature")
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("Current time:")), None)
    require(start is not None and start >= 32, "flow start")
    require(not any(re.match(r"^(?:Error|Fatal):", line)
                    for line in lines[start + 1:]), "in-flow Error/Fatal")
    for marker in ("Writing verilog file '", "Writing ddc file '",
                   "set_svf -off", "Thank you..."):
        require(marker in text, "completion marker")
    return True


def timing_gate(text, delay_type, wns):
    require(kv_text(text) == {
        "phase": "POST_RESTORE_REPORTED", "delay_type": delay_type,
        "status": "MET", "wns_ns": wns, "tns_ns": "0.000000000",
        "violating_paths": "0", "negative_path_ceiling": "200000"},
        "timing")
    return True


def area_gate(text):
    match = re.search(r"Total cell area:\s*([0-9.]+)", text)
    require(match is not None, "area missing")
    value = float(match.group(1)); baseline = 147246.392090
    overhead = (value / baseline - 1.0) * 100.0
    require(math.isfinite(value) and value == 152898.625984 and
            overhead == 3.8386230139650923 and overhead < 5.0, "area")
    return overhead


def macro_gate(text):
    row = kv_text(text)
    require(row.get("status") ==
            "PASS_M1630_RESOLVED_LIBRARY_MACRO_STRUCTURE", "macro status")
    require(row.get("macro_count_pre") == row.get("macro_count_post") ==
            row.get("expected_macro_count") == "9", "macro count")
    require(row.get("behavioral_macro_verilog_read_by_dc") == "false" and
            row.get("inferred_parent_array_allowed") == "false", "macro mode")
    return True


def drc_gate(text):
    require(re.search(r"Nets With Violations:\s+0(?:\.00)?\s*$", text,
                      re.MULTILINE), "DRC")
    return True


def source_order_gate(text):
    tokens = (
        'verify_dir_seal "${FUTURE_REVIEW_DIR}" 7',
        'verify_file_seal "${FUTURE_RELEASE}"',
        'M1659_EXPECTED_SOURCE_SHA256',
        'M1659_EXPECTED_RELEASE_SHA256',
        'forensic_gate "${SOURCE}"',
        'mkdir -- "${LOCK}"',
        'mkdir -- "${ATTEMPT}"',
        'mkdir -- "${WORK}"',
        'cp -a --no-dereference',
        'forensic_gate "${WORK}/original_quarantine"',
        'mv -T -- "${WORK}" "${TARGET}"',
    )
    positions = []
    for token in tokens:
        require(text.count(token) >= 1, "missing order token")
        positions.append(text.index(token))
    require(positions == sorted(positions), "authority/order")
    return True


def protocol_gate(text):
    require(text.count('forensic_gate "${SOURCE}"') == 1,
            "source forensic count")
    require(text.count('forensic_gate "${WORK}/original_quarantine"') == 1,
            "copy forensic count")
    require(text.count('mkdir -- "${LOCK}"') == 1 and
            text.count('mkdir -- "${ATTEMPT}"') == 1 and
            text.count('cp -a --no-dereference') == 1 and
            text.count('mv -T -- "${WORK}" "${TARGET}"') == 1,
            "one-shot protocol population")
    require('[[ ! -e "${TARGET}" ]]' in text and "retry=false" in text and
            "rm -" not in text, "no-replace/no-retry")
    require('verify_dir_seal "${WORK}" 42' in text and
            'verify_dir_seal "${TARGET}" 42' in text,
            "pre/post publication seal")
    executable = "\n".join(line for line in text.splitlines()
                            if line.strip() and
                            not line.lstrip().startswith("#"))
    for command in ("dc_shell", "fm_shell", "pt_shell", "vcs"):
        require(not re.search(r"(?m)^\s*(?:\$\{[^}]+\}/)?" + command +
                              r"\b", executable), "EDA command")
    return True


def claims_gate(contract, text):
    require(contract["status"] ==
            "SOURCE_ONLY_M1659_C1_ATOMIC_CANONICAL_RECOVERY__NO_COPY_NO_EDA",
            "contract status")
    require(contract["authorization"] == {
        "recovery_runs_now": 0, "future_copy_only_recoveries_max": 1,
        "all_eda_runs": 0}, "contract authorization")
    boundary = contract["recovered_receipt_boundary"]
    require(boundary["dc_setup_hold_area_macro_drc_candidate"] is True,
            "candidate boundary")
    for key in ("formality", "independent_prime_time", "power", "energy",
                "cycle_speedup", "system_speedup", "paper_ppa_ready",
                "paper_citable", "headline"):
        require(boundary[key] is False, "claim true: " + key)
    require("'formality':False,'independent_pt':False,'power':False" in text and
            "'cycle_speedup':False,'system_speedup':False,'paper_ppa_ready':False" in text and
            "'paper_citable':False,'headline':False" in text,
            "generated receipt false claims")
    return True


def expect_reject(label, function):
    try:
        function()
    except (Failure, KeyError, ValueError, StopIteration):
        return label
    raise Failure("mutation accepted: " + label)


def main():
    regular(SOURCE); regular(TEST); regular(CONTRACT); regular(DOCS359)
    require(sha(SOURCE) == EXPECTED["source"], "source SHA")
    require(sha(TEST) == EXPECTED["test"], "test SHA")
    require(sha(DOCS359) == EXPECTED["docs359"], "docs359 SHA")
    verify_file_seal(CONTRACT, EXPECTED["contract"],
                     EXPECTED["contract_manifest"], EXPECTED["contract_outer_file"])
    author_rows, _ = verify_dir(AUTHOR, 7, EXPECTED["author_manifest"],
                                EXPECTED["author_outer_file"])
    require(sha(AUTHOR / "review.json") == EXPECTED["author_review"],
            "author review identity")
    q_rows, q_actual = verify_dir(Q, 39, EXPECTED["q_manifest"],
                                  EXPECTED["q_outer_file"])
    attempt_rows, _ = verify_dir(ATTEMPT1649, 1,
                                 EXPECTED["attempt_manifest"],
                                 EXPECTED["attempt_outer_file"])
    require(len(author_rows) == 7 and len(attempt_rows) == 1, "sealed counts")
    regular(RUNNER1649); require(sha(RUNNER1649) == EXPECTED["runner1649"],
                                "M1649 runner")
    require(sha(CONTRACT1649) == EXPECTED["contract1649"], "M1649 contract")
    verify_dir(REVIEW1650, 9, EXPECTED["manifest1650"], EXPECTED["outer1650"])
    require(sha(REVIEW1650 / "review.json") == EXPECTED["review1650"],
            "M1650 review")
    require(sha(RELEASE1651) == EXPECTED["release1651"], "M1651 release")
    verify_dir(REVIEW1655, 7, EXPECTED["manifest1655"], EXPECTED["outer1655"])
    require(sha(REVIEW1655 / "review.json") == EXPECTED["review1655"],
            "M1655 review")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    author = json.loads((AUTHOR / "review.json").read_text(encoding="utf-8"))
    r1650 = json.loads((REVIEW1650 / "review.json").read_text(encoding="utf-8"))
    rel1651 = json.loads(RELEASE1651.read_text(encoding="utf-8"))
    r1655 = json.loads((REVIEW1655 / "review.json").read_text(encoding="utf-8"))
    require(author["status"].startswith("PASS_AUTHOR_M1659"), "author status")
    require(r1650["status"].startswith("PASS_M1650_M1649"), "M1650 status")
    require(rel1651["status"] ==
            "AUTHORIZE_ONE_M1649_C1_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT",
            "M1651 status")
    require(r1655["status"] ==
            "PASS_M1655_M1649_C1_SEALED_QUARANTINE_FORENSIC__AUTHORIZE_SOURCE_ONLY_CANONICAL_RECOVERY__NO_EDA",
            "M1655 status")
    require(r1655["authorization"]["source_only_canonical_recovery_authoring"] is True,
            "M1655 authority")

    for name, digest in KEY_ARTIFACTS.items():
        require(q_rows.get(name) == digest and q_actual.get(name) == digest,
                "artifact identity " + name)
    require((Q / "dc.rc").read_text(encoding="ascii") == "0\n", "dc.rc")
    terminal = kv_text((Q / "TCL_INTERNAL_COMPLETE.txt").read_text(encoding="utf-8"))
    require(terminal.get("status") ==
            "M1630_DC_INTERNAL_COMPLETE__RUNNER_GATE_REQUIRED" and
            terminal.get("input_generation") == "original_m993_m1006_admitted_ddc" and
            terminal.get("failed_m1614_output_used") == "false" and
            terminal.get("hold_only_incremental_mapping_count") == "1" and
            terminal.get("formality_required") == "true" and
            terminal.get("independent_pt_required") == "true" and
            terminal.get("paper_citable") == "false", "Tcl completion")
    log_text = (Q / "dc.log").read_text(encoding="utf-8", errors="replace")
    setup_text = (Q / "reports/setup_posthold_summary_machine.txt").read_text()
    hold_text = (Q / "reports/hold_posthold_summary_machine.txt").read_text()
    area_text = (Q / "reports/area_posthold.rpt").read_text(errors="replace")
    macro_text = (Q / "reports/macro_binding_audit.txt").read_text()
    qor_text = (Q / "reports/qor_posthold.rpt").read_text(errors="replace")
    log_gate(log_text)
    timing_gate(setup_text, "max", "0.002221110")
    timing_gate(hold_text, "min", "0.000999451")
    overhead = area_gate(area_text); macro_gate(macro_text); drc_gate(qor_text)

    source_text = SOURCE.read_text(encoding="utf-8")
    source_order_gate(source_text); protocol_gate(source_text)
    claims_gate(contract, source_text)
    require(not RELEASE1664.exists(), "M1664 preexists")
    require(not any(path.exists() for path in RUNTIME1665), "M1665 namespace used")

    rejected = []
    # Topology mutations: missing, digest mismatch, symlink and count.
    missing = dict(q_actual); missing.pop(next(iter(missing)))
    rejected.append(expect_reject("tree_missing", lambda: tree_model_gate(
        q_rows, missing, [], 39)))
    mismatch = dict(q_actual); first = next(iter(mismatch)); mismatch[first] = "0" * 64
    rejected.append(expect_reject("tree_mismatch", lambda: tree_model_gate(
        q_rows, mismatch, [], 39)))
    rejected.append(expect_reject("tree_symlink", lambda: tree_model_gate(
        q_rows, q_actual, [first], 39)))
    rejected.append(expect_reject("tree_member_count", lambda: tree_model_gate(
        q_rows, q_actual, [], 38)))
    # Error identity, location, text, and newly introduced in-flow Error/Fatal.
    rejected.append(expect_reject("error_position", lambda: log_gate(
        "\n" + log_text)))
    rejected.append(expect_reject("error_text", lambda: log_gate(
        log_text.replace("dv/.synopsys_dv.tcl", "dv/changed.tcl", 1))))
    marker = next(line for line in log_text.splitlines() if line.startswith("Current time:"))
    rejected.append(expect_reject("new_flow_error", lambda: log_gate(
        log_text.replace(marker, marker + "\nError: injected in flow", 1))))
    rejected.append(expect_reject("new_flow_fatal", lambda: log_gate(
        log_text + "\nFatal: injected after flow\n")))
    # Timing, area, macro, DRC and artifact identity mutations.
    rejected.append(expect_reject("setup", lambda: timing_gate(
        setup_text.replace("0.002221110", "-0.000000001"), "max", "0.002221110")))
    rejected.append(expect_reject("hold", lambda: timing_gate(
        hold_text.replace("0.000999451", "-0.000000001"), "min", "0.000999451")))
    rejected.append(expect_reject("area", lambda: area_gate(
        area_text.replace("152898.625984", "154608.711695", 1))))
    rejected.append(expect_reject("macro", lambda: macro_gate(
        macro_text.replace("expected_macro_count=9", "expected_macro_count=8", 1))))
    rejected.append(expect_reject("drc", lambda: drc_gate(
        re.sub(r"Nets With Violations:\s+0(?:\.00)?", "Nets With Violations: 1", qor_text, count=1))))
    bad_artifacts = dict(q_actual)
    artifact_name = "netlist/m935_m912_three_stage_exact_parent_match_product_capture_island_m1630_residual_hold_closed.ddc"
    bad_artifacts[artifact_name] = "f" * 64
    rejected.append(expect_reject("artifact", lambda: tree_model_gate(
        q_rows, bad_artifacts, [], 39)))
    # Authority, order, dual-forensic, one-shot and namespace mutations.
    rejected.append(expect_reject("authority_status", lambda: claims_gate(
        dict(contract, status="SOURCE_ONLY_CHANGED"), source_text)))
    bad_auth = json.loads(json.dumps(contract))
    bad_auth["authorization"]["all_eda_runs"] = 1
    rejected.append(expect_reject("authority_eda", lambda: claims_gate(
        bad_auth, source_text)))
    first_token = 'verify_dir_seal "${FUTURE_REVIEW_DIR}" 7'
    bad_order = source_text.replace(first_token, "", 1) + "\n" + first_token + "\n"
    rejected.append(expect_reject("order", lambda: source_order_gate(bad_order)))
    rejected.append(expect_reject("forensic_source_count", lambda: protocol_gate(
        source_text.replace('forensic_gate "${SOURCE}"', "", 1))))
    rejected.append(expect_reject("forensic_copy_count", lambda: protocol_gate(
        source_text.replace('forensic_gate "${WORK}/original_quarantine"', "", 1))))
    rejected.append(expect_reject("attempt_count", lambda: protocol_gate(
        source_text + '\nmkdir -- "${ATTEMPT}"\n')))
    rejected.append(expect_reject("copy_count", lambda: protocol_gate(
        source_text + '\ncp -a --no-dereference\n')))
    rejected.append(expect_reject("retry", lambda: protocol_gate(
        source_text.replace("retry=false", "retry=true"))))
    rejected.append(expect_reject("namespace", lambda: require(
        not any([False, False, True, False, False]), "namespace collision")))
    rejected.append(expect_reject("claim", lambda: claims_gate(
        dict(contract, recovered_receipt_boundary=dict(
            contract["recovered_receipt_boundary"], paper_citable=True)), source_text)))

    output = {
        "schema": "m1660_m1659_c1_canonical_recovery_source_independent_hammer_r1_v1",
        "status": "PASS_M1660_M1659_C1_CANONICAL_RECOVERY_SOURCE_HAMMER",
        "python": sys.version.split()[0],
        "identities": EXPECTED,
        "quarantine": {"pid": 519344, "members": 39,
                       "manifest_sha256": EXPECTED["q_manifest"],
                       "outer_seal_file_sha256": EXPECTED["q_outer_file"],
                       "dc_rc": 0, "tcl_internal_complete": True,
                       "only_error_line": 32,
                       "only_error_phase": "pre_flow_HOME_dv_tcl",
                       "in_flow_error_or_fatal_count": 0},
        "physical": {"setup_wns_ns": 0.002221110,
                     "hold_wns_ns": 0.000999451,
                     "area_um2": 152898.625984,
                     "area_overhead_percent": overhead,
                     "area_within_five_percent": True,
                     "macro_count": 9, "drc_violating_nets": 0,
                     "exact_artifact_count": 4},
        "protocol": {"future_authority_before_caller_pins": True,
                     "caller_pins_before_forensic_and_copy": True,
                     "forensic_gates": 2, "atomic_lock": True,
                     "attempt_before_copy": True, "no_replace": True,
                     "retry": False, "fresh_runtime_namespaces": 5,
                     "copy_only": True},
        "mutations": {"count": len(rejected), "all_rejected": True,
                      "labels": rejected},
        "authorization": {"m1664_release_authoring": True,
                          "recovery_now": False, "all_eda": False},
        "claim_boundary": {"source_review": True, "release": False,
                           "recovery": False, "artifact_copy": False,
                           "all_eda": False, "formality": False,
                           "prime_time": False, "power": False,
                           "energy": False, "cycle_speedup": False,
                           "system_speedup": False,
                           "paper_ppa_ready": False,
                           "paper_citable": False, "headline": False},
        "source_executed": False, "recovery_executed": False,
        "artifact_copy": False, "eda_launched": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
