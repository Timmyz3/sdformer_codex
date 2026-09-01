#!/usr/bin/env python3
"""Different-author, no-EDA review of the M1652 C2 DC successor source."""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_dc_m1652_m1634_c2_resource_gate_successor_exact_sha_r1.sh"
OLD_RUNNER = HW / "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_axis_logic_only_exact_sha_r1.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
CONTRACT = HW / "contracts/m1652_m1634_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
TEST = HW / "system_simulator/tests/test_m1652_m1634_c2_resource_gate_successor_dc_source.py"
AUTHOR = HW / "reviews/m1652_m1634_c2_resource_gate_successor_dc_source_author_receipt_r1_20260901"
M1635 = HW / "reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_r1_20260901"
M1636 = HW / "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_20260901.json"
M1641 = HW / "reviews/m1641_m1636_m1634_m1609_c2_three_axis_dc_release_hammer_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RELEASE = HW / "contracts/m1654_m1653_m1652_m1634_c2_resource_gate_successor_dc_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1652_m1634_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1652_m1634_c2_resource_gate_successor_three_axis_dc_attempt_consumed"
LOCK = HW / "dc_handoff/runs/.m1652_m1634_c2_resource_gate_successor_three_axis_dc_launch_lock"
WORK_GLOB = ".m1652_m1634_c2_resource_gate_successor_three_axis_dc_work.*"

EXPECTED_ROWS = (
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
)

EXPECTED = {
    RUNNER: "57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3",
    OLD_RUNNER: "da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7",
    FILELIST: "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    CONTRACT: "01ee8cff796705c71a0b3c5875046ca32d08935936026315375da797d02d863c",
    TEST: "c05b5f4cb8ffef4eff4cc24a3ef9426be9d423869a96b2d735e6367d242524b9",
    AUTHOR / "review.json": "15411ee3346403ab3363fb5e751cb9948173c0d4598412068a1915bcd2878c0a",
    AUTHOR / "SHA256SUMS": "d1ef05c829cab66a1988988d1826820c9c6ee2ba32010a8763b00c1fc0a16563",
    AUTHOR / "SHA256SUMS.seal.sha256": "f0b83649261c318dc202cbdcb1309b22e10e4491b91b82caa9bdff6e426fa848",
    M1635 / "review.json": "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620",
    M1635 / "SHA256SUMS": "e47e87a00975172451069984073e83487f1cb97cdd101a240437ed789fac66aa",
    M1635 / "SHA256SUMS.seal.sha256": "9dbcef360c8038403174bbfe05e3c0f3e3f09a7235c78cac1c47ae1a94707614",
    M1636: "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088",
    Path(str(M1636) + ".sha256"): "7f919832b6924bd23a489aa8f642de39c6cd078ad21a14cea0694e3901382fb2",
    Path(str(M1636) + ".sha256.seal.sha256"): "2362be91cc058819fc324e6973c10ca5efaaaeb91f97a57f1d2aa336dc7e394c",
    M1641 / "review.json": "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6",
    M1641 / "SHA256SUMS": "e5c210dad77d008d7e5d59000cfb93c434cc823cdcde3fb0fc1c2ba1405bce50",
    M1641 / "SHA256SUMS.seal.sha256": "a60f1b6eaa44c88a0f59cd1e522bd6ea94295618f6a5902200bccc599d417459",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def verify_tree(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "outer seal drift " + str(root))
    expected = {}
    for row in manifest.read_text(encoding="ascii").splitlines():
        digest, name = row.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in expected, "unsafe manifest row")
        expected[name] = digest
    actual = set()
    for base, dirs, files in os.walk(str(root), followlinks=False):
        base_path = Path(base)
        for name in dirs:
            require(not (base_path / name).is_symlink(), "symlink directory")
        for name in files:
            path = base_path / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
                    "nonregular tree member")
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), "tree coverage drift " + str(root))
    for name, digest in expected.items():
        require(sha256(root / name) == digest, "tree member drift " + name)


def command_text(text, comment="#"):
    return "\n".join(row.split(comment, 1)[0] for row in text.splitlines())


def policy(runner, contract, rows):
    """Validate every intended source property except the known inline defect."""
    require(tuple(rows) == EXPECTED_ROWS, "12-row filelist drift")
    require(runner.count('"${headroom}" -ge 50331648') == 1 and
            '"${headroom}" -ge 67108864' not in runner, "commit gate drift")
    require(runner.count('"${mem_available}" -ge 100663296') == 1,
            "MemAvailable gate drift")
    require(runner.count('"${swap_free}" -ge 16777216') == 1,
            "SwapFree gate drift")
    require("blocked={'dc_shell','dc_shell-t','common_shell_exec','common_shell_exe'}"
            in runner and "same-UID DC collision" in runner, "collision gate drift")
    require(runner.count(
        '"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler') == 1,
        "license gate drift")
    require("axis_names=(k1 k8 k1x8)" in runner and
            "axis_modes=(0 1 2)" in runner and "for index in 0 1 2" in runner,
            "three-axis loop drift")
    require(runner.count('"${DC_SHELL}" -f "${TCL}"') == 1,
            "DC invocation drift")
    require("fresh_all_axes=true" in runner and "old_netlist_reuse=false" in runner,
            "fresh compile contract drift")
    require("TIM-209=0" in runner and "OPT-150=0" in runner and
            "slack (MET)" in runner and
            "This design has no violated constraints." in runner,
            "result predicate drift")
    for artifact in ("mapped.v", "mapped.sdc", '.ddc"', '.svf"',
                     "timing_setup.rpt", "timing_hold_diagnostic.rpt",
                     "area.rpt", "qor.rpt"):
        require(artifact in runner, "artifact predicate drift " + artifact)
    require("hold_diagnostic_only=true" in runner and "automatic_retry':False" in runner,
            "hold/retry boundary drift")
    require("PASS_M1641_M1636_C2_THREE_AXIS_DC_RELEASE_HAMMER__ONE_LAUNCH_ADMITTED"
            in runner, "M1641 binding drift")
    for digest in ("215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620",
                   "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088",
                   "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6"):
        require(digest in runner, "predecessor authority digest drift")
    require("M1652_EXPECTED_DC_RUNNER_SHA256" in runner and
            "M1652_EXPECTED_DC_RELEASE_SHA256" in runner,
            "caller pin drift")
    require(runner.index('verify_dir_seal "${HAMMER_DIR}"') <
            runner.index('mkdir -- "${ATTEMPT}"') <
            runner.index('"${DC_SHELL}" -f "${TCL}"'), "authority/order drift")
    require("rm -rf" not in runner and "pt_shell" not in runner and
            "fm_shell" not in runner, "unauthorized/destructive tool")
    gate = contract["resource_gate"]
    require(gate["old_commit_headroom_min_kib"] == 67108864 and
            gate["commit_headroom_min_kib"] == 50331648 and
            gate["mem_available_min_kib"] == 100663296 and
            gate["swap_free_min_kib"] == 16777216 and
            gate["same_uid_dc_collision_tolerance"] == 0 and
            gate["physical_or_result_condition_changed"] is False,
            "contract resource gate drift")
    fair = contract["fair_three_axis_definition"]
    require(fair["axis_order"] == ["k1", "k8", "k1x8"] and
            fair["frozen_baseline_netlist_reuse"] is False,
            "contract fairness drift")
    auth = contract["authorization"]
    require(auth["dc_runs_now"] == 0 and auth["future_dc_shell_runs_max"] == 3 and
            auth["all_other_eda_runs"] == 0 and auth["retry"] is False,
            "contract authorization drift")
    claim = contract["claim_boundary"]
    for name in ("dc_authorized", "dc_completed", "fresh_mapped_k8",
                 "setup_area", "hold_closed", "power", "energy", "formality",
                 "paper_ppa_ready", "system_speedup", "paper_headline"):
        require(claim[name] is False, "claim opened " + name)


def run_embedded_authorization_preflight(runner):
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", runner, re.S)
    selected = [text for text in snippets if "contract,runner,m1627,m903" in text]
    require(len(selected) == 1, "embedded contract preflight cardinality drift")
    completed = subprocess.run(
        ["/usr/libexec/platform-python3.6", "-I", "-", str(CONTRACT), str(RUNNER),
         str(HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901/review.json"),
         str(HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829/review.json")],
        input=selected[0], universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15, check=False)
    return completed.returncode, completed.stdout


def main():
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink(), "missing/nonregular " + str(path))
        require(sha256(path) == digest, "identity drift " + str(path))
    for root in (AUTHOR, M1635, M1641):
        verify_tree(root)
    runner = RUNNER.read_text(encoding="utf-8")
    rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines() if row.strip()]
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    policy(runner, contract, rows)
    completed = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=10, check=False)
    require(completed.returncode == 0, "bash syntax failure")
    rc, output = run_embedded_authorization_preflight(runner)
    require(rc != 0 and "AssertionError" in output,
            "expected embedded authorization defect was not reproduced")
    require(contract["authorization"] != {
        "dc_runs_now": 0, "future_dc_shell_runs_max": 3, "all_other_eda_runs": 0},
        "contract unexpectedly matches embedded short dictionary")
    require(not RELEASE.exists() and not RESULT.exists() and not ATTEMPT.exists() and
            not LOCK.exists() and not list((HW / "dc_handoff/runs").glob(WORK_GLOB)),
            "future runtime/release namespace is not clean")

    attacks = []
    def reject(label, changed_runner=None, changed_contract=None, changed_rows=None):
        try:
            policy(changed_runner if changed_runner is not None else runner,
                   changed_contract if changed_contract is not None else contract,
                   changed_rows if changed_rows is not None else rows)
        except (AssertionError, KeyError, ValueError):
            attacks.append(label)
            return
        raise AssertionError("mutation escaped " + label)

    runner_attacks = [
        ("commit_floor", '"${headroom}" -ge 50331648', '"${headroom}" -ge 1'),
        ("old_commit_floor", '"${headroom}" -ge 50331648', '"${headroom}" -ge 67108864'),
        ("mem_floor", '"${mem_available}" -ge 100663296', '"${mem_available}" -ge 1'),
        ("swap_floor", '"${swap_free}" -ge 16777216', '"${swap_free}" -ge 1'),
        ("collision", "same-UID DC collision", "collision disabled"),
        ("license", '"${LMUTIL}" lmstat -c 27030@ic.ismd-nemo -f Design-Compiler', "true"),
        ("axis_names", "axis_names=(k1 k8 k1x8)", "axis_names=(k1 k8)"),
        ("axis_modes", "axis_modes=(0 1 2)", "axis_modes=(0 1 1)"),
        ("axis_loop", "for index in 0 1 2", "for index in 0 1"),
        ("dc_call", '"${DC_SHELL}" -f "${TCL}"', "true"),
        ("fresh", "fresh_all_axes=true", "fresh_all_axes=false"),
        ("reuse", "old_netlist_reuse=false", "old_netlist_reuse=true"),
        ("tim209", "TIM-209=0", "TIM-209=1"),
        ("opt150", "OPT-150=0", "OPT-150=1"),
        ("setup", "slack (MET)", "slack (VIOLATED)"),
        ("drc", "This design has no violated constraints.", "violations ignored"),
        ("hold_boundary", "hold_diagnostic_only=true", "hold_diagnostic_only=false"),
        ("retry", "automatic_retry':False", "automatic_retry':True"),
        ("m1641_status", "PASS_M1641_M1636_C2_THREE_AXIS_DC_RELEASE_HAMMER__ONE_LAUNCH_ADMITTED", "PASS_FAKE"),
        ("m1635_digest", "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620", "0" * 64),
        ("m1636_digest", "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088", "0" * 64),
        ("m1641_digest", "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6", "0" * 64),
        ("runner_pin", "M1652_EXPECTED_DC_RUNNER_SHA256", "UNPINNED_RUNNER"),
        ("release_pin", "M1652_EXPECTED_DC_RELEASE_SHA256", "UNPINNED_RELEASE"),
        ("attempt_order", 'mkdir -- "${ATTEMPT}"', "true"),
    ]
    for label, old, new in runner_attacks:
        reject(label, changed_runner=runner.replace(old, new))
    reject("drop_row", changed_rows=rows[:-1])
    reject("reorder_row", changed_rows=[rows[1], rows[0]] + rows[2:])
    reject("predecessor_row", changed_rows=[
        "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"] + rows[1:])

    contract_attacks = [
        (("resource_gate", "commit_headroom_min_kib"), 1),
        (("resource_gate", "old_commit_headroom_min_kib"), 1),
        (("resource_gate", "mem_available_min_kib"), 1),
        (("resource_gate", "swap_free_min_kib"), 1),
        (("resource_gate", "same_uid_dc_collision_tolerance"), 1),
        (("resource_gate", "physical_or_result_condition_changed"), True),
        (("fair_three_axis_definition", "axis_order"), ["k8", "k1", "k1x8"]),
        (("fair_three_axis_definition", "frozen_baseline_netlist_reuse"), True),
        (("authorization", "dc_runs_now"), 1),
        (("authorization", "future_dc_shell_runs_max"), 4),
        (("authorization", "all_other_eda_runs"), 1),
        (("authorization", "retry"), True),
        (("claim_boundary", "dc_authorized"), True),
        (("claim_boundary", "hold_closed"), True),
        (("claim_boundary", "power"), True),
        (("claim_boundary", "paper_headline"), True),
    ]
    for index, (parts, value) in enumerate(contract_attacks):
        candidate = json.loads(json.dumps(contract))
        candidate[parts[0]][parts[1]] = value
        reject("contract_%02d" % index, changed_contract=candidate)

    result = {
        "status": "FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE",
        "score": 91,
        "p0_count": 0,
        "p1_count": 1,
        "p2_count": 1,
        "author_tests": "PASS_13_OF_13_BUT_MISSED_INLINE_PREFLIGHT",
        "expected_inline_preflight_returncode": rc,
        "expected_inline_preflight_assertion_error": True,
        "mutation_attacks_rejected": len(attacks),
        "source_identity_and_predecessor_seals": "PASS",
        "physical_and_result_predicates_unchanged": True,
        "eda_runs": 0,
        "attempts_created": 0,
        "release_authorized": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"hammer_internal_error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
