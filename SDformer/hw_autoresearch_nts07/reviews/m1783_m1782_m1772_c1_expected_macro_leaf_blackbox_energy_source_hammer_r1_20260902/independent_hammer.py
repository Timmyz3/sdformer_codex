#!/usr/bin/env python3
"""Different-author, zero-EDA source hammer for M1782.

This script reads already-created source and forensic artifacts only.  It must
never import or execute the one-shot runner.  In particular, it does not query
a license server or create M1782 attempt/result namespaces.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1782_c1_expected_macro_leaf_blackbox_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1782_independent_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

AUTHOR = HW / "reviews/m1782_m1772_c1_expected_macro_leaf_blackbox_energy_source_author_receipt_r1_20260902"
NET = CHECK.NET
DOC359 = CHECK.DOC359

PINNED = {
    CHECK.CONTRACT: "7bf257abaf94146ac8bd062df489ea5795057336d8f23062a852a34ad880ce8d",
    CHECK.CONTRACT.with_name(CHECK.CONTRACT.name + ".sha256"):
        "89ec82fec4500fd9bca34a5cdd7204b871353271f123c06ce53f206940979518",
    CHECK.CONTRACT.with_name(CHECK.CONTRACT.name + ".sha256.seal.sha256"):
        "a9e3d6284790dcc14019aa4a21983d42eb9f68e34bc4ac06b4f7603099e9c00e",
    CHECK.PT_TCL: "e5c1b5157eba7a58dc7ef3326ba4aab8012a4da8d7dd09ff20a837f4664a4e16",
    CHECK.RUNNER: "4fd47d6ad137eb56f16eca92c6e716e18f9f6fb9f93f0e9f6e962b4ed8418a16",
    CHECK.CHECKER: "4f99a631adbe8e72af77023dd3a1f8941586609cce1b7108979b2ef87232323b",
    CHECK.TEST: "41116179f8c4f5689896b2381c2d311f45910c9da570907d2c0c62563c315016",
    AUTHOR / "receipt.json": "6098aef2b298b00f16082892b502d272c9513be317d02e8d389336e4dc0c8905",
    AUTHOR / "SHA256SUMS": "becedbeff98f522ede25f725ead514c0f188544a3bb7a12fd2f439b5b9145cc1",
    AUTHOR / "SHA256SUMS.seal.sha256": "75a3e4393442a034f3c80114e17bbb7602eb5da90e4a6ab9d45e43795debf540",
    NET: "d990bb416370fd07a1c241849e2fa494b94a179b47687a1a3ff2b1ab92c255e8",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(root):
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         "outer seal mismatch")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        name = fields[1].lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe/duplicate manifest member")
        need((root / rel).is_file() and not (root / rel).is_symlink(),
             "manifest member absent/nonregular")
        need(sha(root / rel) == fields[0], "manifest member drift " + name)
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    need(actual == listed, "sealed population drift")


def inventory(rows, black_count="9", macro_count="9"):
    # The second header records expected_macro_count, not a measured macro
    # count.  The live Tcl separately checks sizeof_collection(macro_cells).
    return "\n".join([
        "black_box_count=" + black_count,
        "expected_macro_count=" + macro_count,
    ] + rows) + "\n"


def row(name, ref=None, hier="false", bb="true"):
    return "name=%s ref=%s is_hierarchical=%s is_black_box=%s" % (
        name, ref if ref is not None else CHECK.MACRO_REF, hier, bb)


def expect_reject(text):
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / "inventory.rpt"
        path.write_text(text)
        try:
            CHECK.validate_black_box_inventory(path)
        except RuntimeError:
            return True
    return False


def main():
    for path, digest in PINNED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "pinned identity drift " + str(path))
    verify_seal(AUTHOR)

    source = CHECK.validate_sources()
    need(source["status"] ==
         "PASS_M1782_EXACT_EXPECTED_MACRO_LEAF_BLACKBOX_SOURCE_ONLY_NO_EDA",
         "author source checker status")
    forensic = CHECK.validate_m1772_failure()
    need(forensic == {
        "automatic_retry": False,
        "failure_phase": "PTPX_post_link_pre_SAIF",
        "mapped_sim": "PASS",
        "measurement_cycles": 253,
        "ptpx_power_result": False,
        "saif_activity_forms_per_tag": 117690,
        "saif_tx_nonzero": 0,
        "vcs_compile": "PASS",
    }, "M1772 forensic tuple drift")

    expected = list(CHECK.EXPECTED_MACRO_NAMES)
    need(expected == [
        "u_parent_scratch/g_slice_%d__u_parent_sram" % index
        for index in range(9)], "expected name set is not fixed [0..8]")
    good = [row(name) for name in expected]
    with tempfile.TemporaryDirectory() as temp_name:
        path = Path(temp_name) / "inventory.rpt"
        path.write_text(inventory(good))
        admitted = CHECK.validate_black_box_inventory(path)
    need(admitted["count"] == 9
         and admitted["ref_name"] == "TS1N28HPCPHVTB128X128M4S"
         and admitted["unexpected_black_boxes"] == 0
         and admitted["missing_expected_macros"] == 0,
         "exact inventory not admitted")

    replacement = good[:-1] + [row("u_parent_scratch/g_slice_9__u_parent_sram")]
    mutations = {
        "missing": inventory(good[:-1], black_count="8"),
        "extra": inventory(good + [row("u_extra")], black_count="10"),
        "same_count_replacement": inventory(replacement),
        "wrong_ref": inventory([row(expected[0], ref="WRONG_REF")] + good[1:]),
        "hierarchical": inventory([row(expected[0], hier="true")] + good[1:]),
        "not_blackbox": inventory([row(expected[0], bb="false")] + good[1:]),
        "duplicate_name": inventory([good[0]] + good[:-1]),
        "wrong_blackbox_header": inventory(good, black_count="8"),
        "wrong_expected_header": inventory(good, macro_count="8"),
        "malformed_row": inventory(["name=broken"] + good[1:]),
    }
    rejected = dict((name, expect_reject(text))
                    for name, text in mutations.items())
    need(all(rejected.values()), "inventory mutation escaped")

    tcl = CHECK.PT_TCL.read_text(errors="strict")
    active = CHECK.strip_tcl_comments(tcl)
    hard_gate_tokens = (
        'get_cells -hierarchical -filter "is_black_box==true"',
        'get_cells -hierarchical -filter "ref_name == $macro_cell"',
        'set expected_names([format {u_parent_scratch/g_slice_%d__u_parent_sram} $index]) 1',
        'if {$black_box_count != $expected_macro_count}',
        'if {$macro_count != $expected_macro_count}',
        'if {![info exists observed_names($expected_name)]}',
        'if {![info exists expected_names($cell_name)]}',
        'if {$ref_name ne $macro_cell}',
        'if {$is_hierarchical ne "false"}',
        'if {$is_black_box ne "true"}',
    )
    need(all(token in active for token in hard_gate_tokens),
         "live Tcl exact-set predicate omitted")
    bypasses = (
        "remove_from_collection $black_boxes",
        "remove_from_collection $macro_cells",
        "set black_boxes {}",
        "set macro_cells {}",
        "report_power $macro_cells",
        "ignore_black_box",
    )
    need(not any(token in active for token in bypasses),
         "black-box/accounting bypass present")
    need(active.index("link_design $design_name") <
         active.index("set black_boxes [get_cells") <
         active.index("read_saif -strip_path"),
         "black-box gate is not post-link/pre-SAIF")
    need("M1782_FAIL_EXACT_NET_ANNOTATION_GATE" in active
         and "M1782_FAIL_EXACT_LEAF_ANNOTATION_GATE" in active
         and "report_power -unit mW" in active,
         "annotation/power gate weakened")

    net_text = NET.read_text(errors="strict")
    instances = re.findall(
        r"^\s*TS1N28HPCPHVTB128X128M4S\s+(g_slice_[0-9]+__u_parent_sram)\s*\(",
        net_text, flags=re.MULTILINE)
    need(instances == ["g_slice_%d__u_parent_sram" % index for index in range(9)],
         "mapped netlist SRAM leaf inventory drift")
    need(net_text.count("TS1N28HPCPHVTB128X128M4S") == 9,
         "mapped netlist macro ref count drift")

    runner = CHECK.RUNNER.read_text(errors="strict")
    need('ATTEMPT = HW / "results/.m1782_' in runner
         and 'PRIVATE = HW / "results/m1782_' in runner
         and 'saif = candidate / "m1782_c1_directed_component.saif"' in runner,
         "fresh M1782 namespaces absent")
    need("m1772_c1_two_bank_public_warmup_energy_r1_20260902.private_build" not in runner,
         "M1772 private build reused")
    need(runner.count("state[\"vcs_compiles\"] += 1") == 1
         and runner.count("state[\"simv_runs\"] += 1") == 1
         and runner.count("state[\"saif_files\"] += 1") == 1
         and runner.count("state[\"ptpx_runs\"] += 1") == 1,
         "fresh1 execution counters not singular")
    need(runner.count('"+define+UNIT_DELAY"') == 1
         and "+initreg" not in runner
         and "+notimingcheck" not in runner
         and "+no_notifier" not in runner
         and "+nospecify" not in runner,
         "mapped simulation bypass/fresh compile drift")
    need("CHECK.validate_saif(saif, runtime[\"measurement_cycles\"])" in runner
         and "CHECK.validate_black_box_inventory(" in runner
         and "CHECK.whole_component_power(" in runner,
         "runtime/SAIF/inventory/power checker chain incomplete")

    for namespace in (
        HW / "results/.m1782_c1_expected_macro_leaf_blackbox_energy_attempt_consumed",
        HW / "results/m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902",
        HW / "results/m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902.failed_or_incomplete.quarantine",
        HW / "results/m1782_c1_expected_macro_leaf_blackbox_energy_r1_20260902.private_build.unsealed_do_not_cite",
    ):
        need(not namespace.exists(), "M1782 execution namespace already exists")

    output = {
        "schema": "m1783_m1782_c1_expected_macro_leaf_blackbox_source_independent_hammer_r1_v1",
        "status": "PASS_M1783_M1782_C1_EXPECTED_MACRO_LEAF_BLACKBOX_SOURCE_HAMMER_NO_EDA",
        "m1772_forensics": forensic,
        "exact_inventory": {
            "count": 9,
            "ref": CHECK.MACRO_REF,
            "names": expected,
            "leaf": True,
            "black_box": True,
        },
        "mutations_rejected": rejected,
        "fresh_execution_budget": source["fresh_execution_budget"],
        "author_claims_all_false": all(value is False
                                         for value in source["claim_boundary"].values()),
        "docs359_sha256": sha(DOC359),
        "eda_or_license_launched": False,
        "attempt_or_result_created": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
