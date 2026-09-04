#!/usr/bin/env python3
"""Independent, read-only hammer for the M1591 C1 macro-area model.

The production builder is executed only with CPython 3.12 into a temporary
directory.  This checker independently parses the frozen inputs and result,
recomputes storage and area arithmetic, tests the real overwrite guard, and
mutates copies only.  It never invokes EDA or modifies production evidence.
"""

from decimal import Decimal, getcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1591_c1_full_storage_macro_area_model.py"
TEST = HW / "system_simulator/tests/test_m1591_c1_full_storage_macro_area_model.py"
RESULT = HW / "results/m1591_c1_full_storage_macro_area_model_r1_20260901/m1591_c1_full_storage_macro_area_model_result_r1.json"
M1102 = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830/m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M993 = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829/m993_recovered_dc_receipt.json"
AREA = HW / "dc_handoff/runs/m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829/original_quarantine/reports/area_hierarchy.rpt"
M1114 = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830/review.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    SOURCE: "da9a7714e66a11188813ec0d3fe85fc30438790a14f099534cefa9fb037f1d78",
    TEST: "81f6b71cd08ffe0ac2bc42e69a6255a0d832afd6cb0bacafd14fd023f59a74bf",
    RESULT: "9b10348228780ca46b950cd4d603971d4d67bf9ba67412937a88c8eb4c8b3a2b",
    M1102: "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91",
    M993: "193a06e847755cca99b9dcf079cd0fee203664203e7d8b1abc8cad72c73007cc",
    AREA: "ff6683e13fe9ad8eaa0e47ff64c2f17037bfb1ee8993290331a4fc355185a94c",
    M1114: "8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
REVIEWED_COMMIT = "0d87f0bd911ef60a91d010baa0ab1ecd4c378c58"
PUBLICATION_LABEL = "[macro area model]"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_text(text):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON token: " + token)),
    )
    require(type(value) is dict, "JSON root is not object")
    return value


def strict_json(path):
    return strict_json_text(path.read_text(encoding="utf-8"))


def decimal_field(pattern, text):
    match = re.search(pattern, text, flags=re.M)
    require(match is not None, "missing area field: " + pattern)
    return Decimal(match.group(1))


def validate_model(value):
    require(set(value) == {
        "area_um2", "claim_boundary", "conservative_macro_rounding",
        "identity", "logical_storage", "schema", "status", "technology",
        "timing",
    }, "result root schema drift")
    require(value["schema"] == "m1591_c1_full_storage_macro_area_model_r1_v1",
            "schema drift")
    require(value["status"] ==
            "PASS_M1591_C1_FULL_STORAGE_CONSERVATIVE_MACRO_AREA_MODEL",
            "status drift")

    old = strict_json(M1102)
    receipt = strict_json(M993)
    capacity = old["raw_cpu_model"]["capacity"]
    parent = capacity["parent_plus_other"]
    metadata = parent["bytes"] - parent["parent_scratch_bytes"]
    macro_bytes = capacity["macro_bytes"]
    expected_counts = {
        "parent_scratch": math.ceil(parent["parent_scratch_bytes"] / macro_bytes),
        "metadata_and_reserve_conservative": math.ceil(metadata / macro_bytes),
        "psum": math.ceil(capacity["psum"]["bytes"] / macro_bytes),
        "weight": math.ceil(capacity["weight"]["bytes"] / macro_bytes),
    }
    require(expected_counts == {
        "parent_scratch": 9,
        "metadata_and_reserve_conservative": 12,
        "psum": 60,
        "weight": 24,
    }, "independent macro decomposition drift")
    total_macros = sum(expected_counts.values())
    logical_total = (parent["bytes"] + capacity["psum"]["bytes"] +
                     capacity["weight"]["bytes"])
    represented = total_macros * macro_bytes
    require((logical_total, represented, capacity["budget_bytes"]) ==
            (214912, 215040, 245760), "independent capacity arithmetic drift")
    require(value["logical_storage"] == {
        "parent_scratch_bytes": 18432,
        "metadata_and_reserve_bytes": 24448,
        "psum_bytes": 122880,
        "weight_bytes": 49152,
        "total_bytes": logical_total,
        "budget_bytes": 245760,
    }, "logical storage result drift")
    rounded = value["conservative_macro_rounding"]
    require(rounded == {
        "counts": expected_counts,
        "total_macro_count": 105,
        "represented_bytes": 215040,
        "rounding_overhead_bytes": 128,
        "budget_margin_after_rounding_bytes": 30720,
    }, "macro rounding result drift")

    report = AREA.read_text(encoding="utf-8")
    total9 = decimal_field(r"^Total cell area:\s+([0-9.]+)$", report)
    macros9 = decimal_field(r"^Macro/Black Box area:\s+([0-9.]+)$", report)
    require(total9 == Decimal("147246.392090") and
            macros9 == Decimal("78825.243164"), "M993 report area drift")
    require(Decimal(str(receipt["total_cell_area_um2_dc_reported"])) == total9 and
            receipt["macro_count"] == 9 and
            receipt["macro_cell"] == "TS1N28HPCPHVTB128X128M4S" and
            receipt["setup"] == {
                "met": True, "tns_ns": 0.0, "top100_reported_paths": 100,
                "violating_paths": 0, "wns_ns": 0.001795,
            }, "M993 receipt/report binding drift")
    logic = total9 - macros9
    each = macros9 / Decimal(9)
    modeled_macros = each * Decimal(105)
    modeled_total = logic + modeled_macros
    area = value["area_um2"]
    require(Decimal(area["dc_logic_excluding_nine_parent_macros"]) == logic and
            Decimal(area["foundry_macro_area_each_from_dc"]) == each and
            Decimal(area["modeled_105_macro_area"]) == modeled_macros and
            Decimal(area["modeled_logic_plus_full_storage"]) == modeled_total and
            Decimal(area["modeled_logic_plus_full_storage_mm2"]) ==
            modeled_total / Decimal(1000000), "modeled area arithmetic drift")
    require(modeled_total ==
            Decimal("988048.98583933333333333333333334"),
            "expected 0.988049 mm2 model drift")

    require(value["technology"] == {
        "nm": 28,
        "macro_cell": "TS1N28HPCPHVTB128X128M4S",
        "macro_geometry": "128x128-bit 1RW single-port",
        "macro_capacity_bytes": 2048,
    }, "technology drift")
    require(value["timing"] == {
        "clock_ns": 3.0,
        "existing_logic_plus_nine_macro_setup_met": True,
        "existing_setup_wns_ns": 0.001795,
        "extra_96_macros_integrated_in_timing_top": False,
    }, "timing boundary drift")
    require(value["claim_boundary"] == {
        "macro_area_model": True,
        "conservative_same_foundry_macro_scaling": True,
        "full_storage_logic_netlist": False,
        "full_storage_timing": False,
        "power": False,
        "energy": False,
        "throughput": False,
        "throughput_per_area": False,
        "system_speedup": False,
        "paper_citable_after_independent_review_with_model_label": False,
    }, "claim boundary drift")
    require(set(value["identity"].values()) == set(PINS[path] for path in
            (M1102, M993, AREA, M1114, DOCS359)), "embedded identity drift")
    return {
        "parent_macros": 9,
        "metadata_macros": 12,
        "psum_macros": 60,
        "weight_macros": 24,
        "total_macros": 105,
        "additional_macros_not_timing_integrated": 96,
        "logical_bytes": logical_total,
        "represented_bytes": represented,
        "rounding_overhead_bytes": represented - logical_total,
        "budget_bytes": capacity["budget_bytes"],
        "budget_margin_bytes": capacity["budget_bytes"] - represented,
        "logic_area_um2": str(logic),
        "macro_area_each_um2": str(each),
        "modeled_full_storage_area_um2": str(modeled_total),
        "modeled_full_storage_area_mm2": str(modeled_total / Decimal(1000000)),
    }


def require_publication_label(value, label):
    validate_model(value)
    require(label == PUBLICATION_LABEL, "missing or wrong macro-area model label")


def expect_reject(value, mutation):
    try:
        validate_model(value)
    except Exception:
        return mutation
    raise RuntimeError("mutation accepted: " + mutation)


def mutate(value, path, replacement):
    output = copy.deepcopy(value)
    cursor = output
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return output


def run():
    getcontext().prec = 32
    for path, digest in PINS.items():
        require(path.is_file() and not path.is_symlink(), "missing/nonregular pin: " + str(path))
        require(sha256(path) == digest, "pin drift: " + str(path))
    subprocess.run(["git", "cat-file", "-e", REVIEWED_COMMIT + "^{commit}"],
                   cwd=str(ROOT), check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    value = strict_json(RESULT)
    arithmetic = validate_model(value)

    test_run = subprocess.run(
        ["/usr/bin/python3.12", "-m", "unittest", "-v", str(TEST)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    require(test_run.returncode == 0 and "Ran 3 tests" in test_run.stdout and
            test_run.stdout.rstrip().endswith("OK"), "author unit tests failed")

    with tempfile.TemporaryDirectory(prefix="m1596_readonly_") as directory:
        rebuilt = Path(directory) / "rebuilt.json"
        rebuild = subprocess.run(
            ["/usr/bin/python3.12", str(SOURCE), "--out", str(rebuilt)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        require(rebuild.returncode == 0 and rebuilt.read_bytes() == RESULT.read_bytes(),
                "production rebuild differs from published result")
        occupied = Path(directory) / "occupied.json"
        occupied.write_text("occupied\n", encoding="utf-8")
        overwrite = subprocess.run(
            ["/usr/bin/python3.12", str(SOURCE), "--out", str(occupied)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        require(overwrite.returncode != 0 and
                occupied.read_text(encoding="utf-8") == "occupied\n" and
                "refuse overwrite" in overwrite.stderr,
                "real production overwrite guard failed")

    attacks = []
    attack_specs = [
        (("conservative_macro_rounding", "counts", "metadata_and_reserve_conservative"), 11, "metadata_macro_undercount"),
        (("conservative_macro_rounding", "total_macro_count"), 104, "total_macro_undercount"),
        (("conservative_macro_rounding", "represented_bytes"), 214912, "physical_equals_logical_forgery"),
        (("conservative_macro_rounding", "rounding_overhead_bytes"), 0, "rounding_overhead_erased"),
        (("logical_storage", "budget_bytes"), 215040, "budget_coordinate_mutation"),
        (("area_um2", "modeled_logic_plus_full_storage_mm2"), "0.147246392090", "nine_macro_area_substitution"),
        (("timing", "extra_96_macros_integrated_in_timing_top"), True, "extra96_timing_forgery"),
        (("claim_boundary", "full_storage_logic_netlist"), True, "full_netlist_forgery"),
        (("claim_boundary", "full_storage_timing"), True, "full_timing_forgery"),
        (("claim_boundary", "power"), True, "power_forgery"),
        (("claim_boundary", "throughput"), True, "throughput_forgery"),
        (("claim_boundary", "throughput_per_area"), True, "throughput_per_area_forgery"),
        (("claim_boundary", "system_speedup"), True, "system_speedup_forgery"),
    ]
    for path, replacement, name in attack_specs:
        attacks.append(expect_reject(mutate(value, path, replacement), name))
    for label, name in (("macro area model", "unbracketed_label"),
                        ("[PPA]", "wrong_label"), (None, "missing_label")):
        try:
            require_publication_label(value, label)
        except Exception:
            attacks.append(name)
        else:
            raise RuntimeError("publication-label mutation accepted: " + name)
    require_publication_label(value, PUBLICATION_LABEL)
    for payload, name in (("{\"a\":1,\"a\":2}", "duplicate_json_key"),
                          ("{\"a\":NaN}", "nonfinite_json")):
        try:
            strict_json_text(payload)
        except Exception:
            attacks.append(name)
        else:
            raise RuntimeError("strict JSON attack accepted: " + name)

    return {
        "schema": "m1596_m1591_independent_macro_area_model_mechanical_checks_r1_v1",
        "status": "PASS_M1596_M1591_INDEPENDENT_MACRO_AREA_MODEL_HAMMER",
        "reviewed_commit": REVIEWED_COMMIT,
        "identity": {path.relative_to(ROOT).as_posix(): digest
                     for path, digest in PINS.items()},
        "runtime": {
            "independent_checker_python36_compatible": True,
            "production_builder_python312_pass": True,
            "production_builder_python36_compatible": False,
            "production_builder_python36_reason":
                "from __future__ import annotations is unsupported by CPython 3.6",
            "author_unittest_cases_passed_python312": 3,
            "published_result_byte_exact_rebuild": True,
            "real_cli_overwrite_refusal_pass": True,
        },
        "arithmetic": arithmetic,
        "publication_boundary": {
            "paper_citable_area_model_after_this_review": True,
            "required_exact_label": PUBLICATION_LABEL,
            "existing_nine_macro_setup_is_not_full_storage_setup": True,
            "additional_96_macros_timing_integrated": False,
            "full_storage_netlist": False,
            "full_storage_timing": False,
            "power": False,
            "energy": False,
            "throughput": False,
            "throughput_per_area": False,
            "system_speedup": False,
        },
        "attacks": {
            "attempted": len(attacks),
            "rejected": len(attacks),
            "names": attacks,
        },
    }


def main():
    value = run()
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(sys.argv) == 1:
        sys.stdout.write(payload)
        return 0
    require(len(sys.argv) == 3 and sys.argv[1] == "--out", "usage: independent_review.py [--out FILE]")
    output = Path(sys.argv[2])
    require(not output.exists(), "refuse overwrite")
    output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
