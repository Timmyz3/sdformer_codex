#!/usr/bin/env python3
"""M1277 source-only M1102<->M623 parent-component binding.

This adapter binds population and parent-access identity while preserving the
fact that the M1102 speedup and M623 energy ablation use different baselines.
It creates no production/canonical energy result and runs no EDA workload.
"""
import argparse
import copy
from decimal import Decimal, InvalidOperation, getcontext
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Dict, Iterable, Optional, Set, Tuple

getcontext().prec = 50
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent

M1102_ROOT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M1102_RESULT = M1102_ROOT / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_RESULT_SHA = "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91"
M1102_MANIFEST_SHA = "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12"
M1102_OUTER_SHA = "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f"

M1114_ROOT = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830"
M1114_ID = (
    "8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
    "3f48f2c91e1feba599fca3eab9f3c8348ed5ca5af1d317de14dd01a548b1c1b7",
    "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4",
)

M617_ROOT = HW / "results/m617_m597_m593_parent_scratch_generated_macro_energy_r5_20260828"
M617_RESULT = M617_ROOT / "m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json"
M617_RESULT_SHA = "be384c45da0efd8b59bd446313352d6375bb0d8d28933d28aca0177b15553e94"
M617_MANIFEST_SHA = "7b88c46225d47a2c02c2fe50dced97f23798bd8631e79ae3524f75c105445859"
M617_OUTER_SHA = "7269c4cc4cadc51fb9c0fb51037f22f3c85408464dabebb1c2f1605696bdce6f"

M623_ROOT = HW / "reviews/m623_m617_m597_m593_parent_scratch_energy_r5_result_hammer_r1_20260828"
M623_ID = (
    "9681239182a27192f69bbc59ec48a2bf9f6336e9c8fc0575924964f69fde6b3a",
    "2631d9813bd064e5c27bb703ea40d9212da47314e5d14b04210173db6a67d212",
    "9d67c8b76dcc3342ad4ce2fee737fa30dff67580aebe3aa584df27d5e7a09cc9",
)

DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

EXPECTED = {
    "samples": 10,
    "tasks": 812160,
    "operators": 4,
    "candidate_reads": 131926088,
    "candidate_writes": 79581608,
    "candidate_forwards": 13717024,
    "m623_forwards_per_output_block": 1714628,
    "output_block_banks": 8,
    "m1102_candidate_cycles": 434242823,
    "m1102_baseline_cycles": 763908050,
    "m617_candidate_cycles": 435293339,
    "m617_all_write_cycles": 456016645,
}


class BindingError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BindingError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha256(path) == expected, "regular-file identity drift: " + str(path))


def reject_constant(token: str) -> Any:
    raise BindingError("nonfinite JSON constant: " + token)


def no_duplicates(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=no_duplicates,
                       parse_constant=reject_constant,
                       parse_float=Decimal)
    require(isinstance(value, dict), "top-level JSON is not an object")
    return value


def strict_load_text(text: str) -> Dict[str, Any]:
    value = json.loads(text, object_pairs_hook=no_duplicates,
                       parse_constant=reject_constant, parse_float=Decimal)
    require(isinstance(value, dict), "top-level JSON is not an object")
    return value


def verify_manifest(root: Path, manifest_sha: str, outer_sha: str,
                    exact_payload: Optional[Set[str]] = None) -> None:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    # M1102 keeps the seal in a nested atomic directory.
    if not manifest.exists():
        manifest = root / ".m1102_atomic_seal/SHA256SUMS"
        outer = root / ".m1102_atomic_seal/SHA256SUMS.seal.sha256"
    regular(manifest, manifest_sha)
    regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            "outer seal content drift: " + str(root))
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "manifest grammar drift")
        relative = fields[1][2:] if fields[1].startswith("./") else fields[1]
        require(relative and not relative.startswith("/") and
                ".." not in Path(relative).parts and relative not in listed,
                "unsafe or duplicate manifest member")
        listed.add(relative)
        regular(root / relative, fields[0])
    if exact_payload is not None:
        require(listed == exact_payload, "manifest payload set drift")


def verify_flat_review(root: Path, identity: Tuple[str, str, str]) -> Dict[str, Any]:
    regular(root / "review.json", identity[0])
    verify_manifest(root, identity[1], identity[2])
    return strict_load(root / "review.json")


def integer(value: Any, expected: int, label: str) -> int:
    require(type(value) is int and value == expected, label + " mismatch")
    return value


def decimal(value: Any, expected: str, label: str) -> Decimal:
    try:
        observed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise BindingError(label + " is not decimal")
    require(observed == Decimal(expected), label + " mismatch")
    return observed


def verify_authorities() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    regular(DOCS359, DOCS359_SHA)
    verify_manifest(M1102_ROOT, M1102_MANIFEST_SHA, M1102_OUTER_SHA, {
        "RUN_COMPLETE.txt",
        "m1102_c1_work8_exact_1rw_full_replay_result_r1.json",
        "m1102_work8_domain_preflight_receipt_r1.json",
    })
    regular(M1102_RESULT, M1102_RESULT_SHA)
    verify_manifest(M617_ROOT, M617_MANIFEST_SHA, M617_OUTER_SHA, {
        "RUN_COMPLETE.txt",
        "m597_m593_m528_parent_scratch_generated_macro_energy_result_r2.json",
        "m597_parent_scratch_energy_rows_r2.csv",
        "m606_terminal_rehash_receipt.json",
        "production_stderr.log",
        "production_stdout.log",
    })
    regular(M617_RESULT, M617_RESULT_SHA)
    return (strict_load(M1102_RESULT), strict_load(M617_RESULT),
            verify_flat_review(M1114_ROOT, M1114_ID),
            verify_flat_review(M623_ROOT, M623_ID))


def row_by_design(result: Dict[str, Any], design: str) -> Dict[str, Any]:
    rows = result.get("rows")
    require(isinstance(rows, list), "M617 rows missing")
    matches = [row for row in rows
               if isinstance(row, dict) and row.get("design") == design]
    require(len(matches) == 1, "M617 design row multiplicity drift: " + design)
    return matches[0]


def build_binding(m1102: Dict[str, Any], m617: Dict[str, Any],
                  m1114: Dict[str, Any], m623: Dict[str, Any]) -> Dict[str, Any]:
    raw = m1102.get("raw_cpu_model")
    require(isinstance(raw, dict), "M1102 raw_cpu_model missing")
    coverage = raw.get("coverage")
    aggregate = raw.get("aggregate")
    require(isinstance(coverage, dict) and isinstance(aggregate, dict),
            "M1102 coverage/aggregate missing")
    candidate_parent = coverage.get("parent", {}).get("candidate", {})
    candidate_axis = aggregate.get("candidate", {})
    zero_axis = aggregate.get("strongest_zero", {})
    bit_axis = aggregate.get("same_coordinate_bit", {})
    require(all(isinstance(value, dict) for value in
                (candidate_parent, candidate_axis, zero_axis, bit_axis)),
            "M1102 axis structure drift")

    integer(m1102.get("work_domain_preflight", {}).get("tasks"),
            EXPECTED["tasks"], "M1102 tasks")
    integer(len(raw.get("samples", [])), EXPECTED["samples"], "M1102 samples")
    integer(candidate_parent.get("reads"), EXPECTED["candidate_reads"],
            "M1102 candidate parent reads")
    integer(candidate_parent.get("writes"), EXPECTED["candidate_writes"],
            "M1102 candidate parent writes")
    integer(candidate_parent.get("forwards"), EXPECTED["candidate_forwards"],
            "M1102 candidate parent forwards")
    for baseline in ("strongest_zero", "same_coordinate_bit"):
        parent = coverage.get("parent", {}).get(baseline, {})
        require(isinstance(parent, dict), baseline + " parent structure")
        integer(parent.get("reads"), 0, baseline + " parent reads")
        integer(parent.get("writes"), 0, baseline + " parent writes")
        integer(parent.get("forwards"), 0, baseline + " parent forwards")
    integer(candidate_axis.get("cycles"), EXPECTED["m1102_candidate_cycles"],
            "M1102 candidate cycles")
    integer(zero_axis.get("cycles"), EXPECTED["m1102_baseline_cycles"],
            "M1102 zero cycles")
    integer(bit_axis.get("cycles"), EXPECTED["m1102_baseline_cycles"],
            "M1102 bit cycles")

    scope = m617.get("scope")
    require(isinstance(scope, dict), "M617 scope missing")
    integer(scope.get("frozen_sampled_inference_count"), EXPECTED["samples"],
            "M617 samples")
    integer(scope.get("sequence_count"), 1, "M617 sequence count")
    require(scope.get("checkpoint") == "H67 ep35" and
            scope.get("operators") == "four bottleneck Conv3x3 only",
            "M617 frozen scope drift")
    all_write = row_by_design(m617, "m504_all_write_1rw_parent_scratch")
    dead = row_by_design(m617, "m528_dead_write_only_1rw_parent_scratch")
    integer(dead.get("read_accesses_s10"), EXPECTED["candidate_reads"],
            "M617 dead reads")
    integer(dead.get("write_accesses_s10"), EXPECTED["candidate_writes"],
            "M617 dead writes")
    integer(dead.get("raw_forwards_per_output_block"),
            EXPECTED["m623_forwards_per_output_block"], "M617 forwards/block")
    integer(dead.get("output_block_banks"), EXPECTED["output_block_banks"],
            "M617 output-block banks")
    require(dead["raw_forwards_per_output_block"] * dead["output_block_banks"] ==
            EXPECTED["candidate_forwards"], "M617 total forwards mismatch")
    integer(dead.get("cycles_s10"), EXPECTED["m617_candidate_cycles"],
            "M617 dead cycles")
    integer(all_write.get("cycles_s10"), EXPECTED["m617_all_write_cycles"],
            "M617 all-write cycles")

    require(m1114.get("status") ==
            "PASS_M1114_M1102_C1_RAW_CPU_SAME_LEDGER_RESULT_HAMMER" and
            m1114.get("admission", {}).get("raw_cpu_same_ledger_speedup_admitted") is True and
            m1114.get("admission", {}).get("ppa_or_energy_admitted") is False,
            "M1114 admission boundary drift")
    decimal(m1114.get("cycle_rederivation", {}).get("candidate_vs_strongest_zero"),
            "1.7591725401987818", "M1114 speedup")
    require(m623.get("status") ==
            "PASS_M623_M617_R5_BOUNDED_GENERATED_MACRO_COMPONENT_RESULT" and
            m623.get("claim_boundary", {}).get("component_only") is True and
            m623.get("claim_boundary", {}).get("c1_total_energy") is False and
            m623.get("claim_boundary", {}).get("date_or_paper_headline") is False,
            "M623 component boundary drift")
    decimal(m623.get("independent_recompute", {}).get("component_reduction_percent"),
            "38.228307918921945", "M623 reduction")

    latest_cycles = Decimal(EXPECTED["m1102_candidate_cycles"])
    m617_cycles = Decimal(EXPECTED["m617_candidate_cycles"])
    cycle_delta = latest_cycles - m617_cycles
    cycle_delta_percent = cycle_delta / m617_cycles * Decimal(100)
    speedup = (Decimal(EXPECTED["m1102_baseline_cycles"]) /
               Decimal(EXPECTED["m1102_candidate_cycles"]))

    return {
        "schema": "m1277_m1102_m623_parent_component_rebind_source_v1",
        "status": "PASS_SOURCE_ONLY_IDENTITY_BINDING__NO_NEW_ENERGY_RESULT",
        "population": {
            "checkpoint": "H67 ep35",
            "samples": EXPECTED["samples"],
            "tasks": EXPECTED["tasks"],
            "operators": "four bottleneck Conv3x3 only",
            "sequence_count": 1,
            "identical_between_m1102_and_m623": True,
        },
        "candidate_parent_identity": {
            "reads_s10": EXPECTED["candidate_reads"],
            "writes_s10": EXPECTED["candidate_writes"],
            "forwards_s10": EXPECTED["candidate_forwards"],
            "m1102_equals_m623": True,
        },
        "cycle_binding": {
            "m1102_candidate_cycles": EXPECTED["m1102_candidate_cycles"],
            "m617_m528_candidate_cycles": EXPECTED["m617_candidate_cycles"],
            "difference_cycles_m1102_minus_m617": int(cycle_delta),
            "difference_percent_of_m617": format(cycle_delta_percent, ".18f"),
            "m1102_baseline_cycles": EXPECTED["m1102_baseline_cycles"],
            "m1102_raw_cpu_speedup_x": format(speedup, ".16f"),
            "m623_leakage_already_on_m1102_cycles": False,
        },
        "baseline_separation": {
            "m1102_speedup_denominator": "strongest_zero_or_same_coordinate_bit",
            "m1102_baseline_parent_accesses": 0,
            "m623_energy_ablation_denominator": "m504_all_write_same_candidate_mechanism",
            "m623_all_write_parent_writes_s10": all_write["write_accesses_s10"],
            "may_claim_candidate_vs_zero_or_bit_energy_reduction": False,
            "may_merge_1p759x_and_38p2283pct_as_one_efficiency_pair": False,
        },
        "claim_boundary": {
            "source_only": True,
            "new_dynamic_energy": False,
            "new_leakage_energy": False,
            "candidate_vs_baseline_energy": False,
            "c1_total_energy": False,
            "system_energy": False,
            "rtl_or_eda": False,
            "paper_ppa_ready": False,
            "allowed_use": "machine-readable provenance bridge for two separately labelled component-table rows",
        },
    }


def expect_reject(name: str, m1102: Dict[str, Any], m617: Dict[str, Any],
                  m1114: Dict[str, Any], m623: Dict[str, Any], mutation) -> str:
    values = [copy.deepcopy(value) for value in (m1102, m617, m1114, m623)]
    mutation(*values)
    try:
        build_binding(*values)
    except BindingError:
        return name
    raise BindingError("attack was accepted: " + name)


def self_test() -> Dict[str, Any]:
    m1102, m617, m1114, m623 = verify_authorities()
    binding = build_binding(m1102, m617, m1114, m623)
    rejected = []
    rejected.append(expect_reject("m1102_read_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: a["raw_cpu_model"]["coverage"]["parent"]["candidate"].__setitem__("reads", 1)))
    rejected.append(expect_reject("m1102_write_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: a["raw_cpu_model"]["coverage"]["parent"]["candidate"].__setitem__("writes", 1)))
    rejected.append(expect_reject("m1102_forward_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: a["raw_cpu_model"]["coverage"]["parent"]["candidate"].__setitem__("forwards", 1)))
    rejected.append(expect_reject("baseline_parent_nonzero", m1102, m617, m1114, m623,
        lambda a, b, c, d: a["raw_cpu_model"]["coverage"]["parent"]["strongest_zero"].__setitem__("reads", 1)))
    rejected.append(expect_reject("m617_write_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: b["rows"][1].__setitem__("write_accesses_s10", 1)))
    rejected.append(expect_reject("m617_scope_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: b["scope"].__setitem__("checkpoint", "final")))
    rejected.append(expect_reject("m1102_cycle_drift", m1102, m617, m1114, m623,
        lambda a, b, c, d: a["raw_cpu_model"]["aggregate"]["candidate"].__setitem__("cycles", 1)))
    rejected.append(expect_reject("m1114_energy_promotion", m1102, m617, m1114, m623,
        lambda a, b, c, d: c["admission"].__setitem__("ppa_or_energy_admitted", True)))
    rejected.append(expect_reject("m623_total_energy_promotion", m1102, m617, m1114, m623,
        lambda a, b, c, d: d["claim_boundary"].__setitem__("c1_total_energy", True)))
    try:
        strict_load_text('{"x":1,"x":2}')
        raise BindingError("duplicate JSON attack was accepted")
    except BindingError:
        rejected.append("duplicate_json")
    try:
        strict_load_text('{"x":NaN}')
        raise BindingError("nonfinite JSON attack was accepted")
    except BindingError:
        rejected.append("nonfinite_json")
    require(len(rejected) == 11 and len(set(rejected)) == 11,
            "self-test attack coverage drift")
    return {
        "schema": "m1277_m1102_m623_parent_component_rebind_selftest_v1",
        "status": "PASS_M1277_SOURCE_FAIL_CLOSED_SELF_TEST__NO_PRODUCTION_RESULT",
        "positive_binding_pass": True,
        "attack_cases_rejected": len(rejected),
        "attacks": rejected,
        "binding": binding,
        "execution_boundary": {
            "canonical_result_written": False,
            "eda_gpu_remote": False,
            "docs359_written": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-binding", action="store_true")
    args = parser.parse_args()
    require(args.self_test ^ args.print_binding,
            "choose exactly one of --self-test or --print-binding")
    if args.self_test:
        payload = self_test()
    else:
        payload = build_binding(*verify_authorities())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as error:
        print("M1277_FAIL_CLOSED: " + str(error))
        raise SystemExit(2)
