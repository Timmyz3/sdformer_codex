#!/usr/bin/env python3
"""Independent, read-only arithmetic hammer for the sealed M722-r2 result.

This script does not replay the CPU model.  It verifies sealed identities and
recomputes the population, cycle, group, traffic, storage, and decision facts
from the already sealed row ledger and report.
"""

from collections import Counter, defaultdict
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path, PurePosixPath


getcontext().prec = 50
ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m722r2_lb_fuse_decoder_cpu_fastkill_r1_20260828"
HANDOFF = HW / "reviews/m722r2_lb_fuse_decoder_cpu_fastkill_author_handoff_r1_20260828"
CONTRACT = HW / "contracts/m722r2_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json"
BASE_CONTRACT = HW / "contracts/m722_lb_fuse_decoder_cpu_fastkill_contract_r1_20260828.json"
ANALYZER = HW / "system_simulator/scripts/analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py"
R1_ANALYZER = HW / "system_simulator/scripts/analyze_m722_lb_fuse_decoder_cpu_fastkill.py"
RUNNER = HW / "system_simulator/scripts/run_m722r2_lb_fuse_decoder_cpu_fastkill_r1_exact_sha.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "contract": "8fbaffd0eb2b7a1ae02298b58c3071a3a9b7ab592c890e2cb156294fd3fe8039",
    "base_contract": "e88cb84794a83026e4c8329ba6a93798a682a421095ce82f201b87e942879545",
    "analyzer": "ed2e1a638ffc533e8b7c9c1ca933e867d1182ca80ed589b2fef547fd39715165",
    "r1_analyzer": "3693fd1078738e8e3e0928080802cf2f276d5cb5951f72134a4482ce364077df",
    "runner": "37b7aa4c2ed98d75eb6064bc5a9d617ae64ff802cb7ee6b4d0f1ce7cbd8ffa82",
    "report": "363f319dc2bb49c1fa295c730fd73f0d0792a2c06d5cd8547d4f500dc03ad5d4",
    "rows": "ba5bc839fac591309baefdd496830dc8b434033e2eeabbfb19a715f8c7664f62",
    "result_manifest": "98f5832c913f9bfacfbface26dca3cbb22bab3b7883da2122ce77e16e5a95292",
    "result_outer": "bf767f0698f33874d8793e8b5478cc54b1cede73627c67157d9d404f093e3357",
    "contract_sidecar": "7b2e78e1cbe489d984133cecf852d1e53a97e8ab964d5c00fc6272d17e9063be",
    "contract_outer": "ef3b373c49a382d0b9c49fecbcfa87edffa6bd5061582cde5bc2743dc757b520",
    "base_contract_sidecar": "1a725de08645903ada5142539e190693b88549ba0fa582054fd384442a866ef2",
    "base_contract_outer": "73d0eb5a229f8345b51c5b9e456680bc08491c3840ea416a2a4d1d0d72287577",
    "handoff_json": "3f6dc4dfd8f8508e2bf199a9a64e458f4f5609dcdd4022f0c2bc86fefee247f1",
    "handoff_manifest": "bfc983aa56e6da2b25e3660133ec4430c61061d5ce18e02a47680c6e7703c3ce",
    "handoff_outer": "ef7a0095b6bec0afb3daaeda082b9417dd0cc5bfb463bb4ad26702395699d5c2",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite token: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    if not member.parts or member.is_absolute() or ".." in member.parts or member.as_posix() != name:
        raise RuntimeError("unsafe sealed member: " + name)
    return member


def verify_directory(root):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    sealed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError("malformed manifest: " + str(root))
        expected, name = fields
        if name in sealed:
            raise RuntimeError("duplicate sealed member: " + name)
        sealed.add(name)
        member = root.joinpath(*safe_member(name).parts)
        if not member.is_file() or member.is_symlink() or sha256(member) != expected:
            raise RuntimeError("sealed member mismatch: " + str(member))
    root_seals = {manifest.resolve(), outer.resolve()}
    actual = set()
    for member in root.rglob("*"):
        if member.is_symlink():
            raise RuntimeError("symlink under sealed root: " + str(member))
        if member.is_file() and member.resolve() not in root_seals:
            actual.add(member.relative_to(root).as_posix())
    if actual != sealed:
        raise RuntimeError("sealed population mismatch: " + str(root))
    fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    if fields != [sha256(manifest), "SHA256SUMS"]:
        raise RuntimeError("outer seal mismatch: " + str(root))
    return {
        "members": len(sealed),
        "manifest_sha256": sha256(manifest),
        "outer_seal_file_sha256": sha256(outer),
    }


def ratio(numerator, denominator):
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def verify_contract_sidecars(contract_path, expected_contract,
                             expected_sidecar, expected_outer):
    sidecar = Path(str(contract_path) + ".sha256")
    outer = Path(str(contract_path) + ".sha256.seal.sha256")
    side_fields = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    outer_fields = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    return (
        sha256(contract_path) == expected_contract and
        sha256(sidecar) == expected_sidecar and
        sha256(outer) == expected_outer and
        side_fields == [expected_contract, Path(contract_path).name] and
        outer_fields == [expected_sidecar, sidecar.name]
    )


def main():
    contract = strict_json(CONTRACT)
    base = strict_json(BASE_CONTRACT)
    report = strict_json(RESULT / "report.json")
    handoff = strict_json(HANDOFF / "handoff.json")
    rows = [json.loads(line) for line in
            (RESULT / "rows.jsonl").read_text(encoding="utf-8").splitlines()]

    result_seal = verify_directory(RESULT)
    handoff_seal = verify_directory(HANDOFF)
    m699 = HW / base["inputs"]["m699_directory"]
    m705 = HW / base["inputs"]["m705_review_directory"]
    m686 = HW / base["inputs"]["m686_weight_directory"]
    m699_seal = verify_directory(m699)
    m705_seal = verify_directory(m705)
    m686_seal = verify_directory(m686)

    module_counts = Counter(row["module"] for row in rows)
    sequence_counts = Counter(row["sequence"] for row in rows)
    time_counts = Counter(int(row["time"]) for row in rows)
    record_counts = Counter(int(row["record_index"]) for row in rows)
    tuple_counts = Counter((row["sequence"], int(row["sequence_sample_id"]),
                            row["module"], int(row["time"])) for row in rows)
    sequence_module_counts = Counter((row["sequence"], row["module"])
                                     for row in rows)
    headline = [row for row in rows if int(row["module_index"]) in (0, 2, 3)]

    def sum_field(selected, *keys):
        total = 0
        for row in selected:
            value = row
            for key in keys:
                value = value[key]
            total += int(value)
        return total

    a1_cycles = sum_field(headline, "a1_cycles", "total")
    lb_cycles = sum_field(headline, "lb_cycles", "total")
    a1_groups = sum_field(headline, "a1_osg_groups")
    lb_groups = sum_field(headline, "lb_direct_groups")
    a1_rmw = sum_field(headline, "traffic", "a1_onchip_psum_rmw_bytes")
    lb_rmw = sum_field(headline, "traffic", "lb_onchip_psum_rmw_bytes")
    a1_commit = sum_field(headline, "traffic", "dense_commit_bytes_a1")
    lb_commit = sum_field(headline, "traffic", "dense_commit_bytes_lb")
    a1_spill = sum_field(rows, "traffic", "a1_offchip_psum_spill_bytes")
    lb_spill = sum_field(rows, "traffic", "lb_offchip_psum_spill_bytes")
    rmw_delta = lb_rmw - a1_rmw

    per_sequence = {}
    for sequence in sorted(sequence_counts):
        selected = [row for row in headline if row["sequence"] == sequence]
        seq_a1 = sum_field(selected, "a1_cycles", "total")
        seq_lb = sum_field(selected, "lb_cycles", "total")
        per_sequence[sequence] = {
            "planes": len(selected),
            "a1_cycles": seq_a1,
            "lb_cycles": seq_lb,
            "a1_over_lb": ratio(seq_a1, seq_lb),
            "lb_slowdown_percent": format(
                (Decimal(seq_lb) / Decimal(seq_a1) - 1) * 100, ".6f"),
        }

    component_sums_valid = all(
        row["a1_cycles"]["total"] == sum(
            value for key, value in row["a1_cycles"].items() if key != "total") and
        row["lb_cycles"]["total"] == sum(
            value for key, value in row["lb_cycles"].items() if key != "total")
        for row in rows)
    port_model_valid = all(
        row["port_model"]["single_1rw_onchip_psum_port"] is True and
        row["port_model"]["serialized_group_service_covers_all_rmw"] is True and
        row["port_model"]["lb_port_conflict_events"] == 0 and
        row["port_model"]["a1_port_operations"] == row["a1_osg_groups"] * 12 and
        row["port_model"]["lb_port_operations"] == row["lb_direct_groups"] * 12
        for row in rows)
    common_commit_valid = all(
        row["a1_cycles"]["dense_output_commit"] ==
        row["lb_cycles"]["dense_output_commit"] and
        row["traffic"]["dense_commit_bytes_a1"] ==
        row["traffic"]["dense_commit_bytes_lb"]
        for row in rows)

    d3_a1 = report["storage"]["a1_no_spill_plans"]["D3"]
    d3_range = report["numeric_exactness"]["dynamic_ranges"]["D3"]
    d0_range = report["numeric_exactness"]["dynamic_ranges"]["D0"]
    checks = {
        "docs359_frozen": sha256(DOCS359) == EXPECTED["docs359"],
        "source_and_contract_identity": (
            verify_contract_sidecars(
                CONTRACT, EXPECTED["contract"], EXPECTED["contract_sidecar"],
                EXPECTED["contract_outer"]) and
            verify_contract_sidecars(
                BASE_CONTRACT, EXPECTED["base_contract"],
                EXPECTED["base_contract_sidecar"], EXPECTED["base_contract_outer"]) and
            sha256(ANALYZER) == EXPECTED["analyzer"] and
            sha256(R1_ANALYZER) == EXPECTED["r1_analyzer"] and
            sha256(RUNNER) == EXPECTED["runner"]),
        "result_identity": (
            sha256(RESULT / "report.json") == EXPECTED["report"] and
            sha256(RESULT / "rows.jsonl") == EXPECTED["rows"] and
            result_seal["manifest_sha256"] == EXPECTED["result_manifest"] and
            result_seal["outer_seal_file_sha256"] == EXPECTED["result_outer"]),
        "author_handoff_identity_and_double_seal": (
            handoff_seal["members"] == 5 and
            sha256(HANDOFF / "handoff.json") == EXPECTED["handoff_json"] and
            handoff_seal["manifest_sha256"] == EXPECTED["handoff_manifest"] and
            handoff_seal["outer_seal_file_sha256"] == EXPECTED["handoff_outer"]),
        "base_inputs_double_sealed_and_bound": (
            m699_seal["outer_seal_file_sha256"] == base["inputs"]["m699_outer_seal_file_sha256"] and
            m705_seal["outer_seal_file_sha256"] == base["inputs"]["m705_outer_seal_file_sha256"] and
            m686_seal["outer_seal_file_sha256"] == base["inputs"]["m686_outer_seal_file_sha256"] and
            sha256(m699 / "manifest.json") == base["inputs"]["m699_manifest_sha256"] and
            sha256(m705 / "review.json") == base["inputs"]["m705_review_sha256"] and
            sha256(m686 / "manifest.json") == base["inputs"]["m686_manifest_sha256"]),
        "population_120_records_1200_planes": (
            len(rows) == 1200 and len(record_counts) == 120 and
            set(record_counts.values()) == {10} and
            set(module_counts.values()) == {300} and
            set(sequence_counts.values()) == {400} and
            set(time_counts.values()) == {120} and
            len(tuple_counts) == 1200 and set(tuple_counts.values()) == {1} and
            set(sequence_module_counts.values()) == {100}),
        "headline_900_and_sequence_stratified": (
            len(headline) == 900 and
            all(value["planes"] == 300 for value in per_sequence.values())),
        "cycle_components_recompute": component_sums_valid,
        "headline_cycle_ratio_recompute": (
            a1_cycles == 21590945350 and lb_cycles == 23377337337 and
            ratio(a1_cycles, lb_cycles) == "0.923584454412"),
        "all_sequences_lose": all(
            Decimal(value["a1_over_lb"]) < 1 for value in per_sequence.values()),
        "group_inflation_recompute": (
            a1_groups == 827946728 and lb_groups == 1170190821 and
            ratio(lb_groups, a1_groups) == "1.413364871707"),
        "rmw_increment_recompute": (
            a1_rmw == 476897315328 and lb_rmw == 549335071872 and
            rmw_delta == 72437756544),
        "same_dense_commit": common_commit_valid and
            a1_commit == lb_commit == 11612160000,
        "same_serialized_1rw_model": port_model_valid,
        "both_zero_offchip_spill": a1_spill == 0 and lb_spill == 0,
        "a1_d3_acc24_240k_zero_spill": (
            d3_a1["accumulator"] == "Acc24" and
            d3_a1["stripe_width"] == 256 and
            d3_a1["stripe_count"] == 2 and
            d3_a1["source_column_overlap"] == 1 and
            d3_a1["total_bytes"] == 243200 and
            d3_a1["total_bytes"] <= report["storage"]["budget_bytes"] == 245760 and
            d3_a1["offchip_psum_spill_bytes"] == 0),
        "acc24_integer_miter_zero": (
            report["numeric_exactness"]["a1_lb_acc24_mismatches"] == 0 and
            report["numeric_exactness"]["integer_arithmetic_exact"] is True and
            all(row["a1_lb_acc24_mismatches"] == 0 for row in rows)),
        "d3_acc16_trace_safe_but_d0_not": (
            d3_range["trace_all_orders_fit_acc16"] is True and
            d3_range["order_independent_abs_prefix_bound"] == 7288 and
            d0_range["final_values_fit_acc16"] is True and
            d0_range["trace_all_orders_fit_acc16"] is False and
            d0_range["order_independent_abs_prefix_bound"] == 62696),
        "local_int8_scope_not_accuracy": all(
            value["not_checkpoint_numeric_admission"] is True
            for value in report["local_int8_probe_identities"].values()),
        "kill_and_claim_boundary": (
            report["status"] == "KILL_NO_RTL__FAIR_A1_ZERO_PSUM_SPILL" and
            report["decision"]["performance_go"] is False and
            report["decision"]["traffic_go"] is False and
            report["decision"]["rtl_authorized_now"] is False and
            all(report["claim_boundary"][key] is False for key in (
                "paper_headline", "system_speedup", "full_decoder_cycle_simulation",
                "accuracy", "checkpoint_numeric_admission", "rtl", "vcs", "eda",
                "energy", "ppa"))),
    }

    recomputed = {
        "population": {
            "records": len(record_counts),
            "planes": len(rows),
            "headline_planes": len(headline),
            "module_planes": dict(sorted(module_counts.items())),
            "sequence_planes": dict(sorted(sequence_counts.items())),
        },
        "headline": {
            "a1_cycles": a1_cycles,
            "lb_cycles": lb_cycles,
            "a1_over_lb": ratio(a1_cycles, lb_cycles),
            "lb_slowdown_percent": format(
                (Decimal(lb_cycles) / Decimal(a1_cycles) - 1) * 100, ".6f"),
            "a1_osg_groups": a1_groups,
            "lb_direct_groups": lb_groups,
            "lb_over_osg_groups": ratio(lb_groups, a1_groups),
            "a1_onchip_psum_rmw_bytes": a1_rmw,
            "lb_onchip_psum_rmw_bytes": lb_rmw,
            "lb_minus_a1_rmw_bytes": rmw_delta,
            "lb_over_a1_rmw": ratio(lb_rmw, a1_rmw),
            "commit_bytes_each": a1_commit,
            "offchip_psum_spill_bytes_each": 0,
        },
        "per_sequence": per_sequence,
        "precision_boundary": {
            "d3_acc16_abs_prefix_bound": d3_range["order_independent_abs_prefix_bound"],
            "d3_acc16_trace_safe": d3_range["trace_all_orders_fit_acc16"],
            "d0_acc16_abs_prefix_bound": d0_range["order_independent_abs_prefix_bound"],
            "d0_final_fit_acc16": d0_range["final_values_fit_acc16"],
            "d0_acc16_all_orders_safe": d0_range["trace_all_orders_fit_acc16"],
            "scope": "complete sealed M705 S3x10 local-INT8 probe only",
        },
        "d3_a1_storage": d3_a1,
    }
    payload = {
        "schema": "m732_m722r2_lb_fuse_fresh_result_hammer_recompute_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": recomputed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
