#!/usr/bin/env python3
"""Independent M454 hammer of the M450 fixed-160B co-pack screen.

No raw M40 payload or manifest is an input.  The auditor verifies upstream
seals from each result directory, makes one independent pass over the sealed
M430 phase CSV, reconstructs all sign-conditioned correction-vector widths
from the frozen INT8 weights, and recomputes the atomic and non-executable
fragment ceilings.
"""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def signed_bits(minimum, maximum):
    for bits in range(1, 9):
        if minimum >= -(1 << (bits - 1)) and \
                maximum <= (1 << (bits - 1)) - 1:
            return bits
    raise RuntimeError("signed INT8 vector width overflow")


def run_seal_check(directory, manifest, seal):
    directory = Path(directory).resolve()
    commands = (["sha256sum", "-c", manifest],
                ["sha256sum", "-c", seal])
    output = []
    for command in commands:
        completed = subprocess.run(command, cwd=str(directory),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        require(completed.returncode == 0,
                "seal check failed in cwd " + str(directory))
        output.append("cwd=" + str(directory) + "\n" + completed.stdout)
    return "".join(output)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M454 output overwrite")
    contract = strict_json(args.contract)
    require(contract["schema"] == "m454_m450_independent_hammer_contract_v1" and
            contract["status"] == "FROZEN_BEFORE_INDEPENDENT_RECOMPUTATION",
            "M454 contract status drift")
    hw_root = args.contract.resolve().parents[1]
    source = Path(__file__).resolve()
    source_sha = sha256(source)
    require(contract["inputs"]["auditor"]["sha256"] == source_sha and
            (hw_root / contract["inputs"]["auditor"]["path"]).resolve() ==
            source, "M454 auditor identity drift")
    paths = {}
    for name, specification in contract["inputs"].items():
        path = hw_root / specification["path"]
        require(path.is_file() and sha256(path) == specification["sha256"],
                "M454 frozen input identity drift: " + name)
        paths[name] = path
    docs_before = sha256(paths["docs359"])
    catalog_before = sha256(paths["m430_catalog"])

    # The checks deliberately execute with each evidence directory as cwd so
    # relative names inside the manifests resolve exactly as sealed.
    seal_logs = {
        "m450": run_seal_check(paths["m450_result"].parent,
                               "SHA256SUMS", "SHA256SUMS.seal.sha256"),
        "m430": run_seal_check(paths["m430_result"].parent,
                               "SHA256SUMS", "SHA256SUMS.seal.sha256"),
        "m442": run_seal_check(paths["m442_seal"].parent,
                               "RUN_MANIFEST.sha256",
                               "RUN_MANIFEST.seal.sha256"),
        "m449": run_seal_check(paths["m449_seal"].parent,
                               "SHA256SUMS", "SHA256SUMS.seal.sha256"),
    }

    # Static code-path check: the analyzer contract has no raw-M40 input and
    # the analyzer AST contains no packed/value payload field or raw path.
    m450_contract = strict_json(paths["m450_contract"])
    analyzer_text = paths["m450_analyzer"].read_text(encoding="utf-8")
    analyzer_ast = ast.parse(analyzer_text)
    string_literals = [node.s for node in ast.walk(analyzer_ast)
                       if isinstance(node, ast.Str)]
    forbidden_literals = (
        "packed_file", "value_payload_file",
        "m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822")
    raw_path_literal_hits = sum(
        any(forbidden in literal for forbidden in forbidden_literals)
        for literal in string_literals)
    raw_contract_inputs = [name for name, specification in
                           m450_contract["inputs"].items()
                           if "m40_h67_ep35_bottleneck_packed_sources" in
                           specification["path"] or
                           "packed_file" in specification["path"] or
                           "value_payload" in specification["path"]]
    require(raw_path_literal_hits == 0 and not raw_contract_inputs,
            "M450 raw payload code path found")

    # Independent enumeration of every source-term/output-block vector and
    # both residual polarities: 4*6912*8*2 = 442,368 cases.
    width_histogram = Counter()
    width_rows = []
    vector_cases = 0
    negative_128_count = 0
    global_minimum = 999
    global_maximum = -999
    for operator in range(4):
        weights = np.fromfile(paths[f"weight_o{operator}"], dtype=np.int8)
        require(weights.size == 6912 * 768,
                "M454 frozen weight extent drift")
        weights = weights.reshape(6912, 8, 96).astype(np.int16)
        negative_128_count += int(np.count_nonzero(weights == -128))
        global_minimum = min(global_minimum, int(weights.min()))
        global_maximum = max(global_maximum, int(weights.max()))
        for direction, multiplier in (("positive", 1), ("negative", -1)):
            for source_index in range(6912):
                for output_block in range(8):
                    vector = weights[source_index, output_block] * multiplier
                    bits = signed_bits(int(vector.min()), int(vector.max()))
                    width_histogram[(operator, direction, bits)] += 1
                    vector_cases += 1
    for (operator, direction, bits), count in sorted(width_histogram.items()):
        width_rows.append({
            "operator": operator,
            "direction": direction,
            "signed_bits_per_lane": bits,
            "payload_bytes": bits * 12,
            "vector_cases": count,
            "fits_wide_16B": int(bits * 12 <= 16),
            "fits_narrow_64B": int(bits * 12 <= 64),
        })
    aggregate_width = Counter()
    for (_, _, bits), count in width_histogram.items():
        aggregate_width[bits] += count
    require(vector_cases == 442368 and negative_128_count == 0 and
            global_minimum == -127 and global_maximum == 127 and
            aggregate_width == Counter({6: 6, 7: 70724, 8: 371638}),
            "M454 independent width population drift")
    atomic_candidates = sum(count for (_, _, bits), count in
                            width_histogram.items() if bits * 12 <= 64)
    require(atomic_candidates == 0,
            "M454 unexpected fixed160 atomic candidate")

    # Full static-codec population audit.
    static_total = static_narrow = static_wide = 0
    with paths["m430_static_codec"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            static_total += 1
            static_narrow += int(row["narrow"])
            static_wide += 1 - int(row["narrow"])
    require((static_total, static_narrow, static_wide) ==
            (442368, 70503, 371865),
            "M454 static codec population mismatch")

    # Exactly one independent review pass over the sealed phase CSV.  This is
    # separate from, and does not alter, M450's recorded single pass.
    phase_count = 0
    phase_aggregate = Counter()
    with paths["m430_phase_csv"].open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase_count += 1
            for field in ("active_rows", "eligible_rows", "pwp_rows",
                          "exact_pwp_rows", "fallback_rows",
                          "correction_ops_per_block", "used_pwp_patterns",
                          "used_center_runs", "early_matcher"):
                phase_aggregate[field] += int(row[field])
    require(phase_count == 17280,
            "M454 independent phase extent mismatch")

    m430 = strict_json(paths["m430_result"])
    require(phase_aggregate["pwp_rows"] ==
            m430["runtime_population"]["pwp_rows"] and
            phase_aggregate["correction_ops_per_block"] ==
            m430["runtime_population"]["correction_ops_per_block"] and
            phase_aggregate["fallback_rows"] ==
            m430["runtime_population"]["fallback_rows"] and
            phase_aggregate["early_matcher"] ==
            m430["runtime_population"]["q32_early_matcher_cycles"],
            "M454 phase aggregate mismatch")
    pwp_issues = phase_aggregate["pwp_rows"] * 8
    correction_issues = phase_aggregate["correction_ops_per_block"] * 8
    runtime_narrow = (
        m430["runtime_population"]["narrow_block_descriptors_tile0"] +
        m430["runtime_population"]["narrow_block_descriptors_tile1"])
    runtime_wide = pwp_issues - runtime_narrow
    require((pwp_issues, correction_issues, runtime_narrow, runtime_wide) ==
            (127277168, 304443912, 18267843, 109009325),
            "M454 runtime issue split mismatch")

    m430_cycles = 517041352
    strong_zero_cycles = 742148386
    atomic_cycles = m430_cycles
    atomic_speedup = 1.0
    slack_bytes = runtime_narrow * 64 + runtime_wide * 16
    pooled96_hidden = min(correction_issues, slack_bytes // 96)
    pooled96_cycles = m430_cycles - pooled96_hidden
    pooled96_speedup = m430_cycles / pooled96_cycles
    pooled72_hidden = min(correction_issues, slack_bytes // 72)
    pooled72_cycles = m430_cycles - pooled72_hidden
    pooled72_speedup = m430_cycles / pooled72_cycles
    require(slack_bytes == 2913291152 and
            pooled96_hidden == 30346782 and
            pooled96_cycles == 486694570 and
            abs(pooled96_speedup - 1.0623528263321287) < 1e-15 and
            pooled72_hidden == 40462377 and
            pooled72_cycles == 476578975 and
            abs(pooled72_speedup - 1.0849017248400437) < 1e-15 and
            pooled72_speedup < 1.10,
            "M454 generosity ceiling mismatch")

    # Post-derivation upstream comparisons.
    m450_result = strict_json(paths["m450_result"])
    m450_audit = strict_json(paths["m450_audit"])
    upstream_histogram = {}
    with paths["m450_histogram"].open(
            "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            upstream_histogram[(int(row["operator"]), row["direction"],
                                int(row["signed_bits_per_lane"]))] = int(
                                    row["vector_cases"])
    width_mismatches = sum(
        upstream_histogram.get(key, 0) != count
        for key, count in width_histogram.items())
    upstream_mismatches = sum((
        width_mismatches != 0,
        m450_result["cycle_points"]["fixed160_atomic_copack_cycles"] !=
            atomic_cycles,
        m450_result["cycle_points"]
            ["fixed160_atomic_copack_speedup_vs_m430"] != atomic_speedup,
        m450_result["non_executable_global_fragment_pooling_ceiling"]
            ["optimistic_cycles"] != pooled96_cycles,
        m450_result["non_executable_global_fragment_pooling_ceiling"]
            ["optimistic_speedup_vs_m430"] != pooled96_speedup,
        m450_audit["phase_count"] != phase_count,
        m450_audit["phase_csv_passes"] != 1,
        m450_audit["raw_payload_reads"] != 0,
    ))
    require(upstream_mismatches == 0,
            "M454 upstream crosscheck mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, log in seal_logs.items():
        (args.output_dir / f"{name}_seal_check_from_result_cwd.log").write_text(
            log, encoding="utf-8")
    write_csv(args.output_dir / "m454_independent_width_histogram.csv",
              width_rows, list(width_rows[0].keys()))
    result = {
        "schema": "m454_m450_independent_recomputation_v1",
        "status": "PASS_M454_INDEPENDENT_NO_GO_CONFIRMED",
        "identity": {
            "auditor": {"path": str(source.relative_to(hw_root)),
                        "sha256": source_sha},
            "contract": {"path": str(args.contract.resolve().relative_to(hw_root)),
                         "sha256": sha256(args.contract)},
            "docs359_before": docs_before,
            "docs359_after": sha256(paths["docs359"]),
            "catalog_before": catalog_before,
            "catalog_after": sha256(paths["m430_catalog"]),
        },
        "seal_checks": {
            name: {"executed_from_result_directory_cwd": True,
                   "mismatches": 0}
            for name in seal_logs
        },
        "raw_m40_boundary": {
            "raw_m40_inputs_in_m450_contract": raw_contract_inputs,
            "raw_payload_path_literals_in_m450_analyzer_ast":
                raw_path_literal_hits,
            "historical_read_count_independently_observable": False,
            "recorded_m450_raw_payload_reads":
                m450_audit["raw_payload_reads"],
            "independent_review_raw_payload_reads": 0,
            "interpretation": "Code path and sealed audit support zero raw reads; historical process I/O cannot be reconstructed from final files alone."
        },
        "width_audit": {
            "vector_cases": vector_cases,
            "weight_minimum": global_minimum,
            "weight_maximum": global_maximum,
            "negative_128_count": negative_128_count,
            "histogram": {str(bits): aggregate_width[bits]
                          for bits in sorted(aggregate_width)},
            "minimum_payload_bytes": 72,
            "maximum_payload_bytes": 96,
            "wide_slack_bytes": 16,
            "narrow_slack_bytes": 64,
            "atomic_candidates": atomic_candidates,
        },
        "phase_audit": {
            "independent_phase_csv_passes": 1,
            "phases": phase_count,
            "aggregates": dict(phase_aggregate),
            "runtime_pwp_issues": pwp_issues,
            "runtime_correction_issues": correction_issues,
            "runtime_narrow_pwp_issues": runtime_narrow,
            "runtime_wide_pwp_issues": runtime_wide,
        },
        "cycles": {
            "strong_zero": strong_zero_cycles,
            "m430_separate": m430_cycles,
            "fixed160_atomic": atomic_cycles,
            "fixed160_speedup_vs_m430": atomic_speedup,
        },
        "non_executable_ceilings": {
            "runtime_slack_bytes": slack_bytes,
            "actual_width_min96_global_fragment": {
                "hidden_corrections": pooled96_hidden,
                "cycles": pooled96_cycles,
                "speedup_vs_m430": pooled96_speedup,
                "executable": False,
            },
            "all_vectors_impossibly_min72_global_fragment": {
                "hidden_corrections": pooled72_hidden,
                "cycles": pooled72_cycles,
                "speedup_vs_m430": pooled72_speedup,
                "executable": False,
            },
        },
        "upstream_crosscheck_mismatches": upstream_mismatches,
        "decision": {
            "fixed160_atomic_copack": "NO_GO",
            "rtl": "NO_GO",
            "minimum_required_speedup": 1.10,
            "even_impossible_72B_ceiling_below_threshold": True,
        },
        "claim_boundary": {
            "four_h67_bottleneck_conv": True,
            "new_rtl": False,
            "rtl_measured": False,
            "resource_normalized": False,
            "system_speedup": False,
            "date_headline": False,
        },
    }
    require(result["identity"]["docs359_before"] ==
            result["identity"]["docs359_after"] ==
            contract["inputs"]["docs359"]["sha256"] and
            result["identity"]["catalog_before"] ==
            result["identity"]["catalog_after"],
            "protected input changed during M454")
    require(source_sha == sha256(source), "M454 auditor changed during run")
    (args.output_dir / "m454_independent_recomputation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_M454_NO_GO_CONFIRMED vectors={} phases={} atomic={} "
          "pool96={:.12f} pool72={:.12f} docs359={}".format(
              vector_cases, phase_count, atomic_cycles, pooled96_speedup,
              pooled72_speedup, result["identity"]["docs359_after"]),
          flush=True)


if __name__ == "__main__":
    main()
