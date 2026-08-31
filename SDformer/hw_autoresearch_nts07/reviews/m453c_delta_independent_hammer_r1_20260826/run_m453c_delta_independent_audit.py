#!/usr/bin/env python3
"""Read-only delta audit of the M453c receipt-only normalization."""

import hashlib
import json
from pathlib import Path


REVIEW_CONTRACT_SHA256 = (
    "02167995ae9956eef71a1cf24bbc09037974f14a10f9b0098dd1f2f05f7f9234")


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


def verify_manifest(manifest_path, seal_path, label):
    manifest_path = Path(manifest_path)
    seal_path = Path(seal_path)
    entries = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        require(line and "  " in line, label + " malformed manifest")
        expected, name = line.split("  ", 1)
        target = Path(name)
        if not target.is_absolute():
            target = manifest_path.parent / target
        require(target.is_file() and sha256(target) == expected,
                label + " inner seal mismatch: " + name)
        entries += 1
    require(entries > 0, label + " empty manifest")
    line = seal_path.read_text(encoding="utf-8").strip()
    require("  " in line, label + " malformed outer seal")
    expected, name = line.split("  ", 1)
    require(Path(name).name == manifest_path.name and
            expected == sha256(manifest_path),
            label + " outer seal mismatch")
    return {
        "entries": entries,
        "manifest_sha256": sha256(manifest_path),
        "outer_seal_file_sha256": sha256(seal_path),
    }


def main():
    review_dir = Path(__file__).resolve().parent
    hw = review_dir.parents[1]
    contract_path = review_dir / "m453c_delta_independent_review_contract_r1.json"
    require(sha256(contract_path) == REVIEW_CONTRACT_SHA256,
            "review contract SHA drift")
    contract = strict_json(contract_path)
    paths = {}
    for name, spec in contract["frozen_inputs"].items():
        path = hw / spec["path"]
        require(path.is_file() and sha256(path) == spec["sha256"],
                "frozen input SHA drift: " + name)
        paths[name] = path

    seals = {
        "m453c": verify_manifest(paths["m453c_manifest"],
                                 paths["m453c_outer_seal"], "M453c"),
        "m453b": verify_manifest(paths["m453b_manifest"],
                                 paths["m453b_outer_seal"], "M453b"),
        "m453b_h2": verify_manifest(paths["m453b_h2_manifest"],
                                    paths["m453b_h2_outer_seal"],
                                    "M453b-H2"),
    }
    closure = strict_json(paths["m453c_receipt"])
    subject = strict_json(paths["m453b_result"])
    h2 = strict_json(paths["m453b_h2_receipt"])
    readme = paths["m453c_readme"].read_text(encoding="utf-8")

    require(closure["schema"] ==
            "m453c_m453b_claim_control_normalization_v1" and
            closure["status"] ==
            "PASS_RECEIPT_ONLY_NORMALIZATION__NO_GO_TREE_LINE",
            "M453c schema/status drift")
    require(closure["sealed_subject"]["result_sha256"] ==
            sha256(paths["m453b_result"]) and
            closure["sealed_subject"]["manifest_sha256"] ==
            sha256(paths["m453b_manifest"]) and
            closure["sealed_subject"]["outer_seal_file_sha256"] ==
            sha256(paths["m453b_outer_seal"]) and
            closure["sealed_subject"]["modified"] is False,
            "M453c subject binding/mutation boundary drift")
    require(closure["independent_hammer"]["review_receipt_sha256"] ==
            sha256(paths["m453b_h2_receipt"]) and
            closure["independent_hammer"]["manifest_sha256"] ==
            sha256(paths["m453b_h2_manifest"]) and
            closure["independent_hammer"]["outer_seal_file_sha256"] ==
            sha256(paths["m453b_h2_outer_seal"]) and
            closure["independent_hammer"]["score_over_100"] == 94 and
            closure["independent_hammer"]["severity_counts"] ==
            {"P0": 0, "P1": 1, "P2": 0},
            "M453c H2 binding/result drift")

    observed = closure["observed_inconsistency"]
    require(observed["stale_field"] == "decision.matcher_rtl" and
            observed["stale_value"] == subject["decision"]["matcher_rtl"] and
            observed["authoritative_subject_status"] == subject["status"] and
            observed["authoritative_subject_next"] ==
            subject["decision"]["next"] and
            observed["authoritative_subject_reason"] ==
            subject["decision"]["reason"] and
            h2["unique_conclusion"] == "NO_GO_TREE_LINE" and
            h2["severity_counts"] == {"P0": 0, "P1": 1, "P2": 0},
            "M453c stale-field/H2 finding drift")

    normalized = closure["normalized_decision"]
    require(normalized["disposition"] == "NO_GO_TREE_LINE" and
            normalized["matcher_rtl"] == "NO_GO_TREE_LINE" and
            normalized["materializer_rtl"] == "NO_GO_TREE_LINE" and
            normalized["m461_execution"] == "NO_GO_TREE_LINE",
            "M453c permanent line-level NO-GO drift")
    exact_numeric = {
        "tree_separate_cycles": subject["cycles"]["m453_tree_separate"],
        "m430_cycles": subject["cycles"]["m430_q32_separate"],
        "tree_separate_promotion_threshold":
            subject["decision"]["tree_separate_threshold"],
        "fused_diagnostic_cycles":
            subject["cycles"]["m453_tree_m451_fused_opportunity"],
        "strong_zero_cycles": subject["cycles"]["strong_zero"],
        "fused_promotion_threshold":
            subject["decision"]["tree_fused_threshold"],
    }
    for name, value in exact_numeric.items():
        require(normalized[name] == value,
                "M453c copied exact numeric drift: " + name)
    rounded_numeric = {
        "speedup_vs_m430":
            subject["comparisons"]["tree_separate_speedup_vs_m430"],
        "fused_diagnostic_speedup_vs_strong_zero":
            subject["comparisons"]["tree_fused_speedup_vs_strong_zero"],
    }
    rounding_errors = {}
    for name, exact in rounded_numeric.items():
        copied = normalized[name]
        require(copied == round(exact, 9) and abs(copied - exact) < 0.5e-9,
                "M453c copied rounded numeric drift: " + name)
        rounding_errors[name] = copied - exact
    require(normalized["tree_separate_threshold_pass"] is False and
            normalized["fused_threshold_pass"] is False and
            normalized["fused_resource_status"] ==
            "RESOURCE_KILLED_BY_M455_M457",
            "M453c threshold/resource-kill drift")

    admission = closure["admission"]
    require(admission["receipt_only_claim_normalization"] is True and
            admission["sealed_subject_modified"] is False and
            all(admission[name] is False for name in (
                "m40_replayed", "analyzer_executed", "new_cycle_result",
                "matcher_rtl", "materializer_rtl", "m461_execution",
                "synopsys", "energy", "ppa", "system_speedup", "headline")),
            "M453c admission boundary drift")
    rule = closure["consumer_rule"]
    require("override only the stale decision.matcher_rtl string" in rule and
            "They do not alter any numerical M453b evidence" in rule and
            "only permitted line-level interpretation is NO_GO_TREE_LINE" in rule,
            "M453c consumer rule scope/permanence drift")
    require("does **not** edit, rerun, or supersede" in readme and
            "must not" in readme and "M461" in readme and
            "matcher RTL" in readme and "materialization RTL" in readme and
            "performance claim" in readme and
            "No M40 replay, analyzer execution, RTL, GPU, or Synopsys" in readme,
            "M453c README no-go/boundary drift")
    require(closure["docs359_sha256_unchanged"] ==
            sha256(paths["docs359"]), "docs359 binding drift")

    report = {
        "status":
            "PASS_RECEIPT_ONLY_CLAIM_CONTROL_CLOSURE__NO_GO_TREE_LINE",
        "score": 98,
        "severity_counts": {"P0": 0, "P1": 0, "P2": 1},
        "unique_conclusion": "NO_GO_TREE_LINE",
        "seals": seals,
        "exact_numeric_mismatches": 0,
        "rounded_numeric": {
            "decimal_places": 9,
            "mismatches": 0,
            "rounding_errors": rounding_errors,
        },
        "consumer_rule": {
            "override_scope": "decision.matcher_rtl only",
            "m461": "PERMANENT_NO_GO_FOR_THIS_LINE",
            "matcher_rtl": "PERMANENT_NO_GO_FOR_THIS_LINE",
            "materializer_rtl": "PERMANENT_NO_GO_FOR_THIS_LINE",
        },
        "new_claims": {
            "performance": False,
            "rtl": False,
            "system": False,
            "energy": False,
            "ppa": False,
            "headline": False,
        },
        "findings": {
            "P0": [],
            "P1": [],
            "P2": [{
                "id": "M453C-DH-P2-001",
                "finding": "The two copied speedup fields are correctly rounded to nine decimals but the M453c JSON does not explicitly label their display precision.",
                "impact": "No gate, cycle count, or claim changes; both values equal round(sealed_value, 9) within 0.5e-9.",
                "recommendation": "Consumers should use the sealed M453b/H2 full-precision ratios for arithmetic and treat the M453c values as display-only context. No subject edit or rerun is required."
            }]
        },
        "independence": {
            "m40_reads": 0,
            "m40_replays": 0,
            "m453b_analyzer_executions": 0,
            "m453b_or_m453c_subject_edits": 0,
            "docs359_edits": 0,
        },
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
