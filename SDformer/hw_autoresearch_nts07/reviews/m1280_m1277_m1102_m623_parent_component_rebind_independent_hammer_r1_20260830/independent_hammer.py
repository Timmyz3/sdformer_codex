#!/usr/bin/env python3
"""M1280 independent read-only hammer for the M1277 provenance bridge."""
from __future__ import annotations

import copy
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any

getcontext().prec = 60
ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = HW / "system_simulator/scripts/build_m1277_m1102_m623_parent_component_rebind_source.py"
CONTRACT = HW / "contracts/m1277_m1102_m623_parent_component_rebind_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1277_m1102_m623_parent_component_rebind_source_receipt_r1_20260830"
DOCS = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    SOURCE: "92b9cc8135b30e1fbba7b15f5e4575cf31cdfb668695de043ad84d5bf51343b1",
    CONTRACT: "59c08c0d0f09df349ded2d033bac33b41eab3488df36073fff8a5309a5f9c0d8",
    AUTHOR / "review.json": "e9833dfc0bd7e0132f13b422c38bf57e64ec72537782b2cc5d9aaf9566070fcb",
    AUTHOR / "review.md": "e6fadcd985108d87a47c22cfd4f36717952758f2cf61a74be6c3fef106888f86",
    AUTHOR / "selftest.json": "af9cc0133dc6b95d1e7ff7de3b6a23746e1333c7bd10e7a697a4f4abc14a4197",
    AUTHOR / "RUN_COMPLETE.txt": "fbd44b32133747151eb9824654a133e6a704af31c2bc346eb813d168b53ccbcb",
    DOCS: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "identity drift: " + str(path))


def strict(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON")
            out[key] = value
        return out
    def reject(token):
        raise HammerError("nonfinite JSON: " + token)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_float=Decimal, parse_constant=reject)
    require(isinstance(value, dict), "JSON root not object")
    return value


def load_source():
    name = "m1280_frozen_m1277"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    require(spec is not None and spec.loader is not None, "cannot import M1277")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_flat_manifest(root: Path, review_sha: str,
                         manifest_sha: str, outer_sha: str) -> None:
    regular(root / "review.json", review_sha)
    regular(root / "SHA256SUMS", manifest_sha)
    regular(root / "SHA256SUMS.seal.sha256", outer_sha)
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "outer seal content drift")
    seen = set()
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in seen and ".." not in Path(relative).parts,
                "unsafe manifest")
        seen.add(relative); regular(root / relative, digest)


def independent_recompute(module, m1102, m617, m1114, m623) -> dict[str, Any]:
    raw = m1102["raw_cpu_model"]
    parent = raw["coverage"]["parent"]
    require(len(raw["samples"]) == 10 and m1102["work_domain_preflight"]["tasks"] == 812160,
            "M1102 population mismatch")
    for design in ("candidate", "strongest_zero", "same_coordinate_bit"):
        total = sum(row["designs"][design]["cycles_after_commit"] for row in raw["samples"])
        require(total == raw["aggregate"][design]["cycles"], "sample cycle sum mismatch")
    require(parent["candidate"] == {"reads": 131926088, "writes": 79581608,
            "forwards": 13717024, "work_cycles": 409734336},
            "candidate parent vector mismatch")
    for design in ("strongest_zero", "same_coordinate_bit"):
        require(all(parent[design][key] == 0 for key in ("reads", "writes", "forwards")),
                "baseline parent accesses not zero")
    rows = {row["design"]: row for row in m617["rows"]}
    require(set(rows) == {"m504_all_write_1rw_parent_scratch",
                          "m528_dead_write_only_1rw_parent_scratch"}, "M617 rows drift")
    all_write = rows["m504_all_write_1rw_parent_scratch"]
    dead = rows["m528_dead_write_only_1rw_parent_scratch"]
    require((dead["read_accesses_s10"], dead["write_accesses_s10"],
             dead["raw_forwards_per_output_block"] * dead["output_block_banks"]) ==
            (131926088, 79581608, 13717024), "M617/M1102 vector mismatch")
    require(all_write["write_accesses_s10"] == 218444544 and
            all_write["read_accesses_s10"] == 131926088, "all-write denominator drift")
    candidate = raw["aggregate"]["candidate"]["cycles"]
    baseline = raw["aggregate"]["strongest_zero"]["cycles"]
    old = dead["cycles_s10"]
    delta = candidate - old
    percent = Decimal(delta) / Decimal(old) * Decimal(100)
    speedup = Decimal(baseline) / Decimal(candidate)
    reduction = Decimal(str(m623["independent_recompute"]["component_reduction_percent"]))
    require((candidate, baseline, old, delta) ==
            (434242823, 763908050, 435293339, -1050516), "cycle rebind drift")
    require(speedup == Decimal(763908050) / Decimal(434242823), "speedup arithmetic drift")
    require(reduction == Decimal("38.228307918921945"), "M623 reduction drift")
    require(m1114["admission"]["ppa_or_energy_admitted"] is False and
            m623["claim_boundary"]["c1_total_energy"] is False and
            m623["claim_boundary"]["system_or_full_network_energy"] is False,
            "upstream energy admission drift")
    return {"samples": 10, "tasks": 812160, "operators": 4, "sequence_count": 1,
            "reads": 131926088, "writes": 79581608, "forwards": 13717024,
            "candidate_cycles": candidate, "baseline_cycles": baseline,
            "m617_cycles": old, "delta_cycles": delta,
            "delta_percent": format(percent, ".18f"),
            "speedup_x": format(speedup, ".16f"),
            "m623_all_write_writes": 218444544,
            "m623_component_reduction_percent": str(reduction)}


def validate_projection(value: dict[str, Any]) -> None:
    require(value["schema"] == "m1277_m1102_m623_parent_component_rebind_source_v1" and
            value["status"] == "PASS_SOURCE_ONLY_IDENTITY_BINDING__NO_NEW_ENERGY_RESULT",
            "schema/status promotion")
    require(value["population"] == {"checkpoint": "H67 ep35", "samples": 10,
            "tasks": 812160, "operators": "four bottleneck Conv3x3 only",
            "sequence_count": 1, "identical_between_m1102_and_m623": True},
            "population/checkpoint promotion")
    require(value["candidate_parent_identity"] == {"reads_s10": 131926088,
            "writes_s10": 79581608, "forwards_s10": 13717024,
            "m1102_equals_m623": True}, "count/identity promotion")
    require(value["cycle_binding"] == {"m1102_candidate_cycles": 434242823,
            "m617_m528_candidate_cycles": 435293339,
            "difference_cycles_m1102_minus_m617": -1050516,
            "difference_percent_of_m617": "-0.241335188453228318",
            "m1102_baseline_cycles": 763908050,
            "m1102_raw_cpu_speedup_x": "1.7591725401987818",
            "m623_leakage_already_on_m1102_cycles": False}, "cycle promotion")
    require(value["baseline_separation"] == {
            "m1102_speedup_denominator": "strongest_zero_or_same_coordinate_bit",
            "m1102_baseline_parent_accesses": 0,
            "m623_energy_ablation_denominator": "m504_all_write_same_candidate_mechanism",
            "m623_all_write_parent_writes_s10": 218444544,
            "may_claim_candidate_vs_zero_or_bit_energy_reduction": False,
            "may_merge_1p759x_and_38p2283pct_as_one_efficiency_pair": False},
            "denominator merge/promotion")
    require(value["claim_boundary"] == {"source_only": True,
            "new_dynamic_energy": False, "new_leakage_energy": False,
            "candidate_vs_baseline_energy": False, "c1_total_energy": False,
            "system_energy": False, "rtl_or_eda": False,
            "paper_ppa_ready": False,
            "allowed_use": "machine-readable provenance bridge for two separately labelled component-table rows"},
            "energy/system promotion")


def attack_projection(base: dict[str, Any]) -> list[str]:
    attacks = []
    cases = [
        ("checkpoint", lambda x: x["population"].__setitem__("checkpoint", "final")),
        ("population_samples", lambda x: x["population"].__setitem__("samples", 30)),
        ("population_tasks", lambda x: x["population"].__setitem__("tasks", 1)),
        ("population_operator", lambda x: x["population"].__setitem__("operators", "full network")),
        ("population_identity", lambda x: x["population"].__setitem__("identical_between_m1102_and_m623", False)),
        ("read_count", lambda x: x["candidate_parent_identity"].__setitem__("reads_s10", 1)),
        ("write_count", lambda x: x["candidate_parent_identity"].__setitem__("writes_s10", 1)),
        ("forward_count", lambda x: x["candidate_parent_identity"].__setitem__("forwards_s10", 1)),
        ("candidate_cycle", lambda x: x["cycle_binding"].__setitem__("m1102_candidate_cycles", 1)),
        ("cycle_delta", lambda x: x["cycle_binding"].__setitem__("difference_cycles_m1102_minus_m617", 0)),
        ("speedup", lambda x: x["cycle_binding"].__setitem__("m1102_raw_cpu_speedup_x", "2.0")),
        ("leakage_rebind", lambda x: x["cycle_binding"].__setitem__("m623_leakage_already_on_m1102_cycles", True)),
        ("denominator_merge", lambda x: x["baseline_separation"].__setitem__("m623_energy_ablation_denominator", "strongest_zero")),
        ("merge_efficiency", lambda x: x["baseline_separation"].__setitem__("may_merge_1p759x_and_38p2283pct_as_one_efficiency_pair", True)),
        ("energy_upgrade", lambda x: x["claim_boundary"].__setitem__("new_dynamic_energy", True)),
        ("total_energy_upgrade", lambda x: x["claim_boundary"].__setitem__("c1_total_energy", True)),
        ("system_upgrade", lambda x: x["claim_boundary"].__setitem__("system_energy", True)),
        ("ppa_upgrade", lambda x: x["claim_boundary"].__setitem__("paper_ppa_ready", True)),
    ]
    for name, mutate in cases:
        value = copy.deepcopy(base); mutate(value)
        try:
            validate_projection(value)
        except HammerError:
            attacks.append(name)
        else:
            raise HammerError("projection attack escaped: " + name)
    return attacks


def write_outputs(review: dict[str, Any], checks: dict[str, Any]) -> None:
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M1280 independent hammer of M1277\n\n"
        "**PASS 100/100, P0/P1/P2 = 0/0/0.** M1102 and M623 bind the same ep35 "
        "ten-sample parent-access vector, while their denominators remain different and nonmergeable.\n\n"
        "Recomputed: 131,926,088 reads; 79,581,608 writes; 13,717,024 forwards; "
        "434,242,823 candidate cycles; 763,908,050 zero/bit cycles; M617 old candidate "
        "435,293,339 cycles; delta -1,050,516 (-0.241335188453228318%).\n\n"
        "The 1.7591725401987818x raw-CPU speedup is versus zero/bit with zero parent "
        "accesses. The 38.22830791892194% generated-macro component reduction is versus "
        "M504 all-write (218,444,544 writes). They cannot be combined into one energy, "
        "efficiency, system, or headline claim. Eighteen independent promotion attacks "
        "were rejected. No canonical input was modified and no EDA/GPU/remote work ran.\n",
        encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        "PASS_M1280_M1277_PARENT_COMPONENT_REBIND_HAMMER__SOURCE_ONLY_NO_ENERGY_PROMOTION\n",
        encoding="utf-8")
    members = sorted(path for path in OUT.iterdir()
                     if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = OUT / "SHA256SUMS"
    manifest.write_text("".join(sha(path) + "  " + path.name + "\n" for path in members),
                        encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    before = {str(path): sha(path) for path in EXPECTED}
    for path, digest in EXPECTED.items(): regular(path, digest)
    require(set(path.name for path in AUTHOR.iterdir()) ==
            {"review.json", "review.md", "selftest.json", "RUN_COMPLETE.txt"},
            "M1277 author receipt member set drift")
    contract = strict(CONTRACT); author = strict(AUTHOR / "review.json")
    require(contract["status"] == "SOURCE_ONLY_MACHINE_READABLE_IDENTITY_BRIDGE__NO_NEW_ENERGY_RESULT" and
            author["status"] == "PASS_M1277_ADDITIVE_SOURCE_RECEIPT_AND_FAIL_CLOSED_SELFTEST__NO_NEW_ENERGY_RESULT",
            "contract/receipt status drift")
    module = load_source()
    m1102, m617, m1114, m623 = module.verify_authorities()
    verify_flat_manifest(module.M1114_ROOT, *module.M1114_ID)
    verify_flat_manifest(module.M623_ROOT, *module.M623_ID)
    recompute = independent_recompute(module, m1102, m617, m1114, m623)
    binding = module.build_binding(m1102, m617, m1114, m623)
    validate_projection(binding)
    attacks = attack_projection(binding)
    source_selftest = module.self_test()
    require(source_selftest == strict(AUTHOR / "selftest.json") and
            source_selftest["attack_cases_rejected"] == 11,
            "M1277 source selftest/receipt mismatch")
    after = {str(path): sha(path) for path in EXPECTED}
    require(before == after, "read-only authority changed")
    checks = {"schema": "m1280_m1277_parent_component_rebind_mechanical_r1_v1",
              "status": "PASS", "recompute": recompute,
              "source_attacks_rejected": 11,
              "independent_projection_attacks_rejected": len(attacks),
              "attacks": attacks, "different_denominators_nonmergeable": True,
              "canonical_inputs_modified": 0, "eda_gpu_remote_runs": 0}
    review = {"schema": "m1280_m1277_parent_component_rebind_independent_hammer_r1_v1",
              "status": "PASS_M1280_M1277_PARENT_COMPONENT_REBIND_SOURCE_ONLY",
              "verdict": "GO_SOURCE_ONLY_SEPARATELY_LABELLED_COMPONENT_ROWS",
              "score": 100, "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
              "identity": {"source_sha256": EXPECTED[SOURCE],
                           "contract_sha256": EXPECTED[CONTRACT],
                           "author_review_sha256": EXPECTED[AUTHOR / "review.json"],
                           "author_selftest_sha256": EXPECTED[AUTHOR / "selftest.json"],
                           "docs359_sha256": EXPECTED[DOCS]},
              "recompute": recompute,
              "denominator_boundary": {
                  "m1102": "zero/bit baseline with zero parent accesses",
                  "m623": "M504 all-write candidate-mechanism ablation",
                  "merge_allowed": False},
              "attacks": {"source": 11, "independent": len(attacks)},
              "admission": {"population_and_parent_identity_bridge": True,
                            "separately_labelled_component_rows": True,
                            "new_energy": False, "candidate_vs_zero_energy": False,
                            "c1_or_system_energy": False, "system_speedup": False,
                            "paper_ppa_ready": False},
              "execution": {"read_only": True, "eda": False, "gpu": False,
                            "remote": False, "docs359_modified": False}}
    write_outputs(review, checks)
    print(review["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
