#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author static hammer for the M1118 Table-A component annex."""
from __future__ import annotations

import copy
from decimal import Decimal, getcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


sys.dont_write_bytecode = True
getcontext().prec = 60
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = ROOT / "system_simulator/config/m1118_h67_table_a_component_annex_r12_20260830.json"
BUILDER = ROOT / "system_simulator/scripts/build_m1118_h67_table_a_component_annex_r12.py"
TESTS = ROOT / "system_simulator/tests/test_m1118_h67_table_a_component_annex_r12.py"
CONTRACT = ROOT / "contracts/m1118_h67_table_a_component_annex_r12_contract_r1_20260830.json"
AUTHOR = ROOT / "reviews/m1118_h67_table_a_component_annex_r12_author_handoff_r1_20260830"
M910 = ROOT / "reviews/m910_m903_table_a_component_annex_r11_static_hammer_r1_20260829"
M903 = ROOT / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M1114 = ROOT / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830"
M1102 = ROOT / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M928 = ROOT / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"
PYTHON = "/opt/anaconda3/envs/pytorch310/bin/python3.10"

EXPECTED = {
    "config": "d4be661225df58a3906f015d867bdc272de48edc6901fefd744813047c513332",
    "builder": "1e72ccebdf8f0ee78a5a885b71fa873e02076b22854657ce69efbf5e4c942c78",
    "tests": "ca123d01123e8cf8591a891c4c8675a456b7b4fa09c412cc082ebd6c7e3df5ae",
    "contract": "e747724a6b5c0533692086eb49a9471cbaa681af04adb5f7df55f2ab2b4e1f11",
    "contract_side": "4465225f2d7c51ab510ab107b4cc82ebb78e3dc499ca51931c3f877b5f0f4439",
    "contract_outer": "0e3507fdeb00e4709eb8ff8de4ed9e7cffd7e17059f29a472f3bcfa1a93e9733",
    "author_outer": "7037fe329c0df2fd8e374bb56ab3bcacab6a3cad1c761175c891ea90782d0d2a",
    "m910_outer": "e611ac036cd3fae1d01469acfbb0b374910db49dce6e95ed619a1299b61355d9",
    "m903_outer": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    "m1114_outer": "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4",
    "m1102_manifest": "6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12",
    "m1102_outer": "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f",
    "m928_outer": "43e6cee08ed52c52d1e46d48afc8b6835fd735e74ce4320b671cd401cf9c17d3",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def strict_load(path: Path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                      parse_float=Decimal)


def verify_flat(directory: Path, expected_outer: str, legacy_pycache: bool = False) -> dict:
    require(directory.exists() and not directory.is_symlink() and stat.S_ISDIR(directory.lstat().st_mode), f"sealed dir {directory}")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer) and sha(outer) == expected_outer, f"seal metadata {directory}")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer content")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]), "manifest grammar")
        name = fields[1].lstrip("*"); rel = Path(name.lstrip("./"))
        name = rel.as_posix()
        require(name and not rel.is_absolute() and ".." not in rel.parts and name not in listed, "manifest path")
        listed[name] = fields[0]
    actual = set(); ignored = []
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        if legacy_pycache and name.startswith("__pycache__/") and stat.S_ISREG(mode) and not member.is_symlink():
            ignored.append(name); continue
        require(not stat.S_ISLNK(mode), f"live symlink {directory}/{name}")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), f"special member {name}")
    require(actual == set(listed), f"exact member coverage {directory}")
    for name, digest in listed.items():
        member = directory / name
        require(regular(member) and sha(member) == digest, f"member identity {name}")
    return {"members": len(listed), "legacy_pycache_ignored": ignored, "manifest_sha256": sha(manifest), "outer_seal_file_sha256": sha(outer)}


def verify_m1102() -> dict:
    seal = M1102 / ".m1102_atomic_seal"; manifest = seal / "SHA256SUMS"; outer = seal / "SHA256SUMS.seal.sha256"
    require(regular(manifest) and regular(outer), "M1102 seal regular")
    require(sha(manifest) == EXPECTED["m1102_manifest"] and sha(outer) == EXPECTED["m1102_outer"], "M1102 seal identity")
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "M1102 outer")
    expected = {"RUN_COMPLETE.txt", "m1102_c1_work8_exact_1rw_full_replay_result_r1.json", "m1102_work8_domain_preflight_receipt_r1.json"}
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); name = name.lstrip("*")
        require(name in expected and name not in listed, "M1102 manifest")
        listed[name] = digest; require(regular(M1102 / name) and sha(M1102 / name) == digest, "M1102 member")
    actual = {path.name for path in M1102.iterdir() if path.name != ".m1102_atomic_seal"}
    require(set(listed) == expected == actual and seal.is_dir() and not seal.is_symlink(), "M1102 exact root")
    return strict_load(M1102 / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json")


def import_builder():
    spec = importlib.util.spec_from_file_location("m1118_independent_subject", BUILDER)
    require(spec is not None and spec.loader is not None, "builder import")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def rederive(authorities: dict[str, dict]) -> dict:
    c1_review = authorities["c1_review"]; c1_result = authorities["c1_result"]
    c2_review = authorities["c2_review"]; c3_review = authorities["c3_review"]
    c1_cycles = c1_review["cycle_rederivation"]
    candidate = sum(c1_cycles["candidate_sample_cycles"]); baseline = sum(c1_cycles["strongest_zero_sample_cycles"])
    require(candidate == c1_result["raw_cpu_model"]["aggregate"]["candidate"]["cycles"] == 434242823, "C1 candidate")
    require(baseline == c1_result["raw_cpu_model"]["aggregate"]["strongest_zero"]["cycles"] == 763908050, "C1 baseline")
    c1_speedup = Decimal(baseline) / Decimal(candidate)
    capacity = c1_result["raw_cpu_model"]["capacity"]
    derived_capacity = capacity["psum"]["bytes"] + capacity["weight"]["bytes"] + capacity["parent_plus_other"]["bytes"]
    require(derived_capacity == 214912 and capacity["budget_bytes"] - derived_capacity == 30848, "C1 capacity")

    fair = c2_review["fair_equal_bandwidth_metrics"]; axes = c2_review["dc_evidence"]["axes"]
    k8 = sum(fair["frozen_directed_vcs_cycles"]["k8"]); k1x8 = sum(fair["frozen_directed_vcs_cycles"]["k1x8"])
    require((k8, k1x8) == (1913, 1945), "C2 cycles")
    c2_speedup = Decimal(k1x8) / Decimal(k8)
    k8_area = Decimal(str(axes["k8"]["area_um2"])); k1x8_area = Decimal(str(axes["k1x8"]["area_um2"]))
    throughput_area = c2_speedup * k1x8_area / k8_area
    area_saving = (Decimal(1) - k8_area / k1x8_area) * Decimal(100)

    dc = c3_review["dc_result"]
    require(dc["cell_area_um2"] == Decimal("62433.503388") and dc["clock_period_ns"] == Decimal("3.0") and dc["setup"]["minimum_reported_slack_ns"] == Decimal("0.0003") and dc["macro_count"] == 0, "C3 metrics")
    return {
        "c1": {"candidate_cycles": candidate, "baseline_cycles": baseline, "speedup_exact": str(c1_speedup), "capacity_bytes": derived_capacity, "budget_bytes": capacity["budget_bytes"], "margin_bytes": 30848},
        "c2": {"k8_cycles": k8, "k1x8_cycles": k1x8, "speedup_exact": str(c2_speedup), "throughput_per_mm2_exact": str(throughput_area), "area_saving_percent_exact": str(area_saving)},
        "c3": {"area_um2": str(dc["cell_area_um2"]), "clock_period_ns": str(dc["clock_period_ns"]), "minimum_setup_slack_ns": str(dc["setup"]["minimum_reported_slack_ns"]), "macro_count": dc["macro_count"], "hold_closed": dc["hold_diagnostic"]["closed"]},
    }


def independent_row_validator(rows: dict, metrics: dict) -> None:
    require(set(rows) == {"c1_exact_1rw_product_capture_raw_cpu_same_ledger", "c2_typed_signed_k8_vs_equal_bandwidth_k1x8", "c3_fixed_t10_logic_only_setup_area"}, "three exact rows")
    c1 = rows["c1_exact_1rw_product_capture_raw_cpu_same_ledger"]
    c2 = rows["c2_typed_signed_k8_vs_equal_bandwidth_k1x8"]
    c3 = rows["c3_fixed_t10_logic_only_setup_area"]
    require(c1["raw_cpu_same_ledger_metrics"]["candidate_cycles"] == metrics["c1"]["candidate_cycles"] and c1["raw_cpu_same_ledger_metrics"]["strongest_zero_cycles"] == metrics["c1"]["baseline_cycles"] and Decimal(c1["raw_cpu_same_ledger_metrics"]["candidate_vs_strongest_zero_x"]) == Decimal(metrics["c1"]["speedup_exact"]).quantize(Decimal("0.0000000000000001")), "C1 projected metrics")
    require(c1["raw_cpu_same_ledger_metrics"]["capacity_ledger_bytes"] == metrics["c1"]["capacity_bytes"], "C1 capacity projected")
    require(Decimal(c2["directed_equal_bandwidth_metrics"]["fair_cycle_speedup_x"]) == Decimal("1.01672765") and Decimal(metrics["c2"]["speedup_exact"]).quantize(Decimal("0.00000001")) == Decimal("1.01672765"), "C2 speedup")
    require(Decimal(c2["directed_equal_bandwidth_metrics"]["fair_throughput_per_mm2_x"]) == Decimal("4.541077998") and Decimal(metrics["c2"]["throughput_per_mm2_exact"]).quantize(Decimal("0.000000001")) == Decimal("4.541077998"), "C2 throughput/area")
    require(Decimal(c2["directed_equal_bandwidth_metrics"]["logic_cell_area_saving_percent"]) == Decimal("77.6104") and Decimal(metrics["c2"]["area_saving_percent_exact"]).quantize(Decimal("0.0001")) == Decimal("77.6104"), "C2 area saving")
    require(Decimal(c3["dc_setup_area"]["cell_area_um2"]) == Decimal(metrics["c3"]["area_um2"]) and Decimal(c3["dc_setup_area"]["minimum_reported_setup_slack_ns"]) == Decimal(metrics["c3"]["minimum_setup_slack_ns"]), "C3 projected metrics")
    require(c1["claim_boundary"]["system_speedup"] is False and c1["claim_boundary"]["rtl_speedup"] is False and c2["claim_boundary"]["system_speedup"] is False and c3["claim_boundary"]["speedup"] is False and c3["claim_boundary"]["hold_closed"] is False, "row claims")


def mutation_attacks(builder, config: dict, canonical_rows: dict, metrics: dict) -> dict:
    results = {}
    def reject_config(label, mutate):
        value = copy.deepcopy(config); mutate(value)
        with tempfile.TemporaryDirectory(prefix="m1118_mut_") as raw:
            path = Path(raw) / "config.json"; path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
            try: builder.build(path)
            except (OSError, ValueError, RuntimeError, AssertionError): results[label] = True
            else: results[label] = False
    with tempfile.TemporaryDirectory(prefix="m1118_json_") as raw:
        dup = Path(raw) / "dup.json"; dup.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        nan = Path(raw) / "nan.json"; nan.write_text('{"schema":NaN}\n', encoding="utf-8")
        for label, path in (("duplicate_key", dup), ("nan", nan)):
            try: builder.build(path)
            except (OSError, ValueError, RuntimeError, AssertionError): results[label] = True
            else: results[label] = False
    reject_config("full_system_claim_escalation", lambda v: v["admission_boundary"].__setitem__("table_a_full_system_production_rows", 1))
    reject_config("c1_rtl_claim_escalation", lambda v: v["additive_component_rows"]["c1_exact_1rw_product_capture_raw_cpu_same_ledger"]["claim_boundary"].__setitem__("rtl_speedup", True))
    reject_config("c3_power_claim_escalation", lambda v: v["additive_component_rows"]["c3_fixed_t10_logic_only_setup_area"]["claim_boundary"].__setitem__("power", True))
    reject_config("c1_authority_drift", lambda v: v["additive_component_rows"]["c1_exact_1rw_product_capture_raw_cpu_same_ledger"]["authority"].__setitem__("review_sha256", "0" * 64))
    reject_config("c3_authority_drift", lambda v: v["additive_component_rows"]["c3_fixed_t10_logic_only_setup_area"]["authority"].__setitem__("outer_seal_file_sha256", "0" * 64))
    reject_config("m910_authority_drift", lambda v: v["sealed_component_annex_r11"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64))
    reject_config("c1_metric_mutation", lambda v: v["additive_component_rows"]["c1_exact_1rw_product_capture_raw_cpu_same_ledger"]["raw_cpu_same_ledger_metrics"].__setitem__("candidate_cycles", 1))
    reject_config("c3_metric_mutation", lambda v: v["additive_component_rows"]["c3_fixed_t10_logic_only_setup_area"]["dc_setup_area"].__setitem__("cell_area_um2", "1"))
    changed = copy.deepcopy(canonical_rows)
    changed["c2_typed_signed_k8_vs_equal_bandwidth_k1x8"]["directed_equal_bandwidth_metrics"]["fair_cycle_speedup_x"] = "9.0"
    try: independent_row_validator(changed, metrics)
    except AssertionError: results["c2_metric_mutation"] = True
    else: results["c2_metric_mutation"] = False

    with tempfile.TemporaryDirectory(prefix="m1118_seal_", dir=ROOT / "reviews") as raw:
        flat = Path(raw); review = flat / "review.json"; review.write_text("{}\n", encoding="utf-8")
        manifest = flat / "SHA256SUMS"; manifest.write_text(f"{sha(review)}  review.json\n", encoding="utf-8")
        outer = flat / "SHA256SUMS.seal.sha256"; outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
        verify_flat(flat, sha(outer))
        extra = flat / "EXTRA"; extra.write_text("x\n", encoding="utf-8")
        try: verify_flat(flat, sha(outer))
        except AssertionError: results["live_seal_extra"] = True
        else: results["live_seal_extra"] = False
        extra.unlink(); real = flat / "manifest.real"; manifest.rename(real); manifest.symlink_to(real.name)
        try: verify_flat(flat, sha(outer))
        except AssertionError: results["live_manifest_symlink"] = True
        else: results["live_manifest_symlink"] = False
    require(all(results.values()), f"mutation escaped {results}")
    return results


def main() -> None:
    fixed = {"config": CONFIG, "builder": BUILDER, "tests": TESTS, "contract": CONTRACT,
             "contract_side": Path(str(CONTRACT) + ".sha256"),
             "contract_outer": Path(str(CONTRACT) + ".sha256.seal.sha256"), "docs359": DOCS359}
    for label, path in fixed.items(): require(regular(path) and sha(path) == EXPECTED[label], f"fixed {label}")
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    m910 = verify_flat(M910, EXPECTED["m910_outer"], legacy_pycache=True)
    m903 = verify_flat(M903, EXPECTED["m903_outer"])
    m1114 = verify_flat(M1114, EXPECTED["m1114_outer"])
    m928 = verify_flat(M928, EXPECTED["m928_outer"])
    c1_result = verify_m1102()
    authorities = {"c1_review": strict_load(M1114 / "review.json"), "c1_result": c1_result,
                   "c2_review": strict_load(M903 / "review.json"), "c3_review": strict_load(M928 / "review.json")}
    metrics = rederive(authorities)
    builder = import_builder(); preview = builder.build(CONFIG)
    require(preview["component_annex_row_count"] == 3 and preview["full_system_table_a_production_rows"] == 0, "3 component / 0 system")
    require(preview["system_speedup_admitted"] is False and preview["paper_ppa_ready"] is False, "preview boundary")
    independent_row_validator(preview["component_annex"], metrics)
    config = strict_load(CONFIG); attacks = mutation_attacks(builder, config, preview["component_annex"], metrics)

    env = os.environ.copy(); env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    test = subprocess.run([PYTHON, "-B", str(TESTS)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env, timeout=180, check=False)
    require(test.returncode == 0 and "Ran 21 tests" in test.stdout and "OK" in test.stdout, "author tests supplemental run")

    status = "PASS_M1118_R12_DIFFERENT_AUTHOR_STATIC_HAMMER__THREE_COMPONENT_ROWS__FULL_SYSTEM_ZERO"
    checks = {"schema": "m1118_r12_different_author_static_hammer_mechanical_v1", "status": status,
        "scope": {"static_only": True, "eda": 0, "gpu": 0, "remote": 0, "production": 0},
        "identity": {**EXPECTED, "author": author, "m910": m910, "m903": m903, "m1114": m1114, "m928": m928},
        "independent_rederivation": metrics,
        "annex": {"row_ids": sorted(preview["component_annex"]), "component_rows": 3, "full_system_rows": 0,
                  "system_speedup": False, "power_or_energy": False, "final_checkpoint_bound": False, "paper_ppa_ready": False},
        "attacks": attacks, "attacks_rejected": len(attacks),
        "supplemental_author_tests": {"return_code": test.returncode, "tests": 21, "failures": 0},
    }
    (OUT / "mechanical_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = {"schema": "m1118_r12_different_author_static_hammer_review_v1", "status": status,
        "verdict": "ADMIT_THREE_BOUNDED_COMPONENT_ANNEX_ROWS__DO_NOT_ADMIT_FULL_SYSTEM_TABLE_A_ROW",
        "score": 100, "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
        "rows": {"count": 3,
            "c1": {"citable": "raw CPU same-ledger component opportunity only", "cycles": [434242823, 763908050], "speedup_x": "1.7591725401987818", "capacity_ledger_bytes": 214912},
            "c2": {"citable": "five directed equal-bandwidth K8-vs-K1x8 logic-only component metrics", "cycles": [1913, 1945], "cycle_speedup_x": "1.01672765", "throughput_per_mm2_x": "4.541077998", "logic_area_saving_percent": "77.6104"},
            "c3": {"citable": "Fixed-T10 28-nm logic-only pre-macro DC setup/area only", "cell_area_um2": "62433.503388", "clock_period_ns": "3.000", "minimum_setup_slack_ns": "+0.0003"}},
        "full_system_table_a_production_rows": 0,
        "forbidden": ["C1 RTL/mapped/system/final-checkpoint speedup", "C1 physical SRAM macro PPA", "C2 full-network/trace-weighted/system or energy claim", "C3 hold/PT/power/energy/throughput/speedup/system claim", "multiplication of local ratios", "paper-PPA-ready or headline claim"],
        "validation": {"independent_metric_rederivation": True, "attacks_rejected": len(attacks), "author_tests_supplemental": "21/21"},
        "identity": {"config_sha256": EXPECTED["config"], "builder_sha256": EXPECTED["builder"], "tests_sha256": EXPECTED["tests"], "contract_sha256": EXPECTED["contract"], "author_handoff_outer_seal_file_sha256": EXPECTED["author_outer"], "docs359_sha256": EXPECTED["docs359"]},
        "claim_boundary": {"component_annex_citable_with_per_row_qualifiers": True, "full_system_row": False, "system_speedup": False, "decoder_complete": False, "power_or_energy": False, "final_checkpoint_bound": False, "paper_ppa_ready": False, "paper_headline": False}}
    (OUT / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(status + "\n", encoding="utf-8")
    (OUT / "STATIC_ONLY_NO_EDA.txt").write_text("Different-author static annex hammer only; no EDA, GPU, remote, or production row was run or created.\n", encoding="utf-8")
    print(f"{status} rows=3 full_system_rows=0 attacks={len(attacks)} eda=0")


if __name__ == "__main__": main()
