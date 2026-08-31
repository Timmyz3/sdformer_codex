#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Build fail-closed M1118 component annex r12; never a system Table-A row."""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
from pathlib import Path
import stat


class AnnexError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
HW = REPO_ROOT / "hw_autoresearch_nts07"
M910_CONFIG = HW / "system_simulator/config/m910_h67_table_a_component_annex_r11_20260829.json"
M910_CONFIG_SHA = "4e8ce01102d18c90ea9ed95544266ddafb4adf27b28653a2028d60365ea81d1b"
M910_BUILDER = HW / "system_simulator/scripts/build_m910_h67_table_a_component_annex_r11.py"
M910_BUILDER_SHA = "a854e60fafdedd7c36cdaea45710250a0065507dc0b9488e4e633b106dfc0194"
M910_TESTS_SHA = "22b3df2f902fd771a3054cb1d87e6ede617e68d8e4e08509137f29f76479dd18"
M910_CONTRACT_SHA = "a97fb69412268f8f16dda5ef17cc8d987a5c491e1094217e040f3c1da70386d4"
M910_ROOT = HW / "reviews/m910_m903_table_a_component_annex_r11_static_hammer_r1_20260829"
M910_SEAL = ("73b352a737befd98cba186fc4c813dcec8db20d95e0d19187366895739232298",
             "5e04ac222fd648cdfe52f41ad138eea43f1253cbe83f85e262943dea37039ea4",
             "e611ac036cd3fae1d01469acfbb0b374910db49dce6e95ed619a1299b61355d9")
M1114_ROOT = HW / "reviews/m1114_m1102_c1_work8_full_replay_result_hammer_r1_20260830"
M1114_SEAL = ("8ced2392215b7bd70b8afcc90efab3f6078c9b3cc9b1a9d7b0c1d5e33d36b8bc",
              "3f48f2c91e1feba599fca3eab9f3c8348ed5ca5af1d317de14dd01a548b1c1b7",
              "f423e3317825cdb02e637e70d12a9b625df2c4519a4041c3ad9b4440a65c9ef4")
M1102_ROOT = HW / "results/m1102_c1_work8_exact_1rw_full_replay_r1_20260830"
M1102_RESULT = M1102_ROOT / "m1102_c1_work8_exact_1rw_full_replay_result_r1.json"
M1102_RESULT_SHA = "a229c21b1469f2482ade412a8965e66018db1e4aaa5d434329994a0572587d91"
M1102_SEAL = ("6af45f4091ab4a88b6a60a70f4caf89ceccccee7857a7debe6d8433f9843ee12",
              "f6c9d12b105991ec4ed046e709a2b4d8d983636882cfdcebaae194bd852be96f")
M928_ROOT = HW / "reviews/m928_m917_m518_r5_fixed_dc_result_hammer_r1_20260829"
M928_SEAL = ("b74ae587cafa9670e92165437ef6e042d8f3256bb9998dc25c5a1976d0e7e1f6",
             "02e27837c9a9cab75f5d9afa45a740374321ebbd32a62ad424087907cdefcd83",
             "43e6cee08ed52c52d1e46d48afc8b6835fd735e74ce4320b671cd401cf9c17d3")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DEFAULT_CONFIG = HW / "system_simulator/config/m1118_h67_table_a_component_annex_r12_20260830.json"
DEFAULT_CONFIG_SHA = "d4be661225df58a3906f015d867bdc272de48edc6901fefd744813047c513332"
C1 = "c1_exact_1rw_product_capture_raw_cpu_same_ledger"
C2 = "c2_typed_signed_k8_vs_equal_bandwidth_k1x8"
C3 = "c3_fixed_t10_logic_only_setup_area"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AnnexError(message)


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "regular identity drift: " + str(path))


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def reject_constant(value):
    raise AnnexError("nonfinite JSON constant: " + value)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates, parse_constant=reject_constant,
                      parse_float=Decimal)


def exact(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + " fields differ")


def decimal_equal(value, expected: str, label: str) -> None:
    try:
        actual = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise AnnexError(label + " not decimal")
    require(actual == Decimal(expected), label + " mismatch")


def verify_flat(root: Path, identity, allow_legacy_pycache=False):
    review_sha, manifest_sha, outer_sha = identity
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(review, review_sha); regular(manifest, manifest_sha); regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            "outer content mismatch: " + str(root))
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2, "manifest grammar")
        rel = fields[1][2:] if fields[1].startswith("./") else fields[1]
        require(rel and not rel.startswith("/") and ".." not in Path(rel).parts and
                rel not in listed and rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
                "unsafe/duplicate manifest member")
        listed.add(rel); regular(root / rel, fields[0])
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if allow_legacy_pycache:
        actual = {rel for rel in actual if not rel.startswith("__pycache__/")}
    require(listed == actual, "flat manifest coverage mismatch: " + str(root))
    return load(review)


def verify_m1102_atomic():
    seal = M1102_ROOT / ".m1102_atomic_seal"
    manifest, outer = seal / "SHA256SUMS", seal / "SHA256SUMS.seal.sha256"
    regular(manifest, M1102_SEAL[0]); regular(outer, M1102_SEAL[1])
    require(outer.read_text(encoding="utf-8") == M1102_SEAL[0] + "  SHA256SUMS\n",
            "M1102 outer content")
    expected = {"RUN_COMPLETE.txt", "m1102_c1_work8_exact_1rw_full_replay_result_r1.json",
                "m1102_work8_domain_preflight_receipt_r1.json"}
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and fields[1] in expected and fields[1] not in listed,
                "M1102 manifest grammar/coverage")
        listed.add(fields[1]); regular(M1102_ROOT / fields[1], fields[0])
    require(listed == expected, "M1102 exact members")
    actual = {path.name for path in M1102_ROOT.iterdir() if path.name != ".m1102_atomic_seal"}
    require(actual == expected and seal.is_dir() and not seal.is_symlink(), "M1102 root coverage")
    regular(M1102_RESULT, M1102_RESULT_SHA)
    return load(M1102_RESULT)


def load_m910():
    regular(M910_CONFIG, M910_CONFIG_SHA); regular(M910_BUILDER, M910_BUILDER_SHA)
    regular(HW / "system_simulator/tests/test_m910_h67_table_a_component_annex_r11.py",
            M910_TESTS_SHA)
    regular(HW / "contracts/m910_h67_table_a_component_annex_r11_contract_r1_20260829.json",
            M910_CONTRACT_SHA)
    # M910's already-sealed independent check was historically imported under
    # Python 3.6, leaving an unsealed __pycache__.  Cache bytes are never loaded
    # here and are excluded only for this frozen predecessor; every listed
    # authority member remains exact-SHA checked.  New M1118 authorities stay
    # exact-flat and do not receive this exception.
    review = verify_flat(M910_ROOT, M910_SEAL, allow_legacy_pycache=True)
    require(review.get("status") ==
            "PASS100_ONE_PRODUCTION_COMPONENT_ROW__FULL_SYSTEM_TABLE_A_ZERO" and
            review.get("score_out_of_100") == 100 and
            review.get("admission_boundary", {}).get("full_system_table_a_production_rows") == 0,
            "M910 hammer boundary")
    spec = importlib.util.spec_from_file_location("m1118_sealed_m910", M910_BUILDER)
    require(spec is not None and spec.loader is not None, "M910 import")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    preview = module.build(M910_CONFIG)
    require(preview["production_component_row_count"] == 1 and
            preview["full_system_table_a_production_rows"] == 0 and
            preview["system_speedup_admitted"] is False, "M910 preview boundary")
    return preview


def validate_c1_authority():
    review = verify_flat(M1114_ROOT, M1114_SEAL)
    result = verify_m1102_atomic()
    require(review.get("status") == "PASS_M1114_M1102_C1_RAW_CPU_SAME_LEDGER_RESULT_HAMMER" and
            review.get("verdict") == "ADMIT_RAW_CPU_SAME_LEDGER_SPEEDUP_ONLY" and
            review.get("score") == 100 and review.get("issue_counts") ==
            {"P0": 0, "P1": 0, "P2": 0}, "M1114 verdict")
    admission = review["admission"]
    require(admission["raw_cpu_same_ledger_cycles_admitted"] is True and
            admission["raw_cpu_same_ledger_speedup_admitted"] is True and
            admission["paper_citable_with_raw_cpu_same_ledger_boundary"] is True and
            all(admission[key] is False for key in
                ("rtl_cycles_admitted", "rtl_speedup_admitted", "matched_rtl_cpu_cycles_admitted",
                 "system_or_decoder_complete_speedup_admitted", "ppa_or_energy_admitted",
                 "paper_ppa_ready")), "M1114 admission escalation")
    cycles = review["cycle_rederivation"]
    require((cycles["candidate_cycles"], cycles["strongest_zero_cycles"],
             cycles["same_coordinate_bit_cycles"]) == (434242823, 763908050, 763908050),
            "M1114 cycles")
    decimal_equal(cycles["candidate_vs_strongest_zero"], "1.7591725401987818", "C1 speedup")
    decimal_equal(cycles["candidate_vs_same_coordinate_bit"], "1.7591725401987818", "C1 bit speedup")
    capacity = review["capacity"]
    require(capacity == {"psum_bytes": 122880, "weight_bytes": 49152,
                         "parent_plus_other_bytes": 42880, "derived_total_bytes": 214912,
                         "budget_bytes": 245760, "margin_bytes": 30848,
                         "raw_cpu_capacity_ledger_arithmetic_admitted": True,
                         "physical_sram_macro_timing_or_power_admitted": False}, "M1114 capacity")
    aggregate = result["raw_cpu_model"]["aggregate"]
    require(aggregate["candidate"]["cycles"] == 434242823 and
            aggregate["strongest_zero"]["cycles"] == 763908050 and
            aggregate["same_coordinate_bit"]["cycles"] == 763908050,
            "M1102 aggregate cycles")
    require(result["claim_boundary"]["speedup_admitted"] is False and
            result["claim_boundary"]["rtl_cycles"] is False and
            result["claim_boundary"]["paper_citable"] is False,
            "M1102 pending-hammer boundary")
    return review


def validate_c3_authority():
    review = verify_flat(M928_ROOT, M928_SEAL)
    require(review.get("status") == "PASS_M928_M917_M518_R5_FIXED_LOGIC_ONLY_DC_RESULT_ADMITTED" and
            review.get("score_out_of_100") == 99 and review.get("p0_count") == 0 and
            review.get("p1_count") == 0 and review.get("p2_count") == 1,
            "M928 verdict")
    dc = review["dc_result"]
    require(dc["design"] == "m518_matched_fixed_t10_atlif" and dc["technology"] ==
            "TSMC 28 nm standard-cell library" and dc["clock_period_ns"] == 3.0 and
            dc["clock_network"] == "ideal" and dc["wireload"] == "ZeroWireload" and
            dc["logic_only"] is True and dc["macro_count"] == 0 and
            dc["cell_count"] == 71898 and dc["combinational_cell_count"] == 61325 and
            dc["sequential_cell_count"] == 10573, "M928 DC identity")
    decimal_equal(dc["cell_area_um2"], "62433.503388", "C3 area")
    decimal_equal(dc["setup"]["minimum_reported_slack_ns"], "0.0003", "C3 setup")
    boundary = review["claim_boundary"]
    require(boundary["logic_only_pre_macro_dc_setup_area_citable"] is True and
            boundary["setup_met_at_3ns_ideal_clock_zerowireload"] is True and
            all(boundary[key] is False for key in
                ("hold_closed", "sta_completed", "macro_inclusive", "power", "energy",
                 "throughput", "speedup", "system", "paper_ppa_ready", "headline")),
            "M928 claim escalation")
    return review


def validate_config(config):
    exact(config, {"schema", "date", "status", "purpose", "sealed_component_annex_r11",
                   "additive_component_rows", "admission_boundary", "protected_file"}, "M1118 config")
    require(config["schema"] == "m1118.h67.table_a_component_annex.r12" and
            config["status"] ==
            "ADDITIVE_COMPONENT_ANNEX__THREE_BOUNDED_ROWS__SYSTEM_TABLE_A_REMAINS_ZERO",
            "M1118 identity")
    expected_m910 = {"config_path": M910_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": M910_CONFIG_SHA,
        "builder_path": M910_BUILDER.relative_to(REPO_ROOT).as_posix(),
        "builder_sha256": M910_BUILDER_SHA, "tests_sha256": M910_TESTS_SHA,
        "contract_sha256": M910_CONTRACT_SHA, "hammer_review_sha256": M910_SEAL[0],
        "hammer_manifest_sha256": M910_SEAL[1], "hammer_outer_seal_file_sha256": M910_SEAL[2]}
    require(config["sealed_component_annex_r11"] == expected_m910, "M910 config authority")
    rows = config["additive_component_rows"]
    require(isinstance(rows, dict) and set(rows) == {C1, C3}, "exactly two additive rows required")
    c1, c3 = rows[C1], rows[C3]
    exact(c1, {"schema", "component_id", "contribution_id", "scope", "evidence_class",
               "authority", "raw_cpu_same_ledger_metrics", "claim_boundary"}, "C1 row")
    exact(c3, {"schema", "component_id", "contribution_id", "scope", "evidence_class",
               "authority", "dc_setup_area", "claim_boundary"}, "C3 row")
    require((c1["schema"], c1["component_id"], c1["contribution_id"], c1["scope"],
             c1["evidence_class"]) ==
            ("m1118.h67.bounded_component_row.r1", "c1_exact_1rw_product_capture_work8", "C1",
             "frozen_h67_four_bottleneck_conv_ten_sample_812160_task",
             "INDEPENDENTLY_HAMMERED_RAW_CPU_SAME_LEDGER_REPLAY"), "C1 identity")
    require(c1["authority"] == {"review_path": (M1114_ROOT / "review.json").relative_to(REPO_ROOT).as_posix(),
        "review_sha256": M1114_SEAL[0], "manifest_sha256": M1114_SEAL[1],
        "outer_seal_file_sha256": M1114_SEAL[2],
        "result_path": M1102_RESULT.relative_to(REPO_ROOT).as_posix(),
        "result_sha256": M1102_RESULT_SHA, "result_manifest_sha256": M1102_SEAL[0],
        "result_outer_seal_file_sha256": M1102_SEAL[1]}, "C1 authority")
    require(c1["raw_cpu_same_ledger_metrics"] == {"samples": 10, "tasks": 812160,
        "candidate_cycles": 434242823, "strongest_zero_cycles": 763908050,
        "same_coordinate_bit_cycles": 763908050,
        "candidate_vs_strongest_zero_x": "1.7591725401987818",
        "candidate_vs_same_coordinate_bit_x": "1.7591725401987818",
        "capacity_ledger_bytes": 214912, "capacity_budget_bytes": 245760,
        "capacity_margin_bytes": 30848}, "C1 metrics")
    require(c1["claim_boundary"] == {"raw_cpu_same_ledger_cycles": True,
        "raw_cpu_same_ledger_component_speedup": True, "raw_cpu_capacity_ledger_arithmetic": True,
        "paper_citable_with_raw_cpu_boundary": True, "rtl_cycles": False,
        "rtl_speedup": False, "mapped_gate": False,
        "physical_sram_macro_timing_or_power": False, "final_checkpoint_bound": False,
        "full_network": False, "decoder_complete": False, "system_speedup": False,
        "power": False, "energy": False, "paper_ppa_ready": False,
        "paper_headline": False}, "C1 claim boundary")
    require((c3["schema"], c3["component_id"], c3["contribution_id"], c3["scope"],
             c3["evidence_class"]) ==
            ("m1118.h67.bounded_component_row.r1", "c3_fixed_t10_atlif", "C3",
             "fixed_t10_component_logic_only_pre_macro_dc",
             "INDEPENDENTLY_HAMMERED_NATIVE_SYNOPSYS_LOGIC_ONLY_PREMACRO_DC"), "C3 identity")
    require(c3["authority"] == {"review_path": (M928_ROOT / "review.json").relative_to(REPO_ROOT).as_posix(),
        "review_sha256": M928_SEAL[0], "manifest_sha256": M928_SEAL[1],
        "outer_seal_file_sha256": M928_SEAL[2]}, "C3 authority")
    require(c3["dc_setup_area"] == {"technology_nm": 28, "clock_period_ns": "3.000",
        "clock_model": "ideal_clock", "wireload_model": "ZeroWireload", "macro_count": 0,
        "cell_area_um2": "62433.503388", "cell_count": 71898,
        "combinational_cell_count": 61325, "sequential_cell_count": 10573,
        "minimum_reported_setup_slack_ns": "+0.0003"}, "C3 DC metrics")
    require(c3["claim_boundary"] == {"fixed_t10_exact_component": True,
        "logic_only_pre_macro": True, "setup_area_citable": True,
        "setup_met_at_3ns_ideal_clock_zerowireload": True, "hold_closed": False,
        "pt_sta_completed": False, "macro_inclusive": False, "throughput": False,
        "speedup": False, "system": False, "power": False, "energy": False,
        "final_checkpoint_activity_bound": False, "paper_ppa_ready": False,
        "paper_headline": False}, "C3 claim boundary")
    require(config["admission_boundary"] == {"table_a_full_system_production_rows": 0,
        "table_a_full_system_ppa_admitted": False, "component_annex_rows_total": 3,
        "inherited_component_rows": 1, "additive_component_rows": 2,
        "raw_cpu_component_rows": 1, "logic_only_pre_macro_component_rows": 2,
        "component_rows_are_not_table_a_full_system_rows": True,
        "system_speedup_admitted": False, "power_or_energy_admitted": False,
        "final_checkpoint_bound": False, "paper_ppa_ready": False,
        "paper_headline_admitted": False}, "M1118 admission boundary")
    require(config["protected_file"] == {"path": DOCS359.relative_to(REPO_ROOT).as_posix(),
        "sha256": DOCS359_SHA}, "protected-file identity")
    return c1, c3


def build(config_path=DEFAULT_CONFIG):
    # Parse and structurally reject caller mutations before touching sealed
    # authorities.  This also guarantees duplicate-key/NaN failures originate
    # at the untrusted boundary.
    path = Path(config_path)
    if path.resolve() == DEFAULT_CONFIG.resolve():
        regular(path, DEFAULT_CONFIG_SHA)
    c1, c3 = validate_config(load(path))
    predecessor = load_m910(); c1_authority = validate_c1_authority(); c3_authority = validate_c3_authority()
    regular(DOCS359, DOCS359_SHA)
    inherited = copy.deepcopy(predecessor["component_annex"])
    require(set(inherited) == {C2}, "M910 inherited row count")
    rows = {C2: inherited[C2], C1: copy.deepcopy(c1), C3: copy.deepcopy(c3)}
    require(len(rows) == 3, "component row count")
    return {"schema": "m1118.h67.table_a_component_annex.r12.preview",
        "status": "PASS_THREE_BOUNDED_COMPONENT_ROWS__FULL_SYSTEM_TABLE_A_ZERO",
        "full_system_table_a": predecessor["full_system_table_a"],
        "full_system_table_a_production_rows": 0, "component_annex": rows,
        "component_annex_row_count": 3,
        "authority_summary": {"c1": c1_authority["status"],
            "c2": "PASS100_ONE_PRODUCTION_COMPONENT_ROW__FULL_SYSTEM_TABLE_A_ZERO",
            "c3": c3_authority["status"]},
        "system_speedup_admitted": False, "power_or_energy_admitted": False,
        "final_checkpoint_bound": False, "paper_ppa_ready": False,
        "paper_headline_admitted": False,
        "protected_file_validated": {"path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                                      "sha256": DOCS359_SHA}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true"); args = parser.parse_args()
    try:
        result = build(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print("M1118_COMPONENT_ANNEX_FAIL: " + str(exc)); return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    else:
        print("M1118_COMPONENT_ANNEX_PASS rows=3 full_system_rows=0 C1=raw_cpu_only C2=bounded_component C3=setup_area_only system=false final_ckpt=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
