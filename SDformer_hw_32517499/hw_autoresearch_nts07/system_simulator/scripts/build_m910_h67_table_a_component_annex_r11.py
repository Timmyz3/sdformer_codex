#!/usr/bin/env python3
"""Validate the additive M910 component annex over canonical-zero Table-A r10.

M903 is real native Synopsys component evidence, but it is not the ten-operator,
17-macro, PT/PTPX-backed full-system evidence required by M698.  This builder
therefore admits one strongly typed component row while preserving zero
full-system Table-A rows and every M903 false-claim boundary.
"""

import argparse
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path


class AnnexError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
M698_BUILDER = HW_ROOT / "system_simulator/scripts/build_m698_h67_paper_metric_registry_r10.py"
M698_BUILDER_SHA256 = "81fdc6e28e3940652f9afa65780d7539fde91d26fdcb6bef49cef9f6a260849e"
M698_CONFIG = HW_ROOT / "system_simulator/config/m698_h67_paper_metric_registry_r10_20260828.json"
M698_CONFIG_SHA256 = "6d9dedb378acfc43330a09315274e4cbe372c2abf1b9749916b606261ab2e5a3"
M706_ROOT = HW_ROOT / "reviews/m706_m698_table_a_registry_r10_fresh_hammer_r1_20260828"
M706_REVIEW_SHA256 = "a1b109235cea7af04a63c88001290d9e785935e77aa1e65f10834c08b6eb8b16"
M706_MANIFEST_SHA256 = "b3aee3e711c99892d3ec13d76010c333072ff7374ebdb66dee6f0885cc0371d9"
M706_OUTER_SHA256 = "960c816fa7ac3b6b47236e3457927fc42a54c1b882e390cb00c05e187524dc73"
M903_ROOT = HW_ROOT / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M903_REVIEW_SHA256 = "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a"
M903_MANIFEST_SHA256 = "e99268c516969eba1cd0ae29131146dc4b5ece2d7197b10924debab0b60d9984"
M903_OUTER_SHA256 = "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m910_h67_table_a_component_annex_r11_20260829.json"
DEFAULT_CONFIG_SHA256 = "4e8ce01102d18c90ea9ed95544266ddafb4adf27b28653a2028d60365ea81d1b"
ROW_ID = "c2_typed_signed_k8_vs_equal_bandwidth_k1x8"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AnnexError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_no_duplicate_object)


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise AnnexError(label + " fields differ")


def _load_module(name, path, expected_sha):
    if _sha256(path) != expected_sha:
        raise AnnexError("sealed dependency SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise AnnexError("cannot import sealed dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M698 = _load_module("m910_sealed_m698", M698_BUILDER, M698_BUILDER_SHA256)


def _verify_double_seal(root, review_sha, manifest_sha, outer_sha):
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if (_sha256(review) != review_sha or _sha256(manifest) != manifest_sha or
            _sha256(outer) != outer_sha):
        raise AnnexError("sealed review root drift: " + str(root))
    if outer.read_text(encoding="utf-8") != manifest_sha + "  SHA256SUMS\n":
        raise AnnexError("outer seal content mismatch: " + str(root))
    listed = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        fields = raw.split("  ", 1)
        if len(fields) != 2:
            raise AnnexError("malformed manifest line: " + str(root))
        rel = fields[1][2:] if fields[1].startswith("./") else fields[1]
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise AnnexError("unsafe manifest path: " + str(root))
        if rel in listed or rel in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            raise AnnexError("manifest duplicate/self inclusion: " + rel)
        listed.add(rel)
        target = root / rel
        if not target.is_file() or target.is_symlink() or _sha256(target) != fields[0]:
            raise AnnexError("manifest member mismatch: " + str(target))
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and not path.is_symlink() and
                 path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    if listed != actual:
        raise AnnexError("manifest coverage mismatch: " + str(root))
    return _load_json(review)


def _validate_predecessor():
    if _sha256(M698_CONFIG) != M698_CONFIG_SHA256:
        raise AnnexError("M698 config drift")
    preview = M698.build(M698_CONFIG)
    if (preview["validated_production_run_count"] != 0 or
            preview["trusted_hammer_authority_count"] != 0 or
            preview["table_a_evidence_bundle_count"] != 0 or
            preview["headline_gate"]["admitted"] or
            preview["analytical_diagnostic"]["admitted"]):
        raise AnnexError("M698 predecessor is not canonical zero")
    m706 = _verify_double_seal(M706_ROOT, M706_REVIEW_SHA256,
                               M706_MANIFEST_SHA256, M706_OUTER_SHA256)
    if (m706.get("verdict") != "GO_REGISTRY_SCAFFOLDING_AND_PRODUCTION_ZERO_ONLY" or
            m706.get("score") != 96 or m706.get("severity_counts") !=
            {"P0": 0, "P1": 0, "P2": 1} or
            m706.get("admission", {}).get("table_a_ppa_row_admitted") is not False or
            m706.get("canonical", {}).get("validated_production_runs") != 0):
        raise AnnexError("M706 predecessor authority mismatch")
    return preview


def _decimal_equal(value, expected, label):
    try:
        actual = Decimal(str(value))
    except Exception:
        raise AnnexError(label + " is not decimal")
    if actual != Decimal(expected):
        raise AnnexError(label + " mismatch")


def _validate_m903_authority():
    review = _verify_double_seal(M903_ROOT, M903_REVIEW_SHA256,
                                 M903_MANIFEST_SHA256, M903_OUTER_SHA256)
    if (review.get("schema") !=
            "m903_m872_m803_c2_r16_three_axis_dc_result_hammer_v1" or
            review.get("status") !=
            "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED" or
            review.get("verdict") != "PASS" or review.get("score_out_of_100") != 100 or
            any(review.get(key) != 0 for key in ("p0_count", "p1_count", "p2_count"))):
        raise AnnexError("M903 authority verdict mismatch")
    axes = review["dc_evidence"]["axes"]
    for axis, expected in (("k1", "124620.173180"),
                           ("k8", "131086.241193"),
                           ("k1x8", "585479.153645")):
        _decimal_equal(axes[axis]["area_um2"], expected, axis + " M903 area")
        if (axes[axis]["tim209"] != 0 or axes[axis]["opt150"] != 0 or
                axes[axis]["setup_violating_paths"] != 0 or
                axes[axis]["artifact_count"] != 7):
            raise AnnexError(axis + " M903 physical gate mismatch")
    fair = review["fair_equal_bandwidth_metrics"]
    if fair["aggregate_sum_cycles"] != {"k1x8": 1945, "k8": 1913}:
        raise AnnexError("M903 cycle sums mismatch")
    _decimal_equal(fair["aggregate_equal_bandwidth_cycle_speedup_k8_vs_k1x8"],
                   "1.0167276529012024", "M903 fair speedup")
    _decimal_equal(fair["aggregate_equal_bandwidth_throughput_per_mm2_ratio_k8_vs_k1x8"],
                   "4.541077997893274", "M903 throughput/mm2")
    boundary = review["claim_boundary"]
    required_true = ("logic_only_pre_macro", "setup_area_citable",
                     "directed_component_equal_bandwidth_cycle_and_throughput_per_area_citable")
    required_false = ("system", "system_speedup", "power", "energy", "ppa",
                      "paper_ppa_ready", "headline")
    if any(boundary.get(key) is not True for key in required_true) or any(
            boundary.get(key) is not False for key in required_false):
        raise AnnexError("M903 claim boundary mismatch")
    return review


def _validate_config(config):
    _exact(config, {"schema", "date", "status", "purpose", "sealed_table_a_registry",
                    "component_rows", "admission_boundary", "protected_file"},
           "M910 config")
    if config["schema"] != "m910.h67.table_a_component_annex.r11":
        raise AnnexError("M910 schema mismatch")
    expected_registry = {
        "config_path": M698_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": M698_CONFIG_SHA256,
        "fresh_hammer_path": (M706_ROOT / "review.json").relative_to(REPO_ROOT).as_posix(),
        "fresh_hammer_sha256": M706_REVIEW_SHA256,
        "fresh_hammer_manifest_sha256": M706_MANIFEST_SHA256,
        "fresh_hammer_outer_seal_file_sha256": M706_OUTER_SHA256,
    }
    if config["sealed_table_a_registry"] != expected_registry:
        raise AnnexError("M698/M706 identity mismatch")
    if config["protected_file"] != {
            "path": DOCS359.relative_to(REPO_ROOT).as_posix(),
            "sha256": DOCS359_SHA256}:
        raise AnnexError("protected-file identity mismatch")
    rows = config["component_rows"]
    if not isinstance(rows, dict) or set(rows) != {ROW_ID}:
        raise AnnexError("component annex must contain exactly the pinned M903 row")
    row = rows[ROW_ID]
    _exact(row, {"schema", "component_id", "contribution_id", "scope",
                 "evidence_class", "authority", "dc_setup_area",
                 "directed_equal_bandwidth_metrics", "claim_boundary"},
           "M910 component row")
    if (row["schema"] != "m910.h67.production_component_row.r1" or
            row["component_id"] != "c2_typed_signed_k8_shared_acc24" or
            row["contribution_id"] != "C2" or
            row["scope"] != "five_frozen_directed_component_workloads" or
            row["evidence_class"] !=
            "NATIVE_SYNOPSYS_LOGIC_ONLY_PREMACRO_PLUS_DIRECTED_VCS"):
        raise AnnexError("component row identity mismatch")
    if row["authority"] != {
            "review_path": (M903_ROOT / "review.json").relative_to(REPO_ROOT).as_posix(),
            "review_sha256": M903_REVIEW_SHA256,
            "manifest_sha256": M903_MANIFEST_SHA256,
            "outer_seal_file_sha256": M903_OUTER_SHA256}:
        raise AnnexError("M903 row authority mismatch")
    dc = row["dc_setup_area"]
    _exact(dc, {"technology_nm", "clock_period_ns", "clock_model",
                "wireload_model", "macro_count", "axes"}, "M910 DC row")
    if (dc["technology_nm"] != 28 or dc["clock_period_ns"] != "3.000" or
            dc["clock_model"] != "ideal_clock" or
            dc["wireload_model"] != "ZeroWireload" or dc["macro_count"] != 0):
        raise AnnexError("M910 DC contract mismatch")
    expected_axes = {
        "k1": ("single_k1_diagnostic_axis", "124620.173180", "+0.0020"),
        "k8": ("typed_signed_channel_split_candidate", "131086.241193", "+0.0013"),
        "k1x8": ("equal_bandwidth_fair_baseline", "585479.153645", "+0.0012"),
    }
    if set(dc["axes"]) != set(expected_axes):
        raise AnnexError("M910 DC axes mismatch")
    for axis, expected in expected_axes.items():
        if dc["axes"][axis] != {"role": expected[0], "cell_area_um2": expected[1],
                                "minimum_setup_slack_ns": expected[2]}:
            raise AnnexError("M910 %s axis mismatch" % axis)
    fair = row["directed_equal_bandwidth_metrics"]
    if fair != {
            "comparison": "k8_vs_k1x8_equal_bandwidth_only",
            "aggregation": "sum_over_five_frozen_directed_component_workloads",
            "k8_sum_cycles": 1913, "k1x8_sum_cycles": 1945,
            "fair_cycle_speedup_x": "1.01672765",
            "fair_throughput_per_mm2_x": "4.541077998",
            "logic_cell_area_saving_percent": "77.6104"}:
        raise AnnexError("M910 fair metric row mismatch")
    expected_claims = {
        "logic_only_pre_macro": True, "directed_component": True,
        "equal_bandwidth_k8_vs_k1x8_only": True, "setup_area_citable": True,
        "component_cycle_and_throughput_per_area_citable": True,
        "hold_diagnostic_only": True, "macro_inclusive": False,
        "full_network": False, "trace_weighted": False,
        "system_speedup": False, "power": False, "energy": False,
        "ppa": False, "paper_ppa_ready": False, "paper_headline": False,
        "k8_vs_single_k1_performance_headline": False,
    }
    if row["claim_boundary"] != expected_claims:
        raise AnnexError("M910 row claim boundary mismatch")
    expected_admission = {
        "table_a_full_system_production_rows": 0,
        "table_a_full_system_ppa_admitted": False,
        "production_component_rows": 1,
        "component_setup_area_and_directed_fair_metrics_admitted": True,
        "component_rows_are_not_table_a_full_system_rows": True,
        "system_speedup_admitted": False, "power_or_energy_admitted": False,
        "paper_ppa_ready": False, "paper_headline_admitted": False,
    }
    if config["admission_boundary"] != expected_admission:
        raise AnnexError("M910 admission boundary mismatch")
    return row


def build(config_path=DEFAULT_CONFIG):
    predecessor = _validate_predecessor()
    authority = _validate_m903_authority()
    if _sha256(DOCS359) != DOCS359_SHA256:
        raise AnnexError("docs/359 SHA drift")
    path = Path(config_path)
    if path == DEFAULT_CONFIG and _sha256(path) != DEFAULT_CONFIG_SHA256:
        raise AnnexError("canonical M910 config SHA drift")
    row = _validate_config(_load_json(path))
    return {
        "schema": "m910.h67.table_a_component_annex.r11.preview",
        "status": "PASS_ONE_PRODUCTION_COMPONENT_ROW__FULL_SYSTEM_TABLE_A_ZERO",
        "full_system_table_a": predecessor["table_a"],
        "full_system_table_a_production_rows": 0,
        "component_annex": {ROW_ID: row},
        "production_component_row_count": 1,
        "source_authority": {
            "schema": authority["schema"], "score_out_of_100": 100,
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "review_sha256": M903_REVIEW_SHA256,
        },
        "system_speedup_admitted": False,
        "power_or_energy_admitted": False,
        "paper_ppa_ready": False,
        "paper_headline_admitted": False,
        "protected_file_validated": {"path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                                     "sha256": DOCS359_SHA256},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print("M910_COMPONENT_ANNEX_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M910_COMPONENT_ANNEX_PASS component_rows=1 full_system_table_a_rows=0 fair_speedup=1.01672765 throughput_per_mm2=4.541077998 system=false power=false energy=false ppa=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
