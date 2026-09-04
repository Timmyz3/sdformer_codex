#!/usr/bin/env python3
"""Additive M1727 successor for the M1721 TSBG/S2 decision source.

M1727 imports the exact, failed-closed M1721 source as an inert implementation
base and replaces every finding identified by the M1725 independent hammer.
The production entry point is unavailable unless the exact double-sealed M1727
contract, a future different-author M1728 source review and a future M1729
one-shot analysis release all validate *before* capture verification is touched.

TSBG retains the exact persistent same-B ordinary LRU comparator and separate
weight-fetch, compute, schedule, commit and roofline axes.  It additionally
reports the B-token Acc24/context/FIFO storage lower bound and the still-unpriced
broadcast/control state.  Therefore its 4-byte captured-weight roofline is only
a diagnostic screening result: there is no hardware weight-quantization or full
same-resource pricing authority in this source.

S2 FC1 retains real captured signed codewords and checkpoint-derived per-output-
block beta bounds.  Its sum absolute output-code debt now weights every beta by
the actual number of output channels represented by that block, including a
partial tail block.  PATCH remains blocked and FC2 remains the M1713 NO-GO.

This file is CPython-3.6 syntax compatible.  Source checks and tests do not run
the analyzer, inspect the capture, start GPU/EDA work or create a result/release.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/test_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
CONTRACT = HW / (
    "contracts/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_contract_r1_20260901.json")
CONTRACT_SIDECAR = Path(str(CONTRACT) + ".sha256")
CONTRACT_OUTER = Path(str(CONTRACT) + ".sha256.seal.sha256")
M1721_SOURCE = HW / (
    "system_simulator/scripts/analyze_m1721_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
M1725_REVIEW = HW / (
    "reviews/m1725_m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_hammer_r1_20260901/review.json")
FUTURE_REVIEW = HW / (
    "reviews/m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_hammer_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1729_m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_analysis_release_r1_20260901.json")
RESULT = HW / (
    "results/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901")
WORK = HW / (
    "results/.m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "r1_20260901.work")

M1721_SOURCE_SHA256 = (
    "7564842899716491f3d8de9b47e6b2abcf6a1a4d39c8fbe6da4e8e4206812df7")
M1725_REVIEW_SHA256 = (
    "fa6d9af98a896db910e31b97715ec5fe9a64190de3ebda573cfde3b227031554")
SCHEMA = "m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_v1"
STATUS = ("DIAGNOSTIC_SCREENING_ONLY__TSBG_B4_B8_RESOURCE_PRICE_INCOMPLETE__"
          "S2_FC1_CHANNEL_MULTIPLICITY_FIXED__NO_PAPER_RESULT")
REVIEW_SCHEMA = (
    "m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_hammer_r1_v1")
REVIEW_STATUS = (
    "PASS_M1728_M1727_SOURCE_HAMMER__M1729_RELEASE_MAY_BE_CREATED")
RELEASE_SCHEMA = (
    "m1729_m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "analysis_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1727_M1707_EP34_TSBG_S2_DECISION_ANALYSIS")

CAPTURED_WEIGHT_BYTES_PER_ELEMENT = 4
CAPTURED_SOURCE_CODE_BYTES = 1
M1727_ACC_BYTES = 3
M1727_OUTPUT_LANES = 96
M1727_SOURCE_GROUP = 16


class M1727Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1727Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1727Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1727Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def verify_sidecar(path, sidecar, outer, label):
    path = Path(path)
    sidecar = Path(sidecar)
    outer = Path(outer)
    require(sidecar.is_file() and not sidecar.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            label + " double seal missing")
    require(sidecar.read_text(encoding="ascii").split() ==
            [sha256(path), path.name], label + " sidecar drift")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sidecar), sidecar.name], label + " outer seal drift")


def verify_sealed_directory(root, label):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(),
            label + " directory missing/non-regular")
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and not sums.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            label + " seal missing")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(sums), sums.name], label + " outer seal drift")
    names = []
    for line in sums.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                label + " malformed manifest")
        digest = fields[0]
        name = fields[1].strip().lstrip("*")
        require(name and name not in names and not Path(name).is_absolute() and
                ".." not in Path(name).parts and Path(name).as_posix() == name,
                label + " unsafe manifest member")
        regular_exact(root / name, digest, label + " member " + name)
        names.append(name)
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file() and
                    path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(sorted(names) == actual, label + " manifest coverage drift")
    require("review.json" in names, label + " review.json missing")
    return {"review_sha256": sha256(root / "review.json"),
            "manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer)}


regular_exact(M1721_SOURCE, M1721_SOURCE_SHA256, "exact failed M1721 base")
_BASE_SPEC = importlib.util.spec_from_file_location("m1727_exact_m1721_base",
                                                     str(M1721_SOURCE))
require(_BASE_SPEC is not None and _BASE_SPEC.loader is not None,
        "cannot import exact M1721 base")
BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(BASE)
regular_exact(M1721_SOURCE, M1721_SOURCE_SHA256,
              "exact failed M1721 base after import")

# Read-only aliases retained for exact LRU/tree tests and future replay.
_reference_lru = BASE._reference_lru
exact_lru_entity_stats = BASE.exact_lru_entity_stats
verify_tree = BASE.verify_tree
BUNDLES = BASE.BUNDLES
S2_EPSILON_RATIO = BASE.S2_EPSILON_RATIO
M1707_RECEIPT = BASE.M1707_RECEIPT

_BASE_TSBG_PAIR_METRICS = BASE.tsbg_pair_metrics
_BASE_S2_PAIR_METRICS = BASE.s2_fc1_pair_metrics
_BASE_FINALIZE_TSBG = BASE.DecisionAccumulator.finalize_tsbg_rows
_BASE_CANONICAL_JSON_BYTES = BASE.canonical_json_bytes
_ACTIVE_AUTHORITY = None


def tsbg_pair_metrics(active_groups, nnz_by_group, output_tiles, row_bytes,
                      base_row, bundle, np):
    """Preserve exact M1721 work axes; resource cost is attached at finalize."""
    row = _BASE_TSBG_PAIR_METRICS(
        active_groups, nnz_by_group, output_tiles, row_bytes,
        base_row, bundle, np)
    require(int(row_bytes) % CAPTURED_WEIGHT_BYTES_PER_ELEMENT == 0,
            "captured 4-byte weight-row screening drift")
    return row


def s2_beta_output_channel_weight(output_channels, beta_by_output_block):
    """Sum beta(block) once for every output channel represented by it."""
    channels = int(output_channels)
    betas = [int(value) for value in beta_by_output_block]
    require(channels > 0 and betas and min(betas) > 0,
            "S2 beta/output-channel coordinate drift")
    widths = [min(BASE.S2_OUTPUT_TILE, channels - begin)
              for begin in range(0, channels, BASE.S2_OUTPUT_TILE)]
    require(len(widths) == len(betas) and min(widths) > 0,
            "S2 output-block multiplicity drift")
    return sum(beta * width for beta, width in zip(betas, widths))


def s2_fc1_pair_metrics(active_groups, nnz_by_group, abs_sum_by_group,
                        output_channels, beta_by_output_block, epsilon, np):
    """Correct sum debt using each beta's represented output multiplicity."""
    row = _BASE_S2_PAIR_METRICS(
        active_groups, nnz_by_group, abs_sum_by_group, output_channels,
        beta_by_output_block, epsilon, np)
    active = np.asarray(active_groups, dtype=np.bool_)
    magnitude = np.asarray(abs_sum_by_group, dtype=np.int32)
    betas = np.asarray(beta_by_output_block, dtype=np.int64)
    eps = float(epsilon)
    threshold = int(math.floor(
        eps * BASE.GROUP_WIDTH * 127.0 + 1.0e-12))
    dropped = active & (magnitude <= threshold)
    if eps == 0.0:
        dropped[:] = False
    weighted_beta = s2_beta_output_channel_weight(
        output_channels, betas.tolist())
    row["sum_abs_output_code_debt"] = (
        int(magnitude[dropped].sum()) * int(weighted_beta))
    row["sum_abs_output_code_debt_includes_output_channel_multiplicity"] = True
    return row


def tsbg_resource_account(bundle):
    """Return explicit lower-bound storage and unresolved pricing axes."""
    bundle = int(bundle)
    require(bundle in BUNDLES, "TSBG bundle resource coordinate drift")
    baseline_acc = M1727_OUTPUT_LANES * M1727_ACC_BYTES
    candidate_acc = bundle * baseline_acc
    baseline_fifo = M1727_SOURCE_GROUP * CAPTURED_SOURCE_CODE_BYTES
    candidate_fifo = bundle * baseline_fifo
    return {
        "baseline_acc24_context_bytes_lower_bound": baseline_acc,
        "candidate_b_token_acc24_context_bytes_lower_bound": candidate_acc,
        "baseline_source_fifo_bytes_lower_bound": baseline_fifo,
        "candidate_b_token_source_fifo_bytes_lower_bound": candidate_fifo,
        "candidate_incremental_state_bytes_lower_bound":
            candidate_acc + candidate_fifo - baseline_acc - baseline_fifo,
        "context_tag_and_broadcast_control_priced": False,
        "full_area_energy_pricing_complete": False,
        "same_resource_claim": False,
        "captured_weight_bytes_per_element_screening":
            CAPTURED_WEIGHT_BYTES_PER_ELEMENT,
        "hardware_weight_quantization_authority": False,
        "screening_only": True}


def finalize_tsbg_rows(self):
    """Attach explicit state lower bounds and fail closed on unpriced logic."""
    rows = _BASE_FINALIZE_TSBG(self)
    for row in rows:
        bundle = int(row["bundle"])
        row.update(tsbg_resource_account(bundle))
        if "aggregate_cycle_gate_ge_1p15" in row:
            row["diagnostic_aggregate_cycle_gate_ge_1p15"] = bool(
                row["aggregate_cycle_gate_ge_1p15"])
            row["aggregate_cycle_gate_ge_1p15"] = False
        if "sequence_cycle_gate_ge_1p05" in row:
            row["diagnostic_sequence_cycle_gate_ge_1p05"] = bool(
                row["sequence_cycle_gate_ge_1p05"])
            row["sequence_cycle_gate_ge_1p05"] = False
        row["diagnostic_energy_branch_weight_reduction_ge_30pct"] = bool(
            row["energy_branch_weight_reduction_ge_30pct"])
        row["diagnostic_energy_branch_cycle_regression_le_5pct"] = bool(
            row["energy_branch_cycle_regression_le_5pct"])
        row["energy_branch_weight_reduction_ge_30pct"] = False
        row["energy_branch_cycle_regression_le_5pct"] = False
    return rows


def validate_source_contract():
    regular_exact(M1721_SOURCE, M1721_SOURCE_SHA256, "exact M1721 base")
    regular_exact(M1725_REVIEW, M1725_REVIEW_SHA256, "exact M1725 failure")
    value = strict_json(CONTRACT)
    verify_sidecar(CONTRACT, CONTRACT_SIDECAR, CONTRACT_OUTER,
                   "M1727 source contract")
    require(value.get("schema") ==
            "m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source_contract_r1_v1" and
            value.get("status") ==
            "SOURCE_ONLY__M1725_SUCCESSOR__NO_CAPTURE_NO_ANALYSIS_NO_RELEASE" and
            value.get("source") == {
                "path": str(SOURCE.relative_to(ROOT)),
                "sha256": sha256(SOURCE)} and
            value.get("test") == {
                "path": str(TEST.relative_to(ROOT)),
                "sha256": sha256(TEST)} and
            value.get("predecessor", {}).get("m1721_source_sha256") ==
                M1721_SOURCE_SHA256 and
            value.get("predecessor", {}).get("m1725_failed_review_sha256") ==
                M1725_REVIEW_SHA256 and
            value.get("future_authority_paths") == {
                "m1728_review": str(FUTURE_REVIEW.relative_to(ROOT)),
                "m1729_release": str(FUTURE_RELEASE.relative_to(ROOT))} and
            value.get("authorization") == {
                "analysis_run": False, "capture_verify": False,
                "capture": False, "gpu": False, "rtl": False,
                "eda": False, "release": False, "paper_result": False} and
            value.get("claim_boundary", {}).get("paper_result") is False,
            "M1727 source contract drift")
    return value


def validate_future_review(review_root, identities):
    seal = verify_sealed_directory(review_root, "M1728 review")
    review = strict_json(Path(review_root) / "review.json")
    require(review.get("schema") == REVIEW_SCHEMA and
            review.get("status") == REVIEW_STATUS and
            review.get("identity") == identities and
            review.get("authorization") == {
                "m1729_release_may_be_created": True,
                "analysis_run": False, "capture_verify": False} and
            review.get("claim_boundary", {}).get("paper_result") is False,
            "M1728 review authority drift")
    seal["review"] = review
    return seal


def validate_future_release(release_path, review_binding, identities):
    release_path = Path(release_path)
    sidecar = Path(str(release_path) + ".sha256")
    outer = Path(str(release_path) + ".sha256.seal.sha256")
    release = strict_json(release_path)
    verify_sidecar(release_path, sidecar, outer, "M1729 release")
    expected_identity = dict(identities)
    expected_identity.update({
        "m1728_review_sha256": review_binding["review_sha256"],
        "m1728_review_outer_seal_file_sha256":
            review_binding["outer_seal_file_sha256"]})
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == expected_identity and
            release.get("authorization") == {
                "analysis_runs": 1, "capture_verifications": 1,
                "result_publications": 1, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            release.get("claim_boundary", {}).get("paper_result") is False,
            "M1729 analysis release authority drift")
    return {"release": release, "release_sha256": sha256(release_path),
            "release_sidecar_sha256": sha256(sidecar),
            "release_outer_seal_file_sha256": sha256(outer)}


def verify_analysis_authority():
    """Verify every source/review/release identity without touching capture."""
    validate_source_contract()
    identities = {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "contract_sha256": sha256(CONTRACT),
        "contract_sidecar_sha256": sha256(CONTRACT_SIDECAR),
        "contract_outer_seal_file_sha256": sha256(CONTRACT_OUTER),
        "m1721_source_sha256": M1721_SOURCE_SHA256,
        "m1725_failed_review_sha256": M1725_REVIEW_SHA256}
    review = validate_future_review(FUTURE_REVIEW, identities)
    release = validate_future_release(
        FUTURE_RELEASE, review, identities)
    return {"identities": identities,
            "m1728_review_sha256": review["review_sha256"],
            "m1728_review_outer_seal_file_sha256":
                review["outer_seal_file_sha256"],
            "m1729_release_sha256": release["release_sha256"],
            "m1729_release_outer_seal_file_sha256":
                release["release_outer_seal_file_sha256"]}


def _authority_canonical_json_bytes(value):
    """Bind the validated authority into the future decision before sealing."""
    if type(value) is dict and value.get("schema") == SCHEMA:
        require(_ACTIVE_AUTHORITY is not None,
                "M1727 result serialization lacks active authority")
        value.setdefault("identity", {}).update({
            "m1727_contract_sha256":
                _ACTIVE_AUTHORITY["identities"]["contract_sha256"],
            "m1728_review_sha256":
                _ACTIVE_AUTHORITY["m1728_review_sha256"],
            "m1729_release_sha256":
                _ACTIVE_AUTHORITY["m1729_release_sha256"]})
        value["analysis_authority"] = {
            "contract_double_sealed": True,
            "different_author_review_double_sealed": True,
            "one_shot_release_double_sealed": True,
            "capture_verified_only_after_release": True}
        value.setdefault("tsbg", {})["same_resource_screening"] = {
            "b_token_acc24_context_fifo_lower_bound_included": True,
            "broadcast_control_physical_price_complete": False,
            "captured_weight_bytes_per_element":
                CAPTURED_WEIGHT_BYTES_PER_ELEMENT,
            "hardware_weight_quantization_authority": False,
            "diagnostic_only": True}
        for row in value.get("tsbg", {}).get("decisions", []):
            row["cycle_path_admitted"] = False
            row["energy_only_path_eligible"] = False
            row["resource_pricing_complete"] = False
            row["screening_only"] = True
    return _BASE_CANONICAL_JSON_BYTES(value)


# Rebind only the imported implementation module; M1721 on disk is unchanged.
BASE.SOURCE = SOURCE
BASE.TEST = TEST
BASE.CONTRACT = CONTRACT
BASE.RESULT = RESULT
BASE.WORK = WORK
BASE.SCHEMA = SCHEMA
BASE.STATUS = STATUS
BASE.tsbg_pair_metrics = tsbg_pair_metrics
BASE.s2_fc1_pair_metrics = s2_fc1_pair_metrics
BASE.DecisionAccumulator.finalize_tsbg_rows = finalize_tsbg_rows
BASE.canonical_json_bytes = _authority_canonical_json_bytes


def run_analysis():
    """One future production run; authority is checked before capture access."""
    global _ACTIVE_AUTHORITY
    require(_ACTIVE_AUTHORITY is None, "M1727 analysis already active")
    authority = verify_analysis_authority()
    require(not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)),
            "fresh M1727 result/work namespace required")
    _ACTIVE_AUTHORITY = authority
    try:
        return BASE.run_analysis()
    finally:
        _ACTIVE_AUTHORITY = None


def source_self_check():
    BASE.verify_static_authorities()
    validate_source_contract()
    require(BUNDLES == (4, 8) and S2_EPSILON_RATIO[0] == 0.0 and
            RESULT != BASE.CAPTURE and WORK != RESULT and
            not os.path.lexists(str(RESULT)) and
            not os.path.lexists(str(WORK)),
            "M1727 coordinate/namespace drift")
    unit_debt = s2_beta_output_channel_weight(32, [1, 1])
    require(unit_debt == 32,
            "M1727 S2 channel multiplicity self-check drift")
    return {
        "status": "PASS_M1727_SOURCE_SELF_CHECK__NO_CAPTURE_NO_ANALYSIS",
        "m1721_unchanged_sha256": sha256(M1721_SOURCE),
        "m1725_failure_bound": True,
        "analysis_authority_required": ["exact_contract", "m1728_review",
                                         "m1729_release"],
        "capture_touched": False,
        "s2_32_output_unit_debt": unit_debt,
        "tsbg": {"bundles": list(BUNDLES),
            "ordinary_lru_same_B": True,
            "b_token_resource_lower_bound_included": True,
            "full_resource_pricing_complete": False,
            "captured_weight_bytes_per_element_screening": 4,
            "hardware_weight_quantization_authority": False,
            "screening_only": True},
        "analysis_executed": False, "gpu_runs": 0, "eda_runs": 0,
        "claim_boundary": {"source_only": True, "cycles": False,
            "traffic": False, "aee": False, "speedup": False,
            "energy": False, "rtl": False, "eda": False,
            "paper_result": False}}


def main(argv=None):
    parser = BASE.argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-self-check", action="store_true")
    mode.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args(argv)
    value = source_self_check() if args.source_self_check else run_analysis()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
