#!/usr/bin/env python3
"""Read-only, different-author hammer for the M1727 decision source.

The hammer uses only synthetic arrays and temporary authority objects.  It
does not inspect M1707 capture data, execute the production analyzer, create a
release/result, or start GPU/RTL/EDA work.
"""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_simulator/scripts/analyze_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
TEST = HW / (
    "system_simulator/tests/test_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
CONTRACT = HW / (
    "contracts/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_contract_r1_20260901.json")
AUTHOR = HW / (
    "reviews/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_author_receipt_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "e0d2fc508a835b667b63a8719af3bf4ad883bfccca5b4c388f4e96ac9c6eaed9",
    TEST: "3b68aa96eba68e397a84459cfdc3199a7b8df6bf646236bf9495e0dd9137071c",
    CONTRACT: "efa110402bee236e4f1d2956ccad364a8de2c52e429d1e58a7c3dbe19f1e55f6",
    Path(str(CONTRACT) + ".sha256"):
        "34261298c564bdbd6c18126d9b0b89ee07b4c4169126c0cc10fbf629dd70e690",
    Path(str(CONTRACT) + ".sha256.seal.sha256"):
        "fe9e050debfed80c9a942f10e2889a4a6753c0c94bbf7073cc1f2607ce7ec168",
    AUTHOR / "author_receipt.json":
        "403505d340e6b89f9b0fb2c9e6fe0fd3cb63a1134245186d4ceae4565d968175",
    AUTHOR / "SHA256SUMS":
        "0e5bef732c7d719633b8d7158fbc1e74813568a1dc95405cab1f9d9922ea4014",
    AUTHOR / "SHA256SUMS.seal.sha256":
        "7294a49abab7959b531432ad33f1ff622f82b528e6ea66c9ba867e926dcbd087",
    DOCS359:
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode) and sha(path) == expected,
            "identity drift: " + str(path))


def verify_seal(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha(sums), "SHA256SUMS"], "outer seal drift")
    listed = set()
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        require(name not in listed and ".." not in Path(name).parts and
                not Path(name).is_absolute(), "unsafe manifest")
        exact(root / name, digest)
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    require(listed == actual, "manifest coverage drift")


def seal_directory(root):
    members = sorted(path.relative_to(root).as_posix()
                     for path in root.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha(root / name), name)
                            for name in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha(sums)), encoding="ascii")


def seal_file(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    sidecar.write_text("{}  {}\n".format(sha(path), path.name),
                       encoding="ascii")
    outer.write_text("{}  {}\n".format(sha(sidecar), sidecar.name),
                     encoding="ascii")


def load_target():
    spec = importlib.util.spec_from_file_location("m1728_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(module):
    return {
        "source_sha256": module.sha256(module.SOURCE),
        "test_sha256": module.sha256(module.TEST),
        "contract_sha256": module.sha256(module.CONTRACT),
        "contract_sidecar_sha256": module.sha256(module.CONTRACT_SIDECAR),
        "contract_outer_seal_file_sha256":
            module.sha256(module.CONTRACT_OUTER),
        "m1721_source_sha256": module.M1721_SOURCE_SHA256,
        "m1725_failed_review_sha256": module.M1725_REVIEW_SHA256,
    }


def make_temp_authority(module, root):
    identity = identities(module)
    review_root = root / "review"
    review_root.mkdir()
    review = {
        "schema": module.REVIEW_SCHEMA,
        "status": module.REVIEW_STATUS,
        "identity": identity,
        "authorization": {"m1729_release_may_be_created": True,
                          "analysis_run": False,
                          "capture_verify": False},
        "claim_boundary": {"paper_result": False},
    }
    (review_root / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    seal_directory(review_root)
    binding = module.validate_future_review(review_root, identity)
    release = root / "release.json"
    release_identity = dict(identity)
    release_identity.update({
        "m1728_review_sha256": binding["review_sha256"],
        "m1728_review_outer_seal_file_sha256":
            binding["outer_seal_file_sha256"],
    })
    release.write_text(json.dumps({
        "schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS,
        "identity": release_identity,
        "authorization": {"analysis_runs": 1, "capture_verifications": 1,
                          "result_publications": 1, "automatic_retry": False,
                          "gpu_runs": 0, "eda_runs": 0,
                          "all_other_runs": 0},
        "claim_boundary": {"paper_result": False},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_file(release)
    return review_root, release, identity, binding


def main():
    for path, digest in EXPECTED.items():
        exact(path, digest)
    verify_seal(AUTHOR)
    module = load_target()
    module.validate_source_contract()
    import numpy as np

    # Exact vector implementation must match a scalar persistent ordinary LRU
    # for varied sequence lengths, source universes, tile counts, and B.
    rng = np.random.RandomState(1728)
    for _ in range(2000):
        rows = int(rng.randint(1, 80))
        width = int(rng.randint(1, 28))
        capacity = int(rng.randint(1, width + 1))
        tiles = int(rng.randint(1, 7))
        active = rng.rand(rows, width) < rng.uniform(0.01, 0.99)
        got = module.exact_lru_entity_stats(active, tiles, capacity, np)
        accesses = []
        for token in range(rows):
            for tile in range(tiles):
                accesses.extend(tile * width + int(group)
                    for group in np.flatnonzero(active[token]).tolist())
        misses, _cache, hits = module._reference_lru(accesses, capacity)
        require((got["accesses"], got["misses"], got["hits"]) ==
                (len(accesses), misses, len(hits)), "vector LRU mismatch")

    # Same B must be used on both sides, even when bundling changes accesses.
    for bundle in (4, 8):
        active = rng.rand(37, 11) < 0.31
        nnz = active.astype(np.int16)
        point = module.tsbg_pair_metrics(
            active, nnz, 3, 1024, 13, bundle, np)
        require(point["ordinary_lru_capacity_rows"] == bundle and
                point["candidate_fetch_not_greater_than_baseline"],
                "same-B LRU comparator drift")

    # B4/B8 lower bounds are exact for 96 Acc24 lanes and 16 int8 sources.
    expected_cost = {
        4: (288, 1152, 16, 64, 912),
        8: (288, 2304, 16, 128, 2128),
    }
    for bundle, expected in expected_cost.items():
        row = module.tsbg_resource_account(bundle)
        got = (row["baseline_acc24_context_bytes_lower_bound"],
               row["candidate_b_token_acc24_context_bytes_lower_bound"],
               row["baseline_source_fifo_bytes_lower_bound"],
               row["candidate_b_token_source_fifo_bytes_lower_bound"],
               row["candidate_incremental_state_bytes_lower_bound"])
        require(got == expected, "TSBG B-token state lower bound drift")
        require(row["captured_weight_bytes_per_element_screening"] == 4 and
                row["hardware_weight_quantization_authority"] is False and
                row["context_tag_and_broadcast_control_priced"] is False and
                row["full_area_energy_pricing_complete"] is False and
                row["same_resource_claim"] is False and
                row["screening_only"] is True,
                "TSBG diagnostic-only resource boundary drift")

    # Random S2 reference: each dropped input magnitude must be counted once
    # for every represented output channel, including the final partial block.
    s2_cases = 0
    for output_channels in list(range(1, 41)) + [47, 48, 63, 64, 65, 95, 96]:
        blocks = (output_channels + 15) // 16
        betas = [int(value) for value in rng.randint(1, 8, size=blocks)]
        active = np.ones((3, 5), dtype=np.bool_)
        nnz = np.ones((3, 5), dtype=np.int16)
        magnitude = rng.randint(1, 3, size=(3, 5)).astype(np.int32)
        row = module.s2_fc1_pair_metrics(
            active, nnz, magnitude, output_channels, betas, 0.01, np)
        widths = [min(16, output_channels - begin)
                  for begin in range(0, output_channels, 16)]
        expected = int(magnitude.sum()) * sum(
            beta * width for beta, width in zip(betas, widths))
        require(row["sum_abs_output_code_debt"] == expected and
                row["sum_abs_output_code_debt_includes_output_channel_multiplicity"]
                is True, "S2 output multiplicity drift")
        s2_cases += 1
    exact_zero = module.s2_fc1_pair_metrics(
        np.ones((1, 1), dtype=np.bool_), np.ones((1, 1), dtype=np.int16),
        np.ones((1, 1), dtype=np.int32), 32, [1, 1], 0.0, np)
    require(exact_zero["dropped_blocks"] == 0 and
            exact_zero["sum_abs_output_code_debt"] == 0,
            "S2 epsilon-zero exact bypass drift")

    # Synthetic finalization must preserve useful diagnostics while forcing
    # every paper/cycle/energy admission gate false until resource pricing.
    class Dummy(object):
        pass
    dummy = Dummy()
    dummy.tsbg = {}
    for bundle in (4, 8):
        active = np.ones((8, 2), dtype=np.bool_)
        nnz = np.ones((8, 2), dtype=np.int16)
        metric = module.tsbg_pair_metrics(
            active, nnz, 2, 1024, 0, bundle, np)
        dummy.tsbg[(bundle, "all", "FC1_FC2")] = dict(metric)
        dummy.tsbg[(bundle, "sequence", "seq_a")] = dict(metric)
        dummy.tsbg[(bundle, "sequence", "seq_b")] = dict(metric)
    finalized = module.finalize_tsbg_rows(dummy)
    require(len(finalized) == 6, "synthetic finalized row population drift")
    for row in finalized:
        require(row.get("same_capacity_ordinary_lru_baseline") is True and
                row.get("aggregate_cycle_gate_ge_1p15", False) is False and
                row.get("sequence_cycle_gate_ge_1p05", False) is False and
                row["energy_branch_weight_reduction_ge_30pct"] is False and
                row["energy_branch_cycle_regression_le_5pct"] is False and
                row["full_area_energy_pricing_complete"] is False and
                row["same_resource_claim"] is False and
                row["screening_only"] is True,
                "TSBG admission did not fail closed")

    # Valid temporary authorities pass.  Independently resealed mutations of
    # reviewed identity and one-shot budget must both fail.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        review, release, identity, binding = make_temp_authority(
            module, root)
        module.validate_future_release(release, binding, identity)
        review_value = json.loads((review / "review.json").read_text())
        review_value["identity"]["source_sha256"] = "0" * 64
        (review / "review.json").write_text(
            json.dumps(review_value, indent=2, sort_keys=True) + "\n")
        seal_directory(review)
        try:
            module.validate_future_review(review, identity)
        except module.M1727Error:
            review_mutation_rejected = True
        else:
            review_mutation_rejected = False
        require(review_mutation_rejected,
                "resealed review identity mutation accepted")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        review, release, identity, binding = make_temp_authority(
            module, root)
        release_value = json.loads(release.read_text())
        release_value["authorization"]["analysis_runs"] = 2
        release.write_text(
            json.dumps(release_value, indent=2, sort_keys=True) + "\n")
        seal_file(release)
        try:
            module.validate_future_release(release, binding, identity)
        except module.M1727Error:
            release_mutation_rejected = True
        else:
            release_mutation_rejected = False
        require(release_mutation_rejected,
                "resealed release budget mutation accepted")

    # Actual production call, with the review directory incomplete and no
    # M1729 release, must stop before a capture-verification sentinel.
    class CaptureTouched(Exception):
        pass
    touched = [False]
    old_capture = module.BASE.verify_capture_identity
    def capture_sentinel(_root):
        touched[0] = True
        raise CaptureTouched("capture must not be touched")
    module.BASE.verify_capture_identity = capture_sentinel
    try:
        try:
            module.run_analysis()
        except (module.M1727Error, OSError):
            unauthorized_rejected = True
        except CaptureTouched:
            unauthorized_rejected = False
        else:
            unauthorized_rejected = False
    finally:
        module.BASE.verify_capture_identity = old_capture
    require(unauthorized_rejected and not touched[0],
            "production authority did not precede capture")
    require(not os.path.lexists(str(module.RESULT)) and
            not os.path.lexists(str(module.WORK)),
            "production namespace changed during source hammer")

    source_text = SOURCE.read_text(encoding="utf-8")
    body = source_text[source_text.index("def run_analysis():"):
                       source_text.index("def source_self_check():")]
    require(body.index("verify_analysis_authority()") <
            body.index("os.path.lexists") and
            "verify_capture_identity" not in body,
            "static authority-before-capture order drift")

    print(json.dumps({
        "status": "PASS_M1728_M1727_SOURCE_HAMMER__M1729_RELEASE_MAY_BE_CREATED",
        "score": 99,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "vector_lru_random_cases": 2000,
        "vector_lru_equal_to_scalar": True,
        "s2_output_multiplicity_cases": s2_cases,
        "s2_epsilon_zero_exact": True,
        "tsbg_b4_b8_state_lower_bound_verified": True,
        "all_admission_forced_false": True,
        "review_identity_mutation_rejected": review_mutation_rejected,
        "release_budget_mutation_rejected": release_mutation_rejected,
        "unauthorized_capture_touched": touched[0],
        "capture_runs": 0,
        "capture_verifications": 0,
        "analysis_runs": 0,
        "result_writes": 0,
        "gpu_runs": 0,
        "eda_runs": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
