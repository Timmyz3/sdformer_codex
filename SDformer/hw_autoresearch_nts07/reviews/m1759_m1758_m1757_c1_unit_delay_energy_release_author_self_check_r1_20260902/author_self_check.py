#!/usr/bin/env python3
"""CPU-only author check for the M1759 one-campaign C1 release."""
import hashlib
import json
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1757_m1701_c1_unit_delay_functional_saif_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1757_c1_m1701_unit_delay_functional_saif_energy_source.py"
CONTRACT = HW / "contracts/m1757_m1701_c1_unit_delay_functional_saif_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1757_m1701_c1_unit_delay_functional_saif_energy_source_author_receipt_r1_20260901"
REVIEW = HW / "reviews/m1758_m1757_m1701_c1_unit_delay_functional_saif_energy_source_hammer_r1_20260901"
RELEASE = HW / "contracts/m1759_m1758_m1757_m1701_c1_unit_delay_functional_saif_energy_launch_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "b7df92c54d20af892264044d9882bbdf43de1cfa79f21d57d11cbb0d613876ea",
    CHECKER: "c1b26c42896822b9903061525636aa2f36ea7a6651c1cba0e14c594808861a7b",
    TEST: "79bd4dfdcfba09e4b6b88f70cdb26041e510504e437d9eab57563925d81d93e2",
    CONTRACT: "505e3f248fee60b757dfea62516d073e01442daf2ad00e3a3b0d350e7cc09a51",
    Path(str(CONTRACT) + ".sha256"): "249443c8828b2baa9a3fe11af8a6d00ed0f9516250305167324c40825700ee90",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "8881ac1225cc05b186a18d175e3547fab43436847e8fa24cd9fb8b05b214f6bf",
    AUTHOR / "receipt.json": "0de5d83ebcd26f94785e732a8ca7564fe6076053b59811b9b6dbb8f114cda8ab",
    AUTHOR / "SHA256SUMS": "f01196bbd36119118851012a66e359baccba4b61a04d4b1fd4c175b93ac4d6b1",
    AUTHOR / "SHA256SUMS.seal.sha256": "23107601acfb6faf8f560b89b0e13f2a0726743f7e036ac28658bdfcc1524a2c",
    REVIEW / "review.json": "01ed6151b603a152d6d40c5547fa2a4030149d8976882d4e319164159b5b7ba4",
    REVIEW / "SHA256SUMS": "584f8180af7816598f3dca13e76626705fb93f3f8db703302e6cc93f0b5699ef",
    REVIEW / "SHA256SUMS.seal.sha256": "793e9c890d9582973fd50ad2708a17214ad542b2a9688f4d70140bcffdb51f0e",
    RELEASE: "c5fca9c2e3a05ad48460baec52403da10c741bcb7071012649fda61ea181d190",
    Path(str(RELEASE) + ".sha256"): "a8110cb46fc601bba632c1d82ccc450066aed5416b35fd24f402a84a764fc4ef",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "73e64711f6ed462da269477d925ba8075c433676ca026093bfc8a74f9d49899e",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CLAIMS = {
    "launch_executed": False,
    "mapped_vcs": False,
    "production_saif": False,
    "ptpx": False,
    "component_power": False,
    "component_energy": False,
    "total_c1_energy": False,
    "energy_per_frame": False,
    "performance": False,
    "system_speedup": False,
    "paper_ppa_ready": False,
    "headline": False,
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
         "seal identity")
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n", "outer content")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest member")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "sealed member")
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and
                 path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population")


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    release_sum = Path(str(RELEASE) + ".sha256")
    release_outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    need(release_sum.read_text() == EXPECTED[RELEASE] + "  " + RELEASE.name + "\n",
         "release sidecar content")
    need(release_outer.read_text() == EXPECTED[release_sum] + "  "
         + release_sum.name + "\n", "release outer content")
    verify_seal(AUTHOR, EXPECTED[AUTHOR / "SHA256SUMS"],
                EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_seal(REVIEW, EXPECTED[REVIEW / "SHA256SUMS"],
                EXPECTED[REVIEW / "SHA256SUMS.seal.sha256"])

    author = strict_json(AUTHOR / "receipt.json")
    review = strict_json(REVIEW / "review.json")
    release = strict_json(RELEASE)
    need(author.get("status") ==
         "PASS_M1757_UNIT_DELAY_FUNCTIONAL_SOURCE_SELF_CHECK__SOURCE_ONLY__REQUEST_M1758_REVIEW",
         "author status")
    need(review.get("status") ==
         "PASS_M1758_M1757_C1_UNIT_DELAY_FUNCTIONAL_SAIF_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_CAMPAIGN",
         "review status")
    need(review.get("score_over_100") == 99 and
         [review.get("p0_count"), review.get("p1_count"), review.get("p2_count")] == [0, 0, 0],
         "review verdict")
    need(release.get("status") ==
         "AUTHORIZE_ONE_M1757_C1_UNIT_DELAY_FUNCTIONAL_SAIF_ENERGY_CAMPAIGN",
         "release status")
    need(release.get("identity") == {
        "runner_sha256": EXPECTED[RUNNER],
        "source_contract_sha256": EXPECTED[CONTRACT],
        "m1758_review_sha256": EXPECTED[REVIEW / "review.json"]},
        "release identity")
    need(release.get("authorization") == {
        "future_m1757_campaigns": 1,
        "automatic_retry": False,
        "vcs_compiles": 1,
        "simv_runs": 1,
        "saif_files": 1,
        "ptpx_runs": 1,
        "alternate_workload_after_attempt": False}, "execution budget")

    geometry = release.get("execution_geometry", {})
    need(geometry == {
        "fresh_compile": True,
        "old_binary_or_csrc_reuse": False,
        "unit_delay_define_count": 1,
        "mapped_top": "m935_m912_three_stage_exact_parent_match_product_capture_island",
        "directed_rows": 64,
        "clock_period_ns": 3.0,
        "support_tiers": [1, 2, 4],
        "automatic_retry": False}, "execution geometry")
    timing = release.get("functional_and_timing_boundary", {})
    need(timing.get("gate_activity_mode") == "UNIT_DELAY_functional"
         and timing.get("functional_gate_saif") is True
         and timing.get("timing_simulation") is False
         and timing.get("timing_signoff_from_simulation") is False
         and timing.get("timing_authority") == "independent_M1740_PrimeTime_prelayout"
         and timing.get("independent_pt_setup_wns_ns") == 0.027871
         and timing.get("independent_pt_hold_wns_ns") == 0.001827
         and timing.get("paper_ppa_ready") is False, "functional/timing boundary")
    power = release.get("power_measurement_boundary", {})
    need(power.get("mapped_top_included") is True
         and power.get("sram_liberty_macro_count") == 9
         and power.get("primary_report") ==
             "whole mapped C1 top including exactly nine linked SRAM Liberty macro instances"
         and power.get("corner_classification") == "mixed_corner_component_estimate"
         and power.get("single_corner_signoff") is False
         and power.get("frame_or_system_energy") is False, "power boundary")
    datasheet = release.get("datasheet_sram_boundary", {})
    need(datasheet == {
        "role": "separate_alternative_sensitivity_only",
        "included_in_primary_whole_top_ptpx": False,
        "added_to_whole_top_ptpx": False,
        "combined_energy_claim_forbidden": True,
        "signoff_role": False}, "datasheet boundary")
    need(release.get("claim_boundary") == CLAIMS, "claim boundary")
    need(release.get("release_execution") == {
        "source_only": True,
        "license_queries": 0,
        "vcs_compiles": 0,
        "simv_runs": 0,
        "saif_files": 0,
        "ptpx_runs": 0,
        "eda_runs": 0,
        "attempt_created": False,
        "result_created": False}, "release execution")

    launch = release.get("launch_environment_exact", {})
    need(launch.get("M1757_EXPECTED_M1759_RELEASE_SHA256") ==
         "SUPPLY_FROM_THIS_RELEASE_SHA256_SIDECAR", "self-reference policy")
    for name, digest in {
            "M1757_EXPECTED_RUNNER_SHA256": EXPECTED[RUNNER],
            "M1757_EXPECTED_SOURCE_CONTRACT_SHA256": EXPECTED[CONTRACT],
            "M1757_EXPECTED_M1758_REVIEW_SHA256": EXPECTED[REVIEW / "review.json"],
            "M1757_EXPECTED_M1758_MANIFEST_SHA256": EXPECTED[REVIEW / "SHA256SUMS"],
            "M1757_EXPECTED_M1758_OUTER_FILE_SHA256": EXPECTED[REVIEW / "SHA256SUMS.seal.sha256"]}.items():
        need(launch.get(name) == digest and re.fullmatch(r"[0-9a-f]{64}", digest),
             "launch pin: " + name)

    for path in (
            HW / "results/.m1757_c1_unit_delay_functional_saif_energy_attempt_consumed",
            HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901",
            HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901.failed_or_incomplete.quarantine",
            HW / "results/m1757_c1_unit_delay_functional_saif_energy_r1_20260901.private_build.unsealed_do_not_cite"):
        need(not os.path.lexists(str(path)), "execution namespace exists")

    print(json.dumps({
        "schema": "m1759_release_author_self_check_r1_v1",
        "status": "PASS_M1759_C1_RELEASE_AUTHOR_SELF_CHECK__NO_EDA",
        "release_sha256": EXPECTED[RELEASE],
        "budget": release["authorization"],
        "gate_activity_mode": "UNIT_DELAY_functional",
        "independent_pt_timing": True,
        "whole_mapped_top": True,
        "sram_liberty_macro_count": 9,
        "datasheet_sram_alternative_only": True,
        "mixed_corner_single_corner_signoff": False,
        "eda_runs": 0,
        "license_queries": 0,
        "attempt_created": False,
        "result_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()
