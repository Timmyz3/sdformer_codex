#!/usr/bin/env python3
"""CPU-only author check for the M1761 one-attempt release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
CHECKER = HW / "system_simulator/scripts/check_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
TEST = HW / "system_simulator/tests/test_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_author_receipt_r1_20260901"
REVIEW = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
RELEASE = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    RUNNER: "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e",
    CHECKER: "b9bb417be8786b69a3d476d75b2a49c0a99b46518ed76b0bad9a572937160312",
    TEST: "6d4a48e14d89c31ecb80be2a009ca469be990920ad6280e8c9440340e9261994",
    CONTRACT: "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21",
    Path(str(CONTRACT) + ".sha256"): "ec8dcccf92d8979b674008ca83edff4ae98f87e127e3212a979801853ac27092",
    Path(str(CONTRACT) + ".sha256.seal.sha256"): "2b7510d270632a1989366870abdb68e1bcb3470e665c486b89be6d4e3f50b8d9",
    AUTHOR / "author_receipt.json": "1557450905ced1464cd23f792e68b7964c80b81bb06cc9e26e207400a1ad4fe8",
    AUTHOR / "SHA256SUMS": "031f3d9c312180fc07fb6cf485f8cdd64b42ec302c07d8bf6df4f115896fdf36",
    AUTHOR / "SHA256SUMS.seal.sha256": "d85a20af6b8a3deaad45b87f893b87fb38b6e56f85d6aaa385b736dc19e3f962",
    REVIEW / "review.json": "987fccddbad6281bb31aa128987118ef4942e210d47201c528ab9be50055329c",
    REVIEW / "SHA256SUMS": "e8921f4612f9b0b8532b43f441ccd2b93c2600e5dca861cefcb6ef293601afcf",
    REVIEW / "SHA256SUMS.seal.sha256": "55caca70cf9670ee8e361c062f4c73e1272c399c990eb4b1e27771008f00830e",
    RELEASE: "bb5b32ead4bd2ff682abfbcedf242b645c20c89d71db0b7eeadc7c18f5191f5e",
    Path(str(RELEASE) + ".sha256"): "71b353b92b87c559b8c6501e4b8834e4c78383c5e29d1a367b1ae277423f7e3d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CLAIMS = dict((key, False) for key in (
    "vcs", "mapped_functionality", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready", "headline"))


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
    manifest = Path(root) / "SHA256SUMS"
    outer = Path(root) / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha, "seal identity")
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n", "outer content")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = Path(root) / name
        need(name not in listed and path.is_file() and not path.is_symlink()
             and sha(path) == digest, "sealed member")
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in Path(root).rglob("*")
                 if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population")


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    release_outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    need(release_outer.is_file() and not release_outer.is_symlink(), "release outer absent")
    need(Path(str(RELEASE) + ".sha256").read_text() ==
         EXPECTED[RELEASE] + "  " + RELEASE.name + "\n", "release sidecar content")
    need(release_outer.read_text() ==
         EXPECTED[Path(str(RELEASE) + ".sha256")] + "  "
         + Path(str(RELEASE) + ".sha256").name + "\n", "release outer content")
    verify_seal(AUTHOR, EXPECTED[AUTHOR / "SHA256SUMS"],
                EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_seal(REVIEW, EXPECTED[REVIEW / "SHA256SUMS"],
                EXPECTED[REVIEW / "SHA256SUMS.seal.sha256"])
    review = strict_json(REVIEW / "review.json")
    release = strict_json(RELEASE)
    need(review.get("status") ==
         "PASS_M1760_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT",
         "review status")
    need(review.get("score") == 98 and review.get("severity") ==
         {"p0": 0, "p1": 0, "p2": 1}, "review verdict")
    need(release.get("status") ==
         "AUTHORIZE_ONE_M1753_C2_THREE_AXIS_MAPPED_DIRECTED_COMPONENT_ENERGY_ATTEMPT",
         "release status")
    need(release.get("identity") == {
        "runner_sha256": EXPECTED[RUNNER],
        "source_contract_sha256": EXPECTED[CONTRACT],
        "m1760_review_sha256": EXPECTED[REVIEW / "review.json"]}, "release identity")
    need(release.get("authorization") == {"future_m1753_attempts": 1,
        "automatic_retry": False, "vcs_compiles": 3, "simv_runs": 15,
        "saif_files": 15, "ptpx_runs": 15}, "execution budget")
    geometry = release.get("execution_geometry", {})
    need(geometry.get("axes") == ["k1", "k8", "k1x8"]
         and geometry.get("cases") == [0, 1, 2, 3, 4]
         and geometry.get("accepted_sources_per_axis") == 261
         and geometry.get("all_fifteen_checked_saif_before_any_ptpx") is True
         and geometry.get("partial_axis_citable") is False
         and geometry.get("automatic_retry") is False, "geometry")
    boundary = release.get("measurement_boundary", {})
    need(boundary.get("workload_class") == "DIRECTED_COMPONENT_NOT_PRODUCTION"
         and boundary.get("whole_mapped_logic_report_power") is True
         and boundary.get("logic_only_premacro") is True
         and boundary.get("trace_frame_or_system_energy") is False,
         "measurement boundary")
    need(set(boundary.get("excluded", [])) == {"weight_sram", "testbench_memory_model",
         "io_phy", "clock_tree", "postlayout_parasitics"}, "memory exclusion")
    disclosure = release.get("mandatory_joint_disclosure", {})
    need(disclosure == {"equal_bandwidth_k8_vs_k1x8_cycle_speedup": 1.0167276529012024,
        "equal_bandwidth_k8_vs_k1x8_throughput_per_mm2": 4.562720096484654,
        "same_table_and_sentence": True, "k8_vs_single_k1_headline_forbidden": True},
        "joint disclosure")
    need(release.get("claim_boundary") == CLAIMS, "claim boundary")
    launch = release.get("launch_environment_exact", {})
    need(launch.get("M1753_EXPECTED_M1761_RELEASE_SHA256") ==
         "SUPPLY_FROM_THIS_RELEASE_SHA256_SIDECAR", "self-reference policy")
    for name, digest in {
            "M1753_EXPECTED_RUNNER_SHA256": EXPECTED[RUNNER],
            "M1753_EXPECTED_SOURCE_CONTRACT_SHA256": EXPECTED[CONTRACT],
            "M1753_EXPECTED_M1760_REVIEW_SHA256": EXPECTED[REVIEW / "review.json"],
            "M1753_EXPECTED_M1760_MANIFEST_SHA256": EXPECTED[REVIEW / "SHA256SUMS"],
            "M1753_EXPECTED_M1760_OUTER_FILE_SHA256": EXPECTED[REVIEW / "SHA256SUMS.seal.sha256"]}.items():
        need(launch.get(name) == digest and re.fullmatch(r"[0-9a-f]{64}", digest),
             "launch pin: " + name)
    for path in (HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed",
                 HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901",
                 HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine"):
        need(not os.path.lexists(path), "execution namespace exists")
    print(json.dumps({"schema": "m1761_release_author_self_check_r1_v1",
        "status": "PASS_M1761_RELEASE_AUTHOR_SELF_CHECK__NO_EDA",
        "release_sha256": EXPECTED[RELEASE],
        "axes": ["k1", "k8", "k1x8"], "cases_per_axis": 5,
        "budget": release["authorization"],
        "all_15_saif_before_ptpx": True,
        "workload_class": "DIRECTED_COMPONENT_NOT_PRODUCTION",
        "whole_mapped_logic_ptpx": True,
        "memory_excluded": True,
        "joint_disclosure": disclosure,
        "eda_runs": 0, "license_queries": 0,
        "attempt_created": False, "result_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()
