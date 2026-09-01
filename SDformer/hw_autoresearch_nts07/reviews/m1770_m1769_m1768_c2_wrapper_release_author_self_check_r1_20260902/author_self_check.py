#!/usr/bin/env python3
"""CPU-only author check for the M1770 one-wrapper-attempt release."""
import hashlib
import json
import os
from pathlib import Path
import re


HW = Path(__file__).resolve().parents[2]
WRAPPER = HW / "dc_handoff/scripts/run_m1768_m1753_c2_python312_wrapper_one_shot.py"
M1768_CONTRACT = HW / "contracts/m1768_m1767_m1753_c2_python312_wrapper_source_contract_r1_20260902.json"
M1768_AUTHOR = HW / "reviews/m1768_m1767_m1753_c2_python312_wrapper_source_author_receipt_r1_20260902"
M1767 = HW / "reviews/m1767_m1761_m1753_c2_python36_preparse_failure_receipt_r1_20260902"
M1769 = HW / "reviews/m1769_m1768_m1753_c2_python312_wrapper_source_hammer_r1_20260902"
M1753 = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
M1753_CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
PYTHON312 = Path("/usr/bin/python3.12")
RELEASE = HW / "contracts/m1770_m1769_m1768_m1753_c2_python312_wrapper_launch_release_r1_20260902.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    WRAPPER: "3a19b42593c22d5b756e9584e5cfd6a94fab9e03e614735f6db230fa1da4443c",
    M1768_CONTRACT: "1a5268690144aa0b61813132f292d50e48fa5a94fa808f2a4bf7c6ea5fe70f8e",
    Path(str(M1768_CONTRACT) + ".sha256"): "c192596b6b33ad8371ee03756d3df00bed7ea3f863d40a70832da339699dfa55",
    Path(str(M1768_CONTRACT) + ".sha256.seal.sha256"): "11843f5ce3c4dde3d73c860065e213dad637bc47beb7a3a987cbc1ac0a542b68",
    M1768_AUTHOR / "author_receipt.json": "80663202ba318eebb1449e5032495445eb9c8181f7763c5829c16d443428d564",
    M1768_AUTHOR / "SHA256SUMS": "52237cce7410819c2b49faf1f799b41d35ee3f94302cc36b7afb89d1a6a5d1f1",
    M1768_AUTHOR / "SHA256SUMS.seal.sha256": "3118d731fd61d19f2e94ee0156be76158abe93c49728193aa45636074e76db2f",
    M1767 / "receipt.json": "330e533d0f545439b7b0539a0c4816e8e77a6c89330620571162ec060c6b3729",
    M1767 / "SHA256SUMS": "ed9fbb6e5a3b30e77b81f74ee64861231336576ce562517a7a999f518c26d474",
    M1767 / "SHA256SUMS.seal.sha256": "80aaf88a542ed1fb9e754172d722ec8cdd7741bfe399ce02be212c30c60f2b71",
    M1769 / "review.json": "e5e34cafa0731de0bab9471bc4d1f7bf0d724ff71c25b895d67e86a33dbe623c",
    M1769 / "SHA256SUMS": "0f065fa6b8abfeac5eabc675e7fd4e03a8405598b949a393949aa4221bed6c9d",
    M1769 / "SHA256SUMS.seal.sha256": "00a90f853106238b8457fa040072c9f70b43c901ecdf2161208902c819db53cf",
    M1753: "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e",
    M1753_CONTRACT: "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21",
    M1760 / "review.json": "987fccddbad6281bb31aa128987118ef4942e210d47201c528ab9be50055329c",
    M1760 / "SHA256SUMS": "e8921f4612f9b0b8532b43f441ccd2b93c2600e5dca861cefcb6ef293601afcf",
    M1760 / "SHA256SUMS.seal.sha256": "55caca70cf9670ee8e361c062f4c73e1272c399c990eb4b1e27771008f00830e",
    M1761: "bb5b32ead4bd2ff682abfbcedf242b645c20c89d71db0b7eeadc7c18f5191f5e",
    Path(str(M1761) + ".sha256"): "71b353b92b87c559b8c6501e4b8834e4c78383c5e29d1a367b1ae277423f7e3d",
    Path(str(M1761) + ".sha256.seal.sha256"): "47df6661b49128152093bce9cfaacf9868e4288d4647011f1b654f4783095074",
    PYTHON312: "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814",
    RELEASE: "01fca60751e481f6fe4df2eb1248a973f9e52c4f7d47df20ae5a346138a4fb98",
    Path(str(RELEASE) + ".sha256"): "50b0378fcb8b045c2fccf867dd78dc138203c6677566da4833f52ad1cf262449",
    Path(str(RELEASE) + ".sha256.seal.sha256"): "6358dca6fc59e1c5cc9130968532e1687064e1c921bdc4ee477b52b6ac547b43",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
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
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha, "seal identity")
    need(outer.read_text() == manifest_sha + "  SHA256SUMS\n", "outer content")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts and name not in listed,
             "unsafe manifest")
        path = root / rel
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "sealed member")
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "sealed population")


def main():
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             "fixed identity drift: " + str(path))
    release_sum = Path(str(RELEASE) + ".sha256")
    release_outer = Path(str(RELEASE) + ".sha256.seal.sha256")
    need(release_sum.read_text() == EXPECTED[RELEASE] + "  " + RELEASE.name + "\n",
         "release sidecar")
    need(release_outer.read_text() == EXPECTED[release_sum] + "  "
         + release_sum.name + "\n", "release outer")
    verify_seal(M1767, EXPECTED[M1767 / "SHA256SUMS"],
                EXPECTED[M1767 / "SHA256SUMS.seal.sha256"])
    verify_seal(M1768_AUTHOR, EXPECTED[M1768_AUTHOR / "SHA256SUMS"],
                EXPECTED[M1768_AUTHOR / "SHA256SUMS.seal.sha256"])
    verify_seal(M1769, EXPECTED[M1769 / "SHA256SUMS"],
                EXPECTED[M1769 / "SHA256SUMS.seal.sha256"])
    verify_seal(M1760, EXPECTED[M1760 / "SHA256SUMS"],
                EXPECTED[M1760 / "SHA256SUMS.seal.sha256"])

    review = strict_json(M1769 / "review.json")
    release = strict_json(RELEASE)
    need(review.get("status") ==
         "PASS_M1769_M1768_C2_PYTHON312_WRAPPER_SOURCE_HAMMER__AUTHORIZE_ONE_WRAPPER_ATTEMPT",
         "review status")
    need(review.get("score_over_100") == 99 and
         [review.get("p0_count"), review.get("p1_count"), review.get("p2_count")] == [0, 0, 0],
         "review verdict")
    need(release.get("status") ==
         "AUTHORIZE_ONE_M1768_C2_PYTHON312_WRAPPER_ATTEMPT", "release status")
    need(release.get("identity") == {
        "wrapper_sha256": EXPECTED[WRAPPER],
        "source_contract_sha256": EXPECTED[M1768_CONTRACT],
        "m1769_review_sha256": EXPECTED[M1769 / "review.json"]}, "release identity")
    need(release.get("interpreter_identity") == {
        "path": "/usr/bin/python3.12", "resolved_path": "/usr/bin/python3.12",
        "implementation": "CPython", "version": "3.12.13",
        "binary_sha256": EXPECTED[PYTHON312]}, "interpreter identity")
    need(release.get("authorization") == {
        "future_m1768_wrapper_attempts": 1,
        "automatic_retry": False,
        "underlying_m1753_campaigns": 1}, "wrapper authorization")
    need(release.get("wrapper_execution_budget") == {
        "interpreter_preflights": 1, "atomic_wrapper_attempts": 1,
        "execve_handoffs": 1, "direct_m1753_bypass": False,
        "automatic_retry": False}, "wrapper execution budget")
    inherited = release.get("inherited_m1761_campaign_budget", {})
    need(inherited == {
        "vcs_compiles": 3, "simv_runs": 15, "saif_files": 15,
        "ptpx_runs": 15, "axes": ["k1", "k8", "k1x8"],
        "cases_per_axis": 5,
        "all_fifteen_checked_saif_before_any_ptpx": True,
        "partial_axis_citable": False, "automatic_retry": False},
        "M1761 inherited budget")
    launch = release.get("launch_environment_exact", {})
    need(launch.get("M1768_EXPECTED_M1770_RELEASE_SHA256") ==
         "SUPPLY_FROM_THIS_RELEASE_SHA256_SIDECAR", "self reference policy")
    for name, digest in {
            "M1768_EXPECTED_WRAPPER_SHA256": EXPECTED[WRAPPER],
            "M1768_EXPECTED_SOURCE_CONTRACT_SHA256": EXPECTED[M1768_CONTRACT],
            "M1768_EXPECTED_M1769_REVIEW_SHA256": EXPECTED[M1769 / "review.json"],
            "M1768_EXPECTED_M1769_MANIFEST_SHA256": EXPECTED[M1769 / "SHA256SUMS"],
            "M1768_EXPECTED_M1769_OUTER_FILE_SHA256": EXPECTED[M1769 / "SHA256SUMS.seal.sha256"],
            "M1753_EXPECTED_RUNNER_SHA256": EXPECTED[M1753],
            "M1753_EXPECTED_SOURCE_CONTRACT_SHA256": EXPECTED[M1753_CONTRACT],
            "M1753_EXPECTED_M1760_REVIEW_SHA256": EXPECTED[M1760 / "review.json"],
            "M1753_EXPECTED_M1760_MANIFEST_SHA256": EXPECTED[M1760 / "SHA256SUMS"],
            "M1753_EXPECTED_M1760_OUTER_FILE_SHA256": EXPECTED[M1760 / "SHA256SUMS.seal.sha256"],
            "M1753_EXPECTED_M1761_RELEASE_SHA256": EXPECTED[M1761]}.items():
        need(launch.get(name) == digest and re.fullmatch(r"[0-9a-f]{64}", digest),
             "launch pin " + name)
    disclosure = release.get("mandatory_joint_disclosure", {})
    need(disclosure == {
        "equal_bandwidth_k8_vs_k1x8_cycle_speedup": 1.0167276529012024,
        "equal_bandwidth_k8_vs_k1x8_throughput_per_mm2": 4.562720096484654,
        "same_table_and_sentence": True,
        "k8_vs_single_k1_headline_forbidden": True}, "joint disclosure")
    need(release.get("claim_boundary") == {
        "launch_executed": False, "wrapper_preflight": False,
        "wrapper_attempt": False, "execve_handoff": False,
        "m1753_attempt": False, "mapped_vcs": False,
        "production_saif": False, "ptpx": False, "power": False,
        "energy": False, "performance": False, "system_speedup": False,
        "paper_ppa_ready": False, "headline": False}, "claim boundary")
    execution = release.get("release_execution", {})
    need(execution.get("wrapper_runs") == 0 and execution.get("m1753_runs") == 0
         and execution.get("license_queries") == 0 and execution.get("eda_runs") == 0
         and execution.get("wrapper_attempt_created") is False
         and execution.get("m1753_attempt_created") is False
         and execution.get("result_created") is False, "release execution")
    for path in (
            HW / "results/.m1768_m1753_c2_python312_wrapper_attempt_consumed",
            HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine",
            HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.private_build.unsealed_do_not_cite"):
        need(not os.path.lexists(str(path)), "execution namespace exists")
    need(not list((HW / "results").glob(
         ".m1768_m1753_c2_python312_wrapper_attempt_stage.*")), "stage residue")
    print(json.dumps({
        "schema": "m1770_release_author_self_check_r1_v1",
        "status": "PASS_M1770_WRAPPER_RELEASE_AUTHOR_SELF_CHECK__NO_EXECUTION",
        "release_sha256": EXPECTED[RELEASE],
        "python_path": "/usr/bin/python3.12",
        "python_version": "3.12.13",
        "wrapper_budget": release["wrapper_execution_budget"],
        "m1753_budget": inherited,
        "all_15_saif_before_ptpx": True,
        "automatic_retry": False,
        "wrapper_runs": 0,
        "m1753_runs": 0,
        "eda_runs": 0,
        "license_queries": 0,
        "attempt_created": False,
        "result_created": False}, sort_keys=True))


if __name__ == "__main__":
    main()
