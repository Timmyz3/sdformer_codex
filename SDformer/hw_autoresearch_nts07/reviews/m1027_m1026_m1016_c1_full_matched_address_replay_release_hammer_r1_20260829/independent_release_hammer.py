#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1027 static release-chain hammer. Never executes M1028/full replay/EDA."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1028_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1026_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
RELEASE_SHA = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
CONTRACT = HW / "contracts/m1016_m1010_c1_full_matched_address_replay_source_contract_r1_20260829.json"
ENGINE = HW / "system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"
CHECKER = HW / "system_simulator/scripts/check_m1016_c1_full_matched_address_replay_source.py"
TESTS = HW / "system_simulator/tests/test_m1016_c1_full_matched_address_replay_source.py"
M1025 = HW / "reviews/m1025_m1016_c1_full_matched_address_replay_source_hammer_r1_20260829"
RESULTS = HW / "results"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "f557c0e65e745500579873d3b7df0d53fad8103d93452a0e7f7bc7327722dc47",
    "release": "96e9685ba9eae5f3545ca0745cfeff9acd2255000238b70f2d5f1b0b1ded1afd",
    "release_sidecar": "ab24498486fb371c67629329b98e9cf91610f6a4de000d58d4a3be5a9bb35553",
    "release_outer": "6b495010bfad7746479c1930cd724961e58c8f0896335a1b99f6c5f0b12e4fc4",
    "contract": "b980f51017778b1958845547601de5d343ba5a1f3db1b046963afa7549644c90",
    "engine": "d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa",
    "checker": "f08f45ca10f41524dd8f4b7f679af11b0621ea3097dbe0d9c642d9f8259f06a3",
    "tests": "f2d92f41eda1bf5f74bc63fbdba3d6315e172ce8175e04542b238cc729c5759c",
    "m1025_review": "6c86079035a52af4a36a3156ed4cdd6cb0bb71b51b198e71eb79512dfc361703",
    "m1025_manifest": "89155180dd298b752b4ef92ae70fe907e393733d23cb0a4d9be1b2a5f2784e16",
    "m1025_outer": "7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff",
    "m410": "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def verify_flat(directory, review_sha, manifest_sha, outer_sha):
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_sha), "M1025 identity drift")
    listed = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("./*")
        require(name not in listed and sha(directory / name) == expected,
                "M1025 member drift")
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual and outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "M1025 exact-set/outer drift")


def release_valid(value):
    authorization = value.get("authorization", {})
    execution = value.get("execution", {})
    return bool(
        value.get("status") == "PASS_M1026_M1016_C1_FULL_REPLAY_LAUNCH_RELEASE"
        and value.get("launch_now") is True
        and value.get("max_attempts") == 1
        and value.get("runner_sha256") == EXPECTED["runner"]
        and value.get("source_contract_sha256") == EXPECTED["contract"]
        and value.get("engine_sha256") == EXPECTED["engine"]
        and value.get("source_checker_sha256") == EXPECTED["checker"]
        and value.get("source_tests_sha256") == EXPECTED["tests"]
        and value.get("m1025_outer_seal_file_sha256") == EXPECTED["m1025_outer"]
        and value.get("m410_rows_sha256") == EXPECTED["m410"]
        and execution.get("hardcoded_authority_paths") is True
        and execution.get("caller_selectable_release_path") is False
        and execution.get("caller_selectable_hammer_path") is False
        and authorization.get("one_m1028_attempt") is True
        and authorization.get("automatic_retry") is False
        and authorization.get("cpu_full_replay") is True
        and authorization.get("m1027_release_hammer_required_before_attempt") is True
        and all(authorization.get(key) is False for key in
                ("vcs", "dc", "formality", "pt", "ptpx", "gpu", "remote"))
    )


def mutation_rejections(release):
    faults = [
        ("status", ("status", "BAD")),
        ("launch_false", ("launch_now", False)),
        ("attempts_two", ("max_attempts", 2)),
        ("runner_pin", ("runner_sha256", "0" * 64)),
        ("contract_pin", ("source_contract_sha256", "0" * 64)),
        ("engine_pin", ("engine_sha256", "0" * 64)),
        ("checker_pin", ("source_checker_sha256", "0" * 64)),
        ("tests_pin", ("source_tests_sha256", "0" * 64)),
        ("m1025_outer", ("m1025_outer_seal_file_sha256", "0" * 64)),
        ("m410_pin", ("m410_rows_sha256", "0" * 64)),
    ]
    results = []
    for name, (key, changed) in faults:
        value = copy.deepcopy(release)
        value[key] = changed
        results.append({"fault": name, "rejected": not release_valid(value)})
    for name, section, key, changed in (
        ("caller_release_path", "execution", "caller_selectable_release_path", True),
        ("caller_hammer_path", "execution", "caller_selectable_hammer_path", True),
        ("hardcoded_paths_false", "execution", "hardcoded_authority_paths", False),
        ("automatic_retry", "authorization", "automatic_retry", True),
        ("vcs_authorized", "authorization", "vcs", True),
        ("ptpx_authorized", "authorization", "ptpx", True),
        ("hammer_not_required", "authorization", "m1027_release_hammer_required_before_attempt", False),
    ):
        value = copy.deepcopy(release)
        value[section][key] = changed
        results.append({"fault": name, "rejected": not release_valid(value)})
    require(all(item["rejected"] for item in results), "release mutation accepted")
    return results


def main():
    require(sha(RUNNER) == EXPECTED["runner"] and sha(RELEASE) == EXPECTED["release"] and
            sha(RELEASE_SHA) == EXPECTED["release_sidecar"] and
            sha(RELEASE_OUTER) == EXPECTED["release_outer"], "new-chain identity drift")
    require(RELEASE_SHA.read_text().split() == [EXPECTED["release"], RELEASE.name] and
            RELEASE_OUTER.read_text().split() == [EXPECTED["release_sidecar"], RELEASE_SHA.name],
            "M1026 sidecar content drift")
    require(sha(CONTRACT) == EXPECTED["contract"] and sha(ENGINE) == EXPECTED["engine"] and
            sha(CHECKER) == EXPECTED["checker"] and sha(TESTS) == EXPECTED["tests"] and
            sha(DOC359) == EXPECTED["docs359"], "M1016/docs authority drift")
    verify_flat(M1025, EXPECTED["m1025_review"], EXPECTED["m1025_manifest"],
                EXPECTED["m1025_outer"])
    m1025 = json.loads((M1025 / "review.json").read_text())
    require(m1025["status"] ==
            "PASS_M1025_M1016_C1_FULL_MATCHED_ADDRESS_REPLAY_SOURCE_HAMMER" and
            m1025["authorization"]["author_execution_release_and_exact_runner"] is True and
            m1025["authorization"]["execute_51840000_replay"] is False,
            "M1025 authority scope drift")
    release = json.loads(RELEASE.read_text(),
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             RuntimeError("nonfinite JSON: " + value)))
    require(release_valid(release), "M1026 release semantics drift")
    faults = mutation_rejections(release)
    runner = RUNNER.read_text()
    for token in ("M1028_EXPECTED_RUNNER_SHA256", "M1028_EXPECTED_M1025_OUTER_SHA256",
                  "M1028_EXPECTED_M1027_OUTER_SHA256", "ATTEMPT_ATOMIC_CONSUME",
                  "cleanup()", "seal_dir()", "PASS_M1028_M1016_RAW_FULL_REPLAY"):
        require(token in runner, "M1028 runner missing: " + token)
    for forbidden in ("M1016_RELEASE_JSON", "M1016_RELEASE_HAMMER_DIR",
                      "/opt/synopsys", "dc_shell", "pt_shell", "nvidia-smi", "ssh "):
        require(forbidden not in runner, "generic/EDA path survived: " + forbidden)
    require("contracts/m1026_m1016_c1_full_matched_address_replay_launch_release" in runner and
            "reviews/m1027_m1026_m1016_c1_full_matched_address_replay_release_hammer" in runner,
            "M1026/M1027 paths not hardcoded")
    stale = list(RESULTS.glob("m1028_m1016_c1_full_matched_address_replay_r1_20260829*"))
    stale += list(RESULTS.glob(".m1028_m1016_c1_full_matched_address_replay*"))
    require(not stale, "M1028 namespace already consumed")
    return {
        "schema": "m1027_m1026_m1016_c1_full_replay_release_hammer_v1",
        "status": "PASS_M1027_M1026_M1016_C1_FULL_REPLAY_RELEASE_HAMMER",
        "verdict": "GO_ONE_M1028_CPU_FULL_REPLAY_ATTEMPT_ONLY",
        "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "identity": {"runner_sha256": sha(RUNNER), "release_sha256": sha(RELEASE),
                     "contract_sha256": sha(CONTRACT), "engine_sha256": sha(ENGINE),
                     "m1025_outer_sha256": EXPECTED["m1025_outer"],
                     "docs359_sha256": sha(DOC359)},
        "fault_injections": faults,
        "execution_chain": {"hardcoded_authority_paths": True,
                            "caller_exact_pins": ["runner", "M1025 outer", "M1027 outer"],
                            "one_shot": True, "cleanup_quarantine": True,
                            "recursive_result_seal": True, "m1028_namespace_fresh": True},
        "authorization": {"m1028_cpu_full_replay": True, "max_attempts": 1,
                          "automatic_retry": False, "eda_gpu_remote": False},
        "scope": {"static_only": True, "m1028_runs": 0,
                  "full_51840000_replays": 0, "eda_runs": 0},
        "claim_boundary": {"capacity_214912B": False, "matched_cycles": False,
                           "speedup": False, "paper_ppa_ready": False}
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
