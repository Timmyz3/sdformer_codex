#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Static/source-only checker for the M1336 C2 one-shot VCS release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HW = Path(__file__).resolve().parents[1]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1336_c2_headline_mapped_production_activity_one_shot_exact_sha.sh"
CONTRACT = HW / "contracts/m1336_c2_headline_mapped_production_activity_vcs_release_source_contract_r1_20260831.json"
TEST = HW / "verif_m1336_c2_activity_release/test_m1336_c2_activity_vcs_release_source.py"
CHECKER = Path(__file__).resolve()
AUTHOR = HW / "reviews/m1336_c2_headline_mapped_production_activity_vcs_release_source_author_r1_20260831"
M1334_AUTHOR = HW / "reviews/m1334_c2_headline_mapped_production_activity_source_author_r1_20260831"
M1335_BLIND = HW / "reviews/m1335_m1334_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M872 = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
SOURCE_CHECKER = HW / "system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
SOURCE_TEST = HW / "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py"
FILELIST_K8 = HW / "dc_handoff/filelists/date_m1334_c2_k8_mapped_production_activity.f"
FILELIST_K1X8 = HW / "dc_handoff/filelists/date_m1334_c2_k1x8_mapped_production_activity.f"
UCLI = HW / "dc_handoff/scripts/m1334_c2_headline_mapped_production_activity.ucli.tcl"
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
LMUTIL = Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil")
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
NET = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
SDC = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.sdc"
EXPECTED = {
    M1334_AUTHOR / "review.json": "0bd2a6fdd75c24efcec2f41cf10bd689c2af3c56b61c968b036803ecc80b1ed9",
    M1334_AUTHOR / "SHA256SUMS": "0d2bd9d33cd1140ed26c28e47521c43672e3456b1e20de2f7260bc63be6872a7",
    M1334_AUTHOR / "SHA256SUMS.seal.sha256": "fe6d57fb982b50a60c16e0ec3f25c0fcf99db1c21f07a6407314d26787ec331d",
    M1335_BLIND / "review.json": "2905fdec0e8799bd3790cadf3ca8c29b901deb564c2a290b83b63990465528c0",
    M1335_BLIND / "SHA256SUMS": "29efd256bdf7d328e7f965afa06d5b3b6a266447b5c6c8737e61e807d5958d55",
    M1335_BLIND / "SHA256SUMS.seal.sha256": "7ca5e39a9abeb85049de24b25b3f019df51f74d8519c105aa092c3e5cfd004b4",
    M903 / "review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    M903 / "SHA256SUMS.seal.sha256": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    M872 / "k8" / NET: "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
    M872 / "k8" / SDC: "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
    M872 / "k1x8" / NET: "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
    M872 / "k1x8" / SDC: "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
    SOURCE_CHECKER: "c9326ff934239e8773e9f991e6bf0be94bba9c9c602be199433c22d1cd4c9da9",
    SOURCE_TEST: "d90a23fb9c7c8f18666d26dcfb2b0ac75ed160bc84ba36506e4b881701117ab8",
    FILELIST_K8: "9030ca8f6e42a21546332f25009e08033e6a6740f5d95fd8c5a36f190ac00e6d",
    FILELIST_K1X8: "cca8a9b0bfe0c32d85f554994ab61c2b78dba425e6dee194fe9f1557b54998e9",
    UCLI: "c90153dfd58ff4e653852a54b31ad3b19cb8fabd993e15c21d9071b555cbebc1",
    VCS: "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    LMUTIL: "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    PYTHON: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CYCLES = {"k8": [51, 131, 486, 1231, 14],
          "k1x8": [53, 133, 499, 1246, 14]}
EVENTS = [20, 41, 90, 110, 0]
ENV_NAMES = (
    "M1336_EXPECTED_RUNNER_SHA256",
    "M1336_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
    "M1336_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
    "M1336_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
    "M1336_EXPECTED_LAUNCH_RELEASE_SHA256",
    "M1336_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
    "M1336_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
    "M1336_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sidecar(path: Path) -> None:
    sums = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    assert sums.read_text().split() == [sha(path), path.name]
    assert outer.read_text().split() == [sha(sums), sums.name]


def verify_dir(root: Path) -> None:
    sums, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    assert root.is_dir() and not root.is_symlink()
    assert outer.read_text().split() == [sha(sums), "SHA256SUMS"]
    listed = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        pure = Path(name)
        assert name not in listed and not pure.is_absolute() and ".." not in pure.parts
        listed[name] = digest
    actual = set()
    for base_text, dirs, files in os.walk(root, followlinks=False):
        base = Path(base_text)
        assert all(not (base / name).is_symlink() for name in dirs + files)
        for name in files:
            path = base / name; rel = path.relative_to(root).as_posix()
            if rel not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
                actual.add(rel)
    assert actual == set(listed)
    for name, digest in listed.items():
        assert sha(root / name) == digest


def env_gate(env: dict[str, str]) -> bool:
    return all(re.fullmatch(r"[0-9a-f]{64}", env.get(name, "")) is not None
               for name in ENV_NAMES)


def exact_runner_gate(text: str, digest: str) -> bool:
    return sha_text(text) == digest


def namespaces() -> list[Path]:
    result = HW / "results/m1336_c2_headline_mapped_production_activity_vcs_r1_20260831"
    return [
        HW / "results/.m1336_c2_headline_mapped_production_activity_vcs_attempt_consumed",
        result,
        Path(str(result) + ".failed_or_incomplete.quarantine"),
        Path(str(result) + ".private_build.unsealed_do_not_cite"),
        Path(str(result) + ".failed_private_build.unsealed_do_not_cite"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-author", action="store_true")
    args = parser.parse_args(); checks = 0
    for path, digest in EXPECTED.items():
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, path
        checks += 1
    verify_dir(M1334_AUTHOR); verify_dir(M1335_BLIND); checks += 2
    blind = json.loads((M1335_BLIND / "review.json").read_text())
    assert blind["status"] == "PASS_SOURCE_ADMITTED__EXACT_SHA_VCS_RELEASE_CONTRACT_MAY_BE_AUTHORED"
    assert blind["predecessor"]["remaining_false_negative_count"] == 0
    assert blind["frozen_m872_m903_anchors"]["k8_cycles"] == CYCLES["k8"]
    assert blind["frozen_m872_m903_anchors"]["k1x8_cycles"] == CYCLES["k1x8"]; checks += 1
    m903 = json.loads((M903 / "review.json").read_text())
    metrics = m903["fair_equal_bandwidth_metrics"]
    assert metrics["frozen_directed_vcs_cycles"] == {
        "k8": CYCLES["k8"], "k1x8": CYCLES["k1x8"]}
    assert metrics["aggregate_sum_cycles"] == {"k8": 1913, "k1x8": 1945}; checks += 1

    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    assert syntax.returncode == 0, syntax.stderr; checks += 1
    contract = json.loads(CONTRACT.read_text()); sidecar(CONTRACT); checks += 1
    assert contract["status"] == "M1336_C2_ACTIVITY_ONE_SHOT_RELEASE_SOURCE_READY__FRESH_M1337_REQUIRED"
    assert contract["identity"]["runner_sha256"] == sha(RUNNER)
    assert contract["identity"]["checker_sha256"] == sha(CHECKER)
    assert contract["identity"]["test_sha256"] == sha(TEST)
    assert contract["workloads"] == {"events": EVENTS, "k8_cycles": CYCLES["k8"],
                                      "k1x8_cycles": CYCLES["k1x8"]}
    assert contract["authorization"]["vcs_now"] is False
    assert contract["future_execution"] == {
        "vcs_compiles": 2, "simv_runs": 10, "automatic_retry": False,
        "compile_timeout_seconds": 1800, "simulation_timeout_seconds": 600,
        "attempt_consumed_before_first_vcs": True}
    for key in ("functional_vcs_verified", "production_saif", "ptpx", "power",
                "energy", "performance", "system_speedup", "paper_ppa_ready",
                "headline"):
        assert contract["claim_boundary"][key] is False
    checks += 1

    runner = RUNNER.read_text()
    assert "ucli.key" not in runner and re.search(r"(^|\s)rm([\s-]|$)", runner) is None
    assert runner.count('"${VCS_BIN}" -full64') == 1
    assert runner.count('./simv +M979_UCLI_SAIF') == 1
    assert "for axis in k8 k1x8; do" in runner
    assert "for case_id in 0 1 2 3 4; do" in runner
    assert "[[ \"${compile_count}\" -eq 2 && \"${sim_count}\" -eq 10 ]]" in runner
    assert "cycles=(51 131 486 1231 14)" in runner
    assert "cycles=(53 133 499 1246 14)" in runner
    assert "events=(20 41 90 110 0)" in runner
    assert runner.index('phase="RESOURCE_PREFLIGHT"') < runner.index('phase="ATTEMPT_CONSUME"')
    assert runner.index('phase="LICENSE_PREFLIGHT"') < runner.index('phase="ATTEMPT_CONSUME"')
    assert "publish_no_replace \"${ATTEMPT_STAGE}\" \"${ATTEMPT}\"" in runner
    assert "seal_dir \"${FAILURE_STAGE}\"" in runner
    assert "seal_dir \"${RESULT_STAGE}\"" in runner
    assert "automatic_retry=false" in runner
    assert "CANDIDATE_UNSEALED_DO_NOT_CITE" in runner
    assert runner.count("M1334_SAIF_FILE") == 1
    checks += 20

    good = {name: "a" * 64 for name in ENV_NAMES}
    assert env_gate(good); checks += 1
    for name in ENV_NAMES:
        bad = dict(good); bad.pop(name); assert not env_gate(bad); checks += 1
        bad = dict(good); bad[name] = "A" * 64; assert not env_gate(bad); checks += 1
    assert exact_runner_gate(runner, contract["identity"]["runner_sha256"]); checks += 1
    assert not exact_runner_gate(runner + "\n", contract["identity"]["runner_sha256"]); checks += 1

    for path in namespaces():
        assert not os.path.lexists(path), path
        checks += 1
    for pattern in (
        ".m1336_c2_headline_mapped_production_activity_vcs_work.*",
        ".m1336_c2_headline_mapped_production_activity_vcs_result_stage.*",
        ".m1336_c2_headline_mapped_production_activity_vcs_attempt_stage.*",
        ".m1336_c2_headline_mapped_production_activity_vcs_failure_stage.*"):
        assert not list((HW / "results").glob(pattern)); checks += 1

    future = [
        HW / "reviews/m1337_m1336_c2_headline_mapped_production_activity_vcs_release_source_blind_hammer_r1_20260831",
        HW / "contracts/m1338_m1336_c2_headline_mapped_production_activity_vcs_launch_release_r1_20260831.json",
        HW / "reviews/m1339_m1338_m1336_c2_headline_mapped_production_activity_vcs_final_launch_hammer_r1_20260831",
    ]
    assert all(not os.path.lexists(path) for path in future); checks += len(future)
    if not args.skip_author:
        verify_dir(AUTHOR); author = json.loads((AUTHOR / "review.json").read_text())
        assert author["status"] == "PASS_M1336_C2_ACTIVITY_RELEASE_SOURCE__FRESH_M1337_HAMMER_REQUIRED"
        assert author["bindings"]["runner_sha256"] == sha(RUNNER)
        assert author["bindings"]["source_contract_sha256"] == sha(CONTRACT)
        checks += 1
    print(json.dumps({
        "schema": "m1336_c2_activity_vcs_release_source_static_check_r1",
        "status": "PASS_M1336_SOURCE_ONLY__FRESH_M1337_REQUIRED__NO_VCS_NO_EDA",
        "checks_passed": checks,
        "future_cardinality": {"vcs_compiles": 2, "simv_runs": 10},
        "one_shot": True, "automatic_retry": False,
        "success_recursive_seal": True, "failure_recursive_seal": True,
        "vcs_runs": 0, "simv_runs": 0, "license_queries": 0,
        "docs359_sha256": sha(DOCS359),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
