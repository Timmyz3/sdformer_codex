#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent static M1012 release hammer. Never launches M1013 or EDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1011_m1001_c2_mapped_gate_saif_launch_release_r2_20260829.json"
RELEASE_SHA = Path(str(RELEASE) + ".sha256")
RELEASE_OUTER = Path(str(RELEASE) + ".sha256.seal.sha256")
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
RUNNER = HW / "dc_handoff/scripts/run_m1013_m1001_c2_mapped_gate_saif_one_shot_r2.sh"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
M1003 = HW / "contracts/m1003_m1001_c2_mapped_gate_saif_launch_release_r1_20260829.json"
M1004 = HW / "reviews/m1004_m1003_m1001_c2_mapped_gate_saif_release_hammer_r1_20260829"
RESULTS = HW / "results"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "release": "7f765d317de1164fb3268d0b801886d05aa826002c8e73087f0df1dd4947ea67",
    "release_sidecar": "f720ebd01636209d37a9ee494851e3a895f51c25c08ceddd73fb5b84735947a6",
    "release_outer": "31d0f41819670d44bcd7310b089508d573bf71013054c0b4940cd44a77819760",
    "contract": "7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6",
    "runner": "d9a7876a53c1becbba0155298b8f05aafba78dfedf42767ff298649fe13a9d14",
    "m1002_review": "e747c73b3add43e7010fc539f9f06d35f5e6e69219a9e66fc7f0e25e511045d7",
    "m1002_manifest": "019b810281f815d44d0024b89556ac7cacaea2c28885aa4ce79ead37761cc6eb",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
    "m1003": "5abef119dc4a84e5a91d8c9e15b46eec9b14e14c103f9c61c4dcb710ca72cfe3",
    "m1004_review": "188ea41b1c4d8cd8531ec1e7fc7b9571a7592e744adcc24aa97edfec2788cd3e",
    "m1004_manifest": "5043b5146da3898eb47b812f5f4dddd5c26a869ad052d23a6517d83e2824981b",
    "m1004_outer": "de404a9b9cbda3ecebe9aa2e493f2ec2fd3c36d6e1abe450ffb1e5874490adcd",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_flat(directory, review_sha, manifest_sha, outer_sha):
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sha(review) == review_sha and sha(manifest) == manifest_sha and
            sha(outer) == outer_sha, "sealed directory identity drift: " + directory.name)
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "sealed member drift: " + str(member))
        listed[rel] = digest
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "outer content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "sealed exact-set drift: " + directory.name)


def namespace_fresh(prefix):
    return not list(RESULTS.glob(prefix))


def main():
    require(sha(RELEASE) == EXPECTED["release"] and
            sha(RELEASE_SHA) == EXPECTED["release_sidecar"] and
            sha(RELEASE_OUTER) == EXPECTED["release_outer"],
            "M1011 release identity drift")
    require(RELEASE_SHA.read_text().split() == [EXPECTED["release"], RELEASE.name] and
            RELEASE_OUTER.read_text().split() == [EXPECTED["release_sidecar"], RELEASE_SHA.name],
            "M1011 release sidecar content drift")
    require(sha(CONTRACT) == EXPECTED["contract"] and
            sha(RUNNER) == EXPECTED["runner"] and
            sha(M1003) == EXPECTED["m1003"] and sha(DOC359) == EXPECTED["docs359"],
            "source/release authority drift")
    verify_flat(M1002, EXPECTED["m1002_review"], EXPECTED["m1002_manifest"],
                EXPECTED["m1002_outer"])
    verify_flat(M1004, EXPECTED["m1004_review"], EXPECTED["m1004_manifest"],
                EXPECTED["m1004_outer"])
    release = json.loads(RELEASE.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        RuntimeError("nonfinite JSON: " + value)))
    m1002 = json.loads((M1002 / "review.json").read_text())
    m1003 = json.loads(M1003.read_text())
    m1004 = json.loads((M1004 / "review.json").read_text())
    require(release["status"] == "PASS_M1011_M1001_LAUNCH_RELEASE_R2" and
            release["launch_now"] is True and release["max_attempts"] == 1 and
            release["runner_sha256"] == EXPECTED["runner"] and
            release["source_contract_sha256"] == EXPECTED["contract"],
            "M1011 top-level content drift")
    require(release["source_hammer"]["outer_seal_file_sha256"] == EXPECTED["m1002_outer"] and
            m1002["status"] == "PASS_M1002_M1001_SOURCE_HAMMER" and
            m1002["p0_count"] == 0, "M1002 authority drift")
    require(release["execution"]["axes"] == ["k1", "k8", "k1x8"] and
            release["execution"]["cases_per_axis"] == 5 and
            release["execution"]["total_gate_simulations"] == 15 and
            release["execution"]["fresh_compile_per_axis"] is True and
            release["execution"]["old_simv_reuse"] is False and
            release["execution"]["dut_only_saif"] is True,
            "execution geometry drift")
    authorization = release["authorization"]
    require(authorization["one_m1013_run"] is True and
            authorization["automatic_retry"] is False and
            authorization["vcs_mapped_gate"] is True and
            authorization["saif_generation"] is True and
            all(authorization[key] is False for key in ("pt", "ptpx", "dc", "gpu_remote")),
            "authorization scope drift")
    runner = RUNNER.read_text()
    for token in ("M1013_EXPECTED_RUNNER_SHA256", "M1013_EXPECTED_M1002_OUTER_SHA256",
                  "M1013_EXPECTED_M1012_OUTER_SHA256", "PASS_M1013_M1001",
                  "m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829"):
        require(token in runner, "M1013 runner missing token: " + token)
    for stale in ("M1005_EXPECTED_", "m1005_m1001_c2_three_axis_mapped_gate_saif_r1"):
        require(stale not in runner, "old execution namespace survived: " + stale)
    require(m1003["source_contract_sha256"] != EXPECTED["contract"] and
            m1004["status"] == "STOP_M1004_M1003_SOURCE_CONTRACT_PIN_DRIFT" and
            m1004["decision"]["m1005_execution_authorized"] is False,
            "old-chain STOP evidence drift")
    m1005_fresh = namespace_fresh("m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829*") and \
        namespace_fresh(".m1005_m1001_c2_three_axis_mapped_gate_saif*")
    m1013_fresh = namespace_fresh("m1013_m1001_c2_three_axis_mapped_gate_saif_r2_20260829*") and \
        namespace_fresh(".m1013_m1001_c2_three_axis_mapped_gate_saif*")
    require(m1005_fresh and m1013_fresh, "M1005/M1013 namespace already consumed")
    return {
        "schema": "m1012_m1011_m1001_c2_mapped_gate_saif_release_hammer_r2_v1",
        "status": "PASS_M1012_M1011_M1001_RELEASE_HAMMER_R2",
        "verdict": "GO_ONE_M1013_VCS_SAIF_ATTEMPT_ONLY",
        "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "identity": {"m1011_release_sha256": sha(RELEASE),
                     "m1013_runner_sha256": sha(RUNNER),
                     "m1001_contract_sha256": sha(CONTRACT),
                     "m1002_outer_sha256": EXPECTED["m1002_outer"],
                     "docs359_sha256": sha(DOC359)},
        "old_chain": {"m1003_wrong_source_contract_pin": True,
                      "m1004_stop_verified": True,
                      "m1005_attempt_consumed": False,
                      "m1005_result_created": False},
        "new_chain": {"release_double_sidecar": "PASS",
                      "axes": 3, "cases": 15,
                      "m1013_namespace_fresh": True,
                      "caller_exact_sha_required": True,
                      "max_attempts": 1},
        "decision": {"m1013_execution_authorized": True,
                     "automatic_retry": False,
                     "authorized_tools": ["VCS", "SAIF"],
                     "pt_ptpx_dc_authorized": False},
        "scope": {"static_release_hammer": True, "m1013_runs": 0,
                  "vcs_runs": 0, "pt_runs": 0, "ptpx_runs": 0,
                  "dc_runs": 0, "gpu_remote_runs": 0},
        "claim_boundary": {"saif_created": False, "power": False,
                           "energy": False, "system_speedup": False,
                           "paper_ppa_ready": False}
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
