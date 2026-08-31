#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent static M1004 release hammer. Never launches M1005 or EDA."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1003_m1001_c2_mapped_gate_saif_launch_release_r1_20260829.json"
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
RUNNER = HW / "dc_handoff/scripts/run_m1005_m1001_c2_mapped_gate_saif_one_shot.sh"
M1002 = HW / "reviews/m1002_m1001_c2_mapped_gate_saif_rekey_source_hammer_r1_20260829"
RESULTS = HW / "results"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "release": "5abef119dc4a84e5a91d8c9e15b46eec9b14e14c103f9c61c4dcb710ca72cfe3",
    "release_sidecar": "0c49e25e000dd2c12207da96512dc6d72eb75d3133a050f7007482119c7a72f7",
    "release_outer": "f79b43c9549ebb325d5ccda680dc1d32ff298cf86ee4bfb8988e7549f498bf65",
    "contract": "7afc4c093b802bdfd97aea101c803735e993c2eef57983311d3eb1a3d6bd36c6",
    "runner": "559ebb252fcda7592106413f5e2106a9156fffc5ce0de216baae45e218c0635d",
    "m1002_review": "e747c73b3add43e7010fc539f9f06d35f5e6e69219a9e66fc7f0e25e511045d7",
    "m1002_manifest": "019b810281f815d44d0024b89556ac7cacaea2c28885aa4ce79ead37761cc6eb",
    "m1002_outer": "d489e1cc3893e9c2a265ad5d35213e349f6eb44a5b4e2e15189711b1c82f5b85",
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


def verify_file_seal(payload):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    require(sha(payload) == EXPECTED["release"] and
            sha(sidecar) == EXPECTED["release_sidecar"] and
            sha(outer) == EXPECTED["release_outer"], "release seal identity drift")
    require(sidecar.read_text().split() == [sha(payload), payload.name] and
            outer.read_text().split() == [sha(sidecar), sidecar.name],
            "release seal content drift")


def verify_directory(directory):
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(sha(manifest) == EXPECTED["m1002_manifest"] and
            sha(outer) == EXPECTED["m1002_outer"], "M1002 seal identity drift")
    require(outer.read_text().split() == [EXPECTED["m1002_manifest"], "SHA256SUMS"],
            "M1002 outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1); rel = rel.lstrip("*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "M1002 member drift")
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "M1002 exact-set drift")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def main():
    verify_file_seal(RELEASE)
    m1002_seal = verify_directory(M1002)
    require(sha(CONTRACT) == EXPECTED["contract"] and
            sha(RUNNER) == EXPECTED["runner"] and
            sha(M1002 / "review.json") == EXPECTED["m1002_review"] and
            sha(DOC359) == EXPECTED["docs359"], "source authority drift")
    release = json.loads(RELEASE.read_text())
    m1002 = json.loads((M1002 / "review.json").read_text())
    require(release["status"] == "PASS_M1003_M1001_LAUNCH_RELEASE" and
            release["launch_now"] is True and release["max_attempts"] == 1 and
            release["runner_sha256"] == EXPECTED["runner"],
            "M1003 top-level scope drift")
    require(release["source_hammer"] == {
        "review_sha256": EXPECTED["m1002_review"],
        "manifest_sha256": EXPECTED["m1002_manifest"],
        "outer_seal_file_sha256": EXPECTED["m1002_outer"]},
        "M1002 binding drift")
    require(m1002["status"] == "PASS_M1002_M1001_SOURCE_HAMMER" and
            m1002["p0_count"] == 0 and
            m1002["decision"]["m1003_release_authoring_authorized"] is True,
            "M1002 authority drift")
    require(release["execution"] == {
        "axes": ["k1", "k8", "k1x8"], "cases_per_axis": 5,
        "total_gate_simulations": 15,
        "k8_cycle_anchors": [51, 131, 486, 1231, 14],
        "k1x8_cycle_anchors": [53, 133, 499, 1246, 14],
        "fresh_compile_per_axis": True, "old_simv_reuse": False,
        "dut_only_saif": True}, "M1003 execution geometry drift")
    require(release["authorization"] == {
        "one_m1005_run": True, "automatic_retry": False,
        "vcs_mapped_gate": True, "saif_generation": True,
        "pt": False, "ptpx": False, "dc": False, "gpu_remote": False},
        "M1003 authorization scope drift")

    release_pin = release["source_contract_sha256"]
    actual_pin = sha(CONTRACT)
    pin_mismatch = release_pin != actual_pin
    require(pin_mismatch, "expected source-contract P0 was not reproduced")
    runner_content_gate_would_pass = release_pin == actual_pin

    result = RESULTS / "m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829"
    attempt = RESULTS / ".m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"
    stale = sorted(path.name for path in RESULTS.glob(
        "m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829*"))
    stale += sorted(path.name for path in RESULTS.glob(
        ".m1005_m1001_c2_three_axis_mapped_gate_saif*"))
    require(not result.exists() and not attempt.exists() and not stale,
            "M1005 namespace not fresh")

    return {
        "schema": "m1004_m1003_m1001_c2_mapped_gate_saif_release_hammer_v1",
        "status": "STOP_M1004_M1003_SOURCE_CONTRACT_PIN_DRIFT",
        "verdict": "STOP",
        "score_out_of_100": 82,
        "p0_count": 1, "p1_count": 0, "p2_count": 0,
        "identity": {"m1003_release_sha256": EXPECTED["release"],
                     "m1005_runner_sha256": EXPECTED["runner"],
                     "m1002_review_sha256": EXPECTED["m1002_review"],
                     "m1002_outer_sha256": EXPECTED["m1002_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "m1002_seal": m1002_seal,
        "positive": {"m1003_double_sidecar": "PASS", "one_shot": True,
                     "total_gate_simulations": 15,
                     "vcs_and_saif_only": True,
                     "pt_ptpx_dc_authorized": False,
                     "m1005_namespace_fresh": True,
                     "m1005_or_eda_executed_by_m1004": False},
        "p0": [{"id": "P0_M1003_SOURCE_CONTRACT_SHA_MISMATCH",
                "release_pin": release_pin, "actual_source_contract_sha256": actual_pin,
                "runner_content_gate_would_pass": runner_content_gate_would_pass,
                "impact": "M1003 neither binds the frozen M1001 source contract nor passes the M1005 pre-EDA release content gate.",
                "required_repair": "Author a new additive release with the exact M1001 contract SHA, then independently hammer that release before any M1005-equivalent run."}],
        "decision": {"m1005_execution_authorized": False,
                     "automatic_retry": False,
                     "eda_runs_authorized": 0},
        "scope": {"static_release_hammer": True, "m1005_runs": 0,
                  "eda_runs": 0, "docs359_modified": False},
        "claim_boundary": {"saif_created": False, "power": False,
                           "energy": False, "paper_ppa_ready": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
