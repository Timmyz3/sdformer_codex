#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent M992 release hammer. Never runs promotion or EDA."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m991_m990_m989_atomic_one_shot_copy_only_promotion_release_r1_20260829.json"
SCRIPT = HW / "dc_handoff/scripts/promote_m989_m962_quarantine_atomic_one_shot_copy_only_r1.sh"
CONTRACT = HW / "contracts/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_contract_r1_20260829.json"
M990 = HW / "reviews/m990_m989_m975_m962_atomic_one_shot_promotion_source_hammer_r1_20260829"
M989 = HW / "reviews/m989_m975_m962_atomic_one_shot_copy_only_promotion_source_r1_20260829"
SOURCE = HW / "dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RUNS = HW / "dc_handoff/runs"

EXPECTED = {
    "release": "4cd6bd1777407dd0b5282713a12b945d53ea3cbcbb63ad0a2409d161c85992e7",
    "release_sidecar": "9ae3f8348f4565c9b579aca89d804d822fa9031f9323f3852e4ed0cce0b40dc9",
    "release_outer": "3e45704f4f43cb1d34eda4e08600116979855e5346c904b593869c0b39ae8450",
    "script": "7b63668f5fb68ac8d60acf4e43925313ab1c0bdc84caeefcbfb0e238871c4be9",
    "contract": "9be7b8640f3f1674de5e603992cf69c4f091000c1c09c9bcfa4f67745a037030",
    "m990_review": "ead5d294339a8877f5b13495dfc8d6d7a88b6f3f9dd1ef9e6221da4bf509b3ee",
    "m990_manifest": "b4e6e8746c5e74bd2869c60d154208e165469379f65c398def78cf7e9288ab1a",
    "m990_outer": "022066afae8ab52cd430183fac310c32dd05f2970703071fc14c58d399da9ca9",
    "m989_review": "801abbe4d19a544b30d0b57f05c28a1f99d5a0fbe594261861d4ea3f05d2e6d3",
    "m989_manifest": "4af382fcc6c6b9ced71a2d34dad2f037221530e09e0c34801930259780221edf",
    "m989_outer": "ac4a58e4516f7e7ceabad14b42077d148855e4c3f4f1d2a20533b887117fdaf7",
    "source_manifest": "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
    "source_outer": "a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997",
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


def verify_file_seal(payload, sidecar_sha, outer_sha):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    require(all(path.is_file() and not path.is_symlink()
                for path in (payload, sidecar, outer)), "release seal missing/symlink")
    require(sha(sidecar) == sidecar_sha and sha(outer) == outer_sha,
            "release seal identity drift")
    require(sidecar.read_text().split() == [sha(payload), payload.name],
            "release sidecar content drift")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name],
            "release outer content drift")


def verify_directory(directory, manifest_sha, outer_sha):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "directory missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(), "directory seal missing")
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "directory seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "directory outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1); rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts,
                "unsafe/duplicate manifest path")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "manifest member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(listed) == actual, "recursive exact-set drift")
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "recursive symlink found")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def main():
    require(sha(RELEASE) == EXPECTED["release"], "M991 release SHA drift")
    require(sha(SCRIPT) == EXPECTED["script"], "promotion script SHA drift")
    require(sha(CONTRACT) == EXPECTED["contract"], "M989 contract SHA drift")
    require(sha(M990 / "review.json") == EXPECTED["m990_review"], "M990 review drift")
    require(sha(M989 / "review.json") == EXPECTED["m989_review"], "M989 review drift")
    require(sha(DOC359) == EXPECTED["docs359"], "docs359 drift")
    verify_file_seal(RELEASE, EXPECTED["release_sidecar"], EXPECTED["release_outer"])
    m990_seal = verify_directory(M990, EXPECTED["m990_manifest"], EXPECTED["m990_outer"])
    m989_seal = verify_directory(M989, EXPECTED["m989_manifest"], EXPECTED["m989_outer"])
    source_seal = verify_directory(SOURCE, EXPECTED["source_manifest"], EXPECTED["source_outer"])

    release = json.loads(RELEASE.read_text())
    contract = json.loads(CONTRACT.read_text())
    m990 = json.loads((M990 / "review.json").read_text())
    m989 = json.loads((M989 / "review.json").read_text())
    require(release["status"] == "AUTHORIZE_ONE_M993_M989_COPY_ONLY_CANONICAL_RECOVERY" and
            release["authorization"] == {"copy_only_promotions": 1, "eda_runs": 0},
            "M991 release scope drift")
    require(release["identity"]["promotion_script_sha256"] == EXPECTED["script"] and
            release["identity"]["source_contract_sha256"] == EXPECTED["contract"] and
            release["identity"]["m990_review_sha256"] == EXPECTED["m990_review"] and
            release["identity"]["source_quarantine_manifest_sha256"] ==
            EXPECTED["source_manifest"] and
            release["identity"]["source_quarantine_outer_seal_file_sha256"] ==
            EXPECTED["source_outer"], "M991 release binding drift")
    require(release["preconditions"] == {
        "m990_p0_count": 0,
        "concurrency_protocol_admitted": True,
        "target_lock_attempt_work_and_failure_identities_fresh": True,
        "copy_only": True,
        "original_failure_marker_preserved": True,
        "independent_m992_release_hammer_required": True,
    }, "M991 precondition drift")
    require(m990["status"] == "PASS_M990_M989_COPY_ONLY_PROMOTION_SOURCE_HAMMER" and
            m990["p0_count"] == 0 and
            m990["decision"]["concurrency_protocol_admitted"] is True and
            m990["decision"]["m991_release_authoring_authorized"] is True and
            m990["decision"]["m993_execution_authorized_now"] is False,
            "M990 authority drift")
    require(m989["status"] ==
            "PASS_M989_ATOMIC_ONE_SHOT_COPY_ONLY_PROMOTION_SOURCE__FUTURE_M990_HAMMER_REQUIRED" and
            contract["authorization"] == {"promotion_runs_now": 0,
                                           "future_copy_only_promotions_max": 1,
                                           "eda_runs": 0}, "M989 source scope drift")

    names = ["m993_m989_m962_m935_macro_aware_dc_recovered_canonical_r1_20260829",
             ".m993_m989_m962_copy_promotion_launch_lock",
             ".m993_m989_m962_copy_promotion_attempt_consumed",
             ".m993_m989_m962_copy_promotion_work",
             "m993_m989_m962_copy_promotion_failed_or_incomplete.quarantine"]
    freshness = {name: not (RUNS / name).exists() and not (RUNS / name).is_symlink()
                 for name in names}
    require(all(freshness.values()), "M993 namespace is not fresh")

    source = SCRIPT.read_text()
    forbidden_invocations = [token for token in
        ("dc_shell -f", "pt_shell", "ptpx", "vcs ", "simv", "fm_shell")
        if token in source.lower()]
    require(not forbidden_invocations, "promotion contains EDA/simulation invocation")
    require('cp -a --no-dereference "${SOURCE}/." "${WORK}/original_quarantine/"' in
            source and 'mv -T -- "${WORK}" "${TARGET}"' in source,
            "copy-only publication tokens missing")
    require('if ! mkdir -- "${LOCK}"; then' in source and
            'if ! mkdir -- "${ATTEMPT}"; then' in source and
            source.index('if ! mkdir -- "${ATTEMPT}"; then') <
            source.index('cp -a --no-dereference'), "one-shot state order drift")
    require("rm -rf" not in source and "rm -r" not in source,
            "recursive mutation found")
    nested_p2 = next(item for item in m990["p2"]
                     if item["id"] == "P2_NESTED_SEAL_FILES_NOT_ROOT_MANIFEST_MEMBERS")
    require("original_quarantine" in nested_p2["finding"],
            "M990 nested-seal boundary absent")

    return {
        "schema": "m992_m991_m990_m989_promotion_release_hammer_v1",
        "status": "PASS_M992_M991_M990_M989_PROMOTION_RELEASE_HAMMER",
        "verdict": "GO_ONE_M993_COPY_ONLY_PROMOTION",
        "score_out_of_100": 98,
        "p0_count": 0, "p1_count": 0, "p2_count": 2,
        "identity": {
            "m991_release_sha256": EXPECTED["release"],
            "promotion_script_sha256": EXPECTED["script"],
            "m989_source_contract_sha256": EXPECTED["contract"],
            "m990_review_sha256": EXPECTED["m990_review"],
            "m989_review_sha256": EXPECTED["m989_review"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "source_seals": {"m990": m990_seal, "m989": m989_seal,
                         "m962_quarantine": source_seal},
        "freshness": freshness,
        "positive": {
            "release_double_sidecar": "PASS",
            "script_and_source_pins": "PASS",
            "m990_p0_zero": True,
            "concurrency_protocol_admitted": True,
            "copy_only": True,
            "eda_invocations": 0,
            "promotion_executed_by_m992": False,
        },
        "p2_boundaries": [
            "Same-UID noncooperating mutation remains outside the cooperating-runner model.",
            "Nested original_quarantine seal files are not root-manifest members. M993 result hammer must independently verify both inner SHA256SUMS identities and exact-set coverage.",
        ],
        "decision": {
            "authorize_m993_execution": True,
            "max_copy_only_promotions": 1,
            "eda_runs_authorized": 0,
            "m993_result_hammer_must_verify_nested_original_quarantine": True,
        },
        "scope": {"release_hammer_only": True, "promotion_executed": False,
                  "eda_runs": 0, "docs359_modified": False},
        "claim_boundary": {"paper_citable_now": False,
                           "setup_area_promoted_now": False,
                           "speedup": False, "system_speedup": False,
                           "paper_ppa_ready": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
