#!/usr/bin/env python3
"""Read-only M987 release/promotion hammer; never invokes promotion."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SCRIPT = HW / "dc_handoff/scripts/promote_m975_m962_quarantine_copy_only_canonical_recovery_r1.sh"
CONTRACT = HW / "contracts/m975_m962_quarantine_copy_only_canonical_recovery_source_contract_r1_20260829.json"
RELEASE = HW / "contracts/m976_m975_m962_quarantine_copy_only_canonical_recovery_release_r1_20260829.json"
M975 = HW / "reviews/m975_m962_quarantine_rc9_forensic_and_promotion_source_hammer_r1_20260829"
SOURCE = HW / "dc_handoff/runs/m962_m935_three_stage_match_macro_aware_dc_3p000ns_r1_20260829.failed_or_incomplete.3868703.quarantine"
TARGET = HW / "dc_handoff/runs/m975_m962_m935_three_stage_match_macro_aware_dc_3p000ns_recovered_canonical_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "script": "1066e375f70732fda9c1ae20c10d0029bc381d03c8361c04c167432824b3eda5",
    "contract": "b46e1dcc6e9691e828fafb811e22e7e523d5b226a28955c9f2a1366424c6211a",
    "review": "5819ac17810e3f3680e1f5cc2320f3f91aa742aa0086bed69ba5a9fd9ad2e429",
    "m975_manifest": "5bf54a8e89394eb143a27e4cdef0b71ea6e6b7560e5cae428021fd884de6a755",
    "m975_outer": "f73df2e7b7bf6bd8bb6b0ae867facc90f95ba81ef09df3c73018e6c7cb97ee9e",
    "release": "70696e6de27079539b210ad2996d09eebcd3265b1a0d5aaaaa63d3d3d2f45867",
    "release_sidecar": "8ddff5479aac89fbd8b0af0c3c99de1619c1757c6d4eef3de2a86c84844b2623",
    "release_outer": "01cc1319d73b977b04b6b62395ae0335847d531cf4c5dd769831c678c29fe460",
    "source_manifest": "9a1649638c0c2aa7b533fdb16cd763c87e6280dfc5a3c291240818cf1022eafe",
    "source_outer": "a213df2a38ff231f9d0dbd78c379ef13b3731caf3b5335c37d6d17bf20927997",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_file_seal(payload):
    sidecar = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    require(all(path.is_file() and not path.is_symlink()
                for path in (payload, sidecar, outer)), "file seal missing")
    require(sidecar.read_text().split() == [sha(payload), payload.name],
            "payload sidecar drift")
    require(outer.read_text().split() == [sha(sidecar), sidecar.name],
            "outer sidecar drift")


def verify_directory(directory, manifest_sha, outer_sha):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(directory.is_dir() and not directory.is_symlink(), "directory missing/symlink")
    require(manifest.is_file() and outer.is_file() and
            not manifest.is_symlink() and not outer.is_symlink(), "directory seals missing")
    require(sha(manifest) == manifest_sha and sha(outer) == outer_sha,
            "directory seal identity drift")
    require(outer.read_text().split() == [manifest_sha, "SHA256SUMS"],
            "directory outer content drift")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1); rel = rel.lstrip("*")
        require(rel not in listed and ".." not in Path(rel).parts, "manifest path drift")
        member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "manifest member drift: " + rel)
        listed[rel] = digest
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(actual == set(listed), "recursive exact-set drift")
    require(not [path for path in directory.rglob("*") if path.is_symlink()],
            "recursive symlink found")
    return {"entries": len(listed), "manifest_sha256": sha(manifest),
            "outer_seal_file_sha256": sha(outer)}


def reproduce_mv_race():
    with tempfile.TemporaryDirectory(prefix="m987_mv_race_") as temporary:
        root = Path(temporary)
        work_a = root / "target.copy_work.1"
        work_b = root / "target.copy_work.2"
        target = root / "target"
        work_a.mkdir(); work_b.mkdir()
        (work_a / "SEALED_A").write_text("a")
        (work_b / "SEALED_B").write_text("b")
        first = subprocess.run(["/usr/bin/mv", "--", str(work_a), str(target)], check=False)
        second = subprocess.run(["/usr/bin/mv", "--", str(work_b), str(target)], check=False)
        nested = (target / work_b.name / "SEALED_B").is_file()
        return {"first_mv_rc": first.returncode, "second_mv_rc": second.returncode,
                "second_work_nested_under_sealed_target": nested,
                "p0_reproduced": first.returncode == 0 and second.returncode == 0 and nested}


def main():
    require(sha(SCRIPT) == EXPECTED["script"], "promotion script SHA drift")
    require(sha(CONTRACT) == EXPECTED["contract"], "source contract SHA drift")
    require(sha(M975 / "review.json") == EXPECTED["review"], "M975 review SHA drift")
    require(sha(RELEASE) == EXPECTED["release"], "M976 release SHA drift")
    require(sha(DOC359) == EXPECTED["docs359"], "docs359 drift")
    verify_file_seal(SCRIPT); verify_file_seal(CONTRACT); verify_file_seal(RELEASE)
    require(sha(Path(str(RELEASE) + ".sha256")) == EXPECTED["release_sidecar"],
            "M976 sidecar file SHA drift")
    require(sha(Path(str(RELEASE) + ".sha256.seal.sha256")) == EXPECTED["release_outer"],
            "M976 outer sidecar file SHA drift")
    m975_seal = verify_directory(M975, EXPECTED["m975_manifest"], EXPECTED["m975_outer"])
    source_seal = verify_directory(SOURCE, EXPECTED["source_manifest"], EXPECTED["source_outer"])

    contract = json.loads(CONTRACT.read_text())
    review = json.loads((M975 / "review.json").read_text())
    release = json.loads(RELEASE.read_text())
    require(contract["status"] == "SOURCE_READY__PROMOTION_NOT_AUTHORIZED_NOW",
            "M975 source contract status")
    require(review["status"] ==
            "PASS_M975_M962_QUARANTINE_FORENSIC_RECOVERY__GO_COPY_ONLY_PROMOTION_SOURCE" and
            review["p0_count"] == 0, "M975 review authority")
    require(release["status"] == "AUTHORIZE_ONE_M975_COPY_ONLY_CANONICAL_RECOVERY" and
            release["authorization"] == {"copy_only_promotions": 1, "eda_runs": 0},
            "M976 release scope")
    require(release["identity"]["promotion_script_sha256"] == EXPECTED["script"] and
            release["identity"]["source_contract_sha256"] == EXPECTED["contract"] and
            release["identity"]["m975_review_sha256"] == EXPECTED["review"],
            "M976 release pin drift")

    script_text = SCRIPT.read_text()
    require('WORK="${TARGET}.copy_work.$$"' in script_text, "PID work naming drift")
    require('mv -- "${WORK}" "${TARGET}"' in script_text, "final mv form drift")
    require("mkdir -- \"${LOCK}\"" not in script_text and "ATTEMPT" not in script_text,
            "expected missing lock/attempt condition not reproduced")
    require("dc_shell -f" not in script_text and "pt_shell" not in script_text and
            "ptpx" not in script_text.lower() and "vcs " not in script_text.lower(),
            "promotion script contains EDA invocation")

    target_fresh = not TARGET.exists() and not TARGET.is_symlink()
    stale_work = sorted(path.name for path in TARGET.parent.glob(TARGET.name + ".copy_work.*"))
    require(target_fresh and not stale_work, "target/work namespace not fresh")
    race = reproduce_mv_race()
    require(race["p0_reproduced"], "TOCTOU nesting race not reproduced")

    return {
        "schema": "m987_m976_m975_m962_quarantine_copy_only_release_promotion_hammer_v1",
        "status": "STOP_M987_PROMOTION_RELEASE_NOT_FAIL_CLOSED",
        "verdict": "STOP",
        "score_out_of_100": 80,
        "p0_count": 1, "p1_count": 0, "p2_count": 0,
        "pins": {"script_sha256": EXPECTED["script"],
                 "source_contract_sha256": EXPECTED["contract"],
                 "m975_review_sha256": EXPECTED["review"],
                 "m976_release_sha256": EXPECTED["release"]},
        "source_quarantine_seal": source_seal,
        "m975_review_seal": m975_seal,
        "target_fresh": target_fresh,
        "stale_work_prefixes": stale_work,
        "p0": {"id": "P0_CONCURRENT_PROMOTION_TOCTOU_CORRUPTS_SEALED_TARGET",
               "evidence": race,
               "cause": "No lock/attempt; PID-specific work roots; final mv lacks -T/no-replace and does not recheck target."},
        "scope": {"promotion_executed_by_m987": False, "eda_runs": 0,
                  "source_modified": False, "target_created": False},
        "claim_boundary": {"setup_area_promoted": False, "paper_citable": False,
                           "speedup": False, "system_speedup": False,
                           "paper_ppa_ready": False},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True, allow_nan=False))
