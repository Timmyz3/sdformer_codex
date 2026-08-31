#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1214 hammer for M1213 C1/R8 VCS release source."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1213_m1210r8_m1162_c1_common_charge_protocol_exact_sha_r8.sh"
CHECKER = HW / "verif_m1213_c1_r8_vcs_release/static_check_m1213_c1_r8_vcs_release_source.py"
CONTRACT = HW / "contracts/m1213_m1212_m1210_c1_r8_vcs_launcher_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1213_m1212_m1210_c1_r8_vcs_launch_release_r1_20260830.json"
M1210_AUTHOR = HW / "reviews/m1210_m1207_c1_r8_random_request_quiesce_author_receipt_r1_20260830"
M1212 = HW / "reviews/m1212_m1210_c1_r8_random_request_quiesce_source_hammer_r1_20260830"
M1213_AUTHOR = HW / "reviews/m1213_m1212_m1210_c1_r8_vcs_release_author_receipt_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    RUNNER: "0eb674169ad79730e41642b2b3c2b3e2571dfc42032f2e96ff8ff05f4b080049",
    CHECKER: "6a14b6a42236aa22abf0d07bee49b9210ab0d695c6d07b702e5871773f67a10d",
    CONTRACT: "c0b21358dad1faf540483fefedde8233fd777423bbb9257259ad10c92793495c",
    RELEASE: "131015e88821cd193bc6dfabb4e0c2f606da8a5f40ea8cfc347744e221e400d0",
    M1213_AUTHOR / "review.json": "a79852d55afee47584578f637fc0f0c40d4aad74aa60c10bfd64e4be2ea6c650",
    M1213_AUTHOR / "SHA256SUMS": "62d00656c239332ce2dd458446af74809ee9ebdf98a0d4f2fa04082ecec2bc23",
    M1213_AUTHOR / "SHA256SUMS.seal.sha256": "86489cbb9daa50b907c33ab20ccc389f50f6e47e4d086be5fed78a53f02bcf97",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
checks = 0


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           AssertionError("nonfinite " + token)))
    require(isinstance(value, dict), "JSON root")
    return value


def verify_seal(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and name not in rows,
                "safe unique member")
        rows[name] = digest
    actual = set()
    for root, dirs, files in os.walk(directory, followlinks=False):
        base = Path(root); dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
        for name in files:
            member = base / name; rel = member.relative_to(directory).as_posix()
            if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
                continue
            if stat.S_ISREG(member.lstat().st_mode): actual.add(rel)
    require(actual == set(rows), "complete recursive membership")
    for name, digest in rows.items(): require(sha(directory / name) == digest, "member SHA")


def sidecar(path: Path) -> None:
    manifest = Path(str(path) + ".sha256"); outer = Path(str(path) + ".sha256.seal.sha256")
    require(manifest.read_text().split() == [sha(path), path.name], "sidecar")
    require(outer.read_text().split() == [sha(manifest), manifest.name], "sidecar outer")


def semantic_review(review: dict, runner_sha: str, contract_sha: str, release_sha: str) -> None:
    require(review.get("schema") ==
            "m1214_m1213_m1210_c1_r8_vcs_release_source_hammer_review_r1_v1", "review schema")
    require(review.get("status") ==
            "PASS_M1214_M1213_C1_R8_ACYCLIC_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH", "review status")
    require(review.get("verdict") == "GO" and review.get("score", 0) >= 95, "review verdict")
    require(review.get("issue_counts") == {"P0": 0, "P1": 0, "P2": 0}, "review issues")
    identity = review.get("identity", {})
    require(identity.get("runner_sha256") == runner_sha and
            identity.get("source_contract_sha256") == contract_sha and
            identity.get("release_sha256") == release_sha, "review identity")
    forbidden = {"hammer_manifest_sha256", "hammer_outer_seal_file_sha256",
                 "manifest_sha256", "outer_seal_file_sha256"}
    require(not forbidden.intersection(identity), "acyclic review identity")
    require(review.get("authorization") ==
            {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}, "authorization")


def main() -> None:
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "exact identity " + str(path))
    for directory in (M1210_AUTHOR, M1212, M1213_AUTHOR): verify_seal(directory)
    for path in (CONTRACT, RELEASE): sidecar(path)
    contract = strict_json(CONTRACT); release = strict_json(RELEASE)
    runner = RUNNER.read_text()
    ast.parse(CHECKER.read_text())
    require(contract["identity"]["runner_sha256"] == sha(RUNNER) and
            release["identity"]["runner_sha256"] == sha(RUNNER), "runner binding")
    require(release["identity"]["source_contract_sha256"] == sha(CONTRACT), "contract binding")
    require(release["required_environment"] == {
        "M1213_EXPECTED_RELEASE_SHA256": "SHA256_OF_THIS_EXACT_RELEASE_JSON",
        "M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256": "SHA256_OF_FRESH_M1214_REVIEW_JSON",
        "M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256": "SHA256_OF_FRESH_M1214_SHA256SUMS",
        "M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256":
            "SHA256_OF_FRESH_M1214_SHA256SUMS_SEAL_FILE"}, "four runtime digests")
    seal = runner.index('verify_recursive_seal "${RELEASE_HAMMER}"')
    review = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256}"')
    manifest = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256}"')
    outer = runner.index('sha_exact "${M1213_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256}"')
    semantic = runner.index("x['schema']=='m1214_m1213_m1210_c1_r8_vcs_release_source_hammer_review_r1_v1'")
    freshness = runner.index('[[ ! -e "${ATTEMPT}"')
    collision = runner.index("blocked={'vcs'")
    attempt = runner.index('/bin/mkdir -- "${ATTEMPT}"')
    compile_call = runner.index('"${VCS_BIN}" -full64')
    sim_call = runner.index('./simv -no_save')
    require(seal < review < manifest < outer < semantic < freshness < collision < attempt < compile_call < sim_call,
            "all gates precede attempt and execution")
    require(runner.count('"${VCS_BIN}" -full64') == 1 and
            runner.count('./simv -no_save') == 1, "exact one compile and sim")
    require("1800s ./simv" in runner and "--kill-after=30s" in runner, "bounded simulation")
    require("automatic_retry=false" in runner and
            runner.count('"${VCS_BIN}" -full64') == 1 and
            runner.count('./simv -no_save') == 1 and
            "retry_compile" not in runner and "retry_sim" not in runner,
            "no execution retry path")
    require("FAILED_OR_INCOMPLETE" in runner and "quarantine" in runner and
            'seal_dir "${WORK}"' in runner, "failure isolation")
    attempt_path = HW / release["unique_attempt"]["attempt_path"]
    result_path = HW / release["unique_attempt"]["result_path"]
    require(not attempt_path.exists() and not result_path.exists(), "fresh attempt/result")
    require(not list((HW / "results").glob(".m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_work.*")), "fresh work")
    require(not list((HW / "results").glob("m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830.failed_or_incomplete.*")), "fresh quarantine")

    base = {"schema": "m1214_m1213_m1210_c1_r8_vcs_release_source_hammer_review_r1_v1",
            "status": "PASS_M1214_M1213_C1_R8_ACYCLIC_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH",
            "verdict": "GO", "score": 98, "issue_counts": {"P0": 0, "P1": 0, "P2": 0},
            "identity": {"runner_sha256": sha(RUNNER), "source_contract_sha256": sha(CONTRACT),
                         "release_sha256": sha(RELEASE)},
            "authorization": {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}}
    semantic_review(base, sha(RUNNER), sha(CONTRACT), sha(RELEASE))
    mutations = 0
    for mutate in (
        lambda x: x.update(status="FAIL"),
        lambda x: x["identity"].update(runner_sha256="0" * 64),
        lambda x: x["identity"].update(manifest_sha256="0" * 64),
        lambda x: x["authorization"].update(vcs_compiles=2),
        lambda x: x["issue_counts"].update(P1=1),
        lambda x: x.update(score=94),
    ):
        bad = copy.deepcopy(base); mutate(bad)
        try: semantic_review(bad, sha(RUNNER), sha(CONTRACT), sha(RELEASE))
        except AssertionError: mutations += 1
    require(mutations == 6, "all semantic mutations rejected")
    print(json.dumps({"schema": "m1214_independent_release_hammer_mechanical_r1_v1",
        "status": "PASS", "checks": checks, "mutations_rejected": mutations,
        "recursive_seals": 3, "sidecars": 2, "fresh_namespace": True,
        "one_compile_one_bounded_sim": True, "automatic_retry": False,
        "vcs_runs": 0, "simv_runs": 0, "all_eda_runs": 0,
        "network_runs": 0, "gpu_runs": 0, "docs359_sha256": sha(DOCS359)},
        indent=2, sort_keys=True))


if __name__ == "__main__": main()
