#!/usr/bin/env python3
"""Fail-closed pre-attempt gate for the additive M1187/R4 VCS launcher."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise AssertionError(f"duplicate key {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_recursive(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        p = Path(name)
        assert name not in listed and not p.is_absolute() and ".." not in p.parts
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
            continue
        if stat.S_ISREG(member.lstat().st_mode):
            actual.add(rel)
    assert actual == set(listed), (actual - set(listed), set(listed) - actual)
    for name, digest in listed.items():
        assert sha(directory / name) == digest


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    assert len(value) == 64 and all(c in "0123456789abcdef" for c in value), name
    return value


def main() -> None:
    assert len(sys.argv) == 8, "contract runner source_hammer release release_hammer source_author r2_quarantine"
    contract, runner, source_hammer, release, release_hammer, source_author, r2_q = map(Path, sys.argv[1:])
    expected_release = required_env("M1187_EXPECTED_RELEASE_SHA256")
    expected_source_review = required_env("M1187_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256")
    expected_source_outer = required_env("M1187_EXPECTED_SOURCE_HAMMER_OUTER_SHA256")
    expected_release_review = required_env("M1187_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256")
    expected_release_outer = required_env("M1187_EXPECTED_RELEASE_HAMMER_OUTER_SHA256")

    for directory in (source_hammer, release_hammer, source_author, r2_q):
        verify_recursive(directory)
    assert sha(release) == expected_release
    assert sha(source_hammer / "review.json") == expected_source_review
    assert sha(source_hammer / "SHA256SUMS.seal.sha256") == expected_source_outer
    assert sha(release_hammer / "review.json") == expected_release_review
    assert sha(release_hammer / "SHA256SUMS.seal.sha256") == expected_release_outer

    c = strict_json(contract)
    r = strict_json(release)
    sh = strict_json(source_hammer / "review.json")
    rh = strict_json(release_hammer / "review.json")
    assert c["status"] == "M1187_R4_SOURCE_READY__FRESH_RELEASE_HAMMER_REQUIRED__NO_VCS_NO_EDA"
    assert r["status"] == "AUTHORIZE_ONE_M1187_R4_FUNCTIONAL_VCS_ATTEMPT_AFTER_FRESH_RELEASE_HAMMER"
    ident = r["identity"]
    assert ident["runner_sha256"] == sha(runner)
    assert ident["source_contract_sha256"] == sha(contract)
    assert ident["source_hammer_review_sha256"] == sha(source_hammer / "review.json")
    assert ident["source_hammer_outer_seal_file_sha256"] == sha(source_hammer / "SHA256SUMS.seal.sha256")
    assert sh["status"] == "PASS_M1182_M1181_M1168R3_VCS_SOURCE_HAMMER__AUTHORIZE_RELEASE"
    assert sh["verdict"] == "GO" and sh["issue_counts"]["P0"] == 0 and sh["issue_counts"]["P1"] == 0

    assert rh["schema"] == "m1188_m1187_m1168r3_c1_vcs_release_hammer_review_r1_v1"
    assert rh["status"] == "PASS_M1188_M1187_C1_VCS_RELEASE_HAMMER__AUTHORIZE_ONE_LAUNCH"
    assert rh["verdict"] == "GO" and rh["score"] >= 95
    assert rh["issue_counts"]["P0"] == 0 and rh["issue_counts"]["P1"] == 0
    assert rh["identity"]["release_sha256"] == sha(release)
    assert rh["identity"]["runner_sha256"] == sha(runner)
    assert rh["identity"]["source_contract_sha256"] == sha(contract)
    assert rh["execution_audit"]["vcs_compiles"] == 0
    assert rh["execution_audit"]["simv_runs"] == 0
    assert rh["execution_audit"]["all_eda_runs"] == 0
    assert rh["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}

    assert r["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}
    assert r["fresh_release_hammer"]["path"] == str(release_hammer)
    assert r["fresh_release_hammer"]["runtime_review_sha_env"] == "M1187_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256"
    assert r["fresh_release_hammer"]["runtime_outer_sha_env"] == "M1187_EXPECTED_RELEASE_HAMMER_OUTER_SHA256"
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "system_speedup", "paper_citable", "headline"):
        assert r["claim_boundary"][key] is False

    print("PASS_M1187_R4_PRE_ATTEMPT_GATE")


if __name__ == "__main__":
    main()
