#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1116 read-only audit of the M1112r2 future launch hash cycle."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1112r2_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = HW / "contracts/m1112r2_c2_async_observation_shadow_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1112r2_c2_async_observation_source_receipt_r1_20260830"
M1114R2 = HW / "reviews/m1114r2_m1112r2_c2_async_observation_engine_hammer_r1_20260830"
OLD_CORRECT = HW / "dc_handoff/scripts/m1091r3_m1090r3_c2_observation_authorized_engine_r1.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE / "mechanical_checks.json"
EXPECTED = {
    "engine": "cd4f3eb4d9c659b14fca143651b2e5a4c0d3147335469b9ec22063b1113980c4",
    "contract": "0f378e5d6100c2d9ae30fcc15a3e3cad53f2fb2d4aa51583c4e53935014b677d",
    "contract_outer": "b2670f2a1f4742235d013f8f7e954db84d80ae2de6d4f6a13e0273e6e10817fa",
    "author_outer": "bafe08fe786b7e51b8f064786ffeb02aa164af39e24674f48ccacaadc0ece2de",
    "m1114_review": "6c162460b25bc9c24eba3b4d697b982b28f504d47567c6f189f5abccb2a2f6a4",
    "m1114_manifest": "6c9b41b242e252e6bc9369922a3fa2e204387d6d9a288444775593bca2990f4a",
    "m1114_outer": "15e1f136aa4d892a965a005f97a5845d81a144634f86c9db432bfdf4bec884a9",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and
            sha(path) == expected, "identity drift " + str(path))


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_sha: str, status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "sealed directory drift")
    regular(directory / "review.json", review_sha)
    regular(directory / "SHA256SUMS", manifest_sha)
    regular(directory / "SHA256SUMS.seal.sha256", outer_sha)
    require((directory / "SHA256SUMS.seal.sha256").read_text().split() ==
            [manifest_sha, "SHA256SUMS"], "outer content drift")
    expected = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(name not in expected and "/" not in name and ".." not in Path(name).parts,
                "non-flat/duplicate manifest")
        regular(directory / name, digest); expected[name] = digest
    actual = {path.name for path in directory.iterdir() if path.is_file() and
              path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(expected), "flat coverage drift")
    require(json.loads((directory / "review.json").read_text())["status"] == status,
            "review status drift")


regular(ENGINE, EXPECTED["engine"])
regular(CONTRACT, EXPECTED["contract"])
regular(Path(str(CONTRACT) + ".sha256.seal.sha256"), EXPECTED["contract_outer"])
regular(AUTHOR / "SHA256SUMS.seal.sha256", EXPECTED["author_outer"])
verify_flat(M1114R2, EXPECTED["m1114_review"], EXPECTED["m1114_manifest"],
            EXPECTED["m1114_outer"],
            "PASS_M1114R2_M1112R2_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA")
regular(DOCS359, EXPECTED["docs359"])

source = ENGINE.read_text(encoding="utf-8")
ast.parse(source)
require('receipt["m1115r2_outer_seal_file_sha256"]' in source and
        'verify_exact_flat(M1115R2, receipt["m1115r2_outer_seal_file_sha256"])' in source and
        'launch_review["identity"]["launch_receipt_outer_seal_file_sha256"] != receipt_outer' in source,
        "future hash-cycle edges not found")
old = OLD_CORRECT.read_text(encoding="utf-8")
require("m1096r2_outer = verify_flat_self_consistent(M1096)" in old and
        'receipt["m1096r2_outer_seal_file_sha256"] = m1096r2_outer' in old,
        "known acyclic predecessor pattern drift")

launcher = HW / "dc_handoff/scripts/run_m1112r2_c2_async_observation_authorized_launch_r1.py"
receipt = HW / "contracts/m1112r2_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
attempt = HW / "results/.m1112r2_c2_async_observation_dc_mapped_vcs_attempt_consumed"
result = HW / "results/m1112r2_c2_async_observation_dc_mapped_vcs_r1_20260830"
require(not any(path.exists() or path.is_symlink() for path in (launcher, receipt, attempt, result)),
        "r2 future/canonical namespace unexpectedly exists")

output = {
    "schema": "m1116_m1112r2_c2_launch_chain_circularity_audit_v1",
    "status": "STOP_M1116_M1112R2_FUTURE_LAUNCH_HASH_CYCLE__ADDITIVE_R3_REQUIRED",
    "score": 70,
    "identity": EXPECTED,
    "p0": {"id": "M1116-P0-01",
        "title": "Launch receipt and future M1115r2 sealed review form an unconstructable SHA256 fixed point",
        "edge_receipt_to_future_hammer_outer": True,
        "edge_future_hammer_review_to_receipt_outer": True,
        "practical_fixed_point_constructible": False,
        "m1114r2_static_engine_findings_still_valid": True,
        "m1114r2_launcher_authoring_go_withdrawn": True},
    "required_repair": {"additive_m1112r3": True,
        "launch_receipt_binds_existing_authorities_only": True,
        "future_launch_hammer_outer_discovered_self_consistently_at_execution": True,
        "future_hammer_review_binds_exact_launcher_and_launch_receipt_outer": True,
        "placeholder_or_hash_cycle_allowed": False},
    "execution": {"launcher_created": False, "launch_receipt_created": False,
        "engine_executed": False, "eda": False, "attempt_created": False,
        "result_created": False, "docs359_modified": False},
    "claim_boundary": {"source_audit_only": True, "mapped_functionality": False,
        "timing": False, "power": False, "performance": False,
        "paper_citable": False}
}
OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output["status"])
