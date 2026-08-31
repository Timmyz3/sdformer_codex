#!/usr/bin/env python3
"""M1521 additive canonical-manifest seal successor over immutable M1516.

M1517 proved that M1516's recursive seal authenticated a self-consistent file
set but not the canonical numeric/protocol semantics of ``manifest.json``.
M1521 closes that edge.  Both the pre-publication seal and the post-publication
verifier internally regenerate the complete expected manifest from the exact
canonical M1458 capture through the frozen M1510 audit and M1516 enrichment.
Every JSON type, field, list position, call identity, canonical output path,
scale/encoding/no-fold flag, claim boundary, and payload SHA must match.

The CLI remains source-only.  M1522 hammer and M1523 release are required for
the inert production hook; no production namespace is created here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE = Path(__file__).resolve()
HERE = SOURCE.parent
HW = HERE.parent.parent
ROOT = HW.parent
TEST = HW / "system_simulator/tests/test_m1521_ep34_decoder_canonical_manifest_seal_successor_source.py"
CONTRACT = HW / "contracts/m1521_ep34_decoder_canonical_manifest_seal_successor_source_contract_r1_20260831.json"
M1516_SOURCE = HERE / "build_m1516_ep34_decoder_positive_plane_materializer_source.py"
M1516_SOURCE_SHA256 = "b712e3246f1cca5ac857017439fc75a7bccc8a87e7e09763a19f0d50806b94ef"
M1516_TEST = HW / "system_simulator/tests/test_m1516_ep34_decoder_positive_plane_materializer_source.py"
M1516_TEST_SHA256 = "aa88c18b4b90c8f01e24053cf96044adda4677f49f340f9a143bdaa3a631cfe6"
M1516_CONTRACT = HW / "contracts/m1516_ep34_decoder_positive_plane_materializer_source_contract_r1_20260831.json"
M1516_CONTRACT_SHA256 = "f5e8536135afd3817305997068761816c840885828d9136fd18b6650b3c7c756"
M1517 = HW / "reviews/m1517_m1516_ep34_decoder_positive_plane_materializer_source_hammer_r1_20260831"
M1517_PINS = (
    "027d5b1a93ac60c39fa6b10fc49c8877e0ae18c15fbad2a65a862e52e8c45da9",
    "541afd3214225c1e566bc8858c02dcab469c134ea8e969ed7721ddc424223ccd",
    "4d3e10dde84f724d56c37751d28f23f6810ae7c911fdc0f495e619bca40f1404",
)
M1517_STATUS = "FAIL_M1517_M1516_SEAL_SEMANTIC_AUTHENTICATION__M1518_BLOCKED"
OUTPUT = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831"
ATTEMPT = HW / "results/.m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831.attempt_consumed"
WORK_PREFIX = ".m1521_ep34_decoder_positive_planes_work."
FUTURE_RELEASE = HW / "contracts/m1523_ep34_decoder_canonical_manifest_materializer_production_release_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

SCHEMA = "m1521_ep34_decoder_canonical_manifest_seal_successor_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1522_HAMMER_AND_M1523_RELEASE_REQUIRED__NO_PRODUCTION"
OUTPUT_SCHEMA = "m1521_ep34_decoder_positive_plane_materialization_r1_v1"
OUTPUT_STATUS = "CANONICAL_MATERIALIZATION_COMPLETE__ADDRESS_TIMED_REPLAY_NOT_RUN"
HAMMER_SCHEMA = "m1522_m1521_ep34_decoder_canonical_manifest_seal_source_hammer_r1_v1"
HAMMER_STATUS = "PASS_M1522_M1521_CANONICAL_MANIFEST_SEAL__M1523_RELEASE_ONLY"
RELEASE_SCHEMA = "m1523_ep34_decoder_canonical_manifest_materializer_production_release_r1_v1"
RELEASE_STATUS = "M1522_CANONICAL_SEAL_HAMMER_BOUND__ONE_M1521_MATERIALIZATION"
ATTEMPT_TOKEN = "M1521_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
RUN_TOKEN = "PASS_M1521_CANONICAL_EP34_DECODER_POSITIVE_PLANE_MATERIALIZATION\n"
EXPECTED_SEALED_MEMBERS = 122
CLAIM_BOUNDARY = {
    "source_only": True,
    "production": False,
    "canonical_manifest_regenerated": True,
    "preseal_full_tree_compare": True,
    "postpublication_full_tree_compare": True,
    "positive_plane_materialization": False,
    "negative_plane_output": False,
    "weight_folding": False,
    "normalization": False,
    "coercion": False,
    "address_timed_replay": False,
    "cycles": False,
    "traffic": False,
    "speedup": False,
    "system_speedup": False,
    "energy": False,
    "rtl": False,
    "eda": False,
    "ppa": False,
    "table_a": False,
}


class M1521Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise M1521Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = Path(path).lstat().st_mode
    except FileNotFoundError as error:
        raise M1521Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not Path(path).is_symlink(),
            label + " must be regular non-symlink")


def regular_exact(path: Path, digest: str, label: str) -> None:
    regular(path, label)
    require(sha256(path) == digest, label + " SHA drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1521Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root is not object")
    return value


def load_exact(name: str, path: Path, digest: str):
    regular_exact(path, digest, name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    regular_exact(path, digest, name + " after import")
    return module


M1516 = load_exact("m1521_frozen_m1516", M1516_SOURCE, M1516_SOURCE_SHA256)


def verify_m1517_failure() -> dict[str, Any]:
    review_sha, manifest_sha, outer_sha = M1517_PINS
    regular_exact(M1517 / "review.json", review_sha, "M1517 review")
    regular_exact(M1517 / M1516.MANIFEST, manifest_sha, "M1517 manifest")
    regular_exact(M1517 / M1516.OUTER, outer_sha, "M1517 outer")
    require((M1517 / M1516.OUTER).read_text().split() ==
            [manifest_sha, M1516.MANIFEST], "M1517 outer content drift")
    members: set[str] = set()
    for line in (M1517 / M1516.MANIFEST).read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "M1517 manifest row malformed")
        digest, relative = fields
        relative = relative.lstrip("*")
        require(relative not in members, "M1517 duplicate member")
        regular_exact(M1516.safe_member(M1517, relative, "M1517 member"), digest,
                      "M1517 member")
        members.add(relative)
    actual = {path.relative_to(M1517).as_posix() for path in M1517.rglob("*")
              if path.is_file() and path.relative_to(M1517).as_posix() not in
              {M1516.MANIFEST, M1516.OUTER}}
    require(actual == members, "M1517 sealed population drift")
    review = strict_json(M1517 / "review.json")
    require(review.get("status") == M1517_STATUS and
            review.get("p0", {}).get("id") ==
            "P0_SEAL_SELF_CONSISTENCY_IS_NOT_CANONICAL_SEMANTIC_AUTHENTICATION" and
            review.get("authorization") == {
                "m1518_release_authoring": False,
                "production_materialization": False},
            "M1517 failure authority drift")
    return review


def exact_tree_equal(observed: Any, expected: Any, path: str = "$" ) -> None:
    require(type(observed) is type(expected), path + " JSON type drift")
    if type(expected) is dict:
        require(set(observed) == set(expected), path + " JSON key set drift")
        for key in sorted(expected):
            exact_tree_equal(observed[key], expected[key], path + "." + key)
    elif type(expected) is list:
        require(len(observed) == len(expected), path + " JSON list length drift")
        for index, (left, right) in enumerate(zip(observed, expected)):
            exact_tree_equal(left, right, path + "[{}]".format(index))
    elif type(expected) is float:
        require(math.isfinite(observed) and observed == expected,
                path + " JSON float drift")
    else:
        require(observed == expected, path + " JSON value drift")


def expected_manifest_from_enriched(enriched: Mapping[str, Any]) -> dict[str, Any]:
    """Pure projection; production can reach it only after canonical derivation."""
    manifest = M1516.build_output_manifest(enriched)
    manifest["schema"] = OUTPUT_SCHEMA
    manifest["status"] = OUTPUT_STATUS
    manifest["m1521_canonical_seal"] = {
        "m1516_source_sha256": M1516_SOURCE_SHA256,
        "m1517_review_sha256": M1517_PINS[0],
        "expected_manifest_regenerated_from_m1458_m1510": True,
        "preseal_full_tree_compare": True,
        "postpublication_full_tree_compare": True,
    }
    return manifest


def derive_canonical_expected() -> tuple[dict[str, Any], dict[str, Any]]:
    """The only production source of expected manifest semantics."""
    audit = M1516.M1510.audit_capture(M1516.CAPTURE)
    enriched = M1516.enrich_audit(audit, M1516.CAPTURE)
    manifest = expected_manifest_from_enriched(enriched)
    return enriched, manifest


def _sealed_payload_files(root: Path) -> list[Path]:
    output = []
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "canonical seal refuses symlink")
        if path.is_file() and path.relative_to(root).as_posix() not in {
                M1516.MANIFEST, M1516.OUTER}:
            output.append(path)
    return output


def _verify_against_expected(root: Path,
                             expected_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Private test seam; production callers cannot supply expected semantics."""
    manifest_seal = root / M1516.MANIFEST
    outer = root / M1516.OUTER
    regular(manifest_seal, "canonical output manifest seal")
    regular(outer, "canonical output outer seal")
    require(outer.read_text().split() ==
            [sha256(manifest_seal), M1516.MANIFEST],
            "canonical output outer seal drift")
    rows: dict[str, str] = {}
    for line in manifest_seal.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "canonical output seal row malformed")
        digest, relative = fields
        relative = relative.lstrip("*")
        require(relative not in rows, "canonical output duplicate seal member")
        member = M1516.safe_member(root, relative, "canonical output member")
        require(sha256(member) == digest, "canonical output member SHA drift")
        rows[relative] = digest
    actual = {path.relative_to(root).as_posix()
              for path in _sealed_payload_files(root)}
    require(actual == set(rows) and len(rows) == EXPECTED_SEALED_MEMBERS,
            "canonical output sealed population drift")

    observed = strict_json(root / "manifest.json")
    exact_tree_equal(observed, dict(expected_manifest))
    records = expected_manifest.get("records")
    require(type(records) is list and len(records) == 120,
            "canonical expected records are not 120")
    expected_paths = []
    for ordinal, record in enumerate(records):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        canonical = M1516.output_plane_name(ordinal, sample, module)
        require(record.get("global_call_ordinal") == ordinal and
                record.get("global_sample_id") == sample and
                record.get("module_ordinal") == module and
                record.get("positive_output") == canonical,
                "canonical expected 30x4 path/order drift")
        require(record.get("layer_scale_word_uint32") ==
                M1516.EXPECTED_SCALE_WORDS[module] and
                record.get("numeric_encoding") == (
                    "bit_times_layer_constant" if module in (0, 1)
                    else "exact_binary") and
                record.get("weight_folding") is False and
                record.get("normalized") is False and
                record.get("coerced") is False and
                record.get("negative_plane_output") is None and
                record.get("negative_plane_all_zero") is True,
                "canonical expected scale/encoding/no-fold drift")
        require(rows.get(canonical) == record.get("positive_output_sha256"),
                "canonical expected payload SHA differs from seal")
        expected_paths.append(canonical)
    require(len(set(expected_paths)) == 120 and
            set(expected_paths) == {name for name in rows if name.startswith("payloads/")},
            "canonical 120 output path population drift")
    boundary = expected_manifest.get("claim_boundary")
    require(type(boundary) is dict and
            all(boundary.get(key) is False for key in (
                "address_timed_replay", "cycles", "traffic", "speedup",
                "system_speedup", "energy", "rtl", "eda", "ppa", "table_a")),
            "canonical expected performance boundary drift")
    return {"manifest_sha256": sha256(manifest_seal),
            "outer_file_sha256": sha256(outer), "members": len(rows),
            "full_tree_equal": True, "canonical_paths": 120}


def verify_materialized_seal(root: Path) -> dict[str, Any]:
    """Post-publication verifier: expected manifest is regenerated internally."""
    _audit, expected = derive_canonical_expected()
    return _verify_against_expected(Path(root), expected)


def _seal_against_expected(root: Path,
                           expected_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Private test seam used after its caller derived the canonical manifest."""
    require(root.is_dir() and not root.is_symlink() and
            not (root / M1516.MANIFEST).exists() and
            not (root / M1516.OUTER).exists(),
            "bad canonical staging seal target")
    observed = strict_json(root / "manifest.json")
    exact_tree_equal(observed, dict(expected_manifest))
    members = _sealed_payload_files(root)
    require(len(members) == EXPECTED_SEALED_MEMBERS,
            "canonical staging member population is not 122")
    M1516.fsync_dir(root / "payloads")
    lines = [sha256(path) + "  " + path.relative_to(root).as_posix()
             for path in members]
    M1516.write_exclusive(root / M1516.MANIFEST,
                          ("\n".join(lines) + "\n").encode(), 0o400)
    M1516.write_exclusive(root / M1516.OUTER,
                          (sha256(root / M1516.MANIFEST) + "  " +
                           M1516.MANIFEST + "\n").encode(), 0o400)
    M1516.fsync_dir(root)
    return _verify_against_expected(root, expected_manifest)


def seal_staging(root: Path) -> dict[str, Any]:
    """Pre-publication seal: expected manifest is regenerated internally."""
    _audit, expected = derive_canonical_expected()
    return _seal_against_expected(Path(root), expected)


def materialize_canonical_once(output: Path = OUTPUT,
                               attempt: Path = ATTEMPT) -> Path:
    """Future one-shot primitive; no external manifest/audit input is accepted."""
    audit, expected = derive_canonical_expected()
    M1516.namespace_fresh(output, attempt, WORK_PREFIX)
    M1516.consume_attempt(attempt)
    staging = output.parent / (WORK_PREFIX + str(os.getpid()) + "." + str(time.time_ns()))
    staging.mkdir(mode=0o700)
    (staging / "payloads").mkdir(mode=0o700)
    try:
        for call, record in zip(audit["calls"], expected["records"]):
            source = M1516.safe_member(
                M1516.CAPTURE, call["support_sign"], "canonical support/sign")
            destination = staging.joinpath(
                *PurePosixPath(record["positive_output"]).parts)
            M1516.copy_positive_plane_exclusive(
                source, destination, record["elements"], record["plane_bytes"],
                record["source_support_sign_sha256"],
                record["source_positive_plane_sha256"],
                record["source_negative_zero_plane_sha256"])
        M1516.write_exclusive(staging / "manifest.json",
                              (json.dumps(expected, indent=2, sort_keys=True,
                                          allow_nan=False) + "\n").encode(), 0o400)
        M1516.write_exclusive(staging / "RUN_COMPLETE.txt", RUN_TOKEN.encode(), 0o400)
        _seal_against_expected(staging, expected)
        M1516.rename_noreplace(staging, output)
        M1516.fsync_dir(output.parent)
        _verify_against_expected(output, expected)
    except BaseException:
        raise  # attempt and stage intentionally remain for forensics
    return output


def verify_m1522_hammer(entry: Any) -> dict[str, Any]:
    require(type(entry) is dict and set(entry) == {
        "path", "review_sha256", "manifest_sha256", "outer_file_sha256"},
            "M1522 hammer entry drift")
    relative = PurePosixPath(entry["path"])
    require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
            "M1522 hammer path unsafe")
    root = ROOT.joinpath(*relative.parts)
    require(root.parent == HW / "reviews", "M1522 hammer not directly under reviews")
    review = M1516.verify_sealed_review(root, (
        entry["review_sha256"], entry["manifest_sha256"],
        entry["outer_file_sha256"]), HAMMER_STATUS)
    require(review.get("schema") == HAMMER_SCHEMA and
            review.get("source_identity") == {
                "source_sha256": sha256(SOURCE), "test_sha256": sha256(TEST),
                "contract_sha256": sha256(CONTRACT)} and
            review.get("authorization") == {
                "m1523_release_authoring": True,
                "production_materialization": False},
            "M1522 hammer authority drift")
    return review


def validate_release_shape(release: Any) -> None:
    require(type(release) is dict and set(release) == {
        "schema", "status", "source_identity", "m1522_source_hammer",
        "one_shot", "output", "claim_boundary"},
            "M1523 release key set drift")
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS,
            "M1523 release schema/status drift")
    require(release.get("source_identity") == {
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
        "contract_path": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT)},
            "M1523 source identity drift")
    require(release.get("one_shot") == {
        "attempt_marker": str(ATTEMPT.relative_to(ROOT)),
        "automatic_retry": False, "maximum_materializations": 1,
        "failure_stage_preserved": True},
            "M1523 one-shot policy drift")
    require(release.get("output") == {
        "path": str(OUTPUT.relative_to(ROOT)), "positive_plane_files": 120,
        "negative_plane_files": 0, "sealed_members": 122,
        "atomic_no_replace": True, "canonical_manifest_required": True},
            "M1523 output policy drift")
    require(release.get("claim_boundary") == {
        "positive_plane_materialization": True,
        "address_timed_replay": False, "cycles": False, "traffic": False,
        "speedup": False, "system_speedup": False, "energy": False,
        "rtl": False, "eda": False, "ppa": False, "table_a": False},
            "M1523 claim boundary drift")


def execute_once(release_path: Path) -> Path:
    require(Path(release_path).resolve() == FUTURE_RELEASE.resolve(),
            "only canonical M1523 release path allowed")
    regular(release_path, "M1523 release")
    release = strict_json(release_path)
    validate_release_shape(release)
    verify_m1522_hammer(release["m1522_source_hammer"])
    M1516.verify_authorities()
    verify_m1517_failure()
    return materialize_canonical_once()


def validate_source_policy() -> dict[str, Any]:
    regular_exact(M1516_SOURCE, M1516_SOURCE_SHA256, "M1516 source")
    regular_exact(M1516_TEST, M1516_TEST_SHA256, "M1516 test")
    regular_exact(M1516_CONTRACT, M1516_CONTRACT_SHA256, "M1516 contract")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    policy = strict_json(CONTRACT)
    require(policy.get("schema") == SCHEMA and
            policy.get("status") == SOURCE_STATUS,
            "M1521 source policy schema/status drift")
    require(policy.get("source") == {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)} and
            policy.get("test") == {
                "path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST)},
            "M1521 source/test identity drift")
    require(policy.get("production_authorized") is False and
            policy.get("future_release") == str(FUTURE_RELEASE.relative_to(ROOT)) and
            policy.get("claim_boundary") == CLAIM_BOUNDARY,
            "M1521 production/future/claim boundary drift")
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    require(args.source_self_check,
            "M1521 is source-only; production materialization CLI is forbidden")
    validate_source_policy()
    verify_m1517_failure()
    M1516.verify_authorities()
    require(not OUTPUT.exists() and not ATTEMPT.exists() and
            not any(OUTPUT.parent.glob(WORK_PREFIX + "*")) and
            not FUTURE_RELEASE.exists(),
            "M1521 production/release namespace already exists")
    print("PASS_M1521_SOURCE_SELF_CHECK__NO_CAPTURE_READ_NO_MATERIALIZATION")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1521Error as error:
        print("M1521_FAIL_CLOSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
