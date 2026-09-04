#!/usr/bin/env python3
"""M1656 source-only successor for one decoder actual-prefix run.

M1646 rejected M1645's private execution gate because two path-existence
checks could reach payload selection.  This successor reuses exact M1645 for
the fixed D0 prefix and all scheduling/miters, but places an exact, recursively
sealed M1646 review check and a semantic, double-sealed M1657/M1658 authority
check before any payload selection/open, RSS construction, attempt or run.

This authoring revision is inert.  It does not open payload, sample payload-run
RSS, create an attempt/result, execute the prefix, or admit cycles/bytes.
Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = Path(__file__).resolve()
TEST = HW / (
    "system_simulator/tests/test_m1656_decoder_actual_prefix_"
    "authorization_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1656_decoder_actual_prefix_authorization_successor_"
    "source_contract_r1_20260901.json")
M1645_SOURCE = HERE / (
    "build_m1645_decoder_compact_actual_prefix_runner_source.py")
M1645_TEST = HW / (
    "system_simulator/tests/"
    "test_m1645_decoder_compact_actual_prefix_runner_source.py")
M1645_CONTRACT = HW / (
    "contracts/m1645_decoder_compact_actual_prefix_runner_"
    "source_contract_r1_20260901.json")
M1646 = HW / (
    "reviews/m1646_m1645_decoder_compact_actual_prefix_runner_source_"
    "independent_review_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

FUTURE_REVIEW = HW / (
    "reviews/m1657_m1656_decoder_actual_prefix_authorization_successor_"
    "source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1658_m1657_m1656_decoder_actual_prefix_one_shot_"
    "release_r1_20260901.json")
RESULT = HW / (
    "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901")
ATTEMPT = HW / (
    "results/.m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.attempt_consumed")
WORK = HW / (
    "results/.m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.work")
FAILURE = HW / (
    "results/m1656_decoder_d0_call0_actual_prefix_three_configuration_"
    "r1_20260901.failed_no_retry")

SCHEMA = "m1656_decoder_actual_prefix_authorization_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M1646_P1_REPAIRED__DIFFERENT_AUTHOR_REVIEW_REQUIRED__"
    "NO_PAYLOAD_NO_EXECUTION")
REVIEW_STATUS = (
    "PASS_M1657_M1656_DECODER_ACTUAL_PREFIX_AUTHORIZATION_SUCCESSOR_"
    "SOURCE__AUTHORIZE_RELEASE_AUTHORING__NO_EXECUTION")
RELEASE_SCHEMA = (
    "m1658_m1657_m1656_decoder_actual_prefix_one_shot_release_r1_v1")
RELEASE_STATUS = (
    "AUTHORIZE_ONE_M1656_D0_CALL0_ACTUAL_PREFIX_THREE_CONFIGURATION_RUN")
ATTEMPT_TOKEN = (
    "M1656_ATTEMPT_CONSUMED__D0_CALL0_DEST0_41__THREE_CONFIGURATIONS__"
    "AUTOMATIC_RETRY_FALSE\n")

M1645_SOURCE_SHA256 = (
    "0869bed30edbae34ed4d58a0959fa7f70962c3b78b383c80bbd96e4782e7d833")
M1645_TEST_SHA256 = (
    "bf0796b01da592b4e206ac3dee48773a325aeed9da70c7dd360d6067e53f48d8")
M1645_CONTRACT_SHA256 = (
    "8beeebe22bdb9d22c2032450dd79fb1578351fb11c55039bc5a533062912f957")
M1646_REVIEW_SHA256 = (
    "95d7b61fe7fd49c241fc20bd9561eec5b4e2d0bb4a2402b6d8b99e6609fbd81d")
M1646_MANIFEST_SHA256 = (
    "e516a35fe13d1e52df1aa1916f57c4f03a23095bd2aad28f14c5eb082ee4d523")
M1646_OUTER_FILE_SHA256 = (
    "20a23ef4cdefe493518f7db90f726c821052b89de6a4a812c9e77c147728b3d3")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")


class M1656Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1656Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1656Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1656Error("nonfinite JSON: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _safe_member(root, name):
    require(type(name) is str and name and "\\" not in name,
            "unsafe sealed member")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "unsafe sealed member path")
    path = root / relative
    require(str(path.resolve()).startswith(str(root.resolve()) + os.sep),
            "sealed member escapes tree")
    return path


def verify_tree(root, expected_review_sha=None, expected_manifest_sha=None,
                expected_outer_file_sha=None, label="review"):
    root = Path(root)
    try:
        mode = root.lstat().st_mode
    except OSError as error:
        raise M1656Error("missing " + label + " tree") from error
    require(stat.S_ISDIR(mode) and not root.is_symlink(),
            label + " tree must be directory non-symlink")
    review = root / "review.json"
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    for item, item_label in ((review, " review"),
                             (manifest, " manifest"),
                             (outer, " outer")):
        try:
            item_mode = item.lstat().st_mode
        except OSError as error:
            raise M1656Error("missing " + label + item_label) from error
        require(stat.S_ISREG(item_mode) and not item.is_symlink(),
                label + item_label + " must be regular non-symlink")
    if expected_review_sha is None:
        expected_review_sha = sha256(review)
    if expected_manifest_sha is None:
        expected_manifest_sha = sha256(manifest)
    if expected_outer_file_sha is None:
        expected_outer_file_sha = sha256(outer)
    regular_exact(review, expected_review_sha, label + " review")
    regular_exact(manifest, expected_manifest_sha, label + " manifest")
    regular_exact(outer, expected_outer_file_sha, label + " outer")
    require(outer.read_text(encoding="ascii") ==
            expected_manifest_sha + "  SHA256SUMS\n",
            label + " outer content drift")
    seen = set()
    sealed_review = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                label + " malformed manifest member")
        digest, name = fields
        require(name not in seen, label + " duplicate manifest member")
        seen.add(name)
        member = _safe_member(root, name)
        regular_exact(member, digest, label + " member " + name)
        if member.resolve() == review.resolve():
            require(digest == expected_review_sha,
                    label + " review not bound by manifest")
            sealed_review = True
    require(sealed_review, label + " review absent from manifest")
    return strict_json(review), expected_manifest_sha, expected_outer_file_sha


def verify_file_double_seal(path, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1656Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for item, item_label in ((sidecar, " sidecar"), (outer, " outer")):
        try:
            item_mode = item.lstat().st_mode
        except OSError as error:
            raise M1656Error("missing " + label + item_label) from error
        require(stat.S_ISREG(item_mode) and not item.is_symlink(),
                label + item_label + " must be regular non-symlink")
    require(sidecar.read_text(encoding="ascii") ==
            sha256(path) + "  " + path.name + "\n" and
            outer.read_text(encoding="ascii") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            label + " double seal drift")
    return sha256(path), sha256(sidecar), sha256(outer)


def load_m1645():
    regular_exact(M1645_SOURCE, M1645_SOURCE_SHA256, "exact M1645 source")
    spec = importlib.util.spec_from_file_location("m1656_exact_m1645",
                                                  str(M1645_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1645")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regular_exact(M1645_SOURCE, M1645_SOURCE_SHA256,
                  "exact M1645 source after import")
    return module


P = load_m1645()


def verify_m1646_no_go_and_disposition():
    review, manifest_sha, outer_sha = verify_tree(
        M1646, M1646_REVIEW_SHA256, M1646_MANIFEST_SHA256,
        M1646_OUTER_FILE_SHA256, "M1646")
    findings = review.get("findings", {})
    p1 = findings.get("p1", [])
    authorization = review.get("authorization", {})
    require(review.get("status") ==
            "NO_GO_M1645_ACTUAL_PREFIX_EXECUTION_RELEASE__ONE_P1_PRESENCE_ONLY_AUTHORIZATION_GATE__SUCCESSOR_SOURCE_REPAIR_ONLY" and
            review.get("verdict") == "NO_GO" and
            review.get("p0_count") == 0 and
            review.get("p1_count") == 1 and
            review.get("p2_count") == 0 and
            type(p1) is list and len(p1) == 1 and
            p1[0].get("id") ==
                "P1_PRESENCE_ONLY_PRIVATE_EXECUTION_AUTHORIZATION" and
            "separately named source-only successor" in
                p1[0].get("required_repair", "") and
            authorization.get("successor_source_repair") is True and
            authorization.get("m1645_execution") is False and
            authorization.get("m1647_release") is False and
            authorization.get("actual_payload") is False and
            authorization.get("actual_prefix") is False and
            review.get("single_next_action", "").startswith(
                "Author a separately named source-only successor"),
            "M1646 status/P1 disposition drift")
    return {"review_sha256": M1646_REVIEW_SHA256,
            "manifest_sha256": manifest_sha,
            "outer_file_sha256": outer_sha,
            "p1_id": p1[0]["id"], "successor_source_repair": True}


def validate_source_contract():
    row = strict_json(SOURCE_CONTRACT)
    require(row.get("schema") == SCHEMA and row.get("status") == STATUS and
            row.get("source") == {"path": str(SOURCE.relative_to(HW)),
                                  "sha256": sha256(SOURCE)} and
            row.get("test") == {"path": str(TEST.relative_to(HW)),
                                "sha256": sha256(TEST)},
            "M1656 source contract identity drift")
    auth = row.get("authorization", {})
    require(auth.get("different_author_review") is True and
            auth.get("release_authoring") is False and
            auth.get("payload") is False and
            auth.get("execution") is False and
            auth.get("attempt_creation") is False,
            "M1656 source contract authorizes runtime")
    return row


def _review_identity():
    return {
        "source_sha256": sha256(SOURCE),
        "test_sha256": sha256(TEST),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
        "m1645_source_sha256": M1645_SOURCE_SHA256,
        "m1645_test_sha256": M1645_TEST_SHA256,
        "m1645_contract_sha256": M1645_CONTRACT_SHA256,
        "m1646_review_sha256": M1646_REVIEW_SHA256,
        "m1646_manifest_sha256": M1646_MANIFEST_SHA256,
        "m1646_outer_file_sha256": M1646_OUTER_FILE_SHA256,
        "checkpoint_sha256": P.CHECKPOINT_SHA256,
        "resource_manifest_sha256": P.RESOURCE_SHA256,
        "docs359_sha256": DOCS359_SHA256,
    }


def _fixed_population():
    return {"decoder_stage": "D0", "call_ordinal": 0,
            "module_ordinal": 0, "timestep": 0,
            "destinations": list(range(42)),
            "output_blocks": [0, 1, 2, 3],
            "configuration_order": list(P.CONFIGS)}


def _namespaces():
    return {"result": str(RESULT.relative_to(HW)),
            "attempt": str(ATTEMPT.relative_to(HW)),
            "work": str(WORK.relative_to(HW)),
            "failure": str(FAILURE.relative_to(HW))}


def validate_future_review_and_release():
    review, review_manifest_sha, review_outer_sha = verify_tree(
        FUTURE_REVIEW, label="M1657")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == _review_identity() and
            review.get("authorization") == {
                "release_authoring": True, "execution": False,
                "payload": False, "automatic_retry": False},
            "M1657 review semantic authority drift")
    release_sha, _sidecar_sha, _outer_sha = verify_file_double_seal(
        FUTURE_RELEASE, "M1658 release")
    release = strict_json(FUTURE_RELEASE)
    release_identity = dict(_review_identity(),
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=review_manifest_sha,
        review_outer_file_sha256=review_outer_sha)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == release_identity and
            release.get("authorization") == {
                "actual_prefix_runs": 1, "payload_opens": 1,
                "attempt_writes": 1, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            release.get("namespaces") == _namespaces() and
            release.get("fixed_population") == _fixed_population() and
            release.get("claim_boundary") == {
                "prefix_only": True, "cycles_pending_hammer": True,
                "bytes_pending_hammer": True, "product_capture": False,
                "l3": False, "full_decoder": False,
                "production": False, "paper_result": False},
            "M1658 release schema/status/identity/authorization/namespace drift")
    return release, release_sha


def require_fresh_namespaces():
    paths = (RESULT, ATTEMPT, WORK, FAILURE)
    require(len(set(paths)) == 4 and
            all("m1656_" in path.name for path in paths),
            "M1656 namespace identity drift")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1656 result/attempt/work/failure namespace not fresh")


def verify_pre_payload_authorities(require_future):
    """The mandatory first runtime gate; performs no payload or RSS action."""
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1645_SOURCE, M1645_SOURCE_SHA256, "exact M1645 source")
    regular_exact(M1645_TEST, M1645_TEST_SHA256, "exact M1645 test")
    regular_exact(M1645_CONTRACT, M1645_CONTRACT_SHA256,
                  "exact M1645 contract")
    disposition = verify_m1646_no_go_and_disposition()
    validate_source_contract()
    require_fresh_namespaces()
    release = None
    release_sha = None
    if require_future:
        release, release_sha = validate_future_review_and_release()
    else:
        require(not os.path.lexists(str(FUTURE_REVIEW)) and
                not os.path.lexists(str(FUTURE_RELEASE)) and
                not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                             ".sha256"))) and
                not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                             ".sha256.seal.sha256"))),
                "future M1657/M1658 authority must be absent at source stage")
    return {"m1646": disposition, "release": release,
            "release_sha256": release_sha}


def consume_attempt(release_sha):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(ATTEMPT), flags, 0o400)
    try:
        value = (ATTEMPT_TOKEN + "release_sha256=" + release_sha +
                 "\nsource_sha256=" + sha256(SOURCE) + "\n")
        os.write(descriptor, value.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_result(root):
    members = sorted(path.relative_to(root) for path in root.rglob("*")
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(root / member), member.as_posix()) for member in members),
        encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")


def _run_authorized_actual_prefix():
    """Future one-shot target. Authority validation is the first call."""
    authority = verify_pre_payload_authorities(require_future=True)
    consume_attempt(authority["release_sha256"])
    WORK.mkdir(mode=0o700)
    published = False
    try:
        # Exact predecessor runtime validation remains after the repaired gate.
        P.R.validate_authorities(True)
        path, shape, payload_sha = P._selected_payload()
        rss = P.RssGate()
        plane = P.ImmutablePrefixPlane(path, shape, payload_sha)
        rss.sample()
        receipts = []
        metrics = []
        for configuration in P.CONFIGS:
            receipt, metric = P._schedule_prefix(configuration, plane, rss)
            receipts.append(receipt)
            metrics.append(metric)
        P.L2.validate_bundle(receipts)
        rss.sample()
        row = {"schema": "m1656_decoder_actual_prefix_result_r1_v1",
            "status": "PREFIX_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED",
            "source_sha256": sha256(SOURCE),
            "release_sha256": authority["release_sha256"],
            "checkpoint": P.CHECKPOINT,
            "checkpoint_sha256": P.CHECKPOINT_SHA256,
            "resource_manifest_sha256": P.RESOURCE_SHA256,
            "fixed_population": _fixed_population(),
            "sessions": [receipt.as_dict() for receipt in receipts],
            "metrics": metrics, "rss": rss.summary(),
            "payload_fd_sha256": plane.opened_sha256,
            "payload_fd_size": plane.opened_size,
            "independent_hammer_pending": True,
            "cycles_pending_hammer": True, "bytes_pending_hammer": True,
            "product_capture": False, "l3": False,
            "full_decoder": False, "production": False,
            "paper_result": False}
        (WORK / "result.json").write_text(json.dumps(
            row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        seal_result(WORK)
        WORK.rename(RESULT)
        published = True
        return row
    except BaseException:
        if WORK.is_dir() and not os.path.lexists(str(FAILURE)):
            WORK.rename(FAILURE)
        raise
    finally:
        require(published or not WORK.exists(),
                "failed M1656 work namespace was not quarantined")


@contextlib.contextmanager
def _source_only_m1645_gate_binding():
    old_review, old_release = P.FUTURE_REVIEW, P.FUTURE_RELEASE
    try:
        P.FUTURE_REVIEW = FUTURE_REVIEW
        P.FUTURE_RELEASE = FUTURE_RELEASE
        yield
    finally:
        P.FUTURE_REVIEW, P.FUTURE_RELEASE = old_review, old_release


def static_self_test():
    preflight = verify_pre_payload_authorities(require_future=False)
    with _source_only_m1645_gate_binding():
        synthetic = P.static_self_test()
    require(synthetic.get("distinct_sessions") == 3 and
            synthetic.get("configurations") == list(P.CONFIGS) and
            all(row.get("kind_counts", {}).get("commit") == 168
                for row in synthetic.get("metrics", [])) and
            synthetic.get("actual_payload") is False and
            synthetic.get("actual_execution") is False,
            "exact M1645 synthetic invariant drift")
    return {"schema": SCHEMA,
            "status": "PASS_M1656_AUTHORIZATION_SUCCESSOR_SOURCE_STATIC_ONLY",
            "m1646": preflight["m1646"],
            "fixed_population": _fixed_population(),
            "configurations": list(P.CONFIGS), "distinct_sessions": 3,
            "commits_per_configuration": 168,
            "rss_absolute_limit_kib": P.RSS_ABSOLUTE_LIMIT_KIB,
            "rss_increment_limit_kib": P.RSS_INCREMENT_LIMIT_KIB,
            "future_review_present": False,
            "future_release_present": False,
            "actual_payload": False, "actual_execution": False,
            "attempt_writes": 0, "cycles_admitted": False,
            "bytes_admitted": False, "gpu": False, "eda": False,
            "paper_result": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "fixed_population": _fixed_population(),
            "exact_reuse": {"m1645_source_sha256": M1645_SOURCE_SHA256,
                "m1646_review_sha256": M1646_REVIEW_SHA256,
                "per_request_miter": "M1638 accept_request_pair",
                "per_destination_miter": "M1638 accept_destination_pair",
                "distinct_sessions": 3,
                "rss_absolute_limit_kib": P.RSS_ABSOLUTE_LIMIT_KIB,
                "rss_increment_limit_kib": P.RSS_INCREMENT_LIMIT_KIB},
            "namespaces": _namespaces(),
            "future_gate": {"review": str(FUTURE_REVIEW.relative_to(HW)),
                "release": str(FUTURE_RELEASE.relative_to(HW)),
                "review_present": os.path.lexists(str(FUTURE_REVIEW)),
                "release_present": os.path.lexists(str(FUTURE_RELEASE))},
            "authorization": {"source_only": True,
                "different_author_review": True,
                "release_authoring": False, "payload": False,
                "execution": False, "attempt_creation": False,
                "automatic_retry": False, "cycles": False,
                "traffic": False, "energy": False, "speedup": False,
                "gpu": False, "rtl": False, "eda": False,
                "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        verify_pre_payload_authorities(require_future=False)
        output = describe()
    elif args.synthetic_self_test:
        output = static_self_test()
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
