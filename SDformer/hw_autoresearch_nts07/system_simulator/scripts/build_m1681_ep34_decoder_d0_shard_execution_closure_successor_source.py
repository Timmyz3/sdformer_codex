#!/usr/bin/env python3
"""M1681 source-only execution closure for the frozen M1671 D0 grid.

M1672 accepted the M1671 grid and request/destination miters but rejected its
declarative execution boundary.  This successor changes neither grid nor
scheduler.  It adds the minimum executable closure required for a later
release: fixed per-shard namespaces, attempt consumption before the first
payload access, immutable opened-FD/hash binding, atomic sealed publication,
read-only resume verification and a reducer which accepts only all 8,700
strictly sealed shard receipts.

No CLI mode reaches the private payload target or reducer.  M1682 must review
this exact source and M1683 must separately release execution.  This source
revision opens no canonical payload and performs no replay.  CPython 3.6 safe.
"""
from __future__ import print_function

import argparse
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
    "system_simulator/tests/"
    "test_m1681_ep34_decoder_d0_shard_execution_closure_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1681_ep34_decoder_d0_shard_execution_closure_successor_"
    "source_contract_r1_20260901.json")
M1671_SOURCE = HERE / (
    "build_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py")
M1671_TEST = HW / (
    "system_simulator/tests/"
    "test_m1671_ep34_decoder_d0_recoverable_shard_successor_source.py")
M1671_CONTRACT = HW / (
    "contracts/m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_contract_r1_20260901.json")
M1672_REVIEW = HW / (
    "reviews/m1672_m1671_ep34_decoder_d0_recoverable_shard_successor_"
    "source_independent_review_r1_20260901")
M1666_REVIEW = HW / (
    "reviews/m1666_m1656_decoder_actual_prefix_result_independent_"
    "hammer_r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
FUTURE_REVIEW = HW / (
    "reviews/m1682_m1681_ep34_decoder_d0_shard_execution_closure_"
    "successor_source_independent_review_r1_20260901")
FUTURE_RELEASE = HW / (
    "contracts/m1683_m1682_m1681_ep34_decoder_d0_shard_execution_"
    "campaign_release_r1_20260901.json")

SCHEMA = "m1681_ep34_decoder_d0_shard_execution_closure_successor_source_r1_v1"
STATUS = (
    "SOURCE_ONLY__M1672_EXECUTION_AND_REDUCER_P1_REPAIRED__"
    "DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_PAYLOAD_NO_EXECUTION")
RESULT_SCHEMA = "m1681_ep34_decoder_d0_sealed_shard_result_r1_v1"
RESULT_STATUS = "SHARD_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED"
REVIEW_STATUS = (
    "PASS_M1682_M1681_DECODER_D0_SHARD_EXECUTION_CLOSURE_SOURCE__"
    "AUTHORIZE_M1683_RELEASE_AUTHORING__NO_EXECUTION")
RELEASE_SCHEMA = (
    "m1683_m1682_m1681_ep34_decoder_d0_shard_execution_campaign_release_r1_v1")
RELEASE_STATUS = "AUTHORIZE_M1681_FULL_D0_8700_SHARD_CAMPAIGN"

M1671_SOURCE_SHA256 = (
    "f6f99909265acac768acf3f1f6340e25d422bde2726cc19b60b4a30c602b8e02")
M1671_TEST_SHA256 = (
    "db1a64ae42b2885f7ebe7bfc7542cab695b63a7e24275da8858d52d98b2675f5")
M1671_CONTRACT_SHA256 = (
    "5745fd1d1c44507cc20208144c78533bdc6838265cd0611b04cfed23eb90aa6f")
M1672_REVIEW_SHA256 = (
    "f9d9a1290e8a616940a14db60cc1d50c9f1e2492a0a9a98ee3538991b90b404d")
M1672_MANIFEST_SHA256 = (
    "b154ba678a2a4850e3c5665fb734da03dbf74a405cb14fac4cdd5400a81efa5f")
M1672_OUTER_FILE_SHA256 = (
    "7608fa6da9dd0ec7a7d33ddfce5645da58aba28ee25934dc689b608a95398e7e")
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
EXPECTED_BYTE_KINDS = frozenset(("external_read", "external_write",
    "weight_read", "weight_write", "psum_read", "psum_write", "compute",
    "commit"))
HEX = frozenset("0123456789abcdef")
RESULT_PREFIX = "m1681_ep34_decoder_d0_shard_"
ATTEMPT_PREFIX = ".m1681_ep34_decoder_d0_shard_"
WORK_SUFFIX = ".work"
FAILURE_SUFFIX = ".failed_no_retry"


class M1681Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1681Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            M1681Error("nonfinite JSON: " + token)))


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise M1681Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


def _safe_relative(name):
    require(type(name) is str and name and "\\" not in name,
            "unsafe sealed member")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == name, "unsafe sealed member path")
    return path


def verify_sealed_tree(root, expected_review=None, expected_manifest=None,
                       expected_outer=None, allow_ignored_pycache=False,
                       label="sealed tree"):
    """Verify exact recursive population; optionally ignore only pyc caches.

    ``__pycache__`` files are never evidence.  For M1666 only, existing
    regular ``__pycache__/*.pyc`` files are explicitly ignored.  Every other
    unsealed member is rejected.  M1681 result trees forbid pycache entirely.
    """
    root = Path(root)
    try:
        mode = root.lstat().st_mode
    except OSError as error:
        raise M1681Error("missing " + label) from error
    require(stat.S_ISDIR(mode) and not root.is_symlink(),
            label + " must be directory non-symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    for path, item_label in ((manifest, " manifest"), (outer, " outer")):
        current = path.lstat().st_mode
        require(stat.S_ISREG(current) and not path.is_symlink(),
                label + item_label + " must be regular")
    if expected_manifest is not None:
        regular_exact(manifest, expected_manifest, label + " manifest")
    if expected_outer is not None:
        regular_exact(outer, expected_outer, label + " outer")
    require(outer.read_text(encoding="ascii") ==
            sha256(manifest) + "  SHA256SUMS\n",
            label + " outer content drift")
    sealed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and
                all(character in HEX for character in fields[0]) and
                fields[1] not in sealed,
                label + " malformed/duplicate manifest member")
        relative = _safe_relative(fields[1])
        require("__pycache__" not in relative.parts,
                label + " manifest may not seal runtime pycache")
        path = root / relative
        regular_exact(path, fields[0], label + " member " + fields[1])
        sealed[fields[1]] = fields[0]
    actual = set()
    ignored = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        path_mode = path.lstat().st_mode
        require(not path.is_symlink(), label + " contains a symlink")
        if path.is_dir():
            require(stat.S_ISDIR(path_mode), label + " special directory")
            continue
        require(stat.S_ISREG(path_mode), label + " contains special file")
        name = relative.as_posix()
        if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        if "__pycache__" in relative.parts:
            require(allow_ignored_pycache and relative.parts[-2] ==
                    "__pycache__" and relative.suffix == ".pyc",
                    label + " contains forbidden unsealed pycache")
            ignored.append(name)
        else:
            actual.add(name)
    require(actual == set(sealed),
            label + " recursive population differs from manifest")
    if expected_review is not None:
        require("review.json" in sealed and
                sealed["review.json"] == expected_review,
                label + " review identity absent from manifest")
    return {"members": len(sealed), "manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer),
            "ignored_unsealed_pycache": sorted(ignored)}


def verify_double_sealed_file(path, label):
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    for item in (sidecar, outer):
        item_mode = item.lstat().st_mode
        require(stat.S_ISREG(item_mode) and not item.is_symlink(),
                label + " seal must be regular non-symlink")
    require(sidecar.read_text(encoding="ascii") ==
            sha256(path) + "  " + path.name + "\n" and
            outer.read_text(encoding="ascii") ==
            sha256(sidecar) + "  " + sidecar.name + "\n",
            label + " double seal drift")
    return sha256(path)


def load_m1671():
    regular_exact(M1671_SOURCE, M1671_SOURCE_SHA256, "exact M1671 source")
    spec = importlib.util.spec_from_file_location("m1681_exact_m1671",
                                                  str(M1671_SOURCE))
    require(spec is not None and spec.loader is not None,
            "cannot import exact M1671")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.SCHEMA ==
            "m1671_ep34_decoder_d0_recoverable_shard_successor_source_r1_v1" and
            tuple(module.CONFIGS) == CONFIGS and module.TOTAL_SHARDS == 8700,
            "M1671 grid/scheduler boundary drift")
    return module


G = load_m1671()


def verify_m1672_no_go():
    seal = verify_sealed_tree(M1672_REVIEW, M1672_REVIEW_SHA256,
        M1672_MANIFEST_SHA256, M1672_OUTER_FILE_SHA256, False, "M1672")
    row = strict_json(M1672_REVIEW / "review.json")
    require(row.get("status") ==
            "FAIL_M1672_M1671_DECODER_FULL_D0_SOURCE__NO_M1673_EXECUTION_RELEASE__SUCCESSOR_EXECUTION_CLOSURE_REQUIRED" and
            row.get("verdict") == "FAIL_CLOSED_NO_M1673_EXECUTION_RELEASE" and
            row.get("p0_count") == 0 and row.get("p1_count") == 3 and
            [item.get("id") for item in row.get("p1", [])] == [
                "P1_NO_EXECUTABLE_SHARD_OR_ATOMIC_PUBLISH_CLOSURE",
                "P1_REDUCER_ACCEPTS_UNSEALED_INCOMPLETE_METRICS",
                "P1_M1666_PREDECESSOR_NOT_RECURSIVELY_CLOSED"] and
            row.get("authorization", {}).get(
                "successor_execution_closure_source") is True and
            row.get("authorization", {}).get("shard_execution") is False,
            "M1672 finding/disposition drift")
    return seal


def verify_m1666_with_explicit_pycache_policy():
    seal = verify_sealed_tree(M1666_REVIEW,
        G.M1666_REVIEW_SHA256, G.M1666_MANIFEST_SHA256,
        G.M1666_OUTER_FILE_SHA256, True, "M1666")
    require(all("/__pycache__/" in "/" + name or
                name.startswith("__pycache__/")
                for name in seal["ignored_unsealed_pycache"]),
            "M1666 ignored member is not pycache")
    return seal


def namespace_paths(ordinal):
    shard = G.shard_descriptor(ordinal)
    token = "{:04d}".format(shard["shard_ordinal"])
    result = HW / "results" / (RESULT_PREFIX + token + "_r1_20260901")
    attempt = HW / "results" / (
        ATTEMPT_PREFIX + token + "_r1_20260901.attempt_consumed")
    work = HW / "results" / (
        "." + RESULT_PREFIX + token + "_r1_20260901" + WORK_SUFFIX)
    failure = HW / "results" / (
        RESULT_PREFIX + token + "_r1_20260901" + FAILURE_SUFFIX)
    require(len(set((result, attempt, work, failure))) == 4,
            "shard namespaces collide")
    return {"result": result, "attempt": attempt,
            "work": work, "failure": failure}


def namespace_strings(ordinal):
    return dict((key, str(path.relative_to(HW)))
                for key, path in namespace_paths(ordinal).items())


def _review_identity():
    return {"source_sha256": sha256(SOURCE),
            "test_sha256": sha256(TEST),
            "source_contract_sha256": sha256(SOURCE_CONTRACT),
            "m1671_source_sha256": M1671_SOURCE_SHA256,
            "m1671_test_sha256": M1671_TEST_SHA256,
            "m1671_contract_sha256": M1671_CONTRACT_SHA256,
            "m1672_review_sha256": M1672_REVIEW_SHA256,
            "m1672_manifest_sha256": M1672_MANIFEST_SHA256,
            "m1672_outer_file_sha256": M1672_OUTER_FILE_SHA256,
            "checkpoint_sha256": G.CHECKPOINT_SHA256,
            "resource_manifest_sha256": G.RESOURCE_SHA256,
            "docs359_sha256": DOCS359_SHA256}


def validate_future_review_and_release():
    review_seal = verify_sealed_tree(FUTURE_REVIEW,
        allow_ignored_pycache=False, label="M1682")
    review = strict_json(FUTURE_REVIEW / "review.json")
    require(review.get("status") == REVIEW_STATUS and
            review.get("score_over_100", 0) >= 95 and
            review.get("p0_count") == 0 and review.get("p1_count") == 0 and
            review.get("identity") == _review_identity() and
            review.get("authorization") == {
                "release_authoring": True, "shard_execution": False,
                "payload_open": False, "automatic_retry": False},
            "M1682 semantic authority drift")
    release_sha = verify_double_sealed_file(FUTURE_RELEASE, "M1683 release")
    release = strict_json(FUTURE_RELEASE)
    identity = dict(_review_identity(),
        review_sha256=sha256(FUTURE_REVIEW / "review.json"),
        review_manifest_sha256=review_seal["manifest_sha256"],
        review_outer_file_sha256=review_seal["outer_file_sha256"])
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == RELEASE_STATUS and
            release.get("identity") == identity and
            release.get("authorization") == {
                "shard_runs": 8700, "payload_opens": 8700,
                "attempt_writes": 8700, "automatic_retry": False,
                "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0} and
            release.get("fixed_grid") == G.fixed_grid() and
            release.get("namespace_examples") == {
                "first": namespace_strings(0),
                "last": namespace_strings(G.TOTAL_SHARDS - 1)} and
            release.get("claim_boundary") == {
                "shard_isolated": True, "monolithic_full_call": False,
                "full_decoder": False, "system_speedup": False,
                "paper_result": False},
            "M1683 release identity/authorization/grid/namespace drift")
    return release_sha


def require_fresh_shard(ordinal):
    paths = namespace_paths(ordinal)
    require(all(not os.path.lexists(str(path)) for path in paths.values()),
            "shard namespace is not fresh; retry is forbidden")
    return paths


def consume_attempt(ordinal, release_sha):
    paths = namespace_paths(ordinal)
    attempt = paths["attempt"]
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(attempt), flags, 0o400)
    try:
        payload = {"schema": SCHEMA, "shard_ordinal": ordinal,
            "shard": G.shard_descriptor(ordinal),
            "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha,
            "automatic_retry": False,
            "payload_opened_before_attempt": False}
        os.write(descriptor, canonical_bytes(payload) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256(attempt)


class ImmutableTimestepPlane(object):
    """Opened-FD/hash-bound immutable snapshot of exactly one D0 timestep."""
    def __init__(self, path, shape, expected_sha256, timestep):
        self.path = Path(path)
        self.shape = tuple(int(value) for value in shape)
        self.expected_sha256 = str(expected_sha256)
        self.timestep = int(timestep)
        require(self.shape == tuple(G.R.INPUT_SHAPES[0]) and
                0 <= self.timestep < G.TIMESTEPS,
                "D0 timestep plane shape/index drift")
        mode = self.path.lstat().st_mode
        require(stat.S_ISREG(mode) and not self.path.is_symlink(),
                "payload must be regular non-symlink")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(self.path), flags)
        stream = None
        try:
            opened = os.fstat(descriptor)
            require(stat.S_ISREG(opened.st_mode),
                    "opened payload is not regular")
            stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
            self.opened_sha256 = digest.hexdigest()
            self.opened_size = int(opened.st_size)
            require(self.opened_sha256 == self.expected_sha256,
                    "opened payload SHA drift")
            channels, height, width = self.shape[2:]
            bits = channels * height * width
            require(bits % 8 == 0 and self.opened_size ==
                    (self.shape[0] * bits) // 8,
                    "payload/timestep byte geometry drift")
            self.timestep_bytes = bits // 8
            stream.seek(self.timestep * self.timestep_bytes)
            self.snapshot = bytes(stream.read(self.timestep_bytes))
            require(len(self.snapshot) == self.timestep_bytes,
                    "payload timestep snapshot truncated")
            stream.close()
            stream = None
        finally:
            if stream is not None:
                stream.close()
            if descriptor >= 0:
                os.close(descriptor)

    def bit(self, channel, y, x):
        channels, height, width = self.shape[2:]
        require(0 <= channel < channels and 0 <= y < height and
                0 <= x < width, "D0 timestep bit coordinate out of range")
        index = (channel * height + y) * width + x
        return (self.snapshot[index >> 3] >> (index & 7)) & 1


def _hex64(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in HEX for character in value),
            label + " must be lowercase hex64")


def metric_final_state(metric, shard):
    return canonical_sha({"configuration": metric["configuration"],
        "shard": shard, "total_cycles": metric["total_cycles"],
        "request_count": metric["request_count"],
        "kind_counts": metric["kind_counts"],
        "byte_counts": metric["byte_counts"],
        "packed_transaction_address_sha256":
            metric["packed_transaction_address_sha256"],
        "packed_commit_sequence_sha256":
            metric["packed_commit_sequence_sha256"],
        "destination_state_chain_sha256":
            metric["destination_state_chain_sha256"]})


def validate_metric(metric, configuration, shard):
    require(type(metric) is dict and
            metric.get("configuration") == configuration and
            metric.get("resource_manifest_sha256") == G.RESOURCE_SHA256 and
            metric.get("per_request_miter") is True and
            metric.get("per_destination_miter") is True and
            metric.get("shard_reset_boundary") is True and
            type(metric.get("total_cycles")) is int and
            metric["total_cycles"] > 0 and
            type(metric.get("request_count")) is int and
            metric["request_count"] > 0,
            "sealed shard metric identity/cycle/request drift")
    kinds = metric.get("kind_counts")
    byte_counts = metric.get("byte_counts")
    require(type(kinds) is dict and kinds and
            all(type(key) is str and key in EXPECTED_BYTE_KINDS and
                type(value) is int and value >= 0
                for key, value in kinds.items()) and
            sum(kinds.values()) == metric["request_count"],
            "sealed shard kind/request ledger drift")
    require(type(byte_counts) is dict and byte_counts and
            set(byte_counts).issubset(EXPECTED_BYTE_KINDS) and
            all(type(value) is int and value >= 0
                for value in byte_counts.values()),
            "sealed shard byte ledger contains negative/noninteger values")
    require(kinds.get("commit") == shard["destination_count"] *
                G.OUTPUT_BLOCKS and
            byte_counts.get("commit") == shard["destination_count"] *
                G.OUTPUT_BLOCKS * G.R.OUTPUT_COMMIT_BYTES,
            "sealed shard commit count/bytes drift")
    for field in ("packed_transaction_address_sha256",
                  "packed_commit_sequence_sha256",
                  "destination_state_chain_sha256", "final_state_sha256"):
        _hex64(metric.get(field), field)
    require(metric["final_state_sha256"] == metric_final_state(metric, shard),
            "sealed shard final-state digest drift")
    return metric


def validate_metric_bundle(metrics, shard):
    require(type(metrics) is list and len(metrics) == 3 and
            [row.get("configuration") for row in metrics] == list(CONFIGS),
            "sealed shard configuration order drift")
    for configuration, metric in zip(CONFIGS, metrics):
        validate_metric(metric, configuration, shard)
    require(len(set(row["packed_commit_sequence_sha256"]
                    for row in metrics)) == 1,
            "sealed shard cross-configuration commit digest drift")
    return True


def seal_work_tree(root):
    root = Path(root)
    members = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        require("__pycache__" not in relative.parts and not path.is_symlink(),
                "result work tree contains pycache/symlink")
        if path.is_file() and path.name not in (
                "SHA256SUMS", "SHA256SUMS.seal.sha256"):
            members.append(relative)
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(root / member), member.as_posix())
        for member in sorted(members)), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")


def validate_shard_receipt(row, ordinal, attempt_sha=None,
                           release_sha=None):
    shard = G.shard_descriptor(ordinal)
    require(type(row) is dict and row.get("schema") == RESULT_SCHEMA and
            row.get("status") == RESULT_STATUS and
            row.get("source_sha256") == sha256(SOURCE) and
            row.get("checkpoint_sha256") == G.CHECKPOINT_SHA256 and
            row.get("resource_manifest_sha256") == G.RESOURCE_SHA256 and
            row.get("shard_ordinal") == ordinal and row.get("shard") == shard and
            row.get("configuration_order") == list(CONFIGS) and
            row.get("automatic_retry") is False and
            row.get("shard_isolated") is True and
            row.get("monolithic_full_call") is False and
            row.get("full_decoder") is False and
            row.get("system_speedup") is False and
            row.get("paper_result") is False,
            "sealed shard receipt identity/claim drift")
    if attempt_sha is not None:
        require(row.get("attempt_sha256") == attempt_sha,
                "sealed shard attempt identity drift")
    if release_sha is not None:
        require(row.get("release_sha256") == release_sha,
                "sealed shard release identity drift")
    _hex64(row.get("payload_fd_sha256"), "payload FD SHA")
    require(type(row.get("payload_fd_size")) is int and
            row["payload_fd_size"] > 0,
            "sealed shard payload extent drift")
    validate_metric_bundle(row.get("metrics"), shard)
    expected_ratios = G.validate_three_configuration_metrics(
        row["metrics"], shard)
    require(row.get("integer_ratio_inputs") == expected_ratios,
            "sealed shard integer ratio inputs drift")
    rss = row.get("rss")
    require(type(rss) is dict and rss.get("absolute_limit_kib") ==
                G.RSS_ABSOLUTE_LIMIT_KIB and
            rss.get("increment_limit_kib") == G.RSS_INCREMENT_LIMIT_KIB and
            type(rss.get("gate_calls")) is int and rss["gate_calls"] > 0,
            "sealed shard RSS receipt drift")
    return row


def verify_sealed_shard(ordinal):
    paths = namespace_paths(ordinal)
    attempt = paths["attempt"]
    result = paths["result"]
    require(attempt.exists() and not attempt.is_symlink(),
            "sealed shard lacks regular attempt")
    attempt_sha = sha256(attempt)
    seal = verify_sealed_tree(result, allow_ignored_pycache=False,
                              label="M1681 shard")
    require(set(path.relative_to(result).as_posix()
                for path in result.rglob("*") if path.is_file() and
                path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")) ==
            {"result.json"}, "shard receipt population must be result.json only")
    row = strict_json(result / "result.json")
    validate_shard_receipt(row, ordinal, attempt_sha,
                           row.get("release_sha256"))
    return {"ordinal": ordinal, "row": row, "seal": seal,
            "attempt_sha256": attempt_sha}


def resume_state():
    counts = {"complete": 0, "pending": 0, "failed_no_retry": 0,
              "interrupted_no_retry": 0}
    completed = hashlib.sha256()
    for ordinal in range(G.TOTAL_SHARDS):
        paths = namespace_paths(ordinal)
        present = dict((key, os.path.lexists(str(path)))
                       for key, path in paths.items())
        if present["result"]:
            require(present == {"result": True, "attempt": True,
                                "work": False, "failure": False},
                    "completed shard namespace topology drift")
            verified = verify_sealed_shard(ordinal)
            completed.update((str(ordinal) + ":" +
                verified["seal"]["manifest_sha256"] + "\n").encode("ascii"))
            counts["complete"] += 1
        elif present["failure"]:
            require(present["attempt"] and not present["result"] and
                    not present["work"], "failed shard topology drift")
            counts["failed_no_retry"] += 1
        elif present["attempt"]:
            require(not present["result"] and not present["failure"],
                    "attempted shard topology drift")
            counts["interrupted_no_retry"] += 1
        else:
            require(not present["work"] and not present["failure"],
                    "work/failure exists without attempt")
            counts["pending"] += 1
    require(sum(counts.values()) == G.TOTAL_SHARDS,
            "resume population conservation failed")
    return {"schema": SCHEMA, "counts": counts,
            "complete_manifest_chain_sha256": completed.hexdigest(),
            "automatic_retry": False}


def _schedule_actual_shard(shard, plane, rss):
    """Exact private payload-to-shard target after attempt consumption."""
    require(type(plane) is ImmutableTimestepPlane and
            plane.timestep == shard["timestep"],
            "canonical immutable timestep plane required")
    cin, cout, hin, win, _hout, wout = G.R.GEOMETRY[0]
    require(wout == G.OUTPUT_WIDTH and (cout + 95) // 96 == G.OUTPUT_BLOCKS,
            "D0 geometry drift")
    metrics = []
    for configuration in CONFIGS:
        session = G.ShardSession(configuration, shard, rss)
        getter = lambda channel, y, x: plane.bit(channel, y, x)
        for destination in range(shard["destination_start"],
                                 shard["destination_stop_exclusive"]):
            oy, ox = divmod(destination, G.OUTPUT_WIDTH)
            contributors = G.R.contributors_for_destination(
                getter, configuration, cin, hin, win, oy, ox)
            for output_block in range(G.OUTPUT_BLOCKS):
                last = ""
                for request_row in G.R.destination_transactions(
                        configuration, 0, shard["timestep"], destination,
                        output_block, contributors, "", session.cache):
                    session.accept(request_row, destination, output_block)
                    if request_row["kind"] == "psum_write":
                        last = request_row["produces"]
                identifier = "{}:c{}:t{}:commit:{}:{}".format(
                    configuration, shard["call_ordinal"],
                    shard["timestep"], destination, output_block)
                address = ((4 << 60) | (shard["timestep"] << 44) |
                    ((destination * G.OUTPUT_BLOCKS + output_block) *
                     G.R.OUTPUT_COMMIT_BYTES))
                commit = G.R.request(identifier, configuration, "commit",
                    [address], [0], G.R.OUTPUT_COMMIT_BYTES,
                    [last] if last else ())
                session.accept(commit, destination, output_block)
            session.finish_destination(destination)
        metric = session.finish()
        metric["final_state_sha256"] = metric_final_state(metric, shard)
        metrics.append(metric)
    validate_metric_bundle(metrics, shard)
    return metrics


def _run_authorized_shard(ordinal):
    """Future private target. Strong authority is the mandatory first call."""
    release_sha = validate_future_review_and_release()
    paths = require_fresh_shard(ordinal)
    attempt_sha = consume_attempt(ordinal, release_sha)
    paths["work"].mkdir(mode=0o700)
    published = False
    try:
        # All population hashing and the first canonical payload selection/open
        # occur after the immutable attempt marker above.
        G.R.validate_authorities(True)
        shard = G.shard_descriptor(ordinal)
        record = G.selected_record(shard)
        payload = (G.R.M1521_ROOT / record["positive_output"]).resolve()
        require(payload.parent == (G.R.M1521_ROOT / "payloads").resolve(),
                "canonical payload path escaped payload directory")
        rss = G.P.RssGate()
        plane = ImmutableTimestepPlane(payload, record["shape"],
            record["positive_output_sha256"], shard["timestep"])
        rss.sample()
        metrics = _schedule_actual_shard(shard, plane, rss)
        rss.sample()
        row = {"schema": RESULT_SCHEMA, "status": RESULT_STATUS,
            "source_sha256": sha256(SOURCE),
            "release_sha256": release_sha,
            "attempt_sha256": attempt_sha,
            "checkpoint_sha256": G.CHECKPOINT_SHA256,
            "resource_manifest_sha256": G.RESOURCE_SHA256,
            "shard_ordinal": ordinal, "shard": shard,
            "configuration_order": list(CONFIGS),
            "metrics": metrics,
            "integer_ratio_inputs":
                G.validate_three_configuration_metrics(metrics, shard),
            "payload_fd_sha256": plane.opened_sha256,
            "payload_fd_size": plane.opened_size,
            "rss": rss.summary(), "automatic_retry": False,
            "shard_isolated": True, "monolithic_full_call": False,
            "full_decoder": False, "system_speedup": False,
            "paper_result": False,
            "independent_result_hammer_pending": True}
        validate_shard_receipt(row, ordinal, attempt_sha, release_sha)
        (paths["work"] / "result.json").write_text(json.dumps(
            row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        seal_work_tree(paths["work"])
        verify_sealed_tree(paths["work"], allow_ignored_pycache=False,
                           label="M1681 work shard")
        paths["work"].rename(paths["result"])
        published = True
        return row
    except BaseException:
        if paths["work"].is_dir() and not os.path.lexists(
                str(paths["failure"])):
            paths["work"].rename(paths["failure"])
        raise
    finally:
        require(published or not paths["work"].exists(),
                "failed shard work tree was not quarantined")


def reduce_complete_sealed_shards():
    """Read only all exact sealed receipts; incomplete populations fail."""
    totals = dict((configuration, {"cycles": 0, "requests": 0,
                                   "bytes": {}})
                  for configuration in CONFIGS)
    manifest_chain = hashlib.sha256()
    for ordinal in range(G.TOTAL_SHARDS):
        verified = verify_sealed_shard(ordinal)
        row = verified["row"]
        require(row["shard"] == G.shard_descriptor(ordinal),
                "reducer shard order drift")
        manifest_chain.update((str(ordinal) + ":" +
            verified["seal"]["manifest_sha256"] + "\n").encode("ascii"))
        for configuration, metric in zip(CONFIGS, row["metrics"]):
            target = totals[configuration]
            target["cycles"] += metric["total_cycles"]
            target["requests"] += metric["request_count"]
            for name in EXPECTED_BYTE_KINDS:
                value = metric["byte_counts"].get(name, 0)
                require(type(value) is int and value >= 0,
                        "reducer byte ledger drift")
                target["bytes"][name] = target["bytes"].get(name, 0) + value
    dense = totals[CONFIGS[0]]["cycles"]
    equal = totals[CONFIGS[1]]["cycles"]
    typed = totals[CONFIGS[2]]["cycles"]
    return {"schema": SCHEMA,
        "status": "COMPLETE_8700_SEALED_SHARDS__INDEPENDENT_HAMMER_REQUIRED",
        "configuration_totals": totals,
        "ratio_of_sums": {
            "dense_to_bit_typed": {"numerator": dense,
                "denominator": typed},
            "bit_equal_to_bit_typed": {"numerator": equal,
                "denominator": typed}},
        "complete_shards": G.TOTAL_SHARDS,
        "sealed_manifest_chain_sha256": manifest_chain.hexdigest(),
        "shard_isolated": True, "monolithic_full_call": False,
        "full_decoder": False, "system_speedup": False,
        "paper_result_pending_independent_hammer": True}


def validate_authorities_source_stage():
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1671_SOURCE, M1671_SOURCE_SHA256, "exact M1671 source")
    regular_exact(M1671_TEST, M1671_TEST_SHA256, "exact M1671 test")
    regular_exact(M1671_CONTRACT, M1671_CONTRACT_SHA256,
                  "exact M1671 contract")
    m1672 = verify_m1672_no_go()
    m1666 = verify_m1666_with_explicit_pycache_policy()
    require(G.validate_grid() == {"calls": 30, "timesteps": 300,
            "destinations": 360000, "shards": 8700,
            "gap_count": 0, "overlap_count": 0}, "M1671 grid drift")
    require(not FUTURE_REVIEW.exists() and
            not os.path.lexists(str(FUTURE_RELEASE)) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) + ".sha256"))) and
            not os.path.lexists(str(Path(str(FUTURE_RELEASE) +
                                         ".sha256.seal.sha256"))),
            "future M1682/M1683 authority must be absent at source stage")
    return {"m1672": m1672, "m1666": m1666,
            "grid": G.validate_grid(), "payload_opened": False,
            "execution": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
        "unchanged_core": {"m1671_source_sha256": M1671_SOURCE_SHA256,
            "grid": G.fixed_grid(),
            "scheduler_changed": False, "grid_changed": False},
        "execution_closure": {
            "private_target": "_run_authorized_shard",
            "namespace_pattern": {
                "first": namespace_strings(0),
                "last": namespace_strings(G.TOTAL_SHARDS - 1)},
            "attempt_before_first_payload_access": True,
            "automatic_retry": False,
            "immutable_opened_fd_hash": True,
            "atomic_seal_then_rename": True,
            "resume_verifier": "resume_state",
            "strict_sealed_reducer": "reduce_complete_sealed_shards",
            "M1666_pycache_policy":
                "ignore only regular unsealed __pycache__/*.pyc as runtime cache; reject every other unsealed member",
            "M1681_result_pycache_policy": "forbidden"},
        "future_gate": {"review": str(FUTURE_REVIEW.relative_to(HW)),
            "release": str(FUTURE_RELEASE.relative_to(HW)),
            "review_present": FUTURE_REVIEW.exists(),
            "release_present": os.path.lexists(str(FUTURE_RELEASE))},
        "claim_boundary": {"source_only": True,
            "payload_opened": False, "shard_execution": False,
            "reducer_execution": False, "cycles": False,
            "traffic": False, "speedup": False, "energy": False,
            "rtl": False, "eda": False, "full_d0_result": False,
            "monolithic_full_call": False, "full_decoder": False,
            "system_speedup": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        output = {"schema": SCHEMA,
            "status": "PASS_M1681_SOURCE_PREFLIGHT__NO_PAYLOAD_NO_EXECUTION",
            "authorities": validate_authorities_source_stage(),
            "claim_boundary": describe()["claim_boundary"]}
    else:
        output = describe()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
