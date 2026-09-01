#!/usr/bin/env python3
"""Different-author source hammer for M1704; never replays or reduces."""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1704_ep34_decoder_d0_execution_authority_adapter_source.py"
TEST = HW / "system_simulator/tests/test_m1704_ep34_decoder_d0_execution_authority_adapter_source.py"
CONTRACT = HW / "contracts/m1704_ep34_decoder_d0_execution_authority_adapter_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1704_ep34_decoder_d0_execution_authority_adapter_source_author_receipt_r1_20260901"
M1688_SOURCE = HW / "system_simulator/scripts/build_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py"
M1688_TEST = HW / "system_simulator/tests/test_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source.py"
M1688_CONTRACT = HW / "contracts/m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_contract_r1_20260901.json"
M1689 = HW / "reviews/m1689_m1688_ep34_decoder_d0_reducer_topology_repair_successor_source_independent_review_r1_20260901"
M1683 = HW / "contracts/m1683_m1682_m1681_ep34_decoder_d0_shard_execution_campaign_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "abc052025a5ed6975fa9d200581b182fa723a7b4b0330bc341b4bc2712204820",
    TEST: "9d804d08d69621b66cd687c9a4a272f344c62eb7dff97204ebd25efe57eb25f4",
    CONTRACT: "1a133259f01addec024530944bec739a9a783fa381fc08f736721219b15bd554",
    M1688_SOURCE: "2ae2725e24c46972f46c54ae71260a8fc637e85c4de0b90f9f91bc42da76abba",
    M1688_TEST: "7a331143f6d486939ed77eb34eef60610e450d131313f6df3340cd76290662cb",
    M1688_CONTRACT: "10f44a589f986c06f560b0353224b83f5ca6f44e5a0ac73599bd40a8dc85271f",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs)


def verify_file_seal(payload):
    payload = Path(payload)
    manifest = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    require(manifest.read_text(encoding="ascii").split() ==
            [sha256(payload), payload.name], "file payload seal drift")
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), manifest.name], "file outer seal drift")


def verify_tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "tree entity drift")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text(encoding="ascii").split() ==
            [sha256(manifest), "SHA256SUMS"], "tree outer seal drift")
    expected = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and
                name not in expected, "unsafe manifest row")
        expected[name] = digest
    actual = set()
    for base, directories, files in os.walk(str(root), followlinks=False):
        parent = Path(base)
        directories[:] = [name for name in directories
                           if not (parent / name).is_symlink()]
        for name in files:
            path = parent / name
            if name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                continue
            mode = path.lstat().st_mode
            require(stat.S_ISREG(mode) and not path.is_symlink(),
                    "nonregular tree member")
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), "tree population drift")
    for name, digest in expected.items():
        require(sha256(root / name) == digest, "tree member SHA drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}


def seal_tree(root):
    root = Path(root)
    rows = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            continue
        require(path.is_file() and not path.is_symlink(),
                "fixture member entity")
        rows.append(sha256(path) + "  " + path.name + "\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(root / "SHA256SUMS") + "  SHA256SUMS\n", encoding="ascii")
    return verify_tree(root)


def seal_file(path):
    path = Path(path)
    manifest = Path(str(path) + ".sha256")
    manifest.write_text(sha256(path) + "  " + path.name + "\n",
                        encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        sha256(manifest) + "  " + manifest.name + "\n", encoding="ascii")


SPEC = importlib.util.spec_from_file_location("m1705_exact_m1704", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Fixture(object):
    def __init__(self, root, review_mutation=None, release_mutation=None):
        root = Path(root)
        self.forbidden = root / "m1683_forbidden.json"
        self.review = root / "m1705_review"
        self.review.mkdir()
        review = {"schema": "m1705_fixture",
            "status": M.REVIEW_STATUS, "score_over_100": 100,
            "p0_count": 0, "p1_count": 0,
            "identity": M._review_identity(),
            "authorization": {"release_authoring": True,
                "shard_execution": False, "payload_open": False,
                "reducer_execution": False, "automatic_retry": False}}
        if review_mutation is not None:
            review_mutation(review)
        (self.review / "review.json").write_text(json.dumps(
            review, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        review_seal = seal_tree(self.review)
        self.release = root / "m1706_release.json"
        release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "identity": dict(M._review_identity(),
                review_sha256=sha256(self.review / "review.json"),
                review_manifest_sha256=review_seal["manifest_sha256"],
                review_outer_file_sha256=review_seal["outer_file_sha256"]),
            "authorization": {"shard_runs": 8700,
                "payload_opens": 8700, "attempt_writes": 8700,
                "automatic_retry": False, "gpu_runs": 0,
                "eda_runs": 0, "all_other_runs": 0},
            "fixed_grid": M.B.G.fixed_grid(),
            "namespace_examples": {"first": M.B.namespace_strings(0),
                "last": M.B.namespace_strings(M.B.G.TOTAL_SHARDS - 1)},
            "reducer": {"source": "M1688",
                "strong_exact_sibling_topology": True,
                "attempt_regular_nonsymlink_mode_0400": True},
            "claim_boundary": {"shard_isolated": True,
                "monolithic_full_call": False, "full_decoder": False,
                "system_speedup": False, "paper_result": False}}
        if release_mutation is not None:
            release_mutation(release)
        self.release.write_text(json.dumps(
            release, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        seal_file(self.release)


def with_fixture(fixture, function):
    old = (M.FUTURE_REVIEW, M.FUTURE_RELEASE, M.FORBIDDEN_M1683_RELEASE)
    M.FUTURE_REVIEW = fixture.review
    M.FUTURE_RELEASE = fixture.release
    M.FORBIDDEN_M1683_RELEASE = fixture.forbidden
    try:
        return function()
    finally:
        (M.FUTURE_REVIEW, M.FUTURE_RELEASE,
         M.FORBIDDEN_M1683_RELEASE) = old


def expect_reject(label, fixture):
    try:
        with_fixture(fixture, M.validate_future_review_and_release)
    except (M.M1704Error, M.B.M1681Error, OSError, KeyError,
            ValueError, AssertionError):
        return label
    raise AssertionError("mutation accepted " + label)


def function(tree, name):
    rows = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name == name]
    require(len(rows) == 1, "function population drift " + name)
    return rows[0]


def attr(node, left, right):
    return (isinstance(node, ast.Attribute) and node.attr == right and
            isinstance(node.value, ast.Name) and node.value.id == left)


def static_adapter_hammer(tree):
    node = function(tree, "_run_authorized_shard")
    require(len(node.args.args) == 1 and node.args.args[0].arg == "ordinal" and
            node.args.vararg is None and node.args.kwarg is None,
            "adapter signature drift")
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, (ast.Str, ast.Constant))):
        body = body[1:]
    require(len(body) == 4 and isinstance(body[0], ast.Expr) and
            isinstance(body[1], ast.Assign) and
            isinstance(body[2], ast.Assign) and isinstance(body[3], ast.Try),
            "adapter statement topology drift")
    require(isinstance(body[1].targets[0], ast.Name) and
            body[1].targets[0].id == "original" and
            attr(body[1].value, "B", "validate_future_review_and_release"),
            "original authority capture drift")
    require(attr(body[2].targets[0], "B",
                 "validate_future_review_and_release") and
            isinstance(body[2].value, ast.Name) and
            body[2].value.id == "validate_future_review_and_release",
            "authority bind drift")
    guarded = body[3]
    require(not guarded.handlers and not guarded.orelse and
            len(guarded.body) == 1 and isinstance(guarded.body[0], ast.Return),
            "synchronous try topology drift")
    call = guarded.body[0].value
    require(isinstance(call, ast.Call) and
            attr(call.func, "B", "_run_authorized_shard") and
            len(call.args) == 1 and isinstance(call.args[0], ast.Name) and
            call.args[0].id == "ordinal" and not call.keywords,
            "frozen shard delegate drift")
    require(len(guarded.finalbody) == 1 and
            isinstance(guarded.finalbody[0], ast.Assign) and
            attr(guarded.finalbody[0].targets[0], "B",
                 "validate_future_review_and_release") and
            isinstance(guarded.finalbody[0].value, ast.Name) and
            guarded.finalbody[0].value.id == "original",
            "finally restore drift")
    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
    delegates = [item for item in calls
                 if attr(item.func, "B", "_run_authorized_shard")]
    helpers = sorted(item.func.id for item in calls
                     if isinstance(item.func, ast.Name))
    require(len(delegates) == 1 and helpers == ["require", "type"],
            "adapter call population drift")
    return {"authority_bind_count": 1, "finally_restore_count": 1,
            "frozen_delegate_count": 1, "synchronous": True}


def static_reducer_hammer(tree):
    node = function(tree, "reduce_complete_sealed_shards")
    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
    require(len(calls) == 1 and
            attr(calls[0].func, "M1688", "reduce_complete_sealed_shards") and
            not calls[0].args and not calls[0].keywords,
            "reducer is not exact M1688 delegation")
    require(not any(isinstance(item, (ast.For, ast.While, ast.With))
                    for item in ast.walk(node)), "adapter reducer logic added")
    return {"delegate": "M1688.reduce_complete_sealed_shards",
            "call_count": 1, "adapter_reducer_logic": False,
            "executed": False}


def static_cli_hammer(tree):
    node = function(tree, "main")
    text = "\n".join(SOURCE.read_text(encoding="utf-8").splitlines()[
        node.lineno - 1:])
    require("_run_authorized_shard(" not in text and
            "reduce_complete_sealed_shards(" not in text,
            "source CLI reaches execution/reducer")
    literals = [item.s for item in ast.walk(node)
                if isinstance(item, ast.Str)]
    require("--describe" in literals and "--preflight" in literals,
            "source CLI mode drift")
    return {"describe": True, "preflight": True,
            "shard_mode": False, "reducer_mode": False}


def dynamic_adapter_hammer():
    original_run = M.B._run_authorized_shard
    original_gate = M.B.validate_future_review_and_release
    calls = []

    def fake_run(ordinal):
        require(M.B.validate_future_review_and_release is
                M.validate_future_review_and_release,
                "authority not rebound during delegate")
        calls.append(ordinal)
        return {"ordinal": ordinal, "synthetic": True}

    M.B._run_authorized_shard = fake_run
    try:
        require(M._run_authorized_shard(0) ==
                {"ordinal": 0, "synthetic": True}, "valid ordinal drift")
    finally:
        M.B._run_authorized_shard = original_run
    require(calls == [0], "delegate call population drift")
    require(M.B.validate_future_review_and_release is original_gate,
            "normal-path authority restore failed")

    def raising_run(ordinal):
        require(M.B.validate_future_review_and_release is
                M.validate_future_review_and_release,
                "exception path authority not rebound")
        raise RuntimeError("synthetic delegate failure")

    M.B._run_authorized_shard = raising_run
    try:
        try:
            M._run_authorized_shard(17)
        except RuntimeError as error:
            require(str(error) == "synthetic delegate failure",
                    "unexpected synthetic failure")
        else:
            raise AssertionError("exception fixture did not raise")
    finally:
        M.B._run_authorized_shard = original_run
    require(M.B.validate_future_review_and_release is original_gate,
            "exception-path authority restore failed")

    rejected = []
    for value in (True, False, -1, 8700, 1.0, "1", None):
        try:
            M._run_authorized_shard(value)
        except M.M1704Error:
            rejected.append(repr(value))
        else:
            raise AssertionError("ordinal accepted " + repr(value))
    require(M.B.validate_future_review_and_release is original_gate,
            "ordinal rejection changed authority")
    return {"synthetic_delegate_calls": 2,
            "normal_restore": True, "exception_restore": True,
            "ordinal_mutations_rejected": rejected,
            "actual_shard_runs": 0, "payload_opens": 0,
            "attempt_writes": 0}


def authority_mutation_hammer():
    labels = []
    with tempfile.TemporaryDirectory() as directory:
        good = Fixture(directory)
        expected = sha256(good.release)
        require(with_fixture(good, M.validate_future_review_and_release) ==
                expected, "exact future authority rejected")

    review_mutations = [
        ("review_status", lambda row: row.update(status="WRONG")),
        ("review_score", lambda row: row.update(score_over_100=94)),
        ("review_p0", lambda row: row.update(p0_count=1)),
        ("review_p1", lambda row: row.update(p1_count=1)),
        ("review_source", lambda row: row["identity"].update(
            source_sha256="0" * 64)),
        ("review_test", lambda row: row["identity"].update(
            test_sha256="0" * 64)),
        ("review_contract", lambda row: row["identity"].update(
            source_contract_sha256="0" * 64)),
        ("review_release_auth", lambda row: row["authorization"].update(
            release_authoring=False)),
        ("review_shard_auth", lambda row: row["authorization"].update(
            shard_execution=True)),
        ("review_payload_auth", lambda row: row["authorization"].update(
            payload_open=True)),
        ("review_reducer_auth", lambda row: row["authorization"].update(
            reducer_execution=True)),
        ("review_retry", lambda row: row["authorization"].update(
            automatic_retry=True)),
        ("review_extra_auth", lambda row: row["authorization"].update(
            extra=True)),
    ]
    for label, mutation in review_mutations:
        with tempfile.TemporaryDirectory() as directory:
            labels.append(expect_reject(label, Fixture(
                directory, review_mutation=mutation)))

    release_mutations = [
        ("release_schema", lambda row: row.update(schema="WRONG")),
        ("release_status", lambda row: row.update(status="WRONG")),
        ("release_identity", lambda row: row["identity"].update(
            source_sha256="0" * 64)),
        ("release_shards", lambda row: row["authorization"].update(
            shard_runs=8699)),
        ("release_payloads", lambda row: row["authorization"].update(
            payload_opens=8699)),
        ("release_attempts", lambda row: row["authorization"].update(
            attempt_writes=8699)),
        ("release_retry", lambda row: row["authorization"].update(
            automatic_retry=True)),
        ("release_gpu", lambda row: row["authorization"].update(gpu_runs=1)),
        ("release_eda", lambda row: row["authorization"].update(eda_runs=1)),
        ("release_other", lambda row: row["authorization"].update(
            all_other_runs=1)),
        ("release_grid", lambda row: row["fixed_grid"].update(shards=8699)),
        ("release_first_namespace", lambda row: row["namespace_examples"].update(
            first={})),
        ("release_last_namespace", lambda row: row["namespace_examples"].update(
            last={})),
        ("release_reducer_source", lambda row: row["reducer"].update(
            source="M1681")),
        ("release_reducer_topology", lambda row: row["reducer"].update(
            strong_exact_sibling_topology=False)),
        ("release_attempt_mode", lambda row: row["reducer"].update(
            attempt_regular_nonsymlink_mode_0400=False)),
        ("release_full_call", lambda row: row["claim_boundary"].update(
            monolithic_full_call=True)),
        ("release_full_decoder", lambda row: row["claim_boundary"].update(
            full_decoder=True)),
        ("release_system", lambda row: row["claim_boundary"].update(
            system_speedup=True)),
        ("release_paper", lambda row: row["claim_boundary"].update(
            paper_result=True)),
    ]
    for label, mutation in release_mutations:
        with tempfile.TemporaryDirectory() as directory:
            labels.append(expect_reject(label, Fixture(
                directory, release_mutation=mutation)))

    for suffix in ("", ".sha256", ".sha256.seal.sha256"):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            forbidden = Path(str(fixture.forbidden) + suffix)
            forbidden.write_text("forbidden\n", encoding="ascii")
            labels.append(expect_reject("m1683" + (suffix or "_payload"),
                                        fixture))
    return {"exact_authority_passed": True,
            "mutations_rejected": len(labels), "labels": labels}


def run_hammer():
    for path, expected in EXPECTED.items():
        require(sha256(path) == expected, "identity drift " + str(path))
    verify_file_seal(CONTRACT)
    author_seal = verify_tree(AUTHOR)
    m1689_seal = verify_tree(M1689)
    m1689_review = strict_json(M1689 / "review.json")
    require(sha256(M1689 / "review.json") == M.M1689_REVIEW_SHA256 and
            m1689_seal["manifest_sha256"] == M.M1689_MANIFEST_SHA256 and
            m1689_seal["outer_file_sha256"] == M.M1689_OUTER_FILE_SHA256,
            "M1689 identity drift")
    require(m1689_review.get("status") ==
            "PASS_M1689_M1688_DECODER_D0_REDUCER_TOPOLOGY_REPAIR_SOURCE__AUTHORIZE_NEWLY_NUMBERED_RELEASE_AUTHORING_ONLY__NO_EXECUTION",
            "M1689 status drift")
    for path in (M1683, Path(str(M1683) + ".sha256"),
                 Path(str(M1683) + ".sha256.seal.sha256")):
        require(not os.path.lexists(str(path)), "M1683 entity exists")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    adapter = static_adapter_hammer(tree)
    reducer = static_reducer_hammer(tree)
    cli = static_cli_hammer(tree)
    dynamic = dynamic_adapter_hammer()
    mutations = authority_mutation_hammer()
    contract = strict_json(CONTRACT)
    require(contract.get("source", {}).get("sha256") == EXPECTED[SOURCE] and
            contract.get("test", {}).get("sha256") == EXPECTED[TEST] and
            contract.get("authorization", {}).get("shard_execution") is False and
            contract.get("authorization", {}).get("reducer_execution") is False and
            contract.get("numbering") == {"source": "M1704",
                "future_review": "M1705", "future_release": "M1706",
                "forbidden_release": "M1683"},
            "contract boundary drift")
    return {"schema": "m1705_m1704_decoder_d0_authority_adapter_hammer_v1",
        "status": "PASS", "runtime": sys.version.split()[0],
        "identity": {"source_sha256": EXPECTED[SOURCE],
            "test_sha256": EXPECTED[TEST],
            "contract_sha256": EXPECTED[CONTRACT],
            "author_manifest_sha256": author_seal["manifest_sha256"],
            "author_outer_file_sha256": author_seal["outer_file_sha256"],
            "m1689_manifest_sha256": m1689_seal["manifest_sha256"],
            "m1689_outer_file_sha256": m1689_seal["outer_file_sha256"]},
        "adapter": adapter, "dynamic_synthetic_adapter": dynamic,
        "reducer": reducer, "cli": cli, "authority": mutations,
        "execution": {"payload_opens": 0, "actual_shard_runs": 0,
            "reducer_runs": 0, "attempt_writes": 0,
            "release_writes": 0, "gpu_runs": 0, "eda_runs": 0}}


def main():
    try:
        output = run_hammer()
        rc = 0
    except BaseException as error:
        output = {"schema": "m1705_m1704_decoder_d0_authority_adapter_hammer_v1",
            "status": "FAIL_CLOSED", "runtime": sys.version.split()[0],
            "error_type": type(error).__name__, "error": str(error),
            "authorization": {"release_authoring": False,
                "shard_execution": False, "reducer_execution": False}}
        rc = 1
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
