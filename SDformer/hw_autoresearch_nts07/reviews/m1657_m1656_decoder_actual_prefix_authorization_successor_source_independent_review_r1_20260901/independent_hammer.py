#!/usr/bin/env python3
"""Different-author M1657 hammer; source/temporary-authority only.

This program never calls the private actual-prefix runner with real authority,
never opens the decoder payload, and never creates a canonical attempt/result.
"""
from __future__ import print_function

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / (
    "system_simulator/scripts/build_m1656_decoder_actual_prefix_"
    "authorization_successor_source.py")
TEST = HW / (
    "system_simulator/tests/test_m1656_decoder_actual_prefix_"
    "authorization_successor_source.py")
CONTRACT = HW / (
    "contracts/m1656_decoder_actual_prefix_authorization_successor_"
    "source_contract_r1_20260901.json")
SPEC = importlib.util.spec_from_file_location("m1657_hammer_source", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def seal_tree(root):
    members = sorted(path for path in root.iterdir()
                     if path.is_file() and path.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(path), path.name)
                                for path in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")


def seal_file(path):
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(sha256(path) + "  " + path.name + "\n",
                       encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        sha256(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


class Authority(object):
    def __init__(self, root, review_mutation=None, release_mutation=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.review = root / "review"
        self.release = root / "release.json"
        self.review.mkdir()
        review = {"status": M.REVIEW_STATUS, "score": 99,
            "p0_count": 0, "p1_count": 0,
            "identity": M._review_identity(),
            "authorization": {"release_authoring": True,
                "execution": False, "payload": False,
                "automatic_retry": False}}
        if review_mutation:
            review_mutation(review)
        (self.review / "review.json").write_text(
            json.dumps(review, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (self.review / "review.md").write_text("synthetic review\n",
                                               encoding="utf-8")
        seal_tree(self.review)
        identity = dict(M._review_identity(),
            review_sha256=sha256(self.review / "review.json"),
            review_manifest_sha256=sha256(self.review / "SHA256SUMS"),
            review_outer_file_sha256=sha256(
                self.review / "SHA256SUMS.seal.sha256"))
        release = {"schema": M.RELEASE_SCHEMA,
            "status": M.RELEASE_STATUS, "identity": identity,
            "authorization": {"actual_prefix_runs": 1,
                "payload_opens": 1, "attempt_writes": 1,
                "automatic_retry": False, "gpu_runs": 0,
                "eda_runs": 0, "all_other_runs": 0},
            "namespaces": M._namespaces(),
            "fixed_population": M._fixed_population(),
            "claim_boundary": {"prefix_only": True,
                "cycles_pending_hammer": True,
                "bytes_pending_hammer": True,
                "product_capture": False, "l3": False,
                "full_decoder": False, "production": False,
                "paper_result": False}}
        if release_mutation:
            release_mutation(release)
        self.release.write_text(
            json.dumps(release, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        seal_file(self.release)


def rejected(callable_):
    try:
        callable_()
    except (RuntimeError, OSError, ValueError, TypeError):
        return True
    return False


def bind(authority):
    old = M.FUTURE_REVIEW, M.FUTURE_RELEASE
    M.FUTURE_REVIEW, M.FUTURE_RELEASE = authority.review, authority.release
    return old


def restore(old):
    M.FUTURE_REVIEW, M.FUTURE_RELEASE = old


def run():
    checks = []
    attacks = []

    expected = {
        SOURCE: "5e1930598b1f107f231b280de0a9dc73d4589171d790f3732ace218ff8c91429",
        TEST: "28f1d4bdbeb63e35d9213eec09ce3746743e2b109559d12a94a9f0ca219fcf2c",
        CONTRACT: "f0e679e976a339889fef7fbf06f29f3beb0bf285e2b6465309da313acb843590",
    }
    for path, digest in expected.items():
        require(sha256(path) == digest, "identity drift: " + path.name)
        checks.append("exact_sha:" + path.name)
    M.validate_source_contract()
    checks.append("source_contract_semantics")
    disposition = M.verify_m1646_no_go_and_disposition()
    require(disposition["p1_id"] ==
            "P1_PRESENCE_ONLY_PRIVATE_EXECUTION_AUTHORIZATION",
            "M1646 P1 disposition drift")
    checks.append("m1646_exact_double_seal_and_p1_disposition")
    M.require_fresh_namespaces()
    checks.append("fresh_distinct_namespaces")

    with tempfile.TemporaryDirectory() as directory:
        authority = Authority(directory)
        old = bind(authority)
        try:
            release, release_sha = M.validate_future_review_and_release()
            require(release_sha == sha256(authority.release),
                    "valid release SHA mismatch")
            require(release["fixed_population"] == M._fixed_population(),
                    "fixed population mismatch")
        finally:
            restore(old)
    checks.append("valid_future_authority")

    review_mutations = [
        ("review_status", lambda row: row.update(status="wrong")),
        ("review_score", lambda row: row.update(score=94)),
        ("review_p0", lambda row: row.update(p0_count=1)),
        ("review_p1", lambda row: row.update(p1_count=1)),
        ("review_source_identity", lambda row:
            row["identity"].update(source_sha256="0" * 64)),
        ("review_release_authoring", lambda row:
            row["authorization"].update(release_authoring=False)),
        ("review_execution", lambda row:
            row["authorization"].update(execution=True)),
        ("review_payload", lambda row:
            row["authorization"].update(payload=True)),
        ("review_retry", lambda row:
            row["authorization"].update(automatic_retry=True)),
    ]
    for name, mutation in review_mutations:
        with tempfile.TemporaryDirectory() as directory:
            authority = Authority(directory, review_mutation=mutation)
            old = bind(authority)
            try:
                require(rejected(M.validate_future_review_and_release), name)
            finally:
                restore(old)
        attacks.append(name)

    release_mutations = [
        ("release_schema", lambda row: row.update(schema="wrong")),
        ("release_status", lambda row: row.update(status="wrong")),
        ("release_review_identity", lambda row:
            row["identity"].update(review_sha256="0" * 64)),
        ("release_checkpoint_identity", lambda row:
            row["identity"].update(checkpoint_sha256="0" * 64)),
        ("release_runs", lambda row:
            row["authorization"].update(actual_prefix_runs=2)),
        ("release_payload_opens", lambda row:
            row["authorization"].update(payload_opens=2)),
        ("release_attempts", lambda row:
            row["authorization"].update(attempt_writes=2)),
        ("release_retry", lambda row:
            row["authorization"].update(automatic_retry=True)),
        ("release_gpu", lambda row:
            row["authorization"].update(gpu_runs=1)),
        ("release_eda", lambda row:
            row["authorization"].update(eda_runs=1)),
        ("release_other", lambda row:
            row["authorization"].update(all_other_runs=1)),
        ("release_result_namespace", lambda row:
            row["namespaces"].update(result="results/wrong")),
        ("release_attempt_namespace", lambda row:
            row["namespaces"].update(attempt="results/wrong")),
        ("release_stage", lambda row:
            row["fixed_population"].update(decoder_stage="D1")),
        ("release_call", lambda row:
            row["fixed_population"].update(call_ordinal=1)),
        ("release_module", lambda row:
            row["fixed_population"].update(module_ordinal=1)),
        ("release_timestep", lambda row:
            row["fixed_population"].update(timestep=1)),
        ("release_destinations", lambda row:
            row["fixed_population"].update(destinations=list(range(41)))),
        ("release_configuration_order", lambda row:
            row["fixed_population"].update(
                configuration_order=list(reversed(M.P.CONFIGS)))),
        ("release_product_capture", lambda row:
            row["claim_boundary"].update(product_capture=True)),
        ("release_full_decoder", lambda row:
            row["claim_boundary"].update(full_decoder=True)),
        ("release_paper", lambda row:
            row["claim_boundary"].update(paper_result=True)),
    ]
    for name, mutation in release_mutations:
        with tempfile.TemporaryDirectory() as directory:
            authority = Authority(directory, release_mutation=mutation)
            old = bind(authority)
            try:
                require(rejected(M.validate_future_review_and_release), name)
            finally:
                restore(old)
        attacks.append(name)

    for name in ("review_tree", "review_json", "review_manifest",
                 "review_outer", "release", "release_sidecar",
                 "release_outer"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = Authority(root / "authority")
            if name == "review_tree":
                saved = root / "saved_review"
                authority.review.rename(saved)
                authority.review.symlink_to(saved, target_is_directory=True)
            else:
                mapping = {"review_json": authority.review / "review.json",
                    "review_manifest": authority.review / "SHA256SUMS",
                    "review_outer": authority.review /
                        "SHA256SUMS.seal.sha256",
                    "release": authority.release,
                    "release_sidecar": Path(str(authority.release) +
                                             ".sha256"),
                    "release_outer": Path(str(authority.release) +
                        ".sha256.seal.sha256")}
                target = mapping[name]
                saved = Path(str(target) + ".saved")
                target.rename(saved)
                target.symlink_to(saved)
            old = bind(authority)
            try:
                require(rejected(M.validate_future_review_and_release),
                        "symlink_" + name)
            finally:
                restore(old)
        attacks.append("symlink_" + name)

    with tempfile.TemporaryDirectory() as directory:
        copied = Path(directory) / "m1646"
        shutil.copytree(str(M.M1646), str(copied))
        review = copied / "review.json"
        row = json.loads(review.read_text(encoding="utf-8"))
        row["authorization"]["m1645_execution"] = True
        review.write_text(json.dumps(row, sort_keys=True) + "\n",
                          encoding="utf-8")
        seal_tree(copied)
        old = M.M1646
        M.M1646 = copied
        try:
            require(rejected(M.verify_m1646_no_go_and_disposition),
                    "M1646 semantic reseal bypassed exact pin")
        finally:
            M.M1646 = old
    attacks.append("m1646_semantic_reseal_against_exact_pin")

    events = []
    originals = (M.verify_pre_payload_authorities, M.consume_attempt,
                 M.P.R.validate_authorities, M.P._selected_payload,
                 M.P.RssGate, M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        M.RESULT, M.ATTEMPT = root / "result", root / "attempt"
        M.WORK, M.FAILURE = root / "work", root / "failure"
        M.verify_pre_payload_authorities = lambda **_kw: (_ for _ in ()).throw(
            M.M1656Error("gate rejected"))
        M.consume_attempt = lambda *_args: events.append("attempt")
        M.P.R.validate_authorities = lambda *_args: events.append("predecessor")
        M.P._selected_payload = lambda: events.append("payload")
        M.P.RssGate = lambda: events.append("rss")
        try:
            require(rejected(M._run_authorized_actual_prefix),
                    "failed authority not rejected")
        finally:
            (M.verify_pre_payload_authorities, M.consume_attempt,
             M.P.R.validate_authorities, M.P._selected_payload,
             M.P.RssGate, M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE) = originals
    require(events == [], "failed authority reached runtime seam")
    attacks.append("failed_authority_blocks_attempt_payload_rss")

    class StopAtRss(RuntimeError):
        pass
    events = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        M.RESULT, M.ATTEMPT = root / "result", root / "attempt"
        M.WORK, M.FAILURE = root / "work", root / "failure"
        M.verify_pre_payload_authorities = lambda **_kw: (
            events.append("authority") or {"release_sha256": "1" * 64})
        M.consume_attempt = lambda *_args: events.append("attempt")
        M.P.R.validate_authorities = lambda *_args: events.append("predecessor")
        M.P._selected_payload = lambda: (
            events.append("payload_select") or (Path("never-opened"), (),
                                                  "0" * 64))
        def stop_rss():
            events.append("rss")
            raise StopAtRss()
        M.P.RssGate = stop_rss
        try:
            require(rejected(M._run_authorized_actual_prefix),
                    "RSS stop not observed")
        finally:
            (M.verify_pre_payload_authorities, M.consume_attempt,
             M.P.R.validate_authorities, M.P._selected_payload,
             M.P.RssGate, M.RESULT, M.ATTEMPT, M.WORK, M.FAILURE) = originals
    require(events == ["authority", "attempt", "predecessor",
                       "payload_select", "rss"],
            "runtime order drift: " + repr(events))
    checks.append("authority_before_attempt_payload_and_rss")

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Name) and
             node.func.id == "_run_authorized_actual_prefix"]
    require(calls == [], "public/CLI surface reaches private runner")
    checks.append("private_runner_unreachable_from_cli")

    return {"schema": "m1657_independent_hammer_r1_v1",
            "status": "PASS", "checks": checks,
            "checks_passed": len(checks), "attacks": attacks,
            "attacks_rejected": len(attacks),
            "actual_payload": False, "actual_execution": False,
            "attempt_writes": 0, "gpu_runs": 0, "eda_runs": 0}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
