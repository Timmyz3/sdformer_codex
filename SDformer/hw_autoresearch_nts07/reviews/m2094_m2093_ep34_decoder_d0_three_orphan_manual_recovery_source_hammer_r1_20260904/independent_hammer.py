#!/usr/bin/env python3
"""Independent, non-production hammer for the M2093 recovery source.

All executable recovery-path tests replace every result, shard, attempt,
work, quarantine, and payload path with a temporary directory.  The only
production-state operation is the documented read-only source preflight.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HW / (
    "system_simulator/scripts/"
    "run_m2093_ep34_decoder_d0_three_orphan_manual_recovery.py")
CONTRACT = HW / (
    "contracts/m2093_ep34_decoder_d0_three_orphan_manual_recovery_"
    "source_contract_r1_20260904.json")
OUT = HERE / "mechanical_checks.json"
EXPECTED_SOURCE = (
    "4238f72026442983d3d8c2bf0ea69d09470c56d5b45784100fb27fa88730b757")
EXPECTED_CONTRACT = (
    "1c2a5fa7b27ddc2abbfab5545c83d959d44c3b6bfca5bd9dea9f42d81fde825e")
EXPECTED_DOCS359 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")
ORDINALS = (7560, 7561, 7562)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    require(sha256(SOURCE) == EXPECTED_SOURCE, "source identity drift")
    require(sha256(CONTRACT) == EXPECTED_CONTRACT, "contract identity drift")
    spec = importlib.util.spec_from_file_location("m2094_reviewed_m2093", str(SOURCE))
    require(spec is not None and spec.loader is not None, "cannot import source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_source()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def seal_tree(root):
    root = Path(root)
    members = []
    for path in root.rglob("*"):
        if path.is_file() and path.name not in (
                "SHA256SUMS", "SHA256SUMS.seal.sha256"):
            members.append(path.relative_to(root))
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(root / member), member.as_posix()) for member in sorted(members)),
        encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")


def double_seal(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(sha256(path) + "  " + path.name + "\n", encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        sha256(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


def subprocess_json(executable, mode):
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([executable, str(SOURCE), mode],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment, check=True,
        universal_newlines=True)
    require(not completed.stderr, "unexpected stderr from " + executable)
    return json.loads(completed.stdout)


def test_versions_and_preflight():
    versions = []
    outputs = {}
    for executable in ("/usr/bin/python3.6", "/usr/bin/python3.12"):
        require(Path(executable).is_file(), "missing interpreter " + executable)
        compile_run = subprocess.run([executable, "-c",
            "compile(open(%r,'rb').read(),%r,'exec')" %
            (str(SOURCE), str(SOURCE))], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        require(not compile_run.stdout and not compile_run.stderr,
                "compile output drift")
        outputs[executable] = {
            "describe": subprocess_json(executable, "--describe"),
            "preflight": subprocess_json(executable, "--preflight")}
        versions.append(executable)
    require(outputs[versions[0]] == outputs[versions[1]],
            "cross-version describe/preflight mismatch")
    preflight = outputs[versions[0]]["preflight"]
    require(preflight["execution"] is False and
            preflight["status"] == "PASS_M2093_SOURCE_PREFLIGHT_NO_EXECUTION",
            "preflight boundary drift")
    require([row["ordinal"] for row in preflight["orphans"]] == list(ORDINALS),
            "preflight orphan set drift")
    require(all(row["attempt_mode"] == "0400" and row["empty_work"] is True
                for row in preflight["orphans"]), "orphan topology drift")
    require(preflight["identity"] == M._identity(), "identity drift")
    return {"interpreters": versions,
        "describe_sha256": hashlib.sha256(canonical(
            outputs[versions[0]]["describe"])).hexdigest(),
        "preflight_sha256": hashlib.sha256(canonical(preflight)).hexdigest(),
        "orphans": preflight["orphans"], "identity": preflight["identity"]}


def test_static_order_and_scope():
    text = SOURCE.read_text(encoding="utf-8")
    require("ORDINALS = (7560, 7561, 7562)" in text, "ordinal scope drift")
    required = [
        "_validate_detached_launch()",
        "release_sha = _validate_future_gate()",
        "orphans = validate_source_topology()",
        "attempt_sha = _consume_outer_attempt(release_sha, orphans)",
        "QUARANTINE.mkdir(parents=True, mode=0o700)",
        "recovered = [_recover_one(ordinal, release_sha)"]
    positions = [text.index(token, text.index("def execute():")) for token in required]
    require(positions == sorted(positions), "execute authority/attempt order drift")
    recover_start = text.index("def _recover_one")
    recover_stop = text.index("\ndef execute", recover_start)
    recover = text[recover_start:recover_stop]
    tokens = ["original_attempt_sha = sha256(paths[\"attempt\"])",
        "M2090._rename_noreplace(paths[\"work\"], quarantine)",
        "paths[\"work\"].mkdir(mode=0o700)",
        "B.G.R.validate_authorities(True)",
        "B.ImmutableTimestepPlane(payload", "B._schedule_actual_shard(",
        "B.validate_shard_receipt(", "B.seal_work_tree(paths[\"work\"])",
        "M2090._rename_noreplace(paths[\"work\"], paths[\"result\"])"]
    rpos = [recover.index(token) for token in tokens]
    require(rpos == sorted(rpos), "recovery preserve/open/compute/publish order drift")
    require("consume_attempt(" not in recover and
            "_run_authorized_shard(" not in recover,
            "manual recovery creates/delegates a new M1681 attempt")
    contract = M.B.strict_json(CONTRACT)
    require(contract["ordinals"] == list(ORDINALS) and
            contract["recovery"]["new_m1681_shard_attempt_writes"] == 0 and
            contract["source_stage_authorization"]["manual_recovery_execution"] == 0 and
            contract["post_review_ceiling"] == {
                "m2095_release_authoring": 1,
                "manual_recovery_execution": 0,
                "reducer_execution": 0}, "source contract authorization drift")
    return {"execute_order_tokens": required,
        "recover_order_tokens": tokens, "ordinals": list(ORDINALS)}


def make_review_release(root, mutate_review=None, mutate_release=None):
    root = Path(root)
    review_dir = root / "review"
    review_dir.mkdir(parents=True)
    review = {"status": M.REVIEW_STATUS, "score_over_100": 95,
        "severity_counts": {"p0": 0, "p1": 0, "p2": 0},
        "identity": M._identity(),
        "authorization": {"m2095_release_authoring": 1,
            "manual_recovery_execution": 0, "reducer_execution": 0}}
    if mutate_review is not None:
        mutate_review(review)
    (review_dir / "review.json").write_text(json.dumps(
        review, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    seal_tree(review_dir)
    seal = M.B.verify_sealed_tree(review_dir, allow_ignored_pycache=False,
                                  label="temporary review")
    release_path = root / "release.json"
    identity = dict(M._identity(),
        review_sha256=sha256(review_dir / "review.json"),
        review_manifest_sha256=seal["manifest_sha256"],
        review_outer_file_sha256=seal["outer_file_sha256"])
    release = {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
        "identity": identity,
        "authorization": {"detached_launcher_runs": 1,
            "manual_recovery_shard_runs": 3, "payload_opens": 3,
            "new_m1681_shard_attempt_writes": 0,
            "outer_orchestration_attempt_writes": 1,
            "automatic_retry": False, "reducer_runs": 0,
            "gpu_runs": 0, "eda_runs": 0},
        "ordinals": list(ORDINALS),
        "claim_boundary": {"manual_recovery_only": True,
            "exact_m1681_compute_and_receipt_schema": True,
            "full_d0_result": False, "full_decoder": False,
            "system_speedup": False, "paper_result": False}}
    if mutate_release is not None:
        mutate_release(release)
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    double_seal(release_path)
    return review_dir, release_path


def test_future_gate():
    cases = []
    with tempfile.TemporaryDirectory(prefix="m2094_gate_") as directory:
        root = Path(directory)
        review, release = make_review_release(root / "good")
        with mock.patch.object(M, "FUTURE_REVIEW", review), \
                mock.patch.object(M, "FUTURE_RELEASE", release):
            require(M._validate_future_gate() == sha256(release),
                    "exact future gate rejected")
        cases.append("exact_gate_accept")
        mutations = [
            ("review_score_94", lambda row: row.__setitem__("score_over_100", 94), None),
            ("review_execution_auth", lambda row: row["authorization"].__setitem__(
                "manual_recovery_execution", 1), None),
            ("review_identity", lambda row: row["identity"].__setitem__(
                "source_sha256", "0" * 64), None),
            ("release_ordinal", None, lambda row: row.__setitem__(
                "ordinals", [7560, 7561, 7563])),
            ("release_reducer", None, lambda row: row["authorization"].__setitem__(
                "reducer_runs", 1)),
            ("release_paper", None, lambda row: row["claim_boundary"].__setitem__(
                "paper_result", True))]
        for index, (name, review_mutation, release_mutation) in enumerate(mutations):
            target = root / ("bad_%02d" % index)
            review, release = make_review_release(
                target, review_mutation, release_mutation)
            rejected = False
            with mock.patch.object(M, "FUTURE_REVIEW", review), \
                    mock.patch.object(M, "FUTURE_RELEASE", release):
                try:
                    M._validate_future_gate()
                except Exception:
                    rejected = True
            require(rejected, "future gate accepted " + name)
            cases.append(name + "_rejected")
    return cases


class FakeRss(object):
    def __init__(self):
        self.calls = 0
    def sample(self):
        self.calls += 1
    def summary(self):
        return {"gate_calls": self.calls}


class FakePlane(object):
    events = None
    outer_attempt = None
    def __init__(self, path, shape, expected_sha, timestep):
        require(Path(path).read_bytes() == b"payload", "temporary payload drift")
        require(sha256(path) == expected_sha, "temporary payload SHA drift")
        require(Path(self.outer_attempt).is_file(), "payload opened before outer attempt")
        self.opened_sha256 = expected_sha
        self.opened_size = Path(path).stat().st_size
        self.events.append("payload_open_%d" % timestep)


class FakeR(object):
    def __init__(self, payload_root, events):
        self.M1521_ROOT = Path(payload_root)
        self.events = events
    def validate_authorities(self, strong):
        require(strong is True, "weak temporary authority")
        self.events.append("authority")


class FakeP(object):
    RssGate = FakeRss


class FakeG(object):
    CHECKPOINT_SHA256 = "1" * 64
    RESOURCE_SHA256 = "2" * 64
    def __init__(self, payload_root, events):
        self.R = FakeR(payload_root, events)
        self.P = FakeP()
    def shard_descriptor(self, ordinal):
        return {"shard_ordinal": ordinal, "timestep": ordinal - 7560}
    def selected_record(self, shard):
        payload = self.R.M1521_ROOT / "payloads" / "payload.bin"
        return {"positive_output": "payloads/payload.bin",
            "positive_output_sha256": sha256(payload), "shape": [1]}
    def validate_three_configuration_metrics(self, metrics, shard):
        return {"cycles": metrics[0]["total_cycles"],
            "ordinal": shard["shard_ordinal"]}


class FakeB(object):
    SCHEMA = "m1681_ep34_decoder_d0_shard_execution_closure_successor_source_r1_v1"
    RESULT_SCHEMA = "m1681_ep34_decoder_d0_sealed_shard_result_r1_v1"
    RESULT_STATUS = "SHARD_COMPLETE__INDEPENDENT_RESULT_HAMMER_REQUIRED"
    CONFIGS = ("DENSE_TYPED_K8", "BIT_EQUAL_SERVICE_K1X8", "BIT_TYPED_K8")
    def __init__(self, root, events, fail_ordinal=None):
        self.root = Path(root)
        self.events = events
        self.fail_ordinal = fail_ordinal
        payload_root = self.root / "payload_root"
        (payload_root / "payloads").mkdir(parents=True)
        (payload_root / "payloads" / "payload.bin").write_bytes(b"payload")
        self.G = FakeG(payload_root, events)
        self.ImmutableTimestepPlane = FakePlane
    def namespace_paths(self, ordinal):
        base = self.root / ("shard_%d" % ordinal)
        return {"attempt": Path(str(base) + ".attempt"),
            "work": Path(str(base) + ".work"),
            "result": Path(str(base) + ".result"),
            "failure": Path(str(base) + ".failure")}
    @staticmethod
    def strict_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    def _schedule_actual_shard(self, shard, plane, rss):
        ordinal = shard["shard_ordinal"]
        self.events.append("schedule_%d" % ordinal)
        if ordinal == self.fail_ordinal:
            raise RuntimeError("injected temporary schedule failure")
        return [{"configuration": item, "total_cycles": ordinal}
                for item in self.CONFIGS]
    def validate_shard_receipt(self, row, ordinal, attempt_sha, release_sha):
        require(row["source_sha256"] == M.M1681_SHA256 and
                row["release_sha256"] == M.M1706_SHA256 and
                row["attempt_sha256"] == attempt_sha and
                row["shard_ordinal"] == ordinal and
                release_sha == M.M1706_SHA256,
                "temporary shard receipt identity drift")
    @staticmethod
    def seal_work_tree(root):
        seal_tree(root)
    @staticmethod
    def verify_sealed_tree(root, allow_ignored_pycache=False, label="tree"):
        require(not allow_ignored_pycache, "temporary pycache policy drift")
        root = Path(root)
        manifest = root / "SHA256SUMS"
        outer = root / "SHA256SUMS.seal.sha256"
        require(outer.read_text(encoding="ascii") ==
                sha256(manifest) + "  SHA256SUMS\n", label + " outer drift")
        return {"manifest_sha256": sha256(manifest),
            "outer_file_sha256": sha256(outer)}
    def verify_sealed_shard(self, ordinal):
        paths = self.namespace_paths(ordinal)
        require(paths["result"].is_dir() and paths["attempt"].is_file(),
                "temporary sealed shard missing")
        return True


def temp_rename_noreplace(source, target):
    source = Path(source)
    target = Path(target)
    if os.path.lexists(str(target)):
        raise FileExistsError(str(target))
    source.rename(target)


def prepare_temp_orphans(fake):
    attempts = {}
    for ordinal in ORDINALS:
        paths = fake.namespace_paths(ordinal)
        row = {"schema": fake.SCHEMA, "shard_ordinal": ordinal,
            "shard": fake.G.shard_descriptor(ordinal),
            "source_sha256": M.M1681_SHA256,
            "release_sha256": M.M1706_SHA256,
            "automatic_retry": False,
            "payload_opened_before_attempt": False}
        paths["attempt"].write_bytes(canonical(row) + b"\n")
        paths["attempt"].chmod(0o400)
        paths["work"].mkdir(mode=0o700)
        attempts[ordinal] = sha256(paths["attempt"])
    return attempts


def run_temp_recovery_case(fail_ordinal=None):
    with tempfile.TemporaryDirectory(prefix="m2094_recovery_") as directory:
        root = Path(directory)
        events = []
        fake = FakeB(root, events, fail_ordinal)
        attempts = prepare_temp_orphans(fake)
        outer_attempt = root / "outer.attempt"
        overall_result = root / "overall.result"
        overall_work = root / "overall.work"
        overall_failure = root / "overall.failure"
        quarantine = root / "quarantine"
        FakePlane.events = events
        FakePlane.outer_attempt = outer_attempt

        def rename_checked(source, target):
            require(outer_attempt.is_file(), "quarantine/publish before outer attempt")
            events.append("rename_" + Path(source).name)
            temp_rename_noreplace(source, target)

        orphan_rows = [{"ordinal": ordinal,
            "attempt_sha256": attempts[ordinal], "attempt_mode": "0400",
            "empty_work": True} for ordinal in ORDINALS]
        patches = [
            mock.patch.object(M, "B", fake),
            mock.patch.object(M, "HW", root),
            mock.patch.object(M, "ATTEMPT", outer_attempt),
            mock.patch.object(M, "RESULT", overall_result),
            mock.patch.object(M, "WORK", overall_work),
            mock.patch.object(M, "FAILURE", overall_failure),
            mock.patch.object(M, "QUARANTINE", quarantine),
            mock.patch.object(M, "_validate_detached_launch", lambda: None),
            mock.patch.object(M, "_validate_future_gate", lambda: "3" * 64),
            mock.patch.object(M, "validate_source_topology", lambda: orphan_rows),
            mock.patch.object(M.M2090, "_rename_noreplace", rename_checked)]
        for patch in patches:
            patch.start()
        try:
            failure_seen = False
            try:
                receipt = M.execute()
            except RuntimeError as error:
                require(fail_ordinal is not None and
                        "injected temporary" in str(error),
                        "unexpected temporary recovery failure")
                failure_seen = True
                receipt = None
            require(failure_seen == (fail_ordinal is not None),
                    "temporary failure disposition drift")
            require(outer_attempt.is_file(), "outer attempt not consumed")
            for ordinal in ORDINALS:
                paths = fake.namespace_paths(ordinal)
                require(paths["attempt"].is_file() and
                        sha256(paths["attempt"]) == attempts[ordinal] and
                        stat.S_IMODE(paths["attempt"].stat().st_mode) == 0o400,
                        "original attempt was not preserved")
            if fail_ordinal is None:
                require(receipt["new_m1681_shard_attempt_writes"] == 0 and
                        overall_result.is_dir() and not overall_failure.exists(),
                        "temporary success publication drift")
                require(all(fake.namespace_paths(ordinal)["result"].is_dir()
                            for ordinal in ORDINALS), "recovered result missing")
                require(all((quarantine /
                    ("shard_%04d_original_empty_work" % ordinal)).is_dir()
                    for ordinal in ORDINALS), "original empty work not preserved")
            else:
                require(not overall_result.exists() and overall_failure.is_dir(),
                        "outer failure evidence drift")
                before = [ordinal for ordinal in ORDINALS if ordinal < fail_ordinal]
                require(all(fake.namespace_paths(item)["result"].is_dir()
                            for item in before), "prior recovery lost on failure")
                require(fake.namespace_paths(fail_ordinal)["failure"].is_dir(),
                        "failed shard evidence missing")
                after = [ordinal for ordinal in ORDINALS if ordinal > fail_ordinal]
                require(all(fake.namespace_paths(item)["work"].is_dir()
                            for item in after), "unexecuted orphan was mutated")
            return {"failure_ordinal": fail_ordinal,
                "events": events, "original_attempts_preserved": True,
                "outer_attempt_before_first_event": True}
        finally:
            for patch in reversed(patches):
                patch.stop()


def test_attempt_and_no_replace():
    with tempfile.TemporaryDirectory(prefix="m2094_atomic_") as directory:
        root = Path(directory)
        attempt = root / "attempt"
        with mock.patch.object(M, "ATTEMPT", attempt):
            first = M._consume_outer_attempt("4" * 64, [])
            first_bytes = attempt.read_bytes()
            rejected = False
            try:
                M._consume_outer_attempt("4" * 64, [])
            except FileExistsError:
                rejected = True
            require(rejected and attempt.read_bytes() == first_bytes and
                    sha256(attempt) == first, "outer attempt O_EXCL drift")
        source = root / "source"
        target = root / "target"
        source.write_text("source", encoding="ascii")
        target.write_text("target", encoding="ascii")
        rejected = False
        try:
            M.M2090._rename_noreplace(source, target)
        except OSError:
            rejected = True
        require(rejected and source.read_text(encoding="ascii") == "source" and
                target.read_text(encoding="ascii") == "target",
                "RENAME_NOREPLACE overwrote target")
    return ["outer_attempt_o_excl", "rename_noreplace_existing_target"]


def test_detached_guard():
    cases = []
    with mock.patch.dict(M.os.environ, {}, clear=True):
        try:
            M._validate_detached_launch()
        except Exception:
            cases.append("missing_token_rejected")
    with mock.patch.dict(M.os.environ, {"M2093_MANUAL_RECOVERY": "1"}, clear=True), \
            mock.patch.object(M.os, "getsid", return_value=1), \
            mock.patch.object(M.os, "getpid", return_value=2):
        try:
            M._validate_detached_launch()
        except Exception:
            cases.append("non_session_leader_rejected")
    with mock.patch.dict(M.os.environ, {"M2093_MANUAL_RECOVERY": "1"}, clear=True), \
            mock.patch.object(M.os, "getsid", return_value=2), \
            mock.patch.object(M.os, "getpid", return_value=2), \
            mock.patch.object(M.os, "isatty", side_effect=lambda fd: fd == 1):
        try:
            M._validate_detached_launch()
        except Exception:
            cases.append("tty_rejected")
    require(len(cases) == 3, "detached guard negative drift")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args(argv)
    result = {"schema": "m2094_m2093_source_hammer_mechanical_r1_v1",
        "status": "PASS_M2094_MECHANICAL_SOURCE_CHECKS",
        "identity": M._identity(),
        "versions_and_read_only_preflight": test_versions_and_preflight(),
        "static_scope_and_order": test_static_order_and_scope(),
        "future_gate_attacks": test_future_gate(),
        "detached_guard_attacks": test_detached_guard(),
        "atomicity_attacks": test_attempt_and_no_replace(),
        "temporary_success": run_temp_recovery_case(None),
        "temporary_partial_failure": run_temp_recovery_case(7561),
        "production_payload_opens": 0, "production_shard_runs": 0,
        "production_namespace_mutations": 0, "gpu_runs": 0, "eda_runs": 0,
        "protected_docs359_sha256": sha256(M.DOCS359)}
    require(result["protected_docs359_sha256"] == EXPECTED_DOCS359,
            "protected docs359 changed")
    Path(args.output).write_text(json.dumps(result, indent=2,
        sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
