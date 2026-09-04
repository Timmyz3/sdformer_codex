#!/usr/bin/env python3
"""Independent, source-only hammer for the M1754 interpreter-bound wrapper.

This program performs no network access and never launches the production
interpreter, capture, analyzer, GPU, or EDA.  All execution-path tests replace
the production namespaces and ``os.execve`` inside an isolated temporary
directory.
"""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/run_m1754_m1747_tsbg_interpreter_bound_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1754_m1747_tsbg_interpreter_bound_one_shot.py"
CONTRACT = HW / "contracts/m1754_m1747_tsbg_interpreter_bound_execution_source_contract_r1_20260901.json"
FAILURE = HW / "results/m1754_m1749_m1747_tsbg_interpreter_failure_receipt_r1_20260901.json"
M1747 = HW / "system_simulator/scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py"
M1749 = HW / "contracts/m1749_m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_analysis_release_r1_20260901.json"
M1748 = HW / "reviews/m1748_m1747_m1727_ep34_tsbg_schema_identity_successor_source_hammer_r1_20260901"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise AssertionError("duplicate key " + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite " + token)))


def seal_file(module, path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    sidecar.write_text("{}  {}\n".format(module.sha256(path), path.name), encoding="ascii")
    outer.write_text("{}  {}\n".format(module.sha256(sidecar), sidecar.name), encoding="ascii")


def seal_dir(module, root):
    names = sorted(p.name for p in root.iterdir()
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(module.sha256(root / name), name)
                            for name in names), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(module.sha256(sums)), encoding="ascii")


def load_module():
    spec = importlib.util.spec_from_file_location("m1754_hammer_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(call, label):
    try:
        call()
    except Exception:
        return
    raise AssertionError("mutation accepted: " + label)


def make_authority(module, root):
    root.mkdir(parents=True, exist_ok=True)
    identities = module.source_identities()
    review = root / "review"
    review.mkdir()
    (review / "review.json").write_text(json.dumps({
        "schema": module.REVIEW_SCHEMA,
        "status": module.REVIEW_STATUS,
        "identity": identities,
        "authorization": {"m1756_release_may_be_created": True,
                          "execution": False, "analysis_run": False},
        "claim_boundary": {"paper_result": False},
    }, sort_keys=True), encoding="utf-8")
    seal_dir(module, review)
    review_binding = module.validate_future_review(review, identities)
    release = root / "release.json"
    release_identity = dict(identities)
    release_identity.update({
        "m1755_review_sha256": review_binding["review_sha256"],
        "m1755_review_outer_seal_file_sha256": review_binding["outer_seal_file_sha256"],
    })
    release.write_text(json.dumps({
        "schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS,
        "identity": release_identity,
        "authorization": {"wrapper_runs": 1, "interpreter_preflights": 1, "execs": 1,
                          "analysis_runs": 1, "capture_verifications": 1,
                          "result_publications": 1, "automatic_retry": False,
                          "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0},
        "claim_boundary": {"paper_result": False},
    }, sort_keys=True), encoding="utf-8")
    seal_file(module, release)
    return identities, review, review_binding, release


def mutate_scalar(value, index):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            return ("0" if value[0] != "0" else "1") + value[1:]
        return value + "__MUT{}".format(index)
    return None


def main():
    module = load_module()
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    checks = 0
    mutations = 0

    # Exact immutable identities and double seals.
    expected = {
        "source": "fd2e22a1d7031427b360ac97a2034b72237677c55414e6ed639aa852c201a027",
        "test": "e495000136482c159e9be55ff24183309dac4bc8907b04560964f6493c8cc103",
        "contract": "c172e0fc0356bed2a2f41727fa0b781057ae50e5567bd1703b86a6f6270d9bed",
        "m1747": "3bc48502ab1cccf579cfc65dc0cba2747e5bd38a8a4df82dda3f626f7283683b",
        "m1749": "6114020ab8d4da7c9a7c6f149496ee3efb1e7d19aeff5e34becaf60c1d465806",
        "failure": "57605ca6fa397429a4673f351cf2b01016dea7ff1dbcb29a01d1cbb4e4f12440",
        "failure_sidecar": "199c647a948df47990078739e0c0ebff0861f7723f63f2d3f77970bd2e90b666",
        "failure_outer": "382beb26c5e017f33041f99eb24cfd7d931d67dc918bfe8081f50cee0e9c8ebe",
        "m1748_review": "f9c3e152bb10d67a1e0b2421565e0f72469804fab4330dae9c00518b684e1c47",
        "m1748_manifest": "10683d2a63035841ef17572a5ca8b57a98eb260cb5b8c39d8d5eabbfb132e594",
        "m1748_outer": "d1ba7c36dff713385fc30817877f3228516f9a6fa862805a44e5f7d6355e07cc",
    }
    actual = {
        "source": sha256(SOURCE), "test": sha256(TEST), "contract": sha256(CONTRACT),
        "m1747": sha256(M1747), "m1749": sha256(M1749), "failure": sha256(FAILURE),
        "failure_sidecar": sha256(Path(str(FAILURE) + ".sha256")),
        "failure_outer": sha256(Path(str(FAILURE) + ".sha256.seal.sha256")),
        "m1748_review": sha256(M1748 / "review.json"),
        "m1748_manifest": sha256(M1748 / "SHA256SUMS"),
        "m1748_outer": sha256(M1748 / "SHA256SUMS.seal.sha256"),
    }
    assert actual == expected
    checks += len(expected)
    module.validate_contract()
    review_binding = module.validate_static()
    assert review_binding == {"review_sha256": expected["m1748_review"],
                              "manifest_sha256": expected["m1748_manifest"],
                              "outer_seal_file_sha256": expected["m1748_outer"]}
    checks += 2

    # The consumed failure is exact, is the /usr/bin/python3 torch-import
    # failure, and precedes every payload replay/publication.
    failure = strict_json(FAILURE)
    assert failure["status"] == "FAILED_CLOSED_BEFORE_PAYLOAD_REPLAY__CPU_TORCH_IMPORT_MISSING__M1747_NO_RETRY"
    assert failure["failed_identity"]["failed_interpreter"] == "/usr/bin/python3"
    assert failure["failed_identity"]["failed_python_version"] == "3.12.3"
    assert failure["observed_failure"]["initial_exception"] == "ModuleNotFoundError: No module named 'torch'"
    assert failure["observed_failure"]["terminal_exception"] == \
        "m1727_exact_m1721_base.M1721Error: future production analysis requires CPU torch"
    assert failure["observed_failure"]["traceback_frames"] == [
        "analyze_m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:581:checkpoint_fc1_betas",
        "analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:484:<module>",
        "analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:478:main",
        "analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:441:run_analysis",
        "analyze_m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:948:run_analysis",
        "analyze_m1721_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_source.py:583:checkpoint_fc1_betas",
    ]
    absence = failure["absence_and_budget"]
    assert absence == {
        "result_path": "hw_autoresearch_nts07/results/m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_20260901",
        "result_absent": True,
        "work_path": "hw_autoresearch_nts07/results/.m1747_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_r1_20260901.work",
        "work_absent": True, "analysis_invocations": 1, "capture_verification_entries": 1,
        "checkpoint_beta_extractions": 0, "payload_replays": 0, "result_publications": 0,
        "gpu_runs": 0, "eda_runs": 0, "automatic_retry": False,
        "m1749_authority_consumed": True,
    }
    checks += 7

    # Required production interpreter is bound statically.  The independent
    # review intentionally does not access the remote host or launch it.
    ids = module.source_identities()
    assert ids["interpreter_path"] == "/opt/conda/envs/sdformerflow/bin/python3.10"
    assert ids["interpreter_sha256"] == "89520a3f2bc6e4f670921bd7a71a66eb0073775e685f6cbefda0dcda7bc42aa0"
    assert (ids["python_version"], ids["torch_version"], ids["numpy_version"]) == \
           ("3.10.20", "2.2.2+cu121", "1.26.4")
    checks += 3

    # AST/source structure: no M1747 import, no network/process launcher, and
    # authority -> preflight -> fresh namespaces -> attempt -> exact execve.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not ({"socket", "requests", "urllib", "paramiko", "subprocess"} & imported)
    assert not any("analyze_m1747" in name for name in imported)
    body = source_text[source_text.index("def run_execution():"):
                       source_text.index("def source_self_check():")]
    ordered = ["verify_authority()", "interpreter_preflight()", "os.path.lexists(str(RESULT))",
               "ATTEMPT.mkdir()", "launch_receipt.json", "os.execve(str(INTERPRETER)"]
    positions = [body.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert 'env["PYTHONNOUSERSITE"] = "1"' in body and 'env["CUDA_VISIBLE_DEVICES"] = ""' in body
    assert '[str(INTERPRETER), str(M1747_SOURCE), "--run-analysis"]' in body
    assert "os.execve" not in source_text[source_text.index("def source_self_check():"):
                                            source_text.index("def main(")]
    checks += 7

    # Future M1755/M1756 authority mutation campaign.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        identities, review, rb, release = make_authority(module, root)
        assert module.validate_future_review(review, identities) == rb
        module.validate_future_release(release, rb, identities)
        checks += 2

        # Every identity member must be exact in the review.
        for index, key in enumerate(sorted(identities)):
            row = json.loads((review / "review.json").read_text(encoding="utf-8"))
            row["identity"][key] = mutate_scalar(row["identity"][key], index)
            (review / "review.json").write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
            seal_dir(module, review)
            expect_reject(lambda: module.validate_future_review(review, identities), "review identity " + key)
            mutations += 1
            identities, review, rb, release = make_authority(module, root / ("reset_i_" + str(index)))

        # Mutate review schema/status/authorization/claim, with valid reseals.
        review_fields = [("schema",), ("status",), ("authorization", "m1756_release_may_be_created"),
                         ("authorization", "execution"), ("authorization", "analysis_run"),
                         ("claim_boundary", "paper_result")]
        for index, path in enumerate(review_fields):
            base = root / ("review_field_" + str(index)); base.mkdir()
            i2, r2, rb2, l2 = make_authority(module, base)
            row = json.loads((r2 / "review.json").read_text(encoding="utf-8"))
            cursor = row
            for key in path[:-1]: cursor = cursor[key]
            cursor[path[-1]] = mutate_scalar(cursor[path[-1]], index)
            (r2 / "review.json").write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
            seal_dir(module, r2)
            expect_reject(lambda r=r2, i=i2: module.validate_future_review(r, i), "review field " + ".".join(path))
            mutations += 1

        # Mutate every release identity, budget, schema/status/claim.
        base = root / "release_mutations"; base.mkdir()
        i2, r2, rb2, l2 = make_authority(module, base)
        release_row = json.loads(l2.read_text(encoding="utf-8"))
        targets = [("identity", key) for key in sorted(release_row["identity"])]
        targets += [("authorization", key) for key in sorted(release_row["authorization"])]
        targets += [("schema",), ("status",), ("claim_boundary", "paper_result")]
        for index, path in enumerate(targets):
            case = root / ("release_case_" + str(index)); case.mkdir()
            ci, cr, crb, cl = make_authority(module, case)
            row = json.loads(cl.read_text(encoding="utf-8")); cursor = row
            for key in path[:-1]: cursor = cursor[key]
            cursor[path[-1]] = mutate_scalar(cursor[path[-1]], index)
            cl.write_text(json.dumps(row, sort_keys=True), encoding="utf-8"); seal_file(module, cl)
            expect_reject(lambda l=cl, b=crb, i=ci: module.validate_future_release(l, b, i),
                          "release field " + ".".join(path))
            mutations += 1

        # Unsealed mutation, duplicate key, and symlink review member attacks.
        base = root / "structural"; base.mkdir()
        si, sr, srb, sl = make_authority(module, base)
        sl.write_text(sl.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expect_reject(lambda: module.validate_future_release(sl, srb, si), "unsealed release")
        mutations += 1
        dup = root / "duplicate.json"; dup.write_text('{"x":1,"x":2}', encoding="utf-8")
        expect_reject(lambda: module.strict_json(dup), "duplicate JSON")
        mutations += 1
        member = sr / "review.json"; member.unlink(); member.symlink_to(CONTRACT)
        seal_dir(module, sr)
        expect_reject(lambda: module.validate_future_review(sr, si), "symlink review member")
        mutations += 1

    # Isolated execution order and fail-before-attempt behavior.
    class StopExec(Exception):
        pass

    original = {name: getattr(module, name) for name in
                ("verify_authority", "interpreter_preflight", "RESULT", "WORK", "ATTEMPT")}
    original_execve = module.os.execve
    original_lexists = module.os.path.lexists
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); order = []
            module.RESULT = root / "result"; module.WORK = root / "work"; module.ATTEMPT = root / "attempt"
            module.verify_authority = lambda: (order.append("authority") or ({"source_sha256": "a"},
                {"review_sha256": "b"}, {"release_sha256": "c"}))
            module.interpreter_preflight = lambda: (order.append("preflight") or {"python": "3.10.20"})
            def lexists(path):
                order.append("fresh:" + Path(path).name)
                return original_lexists(path)
            def execve(path, argv, env):
                order.append("execve")
                assert module.ATTEMPT.is_dir()
                receipt = strict_json(module.ATTEMPT / "launch_receipt.json")
                assert receipt["m1756_release_sha256"] == "c" and receipt["automatic_retry"] is False
                assert path == str(module.INTERPRETER)
                assert argv == [str(module.INTERPRETER), str(module.M1747_SOURCE), "--run-analysis"]
                assert env["PYTHONNOUSERSITE"] == "1" and env["CUDA_VISIBLE_DEVICES"] == ""
                raise StopExec()
            module.os.path.lexists = lexists; module.os.execve = execve
            try: module.run_execution()
            except StopExec: pass
            assert order == ["authority", "preflight", "fresh:result", "fresh:work", "execve"]
            checks += 1
            # The consumed attempt blocks a second run before execve.
            order[:] = []
            expect_reject(module.run_execution, "second wrapper attempt")
            assert "execve" not in order
            mutations += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); module.ATTEMPT = root / "attempt"; events = []
            module.verify_authority = lambda: (_ for _ in ()).throw(module.M1754Error("bad authority"))
            module.interpreter_preflight = lambda: events.append("preflight")
            expect_reject(module.run_execution, "authority failure")
            assert not events and not module.ATTEMPT.exists()
            mutations += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); module.ATTEMPT = root / "attempt"; events = []
            module.verify_authority = lambda: ({}, {}, {})
            module.interpreter_preflight = lambda: (_ for _ in ()).throw(module.M1754Error("bad interpreter"))
            module.os.path.lexists = lambda path: events.append("namespace") or False
            expect_reject(module.run_execution, "interpreter failure")
            assert not events and not module.ATTEMPT.exists()
            mutations += 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); module.RESULT = root / "result"; module.RESULT.mkdir()
            module.WORK = root / "work"; module.ATTEMPT = root / "attempt"
            module.verify_authority = lambda: ({}, {}, {})
            module.interpreter_preflight = lambda: {}
            module.os.path.lexists = original_lexists
            expect_reject(module.run_execution, "occupied result namespace")
            assert not module.ATTEMPT.exists()
            mutations += 1
    finally:
        for name, value in original.items(): setattr(module, name, value)
        module.os.execve = original_execve
        module.os.path.lexists = original_lexists

    # Inert source-self-check is run last against real, fresh namespaces.
    row = module.source_self_check()
    assert row["status"] == "PASS_M1754_SOURCE_SELF_CHECK__NO_EXECUTION"
    assert row["attempt_created"] is False and row["analysis_runs"] == 0
    checks += 2

    output = {
        "schema": "m1755_m1754_independent_hammer_output_r1_v1",
        "status": "PASS",
        "checks": checks,
        "negative_mutations_rejected": mutations,
        "live_remote_interpreter_checked": False,
        "network_access": False,
        "capture_runs": 0,
        "analysis_runs": 0,
        "gpu_runs": 0,
        "eda_runs": 0,
        "production_namespace_writes": 0,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
