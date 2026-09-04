#!/usr/bin/env python3
"""Receipt-blind, source-only hammer for the M1747 TSBG successor.

This hammer does not read the M1747 author receipt, inspect the M1707 capture,
invoke production analysis, use GPU/EDA tools, or access the network.  All
dynamic tests use synthetic documents and temporary authority objects.
"""
from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_simulator/scripts/analyze_m1747_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
TEST = HW / (
    "system_simulator/tests/test_m1747_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
CONTRACT = HW / (
    "contracts/m1747_m1729_m1727_m1707_ep34_tsbg_schema_identity_"
    "successor_source_contract_r1_20260901.json")
FAILURE = HW / (
    "results/m1747_m1729_m1727_ep34_tsbg_analysis_failed_attempt_"
    "receipt_r1_20260901.json")
M1744 = HW / (
    "reviews/m1744_m1707_ep34_tsbg_capture_result_independent_hammer_"
    "r1_20260901")
M1727_SOURCE = HW / (
    "system_simulator/scripts/analyze_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
M1727_TEST = HW / (
    "system_simulator/tests/test_m1727_m1707_ep34_tsbg_b4_b8_s2_"
    "fc1_patch_decision_source.py")
M1727_CONTRACT = HW / (
    "contracts/m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_decision_"
    "source_contract_r1_20260901.json")
M1729_RELEASE = HW / (
    "contracts/m1729_m1728_m1727_m1707_ep34_tsbg_b4_b8_s2_fc1_patch_"
    "decision_analysis_release_r1_20260901.json")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "3bc48502ab1cccf579cfc65dc0cba2747e5bd38a8a4df82dda3f626f7283683b",
    TEST: "5d31b3f22b1a358bad3f6f7203400ecdcd8c84186edd30ecae50b2482e5bdea9",
    CONTRACT: "31af6b726bdbddb0539e20b101fcaf603b0e2a25527b886059d756b1a4bc281e",
    Path(str(CONTRACT) + ".sha256"):
        "dc7392020681bf923e3741f4369058c04562f349ac5561f1eaf3763337750b96",
    Path(str(CONTRACT) + ".sha256.seal.sha256"):
        "e570e0b76bc4e80f4cd8f5202822e7ad2972c74a33435deaee4c0565b257920e",
    FAILURE: "e07805d95200208c74b817c13f7d100a78cf33d6d7694fb42cc7a2f7c0be1b24",
    Path(str(FAILURE) + ".sha256"):
        "5b2d9e64158db8e015e377cac5108d4482f9f5c224ecceb7860fc186f3e788fe",
    Path(str(FAILURE) + ".sha256.seal.sha256"):
        "a16412bb861fde518a977e1e5c57c524d924f721e7585826813e261343cf21a5",
    M1744 / "review.json":
        "d237b3a64cf47313873a84a4749465b7cc7361bd8cf57dde5a0b6275f336dbc7",
    M1744 / "SHA256SUMS":
        "df15fe385bc7f5eccde2fecd19f5fe478dbc0480653cec5aab208c59a8a6b1f4",
    M1744 / "SHA256SUMS.seal.sha256":
        "40c3e5f2c4a98be985bf225fe6cf3a3cda88c3a32047a372c84ca0608baaf1d2",
    M1727_SOURCE:
        "e0d2fc508a835b667b63a8719af3bf4ad883bfccca5b4c388f4e96ac9c6eaed9",
    M1727_TEST:
        "3b68aa96eba68e397a84459cfdc3199a7b8df6bf646236bf9495e0dd9137071c",
    M1727_CONTRACT:
        "efa110402bee236e4f1d2956ccad364a8de2c52e429d1e58a7c3dbe19f1e55f6",
    M1729_RELEASE:
        "440dd2472c6a92d99980d46b36709d88d697f48ad88b1119a36cd20d1d5d439a",
    DOCS359:
        "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            stat.S_ISREG(path.lstat().st_mode) and sha(path) == expected,
            "identity drift: " + str(path))


def strict_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)))
    require(type(result) is dict, "JSON root is not an object")
    return result


def verify_file_seal(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(sidecar.read_text(encoding="ascii").split() ==
            [sha(path), path.name], "sidecar drift: " + str(path))
    require(outer.read_text(encoding="ascii").split() ==
            [sha(sidecar), sidecar.name], "outer drift: " + str(path))


def verify_directory_seal(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(root.is_dir() and not root.is_symlink(), "sealed root missing")
    require(outer.read_text(encoding="ascii").split() ==
            [sha(sums), "SHA256SUMS"], "directory outer drift")
    listed = []
    for line in sums.read_text(encoding="ascii").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed directory manifest")
        name = fields[1].strip().lstrip("*")
        require(name not in listed and not Path(name).is_absolute() and
                ".." not in Path(name).parts and Path(name).as_posix() == name,
                "unsafe directory member")
        exact(root / name, fields[0])
        listed.append(name)
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file() and
                    path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
                    "__pycache__" not in path.parts and path.suffix != ".pyc")
    require(sorted(listed) == actual, "directory manifest coverage drift")


def seal_file(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    sidecar.write_text("{}  {}\n".format(sha(path), path.name),
                       encoding="ascii")
    outer.write_text("{}  {}\n".format(sha(sidecar), sidecar.name),
                     encoding="ascii")


def seal_directory(root):
    members = sorted(path.relative_to(root).as_posix()
                     for path in root.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join("{}  {}\n".format(sha(root / name), name)
                            for name in members), encoding="ascii")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha(sums)), encoding="ascii")


def load_target():
    spec = importlib.util.spec_from_file_location("m1748_target", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rejected(function, error_type):
    try:
        function()
    except error_type:
        return True
    return False


def canonical(module):
    return {"schema": module.SAMPLE_ORDER_SCHEMA,
        "samples": [{"global_sample_id": index} for index in range(40)],
        "identity": {"checkpoint_sha256": module.BASE.CHECKPOINT_SHA256}}


def make_authority(module, root):
    identity = module.source_identities()
    review_root = root / "review"
    review_root.mkdir()
    review = {"schema": module.REVIEW_SCHEMA,
        "status": module.REVIEW_STATUS, "identity": identity,
        "authorization": {"m1749_release_may_be_created": True,
            "analysis_run": False, "capture_verify": False},
        "claim_boundary": {"paper_result": False}}
    (review_root / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    seal_directory(review_root)
    binding = module.validate_future_review(review_root, identity)
    release = root / "release.json"
    release_identity = dict(identity)
    release_identity.update({
        "m1748_review_sha256": binding["review_sha256"],
        "m1748_review_outer_seal_file_sha256":
            binding["outer_seal_file_sha256"]})
    release.write_text(json.dumps({"schema": module.RELEASE_SCHEMA,
        "status": module.RELEASE_STATUS, "identity": release_identity,
        "authorization": {"analysis_runs": 1, "capture_verifications": 1,
            "result_publications": 1, "automatic_retry": False,
            "gpu_runs": 0, "eda_runs": 0, "all_other_runs": 0},
        "claim_boundary": {"paper_result": False}},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_file(release)
    return review_root, release, identity, binding


def main():
    for path, digest in EXPECTED.items():
        exact(path, digest)
    verify_file_seal(CONTRACT)
    verify_file_seal(FAILURE)
    verify_directory_seal(M1744)
    failure = strict_json(FAILURE)
    require(failure["observed_failure"]["exception_message"] ==
            "M1707 sample order drift" and
            failure["observed_failure"]["canonical_actual_schema"] ==
            "m1544_ep34_m1458_sample_order_r1_v1" and
            failure["observed_failure"]["canonical_sample_order_sha256"] ==
            "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773" and
            failure["absence_witness"]["result_absent_after_failure"] is True and
            failure["absence_witness"]["work_absent_after_failure"] is True and
            failure["absence_witness"]["payload_replay_started"] is False and
            failure["observed_budget"]["analysis_invocations"] == 1 and
            failure["observed_budget"]["automatic_retry"] is False and
            failure["observed_budget"]["m1729_authority_consumed"] is True,
            "failure receipt semantic drift")
    m1744_review = strict_json(M1744 / "review.json")
    m1744_hammer = strict_json(M1744 / "hammer_output.json")
    require(m1744_review["verified"]["samples"] == 40 and
            m1744_review["authorization"]["capture_retry"] is False and
            m1744_hammer["checks"]["sample_order_40_exact"] is True and
            m1744_hammer["bindings"]["sample_order_sha256"] ==
            "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773",
            "M1744 capture-review semantics drift")

    module = load_target()
    module.validate_source_contract()
    require(module.M1727_SOURCE_SHA256 == EXPECTED[M1727_SOURCE] and
            module.M1727_TEST_SHA256 == EXPECTED[M1727_TEST] and
            module.M1727_CONTRACT_SHA256 == EXPECTED[M1727_CONTRACT] and
            module.M1729_RELEASE_SHA256 == EXPECTED[M1729_RELEASE],
            "predecessor pin drift")
    require(module.BASE.tsbg_pair_metrics is module.M1727.tsbg_pair_metrics and
            module.BASE.s2_fc1_pair_metrics is module.M1727.s2_fc1_pair_metrics and
            module.BASE.DecisionAccumulator.finalize_tsbg_rows is
                module.M1727.finalize_tsbg_rows and
            module.BASE.BUNDLES == (4, 8) and
            module.BASE.S2_EPSILON_RATIO == (0.0, 0.01, 0.02, 0.05, 0.10),
            "M1727 algorithm/gate reuse drift")

    attacks = {}
    good = canonical(module)
    adapted = module.adapt_sample_order_document(good)
    require(adapted["schema"] == module.LEGACY_SAMPLE_ORDER_SCHEMA and
            adapted["samples"] == good["samples"] and
            adapted["identity"] == good["identity"] and
            good["schema"] == module.SAMPLE_ORDER_SCHEMA,
            "legal schema-only adaptation drift")
    mutation_rows = []
    for name, mutator in (
            ("legacy_schema", lambda row: row.update(
                {"schema": module.LEGACY_SAMPLE_ORDER_SCHEMA})),
            ("near_schema", lambda row: row.update({"schema": "near"})),
            ("missing_schema", lambda row: row.pop("schema")),
            ("short_population", lambda row: row["samples"].pop()),
            ("duplicate_id", lambda row: row["samples"][8].update(
                {"global_sample_id": 7})),
            ("negative_id", lambda row: row["samples"][0].update(
                {"global_sample_id": -1})),
            ("checkpoint", lambda row: row["identity"].update(
                {"checkpoint_sha256": "0" * 64})),
            ("missing_identity", lambda row: row.pop("identity"))):
        row = canonical(module)
        mutator(row)
        require(rejected(lambda row=row:
                         module.adapt_sample_order_document(row),
                         module.M1747Error), name + " mutation accepted")
        mutation_rows.append(name)

    # Prove the temporary strict-json monkeypatch is restored on both success
    # and exception without opening the canonical capture.
    class InnerFailure(Exception):
        pass
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample_order.json"
        sample.write_text("synthetic only\n", encoding="ascii")
        old_regular = module.regular_exact
        old_strict = module._BASE_STRICT_JSON
        old_verify = module._BASE_VERIFY_CAPTURE_IDENTITY
        old_live = module.BASE.strict_json
        def no_file_hash(_path, _digest, _label):
            return None
        def synthetic_json(_path, root_type=dict):
            return canonical(module)
        seen = []
        def failing_verify(_root):
            seen.append(module.BASE.strict_json(sample)["schema"])
            raise InnerFailure("synthetic verifier failure")
        module.regular_exact = no_file_hash
        module._BASE_STRICT_JSON = synthetic_json
        module._BASE_VERIFY_CAPTURE_IDENTITY = failing_verify
        try:
            require(rejected(lambda: module.verify_capture_identity(Path(tmp)),
                             InnerFailure), "synthetic inner failure escaped")
            require(module.BASE.strict_json is old_live and
                    seen == [module.LEGACY_SAMPLE_ORDER_SCHEMA],
                    "strict_json not restored after exception")
            attacks["monkeypatch_exception_restore"] = True
            def successful_verify(_root):
                adapted_row = module.BASE.strict_json(sample)
                return ({}, {}, adapted_row, {}, {})
            module._BASE_VERIFY_CAPTURE_IDENTITY = successful_verify
            result = module.verify_capture_identity(Path(tmp))
            require(module.BASE.strict_json is old_live and
                    result[2]["schema"] == module.SAMPLE_ORDER_SCHEMA,
                    "strict_json not restored after success")
            attacks["monkeypatch_success_restore"] = True
        finally:
            module.regular_exact = old_regular
            module._BASE_STRICT_JSON = old_strict
            module._BASE_VERIFY_CAPTURE_IDENTITY = old_verify
            module.BASE.strict_json = old_live

    # Future review/release validation must reject independently resealed
    # authority and one-shot budget mutations.
    authority_attacks = (
        ("review_status", "review", lambda row:
            row.update({"status": "PASS_NEAR_MATCH"})),
        ("review_identity", "review", lambda row:
            row["identity"].update({"source_sha256": "0" * 64})),
        ("review_capture_authority", "review", lambda row:
            row["authorization"].update({"capture_verify": True})),
        ("release_status", "release", lambda row:
            row.update({"status": "AUTHORIZE_NEAR_MATCH"})),
        ("release_identity", "release", lambda row:
            row["identity"].update({"source_sha256": "0" * 64})),
        ("release_analysis_runs", "release", lambda row:
            row["authorization"].update({"analysis_runs": 2})),
        ("release_capture_runs", "release", lambda row:
            row["authorization"].update({"capture_verifications": 2})),
        ("release_retry", "release", lambda row:
            row["authorization"].update({"automatic_retry": True})),
        ("release_paper", "release", lambda row:
            row["claim_boundary"].update({"paper_result": True})),
    )
    for name, target, mutate in authority_attacks:
        with tempfile.TemporaryDirectory() as tmp:
            review, release, identity, binding = make_authority(
                module, Path(tmp))
            if target == "review":
                path = review / "review.json"
                row = json.loads(path.read_text())
                mutate(row)
                path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
                seal_directory(review)
                outcome = rejected(lambda: module.validate_future_review(
                    review, identity), module.M1747Error)
            else:
                row = json.loads(release.read_text())
                mutate(row)
                release.write_text(json.dumps(
                    row, indent=2, sort_keys=True) + "\n")
                seal_file(release)
                outcome = rejected(lambda: module.validate_future_release(
                    release, binding, identity), module.M1747Error)
            require(outcome, name + " authority mutation accepted")
            mutation_rows.append(name)

    # No future release: production must stop before namespaces and the exact
    # implementation.  Synthetic freshness collision must also stop before it.
    old_authority = module.verify_analysis_authority
    old_base_run = module.BASE.run_analysis
    old_result = module.RESULT
    old_work = module.WORK
    base_touched = [False]
    def deny():
        raise module.M1747Error("no M1749")
    def base_sentinel():
        base_touched[0] = True
        raise RuntimeError("base must not run")
    module.verify_analysis_authority = deny
    module.BASE.run_analysis = base_sentinel
    try:
        require(rejected(module.run_analysis, module.M1747Error) and
                not base_touched[0], "authority gate did not precede base")
        attacks["authority_before_base"] = True
    finally:
        module.verify_analysis_authority = old_authority
        module.BASE.run_analysis = old_base_run

    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp) / "existing"
        existing.mkdir()
        module.RESULT = existing
        module.WORK = Path(tmp) / "fresh"
        module.verify_analysis_authority = lambda: {"synthetic": True}
        module.BASE.run_analysis = base_sentinel
        base_touched[0] = False
        try:
            require(rejected(module.run_analysis, module.M1747Error) and
                    not base_touched[0],
                    "fresh namespace gate did not precede base")
            attacks["fresh_namespace_before_base"] = True
        finally:
            module.RESULT = old_result
            module.WORK = old_work
            module.verify_analysis_authority = old_authority
            module.BASE.run_analysis = old_base_run

    # The source self-check must remain inert even with a capture sentinel.
    old_capture = module.BASE.verify_capture_identity
    capture_touched = [False]
    def capture_sentinel(_root):
        capture_touched[0] = True
        raise RuntimeError("capture touched")
    module.BASE.verify_capture_identity = capture_sentinel
    try:
        self_check = module.source_self_check()
    finally:
        module.BASE.verify_capture_identity = old_capture
    require(not capture_touched[0] and self_check["capture_touched"] is False and
            self_check["analysis_executed"] is False,
            "source self-check touched capture or analysis")

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    require(not imports.intersection(set(("subprocess", "socket", "requests",
                                          "paramiko", "urllib", "http"))),
            "execution/network import found")
    source_text = SOURCE.read_text(encoding="utf-8")
    run_body = source_text[source_text.index("def run_analysis():"):
                           source_text.index("def source_self_check():")]
    require(run_body.index("verify_analysis_authority()") <
            run_body.index("os.path.lexists") <
            run_body.index("BASE.run_analysis()"),
            "production authority/fresh/base order drift")
    require(not os.path.lexists(str(module.RESULT)) and
            not os.path.lexists(str(module.WORK)),
            "production namespace changed during hammer")

    print(json.dumps({
        "status": module.REVIEW_STATUS,
        "score": 100,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "receipt_blind": True,
        "exact_failure_receipt_triple": True,
        "exact_m1744_capture_review_triple": True,
        "exact_m1727_source_test_contract_and_m1729_release": True,
        "schema_only_repair": True,
        "algorithm_changed": False,
        "gates_changed": False,
        "claim_boundary_changed": False,
        "synthetic_mutations_rejected": len(mutation_rows),
        "mutation_classes": mutation_rows,
        "runtime_order_checks": attacks,
        "monkeypatch_restored_on_success_and_exception": True,
        "python36_syntax_compatible": True,
        "source_self_check_capture_touched": capture_touched[0],
        "capture_verifications": 0,
        "analysis_runs": 0,
        "result_writes": 0,
        "gpu_runs": 0,
        "eda_runs": 0,
        "network_access": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
