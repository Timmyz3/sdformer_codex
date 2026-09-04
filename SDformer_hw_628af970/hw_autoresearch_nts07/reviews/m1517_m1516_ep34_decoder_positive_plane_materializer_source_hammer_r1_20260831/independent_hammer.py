#!/usr/bin/env python3
"""Independent CPU/static hammer for M1516; never touches production paths."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1516_ep34_decoder_positive_plane_materializer_source.py"
TEST = HW / "system_simulator/tests/test_m1516_ep34_decoder_positive_plane_materializer_source.py"
CONTRACT = HW / "contracts/m1516_ep34_decoder_positive_plane_materializer_source_contract_r1_20260831.json"
EXPECTED = {
    "source": "b712e3246f1cca5ac857017439fc75a7bccc8a87e7e09763a19f0d50806b94ef",
    "test": "aa88c18b4b90c8f01e24053cf96044adda4677f49f340f9a143bdaa3a631cfe6",
    "contract": "f5e8536135afd3817305997068761816c840885828d9136fd18b6650b3c7c756",
}
TINY_SHAPES = ((1, 1, 1, 1, 9),) * 4
POSITIVE = bytes((0x05, 0x01))
NEGATIVE = bytes((0x00, 0x00))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1517_hammer_target", SOURCE)


def synthetic_audit() -> dict:
    positive_sha = hashlib.sha256(POSITIVE).hexdigest()
    negative_sha = hashlib.sha256(NEGATIVE).hexdigest()
    support_sha = hashlib.sha256(POSITIVE + NEGATIVE).hexdigest()
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 247 + 200 + module,
            "global_sample_id": sample,
            "sequence": "independent_sequence_%d" % (sample // 10),
            "sample_key": "independent_%02d" % sample,
            "module_ordinal": module,
            "module": M.M1510.M1323.MODULES[module],
            "shape": list(TINY_SHAPES[module]),
            "support_sign": "payloads/independent_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": support_sha,
            "source_support_sign_sha256": support_sha,
            "positive_plane_sha256": positive_sha,
            "negative_zero_plane_sha256": negative_sha,
            "positive_plane_bytes": 2,
            "negative_plane_bytes": 2,
            "plane_bytes": 2,
            "positive_word_uint32": M.EXPECTED_SCALE_WORDS[module],
            "negative_count": 0,
            "nonfinite_count": 0,
        })
    layers = [{
        "module_ordinal": ordinal,
        "module": M.M1510.M1323.MODULES[ordinal],
        "calls": 30,
        "word_uint32": M.EXPECTED_SCALE_WORDS[ordinal],
        "word_hex": "0x{:08x}".format(M.EXPECTED_SCALE_WORDS[ordinal]),
        "all_calls_same_word": True,
    } for ordinal in range(4)]
    return {
        "schema": M.M1510.SCHEMA,
        "status": M.M1510.STATUS,
        "capture_seal": {
            "sha256sums_sha256": M.CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": M.CAPTURE_OUTER_SHA256,
        },
        "layer_scale_words": layers,
        "calls": calls,
    }


def build(audit=None):
    old = M.M1510.M1323.SHAPES
    M.M1510.M1323.SHAPES = TINY_SHAPES
    try:
        return M.build_output_manifest(synthetic_audit() if audit is None else audit)
    finally:
        M.M1510.M1323.SHAPES = old


def rejected(function) -> bool:
    try:
        function()
    except BaseException:
        return True
    return False


def write_stage(root: Path, manifest: dict) -> None:
    (root / "payloads").mkdir(parents=True)
    for row in manifest["records"]:
        path = root.joinpath(*Path(row["positive_output"]).parts)
        M.write_exclusive(path, POSITIVE, 0o400)
    M.write_exclusive(root / "manifest.json",
                      (json.dumps(manifest, sort_keys=True) + "\n").encode(), 0o400)
    M.write_exclusive(root / "RUN_COMPLETE.txt", M.RUN_TOKEN.encode(), 0o400)


def run() -> dict:
    checks = {}
    checks["source_sha_exact"] = sha256(SOURCE) == EXPECTED["source"]
    checks["test_sha_exact"] = sha256(TEST) == EXPECTED["test"]
    checks["contract_sha_exact"] = sha256(CONTRACT) == EXPECTED["contract"]
    checks["authority_chain"] = bool(M.verify_authorities())

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(load("m1517_author_tests", TEST))
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    checks["author_tests_12_pass"] = result.wasSuccessful() and result.testsRun == 12

    valid = build()
    checks["valid_manifest_120"] = (
        len(valid["records"]) == 120 and
        [row["numeric_encoding"] for row in valid["records"][:4]] ==
        ["bit_times_layer_constant", "bit_times_layer_constant",
         "exact_binary", "exact_binary"])

    audit = synthetic_audit()
    audit["layer_scale_words"][0]["word_uint32"] ^= 1
    checks["d0_scale_attack_rejected"] = rejected(lambda: build(audit))
    audit = synthetic_audit()
    audit["calls"][1]["positive_word_uint32"] ^= 1
    checks["d1_call_scale_attack_rejected"] = rejected(lambda: build(audit))
    audit = synthetic_audit()
    audit["calls"][1]["global_order"] = audit["calls"][0]["global_order"]
    checks["duplicate_global_order_rejected"] = rejected(lambda: build(audit))

    with tempfile.TemporaryDirectory(prefix="m1517_plane_") as directory:
        root = Path(directory)
        source = root / "support.bin"
        source.write_bytes(POSITIVE + NEGATIVE)
        destination = root / "positive.bin"
        support_sha = hashlib.sha256(POSITIVE + NEGATIVE).hexdigest()
        positive_sha = hashlib.sha256(POSITIVE).hexdigest()
        negative_sha = hashlib.sha256(NEGATIVE).hexdigest()
        M.copy_positive_plane_exclusive(
            source, destination, 9, 2, support_sha, positive_sha, negative_sha)
        checks["positive_split_exact"] = destination.read_bytes() == POSITIVE
        checks["copy_collision_rejected"] = rejected(lambda:
            M.copy_positive_plane_exclusive(
                source, destination, 9, 2, support_sha, positive_sha, negative_sha))
        for label, payload in (
                ("extent", POSITIVE + NEGATIVE + b"x"),
                ("positive_tail", bytes((0x05, 0x81)) + NEGATIVE),
                ("negative_nonzero", POSITIVE + bytes((0x01, 0x00)))):
            attacked = root / (label + ".bin")
            attacked.write_bytes(payload)
            checks[label + "_rejected"] = rejected(lambda p=attacked, q=payload:
                M.copy_positive_plane_exclusive(
                    p, root / (p.stem + ".out"), 9, 2,
                    hashlib.sha256(q).hexdigest(),
                    hashlib.sha256(q[:2]).hexdigest(),
                    hashlib.sha256(q[2:4]).hexdigest()))
        checks["traversal_rejected"] = rejected(
            lambda: M.safe_member(root, "../support.bin", "member"))
        link = root / "link.bin"
        link.symlink_to(source)
        checks["symlink_rejected"] = rejected(
            lambda: M.safe_member(root, "link.bin", "member"))

    with tempfile.TemporaryDirectory(prefix="m1517_attempt_") as directory:
        attempt = Path(directory) / "attempt"
        M.consume_attempt(attempt)
        checks["attempt_o_excl_no_retry"] = rejected(lambda: M.consume_attempt(attempt))

    with tempfile.TemporaryDirectory(prefix="m1517_rename_") as directory:
        root = Path(directory); source = root / "source"; destination = root / "dest"
        source.mkdir(); destination.mkdir()
        checks["rename_noreplace_collision"] = rejected(
            lambda: M.rename_noreplace(source, destination))

    # Independent semantic forgery: mutate critical record/top-level fields before
    # sealing.  A sound result verifier must reject this sealed directory.
    forged = build()
    forged["records"][0]["layer_scale_word_uint32"] = 0x3F800000
    forged["records"][0]["layer_scale_word_hex"] = "0x3f800000"
    forged["records"][0]["numeric_encoding"] = "exact_binary"
    forged["records"][0]["weight_folding"] = True
    forged["records"][0]["normalized"] = True
    forged["records"][0]["coerced"] = True
    forged["records"][1]["capture_global_order"] = forged["records"][0][
        "capture_global_order"]
    forged["claim_boundary"]["cycles"] = True
    with tempfile.TemporaryDirectory(prefix="m1517_forge_") as directory:
        stage = Path(directory) / "stage"; stage.mkdir()
        write_stage(stage, forged)
        semantic_forgery_accepted = not rejected(lambda: M.seal_staging(stage))
        checks["semantic_forgery_rejected"] = not semantic_forgery_accepted

    # Independent path forgery: a noncanonical plane filename is accepted if the
    # record and payload population agree, despite the frozen output-name rule.
    path_forged = build()
    path_forged["records"][0]["positive_output"] = "payloads/renamed_attack.bin"
    with tempfile.TemporaryDirectory(prefix="m1517_pathforge_") as directory:
        stage = Path(directory) / "stage"; stage.mkdir()
        write_stage(stage, path_forged)
        path_forgery_accepted = not rejected(lambda: M.seal_staging(stage))
        checks["canonical_output_path_forgery_rejected"] = not path_forgery_accepted

    # Population itself is correctly enforced.
    with tempfile.TemporaryDirectory(prefix="m1517_population_") as directory:
        stage = Path(directory) / "stage"; stage.mkdir()
        manifest = build()
        write_stage(stage, manifest)
        victim = stage.joinpath(*Path(manifest["records"][-1]["positive_output"]).parts)
        victim.unlink()
        checks["seal_121_member_attack_rejected"] = rejected(lambda: M.seal_staging(stage))

    source_text = SOURCE.read_text(encoding="utf-8")
    execute_text = source_text[source_text.index("def execute_once"):]
    checks["cli_production_forbidden"] = (
        "--materialize" not in source_text and "--production" not in source_text and
        rejected(lambda: M.main([])))
    checks["release_gate_calls_m1517_before_materialization"] = (
        execute_text.index("verify_m1517_hammer") <
        execute_text.index("materialize_prepared_once"))
    checks["production_namespace_absent"] = (
        not M.OUTPUT.exists() and not M.ATTEMPT.exists() and
        not any(M.OUTPUT.parent.glob(M.WORK_PREFIX + "*")) and
        not M.FUTURE_RELEASE.exists())

    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "schema": "m1517_independent_hammer_output_r1_v1",
        "target": {"source_sha256": EXPECTED["source"],
                   "test_sha256": EXPECTED["test"],
                   "contract_sha256": EXPECTED["contract"]},
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "author_test_output": stream.getvalue(),
        "verdict": "FAIL_CLOSED__M1518_RELEASE_BLOCKED" if failed else "PASS",
        "execution": {"production": 0, "gpu": 0, "eda": 0, "remote": 0},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
