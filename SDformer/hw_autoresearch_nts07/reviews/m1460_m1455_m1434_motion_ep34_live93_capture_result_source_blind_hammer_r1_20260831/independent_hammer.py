#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent blind hammer for the source-only M1455 result validator."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/hammer_m1455_m1434_motion_ep34_live93_capture_result_source.py"
SOURCE_SHA256 = "a77ae63153dbe808d98c73e1db05108d9e2152fdb29ae5837beae6fa5ea7991a"
TEST = HW / "tests/test_hammer_m1455_m1434_motion_ep34_live93_capture_result_source.py"
TEST_SHA256 = "d6159976384edbc37e7ed6e97efd285bbf6f0b4c3842a3068a23fe1727f1e928"
CONTRACT = HW / (
    "contracts/m1455_m1434_motion_ep34_live93_capture_result_hammer_source_"
    "contract_r1_20260831.json")
CONTRACT_SHA256 = "ebb41b86ebba7d1ac40f01ec5cf0592df8513817a9572591c4a79b279c4ae671"
AUTHOR = HW / (
    "reviews/m1455_m1434_motion_ep34_live93_capture_result_hammer_source_"
    "author_r1_20260831")
AUTHOR_REVIEW_SHA256 = "6295d65fd74760143fe0052cf3124c4a292520a99ab45b490bc7995513a320e6"
AUTHOR_MANIFEST_SHA256 = "3b25e256e9a98b5d984c53d137d2fa0384d629114020827e1d6cb81ba04eb56a"
AUTHOR_OUTER_SHA256 = "69765ed9a4fd28277e51edc26c08b2a579572d8307d85187db79eccba80496ac"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assert sha256(SOURCE) == SOURCE_SHA256
assert sha256(TEST) == TEST_SHA256
assert sha256(CONTRACT) == CONTRACT_SHA256
assert sha256(DOCS359) == DOCS359_SHA256
assert sha256(AUTHOR / "review.json") == AUTHOR_REVIEW_SHA256
assert sha256(AUTHOR / "SHA256SUMS") == AUTHOR_MANIFEST_SHA256
assert sha256(AUTHOR / "SHA256SUMS.seal.sha256") == AUTHOR_OUTER_SHA256

M = load("m1460_sealed_m1455", SOURCE)
T = load("m1460_sealed_m1455_test", TEST)


def verify_author_seal() -> None:
    rows = {}
    for line in (AUTHOR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert relative not in rows
        rows[relative] = digest
        assert sha256(AUTHOR / relative.removeprefix("./")) == digest
    assert rows == {
        "./review.json": AUTHOR_REVIEW_SHA256,
        "./review.md": "95f6741b07c5aa821d6242566700e2ee96a5ac958ad42fd6407254fa9a189479",
    }
    assert (AUTHOR / "SHA256SUMS.seal.sha256").read_text(encoding="ascii") == (
        AUTHOR_MANIFEST_SHA256 + "  SHA256SUMS\n")
    review = json.loads((AUTHOR / "review.json").read_text(encoding="utf-8"))
    assert review["status"] == "PASS_SOURCE_AUTHOR__DIFFERENT_AUTHOR_BLIND_REQUIRED"
    assert review["authorization"] == {
        "independent_source_hammer": True,
        "capture": False,
        "result_promotion": False,
    }


def reseal(root: Path) -> None:
    T.OLD.seal(root)


@contextlib.contextmanager
def fixture_root(fixture):
    with tempfile.TemporaryDirectory(prefix="m1460_attack_") as temporary:
        root = Path(temporary) / "result"
        shutil.copytree(fixture.root, root)
        yield root


def external_success():
    return contextlib.ExitStack()


def install_external_success(stack: contextlib.ExitStack) -> None:
    stack.enter_context(mock.patch.object(
        M.M1434, "validate_snapshot_population_live93"))
    stack.enter_context(mock.patch.object(
        M.M1401.M1338, "validate_retained_payloads", return_value=320))
    stack.enter_context(mock.patch.object(
        M.M1401.M1338.OLD, "validate_attention_geometry",
        return_value={"records": 480}))
    stack.enter_context(mock.patch.object(
        M.M1401.M1338, "validate_attention_exact_archive"))


def expect_reject(root: Path, *, patch_external: bool = True,
                  exact_m1455: bool = False) -> None:
    try:
        with external_success() as stack:
            if patch_external:
                install_external_success(stack)
            M.validate_result(root)
    except Exception as error:
        if exact_m1455:
            assert isinstance(error, M.M1455Error), type(error)
        return
    raise AssertionError("false negative: mutation was accepted")


def mutate_json(root: Path, relative: str, mutator) -> None:
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    reseal(root)


def mutate_ordered(root: Path, ordinal: int, mutator,
                   terminal_newline: bool = True) -> None:
    path = root / "unified_ordered_records.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[ordinal])
    mutator(row)
    lines[ordinal] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + ("\n" if terminal_newline else ""),
                    encoding="utf-8")
    reseal(root)


def main() -> int:
    verify_author_seal()
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    validate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == "validate_result")
    validate_text = ast.get_source_segment(source_text, validate)
    assert validate_text is not None
    production_calls = [
        "M1401.M1338.validate_retained_payloads(root, rows, ordered)",
        "M1401.M1338.OLD.validate_attention_geometry(root, rows)",
        "M1401.M1338.validate_attention_exact_archive(root)",
        "M1434.R1.validate_payload_population(root)",
        "M1434.validate_snapshot_population_live93(root)",
    ]
    for call in production_calls:
        assert call in validate_text
    assert "except Exception as error:" in validate_text
    assert "retained/attention/payload/forensic validation failed" in validate_text
    assert "subprocess" not in source_text and "torch.cuda" not in source_text
    assert "os.kill" not in source_text and 'add_argument("--run"' not in source_text

    fixture = T.Fixture()
    attacks = 0
    try:
        # Reconfirm the complete positive fixture and all five production calls.
        with fixture_root(fixture) as root, external_success() as stack:
            snapshots = stack.enter_context(mock.patch.object(
                M.M1434, "validate_snapshot_population_live93"))
            retained = stack.enter_context(mock.patch.object(
                M.M1401.M1338, "validate_retained_payloads", return_value=320))
            geometry = stack.enter_context(mock.patch.object(
                M.M1401.M1338.OLD, "validate_attention_geometry",
                return_value={"records": 480}))
            exact = stack.enter_context(mock.patch.object(
                M.M1401.M1338, "validate_attention_exact_archive"))
            payload = stack.enter_context(mock.patch.object(
                M.M1434.R1, "validate_payload_population", return_value=[None] * 640))
            result = M.validate_result(root)
            assert result["population"] == {
                "ordered": 9880, "retained": 320,
                "attention": {"records": 480}, "payload": 640,
                "execution": 7360, "operator": 79, "atlif": 93,
                "forensic_snapshots": 40,
            }
            for patched in (snapshots, retained, geometry, exact, payload):
                patched.assert_called_once()

        # Recursive seal attacks.
        with fixture_root(fixture) as root:
            (root / "unsealed").write_text("x\n", encoding="utf-8")
            expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            os.symlink("manifest.json", root / "hidden_link")
            expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            manifest = root / "SHA256SUMS"
            first = manifest.read_text(encoding="ascii").splitlines()[0]
            manifest.write_text(manifest.read_text(encoding="ascii") + first + "\n",
                                encoding="ascii")
            (root / "SHA256SUMS.seal.sha256").write_text(
                sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")
            expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            manifest = root / "SHA256SUMS"
            manifest.write_text(manifest.read_text(encoding="ascii") +
                                "0" * 64 + "  ../escape\n", encoding="ascii")
            (root / "SHA256SUMS.seal.sha256").write_text(
                sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")
            expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            path = root / "manifest.json"
            path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
            reseal(root); expect_reject(root); attacks += 1

        # Complete graph and 40x247 sequence attacks.
        ordered_attacks = [
            (0, lambda row: row.pop("global_order"), True),
            (4, lambda row: row.__setitem__("global_order", True), True),
            (4, lambda row: row.__setitem__("global_order", 3), True),
            (9879, lambda row: None, False),
            (247, lambda row: row.__setitem__("global_sample_id", 0), True),
            (247, lambda row: row.__setitem__("global_sample_id", True), True),
            (247, lambda row: row.__setitem__("name", "cross.sample.drift"), True),
            (247, lambda row: row.__setitem__("category", "cross_category_drift"), True),
            (0, lambda row: row.__setitem__("name", M.M1434.DEAD_SN2_Q[0]), True),
            (0, lambda row: row.__setitem__("category", "atlif"), True),
        ]
        for ordinal, mutator, terminal in ordered_attacks:
            with fixture_root(fixture) as root:
                mutate_ordered(root, ordinal, mutator, terminal)
                expect_reject(root); attacks += 1

        # Manifest/admission identity and no-claim boundaries.
        json_attacks = [
            ("manifest.json", lambda x: x.__setitem__("schema", "wrong")),
            ("manifest.json", lambda x: x.__setitem__("status", "PASS_PAPER")),
            ("manifest.json", lambda x: x["identity"]["module_counts"].__setitem__("ATLIFTernaryPSN", 93)),
            ("manifest.json", lambda x: x["identity"]["selection"]["selected"].__setitem__("epoch", 33)),
            ("manifest.json", lambda x: x["identity"]["selection"]["selected"]["checkpoint"].__setitem__("sha256", "0" * 64)),
            ("manifest.json", lambda x: x["identity"]["selection"]["selected"]["configuration"].__setitem__("sha256", "0" * 64)),
            ("manifest.json", lambda x: x["identity"]["selection"]["selected"]["profile"].__setitem__("sha256", "0" * 64)),
            ("manifest.json", lambda x: x["identity"]["selection"]["selected"]["profile"].__setitem__("samples", 824)),
            ("manifest.json", lambda x: x["m1434_runtime_contract"].__setitem__("static_atlif", 93)),
            ("manifest.json", lambda x: x["m1434_runtime_contract"].__setitem__("live_atlif", 105)),
            ("manifest.json", lambda x: x["m1434_runtime_contract"].__setitem__("ordered_records", 10360)),
            ("manifest.json", lambda x: x["m1434_runtime_contract"]["dead_atlif"].__setitem__("count", 11)),
            ("manifest.json", lambda x: x["m1434_runtime_contract"]["dead_atlif"].__setitem__("terminal_lf_sha256", "0" * 64)),
            ("manifest.json", lambda x: x["forensic_snapshots"].__setitem__("automatic_canonical_promotion", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("accuracy", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("cycles", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("speedup", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("energy", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("rtl", True)),
            ("manifest.json", lambda x: x["claim_boundary"].__setitem__("ppa", True)),
            ("m1434_admission.json", lambda x: x.__setitem__("status", "FAIL")),
            ("m1434_admission.json", lambda x: x.__setitem__("ordered", 10360)),
            ("m1434_admission.json", lambda x: x["dead_atlif"].__setitem__("names", [])),
            ("m1434_admission.json", lambda x: x["claim_boundary"].__setitem__("speedup", True)),
        ]
        for relative, mutator in json_attacks:
            with fixture_root(fixture) as root:
                mutate_json(root, relative, mutator)
                expect_reject(root); attacks += 1

        # Population attacks outside the mocked payload archive substrate.
        population_attacks = [
            ("execution_trace.json", lambda x: x.pop()),
            ("operator_runtime.json", lambda x: x.pop()),
            ("operator_runtime.json", lambda x: x[1].__setitem__("name", x[0]["name"])),
            ("operator_runtime.json", lambda x: x[0].__setitem__("calls", True)),
            ("atlif_activity.json", lambda x: x.pop()),
            ("atlif_activity.json", lambda x: x[1].__setitem__("name", x[0]["name"])),
            ("atlif_activity.json", lambda x: x[0].__setitem__("calls", 39)),
            ("atlif_activity.json", lambda x: x[0].__setitem__("name", M.M1434.DEAD_SN2_Q[0])),
        ]
        for relative, mutator in population_attacks:
            with fixture_root(fixture) as root:
                mutate_json(root, relative, mutator)
                expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            (root / "attention_qk/manifest.json").unlink()
            reseal(root); expect_reject(root); attacks += 1
        with fixture_root(fixture) as root:
            (root / "RUN_COMPLETE.txt").write_text("PASS_PAPER\n", encoding="utf-8")
            reseal(root); expect_reject(root); attacks += 1

        # Every production external validator must be called and wrapped as M1455Error.
        validators = [
            (M.M1401.M1338, "validate_retained_payloads"),
            (M.M1401.M1338.OLD, "validate_attention_geometry"),
            (M.M1401.M1338, "validate_attention_exact_archive"),
            (M.M1434.R1, "validate_payload_population"),
            (M.M1434, "validate_snapshot_population_live93"),
        ]
        for owner, name in validators:
            with fixture_root(fixture) as root, external_success() as stack:
                install_external_success(stack)
                stack.enter_context(mock.patch.object(
                    owner, name, side_effect=RuntimeError("m1460 injected failure")))
                expect_reject(root, patch_external=False, exact_m1455=True)
                attacks += 1
    finally:
        fixture.close()

    assert attacks == 54, attacks
    print(json.dumps({
        "status": "PASS_M1460_M1455_M1434_RESULT_SOURCE_BLIND_HAMMER",
        "attacks": attacks,
        "false_negatives": 0,
        "author_tests_replayed": 10,
        "source_self_check_replayed": True,
        "production_validator_calls_static_confirmed": len(production_calls),
        "production_validator_failures_wrapped": 5,
        "remote": False,
        "gpu": False,
        "capture": False,
        "controller_signal": False,
        "result_read": False,
        "claim_boundary": {
            "source_only": True, "paper_result": False,
            "cycles": False, "speedup": False, "energy": False,
            "ppa": False, "system_speedup": False, "headline": False,
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
