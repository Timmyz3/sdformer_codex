#!/opt/anaconda3/bin/python3
"""Independent no-EDA hammer for M2176 reset/clear semantic repair."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
PARSER = HW / "system_simulator/scripts/parse_m2176_m2018_ordinary_native_saif_reset_semantics_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2176_m2018_ordinary_native_saif_reset_semantics_preflight_one_shot.py"
TEST = HW / "tests/test_m2176_ordinary_native_saif_reset_semantics_preflight.py"
BASE_PARSER = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
BASE_RUNNER = HW / "dc_handoff/scripts/run_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_one_shot.py"
BASE_TEST = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
CONTRACT = HW / "contracts/m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_contract_r1_20260904.json"
M2173 = HW / "reviews/m2173_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_source_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2176_m2173_ordinary_native_saif_reset_semantics_preflight_source_author_receipt_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED = {
    PARSER: "2dadf88ccfb4f4e43281203c67317b9f0bf91ed1fa3874eadb6015db9244438d",
    RUNNER: "c1dcc5736e093ac5f1424667ac460056fdd561005639fa17d6a23708951867b9",
    TEST: "8ae73003fe81a025abefc63175d28278a94d83b301c5200d4fe2d7a77c360cda",
    BASE_PARSER: "42fd87d6991c46366e80db1d08c20ec5e0d463f3bca8c6050673093d04f3bfe2",
    BASE_RUNNER: "828c743093afe0c1e506bd820d7cd2fcad0169ae0ea9e9ad8308ec1e3c9c27eb",
    CONTRACT: "173444343a4856010728bcd0b81be55c8b639eb9b3ed5d8a1544622c308942a0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_exhaustive_seal(directory: Path) -> dict[str, object]:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert manifest.is_file() and not manifest.is_symlink()
    assert outer.is_file() and not outer.is_symlink()
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        assert name not in entries
        path = directory / name
        assert path.is_file() and not path.is_symlink()
        assert sha256(path) == digest
        entries[name] = digest
    actual = {
        path.name for path in directory.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    assert set(entries) == actual
    outer_digest, outer_name = outer.read_text().split()
    assert outer_name.lstrip("*") == "SHA256SUMS"
    assert outer_digest == sha256(manifest)
    return {
        "directory": str(directory.relative_to(REPO)),
        "member_count": len(entries),
        "manifest_sha256": sha256(manifest),
        "outer_sha256": sha256(outer),
        "exhaustive": True,
    }


def run_source(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX="/tmp/m2177_source_hammer_pycache")
    return subprocess.run(command, cwd=REPO, env=env, check=True,
                          capture_output=True, text=True, timeout=180)


def expect_failure(action, label: str) -> str:
    try:
        action()
    except Exception as exc:  # exact module-local Failure types are intentionally inherited
        return f"{label}: {type(exc).__name__}: {exc}"
    raise AssertionError(f"expected fail-closed rejection escaped: {label}")


def main() -> None:
    assert sha256(DOC359) == DOC359_SHA256
    for path, expected in EXPECTED.items():
        assert path.is_file() and not path.is_symlink()
        assert sha256(path) == expected, path

    contract_sidecar = Path(str(CONTRACT) + ".sha256")
    contract_outer = Path(str(contract_sidecar) + ".seal.sha256")
    assert contract_sidecar.read_text().split() == [sha256(CONTRACT), CONTRACT.name]
    assert contract_outer.read_text().split() == [sha256(contract_sidecar), contract_sidecar.name]
    m2173_seal = verify_exhaustive_seal(M2173)
    author_seal = verify_exhaustive_seal(AUTHOR)
    rejection = json.loads((M2173 / "review.json").read_text())
    assert rejection["status"] == (
        "FAIL_M2173_M2172_SOURCE_HAMMER__RESET_SYNONYM_GATE_INCOMPLETE__M2174_NOT_AUTHORIZED")
    assert rejection["severity_counts"] == {"p0": 1, "p1": 0, "p2": 0}
    assert rejection["authorization"]["m2174_authorized"] is False

    test_run = run_source([sys.executable, "-B", str(TEST)])
    assert "PASS_M2176_SOURCE_TESTS inherited_m2172=42 minimal_failure_mutations=14 accepted_controls=2 eda_runs=0" in test_run.stdout
    parser_static = run_source([sys.executable, "-B", str(PARSER), "static"])
    runner_static = run_source([sys.executable, "-B", str(RUNNER), "--static"])
    assert json.loads(parser_static.stdout)["status"] == "PASS_M2176_STATIC_PARSER"
    assert json.loads(runner_static.stdout)["status"] == "PASS_M2176_STATIC_RUNNER"

    module = load("m2176_parser_independent_m2177", PARSER)
    fixture = load("m2172_fixture_independent_m2177", BASE_TEST)
    runner = load("m2176_runner_independent_m2177", RUNNER)
    assert module.parse_saif is module.BASE.parse_saif
    assert module.parse_balanced_saif is module.BASE.parse_balanced_saif
    assert sha256(module.BASE_PATH) == EXPECTED[BASE_PARSER]

    failures = [
        "Warning: reset ignored.", "Warning: reset rejected.",
        "Error: reset denied.", "Warning: reset unsupported.",
        "Warning: reset failed.", "Error: reset cannot complete.",
        "Warning: reset unable to complete.", "Error: reset remained uncleared.",
        "Warning: reset retained old counters.", "Error: reset remained active.",
        "Warning: reset not cleared.", "Error: reset not reset.",
        "Warning: clear failed.", "Error: clear request denied.",
    ]
    successes = [
        "Info: power reset request accepted and switching counters cleared.",
        "Info: reset completed successfully.",
    ]
    variants = [
        "WARNING -- RESET FAILED!",
        "Error: resetting unsupported.",
        "WARNING: clearing cannot complete.",
        "Error: clear was not cleared.",
        "Warning: reset failure.",
    ]
    assert all(module.reset_failure_lines(line) == [line] for line in failures)
    assert all(module.reset_failure_lines(line) == [] for line in successes)
    assert all(module.reset_failure_lines(line) == [line] for line in variants)

    runtime_rejections: list[str] = []
    with tempfile.TemporaryDirectory(prefix="m2177_runtime_") as raw:
        runtime = Path(raw) / "rtl_sim.log"
        runtime.write_text(fixture.runtime_text())
        assert module.parse_runtime(runtime)["completion_ledger"]["products"] == 29472
        for line in failures:
            runtime.write_text(fixture.runtime_text() + line + "\n")
            runtime_rejections.append(expect_failure(lambda p=runtime: module.parse_runtime(p), line))
        runtime.write_text(fixture.runtime_text() + successes[0] + "\n")
        assert module.parse_runtime(runtime)["completion_ledger"]["products"] == 29472

    original_sha256 = runner.sha256
    runner.sha256 = lambda path: ("0" * 64 if Path(path).resolve() == PARSER.resolve()
                                  else original_sha256(path))
    inventory_failure = expect_failure(lambda: runner.source_validation(require_review=False),
                                       "source inventory drift")
    runner.sha256 = original_sha256
    with tempfile.TemporaryDirectory(prefix="m2177_missing_review_") as raw:
        original_review = runner.REVIEW
        runner.REVIEW = Path(raw) / "absent_review"
        missing_review_failure = expect_failure(
            lambda: runner.source_validation(require_review=True), "missing independent review")
        runner.REVIEW = original_review

    results_dir = HW / "results"
    m2174 = sorted(path.name for path in results_dir.iterdir() if "m2174" in path.name.lower())
    m2178 = sorted(path.name for path in results_dir.iterdir() if "m2178" in path.name.lower())
    assert m2174 == []
    assert m2178 == []

    output = {
        "status": "PASS_M2177_MECHANICAL_SOURCE_HAMMER",
        "execution_invoked": {
            "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
            "raw_saif_files_written": 0, "dc_runs": 0, "ptpx_runs": 0,
            "icc2_runs": 0, "gpu_runs": 0,
        },
        "identity": {
            "docs359_sha256": sha256(DOC359),
            "parser_sha256": sha256(PARSER),
            "runner_sha256": sha256(RUNNER),
            "test_sha256": sha256(TEST),
            "contract_sha256": sha256(CONTRACT),
            "frozen_m2172_parser_sha256": sha256(BASE_PARSER),
            "frozen_m2172_runner_sha256": sha256(BASE_RUNNER),
        },
        "seals": {"m2173_rejection": m2173_seal, "m2176_author_receipt": author_seal},
        "semantic_hammer": {
            "minimal_failure_sentences_rejected": len(runtime_rejections),
            "accepted_controls": len(successes),
            "additional_normalization_variants_rejected": len(variants),
            "inherited_m2172_tests": 42,
            "balanced_parse_saif_exact_function_reuse": True,
            "balanced_parse_balanced_saif_exact_function_reuse": True,
        },
        "fail_closed": {
            "source_inventory_drift_rejected": inventory_failure,
            "missing_review_rejected": missing_review_failure,
        },
        "census": {"m2174_artifacts": m2174, "m2178_artifacts": m2178},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
