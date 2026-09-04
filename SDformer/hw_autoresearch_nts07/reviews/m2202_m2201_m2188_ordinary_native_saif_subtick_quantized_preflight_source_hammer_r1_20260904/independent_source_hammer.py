#!/opt/anaconda3/bin/python3
"""Independent no-EDA M2202 hammer for the M2201 SAIF parser repair."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
PARSER = HW / "system_simulator/scripts/parse_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight.py"
RUNNER = HW / "dc_handoff/scripts/run_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight_one_shot.py"
TEST = HW / "tests/test_m2201_ordinary_native_saif_subtick_quantized_preflight.py"
FIXTURE_SOURCE = HW / "tests/test_m2172_ordinary_native_saif_balanced_scope_preflight.py"
CONTRACT = HW / "contracts/m2201_m2188_ordinary_native_saif_subtick_quantized_preflight_source_contract_r1_20260904.json"
M2188 = HW / "reviews/m2188_m2187_m2185_ordinary_native_saif_gate_level_preflight_failure_result_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2201_m2188_ordinary_native_saif_subtick_quantized_preflight_source_author_receipt_r1_20260904"
OLD_Q = HW / "results/m2187_m2185_m2018_ordinary_native_saif_gate_level_preflight_r1_20260904.failed.3245526.quarantine"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = HW / "results/m2203_m2201_m2018_ordinary_native_saif_subtick_quantized_preflight_r1_20260904"
ATTEMPT = HW / "results/.m2203_m2201_ordinary_native_saif_subtick_quantized_preflight_attempt_consumed"
LOCK = HW / "results/.m2203_m2201_ordinary_native_saif_subtick_quantized_preflight_launch_lock"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_dir(path: Path) -> tuple[int, str, str]:
    assert path.is_dir() and not path.is_symlink()
    assert not list(path.rglob("*")) or not any(item.is_symlink() for item in path.rglob("*"))
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        assert name not in listed
        listed[name] = digest
    actual = sorted(str(item.relative_to(path)) for item in path.rglob("*")
                    if item.is_file() and item.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    assert actual == sorted(listed), (actual, sorted(listed))
    for name, digest in listed.items():
        assert sha(path / name) == digest, name
    assert outer.read_text().split() == [sha(manifest), manifest.name]
    return len(listed), sha(manifest), sha(outer)


def must_fail(callable_) -> None:
    try:
        callable_()
    except P.Failure:
        return
    raise AssertionError("negative control passed")


P = load("m2201_parser_for_independent_m2202", PARSER)
F = load("m2172_fixture_for_independent_m2202", FIXTURE_SOURCE)


def main() -> int:
    m2188_count, m2188_manifest, m2188_outer = verify_dir(M2188)
    author_count, author_manifest, author_outer = verify_dir(AUTHOR)
    assert m2188_count == 6 and author_count == 5
    m2188 = strict_json(M2188 / "review.json")
    assert m2188["status"] == (
        "FAIL_M2188_M2187_RESULT_HAMMER__DIAGNOSTIC_SUBTICK_QUANTIZATION_"
        "PARSER_GAP__M2187_NO_RETRY__M2193_SOURCE_ONLY")
    assert m2188["authorization"]["m2187_retry_authorized"] is False
    assert m2188["authorization"]["reuse_m2187_raw_files"] is False
    assert m2188["root_cause_assessment"]["full_cycle_tolerance_forbidden"] is True

    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    assert sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name]
    assert outer.read_text().split() == [sha(sidecar), sidecar.name]
    contract = strict_json(CONTRACT)
    lineage = contract["m2188_failure_lineage"]
    assert lineage == {
        "review_sha256": sha(M2188 / "review.json"),
        "manifest_sha256": m2188_manifest,
        "outer_sha256": m2188_outer,
        "member_count": 6,
        "m2187_retry_authorized": False,
        "reuse_m2187_raw_files": False,
        "administrative_identity_remap": "M2193-M2196_TO_M2201-M2204",
    }
    inventory = contract["source_inventory"]
    assert len(inventory) == 22
    for relative, expected in inventory.items():
        path = REPO / relative
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == expected, relative
    author_json = strict_json(AUTHOR / "author_receipt.json")
    assert author_json["status"] == "PASS_M2201_SOURCE_AUTHOR_SELFTEST__M2202_REQUIRED__NO_EDA"
    assert author_json["selftests"]["source_inventory_exact"] == "22/22"
    assert author_json["authorization"]["direct_production_authorized"] is False

    # Measurement is not reimplemented or wrapped with tolerance: it is the
    # exact M2176 function, which is itself the frozen M2172 per-record parser.
    source = PARSER.read_text()
    assert 'if role == "measurement":' in source
    assert "return BASE.parse_saif(path, role=role)" in source
    assert P.BASE.parse_saif is P.BASE.BASE.parse_saif

    before = {name: sha(OLD_Q / name) for name in
              ("rtl_prehistory.saif", "rtl_measurement.saif")}
    real_diagnostic = P.parse_saif(OLD_Q / "rtl_prehistory.saif",
                                   role="diagnostic_prehistory")
    real_measurement = P.parse_saif(OLD_Q / "rtl_measurement.saif",
                                    role="measurement")
    after = {name: sha(OLD_Q / name) for name in before}
    assert before == after
    assert real_diagnostic["conservation_mode"] == "uniform_floor_subtick"
    assert abs(real_diagnostic["subtick_residual_raw"] - 0.01) <= 1e-6
    assert real_diagnostic["record_count"] == 93971
    assert real_measurement["record_count"] == 93971
    assert real_measurement["conservation_failures"] == 0
    assert real_measurement["tx_nonzero_record_count"] == 0

    A = P.audit_conservation_fields
    exact = A([(7.0, 3.0, 0.0, 2.0)], duration_raw=10.0,
              role="diagnostic_prehistory")
    subtick = A([(7.0, 3.0, 0.0, 2.0), (6.0, 4.0, 0.0, 4.0)],
                duration_raw=10.01, role="diagnostic_prehistory")
    assert exact["mode"] == "exact"
    assert subtick["mode"] == "uniform_floor_subtick"
    assert math.isclose(subtick["residual_raw"], 0.01,
                        rel_tol=0.0, abs_tol=1e-12)
    assert subtick["full_tick_error_accepted"] is False
    conservation_rejections = [
        ([(6.0, 3.0, 0.0, 2.0)], 10.01, "diagnostic_prehistory"),
        ([(8.0, 3.0, 0.0, 2.0)], 10.01, "diagnostic_prehistory"),
        ([(7.0, 3.0, 0.0, 2.0), (6.0, 3.0, 0.0, 2.0)], 10.01,
         "diagnostic_prehistory"),
        ([(7.5, 2.5, 0.0, 2.0)], 10.01, "diagnostic_prehistory"),
        ([(7.0, 2.5, 0.5, 2.0)], 10.01, "diagnostic_prehistory"),
        ([(7.0, 3.0, 0.0, 2.5)], 10.01, "diagnostic_prehistory"),
        ([(-1.0, 11.0, 0.0, 2.0)], 10.01, "diagnostic_prehistory"),
        ([(7.0, 2.99, 0.0, 2.0)], 10.0, "measurement"),
        ([(6.0, 3.0, 0.0, 2.0)], 10.0, "measurement"),
    ]
    for samples, duration, role in conservation_rejections:
        must_fail(lambda samples=samples, duration=duration, role=role:
                  A(samples, duration_raw=duration, role=role))

    with tempfile.TemporaryDirectory(prefix="m2202_full_saif_") as raw:
        root = Path(raw)
        path = root / "measurement.saif"
        old_records = P.EXPECTED["records"]
        P.EXPECTED["records"] = 32
        try:
            good = F.saif_text(role="measurement", count=32)
            path.write_text(good)
            F.seal_file(path)
            assert P.parse_saif(path, role="measurement")["record_count"] == 32
            full_mutations = {
                "measurement_residual_0p01": good.replace(
                    "(T0 60875.00)", "(T0 60874.99)", 1),
                "measurement_tx": F.saif_text(role="measurement", count=32,
                                                tx_first=1),
                "wrong_hierarchy": F.saif_text(role="measurement", count=32,
                                                 target="impostor"),
                "missing_record": F.saif_text(role="measurement", count=31),
                "missing_critical": F.saif_text(role="measurement", count=32,
                                                  mute_first_critical=True),
            }
            for value in full_mutations.values():
                path.write_text(value)
                F.seal_file(path)
                must_fail(lambda: P.parse_saif(path, role="measurement"))
        finally:
            P.EXPECTED["records"] = old_records

    result_count = sum(path.exists() for path in (RESULT, ATTEMPT, LOCK))
    assert result_count == 0
    assert sha(DOC359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    runner = RUNNER.read_text()
    assert "M2203_ATTEMPT_CONSUMED" in runner
    assert "reuse_old_artifacts\": False" in runner
    assert contract["execution_budget"]["automatic_retry"] is False
    assert contract["execution_authority"]["direct_execution_authorized_now"] is False
    output = {
        "status": "PASS_M2202_INDEPENDENT_MECHANICAL_CHECKS",
        "seals": {
            "m2188_members": m2188_count,
            "m2188_manifest_sha256": m2188_manifest,
            "m2188_outer_sha256": m2188_outer,
            "m2201_author_members": author_count,
            "m2201_author_manifest_sha256": author_manifest,
            "m2201_author_outer_sha256": author_outer,
            "contract_sha256": sha(CONTRACT),
            "contract_sidecar_sha256": sha(sidecar),
            "contract_outer_sha256": sha(outer),
        },
        "identity": {"source_inventory": "22/22", "docs359": sha(DOC359)},
        "lineage": lineage,
        "controls": {
            "real_m2187_read_only": 2,
            "diagnostic_exact_control": 1,
            "diagnostic_uniform_subtick_control": 1,
            "conservation_rejections": 9,
            "full_saif_control": 1,
            "full_saif_rejections": 5,
            "measurement_exact_frozen_m2176_function": True,
            "measurement_0p01_rejected": True,
            "diagnostic_integer_fields_required": True,
            "diagnostic_each_sum_floor_duration": True,
            "diagnostic_uniform_residual": True,
            "diagnostic_strict_residual_interval": True,
        },
        "execution": {"m2203_census": result_count, "vcs_runs": 0,
                      "license_queries": 0, "eda_runs": 0, "gpu_runs": 0,
                      "git_mutations": 0, "source_modifications": 0},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
