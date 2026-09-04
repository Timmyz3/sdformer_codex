#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Receipt-blind M1024 hammer of the M1023 decoder-window r2 repair.

Synthetic/source-only.  It never opens a real decoder payload and never runs
EDA, GPU, remote jobs, or a real measurement window.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "system_simulator/scripts/analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
CHECKER = HW / "system_simulator/scripts/check_m1023_decoder_stratified_block_reset_windows_source_r2.py"
TESTS = HW / "system_simulator/tests/test_m1023_decoder_stratified_block_reset_windows_source_r2.py"
CONTRACT = HW / "contracts/m1023_decoder_stratified_block_reset_windows_source_r2_contract_r1_20260829.json"
M1017 = HW / "reviews/m1017_m1014_decoder_stratified_block_reset_windows_source_hammer_r1_20260829"
RECEIPT = HW / "reviews/m1023_decoder_stratified_block_reset_windows_source_r2_receipt_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "source": "8e9ce843499cbcfdfe1856e5f829218e0329cd299ce25d1ba93e3b45cd74d2b2",
    "checker": "e69c6dd74fc86c7f5d68dc724054ec1b6b0122a3dc45ae9e800bef2d2b6d3536",
    "tests": "e4d92230b62a5241c9d56b72fa09e57075c0114d321831e5c471d6ac18bf65cf",
    "contract": "6f6ce243162bb8e05f4a13701eabf8c56f0a9b251e2769df73930bdefd35229a",
    "m1017_review": "ec916d15481d7e0428e3cfb19912ee325bb5322edb29222c991d73ae1e9cb941",
    "m1017_manifest": "17334fe7d0adfb7027d51db1cf541cae6e806fcdc00af07ff861a324a8c33700",
    "m1017_outer": "44b2bc65932436e2c86402746118fcd0eb4900c6fc608cf007d02fb72a184e9a",
    "receipt_review": "7088ca2eb57d26408593040ed425bd9fe9d05dd256260d195ae939992c56b3ea",
    "receipt_manifest": "f5b18f37294bd0d6b09a2262facde01fbd3b09507cead57c809751d61a90da1c",
    "receipt_outer": "01835ab3c56a26e8593ed365df69f1313a62f29a3b9975fe1fe461c6f2149f99",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def strict_json(path: Path) -> dict:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def verify_flat(directory: Path, expected: tuple[str, str, str]) -> dict:
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) == expected,
            "sealed identity drift: " + directory.name)
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        member = directory / rel
        require(rel not in listed and member.is_file() and not member.is_symlink() and
                sha(member) == digest, "sealed member drift: " + str(member))
        listed.add(rel)
    require(outer.read_text().split() == [expected[1], "SHA256SUMS"],
            "outer seal content drift: " + directory.name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256") and
              "__pycache__" not in path.parts}
    require(listed == actual, "sealed exact-set drift: " + directory.name)
    return strict_json(review)


def load_source():
    spec = importlib.util.spec_from_file_location("m1024_independent_m1023", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expect_rejected(call, fragment: str) -> None:
    try:
        call()
    except RuntimeError as error:
        require(fragment in str(error), "wrong rejection: " + str(error))
        return
    raise RuntimeError("attack accepted; expected " + fragment)


def high_variance(module):
    return module.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])


def main() -> dict:
    for path, key in ((SOURCE, "source"), (CHECKER, "checker"), (TESTS, "tests"),
                      (CONTRACT, "contract"), (DOC359, "docs359")):
        require(sha(path) == EXPECTED[key], key + " identity drift")
    m1017 = verify_flat(M1017, (EXPECTED["m1017_review"], EXPECTED["m1017_manifest"],
                               EXPECTED["m1017_outer"]))
    receipt = verify_flat(RECEIPT, (EXPECTED["receipt_review"], EXPECTED["receipt_manifest"],
                                   EXPECTED["receipt_outer"]))
    require(m1017["status"] == "FAIL_M1017_M1014_SOURCE_HAMMER__BLOCK_EXECUTION_RELEASE" and
            m1017["p0_count"] == 3 and
            m1017["authorization"]["execution_release_authorized"] is False,
            "M1017 negative authority drift")
    require(receipt["status"] == "PASS_M1023_R2_SOURCE_ONLY__M1024_INDEPENDENT_HAMMER_REQUIRED" and
            receipt["launch_now"] is False, "M1023 receipt drift")
    contract = strict_json(CONTRACT)
    require(contract["negative_authority"]["m1017_review_sha256"] == EXPECTED["m1017_review"] and
            contract["negative_authority"]["m1017_outer_seal_file_sha256"] == EXPECTED["m1017_outer"],
            "contract does not bind M1017")
    require(contract["source_identity"]["source"]["sha256"] == EXPECTED["source"] and
            contract["source_identity"]["checker"]["sha256"] == EXPECTED["checker"] and
            contract["source_identity"]["test"]["sha256"] == EXPECTED["tests"],
            "source identities drift")

    python = "/opt/anaconda3/envs/pytorch310/bin/python3.10"
    tests = subprocess.run([python, "-m", "unittest", "-v", str(TESTS)],
                           text=True, capture_output=True, check=True, timeout=60)
    require("Ran 11 tests" in tests.stdout + tests.stderr and
            "OK" in tests.stdout + tests.stderr, "author 11/11 failed")
    selftest = subprocess.run([python, str(SOURCE), "--self-test"],
                              text=True, capture_output=True, check=True, timeout=60)
    require("PASS_M1023_R2_SMALL_SYNTHETIC_P0_REPAIR_SELFTEST" in selftest.stdout,
            "source self-test failed")
    source_validate = subprocess.run([python, str(SOURCE), "--validate-source"],
                                     text=True, capture_output=True, check=True, timeout=60)
    require("PASS_M1023_R2_SOURCE_VALIDATION__NO_REAL_EXECUTION" in source_validate.stdout,
            "source validation failed")
    checker = subprocess.run([python, str(CHECKER), "--check"],
                             text=True, capture_output=True, check=True, timeout=60)
    require("PASS_M1023_R2_SOURCE_STATIC_CHECK__NO_REAL_EXECUTION" in checker.stdout,
            "source checker failed")
    with tempfile.TemporaryDirectory(prefix="m1024_pycache_") as pycache:
        env = os.environ.copy()
        env["PYTHONPYCACHEPREFIX"] = pycache
        subprocess.run([python, "-m", "py_compile", str(SOURCE), str(CHECKER), str(TESTS)],
                       env=env, check=True, timeout=60)

    module = load_source()
    # P0-1: broad case/punctuation/nesting/timing aliases all fail before selection.
    aliases = [
        {"total_cycles": 1}, {"TOTAL_CyClEs": 1}, {"Total-Cycles": 1},
        {"latency": 1}, {"Latency_NS": 1}, {"execution_time": 1},
        {"ElapsedTime": 1}, {"wall_clock": 1}, {"runtime_ms": 1},
        {"SpEeDuP": 1.2}, {"through_put": 1.2}, {"FPS": 1.2},
        {"nested": {"diagnostic": {"totalLatency": 1}}},
        {"nested": [{"time-ms": 1}, {"POINT_SPEEDUP": 2}]},
    ]
    for ordinal, attack in enumerate(aliases):
        row = {"block_id": "attack-{}".format(ordinal), "compute_count": 1}
        row.update(attack)
        expect_rejected(lambda row=row: module.deterministic_select(
            [row], "COMPUTE_REGULAR", 1), "")
    expect_rejected(lambda: module.deterministic_select(
        [{"block_id": "x", "compute_count": 1, "harmless": 1}],
        "COMPUTE_REGULAR", 1), "unknown pre-cycle")
    expect_rejected(lambda: module.deterministic_select(
        [{"block_id": "x", "compute_count": 1, "destination": {"x": 1}}],
        "COMPUTE_REGULAR", 1), "nested metadata")

    # P0-2: compare every canonical boundary/fill/drain field independently.
    body = module.M890.synthetic_transactions(448)
    spec = module.WindowSpec("m1024-reset", "D0", "COMMIT_TAIL", 1)
    normal = module.paired_replay(body, body, spec)
    require(normal["paired_reset_exact_equal"] and
            [row["role"] for row in normal["paired_reset_service_cycle_sequence"]] ==
            ["boundary", "fill", "drain"], "normal reset sequence drift")
    fields = tuple(normal["paired_reset_service_cycle_sequence"][0].keys())
    reset_faults = 0
    for role_index in range(3):
        for field in fields:
            original = module._reset_service_semantics
            calls = [0]

            def attacked(transactions, body_count, role_index=role_index, field=field):
                value = original(transactions, body_count)
                calls[0] += 1
                if calls[0] == 2:
                    old = value[role_index][field]
                    if isinstance(old, bool):
                        new = not old
                    elif isinstance(old, int):
                        new = old + 1
                    elif isinstance(old, str):
                        new = old + "_ASYM"
                    elif isinstance(old, list):
                        new = list(old) + ["ASYM"]
                    else:
                        raise RuntimeError("unsupported reset field type")
                    value[role_index][field] = new
                return value

            module._reset_service_semantics = attacked
            try:
                expect_rejected(lambda: module.paired_replay(body, body, spec),
                                "reset semantic or cycle charge asymmetry")
                reset_faults += 1
            finally:
                module._reset_service_semantics = original

    # P0-3: state thresholds and recursive disclosure audit.
    hard = high_variance(module)
    require(hard["ci95_relative_halfwidth_max"] > 0.10 and
            hard["candidate_total_cycles_estimate"] is None and
            hard["baseline_total_cycles_estimate"] is None and
            hard["paired_speedup_estimate"] is None and
            hard["point_estimate_admitted"] is False,
            "top-level >10% hard stop failed")
    leaked_points = []
    for index, row in enumerate(hard.get("strata", [])):
        for key, value in row.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if (("meancycle" in normalized or normalized.endswith("cyclesestimate") or
                 normalized.endswith("speedupestimate")) and value is not None):
                leaked_points.append("strata[{}].{}={}".format(index, key, value))
    diagnostic = module.apply_ci_publication_gate({
        "candidate_total_cycles_estimate": 100.0,
        "candidate_ci95": [94.0, 106.0],
        "baseline_total_cycles_estimate": 120.0,
        "baseline_ci95": [114.0, 126.0],
        "paired_speedup_estimate": 1.2,
        "paired_speedup_ci95": [1.128, 1.272],
    })
    require(0.05 < diagnostic["ci95_relative_halfwidth_max"] <= 0.10 and
            diagnostic["status"].startswith("DIAGNOSTIC_POINT_ONLY") and
            diagnostic["point_estimate_admitted"] is False and
            diagnostic["paired_speedup_estimate"] == 1.2,
            "5-10% diagnostic state failed")
    precise = module.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
        "candidate_cycles": [10] * 8, "baseline_cycles": [20] * 8,
    }])
    require(precise["ci95_relative_halfwidth_max"] <= 0.05 and
            precise["status"] == "PRECISE_POINT_ELIGIBLE_FOR_LATER_RELEASE" and
            precise["point_estimate_admitted"] is True and
            precise["paper_citable"] is False, "<=5% candidate state failed")

    # Frozen routing and source-only protocol boundaries.
    for layer, index in (("D0", 0), ("D2", 2), ("D3", 3)):
        route = module.frozen_route(layer)
        require(route["module_index"] == index and route["sample_id"] == 0 and
                route["timestep"] == 0 and route["real_payload_opened"] is False,
                "frozen route drift: " + layer)
    expect_rejected(lambda: module.frozen_route("D1"),
                    "D1_STRICT_COMMON_CHARGE_NO_GENERATOR_OR_SCHEDULER_CALL")
    no_commit = [tx for tx in body if tx.kind != "commit"]
    expect_rejected(lambda: module.paired_replay(no_commit, no_commit, spec),
                    "commit stratum has zero commit requests")
    too_large = [replace(body[0], count=module.WINDOW_EXPANDED_REQUEST_CAP - 2)]
    expect_rejected(lambda: module.block_reset_transactions(
        too_large, module.WindowSpec("large", "D0", "COMPUTE_REGULAR", 1),
        "candidate"), "window exceeds 10K")
    expect_rejected(lambda: module.deterministic_select(
        [{"block_id": "x", "compute_count": 1}], "COMPUTE_REGULAR", 33),
        "selection count exceeds frozen bound")
    require(module.PILOT_PER_STRATUM == 8 and module.MAX_PER_STRATUM == 32 and
            module.WINDOW_EXPANDED_REQUEST_CAP == 10000 and
            module.SELECTION_SEED == "M1009_STRATIFIED_WINDOW_R1_20260829",
            "selector/window freeze drift")

    p0 = []
    if leaked_points:
        p0.append({
            "id": "P0_CI_HARD_STOP_RETAINS_STRATUM_CYCLE_POINTS",
            "finding": "The >10% state nulls only three top-level estimates but retains per-stratum candidate_mean_cycles and baseline_mean_cycles, contradicting HARD_STOP_REPORT_BOUNDS_AND_COVERAGE_ONLY and the requirement that all cycle/speedup point fields be null.",
            "observed": leaked_points,
            "required_repair": "Recursively redact every point-cycle and point-speedup field in the >10% publication object while retaining only CI bounds and coverage/sample-count fields; add nested-output tests.",
        })

    return {
        "schema": "m1024_m1023_decoder_stratified_block_reset_windows_source_r2_hammer_r1_v1",
        "date": "2026-08-29",
        "milestone": "M1024",
        "status": ("FAIL_M1024_M1023_R2_SOURCE_HAMMER__BLOCK_EXECUTION_RELEASE"
                   if p0 else "PASS_M1024_M1023_R2_SOURCE_HAMMER"),
        "verdict": ("NO_GO_EXECUTION_RELEASE__AUTHOR_R3_CI_REDACTION_REPAIR"
                    if p0 else "GO_AUTHOR_EXECUTION_RELEASE_AND_RUNNER_ONLY"),
        "score_out_of_100": 86 if p0 else 100,
        "p0_count": len(p0), "p1_count": 0, "p2_count": 0,
        "identity": {
            "source_sha256": sha(SOURCE), "checker_sha256": sha(CHECKER),
            "tests_sha256": sha(TESTS), "contract_sha256": sha(CONTRACT),
            "m1017_review_sha256": sha(M1017 / "review.json"),
            "m1017_manifest_sha256": sha(M1017 / "SHA256SUMS"),
            "m1017_outer_seal_file_sha256": sha(M1017 / "SHA256SUMS.seal.sha256"),
            "m1023_receipt_review_sha256": sha(RECEIPT / "review.json"),
            "m1023_receipt_manifest_sha256": sha(RECEIPT / "SHA256SUMS"),
            "m1023_receipt_outer_seal_file_sha256": sha(RECEIPT / "SHA256SUMS.seal.sha256"),
            "docs359_sha256": sha(DOC359),
        },
        "mechanical": {
            "author_unittest": "PASS_11_OF_11", "source_selftest": "PASS",
            "source_validation": "PASS", "static_checker": "PASS",
            "python_compile": "PASS", "metadata_alias_attacks_rejected": len(aliases),
            "reset_fields": list(fields), "reset_roles": ["boundary", "fill", "drain"],
            "reset_field_role_asymmetries_rejected": reset_faults,
            "ci_above_10_top_level_null": True,
            "ci_above_10_nested_point_leaks": leaked_points,
            "ci_5_to_10_diagnostic_only": True,
            "ci_at_most_5_candidate_only": True,
            "d1_rejected": True, "commit_zero_rejected": True,
            "window_above_10000_rejected": True, "selector_above_32_rejected": True,
            "pilot_exactly_8": True, "frozen_routes": ["D0", "D2", "D3"],
        },
        "p0": p0,
        "authorization": {
            "execution_release_authorized": not p0,
            "execution_runner_authorized": not p0,
            "real_window_execution": False,
            "eda": False, "gpu": False, "remote": False,
            "r3_ci_redaction_repair": bool(p0),
        },
        "scope": {
            "receipt_blind_hammer": True, "synthetic_only": True,
            "real_payload_opened": False, "real_window_execution": False,
            "eda_gpu_remote_used": False, "docs359_modified": False,
        },
        "claim_boundary": {
            "paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_speedup": False,
            "transaction_ratio_is_speedup": False,
        },
    }


if __name__ == "__main__":
    result = main()
    (HERE / "review.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
