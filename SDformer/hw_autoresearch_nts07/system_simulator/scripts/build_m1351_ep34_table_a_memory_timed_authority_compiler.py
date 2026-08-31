#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Memory-timed, authority-bound successor to fail-closed M1342.

This source-only compiler retains M1342's empty production allowlist.  It adds
resolved containment, a sealed address-trace payload, row-invariant direct
logic rates, independent nonzero DRAM/SRAM gates, and an exact per-population
memory-latency/stall receipt.  No production Table-A row is authorized here.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
HW = REPO / "hw_autoresearch_nts07"
M1342_PATH = HW / "system_simulator/scripts/build_m1342_ep34_table_a_authority_compiler.py"
M1342_SHA256 = "ce0089a4af043e5ad74b6a38c142bdd74ad91c36527ffc4a3894e8e905954569"
M1342_TEST = HW / "system_simulator/tests/test_m1342_ep34_table_a_authority_compiler.py"
M1342_TEST_SHA256 = "0b3c9269fd3d07510de944e5300b0cca1cd6454855a80a0ed40f17d59949987f"
M1346_FAIL = HW / "reviews/m1346_m1342_ep34_table_a_authority_compiler_source_blind_hammer_r1_20260831"
M1346_REVIEW_SHA256 = "4d80b9dc292128f11d18b7baa94aebb040cc4d2314d1c6f2c9609e7cc87a99fd"
M1346_MANIFEST_SHA256 = "a63663ee9a06d2b77c9e958dd80f8daed7692506a6dd30f0d9f3e642ebe0e6d7"
M1346_OUTER_SHA256 = "4708175084f0e630d3994d6dcb71c6cedffe74e5ae45885f7c59ae83c903d576"
CONTRACT = HW / "contracts/m1351_ep34_table_a_memory_timed_authority_compiler_source_contract_r1_20260831.json"
TEST = HW / "system_simulator/tests/test_m1351_ep34_table_a_memory_timed_authority_compiler.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
MEMORY_TIMING_SCHEMA = "m1351.table_a.memory_timing_receipt.r1"
SOURCE_SCHEMA = "m1351.ep34.table_a.memory_timed.authority.compiler.source.r1"
PRODUCTION_ALLOWLIST_STATE = "SOURCE_ONLY_UNPOPULATED__ADDITIVE_RELEASE_REQUIRED"


class CompileError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CompileError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat()
    except FileNotFoundError as exc:
        raise CompileError("missing " + label) from exc
    require(stat.S_ISREG(mode.st_mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


regular_exact(M1342_PATH, M1342_SHA256, "M1342 source")
SPEC = importlib.util.spec_from_file_location("m1351_frozen_m1342", M1342_PATH)
M1342 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M1342)
M = M1342.M


def strict_json(path: Path) -> Any:
    return M1342.strict_json(path)


def secure_no_symlink_ancestry(root: Path, path: Path,
                               leaf_must_exist: bool = True) -> Path:
    root_abs = Path(root).absolute()
    try:
        root_resolved = root_abs.resolve(strict=True)
    except OSError as exc:
        raise CompileError("workspace root missing") from exc
    require(root_resolved == root_abs, "workspace root must not traverse symlinks")
    raw = Path(path)
    require(".." not in raw.parts, "parent traversal forbidden: %s" % path)
    candidate = raw if raw.is_absolute() else root_resolved / raw
    try:
        resolved = candidate.resolve(strict=leaf_must_exist)
    except OSError as exc:
        raise CompileError("path resolution failed: %s" % path) from exc
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise CompileError("resolved path escapes workspace: %s" % path) from exc
    current = root_resolved
    for part in relative.parts:
        current = current / part
        if not current.exists() and current == resolved and not leaf_must_exist:
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise CompileError("path ancestry missing: %s" % current) from exc
        require(not stat.S_ISLNK(mode), "symlink path component forbidden: %s" % current)
    return resolved


@contextlib.contextmanager
def secure_m1342_paths() -> Iterator[None]:
    original = M1342.no_symlink_ancestry
    M1342.no_symlink_ancestry = secure_no_symlink_ancestry
    try:
        yield
    finally:
        M1342.no_symlink_ancestry = original


def verify_m1346_failure() -> None:
    regular_exact(M1342_TEST, M1342_TEST_SHA256, "M1342 test")
    regular_exact(M1346_FAIL / "review.json", M1346_REVIEW_SHA256, "M1346 review")
    regular_exact(M1346_FAIL / "SHA256SUMS", M1346_MANIFEST_SHA256,
                  "M1346 manifest")
    regular_exact(M1346_FAIL / "SHA256SUMS.seal.sha256", M1346_OUTER_SHA256,
                  "M1346 outer seal")
    outer = (M1346_FAIL / "SHA256SUMS.seal.sha256").read_text().split()
    require(outer == [M1346_MANIFEST_SHA256, "SHA256SUMS"],
            "M1346 outer semantics drift")
    review = strict_json(M1346_FAIL / "review.json")
    require(review.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and review.get("false_negative_count") == 6
            and review.get("authorization", {}).get("populate_production_allowlist") is False
            and review.get("authorization", {}).get("additive_source_successor") is True,
            "M1346 failure semantics drift")


def load_authorities(config_path: Path, workspace_root: Path,
                     fixture_allowlist: dict[str, dict[str, str]] | None) -> tuple[dict, dict]:
    root = workspace_root.absolute()
    config_path = M1342.regular_readonly_single(root, config_path, "M1351 config")
    config = strict_json(config_path)
    require(config.get("schema") == M1342.SCHEMA and
            config.get("status") in ("SOURCE_FIXTURE", "PRODUCTION_CANDIDATE"),
            "M1351 requires an exact M1342 config")
    production = config["status"] == "PRODUCTION_CANDIDATE"
    if production:
        require(set(M1342.PRODUCTION_AUTHORITY_ALLOWLIST) == set(M1342.ROLES),
                "production authority allowlist is not populated")
        allowlist = M1342.PRODUCTION_AUTHORITY_ALLOWLIST
    else:
        require(fixture_allowlist is not None and
                set(fixture_allowlist) == set(M1342.ROLES),
                "source fixture requires explicit non-production allowlist")
        allowlist = fixture_allowlist
    roots = config.get("authority_roots")
    require(type(roots) is dict and set(roots) == set(M1342.ROLES),
            "authority role set differs")
    authorities = {role: M1342.verify_authority(root, role, roots[role], allowlist,
                                                production)
                   for role in M1342.ROLES}
    return config, authorities


def positive_int(value: Any, label: str) -> int:
    require(type(value) is int and value >= 1, label + " must be positive integer")
    return value


def nonnegative_int(value: Any, label: str) -> int:
    require(type(value) is int and value >= 0, label + " must be nonnegative integer")
    return value


def validate_direct_rate_fairness(path: Path) -> dict[str, Any]:
    payload = strict_json(path)
    direct = payload.get("direct_logic_pj_per_cycle")
    require(type(direct) is dict and set(direct) == set(M.DIRECT_BRANCHES),
            "direct logic branch set differs")
    normalized = {}
    for branch in M.DIRECT_BRANCHES:
        rates = direct[branch]
        require(type(rates) is dict and set(rates) == set(M.ROWS),
                "direct logic row set differs")
        values = [M.finite_nonnegative_number(rates[row], "direct logic rate",
                                               positive=True) for row in M.ROWS]
        require(all(math.isclose(value, values[0], rel_tol=0.0, abs_tol=0.0)
                    for value in values),
                "direct logic rate is not row invariant: " + branch)
        normalized[branch] = values[0]
    return normalized


def validate_transaction_extensions(receipt: dict[str, Any], trace_path: Path,
                                    trace_sha: str, timing: dict[str, Any]) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{64}", trace_sha or "") is not None,
            "address trace SHA grammar invalid")
    require(trace_path.stat().st_size > 0 and sha256(trace_path) == trace_sha,
            "sealed address trace payload SHA/extent mismatch")
    require(receipt.get("address_trace_sha256") == trace_sha,
            "transaction receipt is not bound to sealed address trace")
    rows = receipt.get("rows")
    require(type(rows) is dict and set(rows) == set(M.ROWS),
            "transaction row set differs")
    dram_total = 0
    sram_total = {macro: 0 for macro in M.SRAM_MACROS}
    for row in M.ROWS:
        require(type(rows[row]) is dict and rows[row], "transaction population missing")
        for charge in rows[row].values():
            charge = M.validate_charge(charge, "M1351 transaction")
            dram_total += charge["dram_read_bytes"] + charge["dram_write_bytes"]
            for macro in M.SRAM_MACROS:
                sram_total[macro] += (charge["sram_bytes"][macro]["read_bytes"] +
                                      charge["sram_bytes"][macro]["write_bytes"])
    require(dram_total > 0, "all-zero DRAM plane forbidden")
    require(all(value > 0 for value in sram_total.values()),
            "one or more all-zero SRAM macro planes forbidden")

    require(type(timing) is dict and set(timing) ==
            {"schema", "identity", "address_trace_sha256", "latency_model", "rows"},
            "memory timing receipt fields drift")
    require(timing["schema"] == MEMORY_TIMING_SCHEMA and
            timing["identity"] == "Motion-C12-ep34-final" and
            timing["address_trace_sha256"] == trace_sha,
            "memory timing identity/trace binding drift")
    latency = timing["latency_model"]
    require(type(latency) is dict and set(latency) ==
            {"sram_read_cycles", "sram_write_cycles",
             "dram_read_cycles", "dram_write_cycles"},
            "memory latency model fields drift")
    latency = {key: positive_int(value, key) for key, value in latency.items()}
    timing_rows = timing["rows"]
    require(type(timing_rows) is dict and set(timing_rows) == set(M.ROWS),
            "memory timing row set differs")
    stall_total = 0
    for row in M.ROWS:
        require(type(timing_rows[row]) is dict and
                set(timing_rows[row]) == set(rows[row]),
                "memory timing population coverage differs: " + row)
        for key, charge in rows[row].items():
            item = timing_rows[row][key]
            require(type(item) is dict and set(item) ==
                    {"address_timed_cycles", "memory_stall_cycles",
                     "sram_stall_cycles", "dram_stall_cycles"},
                    "memory timing item fields drift")
            address_cycles = positive_int(item["address_timed_cycles"],
                                          "address-timed cycles")
            memory_stall = nonnegative_int(item["memory_stall_cycles"],
                                           "memory stall cycles")
            sram_stall = nonnegative_int(item["sram_stall_cycles"],
                                         "SRAM stall cycles")
            dram_stall = nonnegative_int(item["dram_stall_cycles"],
                                         "DRAM stall cycles")
            require(memory_stall == sram_stall + dram_stall,
                    "memory stall partition mismatch")
            require(address_cycles == charge["cycles"] and memory_stall <= address_cycles,
                    "memory timing cycles differ from conserved transaction charge")
            stall_total += memory_stall
    return {"dram_bytes": dram_total, "sram_bytes_by_macro": sram_total,
            "latency_model": latency, "memory_stall_cycles": stall_total}


def preflight(config_path: Path, workspace_root: Path,
              fixture_allowlist: dict[str, dict[str, str]] | None) -> dict[str, Any]:
    with secure_m1342_paths():
        _config, authorities = load_authorities(config_path, workspace_root,
                                                fixture_allowlist)
        energy_payloads = authorities["energy_producer"]["payloads"]
        energy_spec = energy_payloads.get("partitioned_energy")
        require(energy_spec is not None, "partitioned energy payload missing")
        energy_path = M1342.regular_readonly_single(workspace_root,
                                                    Path(energy_spec["path"]),
                                                    "partitioned energy")
        direct_rates = validate_direct_rate_fairness(energy_path)

        transaction_payloads = authorities["transaction_receipt"]["payloads"]
        require(set(transaction_payloads) >=
                {"transaction_receipt", "address_trace", "memory_timing"},
                "transaction authority lacks trace/timing payloads")
        receipt_spec = transaction_payloads["transaction_receipt"]
        trace_spec = transaction_payloads["address_trace"]
        timing_spec = transaction_payloads["memory_timing"]
        receipt_path = M1342.regular_readonly_single(
            workspace_root, Path(receipt_spec["path"]), "transaction receipt")
        trace_path = M1342.regular_readonly_single(
            workspace_root, Path(trace_spec["path"]), "address trace")
        timing_path = M1342.regular_readonly_single(
            workspace_root, Path(timing_spec["path"]), "memory timing receipt")
        require(sha256(trace_path) == trace_spec["sha256"] and
                sha256(timing_path) == timing_spec["sha256"],
                "transaction authority payload SHA drift")
        receipt = strict_json(receipt_path)
        timing = strict_json(timing_path)
        memory = validate_transaction_extensions(receipt, trace_path,
                                                 trace_spec["sha256"], timing)
    return {"direct_logic_pj_per_cycle": direct_rates,
            "address_trace_sha256": trace_spec["sha256"],
            "memory_timing_sha256": timing_spec["sha256"], **memory}


def build(config_path: Path, workspace_root: Path,
          fixture_allowlist: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    verify_m1346_failure()
    preflight_result = preflight(config_path, workspace_root, fixture_allowlist)
    with secure_m1342_paths():
        result = M1342.build(config_path, workspace_root, fixture_allowlist)
    require(result["status"] == "PASS_SOURCE_FIXTURE_NOT_PRODUCTION",
            "M1351 source cannot admit production")
    result = dict(result)
    result.update({"schema": "m1351.ep34.table_a.memory_timed.authority.output.r1",
                   "status": "PASS_SOURCE_FIXTURE_MEMORY_TIMED_NOT_PRODUCTION",
                   "m1351_source_sha256": sha256(SCRIPT),
                   "memory_timing": preflight_result})
    result["claim_boundary"] = {**result["claim_boundary"],
        "resolved_containment": True, "direct_logic_rate_row_invariant": True,
        "sealed_address_trace": True, "dram_and_each_sram_nonzero": True,
        "memory_latency_and_stalls_bound": True, "paper_headline_admitted": False}
    return result


def validate_source_policy() -> dict[str, Any]:
    verify_m1346_failure()
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") == SOURCE_SCHEMA and
            contract.get("status") == PRODUCTION_ALLOWLIST_STATE and
            contract.get("production_authorized") is False,
            "M1351 source policy drift")
    require(contract.get("source") == {"path": str(SCRIPT.relative_to(REPO)),
                                       "sha256": sha256(SCRIPT)} and
            contract.get("test") == {"path": str(TEST.relative_to(REPO)),
                                     "sha256": sha256(TEST)},
            "M1351 source/test identity drift")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.source_self_check:
            require(args.config is None and args.workspace_root is None and args.output is None,
                    "source self-check cannot accept production paths")
            validate_source_policy()
            print("PASS_M1351_SOURCE_SELF_CHECK__NO_TABLE_A_NO_EDA")
            return 0
        require(args.config is not None and args.workspace_root is not None and
                args.output is not None, "build requires config/workspace/output")
        result = build(args.config, args.workspace_root)
        output = secure_no_symlink_ancestry(args.workspace_root, args.output,
                                            leaf_must_exist=False)
        require(output.parent.is_dir(), "output parent missing")
        descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
        return 0
    except (CompileError, M1342.CompileError, M.CompileError,
            OSError, ValueError) as exc:
        print("M1351_FAIL_CLOSED: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
