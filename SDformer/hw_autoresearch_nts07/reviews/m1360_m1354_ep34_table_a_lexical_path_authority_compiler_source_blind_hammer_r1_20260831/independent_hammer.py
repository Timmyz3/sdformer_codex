#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author source-only hammer for M1354; never emits Table-A."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
SOURCE = HW / "system_simulator/scripts/build_m1354_ep34_table_a_lexical_path_authority_compiler.py"
TEST = HW / "system_simulator/tests/test_m1354_ep34_table_a_lexical_path_authority_compiler.py"
CONTRACT = HW / "contracts/m1354_ep34_table_a_lexical_path_authority_compiler_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1354_ep34_table_a_lexical_path_authority_compiler_source_author_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = "/opt/anaconda3/envs/pytorch310/bin/python3.10"
REPLAY = (
    (HW / "system_simulator/tests/test_m1340_ep34_table_a_common_charge_compiler.py", 10),
    (HW / "system_simulator/tests/test_m1342_ep34_table_a_authority_compiler.py", 16),
    (HW / "system_simulator/tests/test_m1351_ep34_table_a_memory_timed_authority_compiler.py", 13),
    (TEST, 6),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load("m1360_target_m1354", SOURCE)
T = load("m1360_fixture_m1351", N.M1351_TEST)
M = N.M1351.M


def run_test(path: Path, expected: int) -> dict:
    run = subprocess.run([PYTHON, "-B", str(path)], cwd=ROOT,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, check=False)
    return {"expected": expected, "returncode": run.returncode,
            "passed": run.returncode == 0 and f"Ran {expected} tests" in run.stdout and
                      "OK" in run.stdout}


def fixture():
    value = T.F.Fixture(); T.add_m1351_authorities(value); return value


def rejects(call: Callable[[], object]) -> bool:
    try:
        call(); return False
    except Exception:
        return True


def rewrite_role_json(fx, role: str, member: str, payload_name: str,
                      mutate: Callable[[dict], None]) -> None:
    path = fx.role_dirs[role] / member
    value = json.loads(path.read_text(encoding="utf-8")); mutate(value)
    T.rewrite_json(path, value)
    fx.role_payloads[role][payload_name]["sha256"] = T.digest(path)
    fx.seal(role)


def build_attack(mutate: Callable[[object], None]) -> bool:
    fx = fixture()
    try:
        mutate(fx)
        return rejects(lambda: N.build(fx.config_path(), fx.root, fx.allowlist))
    finally:
        fx.close()


def extension_attack(mutate: Callable[[dict, Path, dict], None]) -> bool:
    fx = fixture()
    try:
        receipt, trace, timing = T.Tests.extension_inputs(fx)
        mutate(receipt, trace, timing)
        return rejects(lambda: N.M1351.validate_transaction_extensions(
            receipt, trace, T.digest(trace), timing))
    finally:
        fx.close()


def path_attacks() -> dict[str, bool]:
    result = {}
    with tempfile.TemporaryDirectory(prefix="m1360_paths_") as temporary:
        root = Path(temporary)
        genuine = root / "genuine"; genuine.mkdir()
        leaf = genuine / "config.json"; leaf.write_text("{}\n", encoding="utf-8")
        outside_dir = Path(temporary).parent / (root.name + "_outside")
        outside_dir.mkdir(); outside = outside_dir / "outside.json"
        outside.write_text("{}\n", encoding="utf-8")
        try:
            alias_leaf = root / "alias.json"; alias_leaf.symlink_to(leaf)
            result["leaf_symlink_to_genuine"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, alias_leaf))
            alias_ancestor = root / "alias_dir"; alias_ancestor.symlink_to(genuine, target_is_directory=True)
            result["ancestor_symlink_to_genuine"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, alias_ancestor / leaf.name))
            external_leaf = root / "external.json"; external_leaf.symlink_to(outside)
            result["leaf_symlink_escape"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, external_leaf))
            broken = root / "broken.json"; broken.symlink_to("missing.json")
            result["broken_symlink_leaf"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, broken))
            broken_dir = root / "broken_dir"; broken_dir.symlink_to("missing_dir", target_is_directory=True)
            result["broken_symlink_ancestor"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, broken_dir / "x.json"))
            root_alias = Path(temporary).parent / (root.name + "_alias")
            root_alias.symlink_to(root, target_is_directory=True)
            try:
                result["workspace_root_symlink_to_genuine"] = rejects(
                    lambda: N.lexical_lstat_then_resolved_containment(root_alias, root_alias / "genuine/config.json"))
            finally:
                root_alias.unlink()
            result["parent_dotdot"] = rejects(lambda: N.lexical_lstat_then_resolved_containment(
                root, root / "genuine" / ".." / "genuine/config.json"))
            result["absolute_resolved_escape"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(root, outside))
            result["missing_ancestor_for_future_leaf"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(
                    root, root / "missing_parent/output.json", leaf_must_exist=False))
            nondir = root / "not_a_dir"; nondir.write_text("x", encoding="utf-8")
            result["non_directory_ancestor"] = rejects(
                lambda: N.lexical_lstat_then_resolved_containment(
                    root, nondir / "output.json", leaf_must_exist=False))

            race = root / "race.json"; race.write_text("{}\n", encoding="utf-8")
            original_resolve = Path.resolve; swapped = {"done": False}
            def racing_resolve(self, strict=False):
                if self == race and not swapped["done"]:
                    swapped["done"] = True
                    self.unlink(); self.symlink_to(leaf)
                return original_resolve(self, strict=strict)
            Path.resolve = racing_resolve
            try:
                result["toctou_leaf_swapped_to_genuine_symlink_after_lstat"] = rejects(
                    lambda: N.lexical_lstat_then_resolved_containment(root, race))
            finally:
                Path.resolve = original_resolve
        finally:
            for path in sorted(outside_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink(): path.unlink()
            outside_dir.rmdir()
    return result


def main() -> int:
    replay = {path.stem: run_test(path, count) for path, count in REPLAY}
    self_check = subprocess.run(
        [PYTHON, "-B", str(SOURCE), "--source-self-check"], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    attacks = path_attacks()

    attacks["energy_direct_row_smuggle"] = build_attack(lambda fx: rewrite_role_json(
        fx, "energy_producer", "partitioned_energy.json", "partitioned_energy",
        lambda value: value["direct_logic_pj_per_cycle"][M.DIRECT_BRANCHES[0]].update({"Ours": 1e-9})))
    attacks["energy_extra_field"] = build_attack(lambda fx: rewrite_role_json(
        fx, "energy_producer", "partitioned_energy.json", "partitioned_energy",
        lambda value: value.update({"smuggled_rate": 1.0})))
    attacks["energy_missing_branch"] = build_attack(lambda fx: rewrite_role_json(
        fx, "energy_producer", "partitioned_energy.json", "partitioned_energy",
        lambda value: value["direct_logic_pj_per_cycle"].pop(M.DIRECT_BRANCHES[0])))
    attacks["energy_zero_common_rate"] = build_attack(lambda fx: rewrite_role_json(
        fx, "energy_producer", "partitioned_energy.json", "partitioned_energy",
        lambda value: value.update({"common_logic_pj_per_cycle": 0})))

    attacks["trace_receipt_digest_mismatch"] = extension_attack(
        lambda receipt, trace, timing: receipt.update({"address_trace_sha256": "0" * 64}))
    def empty_trace(receipt, trace, timing):
        trace.chmod(0o644); trace.write_bytes(b""); trace.chmod(0o444)
        digest = T.digest(trace); receipt["address_trace_sha256"] = digest
        timing["address_trace_sha256"] = digest
    attacks["trace_empty_even_if_rebound"] = extension_attack(empty_trace)
    attacks["trace_extra_receipt_field"] = build_attack(lambda fx: rewrite_role_json(
        fx, "transaction_receipt", "transaction.json", "transaction_receipt",
        lambda value: value.update({"trace_alias_sha256": "0" * 64})))
    attacks["timing_trace_digest_mismatch"] = extension_attack(
        lambda receipt, trace, timing: timing.update({"address_trace_sha256": "0" * 64}))
    attacks["latency_extra_field"] = extension_attack(
        lambda receipt, trace, timing: timing["latency_model"].update({"queue_cycles": 1}))
    attacks["latency_zero"] = extension_attack(
        lambda receipt, trace, timing: timing["latency_model"].update({"dram_read_cycles": 0}))
    attacks["latency_string"] = extension_attack(
        lambda receipt, trace, timing: timing["latency_model"].update({"dram_read_cycles": "24"}))

    for macro in M.SRAM_MACROS:
        def zero_macro(receipt, trace, timing, victim=macro):
            for row in receipt["rows"].values():
                for charge in row.values():
                    charge["sram_bytes"][victim] = {"read_bytes": 0, "write_bytes": 0}
        attacks["sram_zero_plane_" + macro] = extension_attack(zero_macro)
    def remove_macro(receipt, trace, timing):
        first = next(iter(next(iter(receipt["rows"].values())).values()))
        first["sram_bytes"].pop(M.SRAM_MACROS[0])
    attacks["sram_missing_macro"] = extension_attack(remove_macro)
    def extra_macro(receipt, trace, timing):
        first = next(iter(next(iter(receipt["rows"].values())).values()))
        first["sram_bytes"]["smuggled_17"] = {"read_bytes": 1, "write_bytes": 1}
    attacks["sram_extra_macro"] = extension_attack(extra_macro)

    def zero_dram(receipt, trace, timing):
        for row in receipt["rows"].values():
            for charge in row.values():
                charge["dram_read_bytes"] = charge["dram_write_bytes"] = 0
    attacks["dram_all_zero"] = extension_attack(zero_dram)
    def negative_dram(receipt, trace, timing):
        first_row = next(iter(receipt["rows"].values()))
        next(iter(first_row.values()))["dram_read_bytes"] = -1
    attacks["dram_negative"] = extension_attack(negative_dram)

    def population_missing(receipt, trace, timing):
        row = M.ROWS[0]; timing["rows"][row].pop(next(iter(timing["rows"][row])))
    attacks["population_missing_timing_key"] = extension_attack(population_missing)
    def population_extra(receipt, trace, timing):
        row = M.ROWS[0]; timing["rows"][row]["smuggled:99"] = copy.deepcopy(
            next(iter(timing["rows"][row].values())))
    attacks["population_extra_timing_key"] = extension_attack(population_extra)
    attacks["population_missing_row"] = extension_attack(
        lambda receipt, trace, timing: timing["rows"].pop(M.ROWS[0]))

    def stall_partition(receipt, trace, timing):
        item = next(iter(timing["rows"][M.ROWS[0]].values())); item["dram_stall_cycles"] += 1
    attacks["stall_partition_mismatch"] = extension_attack(stall_partition)
    def stall_exceeds(receipt, trace, timing):
        item = next(iter(timing["rows"][M.ROWS[0]].values()))
        item["memory_stall_cycles"] = item["address_timed_cycles"] + 1
        item["sram_stall_cycles"] = item["memory_stall_cycles"]
        item["dram_stall_cycles"] = 0
    attacks["stall_exceeds_address_cycles"] = extension_attack(stall_exceeds)
    def negative_stall(receipt, trace, timing):
        next(iter(timing["rows"][M.ROWS[0]].values()))["memory_stall_cycles"] = -1
    attacks["stall_negative"] = extension_attack(negative_stall)
    def extra_timing_field(receipt, trace, timing):
        next(iter(timing["rows"][M.ROWS[0]].values()))["smuggled"] = 0
    attacks["timing_item_extra_field"] = extension_attack(extra_timing_field)

    attacks["allowlist_missing_role"] = build_attack(
        lambda fx: fx.allowlist.pop("energy_producer"))
    attacks["allowlist_extra_role"] = build_attack(
        lambda fx: fx.allowlist.update({"smuggled": copy.deepcopy(fx.allowlist["energy_producer"])}))
    attacks["allowlist_extra_field"] = build_attack(
        lambda fx: fx.allowlist["energy_producer"].update({"alias_sha256": "0" * 64}))
    attacks["allowlist_review_sha_drift"] = build_attack(
        lambda fx: fx.allowlist["energy_producer"].update({"review_sha256": "0" * 64}))
    attacks["production_candidate_with_fixture_allowlist"] = build_attack(
        lambda fx: fx.config.update({"status": "PRODUCTION_CANDIDATE"}))

    false_negatives = [name for name, rejected in attacks.items() if not rejected]
    author_ok = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=AUTHOR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False).returncode == 0
    checks = {
        "replay_45": all(row["passed"] for row in replay.values()) and
                     sum(row["expected"] for row in replay.values()) == 45,
        "source_self_check": self_check.returncode == 0 and
            "PASS_M1354_SOURCE_SELF_CHECK__NO_PRODUCTION_NO_TABLE_A_NO_EDA" in self_check.stdout,
        "author_seal": author_ok,
        "docs359": sha(DOCS359) == N.DOCS359_SHA256,
        "production_allowlist_empty": N.M1351.M1342.PRODUCTION_AUTHORITY_ALLOWLIST == {},
    }
    result = {
        "schema": "m1360_m1354_ep34_table_a_lexical_path_blind_hammer_r1_v1",
        "verdict": "PASS_SOURCE_BLIND__NO_PRODUCTION" if
                   all(checks.values()) and not false_negatives else
                   "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED",
        "replay": replay, "checks": checks,
        "fresh_hammer": {"attacks": len(attacks), "rejected": sum(attacks.values()),
                         "false_negative_count": len(false_negatives),
                         "false_negatives": false_negatives, "results": attacks},
        "target": {"source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
                   "contract_sha256": sha(CONTRACT),
                   "author_review_sha256": sha(AUTHOR / "review.json"),
                   "author_manifest_sha256": sha(AUTHOR / "SHA256SUMS"),
                   "author_outer_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256")},
        "execution": {"production": False, "table_a_rows": 0, "gpu": 0,
                      "vcs": 0, "eda": 0},
        "docs359_sha256": sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
