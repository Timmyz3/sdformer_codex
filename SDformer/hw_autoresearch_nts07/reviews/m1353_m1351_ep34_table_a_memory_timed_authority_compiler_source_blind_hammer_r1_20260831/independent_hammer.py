#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author fail-closed hammer for the M1351 source compiler."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1351_ep34_table_a_memory_timed_authority_compiler.py"
M1351_TEST = HW / "system_simulator/tests/test_m1351_ep34_table_a_memory_timed_authority_compiler.py"
M1342_TEST = HW / "system_simulator/tests/test_m1342_ep34_table_a_authority_compiler.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load("m1353_bound_m1351", SOURCE)
T = load("m1353_bound_m1351_test", M1351_TEST)
F = load("m1353_bound_m1342_fixture", M1342_TEST)
M = N.M


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_json(path: Path, payload: object) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o444)


def fixture():
    value = F.Fixture()
    T.add_m1351_authorities(value)
    return value


def extension_inputs(value):
    root = value.role_dirs["transaction_receipt"]
    return (copy.deepcopy(json.loads((root / "transaction.json").read_text())),
            root / "address_trace.csv",
            copy.deepcopy(json.loads((root / "memory_timing.json").read_text())))


def main() -> int:
    attacks = []

    def record(name: str, thunk) -> None:
        rejected = False
        error_type = ""
        message = ""
        try:
            thunk()
        except Exception as exc:  # Any exception is fail-closed; record exact type.
            rejected = True
            error_type = type(exc).__name__
            message = str(exc)
        attacks.append({"attack": name, "rejected": rejected,
                        "error_type": error_type, "message": message})

    # Path containment: absolute, relative/dotdot, and symlink escape attempts.
    with tempfile.TemporaryDirectory(prefix="m1353_outside_") as outside:
        outside_file = Path(outside) / "outside.json"
        outside_file.write_text("{}\n"); outside_file.chmod(0o444)
        value = fixture()
        try:
            record("absolute_config_escape", lambda: N.build(
                outside_file, value.root, value.allowlist))
            record("relative_dotdot_config_escape", lambda: N.build(
                Path("../") / outside_file.name, value.root, value.allowlist))
            record("absolute_authority_root_escape", lambda: (
                value.config["authority_roots"].__setitem__(
                    "final_identity", str(Path(outside).resolve())),
                value.allowlist["final_identity"].__setitem__(
                    "root", str(Path(outside).resolve())),
                N.build(value.config_path(), value.root, value.allowlist))[-1])
        finally:
            value.close()

    value = fixture()
    try:
        real_config = value.config_path()
        config_alias = value.root / "config_alias.json"
        config_alias.symlink_to(real_config.name)
        record("symlink_config_escape", lambda: N.build(
            config_alias, value.root, value.allowlist))

        alias = value.root / "authority_alias"
        alias.symlink_to("authorities", target_is_directory=True)
        value.config["authority_roots"]["final_identity"] = "authority_alias/final_identity"
        value.allowlist["final_identity"]["root"] = "authority_alias/final_identity"
        record("symlink_authority_escape", lambda: N.build(
            value.config_path(), value.root, value.allowlist))
    finally:
        value.close()

    # Energy fairness: the same logical work cannot buy B0 and Ours arbitrary rates.
    value = fixture()
    try:
        path = value.role_dirs["energy_producer"] / "partitioned_energy.json"
        payload = json.loads(path.read_text())
        branch = M.DIRECT_BRANCHES[0]
        payload["direct_logic_pj_per_cycle"][branch]["B0"] = 1000.0
        payload["direct_logic_pj_per_cycle"][branch]["Ours"] = 0.001
        rewrite_json(path, payload)
        value.role_payloads["energy_producer"]["partitioned_energy"]["sha256"] = digest(path)
        value.seal("energy_producer")
        record("arbitrary_b0_ours_direct_logic_rate", lambda: N.build(
            value.config_path(), value.root, value.allowlist))
    finally:
        value.close()

    # Trace grammar, extent and triple binding.
    value = fixture()
    try:
        receipt, trace, timing = extension_inputs(value)
        record("nonhex_trace_sha", lambda: N.validate_transaction_extensions(
            receipt, trace, "z" * 64, timing))

        original = trace.read_bytes()
        trace.chmod(0o644); trace.write_bytes(b""); trace.chmod(0o444)
        record("empty_trace_payload", lambda: N.validate_transaction_extensions(
            receipt, trace, hashlib.sha256(b"").hexdigest(), timing))
        trace.chmod(0o644); trace.write_bytes(b"replaced-address-trace\n"); trace.chmod(0o444)
        record("replaced_trace_without_reseal", lambda: N.validate_transaction_extensions(
            receipt, trace, hashlib.sha256(original).hexdigest(), timing))
        trace.chmod(0o644); trace.write_bytes(original); trace.chmod(0o444)

        receipt_bad = copy.deepcopy(receipt)
        receipt_bad["address_trace_sha256"] = "0" * 64
        record("receipt_trace_sha_mismatch", lambda: N.validate_transaction_extensions(
            receipt_bad, trace, digest(trace), timing))
        timing_bad = copy.deepcopy(timing)
        timing_bad["address_trace_sha256"] = "1" * 64
        record("timing_trace_sha_mismatch", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), timing_bad))
    finally:
        value.close()

    # Transaction plane completeness.
    value = fixture()
    try:
        receipt, trace, timing = extension_inputs(value)
        dram_zero = copy.deepcopy(receipt)
        for row in dram_zero["rows"].values():
            for charge in row.values():
                charge["dram_read_bytes"] = charge["dram_write_bytes"] = 0
        record("all_zero_dram_plane", lambda: N.validate_transaction_extensions(
            dram_zero, trace, digest(trace), timing))

        sram_zero = copy.deepcopy(receipt)
        for row in sram_zero["rows"].values():
            for charge in row.values():
                for access in charge["sram_bytes"].values():
                    access["read_bytes"] = access["write_bytes"] = 0
        record("all_zero_sram_plane", lambda: N.validate_transaction_extensions(
            sram_zero, trace, digest(trace), timing))

        macro_zero = copy.deepcopy(receipt)
        victim = M.SRAM_MACROS[-1]
        for row in macro_zero["rows"].values():
            for charge in row.values():
                charge["sram_bytes"][victim] = {"read_bytes": 0, "write_bytes": 0}
        record("one_sram_macro_all_zero", lambda: N.validate_transaction_extensions(
            macro_zero, trace, digest(trace), timing))
    finally:
        value.close()

    # Latency, population, conserved cycles and stall constraints.
    value = fixture()
    try:
        receipt, trace, timing = extension_inputs(value)
        missing_latency = copy.deepcopy(timing)
        del missing_latency["latency_model"]["dram_write_cycles"]
        record("missing_latency_member", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), missing_latency))
        zero_latency = copy.deepcopy(timing)
        zero_latency["latency_model"]["sram_read_cycles"] = 0
        record("zero_latency", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), zero_latency))

        row = M.ROWS[0]
        key = next(iter(timing["rows"][row]))
        population_missing = copy.deepcopy(timing)
        del population_missing["rows"][row][key]
        record("timing_population_missing", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), population_missing))

        cycle_mismatch = copy.deepcopy(timing)
        cycle_mismatch["rows"][row][key]["address_timed_cycles"] += 1
        record("address_cycles_not_equal_charge", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), cycle_mismatch))

        partition_bad = copy.deepcopy(timing)
        partition_bad["rows"][row][key]["dram_stall_cycles"] += 1
        record("stall_partition_mismatch", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), partition_bad))

        bound_bad = copy.deepcopy(timing)
        cycles = bound_bad["rows"][row][key]["address_timed_cycles"]
        bound_bad["rows"][row][key]["memory_stall_cycles"] = cycles + 1
        bound_bad["rows"][row][key]["sram_stall_cycles"] = cycles + 1
        bound_bad["rows"][row][key]["dram_stall_cycles"] = 0
        record("stall_exceeds_address_cycles", lambda: N.validate_transaction_extensions(
            receipt, trace, digest(trace), bound_bad))
    finally:
        value.close()

    # Caller-supplied and even monkey-patched fixture allowlists cannot self-admit production.
    value = fixture()
    original_allowlist = N.M1342.PRODUCTION_AUTHORITY_ALLOWLIST
    try:
        value.config["status"] = "PRODUCTION_CANDIDATE"
        record("caller_fixture_allowlist_invented_for_production", lambda: N.build(
            value.config_path(), value.root, value.allowlist))
        N.M1342.PRODUCTION_AUTHORITY_ALLOWLIST = copy.deepcopy(value.allowlist)
        record("code_allowlist_invented_from_fixture_authorities", lambda: N.build(
            value.config_path(), value.root, None))
    finally:
        N.M1342.PRODUCTION_AUTHORITY_ALLOWLIST = original_allowlist
        value.close()

    false_negatives = [row["attack"] for row in attacks if not row["rejected"]]
    output = {
        "schema": "m1353_m1351_table_a_memory_timed_authority_source_blind_hammer_r1_v1",
        "status": "PASS" if not false_negatives else "FAIL_DO_NOT_CITE",
        "attack_count": len(attacks),
        "rejected_count": len(attacks) - len(false_negatives),
        "false_negative_count": len(false_negatives),
        "false_negatives": false_negatives,
        "production_candidate_emitted": False,
        "table_a_row_emitted": False,
        "attacks": attacks,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not false_negatives else 1


if __name__ == "__main__":
    raise SystemExit(main())
