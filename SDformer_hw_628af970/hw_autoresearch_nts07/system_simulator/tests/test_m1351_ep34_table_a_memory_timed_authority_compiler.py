from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "build_m1351_ep34_table_a_memory_timed_authority_compiler.py")
OLD_TEST = ROOT / (
    "hw_autoresearch_nts07/system_simulator/tests/"
    "test_m1342_ep34_table_a_authority_compiler.py")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


N = load("m1351_source", SOURCE)
F = load("m1351_m1342_fixture", OLD_TEST)
M = N.M


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_json(path: Path, value: object) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o444)


def add_m1351_authorities(fixture: F.Fixture) -> None:
    energy_path = fixture.role_dirs["energy_producer"] / "partitioned_energy.json"
    energy = json.loads(energy_path.read_text())
    for branch_index, branch in enumerate(M.DIRECT_BRANCHES):
        energy["direct_logic_pj_per_cycle"][branch] = {
            row: 2.0 + branch_index * 0.1 for row in M.ROWS}
    rewrite_json(energy_path, energy)
    fixture.role_payloads["energy_producer"]["partitioned_energy"]["sha256"] = digest(
        energy_path)
    fixture.seal("energy_producer")

    trace = b"fixture-address-trace\nrow0 R SRAM0 0x0000 32\n"
    trace_spec = fixture.payload("transaction_receipt", "address_trace.csv", trace,
                                 "text/csv", "address_trace")
    receipt_path = fixture.role_dirs["transaction_receipt"] / "transaction.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["address_trace_sha256"] = trace_spec["sha256"]
    rewrite_json(receipt_path, receipt)
    fixture.role_payloads["transaction_receipt"]["transaction_receipt"]["sha256"] = digest(
        receipt_path)
    timing = {"schema": N.MEMORY_TIMING_SCHEMA,
              "identity": "Motion-C12-ep34-final",
              "address_trace_sha256": trace_spec["sha256"],
              "latency_model": {"sram_read_cycles": 1, "sram_write_cycles": 1,
                                "dram_read_cycles": 24, "dram_write_cycles": 26},
              "rows": {row: {} for row in M.ROWS}}
    for row in M.ROWS:
        for key, charge in receipt["rows"][row].items():
            timing["rows"][row][key] = {
                "address_timed_cycles": charge["cycles"],
                "memory_stall_cycles": 3,
                "sram_stall_cycles": 1,
                "dram_stall_cycles": 2,
            }
    fixture.payload_json("transaction_receipt", "memory_timing.json", timing,
                         "memory_timing")
    fixture.seal("transaction_receipt")


class Tests(unittest.TestCase):
    def fixture(self):
        fixture = F.Fixture()
        self.addCleanup(fixture.close)
        add_m1351_authorities(fixture)
        return fixture

    def test_01_good_source_fixture_is_memory_timed_and_not_production(self):
        fixture = self.fixture()
        result = N.build(fixture.config_path(), fixture.root, fixture.allowlist)
        self.assertEqual(result["status"],
                         "PASS_SOURCE_FIXTURE_MEMORY_TIMED_NOT_PRODUCTION")
        self.assertGreater(result["memory_timing"]["dram_bytes"], 0)
        self.assertEqual(set(result["memory_timing"]["sram_bytes_by_macro"]),
                         set(M.SRAM_MACROS))
        self.assertTrue(result["claim_boundary"]["memory_latency_and_stalls_bound"])
        self.assertFalse(result["claim_boundary"]["paper_headline_admitted"])

    def test_02_parent_traversal_is_rejected_after_resolution(self):
        fixture = self.fixture()
        config = fixture.config_path()
        traversing = fixture.root / "does_not_exist" / ".." / config.name
        with self.assertRaisesRegex(N.CompileError, "parent traversal"):
            N.build(traversing, fixture.root, fixture.allowlist)

    def test_03_direct_logic_rate_must_be_row_invariant(self):
        fixture = self.fixture()
        path = fixture.role_dirs["energy_producer"] / "partitioned_energy.json"
        value = json.loads(path.read_text())
        value["direct_logic_pj_per_cycle"][M.DIRECT_BRANCHES[0]]["Ours"] = 1e-9
        rewrite_json(path, value)
        fixture.role_payloads["energy_producer"]["partitioned_energy"]["sha256"] = digest(path)
        fixture.seal("energy_producer")
        with self.assertRaisesRegex(N.CompileError, "row invariant"):
            N.build(fixture.config_path(), fixture.root, fixture.allowlist)

    def test_04_address_trace_sha_requires_lowercase_hex(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        with self.assertRaisesRegex(N.CompileError, "grammar"):
            N.validate_transaction_extensions(receipt, trace, "g" * 64, timing)

    def test_05_receipt_must_bind_sealed_trace_payload(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        receipt["address_trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(N.CompileError, "not bound"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_06_dram_plane_cannot_be_all_zero(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        for row in receipt["rows"].values():
            for charge in row.values():
                charge["dram_read_bytes"] = charge["dram_write_bytes"] = 0
        with self.assertRaisesRegex(N.CompileError, "DRAM"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_07_sram_planes_cannot_be_all_zero(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        for row in receipt["rows"].values():
            for charge in row.values():
                for access in charge["sram_bytes"].values():
                    access["read_bytes"] = access["write_bytes"] = 0
        with self.assertRaisesRegex(N.CompileError, "SRAM"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_08_each_sram_macro_plane_must_be_nonzero(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        victim = M.SRAM_MACROS[0]
        for row in receipt["rows"].values():
            for charge in row.values():
                charge["sram_bytes"][victim] = {"read_bytes": 0, "write_bytes": 0}
        with self.assertRaisesRegex(N.CompileError, "SRAM"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_09_memory_timing_payload_is_mandatory(self):
        fixture = self.fixture()
        del fixture.role_payloads["transaction_receipt"]["memory_timing"]
        fixture.seal("transaction_receipt")
        with self.assertRaisesRegex(N.CompileError, "trace/timing"):
            N.build(fixture.config_path(), fixture.root, fixture.allowlist)

    def test_10_address_timed_cycles_equal_conserved_cycles(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        row = M.ROWS[0]; key = next(iter(timing["rows"][row]))
        timing["rows"][row][key]["address_timed_cycles"] += 1
        with self.assertRaisesRegex(N.CompileError, "differ"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_11_memory_stall_partition_is_exact(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        row = M.ROWS[0]; key = next(iter(timing["rows"][row]))
        timing["rows"][row][key]["dram_stall_cycles"] += 1
        with self.assertRaisesRegex(N.CompileError, "partition"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_12_latency_model_cannot_be_zero(self):
        fixture = self.fixture()
        receipt, trace, timing = self.extension_inputs(fixture)
        timing["latency_model"]["dram_read_cycles"] = 0
        with self.assertRaisesRegex(N.CompileError, "positive"):
            N.validate_transaction_extensions(receipt, trace, digest(trace), timing)

    def test_13_m1346_failure_and_production_boundary_are_exact(self):
        N.verify_m1346_failure()
        fixture = self.fixture()
        fixture.config["status"] = "PRODUCTION_CANDIDATE"
        with self.assertRaisesRegex(N.CompileError, "allowlist is not populated"):
            N.build(fixture.config_path(), fixture.root, fixture.allowlist)

    @staticmethod
    def extension_inputs(fixture):
        root = fixture.role_dirs["transaction_receipt"]
        receipt = json.loads((root / "transaction.json").read_text())
        trace = root / "address_trace.csv"
        timing = json.loads((root / "memory_timing.json").read_text())
        return copy.deepcopy(receipt), trace, copy.deepcopy(timing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
