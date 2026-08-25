import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m23_physical_memory_schedule.py"
SPEC = importlib.util.spec_from_file_location("m23_physical_memory_schedule", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M23PhysicalMemoryScheduleTest(unittest.TestCase):
    def config(self):
        return {
            "dram_burst_bytes": 64,
            "dram_global_bursts_per_tick": 3,
            "dram_banks": 16,
            "dram_row_bytes": 8192,
            "dram_read_ports_per_bank": 1,
            "dram_write_ports_per_bank": 1,
            "sram_word_bytes": 4,
            "sram_banks": 24,
            "sram_row_bytes": 96,
            "sram_read_ports_per_bank": 1,
            "sram_write_ports_per_bank": 1,
            "sram_allocation_alignment_bytes": 96,
            "sram_physical_base_address": 0x80000040,
        }

    def row(self, transaction_id, tier, direction, phase, object_id, span, count,
            address, pattern="CONTIGUOUS", order=None):
        if order is None:
            order = transaction_id
        ticks = MODULE.ceil_div(count, 96 if tier == "SRAM" else 192)
        return {
            "transaction_id": transaction_id,
            "identity": "local_ep44",
            "variant": "local_line",
            "request_issue_order": order,
            "previous_in_trace_order": order - 1 if order else -1,
            "sample_id": 0,
            "call_index": order // 2,
            "event_kind": "operator",
            "name": "fixture.op{}".format(order // 2),
            "temporal_step": -1,
            "tier": tier,
            "direction": direction,
            "phase": phase,
            "address": "0x{:016x}".format(address),
            "address_integer": address,
            "object_id": object_id,
            "object_span_bytes": span,
            "byte_count": count,
            "serialized_service_start": 0,
            "serialized_service_end_exclusive": ticks,
            "service_bytes_per_cycle": 96 if tier == "SRAM" else 192,
            "address_pattern": pattern,
            "evidence_class": "FIXTURE",
        }

    def fixture(self):
        rows = [
            self.row(0, "DRAM", "READ", "dram_activation_read", "dram:a", 65, 65,
                     0x100000000000),
            self.row(1, "SRAM", "WRITE", "sram_activation_fill", "sram:a", 65, 65,
                     0x80000000),
            self.row(2, "SRAM", "READ", "activation_dense_read", "sram:a", 65, 65,
                     0x80000000),
            self.row(3, "SRAM", "READ", "coefficient_term_read", "sram:w", 12, 240,
                     0x80000080, "CYCLIC_WEIGHT_OBJECT_COMPRESSED"),
            self.row(4, "SRAM", "WRITE", "operator_acc_write", "sram:o", 128, 128,
                     0x80000100),
        ]
        transport = sum(row["serialized_service_end_exclusive"] for row in rows)
        summary = {
            "config": {"dram_bytes_per_cycle": 192, "sram_bytes_per_cycle": 96},
            "identities": {
                "local_ep44": {
                    "attention_coverage_status": "MISSING_FROM_EXECUTION_TRACE_NOT_ZERO_COST",
                    "attention_execution_records": 0,
                    "sample_count": 1,
                    "profile_identity": {"module_counts": {"ShiftmaxAttention": 2}},
                    "variants": {
                        "local_line": {
                            "sram_logical_span_bytes": 1024,
                            "totals": {"serialized_byte_service_ticks": transport},
                        }
                    },
                }
            }
        }
        return summary, rows

    def test_contract_and_m22_input_are_fail_closed(self):
        repo = Path(__file__).resolve().parents[3]
        path = repo / "hw_autoresearch_nts07/contracts/m23_m22_input_contract_r1_20260822.json"
        digest = MODULE.sha256(path)
        contract, actual, artifact = MODULE.load_contract(path, digest, repo)
        self.assertEqual(actual, digest)
        self.assertEqual(contract["transaction_records"], 208425)
        summary, rows = MODULE.load_m22(contract, artifact)
        self.assertEqual(len(rows), 208425)
        self.assertEqual(summary["transactions_sha256"], contract["transactions_sha256"])
        with self.assertRaisesRegex(ValueError, "contract SHA mismatch"):
            MODULE.load_contract(path, "0" * 64, repo)

        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "contract.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["transactions_sha256"] = "f" * 64
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            tampered_contract, _sha, tampered_artifact = MODULE.load_contract(
                tampered, MODULE.sha256(tampered), repo
            )
            with self.assertRaisesRegex(ValueError, "manifest/summary identity mismatch"):
                MODULE.load_m22(tampered_contract, tampered_artifact)

    def test_allocator_reuses_dead_buffers_and_rejects_overlap(self):
        objects = [
            {"instance_id": "a", "object_span_bytes": 100,
             "first_issue_order": 0, "last_issue_order": 2, "call_index": 0,
             "category": "activation_input"},
            {"instance_id": "b", "object_span_bytes": 80,
             "first_issue_order": 3, "last_issue_order": 5, "call_index": 1,
             "category": "operator_output"},
            {"instance_id": "c", "object_span_bytes": 32,
             "first_issue_order": 1, "last_issue_order": 4, "call_index": 0,
             "category": "atlif_state"},
        ]
        allocations, totals = MODULE.allocate_lifetimes(objects, 16, physical_base=0)
        by_name = {row["instance_id"]: row for row in allocations}
        self.assertEqual(by_name["a"]["physical_offset"], by_name["b"]["physical_offset"])
        self.assertTrue(by_name["b"]["reused_region"])
        self.assertLessEqual(totals["peak_live_aligned_bytes"], totals["allocator_capacity_bytes"])
        self.assertEqual(totals["allocator_policy"], "COALESCED_BEST_FIT_BY_SIZE_THEN_BASE")
        broken = [dict(row) for row in allocations]
        broken_by_name = {row["instance_id"]: row for row in broken}
        broken_by_name["c"]["physical_offset"] = broken_by_name["a"]["physical_offset"]
        with self.assertRaisesRegex(ValueError, "allocator overlap"):
            MODULE.validate_allocator_nonoverlap(broken)

    def test_allocator_is_explicitly_best_fit_not_first_fit(self):
        objects = [
            {"instance_id": "a", "object_span_bytes": 100,
             "first_issue_order": 0, "last_issue_order": 2, "call_index": 0,
             "category": "activation_input"},
            {"instance_id": "spacer", "object_span_bytes": 20,
             "first_issue_order": 0, "last_issue_order": 10, "call_index": 0,
             "category": "atlif_state"},
            {"instance_id": "b", "object_span_bytes": 50,
             "first_issue_order": 1, "last_issue_order": 2, "call_index": 0,
             "category": "operator_output"},
            {"instance_id": "new", "object_span_bytes": 40,
             "first_issue_order": 3, "last_issue_order": 4, "call_index": 1,
             "category": "activation_input"},
        ]
        allocations, _totals = MODULE.allocate_lifetimes(objects, 1, physical_base=0)
        by_name = {row["instance_id"]: row for row in allocations}
        self.assertEqual(by_name["a"]["physical_offset"], 0)
        self.assertEqual(by_name["spacer"]["physical_offset"], 100)
        self.assertEqual(by_name["b"]["physical_offset"], 120)
        # At issue 3, free blocks are 100B@0 and 50B@120. Best-fit takes 50B@120.
        self.assertEqual(by_name["new"]["physical_offset"], 120)

    def test_fixed_burst_and_cyclic_bank_conservation(self):
        config = self.config()
        dram = self.row(0, "DRAM", "READ", "dram_activation_read", "dram:a", 65, 65,
                        0x100000000000)
        scheduled = MODULE.schedule_transaction(dram, None, config, -1, -1)
        self.assertEqual(scheduled["compressed_request_count"], 2)
        self.assertEqual(scheduled["payload_bytes"], 65)
        self.assertEqual(scheduled["transferred_bytes_with_edge_padding"], 128)

        row = self.row(1, "SRAM", "READ", "coefficient_term_read", "sram:w", 12, 240,
                       0x80000000, "CYCLIC_WEIGHT_OBJECT_COMPRESSED")
        allocation = {
            "instance_id": "w", "logical_base_address": 0x80000000,
            "physical_base_address": 0x80000040, "object_span_bytes": 12,
        }
        cyclic = MODULE.schedule_transaction(row, allocation, config, 0, -1)
        counts = MODULE.cyclic_bank_counts(0, 60, 3, 24)
        self.assertEqual(sum(counts), cyclic["compressed_request_count"])
        self.assertEqual(max(counts), 20)
        self.assertGreater(cyclic["bank_conflict_stall_ticks_lower_bound"], 0)
        self.assertEqual(cyclic["row_index_start"], cyclic["row_index_end"])
        self.assertEqual(cyclic["visited_row_count_lower_bound"], 1)

        contiguous = self.row(
            2, "SRAM", "READ", "activation_dense_read", "sram:x", 192, 8,
            0x80000000 + 92, "CONTIGUOUS"
        )
        crossed = MODULE.schedule_transaction(
            contiguous,
            {"instance_id": "x", "logical_base_address": 0x80000000,
             "physical_base_address": 0x80000040, "object_span_bytes": 192},
            config, 1, -1,
        )
        self.assertEqual(crossed["visited_row_count_lower_bound"], 2)
        self.assertEqual(crossed["row_index_end"] - crossed["row_index_start"], 1)

    def test_unknown_row_selection_has_honest_conflict_envelope(self):
        row = self.row(0, "SRAM", "READ", "motion_previous_acc_read", "sram:o", 4096,
                       384, 0x80000000, "ROW_SELECTED_WITHIN_PREVIOUS_TIMESTEP")
        allocation = {
            "instance_id": "o", "logical_base_address": 0x80000000,
            "physical_base_address": 0x80000040, "object_span_bytes": 4096,
        }
        result = MODULE.schedule_transaction(row, allocation, self.config(), -1, -1)
        self.assertEqual(result["bank_request_count_rle"], "UNKNOWN_ROW_SELECTION")
        self.assertEqual(result["physical_address"], "UNKNOWN_WITHIN_PHYSICAL_OBJECT")
        self.assertEqual(result["row_index_start"], "")
        self.assertEqual(result["visited_row_interval_rle"], "UNKNOWN_WITHIN_OBJECT_ROW_ENVELOPE")
        self.assertEqual(result["bank_service_ticks_lower_bound"], 4)
        self.assertEqual(result["bank_service_ticks_upper_bound"], 96)
        self.assertGreater(result["bank_conflict_stall_ticks_upper_bound"], 0)

    def test_regeneration_is_deterministic_and_attention_is_not_free(self):
        summary, rows = self.fixture()
        meta = {"contract_sha256": "e" * 64}
        first = MODULE.build(summary, rows, self.config(), meta)
        second = MODULE.build(summary, rows, self.config(), meta)
        self.assertEqual(first, second)
        payload, allocations, schedule = first
        self.assertEqual(payload["schedule_records"], len(rows))
        self.assertEqual(len(schedule), len(rows))
        self.assertTrue(allocations)
        attention = payload["identities"]["local_ep44"]["attention"]
        self.assertEqual(attention["status"], "MISSING_TRACE_UNKNOWN_NONZERO_NOT_SCHEDULED")
        self.assertEqual(attention["minimum_missing_module_calls"], 2)
        self.assertEqual(attention["unmodeled_physical_bytes"], "UNKNOWN_NONZERO")
        claims = " ".join(payload["claim_boundary"]["forbidden"])
        self.assertIn("system cycles", claims)
        self.assertIn("DRAMsim3", claims)
        service = payload["identities"]["local_ep44"]["variants"]["local_line"][
            "transport_and_bank_service"
        ]
        self.assertEqual(service["dram_read_payload_bytes"], 65)
        self.assertEqual(service["sram_read_payload_bytes"], 305)
        self.assertEqual(service["sram_write_payload_bytes"], 193)

    def test_frozen_r5_output_receipt_and_source_identity(self):
        repo = Path(__file__).resolve().parents[3]
        receipt_path = repo / "hw_autoresearch_nts07/contracts/m23_output_receipt_r5_20260822.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "m23_frozen_output_receipt_v1")
        self.assertEqual(
            receipt["status"], "FROZEN_PHYSICAL_MEMORY_ENVELOPE_NOT_SYSTEM_SPEEDUP"
        )
        artifact = repo / receipt["artifact_directory"]
        for name, expected in receipt["files_sha256"].items():
            self.assertEqual(MODULE.sha256(artifact / name), expected)
        summary = json.loads((artifact / "m23_summary.json").read_text(encoding="utf-8"))
        output_manifest = json.loads(
            (artifact / "m23_output_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["schedule_records"], receipt["schedule_records"])
        self.assertEqual(summary["allocation_records"], receipt["allocation_records"])
        self.assertEqual(
            summary["input"]["m22_transactions_sha256"], receipt["m22_transactions_sha256"]
        )
        self.assertEqual(
            output_manifest["sources_sha256"][SCRIPT.name], MODULE.sha256(SCRIPT)
        )
        self.assertEqual(
            output_manifest["sources_sha256"][Path(__file__).name], MODULE.sha256(Path(__file__))
        )


if __name__ == "__main__":
    unittest.main()
