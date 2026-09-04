import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "hw_autoresearch_nts07/system_simulator/scripts/"
    "analyze_m38_rst_math_crc_and_cycle_r2.py"
)
RESULT = (
    ROOT / "hw_autoresearch_nts07/results/"
    "m38_rst_math_crc_and_cycle_r2_20260822/"
    "m38_rst_math_crc_and_cycle.json"
)
SPEC = importlib.util.spec_from_file_location("m38r2", str(SCRIPT))
M38 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M38)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M38R2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M38.build()
        cls.contract = json.loads(
            M38.DEFAULT_CONTRACT.read_text(encoding="utf-8")
        )
        cls.golden = cls.contract["canonical_configuration_frame"]["golden_frame"]
        cls.golden_config = {
            "right_factor_q8": cls.golden["right_factor_q8"],
            "left_ternary_code": cls.golden["left_ternary_code"],
            "bias_q24": cls.golden["bias_q24"],
            "threshold_q24": cls.golden["threshold_q24"],
            "stage1_requant_shift_u5": cls.golden["stage1_requant_shift_u5"],
            "generation_u16": cls.golden["generation_u16"],
        }

    def test_python36_complete_math_and_widths(self):
        scalar = self.result["scalar_ternary_audit"]
        rank3 = self.result["rank3_q24_threshold_audit"]
        self.assertEqual(scalar["pairs_checked"], 768)
        self.assertEqual(scalar["product_range"], [-128, 128])
        self.assertEqual(scalar["minimum_signed_product_bits"], 9)
        self.assertEqual(
            scalar["negative_minimum_negation_witness"]["result"], 128
        )
        self.assertRegex(scalar["rows_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(rank3["rank3_sum_range"], [-384, 384])
        self.assertEqual(rank3["minimum_signed_rank3_sum_bits"], 10)
        self.assertEqual(
            rank3["mathematical_minimum_bias_plus_rank_sum_bits"], 25
        )
        self.assertEqual(rank3["implemented_pre_saturation_bits_target"], 26)
        self.assertEqual(rank3["threshold_equality_event"], 1)
        self.assertEqual(rank3["threshold_just_below_event"], 0)
        self.assertRegex(rank3["saturation_rows_sha256"], r"^[0-9a-f]{64}$")
        for value in range(-128, 128):
            for code, coefficient in ((0, 0), (1, 1), (2, -1)):
                self.assertEqual(
                    M38.ternary_product(value, code), value * coefficient
                )

    def test_recursive_final_anchor_identity_and_logs(self):
        anchors = self.result["recursive_anchor_audit"]
        m31 = anchors["m31_r3"]
        m37 = anchors["m37_r7"]
        self.assertEqual(m31["receipt_schema"], M38.M31_RECEIPT_SCHEMA)
        self.assertEqual(m31["live_source_count"], 6)
        self.assertEqual(m31["assert_property_count"], 24)
        self.assertEqual(m31["log_audit"]["cover_property_count"], 4)
        self.assertEqual(len(m31["log_audit"]["cover_nonzero_match_counts"]), 4)
        self.assertTrue(m31["r2_stale_live_source_drift_eliminated"])
        self.assertEqual(m37["receipt_schema"], M38.M37_RECEIPT_SCHEMA)
        self.assertEqual(m37["live_source_count"], 5)
        self.assertEqual(m37["recursive_input_manifest_count"], 8)
        self.assertEqual(m37["assert_property_count"], 21)
        self.assertEqual(m37["log_audit"]["cover_property_count"], 8)
        self.assertEqual(len(m37["log_audit"]["cover_nonzero_match_counts"]), 8)
        self.assertTrue(m37["observed_receipt_population_fully_reconciled"])
        self.assertTrue(m37["r1_stale_receipt_rejected"])

    def test_nested_receipt_source_drift_fails_closed(self):
        contract = copy.deepcopy(self.contract)
        receipt_path = M38.resolve(
            contract["inputs"]["m31_vcs_receipt"]["path"]
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["files"]["unified_core_rtl"] = [
            "/definitely/missing_m31_core.sv", "0" * 64
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_receipt = directory / "fake_m31_receipt.json"
            fake_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            contract["inputs"]["m31_vcs_receipt"] = {
                "path": str(fake_receipt), "sha256": digest(fake_receipt)
            }
            fake_contract = directory / "fake_contract.json"
            fake_contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt live source drift"):
                M38.build(fake_contract)

    def test_exact_status_and_stale_m37_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            contract = copy.deepcopy(self.contract)
            receipt_path = M38.resolve(
                contract["inputs"]["m31_vcs_receipt"]["path"]
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["status"] += "_ALMOST"
            fake_receipt = directory / "bad_status.json"
            fake_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            contract["inputs"]["m31_vcs_receipt"] = {
                "path": str(fake_receipt), "sha256": digest(fake_receipt)
            }
            fake_contract = directory / "bad_status_contract.json"
            fake_contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact receipt status"):
                M38.build(fake_contract)

            contract = copy.deepcopy(self.contract)
            stale = ROOT / (
                "hw_autoresearch_nts07/contracts/"
                "m37_output_receipt_r1_20260822.json"
            )
            contract["inputs"]["m37_vcs_receipt"] = {
                "path": str(stale), "sha256": digest(stale)
            }
            stale_contract = directory / "stale_m37_contract.json"
            stale_contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact receipt schema"):
                M38.build(stale_contract)

    def test_contract_protocol_metadata_and_receipt_population_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, mutate, message in (
                    ("frame", lambda item: item[
                        "canonical_configuration_frame"].__setitem__(
                            "load_fragment_count", 9),
                     "canonical frame metadata drift"),
                    ("cycle", lambda item: item[
                        "abstract_cycle_protocol"].__setitem__(
                            "fifo_push_ports", 2),
                     "abstract cycle protocol drift")):
                contract = copy.deepcopy(self.contract)
                mutate(contract)
                path = directory / (name + "_contract.json")
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    M38.build(path)

            contract = copy.deepcopy(self.contract)
            receipt_path = M38.resolve(
                contract["inputs"]["m31_vcs_receipt"]["path"]
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["observed"]["unreviewed_extra"] = 1
            fake_receipt = directory / "extra_observed.json"
            fake_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            contract["inputs"]["m31_vcs_receipt"] = {
                "path": str(fake_receipt), "sha256": digest(fake_receipt)
            }
            path = directory / "extra_observed_contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observed receipt key"):
                M38.build(path)

    def test_canonical_crc_standard_golden_and_widths(self):
        audit = self.result["canonical_crc_and_fragment_protocol_audit"]
        payload = M38.pack_protected_payload(self.golden_config)
        frame = M38.pack_configuration_frame(self.golden_config)
        self.assertEqual(M38.crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(payload.hex(), self.golden["protected_payload_hex"])
        self.assertEqual(M38.crc32c(payload), 0x4FBC4933)
        self.assertEqual(frame.hex(), self.golden["serialized_frame_hex"])
        self.assertEqual(M38.decode_configuration_frame(frame), self.golden_config)
        self.assertEqual(len(payload) * 8, 592)
        self.assertEqual(len(frame) * 8, 624)
        self.assertEqual(audit["logical_context_bits_excluding_pad"], 617)
        self.assertEqual(audit["protected_bits_before_padding"], 585)
        self.assertEqual(audit["zero_pad_bits_before_crc"], 7)
        self.assertEqual(audit["serialized_context_bits_including_pad"], 624)

    def test_strict_fragments_generation_and_active_atomicity(self):
        frame = M38.pack_configuration_frame(self.golden_config)
        fragments = M38.make_fragments(frame)
        loader = M38.StrictFragmentLoader()
        for fragment in fragments:
            activated = loader.accept(fragment, datapath_drained=True)
        self.assertTrue(activated)
        self.assertEqual(loader.active_config, self.golden_config)

        with self.assertRaisesRegex(ValueError, "fragment order"):
            broken = M38.StrictFragmentLoader()
            broken.accept(fragments[1], datapath_drained=True)
        with self.assertRaisesRegex(ValueError, "restart requires fragment zero"):
            broken.accept(fragments[2], datapath_drained=True)

        corrupted = bytearray(frame)
        corrupted[5] ^= 0x80
        active = copy.deepcopy(self.golden_config)
        bad_crc = M38.StrictFragmentLoader(active)
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            for fragment in M38.make_fragments(bytes(corrupted)):
                bad_crc.accept(fragment, datapath_drained=True)
        self.assertEqual(bad_crc.active_config, active)

        self.assertTrue(M38.generation_is_newer(1, 0xFFFE))
        self.assertFalse(M38.generation_is_newer(7, 7))
        self.assertFalse(M38.generation_is_newer(0x8000, 0))
        self.assertEqual(
            set(self.result["canonical_crc_and_fragment_protocol_audit"]
                ["negative_protocol_cases_rejected"]),
            {"out_of_order", "duplicate_fragment", "nonzero_unused_high_bits",
             "bad_crc", "illegal_ternary", "stale_generation",
             "undrained_activation", "incomplete_frame"},
        )

    def test_abstract_no_stall_finite_n_and_ii5(self):
        model, accepts, dones = M38.run_no_stall_tiles(32)
        self.assertEqual(accepts, list(range(0, 160, 5)))
        self.assertEqual(dones, list(range(9, 165, 5)))
        self.assertEqual(dones[-1] + 1, 5 + 5 * 32)
        self.assertLessEqual(model.maximum_occupancy_plus_reserved, 16)
        self.assertTrue(all(
            event["done"] == (
                event["m38_push"] and event.get("m38_push_beat") == 4
            ) for event in model.history
        ))

    def test_pending_materialize_is_old_read_new_write_and_live(self):
        model, event = M38.run_pending_trace()
        self.assertTrue(event["slot_pop"])
        self.assertTrue(event["slot_push"])
        self.assertTrue(event["pending_materialize"])
        self.assertEqual(event["slot_old_read_tag"], "tile0")
        self.assertEqual(event["slot_new_write_tag"], "tile1")
        self.assertEqual(model.done_tags, ["tile0", "tile1"])
        self.assertIsNone(model.pending)

    def test_fifo_reservation_arbitration_and_counterexample(self):
        state_space = M38.audit_credit_state_space()
        self.assertEqual(state_space["states_checked"], 578)
        self.assertEqual(state_space["maximum_occupancy_plus_reserved"], 16)
        event, model = M38.run_writer_conflict_prevention()
        self.assertTrue(event["m38_push"])
        self.assertTrue(event["other_writer_denied"])
        self.assertEqual(len(model.fifo), 16)
        self.assertEqual(model.reserved, 0)
        self.assertEqual(
            self.result["abstract_integrated_cycle_audit"]
                ["buggy_unreserved_shared_writer_occupancy_trace"],
            [13, 14, 15, 16, 17],
        )

    def test_eventual_sink_full_pop_push_and_context_drain(self):
        model, cycles = M38.run_eventual_sink_liveness(40, 90)
        self.assertEqual(model.done_tags, list(range(40)))
        self.assertTrue(model.drained())
        self.assertLess(cycles, 2000)
        self.assertLessEqual(model.maximum_occupancy_plus_reserved, 16)
        full = M38.run_full_pop_push_and_context_drain()
        self.assertTrue(full["full_old_read_new_write"])
        sequence = M38.run_t10_t2_t10_drain_sequence()
        self.assertEqual(sequence["mode_sequence"], ["T10", "T2", "T10"])
        self.assertEqual(sequence["undrained_switch_rejections"], 2)

    def test_claims_remain_closed_and_frozen_result_rebuilds(self):
        admission = self.result["admission"]
        self.assertTrue(admission["recursive_anchor_identity_admitted"])
        self.assertTrue(admission["canonical_crc32c_frame_admitted"])
        self.assertTrue(admission["abstract_integrated_cycle_safety_and_liveness_admitted"])
        self.assertFalse(admission["integrated_rtl_admitted"])
        self.assertFalse(admission["integrated_rtl_vcs_admitted"])
        self.assertFalse(admission["dc_sta_formality_admitted"])
        self.assertFalse(admission["area_power_energy_admitted"])
        self.assertFalse(admission["system_speedup_admitted"])
        self.assertFalse(admission["headline_admitted"])
        frozen = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(frozen, self.result)

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m38_r2.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M38.write_output(output, {"bad": "overwrite"})
            self.assertEqual(output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
