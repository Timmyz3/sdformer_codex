#!/usr/bin/env python3
"""Positive and adversarial regression for the M38-r3 reference model."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
SCRIPT = HW_ROOT / "system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r3.py"
CONTRACT = HW_ROOT / "contracts/m38_rst_math_input_contract_r3_20260822.json"
SPEC = importlib.util.spec_from_file_location("m38r3", str(SCRIPT))
M38 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M38)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M38R3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = M38.build(CONTRACT)
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.frame = M38.pack_configuration_frame(M38.GOLDEN_CONFIG)
        cls.fragments = M38.make_fragments(cls.frame)

    def write_contract(self, root, payload):
        path = root / "contract.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_status_and_both_independent_reviews_are_bound(self):
        self.assertEqual(
            self.result["status"],
            "PASS_M38_R3_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY")
        reviews = self.result["independent_review_admission_audit"]
        self.assertTrue(reviews["m31_r4"]["admitted"])
        self.assertTrue(reviews["m37_r8"]["admitted"])
        self.assertTrue(self.result["admission"]["recursive_anchor_identity_admitted"])
        self.assertFalse(self.result["admission"]["system_speedup_admitted"])

    def test_current_m31_r4_and_m37_r8_receipts_runs_are_recursive(self):
        anchors = self.result["recursive_anchor_audit"]
        m31 = anchors["m31_r4"]
        self.assertEqual(m31["receipt_schema"], "m31_output_receipt_v4")
        self.assertEqual(m31["source_files"], 6)
        self.assertEqual(m31["assert_properties"], 24)
        self.assertEqual(m31["cover_matches"], [32, 1, 26, 85])
        self.assertEqual(m31["run_basename"],
                         "m31_unified_t10_t2_vcs_r4_static_phase_20260822")
        self.assertEqual(m31["frozen_input_snapshot_ledger_sha256"],
                         "41009ec9ec86d4e19489bd49816634ca148340a0f19f784bd2d18bf2d3d0f22d")
        self.assertEqual(m31["frozen_input_snapshot_members"], 10)
        m37 = anchors["m37_r8"]
        self.assertEqual(m37["receipt_schema"], "m37_output_receipt_v3")
        self.assertEqual(m37["source_files"], 5)
        self.assertEqual(m37["assert_properties"], 21)
        self.assertEqual(m37["cover_matches"],
                         [220, 1271, 249, 117, 245, 571, 133, 210])
        self.assertEqual(m37["run_basename"],
                         "m37_csd_reconstruct_t10_vcs_r8_20260822")

    def test_independent_machine_admissions_are_vcs_only(self):
        reviews = self.result["independent_review_admission_audit"]
        m31 = reviews["m31_r4"]
        self.assertEqual(m31["sha256"],
                         "e8bd1b6452280396a5c8fc83ce79f34d1ae08256f97b469613207418dcfd0ff6")
        self.assertEqual(m31["validator_sha256"],
                         "ac0f932f98541c8458d14895494608c0a4fe2c6caf9842ac0a7af9ad495cb9f8")
        admission_path = M38.resolve(m31["path"])
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        self.assertTrue(admission["admission"]["current_r4_vcs_source_admitted"])
        for key in ("dc_sta_admitted", "formality_admitted", "headline_admitted",
                    "ppa_power_energy_admitted", "system_speedup_admitted"):
            self.assertFalse(admission["admission"][key])
        m37 = reviews["m37_r8"]
        self.assertEqual(m37["sha256"],
                         "f133b96a458686e17f94ecf52c26db3c9b753ef7145f4b396a9f047acfda0fa2")
        self.assertEqual(m37["validator_sha256"],
                         "7be9c7e5bba4ffb0fb972be948019dce5354362bc1da4e8d3e68057b0c4cce07")
        admission_path = M38.resolve(m37["path"])
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        self.assertTrue(admission["admitted"]["standalone_r8_vcs_functional"])
        self.assertTrue(admission["admitted"]["exact_sha_bound_source_intent"])
        self.assertTrue(admission["historical_manifest_resolution"]
                        ["live_r9_may_not_be_used_to_validate_r8"])
        for key in ("physical_zero_multiplier", "dc", "sta", "formality",
                    "ppa", "power", "energy", "system", "headline"):
            self.assertFalse(admission["admitted"][key])

    def test_scalar_full_domain_and_constructive_rank_sums(self):
        scalar = self.result["scalar_ternary_audit"]
        self.assertEqual(scalar["pairs_checked"], 768)
        self.assertEqual(scalar["product_range"], [-128, 128])
        for value in range(-128, 128):
            for code, coefficient in ((0, 0), (1, 1), (2, -1)):
                self.assertEqual(M38.ternary_product(value, code), value * coefficient)
        for total in range(-384, 385):
            terms = M38.constructive_rank3_decomposition(total)
            self.assertEqual(len(terms), 3)
            self.assertEqual(sum(M38.ternary_product(
                row["q8"], row["ternary_code"]) for row in terms), total)
        rank = self.result["rank3_q24_threshold_audit"]
        self.assertEqual(rank["constructive_rank_sum_values_checked"], 769)
        self.assertFalse(rank["all_legal_rank_triples_exhaustively_checked"])
        self.assertIn("EVERY_INTEGER_RANK_SUM", rank["statement"])

    def test_q24_saturation_and_threshold_boundaries(self):
        self.assertEqual(M38.saturate_q24(-(1 << 23) - 384), -(1 << 23))
        self.assertEqual(M38.saturate_q24((1 << 23) - 1 + 384), (1 << 23) - 1)
        rows = self.result["rank3_q24_threshold_audit"][
            "q24_saturation_threshold_cases"]
        self.assertTrue(any(row["raw"] == row["threshold"] and row["event"] == 1
                            for row in rows))
        self.assertTrue(any(row["raw"] == row["threshold"] - 1
                            and row["event"] == 0 for row in rows))

    def test_crc_golden_and_crc_correct_nonzero_pad_rejected(self):
        audit = self.result["canonical_crc_and_fragment_protocol_audit"]
        self.assertEqual(M38.crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(M38.decode_configuration_frame(self.frame), M38.GOLDEN_CONFIG)
        self.assertEqual(audit["golden_serialized_frame_sha256"],
                         "d77db6f549d0c851715b6353e1916670b36022bc19f05cc72b72a1dad6f97102")
        self.assertIn("crc_correct_nonzero_pad", audit["negative_cases_rejected"])
        payload = bytearray(self.frame[:74])
        payload[73] |= 1 << 1
        forged = bytes(payload) + M38.crc32c(payload).to_bytes(4, "little")
        with self.assertRaisesRegex(ValueError, "zero padding"):
            M38.decode_configuration_frame(forged)

    def test_fragment_exact_keys_types_ranges_and_wrong_valid(self):
        negative = self.result["canonical_crc_and_fragment_protocol_audit"][
            "negative_cases_rejected"]
        required = {
            "wrong_valid_nonlast", "wrong_valid_last", "extra_key", "boolean_index",
            "out_of_range_index", "out_of_order", "nonzero_duplicate",
            "nonzero_unused_high_bits", "bad_crc", "crc_correct_nonzero_pad",
            "illegal_ternary", "equal_generation", "ambiguous_delta_0x8000",
            "stale_delta_0x8001", "undrained_activation", "incomplete_frame"}
        self.assertEqual(set(negative), required)
        with self.assertRaisesRegex(ValueError, "population drift"):
            M38.validate_fragment(dict(self.fragments[0], extra=1))
        with self.assertRaisesRegex(ValueError, "type violation"):
            M38.validate_fragment(dict(self.fragments[0], data_u64=True))
        with self.assertRaisesRegex(ValueError, "valid-bit"):
            M38.validate_fragment(dict(self.fragments[9], valid_bits=64))

    def test_loader_failure_fragment0_recovery_and_active_atomicity(self):
        active = copy.deepcopy(M38.GOLDEN_CONFIG)
        active["generation_u16"] = (active["generation_u16"] - 1) & 0xFFFF
        loader = M38.StrictFragmentLoader(active)
        with self.assertRaisesRegex(ValueError, "order"):
            loader.accept(self.fragments[1], datapath_drained=True)
        self.assertEqual(loader.active_config, active)
        for fragment in self.fragments:
            activated = loader.accept(fragment, datapath_drained=True)
        self.assertTrue(activated)
        self.assertEqual(loader.active_config, M38.GOLDEN_CONFIG)

    def test_generation_half_range_equal_and_wrap(self):
        self.assertTrue(M38.generation_is_newer(0x7FFF, 0))
        self.assertFalse(M38.generation_is_newer(0x8000, 0))
        self.assertFalse(M38.generation_is_newer(7, 7))
        self.assertTrue(M38.generation_is_newer(1, 0xFFFE))

    def test_invalid_offers_are_state_atomic_before_fifo_pop(self):
        audit = self.result["offer_validation_atomicity_audit"]
        self.assertEqual(audit["state_atomic_rejections"], 11)
        self.assertTrue(audit["all_snapshots_exactly_equal_before_after"])
        model = M38.IntegratedCycleModel()
        model.switch_context("T10", 3)
        model.seed_fifo(4)
        before = model.canonical_snapshot()
        with self.assertRaisesRegex(ValueError, "population drift"):
            model.step(sink_ready=True,
                       t10_offer={"tag": 1, "generation": 3, "extra": 0})
        self.assertEqual(model.canonical_snapshot(), before)
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            model.step(sink_ready=True, other_writer_offer={
                "writer": "OTHER", "mode": "T2", "tag": 1, "beat": 0,
                "generation": 3})
        self.assertEqual(model.canonical_snapshot(), before)

    def test_complete_reachable_state_bfs(self):
        bfs = self.result["finite_reachable_state_audit"]
        self.assertEqual(bfs["graph_scope"],
                         "COMPLETE_FIXPOINT_FINITE_ABSTRACT_REACHABLE_STATE_GRAPH")
        self.assertEqual(bfs["reachable_states"], 669)
        self.assertEqual(bfs["transitions_checked"], 10438)
        self.assertEqual(bfs["reserved_values_reached"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(bfs["stage1_phases_reached"], ["idle", 1, 2, 3, 4])
        self.assertEqual(bfs["reconstruction_phases_reached"],
                         ["idle", 0, 1, 2, 3, 4])
        self.assertEqual(bfs["maximum_occupancy_plus_reserved"], 16)
        self.assertTrue(bfs["reservation_relation_holds"])
        self.assertTrue(bfs["single_writer_holds"])
        self.assertTrue(bfs["all_reachable_states_have_directed_drain_path"])
        self.assertFalse(bfs["general_fairness_liveness_admitted"])

    def test_finite_n_exact_ratios(self):
        rows = self.result["abstract_cycle_regression_audit"]["finite_n_regressions"]
        expected = {
            1: (10, Fraction(1, 1)), 2: (15, Fraction(4, 3)),
            3: (20, Fraction(3, 2)), 32: (165, Fraction(64, 33)),
            100: (505, Fraction(200, 101))}
        self.assertEqual([row["tiles"] for row in rows], [1, 2, 3, 32, 100])
        for row in rows:
            cycles, ratio = expected[row["tiles"]]
            self.assertEqual(row["parallel_commit_cycles"], cycles)
            self.assertEqual(Fraction(**row["exact_ratio"]), ratio)

    def test_stall_pending_full_pop_push_and_context_sequence(self):
        audit = self.result["abstract_cycle_regression_audit"]
        stalls = audit["eventual_sink_regressions"]
        self.assertEqual([row["stalled_cycles"] for row in stalls], [0, 90, 500])
        self.assertTrue(all(row["completion_and_drain_cycles"] < 10000 for row in stalls))
        pending = audit["pending_old_read_new_write"]
        self.assertEqual((pending["old_slot_tag"], pending["new_slot_tag"]), (0, 1))
        self.assertEqual(pending["done_tags"], [0, 1])
        self.assertTrue(audit["full_fifo_pop_push"][
            "old_head_returned_new_tail_written"])
        self.assertEqual(audit["writer_conflict"]["final_reserved"], 0)
        self.assertTrue(audit["writer_conflict"]["other_writer_denied"])
        self.assertEqual(audit["T10_T2_T10"]["mode_sequence"], ["T10", "T2", "T10"])

    def test_contract_population_identity_claim_and_protocol_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            top = copy.deepcopy(self.contract); top["extra"] = 1
            cases.append((top, "contract population drift"))
            identity = copy.deepcopy(self.contract); identity["identity"] += "_bad"
            cases.append((identity, "identity drift"))
            claim = copy.deepcopy(self.contract); claim["claim_boundary"] += " bad"
            cases.append((claim, "claim boundary drift"))
            fragment = copy.deepcopy(self.contract)
            fragment["canonical_configuration_frame"]["fragment_exact_keys"].append("bad")
            cases.append((fragment, "fragment schema drift"))
            reachable = copy.deepcopy(self.contract)
            reachable["reachable_state_model"]["reserved_domain"] = [0, 16]
            cases.append((reachable, "reachable-state contract drift"))
            for index, (payload, message) in enumerate(cases):
                path = root / "contract_{}.json".format(index)
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    M38.build(path)

    def test_forged_review_admission_boundaries_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = copy.deepcopy(self.contract)
            spec = contract["independent_review_admissions"]["m31_r4"]
            admission_path = M38.resolve(spec["path"])
            admission = json.loads(admission_path.read_text(encoding="utf-8"))
            admission["admission"]["system_speedup_admitted"] = True
            forged = root / "forged_review.json"
            forged.write_text(json.dumps(admission), encoding="utf-8")
            spec["path"] = str(forged)
            spec["sha256"] = digest(forged)
            with self.assertRaisesRegex(ValueError, "review admission boundary drift"):
                M38.build(self.write_contract(root, contract))
            contract = copy.deepcopy(self.contract)
            spec = contract["independent_review_admissions"]["m37_r8"]
            admission_path = M38.resolve(spec["path"])
            admission = json.loads(admission_path.read_text(encoding="utf-8"))
            admission["admitted"]["system"] = True
            forged = root / "forged_m37_review.json"
            forged.write_text(json.dumps(admission), encoding="utf-8")
            spec["path"] = str(forged)
            spec["sha256"] = digest(forged)
            with self.assertRaisesRegex(ValueError,
                                        "M37 review admission boundary drift"):
                M38.build(self.write_contract(root, contract))

    def test_claims_remain_model_only(self):
        admission = self.result["admission"]
        self.assertTrue(admission["both_independent_review_admissions_bound"])
        self.assertTrue(admission["recursive_anchor_identity_admitted"])
        self.assertTrue(admission["finite_abstract_reachable_state_safety_admitted"])
        self.assertTrue(admission["directed_drain_liveness_admitted"])
        self.assertFalse(admission["all_legal_rank_triples_exhaustively_checked"])
        self.assertFalse(admission["general_fairness_or_hardware_liveness_admitted"])
        for key in ("integrated_rtl_admitted", "integrated_rtl_vcs_admitted",
                    "dc_sta_formality_admitted", "area_power_energy_admitted",
                    "memory_and_system_cycles_admitted", "system_speedup_admitted",
                    "headline_admitted"):
            self.assertFalse(admission[key])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupied.json"
            path.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                M38.write_output(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
