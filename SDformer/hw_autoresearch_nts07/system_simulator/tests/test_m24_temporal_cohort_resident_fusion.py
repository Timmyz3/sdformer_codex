import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m24_temporal_cohort_resident_fusion.py"
SPEC = importlib.util.spec_from_file_location("m24_temporal_cohort", str(SCRIPT))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "hw_autoresearch_nts07/contracts/m24_temporal_cohort_input_contract_r1_20260822.json"
GAP = REPO / "hw_autoresearch_nts07/contracts/m24_exact_temporal_bitmap_gap_contract_r1_20260822.json"


def npy_u8(rows):
    columns = len(rows[0])
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d), }" % (
        len(rows), columns,
    )
    padding = 16 - ((10 + len(header) + 1) % 16)
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    return (
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes))
        + header_bytes + b"".join(rows)
    )


class M24TemporalCohortResidentFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.contract_sha, cls.paths = MODULE.load_contract(
            CONTRACT, MODULE.sha256(CONTRACT), REPO
        )
        cls.payload, cls.operator_rows = MODULE.build(
            cls.contract, cls.contract_sha, cls.paths, REPO, GAP
        )

    def test_minimal_numpy_reader_is_python36_compatible_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bits.npz"
            rows = [bytes([0x01, 0x80]), bytes([0x55, 0xAA])]
            with zipfile.ZipFile(str(archive), "w") as handle:
                handle.writestr("packed_current_bits.npy", npy_u8(rows))
            self.assertEqual(
                MODULE.read_npy_u8(archive, "packed_current_bits.npy"), rows
            )
            self.assertEqual(MODULE.bitmap(rows[0]), 0x8001)

    def test_contract_and_gap_are_content_bound_and_fail_closed(self):
        self.assertEqual(self.contract_sha, MODULE.sha256(CONTRACT))
        with self.assertRaisesRegex(ValueError, "input contract SHA mismatch"):
            MODULE.load_contract(CONTRACT, "0" * 64, REPO)
        gap = json.loads(GAP.read_text(encoding="utf-8"))
        self.assertEqual(
            gap["status"],
            "BLOCKED_UNTIL_EXHAUSTIVE_STREAMING_COHORT_CENSUS_EXISTS",
        )
        self.assertIn("--dual-line-cohort-census-dir", json.dumps(gap))

    def test_exact_sampled_masks_do_not_masquerade_as_57_operator_census(self):
        payload = self.payload
        self.assertEqual(payload["headline_gate"]["requested_eligible_operators"], 57)
        self.assertEqual(payload["headline_gate"]["operator_topology"], 79)
        self.assertFalse(payload["headline_gate"]["admitted"])
        self.assertGreater(
            payload["headline_gate"]["observed_maximum_fallback_coefficient_fraction"],
            0.99,
        )
        self.assertEqual(payload["identities"]["h67_ep35"]["exact_operator_names"], 31)
        self.assertEqual(payload["identities"]["local_ep44"]["exact_operator_names"], 36)
        for identity in payload["identities"].values():
            self.assertEqual(identity["operator_topology"], 79)
            for line in identity["line_metrics"].values():
                self.assertGreater(line["fallback_coefficient_fraction"], 0.99)
                self.assertFalse(line["headline_coverage_admitted"])

    def test_cohort_conserves_updates_and_reports_two_fair_schedules(self):
        for identity in self.payload["identities"].values():
            local = identity["line_metrics"]["local_line"]
            motion = identity["line_metrics"]["motion_selector_shared_state"]
            for line in (local, motion):
                self.assertEqual(
                    line["coefficient_scalar_reads_step_major"],
                    line["destination_scalar_updates"],
                )
                self.assertEqual(
                    line["positive_destination_scalar_updates"]
                    + line["negative_destination_scalar_updates"],
                    line["destination_scalar_updates"],
                )
                self.assertLess(
                    line["coefficient_scalar_reads_cohort"],
                    line["coefficient_scalar_reads_step_major"],
                )
                self.assertLess(
                    line["serialized_read_plus_update_operation_envelope"]["sampled_component_speedup"],
                    2.0,
                )
                self.assertEqual(
                    line["fully_overlapped_read_update_envelope"]["sampled_component_speedup"],
                    1.0,
                )
                self.assertEqual(
                    line["equal_resource_baseline"]["resident_capacity_bits"],
                    line["cohort_resident_peak_bits"],
                )
                strongest = line["strongest_composable_same_resource_baseline"]
                self.assertEqual(strongest["coefficient_scalar_reads"], line["coefficient_scalar_reads_cohort"])
                self.assertEqual(strongest["operation_envelope_speedup_of_cohort_masks"], 1.0)
                self.assertEqual(strongest["traffic_reduction_of_cohort_masks"], 0.0)
            self.assertGreater(
                motion["selector_control_bits"], local["selector_control_bits"]
            )
            self.assertGreater(motion["negative_destination_scalar_updates"], 0)
            self.assertEqual(local["negative_destination_scalar_updates"], 0)

    def test_bn_liveness_retains_m22_two_movement_boundary(self):
        fusion = self.payload["resident_fusion"]
        self.assertEqual(fusion["exact_h67_single_sample_edges"], 13)
        self.assertEqual(fusion["exact_h67_single_sample_elements"], 552960000)
        self.assertEqual(fusion["strict_transactions_deleted_from_canonical_m22"], 0)
        self.assertEqual(fusion["strict_bytes_deleted_from_canonical_m22"], 0)
        self.assertEqual(fusion["strongest_composable_baseline_speedup"], 1.0)
        self.assertFalse(
            fusion["liveness"]["direct_pre_barrier_atlif_residency_admitted"]
        )
        self.assertIn(
            "not_cycles",
            "m22_serialized_byte_service_ticks_copied_as_logical_ticks_not_cycles",
        )

    def test_amdahl_constants_are_reconciled_not_silently_mixed(self):
        audit = self.payload["amdahl_and_2x"]
        self.assertAlmostEqual(
            audit["recomputed_eligible_engine_speedup_required_for_2x"],
            7.782511971133631,
        )
        self.assertAlmostEqual(audit["frozen_legacy_target_for_2x"], 7.687553)
        self.assertFalse(audit["threshold_consistency_admitted"])
        self.assertIn("NO_GO", audit["two_x_gate"])
        for identity in audit["sampled_mask_what_if_not_headline"].values():
            for line in identity.values():
                self.assertEqual(line["strongest_same_resource_component_speedup"], 1.0)
                self.assertEqual(line["strongest_same_resource_hypothetical_full_system_speedup"], 1.0)

    def test_output_manifest_and_receipt_bind_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "m24"
            MODULE.write_outputs(
                output, self.payload, self.operator_rows, CONTRACT, GAP
            )
            manifest_path = output / "m24_output_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["claim"],
                "EXACT_SAMPLED_COHORT_DSE_AND_GAP_CONTRACT_NOT_SYSTEM_CYCLES_OR_SPEEDUP",
            )
            self.assertTrue(all(not Path(path).is_absolute() for path in manifest["sources"]))
            for name, entry in manifest["artifacts"].items():
                self.assertEqual(entry["sha256"], MODULE.sha256(output / name))
                self.assertEqual(entry["bytes"], (output / name).stat().st_size)
            receipt = {}
            for line in (output / "m24_evidence.sha256").read_text(encoding="utf-8").splitlines():
                digest, name = line.split(None, 1)
                receipt[name] = digest
            self.assertEqual(
                receipt["m24_output_manifest.json"], MODULE.sha256(manifest_path)
            )

    def test_rebuild_is_deterministic(self):
        second, rows = MODULE.build(
            self.contract, self.contract_sha, self.paths, REPO, GAP
        )
        self.assertEqual(second, self.payload)
        self.assertEqual(rows, self.operator_rows)
        self.assertEqual(
            second["content_sha256_excluding_this_field"],
            MODULE.canonical_sha256({
                key: value for key, value in second.items()
                if key != "content_sha256_excluding_this_field"
            }),
        )


if __name__ == "__main__":
    unittest.main()
