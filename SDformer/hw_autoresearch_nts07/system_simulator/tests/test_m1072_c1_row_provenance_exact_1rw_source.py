#!/usr/bin/env python3
"""Directed M1065/file-integrity attacks for source-only M1072."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/run_m1072_c1_row_provenance_exact_1rw_source.py"
SPEC = importlib.util.spec_from_file_location("m1072_test_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1072 source")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1072Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with M.CanonicalRowReader() as reader:
            cls.real0 = reader.derive(0)
            cls.raw1, _ = reader.raw_for_task(1)

    def resign(self, record):
        return replace(
            record,
            provenance_sha256=hashlib.sha256(
                M._canonical_provenance_payload(M.record_payload(record))
            ).hexdigest(),
        )

    def test_small_oracle(self):
        value = M.small_oracle()
        self.assertEqual(value["status"],
                         "PASS_M1072_SMALL_ORACLE__M1073_REQUIRED_NO_FULL_REPLAY")
        self.assertTrue(all(value["m1065_attacks_rejected"].values()))

    def test_unique_production_iterator_has_zero_arguments(self):
        self.assertEqual(len(inspect.signature(
            M.iter_canonical_full_replay_results).parameters), 0)
        self.assertTrue(inspect.isgeneratorfunction(
            M.iter_canonical_full_replay_results))

    def test_task0_frozen_anchor(self):
        self.assertEqual(self.real0.shared_preprocess_cycles, 210)
        self.assertEqual(self.real0.works, {
            "candidate": 1664, "strongest_zero": 4392,
            "same_coordinate_bit": 4392,
        })
        self.assertEqual(self.real0.raw_row_bytes_sha256,
                         "169d433ac7ab62aabe2cc48786139fea70eb58724dd2eb431778135ebaec794b")
        self.assertEqual(self.real0.masks_le16_sha256,
                         "bb602234a1a5183d09d8307214e7b0085d2b993b38483db7ab1acc5d4966b5d9")

    def test_manual_0_999999_work_and_preprocess_forgery_rejected(self):
        forged = replace(
            self.real0,
            shared_preprocess_cycles=0,
            works={"candidate": 0, "strongest_zero": 999_999,
                   "same_coordinate_bit": 999_999},
            parents={
                "candidate": {"reads": 0, "writes": 0, "forwards": 0,
                              "work_cycles": 0},
                "strongest_zero": {"reads": 0, "writes": 0, "forwards": 0,
                                   "work_cycles": 999_999},
                "same_coordinate_bit": {"reads": 0, "writes": 0,
                                        "forwards": 0, "work_cycles": 999_999},
            },
        )
        forged = self.resign(forged)
        M.validate_record_shape(forged)
        with self.assertRaisesRegex(RuntimeError, "canonical row"):
            M.validate_external_records_against_frozen([forged])

    def test_all_zero_mask_record_rejected(self):
        zero = M.derive_record_from_exact_raw(
            0, b"00000000\n" * 64, self.real0.file_offset
        )
        self.assertEqual(zero.shared_preprocess_cycles, 146)
        self.assertTrue(all(value == 0 for value in zero.works.values()))
        with self.assertRaisesRegex(RuntimeError, "canonical row"):
            M.validate_external_records_against_frozen([zero])

    def test_row_reorder_record_rejected(self):
        reordered = M.derive_record_from_exact_raw(
            0, self.raw1, self.real0.file_offset
        )
        self.assertNotEqual(reordered.raw_row_bytes_sha256,
                            self.real0.raw_row_bytes_sha256)
        with self.assertRaisesRegex(RuntimeError, "canonical row"):
            M.validate_external_records_against_frozen([reordered])

    def test_wrong_digest_and_provenance_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            M.validate_record_shape(replace(self.real0, provenance_sha256="0" * 64))
        changed = replace(self.real0, raw_row_bytes_sha256="0" * 64)
        changed = self.resign(changed)
        with self.assertRaisesRegex(RuntimeError, "canonical row"):
            M.validate_external_records_against_frozen([changed])

    def test_short_pread_rejected(self):
        reader = M.CanonicalRowReader()
        try:
            with mock.patch.object(M.os, "pread", return_value=b""):
                with self.assertRaisesRegex(RuntimeError, "short pread"):
                    reader.raw_for_task(0)
        finally:
            reader.close()

    def test_file_stat_drift_rejected(self):
        reader = M.CanonicalRowReader()
        try:
            actual = M.os.fstat(reader._fd)
            fake = SimpleNamespace(
                st_dev=actual.st_dev, st_ino=actual.st_ino,
                st_size=actual.st_size, st_mtime_ns=actual.st_mtime_ns + 1,
                st_ctime_ns=actual.st_ctime_ns, st_mode=actual.st_mode,
            )
            with mock.patch.object(M.os, "fstat", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "file drift"):
                    reader._verify_unchanged(final_hash=False)
        finally:
            reader.close()

    def test_canonical_file_path_size_sha(self):
        self.assertEqual(M.ROWS.stat().st_size, 466_560_000)
        self.assertEqual(M.ROWS_SHA,
                         "6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334")
        self.assertFalse(M.ROWS.is_symlink())

    def test_external_empty_records_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            M.validate_external_records_against_frozen([])

    def test_coverage_constructor_and_empty_proof_closed(self):
        self.assertEqual(len(inspect.signature(M.ProvenanceCoverage).parameters), 0)
        with self.assertRaises(TypeError):
            M.ProvenanceCoverage(next_task_id=M.TASKS)
        self.assertFalse(M.ProvenanceCoverage().proof()["full_coverage_pass"])

    def test_parent_summary_bound_to_work(self):
        for design in M.DESIGNS:
            self.assertEqual(self.real0.parents[design]["work_cycles"],
                             self.real0.works[design])
        self.assertEqual(self.real0.parents["candidate"], {
            "reads": 408, "writes": 248, "forwards": 32,
            "work_cycles": 1664,
        })

    def test_exact_contract_and_fake_contract_rejection(self):
        self.assertEqual(M.validate_sealed_contract()["contract_sha256"],
                         M.CONTRACT_SHA)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
            json.dump({"status": "PASS_M1072_SEALED_SOURCE_CONTRACT__M1073_REQUIRED_NO_LAUNCH",
                       "launch_now": False}, stream)
            stream.flush()
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                M.validate_sealed_contract(Path(stream.name))

    def test_capacity_and_claim_boundary_closed(self):
        capacity = M.M1064.derive_physical_capacity()
        self.assertEqual(capacity["derived_total_bytes"], 214_912)
        self.assertFalse(capacity["capacity_only_214912B_admitted"])
        boundary = M.small_oracle()["claim_boundary"]
        for key in ("m1073_passed", "launch_now", "full_51840000_replay",
                    "full_trace_port_feasibility", "capacity_only_214912B_admitted",
                    "matched_cycles_admitted", "speedup_admitted", "rtl_cycles",
                    "paper_ppa_ready"):
            self.assertFalse(boundary[key])


if __name__ == "__main__":
    unittest.main()
