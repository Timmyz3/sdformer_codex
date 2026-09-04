from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.report_qfit_local5_vector_fair_baseline import build_report


class VectorFairReportTest(unittest.TestCase):
    def _ledger(self, inplace: int, seed: int, cycles: int, suffix: str = "aa") -> str:
        return (
            "PASS Local5 multi-tile memo=0 "
            f"inplace={inplace} acc_backend=1 tx_service=1 seed={seed} "
            f"cycles={cycles} token=4050 token_delay_sum=100 "
            "weight_delay_sum=200 result_service=43200 hits=0 fallback=0 "
            "replay_records=0 partial=0 final=43200 child_results=0 "
            "weight_cycles=1 frontend_cycles=2 readout_cycles=3 "
            "release_cycles=4 rmw_cycles=0 drain_cycles=0 scheduler_cycles=5 "
            f"vector=1 token_service_hash={suffix} "
            f"weight_service_hash={suffix} result_service_hash={suffix}\n"
        )

    def _tree(self, mismatch: bool = False) -> tuple[tempfile.TemporaryDirectory, Path]:
        holder = tempfile.TemporaryDirectory()
        out = Path(holder.name)
        for seed in (17717, 44257, 48879):
            for name, inplace, cycles in (
                ("b0v_materialize", 0, 310000),
                ("b2v_resident", 1, 295000),
            ):
                suffix = "bb" if mismatch and name.startswith("b2v") else "aa"
                text = self._ledger(inplace, seed, cycles, suffix)
                (out / f"{name}_seed_{seed}_iverilog.log").write_text(text)
                (out / f"{name}_seed_{seed}_verilator_sva.log").write_text(text)
        (out / "tool_versions.txt").write_text("tools\n")
        source = out / "source.txt"
        source.write_text("source\n")
        import hashlib
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (out / "source_sha256.txt").write_text(f"{digest}  {source}\n")
        return holder, out

    def test_expected_cycle_reject(self) -> None:
        holder, out = self._tree()
        try:
            oracle = out / "metadata.json"
            oracle.write_text("{}\n")
            with patch(
                "scripts.report_qfit_local5_vector_fair_baseline.ROOT",
                out.parent,
            ), patch(
                "scripts.report_qfit_local5_vector_fair_baseline.Path.is_file",
                return_value=True,
            ), patch(
                "scripts.report_qfit_local5_vector_fair_baseline.sha256",
                return_value="0" * 64,
            ), patch(
                "scripts.report_qfit_local5_vector_fair_baseline.verify_sha_manifest",
                return_value=1,
            ):
                report = build_report(out)
            self.assertEqual(report["status"], "REJECT_VECTOR_RESIDENCY")
            self.assertFalse(report["comparison"]["cycle_gate_pass"])
        finally:
            holder.cleanup()

    def test_service_hash_mismatch_fails(self) -> None:
        holder, out = self._tree(mismatch=True)
        try:
            with self.assertRaisesRegex(ValueError, "service identity mismatch"):
                with patch(
                    "scripts.report_qfit_local5_vector_fair_baseline.Path.is_file",
                    return_value=True,
                ):
                    build_report(out)
        finally:
            holder.cleanup()


if __name__ == "__main__":
    unittest.main()
