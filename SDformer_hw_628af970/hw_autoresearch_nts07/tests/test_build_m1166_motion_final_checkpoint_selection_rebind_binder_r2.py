from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1166_motion_final_checkpoint_selection_rebind_binder_r2.py"
SPEC = importlib.util.spec_from_file_location("m1166_binder_r2", SCRIPT)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1166TypedZeroTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.config = self.root / "config.yml"
        self.config.write_text("name: typed-zero-fixture\n", encoding="utf-8")
        self.ranking = self.run / "profile_ranking_valid825.md"
        self.aees = {9: "1.50", 14: "1.40", 19: "1.30", 24: "1.20", 29: "1.10"}
        self.policy = M.R1.RunPolicy(
            run_dir=self.run, config=self.config, ranking=self.ranking,
            config_sha256=sha(self.config),
        )
        for epoch in M.R1.EPOCHS:
            self._write_epoch(epoch)
        lines = ["# Standard Valid825 Ranking", "", "Ranking mode: `aee`.", "",
                 "| rank | epoch | AEE |", "|---:|---:|---:|"]
        for rank, epoch in enumerate((29, 24, 19, 14, 9), 1):
            lines.append(f"| {rank} | {epoch} | {self.aees[epoch]} |")
        self.ranking.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _profile(self, epoch: int) -> Path:
        return self.run / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"

    def _write_epoch(self, epoch: int) -> None:
        checkpoint = self.run / f"checkpoint_epoch{epoch}.pth"
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
        ckpt_stat = checkpoint.stat()
        profile_dir = self.run / "standard_valid825" / f"epoch{epoch}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        value = {
            "samples": 825,
            "artifact_identity": {
                "config_path": str(self.config.resolve()),
                "config_sha256": sha(self.config),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": ckpt_stat.st_size,
                "checkpoint_mtime_ns": ckpt_stat.st_mtime_ns,
                "checkpoint_sha256": sha(checkpoint),
            },
            "checkpoint_load_audit": {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
                "missing_count": 0,
                "unexpected_count": 0,
                "overlay_missing_count": 0,
                "overlay_unexpected_count": 0,
            },
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "metrics": {
                "AEE": self.aees[epoch], "AAE": "9", "AAE_Benchmark": "8",
                "AEE_PE1": "0.4", "AEE_PE2": "0.2", "AEE_PE3": "0.1",
                "AEE_outliers": "0.1", "DSEC_Fl": "5",
            },
            "total_spikes": 20_000_000_000 + epoch,
            "global_firing_rate": 0.05,
            "dense_flops": 1000.0,
            "effective_flops": 200.0,
            "energy_uj": 123.0,
        }
        self._profile(epoch).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def test_canonical_and_output_namespace(self) -> None:
        result = M.build(self.policy)
        self.assertEqual(result["selected"]["epoch"], 29)
        self.assertEqual(result["source_hardening"]["typed_zero_rule"],
                         "type(value) is int and value == 0")
        output = self.root / "receipt"
        M.write_receipt(output, result)
        self.assertEqual(
            (output / "RUN_COMPLETE.txt").read_text(),
            "PASS_M1166_FINAL_CHECKPOINT_SELECTED_R2_TYPED_ZERO__"
            "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n",
        )

    def test_four_counter_type_attacks(self) -> None:
        for key in M.LOAD_AUDIT_ZERO_KEYS:
            for label, forged in (("false", False), ("true", True), ("string", "0"),
                                  ("float", 0.0)):
                with self.subTest(key=key, forged=label):
                    path = self._profile(19)
                    canonical = path.read_text(encoding="utf-8")
                    value = json.loads(canonical)
                    value["checkpoint_load_audit"][key] = forged
                    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
                    with self.assertRaises(M.R1.BinderError):
                        M.build(self.policy)
                    path.write_text(canonical, encoding="utf-8")

    def test_r1_dependency_identity_is_pinned(self) -> None:
        self.assertEqual(sha(M.R1_SOURCE), M.R1_SOURCE_SHA256)

    def test_extra_epoch_profile_directory_rejected(self) -> None:
        extra = self.run / "standard_valid825" / "epoch99"
        extra.mkdir()
        (extra / "spike_profile.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(M.R1.BinderError):
            M.build(self.policy)

    def test_duplicate_or_mixed_ranking_mode_rejected(self) -> None:
        canonical = self.ranking.read_text(encoding="utf-8")
        for forged in (
            canonical.replace("Ranking mode: `aee`.\n", "Ranking mode: `aee`.\nRanking mode: `aee`.\n"),
            canonical.replace("Ranking mode: `aee`.\n", "Ranking mode: `candidate`.\nRanking mode: `aee`.\n"),
        ):
            with self.subTest(forged=forged.splitlines()[2:4]):
                self.ranking.write_text(forged, encoding="utf-8")
                with self.assertRaises(M.R1.BinderError):
                    M.build(self.policy)
        self.ranking.write_text(canonical, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
