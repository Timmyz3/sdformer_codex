from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1163_motion_final_checkpoint_selection_rebind_binder.py"
SPEC = importlib.util.spec_from_file_location("m1163_binder", SCRIPT)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1163BinderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.config = self.root / "config.yml"
        self.ranking = self.run_dir / "profile_ranking_valid825.md"
        self.run_dir.mkdir()
        self.config.write_text("name: m1163-fixture\n", encoding="utf-8")
        self.policy = M.RunPolicy(
            run_dir=self.run_dir,
            config=self.config,
            ranking=self.ranking,
            config_sha256=sha(self.config),
        )
        self.aees = {9: "1.50", 14: "1.40", 19: "1.30", 24: "1.20", 29: "1.10"}
        for epoch in M.EPOCHS:
            self._write_epoch(epoch)
        self._write_ranking("aee", (29, 24, 19, 14, 9))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_epoch(self, epoch: int) -> None:
        checkpoint = self.run_dir / f"checkpoint_epoch{epoch}.pth"
        checkpoint.write_bytes((f"checkpoint-{epoch}-" * 4).encode())
        stat = checkpoint.stat()
        profile_dir = self.run_dir / "standard_valid825" / f"epoch{epoch}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = {
            "samples": 825,
            "artifact_identity": {
                "config_path": str(self.config.resolve()),
                "config_sha256": sha(self.config),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
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
                "AEE": self.aees[epoch], "AAE": "9.0", "AAE_Benchmark": "8.0",
                "AEE_PE1": "0.4", "AEE_PE2": "0.2", "AEE_PE3": "0.1",
                "AEE_outliers": "0.1", "DSEC_Fl": "5.0",
            },
            "total_spikes": 20_000_000_000 + epoch,
            "global_firing_rate": 0.05,
            "dense_flops": 1000.0,
            "effective_flops": 200.0,
            "energy_uj": 123.0,
        }
        (profile_dir / "spike_profile.json").write_text(
            json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_ranking(self, mode: str, epochs: tuple[int, ...]) -> None:
        lines = ["# Standard Valid825 Ranking", "", f"Ranking mode: `{mode}`.", "",
                 "| rank | epoch | AEE |", "|---:|---:|---:|"]
        for rank, epoch in enumerate(epochs, 1):
            lines.append(f"| {rank} | {epoch} | {self.aees[epoch]} |")
        self.ranking.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _profile_path(self, epoch: int) -> Path:
        return self.run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"

    def _mutate_profile(self, epoch: int, mutation) -> None:
        path = self._profile_path(epoch)
        value = json.loads(path.read_text())
        mutation(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def assert_rejected(self, function) -> None:
        with self.assertRaises(M.BinderError):
            function()

    def test_canonical_build_and_seal(self) -> None:
        result = M.build(self.policy)
        self.assertEqual(result["selected"]["epoch"], 29)
        self.assertEqual(len(result["five_checkpoint_metric_table"]), 5)
        self.assertEqual([row["id"] for row in result["e0_e8_invalidation_and_rebind_targets"]],
                         [f"E{index}" for index in range(9)])
        self.assertFalse(result["claim_boundary"]["hardware_rebind_authorized"])
        output = self.root / "receipt"
        M.write_receipt(output, result)
        self.assertEqual(
            (output / "RUN_COMPLETE.txt").read_text(),
            "PASS_M1163_FINAL_CHECKPOINT_SELECTED__INDEPENDENT_HAMMER_REQUIRED__"
            "NO_HARDWARE_REBIND_AUTHORITY\n",
        )
        manifest = output / "SHA256SUMS"
        self.assertEqual(
            (output / "SHA256SUMS.seal.sha256").read_text(),
            f"{sha(manifest)}  SHA256SUMS\n",
        )

    def test_tie_break_is_lowest_epoch(self) -> None:
        self.aees[9] = self.aees[14] = "1.00"
        self._write_epoch(9); self._write_epoch(14)
        self._write_ranking("aee", (9, 14, 29, 24, 19))
        self.assertEqual(M.build(self.policy)["selected"]["epoch"], 9)

    def test_missing_fifth_profile_rejected(self) -> None:
        self._profile_path(29).unlink()
        self.assert_rejected(lambda: M.build(self.policy))

    def test_samples_824_rejected(self) -> None:
        self._mutate_profile(19, lambda value: value.__setitem__("samples", 824))
        self.assert_rejected(lambda: M.build(self.policy))

    def test_artifact_identity_drift_rejected(self) -> None:
        self._mutate_profile(14, lambda value: value["artifact_identity"].__setitem__(
            "checkpoint_sha256", "0" * 64))
        self.assert_rejected(lambda: M.build(self.policy))

    def test_all_load_audit_axes_rejected(self) -> None:
        for key in ("missing_count", "unexpected_count", "overlay_missing_count",
                    "overlay_unexpected_count"):
            with self.subTest(key=key):
                original = self._profile_path(24).read_text()
                self._mutate_profile(24, lambda value, key=key:
                                     value["checkpoint_load_audit"].__setitem__(key, 1))
                self.assert_rejected(lambda: M.build(self.policy))
                self._profile_path(24).write_text(original)

    def test_module_count_drift_rejected(self) -> None:
        self._mutate_profile(9, lambda value: value["module_counts"].__setitem__(
            "ATLIFTernaryPSN", 104))
        self.assert_rejected(lambda: M.build(self.policy))

    def test_candidate_ranking_mode_rejected(self) -> None:
        self._write_ranking("candidate", (29, 24, 19, 14, 9))
        self.assert_rejected(lambda: M.build(self.policy))

    def test_incomplete_or_wrong_ranking_rejected(self) -> None:
        self._write_ranking("aee", (29, 24, 19, 14))
        self.assert_rejected(lambda: M.build(self.policy))
        self._write_ranking("aee", (24, 29, 19, 14, 9))
        self.assert_rejected(lambda: M.build(self.policy))

    def test_nonfinite_and_duplicate_json_rejected(self) -> None:
        path = self._profile_path(9)
        path.write_text('{"samples":825,"samples":825}\n', encoding="utf-8")
        self.assert_rejected(lambda: M.build(self.policy))
        self._write_epoch(9)
        text = self._profile_path(9).read_text().replace('"AEE": "1.50"', '"AEE": NaN')
        self._profile_path(9).write_text(text, encoding="utf-8")
        self.assert_rejected(lambda: M.build(self.policy))

    def test_config_and_checkpoint_drift_rejected(self) -> None:
        self.config.write_text("name: forged\n", encoding="utf-8")
        self.assert_rejected(lambda: M.build(self.policy))
        self.config.write_text("name: m1163-fixture\n", encoding="utf-8")
        checkpoint = self.run_dir / "checkpoint_epoch29.pth"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
        self.assert_rejected(lambda: M.build(self.policy))

    def test_symlink_profile_rejected(self) -> None:
        path = self._profile_path(14)
        target = path.with_name("real_profile.json")
        path.rename(target)
        path.symlink_to(target)
        self.assert_rejected(lambda: M.build(self.policy))

    def test_output_overwrite_rejected(self) -> None:
        output = self.root / "receipt"
        output.mkdir()
        self.assert_rejected(lambda: M.write_receipt(output, M.build(self.policy)))


if __name__ == "__main__":
    unittest.main()
