from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" /
    "build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
)
SPEC = importlib.util.spec_from_file_location("m1228_cross_run_binder", SCRIPT)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1228CrossRunBinderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_run = self.root / "old_run"
        self.new_run = self.root / "new_run"
        self.old_run.mkdir()
        self.new_run.mkdir()
        self.old_config = self.root / "configs" / "old.yml"
        self.new_config = self.root / "configs" / "new.yml"
        self.old_config.parent.mkdir()
        self.old_config.write_text("experiment: old\n", encoding="utf-8")
        self.new_config.write_text("experiment: new\n", encoding="utf-8")
        self.new_manifest = self.root / "configs" / "new.json"
        self.new_manifest.write_text(
            json.dumps({"evaluation_epochs": [30, 32, 34], "label": "new"}) + "\n",
            encoding="utf-8",
        )

        old_checkpoint = self._checkpoint(self.old_run, 29)
        for epoch in (30, 32, 34):
            self._checkpoint(self.new_run, epoch)
        self.policy = M.CrossRunPolicy(
            candidates=(
                M.CandidatePolicy(
                    "legacy_ep29", self.old_run, self.old_config, sha(self.old_config), 29,
                    sha(old_checkpoint),
                ),
                M.CandidatePolicy(
                    "resume_ep30", self.new_run, self.new_config, sha(self.new_config), 30
                ),
                M.CandidatePolicy(
                    "resume_ep32", self.new_run, self.new_config, sha(self.new_config), 32
                ),
                M.CandidatePolicy(
                    "resume_ep34", self.new_run, self.new_config, sha(self.new_config), 34
                ),
            ),
            new_run_manifest=self.new_manifest,
            new_evaluation_epochs=(30, 32, 34),
        )
        for candidate, aee in zip(self.policy.candidates, (1.20, 1.10, 1.00, 1.00)):
            self._write_profile(candidate, aee)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _identity(path: Path) -> dict[str, object]:
        stat_result = path.stat()
        return {
            "absolute_path": str(path.resolve()),
            "size_bytes": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "sha256": sha(path),
        }

    @staticmethod
    def _checkpoint(run: Path, epoch: int) -> Path:
        path = run / f"checkpoint_epoch{epoch}.pth"
        path.write_bytes((f"checkpoint-{epoch}-" * 8).encode("ascii"))
        return path

    def _profile_path(self, candidate: object) -> Path:
        return (
            candidate.run_dir / "standard_valid825" /
            f"epoch{candidate.epoch}" / "spike_profile.json"
        )

    def _write_profile(self, candidate: object, aee: float) -> None:
        checkpoint = self._identity(
            candidate.run_dir / f"checkpoint_epoch{candidate.epoch}.pth"
        )
        config = self._identity(candidate.config)
        profile = {
            "samples": 825,
            "artifact_identity": {
                "config_path": config["absolute_path"],
                "config_sha256": config["sha256"],
                "checkpoint_path": checkpoint["absolute_path"],
                "checkpoint_size": checkpoint["size_bytes"],
                "checkpoint_mtime_ns": checkpoint["mtime_ns"],
                "checkpoint_sha256": checkpoint["sha256"],
            },
            "checkpoint_load_audit": {
                "checkpoint": checkpoint["absolute_path"],
                "missing_count": 0,
                "unexpected_count": 0,
                "overlay_missing_count": 0,
                "overlay_unexpected_count": 0,
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
            },
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "metrics": {
                "AEE": aee,
                "AAE": 5.0 + aee,
                "AAE_Benchmark": 4.0 + aee,
                "AEE_PE1": 1.0,
                "AEE_PE2": 2.0,
                "AEE_PE3": 3.0,
                "AEE_outliers": 4.0,
                "DSEC_Fl": 0.5,
            },
            "total_spikes": 1000 + candidate.epoch,
            "global_firing_rate": 0.05,
            "dense_flops": 1000.0,
            "effective_flops": 100.0,
            "energy_uj": 10.0,
        }
        path = self._profile_path(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8")

    def _mutate_profile(self, candidate_index: int, mutation) -> None:
        path = self._profile_path(self.policy.candidates[candidate_index])
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def test_cross_run_selection_and_selected_config_are_bound(self) -> None:
        result = M.build(self.policy)
        self.assertEqual(len(result["candidate_population"]), 4)
        self.assertEqual(result["selected"]["candidate_id"], "resume_ep32")
        self.assertEqual(result["selected"]["epoch"], 32)
        self.assertEqual(result["selected"]["configuration"]["sha256"], sha(self.new_config))
        self.assertEqual(result["selected"]["checkpoint"]["sha256"], sha(
            self.new_run / "checkpoint_epoch32.pth"
        ))
        targets = result["e0_e8_activation_dependent_invalidation_and_rebind_targets"]
        self.assertEqual([row["id"] for row in targets], [f"E{i}" for i in range(9)])
        self.assertTrue(all("selected checkpoint SHA/size/mtime" in row["dependency"]
                            for row in targets))

    def test_lowest_epoch_is_the_only_aee_tie_break(self) -> None:
        self._write_profile(self.policy.candidates[0], 0.75)
        self._write_profile(self.policy.candidates[1], 0.75)
        self._write_profile(self.policy.candidates[2], 0.80)
        self._write_profile(self.policy.candidates[3], 0.90)
        result = M.build(self.policy)
        self.assertEqual(result["selected"]["candidate_id"], "legacy_ep29")
        self.assertEqual(result["selected"]["epoch"], 29)
        self.assertEqual(result["selected"]["configuration"]["sha256"], sha(self.old_config))

    def test_receipt_has_selected_checkpoint_config_and_double_seal(self) -> None:
        output = self.root / "receipt"
        M.write_receipt(output, M.build(self.policy))
        expected = {
            "RUN_COMPLETE.txt", "e0_e8_activation_rebind_targets.json",
            "final_checkpoint_selection.json", "four_checkpoint_metrics.csv",
            "selected_checkpoint_and_config.json", "SHA256SUMS",
            "SHA256SUMS.seal.sha256",
        }
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        selected = json.loads((output / "selected_checkpoint_and_config.json").read_text())
        self.assertEqual(selected["epoch"], 32)
        manifest_rows = {}
        for line in (output / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            manifest_rows[name] = digest
        for name, digest in manifest_rows.items():
            self.assertEqual(sha(output / name), digest)
        self.assertEqual(
            (output / "SHA256SUMS.seal.sha256").read_text(),
            f"{sha(output / 'SHA256SUMS')}  SHA256SUMS\n",
        )
        with self.assertRaises(M.BinderError):
            M.write_receipt(output, M.build(self.policy))

    def test_missing_any_candidate_fails_closed(self) -> None:
        (self.new_run / "checkpoint_epoch34.pth").unlink()
        with self.assertRaises(M.BinderError):
            M.build(self.policy)

    def test_exact_candidate_population_and_cross_run_config_rules(self) -> None:
        attacks = (
            replace(self.policy, candidates=self.policy.candidates[:-1]),
            replace(self.policy, candidates=tuple(reversed(self.policy.candidates))),
            replace(self.policy, candidates=(
                self.policy.candidates[0],
                replace(self.policy.candidates[1], run_dir=self.old_run),
                *self.policy.candidates[2:],
            )),
            replace(self.policy, candidates=(
                self.policy.candidates[0],
                replace(self.policy.candidates[1], config=self.old_config),
                *self.policy.candidates[2:],
            )),
        )
        for attack in attacks:
            with self.subTest(attack=repr(attack.candidates)):
                with self.assertRaises(M.BinderError):
                    M.build(attack)

    def test_new_manifest_epoch_population_and_types_are_exact(self) -> None:
        originals = self.new_manifest.read_text()
        for value in ([30, 32], [30, 32, 34, 35], [30.0, 32, 34], [34, 32, 30]):
            with self.subTest(value=value):
                self.new_manifest.write_text(
                    json.dumps({"evaluation_epochs": value}) + "\n", encoding="utf-8"
                )
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
        self.new_manifest.write_text(originals, encoding="utf-8")

    def test_duplicate_manifest_json_key_is_rejected(self) -> None:
        self.new_manifest.write_text(
            '{"evaluation_epochs":[30,32,34],"evaluation_epochs":[30,32,34]}\n',
            encoding="utf-8",
        )
        with self.assertRaises(M.BinderError):
            M.build(self.policy)

    def test_config_and_legacy_checkpoint_sha_pins_reject_drift(self) -> None:
        attacks = (
            replace(self.policy, candidates=(
                replace(self.policy.candidates[0], config_sha256="0" * 64),
                *self.policy.candidates[1:],
            )),
            replace(self.policy, candidates=(
                replace(self.policy.candidates[0], expected_checkpoint_sha256="0" * 64),
                *self.policy.candidates[1:],
            )),
            replace(self.policy, candidates=(
                self.policy.candidates[0],
                replace(self.policy.candidates[1], config_sha256="0" * 64),
                *self.policy.candidates[2:],
            )),
        )
        for attack in attacks:
            with self.subTest(attack=repr(attack.candidates)):
                with self.assertRaises(M.BinderError):
                    M.build(attack)

    def test_samples_schema_rejects_float_bool_and_wrong_integer(self) -> None:
        path = self._profile_path(self.policy.candidates[2])
        canonical = path.read_text()
        for value in (825.0, True, 824, "825"):
            with self.subTest(value=value):
                profile = json.loads(canonical)
                profile["samples"] = value
                path.write_text(json.dumps(profile) + "\n")
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
        path.write_text(canonical)

    def test_all_artifact_identity_fields_are_exact(self) -> None:
        path = self._profile_path(self.policy.candidates[1])
        canonical = path.read_text()
        fields = tuple(sorted(M.ARTIFACT_IDENTITY_KEYS))
        for field in fields:
            with self.subTest(field=field):
                profile = json.loads(canonical)
                current = profile["artifact_identity"][field]
                profile["artifact_identity"][field] = (
                    current + 1 if type(current) is int else str(current) + ".drift"
                )
                path.write_text(json.dumps(profile) + "\n")
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
        profile = json.loads(canonical)
        profile["artifact_identity"]["extra"] = "forbidden"
        path.write_text(json.dumps(profile) + "\n")
        with self.assertRaises(M.BinderError):
            M.build(self.policy)
        path.write_text(canonical)

    def test_all_load_audit_zero_fields_are_typed(self) -> None:
        path = self._profile_path(self.policy.candidates[1])
        canonical = path.read_text()
        for field in M.LOAD_AUDIT_ZERO_KEYS:
            for value in (False, True, 0.0, "0", 1):
                with self.subTest(field=field, value=value):
                    profile = json.loads(canonical)
                    profile["checkpoint_load_audit"][field] = value
                    path.write_text(json.dumps(profile) + "\n")
                    with self.assertRaises(M.BinderError):
                        M.build(self.policy)
        path.write_text(canonical)

    def test_module_counts_are_exact_typed_and_no_extra_keys(self) -> None:
        path = self._profile_path(self.policy.candidates[3])
        canonical = path.read_text()
        attacks = (
            {"ATLIFTernaryPSN": 105.0, "ShiftmaxAttention": 12},
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": True},
            {"ATLIFTernaryPSN": 104, "ShiftmaxAttention": 12},
            {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12, "extra": 1},
        )
        for counts in attacks:
            with self.subTest(counts=counts):
                profile = json.loads(canonical)
                profile["module_counts"] = counts
                path.write_text(json.dumps(profile) + "\n")
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
        path.write_text(canonical)

    def test_aee_nonfinite_and_nonnumeric_values_are_rejected(self) -> None:
        path = self._profile_path(self.policy.candidates[1])
        canonical = path.read_text()
        for value in (float("nan"), float("inf"), {}, True):
            with self.subTest(value=repr(value)):
                profile = json.loads(canonical)
                profile["metrics"]["AEE"] = value
                path.write_text(json.dumps(profile) + "\n")
                with self.assertRaises(M.BinderError):
                    M.build(self.policy)
        path.write_text(canonical)

    def test_profile_symlink_is_rejected(self) -> None:
        candidate = self.policy.candidates[1]
        path = self._profile_path(candidate)
        target = path.with_name("real_profile.json")
        path.rename(target)
        path.symlink_to(target.name)
        with self.assertRaises(M.BinderError):
            M.build(self.policy)

    def test_source_has_no_model_gpu_remote_or_eda_execution_imports(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for forbidden in (
            "import torch", "import cupy", "import subprocess", "paramiko",
            "dc_shell", "vcs -full64", "ssh ",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("PRODUCTION_POLICY", text)
        self.assertIn("OLD_EP29_CHECKPOINT_SHA256", text)
        self.assertIn("NEW_CONFIG_SHA256", text)


if __name__ == "__main__":
    unittest.main()
