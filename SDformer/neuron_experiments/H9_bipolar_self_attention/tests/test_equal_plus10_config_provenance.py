from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "entrypoints/make_dsec_fullres_w15_equal_plus10_configs.py"
)
SPEC = importlib.util.spec_from_file_location("equal_plus10_configs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EqualPlus10ConfigProvenanceTest(unittest.TestCase):
    def build(self, source: Path, offset: int, label: int) -> dict:
        return MODULE.build(
            source,
            experiment="test_equal_plus10",
            epoch_offset=offset,
            source_checkpoint_label=label,
        )

    def test_h67_replaces_stale_rescue_provenance(self) -> None:
        config = self.build(MODULE.H67_SOURCE, 1, 30)
        MODULE.validate(config, epoch_offset=1, source_checkpoint_label=30)
        runtime = config["runtime"]
        self.assertNotIn("resume_source_epoch", runtime)
        self.assertEqual(runtime["resume_source_budget"], 30)
        self.assertEqual(runtime["resume_source_checkpoint_label"], 30)

    def test_nb0_records_zero_based_checkpoint_label(self) -> None:
        config = self.build(MODULE.NB0_SOURCE, 0, 29)
        MODULE.validate(config, epoch_offset=0, source_checkpoint_label=29)
        runtime = config["runtime"]
        self.assertEqual(runtime["resume_source_budget"], 30)
        self.assertEqual(runtime["resume_source_checkpoint_label"], 29)

    def test_local5_extension_is_preregistered_from_its_own_ep29(self) -> None:
        config = self.build(MODULE.LOCAL5_SOURCE, 0, 29)
        MODULE.validate(config, epoch_offset=0, source_checkpoint_label=29)
        self.assertEqual(config["loader"]["n_epochs"], 40)
        self.assertEqual(config["runtime"]["force_save_epochs"], [34, 39])
        self.assertEqual(config["runtime"]["state_save_epochs"], [34, 39])
        self.assertEqual(config["runtime"]["rescue_init"], "own_crop_rank1_epoch29")


if __name__ == "__main__":
    unittest.main()
