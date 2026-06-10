from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))


class DummySpikingNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresh = nn.Parameter(torch.tensor(1.0))


class DummyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.spiking_neuron = DummySpikingNeuron()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sn_q = DummyWrapper()
        self.linear = nn.Linear(1, 1, bias=False)


class H9LoadAuditTest(unittest.TestCase):
    def _save(self, state_dict):
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        tmp.close()
        torch.save(state_dict, tmp.name)
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_h9_config_rejects_checkpoint_without_overlay_keys(self):
        from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

        model = DummyModel()
        checkpoint = self._save({"linear.weight": torch.ones_like(model.linear.weight)})

        with self.assertRaisesRegex(RuntimeError, "does not contain H9 overlay"):
            load_checkpoint_with_h9_audit(
                checkpoint,
                model,
                torch.device("cpu"),
                config={"atlif_ternary_psn": {"enabled": True}},
            )

    def test_baseline_config_rejects_checkpoint_with_overlay_keys(self):
        from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

        model = DummyModel()
        checkpoint = self._save(model.state_dict())

        with self.assertRaisesRegex(RuntimeError, "requires an H9 config"):
            load_checkpoint_with_h9_audit(checkpoint, model, torch.device("cpu"), config={})

    def test_h9_config_loads_matching_overlay_checkpoint(self):
        from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

        model = DummyModel()
        checkpoint = self._save(model.state_dict())

        loaded = load_checkpoint_with_h9_audit(
            checkpoint,
            model,
            torch.device("cpu"),
            config={"atlif_ternary_psn": {"enabled": True}},
        )

        self.assertIs(loaded, model)


if __name__ == "__main__":
    unittest.main()
