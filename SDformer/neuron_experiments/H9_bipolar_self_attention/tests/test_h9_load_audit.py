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
        self.assertEqual(loaded._h9_load_audit["checkpoint_overlay_keys"], 1)
        self.assertEqual(loaded._h9_load_audit["missing_count"], 0)
        self.assertEqual(loaded._h9_load_audit["unexpected_count"], 0)

    def test_v1_remap_applies_interpolated_state_dict(self):
        from models.STSwinNet_SNN.h9_load_audit import load_checkpoint_with_h9_audit

        source = DummyModel()
        with torch.no_grad():
            source.linear.weight.fill_(3.25)
            source.sn_q.spiking_neuron.thresh.fill_(1.75)
        checkpoint = self._save(source.state_dict())

        target = DummyModel()
        with torch.no_grad():
            target.linear.weight.zero_()
            target.sn_q.spiking_neuron.thresh.zero_()
        loaded = load_checkpoint_with_h9_audit(
            checkpoint,
            target,
            torch.device("cpu"),
            config={"atlif_ternary_psn": {"enabled": True}},
            remap="v1",
        )

        self.assertIs(loaded, target)
        self.assertTrue(torch.equal(target.linear.weight, source.linear.weight))
        self.assertTrue(
            torch.equal(
                target.sn_q.spiking_neuron.thresh,
                source.sn_q.spiking_neuron.thresh,
            )
        )

    def test_lc4_coefficients_are_overlay_owned(self):
        from models.STSwinNet_SNN.h9_load_audit import is_h9_overlay_key

        self.assertTrue(
            is_h9_overlay_key(
                "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0."
                "attn._h9_lc4_coefficients"
            )
        )

    def test_cf10_beta_is_overlay_owned_and_uses_new_module_lr(self):
        from models.STSwinNet_SNN.h28_optimizer import build_optimizer
        from models.STSwinNet_SNN.h9_load_audit import is_h9_overlay_key

        key = (
            "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0."
            "attn._h9_cf10_beta"
        )
        self.assertTrue(is_h9_overlay_key(key))

        class CF10Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.register_parameter("_h9_cf10_beta", nn.Parameter(torch.zeros(3, 2)))
                self.backbone = nn.Linear(2, 2, bias=False)

        optimizer = build_optimizer(
            CF10Model(),
            {
                "optimizer": {
                    "name": "AdamW",
                    "lr": 2.0e-5,
                    "wd": 0.01,
                    "param_groups": {
                        "enabled": True,
                        "backbone_lr": 2.0e-6,
                        "new_module_lr": 5.0e-5,
                    },
                }
            },
        )
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertIn("new_module", groups)
        self.assertEqual(groups["new_module"]["lr"], 5.0e-5)


if __name__ == "__main__":
    unittest.main()
