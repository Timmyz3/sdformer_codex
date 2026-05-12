import unittest
from pathlib import Path
import sys

import torch
import yaml


EXPERIMENTS = {
    "E0_psn_baseline": "psn",
    "E1_exp_sn": "exp_sn",
    "E2_exp_atlif": "exp_atlif",
    "E3_exp_lmh": "exp_lmh",
    "E4_exp_tslif": "exp_tslif",
    "E5_exp_tsn": "exp_tsn",
    "F1_fused_adaptive_psn": "fused_adaptive_psn",
    "F2_fused_lmh_atlif": "fused_lmh_atlif",
    "F3_fused_adaptive_tslif": "fused_adaptive_tslif",
    "F4_fused_lmh_tslif": "fused_lmh_tslif",
    "F5_fused_signed_hybrid": "fused_signed_hybrid",
}


class NeuronExperimentScaffoldTest(unittest.TestCase):
    def test_all_experiments_have_entrypoints_configs_and_overlay(self):
        root = Path("neuron_experiments")

        for exp_id, neuron_type in EXPERIMENTS.items():
            with self.subTest(exp_id=exp_id):
                exp = root / exp_id
                self.assertTrue((exp / "README.md").is_file())
                self.assertTrue((exp / "entrypoints" / "train.py").is_file())
                self.assertTrue((exp / "entrypoints" / "eval.py").is_file())
                self.assertTrue((exp / "configs" / "smoke.yml").is_file())
                self.assertTrue((exp / "configs" / "subset.yml").is_file())
                self.assertTrue((exp / "configs" / "full.yml").is_file())
                self.assertTrue((exp / "results" / "metrics.md").is_file())
                self.assertTrue((exp / "results" / "run_commands.md").is_file())

                with (exp / "configs" / "smoke.yml").open() as f:
                    config = yaml.safe_load(f)
                self.assertEqual(config["spiking_neuron"]["neuron_type"], neuron_type)
                self.assertEqual(config["runtime"]["snn_backend"], "torch")
                overrides = config["data"]["sequence_list_overrides"]
                self.assertTrue(Path(overrides["train"]).is_file())
                self.assertTrue(Path(overrides["valid"]).is_file())
                self.assertEqual(config["test"]["sample"], 1)

                if neuron_type == "psn":
                    continue

                self.assertTrue((exp / "overlay" / "models" / "__init__.py").is_file())
                self.assertTrue(
                    (exp / "overlay" / "models" / "STSwinNet_SNN" / "Spiking_modules.py").is_file()
                )
                self.assertTrue(
                    (
                        exp
                        / "overlay"
                        / "models"
                        / "STSwinNet_SNN"
                        / "experimental_neurons"
                        / "factory.py"
                    ).is_file()
                )

    def test_overlay_factory_instantiates_all_experimental_neurons(self):
        root = Path.cwd()
        overlay = root / "neuron_experiments" / "E1_exp_sn" / "overlay"
        baseline = root / "third_party" / "SDformerFlow"
        original_path = list(sys.path)
        try:
            sys.path[:0] = [str(overlay), str(baseline), str(root)]
            from models.STSwinNet_SNN.experimental_neurons.factory import build_experimental_neuron

            x = torch.randn(4, 2, 3, 5, 5, requires_grad=True)
            for neuron_type in EXPERIMENTS.values():
                if neuron_type == "psn":
                    continue
                with self.subTest(neuron_type=neuron_type):
                    node = build_experimental_neuron(neuron_type, num_steps=4, v_th=1.0)
                    y = node(x)
                    self.assertEqual(y.shape, x.shape)
                    y.mean().backward(retain_graph=True)
                    self.assertIsNotNone(x.grad)
                    x.grad.zero_()
        finally:
            sys.path = original_path
            for module_name in list(sys.modules):
                if module_name == "models" or module_name.startswith("models.STSwinNet_SNN.experimental_neurons"):
                    sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
