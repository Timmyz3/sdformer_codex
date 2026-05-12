import unittest

import torch
from torch import nn

from tools.profile_sops import MetricAccumulator, SpikeActivityProfiler, estimate_sops, flatten_numeric_tree, parse_human_number


class ProfileSopsTest(unittest.TestCase):
    def test_flatten_numeric_tree_sums_nested_records(self):
        record = {"a": 2, "b": {"c": 3.5, "d": {"e": 4}}, "skip": "x"}

        self.assertEqual(flatten_numeric_tree(record), 9.5)

    def test_parse_human_number_accepts_scaled_suffixes(self):
        self.assertEqual(parse_human_number("42.63G"), 42.63e9)
        self.assertEqual(parse_human_number("1.5t"), 1.5e12)
        self.assertEqual(parse_human_number("2500"), 2500.0)

    def test_estimate_sops_multiplies_dense_ops_by_firing_rate(self):
        self.assertEqual(estimate_sops(1000.0, 0.25), 250.0)

    def test_spike_activity_profiler_records_spiking_wrapper_outputs(self):
        class Spiking_neuron(nn.Module):
            def forward(self, x):
                return (x > 0).float()

        model = nn.Sequential(Spiking_neuron())
        profiler = SpikeActivityProfiler(model, module_name_patterns=("Spiking_neuron",))
        profiler.attach()
        try:
            _ = model(torch.tensor([[1.0, -1.0], [-1.0, -1.0]]))
        finally:
            profiler.close()

        summary = profiler.summary()
        self.assertEqual(summary["total_spikes"], 1)
        self.assertEqual(summary["total_elements"], 4)
        self.assertEqual(summary["global_firing_rate"], 0.25)
        self.assertEqual(len(profiler.layer_rows()), 1)

    def test_metric_accumulator_averages_aee_and_auxiliary_rates(self):
        acc = MetricAccumulator(["AEE", "AAE"])
        acc.update_aee(
            (
                torch.tensor([1.0, 3.0]),
                torch.tensor([0.1, 0.3]),
                torch.tensor([0.2, 0.4]),
                torch.tensor([0.3, 0.5]),
                torch.tensor([0.4, 0.6]),
            )
        )
        acc.update_scalar("AAE", torch.tensor(10.0), count=2)

        summary = acc.summary()
        self.assertEqual(summary["AEE"], 2.0)
        self.assertAlmostEqual(summary["AEE_PE1"], 0.2)
        self.assertAlmostEqual(summary["AEE_PE2"], 0.3)
        self.assertAlmostEqual(summary["AEE_PE3"], 0.4)
        self.assertAlmostEqual(summary["AEE_outliers"], 0.5)
        self.assertEqual(summary["AAE"], 5.0)


if __name__ == "__main__":
    unittest.main()
