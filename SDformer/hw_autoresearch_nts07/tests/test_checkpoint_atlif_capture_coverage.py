from __future__ import annotations

import unittest

import torch
from torch import nn

from hw_autoresearch_nts07.scripts.generate_checkpoint_atlif_dptme_vectors import Capture


class ATLIFTernaryPSN(nn.Module):
    def __init__(self, temporal: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(temporal), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(temporal), requires_grad=False)
        self.thresh = nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_mode = "binary"
        self.threshold_mode = "official_atlif"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.bias[:, None] + torch.matmul(self.weight, value)
        return hidden.ge(self.thresh.reshape(1, 1)).to(value.dtype)


class CoverageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, ATLIFTernaryPSN, bool]] = []
        for index in range(45):
            self.entries.append((f"live_t10_{index}.spiking_neuron", ATLIFTernaryPSN(10), True))
        for index in range(36):
            self.entries.append((f"live_t2_{index}.spiking_neuron", ATLIFTernaryPSN(2), True))
        for index in range(12):
            self.entries.append(
                (
                    f"stage{index}.attn.attn_sn.spiking_neuron",
                    ATLIFTernaryPSN(2),
                    True,
                )
            )
        for index in range(12):
            self.entries.append((f"uncalled_{index}.spiking_neuron", ATLIFTernaryPSN(2), False))
        self.modules_list = nn.ModuleList(module for _, module, _ in self.entries)

    def named_modules(self, *args, **kwargs):
        yield "", self
        for name, module, _ in self.entries:
            yield name, module

    def forward(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for _, module, called in self.entries:
            if not called:
                continue
            temporal = module.weight.shape[0]
            columns = 128 if temporal == 10 else 160
            module(torch.randn(temporal, columns, generator=generator))


class CheckpointAtlifCaptureCoverageTest(unittest.TestCase):
    def test_capture_records_installed_called_dead_and_replayed_sets(self) -> None:
        model = CoverageModel()
        capture = Capture()
        capture.attach(model)
        try:
            model()
        finally:
            capture.close()
        replayed = {row["name"] for row in capture.rows}
        self.assertEqual(len(capture.installed_names), 105)
        self.assertEqual(len(capture.called_names), 93)
        self.assertEqual(len(capture.dead_called_names), 12)
        self.assertEqual(len(replayed), 81)
        self.assertEqual(capture.called_names - capture.dead_called_names, replayed)


if __name__ == "__main__":
    unittest.main()
