#!/usr/bin/env python3
"""CPU unit tests for the M29 checkpoint-migratable ATLIF factorization."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


OVERLAY = Path(__file__).resolve().parents[1] / "overlay"
if str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))

from models.STSwinNet_SNN.atlif_ternary_psn import (  # noqa: E402
    ATLIFTernaryPSN,
    apply_trainable_mode,
    atlif_temporal_factor_diagnostics,
    atlif_ternary_summary,
    materialize_temporal_factor_state_dict,
)
from models.STSwinNet_SNN.h9_load_audit import is_h9_overlay_key  # noqa: E402
from neuron_experiments.H9_bipolar_self_attention.entrypoints.make_m29_h67_rank3_factor_config import (  # noqa: E402
    build_config,
)


class _BasePSN(nn.Module):
    def __init__(self, temporal_steps: int) -> None:
        super().__init__()
        self.T = temporal_steps
        self.weight = nn.Parameter(
            torch.arange(temporal_steps * temporal_steps, dtype=torch.float32).reshape(
                temporal_steps, temporal_steps
            )
            / 17.0
        )
        self.bias = nn.Parameter(
            torch.linspace(-0.25, 0.25, temporal_steps).reshape(temporal_steps, 1)
        )


class _RankOneBasePSN(_BasePSN):
    def __init__(self, temporal_steps: int) -> None:
        super().__init__(temporal_steps)
        with torch.no_grad():
            left = torch.arange(1, temporal_steps + 1, dtype=torch.float32)
            right = torch.arange(temporal_steps, 0, -1, dtype=torch.float32)
            self.weight.copy_(left.reshape(-1, 1) * right.reshape(1, -1))


class M29ATLIFTemporalFactorizationTest(unittest.TestCase):
    def test_rank_zero_keeps_dense_forward_and_state_schema(self) -> None:
        base = _BasePSN(4)
        module = ATLIFTernaryPSN(T=4, base_psn=base, temporal_factor_rank=0)
        x = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        )
        expected_h = torch.addmm(module.bias, module.weight, x)
        expected = (expected_h >= module.thresh).float() * module.thresh
        self.assertTrue(torch.equal(module(x), expected))
        self.assertNotIn("temporal_factor_left", module.state_dict())
        self.assertTrue(module.weight.requires_grad)

    def test_pre_m29_pickled_module_can_still_extract_dense_state(self) -> None:
        module = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=0
        )
        for name in (
            "temporal_factor_requested_rank",
            "temporal_factor_rank",
            "temporal_factor_init",
            "temporal_factor_load_source",
        ):
            delattr(module, name)
        state = module.state_dict()
        self.assertIn("weight", state)
        self.assertNotIn("temporal_factor_left", state)

    def test_balanced_svd_reconstructs_exact_low_rank_weight(self) -> None:
        module = ATLIFTernaryPSN(
            T=4, base_psn=_RankOneBasePSN(4), temporal_factor_rank=1
        )
        self.assertTrue(
            torch.allclose(module.temporal_effective_weight(), module.weight, atol=2e-6)
        )
        self.assertFalse(module.weight.requires_grad)
        self.assertEqual(tuple(module.temporal_factor_left.shape), (4, 1))
        self.assertEqual(tuple(module.temporal_factor_right.shape), (1, 4))

    def test_profitable_rank_forward_uses_only_factors_and_trains_them(self) -> None:
        module = ATLIFTernaryPSN(
            T=4,
            base_psn=_BasePSN(4),
            temporal_factor_rank=1,
            thresh=0.2,
            output_mode="binary",
        )
        x = torch.randn(4, 7, requires_grad=True)
        expected_h = torch.addmm(
            module.bias,
            module.temporal_factor_left,
            torch.mm(module.temporal_factor_right, x),
        )
        expected = (expected_h >= module.thresh).float() * module.thresh
        actual = module(x)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertIsNotNone(module.temporal_factor_left.grad)
        self.assertIsNotNone(module.temporal_factor_right.grad)
        self.assertIsNone(module.weight.grad)

    def test_dense_checkpoint_migrates_without_missing_keys(self) -> None:
        dense = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=0
        )
        factor = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        result = factor.load_state_dict(copy.deepcopy(dense.state_dict()), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(factor.temporal_factor_load_source, "balanced_svd_dense")
        singular = torch.linalg.svdvals(dense.weight.detach())
        expected_error = torch.sqrt(torch.sum(singular[1:] ** 2))
        actual_error = torch.linalg.norm(
            factor.temporal_effective_weight() - dense.weight.detach()
        )
        self.assertTrue(torch.allclose(actual_error, expected_error, atol=2e-6))

    def test_factor_checkpoint_roundtrip_is_strict_and_does_not_reinitialize(self) -> None:
        source = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        with torch.no_grad():
            source.temporal_factor_left.add_(0.125)
            source.temporal_factor_right.sub_(0.25)
        destination = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        result = destination.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(destination.temporal_factor_load_source, "checkpoint_factors")
        self.assertTrue(
            torch.equal(destination.temporal_factor_left, source.temporal_factor_left)
        )
        self.assertTrue(
            torch.equal(destination.temporal_factor_right, source.temporal_factor_right)
        )

    def test_factor_state_refreshes_dense_migration_weight_and_explicit_export(self) -> None:
        factor = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        with torch.no_grad():
            factor.temporal_factor_left.add_(0.75)
        self.assertFalse(
            torch.allclose(factor.weight, factor.temporal_effective_weight())
        )
        factor_state = factor.state_dict()
        self.assertTrue(
            torch.equal(factor_state["weight"], factor.temporal_effective_weight())
        )
        exported = materialize_temporal_factor_state_dict(factor)
        self.assertNotIn("temporal_factor_left", exported)
        self.assertNotIn("temporal_factor_right", exported)
        dense = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=0
        )
        result = dense.load_state_dict(exported, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(torch.equal(dense.weight, factor.temporal_effective_weight()))

    def test_factor_checkpoint_rejects_inconsistent_dense_migration_weight(self) -> None:
        source = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        state = source.state_dict()
        state["weight"] = state["weight"] + 1.0
        destination = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        with self.assertRaisesRegex(RuntimeError, "disagrees with L@R"):
            destination.load_state_dict(state, strict=True)

    def test_factor_keys_are_fail_close_overlay_keys(self) -> None:
        self.assertTrue(
            is_h9_overlay_key(
                "model.block.spiking_neuron.temporal_factor_left"
            )
        )
        self.assertTrue(
            is_h9_overlay_key(
                "model.block.spiking_neuron.temporal_factor_right"
            )
        )

    def test_incomplete_factor_checkpoint_fails_closed(self) -> None:
        module = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        state = dict(module.state_dict())
        del state["temporal_factor_right"]
        with self.assertRaisesRegex(RuntimeError, "incomplete ATLIF temporal factor pair"):
            module.load_state_dict(state, strict=False)

    def test_unprofitable_rank_uses_dense_fallback_and_invalid_init_fails(self) -> None:
        module = ATLIFTernaryPSN(
            T=2, base_psn=_BasePSN(2), temporal_factor_rank=3
        )
        self.assertEqual(module.temporal_factor_requested_rank, 3)
        self.assertEqual(module.temporal_factor_rank, 0)
        self.assertNotIn("temporal_factor_left", module.state_dict())
        self.assertTrue(module.weight.requires_grad)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ATLIFTernaryPSN(T=4, base_psn=_BasePSN(4), temporal_factor_rank=-1)
        with self.assertRaisesRegex(ValueError, "balanced_svd"):
            ATLIFTernaryPSN(
                T=4,
                base_psn=_BasePSN(4),
                temporal_factor_rank=1,
                temporal_factor_init="random",
            )

    def test_summary_exposes_factor_scope_without_speedup_claim(self) -> None:
        model = nn.Sequential(
            ATLIFTernaryPSN(T=4, base_psn=_BasePSN(4), temporal_factor_rank=1),
            ATLIFTernaryPSN(T=4, base_psn=_BasePSN(4), temporal_factor_rank=0),
            ATLIFTernaryPSN(T=2, base_psn=_BasePSN(2), temporal_factor_rank=3),
        )
        summary = atlif_ternary_summary(model)
        self.assertEqual(summary["temporal_factorized_modules"], 1)
        self.assertEqual(summary["temporal_factor_requested_modules"], 2)
        self.assertEqual(summary["temporal_factor_dense_fallback_modules"], 1)
        self.assertEqual(summary["temporal_factor_rank_min"], 1)
        self.assertEqual(summary["temporal_factor_rank_max"], 1)
        self.assertEqual(summary["temporal_factor_parameter_entries"], 8)
        self.assertNotIn("speedup", summary)

    def test_temporal_factor_atlif_mode_does_not_train_dense_fallback(self) -> None:
        factor = ATLIFTernaryPSN(
            T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
        )
        fallback = ATLIFTernaryPSN(
            T=2, base_psn=_BasePSN(2), temporal_factor_rank=3
        )
        model = nn.Sequential(factor, fallback)
        result = apply_trainable_mode(
            model,
            {"trainable": "temporal_factor_atlif"},
        )
        self.assertEqual(result["mode"], "temporal_factor_atlif")
        self.assertEqual(result["trainable_parameters"], 13)
        self.assertTrue(factor.temporal_factor_left.requires_grad)
        self.assertTrue(factor.temporal_factor_right.requires_grad)
        self.assertTrue(factor.bias.requires_grad)
        self.assertTrue(factor.thresh.requires_grad)
        self.assertFalse(factor.weight.requires_grad)
        self.assertFalse(fallback.weight.requires_grad)
        self.assertFalse(fallback.bias.requires_grad)
        self.assertFalse(fallback.thresh.requires_grad)

    def test_factor_diagnostics_expose_scale_condition_and_reference_error(self) -> None:
        model = nn.Sequential(
            ATLIFTernaryPSN(
                T=4, base_psn=_BasePSN(4), temporal_factor_rank=1
            )
        )
        diagnostics = atlif_temporal_factor_diagnostics(model)
        self.assertEqual(diagnostics["temporal_factorized_modules"], 1)
        self.assertGreater(diagnostics["left_absmax_max"], 0.0)
        self.assertGreater(diagnostics["right_absmax_max"], 0.0)
        self.assertGreaterEqual(
            diagnostics["left_right_norm_balance_ratio_max"], 1.0
        )
        self.assertGreaterEqual(
            diagnostics["latent_component_balance_ratio_max"], 1.0
        )
        self.assertGreaterEqual(diagnostics["effective_rank_condition_max"], 1.0)
        self.assertGreaterEqual(
            diagnostics["dense_reference_relative_error_mean"], 0.0
        )

    def test_m29_config_is_accuracy_feasibility_and_not_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"m29-test-checkpoint")
            cfg = build_config(
                {
                    "atlif_ternary_psn": {},
                    "loader": {},
                    "optimizer": {"param_groups": {}},
                    "runtime": {},
                },
                checkpoint,
            )
        self.assertEqual(cfg["atlif_ternary_psn"]["temporal_factor_rank"], 3)
        self.assertEqual(
            cfg["atlif_ternary_psn"]["trainable"],
            "temporal_factor_atlif",
        )
        self.assertEqual(cfg["loader"]["n_epochs"], 5)
        self.assertEqual(
            cfg["runtime"]["m29_scope"],
            "floating_factor_valid40_internal_screen_amp_before_int8_qat",
        )
        self.assertIn("valid40 internal", cfg["note"])
        self.assertIn("not a hardware speedup result", cfg["note"])

    def test_train_loader_and_watcher_are_factor_fail_close(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        train_source = (repo / "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('".temporal_factor_left"', train_source)
        self.assertIn("factor checkpoint rank/shape does not match config", train_source)
        watcher = (
            repo
            / "hw_autoresearch_nts07/system_handoff/scripts/run_m29_h67_rank3_after_remote_queue.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("checkpoint_epoch40.pth", watcher)
        self.assertNotIn("checkpoint_epoch4.pth", watcher)
        self.assertIn("sdformer_a800_training_global.lock", watcher)
        self.assertIn("sdformer_date_fullres_factorial_controls_20260821.lock", watcher)
        self.assertIn("--receipt", watcher)
        self.assertIn("--output \"$PREFLIGHT_REL\"", watcher)
        self.assertEqual(watcher.count("verify_frozen_inputs"), 3)
        self.assertIn(
            "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
            watcher,
        )
        self.assertIn(
            "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
            watcher,
        )
        self.assertIn("make_m29_h67_rank3_run_receipt.py", watcher)
        self.assertIn("--phase launch", watcher)
        self.assertIn("--phase postflight", watcher)
        self.assertNotIn('exec "$PYTHON_BIN" -u', watcher)


if __name__ == "__main__":
    unittest.main()
