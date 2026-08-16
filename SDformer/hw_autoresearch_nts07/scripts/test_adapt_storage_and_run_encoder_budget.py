#!/usr/bin/env python3
"""unit tests for storage schema adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adapt_storage_and_run_encoder_budget import (
    classify_storage_schema,
    resolve_storage,
)


class TestSchema(unittest.TestCase):
    def test_classify_contract(self) -> None:
        data = {
            "models": {
                "H67": {
                    "atlif_execution_graph": {"live_temporal_macs_per_frame": 1},
                    "activation_evidence": {"long_skip_elements_s0_s2": 1},
                }
            }
        }
        self.assertEqual(classify_storage_schema(data), "encoder_storage_contract")

    def test_classify_ablation(self) -> None:
        data = {"状态": "通过", "结果": [{"设计": "H67"}]}
        self.assertEqual(classify_storage_schema(data), "storage_ablation_yosys")

    def test_redirect_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ablation = root / "ablation.json"
            contract = root / "contract.json"
            ablation.write_text(
                json.dumps({"状态": "通过", "结果": []}), encoding="utf-8"
            )
            contract.write_text(
                json.dumps(
                    {
                        "models": {
                            "H67": {
                                "atlif_execution_graph": {
                                    "live_temporal_macs_per_frame": 10,
                                    "live_output_elements_per_frame": 20,
                                },
                                "activation_evidence": {
                                    "long_skip_elements_s0_s2": 30
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            data, path, label, notes = resolve_storage(
                ablation, contract_fallback=contract
            )
            self.assertEqual(label, "redirected_ablation_to_contract")
            self.assertEqual(path, contract)
            self.assertIn("atlif_execution_graph", data["models"]["H67"])
            self.assertTrue(any("重定向" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
