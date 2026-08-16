from __future__ import annotations

import copy
import hashlib
import json
import unittest

from hw_autoresearch_nts07.scripts.report_checkpoint_atlif_dptme_rtl import (
    validate_commands,
    validate_site_coverage,
)


def coverage_row(names: set[str]) -> dict:
    values = sorted(names)
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()
    return {"count": len(values), "names": values, "sha256": hashlib.sha256(payload).hexdigest()}


def make_manifest() -> dict:
    commands = []
    for index in range(81):
        temporal = 10 if index < 45 else 2
        commands.append(
            {
                "tag": index + 1,
                "name": f"site_{index:02d}",
                "scenario": "mixed_ordinary_near_threshold_max_amplitude",
                "scenario_lane_counts": (
                    {"ordinary": 10, "near_threshold": 10, "max_amplitude": 12}
                    if temporal == 10
                    else {"ordinary": 53, "near_threshold": 53, "max_amplitude": 54}
                ),
                "temporal_steps": temporal,
                "x_scale": 0.125,
                "weight_scale": 0.0625,
                "accumulator_scale": 0.0078125,
                "captured_events": 320,
                "fixed_vs_float_event_mismatches": 1,
                "model_reference_mismatches": 0,
                "hidden_min": -1024,
                "hidden_max": 2048,
                "clip_counts": {"input": 0, "weight": 0, "bias": 0, "threshold": 0},
                "accumulator_overflow_count": 0,
                "output_contract": "one_bit_event_plus_checkpoint_static_threshold_scale",
            }
        )
    replayed = {row["name"] for row in commands}
    dead = {f"stage.block{index}.attn.attn_sn.spiking_neuron" for index in range(12)}
    called = replayed | dead
    installed = called | {f"uncalled_site_{index}" for index in range(12)}
    return {
        "commands": commands,
        "site_coverage": {
            "installed": coverage_row(installed),
            "called": coverage_row(called),
            "dead_called": coverage_row(dead),
            "replayed": coverage_row(replayed),
        },
        "summary": {
            "captured_events": 25_920,
            "fixed_vs_float_event_mismatches": 81,
            "fixed_vs_float_event_mismatch_ratio": 81 / 25_920,
            "model_reference_mismatches": 0,
        },
    }


class CheckpointAtlifDptmeManifestTest(unittest.TestCase):
    def test_accepts_complete_manifest(self) -> None:
        manifest = make_manifest()
        validate_commands(manifest)
        validate_site_coverage(manifest)

    def test_rejects_site_coverage_sha_drift(self) -> None:
        manifest = make_manifest()
        manifest["site_coverage"]["called"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "coverage SHA"):
            validate_site_coverage(manifest)

    def test_rejects_wrong_dead_site_class(self) -> None:
        manifest = make_manifest()
        dead = set(manifest["site_coverage"]["dead_called"]["names"])
        dead.pop()
        dead.add("not_an_attention_dead_site")
        manifest["site_coverage"]["dead_called"] = coverage_row(dead)
        with self.assertRaisesRegex(RuntimeError, "set relation|site coverage"):
            validate_site_coverage(manifest)

    def test_rejects_duplicate_site(self) -> None:
        manifest = make_manifest()
        manifest["commands"][1]["name"] = manifest["commands"][0]["name"]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_commands(manifest)

    def test_rejects_clipping(self) -> None:
        manifest = make_manifest()
        manifest["commands"][7]["clip_counts"]["input"] = 1
        with self.assertRaisesRegex(RuntimeError, "clipping/overflow"):
            validate_commands(manifest)

    def test_rejects_summary_mismatch(self) -> None:
        manifest = copy.deepcopy(make_manifest())
        manifest["summary"]["fixed_vs_float_event_mismatches"] = 0
        with self.assertRaisesRegex(RuntimeError, "totals"):
            validate_commands(manifest)


if __name__ == "__main__":
    unittest.main()
