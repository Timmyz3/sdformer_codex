"""Generate the equal-budget H60/no-motion control and AAE audit configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
H67 = GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml"
H81 = GEN / "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30.yml"
NB0 = REPO / "configs/generated/upstream_baseline_stride.yml"
NB0_AAE = GEN / "nb0_benchmark_aae_valid825.yml"
H67_AAE = GEN / "h67_motionxor_benchmark_aae_valid825.yml"
H81_AAE = GEN / "h81_nomotion_benchmark_aae_valid825.yml"
MANIFEST = GEN / "h81_equal_budget_control_manifest.json"


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def benchmark_config(config: dict, experiment: str) -> dict:
    result = deepcopy(config)
    result["experiment"] = experiment
    result["metrics"]["name"] = ["AEE", "AAE", "AAE_Benchmark"]
    result.setdefault("test", {}).update({"sample": 825, "n_valid": 1})
    return result


def main() -> int:
    h67 = load(H67)
    h81 = deepcopy(h67)
    h81["experiment"] = H81.stem
    h81["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    h81["note"] = (
        "H81 reviewer control: exact H67 full30 budget, warm start, all12 H60, and "
        "all-binary ATLIF settings, with only the Motion-XOR term disabled."
    )
    dump(H81, h81)

    dump(NB0_AAE, benchmark_config(load(NB0), NB0_AAE.stem))
    dump(H67_AAE, benchmark_config(h67, H67_AAE.stem))
    dump(H81_AAE, benchmark_config(h81, H81_AAE.stem))

    rows = {
        "control": str(H81),
        "benchmark_aae_configs": {
            "NB0": str(NB0_AAE),
            "H67": str(H67_AAE),
            "H81": str(H81_AAE),
        },
        "allowed_training_differences_vs_h67": [
            "experiment",
            "note",
            "bsa_attention.binary_motion_xor_alpha",
        ],
    }
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for path in (H81, NB0_AAE, H67_AAE, H81_AAE, MANIFEST):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
