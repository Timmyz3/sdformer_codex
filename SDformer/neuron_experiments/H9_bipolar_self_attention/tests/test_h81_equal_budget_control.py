from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "neuron_experiments/H9_bipolar_self_attention/configs/generated"


def load(name: str) -> dict:
    return yaml.safe_load((GEN / name).read_text(encoding="utf-8"))


def test_h81_differs_from_h67_only_by_motion_term_and_metadata():
    h67 = load("h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml")
    h81 = load("h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30.yml")

    assert h67["bsa_attention"]["binary_motion_xor_alpha"] == 0.25
    assert h81["bsa_attention"]["binary_motion_xor_alpha"] == 0.0

    normalized_h67 = deepcopy(h67)
    normalized_h81 = deepcopy(h81)
    for config in (normalized_h67, normalized_h81):
        config.pop("experiment")
        config.pop("note")
        config["bsa_attention"].pop("binary_motion_xor_alpha")

    assert normalized_h81 == normalized_h67


def test_benchmark_configs_preserve_legacy_and_official_aae():
    for name in (
        "nb0_benchmark_aae_valid825.yml",
        "h67_motionxor_benchmark_aae_valid825.yml",
        "h81_nomotion_benchmark_aae_valid825.yml",
    ):
        assert load(name)["metrics"]["name"] == ["AEE", "AAE", "AAE_Benchmark"]
