"""Generate H21 SpikeVideoFormer Hamming attention configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(name: str, config: dict) -> None:
    with (CONFIG_DIR / name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def common(base: dict, experiment: str) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg.setdefault("runtime", {})["max_train_steps"] = 120
    cfg["runtime"]["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["pin_memory"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    return cfg


def main() -> None:
    base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_guard120.yml")

    cfg = common(base, "h21a_spikevideo_hamming_binary_direct_guard120")
    cfg["bsa_attention"]["mode"] = "hamming_binary_direct"
    cfg["bsa_attention"]["value_mode"] = "threshold"
    cfg["note"] = "H21a direct SpikeVideoFormer Hamming attention: binary Q/K mapped to {-1,+1}, K reused as V."
    write_config("h21a_spikevideo_hamming_binary_direct_guard120.yml", cfg)

    cfg = common(base, "h21b_hamming_ternary_active_direct_guard120")
    cfg["bsa_attention"]["mode"] = "hamming_ternary_active_direct"
    cfg["bsa_attention"]["value_mode"] = "threshold"
    cfg["note"] = "H21b direct ternary-active Hamming attention: silence stays 0, active signs drive Hamming path."
    write_config("h21b_hamming_ternary_active_direct_guard120.yml", cfg)

    cfg = common(base, "h21c_hamming_binary_signv_guard120")
    cfg["bsa_attention"]["mode"] = "hamming_binary_direct"
    cfg["bsa_attention"]["value_mode"] = "sign"
    cfg["note"] = "H21c direct SpikeVideoFormer Hamming attention with sign-only K as V proxy."
    write_config("h21c_hamming_binary_signv_guard120.yml", cfg)


if __name__ == "__main__":
    main()
