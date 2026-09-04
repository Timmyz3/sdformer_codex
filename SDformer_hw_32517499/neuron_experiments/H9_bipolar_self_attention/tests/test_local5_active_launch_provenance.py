from __future__ import annotations

from neuron_experiments.H9_bipolar_self_attention.entrypoints.capture_local5_active_launch_provenance_20260805 import (
    option_value,
    parse_proc_stat,
)


def test_parse_proc_stat_handles_spaces_in_comm() -> None:
    fields = ["R", "42"] + ["0"] * 17 + ["123456"] + ["0"] * 5
    facts = parse_proc_stat("99 (worker with spaces) " + " ".join(fields))
    assert facts == {
        "pid": 99,
        "comm": "worker with spaces",
        "state": "R",
        "ppid": 42,
        "start_ticks": 123456,
    }


def test_option_value_supports_separate_and_equal_forms() -> None:
    argv = ["python", "train.py", "--config", "/tmp/a.yml", "--finetune=1"]
    assert option_value(argv, "--config") == "/tmp/a.yml"
    assert option_value(argv, "--finetune") == "1"
    assert option_value(argv, "--resume") is None
