#!/opt/anaconda3/bin/python
"""Source-only attacks for M2057. This test never invokes simv, VCS, or lmutil."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / (
    "system_simulator/scripts/"
    "parse_m2057_m2053_ep34_tsbg_full40_missing3_successor.py"
)


def load_source():
    spec = importlib.util.spec_from_file_location("m2057_source", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_log(row: dict, base_cycles: int, tsbg_cycles: int) -> str:
    live = row["live_rows"]
    return (
        "sva_tsbg.cp_bridge_negative, 1 attempts, 1 match\n"
        "sva_tsbg.cp_stale_attack, 1 attempts, 1 match\n"
        "sva_tsbg.cp_reset_recovery_minimum_one_cycle, 1 attempts, 1 match\n"
        "PASS_M2051_EP34_TSBG_FULL40_CYCLE "
        f"workload_slot={row['slot']} sample_id={row['sample_id']} "
        f"layer={row['layer_id']} is_fc2={row['is_fc2']} "
        f"token_start={row['token_start']} source_groups={row['source_groups']} "
        f"physical_groups=48 rows={live} issues={row['issues']} "
        f"products={row['products']} commits=24 base_cycles={base_cycles} "
        f"tsbg_cycles={tsbg_cycles} bundles_base={row['base_misses'] * 12} "
        f"bundles_tsbg={row['tsbg_misses'] * 12} "
        f"scalar_base={row['base_misses'] * 96} "
        f"scalar_tsbg={row['tsbg_misses'] * 96} stale=1 "
        f"retired_replay={int(live != 0)} replay_accept=0 reset=2 recovery=1 "
        "real_weights=false system_speedup=false\n"
    )


def expect_reject(fn, label: str) -> None:
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError(f"attack unexpectedly accepted: {label}")


def main() -> None:
    source = load_source()
    _, fixture, old_rows = source.audit_old()
    assert len(old_rows) == 1917
    with tempfile.TemporaryDirectory(
        prefix=".m2057_source_test.", dir=HW / "results"
    ) as tmp:
        tmp_path = Path(tmp)
        new = tmp_path / "new"
        merged = tmp_path / "merged"
        new.mkdir()
        merged.mkdir()
        (new / "M2057_RUN_COMMANDS.txt").write_text(source.expected_commands())
        for slot in source.MISSING:
            (new / f"sim_slot{slot}.log").write_text(
                synthetic_log(fixture["rows"][slot], 1000, 500)
            )
        shutil.copy2(source.OLD_RAW / "vcs_compile.log", merged / "vcs_compile.log")
        shutil.copy2(new / "M2057_RUN_COMMANDS.txt", merged / "M2057_RUN_COMMANDS.txt")
        for slot in range(1920):
            src = (new if slot in source.MISSING else source.OLD_RAW) / f"sim_slot{slot}.log"
            if slot in source.MISSING:
                shutil.copy2(src, merged / f"sim_slot{slot}.log")
            else:
                os.link(src, merged / f"sim_slot{slot}.log")
        output = merged / "result.json"
        source.merge(new, merged, output)
        result = json.loads(output.read_text())
        assert len(result["rows"]) == 1920
        assert result["cross_attempt_boundary"]["parent_logs_inherited"] == 1917
        assert result["cross_attempt_boundary"]["successor_logs"] == 3
        assert result["cross_attempt_boundary"]["successor_runtime_switch"] == "-no_save"
        assert result["claim_boundary"]["paper_admitted"] is False

        command = new / "M2057_RUN_COMMANDS.txt"
        valid_command = command.read_text()
        command.write_text(valid_command.replace("-no_save ", "", 1))
        expect_reject(lambda: source.merge(new, merged, output), "missing -no_save receipt")
        command.write_text(valid_command)

        attacked = new / "sim_slot86.log"
        valid_log = attacked.read_text()
        attacked.write_text(valid_log + "ASLR will be switched off and simv re-executed\n")
        expect_reject(lambda: source.merge(new, merged, output), "ASLR re-exec successor")
        attacked.write_text(valid_log)

        merged_log = merged / "sim_slot86.log"
        merged_log.write_text(valid_log + "foreign mutation\n")
        expect_reject(lambda: source.merge(new, merged, output), "merged-source mutation")

    print("PASS_M2057_SOURCE_TEST valid_merge=1 attacks=3 old_logs=1920 new_logs=3")


if __name__ == "__main__":
    main()
