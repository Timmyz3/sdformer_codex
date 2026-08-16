from pathlib import Path

import pytest

from scripts.report_local5_stencil_compiler_integration import dilation_pass, ledger


def test_dilation_pass_rejects_error_before_pass(tmp_path: Path) -> None:
    log = tmp_path / "false_pass.log"
    log.write_text(
        "ERROR: bank collision\n"
        "PASS dilation_miter d=2 seed=17 retire=375 stalls=34 pending=2\n"
    )

    with pytest.raises(ValueError, match="bad marker"):
        dilation_pass(log)


def test_dilation_pass_accepts_clean_log(tmp_path: Path) -> None:
    log = tmp_path / "clean.log"
    log.write_text(
        "PASS dilation_miter d=2 seed=99 retire=4 stalls=1 pending=0\n"
    )

    assert dilation_pass(log) == {
        "dilation": 2,
        "seed": 99,
        "retire": 4,
        "stalls": 1,
        "pending": 0,
    }


def test_ledger_rejects_missing_readmem_vectors(tmp_path: Path) -> None:
    log = tmp_path / "missing_vectors.log"
    log.write_text(
        "%Warning: input_q.memh:0: $readmem file not found\n"
        "PASS Local5 score-to-projection backend=0 latency=1 "
        "groups=100 total_cycles=155791\n"
    )

    with pytest.raises(ValueError, match="bad marker"):
        ledger(log)
