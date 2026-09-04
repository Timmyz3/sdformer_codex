from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.report_local5_real_multitile_shared_front import parse_terminal


PASS = (
    "PASS Local5 multi-tile memo=1 cycles=100 token=1350 hits=6 fallback=0 "
    "replay_records=12 partial=9 final=3 weight_cycles=1 frontend_cycles=2 "
    "readout_cycles=3 release_cycles=4 rmw_cycles=5 drain_cycles=6 "
    "scheduler_cycles=7\n"
)


def test_terminal_is_unique_and_fail_closed() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "run.log"
        path.write_text(PASS, encoding="utf-8")
        assert parse_terminal(path)["hits"] == 6
        path.write_text(PASS + "FATAL: late error\n", encoding="utf-8")
        try:
            parse_terminal(path)
        except ValueError as error:
            assert "bad marker" in str(error)
        else:
            raise AssertionError("late fatal marker was accepted")
