#!/usr/bin/env python3
"""Create the no-EDA M2053 successor after the M2052 parser source failure."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
PARSER_IN = HW / "system_simulator/scripts/parse_m2052_ep34_tsbg_full40_vcs.py"
RUNNER_IN = HW / "dc_handoff/scripts/run_m2052_m2051_ep34_tsbg_full40_vcs_one_shot.sh"
PARSER_OUT = HW / "system_simulator/scripts/parse_m2053_ep34_tsbg_full40_vcs.py"
RUNNER_OUT = HW / "dc_handoff/scripts/run_m2053_m2051_ep34_tsbg_full40_vcs_one_shot.sh"
PARSER_IN_SHA = "d4aec2ed6af755f91cba119760ff2c81ae1c8763655b2b5fb2fe6c3aed5ed5ea"
RUNNER_IN_SHA = "0727765430276ae9103626a69f2d79242f37b7103e0a3ae0e45dcb8ab406f4c9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"anchor cardinality drift: {old!r}")
    return text.replace(old, new)


def main() -> int:
    if sha256(PARSER_IN) != PARSER_IN_SHA or sha256(RUNNER_IN) != RUNNER_IN_SHA:
        raise RuntimeError("frozen M2052 source identity drift")

    parser = PARSER_IN.read_text(encoding="utf-8")
    parser = parser.replace("M2052 ep34", "M2053 ep34")
    parser = replace_once(
        parser,
        'text.count("M2048_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE") == 1',
        'text.count("M2051_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE") == 1',
    )
    parser = replace_once(
        parser,
        '"geomean_workload_speedup": math.prod(row["speedup"] for row in rows) **\n'
        '                                     (1.0 / len(rows)),',
        '"geomean_workload_speedup": math.exp(\n'
        '            sum(math.log(row["speedup"]) for row in rows) / len(rows)\n'
        '        ),',
    )
    parser = parser.replace("m2052_ep34_tsbg_full40_vcs_result_r1_v1",
                            "m2053_ep34_tsbg_full40_vcs_result_r1_v1")
    PARSER_OUT.write_text(parser, encoding="utf-8")
    PARSER_OUT.chmod(0o775)

    runner = RUNNER_IN.read_text(encoding="utf-8")
    runner = runner.replace(PARSER_IN.name, PARSER_OUT.name)
    runner = runner.replace("m2052_m2051_ep34_tsbg_full40_vcs",
                            "m2053_m2051_ep34_tsbg_full40_vcs")
    runner = runner.replace("M2052", "M2053")
    runner = replace_once(runner, PARSER_IN_SHA, sha256(PARSER_OUT))
    RUNNER_OUT.write_text(runner, encoding="utf-8")
    RUNNER_OUT.chmod(0o775)
    print(f"parser={PARSER_OUT} sha256={sha256(PARSER_OUT)}")
    print(f"runner={RUNNER_OUT} sha256={sha256(RUNNER_OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
