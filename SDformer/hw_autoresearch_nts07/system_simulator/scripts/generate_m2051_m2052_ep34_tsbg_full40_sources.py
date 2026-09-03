#!/usr/bin/env python3
"""Derive the full-40-sample M2051 fixture and M2052 VCS campaign sources.

The derivation is intentionally mechanical from the committed M2048/M2050
sources.  It changes only the predeclared sample cohort and corresponding
cardinalities/names; RTL, ports, cache, backpressure, token regions, and layer
inventory remain byte-identical inputs to the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"anchor cardinality drift: {old!r}")
    return text.replace(old, new)


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o775)


def derive_builder() -> Path:
    source = HW / "system_simulator/scripts/build_m2048_ep34_tsbg_multilayer_token_fixture.py"
    output = HW / "system_simulator/scripts/build_m2051_ep34_tsbg_full40_fixture.py"
    text = source.read_text(encoding="utf-8")
    text = text.replace("m2048", "m2051").replace("M2048", "M2051")
    text = text.replace("multilayer_token_s192", "full40_s1920")
    text = replace_once(
        text,
        "The fixed selection uses the first captured sample from each of four DSEC\n"
        "sequences, every FC1/FC2 layer supported by the existing G48 frontend, and the\n",
        "The fixed selection uses all forty captured samples from four DSEC sequences,\n"
        "every FC1/FC2 layer supported by the existing G48 frontend, and the\n",
    )
    text = replace_once(text, "TARGET_SAMPLES = (0, 10, 20, 30)",
                        "TARGET_SAMPLES = tuple(range(40))")
    text = replace_once(
        text,
        "    need(all(int(row[\"sequence_sample_id\"]) == 0 for row in samples),\n"
        "         \"fixed first-sequence-sample rule drift\")\n"
        "    need(len({row[\"sequence\"] for row in samples}) == 4,\n"
        "         \"four-sequence selection drift\")\n",
        "    need([int(row[\"global_sample_id\"]) for row in samples] == list(range(40)),\n"
        "         \"full capture sample order drift\")\n"
        "    sequence_counts = {}\n"
        "    for row in samples:\n"
        "        sequence_counts[row[\"sequence\"]] = sequence_counts.get(row[\"sequence\"], 0) + 1\n"
        "    need(len(sequence_counts) == 4 and set(sequence_counts.values()) == {10},\n"
        "         \"four-sequence ten-sample cohort drift\")\n",
    )
    text = replace_once(
        text,
        "    need(len(rows) == 192 and len(fixture_words) == 192 * CONTEXTS * MAX_GROUPS,\n",
        "    need(len(rows) == 1920 and len(fixture_words) == 1920 * CONTEXTS * MAX_GROUPS,\n",
    )
    text = replace_once(
        text,
        '            "first captured sample in each of four sequences; all FC1/FC2 layers "\n',
        '            "all forty captured samples in four sequences; all FC1/FC2 layers "\n',
    )
    text = replace_once(
        text,
        '            "workloads": 192, "sequences": 4, "samples": 4,\n',
        '            "workloads": 1920, "sequences": 4, "samples": 40,\n',
    )
    write(output, text, executable=True)
    return output


def derive_tb() -> Path:
    source = HW / "tb_m2018/tb_m2048_ep34_tsbg_multilayer_token_cycle.sv"
    output = HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv"
    text = source.read_text(encoding="utf-8")
    text = text.replace("tb_m2048_ep34_tsbg_multilayer_token_cycle",
                        "tb_m2051_ep34_tsbg_full40_cycle")
    text = text.replace("m2048_ep34_tsbg_multilayer_token_s192",
                        "m2051_ep34_tsbg_full40_s1920")
    text = text.replace("PASS_M2048_EP34_TSBG_MULTILAYER_TOKEN_CYCLE",
                        "PASS_M2051_EP34_TSBG_FULL40_CYCLE")
    text = text.replace("M2048", "M2051")
    text = replace_once(text, "localparam int WORKLOADS=192;",
                        "localparam int WORKLOADS=1920;")
    text = replace_once(text, "logic [31:0] fixture_word [0:36863];",
                        "logic [31:0] fixture_word [0:368639];")
    text = replace_once(text, "logic [447:0] stats_word [0:191];",
                        "logic [447:0] stats_word [0:1919];")
    text = text.replace("outside 0..191", "outside 0..1919")
    write(output, text)
    return output


def derive_parser() -> Path:
    source = HW / "system_simulator/scripts/parse_m2049_ep34_tsbg_multilayer_token_vcs.py"
    output = HW / "system_simulator/scripts/parse_m2052_ep34_tsbg_full40_vcs.py"
    text = source.read_text(encoding="utf-8")
    text = text.replace("M2050 ep34 multilayer/token", "M2052 ep34 full-40-sample")
    text = text.replace("m2048_ep34_tsbg_multilayer_token_s192",
                        "m2051_ep34_tsbg_full40_s1920")
    text = text.replace("tb_m2048_ep34_tsbg_multilayer_token_cycle",
                        "tb_m2051_ep34_tsbg_full40_cycle")
    text = text.replace("iscas_m2048_ep34_tsbg_multilayer_token_cycle_vcs.f",
                        "iscas_m2051_ep34_tsbg_full40_cycle_vcs.f")
    text = text.replace("PASS_M2048_EP34_TSBG_MULTILAYER_TOKEN_CYCLE",
                        "PASS_M2051_EP34_TSBG_FULL40_CYCLE")
    text = replace_once(text, '"samples": 4, "sequences": 4, "sources_per_group": 16,',
                        '"samples": 40, "sequences": 4, "sources_per_group": 16,')
    text = replace_once(text, '"supported_layers": 16, "workloads": 192,',
                        '"supported_layers": 16, "workloads": 1920,')
    text = replace_once(text, 'assert len(fixture["rows"]) == 192',
                        'assert len(fixture["rows"]) == 1920')
    text = replace_once(text, "for slot in range(192)]", "for slot in range(1920)]")
    text = text.replace("m2050_ep34_tsbg_multilayer_token_vcs_result_r1_v1",
                        "m2052_ep34_tsbg_full40_vcs_result_r1_v1")
    text = replace_once(text, '"samples": [0, 10, 20, 30],',
                        '"samples": list(range(40)),')
    text = replace_once(text, '"workloads": 192,', '"workloads": 1920,')
    write(output, text, executable=True)
    return output


def derive_filelist(tb: Path) -> Path:
    source = HW / "dc_handoff/filelists/iscas_m2048_ep34_tsbg_multilayer_token_cycle_vcs.f"
    output = HW / "dc_handoff/filelists/iscas_m2051_ep34_tsbg_full40_cycle_vcs.f"
    text = source.read_text(encoding="utf-8")
    old = "hw_autoresearch_nts07/tb_m2018/tb_m2048_ep34_tsbg_multilayer_token_cycle.sv"
    new = "hw_autoresearch_nts07/tb_m2018/" + tb.name
    text = replace_once(text, old, new)
    write(output, text)
    return output


def derive_runner(tb: Path, parser: Path, filelist: Path) -> Path:
    source = HW / "dc_handoff/scripts/run_m2050_m2048_ep34_tsbg_multilayer_token_vcs_one_shot.sh"
    output = HW / "dc_handoff/scripts/run_m2052_m2051_ep34_tsbg_full40_vcs_one_shot.sh"
    text = source.read_text(encoding="utf-8")
    text = text.replace("iscas_m2048_ep34_tsbg_multilayer_token_cycle_vcs.f",
                        filelist.name)
    text = text.replace("tb_m2048_ep34_tsbg_multilayer_token_cycle.sv", tb.name)
    text = text.replace("m2048_ep34_tsbg_multilayer_token_s192",
                        "m2051_ep34_tsbg_full40_s1920")
    text = text.replace("parse_m2049_ep34_tsbg_multilayer_token_vcs.py", parser.name)
    text = text.replace("tb_m2048_ep34_tsbg_multilayer_token_cycle",
                        "tb_m2051_ep34_tsbg_full40_cycle")
    text = text.replace("m2050_m2048_ep34_tsbg_multilayer_token_vcs",
                        "m2052_m2051_ep34_tsbg_full40_vcs")
    text = text.replace("M2050", "M2052")
    text = text.replace("EP34_TSBG_MULTILAYER_TOKEN", "EP34_TSBG_FULL40")
    text = replace_once(text, "simv_runs=192", "simv_runs=1920")
    text = replace_once(text, "/usr/bin/seq 0 191 |", "/usr/bin/seq 0 1919 |")
    text = replace_once(text, "for slot in $(/usr/bin/seq 0 191); do",
                        "for slot in $(/usr/bin/seq 0 1919); do")
    pins = {
        "500b6b19efd95bb93beeb5ecaf78ee88c3ba5544e3998ccd426e870db8836b62": sha256(tb),
        "2ff386a999979916f98876029a83d48558445fa1e32421a2387c6555d4300235":
            sha256(HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"),
        "1858c94b1fc411e691152f848f8dd3a5b5955001828236cba902d04b9639014b":
            sha256(HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh"),
        "940f243fa218a6154887f129e622e7a219f96f97b25744b93a68d2ef60532900":
            sha256(HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"),
        "dbbb0acfb9ee80a813c4256e6b8abe431df433c94419ca3f7be96a872d6908b5": sha256(filelist),
        "bd951471b272d4ddcb6dfa61904003e94a59576265e17b06ff687bf84052886b": sha256(parser),
    }
    for old, new in pins.items():
        text = replace_once(text, old, new)
    write(output, text, executable=True)
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", action="store_true")
    args = ap.parse_args()
    builder = derive_builder()
    print(f"builder={builder} sha256={sha256(builder)}")
    if args.finish:
        required = [
            HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh",
            HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920_stats.memh",
            HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json",
        ]
        if not all(path.is_file() for path in required):
            raise RuntimeError("run the derived M2051 builder before --finish")
        tb = derive_tb()
        parser = derive_parser()
        filelist = derive_filelist(tb)
        runner = derive_runner(tb, parser, filelist)
        for path in (tb, parser, filelist, runner):
            print(f"source={path} sha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
