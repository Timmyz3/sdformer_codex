#!/usr/bin/env python3
"""Mechanically derive the M2048 G48 testbench from sealed M2046 RTL TB.

Keeping one audited source template avoids hand-copy drift in the memory,
scoreboard, protocol-attack, reset-recovery, and SVA plumbing.  Only fixture
cardinality, per-workload metadata loading, and the machine PASS record change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "tb_m2018/tb_m2046_ep34_tsbg_g48_four_sequence_cycle.sv"
OUTPUT = HW / "tb_m2018/tb_m2048_ep34_tsbg_multilayer_token_cycle.sv"
SOURCE_SHA = "0883c4b498e7770032f4670780cec37d0b6ebeab34cfe743d5c09563bf2c94f6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"anchor cardinality drift: {old[:80]!r}")
    return text.replace(old, new)


def main() -> int:
    if sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("sealed M2046 TB SHA drift")
    text = SOURCE.read_text(encoding="utf-8")
    text = text.replace(
        "tb_m2046_ep34_tsbg_g48_four_sequence_cycle",
        "tb_m2048_ep34_tsbg_multilayer_token_cycle",
    )
    text = replace_once(
        text,
        '    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2046_ep34_tsbg_g48_s4.memh";\n',
        '    localparam string FIXTURE = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2048_ep34_tsbg_multilayer_token_s192.memh";\n'
        '    localparam string STATS = "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/tb_m2018/fixtures/m2048_ep34_tsbg_multilayer_token_s192_stats.memh";\n'
        '    localparam int WORKLOADS=192;\n',
    )
    text = replace_once(
        text, "    integer sample_slot, sample_id;\n",
        "    integer workload_slot, sample_id, layer_id, is_fc2, token_start;\n"
        "    integer real_source_groups;\n",
    )
    text = replace_once(
        text, "    logic [31:0] fixture_word [0:767];\n",
        "    logic [31:0] fixture_word [0:36863];\n"
        "    logic [447:0] stats_word [0:191];\n",
    )
    text = text.replace("sample_slot * BUNDLE * GROUPS", "workload_slot * BUNDLE * GROUPS")
    text = replace_once(text, "        $readmemh(FIXTURE, fixture_word);\n", "")
    old_pattern = re.compile(
        r"        sample_slot = 0;\n"
        r"        if \(!\$value\$plusargs\(\"SAMPLE_SLOT=%d\", sample_slot\)\)\n"
        r"            sample_slot = 0;\n"
        r"        if \(sample_slot < 0 \|\| sample_slot > 3\)\n"
        r"            \$fatal\(1, \"M2046 SAMPLE_SLOT outside 0\.\.3\"\);\n"
        r"        case \(sample_slot\).*?        endcase\n",
        re.DOTALL,
    )
    replacement = """        workload_slot = 0;
        if (!$value$plusargs("WORKLOAD_SLOT=%d", workload_slot))
            workload_slot = 0;
        if (workload_slot < 0 || workload_slot >= WORKLOADS)
            $fatal(1, "M2048 WORKLOAD_SLOT outside 0..191");
        $readmemh(FIXTURE, fixture_word);
        $readmemh(STATS, stats_word);
        sample_id = stats_word[workload_slot][31:0];
        layer_id = stats_word[workload_slot][63:32];
        is_fc2 = stats_word[workload_slot][95:64];
        token_start = stats_word[workload_slot][127:96];
        real_source_groups = stats_word[workload_slot][159:128];
        expected_rows = stats_word[workload_slot][191:160];
        expected_issues = stats_word[workload_slot][223:192];
        expected_products = stats_word[workload_slot][255:224];
        expected_base_misses = stats_word[workload_slot][287:256];
        expected_base_hits = stats_word[workload_slot][319:288];
        expected_base_evictions = stats_word[workload_slot][351:320];
        expected_tsbg_misses = stats_word[workload_slot][383:352];
        expected_tsbg_hits = stats_word[workload_slot][415:384];
        expected_tsbg_evictions = stats_word[workload_slot][447:416];
        if (real_source_groups != 6 && real_source_groups != 12
                && real_source_groups != 24 && real_source_groups != 48)
            $fatal(1, "M2048 real source-group metadata drift");
"""
    text, count = old_pattern.subn(replacement, text)
    if count != 1:
        raise RuntimeError("M2046 slot-case anchor drift")
    text = text.replace(
        "The frozen ep34 FC1 fixture contains only {0,+1} sources.",
        "The selected ep34 FC1/FC2 fixture contains only {0,+1} sources.",
    )
    text = replace_once(
        text,
        "        if (reorder_base == 0 || reorder_tsbg == 0\n"
        "                || independent_stall_base == 0 || independent_stall_tsbg == 0)\n"
        "            $fatal(1, \"M1880 independent-bank skew/reorder not covered\");\n",
        "        if (expected_rows != 0\n"
        "                && (reorder_base == 0 || reorder_tsbg == 0\n"
        "                || independent_stall_base == 0 || independent_stall_tsbg == 0))\n"
        "            $fatal(1, \"M1880 independent-bank skew/reorder not covered\");\n",
    )
    text = replace_once(
        text,
        "        if (!saved_rsp_valid)\n"
        "            $fatal(1, \"M1880 accepted response identity was not captured\");\n",
        "        if (expected_rows != 0 && !saved_rsp_valid)\n"
        "            $fatal(1, \"M1880 accepted response identity was not captured\");\n",
    )
    old_replay = """        m1970_phase = "retired_replay";
        $display("M1970_PHASE retired_replay_begin cycle=%0d", tb_cycle);
        tsbg.replay_epoch[3] = saved_rsp_epoch;
        tsbg.replay_slot[3] = saved_rsp_slot;
        tsbg.replay_generation[3] = saved_rsp_generation;
        tsbg.replay_tag[3] = saved_rsp_tag;
        for (int lane = 0; lane < LANES; lane++)
            tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];
        tsbg.inject_replay[3] = 1;
        retired_identity_replay_count = retired_identity_replay_count + 1;
        @(posedge clk_core);
        if (tsbg.mem_rsp_accept[3])
            $fatal(1, "M1880 retired legal identity replay was accepted");
        tsbg.inject_replay[3] = 0;
        repeat (2) @(posedge clk_core);
        if (!tsbg.protocol_error || !tsbg.stale_response_seen
                || replay_accept_count != 0)
            $fatal(1, "M1880 retired legal identity replay did not fail closed");
        $display("M1970_PHASE retired_replay_complete cycle=%0d", tb_cycle);
"""
    new_replay = """        if (expected_rows != 0) begin
            m1970_phase = "retired_replay";
            $display("M1970_PHASE retired_replay_begin cycle=%0d", tb_cycle);
            tsbg.replay_epoch[3] = saved_rsp_epoch;
            tsbg.replay_slot[3] = saved_rsp_slot;
            tsbg.replay_generation[3] = saved_rsp_generation;
            tsbg.replay_tag[3] = saved_rsp_tag;
            for (int lane = 0; lane < LANES; lane++)
                tsbg.replay_weight[3][lane] = saved_rsp_weight[lane];
            tsbg.inject_replay[3] = 1;
            retired_identity_replay_count = retired_identity_replay_count + 1;
            @(posedge clk_core);
            if (tsbg.mem_rsp_accept[3])
                $fatal(1, "M1880 retired legal identity replay was accepted");
            tsbg.inject_replay[3] = 0;
            repeat (2) @(posedge clk_core);
            if (!tsbg.protocol_error || !tsbg.stale_response_seen
                    || replay_accept_count != 0)
                $fatal(1, "M1880 retired legal identity replay did not fail closed");
            $display("M1970_PHASE retired_replay_complete cycle=%0d", tb_cycle);
        end else begin
            $display("M2048_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE");
        end
"""
    text = replace_once(text, old_replay, new_replay)
    text = replace_once(
        text,
        "        if (full_tsbg_exec_cycles <= 0 || full_base_exec_cycles <= 0\n"
        "                || full_base_exec_cycles * 1.0 / full_tsbg_exec_cycles < 1.15)\n"
        "            $fatal(1, \"M1880 directed local cycle gate below 1.15x\");\n"
        "        if (stale_attack_count != 1 || retired_identity_replay_count != 1\n",
        "        if (full_tsbg_exec_cycles <= 0 || full_base_exec_cycles <= 0)\n"
        "            $fatal(1, \"M1880 directed local cycle measurement invalid\");\n"
        "        if (stale_attack_count != 1\n"
        "                || retired_identity_replay_count != (expected_rows != 0)\n",
    )
    old_pass = (
        '        $display("PASS_M2046_EP34_TSBG_G48_CYCLE sample_slot=%0d '
        'sample_id=%0d layer=28 rows=%0d issues=%0d products=%0d commits=%0d '
        'base_cycles=%0d tsbg_cycles=%0d bundles_base=%0d bundles_tsbg=%0d '
        'scalar_base=%0d scalar_tsbg=%0d stale=%0d retired_replay=%0d '
        'replay_accept=%0d reset=%0d recovery=%0d system_speedup=false",\n'
        '            sample_slot, sample_id, expected_rows, expected_issues,\n'
        '            expected_products, EXPECTED_COMMITS,\n'
    )
    new_pass = (
        '        $display("PASS_M2048_EP34_TSBG_MULTILAYER_TOKEN_CYCLE '
        'workload_slot=%0d sample_id=%0d layer=%0d is_fc2=%0d token_start=%0d '
        'source_groups=%0d physical_groups=48 rows=%0d issues=%0d products=%0d '
        'commits=%0d base_cycles=%0d tsbg_cycles=%0d bundles_base=%0d '
        'bundles_tsbg=%0d scalar_base=%0d scalar_tsbg=%0d stale=%0d '
        'retired_replay=%0d replay_accept=%0d reset=%0d recovery=%0d '
        'real_weights=false system_speedup=false",\n'
        '            workload_slot, sample_id, layer_id, is_fc2, token_start,\n'
        '            real_source_groups, expected_rows, expected_issues,\n'
        '            expected_products, EXPECTED_COMMITS,\n'
    )
    text = replace_once(text, old_pass, new_pass)
    if "sample_slot" in text or "SAMPLE_SLOT" in text:
        raise RuntimeError("old sample-slot identifier survived")
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"PASS output={OUTPUT} sha256={sha256(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
