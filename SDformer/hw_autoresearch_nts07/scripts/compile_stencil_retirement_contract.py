#!/usr/bin/env python3
"""Compile a fixed stencil and raster order into exact retirement rules."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local5_stencil_retirement_compiler_20260814"
TOPOLOGIES = {
    "cross_r1": ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    "cross_r2": ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)),
    # A deliberately asymmetric, non-cross stencil used only to test whether
    # the generated contract still executes without hand-written role logic.
    "asym5": ((0, 0), (-1, -1), (-1, 0), (0, -1), (1, 0)),
}

BAD_OUTPUT_RE = re.compile(
    r"%Error|\bERROR:|\bFATAL:|Assertion failed|MISMATCH|\bFAIL\b"
)


@dataclass(frozen=True, order=True)
class Offset:
    dy: int
    dx: int


def valid(y: int, x: int, height: int, width: int) -> bool:
    return 0 <= y < height and 0 <= x < width


def consumers(
    source: tuple[int, int], offsets: tuple[Offset, ...], height: int, width: int
) -> list[tuple[int, int, Offset]]:
    sy, sx = source
    rows = []
    for offset in offsets:
        dy = sy - offset.dy
        dx = sx - offset.dx
        if valid(dy, dx, height, width):
            rows.append((dy, dx, offset))
    return rows


def exact_schedule(
    offsets: tuple[Offset, ...], height: int, width: int
) -> dict[tuple[int, int], list[tuple[int, int, Offset]]]:
    schedule = {(y, x): [] for y in range(height) for x in range(width)}
    for sy in range(height):
        for sx in range(width):
            visits = consumers((sy, sx), offsets, height, width)
            if not visits:
                raise AssertionError("self-containing topology lost a source")
            dy, dx, role = max(visits, key=lambda row: (row[0], row[1]))
            schedule[(dy, dx)].append((sy, sx, role))
    priority = tuple(sorted(offsets))
    for events in schedule.values():
        events.sort(key=lambda row: priority.index(row[2]))
    return schedule


def compile_rules(
    offsets: tuple[Offset, ...], height: int, width: int
) -> tuple[list[Offset], dict[tuple[int, int], list[tuple[int, int, Offset]]]]:
    # A consumer at d=s-offset is later in raster order when offset is smaller
    # in lexicographic (dy, dx) order.  An offset retires a source only when all
    # earlier offsets would place their consumers outside the image.
    priority = tuple(sorted(offsets))
    active_roles = []
    schedule = {(y, x): [] for y in range(height) for x in range(width)}
    for offset in priority:
        role_used = False
        earlier = priority[: priority.index(offset)]
        for y in range(height):
            for x in range(width):
                sy = y + offset.dy
                sx = x + offset.dx
                if not valid(sy, sx, height, width):
                    continue
                if any(
                    valid(sy - other.dy, sx - other.dx, height, width)
                    for other in earlier
                ):
                    continue
                schedule[(y, x)].append((sy, sx, offset))
                role_used = True
        if role_used:
            active_roles.append(offset)
    return active_roles, schedule


def find_affine_bank(offsets: tuple[Offset, ...]) -> dict[str, int | list[int]]:
    for banks in range(len(offsets), 4 * len(offsets) + 1):
        for ax in range(banks):
            for by in range(banks):
                colors = [
                    (ax * offset.dx + by * offset.dy) % banks
                    for offset in offsets
                ]
                if len(set(colors)) == len(offsets):
                    return {"banks": banks, "ax": ax, "by": by, "colors": colors}
    raise ValueError("no affine bank map in search range")


def relation_row_span(offsets: tuple[Offset, ...]) -> int:
    dys = [offset.dy for offset in offsets]
    return max(dys) - min(dys) + 1


def sv_int(value: int) -> str:
    return str(value) if value >= 0 else f"-{abs(value)}"


def consumer_valid_expr(source_suffix: str, other: Offset) -> str:
    y_expr = f"(sy_{source_suffix} - ({sv_int(other.dy)}))"
    x_expr = f"(sx_{source_suffix} - ({sv_int(other.dx)}))"
    return (
        f"(({y_expr}) >= 0) && (({y_expr}) < HEIGHT) && "
        f"(({x_expr}) >= 0) && (({x_expr}) < WIDTH)"
    )


def emit_sv(name: str, offsets: tuple[Offset, ...], active: list[Offset]) -> Path:
    module = f"generated_{name}_retirement_rules"
    path = OUT / f"{module}.sv"
    lines = [
        "`default_nettype none",
        "`timescale 1ns/1ps",
        "",
        f"module {module} #(",
        "    parameter int HEIGHT = 15,",
        "    parameter int WIDTH = 15,",
        "    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),",
        "    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH)",
        ") (",
        "    input  logic in_valid,",
        "    input  logic [Y_W-1:0] in_y,",
        "    input  logic [X_W-1:0] in_x,",
        f"    output logic [{len(active)-1}:0] event_valid,",
        f"    output logic [{len(active)}*Y_W-1:0] event_y,",
        f"    output logic [{len(active)}*X_W-1:0] event_x",
        ");",
        "",
    ]
    priority = tuple(sorted(offsets))
    for offset in active:
        suffix = f"{offset.dy}_{offset.dx}".replace("-", "m")
        lines.extend(
            [
                f"    integer sy_{suffix};",
                f"    integer sx_{suffix};",
            ]
        )
    lines.extend(["", "    always_comb begin"])
    for offset in active:
        suffix = f"{offset.dy}_{offset.dx}".replace("-", "m")
        lines.extend(
            [
                f"        sy_{suffix} = $signed({{{{(32-Y_W){{1'b0}}}}, in_y}}) + ({sv_int(offset.dy)});",
                f"        sx_{suffix} = $signed({{{{(32-X_W){{1'b0}}}}, in_x}}) + ({sv_int(offset.dx)});",
            ]
        )
    lines.extend(["        event_valid = '0;", "        event_y = '0;", "        event_x = '0;"])
    for index, offset in enumerate(active):
        suffix = f"{offset.dy}_{offset.dx}".replace("-", "m")
        valid_expr = " && ".join(
            [
                f"(sy_{suffix} >= 0)",
                f"(sy_{suffix} < HEIGHT)",
                f"(sx_{suffix} >= 0)",
                f"(sx_{suffix} < WIDTH)",
            ]
        )
        earlier = priority[: priority.index(offset)]
        blockers = [consumer_valid_expr(suffix, other) for other in earlier]
        condition = f"in_valid && {valid_expr}"
        if blockers:
            condition += " && !(" + " || ".join(blockers) + ")"
        lines.extend(
            [
                f"        if ({condition}) begin",
                f"            event_valid[{index}] = 1'b1;",
                f"            event_y[{index}*Y_W +: Y_W] = Y_W'(sy_{suffix});",
                f"            event_x[{index}*X_W +: X_W] = X_W'(sx_{suffix});",
                "        end",
            ]
        )
    lines.extend(["    end", "endmodule", "", "`default_nettype wire"])
    path.write_text("\n".join(lines) + "\n")
    return path


def emit_scheduler_wrapper(name: str, active_count: int) -> Path:
    module = f"generated_{name}_retirement_scheduler"
    rules_module = f"generated_{name}_retirement_rules"
    count_w = max(1, math.ceil(math.log2(active_count + 1)))
    path = OUT / f"{module}.sv"
    lines = [
        "`default_nettype none",
        "`timescale 1ns/1ps",
        "",
        f"module {module} #(",
        "    parameter int HEIGHT = 15,",
        "    parameter int WIDTH = 15,",
        "    parameter int TIME_PLANES = 2,",
        "    parameter int Y_W = (HEIGHT <= 1) ? 1 : $clog2(HEIGHT),",
        "    parameter int X_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH),",
        "    parameter int PLANE_W = (TIME_PLANES <= 1) ? 1 : $clog2(TIME_PLANES),",
        "    parameter int SOURCE_ID_W =",
        "        (HEIGHT * WIDTH * TIME_PLANES <= 1)",
        "        ? 1 : $clog2(HEIGHT * WIDTH * TIME_PLANES)",
        ") (",
        "    input  logic clk_core,",
        "    input  logic rst_core,",
        "    input  logic plane_start,",
        "    input  logic [PLANE_W-1:0] plane_id,",
        "    input  logic in_valid,",
        "    output logic in_ready,",
        "    input  logic [" + str(active_count - 1) + ":0] in_candidate_valid,",
        "    input  logic [Y_W-1:0] in_y,",
        "    input  logic [X_W-1:0] in_x,",
        "    output logic retire_valid,",
        "    input  logic retire_ready,",
        "    output logic [SOURCE_ID_W-1:0] retire_source_id,",
        "    output logic [Y_W-1:0] retire_y,",
        "    output logic [X_W-1:0] retire_x,",
        "    output logic plane_idle,",
        "    output logic [31:0] perf_producer_stalls,",
        f"    output logic [{count_w-1}:0] perf_max_pending",
        ");",
        "",
        f"    logic [{active_count-1}:0] rule_valid;",
        f"    logic [{active_count}*Y_W-1:0] rule_y;",
        f"    logic [{active_count}*X_W-1:0] rule_x;",
        "",
        f"    {rules_module} #(",
        "        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .Y_W(Y_W), .X_W(X_W)",
        "    ) u_rules (",
        "        .in_valid, .in_y, .in_x,",
        "        .event_valid(rule_valid), .event_y(rule_y), .event_x(rule_x)",
        "    );",
        "",
        "    qfit_compiled_retirement_shell #(",
        "        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),",
        f"        .RULES({active_count}), .Y_W(Y_W), .X_W(X_W),",
        "        .PLANE_W(PLANE_W), .SOURCE_ID_W(SOURCE_ID_W)",
        "    ) u_shell (",
        "        .clk_core, .rst_core, .plane_start, .plane_id,",
        "        .in_valid, .in_ready, .rule_valid, .rule_y, .rule_x,",
        "        .rule_candidate_valid(in_candidate_valid),",
        "        .retire_valid, .retire_ready, .retire_source_id,",
        "        .retire_y, .retire_x, .plane_idle,",
        "        .perf_producer_stalls, .perf_max_pending",
        "    );",
        "",
        "endmodule",
        "",
        "`default_nettype wire",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def emit_expected_vectors(
    name: str,
    active: list[Offset],
    schedule: dict[tuple[int, int], list[tuple[int, int, Offset]]],
    height: int,
    width: int,
) -> tuple[Path, int]:
    y_width = max(1, math.ceil(math.log2(height)))
    x_width = max(1, math.ceil(math.log2(width)))
    active_count = len(active)
    packed_width = active_count * (1 + y_width + x_width)
    hex_digits = math.ceil(packed_width / 4)
    words = []
    for y in range(height):
        for x in range(width):
            by_role = {role: (sy, sx) for sy, sx, role in schedule[(y, x)]}
            valid_bits = 0
            y_bits = 0
            x_bits = 0
            for index, role in enumerate(active):
                if role not in by_role:
                    continue
                sy, sx = by_role[role]
                valid_bits |= 1 << index
                y_bits |= sy << (index * y_width)
                x_bits |= sx << (index * x_width)
            packed = (
                valid_bits
                | (y_bits << active_count)
                | (x_bits << (active_count + active_count * y_width))
            )
            words.append(f"{packed:0{hex_digits}x}")
    path = OUT / f"expected_{name}_15x15.memh"
    path.write_text("\n".join(words) + "\n")
    return path, packed_width


def emit_tb(name: str, active_count: int, expected_path: Path) -> Path:
    module = f"generated_{name}_retirement_rules"
    tb_module = f"tb_{module}"
    path = OUT / f"{tb_module}.sv"
    lines = [
        "`default_nettype none",
        "`timescale 1ns/1ps",
        "",
        f"module {tb_module};",
        "    localparam int HEIGHT = 15;",
        "    localparam int WIDTH = 15;",
        "    localparam int Y_W = $clog2(HEIGHT);",
        "    localparam int X_W = $clog2(WIDTH);",
        f"    localparam int ACTIVE = {active_count};",
        "    localparam int PACK_W = ACTIVE * (1 + Y_W + X_W);",
        "",
        "    logic in_valid;",
        "    logic [Y_W-1:0] in_y;",
        "    logic [X_W-1:0] in_x;",
        "    logic [ACTIVE-1:0] event_valid;",
        "    logic [ACTIVE*Y_W-1:0] event_y;",
        "    logic [ACTIVE*X_W-1:0] event_x;",
        "    logic [PACK_W-1:0] expected [0:HEIGHT*WIDTH-1];",
        "    logic [PACK_W-1:0] actual;",
        "    integer y;",
        "    integer x;",
        "    integer index;",
        "",
        f"    {module} #(",
        "        .HEIGHT(HEIGHT),",
        "        .WIDTH(WIDTH)",
        "    ) dut (.*);",
        "",
        "    assign actual = {event_x, event_y, event_valid};",
        "",
        "    initial begin",
        f"        $readmemh(\"{expected_path}\", expected);",
        "        in_valid = 1'b0;",
        "        in_y = '0;",
        "        in_x = '0;",
        "        #1;",
        "        if (actual !== '0) $fatal(1, \"invalid input emitted event\");",
        "        index = 0;",
        "        for (y = 0; y < HEIGHT; y = y + 1) begin",
        "            for (x = 0; x < WIDTH; x = x + 1) begin",
        "                in_valid = 1'b1;",
        "                in_y = Y_W'(y);",
        "                in_x = X_W'(x);",
        "                #1;",
        "                if (actual !== expected[index]) begin",
        "                    $fatal(1, \"mismatch topology=%s index=%0d y=%0d x=%0d actual=%h expected=%h\",",
        f"                           \"{name}\", index, y, x, actual, expected[index]);",
        "                end",
        "                index = index + 1;",
        "            end",
        "        end",
        "        in_valid = 1'b0;",
        "        #1;",
        "        if (actual !== '0) $fatal(1, \"post-scan invalid input emitted event\");",
        f"        $display(\"PASS topology={name} cases=%0d\", index);",
        "        $finish;",
        "    end",
        "endmodule",
        "",
        "`default_nettype wire",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def emit_scheduler_tb(
    name: str,
    active: list[Offset],
    schedule: dict[tuple[int, int], list[tuple[int, int, Offset]]],
) -> Path:
    module = f"generated_{name}_retirement_scheduler"
    tb_module = f"tb_{module}"
    path = OUT / f"{tb_module}.sv"
    active_index = {offset: index for index, offset in enumerate(active)}
    count_w = max(1, math.ceil(math.log2(len(active) + 1)))
    lines = [
        "`default_nettype none",
        "`timescale 1ns/1ps",
        "",
        f"module {tb_module};",
        "    localparam int HEIGHT = 15;",
        "    localparam int WIDTH = 15;",
        "    localparam int TIME_PLANES = 2;",
        "    localparam int TOKENS = HEIGHT * WIDTH;",
        "    localparam int TOTAL = TOKENS * TIME_PLANES;",
        "    localparam int Y_W = $clog2(HEIGHT);",
        "    localparam int X_W = $clog2(WIDTH);",
        "    localparam int SOURCE_ID_W = $clog2(TOTAL);",
        f"    localparam int RULES = {len(active)};",
        f"    localparam int COUNT_W = {count_w};",
        "",
        "    logic clk_core;",
        "    logic rst_core;",
        "    logic plane_start;",
        "    logic plane_id;",
        "    logic in_valid;",
        "    logic in_ready;",
        "    logic [RULES-1:0] in_candidate_valid;",
        "    logic [Y_W-1:0] in_y;",
        "    logic [X_W-1:0] in_x;",
        "    logic retire_valid;",
        "    logic retire_ready;",
        "    logic [SOURCE_ID_W-1:0] retire_source_id;",
        "    logic [Y_W-1:0] retire_y;",
        "    logic [X_W-1:0] retire_x;",
        "    logic plane_idle;",
        "    logic [31:0] perf_producer_stalls;",
        "    logic [COUNT_W-1:0] perf_max_pending;",
        "    bit seen [0:TOTAL-1];",
        "    int expected_ids [0:TOTAL-1];",
        "    int expected_count;",
        "    int retire_count;",
        "    int seed;",
        "    int long_stall_remaining;",
        "    bit long_stall_started;",
        "",
        f"    {module} #(",
        "        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES)",
        "    ) dut (.*);",
        "",
        "    always #5 clk_core = ~clk_core;",
        "",
        "    function automatic bit source_active(input int p, input int y, input int x);",
        "        int sid;",
        "        sid = p * TOKENS + y * WIDTH + x;",
        "        if (seed == 99)",
        "            source_active = x == 3 && (y == 0 || y == 10);",
        "        else",
        "            source_active = ((sid * 13 + seed * 5 + y * 3 + x) % 7) != 0;",
        "    endfunction",
        "",
        "    function automatic bit source_valid(input int y, input int x);",
        "        source_valid = y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH;",
        "    endfunction",
        "",
        "    function automatic bit candidate_for_rule(",
        "        input int rule, input int p, input int y, input int x",
        "    );",
        "        int sy;",
        "        int sx;",
        "        sy = y;",
        "        sx = x;",
        "        case (rule)",
    ]
    for index, offset in enumerate(active):
        lines.extend(
            [
                f"            {index}: begin sy = y + ({sv_int(offset.dy)}); sx = x + ({sv_int(offset.dx)}); end",
            ]
        )
    lines.extend(
        [
            "            default: begin sy = -1; sx = -1; end",
            "        endcase",
            "        candidate_for_rule = source_valid(sy, sx) && source_active(p, sy, sx);",
            "    endfunction",
            "",
            "    task automatic drive_plane(input int p);",
            "        int accepted;",
            "        bit handshake;",
            "        while (!plane_idle) @(negedge clk_core);",
            "        @(negedge clk_core);",
            "        plane_id = p[0];",
            "        plane_start = 1'b1;",
            "        in_valid = 1'b0;",
            "        @(negedge clk_core);",
            "        plane_start = 1'b0;",
            "        accepted = 0;",
            "        while (accepted < TOKENS) begin",
            "            in_valid = ($urandom_range(0, 4) != 0);",
            "            in_y = Y_W'(accepted / WIDTH);",
            "            in_x = X_W'(accepted % WIDTH);",
            "            for (int rule = 0; rule < RULES; rule = rule + 1)",
            "                in_candidate_valid[rule] = candidate_for_rule(",
            "                    rule, p, accepted / WIDTH, accepted % WIDTH",
            "                );",
            "            @(posedge clk_core);",
            "            handshake = in_valid && in_ready;",
            "            @(negedge clk_core);",
            "            if (handshake) accepted = accepted + 1;",
            "        end",
            "        in_valid = 1'b0;",
            "        in_candidate_valid = '0;",
            "        while (!plane_idle) @(negedge clk_core);",
            "    endtask",
            "",
            "    always_ff @(posedge clk_core) begin",
            "        if (rst_core) begin",
            "            retire_ready <= 1'b0;",
            "            long_stall_remaining <= 0;",
            "            long_stall_started <= 1'b0;",
            "        end else if (retire_valid && !long_stall_started) begin",
            "            retire_ready <= 1'b0;",
            "            long_stall_remaining <= 25;",
            "            long_stall_started <= 1'b1;",
            "        end else if (long_stall_remaining > 0) begin",
            "            retire_ready <= 1'b0;",
            "            long_stall_remaining <= long_stall_remaining - 1;",
            "        end else begin",
            "            retire_ready <= ($urandom_range(0, 5) != 0);",
            "        end",
            "    end",
            "",
            "    always_ff @(posedge clk_core) begin",
            "        if (!rst_core && retire_valid && retire_ready) begin",
            "            int sid;",
            "            sid = int'(retire_source_id);",
            "            if (sid >= TOTAL) $fatal(1, \"retirement id out of range sid=%0d\", sid);",
            "            if (seen[sid]) $fatal(1, \"duplicate retirement sid=%0d\", sid);",
            "            if (!source_active(sid / TOKENS, (sid % TOKENS) / WIDTH, sid % WIDTH))",
            "                $fatal(1, \"inactive retirement sid=%0d\", sid);",
            "            if (retire_count >= expected_count || sid != expected_ids[retire_count])",
            "                $fatal(1, \"ordered retirement mismatch index=%0d sid=%0d expected=%0d\",",
            "                       retire_count, sid, expected_ids[retire_count]);",
            "            seen[sid] <= 1'b1;",
            "            retire_count <= retire_count + 1;",
            "        end",
            "    end",
            "",
            "    initial begin",
            "        clk_core = 1'b0;",
            "        rst_core = 1'b1;",
            "        plane_start = 1'b0;",
            "        plane_id = 1'b0;",
            "        in_valid = 1'b0;",
            "        in_candidate_valid = '0;",
            "        in_y = '0;",
            "        in_x = '0;",
            "        retire_ready = 1'b0;",
            "        expected_count = 0;",
            "        retire_count = 0;",
            "        seed = 17;",
            "        void'($value$plusargs(\"SEED=%d\", seed));",
            "        long_stall_remaining = 0;",
            "        long_stall_started = 1'b0;",
            "        for (int sid = 0; sid < TOTAL; sid = sid + 1) seen[sid] = 1'b0;",
        ]
    )
    for plane in range(2):
        for y in range(15):
            for x in range(15):
                for sy, sx, role in schedule[(y, x)]:
                    if role not in active_index:
                        raise AssertionError("schedule emitted inactive rule")
                    source_id = plane * 225 + sy * 15 + sx
                    lines.extend(
                        [
                            f"        if (source_active({plane}, {sy}, {sx})) begin",
                            f"            expected_ids[expected_count] = {source_id};",
                            "            expected_count = expected_count + 1;",
                            "        end",
                        ]
                    )
    lines.extend(
        [
            "        repeat (4) @(negedge clk_core);",
            "        rst_core = 1'b0;",
            "        drive_plane(0);",
            "        drive_plane(1);",
            "        repeat (4) @(negedge clk_core);",
            "        if (retire_count != expected_count)",
            "            $fatal(1, \"population mismatch got=%0d expected=%0d\", retire_count, expected_count);",
            f"        $display(\"PASS compiled_scheduler topology={name} seed=%0d retire=%0d stalls=%0d pending=%0d\",",
            "                 seed, retire_count, perf_producer_stalls, perf_max_pending);",
            "        $finish;",
            "    end",
            "",
            "    initial begin",
            "        repeat (20000) @(posedge clk_core);",
            "        $fatal(1, \"timeout\");",
            "    end",
            "endmodule",
            "",
            "`default_nettype wire",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def run_logged(command: list[str], log_path: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = completed.stdout or ""
    log_path.write_text("$ " + " ".join(command) + "\n" + text)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
    if BAD_OUTPUT_RE.search(text):
        raise RuntimeError(f"fail-closed marker in command output: {' '.join(command)}")
    return text


def verify_generated_rtl(name: str, sv_path: Path, tb_path: Path) -> dict[str, object]:
    module = f"generated_{name}_retirement_rules"
    tb_module = f"tb_{module}"
    icarus_bin = OUT / f"{tb_module}.vvp"
    icarus_compile_log = OUT / f"{name}_iverilog_compile.log"
    icarus_run_log = OUT / f"{name}_iverilog_run.log"
    verilator_dir = OUT / f"verilator_{name}"
    verilator_compile_log = OUT / f"{name}_verilator_compile.log"
    verilator_run_log = OUT / f"{name}_verilator_run.log"
    yosys_log = OUT / f"{name}_yosys_check.log"

    run_logged(
        ["iverilog", "-g2012", "-s", tb_module, "-o", str(icarus_bin), str(sv_path), str(tb_path)],
        icarus_compile_log,
    )
    icarus_text = run_logged(["vvp", str(icarus_bin)], icarus_run_log)
    verilator_text = run_logged(
        [
            "verilator",
            "--binary",
            "--timing",
            "--assert",
            "-Wall",
            "-Wno-fatal",
            "--top-module",
            tb_module,
            "--Mdir",
            str(verilator_dir),
            str(sv_path),
            str(tb_path),
        ],
        verilator_compile_log,
    )
    verilator_run_text = run_logged(
        [str(verilator_dir / f"V{tb_module}")], verilator_run_log
    )
    yosys_text = run_logged(
        [
            "yosys",
            "-p",
            f"read_verilog -sv {sv_path}; hierarchy -top {module}; proc; opt; check -assert",
        ],
        yosys_log,
    )
    if "PASS topology=" not in icarus_text or "PASS topology=" not in verilator_run_text:
        raise AssertionError(f"missing simulator PASS marker for {name}")
    if "0 problems" not in yosys_text:
        raise AssertionError(f"Yosys check did not report zero problems for {name}")
    warning_lines = [line for line in verilator_text.splitlines() if "%Warning" in line]
    return {
        "iverilog": "PASS",
        "verilator_assert": "PASS",
        "yosys_check": "PASS",
        "verilator_warning_count": len(warning_lines),
        "logs": {
            "iverilog_compile": str(icarus_compile_log.relative_to(ROOT)),
            "iverilog_run": str(icarus_run_log.relative_to(ROOT)),
            "verilator_compile": str(verilator_compile_log.relative_to(ROOT)),
            "verilator_run": str(verilator_run_log.relative_to(ROOT)),
            "yosys_check": str(yosys_log.relative_to(ROOT)),
        },
    }


def verify_generated_scheduler(
    name: str,
    rules_path: Path,
    wrapper_path: Path,
    tb_path: Path,
) -> dict[str, object]:
    module = f"generated_{name}_retirement_scheduler"
    tb_module = f"tb_{module}"
    shell = ROOT / "rtl_qfit/qfit_compiled_retirement_shell.sv"
    icarus_bin = OUT / f"{tb_module}.vvp"
    icarus_compile = OUT / f"{name}_scheduler_iverilog_compile.log"
    verilator_dir = OUT / f"verilator_{name}_scheduler"
    verilator_compile = OUT / f"{name}_scheduler_verilator_compile.log"
    yosys_log = OUT / f"{name}_scheduler_yosys_check.log"

    sources = [str(rules_path), str(shell), str(wrapper_path), str(tb_path)]
    run_logged(
        ["iverilog", "-g2012", "-s", tb_module, "-o", str(icarus_bin), *sources],
        icarus_compile,
    )
    icarus_runs = {}
    for seed in (17, 99):
        path = OUT / f"{name}_scheduler_iverilog_seed{seed}.log"
        text = run_logged(["vvp", str(icarus_bin), f"+SEED={seed}"], path)
        if f"PASS compiled_scheduler topology={name}" not in text:
            raise AssertionError(f"missing Icarus scheduler PASS for {name} seed{seed}")
        icarus_runs[str(seed)] = str(path.relative_to(ROOT))

    verilator_text = run_logged(
        [
            "verilator",
            "--binary",
            "--timing",
            "--assert",
            "-Wall",
            "-Wno-fatal",
            "-Wno-BLKSEQ",
            "-Wno-UNUSEDSIGNAL",
            "--top-module",
            tb_module,
            "--Mdir",
            str(verilator_dir),
            *sources,
        ],
        verilator_compile,
    )
    verilator_runs = {}
    for seed in (17, 99):
        path = OUT / f"{name}_scheduler_verilator_seed{seed}.log"
        text = run_logged(
            [str(verilator_dir / f"V{tb_module}"), f"+SEED={seed}"], path
        )
        if f"PASS compiled_scheduler topology={name}" not in text:
            raise AssertionError(
                f"missing Verilator scheduler PASS for {name} seed{seed}"
            )
        verilator_runs[str(seed)] = str(path.relative_to(ROOT))

    yosys_text = run_logged(
        [
            "yosys",
            "-p",
            (
                f"read_verilog -sv {rules_path} {shell} {wrapper_path}; "
                f"hierarchy -top {module}; proc; opt; check -assert"
            ),
        ],
        yosys_log,
    )
    if "0 problems" not in yosys_text:
        raise AssertionError(f"scheduler Yosys check did not pass for {name}")
    warning_lines = [line for line in verilator_text.splitlines() if "%Warning" in line]
    return {
        "iverilog_seeds": [17, 99],
        "verilator_assert_seeds": [17, 99],
        "ordered_retirement": "PASS",
        "sparse_candidate_filter": "PASS",
        "long_backpressure": "PASS",
        "yosys_check": "PASS",
        "verilator_warning_count": len(warning_lines),
        "logs": {
            "iverilog_compile": str(icarus_compile.relative_to(ROOT)),
            "iverilog_runs": icarus_runs,
            "verilator_compile": str(verilator_compile.relative_to(ROOT)),
            "verilator_runs": verilator_runs,
            "yosys_check": str(yosys_log.relative_to(ROOT)),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compiled = {}
    exhaustive_cases = 0
    for name, raw_offsets in TOPOLOGIES.items():
        offsets = tuple(Offset(*row) for row in raw_offsets)
        if Offset(0, 0) not in offsets or len(set(offsets)) != len(offsets):
            raise ValueError(f"invalid topology {name}")
        active15, generated15 = compile_rules(offsets, 15, 15)
        exact15 = exact_schedule(offsets, 15, 15)
        if generated15 != exact15:
            raise AssertionError(f"15x15 schedule mismatch for {name}")

        checked = []
        min_size = max(3, relation_row_span(offsets))
        for height in range(min_size, 18):
            for width in range(min_size, 18):
                active, generated = compile_rules(offsets, height, width)
                exact = exact_schedule(offsets, height, width)
                if generated != exact:
                    raise AssertionError(f"schedule mismatch {name} {height}x{width}")
                if active != active15:
                    raise AssertionError(f"active rule drift {name} {height}x{width}")
                checked.append([height, width])
                exhaustive_cases += height * width

        bank = find_affine_bank(offsets)
        sv_path = emit_sv(name, offsets, active15)
        expected_path, packed_width = emit_expected_vectors(
            name, active15, exact15, 15, 15
        )
        tb_path = emit_tb(name, len(active15), expected_path)
        verification = verify_generated_rtl(name, sv_path, tb_path)
        if verification["verilator_warning_count"] != 0:
            raise AssertionError(
                f"generated RTL has Verilator warnings for {name}: "
                f"{verification['verilator_warning_count']}"
            )
        wrapper_path = emit_scheduler_wrapper(name, len(active15))
        scheduler_tb_path = emit_scheduler_tb(name, active15, exact15)
        scheduler_verification = verify_generated_scheduler(
            name,
            sv_path,
            wrapper_path,
            scheduler_tb_path,
        )
        if scheduler_verification["verilator_warning_count"] != 0:
            raise AssertionError(
                f"generated scheduler has Verilator warnings for {name}: "
                f"{scheduler_verification['verilator_warning_count']}"
            )
        dynamic_generation_bits = math.ceil(
            math.log2(math.ceil(15 / relation_row_span(offsets)) + 2)
        )
        dynamic_state_bits = (
            bank["banks"]
            * relation_row_span(offsets)
            * math.ceil(15 / bank["banks"])
            * (3 + dynamic_generation_bits)
            + relation_row_span(offsets) * dynamic_generation_bits
        )
        compiled[name] = {
            "offsets": [[row.dy, row.dx] for row in offsets],
            "retirement_priority": [[row.dy, row.dx] for row in sorted(offsets)],
            "active_retirement_rules": [[row.dy, row.dx] for row in active15],
            "active_rule_count": len(active15),
            "relation_row_span": relation_row_span(offsets),
            "affine_bank_map": bank,
            "checked_grids": checked,
            "retired_sources_15x15": sum(len(rows) for rows in generated15.values()),
            "max_events_per_destination_15x15": max(len(rows) for rows in generated15.values()),
            "dynamic_tracker_state_bits_15x15_model": dynamic_state_bits,
            "generated_sv": str(sv_path.relative_to(ROOT)),
            "generated_sv_sha256": hashlib.sha256(sv_path.read_bytes()).hexdigest(),
            "expected_vector": str(expected_path.relative_to(ROOT)),
            "expected_vector_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
            "packed_miter_width": packed_width,
            "generated_tb": str(tb_path.relative_to(ROOT)),
            "generated_tb_sha256": hashlib.sha256(tb_path.read_bytes()).hexdigest(),
            "verification": verification,
            "generated_scheduler": str(wrapper_path.relative_to(ROOT)),
            "generated_scheduler_sha256": hashlib.sha256(
                wrapper_path.read_bytes()
            ).hexdigest(),
            "generated_scheduler_tb": str(scheduler_tb_path.relative_to(ROOT)),
            "generated_scheduler_tb_sha256": hashlib.sha256(
                scheduler_tb_path.read_bytes()
            ).hexdigest(),
            "scheduler_verification": scheduler_verification,
        }

    report = {
        "schema": "fixed_stencil_retirement_compiler_v1",
        "status": "CONDITIONAL_AS_208_CONTRACT_EVIDENCE",
        "evidence": "[finite-grid exhaustive regression] + [generated RTL dual-simulator miter] + [Yosys structural check]",
        "contract": {
            "input": "fixed source-relative offset set plus row-major destination order",
            "output": "last-consumer priority, live relation row span, minimal searched affine bank map, and synthesizable retirement-rule RTL",
            "last_consumer_theorem": "the latest consumer d=s-offset is induced by the lexicographically smallest valid offset; later-priority rules fire only when every earlier consumer is outside the image",
            "runtime_effect": "replace per-source consumer counts and generation tags with O(K) topology rules; relation payload frontier remains unchanged",
        },
        "topologies": compiled,
        "exhaustive_destination_cases": exhaustive_cases,
        "claim_boundary": [
            "cross_r1 is the current Local5 topology; cross_r2 demonstrates compiler generality but is not a trained network result",
            "the affine bank search proves conflict-free offset colors, not a globally minimal physical PPA",
            "generated rules are consumed by one topology-independent ordered pending/backpressure shell; integration_report separately verifies trained cross_r1 through the production tile",
            "no DC, SAIF, SRAM macro, full encoder, or accuracy result",
            "does not modify docs/359 frozen columns",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Fixed-stencil retirement compiler screen",
        "",
        f"- Verdict: `{report['status']}`.",
        f"- Exhaustive finite-grid destination cases: `{exhaustive_cases}`; mismatch `0`.",
        "- Compiler boundary: fixed source-relative offsets + row-major order -> last-consumer rules + relation row span + affine bank map + generated synthesizable rule RTL.",
        "",
        "| topology | offsets | active rules | row span | affine bank | 15x15 retired | max events/cycle | dynamic state model |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, row in compiled.items():
        bank = row["affine_bank_map"]
        lines.append(
            f"| {name} | {len(row['offsets'])} | {row['active_rule_count']} | {row['relation_row_span']} | "
            f"B{bank['banks']}: ({bank['ax']}x+{bank['by']}y) mod {bank['banks']} | "
            f"{row['retired_sources_15x15']} | {row['max_events_per_destination_15x15']} | "
            f"{row['dynamic_tracker_state_bits_15x15_model']} bit |"
        )
    lines.extend([
        "",
        "## Architectural reading",
        "",
        "The current three-event FCSR is one compiler instance, not a hand-written special case: for cross_r1, only north-source, bottom-row west-source, and final self rules can be last consumers. The same generated contract directly drives a topology-independent ordered shell for cross_r1, cross_r2, and the non-cross asymmetric asym5 stencil; no topology-specific role mapping remains in that sidecar path.",
        "",
        "## Generated RTL verification",
        "",
        "All topology instances are checked against independent exact-schedule vectors for every destination in a 15x15 raster. Their generated schedulers additionally run two planes with sparse candidate masks, random input gaps, a long output stall, and random backpressure. Icarus and Verilator preserve the independent ordered retirement list, Verilator reports zero warnings, and Yosys `check -assert` reports zero structural problems.",
        "",
        "## Boundary",
        "",
        "cross_r2 and asym5 prove executable contract generality only; neither is a trained Local5 accuracy result. The separate integration report covers the trained cross_r1 tile; matched RF/SRAM physical comparison and full-encoder evidence remain missing. Frozen main-table numbers are unchanged.",
    ])
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
