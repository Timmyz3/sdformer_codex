`timescale 1ns/1ps
`default_nettype none

`ifndef QFIT_PROJECTION_DUT
`define QFIT_PROJECTION_DUT qfit_tcfm5_projection_top
`endif

module tb_qfit_tcfm5_projection_top;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 6;
    localparam int TIME_PLANES = 1;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 2;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = 1;
    localparam int LANE_W = $clog2(HEAD_DIM);
    localparam int OUT_W = $clog2(OUT_DIM);

    logic clk_core;
    logic rst_core;
    logic weight_valid;
    logic weight_ready;
    logic [LANE_W-1:0] weight_lane;
    logic [OUT_W-1:0] weight_out;
    logic signed [W_W-1:0] weight_data;
    logic weight_last;
    logic weight_context_release;
    logic weight_context_release_ready;
    logic run_start;
    logic run_accumulate;
    logic run_busy;
    logic run_done;
    logic term_valid;
    logic term_ready;
    logic [PLANE_W-1:0] term_source_plane;
    logic [Y_W-1:0] term_source_y;
    logic [X_W-1:0] term_source_x;
    logic [LANE_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [4:0] term_destination_mask;
    logic term_window_last;
    logic window_close;
    logic window_close_ready;
    logic read_valid;
    logic read_ready;
    logic [PLANE_W-1:0] read_plane;
    logic [Y_W-1:0] read_y;
    logic [X_W-1:0] read_x;
    logic [OUT_W-1:0] read_out;
    logic read_data_valid;
    logic signed [ACC_W-1:0] read_data;
    logic protocol_error;
    logic [31:0] perf_product_terms;
    logic [31:0] perf_destination_updates;

    int signed expected [0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    int signed weight_scale;

    `QFIT_PROJECTION_DUT #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic int role_y(input int sy, input int role);
        case (role)
            1: role_y = sy + 1;
            2: role_y = sy - 1;
            default: role_y = sy;
        endcase
    endfunction

    function automatic int role_x(input int sx, input int role);
        case (role)
            3: role_x = sx + 1;
            4: role_x = sx - 1;
            default: role_x = sx;
        endcase
    endfunction

    task automatic load_weight(
        input int lane,
        input int out,
        input int signed value,
        input bit last
    );
        @(negedge clk_core);
        weight_lane = LANE_W'(lane);
        weight_out = OUT_W'(out);
        weight_data = W_W'(value);
        weight_last = last;
        weight_valid = 1'b1;
        @(posedge clk_core);
        if (!weight_ready)
            $fatal(1, "weight not ready");
        @(negedge clk_core);
        weight_valid = 1'b0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
    endtask

    task automatic send_term(
        input int sy,
        input int sx,
        input int lane,
        input int gate,
        input logic [4:0] role_mask,
        input bit last
    );
        @(negedge clk_core);
        term_source_plane = '0;
        term_source_y = Y_W'(sy);
        term_source_x = X_W'(sx);
        term_lane = LANE_W'(lane);
        term_gate = GATE_W'(gate);
        term_destination_mask = role_mask;
        term_window_last = last;
        term_valid = 1'b1;
        while (!term_ready)
            @(negedge clk_core);
        @(posedge clk_core);
        for (int role = 0; role < 5; role = role + 1) begin
            int dy;
            int dx;
            dy = role_y(sy, role);
            dx = role_x(sx, role);
            if (role_mask[role]) begin
                if (dy < 0 || dy >= HEIGHT || dx < 0 || dx >= WIDTH)
                    $fatal(1, "test generated invalid role");
                for (int out = 0; out < OUT_DIM; out = out + 1) begin
                    int signed weight;
                    weight = weight_scale
                           * (lane + 1) * (out == 0 ? 1 : -2);
                    expected[dy][dx][out] =
                        expected[dy][dx][out] + gate * weight;
                end
            end
        end
        @(negedge clk_core);
        term_valid = 1'b0;
        term_window_last = 1'b0;
    endtask

    task automatic check_acc(
        input int y,
        input int x,
        input int out
    );
        @(negedge clk_core);
        read_plane = '0;
        read_y = Y_W'(y);
        read_x = X_W'(x);
        read_out = OUT_W'(out);
        read_valid = 1'b1;
        @(posedge clk_core);
        if (!read_ready)
            $fatal(1, "read not ready");
        @(negedge clk_core);
        read_valid = 1'b0;
        wait (read_data_valid);
        #1;
        if (read_data !== ACC_W'(expected[y][x][out]))
            $fatal(
                1,
                "acc mismatch y=%0d x=%0d out=%0d got=%0d exp=%0d",
                y,
                x,
                out,
                read_data,
                expected[y][x][out]
            );
        @(negedge clk_core);
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
        run_start = 1'b0;
        run_accumulate = 1'b0;
        term_valid = 1'b0;
        term_source_plane = '0;
        term_source_y = '0;
        term_source_x = '0;
        term_lane = '0;
        term_gate = '0;
        term_destination_mask = '0;
        term_window_last = 1'b0;
        window_close = 1'b0;
        read_valid = 1'b0;
        read_plane = '0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        weight_scale = 1;
        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                for (int out = 0; out < OUT_DIM; out = out + 1)
                    expected[y][x][out] = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            for (int out = 0; out < OUT_DIM; out = out + 1) begin
                load_weight(
                    lane,
                    out,
                    (lane + 1) * (out == 0 ? 1 : -2),
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1
                );
            end
        end
        @(negedge clk_core);
        run_start = 1'b1;
        @(negedge clk_core);
        run_start = 1'b0;
        wait (term_ready);

`ifdef QFIT_EXHAUSTIVE_MASKS
        // Exhaust every non-empty role mask at an interior source. Linear-5
        // must replay conflicting masks without losing or duplicating updates.
        for (int mask = 1; mask < 32; mask = mask + 1)
            send_term(
                1,
                2,
                mask % HEAD_DIM,
                mask,
                5'(mask),
                1'b0
            );
        // Boundary subset also checks close/drain after a replay-heavy stream.
        send_term(0, 0, 0, 11, 5'b01011, 1'b1);
`else
        // Interior source: all five roles map to five distinct banks.
        send_term(1, 2, 1, 7, 5'b11111, 1'b0);
        // Same source, another lane and a subset multicast.
        send_term(1, 2, 3, 256, 5'b10101, 1'b0);
        // Corner source exercises valid subset and closes the window.
        send_term(0, 0, 0, 11, 5'b01011, 1'b1);
`endif
        wait (run_done);
        repeat (2) @(negedge clk_core);

        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                for (int out = 0; out < OUT_DIM; out = out + 1)
                    check_acc(y, x, out);
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
`ifdef QFIT_EXHAUSTIVE_MASKS
        if (
            perf_product_terms != 32
            || perf_destination_updates != 83
        )
`else
        if (
            perf_product_terms != 3
            || perf_destination_updates != 11
        )
`endif
            $fatal(
                1,
                "counter mismatch terms=%0d updates=%0d",
                perf_product_terms,
                perf_destination_updates
            );

`ifdef QFIT_WEIGHT_CONTEXT_RELOAD
        wait (weight_context_release_ready);
        @(negedge clk_core);
        weight_context_release = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        weight_context_release = 1'b0;
        wait (weight_ready);
        weight_scale = -1;
        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                for (int out = 0; out < OUT_DIM; out = out + 1)
                    expected[y][x][out] = 0;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            for (int out = 0; out < OUT_DIM; out = out + 1) begin
                load_weight(
                    lane,
                    out,
                    weight_scale
                    * (lane + 1) * (out == 0 ? 1 : -2),
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1
                );
            end
        end
        @(negedge clk_core);
        run_start = 1'b1;
        @(negedge clk_core);
        run_start = 1'b0;
        wait (term_ready);
        send_term(1, 2, 1, 7, 5'b11111, 1'b0);
        send_term(1, 2, 3, 256, 5'b10101, 1'b0);
        send_term(0, 0, 0, 11, 5'b01011, 1'b1);
        wait (run_done);
        repeat (2) @(negedge clk_core);
        for (int y = 0; y < HEIGHT; y = y + 1)
            for (int x = 0; x < WIDTH; x = x + 1)
                for (int out = 0; out < OUT_DIM; out = out + 1)
                    check_acc(y, x, out);
        if (protocol_error || perf_product_terms != 3
            || perf_destination_updates != 11)
            $fatal(1, "second weight context failed");
        $display(
            "PASS projection backend contexts=2 terms=%0d updates=%0d",
            perf_product_terms,
            perf_destination_updates
        );
`else
        $display(
            "PASS projection backend contexts=1 terms=%0d updates=%0d",
            perf_product_terms,
            perf_destination_updates
        );
`endif
        $finish;
    end
endmodule

`default_nettype wire
