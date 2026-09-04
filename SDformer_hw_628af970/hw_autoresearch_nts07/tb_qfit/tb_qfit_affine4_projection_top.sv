`timescale 1ns/1ps
`default_nettype none

module tb_qfit_affine4_projection_top;
    localparam int HEIGHT = 4;
    localparam int WIDTH = 6;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 3;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int Y_W = $clog2(HEIGHT);
    localparam int X_W = $clog2(WIDTH);
    localparam int PLANE_W = $clog2(TIME_PLANES);
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
    logic run_start;
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
    logic [31:0] perf_replay_updates;

    integer signed expected
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    logic [31:0] mask_seen;
    integer accepted_terms;
    integer accepted_updates;
    integer accepted_replays;

    qfit_affine4_projection_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W)
    ) dut (.*);

    /* verilator lint_off BLKSEQ */
    always #5 clk_core = ~clk_core;
    /* verilator lint_on BLKSEQ */

    function automatic integer signed weight_value(
        input integer lane,
        input integer out
    );
        case (out)
            0: weight_value = lane + 1;
            1: weight_value = -(lane + 2);
            default: weight_value = 3 * lane - 2;
        endcase
    endfunction

    function automatic integer destination_y(
        input integer source_y,
        input integer role
    );
        case (role)
            1: destination_y = source_y + 1;
            2: destination_y = source_y - 1;
            default: destination_y = source_y;
        endcase
    endfunction

    function automatic integer destination_x(
        input integer source_x,
        input integer role
    );
        case (role)
            3: destination_x = source_x + 1;
            4: destination_x = source_x - 1;
            default: destination_x = source_x;
        endcase
    endfunction

    function automatic integer mask_popcount(input logic [4:0] mask);
        integer count;
        begin
            count = 0;
            for (integer role = 0; role < 5; role = role + 1)
                count = count + (mask[role] ? 1 : 0);
            mask_popcount = count;
        end
    endfunction

    task automatic clear_scoreboard;
        begin
            for (
                integer plane = 0;
                plane < TIME_PLANES;
                plane = plane + 1
            )
                for (integer y = 0; y < HEIGHT; y = y + 1)
                    for (integer x = 0; x < WIDTH; x = x + 1)
                        for (
                            integer out = 0;
                            out < OUT_DIM;
                            out = out + 1
                        )
                            expected[plane][y][x][out] = 0;
            accepted_terms = 0;
            accepted_updates = 0;
            accepted_replays = 0;
        end
    endtask

    task automatic record_term(
        input logic [PLANE_W-1:0] plane,
        input integer sy,
        input integer sx,
        input integer lane,
        input integer gate,
        input logic [4:0] mask
    );
        integer dy;
        integer dx;
        begin
            accepted_terms = accepted_terms + 1;
            accepted_updates = accepted_updates + mask_popcount(mask);
            if (mask[1] && mask[2])
                accepted_replays = accepted_replays + 1;
            mask_seen[mask] = 1'b1;
            for (integer role = 0; role < 5; role = role + 1) begin
                if (mask[role]) begin
                    dy = destination_y(sy, role);
                    dx = destination_x(sx, role);
                    if (dy < 0 || dy >= HEIGHT || dx < 0 || dx >= WIDTH)
                        $fatal(
                            1,
                            "scoreboard received invalid boundary role=%0d",
                            role
                        );
                    for (
                        integer out = 0;
                        out < OUT_DIM;
                        out = out + 1
                    )
                        expected[plane][dy][dx][out] =
                            expected[plane][dy][dx][out]
                            + gate * weight_value(lane, out);
                end
            end
        end
    endtask

    task automatic load_weight(
        input integer lane,
        input integer out,
        input bit last
    );
        begin
            @(negedge clk_core);
            weight_lane = LANE_W'(lane);
            weight_out = OUT_W'(out);
            weight_data = W_W'(weight_value(lane, out));
            weight_last = last;
            weight_valid = 1'b1;
            @(posedge clk_core);
            if (!weight_ready)
                $fatal(1, "weight port unexpectedly blocked");
            @(negedge clk_core);
            weight_valid = 1'b0;
            weight_last = 1'b0;
        end
    endtask

    task automatic start_run;
        begin
            @(negedge clk_core);
            run_start = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            run_start = 1'b0;
            while (!term_ready)
                @(negedge clk_core);
        end
    endtask

    task automatic drive_term_wait(
        input logic [PLANE_W-1:0] plane,
        input integer sy,
        input integer sx,
        input integer lane,
        input integer gate,
        input logic [4:0] mask,
        input bit last,
        input bit legal_work
    );
        bit accepted;
        begin
            @(negedge clk_core);
            term_source_plane = PLANE_W'(plane);
            term_source_y = Y_W'(sy);
            term_source_x = X_W'(sx);
            term_lane = LANE_W'(lane);
            term_gate = GATE_W'(gate);
            term_destination_mask = mask;
            term_window_last = last;
            term_valid = 1'b1;
            accepted = 1'b0;
            while (!accepted) begin
                @(posedge clk_core);
                if (term_ready) begin
                    accepted = 1'b1;
                    if (legal_work)
                        record_term(plane, sy, sx, lane, gate, mask);
                    else
                        mask_seen[mask] = 1'b1;
                end
            end
            @(negedge clk_core);
            term_valid = 1'b0;
            term_window_last = 1'b0;
        end
    endtask

    task automatic drive_conflict_free_mask_sweep;
        integer sequence_idx;
        logic [PLANE_W-1:0] plane;
        integer sy;
        integer sx;
        integer lane;
        integer gate;
        begin
            sequence_idx = 0;
            for (integer mask = 1; mask < 32; mask = mask + 1) begin
                if ((mask & 32'h0000_0006) != 32'h0000_0006) begin
                    plane = PLANE_W'(sequence_idx & 1);
                    sy = 1 + (sequence_idx & 1);
                    sx = 1 + (sequence_idx % 4);
                    lane = sequence_idx % HEAD_DIM;
                    gate = 1 + ((sequence_idx * 17) % 255);
                    @(negedge clk_core);
                    term_source_plane = PLANE_W'(plane);
                    term_source_y = Y_W'(sy);
                    term_source_x = X_W'(sx);
                    term_lane = LANE_W'(lane);
                    term_gate = GATE_W'(gate);
                    term_destination_mask = 5'(mask);
                    term_window_last = 1'b0;
                    term_valid = 1'b1;
                    @(posedge clk_core);
                    if (!term_ready)
                        $fatal(
                            1,
                            "conflict-free term inserted a bubble mask=%02h",
                            mask
                        );
                    record_term(
                        plane,
                        sy,
                        sx,
                        lane,
                        gate,
                        5'(mask)
                    );
                    sequence_idx = sequence_idx + 1;
                end
            end
            @(negedge clk_core);
            term_valid = 1'b0;
        end
    endtask

    task automatic close_window;
        begin
            while (!window_close_ready)
                @(negedge clk_core);
            window_close = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            window_close = 1'b0;
        end
    endtask

    task automatic check_counters;
        begin
            if (
                perf_product_terms !== 32'(accepted_terms)
                || perf_destination_updates !== 32'(accepted_updates)
                || perf_replay_updates !== 32'(accepted_replays)
            )
                $fatal(
                    1,
                    "counter mismatch terms=%0d/%0d updates=%0d/%0d replay=%0d/%0d",
                    perf_product_terms,
                    accepted_terms,
                    perf_destination_updates,
                    accepted_updates,
                    perf_replay_updates,
                    accepted_replays
                );
        end
    endtask

    task automatic check_acc(
        input integer plane,
        input integer y,
        input integer x,
        input integer out
    );
        begin
            @(negedge clk_core);
            read_plane = PLANE_W'(plane);
            read_y = Y_W'(y);
            read_x = X_W'(x);
            read_out = OUT_W'(out);
            read_valid = 1'b1;
            @(posedge clk_core);
            if (!read_ready)
                $fatal(1, "read port unexpectedly blocked");
            #1;
            if (
                !read_data_valid
                || read_data !== ACC_W'(expected[plane][y][x][out])
            )
                $fatal(
                    1,
                    "acc mismatch p=%0d y=%0d x=%0d out=%0d got=%0d exp=%0d",
                    plane,
                    y,
                    x,
                    out,
                    read_data,
                    expected[plane][y][x][out]
                );
            @(negedge clk_core);
            read_valid = 1'b0;
        end
    endtask

    task automatic check_all_accumulators;
        begin
            for (
                integer plane = 0;
                plane < TIME_PLANES;
                plane = plane + 1
            )
                for (integer y = 0; y < HEIGHT; y = y + 1)
                    for (integer x = 0; x < WIDTH; x = x + 1)
                        for (
                            integer out = 0;
                            out < OUT_DIM;
                            out = out + 1
                        )
                            check_acc(plane, y, x, out);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        run_start = 1'b0;
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
        mask_seen = '0;
        clear_scoreboard();

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                load_weight(
                    lane,
                    out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1
                );

        // Zero mask is the 32nd mask value and is checked as a no-work error.
        start_run();
        drive_term_wait(0, 1, 2, 0, 7, 5'b00000, 1'b1, 1'b0);
        wait (run_done);
        if (!protocol_error)
            $fatal(1, "zero destination mask did not raise protocol_error");
        if (
            perf_product_terms != 0
            || perf_destination_updates != 0
            || perf_replay_updates != 0
        )
            $fatal(1, "zero destination mask produced work");

        // Interior mask sweep: 23 conflict-free masks are accepted every cycle.
        clear_scoreboard();
        start_run();
        drive_conflict_free_mask_sweep();
        for (integer mask = 1; mask < 32; mask = mask + 1) begin
            if ((mask & 32'h0000_0006) == 32'h0000_0006)
                drive_term_wait(
                    PLANE_W'(mask & 1),
                    1 + (mask & 1),
                    1 + (mask % 4),
                    mask % HEAD_DIM,
                    1 + ((mask * 13) % 255),
                    5'(mask),
                    mask == 31,
                    1'b1
                );
        end
        wait (run_done);
        #1;
        if (run_busy)
            $fatal(1, "run_busy remained asserted with run_done");
        if (protocol_error)
            $fatal(1, "unexpected protocol_error during mask sweep");
        check_counters();
        if (
            accepted_terms != 31
            || accepted_updates != 80
            || accepted_replays != 8
        )
            $fatal(
                1,
                "mask sweep conservation mismatch terms=%0d updates=%0d replay=%0d",
                accepted_terms,
                accepted_updates,
                accepted_replays
            );
        check_all_accumulators();

        // Boundary coordinates, both planes, and explicit close-drain.
        clear_scoreboard();
        start_run();
        drive_term_wait(0, 0, 0, 0, 9, 5'b01011, 1'b0, 1'b1);
        drive_term_wait(1, 0, WIDTH - 1, 1, 33, 5'b10011, 1'b0, 1'b1);
        drive_term_wait(0, HEIGHT - 1, 0, 2, 127, 5'b01101, 1'b0, 1'b1);
        drive_term_wait(
            1,
            HEIGHT - 1,
            WIDTH - 1,
            3,
            256,
            5'b10101,
            1'b0,
            1'b1
        );
        close_window();
        wait (run_done);
        if (protocol_error)
            $fatal(1, "unexpected protocol_error during boundary run");
        check_counters();
        check_all_accumulators();

        if (mask_seen !== 32'hffff_ffff)
            $fatal(1, "not all destination masks covered: %08h", mask_seen);

        $display(
            "PASS qfit_affine4_projection masks=32 main_terms=31 main_updates=80 main_replays=8 boundary_checks=%0d",
            TIME_PLANES * HEIGHT * WIDTH * OUT_DIM
        );
        $finish;
    end
endmodule

`default_nettype wire
