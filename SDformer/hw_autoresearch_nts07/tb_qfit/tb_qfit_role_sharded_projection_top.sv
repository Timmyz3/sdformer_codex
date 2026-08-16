`timescale 1ns/1ps
`default_nettype none

module tb_qfit_role_sharded_projection_top;
    localparam int HEIGHT = 3;
    localparam int WIDTH = 4;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 4;
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

    integer signed expected
        [0:TIME_PLANES-1][0:HEIGHT-1][0:WIDTH-1][0:OUT_DIM-1];
    logic [31:0] mask_seen;
    integer consecutive_terms;
    integer five_role_accepts;

    qfit_role_sharded_projection_top #(
        .HEIGHT(HEIGHT),
        .WIDTH(WIDTH),
        .TIME_PLANES(TIME_PLANES),
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    function automatic integer signed weight_value(
        input integer lane,
        input integer out
    );
        case (out)
            0: weight_value = lane + 1;
            1: weight_value = -2 * (lane + 1);
            2: weight_value = 3 * (lane + 1);
            default: weight_value = -4 * (lane + 1);
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

    task automatic record_expected(
        input logic [PLANE_W-1:0] plane,
        input integer source_y,
        input integer source_x,
        input integer lane,
        input integer gate,
        input logic [4:0] mask
    );
        integer dy;
        integer dx;
        begin
            for (integer role = 0; role < 5; role = role + 1) begin
                dy = destination_y(source_y, role);
                dx = destination_x(source_x, role);
                if (
                    mask[role]
                    && dy >= 0
                    && dy < HEIGHT
                    && dx >= 0
                    && dx < WIDTH
                ) begin
                    for (
                        integer out = 0;
                        out < OUT_DIM;
                        out = out + 1
                    ) begin
                        expected[plane][dy][dx][out] =
                            expected[plane][dy][dx][out]
                            + gate * weight_value(lane, out);
                    end
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
                $fatal(1, "weight interface rejected legal load");
            @(negedge clk_core);
            weight_valid = 1'b0;
            weight_last = 1'b0;
        end
    endtask

    task automatic send_term(
        input logic [PLANE_W-1:0] plane,
        input integer source_y,
        input integer source_x,
        input integer lane,
        input integer gate,
        input logic [4:0] mask
    );
        begin
            @(negedge clk_core);
            term_source_plane = PLANE_W'(plane);
            term_source_y = Y_W'(source_y);
            term_source_x = X_W'(source_x);
            term_lane = LANE_W'(lane);
            term_gate = GATE_W'(gate);
            term_destination_mask = mask;
            term_window_last = 1'b0;
            term_valid = 1'b1;
            while (!term_ready)
                @(negedge clk_core);
            @(posedge clk_core);
            record_expected(
                plane,
                source_y,
                source_x,
                lane,
                gate,
                mask
            );
            mask_seen[mask] = 1'b1;
            if (mask == 5'b11111)
                five_role_accepts = five_role_accepts + 1;
            @(negedge clk_core);
            term_valid = 1'b0;
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
                $fatal(1, "read interface rejected legal address");
            @(negedge clk_core);
            read_valid = 1'b0;
            if (!read_data_valid)
                wait (read_data_valid);
            #1;
            if (read_data !== ACC_W'(expected[plane][y][x][out])) begin
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
            end
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
        consecutive_terms = 0;
        five_role_accepts = 0;
        for (integer plane = 0; plane < TIME_PLANES; plane = plane + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1)
                    for (
                        integer out = 0;
                        out < OUT_DIM;
                        out = out + 1
                    )
                        expected[plane][y][x][out] = 0;

        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                load_weight(
                    lane,
                    out,
                    lane == HEAD_DIM - 1 && out == OUT_DIM - 1
                );

        @(negedge clk_core);
        run_start = 1'b1;
        @(negedge clk_core);
        run_start = 1'b0;
        wait (term_ready);

        // Interior source covers all 31 legal non-empty masks.
        for (integer mask = 1; mask < 32; mask = mask + 1)
            send_term(
                PLANE_W'(mask & 1),
                1,
                1,
                mask % HEAD_DIM,
                mask + 3,
                5'(mask)
            );

        // Twelve no-gap full-role terms prove one accepted term per cycle.
        @(negedge clk_core);
        term_valid = 1'b1;
        term_destination_mask = 5'b11111;
        term_window_last = 1'b0;
        for (integer item = 0; item < 12; item = item + 1) begin
            term_source_plane = PLANE_W'(item & 1);
            term_source_y = Y_W'(1);
            term_source_x = X_W'(1 + (item & 1));
            term_lane = LANE_W'(item % HEAD_DIM);
            term_gate = GATE_W'(17 + item);
            @(posedge clk_core);
            if (!term_ready)
                $fatal(1, "bubble in conflict-free role-sharded stream");
            record_expected(
                PLANE_W'(item & 1),
                1,
                1 + (item & 1),
                item % HEAD_DIM,
                17 + item,
                5'b11111
            );
            consecutive_terms = consecutive_terms + 1;
            five_role_accepts = five_role_accepts + 1;
            @(negedge clk_core);
        end
        term_valid = 1'b0;

        // Opposite boundaries use only geometrically legal roles.
        send_term(0, 0, 0, 1, 23, 5'b01011);
        send_term(1, HEIGHT - 1, WIDTH - 1, 2, 29, 5'b10101);

        // Explicit close must drain the final synchronous RMW writeback.
        @(negedge clk_core);
        window_close = 1'b1;
        if (!window_close_ready)
            $fatal(1, "close not accepted after final term");
        @(posedge clk_core);
        @(negedge clk_core);
        window_close = 1'b0;
        wait (run_done);
        #1;

        if (run_busy)
            $fatal(1, "run_done overlaps run_busy");
        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (mask_seen[0])
            $fatal(1, "empty mask was unexpectedly issued");
        if (mask_seen[31:1] !== {31{1'b1}})
            $fatal(1, "not all 31 non-empty masks were covered");
        if (consecutive_terms != 12)
            $fatal(1, "continuous term count mismatch");
        if (five_role_accepts < 13)
            $fatal(1, "five-role same-cycle coverage missing");
        if (
            perf_product_terms != 45
            || perf_destination_updates != 146
        ) begin
            $fatal(
                1,
                "counter mismatch terms=%0d updates=%0d",
                perf_product_terms,
                perf_destination_updates
            );
        end

        for (integer plane = 0; plane < TIME_PLANES; plane = plane + 1)
            for (integer y = 0; y < HEIGHT; y = y + 1)
                for (integer x = 0; x < WIDTH; x = x + 1)
                    for (
                        integer out = 0;
                        out < OUT_DIM;
                        out = out + 1
                    )
                        check_acc(plane, y, x, out);

        $display(
            "PASS role_sharded masks=31 terms=%0d updates=%0d continuous=%0d",
            perf_product_terms,
            perf_destination_updates,
            consecutive_terms
        );
        $finish;
    end
endmodule

`default_nettype wire
