`timescale 1ns/1ps
`default_nettype none

module tb_qfit_lane_product_cache_trace #(
    parameter int WAYS = 4,
    parameter int NO_REPLACE = 0
);
    localparam int LANES = 32;
    localparam int OUT_DIM = 4;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int PLANE_W = 1;
    localparam int Y_W = 4;
    localparam int X_W = 4;
    localparam int DEST_MASK_W = 5;
    localparam int LANE_W = $clog2(LANES);
    localparam int OUT_W = $clog2(OUT_DIM);
    localparam int PRODUCT_W = OUT_DIM * ACC_W;
    localparam int MAX_TERMS = 4096;
    localparam bit NO_REPLACE_B = NO_REPLACE != 0;

    logic clk_core;
    logic rst_core;
    logic weight_valid;
    logic weight_ready;
    logic [LANE_W-1:0] weight_lane;
    logic [OUT_W-1:0] weight_out;
    logic signed [W_W-1:0] weight_data;
    logic weight_last;
    logic epoch_start_valid;
    logic epoch_start_ready;
    logic epoch_close_valid;
    logic epoch_close_ready;
    logic epoch_active;
    logic epoch_done;
    logic in_valid;
    logic in_ready;
    logic [LANE_W-1:0] in_lane;
    logic [GATE_W-1:0] in_gate;
    logic [PLANE_W-1:0] in_source_plane;
    logic [Y_W-1:0] in_source_y;
    logic [X_W-1:0] in_source_x;
    logic [DEST_MASK_W-1:0] in_destination_mask;
    logic in_window_last;
    logic out_valid;
    logic out_ready;
    logic [LANE_W-1:0] out_lane;
    logic [GATE_W-1:0] out_gate;
    logic [PLANE_W-1:0] out_source_plane;
    logic [Y_W-1:0] out_source_y;
    logic [X_W-1:0] out_source_x;
    logic [DEST_MASK_W-1:0] out_destination_mask;
    logic out_window_last;
    logic [PRODUCT_W-1:0] out_product;
    logic protocol_error;
    logic [31:0] perf_accepted_terms;
    logic [31:0] perf_cache_hits;
    logic [31:0] perf_cache_misses;
    logic [31:0] perf_tag_compares;
    logic [31:0] perf_lru_writes;
    logic [31:0] perf_product_reads;
    logic [31:0] perf_product_writes;
    logic [31:0] perf_product_starts;
    logic [31:0] perf_weight_reads;
    logic [31:0] perf_output_stalls;

    integer exp_plane [0:MAX_TERMS-1];
    integer exp_y [0:MAX_TERMS-1];
    integer exp_x [0:MAX_TERMS-1];
    integer exp_lane [0:MAX_TERMS-1];
    integer exp_gate [0:MAX_TERMS-1];
    integer exp_mask [0:MAX_TERMS-1];
    integer exp_last [0:MAX_TERMS-1];
    integer exp_product [0:MAX_TERMS-1][0:OUT_DIM-1];
    integer exp_count;
    integer exp_head;
    logic [15:0] ready_lfsr;

    qfit_lane_product_cache_leaf #(
        .LANES(LANES),
        .WAYS(WAYS),
        .NO_REPLACE(NO_REPLACE_B),
        .OUT_DIM(OUT_DIM),
        .GATE_W(GATE_W),
        .W_W(W_W),
        .ACC_W(ACC_W),
        .PLANE_W(PLANE_W),
        .Y_W(Y_W),
        .X_W(X_W),
        .DEST_MASK_W(DEST_MASK_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer expected_weight(
        input integer lane,
        input integer out
    );
        if (lane == 0 && out == 0)
            expected_weight = -128;
        else if (lane == 0 && out == 1)
            expected_weight = 127;
        else
            expected_weight = ((lane % 7) - 3) * (out + 1);
    endfunction

    task automatic load_weight(
        input integer lane,
        input integer out,
        input logic last
    );
        @(negedge clk_core);
        weight_lane = LANE_W'(lane);
        weight_out = OUT_W'(out);
        weight_data = W_W'(expected_weight(lane, out));
        weight_last = last;
        weight_valid = 1'b1;
        #1;
        while (!weight_ready) begin
            @(negedge clk_core);
            #1;
        end
        @(negedge clk_core);
        weight_valid = 1'b0;
        weight_last = 1'b0;
    endtask

    task automatic start_epoch;
        @(negedge clk_core);
        epoch_start_valid = 1'b1;
        #1;
        while (!epoch_start_ready) begin
            @(negedge clk_core);
            #1;
        end
        @(negedge clk_core);
        epoch_start_valid = 1'b0;
    endtask

    task automatic close_epoch;
        @(negedge clk_core);
        epoch_close_valid = 1'b1;
        #1;
        while (!epoch_close_ready) begin
            @(negedge clk_core);
            #1;
        end
        @(negedge clk_core);
        epoch_close_valid = 1'b0;
    endtask

    always @(posedge clk_core) begin
        if (rst_core) begin
            ready_lfsr <= 16'h1ace;
            out_ready <= 1'b0;
            exp_head <= 0;
        end else begin
            ready_lfsr <= {
                ready_lfsr[14:0],
                ready_lfsr[15] ^ ready_lfsr[13]
                    ^ ready_lfsr[12] ^ ready_lfsr[10]
            };
            out_ready <= ready_lfsr[0] || ready_lfsr[3];
            if (out_valid && out_ready) begin
                if (exp_head >= exp_count)
                    $fatal(1, "trace出现额外输出");
                if (
                    integer'(out_source_plane) != exp_plane[exp_head]
                    || integer'(out_source_y) != exp_y[exp_head]
                    || integer'(out_source_x) != exp_x[exp_head]
                    || integer'(out_lane) != exp_lane[exp_head]
                    || integer'(out_gate) != exp_gate[exp_head]
                    || integer'(out_destination_mask) != exp_mask[exp_head]
                    || integer'(out_window_last) != exp_last[exp_head]
                )
                    $fatal(1, "trace payload失配 index=%0d", exp_head);
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    if (
                        signed'(out_product[out*ACC_W +: ACC_W])
                        != exp_product[exp_head][out]
                    )
                        $fatal(
                            1,
                            "trace product失配 index=%0d out=%0d got=%0d exp=%0d",
                            exp_head,
                            out,
                            signed'(out_product[out*ACC_W +: ACC_W]),
                            exp_product[exp_head][out]
                        );
                exp_head <= exp_head + 1;
            end
        end
    end

    initial begin
        string trace_path;
        integer trace_fd;
        integer header_count;
        reg [8*128-1:0] header_line;
        integer seq;
        integer plane;
        integer y;
        integer x;
        integer lane;
        integer gate;
        integer mask;
        integer window_last;
        integer expected_misses;
        integer expected_writes;
        integer timeout;

        if (!$value$plusargs("TRACE_CSV=%s", trace_path))
            $fatal(1, "缺少+TRACE_CSV");
        if (!$value$plusargs("EXPECTED_MISSES=%d", expected_misses))
            $fatal(1, "缺少+EXPECTED_MISSES");
        if (!$value$plusargs("EXPECTED_WRITES=%d", expected_writes))
            $fatal(1, "缺少+EXPECTED_WRITES");
        trace_fd = $fopen(trace_path, "r");
        if (trace_fd == 0)
            $fatal(1, "无法打开trace: %s", trace_path);
        header_count = $fgets(header_line, trace_fd);
        if (header_count == 0)
            $fatal(1, "trace缺少CSV表头");

        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        epoch_start_valid = 1'b0;
        epoch_close_valid = 1'b0;
        in_valid = 1'b0;
        in_lane = '0;
        in_gate = '0;
        in_source_plane = '0;
        in_source_y = '0;
        in_source_x = '0;
        in_destination_mask = '0;
        in_window_last = 1'b0;
        out_ready = 1'b0;
        exp_count = 0;
        exp_head = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        for (integer load_lane = 0; load_lane < LANES; load_lane++)
            for (integer out = 0; out < OUT_DIM; out++)
                load_weight(
                    load_lane,
                    out,
                    load_lane == LANES - 1 && out == OUT_DIM - 1
                );

        start_epoch();
        while (
            $fscanf(
                trace_fd,
                "%d,%d,%d,%d,%d,%d,%d,%d\n",
                seq,
                plane,
                y,
                x,
                lane,
                gate,
                mask,
                window_last
            ) == 8
        ) begin
            if (exp_count >= MAX_TERMS)
                $fatal(1, "trace超过MAX_TERMS=%0d", MAX_TERMS);
            if (seq != exp_count)
                $fatal(1, "trace seq不连续 got=%0d exp=%0d", seq, exp_count);
            @(negedge clk_core);
            in_source_plane = PLANE_W'(plane);
            in_source_y = Y_W'(y);
            in_source_x = X_W'(x);
            in_lane = LANE_W'(lane);
            in_gate = GATE_W'(gate);
            in_destination_mask = DEST_MASK_W'(mask);
            in_window_last = window_last != 0;
            in_valid = 1'b1;
            #1;
            while (!in_ready) begin
                @(negedge clk_core);
                #1;
            end
            exp_plane[exp_count] = plane;
            exp_y[exp_count] = y;
            exp_x[exp_count] = x;
            exp_lane[exp_count] = lane;
            exp_gate[exp_count] = gate;
            exp_mask[exp_count] = mask;
            exp_last[exp_count] = window_last;
            for (integer out = 0; out < OUT_DIM; out++)
                exp_product[exp_count][out] =
                    gate * expected_weight(lane, out);
            exp_count = exp_count + 1;
        end
        @(negedge clk_core);
        in_valid = 1'b0;
        in_window_last = 1'b0;
        $fclose(trace_fd);

        if (exp_count == 0)
            $fatal(1, "trace没有term");
        close_epoch();

        timeout = 0;
        while (!epoch_done && timeout < 10000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (!epoch_done)
            $fatal(1, "trace epoch超时");
        if (exp_head != exp_count)
            $fatal(
                1,
                "trace退休数错误 got=%0d exp=%0d",
                exp_head,
                exp_count
            );
        if (protocol_error)
            $fatal(1, "合法trace触发protocol_error");

        if (
            perf_accepted_terms != exp_count
            || perf_cache_misses != expected_misses
            || perf_cache_hits != exp_count - expected_misses
            || perf_tag_compares != exp_count * WAYS
            || perf_product_reads != exp_count - expected_misses
            || perf_product_writes != expected_writes
            || perf_product_starts != expected_misses
            || perf_weight_reads != expected_misses * OUT_DIM
            || (NO_REPLACE_B && perf_lru_writes != 0)
            || perf_output_stalls == 0
        )
            $fatal(
                1,
                "trace计数错误 W%0d accepted=%0d hit=%0d miss=%0d cmp=%0d rd=%0d wr=%0d start=%0d weight=%0d stall=%0d",
                WAYS,
                perf_accepted_terms,
                perf_cache_hits,
                perf_cache_misses,
                perf_tag_compares,
                perf_product_reads,
                perf_product_writes,
                perf_product_starts,
                perf_weight_reads,
                perf_output_stalls
            );

        $display(
            "PASS lane-product-cache trace W%0d policy=%0d terms=%0d hit=%0d miss=%0d compare=%0d writes=%0d lru_writes=%0d starts=%0d stalls=%0d",
            WAYS,
            NO_REPLACE,
            perf_accepted_terms,
            perf_cache_hits,
            perf_cache_misses,
            perf_tag_compares,
            perf_product_writes,
            perf_lru_writes,
            perf_product_starts,
            perf_output_stalls
        );
        $finish;
    end
endmodule

`default_nettype wire
