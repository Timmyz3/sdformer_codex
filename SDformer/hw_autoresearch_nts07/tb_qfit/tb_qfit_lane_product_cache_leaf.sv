`timescale 1ns/1ps
`default_nettype none

module tb_qfit_lane_product_cache_leaf #(
    parameter int WAYS = 4,
    parameter int NO_REPLACE = 0
);
    localparam int LANES = 4;
    localparam int OUT_DIM = 2;
    localparam int GATE_W = 9;
    localparam int W_W = 8;
    localparam int ACC_W = 32;
    localparam int PLANE_W = 1;
    localparam int Y_W = 2;
    localparam int X_W = 4;
    localparam int DEST_MASK_W = 5;
    localparam int LANE_W = $clog2(LANES);
    localparam int OUT_W = $clog2(OUT_DIM);
    localparam int PRODUCT_W = OUT_DIM * ACC_W;
    localparam int MAX_EXPECTED = 64;
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

    integer exp_count;
    integer exp_head;
    integer exp_lane [0:MAX_EXPECTED-1];
    integer exp_gate [0:MAX_EXPECTED-1];
    integer exp_x [0:MAX_EXPECTED-1];
    integer exp_mask [0:MAX_EXPECTED-1];
    integer exp_last [0:MAX_EXPECTED-1];
    integer exp_product [0:MAX_EXPECTED-1][0:OUT_DIM-1];
    integer cycle_count;

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
            expected_weight = (lane + 1) * (out == 0 ? 1 : -2);
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
        if (!epoch_active)
            $fatal(1, "epoch did not become active");
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

    task automatic send_term(
        input integer lane,
        input integer gate,
        input integer x,
        input integer mask,
        input logic last
    );
        @(negedge clk_core);
        in_lane = LANE_W'(lane);
        in_gate = GATE_W'(gate);
        in_source_plane = '0;
        in_source_y = Y_W'(lane);
        in_source_x = X_W'(x);
        in_destination_mask = DEST_MASK_W'(mask);
        in_window_last = last;
        in_valid = 1'b1;
        #1;
        while (!in_ready) begin
            @(negedge clk_core);
            #1;
        end
        if (exp_count == MAX_EXPECTED)
            $fatal(1, "expected queue overflow");
        exp_lane[exp_count] = lane;
        exp_gate[exp_count] = gate;
        exp_x[exp_count] = x;
        exp_mask[exp_count] = mask;
        exp_last[exp_count] = integer'(last);
        for (integer out = 0; out < OUT_DIM; out = out + 1)
            exp_product[exp_count][out] =
                gate * expected_weight(lane, out);
        exp_count = exp_count + 1;
        @(negedge clk_core);
        in_valid = 1'b0;
        in_window_last = 1'b0;
    endtask

    task automatic wait_epoch_done;
        integer timeout;
        timeout = 0;
        while (!epoch_done && timeout < 2000) begin
            @(negedge clk_core);
            timeout = timeout + 1;
        end
        if (!epoch_done)
            $fatal(1, "cache epoch timeout");
        if (exp_head != exp_count)
            $fatal(
                1,
                "output queue mismatch head=%0d count=%0d",
                exp_head,
                exp_count
            );
    endtask

    always @(posedge clk_core) begin
        cycle_count <= cycle_count + 1;
        if (rst_core) begin
            out_ready <= 1'b0;
        end else begin
            out_ready <= (cycle_count % 4) != 1;
            if (out_valid && out_ready) begin
                if (exp_head >= exp_count)
                    $fatal(1, "unexpected cache output");
                if (
                    integer'(out_lane) != exp_lane[exp_head]
                    || integer'(out_gate) != exp_gate[exp_head]
                    || integer'(out_source_plane) != 0
                    || integer'(out_source_y) != exp_lane[exp_head]
                    || integer'(out_source_x) != exp_x[exp_head]
                    || integer'(out_destination_mask)
                        != exp_mask[exp_head]
                    || integer'(out_window_last)
                        != exp_last[exp_head]
                )
                    $fatal(1, "cache payload mismatch index=%0d", exp_head);
                for (integer out = 0; out < OUT_DIM; out = out + 1)
                    if (
                        signed'(out_product[out*ACC_W +: ACC_W])
                        != exp_product[exp_head][out]
                    )
                        $fatal(
                            1,
                            "product mismatch index=%0d out=%0d got=%0d exp=%0d",
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
        cycle_count = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;

        // An early weight_last is rejected; a canonical reload recovers.
        load_weight(0, 0, 1'b1);
        if (!protocol_error || epoch_start_ready)
            $fatal(1, "非法weight_last未被拒绝");

        for (integer lane = 0; lane < LANES; lane = lane + 1)
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                load_weight(
                    lane,
                    out,
                    lane == LANES - 1 && out == OUT_DIM - 1
                );

        start_epoch();
        for (integer gate = 1; gate <= WAYS; gate = gate + 1)
            send_term(0, gate, gate - 1, 1 << ((gate - 1) % 5), 0);
        send_term(0, 1, WAYS, 16, 0);
        send_term(0, WAYS + 1, WAYS + 1, 3, 0);
        send_term(0, 2, WAYS + 2, 5, 0);
        send_term(0, 1, WAYS + 3, 9, 0);
        send_term(1, 1, WAYS + 4, 17, 0);
        send_term(1, 1, WAYS + 5, 6, 1);
        close_epoch();
        wait_epoch_done();

        if (protocol_error)
            $fatal(1, "legal cache epoch raised protocol_error");
        if (
            perf_accepted_terms != WAYS + 6
            || perf_cache_hits != (NO_REPLACE_B ? 4 : 3)
            || perf_cache_misses != (NO_REPLACE_B ? WAYS + 2 : WAYS + 3)
            || perf_tag_compares != WAYS * (WAYS + 6)
            || perf_product_reads != (NO_REPLACE_B ? 4 : 3)
            || perf_product_writes != (NO_REPLACE_B ? WAYS + 1 : WAYS + 3)
            || perf_product_starts != (NO_REPLACE_B ? WAYS + 2 : WAYS + 3)
            || perf_weight_reads
                != OUT_DIM * (NO_REPLACE_B ? WAYS + 2 : WAYS + 3)
        )
            $fatal(
                1,
                "counter mismatch accepted=%0d hit=%0d miss=%0d cmp=%0d rd=%0d wr=%0d start=%0d weight=%0d",
                perf_accepted_terms,
                perf_cache_hits,
                perf_cache_misses,
                perf_tag_compares,
                perf_product_reads,
                perf_product_writes,
                perf_product_starts,
                perf_weight_reads
            );
        $display(
            "PASS lane-product-cache W%0d epoch1 accepted=%0d hit=%0d miss=%0d compares=%0d stalls=%0d",
            WAYS,
            perf_accepted_terms,
            perf_cache_hits,
            perf_cache_misses,
            perf_tag_compares,
            perf_output_stalls
        );

        // A new epoch invalidates all cache entries.
        exp_count = 0;
        exp_head = 0;
        start_epoch();
        send_term(0, 511, 10, 1, 1);
        close_epoch();
        wait_epoch_done();
        if (perf_cache_hits != 0 || perf_cache_misses != 1)
            $fatal(1, "epoch invalidation failed");

        $display(
            "PASS lane-product-cache W%0d epoch2 hit=%0d miss=%0d stalls=%0d lru_writes=%0d",
            WAYS,
            perf_cache_hits,
            perf_cache_misses,
            perf_output_stalls,
            perf_lru_writes
        );
        $finish;
    end
endmodule

`default_nettype wire
