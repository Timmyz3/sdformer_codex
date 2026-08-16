`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_score_quotient;
    localparam int PAIRS = 300;
    logic clk_core;
    logic rst_core;
    logic in_valid;
    logic in_ready;
    logic [8:0] in_pair_id;
    logic [63:0] in_q_pair;
    logic [63:0] in_k_pair;
    logic out_valid;
    logic out_ready;
    logic [8:0] out_pair_id;
    logic signed [15:0] out_score_q7;
    logic [1:0] out_temporal_mask;
    logic [1:0] out_active_mask;
    logic out_last;
    logic [31:0] perf_pairs;
    logic [31:0] perf_descriptors;
    logic [31:0] perf_equal_pairs;
    integer expected_descriptors;
    integer expected_equal;
    integer stall_counter;

    h67_temporal_score_quotient dut (.*);
    always #5 clk_core = ~clk_core;

    function automatic integer popcount32(input logic [31:0] value);
        integer count;
        count = 0;
        for (integer lane = 0; lane < 32; lane = lane + 1)
            count = count + value[lane];
        popcount32 = count;
    endfunction

    function automatic integer h67_score(
        input logic [31:0] q,
        input logic [31:0] k,
        input logic [31:0] peer
    );
        integer overlap;
        integer same_zero;
        integer motion;
        integer base;
        integer remainder;
        begin
            overlap = popcount32(q & k);
            same_zero = popcount32(~q & ~k);
            motion = popcount32(k ^ peer);
            base = 4 * overlap + motion + same_zero / 16;
            remainder = same_zero % 16;
            h67_score = base
                + ((remainder > 8) || ((remainder == 8) && (base & 1)));
        end
    endfunction

    task automatic send_and_check(
        input integer pair_id,
        input logic [31:0] q0,
        input logic [31:0] q1,
        input logic [31:0] k0,
        input logic [31:0] k1
    );
        integer score0;
        integer score1;
        logic [1:0] active;
        begin
            score0 = h67_score(q0, k0, k1);
            score1 = h67_score(q1, k1, k0);
            active = {|k1, |k0};
            @(negedge clk_core);
            while (!in_ready) @(negedge clk_core);
            in_pair_id = pair_id[8:0];
            in_q_pair = {q1, q0};
            in_k_pair = {k1, k0};
            in_valid = 1'b1;
            @(negedge clk_core);
            in_valid = 1'b0;

            do @(posedge clk_core); while (!(out_valid && out_ready));
            if (out_pair_id != pair_id[8:0])
                $fatal(1, "pair id mismatch");
            if (score0 == score1) begin
                if (
                    out_score_q7 != score0
                    || out_temporal_mask != 2'b11
                    || out_active_mask != active
                    || !out_last
                )
                    $fatal(
                        1,
                        "equal quotient mismatch pair=%0d score=%0d/%0d tmask=%b amask=%b/%b last=%b",
                        pair_id,
                        out_score_q7,
                        score0,
                        out_temporal_mask,
                        out_active_mask,
                        active,
                        out_last
                    );
                expected_descriptors = expected_descriptors + 1;
                expected_equal = expected_equal + 1;
            end else begin
                if (
                    out_score_q7 != score0
                    || out_temporal_mask != 2'b01
                    || out_active_mask != {1'b0, active[0]}
                    || out_last
                )
                    $fatal(
                        1,
                        "first descriptor mismatch pair=%0d score=%0d/%0d tmask=%b amask=%b/%b last=%b",
                        pair_id,
                        out_score_q7,
                        score0,
                        out_temporal_mask,
                        out_active_mask,
                        {1'b0, active[0]},
                        out_last
                    );
                do @(posedge clk_core); while (!(out_valid && out_ready));
                if (
                    out_score_q7 != score1
                    || out_temporal_mask != 2'b10
                    || out_active_mask != {active[1], 1'b0}
                    || !out_last
                )
                    $fatal(
                        1,
                        "second descriptor mismatch pair=%0d score=%0d/%0d tmask=%b amask=%b/%b last=%b",
                        pair_id,
                        out_score_q7,
                        score1,
                        out_temporal_mask,
                        out_active_mask,
                        {active[1], 1'b0},
                        out_last
                    );
                expected_descriptors = expected_descriptors + 2;
            end
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            out_ready <= 1'b0;
            stall_counter <= 0;
        end else begin
            stall_counter <= stall_counter + 1;
            out_ready <= (stall_counter % 5) != 2;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        in_valid = 1'b0;
        in_pair_id = '0;
        in_q_pair = '0;
        in_k_pair = '0;
        out_ready = 1'b0;
        expected_descriptors = 0;
        expected_equal = 0;
        stall_counter = 0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;

        send_and_check(0, 32'h0, 32'h0, 32'h0, 32'h0);
        send_and_check(1, 32'hffff0000, 32'hffff0000, 32'h00ff00ff, 32'h00ff00ff);
        send_and_check(2, 32'h00000001, 32'h80000000, 32'h00000001, 32'h00000000);
        for (int pair_id = 3; pair_id < PAIRS; pair_id = pair_id + 1)
            send_and_check(
                pair_id,
                $urandom,
                $urandom,
                $urandom,
                $urandom
            );

        repeat (3) @(negedge clk_core);
        if (
            perf_pairs != PAIRS
            || perf_descriptors != expected_descriptors
            || perf_equal_pairs != expected_equal
        )
            $fatal(
                1,
                "counter mismatch pairs=%0d descriptors=%0d/%0d equal=%0d/%0d",
                perf_pairs,
                perf_descriptors,
                expected_descriptors,
                perf_equal_pairs,
                expected_equal
            );
        $display(
            "PASS temporal quotient pairs=%0d descriptors=%0d equal=%0d",
            perf_pairs,
            perf_descriptors,
            perf_equal_pairs
        );
        $finish;
    end
endmodule

`default_nettype wire
