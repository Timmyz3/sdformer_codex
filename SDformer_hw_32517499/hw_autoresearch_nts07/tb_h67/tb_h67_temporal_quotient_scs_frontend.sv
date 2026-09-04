`timescale 1ns/1ps
`default_nettype none

module tb_h67_temporal_quotient_scs_frontend;
    localparam int PAIRS = 40;
    localparam int MAX_DESCRIPTORS = 80;
    localparam int MAX_SCORE = 162;
    localparam int COUNT_W = $clog2(2 * MAX_DESCRIPTORS + 1);
    localparam int CLASS_W = $clog2(MAX_SCORE + 1);

    logic clk_core;
    logic rst_core;
    logic window_start;
    logic window_seal;
    logic seal_ready;
    logic window_done;
    logic pair_valid;
    logic pair_ready;
    logic [8:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic class_valid;
    logic class_ready;
    logic [CLASS_W-1:0] class_score;
    logic [COUNT_W-1:0] class_multiplicity;
    logic class_last;
    logic active_valid;
    logic active_ready;
    logic [8:0] active_pair_id;
    logic signed [15:0] active_score_q7;
    logic [1:0] active_temporal_mask;
    logic [1:0] active_k_mask;
    logic active_last;
    logic signed [15:0] row_max_q7;
    logic protocol_error;
    logic [31:0] perf_pairs;
    logic [31:0] perf_quotient_descriptors;
    logic [31:0] perf_original_tokens;
    logic [31:0] perf_active_entries;
    logic [31:0] perf_equal_pairs;

    integer expected_hist [0:MAX_SCORE];
    integer expected_descriptors;
    integer expected_equal;
    integer expected_active_entries;
    integer expected_row_max;
    integer expected_class_count;
    integer expected_token_sum;
    integer observed_class_count;
    integer observed_token_sum;
    integer observed_active_count;
    integer cycle_count;
    integer exp_active_pair [0:MAX_DESCRIPTORS-1];
    integer exp_active_score [0:MAX_DESCRIPTORS-1];
    logic [1:0] exp_active_temporal [0:MAX_DESCRIPTORS-1];
    logic [1:0] exp_active_mask [0:MAX_DESCRIPTORS-1];
    logic [MAX_SCORE:0] seen_class;

    h67_temporal_quotient_scs_frontend #(
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer popcount32(input logic [31:0] value);
        integer count;
        begin
            count = 0;
            for (integer lane = 0; lane < 32; lane = lane + 1)
                count = count + value[lane];
            popcount32 = count;
        end
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
                + ((remainder > 8) || ((remainder == 8) && ((base & 1) != 0)));
        end
    endfunction

    task automatic record_active(
        input integer id,
        input integer score,
        input logic [1:0] temporal,
        input logic [1:0] mask
    );
        begin
            if (mask != 0) begin
                exp_active_pair[expected_active_entries] = id;
                exp_active_score[expected_active_entries] = score;
                exp_active_temporal[expected_active_entries] = temporal;
                exp_active_mask[expected_active_entries] = mask;
                expected_active_entries = expected_active_entries + 1;
            end
        end
    endtask

    task automatic send_pair(
        input integer id,
        input logic [31:0] q0,
        input logic [31:0] q1,
        input logic [31:0] k0,
        input logic [31:0] k1
    );
        integer score0;
        integer score1;
        logic [1:0] active_mask;
        begin
            score0 = h67_score(q0, k0, k1);
            score1 = h67_score(q1, k1, k0);
            active_mask = {|k1, |k0};
            expected_hist[score0] = expected_hist[score0] + 1;
            expected_hist[score1] = expected_hist[score1] + 1;
            if (score0 > expected_row_max) expected_row_max = score0;
            if (score1 > expected_row_max) expected_row_max = score1;
            if (score0 == score1) begin
                expected_descriptors = expected_descriptors + 1;
                expected_equal = expected_equal + 1;
                record_active(id, score0, 2'b11, active_mask);
            end else begin
                expected_descriptors = expected_descriptors + 2;
                record_active(id, score0, 2'b01, {1'b0, active_mask[0]});
                record_active(id, score1, 2'b10, {active_mask[1], 1'b0});
            end
            @(negedge clk_core);
            while (!pair_ready) @(negedge clk_core);
            pair_id = id[8:0];
            q_pair = {q1, q0};
            k_pair = {k1, k0};
            pair_valid = 1'b1;
            @(negedge clk_core);
            pair_valid = 1'b0;
        end
    endtask

    always @(negedge clk_core) begin
        if (rst_core) begin
            class_ready <= 1'b0;
            active_ready <= 1'b0;
            cycle_count <= 0;
        end else begin
            class_ready <= (cycle_count % 4) != 1;
            active_ready <= (cycle_count % 5) != 2;
            cycle_count <= cycle_count + 1;
        end
    end

    always @(posedge clk_core) begin
        if (!rst_core && class_valid && class_ready) begin
            if (
                seen_class[class_score]
                || class_multiplicity != expected_hist[class_score]
            )
                $fatal(1, "class mismatch score=%0d count=%0d/%0d",
                    class_score, class_multiplicity, expected_hist[class_score]);
            seen_class[class_score] = 1'b1;
            observed_class_count = observed_class_count + 1;
            observed_token_sum = observed_token_sum + class_multiplicity;
            if (class_last != (observed_class_count == expected_class_count))
                $fatal(1, "class last mismatch");
        end
        if (!rst_core && active_valid && active_ready) begin
            if (
                active_pair_id != exp_active_pair[observed_active_count]
                || active_score_q7 != exp_active_score[observed_active_count]
                || active_temporal_mask != exp_active_temporal[observed_active_count]
                || active_k_mask != exp_active_mask[observed_active_count]
            )
                $fatal(1, "active directory mismatch index=%0d", observed_active_count);
            observed_active_count = observed_active_count + 1;
            if (active_last != (observed_active_count == expected_active_entries))
                $fatal(1, "active last mismatch");
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        window_start = 1'b0;
        window_seal = 1'b0;
        pair_valid = 1'b0;
        pair_id = '0;
        q_pair = '0;
        k_pair = '0;
        class_ready = 1'b0;
        active_ready = 1'b0;
        expected_descriptors = 0;
        expected_equal = 0;
        expected_active_entries = 0;
        expected_row_max = 0;
        expected_class_count = 0;
        expected_token_sum = 2 * PAIRS;
        observed_class_count = 0;
        observed_token_sum = 0;
        observed_active_count = 0;
        cycle_count = 0;
        seen_class = '0;
        for (integer score = 0; score <= MAX_SCORE; score = score + 1)
            expected_hist[score] = 0;
        repeat (3) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        window_start = 1'b1;
        @(negedge clk_core);
        window_start = 1'b0;
        send_pair(0, '0, '0, '0, '0);
        send_pair(1, 32'hffff0000, 32'hffff0000,
                     32'h00ff00ff, 32'h00ff00ff);
        send_pair(2, 32'h1, 32'h80000000, 32'h1, '0);
        for (integer id = 3; id < PAIRS; id = id + 1)
            send_pair(
                id,
                $urandom,
                $urandom,
                (id % 4 == 0) ? '0 : $urandom,
                (id % 5 == 0) ? '0 : $urandom
            );
        for (integer score = 0; score <= MAX_SCORE; score = score + 1)
            if (expected_hist[score] != 0)
                expected_class_count = expected_class_count + 1;
        @(negedge clk_core);
        while (!seal_ready) @(negedge clk_core);
        window_seal = 1'b1;
        @(negedge clk_core);
        window_seal = 1'b0;
        wait (window_done);
        repeat (2) @(negedge clk_core);
        if (
            protocol_error
            || perf_pairs != PAIRS
            || perf_quotient_descriptors != expected_descriptors
            || perf_original_tokens != expected_token_sum
            || perf_active_entries != expected_active_entries
            || perf_equal_pairs != expected_equal
            || row_max_q7 != expected_row_max
            || observed_token_sum != expected_token_sum
            || observed_active_count != expected_active_entries
        )
            $fatal(
                1,
                "final mismatch pairs=%0d desc=%0d/%0d tokens=%0d/%0d active=%0d/%0d equal=%0d/%0d max=%0d/%0d",
                perf_pairs,
                perf_quotient_descriptors,
                expected_descriptors,
                perf_original_tokens,
                expected_token_sum,
                perf_active_entries,
                expected_active_entries,
                perf_equal_pairs,
                expected_equal,
                row_max_q7,
                expected_row_max
            );
        $display(
            "PASS quotient weighted-SCS pairs=%0d descriptors=%0d active=%0d classes=%0d",
            perf_pairs,
            perf_quotient_descriptors,
            perf_active_entries,
            observed_class_count
        );
        $finish;
    end
endmodule

`default_nettype wire
