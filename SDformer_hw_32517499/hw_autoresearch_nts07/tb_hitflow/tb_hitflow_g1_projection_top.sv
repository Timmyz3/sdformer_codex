`timescale 1ns/1ps
`default_nettype none

// End-to-end G1 projection: NMF directory path vs direct dense integer model.
module tb_hitflow_g1_projection_top;
    localparam int TOKENS = 6;
    localparam int LANES = 4;
    localparam int SLOTS = 4;
    localparam int GATE_W = 9;
    localparam int WEIGHT_W = 8;
    localparam int PRODUCT_W = 17;
    localparam int ACC_W = 32;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int SEGMENT_TOKENS = 2;
    localparam int TAG_W = 16;
    localparam int TOKEN_ID_W = $clog2(TOKENS);
    localparam int LANE_ID_W = $clog2(LANES);

    logic clk_core = 1'b0;
    logic rst_core;
    logic group_valid;
    logic group_ready;
    logic [TAG_W-1:0] group_tag;
    logic token_valid;
    logic token_ready;
    logic [TOKEN_ID_W-1:0] token_id;
    logic [GATE_W-1:0] token_gate_code;
    logic [LANES-1:0] token_k_bits;
    logic token_last;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [LANE_ID_W-1:0] weight_req_input_channel;
    logic [3:0] weight_req_output_tile;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [LANE_ID_W-1:0] weight_rsp_input_channel;
    logic [3:0] weight_rsp_output_tile;
    logic [(OUT_TILE*WEIGHT_W)-1:0] weight_rsp_weights;
    logic bias_req_valid;
    logic bias_req_ready;
    logic [TOKEN_ID_W-1:0] bias_req_token_id;
    logic [(OUT_TILE*PRODUCT_W)-1:0] bias_req_values;
    logic [BANKS-1:0] final_valid;
    logic [BANKS-1:0] final_ready;
    logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [TAG_W-1:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic group_done_valid;
    logic group_done_ready;
    logic [TAG_W-1:0] group_done_tag;
    logic overflow_seen;
    logic protocol_error;
    logic accumulator_overflow;
    logic [31:0] count_tokens;
    logic [31:0] count_terms;
    logic [31:0] count_products;
    logic [31:0] count_bias_commits;

    logic [GATE_W-1:0] gates [0:TOKENS-1];
    logic [LANES-1:0] kbits [0:TOKENS-1];
    logic signed [WEIGHT_W-1:0] weights [0:LANES-1][0:OUT_TILE-1];
    logic signed [PRODUCT_W-1:0] biases [0:TOKENS-1][0:OUT_TILE-1];
    logic signed [ACC_W-1:0] expected [0:TOKENS-1][0:OUT_TILE-1];
    logic [TOKENS-1:0] final_seen;
    integer errors;

    initial begin
        forever #1 clk_core = ~clk_core;
    end

    hitflow_g1_projection_top #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .SLOTS(SLOTS),
        .GATE_W(GATE_W),
        .WEIGHT_W(WEIGHT_W),
        .PRODUCT_W(PRODUCT_W),
        .ACC_W(ACC_W),
        .OUT_TILE(OUT_TILE),
        .BANKS(BANKS),
        .SEGMENT_TOKENS(SEGMENT_TOKENS),
        .TAG_W(TAG_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(LANE_ID_W),
        .INPUT_CH_W(LANE_ID_W),
        .OUTPUT_TILE_W(4)
    ) dut (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .group_valid(group_valid),
        .group_ready(group_ready),
        .group_tag(group_tag),
        .token_valid(token_valid),
        .token_ready(token_ready),
        .token_id(token_id),
        .token_gate_code(token_gate_code),
        .token_k_bits(token_k_bits),
        .token_last(token_last),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready),
        .weight_req_tag(weight_req_tag),
        .weight_req_input_channel(weight_req_input_channel),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready),
        .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_channel(weight_rsp_input_channel),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_weights(weight_rsp_weights),
        .bias_req_valid(bias_req_valid),
        .bias_req_ready(bias_req_ready),
        .bias_req_token_id(bias_req_token_id),
        .bias_req_values(bias_req_values),
        .final_valid(final_valid),
        .final_ready(final_ready),
        .final_token_ids(final_token_ids),
        .final_tag(final_tag),
        .final_values(final_values),
        .group_done_valid(group_done_valid),
        .group_done_ready(group_done_ready),
        .group_done_tag(group_done_tag),
        .overflow_seen(overflow_seen),
        .protocol_error(protocol_error),
        .accumulator_overflow(accumulator_overflow),
        .count_tokens(count_tokens),
        .count_terms(count_terms),
        .count_products(count_products),
        .count_bias_commits(count_bias_commits)
    );

    // Simple weight slave: always ready, respond one cycle later, hold until taken.
    always_ff @(posedge clk_core) begin
        integer o;
        if (rst_core) begin
            weight_req_ready <= 1'b1;
            weight_rsp_valid <= 1'b0;
            weight_rsp_tag <= '0;
            weight_rsp_input_channel <= '0;
            weight_rsp_output_tile <= '0;
            weight_rsp_weights <= '0;
        end else begin
            weight_req_ready <= 1'b1;
            if (weight_rsp_valid && weight_rsp_ready) begin
                weight_rsp_valid <= 1'b0;
            end
            if (weight_req_valid && weight_req_ready) begin
                weight_rsp_valid <= 1'b1;
                weight_rsp_tag <= weight_req_tag;
                weight_rsp_input_channel <= weight_req_input_channel;
                weight_rsp_output_tile <= weight_req_output_tile;
                for (o = 0; o < OUT_TILE; o = o + 1) begin
                    weight_rsp_weights[(o*WEIGHT_W) +: WEIGHT_W] <=
                        weights[weight_req_input_channel][o];
                end
            end
        end
    end

    // Bias provider.
    always_comb begin
        bias_req_ready = bias_req_valid;
        for (int o = 0; o < OUT_TILE; o = o + 1) begin
            bias_req_values[(o*PRODUCT_W) +: PRODUCT_W] =
                biases[bias_req_token_id][o];
        end
    end

    always_ff @(posedge clk_core) begin
        integer b;
        integer o;
        integer tid;
        logic signed [ACC_W-1:0] got;
        if (rst_core) begin
            final_seen <= '0;
        end else begin
            for (b = 0; b < BANKS; b = b + 1) begin
                if (final_valid[b] && final_ready[b]) begin
                    tid = final_token_ids[(b*TOKEN_ID_W) +: TOKEN_ID_W];
                    if (final_seen[tid]) begin
                        $error("token %0d final repeated", tid);
                        errors = errors + 1;
                    end
                    final_seen[tid] <= 1'b1;
                    for (o = 0; o < OUT_TILE; o = o + 1) begin
                        got = $signed(final_values[
                            (b*OUT_TILE*ACC_W) + (o*ACC_W) +: ACC_W
                        ]);
                        if (got !== expected[tid][o]) begin
                            $error("token %0d out %0d got %0d expected %0d",
                                   tid, o, got, expected[tid][o]);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end
    end

    task automatic recompute_expected;
        integer t, l, o;
        begin
            for (t = 0; t < TOKENS; t = t + 1) begin
                for (o = 0; o < OUT_TILE; o = o + 1) begin
                    expected[t][o] = biases[t][o];
                    for (l = 0; l < LANES; l = l + 1) begin
                        if (kbits[t][l]) begin
                            expected[t][o] = expected[t][o] +
                                $signed({1'b0, gates[t]}) * weights[l][o];
                        end
                    end
                end
            end
        end
    endtask

    task automatic run_group(input logic [15:0] tag);
        integer t;
        integer guard;
        begin
            recompute_expected();
            final_seen = '0;
            group_tag = tag;
            group_valid = 1'b1;
            guard = 0;
            do begin
                @(posedge clk_core);
                guard = guard + 1;
                if (guard > 1000) $fatal(1, "timeout group_ready tag=%h top=%0d nmf=%0d",
                                          tag, dut.state_q, dut.u_nmf.state_q);
            end while (!group_ready);
            #0.1 group_valid = 1'b0;

            for (t = 0; t < TOKENS; t = t + 1) begin
                token_id = t[TOKEN_ID_W-1:0];
                token_gate_code = gates[t];
                token_k_bits = kbits[t];
                token_last = (t == (TOKENS - 1));
                token_valid = 1'b1;
                guard = 0;
                do begin
                    @(posedge clk_core);
                    guard = guard + 1;
                    if (guard > 1000) $fatal(1, "timeout token %0d tag=%h", t, tag);
                end while (!token_ready);
                #0.1 token_valid = 1'b0;
                token_last = 1'b0;
            end

            guard = 0;
            while (!group_done_valid) begin
                @(posedge clk_core);
                guard = guard + 1;
                if (guard > 5000) begin
                    $fatal(1, "timeout group_done tag=%h top=%0d nmf=%0d",
                           tag, dut.state_q, dut.u_nmf.state_q);
                end
            end
            if (group_done_tag !== tag) begin
                $error("group done tag mismatch");
                errors = errors + 1;
            end
            if (overflow_seen) begin
                $error("unexpected overflow/fallback");
                errors = errors + 1;
            end
            if (protocol_error) begin
                $error("protocol_error sticky");
                errors = errors + 1;
            end
            if (final_seen != {TOKENS{1'b1}}) begin
                $error("missing finals seen=%b", final_seen);
                errors = errors + 1;
            end
            // Per-group bias commits are cumulative in the DUT counter; only
            // check that finals covered all tokens.
            group_done_ready = 1'b1;
            guard = 0;
            do begin
                @(posedge clk_core);
                guard = guard + 1;
                if (guard > 20) $fatal(1, "timeout leaving ST_DONE");
            end while (group_done_valid);
            group_done_ready = 1'b0;
            @(posedge clk_core);
        end
    endtask

    initial begin
        errors = 0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = '0;
        token_valid = 1'b0;
        token_id = '0;
        token_gate_code = '0;
        token_k_bits = '0;
        token_last = 1'b0;
        final_ready = {BANKS{1'b1}};
        group_done_ready = 1'b0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_weights = '0;

        // Weights and biases.
        weights[0][0] = 8'sd2;   weights[0][1] = -8'sd1;
        weights[1][0] = 8'sd4;   weights[1][1] = 8'sd3;
        weights[2][0] = -8'sd5;  weights[2][1] = 8'sd7;
        weights[3][0] = 8'sd1;   weights[3][1] = -8'sd8;
        for (int t = 0; t < TOKENS; t = t + 1) begin
            biases[t][0] = PRODUCT_W'(t + 1);
            biases[t][1] = -PRODUCT_W'(t);
        end

        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        // Case 1: shared gate+lane multicast (tokens 0 and 1 same gate, lane0).
        gates[0] = 9'd10; kbits[0] = 4'b0001;
        gates[1] = 9'd10; kbits[1] = 4'b0001;
        gates[2] = 9'd20; kbits[2] = 4'b0010;
        gates[3] = 9'd30; kbits[3] = 4'b0100;
        gates[4] = 9'd0;  kbits[4] = 4'b1111; // gate0 filtered
        gates[5] = 9'd40; kbits[5] = 4'b0000; // kzero filtered
        $display("CASE1 shared-gate multicast + filters");
        run_group(16'hA001);
        if (count_terms < 1) begin
            $error("expected directory terms");
            errors = errors + 1;
        end

        // Case 2: denser unique-ish gates within SLOTS.
        gates[0] = 9'd3;  kbits[0] = 4'b0101;
        gates[1] = 9'd3;  kbits[1] = 4'b1001;
        gates[2] = 9'd5;  kbits[2] = 4'b0011;
        gates[3] = 9'd7;  kbits[3] = 4'b1110;
        gates[4] = 9'd9;  kbits[4] = 4'b0001;
        gates[5] = 9'd9;  kbits[5] = 4'b0100;
        $display("CASE2 multi-lane shared gates");
        run_group(16'hA002);

        // Case 3: max gate * int8 edge product (≤SLOTS unique gates).
        gates[0] = 9'd256; kbits[0] = 4'b0001;
        gates[1] = 9'd256; kbits[1] = 4'b0001;
        gates[2] = 9'd1;   kbits[2] = 4'b0010;
        gates[3] = 9'd2;   kbits[3] = 4'b0100;
        gates[4] = 9'd3;   kbits[4] = 4'b1000;
        gates[5] = 9'd3;   kbits[5] = 4'b0001; // share gate with token4
        weights[0][0] = -8'sd128;
        weights[0][1] = 8'sd127;
        weights[1][0] = 8'sd1;
        weights[1][1] = 8'sd2;
        weights[2][0] = 8'sd3;
        weights[2][1] = 8'sd4;
        weights[3][0] = 8'sd5;
        weights[3][1] = 8'sd6;
        $display("CASE3 gate=256 edge products");
        run_group(16'hA003);

        if (errors == 0) begin
            $display("PASS: G1 projection top direct/NMF integer equivalence");
            $finish;
        end else begin
            $fatal(1, "FAIL: %0d errors", errors);
        end
    end
endmodule

`default_nettype wire
