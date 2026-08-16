`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_direct_raw_multihead_projection_top;
    localparam int TOKENS = 162;
    localparam int LANES = 32;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic tile_start_valid, tile_start_ready;
    logic [15:0] tile_start_tag;
    logic [3:0] tile_start_output_tile;
    logic [2:0] tile_start_head_count;
    logic head_start_valid, head_start_ready;
    logic [15:0] head_start_tag, head_start_payload_tag;
    logic [2:0] head_start_index;
    logic [5:0] head_start_input_channel_base;
    logic head_start_last;
    logic raw_word_valid, raw_word_ready;
    logic [63:0] raw_word_data;
    logic [6:0] raw_word_index;
    logic raw_word_last;
    logic decoder_done_valid, decoder_done_ready;
    logic [15:0] decoder_done_payload_tag;
    logic decoder_done_error;
    logic head_done_valid, head_done_ready;
    logic [15:0] head_done_tag;
    logic [2:0] head_done_index;
    logic head_done_last, head_done_error;
    logic weight_req_valid, weight_req_ready;
    logic [15:0] weight_req_tag;
    logic [5:0] weight_req_input_channel;
    logic [3:0] weight_req_output_tile;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [15:0] weight_rsp_tag;
    logic [5:0] weight_rsp_input_channel;
    logic [3:0] weight_rsp_output_tile;
    logic [15:0] weight_rsp_weights;
    logic bias_req_valid, bias_req_ready;
    logic [15:0] bias_req_tag, bias_rsp_tag;
    logic [3:0] bias_req_output_tile;
    logic [7:0] bias_req_token_id, bias_rsp_token_id;
    logic bias_req_allow, bias_rsp_valid, bias_rsp_ready;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values, bias_lookup_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [(BANKS*8)-1:0] final_token_ids;
    logic [15:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic tile_done_valid, tile_done_ready;
    logic [15:0] tile_done_tag;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_heads, count_terms, count_completed_terms;
    logic [31:0] count_bias_commits, count_raw_records, count_raw_events;
    logic [6641:0] raw_packed_bits;
    logic [63:0] raw_stream_words [0:103];
    integer cycle_count, final_count, mismatch_count;
    integer head_completion_count, decoder_completion_count, bias_requests;

    gatestack_direct_raw_multihead_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .EVENT_WAYS(4),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS), .SEGMENT_TOKENS(18),
        .TAG_W(16), .INPUT_CH_W(6), .OUTPUT_TILE_W(4),
        .HEAD_COUNT_W(3), .TOKEN_ID_W(8), .LANE_ID_W(5)
    ) dut (.*);

    gatestack_bias_sram_model #(
        .TAG_W(16), .OUTPUT_TILE_W(4), .TOKEN_ID_W(8),
        .OUT_TILE(OUT_TILE), .ACC_W(ACC_W)
    ) bias_sram (
        .clk_core, .rst_core, .req_allow(bias_req_allow),
        .bias_req_valid, .bias_req_ready, .bias_req_tag,
        .bias_req_output_tile,
        .bias_req_token_id, .lookup_values(bias_lookup_values),
        .bias_rsp_valid, .bias_rsp_ready, .bias_rsp_tag,
        .bias_rsp_token_id, .bias_rsp_values
    );

    always #5 clk_core <= ~clk_core;

    function automatic signed [7:0] weight_lane(
        input logic [5:0] channel,
        input logic lane_value
    );
        begin
            case (channel)
                6'd2: weight_lane = lane_value ? -8'sd1 : 8'sd3;
                default: weight_lane = '0;
            endcase
        end
    endfunction

    task automatic send_raw_word(
        input logic [63:0] data_value,
        input logic [6:0] index_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            raw_word_data = data_value;
            raw_word_index = index_value;
            raw_word_last = last_value;
            raw_word_valid = 1'b1;
            do @(posedge clk_core); while (!raw_word_ready);
            @(negedge clk_core);
            raw_word_valid = 1'b0;
            raw_word_last = 1'b0;
        end
    endtask

    initial begin : weight_model
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_weights = '0;
        wait (!rst_core);
        forever begin
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid);
            weight_rsp_tag = weight_req_tag;
            weight_rsp_input_channel = weight_req_input_channel;
            weight_rsp_output_tile = weight_req_output_tile;
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            repeat (1) @(posedge clk_core);
            @(negedge clk_core);
            weight_rsp_weights = {
                weight_lane(weight_rsp_input_channel, 1'b1),
                weight_lane(weight_rsp_input_channel, 1'b0)
            };
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    end

    always_comb begin
        head_done_ready = (cycle_count % 5) != 2;
        decoder_done_ready = (cycle_count % 4) != 1;
        bias_req_allow = (cycle_count % 4) != 1;
        bias_lookup_values[31:0] = 32'(10 + 32'(bias_req_token_id));
        bias_lookup_values[63:32] = 32'(-20 - 32'(bias_req_token_id));
        final_ready[0] = (cycle_count % 5) != 2;
        final_ready[1] = (cycle_count % 3) != 1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            final_count <= 0;
            mismatch_count <= 0;
            head_completion_count <= 0;
            decoder_completion_count <= 0;
            bias_requests <= 0;
        end else begin
            integer fires;
            fires = 0;
            cycle_count <= cycle_count + 1;
            if (decoder_done_valid && decoder_done_ready) begin
                if (decoder_done_payload_tag != 16'h6800 || decoder_done_error)
                    $fatal(1, "Direct RAW decoder completion mismatch");
                decoder_completion_count <= decoder_completion_count + 1;
            end
            if (head_done_valid && head_done_ready) begin
                if (head_done_tag != 16'h7800 || head_done_index != 0 ||
                    !head_done_last || head_done_error)
                    $fatal(1, "Direct RAW head completion mismatch");
                head_completion_count <= head_completion_count + 1;
            end
            if (bias_req_valid && bias_req_ready)
                bias_requests <= bias_requests + 1;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    integer token_value;
                    integer signed expected0, expected1;
                    integer signed actual0, actual1;
                    token_value = 32'(final_token_ids[(bank*8) +: 8]);
                    expected0 = 10 + token_value;
                    expected1 = -20 - token_value;
                    if (token_value == 2 || token_value == 159) begin
                        expected0 = expected0 + 12;
                        expected1 = expected1 - 4;
                    end
                    actual0 = $signed(final_values[(bank*64) +: 32]);
                    actual1 = $signed(final_values[(bank*64)+32 +: 32]);
                    if (actual0 != expected0 || actual1 != expected1 ||
                        final_tag != 16'h7800) begin
                        mismatch_count <= mismatch_count + 1;
                        $fatal(1, "Direct RAW token=%0d got=(%0d,%0d) expected=(%0d,%0d)",
                               token_value, actual0, actual1, expected0, expected1);
                    end
                    fires = fires + 1;
                end
            end
            final_count <= final_count + fires;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = 1'b0;
        tile_start_tag = 16'h7800;
        tile_start_output_tile = 4'd1;
        tile_start_head_count = 3'd1;
        head_start_valid = 1'b0;
        head_start_tag = 16'h7800;
        head_start_payload_tag = 16'h6800;
        head_start_index = '0;
        head_start_input_channel_base = '0;
        head_start_last = 1'b1;
        raw_word_valid = 1'b0;
        raw_word_data = '0;
        raw_word_index = '0;
        raw_word_last = 1'b0;
        tile_done_ready = 1'b0;
        raw_packed_bits = '0;
        raw_packed_bits[(2*41) +: 32] = 32'h0000_0004;
        raw_packed_bits[(2*41)+32 +: 9] = 9'd4;
        raw_packed_bits[(159*41) +: 32] = 32'h0000_0004;
        raw_packed_bits[(159*41)+32 +: 9] = 9'd4;
        for (int word = 0; word < 104; word = word + 1)
            raw_stream_words[word] = raw_packed_bits[(word*64) +: 64];
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        tile_start_valid = 1'b1;
        do @(posedge clk_core); while (!tile_start_ready);
        @(negedge clk_core);
        tile_start_valid = 1'b0;

        @(negedge clk_core);
        head_start_valid = 1'b1;
        do @(posedge clk_core); while (!head_start_ready);
        @(negedge clk_core);
        head_start_valid = 1'b0;
        for (int word = 0; word < 104; word = word + 1)
            send_raw_word(raw_stream_words[word], 7'(word), word == 103);

        wait (tile_done_valid);
        @(negedge clk_core);
        if (tile_done_tag != 16'h7800 || protocol_error ||
            accumulator_overflow || mismatch_count != 0 ||
            count_heads != 1 || count_terms != 2 ||
            count_completed_terms != 2 || count_bias_commits != TOKENS ||
            count_raw_records != TOKENS || count_raw_events != 2 ||
            head_completion_count != 1 || decoder_completion_count != 1 ||
            bias_requests != TOKENS || final_count != TOKENS)
            $fatal(1, "Direct RAW count mismatch heads=%0d terms=%0d/%0d raw=%0d/%0d bias=%0d final=%0d protocol=%b",
                   count_heads, count_terms, count_completed_terms,
                   count_raw_records, count_raw_events, count_bias_commits,
                   final_count, protocol_error);
        tile_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: physically-stripped Direct RAW41 heads=%0d terms=%0d records=%0d events=%0d finals=%0d cycles=%0d",
                 count_heads, count_terms, count_raw_records,
                 count_raw_events, final_count, cycle_count);
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk_core);
        $fatal(1, "Direct RAW TB timeout");
    end
endmodule

`default_nettype wire
