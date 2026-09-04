`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_decoder_projection_top;
    localparam int TOKENS = 8;
    localparam int LANES = 32;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [15:0] group_tag;
    logic [1:0] group_route_select;
    logic [5:0] group_input_channel_base;
    logic [3:0] group_output_tile;
    logic [7:0] resident_term_count;
    logic [12:0] resident_event_count;
    logic resident_descriptor_valid, resident_descriptor_ready;
    logic [8:0] resident_descriptor_gate_code;
    logic [4:0] resident_descriptor_lane_id;
    logic [7:0] resident_descriptor_destination_count;
    logic [6:0] resident_descriptor_term_index;
    logic resident_descriptor_last;
    logic resident_word_valid, resident_word_ready;
    logic [63:0] resident_word_data;
    logic [6:0] resident_word_index;
    logic resident_word_last;
    logic ipd_word_valid, ipd_word_ready;
    logic [63:0] ipd_word_data;
    logic [6:0] ipd_word_index;
    logic ipd_word_last;
    logic raw_word_valid, raw_word_ready;
    logic [63:0] raw_word_data;
    logic [6:0] raw_word_index;
    logic raw_word_last;
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
    logic group_done_valid, group_done_ready;
    logic [15:0] group_done_tag;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_terms, count_completed_terms, count_bias_commits;
    logic [383:0] raw_packed_bits;
    logic [63:0] raw_stream_words [0:5];
    integer cycle_count;
    integer current_route;
    integer current_token_a, current_token_b;
    integer current_product0, current_product1;
    integer finals_in_group;
    integer mismatch_count;
    integer bias_requests;

    gatestack_decoder_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .MAX_TERMS(128),
        .RESIDENT_TERMS(80), .EVENT_WAYS(4), .OUT_TILE(OUT_TILE),
        .BANKS(BANKS), .SEGMENT_TOKENS(4), .TAG_W(16),
        .INPUT_CH_W(6), .OUTPUT_TILE_W(4), .TOKEN_ID_W(8),
        .LANE_ID_W(5), .RES_TERM_IDX_W(7), .IPD_TERM_IDX_W(7)
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

    function automatic logic [31:0] descriptor(
        input logic [8:0] gate_value,
        input logic [4:0] lane_value,
        input logic [7:0] count_value
    );
        descriptor = 32'(gate_value) | (32'(lane_value) << 9) |
                     (32'(count_value) << 14);
    endfunction

    function automatic logic [63:0] header0(input logic [15:0] tag_value);
        header0 = (64'(tag_value) << 32) | (64'(1) << 20) |
                  (64'(1) << 16) | 64'h4753;
    endfunction

    function automatic logic [63:0] header1;
        header1 = 64'(208) | (64'(1) << 13) | (64'(2) << 21) |
                  (64'(1) << 34) | (64'(2) << 37) | (64'(24) << 45);
    endfunction

    function automatic signed [7:0] weight_lane(
        input logic [5:0] channel,
        input logic lane_value
    );
        begin
            case (channel)
                6'd0: weight_lane = lane_value ? 8'sd2 : 8'sd1;
                6'd1: weight_lane = lane_value ? 8'sd4 : -8'sd2;
                6'd2: weight_lane = lane_value ? -8'sd1 : 8'sd3;
                default: weight_lane = '0;
            endcase
        end
    endfunction

    task automatic start_group(
        input logic [1:0] route_value,
        input logic [15:0] tag_value,
        input logic [8:0] gate_value,
        input logic [4:0] lane_value,
        input logic [7:0] token_a,
        input logic [7:0] token_b
    );
        begin
            current_route = 32'(route_value);
            current_token_a = 32'(token_a);
            current_token_b = 32'(token_b);
            current_product0 = $signed({1'b0, gate_value}) *
                $signed(weight_lane(6'(lane_value), 1'b0));
            current_product1 = $signed({1'b0, gate_value}) *
                $signed(weight_lane(6'(lane_value), 1'b1));
            finals_in_group = 0;
            @(negedge clk_core);
            group_route_select = route_value;
            group_tag = tag_value;
            group_valid = 1'b1;
            do @(posedge clk_core); while (!group_ready);
            @(negedge clk_core);
            group_valid = 1'b0;
        end
    endtask

    task automatic finish_group(input logic [15:0] tag_value);
        begin
            wait (group_done_valid);
            @(negedge clk_core);
            if (finals_in_group != TOKENS || group_done_tag != tag_value ||
                protocol_error || accumulator_overflow)
                $fatal(1, "decoder route %0d completion mismatch finals=%0d protocol=%b",
                       current_route, finals_in_group, protocol_error);
            group_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            group_done_ready = 1'b0;
        end
    endtask

    task automatic send_ipd_word(
        input logic [63:0] data_value,
        input logic [6:0] index_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            ipd_word_data = data_value;
            ipd_word_index = index_value;
            ipd_word_last = last_value;
            ipd_word_valid = 1'b1;
            do @(posedge clk_core); while (!ipd_word_ready);
            @(negedge clk_core);
            ipd_word_valid = 1'b0;
            ipd_word_last = 1'b0;
        end
    endtask

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
        bias_req_allow = (cycle_count % 4) != 1;
        bias_lookup_values[31:0] = 32'(10 + 32'(bias_req_token_id));
        bias_lookup_values[63:32] = 32'(-20 - 32'(bias_req_token_id));
        final_ready[0] = (cycle_count % 5) != 2;
        final_ready[1] = (cycle_count % 3) != 1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            mismatch_count <= 0;
            bias_requests <= 0;
        end else begin
            integer final_fire_count;
            final_fire_count = 0;
            cycle_count <= cycle_count + 1;
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
                    if (token_value == current_token_a ||
                        token_value == current_token_b) begin
                        expected0 = expected0 + current_product0;
                        expected1 = expected1 + current_product1;
                    end
                    actual0 = $signed(final_values[(bank*64) +: 32]);
                    actual1 = $signed(final_values[(bank*64)+32 +: 32]);
                    if (actual0 != expected0 || actual1 != expected1 ||
                        final_tag != 16'h7500 + 16'(current_route)) begin
                        mismatch_count <= mismatch_count + 1;
                        $fatal(1, "route=%0d token=%0d got=(%0d,%0d) expected=(%0d,%0d)",
                               current_route, token_value, actual0, actual1,
                               expected0, expected1);
                    end
                    final_fire_count = final_fire_count + 1;
                end
            end
            finals_in_group <= finals_in_group + final_fire_count;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = '0;
        group_route_select = '0;
        group_input_channel_base = '0;
        group_output_tile = 4'd1;
        resident_term_count = 8'd1;
        resident_event_count = 13'd2;
        resident_descriptor_valid = 1'b0;
        resident_descriptor_gate_code = '0;
        resident_descriptor_lane_id = '0;
        resident_descriptor_destination_count = '0;
        resident_descriptor_term_index = '0;
        resident_descriptor_last = 1'b0;
        resident_word_valid = 1'b0;
        resident_word_data = '0;
        resident_word_index = '0;
        resident_word_last = 1'b0;
        ipd_word_valid = 1'b0;
        ipd_word_data = '0;
        ipd_word_index = '0;
        ipd_word_last = 1'b0;
        raw_word_valid = 1'b0;
        raw_word_data = '0;
        raw_word_index = '0;
        raw_word_last = 1'b0;
        group_done_ready = 1'b0;
        current_route = 0;
        current_token_a = 0;
        current_token_b = 0;
        current_product0 = 0;
        current_product1 = 0;
        finals_in_group = 0;
        raw_packed_bits = '0;
        raw_packed_bits[(2*41) +: 32] = 32'h0000_0004;
        raw_packed_bits[(2*41)+32 +: 9] = 9'd4;
        raw_packed_bits[(5*41) +: 32] = 32'h0000_0004;
        raw_packed_bits[(5*41)+32 +: 9] = 9'd4;
        for (int word = 0; word < 6; word = word + 1)
            raw_stream_words[word] = raw_packed_bits[(word*64) +: 64];
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        // Resident descriptors plus token-only replay.
        start_group(2'd0, 16'h7500, 9'd2, 5'd0, 8'd0, 8'd7);
        @(negedge clk_core);
        resident_descriptor_gate_code = 9'd2;
        resident_descriptor_lane_id = 5'd0;
        resident_descriptor_destination_count = 8'd2;
        resident_descriptor_term_index = 7'd0;
        resident_descriptor_last = 1'b1;
        resident_descriptor_valid = 1'b1;
        do @(posedge clk_core); while (!resident_descriptor_ready);
        @(negedge clk_core);
        resident_descriptor_valid = 1'b0;
        resident_word_data = 64'h0000_0000_0000_0700;
        resident_word_index = 7'd0;
        resident_word_last = 1'b1;
        resident_word_valid = 1'b1;
        do @(posedge clk_core); while (!resident_word_ready);
        @(negedge clk_core);
        resident_word_valid = 1'b0;
        resident_word_last = 1'b0;
        finish_group(16'h7500);

        // Sequential IPD32W full payload.
        start_group(2'd1, 16'h7501, 9'd3, 5'd1, 8'd1, 8'd6);
        send_ipd_word(header0(16'h7501), 7'd0, 1'b0);
        send_ipd_word(header1(), 7'd1, 1'b0);
        send_ipd_word({32'd0, descriptor(9'd3, 5'd1, 8'd2)},
                      7'd2, 1'b0);
        send_ipd_word(64'h0000_0000_0000_0601, 7'd3, 1'b1);
        finish_group(16'h7501);

        // RAW has active tokens 2/5 followed by K-zero records 6/7. The tail
        // retimer must still mark token 5 as the true final event.
        start_group(2'd2, 16'h7502, 9'd4, 5'd2, 8'd2, 8'd5);
        for (int word = 0; word < 6; word = word + 1)
            send_raw_word(raw_stream_words[word], 7'(word), word == 5);
        finish_group(16'h7502);

        if (mismatch_count != 0 || count_terms != 4 ||
            count_completed_terms != 4 || count_bias_commits != 24 ||
            bias_requests != 24 || protocol_error)
            $fatal(1, "decoder projection cumulative mismatch terms=%0d completed=%0d bias=%0d protocol=%b",
                   count_terms, count_completed_terms,
                   count_bias_commits, protocol_error);
        $display("PASS: actual three-decoder projection terms=%0d completed=%0d bias=%0d cycles=%0d",
                 count_terms, count_completed_terms,
                 count_bias_commits, cycle_count);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "decoder projection TB timeout");
    end
endmodule

`default_nettype wire
