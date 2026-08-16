`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_multihead_tile_projection_top;
    localparam int TOKENS = 8;
    localparam int LANES = 4;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic tile_start_valid, tile_start_ready;
    logic [15:0] tile_start_tag;
    logic [3:0] tile_start_output_tile;
    logic [2:0] tile_start_head_count;
    logic head_start_valid, head_start_ready;
    logic [15:0] head_start_tag;
    logic [2:0] head_start_index;
    logic [5:0] head_start_input_channel_base;
    logic head_start_last;
    logic term_valid, term_ready;
    logic [8:0] term_gate_code;
    logic [1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic event_valid, event_ready;
    logic [8:0] event_gate_code;
    logic [1:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [11:0] event_token_ids;
    logic [2:0] event_count;
    logic [12:0] event_issue_seq;
    logic event_term_first, event_term_last, event_head_last;
    logic source_done_valid, source_done_ready;
    logic [15:0] source_done_tag;
    logic source_done_error;
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
    logic [2:0] bias_req_token_id, bias_rsp_token_id;
    logic bias_req_allow, bias_rsp_valid, bias_rsp_ready;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values, bias_lookup_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [(BANKS*3)-1:0] final_token_ids;
    logic [15:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic tile_done_valid, tile_done_ready;
    logic [15:0] tile_done_tag;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_heads, count_terms;
    logic [31:0] count_completed_terms, count_bias_commits;
    integer signed observed [0:TOKENS-1][0:OUT_TILE-1];
    logic [TOKENS-1:0] observed_valid;
    integer cycle_count, head_done_count, bias_requests, final_count;

    gatestack_multihead_tile_projection_top #(
        .TOKENS(TOKENS), .LANES(LANES), .EVENT_WAYS(4),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS), .SEGMENT_TOKENS(4),
        .TAG_W(16), .INPUT_CH_W(6), .OUTPUT_TILE_W(4),
        .HEAD_COUNT_W(3), .TOKEN_ID_W(3), .LANE_ID_W(2)
    ) dut (.*);

    gatestack_bias_sram_model #(
        .TAG_W(16), .OUTPUT_TILE_W(4), .TOKEN_ID_W(3),
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
        input logic output_lane
    );
        begin
            case (channel)
                6'd0:  weight_lane = output_lane ? 8'sd2 : 8'sd1;
                6'd9:  weight_lane = output_lane ? -8'sd1 : 8'sd3;
                6'd11: weight_lane = output_lane ? 8'sd4 : -8'sd2;
                default: weight_lane = '0;
            endcase
        end
    endfunction

    task automatic start_head(
        input logic [2:0] index_value,
        input logic [5:0] base_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            head_start_index = index_value;
            head_start_input_channel_base = base_value;
            head_start_last = last_value;
            head_start_valid = 1'b1;
            do @(posedge clk_core); while (!head_start_ready);
            @(negedge clk_core);
            head_start_valid = 1'b0;
        end
    endtask

    task automatic send_term(
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [7:0] destination_value,
        input logic [12:0] sequence_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            term_gate_code = gate_value;
            term_lane_id = lane_value;
            term_destination_count = destination_value;
            term_issue_seq = sequence_value;
            term_head_last = last_value;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;
        end
    endtask

    task automatic send_event(
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [3:0] valid_value,
        input logic [11:0] ids_value,
        input logic [2:0] count_value,
        input logic [12:0] sequence_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            event_gate_code = gate_value;
            event_lane_id = lane_value;
            event_token_valid = valid_value;
            event_token_ids = ids_value;
            event_count = count_value;
            event_issue_seq = sequence_value;
            event_term_first = 1'b1;
            event_term_last = 1'b1;
            event_head_last = last_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic finish_source;
        begin
            @(negedge clk_core);
            source_done_valid = 1'b1;
            do @(posedge clk_core); while (!source_done_ready);
            @(negedge clk_core);
            source_done_valid = 1'b0;
        end
    endtask

    task automatic wait_head_done(
        input logic [2:0] index_value,
        input logic last_value
    );
        begin
            wait (head_done_valid);
            if (head_done_tag != 16'h7601 || head_done_index != index_value ||
                head_done_last != last_value || head_done_error)
                $fatal(1, "head completion mismatch index=%0d", index_value);
            do @(posedge clk_core); while (!head_done_ready);
            @(negedge clk_core);
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
            repeat (2) @(posedge clk_core);
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
        head_done_ready = (cycle_count % 4) != 1;
        bias_req_allow = (cycle_count % 5) != 2;
        if (bias_req_output_tile == 4'd2) begin
            bias_lookup_values[31:0] = 32'(100 + 32'(bias_req_token_id));
            bias_lookup_values[63:32] = 32'(-50 - 32'(bias_req_token_id));
        end else begin
            bias_lookup_values = 'x;
        end
        final_ready[0] = (cycle_count % 3) != 1;
        final_ready[1] = (cycle_count % 4) != 2;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            head_done_count <= 0;
            bias_requests <= 0;
            final_count <= 0;
            observed_valid <= '0;
            for (int token = 0; token < TOKENS; token = token + 1)
                for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                    observed[token][lane] <= 0;
        end else begin
            integer final_fire_count;
            final_fire_count = 0;
            cycle_count <= cycle_count + 1;
            if (head_done_valid && head_done_ready)
                head_done_count <= head_done_count + 1;
            if (bias_req_valid && bias_req_ready) begin
                if (head_done_count != 3)
                    $fatal(1, "bias issued before all heads completed");
                bias_requests <= bias_requests + 1;
            end
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    if (final_tag != 16'h7601)
                        $fatal(1, "final tag mismatch");
                    observed_valid[final_token_ids[(bank*3) +: 3]] <= 1'b1;
                    observed[final_token_ids[(bank*3) +: 3]][0] <=
                        $signed(final_values[(bank*64) +: 32]);
                    observed[final_token_ids[(bank*3) +: 3]][1] <=
                        $signed(final_values[(bank*64)+32 +: 32]);
                    final_fire_count = final_fire_count + 1;
                end
            end
            final_count <= final_count + final_fire_count;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = 1'b0;
        tile_start_tag = 16'h7601;
        tile_start_output_tile = 4'd2;
        tile_start_head_count = 3'd3;
        head_start_valid = 1'b0;
        head_start_tag = 16'h7601;
        head_start_index = '0;
        head_start_input_channel_base = '0;
        head_start_last = 1'b0;
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        event_valid = 1'b0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_issue_seq = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        source_done_valid = 1'b0;
        source_done_tag = 16'h7601;
        source_done_error = 1'b0;
        tile_done_ready = 1'b0;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        tile_start_valid = 1'b1;
        do @(posedge clk_core); while (!tile_start_ready);
        @(negedge clk_core);
        tile_start_valid = 1'b0;

        start_head(3'd0, 6'd0, 1'b0);
        send_term(9'd2, 2'd0, 8'd2, 13'd0, 1'b1);
        send_event(9'd2, 2'd0, 4'b0011,
                   {3'd0, 3'd0, 3'd1, 3'd0}, 3'd2, 13'd0, 1'b1);
        finish_source();
        wait_head_done(3'd0, 1'b0);

        // Empty middle head verifies that an empty session neither clears the
        // accumulator nor triggers bias.
        start_head(3'd1, 6'd4, 1'b0);
        finish_source();
        wait_head_done(3'd1, 1'b0);

        start_head(3'd2, 6'd8, 1'b1);
        send_term(9'd3, 2'd1, 8'd2, 13'd0, 1'b0);
        send_event(9'd3, 2'd1, 4'b0011,
                   {3'd0, 3'd0, 3'd2, 3'd1}, 3'd2, 13'd0, 1'b0);
        send_term(9'd4, 2'd3, 8'd1, 13'd1, 1'b1);
        send_event(9'd4, 2'd3, 4'b0001,
                   {3'd0, 3'd0, 3'd0, 3'd2}, 3'd1, 13'd1, 1'b1);
        finish_source();
        wait_head_done(3'd2, 1'b1);

        wait (tile_done_valid);
        @(negedge clk_core);
        if (tile_done_tag != 16'h7601 || protocol_error ||
            accumulator_overflow || observed_valid != {TOKENS{1'b1}} ||
            head_done_count != 3 || count_heads != 3 || count_terms != 3 ||
            count_completed_terms != 3 || count_bias_commits != TOKENS ||
            bias_requests != TOKENS || final_count != TOKENS)
            $fatal(1, "multihead status mismatch heads=%0d terms=%0d bias=%0d finals=%0d protocol=%b",
                   count_heads, count_terms, count_bias_commits,
                   final_count, protocol_error);
        for (int token = 0; token < TOKENS; token = token + 1) begin
            integer signed expected0, expected1;
            expected0 = 100 + token;
            expected1 = -50 - token;
            if (token == 0) begin expected0 = expected0 + 2; expected1 = expected1 + 4; end
            if (token == 1) begin expected0 = expected0 + 11; expected1 = expected1 + 1; end
            if (token == 2) begin expected0 = expected0 + 1; expected1 = expected1 + 13; end
            if (observed[token][0] != expected0 ||
                observed[token][1] != expected1)
                $fatal(1, "token %0d got=(%0d,%0d) expected=(%0d,%0d)",
                       token, observed[token][0], observed[token][1],
                       expected0, expected1);
        end
        tile_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: GateStack multihead tile heads=%0d terms=%0d bias=%0d cycles=%0d",
                 count_heads, count_terms, count_bias_commits, cycle_count);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "multihead tile TB timeout");
    end
endmodule

`default_nettype wire
