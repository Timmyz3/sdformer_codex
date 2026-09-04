`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_routed_multihead_tile_projection_top;
    localparam int TOKENS = 8;
    localparam int SOURCES = 3;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int TOKEN_ID_W = 3;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic tile_start_valid, tile_start_ready;
    logic [15:0] tile_start_tag;
    logic [3:0] tile_start_output_tile;
    logic [2:0] tile_start_head_count;
    logic head_start_valid, head_start_ready;
    logic [15:0] head_start_tag;
    logic [2:0] head_start_index;
    logic [1:0] head_start_route_select;
    logic [5:0] head_start_input_channel_base;
    logic head_start_last;
    logic [SOURCES-1:0] source_term_valid, source_term_ready;
    logic [(SOURCES*9)-1:0] source_term_gate_code;
    logic [(SOURCES*2)-1:0] source_term_lane_id;
    logic [(SOURCES*8)-1:0] source_term_destination_count;
    logic [SOURCES-1:0] source_term_head_last;
    logic [SOURCES-1:0] source_event_valid, source_event_ready;
    logic [(SOURCES*9)-1:0] source_event_gate_code;
    logic [(SOURCES*2)-1:0] source_event_lane_id;
    logic [(SOURCES*4)-1:0] source_event_token_valid;
    logic [(SOURCES*4*TOKEN_ID_W)-1:0] source_event_token_ids;
    logic [(SOURCES*3)-1:0] source_event_count;
    logic [SOURCES-1:0] source_event_term_first;
    logic [SOURCES-1:0] source_event_term_last;
    logic [SOURCES-1:0] source_event_head_last;
    logic [SOURCES-1:0] source_done_valid, source_done_ready;
    logic [(SOURCES*16)-1:0] source_done_tag;
    logic [SOURCES-1:0] source_done_error;
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
    logic [(BANKS*TOKEN_ID_W)-1:0] final_token_ids;
    logic [15:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic tile_done_valid, tile_done_ready;
    logic [15:0] tile_done_tag;
    logic route_active;
    logic [1:0] route_active_select;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_heads, count_terms;
    logic [31:0] count_completed_terms, count_bias_commits;
    integer signed observed [0:TOKENS-1][0:OUT_TILE-1];
    logic [TOKENS-1:0] observed_valid;
    integer cycle_count, head_completions, bias_requests, final_count;

    gatestack_routed_multihead_tile_projection_top #(
        .TOKENS(TOKENS), .LANES(4), .SOURCES(SOURCES),
        .EVENT_WAYS(4), .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(4), .TAG_W(16), .INPUT_CH_W(6),
        .OUTPUT_TILE_W(4), .HEAD_COUNT_W(3), .TOKEN_ID_W(TOKEN_ID_W),
        .LANE_ID_W(2), .ROUTE_W(2)
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
                6'd5:  weight_lane = output_lane ? -8'sd1 : 8'sd3;
                6'd11: weight_lane = output_lane ? 8'sd4 : -8'sd2;
                default: weight_lane = '0;
            endcase
        end
    endfunction

    task automatic send_routed_head(
        input integer route_value,
        input logic [2:0] index_value,
        input logic [5:0] base_value,
        input logic last_value,
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [2:0] token_a,
        input logic [2:0] token_b,
        input logic [2:0] destination_count
    );
        begin
            @(negedge clk_core);
            head_start_index = index_value;
            head_start_route_select = 2'(route_value);
            head_start_input_channel_base = base_value;
            head_start_last = last_value;
            head_start_valid = 1'b1;
            do @(posedge clk_core); while (!head_start_ready);
            @(negedge clk_core);
            head_start_valid = 1'b0;
            if (!route_active || route_active_select != 2'(route_value))
                $fatal(1, "route did not lock source %0d", route_value);

            source_term_gate_code[(route_value*9) +: 9] = gate_value;
            source_term_lane_id[(route_value*2) +: 2] = lane_value;
            source_term_destination_count[(route_value*8) +: 8] =
                8'(destination_count);
            source_term_head_last[route_value] = 1'b1;
            source_term_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_term_ready[route_value]);
            @(negedge clk_core);
            source_term_valid[route_value] = 1'b0;

            source_event_gate_code[(route_value*9) +: 9] = gate_value;
            source_event_lane_id[(route_value*2) +: 2] = lane_value;
            source_event_token_valid[(route_value*4) +: 4] =
                destination_count == 1 ? 4'b0001 : 4'b0011;
            source_event_token_ids[(route_value*12) +: 12] =
                {3'd0, 3'd0, token_b, token_a};
            source_event_count[(route_value*3) +: 3] = destination_count;
            source_event_term_first[route_value] = 1'b1;
            source_event_term_last[route_value] = 1'b1;
            source_event_head_last[route_value] = 1'b1;
            source_event_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_event_ready[route_value]);
            @(negedge clk_core);
            source_event_valid[route_value] = 1'b0;

            source_done_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_done_ready[route_value]);
            @(negedge clk_core);
            source_done_valid[route_value] = 1'b0;
            wait (head_done_valid);
            if (head_done_tag != 16'h7701 || head_done_index != index_value ||
                head_done_last != last_value || head_done_error)
                $fatal(1, "head done mismatch for route %0d", route_value);
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
        head_done_ready = (cycle_count % 4) != 1;
        bias_req_allow = (cycle_count % 5) != 2;
        bias_lookup_values[31:0] = 32'(100 + 32'(bias_req_token_id));
        bias_lookup_values[63:32] = 32'(-50 - 32'(bias_req_token_id));
        final_ready[0] = (cycle_count % 3) != 1;
        final_ready[1] = (cycle_count % 4) != 2;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            head_completions <= 0;
            bias_requests <= 0;
            final_count <= 0;
            observed_valid <= '0;
            for (int token = 0; token < TOKENS; token = token + 1)
                for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                    observed[token][lane] <= 0;
        end else begin
            integer fires;
            fires = 0;
            cycle_count <= cycle_count + 1;
            if (head_done_valid && head_done_ready)
                head_completions <= head_completions + 1;
            if (bias_req_valid && bias_req_ready) begin
                if (head_completions != 3)
                    $fatal(1, "routed bias issued before three heads");
                bias_requests <= bias_requests + 1;
            end
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    if (final_tag != 16'h7701)
                        $fatal(1, "routed final tag mismatch");
                    observed_valid[final_token_ids[(bank*3) +: 3]] <= 1'b1;
                    observed[final_token_ids[(bank*3) +: 3]][0] <=
                        $signed(final_values[(bank*64) +: 32]);
                    observed[final_token_ids[(bank*3) +: 3]][1] <=
                        $signed(final_values[(bank*64)+32 +: 32]);
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
        tile_start_tag = 16'h7701;
        tile_start_output_tile = 4'd2;
        tile_start_head_count = 3'd3;
        head_start_valid = 1'b0;
        head_start_tag = 16'h7701;
        head_start_index = '0;
        head_start_route_select = '0;
        head_start_input_channel_base = '0;
        head_start_last = 1'b0;
        source_term_valid = '0;
        source_term_gate_code = '0;
        source_term_lane_id = '0;
        source_term_destination_count = '0;
        source_term_head_last = '0;
        source_event_valid = '0;
        source_event_gate_code = '0;
        source_event_lane_id = '0;
        source_event_token_valid = '0;
        source_event_token_ids = '0;
        source_event_count = '0;
        source_event_term_first = '0;
        source_event_term_last = '0;
        source_event_head_last = '0;
        source_done_valid = '0;
        source_done_tag = {3{16'h7701}};
        source_done_error = '0;
        tile_done_ready = 1'b0;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        tile_start_valid = 1'b1;
        do @(posedge clk_core); while (!tile_start_ready);
        @(negedge clk_core);
        tile_start_valid = 1'b0;

        send_routed_head(0, 3'd0, 6'd0, 1'b0,
                         9'd2, 2'd0, 3'd0, 3'd0, 3'd1);
        send_routed_head(1, 3'd1, 6'd4, 1'b0,
                         9'd3, 2'd1, 3'd0, 3'd1, 3'd2);
        send_routed_head(2, 3'd2, 6'd8, 1'b1,
                         9'd4, 2'd3, 3'd1, 3'd0, 3'd1);

        wait (tile_done_valid);
        @(negedge clk_core);
        if (tile_done_tag != 16'h7701 || protocol_error ||
            accumulator_overflow || observed_valid != {TOKENS{1'b1}} ||
            count_heads != 3 || count_terms != 3 ||
            count_completed_terms != 3 || count_bias_commits != TOKENS ||
            head_completions != 3 || bias_requests != TOKENS ||
            final_count != TOKENS || route_active)
            $fatal(1, "routed multihead status mismatch");
        for (int token = 0; token < TOKENS; token = token + 1) begin
            integer signed expected0, expected1;
            expected0 = 100 + token;
            expected1 = -50 - token;
            if (token == 0) begin expected0 = expected0 + 11; expected1 = expected1 + 1; end
            if (token == 1) begin expected0 = expected0 + 1; expected1 = expected1 + 13; end
            if (observed[token][0] != expected0 || observed[token][1] != expected1)
                $fatal(1, "routed token %0d mismatch got=(%0d,%0d) expected=(%0d,%0d)",
                       token, observed[token][0], observed[token][1],
                       expected0, expected1);
        end
        tile_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: GateStack routed multihead heads=%0d terms=%0d bias=%0d cycles=%0d",
                 count_heads, count_terms, count_bias_commits, cycle_count);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "routed multihead TB timeout");
    end
endmodule

`default_nettype wire
