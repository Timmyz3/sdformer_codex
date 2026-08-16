`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_routed_single_head_projection_top;
    localparam int TOKENS = 8;
    localparam int SOURCES = 3;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [15:0] group_tag;
    logic [1:0] group_route_select;
    logic [5:0] group_input_channel_base;
    logic [3:0] group_output_tile;
    logic [SOURCES-1:0] source_term_valid, source_term_ready;
    logic [(SOURCES*9)-1:0] source_term_gate_code;
    logic [(SOURCES*2)-1:0] source_term_lane_id;
    logic [(SOURCES*8)-1:0] source_term_destination_count;
    logic [SOURCES-1:0] source_term_head_last;
    logic [SOURCES-1:0] source_event_valid, source_event_ready;
    logic [(SOURCES*9)-1:0] source_event_gate_code;
    logic [(SOURCES*2)-1:0] source_event_lane_id;
    logic [(SOURCES*4)-1:0] source_event_token_valid;
    logic [(SOURCES*4*3)-1:0] source_event_token_ids;
    logic [(SOURCES*3)-1:0] source_event_count;
    logic [SOURCES-1:0] source_event_term_first;
    logic [SOURCES-1:0] source_event_term_last;
    logic [SOURCES-1:0] source_event_head_last;
    logic [SOURCES-1:0] source_done_valid, source_done_ready;
    logic [(SOURCES*16)-1:0] source_done_tag;
    logic [SOURCES-1:0] source_done_error;
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
    logic group_done_valid, group_done_ready;
    logic [15:0] group_done_tag;
    logic route_active;
    logic [1:0] route_active_select;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_terms, count_completed_terms, count_bias_commits;
    integer cycle_count;
    integer current_route;
    integer current_token_a;
    integer current_token_b;
    integer current_product0;
    integer current_product1;
    integer finals_in_group;
    integer mismatches;
    integer bias_requests;

    gatestack_routed_single_head_projection_top #(
        .TOKENS(TOKENS), .LANES(4), .SOURCES(SOURCES),
        .EVENT_WAYS(4), .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(4), .TAG_W(16), .INPUT_CH_W(6),
        .OUTPUT_TILE_W(4), .TOKEN_ID_W(3), .LANE_ID_W(2)
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
        input logic lane
    );
        begin
            case (channel)
                6'd4: weight_lane = lane ? 8'sd2 : 8'sd1;
                6'd5: weight_lane = lane ? 8'sd4 : -8'sd2;
                6'd6: weight_lane = lane ? -8'sd1 : 8'sd3;
                default: weight_lane = '0;
            endcase
        end
    endfunction

    task automatic run_route(
        input logic [1:0] route_value,
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [2:0] token_a,
        input logic [2:0] token_b
    );
        logic [15:0] tag_value;
        begin
            tag_value = 16'h7300 + 16'(route_value);
            current_route = 32'(route_value);
            current_token_a = 32'(token_a);
            current_token_b = 32'(token_b);
            current_product0 = $signed({1'b0, gate_value}) *
                $signed(weight_lane(6'(4 + 32'(lane_value)), 1'b0));
            current_product1 = $signed({1'b0, gate_value}) *
                $signed(weight_lane(6'(4 + 32'(lane_value)), 1'b1));
            finals_in_group = 0;

            @(negedge clk_core);
            group_tag = tag_value;
            group_route_select = route_value;
            group_valid = 1'b1;
            do @(posedge clk_core); while (!group_ready);
            @(negedge clk_core);
            group_valid = 1'b0;

            source_term_gate_code[(route_value*9) +: 9] = gate_value;
            source_term_lane_id[(route_value*2) +: 2] = lane_value;
            source_term_destination_count[(route_value*8) +: 8] = 8'd2;
            source_term_head_last[route_value] = 1'b1;
            source_term_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_term_ready[route_value]);
            @(negedge clk_core);
            source_term_valid[route_value] = 1'b0;

            source_event_gate_code[(route_value*9) +: 9] = gate_value;
            source_event_lane_id[(route_value*2) +: 2] = lane_value;
            source_event_token_valid[(route_value*4) +: 4] = 4'b0011;
            source_event_token_ids[(route_value*12) +: 12] =
                {3'd0, 3'd0, token_b, token_a};
            source_event_count[(route_value*3) +: 3] = 3'd2;
            source_event_term_first[route_value] = 1'b1;
            source_event_term_last[route_value] = 1'b1;
            source_event_head_last[route_value] = 1'b1;
            source_event_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_event_ready[route_value]);
            @(negedge clk_core);
            source_event_valid[route_value] = 1'b0;

            source_done_tag[(route_value*16) +: 16] = tag_value;
            source_done_valid[route_value] = 1'b1;
            do @(posedge clk_core); while (!source_done_ready[route_value]);
            @(negedge clk_core);
            source_done_valid[route_value] = 1'b0;

            wait (group_done_valid);
            @(negedge clk_core);
            if (finals_in_group != TOKENS || group_done_tag != tag_value ||
                protocol_error || accumulator_overflow) begin
                $display("route=%0d finals=%0d tag=%h expected=%h protocol=%b overflow=%b terms=%0d completed=%0d bias=%0d",
                         route_value, finals_in_group, group_done_tag,
                         tag_value, protocol_error, accumulator_overflow,
                         count_terms, count_completed_terms,
                         count_bias_commits);
                $fatal(1, "route %0d completion mismatch", route_value);
            end
            group_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            group_done_ready = 1'b0;
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
        bias_req_allow = (cycle_count % 4) != 2;
        bias_lookup_values[31:0] = 32'(10 + 32'(bias_req_token_id));
        bias_lookup_values[63:32] = 32'(-20 - 32'(bias_req_token_id));
        final_ready[0] = (cycle_count % 5) != 1;
        final_ready[1] = (cycle_count % 3) != 0;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            mismatches <= 0;
            bias_requests <= 0;
        end else begin
            integer final_fire_count;
            final_fire_count = 0;
            cycle_count <= cycle_count + 1;
            if (bias_req_valid && bias_req_ready)
                bias_requests <= bias_requests + 1;
            if (route_active) begin
                if (route_active_select != 2'(current_route))
                    $fatal(1, "route select changed in session");
                for (int source = 0; source < SOURCES; source = source + 1)
                    if (source != current_route &&
                        (source_term_ready[source] ||
                         source_event_ready[source] || source_done_ready[source]))
                        $fatal(1, "unselected source received ready");
            end
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    integer token_value;
                    integer signed expected0;
                    integer signed expected1;
                    integer signed actual0;
                    integer signed actual1;
                    token_value = 32'(final_token_ids[(bank*3) +: 3]);
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
                        final_tag != 16'h7300 + 16'(current_route)) begin
                        mismatches <= mismatches + 1;
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
        group_input_channel_base = 6'd4;
        group_output_tile = 4'd1;
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
        source_done_tag = '0;
        source_done_error = '0;
        group_done_ready = 1'b0;
        current_route = 0;
        current_token_a = 0;
        current_token_b = 0;
        current_product0 = 0;
        current_product1 = 0;
        finals_in_group = 0;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;
        run_route(2'd0, 9'd2, 2'd0, 3'd0, 3'd7);
        run_route(2'd1, 9'd3, 2'd1, 3'd1, 3'd6);
        run_route(2'd2, 9'd4, 2'd2, 3'd2, 3'd5);
        if (mismatches != 0 || count_terms != 3 ||
            count_completed_terms != 3 || count_bias_commits != 24 ||
            bias_requests != 24)
            $fatal(1, "routed projection cumulative counters mismatch");
        $display("PASS: three-route shared projection terms=%0d completed=%0d bias=%0d cycles=%0d",
                 count_terms, count_completed_terms, count_bias_commits,
                 cycle_count);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "routed projection TB timeout");
    end
endmodule

`default_nettype wire
