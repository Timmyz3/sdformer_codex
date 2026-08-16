`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_single_head_projection_top #(
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0,
    parameter bit IMPLICIT_BIAS_FINALIZE_ENABLE = 1'b0
);
    localparam int TOKENS = 8;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [15:0] group_tag;
    logic [5:0] group_input_channel_base;
    logic [3:0] group_output_tile;
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
    logic bias_rsp_valid, bias_rsp_ready;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [(BANKS*3)-1:0] final_token_ids;
    logic [15:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic group_done_valid, group_done_ready;
    logic [15:0] group_done_tag;
    logic protocol_error, accumulator_overflow;
    logic [31:0] count_terms, count_completed_terms, count_bias_commits;
    integer signed observed [0:TOKENS-1][0:OUT_TILE-1];
    logic [TOKENS-1:0] observed_valid;
    integer cycle_count;
    integer bias_requests;
    integer bias_responses;
    integer bias_rsp_stall_cycles;
    integer final_stall_cycles;
    integer bias_req_cycle_q;
    logic bias_pending_q;
    logic bias_delay_q;
    logic bias_mismatch_sent_q;
    logic hold_bank0_q;

    gatestack_single_head_projection_top #(
        .TOKENS(TOKENS), .LANES(4), .EVENT_WAYS(4),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS), .SEGMENT_TOKENS(4),
        .TAG_W(16), .INPUT_CH_W(6), .OUTPUT_TILE_W(4),
        .TOKEN_ID_W(3), .LANE_ID_W(2),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .IMPLICIT_BIAS_FINALIZE_ENABLE(IMPLICIT_BIAS_FINALIZE_ENABLE)
    ) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_term(
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [7:0] destination_value,
        input logic [12:0] sequence_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            term_gate_code = gate_value;
            term_lane_id = lane_value;
            term_destination_count = destination_value;
            term_issue_seq = sequence_value;
            term_head_last = head_last_value;
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
        input logic first_value,
        input logic last_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            event_gate_code = gate_value;
            event_lane_id = lane_value;
            event_token_valid = valid_value;
            event_token_ids = ids_value;
            event_count = count_value;
            event_issue_seq = sequence_value;
            event_term_first = first_value;
            event_term_last = last_value;
            event_head_last = head_last_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    function automatic signed [7:0] weight_lane(
        input logic [5:0] channel,
        input logic lane
    );
        begin
            case (channel)
                6'd5: weight_lane = lane ? 8'sd4 : -8'sd2;
                6'd7: weight_lane = lane ? 8'sd6 : 8'sd1;
                default: weight_lane = '0;
            endcase
        end
    endfunction

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

    assign bias_req_ready = !bias_pending_q && (cycle_count % 4) != 1;
    assign final_ready[0] = !hold_bank0_q;
    assign final_ready[1] = 1'b1;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            bias_requests <= 0;
            bias_responses <= 0;
            bias_rsp_stall_cycles <= 0;
            final_stall_cycles <= 0;
            bias_req_cycle_q <= -1;
            bias_pending_q <= 1'b0;
            bias_delay_q <= 1'b0;
            bias_mismatch_sent_q <= 1'b0;
            bias_rsp_valid <= 1'b0;
            bias_rsp_tag <= '0;
            bias_rsp_token_id <= '0;
            bias_rsp_values <= '0;
            hold_bank0_q <= 1'b1;
            observed_valid <= '0;
            for (int token = 0; token < TOKENS; token = token + 1)
                for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                    observed[token][lane] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (bias_req_valid && bias_req_ready) begin
                bias_requests <= bias_requests + 1;
                bias_pending_q <= 1'b1;
                bias_delay_q <= 1'b1;
                bias_req_cycle_q <= cycle_count;
                if (!bias_mismatch_sent_q) begin
                    bias_rsp_tag <= bias_req_tag ^ 16'h0001;
                    bias_rsp_token_id <= bias_req_token_id + 1'b1;
                end else begin
                    bias_rsp_tag <= bias_req_tag;
                    bias_rsp_token_id <= bias_req_token_id;
                end
                bias_rsp_values[31:0] <=
                    (BIAS_STATIONARY_ENABLE ||
                     IMPLICIT_BIAS_FINALIZE_ENABLE) ?
                    32'(200000) : 32'(200000 + 32'(bias_req_token_id));
                bias_rsp_values[63:32] <=
                    (BIAS_STATIONARY_ENABLE ||
                     IMPLICIT_BIAS_FINALIZE_ENABLE) ?
                    32'(-300000) : 32'(-300000 - 32'(bias_req_token_id));
                if (bias_req_tag != 16'h7201 || bias_req_output_tile != 4'd2)
                    $fatal(1, "bias request identity mismatch");
            end
            if (bias_delay_q) begin
                bias_delay_q <= 1'b0;
                bias_rsp_valid <= 1'b1;
                if (cycle_count <= bias_req_cycle_q)
                    $fatal(1, "bias response violated minimum one-cycle latency");
            end
            if (bias_rsp_valid && !bias_rsp_ready) begin
                bias_rsp_stall_cycles <= bias_rsp_stall_cycles + 1;
                if (!BIAS_STATIONARY_ENABLE &&
                    !IMPLICIT_BIAS_FINALIZE_ENABLE &&
                    bias_rsp_token_id == 3'd2 && bias_rsp_stall_cycles >= 2)
                    hold_bank0_q <= 1'b0;
            end
            if (final_valid[0] && !final_ready[0]) begin
                final_stall_cycles <= final_stall_cycles + 1;
                if ((BIAS_STATIONARY_ENABLE ||
                     IMPLICIT_BIAS_FINALIZE_ENABLE) &&
                    final_stall_cycles >= 2)
                    hold_bank0_q <= 1'b0;
            end
            if (bias_rsp_valid && bias_rsp_ready) begin
                if (!bias_mismatch_sent_q && count_bias_commits != 0)
                    $fatal(1, "mismatched bias response committed data");
                bias_rsp_valid <= 1'b0;
                bias_pending_q <= 1'b0;
                bias_responses <= bias_responses + 1;
                if (!bias_mismatch_sent_q)
                    bias_mismatch_sent_q <= 1'b1;
            end
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    if (final_tag != 16'h7201)
                        $fatal(1, "final tag mismatch");
                    observed_valid[final_token_ids[(bank*3) +: 3]] <= 1'b1;
                    observed[final_token_ids[(bank*3) +: 3]][0] <=
                        $signed(final_values[(bank*64) +: 32]);
                    observed[final_token_ids[(bank*3) +: 3]][1] <=
                        $signed(final_values[(bank*64)+32 +: 32]);
                end
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = 16'h7201;
        group_input_channel_base = 6'd4;
        group_output_tile = 4'd2;
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
        source_done_tag = 16'h7201;
        source_done_error = 1'b0;
        group_done_ready = 1'b0;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;

        send_term(9'd3, 2'd1, 8'd3, 13'd0, 1'b0);
        send_event(9'd3, 2'd1, 4'b0011, {3'd0,3'd0,3'd2,3'd0},
                   3'd2, 13'd0, 1'b1, 1'b0, 1'b0);
        send_event(9'd3, 2'd1, 4'b0001, {3'd0,3'd0,3'd0,3'd5},
                   3'd1, 13'd0, 1'b0, 1'b1, 1'b0);
        send_term(9'd2, 2'd3, 8'd2, 13'd1, 1'b1);
        send_event(9'd2, 2'd3, 4'b0011, {3'd0,3'd0,3'd7,3'd0},
                   3'd2, 13'd1, 1'b1, 1'b1, 1'b1);
        @(negedge clk_core);
        source_done_valid = 1'b1;
        do @(posedge clk_core); while (!source_done_ready);
        @(negedge clk_core);
        source_done_valid = 1'b0;

        wait (group_done_valid);
        @(negedge clk_core);
        if (observed_valid != {TOKENS{1'b1}} || !protocol_error ||
            accumulator_overflow || count_terms != 2 ||
            count_completed_terms != 2 || count_bias_commits != TOKENS ||
            bias_requests != ((BIAS_STATIONARY_ENABLE ||
                               IMPLICIT_BIAS_FINALIZE_ENABLE) ?
                              2 : TOKENS + 1) ||
            bias_responses != ((BIAS_STATIONARY_ENABLE ||
                                IMPLICIT_BIAS_FINALIZE_ENABLE) ?
                               2 : TOKENS + 1) ||
            !bias_mismatch_sent_q ||
            ((BIAS_STATIONARY_ENABLE ||
              IMPLICIT_BIAS_FINALIZE_ENABLE) ?
             (final_stall_cycles < 2) : (bias_rsp_stall_cycles < 2)))
            $fatal(1, "single-head projection status mismatch");
        for (int token = 0; token < TOKENS; token = token + 1) begin
            integer signed expected0;
            integer signed expected1;
            expected0 = 200000 +
                ((BIAS_STATIONARY_ENABLE ||
                  IMPLICIT_BIAS_FINALIZE_ENABLE) ? 0 : token);
            expected1 = -300000 -
                ((BIAS_STATIONARY_ENABLE ||
                  IMPLICIT_BIAS_FINALIZE_ENABLE) ? 0 : token);
            if (token == 0) begin expected0 = expected0 - 4; expected1 = expected1 + 24; end
            if (token == 2 || token == 5) begin expected0 = expected0 - 6; expected1 = expected1 + 12; end
            if (token == 7) begin expected0 = expected0 + 2; expected1 = expected1 + 12; end
            if (observed[token][0] != expected0 ||
                observed[token][1] != expected1)
                $fatal(1, "token %0d mismatch got=(%0d,%0d) expected=(%0d,%0d)",
                       token, observed[token][0], observed[token][1],
                       expected0, expected1);
        end
        if (group_done_tag != 16'h7201)
            $fatal(1, "group done tag mismatch");
        group_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: single-head req/rsp projection bsf=%0d ibf=%0d terms=%0d completed=%0d bias=%0d req=%0d rsp=%0d stalls=%0d final_stalls=%0d cycles=%0d",
                 BIAS_STATIONARY_ENABLE, IMPLICIT_BIAS_FINALIZE_ENABLE,
                 count_terms, count_completed_terms, count_bias_commits,
                 bias_requests, bias_responses, bias_rsp_stall_cycles,
                 final_stall_cycles, cycle_count);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "single-head projection TB timeout");
    end
endmodule

`default_nettype wire
