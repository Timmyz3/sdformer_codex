`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_tdr_multicast_backend;
    localparam int TOKENS = 8;
    localparam int LANES = 4;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int TOKEN_ID_W = 3;
    localparam int PRODUCT_W = 17;
    logic clk_core;
    logic rst_core;
    logic session_start_valid;
    logic session_start_ready;
    logic [15:0] session_tag;
    logic [5:0] session_input_channel_base;
    logic [3:0] session_output_tile;
    logic term_valid;
    logic term_ready;
    logic [8:0] term_gate_code;
    logic [1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [8:0] event_gate_code;
    logic [1:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [11:0] event_token_ids;
    logic [2:0] event_count;
    logic [12:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic source_done_valid;
    logic source_done_ready;
    logic [15:0] source_done_tag;
    logic source_done_error;
    logic decoder_done_valid;
    logic decoder_done_ready;
    logic [15:0] decoder_done_tag;
    logic decoder_done_error;
    logic weight_req_valid;
    logic weight_req_ready;
    logic [15:0] weight_req_tag;
    logic [5:0] weight_req_input_channel;
    logic [3:0] weight_req_output_tile;
    logic weight_rsp_valid;
    logic weight_rsp_ready;
    logic [15:0] weight_rsp_tag;
    logic [5:0] weight_rsp_input_channel;
    logic [3:0] weight_rsp_output_tile;
    logic [15:0] weight_rsp_weights;
    logic [1:0] update_valid;
    logic [1:0] update_ready;
    logic [5:0] update_token_ids;
    logic [15:0] update_tag;
    logic [(OUT_TILE*PRODUCT_W)-1:0] update_values;
    logic backend_done_valid;
    logic backend_done_ready;
    logic [15:0] backend_done_tag;
    logic backend_done_error;
    logic protocol_error;
    logic [13:0] outstanding_terms;
    logic [31:0] count_sessions;
    logic [31:0] count_terms;
    logic [31:0] count_completed_terms;
    logic [31:0] count_empty_sessions;
    integer signed observed [0:TOKENS-1][0:OUT_TILE-1];
    integer cycle_count;
    integer decoder_cycle;
    integer backend_cycle;
    integer first_completion_latency;

    gatestack_tdr_multicast_backend #(
        .TOKENS(TOKENS), .LANES(LANES), .EVENT_WAYS(4),
        .OUT_TILE(OUT_TILE), .BANKS(BANKS), .SEGMENT_TOKENS(4),
        .TAG_W(16), .INPUT_CH_W(6), .OUTPUT_TILE_W(4),
        .TOKEN_ID_W(TOKEN_ID_W), .LANE_ID_W(2)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

`ifdef DEBUG_TDR_BACKEND
    always @(posedge clk_core) begin
        if (!rst_core && (session_start_valid || term_valid || event_valid ||
            source_done_valid || weight_req_valid || weight_rsp_valid ||
            update_valid != 0 || decoder_done_valid || backend_done_valid)) begin
            $display("DBG c=%0d ss=%b/%b term=%b/%b ev=%b/%b srcdone=%b/%b wreq=%b/%b wrsp=%b/%b upd=%b/%b dec=%b back=%b out=%0d",
                     cycle_count, session_start_valid, session_start_ready,
                     term_valid, term_ready, event_valid, event_ready,
                     source_done_valid, source_done_ready,
                     weight_req_valid, weight_req_ready,
                     weight_rsp_valid, weight_rsp_ready,
                     update_valid, update_ready, decoder_done_valid,
                     backend_done_valid, outstanding_terms);
        end
    end
`endif

    task automatic start_session(input logic [15:0] tag_value);
        begin
            @(negedge clk_core);
            session_tag = tag_value;
            session_start_valid = 1'b1;
            do @(posedge clk_core); while (!session_start_ready);
            @(negedge clk_core);
            session_start_valid = 1'b0;
        end
    endtask

    task automatic send_term(
        input logic [8:0] gate_value,
        input logic [1:0] lane_value,
        input logic [7:0] destinations,
        input logic [12:0] sequence_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            term_gate_code = 9'(gate_value);
            term_lane_id = 2'(lane_value);
            term_destination_count = 8'(destinations);
            term_issue_seq = 13'(sequence_value);
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
        input logic [3:0] valid_mask,
        input logic [2:0] id0,
        input logic [2:0] id1,
        input logic [2:0] id2,
        input logic [2:0] id3,
        input logic [2:0] count_value,
        input logic [12:0] sequence_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            event_gate_code = 9'(gate_value);
            event_lane_id = 2'(lane_value);
            event_token_valid = valid_mask;
            event_token_ids = {3'(id3), 3'(id2), 3'(id1), 3'(id0)};
            event_count = 3'(count_value);
            event_issue_seq = 13'(sequence_value);
            event_term_first = first_value;
            event_term_last = last_value;
            event_head_last = head_last_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic finish_source(input logic [15:0] tag_value);
        begin
            @(negedge clk_core);
            source_done_tag = tag_value;
            source_done_valid = 1'b1;
            do @(posedge clk_core); while (!source_done_ready);
            if (!decoder_done_valid || decoder_done_tag != tag_value)
                $fatal(1, "decoder completion forwarding mismatch");
            decoder_cycle = cycle_count;
            @(negedge clk_core);
            source_done_valid = 1'b0;
        end
    endtask

    function automatic signed [7:0] weight_lane(
        input logic [5:0] channel,
        input int lane
    );
        begin
            case (channel)
                6'd4: weight_lane = lane == 0 ? 8'sd7 : -8'sd3;
                6'd5: weight_lane = lane == 0 ? -8'sd2 : 8'sd4;
                6'd7: weight_lane = lane == 0 ? 8'sd1 : 8'sd6;
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
            weight_rsp_weights[7:0] = weight_lane(
                weight_rsp_input_channel, 0);
            weight_rsp_weights[15:8] = weight_lane(
                weight_rsp_input_channel, 1);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            for (int token = 0; token < TOKENS; token = token + 1)
                for (int lane = 0; lane < OUT_TILE; lane = lane + 1)
                    observed[token][lane] <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (update_valid[bank] && update_ready[bank]) begin
                    if (update_tag != 16'h7101) begin
                        $fatal(1, "update tag mismatch");
                    end
                    observed[update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]][0]
                        <= observed[update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]][0] +
                           $signed({{15{update_values[16]}}, update_values[16:0]});
                    observed[update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]][1]
                        <= observed[update_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]][1] +
                           $signed({{15{update_values[33]}}, update_values[33:17]});
                end
            end
            if (backend_done_valid) begin
                if (outstanding_terms != 0 || count_completed_terms != 3)
                    $fatal(1, "backend completed before all terms drained");
            end
        end
    end

    always_comb begin
        update_ready[0] = (cycle_count % 4) != 1;
        update_ready[1] = (cycle_count % 5) != 2;
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        session_start_valid = 1'b0;
        session_tag = '0;
        session_input_channel_base = 6'd4;
        session_output_tile = 4'd2;
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
        source_done_tag = '0;
        source_done_error = 1'b0;
        decoder_done_ready = 1'b1;
        backend_done_ready = 1'b0;
        decoder_cycle = -1;
        backend_cycle = -1;
        first_completion_latency = -1;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        start_session(16'h7101);
        send_term(3, 1, 3, 0, 1'b0);
        send_event(3, 1, 4'b0011, 0, 2, 0, 0, 2, 0, 1'b1, 1'b0, 1'b0);
        send_event(3, 1, 4'b0001, 5, 0, 0, 0, 1, 0, 1'b0, 1'b1, 1'b0);
        send_term(5, 0, 1, 1, 1'b0);
        send_event(5, 0, 4'b0001, 1, 0, 0, 0, 1, 1, 1'b1, 1'b1, 1'b0);
        send_term(2, 3, 2, 2, 1'b1);
        send_event(2, 3, 4'b0011, 0, 7, 0, 0, 2, 2, 1'b1, 1'b1, 1'b1);
        finish_source(16'h7101);

        wait (backend_done_valid);
        @(negedge clk_core);
        backend_cycle = cycle_count;
        if (backend_done_error || protocol_error || decoder_done_error ||
            backend_done_tag != 16'h7101 ||
            backend_cycle <= decoder_cycle) begin
            $display("backend_error=%b protocol=%b decoder_error=%b backend_cycle=%0d decoder_cycle=%0d",
                     backend_done_error, protocol_error, decoder_done_error,
                     backend_cycle, decoder_cycle);
            $fatal(1, "completion/error contract mismatch");
        end
        first_completion_latency = backend_cycle - decoder_cycle;
        if (observed[0][0] != -4 || observed[0][1] != 24 ||
            observed[1][0] != 35 || observed[1][1] != -15 ||
            observed[2][0] != -6 || observed[2][1] != 12 ||
            observed[5][0] != -6 || observed[5][1] != 12 ||
            observed[7][0] != 2 || observed[7][1] != 12)
            $fatal(1, "multicast numerical result mismatch");
        backend_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        backend_done_ready = 1'b0;

        // Empty replay is legal and must not issue a weight request.
        start_session(16'h7102);
        finish_source(16'h7102);
        wait (backend_done_valid);
        if (backend_done_error || count_empty_sessions != 1 ||
            count_sessions != 2 || count_terms != 3 ||
            count_completed_terms != 3)
            $fatal(1, "empty session accounting mismatch");
        $display("PASS: TDR multicast backend sessions=%0d terms=%0d completed=%0d empty=%0d decoder_to_backend=%0d",
                 count_sessions, count_terms, count_completed_terms,
                 count_empty_sessions, first_completion_latency);
        $finish;
    end

    initial begin
        repeat (6000) @(posedge clk_core);
        $fatal(1, "TDR multicast backend TB timeout");
    end
endmodule

`default_nettype wire
