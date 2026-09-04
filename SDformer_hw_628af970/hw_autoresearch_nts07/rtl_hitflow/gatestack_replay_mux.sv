`timescale 1ns/1ps
`default_nettype none

// Session-locked mux for resident, sequential IPD32W and RAW-adapted issue
// streams. Arbitration happens once per head; no per-cycle path switching.
module gatestack_replay_mux #(
    parameter int SOURCES          = 3,
    parameter int EVENT_WAYS       = 4,
    parameter int TOKEN_ID_W       = 8,
    parameter int LANE_ID_W        = 5,
    parameter int ISSUE_SEQ_W      = 13,
    parameter int TAG_W            = 32,
    parameter int WAY_COUNT_W      = $clog2(EVENT_WAYS + 1),
    parameter int ROUTE_W          = (SOURCES <= 1) ? 1 : $clog2(SOURCES),
    parameter int COUNTER_W        = 32
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,

    input  logic                                      route_start_valid,
    output logic                                      route_start_ready,
    input  logic [ROUTE_W-1:0]                        route_start_select,
    output logic                                      route_active,
    output logic [ROUTE_W-1:0]                        route_active_select,

    input  logic [SOURCES-1:0]                        source_term_valid,
    output logic [SOURCES-1:0]                        source_term_ready,
    input  logic [(SOURCES*9)-1:0]                    source_term_gate_code,
    input  logic [(SOURCES*LANE_ID_W)-1:0]            source_term_lane_id,
    input  logic [(SOURCES*8)-1:0]                    source_term_destination_count,
    input  logic [SOURCES-1:0]                        source_term_head_last,

    input  logic [SOURCES-1:0]                        source_event_valid,
    output logic [SOURCES-1:0]                        source_event_ready,
    input  logic [(SOURCES*9)-1:0]                    source_event_gate_code,
    input  logic [(SOURCES*LANE_ID_W)-1:0]            source_event_lane_id,
    input  logic [(SOURCES*EVENT_WAYS)-1:0]           source_event_token_valid,
    input  logic [(SOURCES*EVENT_WAYS*TOKEN_ID_W)-1:0] source_event_token_ids,
    input  logic [(SOURCES*WAY_COUNT_W)-1:0]          source_event_count,
    input  logic [SOURCES-1:0]                        source_event_term_first,
    input  logic [SOURCES-1:0]                        source_event_term_last,
    input  logic [SOURCES-1:0]                        source_event_head_last,

    input  logic [SOURCES-1:0]                        source_done_valid,
    output logic [SOURCES-1:0]                        source_done_ready,
    input  logic [(SOURCES*TAG_W)-1:0]                source_done_tag,
    input  logic [SOURCES-1:0]                        source_done_error,

    output logic                                      term_valid,
    input  logic                                      term_ready,
    output logic [8:0]                                term_gate_code,
    output logic [LANE_ID_W-1:0]                      term_lane_id,
    output logic [7:0]                                term_destination_count,
    output logic [ISSUE_SEQ_W-1:0]                    term_issue_seq,
    output logic                                      term_head_last,

    output logic                                      event_valid,
    input  logic                                      event_ready,
    output logic [8:0]                                event_gate_code,
    output logic [LANE_ID_W-1:0]                      event_lane_id,
    output logic [EVENT_WAYS-1:0]                     event_token_valid,
    output logic [(EVENT_WAYS*TOKEN_ID_W)-1:0]        event_token_ids,
    output logic [WAY_COUNT_W-1:0]                    event_count,
    output logic [ISSUE_SEQ_W-1:0]                    event_issue_seq,
    output logic                                      event_term_first,
    output logic                                      event_term_last,
    output logic                                      event_head_last,

    output logic                                      done_valid,
    input  logic                                      done_ready,
    output logic [TAG_W-1:0]                          done_tag,
    output logic                                      done_error,

    output logic                                      protocol_error,
    output logic [COUNTER_W-1:0]                      count_completed_heads,
    output logic [(SOURCES*COUNTER_W)-1:0]            count_route_heads
);

    logic active_q;
    logic [ROUTE_W-1:0] route_select_q;
    logic [ISSUE_SEQ_W-1:0] term_seq_q;
    logic [ISSUE_SEQ_W-1:0] event_seq_q;
    logic route_start_fire;
    logic term_fire;
    logic event_fire;
    logic done_fire;
    logic selected_done_valid;
    logic selected_done_error;
    logic sequence_balanced;
    logic [8:0] source_term_gate_array [0:SOURCES-1];
    logic [LANE_ID_W-1:0] source_term_lane_array [0:SOURCES-1];
    logic [7:0] source_term_count_array [0:SOURCES-1];
    logic [8:0] source_event_gate_array [0:SOURCES-1];
    logic [LANE_ID_W-1:0] source_event_lane_array [0:SOURCES-1];
    logic [EVENT_WAYS-1:0] source_event_valid_array [0:SOURCES-1];
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] source_event_ids_array [0:SOURCES-1];
    logic [WAY_COUNT_W-1:0] source_event_count_array [0:SOURCES-1];
    logic [TAG_W-1:0] source_done_tag_array [0:SOURCES-1];
    logic [COUNTER_W-1:0] count_route_heads_q [0:SOURCES-1];

    for (genvar source = 0; source < SOURCES; source = source + 1) begin : g_unpack
        assign source_term_gate_array[source] =
            source_term_gate_code[(source*9) +: 9];
        assign source_term_lane_array[source] =
            source_term_lane_id[(source*LANE_ID_W) +: LANE_ID_W];
        assign source_term_count_array[source] =
            source_term_destination_count[(source*8) +: 8];
        assign source_event_gate_array[source] =
            source_event_gate_code[(source*9) +: 9];
        assign source_event_lane_array[source] =
            source_event_lane_id[(source*LANE_ID_W) +: LANE_ID_W];
        assign source_event_valid_array[source] =
            source_event_token_valid[(source*EVENT_WAYS) +: EVENT_WAYS];
        assign source_event_ids_array[source] = source_event_token_ids[
            (source*EVENT_WAYS*TOKEN_ID_W) +: (EVENT_WAYS*TOKEN_ID_W)];
        assign source_event_count_array[source] =
            source_event_count[(source*WAY_COUNT_W) +: WAY_COUNT_W];
        assign source_done_tag_array[source] =
            source_done_tag[(source*TAG_W) +: TAG_W];
        assign count_route_heads[(source*COUNTER_W) +: COUNTER_W] =
            count_route_heads_q[source];
    end

    assign route_start_ready = !active_q;
    assign route_start_fire = route_start_valid && route_start_ready;
    assign route_active = active_q;
    assign route_active_select = route_select_q;
    assign term_issue_seq = term_seq_q;
    assign event_issue_seq = event_seq_q;
    assign term_fire = term_valid && term_ready;
    assign event_fire = event_valid && event_ready;
    assign done_fire = done_valid && done_ready;
    assign sequence_balanced = term_seq_q == event_seq_q;

    always_comb begin
        source_term_ready = '0;
        source_event_ready = '0;
        source_done_ready = '0;
        term_valid = 1'b0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_head_last = 1'b0;
        event_valid = 1'b0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        done_valid = 1'b0;
        done_tag = '0;
        selected_done_valid = 1'b0;
        selected_done_error = 1'b0;
        if (active_q && 32'(route_select_q) < SOURCES) begin
            term_valid = source_term_valid[route_select_q];
            term_gate_code = source_term_gate_array[route_select_q];
            term_lane_id = source_term_lane_array[route_select_q];
            term_destination_count = source_term_count_array[route_select_q];
            term_head_last = source_term_head_last[route_select_q];
            source_term_ready[route_select_q] = term_ready;

            event_valid = source_event_valid[route_select_q];
            event_gate_code = source_event_gate_array[route_select_q];
            event_lane_id = source_event_lane_array[route_select_q];
            event_token_valid = source_event_valid_array[route_select_q];
            event_token_ids = source_event_ids_array[route_select_q];
            event_count = source_event_count_array[route_select_q];
            event_term_first = source_event_term_first[route_select_q];
            event_term_last = source_event_term_last[route_select_q];
            event_head_last = source_event_head_last[route_select_q];
            source_event_ready[route_select_q] = event_ready;

            selected_done_valid = source_done_valid[route_select_q];
            selected_done_error = source_done_error[route_select_q];
            done_valid = selected_done_valid;
            done_tag = source_done_tag_array[route_select_q];
            source_done_ready[route_select_q] = done_ready;
        end
        done_error = selected_done_error ||
                     (selected_done_valid && !sequence_balanced);
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            route_select_q <= '0;
            term_seq_q <= '0;
            event_seq_q <= '0;
            protocol_error <= 1'b0;
            count_completed_heads <= '0;
            for (int source = 0; source < SOURCES; source = source + 1) begin
                count_route_heads_q[source] <= '0;
            end
        end else begin
            if (route_start_fire) begin
                if (32'(route_start_select) >= SOURCES) begin
                    protocol_error <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    route_select_q <= route_start_select;
                    term_seq_q <= '0;
                    event_seq_q <= '0;
                end
            end
            if (term_fire) begin
                term_seq_q <= term_seq_q + 1'b1;
            end
            if (event_fire && event_term_last) begin
                event_seq_q <= event_seq_q + 1'b1;
            end
            if (done_fire) begin
                active_q <= 1'b0;
                count_completed_heads <= count_completed_heads + 1'b1;
                count_route_heads_q[route_select_q] <=
                    count_route_heads_q[route_select_q] + 1'b1;
                if (done_error) begin
                    protocol_error <= 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
