`timescale 1ns/1ps
`default_nettype none

// One-event delay that marks the true final RAW event even when trailing token
// records are K-zero. Decoder done is held until the retained event is issued.
module gatestack_raw_tail_retimer #(
    parameter int TAG_W       = 32,
    parameter int LANE_ID_W   = 5,
    parameter int TOKEN_ID_W  = 8,
    parameter int COUNTER_W   = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [8:0]                   input_gate_code,
    input  logic [LANE_ID_W-1:0]         input_lane_id,
    input  logic [TOKEN_ID_W-1:0]        input_token_id,
    input  logic                         input_done_valid,
    output logic                         input_done_ready,
    input  logic [TAG_W-1:0]             input_done_tag,
    input  logic                         input_done_error,
    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [8:0]                   output_gate_code,
    output logic [LANE_ID_W-1:0]         output_lane_id,
    output logic [TOKEN_ID_W-1:0]        output_token_id,
    output logic                         output_head_last,
    output logic                         output_done_valid,
    input  logic                         output_done_ready,
    output logic [TAG_W-1:0]             output_done_tag,
    output logic                         output_done_error,
    output logic                         protocol_error,
    output logic [COUNTER_W-1:0]         count_inputs,
    output logic [COUNTER_W-1:0]         count_outputs,
    output logic [COUNTER_W-1:0]         count_empty_sessions
);
    logic buffer_valid_q;
    logic [8:0] gate_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [TOKEN_ID_W-1:0] token_q;
    logic session_input_seen_q;
    logic input_fire;
    logic output_fire;
    logic done_fire;

    assign output_valid = buffer_valid_q &&
                          (input_valid || input_done_valid);
    assign output_gate_code = gate_q;
    assign output_lane_id = lane_q;
    assign output_token_id = token_q;
    assign output_head_last = output_valid && input_done_valid && !input_valid;
    assign output_fire = output_valid && output_ready;
    assign input_ready = !buffer_valid_q || (output_fire && input_valid);
    assign input_fire = input_valid && input_ready;

    assign output_done_valid = input_done_valid && !buffer_valid_q;
    assign output_done_tag = input_done_tag;
    assign output_done_error = input_done_error;
    assign input_done_ready = output_done_ready && !buffer_valid_q;
    assign done_fire = output_done_valid && output_done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            buffer_valid_q <= 1'b0;
            gate_q <= '0;
            lane_q <= '0;
            token_q <= '0;
            session_input_seen_q <= 1'b0;
            protocol_error <= 1'b0;
            count_inputs <= '0;
            count_outputs <= '0;
            count_empty_sessions <= '0;
        end else begin
            if (output_fire) begin
                buffer_valid_q <= 1'b0;
                count_outputs <= count_outputs + 1'b1;
            end
            if (input_fire) begin
                buffer_valid_q <= 1'b1;
                gate_q <= input_gate_code;
                lane_q <= input_lane_id;
                token_q <= input_token_id;
                session_input_seen_q <= 1'b1;
                count_inputs <= count_inputs + 1'b1;
            end
            if (done_fire) begin
                if (!session_input_seen_q)
                    count_empty_sessions <= count_empty_sessions + 1'b1;
                session_input_seen_q <= 1'b0;
            end
            if ((input_done_valid && input_valid) ||
                (done_fire && count_inputs != count_outputs))
                protocol_error <= 1'b1;
        end
    end
endmodule

`default_nettype wire
