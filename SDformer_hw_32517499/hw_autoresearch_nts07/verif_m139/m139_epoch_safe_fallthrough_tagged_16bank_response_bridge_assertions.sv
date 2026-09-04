`timescale 1ns/1ps
`default_nettype none

module m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic request_accept,
    input logic macro_flush_req,
    input logic macro_flush_ack,
    input logic macro_request_valid,
    input logic [127:0] macro_bank_row_addresses,
    input logic [15:0] macro_request_token,
    input logic macro_response_valid,
    input logic [15:0] macro_response_token,
    input logic response_valid,
    input logic response_ready,
    input logic [511:0] response_logical_words,
    input logic response_start,
    input logic response_last,
    input logic [3:0] response_width,
    input logic [31:0] response_tag,
    input logic [15:0] response_token,
    input logic response_accept,
    input logic protocol_error,
    input logic recovery_active,
    input logic pending_response,
    input logic [1:0] buffered_responses,
    input logic [1:0] recovery_state_q
);
`ifdef SVA_RUNTIME_ENABLED
    localparam logic [1:0] REC_WAIT_ACK_LOW  = 2'd0;
    localparam logic [1:0] REC_WAIT_ACK_HIGH = 2'd1;
    localparam logic [1:0] REC_WAIT_ACK_DROP = 2'd2;
    localparam logic [1:0] REC_RUN           = 2'd3;

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_recovery_quiet: assert property (
        recovery_active |-> !request_ready && !request_accept
                         && !macro_request_valid && !response_valid
                         && !response_accept && !pending_response
                         && buffered_responses == 0);
    ap_flush_request_decode: assert property (
        macro_flush_req == (recovery_state_q inside
                            {REC_WAIT_ACK_LOW, REC_WAIT_ACK_HIGH}));
    ap_wait_low_transition: assert property (
        recovery_state_q == REC_WAIT_ACK_LOW && !macro_flush_ack
        |=> recovery_state_q == REC_WAIT_ACK_HIGH || protocol_error);
    ap_wait_high_transition: assert property (
        recovery_state_q == REC_WAIT_ACK_HIGH && macro_flush_ack
        && !macro_response_valid
        |=> recovery_state_q == REC_WAIT_ACK_DROP || protocol_error);
    ap_wait_drop_transition: assert property (
        recovery_state_q == REC_WAIT_ACK_DROP && !macro_flush_ack
        && !macro_response_valid
        |=> recovery_state_q == REC_RUN || protocol_error);
    ap_completion_collision_faults: assert property (
        recovery_state_q == REC_WAIT_ACK_HIGH && macro_flush_ack
        && macro_response_valid |-> protocol_error);
    ap_post_completion_response_faults: assert property (
        recovery_state_q == REC_WAIT_ACK_DROP && macro_response_valid
        |-> protocol_error);
    ap_accept_requires_run: assert property (
        request_accept |-> recovery_state_q == REC_RUN
                         && request_valid && request_ready);
    ap_macro_only_on_accept: assert property (
        macro_request_valid |-> request_accept);
    ap_response_accept_definition: assert property (
        response_accept |-> response_valid && response_ready);
    ap_no_output_on_error: assert property (
        protocol_error |-> !request_ready && !request_accept
                        && !macro_request_valid && !response_valid
                        && !response_accept);
    ap_full_stalled_response_stable: assert property (
        response_valid && !response_ready && !protocol_error
        |=> protocol_error || (response_valid
            && $stable({response_logical_words, response_start,
                        response_last, response_width, response_tag,
                        response_token})));
    ap_single_skid_bounded: assert property (buffered_responses <= 1);
    ap_error_sticky: assert property (protocol_error |=> protocol_error);
    ap_reset_quiet_and_flush: assert property (
        @(posedge clk_core) disable iff (1'b0)
        rst_core |-> macro_flush_req && !request_accept
                    && !macro_request_valid && !response_valid
                    && !protocol_error);

    cp_minimum_flush: cover property (
        recovery_state_q == REC_WAIT_ACK_LOW && !macro_flush_ack
        ##1 recovery_state_q == REC_WAIT_ACK_HIGH && macro_flush_ack
        ##1 recovery_state_q == REC_WAIT_ACK_DROP && !macro_flush_ack
        ##1 recovery_state_q == REC_RUN);
    cp_initial_high_ack_rejected: cover property (
        recovery_state_q == REC_WAIT_ACK_LOW && macro_flush_ack[*2]
        ##1 !macro_flush_ack);
    cp_drain_response_dropped: cover property (
        recovery_active && macro_response_valid && !response_valid);
    cp_completion_collision: cover property (
        recovery_state_q == REC_WAIT_ACK_HIGH && macro_flush_ack
        && macro_response_valid && protocol_error);
    cp_post_completion_response: cover property (
        recovery_state_q == REC_WAIT_ACK_DROP && macro_response_valid
        && protocol_error);
    cp_first_token_zero: cover property (
        $past(recovery_state_q) != REC_RUN
        && recovery_state_q == REC_RUN ##1 request_accept
        && macro_request_token == 0);
    cp_contiguous_eight_requests: cover property (request_accept[*8]);
    cp_skid_capture_release: cover property (
        response_valid && !response_ready ##1 buffered_responses == 1
        ##1 response_accept);
    cp_token_wrap: cover property (
        request_accept && macro_request_token == 16'hffff
        ##1 request_accept && macro_request_token == 16'h0000);
`endif
endmodule

bind m139_epoch_safe_fallthrough_tagged_16bank_response_bridge
    m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_assertions
    m139_sva (.*);

`default_nettype wire
