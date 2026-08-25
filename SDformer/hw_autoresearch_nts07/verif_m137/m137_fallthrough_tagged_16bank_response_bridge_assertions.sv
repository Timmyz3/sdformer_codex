`timescale 1ns/1ps
`default_nettype none

module m137_fallthrough_tagged_16bank_response_bridge_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic [11:0] logical_base_word,
    input logic request_accept,
    input logic macro_request_valid,
    input logic [15:0] macro_request_token,
    input logic macro_response_valid,
    input logic [15:0] macro_response_token,
    input logic response_valid,
    input logic response_ready,
    input logic [511:0] response_logical_words,
    input logic [31:0] response_tag,
    input logic [15:0] response_token,
    input logic response_accept,
    input logic protocol_error,
    input logic pending_response,
    input logic [1:0] buffered_responses
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_accept_drives_macro: assert property (
        request_accept |-> macro_request_valid);
    ap_macro_only_on_accept: assert property (
        macro_request_valid |-> request_accept);
    ap_accept_requires_ready: assert property (
        request_accept |-> request_valid && request_ready);
    ap_illegal_request_quarantined: assert property (
        request_valid && ({1'b0, logical_base_word} + 13'd15 >= 13'd3680)
        |-> !request_accept && protocol_error);
    ap_fixed_one_cycle_response_or_fault: assert property (
        request_accept |=> ((macro_response_valid
                             && macro_response_token == $past(macro_request_token))
                            || protocol_error));
    ap_pending_requires_response_or_fault: assert property (
        pending_response |-> macro_response_valid || protocol_error);
    ap_unsolicited_response_faults: assert property (
        macro_response_valid && !pending_response |-> protocol_error);
    ap_no_output_on_error: assert property (
        protocol_error |-> !response_valid && !response_accept && !request_accept);
    ap_response_accept_definition: assert property (
        response_accept |-> response_valid && response_ready);
    ap_fallthrough_accept: assert property (
        macro_response_valid && pending_response && buffered_responses == 0
        && response_ready && !protocol_error |-> response_accept);
    ap_stalled_response_stable: assert property (
        response_valid && !response_ready && !protocol_error
        |=> protocol_error || (response_valid
            && $stable({response_logical_words, response_tag, response_token})));
    ap_single_skid_bounded: assert property (buffered_responses <= 1);
    ap_error_sticky: assert property (protocol_error |=> protocol_error);
    ap_reset_quiet: assert property (@(posedge clk_core) disable iff (1'b0)
        rst_core |-> !request_accept && !macro_request_valid
                    && !response_valid && !protocol_error);

    cp_contiguous_eight_requests: cover property (request_accept[*8]);
    cp_fallthrough_and_next_request: cover property (
        response_accept && request_accept && buffered_responses == 0);
    cp_skid_capture_under_stall: cover property (
        response_valid && !response_ready ##1 buffered_responses == 1);
    cp_skid_release_and_request: cover property (
        buffered_responses == 1 && response_accept && request_accept);
    cp_cross_row_request: cover property (
        request_accept && logical_base_word[3:0] != 0);
    cp_wrong_token_quarantine: cover property (
        macro_response_valid && pending_response
        && macro_response_token != $past(macro_request_token)
        ##0 protocol_error);
    cp_unsolicited_quarantine: cover property (
        macro_response_valid && !pending_response ##0 protocol_error);
`endif
endmodule

bind m137_fallthrough_tagged_16bank_response_bridge
    m137_fallthrough_tagged_16bank_response_bridge_assertions m137_sva (.*);

`default_nettype wire
