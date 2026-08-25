`timescale 1ns/1ps
`default_nettype none

module m186_fc2_k8_fixed_bank_issue_island_assertions #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12
) (
    input logic clk_core,
    input logic rst_core,
    input logic header_valid,
    input logic header_ready,
    input logic header_accept,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic descriptor_accept,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic weight_request_accept,
    input logic [TAG_BITS-1:0] weight_request_tag,
    input logic [2:0] weight_request_output_block,
    input logic [3:0] weight_request_source_count,
    input logic [7:0] weight_request_bank_valid,
    input logic [CHANNEL_BITS-1:0] weight_request_source_channel [0:7],
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic weight_response_accept,
    input logic result_valid,
    input logic result_ready,
    input logic result_accept,
    input logic [TAG_BITS-1:0] result_token_tag,
    input logic [2:0] result_output_block,
    input logic [3:0] result_source_count,
    input logic [7:0] result_bank_mask,
    input logic signed [23:0] result_accumulator [0:LANES-1],
    input logic token_done_valid,
    input logic token_done_ready,
    input logic token_done_accept,
    input logic [TAG_BITS-1:0] token_done_tag,
    input logic token_done_had_event,
    input logic protocol_error,
    input logic numeric_overflow,
    input logic busy,
    input logic pending_valid_q,
    input logic m184_group_accept,
    input logic m184_done_valid,
    input logic m185_issue_accept,
    input logic m185_busy
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_handshake:
        assert property (header_accept == (header_valid && header_ready));
    ap_descriptor_handshake:
        assert property (descriptor_accept
            == (descriptor_valid && descriptor_ready));
    ap_request_handshake:
        assert property (weight_request_accept
            == (weight_request_valid && weight_request_ready));
    ap_response_handshake:
        assert property (weight_response_accept
            == (weight_response_valid && weight_response_ready));
    ap_result_handshake:
        assert property (result_accept == (result_valid && result_ready));
    ap_done_handshake:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_frontend_request_identity:
        assert property (m184_group_accept == weight_request_accept);
    ap_response_issue_identity:
        assert property (m185_issue_accept == weight_response_accept);
    ap_response_requires_pending:
        assert property (weight_response_accept |-> pending_valid_q);
    ap_request_nonempty:
        assert property (weight_request_accept
            |-> weight_request_bank_valid != 0
                && weight_request_source_count
                    == $countones(weight_request_bank_valid));
    ap_done_after_arithmetic_empty:
        assert property (token_done_valid |-> !pending_valid_q && !m185_busy);
    ap_busy_covers_state:
        assert property ((pending_valid_q || result_valid
            || token_done_valid) |-> busy);
    ap_protocol_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_overflow_sticky:
        assert property (numeric_overflow |=> numeric_overflow);
    ap_fault_stops_new_requests:
        assert property ((protocol_error || numeric_overflow)
            |=> !header_ready && !descriptor_ready && !weight_request_valid);
    ap_hold_request_on_stall:
        assert property (weight_request_valid && !weight_request_ready |=>
            $stable({weight_request_valid, weight_request_tag,
                     weight_request_output_block,
                     weight_request_source_count,
                     weight_request_bank_valid,
                     weight_request_source_channel[0],
                     weight_request_source_channel[1],
                     weight_request_source_channel[2],
                     weight_request_source_channel[3],
                     weight_request_source_channel[4],
                     weight_request_source_channel[5],
                     weight_request_source_channel[6],
                     weight_request_source_channel[7]}));
    ap_hold_result_on_stall:
        assert property (result_valid && !result_ready |=>
            $stable({result_valid, result_token_tag, result_output_block,
                     result_source_count, result_bank_mask}));
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_identity
            ap_request_bank_identity:
                assert property (weight_request_valid
                    && weight_request_bank_valid[bank]
                    |-> weight_request_source_channel[bank][2:0]
                        == bank[2:0]);
        end
        for (genvar lane = 0; lane < LANES; lane++) begin : g_hold_result_lane
            ap_hold_result_accumulator:
                assert property (result_valid && !result_ready |=>
                    $stable(result_accumulator[lane]));
        end
    endgenerate

    cp_nonprefix_request:
        cover property (weight_request_accept
            && weight_request_bank_valid != 0
            && weight_request_bank_valid != 8'hff
            && weight_request_bank_valid[7]
            && !weight_request_bank_valid[6]);
    cp_request_stall_then_accept:
        cover property (weight_request_valid && !weight_request_ready
            ##1 weight_request_accept);
    cp_response_stall_then_accept:
        cover property (weight_response_valid && !weight_response_ready
            ##1 weight_response_accept);
    cp_result_stall_then_accept:
        cover property (result_valid && !result_ready ##1 result_accept);
    cp_same_cycle_response_request_replace:
        cover property (weight_response_accept && weight_request_accept);
    cp_done_waits_for_arithmetic:
        cover property (m184_done_valid && !token_done_valid
            ##[1:20] token_done_valid);
    cp_zero_token_done:
        cover property (token_done_accept && !token_done_had_event);
    cp_nonzero_token_done:
        cover property (token_done_accept && token_done_had_event);
    cp_protocol_fault:
        cover property (protocol_error);
    cp_numeric_overflow:
        cover property (numeric_overflow && result_valid);
`endif
endmodule

`default_nettype wire
