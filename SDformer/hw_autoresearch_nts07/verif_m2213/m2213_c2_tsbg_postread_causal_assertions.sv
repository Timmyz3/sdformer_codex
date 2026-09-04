`timescale 1ns/1ps
`default_nettype none

module m2213_c2_tsbg_postread_causal_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic [7:0] mem_req_valid,
    input logic [7:0] mem_req_ready,
    input logic [7:0] mem_req_accept,
    input logic [7:0] mem_rsp_valid,
    input logic [7:0] mem_rsp_ready,
    input logic [7:0] mem_rsp_accept,
    input logic bridge_valid,
    input logic bridge_ready,
    input logic bridge_accept,
    input logic commit_valid,
    input logic commit_ready,
    input logic commit_accept,
    input logic [2:0] commit_context,
    input logic [2:0] commit_slice,
    input logic [23:0] commit_tag,
    input logic commit_terminal,
    input logic signed [23:0] commit_accumulator [0:15],
    input logic bundle_done_valid,
    input logic protocol_error,
    input logic stale_response_seen,
    input logic numeric_overflow,
    input logic [31:0] debug_postread_row_count,
    input logic [31:0] debug_postread_bundle_request_count,
    input logic [31:0] debug_postread_bundle_response_count,
    input logic [31:0] debug_postread_bank_request_count,
    input logic [31:0] debug_postread_bank_response_count,
    input logic [31:0] debug_postread_identity_accept_count
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_no_protocol_error: assert property (!protocol_error);
    ap_no_stale_response: assert property (!stale_response_seen);
    ap_no_numeric_overflow: assert property (!numeric_overflow);
    ap_req_accept_definition: assert property (
        mem_req_accept == (mem_req_valid & mem_req_ready));
    ap_rsp_accept_definition: assert property (
        mem_rsp_accept == (mem_rsp_valid & mem_rsp_ready));
    ap_bridge_accept_definition: assert property (
        bridge_accept == (bridge_valid && bridge_ready));
    ap_commit_accept_definition: assert property (
        commit_accept == (commit_valid && commit_ready));
    ap_commit_identity_domain: assert property (
        commit_valid |-> commit_context < 4 && commit_slice < 6);
    ap_terminal_identity: assert property (
        commit_valid |-> commit_terminal == (commit_slice == 5));
    ap_commit_header_hold: assert property (
        commit_valid && !commit_ready |=> commit_valid
        && $stable({commit_context, commit_slice, commit_tag,
                    commit_terminal}));
    ap_commit_payload_hold: assert property (
        commit_valid && !commit_ready |=> $stable(commit_accumulator));

    ap_postread_request_not_behind_response: assert property (
        debug_postread_bundle_response_count
            <= debug_postread_bundle_request_count);
    ap_postread_bank_request_not_behind_response: assert property (
        debug_postread_bank_response_count
            <= debug_postread_bank_request_count);
    ap_postread_identity_accept_exact: assert property (
        debug_postread_identity_accept_count
            == debug_postread_bundle_response_count);
    ap_postread_bundle_bound: assert property (
        debug_postread_bundle_request_count
            <= debug_postread_row_count * 12);
    ap_postread_bank_bound: assert property (
        debug_postread_bank_request_count
            <= debug_postread_row_count * 96);
    ap_postread_done_has_real_reads: assert property (
        bundle_done_valid |-> debug_postread_row_count > 0
        && debug_postread_bundle_request_count
            == debug_postread_row_count * 12
        && debug_postread_bundle_response_count
            == debug_postread_row_count * 12
        && debug_postread_bank_request_count
            == debug_postread_row_count * 96
        && debug_postread_bank_response_count
            == debug_postread_row_count * 96
        && debug_postread_identity_accept_count
            == debug_postread_row_count * 12);

    cp_real_postread_request: cover property (
        debug_postread_row_count > 0 && |mem_req_accept);
    cp_real_postread_response: cover property (
        debug_postread_row_count > 0 && |mem_rsp_accept);
    cp_postread_commit_terminal: cover property (
        debug_postread_row_count > 0 && commit_accept && commit_terminal);
endmodule

`default_nettype wire
