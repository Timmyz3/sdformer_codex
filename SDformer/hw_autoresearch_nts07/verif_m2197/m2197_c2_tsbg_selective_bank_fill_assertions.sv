`timescale 1ns/1ps
`default_nettype none

module m2197_c2_tsbg_selective_bank_fill_assertions #(
    parameter int LANES = 16,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic rst_core,
    input logic load_accept,
    input logic [15:0] load_source_active,
    input logic [7:0] mem_req_valid,
    input logic [7:0] mem_req_ready,
    input logic [7:0] mem_req_accept,
    input logic [EPOCH_BITS-1:0] mem_req_epoch [0:7],
    input logic [2:0] mem_req_slot [0:7],
    input logic [GENERATION_BITS-1:0] mem_req_generation [0:7],
    input logic [TAG_BITS-1:0] mem_req_tag [0:7],
    input logic [2:0] mem_req_output_block [0:7],
    input logic [2:0] mem_req_slice [0:7],
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    input logic [7:0] mem_rsp_valid,
    input logic [7:0] mem_rsp_ready,
    input logic [7:0] mem_rsp_accept,
    input logic bridge_valid,
    input logic bridge_ready,
    input logic [7:0] bridge_bank_valid,
    input logic signed [1:0] bridge_source_value [0:7],
    input logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1],
    input logic commit_valid,
    input logic commit_ready,
    input logic [2:0] commit_context,
    input logic [TAG_BITS-1:0] commit_tag,
    input logic [2:0] commit_slice,
    input logic signed [23:0] commit_accumulator [0:LANES-1],
    input logic commit_terminal,
    input logic protocol_error,
    input logic stale_response_seen,
    input logic numeric_overflow,
    input logic [31:0] debug_partial_hit_count,
    input logic [31:0] debug_cache_eviction_count,
    input logic [31:0] debug_refill_bank_request_count,
    input logic [31:0] debug_scalar_bank_request_count,
    input logic [31:0] debug_zero_descriptor_skip_count
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
    ap_commit_context: assert property (commit_valid |-> commit_context < 4);
    ap_commit_slice: assert property (commit_valid |-> commit_slice < 6);
    ap_terminal_true: assert property (
        commit_valid && commit_terminal |-> commit_slice == 5);
    ap_terminal_false: assert property (
        commit_valid && commit_slice != 5 |-> !commit_terminal);
    ap_commit_header_hold: assert property (
        commit_valid && !commit_ready |=> commit_valid
        && $stable({commit_context, commit_tag, commit_slice, commit_terminal}));
    ap_commit_payload_hold: assert property (
        commit_valid && !commit_ready |=> $stable(commit_accumulator));
    ap_refill_counter_monotonic: assert property (
        debug_refill_bank_request_count >= $past(debug_refill_bank_request_count));
    ap_partial_counter_monotonic: assert property (
        debug_partial_hit_count >= $past(debug_partial_hit_count));
    ap_eviction_counter_monotonic: assert property (
        debug_cache_eviction_count >= $past(debug_cache_eviction_count));
    ap_adapter_counter_not_ahead: assert property (
        debug_scalar_bank_request_count <= debug_refill_bank_request_count);

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank
            ap_request_bank_identity: assert property (
                mem_req_valid[bank] |-> mem_req_source_channel[bank][2:0] == bank);
            ap_request_hold: assert property (
                mem_req_valid[bank] && !mem_req_ready[bank] |=>
                    mem_req_valid[bank]
                    && $stable({mem_req_epoch[bank], mem_req_slot[bank],
                                mem_req_generation[bank], mem_req_tag[bank],
                                mem_req_output_block[bank], mem_req_slice[bank],
                                mem_req_source_channel[bank]}));
            ap_source_domain: assert property (
                bridge_valid && bridge_bank_valid[bank] |->
                    bridge_source_value[bank] inside {-2'sd1, 2'sd1});
            ap_inactive_source_zero: assert property (
                bridge_valid && !bridge_bank_valid[bank] |->
                    bridge_source_value[bank] == 0);
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_inactive_product_zero: assert property (
                    bridge_valid && !bridge_bank_valid[bank] |->
                        bridge_effective_weight[bank][lane] == 0);
            end
        end
    endgenerate

    cp_partial_refill: cover property (debug_partial_hit_count > 0);
    cp_eviction: cover property (debug_cache_eviction_count > 0);
    cp_selective_request: cover property (
        mem_req_valid != 0 && mem_req_valid != 8'hff);
    cp_independent_bank_backpressure: cover property (
        mem_req_valid != 0 && (mem_req_valid & ~mem_req_ready) != 0
        && (mem_req_valid & mem_req_ready) != 0);
    cp_response_reorder: cover property (
        (|mem_rsp_accept[7:4]) ##[1:20] (|mem_rsp_accept[3:0]));
    cp_bridge_backpressure: cover property (bridge_valid && !bridge_ready);
    cp_commit_backpressure: cover property (commit_valid && !commit_ready);
    cp_positive_source: cover property (
        bridge_valid && bridge_bank_valid[0] && bridge_source_value[0] == 1);
    cp_negative_source: cover property (
        bridge_valid && bridge_bank_valid[0] && bridge_source_value[0] == -1);
    cp_terminal: cover property (commit_valid && commit_ready && commit_terminal);
    cp_zero_descriptor_skip: cover property (debug_zero_descriptor_skip_count > 0);
endmodule

`default_nettype wire

