`timescale 1ns/1ps
`default_nettype none

module m1874_c2_tsbg_b4_real_channel_signed_frontend_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SOURCE_GROUPS = 48,
    parameter int TOKEN_CONTEXTS = 4,
    parameter int LANES = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic load_valid,
    input logic load_ready,
    input logic load_accept,
    input logic [2:0] load_context,
    input logic [7:0] mem_req_valid,
    input logic [7:0] mem_req_ready,
    input logic [EPOCH_BITS-1:0] mem_req_epoch [0:7],
    input logic [2:0] mem_req_slot [0:7],
    input logic [GENERATION_BITS-1:0] mem_req_generation [0:7],
    input logic [TAG_BITS-1:0] mem_req_tag [0:7],
    input logic [2:0] mem_req_output_block [0:7],
    input logic [2:0] mem_req_slice [0:7],
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    input logic [7:0] mem_req_accept,
    input logic [7:0] mem_rsp_valid,
    input logic [7:0] mem_rsp_ready,
    input logic [EPOCH_BITS-1:0] mem_rsp_epoch [0:7],
    input logic [2:0] mem_rsp_slot [0:7],
    input logic [GENERATION_BITS-1:0] mem_rsp_generation [0:7],
    input logic [TAG_BITS-1:0] mem_rsp_tag [0:7],
    input logic signed [7:0] mem_rsp_weight [0:7][0:LANES-1],
    input logic [7:0] mem_rsp_accept,
    input logic bridge_valid,
    input logic bridge_ready,
    input logic [2:0] bridge_context,
    input logic [5:0] bridge_group,
    input logic bridge_half,
    input logic [2:0] bridge_slice,
    input logic [7:0] bridge_bank_valid,
    input logic [CHANNEL_BITS-1:0] bridge_source_channel [0:7],
    input logic signed [1:0] bridge_source_value [0:7],
    input logic signed [8:0] bridge_effective_weight [0:7][0:LANES-1],
    input logic bridge_accept,
    input logic commit_valid,
    input logic commit_ready,
    input logic [2:0] commit_context,
    input logic [TAG_BITS-1:0] commit_tag,
    input logic [2:0] commit_slice,
    input logic signed [23:0] commit_accumulator [0:LANES-1],
    input logic commit_terminal,
    input logic commit_accept,
    input logic protocol_error,
    input logic stale_response_seen,
    input logic numeric_overflow,
    input logic [31:0] debug_cache_eviction_count,
    input logic [31:0] debug_weight_bundle_beat_count
);
    localparam int PRODUCTION_ACC24_ABS_BOUND = 48 * 16 * 128;
    localparam int ELABORATED_ACC24_ABS_BOUND = SOURCE_GROUPS * 16 * 128;

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_load_accept_definition: assert property (
        load_accept |-> load_valid && load_ready);
    ap_load_context_is_b4: assert property (
        load_accept |-> load_context < TOKEN_CONTEXTS);
    ap_bridge_accept_definition: assert property (
        bridge_accept |-> bridge_valid && bridge_ready);
    ap_bridge_context_is_b4: assert property (
        bridge_valid |-> bridge_context < TOKEN_CONTEXTS);
    ap_commit_accept_definition: assert property (
        commit_accept |-> commit_valid && commit_ready);
    ap_commit_context_is_b4: assert property (
        commit_valid |-> commit_context < TOKEN_CONTEXTS);
    ap_bridge_nonempty: assert property (
        bridge_valid |-> bridge_bank_valid != 0);
    ap_terminal_is_slice5: assert property (
        commit_valid && commit_terminal |-> commit_slice == 3'd5);
    ap_nonterminal_before_slice5: assert property (
        commit_valid && commit_slice != 3'd5 |-> !commit_terminal);
    ap_fault_is_sticky: assert property (
        protocol_error |=> protocol_error);
    ap_fault_closes_load: assert property (
        protocol_error |-> !load_ready);
    ap_no_legal_overflow: assert property (!numeric_overflow);

    // Static proof obligation requested by M1781.  98,304 is the maximum
    // absolute legal accumulation and is strictly inside signed Acc24.
    initial begin
        if (SOURCE_GROUPS < 1 || SOURCE_GROUPS > 48
                || TOKEN_CONTEXTS != 4
                || PRODUCTION_ACC24_ABS_BOUND != 98304
                || ELABORATED_ACC24_ABS_BOUND > PRODUCTION_ACC24_ABS_BOUND
                || PRODUCTION_ACC24_ABS_BOUND >= (1 << 23))
            $fatal(1, "M1874 Acc24 static bound invalid");
    end

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_protocol
            ap_bank_request_accept: assert property (
                mem_req_accept[bank] |-> mem_req_valid[bank]
                    && mem_req_ready[bank]);
            ap_bank_response_accept: assert property (
                mem_rsp_accept[bank] |-> mem_rsp_valid[bank]
                    && mem_rsp_ready[bank]);
            ap_bank_request_stable: assert property (
                mem_req_valid[bank] && !mem_req_ready[bank] |=>
                    mem_req_valid[bank]
                    && $stable({mem_req_epoch[bank], mem_req_slot[bank],
                        mem_req_generation[bank], mem_req_tag[bank],
                        mem_req_output_block[bank], mem_req_slice[bank],
                        mem_req_source_channel[bank]}));
            ap_bank_response_stable: assert property (
                mem_rsp_valid[bank] && !mem_rsp_ready[bank]
                    && !protocol_error |=>
                    mem_rsp_valid[bank]
                    && $stable({mem_rsp_epoch[bank], mem_rsp_slot[bank],
                        mem_rsp_generation[bank], mem_rsp_tag[bank]})
                    && $stable(mem_rsp_weight[bank]));
            ap_source_bank_binding: assert property (
                bridge_valid && bridge_bank_valid[bank] |->
                    bridge_source_channel[bank][2:0] == bank[2:0]);
            ap_zero_does_not_issue: assert property (
                bridge_valid && !bridge_bank_valid[bank] |->
                    bridge_source_value[bank] == 0);
            ap_nonzero_is_typed_unit: assert property (
                bridge_valid && bridge_bank_valid[bank] |->
                    (bridge_source_value[bank] == 2'sd1
                     || bridge_source_value[bank] == -2'sd1));
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                ap_inactive_effective_weight_zero: assert property (
                    bridge_valid && !bridge_bank_valid[bank] |->
                        bridge_effective_weight[bank][lane] == 0);
            end
        end
    endgenerate

    ap_bridge_header_stable: assert property (
        bridge_valid && !bridge_ready |=> bridge_valid
            && $stable({bridge_context, bridge_group, bridge_half,
                        bridge_slice, bridge_bank_valid}));
    ap_bridge_payload_stable: assert property (
        bridge_valid && !bridge_ready |=>
            $stable(bridge_source_channel)
            && $stable(bridge_source_value)
            && $stable(bridge_effective_weight));
    ap_commit_header_stable: assert property (
        commit_valid && !commit_ready |=> commit_valid
            && $stable({commit_context, commit_tag, commit_slice,
                        commit_terminal}));
    ap_commit_payload_stable: assert property (
        commit_valid && !commit_ready |=> $stable(commit_accumulator));

    cp_independent_bank_backpressure: cover property (
        mem_req_valid[0] && mem_req_ready[0]
        && mem_req_valid[1] && !mem_req_ready[1]);
    cp_bank_response_reorder: cover property (
        mem_rsp_accept[7] ##[1:32] mem_rsp_accept[0]);
    cp_bridge_positive: cover property (
        bridge_accept && bridge_source_value[0] == 2'sd1);
    cp_bridge_negative: cover property (
        bridge_accept && bridge_source_value[0] == -2'sd1);
    cp_bridge_stall: cover property (bridge_valid && !bridge_ready);
    cp_commit_stall: cover property (commit_valid && !commit_ready);
    cp_terminal: cover property (commit_accept && commit_terminal);
    cp_cache_eviction: cover property (debug_cache_eviction_count > 0);
    cp_weight_bundle: cover property (debug_weight_bundle_beat_count > 0);
    cp_stale_attack: cover property (stale_response_seen && protocol_error);
    cp_reset_recovery_minimum_one_cycle: cover property (disable iff (1'b0)
        protocol_error ##[1:8] rst_core[*1:8] ##1 !rst_core
        ##[1:300000] (commit_accept && commit_terminal && !protocol_error));
endmodule

`default_nettype wire
