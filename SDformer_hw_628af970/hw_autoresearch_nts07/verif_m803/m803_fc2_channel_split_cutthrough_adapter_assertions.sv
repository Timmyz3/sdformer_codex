`timescale 1ns/1ps
`default_nettype none

module m803_fc2_channel_split_cutthrough_adapter_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16
) (
    input logic clk_core, input logic rst_core,
    input logic core_req_valid, input logic core_req_ready,
    input logic core_req_accept,
    input logic [2:0] core_req_slot,
    input logic [3:0] core_req_source_count,
    input logic [7:0] core_req_bank_valid,
    input logic [CHANNEL_BITS-1:0] core_req_source_channel [0:7],
    input logic [7:0] bank_req_valid, input logic [7:0] bank_req_ready,
    input logic [7:0] bank_req_accept,
    input logic [EPOCH_BITS-1:0] bank_req_epoch [0:7],
    input logic [2:0] bank_req_slot [0:7],
    input logic [GENERATION_BITS-1:0] bank_req_generation [0:7],
    input logic [TAG_BITS-1:0] bank_req_tag [0:7],
    input logic [2:0] bank_req_output_block [0:7],
    input logic [2:0] bank_req_slice [0:7],
    input logic [CHANNEL_BITS-1:0] bank_req_source_channel [0:7],
    input logic [7:0] bank_rsp_valid, input logic [7:0] bank_rsp_ready,
    input logic [7:0] bank_rsp_accept,
    input logic core_rsp_valid, input logic core_rsp_ready,
    input logic core_rsp_accept,
    input logic complete_found,
    input logic complete_cutthrough,
    input logic [EPOCH_BITS-1:0] core_rsp_epoch,
    input logic [2:0] core_rsp_slot,
    input logic [GENERATION_BITS-1:0] core_rsp_generation,
    input logic [TAG_BITS-1:0] core_rsp_tag,
    input logic [7:0] core_rsp_bank_valid,
    input logic signed [7:0] core_rsp_weight [0:7][0:SLICE_LANES-1],
    input logic illegal_request, input logic illegal_response,
    input logic fault_q, input logic request_channel_open,
    input logic response_channel_open,
    input logic [7:0] pending_mask_q,
    input logic rsp_hold_valid_q,
    input logic protocol_error, input logic [3:0] debug_live_slots,
    input logic [31:0] debug_bundle_request_count,
    input logic [31:0] debug_bank_request_count,
    input logic [31:0] debug_bank_response_count,
    input logic [31:0] debug_bundle_response_count
);
`ifdef SVA_RUNTIME_ENABLED
    logic [SLICE_LANES*64-1:0] core_rsp_weight_flat;
    logic [3:0] request_mask_count;
    logic request_channel_mismatch;
    always_comb begin
        request_mask_count = 0;
        request_channel_mismatch = 0;
        for (int bank = 0; bank < 8; bank++) begin
            request_mask_count = request_mask_count
                + core_req_bank_valid[bank];
            if (core_req_bank_valid[bank]
                    && core_req_source_channel[bank][2:0] != bank[2:0])
                request_channel_mismatch = 1;
        end
        for (int bank = 0; bank < 8; bank++) begin
            for (int lane = 0; lane < SLICE_LANES; lane++) begin
                core_rsp_weight_flat[(bank*SLICE_LANES+lane)*8+:8]
                    = core_rsp_weight[bank][lane];
            end
        end
    end

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_reset_clears_sticky_and_ledgers:
        assert property (@(posedge clk_core) disable iff (1'b0) rst_core
            |=> !protocol_error && !fault_q && debug_live_slots == 0
                && debug_bundle_request_count == 0
                && debug_bank_request_count == 0
                && debug_bank_response_count == 0
                && debug_bundle_response_count == 0);

    ap_core_req_accept_definition:
        assert property (core_req_accept == (core_req_valid && core_req_ready));
    ap_bank_req_accept_definition:
        assert property (bank_req_accept == (bank_req_valid & bank_req_ready));
    ap_bank_rsp_accept_definition:
        assert property (bank_rsp_accept == (bank_rsp_valid & bank_rsp_ready));
    ap_core_rsp_accept_definition:
        assert property (core_rsp_accept == (core_rsp_valid && core_rsp_ready));
    ap_core_request_mask_nonzero:
        assert property (core_req_accept |-> core_req_bank_valid != 0);
    ap_core_response_mask_nonzero:
        assert property (core_rsp_valid |-> core_rsp_bank_valid != 0);
    ap_live_slot_bound:
        assert property (debug_live_slots <= 8);
    ap_bank_response_le_request:
        assert property (debug_bank_response_count
            <= debug_bank_request_count);
    ap_bundle_response_le_request:
        assert property (debug_bundle_response_count
            <= debug_bundle_request_count);
    ap_bundle_ownership_conservation:
        assert property ((debug_bundle_request_count
            - debug_bundle_response_count) == debug_live_slots);
    ap_same_cycle_reuse_conserves_ownership:
        assert property (core_rsp_accept && core_req_accept
            && core_rsp_slot == core_req_slot
            |=> debug_live_slots == $past(debug_live_slots)
                && debug_bundle_request_count
                    == $past(debug_bundle_request_count) + 1
                && debug_bundle_response_count
                    == $past(debug_bundle_response_count) + 1);
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);
    ap_no_accept_after_sticky_fault:
        assert property (fault_q |-> !core_req_accept
            && bank_req_accept == 0 && bank_rsp_accept == 0
            && !core_rsp_accept);
    ap_request_fault_has_no_request_side_effect:
        assert property (illegal_request |-> !core_req_accept
            && bank_req_accept == 0);
    ap_response_fault_closes_both_channels:
        assert property (illegal_response |-> !core_req_accept
            && bank_req_accept == 0 && bank_rsp_accept == 0
            && !core_rsp_accept);
    ap_channel_open_definition:
        assert property (response_channel_open == (!fault_q
            && !illegal_response) && request_channel_open
            == (!fault_q && !illegal_response && !illegal_request));
    ap_legal_response_not_withdrawn_by_request_fault:
        assert property (illegal_request && !illegal_response && !fault_q
            && complete_found |-> core_rsp_valid);
    ap_accepted_response_uses_response_enable:
        assert property (core_rsp_accept |-> response_channel_open);
    ap_core_response_stable_under_stall:
        assert property (core_rsp_valid && !core_rsp_ready
            |=> protocol_error || (core_rsp_valid
                && $stable({core_rsp_epoch,core_rsp_slot,
                    core_rsp_generation,core_rsp_tag,core_rsp_bank_valid,
                    core_rsp_weight_flat})));
    for (genvar bank = 0; bank < 8; bank++) begin : g_bank
        ap_bank_request_stable_under_stall:
            assert property (bank_req_valid[bank] && !bank_req_ready[bank]
                |=> protocol_error || (bank_req_valid[bank]
                    && $stable({bank_req_epoch[bank],bank_req_slot[bank],
                        bank_req_generation[bank],bank_req_tag[bank],
                        bank_req_output_block[bank],bank_req_slice[bank],
                        bank_req_source_channel[bank]})));
    end

    cp_full_eight_bank_request:
        cover property (core_req_accept && core_req_bank_valid == 8'hff);
    cp_partial_request_distribution:
        cover property (core_req_accept && bank_req_accept != 0
            && bank_req_accept != core_req_bank_valid);
    cp_pending_request_stall:
        cover property (bank_req_valid != 0 && bank_req_accept == 0);
    cp_eight_responses_same_cycle:
        cover property (bank_rsp_accept == 8'hff);
    cp_out_of_order_bundle_response:
        cover property (core_rsp_accept
            ##1 core_rsp_accept && core_rsp_slot < $past(core_rsp_slot));
    cp_core_response_stall:
        cover property (core_rsp_valid && !core_rsp_ready);
    cp_same_cycle_slot_reuse:
        cover property (core_rsp_accept && core_req_accept
            && core_rsp_slot == core_req_slot);
    cp_cutthrough_bundle_response:
        cover property (core_rsp_accept && complete_cutthrough);
    cp_protocol_attack:
        cover property (protocol_error);
    cp_legal_response_illegal_request_same_cycle:
        cover property (illegal_request && !illegal_response
            && core_rsp_accept && bank_req_accept == 0);
    cp_source_count_mismatch_attack:
        cover property (illegal_request && core_req_bank_valid != 0
            && core_req_source_count != request_mask_count);
    cp_zero_mask_attack:
        cover property (illegal_request && core_req_bank_valid == 0);
    cp_channel_bank_mismatch_attack:
        cover property (illegal_request && core_req_bank_valid != 0
            && request_channel_mismatch);
    cp_illegal_response_legal_request_same_cycle:
        cover property (illegal_response && core_req_valid
            && !illegal_request && core_req_accept == 0
            && bank_rsp_accept == 0);
    cp_pending_drain_request_attack:
        cover property (pending_mask_q != 0 && illegal_request
            && bank_req_accept == 0);
    cp_response_backpressure_then_attack:
        cover property (core_rsp_valid && !core_rsp_ready
            ##1 illegal_request && !core_rsp_accept);
    cp_held_response_request_attack_retire:
        cover property (rsp_hold_valid_q && illegal_request
            && core_rsp_accept);
    cp_cutthrough_request_attack_retire:
        cover property (complete_cutthrough && illegal_request
            && core_rsp_accept);
    cp_sticky_fault_quiescent:
        cover property (protocol_error ##1 fault_q
            && !core_req_accept && bank_req_accept == 0
            && bank_rsp_accept == 0 && !core_rsp_accept);
`endif
endmodule

bind m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter
    m803_fc2_channel_split_cutthrough_adapter_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SLICE_LANES(SLICE_LANES)
    ) m803_sva (.*);

`default_nettype wire
