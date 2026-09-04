`timescale 1ns/1ps
`default_nettype none

module m488_fc2_bundle_to_8bank_adapter_assertions #(
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
    input logic [7:0] core_req_bank_valid,
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
    input logic [EPOCH_BITS-1:0] core_rsp_epoch,
    input logic [2:0] core_rsp_slot,
    input logic [GENERATION_BITS-1:0] core_rsp_generation,
    input logic [TAG_BITS-1:0] core_rsp_tag,
    input logic [7:0] core_rsp_bank_valid,
    input logic signed [7:0] core_rsp_weight [0:7][0:SLICE_LANES-1],
    input logic protocol_error, input logic [3:0] debug_live_slots,
    input logic [31:0] debug_bundle_request_count,
    input logic [31:0] debug_bank_request_count,
    input logic [31:0] debug_bank_response_count,
    input logic [31:0] debug_bundle_response_count
);
`ifdef SVA_RUNTIME_ENABLED
    logic [SLICE_LANES*64-1:0] core_rsp_weight_flat;
    always_comb begin
        for (int bank = 0; bank < 8; bank++) begin
            for (int lane = 0; lane < SLICE_LANES; lane++) begin
                core_rsp_weight_flat[(bank*SLICE_LANES+lane)*8+:8]
                    = core_rsp_weight[bank][lane];
            end
        end
    end

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

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
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);
    ap_no_accept_while_faulted:
        assert property (protocol_error |-> !core_req_accept
            && bank_req_accept == 0 && bank_rsp_accept == 0
            && !core_rsp_accept);
    ap_core_response_stable_under_stall:
        assert property (core_rsp_valid && !core_rsp_ready
            |=> core_rsp_valid && $stable({core_rsp_epoch,core_rsp_slot,
                core_rsp_generation,core_rsp_tag,core_rsp_bank_valid,
                core_rsp_weight_flat}));
    for (genvar bank = 0; bank < 8; bank++) begin : g_bank
        ap_bank_request_stable_under_stall:
            assert property (bank_req_valid[bank] && !bank_req_ready[bank]
                |=> bank_req_valid[bank]
                    && $stable({bank_req_epoch[bank],bank_req_slot[bank],
                        bank_req_generation[bank],bank_req_tag[bank],
                        bank_req_output_block[bank],bank_req_slice[bank],
                        bank_req_source_channel[bank]}));
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
    cp_retire_then_slot_reuse:
        cover property (core_rsp_accept && core_req_valid
            && !core_req_accept && core_rsp_slot == core_req_slot
            ##1 core_req_accept && core_req_slot == $past(core_rsp_slot));
    cp_protocol_attack:
        cover property (protocol_error);
`endif
endmodule

bind m488_fc2_bundle_to_8bank_adapter
    m488_fc2_bundle_to_8bank_adapter_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SLICE_LANES(SLICE_LANES)
    ) m488_sva (.*);

`default_nettype wire
