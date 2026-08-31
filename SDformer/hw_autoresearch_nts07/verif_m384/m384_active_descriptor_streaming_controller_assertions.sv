`timescale 1ns/1ps
`default_nettype none

module m384_active_descriptor_streaming_controller_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core,
    input logic reset_n,
    input logic config_reload,
    input logic config_reload_accept,
    input logic phase_valid,
    input logic phase_ready,
    input logic phase_accept,
    input logic row_valid,
    input logic row_ready,
    input logic row_accept,
    input logic [11:0] row_id,
    input logic [15:0] row_original,
    input logic [6:0] row_center_id,
    input logic [4:0] row_distance,
    input logic row_use_pwp,
    input logic row_last,
    input logic descriptor_write_valid,
    input logic descriptor_write_ready,
    input logic descriptor_write_accept,
    input logic [TAG_BITS-1:0] descriptor_write_tag,
    input logic descriptor_write_bank,
    input logic [11:0] descriptor_write_address,
    input logic [47:0] descriptor_write_data,
    input logic phase_seal_valid,
    input logic phase_seal_ready,
    input logic phase_seal_accept,
    input logic [TAG_BITS-1:0] phase_seal_tag,
    input logic phase_seal_bank,
    input logic [11:0] phase_seal_active_count,
    input logic [31:0] phase_seal_used_center_bitmap,
    input logic phase_seal_empty,
    input logic pwp_run_valid,
    input logic pwp_run_ready,
    input logic pwp_run_accept,
    input logic [4:0] pwp_run_start_center,
    input logic [5:0] pwp_run_length_centers,
    input logic [15:0] pwp_run_tile0_address,
    input logic [15:0] pwp_run_tile1_address,
    input logic [15:0] pwp_run_bytes,
    input logic pwp_run_last,
    input logic tile1_prefetch_valid,
    input logic tile1_prefetch_ready,
    input logic tile1_prefetch_accept,
    input logic [TAG_BITS-1:0] tile1_prefetch_tag,
    input logic tile1_prefetch_bank,
    input logic [15:0] tile1_prefetch_weight_address,
    input logic [15:0] tile1_prefetch_pwp_base_address,
    input logic [31:0] tile1_prefetch_used_center_bitmap,
    input logic tile1_prefetch_done_valid,
    input logic tile1_prefetch_done_ready,
    input logic tile1_prefetch_done_accept,
    input logic [TAG_BITS-1:0] tile1_prefetch_done_tag,
    input logic tile1_prefetch_done_bank,
    input logic replay_start_valid,
    input logic replay_start_ready,
    input logic replay_start_accept,
    input logic replay_start_tile,
    input logic descriptor_read_req_valid,
    input logic descriptor_read_req_ready,
    input logic descriptor_read_req_accept,
    input logic [TAG_BITS-1:0] descriptor_read_req_tag,
    input logic descriptor_read_req_bank,
    input logic [11:0] descriptor_read_req_address,
    input logic descriptor_read_rsp_valid,
    input logic descriptor_read_rsp_ready,
    input logic descriptor_read_rsp_accept,
    input logic [TAG_BITS-1:0] descriptor_read_rsp_tag,
    input logic descriptor_read_rsp_bank,
    input logic [11:0] descriptor_read_rsp_address,
    input logic [47:0] descriptor_read_rsp_data,
    input logic bundle_valid,
    input logic bundle_ready,
    input logic bundle_accept,
    input logic [TAG_BITS-1:0] bundle_tag,
    input logic bundle_tile,
    input logic [11:0] bundle_row_id,
    input logic [15:0] bundle_original,
    input logic [6:0] bundle_center_id,
    input logic [15:0] bundle_center,
    input logic [4:0] bundle_distance,
    input logic bundle_use_pwp,
    input logic bundle_fallback_bit_sparse,
    input logic [15:0] bundle_plus_mask,
    input logic [15:0] bundle_minus_mask,
    input logic replay_done_valid,
    input logic replay_done_ready,
    input logic replay_done_accept,
    input logic [TAG_BITS-1:0] replay_done_tag,
    input logic replay_done_tile,
    input logic [11:0] replay_done_count,
    input logic phase_done_valid,
    input logic phase_done_ready,
    input logic phase_done_accept,
    input logic [TAG_BITS-1:0] phase_done_tag,
    input logic [11:0] phase_done_active_count,
    input logic [31:0] phase_done_used_center_bitmap,
    input logic phase_done_empty,
    input logic protocol_error,
    input logic [3:0] debug_state,
    input logic [11:0] debug_active_count,
    input logic [3:0] debug_fifo_occupancy,
    input logic [3:0] debug_outstanding_reads,
    input logic [3:0] debug_credit_used,
    input logic [1:0] debug_replays_completed
);
`ifdef SVA_RUNTIME_ENABLED
    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer bit_index;
        logic [4:0] count;
        begin
            count = '0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                count = count + value[bit_index];
            popcount16 = count;
        end
    endfunction

    ap_phase_accept: assert property (@(posedge clk_core) disable iff(!reset_n)
        phase_accept == (phase_valid && phase_ready));
    ap_row_accept: assert property (@(posedge clk_core) disable iff(!reset_n)
        row_accept == (row_valid && row_ready));
    ap_write_accept: assert property (@(posedge clk_core) disable iff(!reset_n)
        descriptor_write_accept ==
            (descriptor_write_valid && descriptor_write_ready));
    ap_seal_accept: assert property (@(posedge clk_core) disable iff(!reset_n)
        phase_seal_accept == (phase_seal_valid && phase_seal_ready));
    ap_replay_start_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        replay_start_accept == (replay_start_valid && replay_start_ready));
    ap_pwp_run_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_accept == (pwp_run_valid && pwp_run_ready));
    ap_tile1_prefetch_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        tile1_prefetch_accept ==
            (tile1_prefetch_valid && tile1_prefetch_ready));
    ap_tile1_prefetch_done_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        tile1_prefetch_done_accept ==
            (tile1_prefetch_done_valid && tile1_prefetch_done_ready));
    ap_tile0_start_prefetch_atomic: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        replay_start_accept && !replay_start_tile
        |-> tile1_prefetch_accept);
    ap_read_req_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_req_accept ==
            (descriptor_read_req_valid && descriptor_read_req_ready));
    ap_read_rsp_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_rsp_accept ==
            (descriptor_read_rsp_valid && descriptor_read_rsp_ready));
    ap_bundle_accept: assert property (@(posedge clk_core) disable iff(!reset_n)
        bundle_accept == (bundle_valid && bundle_ready));
    ap_replay_done_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        replay_done_accept == (replay_done_valid && replay_done_ready));
    ap_phase_done_accept: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        phase_done_accept == (phase_done_valid && phase_done_ready));

    ap_fault_sticky: assert property (@(posedge clk_core) disable iff(!reset_n)
        $past(protocol_error) |-> protocol_error);
    ap_fault_fail_closed: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        protocol_error |-> !(phase_accept || row_accept
            || descriptor_write_accept || phase_seal_accept
            || pwp_run_accept || tile1_prefetch_accept
            || tile1_prefetch_done_accept || replay_start_accept
            || descriptor_read_req_accept
            || descriptor_read_rsp_accept || bundle_accept
            || replay_done_accept || phase_done_accept
            || config_reload_accept));

    ap_zero_row_no_write: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        row_accept && row_original == 0 |-> !descriptor_write_accept);
    ap_active_row_atomic_write: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        row_accept && row_original != 0 |-> descriptor_write_accept);
    ap_write_has_active_row: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_write_accept |-> row_accept && row_original != 0);
    ap_packed_descriptor: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_write_valid |-> descriptor_write_data ==
            {7'b0,row_use_pwp,row_distance,row_center_id,
             row_original,row_id});
    ap_reserved_zero: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_write_valid |-> descriptor_write_data[47:41] == 0);
    ap_pop1_fallback_retained: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        row_accept && popcount16(row_original) == 1
        |-> descriptor_write_accept && !row_use_pwp);

    ap_write_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_write_valid && !descriptor_write_ready
        |=> protocol_error || (descriptor_write_valid &&
            $stable({descriptor_write_tag,descriptor_write_bank,
                     descriptor_write_address,descriptor_write_data})));
    ap_seal_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        phase_seal_valid && !phase_seal_ready
        |=> protocol_error || (phase_seal_valid &&
            $stable({phase_seal_tag,phase_seal_bank,
                     phase_seal_active_count,
                     phase_seal_used_center_bitmap,phase_seal_empty})));
    ap_pwp_run_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid && !pwp_run_ready
        |=> protocol_error || (pwp_run_valid &&
            $stable({pwp_run_start_center,pwp_run_length_centers,
                     pwp_run_tile0_address,pwp_run_tile1_address,
                     pwp_run_bytes,pwp_run_last})));
    ap_pwp_run_nonzero: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid |-> pwp_run_length_centers inside {[1:32]});
    ap_pwp_run_tile0_direct_address: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid |-> pwp_run_tile0_address ==
            16'd6240 + pwp_run_start_center * 16'd640);
    ap_pwp_run_tile1_direct_address: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid |-> pwp_run_tile1_address ==
            16'd38912 + pwp_run_start_center * 16'd640);
    ap_pwp_run_bytes: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid |-> pwp_run_bytes ==
            pwp_run_length_centers * 16'd640);
    ap_pwp_run_bounds: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_valid |->
            pwp_run_start_center + pwp_run_length_centers <= 32
            && pwp_run_tile0_address + pwp_run_bytes <= 16'd26720
            && pwp_run_tile1_address + pwp_run_bytes <= 16'd59392);
    ap_tile1_prefetch_layout: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        tile1_prefetch_valid |->
            tile1_prefetch_weight_address == 16'd32768
            && tile1_prefetch_pwp_base_address == 16'd38912);
    ap_tile1_prefetch_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        tile1_prefetch_valid && !tile1_prefetch_ready
        |=> protocol_error || (tile1_prefetch_valid &&
            $stable({tile1_prefetch_tag,tile1_prefetch_bank,
                     tile1_prefetch_weight_address,
                     tile1_prefetch_pwp_base_address,
                     tile1_prefetch_used_center_bitmap})));
    ap_read_req_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_req_valid && !descriptor_read_req_ready
        |=> protocol_error || (descriptor_read_req_valid &&
            $stable({descriptor_read_req_tag,descriptor_read_req_bank,
                     descriptor_read_req_address})));
    ap_read_rsp_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_rsp_valid && !descriptor_read_rsp_ready
        |=> protocol_error || (descriptor_read_rsp_valid &&
            $stable({descriptor_read_rsp_tag,descriptor_read_rsp_bank,
                     descriptor_read_rsp_address,
                     descriptor_read_rsp_data})));
    ap_bundle_stable: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid && !bundle_ready
        |=> protocol_error || (bundle_valid &&
            $stable({bundle_tag,bundle_tile,bundle_row_id,bundle_original,
                     bundle_center_id,bundle_center,bundle_distance,
                     bundle_use_pwp,bundle_fallback_bit_sparse,
                     bundle_plus_mask,bundle_minus_mask})));

    ap_credit_bound: assert property (@(posedge clk_core) disable iff(!reset_n)
        debug_fifo_occupancy <= 8 && debug_outstanding_reads <= 8
        && debug_credit_used <= 8
        && debug_credit_used == debug_fifo_occupancy
            + debug_outstanding_reads);
    ap_request_address_bound: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_req_valid
        |-> descriptor_read_req_address < debug_active_count);
    ap_bundle_flag_partition: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid |-> bundle_use_pwp ^ bundle_fallback_bit_sparse);
    ap_bundle_distance: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid |-> bundle_distance ==
            popcount16(bundle_original ^ bundle_center));
    ap_bundle_use_rule: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid && bundle_use_pwp |->
            ({1'b0,bundle_distance}+6'd1) <
            {1'b0,popcount16(bundle_original)});
    ap_bundle_fallback_rule: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid && bundle_fallback_bit_sparse |->
            ({1'b0,bundle_distance}+6'd1) >=
            {1'b0,popcount16(bundle_original)});
    ap_bundle_signed_residual: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid && bundle_use_pwp |->
            bundle_plus_mask == (bundle_original & ~bundle_center)
            && bundle_minus_mask == (bundle_center & ~bundle_original)
            && (bundle_plus_mask & bundle_minus_mask) == 0);
    ap_bundle_fallback_payload: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        bundle_valid && bundle_fallback_bit_sparse |->
            bundle_plus_mask == bundle_original && bundle_minus_mask == 0);
    ap_replay_done_count: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        replay_done_valid |-> replay_done_count == debug_active_count);
    ap_phase_done_replay_extent: assert property (
        @(posedge clk_core) disable iff(!reset_n)
        phase_done_valid && !phase_done_empty
        |-> debug_replays_completed == 2);

    cp_reload: cover property (@(posedge clk_core) disable iff(!reset_n)
        config_reload_accept);
    cp_zero_phase: cover property (@(posedge clk_core) disable iff(!reset_n)
        phase_seal_accept && phase_seal_empty);
    cp_active_one: cover property (@(posedge clk_core) disable iff(!reset_n)
        phase_seal_accept && phase_seal_active_count == 1);
    cp_active_2400: cover property (@(posedge clk_core) disable iff(!reset_n)
        phase_seal_accept && phase_seal_active_count == 2400);
    cp_active_3000: cover property (@(posedge clk_core) disable iff(!reset_n)
        phase_seal_accept && phase_seal_active_count == 3000);
    cp_pop1_fallback: cover property (@(posedge clk_core) disable iff(!reset_n)
        bundle_accept && popcount16(bundle_original) == 1
        && bundle_fallback_bit_sparse);
    cp_mixed_residual: cover property (@(posedge clk_core) disable iff(!reset_n)
        bundle_accept && bundle_use_pwp && bundle_plus_mask != 0
        && bundle_minus_mask != 0);
    cp_single_pwp_run: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_accept && pwp_run_length_centers == 1);
    cp_full_pwp_run: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_accept && pwp_run_start_center == 0
        && pwp_run_length_centers == 32 && pwp_run_last);
    cp_multi_pwp_run: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        pwp_run_accept && !pwp_run_last ##[1:12]
        pwp_run_accept && pwp_run_last);
    cp_tile1_prefetch_overlap_start: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        replay_start_accept && !replay_start_tile
        && tile1_prefetch_accept);
    cp_tile1_prefetch_done: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        tile1_prefetch_done_accept);
    cp_fifo_full: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_fifo_occupancy == 8);
    cp_outstanding_full: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        debug_outstanding_reads == 8);
    cp_simultaneous_push_pop: cover property (
        @(posedge clk_core) disable iff(!reset_n)
        descriptor_read_rsp_accept && bundle_accept);
    cp_tile0_done: cover property (@(posedge clk_core) disable iff(!reset_n)
        replay_done_accept && !replay_done_tile);
    cp_tile1_done: cover property (@(posedge clk_core) disable iff(!reset_n)
        replay_done_accept && replay_done_tile);
    cp_protocol_attack: cover property (@(posedge clk_core) disable iff(!reset_n)
        protocol_error && debug_state == 15);
`endif
endmodule

`default_nettype wire
