`timescale 1ns/1ps
`default_nettype none
module m476_dual_slot_parent_queue_assertions #(
    parameter int LANES = 96,
    parameter int ROW_BITS = 6
) (
    input logic clk_core, reset_n, protocol_error,
    input logic prefetch_valid, prefetch_ready,
    input logic [ROW_BITS-1:0] prefetch_parent_id,
    input logic scratch_read_enable,
    input logic [ROW_BITS-1:0] scratch_read_address,
    input logic issue_valid, issue_ready, issue_first, issue_last,
    input logic issue_parent_valid,
    input logic [ROW_BITS-1:0] issue_row_id, issue_parent_id,
    input logic [LANES*12-1:0] issue_residual_data,
    input logic [LANES*19-1:0] issue_psum_prior,
    input logic scratch_write_enable,
    input logic [ROW_BITS-1:0] scratch_write_address,
    input logic [LANES*12-1:0] scratch_write_data,
    input logic psum_write_valid, psum_write_ready,
    input logic [ROW_BITS-1:0] psum_write_address,
    input logic [LANES*19-1:0] psum_write_data,
    input logic row_complete,
    input logic [1:0] parent_queue_occupancy,
    input logic parent_queue_full,
    input logic debug_forward_event, debug_scratch_read_event,
    input logic debug_read_response_event, debug_dual_enqueue_event,
    input logic debug_overflow_block_event
);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff(!reset_n)
        protocol_error |=> protocol_error);
    ap_issue_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        issue_valid && !issue_ready |=> issue_valid && $stable({issue_row_id,
            issue_first,issue_last,issue_parent_valid,issue_parent_id,
            issue_residual_data,issue_psum_prior}));
    ap_prefetch_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        prefetch_valid && !prefetch_ready |=> prefetch_valid &&
            $stable(prefetch_parent_id));
    ap_psum_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        psum_write_valid && !psum_write_ready |=> psum_write_valid &&
            $stable({psum_write_address,psum_write_data}));
    ap_write_coupling: assert property (@(posedge clk_core) disable iff(!reset_n)
        scratch_write_enable |-> psum_write_valid && psum_write_ready &&
            row_complete && scratch_write_address == psum_write_address);
    ap_complete_coupling: assert property (@(posedge clk_core) disable iff(!reset_n)
        row_complete |-> scratch_write_enable);
    ap_forward_suppresses_read: assert property (@(posedge clk_core)
        disable iff(!reset_n) debug_forward_event |-> !scratch_read_enable);
    ap_read_event_exact: assert property (@(posedge clk_core) disable iff(!reset_n)
        debug_scratch_read_event == scratch_read_enable);
    ap_queue_bound: assert property (@(posedge clk_core) disable iff(!reset_n)
        parent_queue_occupancy <= 2);
    ap_queue_full_exact: assert property (@(posedge clk_core) disable iff(!reset_n)
        parent_queue_full == (parent_queue_occupancy == 2));
    ap_full_does_not_credit_consume: assert property (@(posedge clk_core)
        disable iff(!reset_n) parent_queue_full |-> !prefetch_ready);
    ap_parent_issue_has_head: assert property (@(posedge clk_core)
        disable iff(!reset_n) issue_valid && issue_ready &&
        issue_parent_valid |-> parent_queue_occupancy > 0);
    ap_dual_enqueue_definition: assert property (@(posedge clk_core)
        disable iff(!reset_n) debug_dual_enqueue_event |->
        debug_read_response_event && debug_forward_event);
    ap_overflow_atomic_block: assert property (@(posedge clk_core)
        disable iff(!reset_n) debug_overflow_block_event |-> !issue_ready &&
        !scratch_write_enable && !psum_write_valid && !row_complete);

    cp_forward: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_forward_event);
    cp_macro_read: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_scratch_read_event);
    cp_read_response: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_read_response_event);
    cp_dual_enqueue: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_dual_enqueue_event);
    cp_queue_full: cover property (@(posedge clk_core) disable iff(!reset_n)
        parent_queue_full);
    cp_full_consume_no_prefetch_credit: cover property (@(posedge clk_core)
        disable iff(!reset_n) parent_queue_full && issue_valid && issue_ready &&
        issue_last && issue_parent_valid && !prefetch_ready);
    cp_back_to_back_completion: cover property (@(posedge clk_core)
        disable iff(!reset_n) row_complete ##1 row_complete);
    cp_output_stall: cover property (@(posedge clk_core) disable iff(!reset_n)
        psum_write_valid && !psum_write_ready);
    cp_overflow_atomic_block: cover property (@(posedge clk_core)
        disable iff(!reset_n) debug_overflow_block_event && !issue_ready &&
        !scratch_write_enable && !psum_write_valid);
endmodule
`default_nettype wire
