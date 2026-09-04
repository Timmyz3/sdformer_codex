`timescale 1ns/1ps
`default_nettype none
module m474_fused_parent_dual_update_assertions #(
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
    input logic row_complete, debug_forward_event,
    input logic debug_scratch_read_event, debug_overflow_block_event,
    input logic [31:0] debug_forward_hits, debug_scratch_reads,
    input logic [31:0] debug_stall_cycles
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
    ap_one_ahead_id_correlated: assert property (@(posedge clk_core)
        disable iff(!reset_n) $past(scratch_read_enable) && issue_valid &&
        issue_ready && issue_first && issue_parent_valid |->
        issue_parent_id == $past(scratch_read_address));
    ap_overflow_atomic_block: assert property (@(posedge clk_core)
        disable iff(!reset_n) debug_overflow_block_event |-> !issue_ready &&
        !scratch_write_enable && !psum_write_valid && !row_complete);

    cp_forward: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_forward_hits > 0);
    cp_macro_read: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_scratch_reads > 0);
    cp_exact_parent: cover property (@(posedge clk_core) disable iff(!reset_n)
        issue_valid && issue_ready && issue_first && issue_last &&
        issue_parent_valid && issue_residual_data == '0);
    cp_partial_parent: cover property (@(posedge clk_core) disable iff(!reset_n)
        issue_valid && issue_ready && issue_parent_valid &&
        issue_residual_data != '0);
    cp_output_stall: cover property (@(posedge clk_core) disable iff(!reset_n)
        psum_write_valid && !psum_write_ready);
    cp_back_to_back_completion: cover property (@(posedge clk_core)
        disable iff(!reset_n) row_complete ##1 row_complete);
    cp_one_ahead_macro_read: cover property (@(posedge clk_core)
        disable iff(!reset_n) scratch_read_enable ##1 issue_valid &&
        issue_ready && issue_first && issue_parent_valid &&
        issue_parent_id == $past(scratch_read_address));
    cp_overflow_atomic_block: cover property (@(posedge clk_core)
        disable iff(!reset_n) debug_overflow_block_event && !issue_ready &&
        !scratch_write_enable && !psum_write_valid);
    cp_stall_counter: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_stall_cycles > 0);
endmodule
`default_nettype wire
