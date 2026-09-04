`timescale 1ns/1ps
`default_nettype none
module m467_conv3x3_execution_island_assertions #(
    parameter int TAG_BITS = 24
) (
    input logic clk_core, reset_n, protocol_error,
    input logic descriptor_write_valid, descriptor_write_ready,
    input logic [TAG_BITS-1:0] descriptor_write_tag,
    input logic [11:0] descriptor_write_address,
    input logic [47:0] descriptor_write_data,
    input logic descriptor_read_valid, descriptor_read_ready,
    input logic [11:0] descriptor_read_address,
    input logic payload_request_valid, payload_request_ready,
    input logic payload_request_tile, payload_request_pwp, payload_request_narrow,
    input logic [2:0] payload_request_block,
    input logic accumulator_write_valid, accumulator_write_ready,
    input logic [14:0] accumulator_write_address,
    input logic [1823:0] accumulator_write_data,
    input logic accumulator_read_valid, accumulator_read_ready,
    input logic [14:0] accumulator_read_address,
    input logic commit_valid, commit_ready,
    input logic [14:0] commit_address,
    input logic [1823:0] commit_data,
    input logic phase_done_valid, phase_done_ready,
    input logic [31:0] debug_forward_hits,
    input logic [31:0] debug_zero_initializations,
    input logic [31:0] debug_zero_commits,
    input logic [7:0] debug_zero_init_slot_mask,
    input logic debug_row_live_set_event, debug_row_live_clear_event,
    input logic debug_forward_event, debug_operator_boundary_pending
);
    ap_fault_sticky: assert property (@(posedge clk_core) disable iff(!reset_n)
        protocol_error |=> protocol_error);
    ap_dw_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        descriptor_write_valid && !descriptor_write_ready |=>
        descriptor_write_valid && $stable({descriptor_write_tag,
            descriptor_write_address,descriptor_write_data}));
    ap_dr_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        descriptor_read_valid && !descriptor_read_ready |=>
        descriptor_read_valid && $stable(descriptor_read_address));
    ap_payload_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        payload_request_valid && !payload_request_ready |=>
        payload_request_valid && $stable({payload_request_tile,
            payload_request_block,payload_request_pwp}));
    ap_commit_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        commit_valid && !commit_ready |=> commit_valid &&
        $stable({commit_address,commit_data}));
    ap_done_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        phase_done_valid && !phase_done_ready |=> phase_done_valid);
    ap_acc_read_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        accumulator_read_valid && !accumulator_read_ready |=>
        accumulator_read_valid && $stable(accumulator_read_address));
    ap_acc_write_stable: assert property (@(posedge clk_core) disable iff(!reset_n)
        accumulator_write_valid && !accumulator_write_ready |=>
        accumulator_write_valid && $stable({accumulator_write_address,accumulator_write_data}));
    ap_row_live_set_only_slot7: assert property (@(posedge clk_core) disable iff(!reset_n)
        debug_row_live_set_event |-> accumulator_write_address[14:12] == 7);
    ap_row_live_clear_only_slot7: assert property (@(posedge clk_core) disable iff(!reset_n)
        debug_row_live_clear_event |-> commit_address[14:12] == 7);
    ap_boundary_suppresses_stale: assert property (@(posedge clk_core) disable iff(!reset_n)
        debug_operator_boundary_pending |-> !accumulator_read_valid && !debug_forward_event);
    cp_pwp: cover property (@(posedge clk_core) disable iff(!reset_n)
        payload_request_valid && payload_request_ready && payload_request_pwp);
    cp_narrow_pwp: cover property (@(posedge clk_core) disable iff(!reset_n)
        payload_request_valid && payload_request_ready && payload_request_pwp &&
        payload_request_narrow);
    cp_fallback_weight: cover property (@(posedge clk_core) disable iff(!reset_n)
        payload_request_valid && payload_request_ready && !payload_request_pwp);
    cp_tile1_block3: cover property (@(posedge clk_core) disable iff(!reset_n)
        payload_request_valid && payload_request_ready &&
        payload_request_tile && payload_request_block == 3);
    cp_forward: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_forward_hits > 0);
    cp_distinct_row: cover property (@(posedge clk_core) disable iff(!reset_n)
        accumulator_write_valid && accumulator_write_ready &&
        accumulator_write_address[11:0] == 1);
    cp_zero_initialization: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_zero_initializations > 0);
    cp_zero_commit: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_zero_commits > 0);
    cp_zero_init_all_slots: cover property (@(posedge clk_core) disable iff(!reset_n)
        debug_zero_init_slot_mask == 8'hff);
    cp_acc_read_stall: cover property (@(posedge clk_core) disable iff(!reset_n)
        accumulator_read_valid && !accumulator_read_ready);
    cp_acc_write_stall: cover property (@(posedge clk_core) disable iff(!reset_n)
        accumulator_write_valid && !accumulator_write_ready);
    cp_commit_stall: cover property (@(posedge clk_core) disable iff(!reset_n)
        commit_valid && !commit_ready);
endmodule
`default_nettype wire
