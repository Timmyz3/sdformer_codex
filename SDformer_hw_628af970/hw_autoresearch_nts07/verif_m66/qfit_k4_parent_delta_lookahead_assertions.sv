`timescale 1ns/1ps
`default_nettype none

// Seam-specific properties not present in the inherited M54 assertion set.
module qfit_k4_parent_delta_lookahead_assertions (
    input logic        clk_core,
    input logic        rst_core,
    input logic        launch_valid,
    input logic        launch_ready,
    input logic        launch_accept,
    input logic [2:0]  launch_context_count,
    input logic [15:0] launch_contexts,
    input logic        launch_legal,
    input logic        launch_zero,
    input logic        final_response_success,
    input logic        command_accept,
    input logic [3:0]  command_accept_context,
    input logic        request_accept,
    input logic        output_accept,
    input logic [2:0]  complete_push_count,
    input logic [4:0]  response_metadata_occupancy,
    input logic [4:0]  complete_occupancy,
    input logic        group_active,
    input logic [2:0]  active_count_state,
    input logic [15:0] active_contexts_state,
    input logic [15:0] context_allocated_vector,
    input logic [15:0] context_launched_vector,
    input logic        protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    logic [15:0] next_context_mask;
    logic [15:0] old_context_mask;
    always_comb begin
        next_context_mask = '0;
        old_context_mask = '0;
        for (int slot = 0; slot < 4; slot++) begin
            if (slot < launch_context_count)
                next_context_mask[launch_contexts[slot*4 +: 4]] = 1'b1;
            if (slot < active_count_state)
                old_context_mask[active_contexts_state[slot*4 +: 4]] = 1'b1;
        end
    end

    let legal_seam = launch_accept && launch_legal && final_response_success;

    ap_seam_requires_nonzero_next: assert property (
        legal_seam |-> !launch_zero && launch_context_count inside {[1:4]});
    ap_zero_next_never_uses_seam: assert property (
        final_response_success && launch_valid && launch_legal && launch_zero
        |-> !launch_ready && !launch_accept);
    ap_legal_seam_has_disjoint_old_new_contexts: assert property (
        legal_seam |-> (next_context_mask & old_context_mask) == 0);
    ap_legal_seam_has_no_old_group_request: assert property (
        legal_seam |-> !request_accept);
    ap_final_response_is_last_metadata_entry: assert property (
        final_response_success |-> response_metadata_occupancy == 1
            && !request_accept);
    ap_final_response_drains_metadata: assert property (
        final_response_success |=> response_metadata_occupancy == 0);
    ap_seam_pushes_exact_old_group_count: assert property (
        legal_seam |-> complete_push_count == active_count_state);
    ap_seam_completion_credit_bound: assert property (
        legal_seam |-> ({1'b0, complete_occupancy}
            + active_count_state - output_accept) <= 16);
    ap_seam_keeps_new_group_active: assert property (
        legal_seam |=> protocol_error || group_active);
    ap_seam_loads_new_group_state: assert property (
        legal_seam |=> protocol_error
            || (active_count_state == $past(launch_context_count)
                && active_contexts_state == $past(launch_contexts)));
    ap_seam_frees_old_and_launches_new_contexts: assert property (
        legal_seam |=> protocol_error
            || (((context_allocated_vector & $past(old_context_mask)) == 0)
                && ((context_allocated_vector & $past(next_context_mask))
                    == $past(next_context_mask))
                && ((context_launched_vector & $past(next_context_mask))
                    == $past(next_context_mask))));
    ap_seam_parallel_command_remains_allocated: assert property (
        legal_seam && command_accept |=> protocol_error
            || context_allocated_vector[$past(command_accept_context)]);
    ap_nonseam_final_clears_active: assert property (
        final_response_success && !launch_accept |=> !group_active);

    cp_seam_k1: cover property (legal_seam && launch_context_count == 1);
    cp_seam_k2: cover property (legal_seam && launch_context_count == 2);
    cp_seam_k3: cover property (legal_seam && launch_context_count == 3);
    cp_seam_k4: cover property (legal_seam && launch_context_count == 4);
    cp_zero_next_waits: cover property (
        final_response_success && launch_valid && launch_legal && launch_zero
        && !launch_accept);
    cp_seam_with_command_accept: cover property (legal_seam && command_accept);
    cp_seam_with_output_accept: cover property (legal_seam && output_accept);
    cp_seam_with_completion_push: cover property (
        legal_seam && complete_push_count inside {[1:4]});
    cp_seam_with_command_and_output: cover property (
        legal_seam && command_accept && output_accept);

    initial begin
        $display("M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1");
    end
endmodule

`default_nettype wire
