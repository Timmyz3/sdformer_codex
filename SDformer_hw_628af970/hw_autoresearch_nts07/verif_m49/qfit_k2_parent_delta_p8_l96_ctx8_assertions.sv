`timescale 1ns/1ps
`default_nettype none

module qfit_k2_parent_delta_p8_l96_ctx8_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic command_accept,
    input logic launch_valid,
    input logic launch_ready,
    input logic launch_accept,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [7:0] weight_request_bank_valid,
    input logic [39:0] weight_request_bank_addr,
    input logic [2:0] weight_request_context0,
    input logic weight_request_context1_valid,
    input logic [2:0] weight_request_context1,
    input logic [7:0] weight_request_context0_valid,
    input logic [7:0] weight_request_context0_subtract,
    input logic [7:0] weight_request_context1_valid_by_bank,
    input logic [7:0] weight_request_context1_subtract,
    input logic weight_request_last,
    input logic request_accept,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic response_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [47:0] output_tag,
    input logic [8:0] output_source_count,
    input logic [1823:0] output_acc,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic [3:0] context_occupancy,
    input logic [4:0] response_metadata_occupancy,
    input logic [4:0] complete_occupancy,
    input logic group_active,
    input logic [1:0] complete_push_count,
    input logic final_response_success,
    input logic zero_launch_success
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_command_accept: assert property (
        command_accept == (command_valid && command_ready));
    ap_launch_accept: assert property (
        launch_accept == (launch_valid && launch_ready));
    ap_request_accept: assert property (
        request_accept == (weight_request_valid && weight_request_ready));
    ap_response_accept: assert property (
        response_accept == (weight_response_valid && weight_response_ready));
    ap_output_accept: assert property (
        output_accept == (output_valid && output_ready));
    ap_context_bound: assert property (context_occupancy <= 8);
    ap_metadata_bound: assert property (response_metadata_occupancy <= 16);
    ap_complete_bound: assert property (complete_occupancy <= 16);
    ap_request_has_union_source: assert property (
        weight_request_valid |-> |weight_request_bank_valid);
    ap_destination0_subset: assert property (
        weight_request_valid |->
        (weight_request_context0_valid & ~weight_request_bank_valid) == 0);
    ap_destination1_subset: assert property (
        weight_request_valid |->
        (weight_request_context1_valid_by_bank
         & ~weight_request_bank_valid) == 0);
    ap_subtract0_subset: assert property (
        weight_request_valid |->
        (weight_request_context0_subtract
         & ~weight_request_context0_valid) == 0);
    ap_subtract1_subset: assert property (
        weight_request_valid |->
        (weight_request_context1_subtract
         & ~weight_request_context1_valid_by_bank) == 0);
    ap_k2_distinct: assert property (
        weight_request_valid && weight_request_context1_valid
        |-> weight_request_context0 != weight_request_context1);
    ap_response_requires_metadata: assert property (
        response_accept |-> response_metadata_occupancy != 0);
    ap_request_stable_under_stall: assert property (
        weight_request_valid && !weight_request_ready && !protocol_error
        |=> protocol_error || (weight_request_valid &&
            $stable({weight_request_bank_valid, weight_request_bank_addr,
                     weight_request_context0,
                     weight_request_context1_valid,
                     weight_request_context1,
                     weight_request_context0_valid,
                     weight_request_context0_subtract,
                     weight_request_context1_valid_by_bank,
                     weight_request_context1_subtract,
                     weight_request_last})));
    ap_output_stable_under_stall: assert property (
        output_valid && !output_ready && !protocol_error
        |=> protocol_error || (output_valid
            && $stable({output_tag, output_source_count, output_acc})));
    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_fault_closes_all_handshakes: assert property (
        protocol_error |-> (!command_ready && !launch_ready
            && !weight_request_valid && !weight_response_ready
            && !output_valid));
    ap_busy_definition: assert property (
        busy == ((context_occupancy != 0)
              || (response_metadata_occupancy != 0)
              || (complete_occupancy != 0) || group_active));
    ap_atomic_k2_push: assert property (
        final_response_success && weight_request_context1_valid
        |-> complete_push_count == 2);
    ap_final_releases_group: assert property (
        final_response_success |=> !group_active);
    ap_context_conservation: assert property (
        !protocol_error |=> protocol_error ||
        context_occupancy == ($past(context_occupancy)
            + ($past(command_accept) ? 1 : 0)
            - $past(complete_push_count)));
    ap_metadata_conservation: assert property (
        !protocol_error |=> protocol_error ||
        response_metadata_occupancy ==
            ($past(response_metadata_occupancy)
             + ($past(request_accept) ? 1 : 0)
             - ($past(response_accept) ? 1 : 0)));
    ap_complete_conservation: assert property (
        !protocol_error |=> protocol_error ||
        complete_occupancy == ($past(complete_occupancy)
            + $past(complete_push_count)
            - ($past(output_accept) ? 1 : 0)));

    cp_context8_full: cover property (context_occupancy == 8);
    cp_metadata16_full: cover property (response_metadata_occupancy == 16);
    cp_complete16_full: cover property (complete_occupancy == 16);
    cp_metadata_full_pop_push: cover property (
        response_metadata_occupancy == 16
        && request_accept && response_accept);
    cp_atomic_k2_push: cover property (complete_push_count == 2);
    cp_complete_credit_pop_push2: cover property (
        complete_occupancy == 15 && output_accept
        && complete_push_count == 2);
    cp_k1_request: cover property (
        request_accept && !weight_request_context1_valid);
    cp_k2_shared_bank: cover property (
        request_accept && weight_request_context1_valid
        && |(weight_request_context0_valid
             & weight_request_context1_valid_by_bank));
    cp_k2_partial_share: cover property (
        request_accept && weight_request_context1_valid
        && |(weight_request_context0_valid
             & weight_request_context1_valid_by_bank)
        && |(weight_request_context0_valid
             ^ weight_request_context1_valid_by_bank));
    cp_k2_no_share_cycle: cover property (
        request_accept && weight_request_context1_valid
        && !(|(weight_request_context0_valid
              & weight_request_context1_valid_by_bank)));
    cp_request_stall: cover property (
        weight_request_valid && !weight_request_ready);
    cp_response_stall: cover property (
        weight_response_valid && !weight_response_ready);
    cp_output_stall: cover property (output_valid && !output_ready);
    cp_zero_launch: cover property (zero_launch_success);
    cp_fault: cover property (protocol_error);

    initial $display("M49_ASSERTION_MODULE_ACTIVE=1");
endmodule

bind qfit_k2_parent_delta_p8_l96_ctx8
    qfit_k2_parent_delta_p8_l96_ctx8_assertions m49_assertions (.*);

`default_nettype wire
