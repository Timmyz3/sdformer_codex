`timescale 1ns/1ps
`default_nettype none

module qfit_k4_parent_delta_p8_l96_ctx16_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic command_accept,
    input logic [3:0] command_accept_context,
    input logic launch_valid,
    input logic launch_ready,
    input logic launch_accept,
    input logic [2:0] launch_context_count,
    input logic [15:0] launch_contexts,
    input logic launch_legal,
    input logic launch_zero,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [15:0] weight_request_tag,
    input logic [2:0] weight_request_context_count,
    input logic [15:0] weight_request_contexts,
    input logic [7:0] weight_request_bank_valid,
    input logic [39:0] weight_request_bank_addr,
    input logic [31:0] weight_request_context_valid,
    input logic [31:0] weight_request_context_subtract,
    input logic weight_request_last,
    input logic request_accept,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic [15:0] weight_response_tag,
    input logic [2:0] weight_response_context_count,
    input logic [15:0] weight_response_contexts,
    input logic [7:0] weight_response_bank_valid,
    input logic response_accept,
    input logic response_contract_valid,
    input logic response_acc_overflow,
    input logic output_valid,
    input logic output_ready,
    input logic [47:0] output_tag,
    input logic [8:0] output_source_count,
    input logic [1823:0] output_acc,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic [4:0] context_occupancy,
    input logic [4:0] response_metadata_occupancy,
    input logic [4:0] complete_occupancy,
    input logic group_active,
    input logic [2:0] complete_push_count,
    input logic final_response_success,
    input logic zero_launch_success,
    input logic [15:0] context_allocated_vector,
    input logic [15:0] context_launched_vector,
    input logic [3:0] meta_head,
    input logic [3:0] meta_tail,
    input logic [3:0] complete_head,
    input logic [3:0] complete_tail
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

    ap_context_bound: assert property (context_occupancy <= 16);
    ap_metadata_bound: assert property (response_metadata_occupancy <= 16);
    ap_complete_bound: assert property (complete_occupancy <= 16);
    ap_context_popcount: assert property (
        context_occupancy == $countones(context_allocated_vector));
    ap_launched_subset: assert property (
        (context_launched_vector & ~context_allocated_vector) == 0);
    ap_command_allocates_free: assert property (
        command_accept |-> !context_allocated_vector[command_accept_context]);

    ap_request_has_union_source: assert property (
        weight_request_valid |-> |weight_request_bank_valid);
    ap_request_count_legal: assert property (
        weight_request_valid |-> weight_request_context_count inside {[1:4]});
    ap_request_contexts_distinct: assert property (
        weight_request_valid |->
        (weight_request_context_count < 2
         || weight_request_contexts[3:0] != weight_request_contexts[7:4])
        && (weight_request_context_count < 3
         || (weight_request_contexts[3:0] != weight_request_contexts[11:8]
          && weight_request_contexts[7:4] != weight_request_contexts[11:8]))
        && (weight_request_context_count < 4
         || (weight_request_contexts[3:0] != weight_request_contexts[15:12]
          && weight_request_contexts[7:4] != weight_request_contexts[15:12]
          && weight_request_contexts[11:8] != weight_request_contexts[15:12])));
    generate
        for (genvar slot = 0; slot < 4; slot++) begin : g_slot_subset
            ap_valid_subset: assert property (
                weight_request_valid |->
                (weight_request_context_valid[slot*8 +: 8]
                 & ~weight_request_bank_valid) == 0);
            ap_subtract_subset: assert property (
                weight_request_valid |->
                (weight_request_context_subtract[slot*8 +: 8]
                 & ~weight_request_context_valid[slot*8 +: 8]) == 0);
            ap_inactive_slot_zero: assert property (
                weight_request_valid && slot >= weight_request_context_count
                |-> weight_request_context_valid[slot*8 +: 8] == 0
                    && weight_request_context_subtract[slot*8 +: 8] == 0);
        end
    endgenerate

    ap_request_stable_under_stall: assert property (
        weight_request_valid && !weight_request_ready && !protocol_error
        |=> protocol_error || (weight_request_valid &&
            $stable({weight_request_tag, weight_request_context_count,
                     weight_request_contexts, weight_request_bank_valid,
                     weight_request_bank_addr, weight_request_context_valid,
                     weight_request_context_subtract, weight_request_last})));
    ap_output_stable_under_stall: assert property (
        output_valid && !output_ready && !protocol_error
        |=> protocol_error || (output_valid &&
            $stable({output_tag, output_source_count, output_acc})));
    ap_response_requires_metadata: assert property (
        response_accept |-> response_metadata_occupancy != 0);
    ap_final_reserves_exact_k: assert property (
        final_response_success |-> complete_push_count inside {[1:4]});
    ap_zero_reserves_exact_k: assert property (
        zero_launch_success |-> complete_push_count == launch_context_count);
    ap_push_only_final_or_zero: assert property (
        complete_push_count != 0 |-> final_response_success || zero_launch_success);

    ap_meta_enqueue_only_moves_tail: assert property (
        request_accept |=> meta_tail == $past(meta_tail) + 1'b1);
    ap_meta_no_enqueue_holds_tail: assert property (
        !request_accept |=> meta_tail == $past(meta_tail));
    ap_meta_dequeue_only_moves_head: assert property (
        response_accept |=> meta_head == $past(meta_head) + 1'b1);
    ap_meta_no_dequeue_holds_head: assert property (
        !response_accept |=> meta_head == $past(meta_head));
    ap_meta_enq_only_count: assert property (
        request_accept && !response_accept
        |=> response_metadata_occupancy ==
            $past(response_metadata_occupancy) + 1'b1);
    ap_meta_deq_only_count: assert property (
        !request_accept && response_accept
        |=> response_metadata_occupancy ==
            $past(response_metadata_occupancy) - 1'b1);
    ap_meta_equal_count: assert property (
        request_accept == response_accept
        |=> response_metadata_occupancy ==
            $past(response_metadata_occupancy));

    ap_complete_push_moves_tail: assert property (
        complete_push_count != 0
        |=> complete_tail == $past(complete_tail)
            + $past(complete_push_count));
    ap_complete_no_push_holds_tail: assert property (
        complete_push_count == 0
        |=> complete_tail == $past(complete_tail));
    ap_complete_pop_moves_head: assert property (
        output_accept |=> complete_head == $past(complete_head) + 1'b1);
    ap_complete_no_pop_holds_head: assert property (
        !output_accept |=> complete_head == $past(complete_head));
    ap_complete_count_conservation: assert property (
        1'b1 |=> complete_occupancy ==
            $past(complete_occupancy) + $past(complete_push_count)
            - ($past(output_accept) ? 1 : 0));

    ap_fault_sticky: assert property (protocol_error |=> protocol_error);
    ap_fault_closes_interfaces: assert property (
        protocol_error |-> !command_ready && !launch_ready
            && !weight_request_valid && !weight_response_ready && !output_valid);
    ap_unexpected_response_faults: assert property (
        weight_response_valid && response_metadata_occupancy == 0
        |=> protocol_error);
    ap_bad_accepted_response_faults: assert property (
        response_accept && (!response_contract_valid || response_acc_overflow)
        |=> protocol_error);
    ap_bad_launch_faults: assert property (
        launch_accept && !launch_legal |=> protocol_error);

    cp_context16: cover property (context_occupancy == 16);
    cp_meta16: cover property (response_metadata_occupancy == 16);
    cp_complete16: cover property (complete_occupancy == 16);
    cp_push4: cover property (complete_push_count == 4);
    cp_complete13_pop_push4: cover property (
        complete_occupancy == 13 && output_accept && complete_push_count == 4);
    cp_meta_tail_wrap: cover property (
        request_accept && meta_tail == 4'hf ##1 meta_tail == 4'h0);
    cp_complete_tail_wrap: cover property (
        complete_push_count != 0 && complete_tail >
            (4'hf - complete_push_count) ##1 complete_tail < $past(complete_tail));
    cp_k1: cover property (
        request_accept && weight_request_context_count == 1);
    cp_k2: cover property (
        request_accept && weight_request_context_count == 2);
    cp_k2_full_share: cover property (
        request_accept && weight_request_context_count == 2
        && weight_request_context_valid[7:0] == weight_request_bank_valid
        && weight_request_context_valid[15:8] == weight_request_bank_valid);
    cp_k2_partial_share: cover property (
        request_accept && weight_request_context_count == 2
        && |(weight_request_context_valid[7:0]
             & weight_request_context_valid[15:8])
        && weight_request_context_valid[7:0]
            != weight_request_context_valid[15:8]);
    cp_k2_no_share: cover property (
        request_accept && weight_request_context_count == 2
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[15:8]) == 0);
    cp_k3: cover property (
        request_accept && weight_request_context_count == 3);
    cp_k3_full_share: cover property (
        request_accept && weight_request_context_count == 3
        && weight_request_context_valid[7:0] == weight_request_bank_valid
        && weight_request_context_valid[15:8] == weight_request_bank_valid
        && weight_request_context_valid[23:16] == weight_request_bank_valid);
    cp_k3_partial_share: cover property (
        request_accept && weight_request_context_count == 3
        && |(weight_request_context_valid[7:0]
             & weight_request_context_valid[15:8])
        && (weight_request_context_valid[7:0]
            != weight_request_context_valid[15:8]
         || weight_request_context_valid[15:8]
            != weight_request_context_valid[23:16]));
    cp_k3_no_share: cover property (
        request_accept && weight_request_context_count == 3
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[15:8]) == 0
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[23:16]) == 0
        && (weight_request_context_valid[15:8]
            & weight_request_context_valid[23:16]) == 0);
    cp_k4: cover property (
        request_accept && weight_request_context_count == 4);
    cp_k4_full_share: cover property (
        request_accept && weight_request_context_count == 4
        && weight_request_context_valid[7:0] == weight_request_bank_valid
        && weight_request_context_valid[15:8] == weight_request_bank_valid
        && weight_request_context_valid[23:16] == weight_request_bank_valid
        && weight_request_context_valid[31:24] == weight_request_bank_valid);
    cp_k4_partial_share: cover property (
        request_accept && weight_request_context_count == 4
        && |(weight_request_context_valid[7:0]
             & weight_request_context_valid[15:8])
        && weight_request_context_valid[7:0] !=
           weight_request_context_valid[15:8]);
    cp_k4_no_share: cover property (
        request_accept && weight_request_context_count == 4
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[15:8]) == 0
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[23:16]) == 0
        && (weight_request_context_valid[7:0]
            & weight_request_context_valid[31:24]) == 0
        && (weight_request_context_valid[15:8]
            & weight_request_context_valid[23:16]) == 0
        && (weight_request_context_valid[15:8]
            & weight_request_context_valid[31:24]) == 0
        && (weight_request_context_valid[23:16]
            & weight_request_context_valid[31:24]) == 0);
    cp_request_stall: cover property (
        weight_request_valid && !weight_request_ready);
    cp_response_stall: cover property (
        weight_response_valid && !weight_response_ready);
    cp_output_stall: cover property (output_valid && !output_ready);
    cp_zero_k1: cover property (zero_launch_success && launch_context_count == 1);
    cp_zero_k2: cover property (zero_launch_success && launch_context_count == 2);
    cp_zero_k3: cover property (zero_launch_success && launch_context_count == 3);
    cp_zero_k4: cover property (zero_launch_success && launch_context_count == 4);
    cp_unexpected_response: cover property (
        weight_response_valid && response_metadata_occupancy == 0
        ##1 protocol_error);
    cp_duplicate_context_launch: cover property (
        launch_accept && !launch_legal ##1 protocol_error);
    cp_response_mismatch: cover property (
        response_accept && !response_contract_valid ##1 protocol_error);
    cp_overflow: cover property (
        response_accept && response_acc_overflow ##1 protocol_error);
    cp_fault: cover property (protocol_error);

    initial begin
        $display("M54_ASSERTION_MODULE_ACTIVE=1");
    end
endmodule

`default_nettype wire
