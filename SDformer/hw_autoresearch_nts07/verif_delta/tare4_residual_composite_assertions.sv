`timescale 1ns/1ps
`default_nettype none

module tare4_residual_composite_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [15:0] out_tag,
    input logic out_mode_meta,
    input logic [1:0] out_kind,
    input logic [5:0] out_update_count,
    input logic [12:0] out_raw16,
    input logic [8:0] out_score_q7,
    input logic signed [12:0] selected_raw_signed,
    input logic class_out_valid,
    input logic result_slot_ready,
    input logic [31:0] class_out_mask,
    input logic [5:0] class_out_count,
    input logic in_valid,
    input logic in_ready,
    input logic [15:0] in_tag,
    input logic [31:0] in_q_anchor,
    input logic [31:0] in_k_anchor,
    input logic [31:0] in_q_target,
    input logic [31:0] in_k_target,
    input logic [9:0] in_bias_raw16,
    input logic in_mode_meta,
    input logic replay_cycle
);

    function automatic logic [8:0] rne_div16(
        input logic [12:0] raw
    );
        logic [8:0] quotient;
        logic [3:0] remainder;
        logic increment;
        quotient = raw[12:4];
        remainder = raw[3:0];
        increment =
            remainder > 4'd8 ||
            (remainder == 4'd8 && quotient[0]);
        return quotient + 9'(increment);
    endfunction

    property p_output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=> out_valid &&
            $stable({
                out_tag,
                out_mode_meta,
                out_kind,
                out_update_count,
                out_raw16,
                out_score_q7
            });
    endproperty

    property p_output_raw_legal;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_raw16 <= 13'd2560;
    endproperty

    property p_selected_raw_nonnegative;
        @(posedge clk_core) disable iff (rst_core)
        class_out_valid && result_slot_ready |->
            selected_raw_signed >= 0;
    endproperty

    property p_kind_count_contract;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |->
            (
                (out_kind == 2'd0 && out_update_count == 0) ||
                (
                    out_kind == 2'd1 &&
                    out_update_count > 0 &&
                    out_update_count <= 4
                ) ||
                (out_kind == 2'd2 && out_update_count > 4)
            );
    endproperty

    property p_classifier_mask_count_consistent;
        @(posedge clk_core) disable iff (rst_core)
        class_out_valid |->
            $countones(class_out_mask) == class_out_count;
    endproperty

    property p_bias_range_contract;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && in_ready |-> in_bias_raw16 <= 10'd512;
    endproperty

    property p_q7_matches_raw_rne;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_score_q7 == rne_div16(out_raw16);
    endproperty

    property p_replay_blocks_new_input;
        @(posedge clk_core) disable iff (rst_core)
        replay_cycle |-> !in_ready;
    endproperty

    property p_input_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        in_valid && !in_ready |=> in_valid &&
            $stable({
                in_tag,
                in_q_anchor,
                in_k_anchor,
                in_q_target,
                in_k_target,
                in_bias_raw16,
                in_mode_meta
            });
    endproperty

    assert property (p_output_stable_under_stall);
    assert property (p_output_raw_legal);
    assert property (p_selected_raw_nonnegative);
    assert property (p_kind_count_contract);
    assert property (p_classifier_mask_count_consistent);
    assert property (p_bias_range_contract);
    assert property (p_q7_matches_raw_rne);
    assert property (p_replay_blocks_new_input);
    assert property (p_input_stable_under_stall);

endmodule

`default_nettype wire
