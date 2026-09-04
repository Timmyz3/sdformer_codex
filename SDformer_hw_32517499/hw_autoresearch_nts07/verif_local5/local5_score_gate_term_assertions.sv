`timescale 1ns/1ps
`default_nettype none

module local5_score_gate_term_assertions #(
    parameter int HEAD_DIM  = 32,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int GATE_W    = 9,
    parameter int LANE_ID_W = 5,
    parameter int MULT_W    = 3,
    parameter int CMD_SEQ_W = 16
) (
    input logic                    clk_core,
    input logic                    rst_core,
    input logic                    anchor_valid,
    input logic                    anchor_ready,
    input logic                    row_anchor_valid,
    input logic [TAG_W-1:0]        anchor_tag,
    input logic [DEST_W-1:0]       anchor_dest_id,
    input logic [HEAD_DIM-1:0]     anchor_q_bits,
    input logic [HEAD_DIM-1:0]     anchor_k_bits,
    input logic [4:0]              anchor_valid_mask,
    input logic                    cmd_valid,
    input logic                    cmd_ready,
    input logic [TAG_W-1:0]        cmd_group_tag,
    input logic [CMD_SEQ_W-1:0]    cmd_sequence,
    input logic [GATE_W-1:0]       cmd_gate_code,
    input logic [LANE_ID_W-1:0]    cmd_lane_id,
    input logic [DEST_W-1:0]       cmd_destination_token,
    input logic [MULT_W-1:0]       cmd_multiplicity,
    input logic                    cmd_term_first,
    input logic                    cmd_term_last,
    input logic                    cmd_head_last,
    input logic                    stencil_done_valid,
    input logic                    stencil_done_ready,
    input logic [TAG_W-1:0]        stencil_done_tag,
    input logic [15:0]             perf_tare_issues,
    input logic [15:0]             perf_tare_zero,
    input logic [15:0]             perf_tare_sparse,
    input logic [15:0]             perf_tare_dense
);
    property p_anchor_stable_while_stalled;
        @(posedge clk_core) disable iff (rst_core)
        anchor_valid && !anchor_ready |=>
            anchor_valid &&
            $stable({
                anchor_tag, anchor_dest_id, anchor_q_bits,
                anchor_k_bits, anchor_valid_mask
            });
    endproperty

    property p_cmd_stable_while_stalled;
        @(posedge clk_core) disable iff (rst_core)
        cmd_valid && !cmd_ready |=>
            cmd_valid &&
            $stable({
                cmd_group_tag, cmd_sequence, cmd_gate_code, cmd_lane_id,
                cmd_destination_token, cmd_multiplicity,
                cmd_term_first, cmd_term_last, cmd_head_last
            });
    endproperty

    property p_tare_classification_never_exceeds_issues;
        @(posedge clk_core) disable iff (rst_core)
        perf_tare_zero + perf_tare_sparse + perf_tare_dense <=
            perf_tare_issues;
    endproperty

    property p_done_stable_while_stalled;
        @(posedge clk_core) disable iff (rst_core)
        stencil_done_valid && !stencil_done_ready |=>
            stencil_done_valid && $stable(stencil_done_tag);
    endproperty

    property p_no_anchor_before_done_retire;
        @(posedge clk_core) disable iff (rst_core)
        stencil_done_valid |-> !anchor_ready && !row_anchor_valid;
    endproperty

    property p_tare_classification_complete_at_stencil_done;
        @(posedge clk_core) disable iff (rst_core)
        stencil_done_valid |->
            perf_tare_issues ==
                perf_tare_zero + perf_tare_sparse + perf_tare_dense;
    endproperty

    assert property (p_anchor_stable_while_stalled);
    assert property (p_cmd_stable_while_stalled);
    assert property (p_done_stable_while_stalled);
    assert property (p_no_anchor_before_done_retire);
    assert property (p_tare_classification_never_exceeds_issues);
    assert property (p_tare_classification_complete_at_stencil_done);
endmodule

bind local5_score_gate_term_top local5_score_gate_term_assertions #(
    .HEAD_DIM(HEAD_DIM),
    .TAG_W(TAG_W),
    .DEST_W(DEST_W),
    .GATE_W(GATE_W),
    .LANE_ID_W(LANE_ID_W),
    .MULT_W(MULT_W),
    .CMD_SEQ_W(CMD_SEQ_W)
) u_local5_score_gate_term_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .anchor_valid(anchor_valid),
    .anchor_ready(anchor_ready),
    .row_anchor_valid(row_anchor_valid),
    .anchor_tag(anchor_tag),
    .anchor_dest_id(anchor_dest_id),
    .anchor_q_bits(anchor_q_bits),
    .anchor_k_bits(anchor_k_bits),
    .anchor_valid_mask(anchor_valid_mask),
    .cmd_valid(cmd_valid),
    .cmd_ready(cmd_ready),
    .cmd_group_tag(cmd_group_tag),
    .cmd_sequence(cmd_sequence),
    .cmd_gate_code(cmd_gate_code),
    .cmd_lane_id(cmd_lane_id),
    .cmd_destination_token(cmd_destination_token),
    .cmd_multiplicity(cmd_multiplicity),
    .cmd_term_first(cmd_term_first),
    .cmd_term_last(cmd_term_last),
    .cmd_head_last(cmd_head_last),
    .stencil_done_valid(stencil_done_valid),
    .stencil_done_ready(stencil_done_ready),
    .stencil_done_tag(stencil_done_tag),
    .perf_tare_issues(perf_tare_issues),
    .perf_tare_zero(perf_tare_zero),
    .perf_tare_sparse(perf_tare_sparse),
    .perf_tare_dense(perf_tare_dense)
);

`default_nettype wire
