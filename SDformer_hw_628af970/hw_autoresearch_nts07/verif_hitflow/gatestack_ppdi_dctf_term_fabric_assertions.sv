`timescale 1ns/1ps
`default_nettype none

module gatestack_ppdi_dctf_term_fabric_assertions #(
    parameter int Q = 4,
    parameter int GROUP_TAG_W = 16,
    parameter int SEQUENCE_W = 16,
    parameter int TERM_ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int DEST_TOKEN_W = 8
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [GROUP_TAG_W-1:0] cmd_group_tag,
    input logic [SEQUENCE_W-1:0] cmd_sequence,
    input logic [TERM_ISSUE_SEQ_W-1:0] cmd_term_issue_seq,
    input logic cmd_term_first,
    input logic cmd_term_last,
    input logic cmd_head_last,
    input logic [INPUT_CH_W-1:0] cmd_input_channel,
    input logic [LOGICAL_SUPERTILE_W-1:0] cmd_logical_supertile,
    input logic [GATE_CODE_W-1:0] cmd_gate_code,
    input logic [LANE_ID_W-1:0] cmd_lane_id,
    input logic [1:0] cmd_destination_valid,
    input logic [(2*DEST_TOKEN_W)-1:0] cmd_destination_tokens,
    input logic [2:0] bank_valid,
    input logic [2:0] bank_ready,
    input logic [(3*GROUP_TAG_W)-1:0] bank_group_tags,
    input logic [(3*SEQUENCE_W)-1:0] bank_sequences,
    input logic [(3*TERM_ISSUE_SEQ_W)-1:0] bank_term_issue_seqs,
    input logic [2:0] bank_term_first,
    input logic [2:0] bank_term_last,
    input logic [2:0] bank_head_last,
    input logic [(3*INPUT_CH_W)-1:0] bank_input_channels,
    input logic [(3*LOGICAL_SUPERTILE_W)-1:0] bank_logical_supertiles,
    input logic [(3*GATE_CODE_W)-1:0] bank_gate_codes,
    input logic [(3*LANE_ID_W)-1:0] bank_lane_ids,
    input logic [5:0] bank_destination_valid,
    input logic [(6*DEST_TOKEN_W)-1:0] bank_destination_tokens,
    input logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0] occupancy,
    input logic retire_valid,
    input logic [2:0] head_mask_after_fire
);
    default clocking cb @(posedge clk_core); endclocking

    property p_source_holds;
        disable iff (rst_core || flush)
        cmd_valid && !cmd_ready |=> cmd_valid &&
            $stable({cmd_group_tag, cmd_sequence, cmd_term_issue_seq,
                     cmd_term_first, cmd_term_last, cmd_head_last,
                     cmd_input_channel, cmd_logical_supertile,
                     cmd_gate_code, cmd_lane_id, cmd_destination_valid,
                     cmd_destination_tokens});
    endproperty
    assert property (p_source_holds);

    property p_nonempty_destination;
        disable iff (rst_core || flush)
        cmd_valid && cmd_ready |-> (cmd_destination_valid != 2'b00);
    endproperty
    assert property (p_nonempty_destination);

    property p_even_contract;
        disable iff (rst_core || flush)
        cmd_valid && cmd_ready && cmd_destination_valid[0] |->
            !cmd_destination_tokens[0];
    endproperty
    assert property (p_even_contract);

    property p_odd_contract;
        disable iff (rst_core || flush)
        cmd_valid && cmd_ready && cmd_destination_valid[1] |->
            cmd_destination_tokens[DEST_TOKEN_W];
    endproperty
    assert property (p_odd_contract);

    property p_occupancy_bound;
        disable iff (rst_core || flush)
        occupancy <= $bits(occupancy)'(Q);
    endproperty
    assert property (p_occupancy_bound);

    property p_retire_complete;
        disable iff (rst_core || flush)
        retire_valid |-> (&head_mask_after_fire);
    endproperty
    assert property (p_retire_complete);

    generate
        for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
            property p_bank_holds;
                disable iff (rst_core || flush)
                bank_valid[bank] && !bank_ready[bank] |=>
                    bank_valid[bank] &&
                    $stable({bank_group_tags[(bank*GROUP_TAG_W) +:
                                             GROUP_TAG_W],
                             bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W],
                             bank_term_issue_seqs[
                                 (bank*TERM_ISSUE_SEQ_W) +:
                                 TERM_ISSUE_SEQ_W],
                             bank_term_first[bank], bank_term_last[bank],
                             bank_head_last[bank],
                             bank_input_channels[(bank*INPUT_CH_W) +:
                                                 INPUT_CH_W],
                             bank_logical_supertiles[
                                 (bank*LOGICAL_SUPERTILE_W) +:
                                 LOGICAL_SUPERTILE_W],
                             bank_gate_codes[(bank*GATE_CODE_W) +:
                                             GATE_CODE_W],
                             bank_lane_ids[(bank*LANE_ID_W) +: LANE_ID_W],
                             bank_destination_valid[(bank*2) +: 2],
                             bank_destination_tokens[
                                 (bank*2*DEST_TOKEN_W) +:
                                 (2*DEST_TOKEN_W)]});
            endproperty
            assert property (p_bank_holds);
        end
    endgenerate
endmodule

`default_nettype wire
