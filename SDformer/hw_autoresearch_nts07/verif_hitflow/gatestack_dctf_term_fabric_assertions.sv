`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf_term_fabric_assertions #(
    parameter int Q = 4,
    parameter int GROUP_TAG_W = 16,
    parameter int SEQUENCE_W = 16,
    parameter int TERM_ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int GATE_CODE_W = 2,
    parameter int LANE_ID_W = 7,
    parameter int DEST_TOKEN_W = 8,
    parameter int COUNTER_W = 32
) (
    input logic                        clk_core,
    input logic                        rst_core,
    input logic                        flush,
    input logic                        cmd_valid,
    input logic                        cmd_ready,
    input logic [SEQUENCE_W-1:0]       cmd_sequence,
    input logic                        cmd_term_last,
    input logic                        cmd_head_last,
    input logic [2:0]                  bank_valid,
    input logic [2:0]                  bank_ready,
    input logic [(3*GROUP_TAG_W)-1:0]  bank_group_tags,
    input logic [(3*SEQUENCE_W)-1:0]   bank_sequences,
    input logic [(3*TERM_ISSUE_SEQ_W)-1:0] bank_term_issue_seqs,
    input logic [2:0]                  bank_term_first,
    input logic [2:0]                  bank_term_last,
    input logic [2:0]                  bank_head_last,
    input logic [(3*INPUT_CH_W)-1:0]   bank_input_channels,
    input logic [(3*LOGICAL_SUPERTILE_W)-1:0] bank_logical_supertiles,
    input logic [(3*GATE_CODE_W)-1:0]  bank_gate_codes,
    input logic [(3*LANE_ID_W)-1:0]    bank_lane_ids,
    input logic [(3*DEST_TOKEN_W)-1:0] bank_destination_tokens,
    input logic                        retire_valid,
    input logic [SEQUENCE_W-1:0]       retire_sequence,
    input logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0] occupancy,
    input logic [COUNTER_W-1:0]        count_accepted,
    input logic [(3*COUNTER_W)-1:0]    count_bank_consumed,
    input logic [COUNTER_W-1:0]        count_retired
);
    localparam int OCC_W = (Q < 2) ? 1 : $clog2(Q + 1);
    logic [COUNTER_W-1:0] epoch_accepted_base_q;
    logic [COUNTER_W-1:0] epoch_retired_base_q;
    logic accepted_seen_q;
    logic [SEQUENCE_W-1:0] last_accepted_sequence_q;
    logic retire_seen_q;
    logic [SEQUENCE_W-1:0] last_retire_sequence_q;
    logic [2:0] bank_seen_q;
    logic [SEQUENCE_W-1:0] last_bank_sequence_q [0:2];

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            epoch_accepted_base_q <= '0;
            epoch_retired_base_q <= '0;
            accepted_seen_q <= 1'b0;
            last_accepted_sequence_q <= '0;
            retire_seen_q <= 1'b0;
            last_retire_sequence_q <= '0;
            bank_seen_q <= '0;
            for (int bank = 0; bank < 3; bank = bank + 1)
                last_bank_sequence_q[bank] <= '0;
        end else if (flush) begin
            epoch_accepted_base_q <= count_accepted;
            epoch_retired_base_q <= count_retired;
            accepted_seen_q <= 1'b0;
            last_accepted_sequence_q <= '0;
            retire_seen_q <= 1'b0;
            last_retire_sequence_q <= '0;
            bank_seen_q <= '0;
            for (int bank = 0; bank < 3; bank = bank + 1)
                last_bank_sequence_q[bank] <= '0;
        end else begin
            if (cmd_valid && cmd_ready) begin
                accepted_seen_q <= 1'b1;
                last_accepted_sequence_q <= cmd_sequence;
            end
            if (retire_valid) begin
                retire_seen_q <= 1'b1;
                last_retire_sequence_q <= retire_sequence;
            end
            for (int bank = 0; bank < 3; bank = bank + 1) begin
                if (bank_valid[bank] && bank_ready[bank]) begin
                    bank_seen_q[bank] <= 1'b1;
                    last_bank_sequence_q[bank] <=
                        bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W];
                end
            end
        end
    end

    property p_flush_blocks_outputs;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> (bank_valid == '0) && !retire_valid;
    endproperty

    property p_flush_clears_queue;
        @(posedge clk_core) disable iff (rst_core)
        flush |=> (occupancy == '0) && (bank_valid == '0) && !retire_valid;
    endproperty

    property p_occupancy_in_range;
        @(posedge clk_core) disable iff (rst_core)
        occupancy <= OCC_W'(Q);
    endproperty

    property p_retired_never_exceeds_accepted;
        @(posedge clk_core) disable iff (rst_core)
        count_retired <= count_accepted;
    endproperty

    property p_epoch_count_conservation;
        @(posedge clk_core) disable iff (rst_core || flush)
        (count_accepted - epoch_accepted_base_q) ==
        (count_retired - epoch_retired_base_q) + COUNTER_W'(occupancy);
    endproperty

    property p_retire_sequence_order;
        @(posedge clk_core) disable iff (rst_core || flush)
        retire_valid && retire_seen_q |->
            retire_sequence == (last_retire_sequence_q + 1'b1);
    endproperty

    property p_accepted_sequence_order;
        @(posedge clk_core) disable iff (rst_core || flush)
        cmd_valid && cmd_ready && accepted_seen_q |->
            cmd_sequence == (last_accepted_sequence_q + 1'b1);
    endproperty

    property p_input_head_last_is_term_last;
        @(posedge clk_core) disable iff (rst_core)
        cmd_valid && cmd_head_last |-> cmd_term_last;
    endproperty

    assert property (p_flush_blocks_outputs);
    assert property (p_flush_clears_queue);
    assert property (p_occupancy_in_range);
    assert property (p_retired_never_exceeds_accepted);
    assert property (p_epoch_count_conservation);
    assert property (p_retire_sequence_order);
    assert property (p_accepted_sequence_order);
    assert property (p_input_head_last_is_term_last);

    generate
        for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
            property p_bank_output_stable_under_backpressure;
                @(posedge clk_core) disable iff (rst_core)
                bank_valid[bank] && !bank_ready[bank] && !flush |=>
                    flush ||
                    (bank_valid[bank] &&
                     $stable({bank_group_tags[
                                  (bank*GROUP_TAG_W) +: GROUP_TAG_W],
                              bank_sequences[
                                  (bank*SEQUENCE_W) +: SEQUENCE_W],
                              bank_term_issue_seqs[
                                  (bank*TERM_ISSUE_SEQ_W) +:
                                  TERM_ISSUE_SEQ_W],
                              bank_term_first[bank],
                              bank_term_last[bank],
                              bank_head_last[bank],
                              bank_input_channels[
                                  (bank*INPUT_CH_W) +: INPUT_CH_W],
                              bank_logical_supertiles[
                                  (bank*LOGICAL_SUPERTILE_W) +:
                                  LOGICAL_SUPERTILE_W],
                              bank_gate_codes[
                                  (bank*GATE_CODE_W) +: GATE_CODE_W],
                              bank_lane_ids[
                                  (bank*LANE_ID_W) +: LANE_ID_W],
                              bank_destination_tokens[
                                  (bank*DEST_TOKEN_W) +: DEST_TOKEN_W]}));
            endproperty

            property p_bank_consumed_never_exceeds_accepted;
                @(posedge clk_core) disable iff (rst_core)
                count_bank_consumed[(bank*COUNTER_W) +: COUNTER_W] <=
                    count_accepted;
            endproperty

            property p_bank_sequence_order;
                @(posedge clk_core) disable iff (rst_core || flush)
                bank_valid[bank] && bank_ready[bank] && bank_seen_q[bank] |->
                    bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W] ==
                    (last_bank_sequence_q[bank] + 1'b1);
            endproperty

            property p_bank_head_last_is_term_last;
                @(posedge clk_core) disable iff (rst_core)
                bank_valid[bank] && bank_head_last[bank] |->
                    bank_term_last[bank];
            endproperty

            assert property (p_bank_output_stable_under_backpressure);
            assert property (p_bank_consumed_never_exceeds_accepted);
            assert property (p_bank_sequence_order);
            assert property (p_bank_head_last_is_term_last);
        end
    endgenerate

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    cmd_valid && cmd_ready && retire_valid);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    &(bank_valid & bank_ready));
    cover property (@(posedge clk_core) disable iff (rst_core)
                    flush && (occupancy != '0));
endmodule

`default_nettype wire
