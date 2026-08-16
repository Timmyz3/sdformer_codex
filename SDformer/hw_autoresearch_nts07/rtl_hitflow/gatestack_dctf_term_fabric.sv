`timescale 1ns/1ps
`default_nettype none

// Distributed Command Term Fabric (DCTF): one narrow command stream is
// consumed exactly once and in order by each of three independent banks.
module gatestack_dctf_term_fabric #(
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
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         flush,

    input  logic                         cmd_valid,
    output logic                         cmd_ready,
    input  logic [GROUP_TAG_W-1:0]       cmd_group_tag,
    input  logic [SEQUENCE_W-1:0]        cmd_sequence,
    input  logic [TERM_ISSUE_SEQ_W-1:0]  cmd_term_issue_seq,
    input  logic                         cmd_term_first,
    input  logic                         cmd_term_last,
    input  logic                         cmd_head_last,
    input  logic [INPUT_CH_W-1:0]        cmd_input_channel,
    input  logic [LOGICAL_SUPERTILE_W-1:0] cmd_logical_supertile,
    input  logic [GATE_CODE_W-1:0]       cmd_gate_code,
    input  logic [LANE_ID_W-1:0]         cmd_lane_id,
    input  logic [DEST_TOKEN_W-1:0]      cmd_destination_token,

    output logic [2:0]                   bank_valid,
    input  logic [2:0]                   bank_ready,
    output logic [(3*GROUP_TAG_W)-1:0]   bank_group_tags,
    output logic [(3*SEQUENCE_W)-1:0]    bank_sequences,
    output logic [(3*TERM_ISSUE_SEQ_W)-1:0] bank_term_issue_seqs,
    output logic [2:0]                   bank_term_first,
    output logic [2:0]                   bank_term_last,
    output logic [2:0]                   bank_head_last,
    output logic [(3*INPUT_CH_W)-1:0]    bank_input_channels,
    output logic [(3*LOGICAL_SUPERTILE_W)-1:0] bank_logical_supertiles,
    output logic [(3*GATE_CODE_W)-1:0]   bank_gate_codes,
    output logic [(3*LANE_ID_W)-1:0]     bank_lane_ids,
    output logic [(3*DEST_TOKEN_W)-1:0]  bank_destination_tokens,

    output logic                         retire_valid,
    output logic [GROUP_TAG_W-1:0]       retire_group_tag,
    output logic [SEQUENCE_W-1:0]        retire_sequence,
    output logic [TERM_ISSUE_SEQ_W-1:0]  retire_term_issue_seq,
    output logic                         retire_term_first,
    output logic                         retire_term_last,
    output logic                         retire_head_last,

    output logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0] occupancy,
    output logic [COUNTER_W-1:0]         count_accepted,
    output logic [(3*COUNTER_W)-1:0]     count_bank_consumed,
    output logic [COUNTER_W-1:0]         count_retired,
    output logic [COUNTER_W-1:0]         count_input_stall,
    output logic [(3*COUNTER_W)-1:0]     count_bank_stall,
    output logic [COUNTER_W-1:0]         max_occupancy,
    output logic [COUNTER_W-1:0]         count_skew_cycles
);
    localparam BANK_COUNT = 32'd3;
    localparam BANK_LIMIT = BANK_COUNT;
    localparam QUEUE_LIMIT = Q;
    localparam int PTR_W = (Q < 2) ? 1 : $clog2(Q);
    localparam int OCC_W = (Q < 2) ? 1 : $clog2(Q + 1);

    logic [GROUP_TAG_W-1:0] group_tag_mem_q [0:Q-1];
    logic [GROUP_TAG_W-1:0] group_tag_mem_d [0:Q-1];
    logic [SEQUENCE_W-1:0] sequence_mem_q [0:Q-1];
    logic [SEQUENCE_W-1:0] sequence_mem_d [0:Q-1];
    logic [TERM_ISSUE_SEQ_W-1:0] term_issue_seq_mem_q [0:Q-1];
    logic [TERM_ISSUE_SEQ_W-1:0] term_issue_seq_mem_d [0:Q-1];
    logic term_first_mem_q [0:Q-1];
    logic term_first_mem_d [0:Q-1];
    logic term_last_mem_q [0:Q-1];
    logic term_last_mem_d [0:Q-1];
    logic head_last_mem_q [0:Q-1];
    logic head_last_mem_d [0:Q-1];
    logic [INPUT_CH_W-1:0] input_channel_mem_q [0:Q-1];
    logic [INPUT_CH_W-1:0] input_channel_mem_d [0:Q-1];
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile_mem_q [0:Q-1];
    logic [LOGICAL_SUPERTILE_W-1:0] logical_supertile_mem_d [0:Q-1];
    logic [GATE_CODE_W-1:0] gate_code_mem_q [0:Q-1];
    logic [GATE_CODE_W-1:0] gate_code_mem_d [0:Q-1];
    logic [LANE_ID_W-1:0] lane_id_mem_q [0:Q-1];
    logic [LANE_ID_W-1:0] lane_id_mem_d [0:Q-1];
    logic [DEST_TOKEN_W-1:0] destination_mem_q [0:Q-1];
    logic [DEST_TOKEN_W-1:0] destination_mem_d [0:Q-1];
    logic [BANK_COUNT-1:0] consume_mask_q [0:Q-1];
    logic [BANK_COUNT-1:0] consume_mask_d [0:Q-1];

    logic [PTR_W-1:0] head_ptr_q, head_ptr_d;
    logic [PTR_W-1:0] tail_ptr_q, tail_ptr_d;
    logic [PTR_W-1:0] bank_ptr_q [0:BANK_COUNT-1];
    logic [PTR_W-1:0] bank_ptr_d [0:BANK_COUNT-1];
    logic [OCC_W-1:0] occupancy_q, occupancy_d;
    logic [OCC_W-1:0] bank_pending_q [0:BANK_COUNT-1];
    logic [OCC_W-1:0] bank_pending_d [0:BANK_COUNT-1];
    logic [COUNTER_W-1:0] bank_consumed_q [0:BANK_COUNT-1];
    logic [COUNTER_W-1:0] bank_stall_q [0:BANK_COUNT-1];
    logic [BANK_COUNT-1:0] bank_fire;
    logic [BANK_COUNT-1:0] head_consume_now;
    logic [BANK_COUNT-1:0] head_mask_after_fire;
    logic cmd_fire;
    logic retire_fire;
    logic skew_active;

    always_comb begin
        bank_valid = '0;
        bank_group_tags = '0;
        bank_sequences = '0;
        bank_term_issue_seqs = '0;
        bank_term_first = '0;
        bank_term_last = '0;
        bank_head_last = '0;
        bank_input_channels = '0;
        bank_logical_supertiles = '0;
        bank_gate_codes = '0;
        bank_lane_ids = '0;
        bank_destination_tokens = '0;
        count_bank_consumed = '0;
        count_bank_stall = '0;
        for (int bank = 32'd0; bank < BANK_LIMIT;
             bank = bank + 32'd1) begin
            bank_valid[bank] = !flush && (bank_pending_q[bank] != '0);
            bank_group_tags[(bank*GROUP_TAG_W) +: GROUP_TAG_W] =
                group_tag_mem_q[bank_ptr_q[bank]];
            bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W] =
                sequence_mem_q[bank_ptr_q[bank]];
            bank_term_issue_seqs[
                (bank*TERM_ISSUE_SEQ_W) +: TERM_ISSUE_SEQ_W] =
                term_issue_seq_mem_q[bank_ptr_q[bank]];
            bank_term_first[bank] = term_first_mem_q[bank_ptr_q[bank]];
            bank_term_last[bank] = term_last_mem_q[bank_ptr_q[bank]];
            bank_head_last[bank] = head_last_mem_q[bank_ptr_q[bank]];
            bank_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] =
                input_channel_mem_q[bank_ptr_q[bank]];
            bank_logical_supertiles[
                (bank*LOGICAL_SUPERTILE_W) +: LOGICAL_SUPERTILE_W] =
                logical_supertile_mem_q[bank_ptr_q[bank]];
            bank_gate_codes[(bank*GATE_CODE_W) +: GATE_CODE_W] =
                gate_code_mem_q[bank_ptr_q[bank]];
            bank_lane_ids[(bank*LANE_ID_W) +: LANE_ID_W] =
                lane_id_mem_q[bank_ptr_q[bank]];
            bank_destination_tokens[
                (bank*DEST_TOKEN_W) +: DEST_TOKEN_W] =
                destination_mem_q[bank_ptr_q[bank]];
            count_bank_consumed[(bank*COUNTER_W) +: COUNTER_W] =
                bank_consumed_q[bank];
            count_bank_stall[(bank*COUNTER_W) +: COUNTER_W] =
                bank_stall_q[bank];
        end
    end

    assign bank_fire = bank_valid & bank_ready;
    assign cmd_fire = cmd_valid && cmd_ready;
    assign occupancy = occupancy_q;
    assign retire_group_tag = (occupancy_q != '0) ?
                              group_tag_mem_q[head_ptr_q] : '0;
    assign retire_sequence = (occupancy_q != '0) ?
                             sequence_mem_q[head_ptr_q] : '0;
    assign retire_term_issue_seq = (occupancy_q != '0) ?
                                   term_issue_seq_mem_q[head_ptr_q] : '0;
    assign retire_term_first = (occupancy_q != '0) ?
                               term_first_mem_q[head_ptr_q] : 1'b0;
    assign retire_term_last = (occupancy_q != '0) ?
                              term_last_mem_q[head_ptr_q] : 1'b0;
    assign retire_head_last = (occupancy_q != '0) ?
                              head_last_mem_q[head_ptr_q] : 1'b0;

    always_comb begin
        head_consume_now = '0;
        for (int bank = 32'd0; bank < BANK_LIMIT;
             bank = bank + 32'd1) begin
            if (bank_fire[bank] && (bank_ptr_q[bank] == head_ptr_q))
                head_consume_now[bank] = 1'b1;
        end
        head_mask_after_fire = consume_mask_q[head_ptr_q] | head_consume_now;
    end

    assign retire_fire = !flush && (occupancy_q != '0) &&
                         (&head_mask_after_fire);
    assign retire_valid = retire_fire;
    // A same-cycle head retirement makes one slot available even when full.
    assign cmd_ready = !flush &&
                       ((occupancy_q < OCC_W'(Q)) || retire_fire);
    assign skew_active = (bank_pending_q[0] != bank_pending_q[1]) ||
                         (bank_pending_q[1] != bank_pending_q[2]);

    always_comb begin
        head_ptr_d = head_ptr_q;
        tail_ptr_d = tail_ptr_q;
        occupancy_d = occupancy_q;
        for (int entry = 32'd0; entry < QUEUE_LIMIT;
             entry = entry + 32'd1) begin
            group_tag_mem_d[entry] = group_tag_mem_q[entry];
            sequence_mem_d[entry] = sequence_mem_q[entry];
            term_issue_seq_mem_d[entry] = term_issue_seq_mem_q[entry];
            term_first_mem_d[entry] = term_first_mem_q[entry];
            term_last_mem_d[entry] = term_last_mem_q[entry];
            head_last_mem_d[entry] = head_last_mem_q[entry];
            input_channel_mem_d[entry] = input_channel_mem_q[entry];
            logical_supertile_mem_d[entry] =
                logical_supertile_mem_q[entry];
            gate_code_mem_d[entry] = gate_code_mem_q[entry];
            lane_id_mem_d[entry] = lane_id_mem_q[entry];
            destination_mem_d[entry] = destination_mem_q[entry];
            consume_mask_d[entry] = consume_mask_q[entry];
        end
        for (int bank = 32'd0; bank < BANK_LIMIT;
             bank = bank + 32'd1) begin
            bank_ptr_d[bank] = bank_ptr_q[bank];
            bank_pending_d[bank] = bank_pending_q[bank];
            if (bank_fire[bank]) begin
                consume_mask_d[bank_ptr_q[bank]][bank] = 1'b1;
                if (bank_ptr_q[bank] == PTR_W'(Q - 1))
                    bank_ptr_d[bank] = '0;
                else
                    bank_ptr_d[bank] = bank_ptr_q[bank] + 1'b1;
                bank_pending_d[bank] = bank_pending_d[bank] - 1'b1;
            end
        end

        if (retire_fire) begin
            consume_mask_d[head_ptr_q] = '0;
            if (head_ptr_q == PTR_W'(Q - 1))
                head_ptr_d = '0;
            else
                head_ptr_d = head_ptr_q + 1'b1;
            occupancy_d = occupancy_d - 1'b1;
        end

        // This assignment is deliberately last: at full occupancy, retire and
        // accept may reuse the same physical slot in one cycle.
        if (cmd_fire) begin
            group_tag_mem_d[tail_ptr_q] = cmd_group_tag;
            sequence_mem_d[tail_ptr_q] = cmd_sequence;
            term_issue_seq_mem_d[tail_ptr_q] = cmd_term_issue_seq;
            term_first_mem_d[tail_ptr_q] = cmd_term_first;
            term_last_mem_d[tail_ptr_q] = cmd_term_last;
            head_last_mem_d[tail_ptr_q] = cmd_head_last;
            input_channel_mem_d[tail_ptr_q] = cmd_input_channel;
            logical_supertile_mem_d[tail_ptr_q] = cmd_logical_supertile;
            gate_code_mem_d[tail_ptr_q] = cmd_gate_code;
            lane_id_mem_d[tail_ptr_q] = cmd_lane_id;
            destination_mem_d[tail_ptr_q] = cmd_destination_token;
            consume_mask_d[tail_ptr_q] = '0;
            if (tail_ptr_q == PTR_W'(Q - 1))
                tail_ptr_d = '0;
            else
                tail_ptr_d = tail_ptr_q + 1'b1;
            occupancy_d = occupancy_d + 1'b1;
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1)
                bank_pending_d[bank] = bank_pending_d[bank] + 1'b1;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            head_ptr_q <= '0;
            tail_ptr_q <= '0;
            occupancy_q <= '0;
            for (int entry = 32'd0; entry < QUEUE_LIMIT;
                 entry = entry + 32'd1) begin
                group_tag_mem_q[entry] <= '0;
                sequence_mem_q[entry] <= '0;
                term_issue_seq_mem_q[entry] <= '0;
                term_first_mem_q[entry] <= 1'b0;
                term_last_mem_q[entry] <= 1'b0;
                head_last_mem_q[entry] <= 1'b0;
                input_channel_mem_q[entry] <= '0;
                logical_supertile_mem_q[entry] <= '0;
                gate_code_mem_q[entry] <= '0;
                lane_id_mem_q[entry] <= '0;
                destination_mem_q[entry] <= '0;
                consume_mask_q[entry] <= '0;
            end
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                bank_ptr_q[bank] <= '0;
                bank_pending_q[bank] <= '0;
            end
        end else if (flush) begin
            head_ptr_q <= '0;
            tail_ptr_q <= '0;
            occupancy_q <= '0;
            for (int entry = 32'd0; entry < QUEUE_LIMIT;
                 entry = entry + 32'd1)
                consume_mask_q[entry] <= '0;
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                bank_ptr_q[bank] <= '0;
                bank_pending_q[bank] <= '0;
            end
        end else begin
            head_ptr_q <= head_ptr_d;
            tail_ptr_q <= tail_ptr_d;
            occupancy_q <= occupancy_d;
            for (int entry = 32'd0; entry < QUEUE_LIMIT;
                 entry = entry + 32'd1) begin
                group_tag_mem_q[entry] <= group_tag_mem_d[entry];
                sequence_mem_q[entry] <= sequence_mem_d[entry];
                term_issue_seq_mem_q[entry] <= term_issue_seq_mem_d[entry];
                term_first_mem_q[entry] <= term_first_mem_d[entry];
                term_last_mem_q[entry] <= term_last_mem_d[entry];
                head_last_mem_q[entry] <= head_last_mem_d[entry];
                input_channel_mem_q[entry] <= input_channel_mem_d[entry];
                logical_supertile_mem_q[entry] <=
                    logical_supertile_mem_d[entry];
                gate_code_mem_q[entry] <= gate_code_mem_d[entry];
                lane_id_mem_q[entry] <= lane_id_mem_d[entry];
                destination_mem_q[entry] <= destination_mem_d[entry];
                consume_mask_q[entry] <= consume_mask_d[entry];
            end
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                bank_ptr_q[bank] <= bank_ptr_d[bank];
                bank_pending_q[bank] <= bank_pending_d[bank];
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            count_accepted <= '0;
            count_retired <= '0;
            count_input_stall <= '0;
            max_occupancy <= '0;
            count_skew_cycles <= '0;
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                bank_consumed_q[bank] <= '0;
                bank_stall_q[bank] <= '0;
            end
        end else if (!flush) begin
            if (cmd_fire)
                count_accepted <= count_accepted + 1'b1;
            if (retire_fire)
                count_retired <= count_retired + 1'b1;
            if (cmd_valid && !cmd_ready)
                count_input_stall <= count_input_stall + 1'b1;
            if (COUNTER_W'(occupancy_d) > max_occupancy)
                max_occupancy <= COUNTER_W'(occupancy_d);
            if (skew_active)
                count_skew_cycles <= count_skew_cycles + 1'b1;
            for (int bank = 32'd0; bank < BANK_LIMIT;
                 bank = bank + 32'd1) begin
                if (bank_fire[bank])
                    bank_consumed_q[bank] <=
                        bank_consumed_q[bank] + 1'b1;
                if (bank_valid[bank] && !bank_ready[bank])
                    bank_stall_q[bank] <= bank_stall_q[bank] + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
