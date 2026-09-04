`timescale 1ns/1ps
`default_nettype none

// Ordered three-consumer multicast fabric for parity-paired commands. Each
// bank consumes a command independently; the queue retires it only after all
// three banks have accepted the complete two-destination payload once.
module gatestack_ppdi_dctf_term_fabric #(
    parameter int Q = 4,
    parameter int GROUP_TAG_W = 16,
    parameter int SEQUENCE_W = 16,
    parameter int TERM_ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int LOGICAL_SUPERTILE_W = 8,
    parameter int GATE_CODE_W = 9,
    parameter int LANE_ID_W = 5,
    parameter int DEST_TOKEN_W = 8,
    parameter int COUNTER_W = 32
) (
    input  logic                                      clk_core,
    input  logic                                      rst_core,
    input  logic                                      flush,

    input  logic                                      cmd_valid,
    output logic                                      cmd_ready,
    input  logic [GROUP_TAG_W-1:0]                    cmd_group_tag,
    input  logic [SEQUENCE_W-1:0]                     cmd_sequence,
    input  logic [TERM_ISSUE_SEQ_W-1:0]               cmd_term_issue_seq,
    input  logic                                      cmd_term_first,
    input  logic                                      cmd_term_last,
    input  logic                                      cmd_head_last,
    input  logic [INPUT_CH_W-1:0]                     cmd_input_channel,
    input  logic [LOGICAL_SUPERTILE_W-1:0]            cmd_logical_supertile,
    input  logic [GATE_CODE_W-1:0]                    cmd_gate_code,
    input  logic [LANE_ID_W-1:0]                      cmd_lane_id,
    input  logic [1:0]                                cmd_destination_valid,
    input  logic [(2*DEST_TOKEN_W)-1:0]               cmd_destination_tokens,

    output logic [2:0]                                bank_valid,
    input  logic [2:0]                                bank_ready,
    output logic [(3*GROUP_TAG_W)-1:0]                bank_group_tags,
    output logic [(3*SEQUENCE_W)-1:0]                 bank_sequences,
    output logic [(3*TERM_ISSUE_SEQ_W)-1:0]           bank_term_issue_seqs,
    output logic [2:0]                                bank_term_first,
    output logic [2:0]                                bank_term_last,
    output logic [2:0]                                bank_head_last,
    output logic [(3*INPUT_CH_W)-1:0]                 bank_input_channels,
    output logic [(3*LOGICAL_SUPERTILE_W)-1:0]        bank_logical_supertiles,
    output logic [(3*GATE_CODE_W)-1:0]                bank_gate_codes,
    output logic [(3*LANE_ID_W)-1:0]                  bank_lane_ids,
    output logic [5:0]                                bank_destination_valid,
    output logic [(6*DEST_TOKEN_W)-1:0]               bank_destination_tokens,

    output logic                                      retire_valid,
    output logic [GROUP_TAG_W-1:0]                    retire_group_tag,
    output logic [SEQUENCE_W-1:0]                     retire_sequence,
    output logic [TERM_ISSUE_SEQ_W-1:0]               retire_term_issue_seq,
    output logic                                      retire_term_first,
    output logic                                      retire_term_last,
    output logic                                      retire_head_last,

    output logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0]   occupancy,
    output logic [COUNTER_W-1:0]                      count_accepted,
    output logic [(3*COUNTER_W)-1:0]                  count_bank_consumed,
    output logic [COUNTER_W-1:0]                      count_retired,
    output logic [COUNTER_W-1:0]                      count_input_stall,
    output logic [(3*COUNTER_W)-1:0]                  count_bank_stall,
    output logic [COUNTER_W-1:0]                      max_occupancy,
    output logic [COUNTER_W-1:0]                      count_skew_cycles
);
    localparam int BANKS = 3;
    localparam int PTR_W = (Q < 2) ? 1 : $clog2(Q);
    localparam int OCC_W = (Q < 2) ? 1 : $clog2(Q + 1);

    logic [GROUP_TAG_W-1:0] group_tag_mem_q [0:Q-1];
    logic [SEQUENCE_W-1:0] sequence_mem_q [0:Q-1];
    logic [TERM_ISSUE_SEQ_W-1:0] issue_mem_q [0:Q-1];
    logic term_first_mem_q [0:Q-1];
    logic term_last_mem_q [0:Q-1];
    logic head_last_mem_q [0:Q-1];
    logic [INPUT_CH_W-1:0] input_channel_mem_q [0:Q-1];
    logic [LOGICAL_SUPERTILE_W-1:0] supertile_mem_q [0:Q-1];
    logic [GATE_CODE_W-1:0] gate_mem_q [0:Q-1];
    logic [LANE_ID_W-1:0] lane_mem_q [0:Q-1];
    logic [1:0] destination_valid_mem_q [0:Q-1];
    logic [(2*DEST_TOKEN_W)-1:0] destination_mem_q [0:Q-1];
    logic [BANKS-1:0] consume_mask_q [0:Q-1];

    logic [PTR_W-1:0] head_ptr_q;
    logic [PTR_W-1:0] tail_ptr_q;
    logic [PTR_W-1:0] bank_ptr_q [0:BANKS-1];
    logic [OCC_W-1:0] occupancy_q;
    logic [OCC_W-1:0] bank_pending_q [0:BANKS-1];
    logic [COUNTER_W-1:0] bank_consumed_q [0:BANKS-1];
    logic [COUNTER_W-1:0] bank_stall_q [0:BANKS-1];
    logic [BANKS-1:0] bank_fire;
    logic [BANKS-1:0] head_consume_now;
    logic [BANKS-1:0] head_mask_after_fire;
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
        bank_destination_valid = '0;
        bank_destination_tokens = '0;
        count_bank_consumed = '0;
        count_bank_stall = '0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_valid[bank] = !flush && (bank_pending_q[bank] != '0);
            bank_group_tags[(bank*GROUP_TAG_W) +: GROUP_TAG_W] =
                group_tag_mem_q[bank_ptr_q[bank]];
            bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W] =
                sequence_mem_q[bank_ptr_q[bank]];
            bank_term_issue_seqs[(bank*TERM_ISSUE_SEQ_W) +:
                                 TERM_ISSUE_SEQ_W] =
                issue_mem_q[bank_ptr_q[bank]];
            bank_term_first[bank] = term_first_mem_q[bank_ptr_q[bank]];
            bank_term_last[bank] = term_last_mem_q[bank_ptr_q[bank]];
            bank_head_last[bank] = head_last_mem_q[bank_ptr_q[bank]];
            bank_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] =
                input_channel_mem_q[bank_ptr_q[bank]];
            bank_logical_supertiles[(bank*LOGICAL_SUPERTILE_W) +:
                                    LOGICAL_SUPERTILE_W] =
                supertile_mem_q[bank_ptr_q[bank]];
            bank_gate_codes[(bank*GATE_CODE_W) +: GATE_CODE_W] =
                gate_mem_q[bank_ptr_q[bank]];
            bank_lane_ids[(bank*LANE_ID_W) +: LANE_ID_W] =
                lane_mem_q[bank_ptr_q[bank]];
            bank_destination_valid[(bank*2) +: 2] =
                destination_valid_mem_q[bank_ptr_q[bank]];
            bank_destination_tokens[(bank*2*DEST_TOKEN_W) +:
                                    (2*DEST_TOKEN_W)] =
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
                                   issue_mem_q[head_ptr_q] : '0;
    assign retire_term_first = (occupancy_q != '0) ?
                               term_first_mem_q[head_ptr_q] : 1'b0;
    assign retire_term_last = (occupancy_q != '0) ?
                              term_last_mem_q[head_ptr_q] : 1'b0;
    assign retire_head_last = (occupancy_q != '0) ?
                              head_last_mem_q[head_ptr_q] : 1'b0;

    always_comb begin
        head_consume_now = '0;
        for (int bank = 0; bank < BANKS; bank = bank + 1) begin
            if (bank_fire[bank] && (bank_ptr_q[bank] == head_ptr_q))
                head_consume_now[bank] = 1'b1;
        end
        head_mask_after_fire = consume_mask_q[head_ptr_q] |
                               head_consume_now;
    end

    assign retire_fire = !flush && (occupancy_q != '0) &&
                         (&head_mask_after_fire);
    assign retire_valid = retire_fire;
    assign cmd_ready = !flush &&
                       ((occupancy_q < OCC_W'(Q)) || retire_fire);
    assign skew_active = (bank_pending_q[0] != bank_pending_q[1]) ||
                         (bank_pending_q[1] != bank_pending_q[2]);

    always_ff @(posedge clk_core) begin
        if (rst_core || flush) begin
            head_ptr_q <= '0;
            tail_ptr_q <= '0;
            occupancy_q <= '0;
            for (int entry = 0; entry < Q; entry = entry + 1) begin
                consume_mask_q[entry] <= '0;
                if (rst_core) begin
                    group_tag_mem_q[entry] <= '0;
                    sequence_mem_q[entry] <= '0;
                    issue_mem_q[entry] <= '0;
                    term_first_mem_q[entry] <= 1'b0;
                    term_last_mem_q[entry] <= 1'b0;
                    head_last_mem_q[entry] <= 1'b0;
                    input_channel_mem_q[entry] <= '0;
                    supertile_mem_q[entry] <= '0;
                    gate_mem_q[entry] <= '0;
                    lane_mem_q[entry] <= '0;
                    destination_valid_mem_q[entry] <= '0;
                    destination_mem_q[entry] <= '0;
                end
            end
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                bank_ptr_q[bank] <= '0;
                bank_pending_q[bank] <= '0;
            end
        end else begin
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (bank_fire[bank]) begin
                    consume_mask_q[bank_ptr_q[bank]][bank] <= 1'b1;
                    bank_ptr_q[bank] <=
                        (bank_ptr_q[bank] == PTR_W'(Q - 1)) ?
                        '0 : bank_ptr_q[bank] + 1'b1;
                    bank_pending_q[bank] <= bank_pending_q[bank] - 1'b1;
                end
            end

            if (retire_fire) begin
                consume_mask_q[head_ptr_q] <= '0;
                head_ptr_q <= (head_ptr_q == PTR_W'(Q - 1)) ?
                              '0 : head_ptr_q + 1'b1;
                occupancy_q <= occupancy_q - 1'b1;
            end

            // At full occupancy a retiring head and a new tail may share the
            // same physical slot; command assignment therefore comes last.
            if (cmd_fire) begin
                group_tag_mem_q[tail_ptr_q] <= cmd_group_tag;
                sequence_mem_q[tail_ptr_q] <= cmd_sequence;
                issue_mem_q[tail_ptr_q] <= cmd_term_issue_seq;
                term_first_mem_q[tail_ptr_q] <= cmd_term_first;
                term_last_mem_q[tail_ptr_q] <= cmd_term_last;
                head_last_mem_q[tail_ptr_q] <= cmd_head_last;
                input_channel_mem_q[tail_ptr_q] <= cmd_input_channel;
                supertile_mem_q[tail_ptr_q] <= cmd_logical_supertile;
                gate_mem_q[tail_ptr_q] <= cmd_gate_code;
                lane_mem_q[tail_ptr_q] <= cmd_lane_id;
                destination_valid_mem_q[tail_ptr_q] <=
                    cmd_destination_valid;
                destination_mem_q[tail_ptr_q] <= cmd_destination_tokens;
                consume_mask_q[tail_ptr_q] <= '0;
                tail_ptr_q <= (tail_ptr_q == PTR_W'(Q - 1)) ?
                              '0 : tail_ptr_q + 1'b1;
                occupancy_q <= occupancy_q + 1'b1;
                for (int bank = 0; bank < BANKS; bank = bank + 1)
                    bank_pending_q[bank] <= bank_pending_q[bank] + 1'b1;
            end

            if (cmd_fire && retire_fire)
                occupancy_q <= occupancy_q;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (cmd_fire && bank_fire[bank])
                    bank_pending_q[bank] <= bank_pending_q[bank];
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
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
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
            if (COUNTER_W'(occupancy_q) > max_occupancy)
                max_occupancy <= COUNTER_W'(occupancy_q);
            if (skew_active)
                count_skew_cycles <= count_skew_cycles + 1'b1;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (bank_fire[bank])
                    bank_consumed_q[bank] <= bank_consumed_q[bank] + 1'b1;
                if (bank_valid[bank] && !bank_ready[bank])
                    bank_stall_q[bank] <= bank_stall_q[bank] + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
