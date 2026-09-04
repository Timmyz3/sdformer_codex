`timescale 1ns/1ps
`default_nettype none

// Testbench stimulus and scoreboard state intentionally use blocking updates
// at clock edges to avoid NBA races with the following half-cycle driver.
/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_dctf_term_fabric #(
    parameter int Q = 4
);
    localparam int GROUP_TAG_W = 16;
    localparam int SEQUENCE_W = 16;
    localparam int TERM_ISSUE_SEQ_W = 13;
    localparam int INPUT_CH_W = 10;
    localparam int LOGICAL_SUPERTILE_W = 6;
    localparam int GATE_CODE_W = 2;
    localparam int LANE_ID_W = 7;
    localparam int DEST_TOKEN_W = 8;
    localparam int COUNTER_W = 32;
    localparam int OCC_W = $clog2(Q + 1);
    localparam int TARGET_ACCEPTED = 260;
    localparam int MAX_EXPECTED = 512;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic cmd_valid;
    logic cmd_ready;
    logic [GROUP_TAG_W-1:0] cmd_group_tag;
    logic [SEQUENCE_W-1:0] cmd_sequence;
    logic [TERM_ISSUE_SEQ_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel;
    logic [LOGICAL_SUPERTILE_W-1:0] cmd_logical_supertile;
    logic [GATE_CODE_W-1:0] cmd_gate_code;
    logic [LANE_ID_W-1:0] cmd_lane_id;
    logic [DEST_TOKEN_W-1:0] cmd_destination_token;
    logic [2:0] bank_valid;
    logic [2:0] bank_ready;
    logic [(3*GROUP_TAG_W)-1:0] bank_group_tags;
    logic [(3*SEQUENCE_W)-1:0] bank_sequences;
    logic [(3*TERM_ISSUE_SEQ_W)-1:0] bank_term_issue_seqs;
    logic [2:0] bank_term_first;
    logic [2:0] bank_term_last;
    logic [2:0] bank_head_last;
    logic [(3*INPUT_CH_W)-1:0] bank_input_channels;
    logic [(3*LOGICAL_SUPERTILE_W)-1:0] bank_logical_supertiles;
    logic [(3*GATE_CODE_W)-1:0] bank_gate_codes;
    logic [(3*LANE_ID_W)-1:0] bank_lane_ids;
    logic [(3*DEST_TOKEN_W)-1:0] bank_destination_tokens;
    logic retire_valid;
    logic [GROUP_TAG_W-1:0] retire_group_tag;
    logic [SEQUENCE_W-1:0] retire_sequence;
    logic [TERM_ISSUE_SEQ_W-1:0] retire_term_issue_seq;
    logic retire_term_first;
    logic retire_term_last;
    logic retire_head_last;
    logic [OCC_W-1:0] occupancy;
    logic [COUNTER_W-1:0] count_accepted;
    logic [(3*COUNTER_W)-1:0] count_bank_consumed;
    logic [COUNTER_W-1:0] count_retired;
    logic [COUNTER_W-1:0] count_input_stall;
    logic [(3*COUNTER_W)-1:0] count_bank_stall;
    logic [COUNTER_W-1:0] max_occupancy;
    logic [COUNTER_W-1:0] count_skew_cycles;

    integer cycle_count;
    integer accepted_seen;
    integer retired_seen;
    integer consumed_seen [0:2];
    integer expected_head [0:2];
    integer expected_tail [0:2];
    integer retire_head;
    integer retire_tail;
    integer input_stall_seen;
    integer bank_stall_seen [0:2];
    integer mismatch_count;
    integer next_sequence;
    integer bank2_long_stall_cycles;
    integer skew_distance;
    logic [31:0] lfsr_q;
    integer flush_seen /* verilator public_flat_rw */;
    integer random_flush_seen /* verilator public_flat_rw */;
    integer flush_inflight_seen /* verilator public_flat_rw */;
    integer full_seen /* verilator public_flat_rw */;
    integer all_three_consume_seen /* verilator public_flat_rw */;
    integer input_retire_same_cycle_seen /* verilator public_flat_rw */;
    integer fast_bank_ahead_seen /* verilator public_flat_rw */;
    logic [2:0] held_valid_q;
    logic [(3*GROUP_TAG_W)-1:0] held_group_tags_q;
    logic [(3*SEQUENCE_W)-1:0] held_sequences_q;
    logic [(3*TERM_ISSUE_SEQ_W)-1:0] held_term_issue_seqs_q;
    logic [2:0] held_term_first_q;
    logic [2:0] held_term_last_q;
    logic [2:0] held_head_last_q;
    logic [(3*INPUT_CH_W)-1:0] held_input_channels_q;
    logic [(3*LOGICAL_SUPERTILE_W)-1:0] held_logical_supertiles_q;
    logic [(3*GATE_CODE_W)-1:0] held_gate_codes_q;
    logic [(3*LANE_ID_W)-1:0] held_lane_ids_q;
    logic [(3*DEST_TOKEN_W)-1:0] held_destinations_q;

    logic [GROUP_TAG_W-1:0] exp_group_tag [0:2][0:MAX_EXPECTED-1];
    logic [SEQUENCE_W-1:0] exp_sequence [0:2][0:MAX_EXPECTED-1];
    logic [TERM_ISSUE_SEQ_W-1:0] exp_term_issue_seq
        [0:2][0:MAX_EXPECTED-1];
    logic exp_term_first [0:2][0:MAX_EXPECTED-1];
    logic exp_term_last [0:2][0:MAX_EXPECTED-1];
    logic exp_head_last [0:2][0:MAX_EXPECTED-1];
    logic [INPUT_CH_W-1:0] exp_input_channel [0:2][0:MAX_EXPECTED-1];
    logic [LOGICAL_SUPERTILE_W-1:0] exp_logical_supertile
        [0:2][0:MAX_EXPECTED-1];
    logic [GATE_CODE_W-1:0] exp_gate_code [0:2][0:MAX_EXPECTED-1];
    logic [LANE_ID_W-1:0] exp_lane_id [0:2][0:MAX_EXPECTED-1];
    logic [DEST_TOKEN_W-1:0] exp_destination [0:2][0:MAX_EXPECTED-1];
    logic [GROUP_TAG_W-1:0] exp_retire_tag [0:MAX_EXPECTED-1];
    logic [SEQUENCE_W-1:0] exp_retire_sequence [0:MAX_EXPECTED-1];
    logic [TERM_ISSUE_SEQ_W-1:0] exp_retire_term_issue_seq
        [0:MAX_EXPECTED-1];
    logic exp_retire_term_first [0:MAX_EXPECTED-1];
    logic exp_retire_term_last [0:MAX_EXPECTED-1];
    logic exp_retire_head_last [0:MAX_EXPECTED-1];

    gatestack_dctf_term_fabric #(
        .Q(Q),
        .GROUP_TAG_W(GROUP_TAG_W),
        .SEQUENCE_W(SEQUENCE_W),
        .TERM_ISSUE_SEQ_W(TERM_ISSUE_SEQ_W),
        .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(LOGICAL_SUPERTILE_W),
        .GATE_CODE_W(GATE_CODE_W),
        .LANE_ID_W(LANE_ID_W),
        .DEST_TOKEN_W(DEST_TOKEN_W),
        .COUNTER_W(COUNTER_W)
    ) dut (
        .clk_core,
        .rst_core,
        .flush,
        .cmd_valid,
        .cmd_ready,
        .cmd_group_tag,
        .cmd_sequence,
        .cmd_term_issue_seq,
        .cmd_term_first,
        .cmd_term_last,
        .cmd_head_last,
        .cmd_input_channel,
        .cmd_logical_supertile,
        .cmd_gate_code,
        .cmd_lane_id,
        .cmd_destination_token,
        .bank_valid,
        .bank_ready,
        .bank_group_tags,
        .bank_sequences,
        .bank_term_issue_seqs,
        .bank_term_first,
        .bank_term_last,
        .bank_head_last,
        .bank_input_channels,
        .bank_logical_supertiles,
        .bank_gate_codes,
        .bank_lane_ids,
        .bank_destination_tokens,
        .retire_valid,
        .retire_group_tag,
        .retire_sequence,
        .retire_term_issue_seq,
        .retire_term_first,
        .retire_term_last,
        .retire_head_last,
        .occupancy,
        .count_accepted,
        .count_bank_consumed,
        .count_retired,
        .count_input_stall,
        .count_bank_stall,
        .max_occupancy,
        .count_skew_cycles
    );

    always #5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            lfsr_q <= 32'h1ace_b00c;
        end else begin
            cycle_count <= cycle_count + 1;
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
        end
    end

    always @(negedge clk_core) begin
        if (rst_core) begin
            flush = 1'b0;
            cmd_valid = 1'b0;
            bank_ready = '0;
        end else begin
            flush = (cycle_count == 60) ||
                    ((cycle_count > 90) && (cycle_count < 260) &&
                     (random_flush_seen < 2) &&
                     ((lfsr_q[3:0] == 4'h0) ||
                      ((cycle_count == 173) &&
                       (random_flush_seen == 0))));
            if (cycle_count == 60)
                next_sequence = 1000;

            if (accepted_seen < TARGET_ACCEPTED) begin
                cmd_valid = (cycle_count % 11) != 0;
                cmd_sequence = SEQUENCE_W'(next_sequence);
                cmd_term_issue_seq =
                    TERM_ISSUE_SEQ_W'(next_sequence / 4);
                if ((next_sequence % 17) == 0) begin
                    cmd_term_first = 1'b1;
                    cmd_term_last = 1'b1;
                end else begin
                    cmd_term_first = (next_sequence % 4) == 0;
                    cmd_term_last = (next_sequence % 4) == 3;
                end
                cmd_head_last = cmd_term_last &&
                                (((next_sequence / 4) % 3) == 2);
                cmd_group_tag = GROUP_TAG_W'((next_sequence < 1000) ?
                                             16'h0110 : 16'h0220);
                cmd_input_channel = INPUT_CH_W'((next_sequence * 7) % 512);
                cmd_logical_supertile = LOGICAL_SUPERTILE_W'(
                    ((next_sequence / 2) * 11) % 37);
                cmd_gate_code = GATE_CODE_W'(next_sequence % 3);
                cmd_lane_id = LANE_ID_W'((next_sequence * 5) % 96);
                cmd_destination_token =
                    DEST_TOKEN_W'((next_sequence * 13) % 162);
            end else begin
                cmd_valid = 1'b0;
            end

            if (accepted_seen >= TARGET_ACCEPTED) begin
                bank_ready = 3'b111;
            end else if ((cycle_count % 19) == 0) begin
                bank_ready = 3'b111;
            end else begin
                bank_ready[0] = (cycle_count % 7) != 0;
                bank_ready[1] = lfsr_q[0] | lfsr_q[3];
                if ((cycle_count >= 20) && (cycle_count < 75))
                    bank_ready[2] = 1'b0;
                else
                    bank_ready[2] = lfsr_q[1] | lfsr_q[5];
            end
        end
    end

    always @(posedge clk_core) begin : p_scoreboard
        integer bank;
        integer slot;
        if (!rst_core) begin
            if (flush) begin
                flush_seen = 1;
                if (cycle_count != 60)
                    random_flush_seen = random_flush_seen + 1;
                if (occupancy != '0)
                    flush_inflight_seen = 1;
                for (bank = 0; bank < 3; bank = bank + 1)
                    expected_head[bank] = expected_tail[bank];
                retire_head = retire_tail;
                if ((bank_valid != '0) || retire_valid) begin
                    $error("flush周期仍有输出 valid");
                    mismatch_count = mismatch_count + 1;
                end
            end else begin
                if (cmd_valid && cmd_ready) begin
                    if (accepted_seen >= MAX_EXPECTED) begin
                        $fatal(1, "scoreboard容量不足");
                    end
                    for (bank = 0; bank < 3; bank = bank + 1) begin
                        slot = expected_tail[bank];
                        exp_group_tag[bank][slot] = cmd_group_tag;
                        exp_sequence[bank][slot] = cmd_sequence;
                        exp_term_issue_seq[bank][slot] =
                            cmd_term_issue_seq;
                        exp_term_first[bank][slot] = cmd_term_first;
                        exp_term_last[bank][slot] = cmd_term_last;
                        exp_head_last[bank][slot] = cmd_head_last;
                        exp_input_channel[bank][slot] = cmd_input_channel;
                        exp_logical_supertile[bank][slot] =
                            cmd_logical_supertile;
                        exp_gate_code[bank][slot] = cmd_gate_code;
                        exp_lane_id[bank][slot] = cmd_lane_id;
                        exp_destination[bank][slot] = cmd_destination_token;
                        expected_tail[bank] = expected_tail[bank] + 1;
                    end
                    exp_retire_tag[retire_tail] = cmd_group_tag;
                    exp_retire_sequence[retire_tail] = cmd_sequence;
                    exp_retire_term_issue_seq[retire_tail] =
                        cmd_term_issue_seq;
                    exp_retire_term_first[retire_tail] = cmd_term_first;
                    exp_retire_term_last[retire_tail] = cmd_term_last;
                    exp_retire_head_last[retire_tail] = cmd_head_last;
                    retire_tail = retire_tail + 1;
                    accepted_seen = accepted_seen + 1;
                    next_sequence = next_sequence + 1;
                end

                for (bank = 0; bank < 3; bank = bank + 1) begin
                    if (bank_valid[bank] && bank_ready[bank]) begin
                        if (expected_head[bank] >= expected_tail[bank]) begin
                            $error("bank%0d输出了scoreboard中不存在的命令", bank);
                            mismatch_count = mismatch_count + 1;
                        end else begin
                            slot = expected_head[bank];
                            if (bank_group_tags[(bank*GROUP_TAG_W) +:
                                                GROUP_TAG_W] !==
                                exp_group_tag[bank][slot] ||
                                bank_sequences[(bank*SEQUENCE_W) +:
                                               SEQUENCE_W] !==
                                exp_sequence[bank][slot] ||
                                bank_term_issue_seqs[
                                    (bank*TERM_ISSUE_SEQ_W) +:
                                    TERM_ISSUE_SEQ_W] !==
                                exp_term_issue_seq[bank][slot] ||
                                bank_term_first[bank] !==
                                exp_term_first[bank][slot] ||
                                bank_term_last[bank] !==
                                exp_term_last[bank][slot] ||
                                bank_head_last[bank] !==
                                exp_head_last[bank][slot] ||
                                bank_input_channels[(bank*INPUT_CH_W) +:
                                                    INPUT_CH_W] !==
                                exp_input_channel[bank][slot] ||
                                bank_logical_supertiles[
                                    (bank*LOGICAL_SUPERTILE_W) +:
                                    LOGICAL_SUPERTILE_W] !==
                                exp_logical_supertile[bank][slot] ||
                                bank_gate_codes[(bank*GATE_CODE_W) +:
                                                GATE_CODE_W] !==
                                exp_gate_code[bank][slot] ||
                                bank_lane_ids[(bank*LANE_ID_W) +:
                                              LANE_ID_W] !==
                                exp_lane_id[bank][slot] ||
                                bank_destination_tokens[
                                    (bank*DEST_TOKEN_W) +: DEST_TOKEN_W] !==
                                exp_destination[bank][slot]) begin
                                $error("bank%0d payload失配，期望sequence=%0d，实际=%0d",
                                       bank, exp_sequence[bank][slot],
                                       bank_sequences[(bank*SEQUENCE_W) +:
                                                      SEQUENCE_W]);
                                mismatch_count = mismatch_count + 1;
                            end
                            expected_head[bank] = expected_head[bank] + 1;
                        end
                        consumed_seen[bank] = consumed_seen[bank] + 1;
                    end
                    if (bank_valid[bank] && !bank_ready[bank])
                        bank_stall_seen[bank] = bank_stall_seen[bank] + 1;
                end

                if (retire_valid) begin
                    if (retire_head >= retire_tail) begin
                        $error("retire输出了scoreboard中不存在的命令");
                        mismatch_count = mismatch_count + 1;
                    end else begin
                        if ((retire_group_tag !== exp_retire_tag[retire_head]) ||
                            (retire_sequence !==
                             exp_retire_sequence[retire_head]) ||
                            (retire_term_issue_seq !==
                             exp_retire_term_issue_seq[retire_head]) ||
                            (retire_term_first !==
                             exp_retire_term_first[retire_head]) ||
                            (retire_term_last !==
                             exp_retire_term_last[retire_head]) ||
                            (retire_head_last !==
                             exp_retire_head_last[retire_head])) begin
                            $error("retire顺序失配，期望sequence=%0d，实际=%0d",
                                   exp_retire_sequence[retire_head],
                                   retire_sequence);
                            mismatch_count = mismatch_count + 1;
                        end
                        retire_head = retire_head + 1;
                    end
                    retired_seen = retired_seen + 1;
                end

                if (cmd_valid && !cmd_ready)
                    input_stall_seen = input_stall_seen + 1;
                if (occupancy == OCC_W'(Q))
                    full_seen = 1;
                if (&(bank_valid & bank_ready))
                    all_three_consume_seen = 1;
                if (cmd_valid && cmd_ready && retire_valid)
                    input_retire_same_cycle_seen = 1;
                skew_distance = consumed_seen[0] - consumed_seen[2];
                if (skew_distance > 1)
                    fast_bank_ahead_seen = 1;
                if ((cycle_count >= 20) && (cycle_count < 75) &&
                    bank_valid[2] && !bank_ready[2])
                    bank2_long_stall_cycles = bank2_long_stall_cycles + 1;
            end

            for (bank = 0; bank < 3; bank = bank + 1) begin
                if (held_valid_q[bank] && !flush) begin
                    if (!bank_valid[bank] ||
                        bank_group_tags[(bank*GROUP_TAG_W) +: GROUP_TAG_W] !==
                            held_group_tags_q[(bank*GROUP_TAG_W) +:
                                              GROUP_TAG_W] ||
                        bank_sequences[(bank*SEQUENCE_W) +: SEQUENCE_W] !==
                            held_sequences_q[(bank*SEQUENCE_W) +:
                                             SEQUENCE_W] ||
                        bank_term_issue_seqs[
                            (bank*TERM_ISSUE_SEQ_W) +:
                            TERM_ISSUE_SEQ_W] !==
                            held_term_issue_seqs_q[
                                (bank*TERM_ISSUE_SEQ_W) +:
                                TERM_ISSUE_SEQ_W] ||
                        bank_term_first[bank] !== held_term_first_q[bank] ||
                        bank_term_last[bank] !== held_term_last_q[bank] ||
                        bank_head_last[bank] !== held_head_last_q[bank] ||
                        bank_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] !==
                            held_input_channels_q[(bank*INPUT_CH_W) +:
                                                  INPUT_CH_W] ||
                        bank_logical_supertiles[
                            (bank*LOGICAL_SUPERTILE_W) +:
                            LOGICAL_SUPERTILE_W] !==
                            held_logical_supertiles_q[
                                (bank*LOGICAL_SUPERTILE_W) +:
                                LOGICAL_SUPERTILE_W] ||
                        bank_gate_codes[(bank*GATE_CODE_W) +: GATE_CODE_W] !==
                            held_gate_codes_q[(bank*GATE_CODE_W) +:
                                              GATE_CODE_W] ||
                        bank_lane_ids[(bank*LANE_ID_W) +: LANE_ID_W] !==
                            held_lane_ids_q[(bank*LANE_ID_W) +: LANE_ID_W] ||
                        bank_destination_tokens[
                            (bank*DEST_TOKEN_W) +: DEST_TOKEN_W] !==
                            held_destinations_q[
                                (bank*DEST_TOKEN_W) +: DEST_TOKEN_W]) begin
                        $error("bank%0d反压期间输出不稳定", bank);
                        mismatch_count = mismatch_count + 1;
                    end
                end
            end
            held_valid_q = bank_valid & ~bank_ready;
            held_group_tags_q = bank_group_tags;
            held_sequences_q = bank_sequences;
            held_term_issue_seqs_q = bank_term_issue_seqs;
            held_term_first_q = bank_term_first;
            held_term_last_q = bank_term_last;
            held_head_last_q = bank_head_last;
            held_input_channels_q = bank_input_channels;
            held_logical_supertiles_q = bank_logical_supertiles;
            held_gate_codes_q = bank_gate_codes;
            held_lane_ids_q = bank_lane_ids;
            held_destinations_q = bank_destination_tokens;
        end
    end

    initial begin : p_test
        integer bank;
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        cmd_valid = 1'b0;
        cmd_group_tag = '0;
        cmd_sequence = '0;
        cmd_term_issue_seq = '0;
        cmd_term_first = 1'b0;
        cmd_term_last = 1'b0;
        cmd_head_last = 1'b0;
        cmd_input_channel = '0;
        cmd_logical_supertile = '0;
        cmd_gate_code = '0;
        cmd_lane_id = '0;
        cmd_destination_token = '0;
        bank_ready = '0;
        cycle_count = 0;
        accepted_seen = 0;
        retired_seen = 0;
        retire_head = 0;
        retire_tail = 0;
        input_stall_seen = 0;
        mismatch_count = 0;
        next_sequence = 0;
        bank2_long_stall_cycles = 0;
        skew_distance = 0;
        lfsr_q = 32'h1ace_b00c;
        flush_seen = 0;
        random_flush_seen = 0;
        flush_inflight_seen = 0;
        full_seen = 0;
        all_three_consume_seen = 0;
        input_retire_same_cycle_seen = 0;
        fast_bank_ahead_seen = 0;
        held_valid_q = '0;
        held_group_tags_q = '0;
        held_sequences_q = '0;
        held_term_issue_seqs_q = '0;
        held_term_first_q = '0;
        held_term_last_q = '0;
        held_head_last_q = '0;
        held_input_channels_q = '0;
        held_logical_supertiles_q = '0;
        held_gate_codes_q = '0;
        held_lane_ids_q = '0;
        held_destinations_q = '0;
        for (bank = 0; bank < 3; bank = bank + 1) begin
            consumed_seen[bank] = 0;
            expected_head[bank] = 0;
            expected_tail[bank] = 0;
            bank_stall_seen[bank] = 0;
        end

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        while ((accepted_seen < TARGET_ACCEPTED) || (occupancy != '0)) begin
            @(posedge clk_core);
            if (cycle_count > 5000)
                $fatal(1, "DCTF回归超时");
        end
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);

        for (bank = 0; bank < 3; bank = bank + 1) begin
            if (expected_head[bank] != expected_tail[bank])
                $fatal(1, "bank%0d scoreboard未排空 head=%0d tail=%0d",
                       bank, expected_head[bank], expected_tail[bank]);
            if (count_bank_consumed[(bank*COUNTER_W) +: COUNTER_W] !==
                COUNTER_W'(consumed_seen[bank]))
                $fatal(1, "bank%0d消费计数失配 RTL=%0d TB=%0d", bank,
                       count_bank_consumed[(bank*COUNTER_W) +: COUNTER_W],
                       consumed_seen[bank]);
            if (count_bank_stall[(bank*COUNTER_W) +: COUNTER_W] !==
                COUNTER_W'(bank_stall_seen[bank]))
                $fatal(1, "bank%0d stall计数失配 RTL=%0d TB=%0d", bank,
                       count_bank_stall[(bank*COUNTER_W) +: COUNTER_W],
                       bank_stall_seen[bank]);
        end
        if (retire_head != retire_tail)
            $fatal(1, "retire scoreboard未排空 head=%0d tail=%0d",
                   retire_head, retire_tail);
        if (count_accepted !== COUNTER_W'(accepted_seen))
            $fatal(1, "accepted计数失配 RTL=%0d TB=%0d",
                   count_accepted, accepted_seen);
        if (count_retired !== COUNTER_W'(retired_seen))
            $fatal(1, "retired计数失配 RTL=%0d TB=%0d",
                   count_retired, retired_seen);
        if (count_input_stall !== COUNTER_W'(input_stall_seen))
            $fatal(1, "input stall计数失配 RTL=%0d TB=%0d",
                   count_input_stall, input_stall_seen);
        if (accepted_seen < 200 || mismatch_count != 0 || flush_seen == 0 ||
            random_flush_seen == 0 ||
            flush_inflight_seen == 0 || full_seen == 0 ||
            all_three_consume_seen == 0 ||
            input_retire_same_cycle_seen == 0 ||
            fast_bank_ahead_seen == 0 ||
            (bank2_long_stall_cycles < 20) ||
            (max_occupancy != COUNTER_W'(Q)) ||
            (count_skew_cycles == '0)) begin
            $fatal(1,
                "覆盖门槛失败 accepted=%0d mismatch=%0d flush=%0d random_flush=%0d in_flight=%0d full=%0d all3=%0d in_ret=%0d ahead=%0d bank2stall=%0d maxocc=%0d skew=%0d",
                accepted_seen, mismatch_count, flush_seen, random_flush_seen,
                flush_inflight_seen, full_seen, all_three_consume_seen,
                input_retire_same_cycle_seen, fast_bank_ahead_seen,
                bank2_long_stall_cycles, max_occupancy, count_skew_cycles);
        end

        $display("PASS DCTF q=%0d cycles=%0d accepted=%0d consumed={%0d,%0d,%0d} retired=%0d input_stall=%0d bank_stall={%0d,%0d,%0d} max_occupancy=%0d skew_cycles=%0d flush_inflight=%0d random_flush=%0d",
                 Q, cycle_count, accepted_seen,
                 consumed_seen[0], consumed_seen[1],
                 consumed_seen[2], retired_seen, input_stall_seen,
                 bank_stall_seen[0], bank_stall_seen[1],
                 bank_stall_seen[2], max_occupancy, count_skew_cycles,
                 flush_inflight_seen, random_flush_seen);
        $finish;
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */
`default_nettype wire
