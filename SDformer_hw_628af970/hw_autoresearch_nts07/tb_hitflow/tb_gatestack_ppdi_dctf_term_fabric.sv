`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */
module tb_gatestack_ppdi_dctf_term_fabric #(
    parameter int Q = 4
);
    localparam int TAG_W = 12;
    localparam int SEQ_W = 10;
    localparam int ISSUE_W = 7;
    localparam int INPUT_CH_W = 8;
    localparam int SUPERTILE_W = 5;
    localparam int GATE_W = 9;
    localparam int LANE_W = 5;
    localparam int TOKEN_W = 6;
    localparam int COUNTER_W = 32;
    localparam int OCC_W = (Q < 2) ? 1 : $clog2(Q + 1);
    localparam int TARGET = 180;
    localparam int DEPTH = 256;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic cmd_valid;
    logic cmd_ready;
    logic [TAG_W-1:0] cmd_group_tag;
    logic [SEQ_W-1:0] cmd_sequence;
    logic [ISSUE_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel;
    logic [SUPERTILE_W-1:0] cmd_logical_supertile;
    logic [GATE_W-1:0] cmd_gate_code;
    logic [LANE_W-1:0] cmd_lane_id;
    logic [1:0] cmd_destination_valid;
    logic [(2*TOKEN_W)-1:0] cmd_destination_tokens;
    logic [2:0] bank_valid;
    logic [2:0] bank_ready;
    logic [(3*TAG_W)-1:0] bank_group_tags;
    logic [(3*SEQ_W)-1:0] bank_sequences;
    logic [(3*ISSUE_W)-1:0] bank_term_issue_seqs;
    logic [2:0] bank_term_first;
    logic [2:0] bank_term_last;
    logic [2:0] bank_head_last;
    logic [(3*INPUT_CH_W)-1:0] bank_input_channels;
    logic [(3*SUPERTILE_W)-1:0] bank_logical_supertiles;
    logic [(3*GATE_W)-1:0] bank_gate_codes;
    logic [(3*LANE_W)-1:0] bank_lane_ids;
    logic [5:0] bank_destination_valid;
    logic [(6*TOKEN_W)-1:0] bank_destination_tokens;
    logic retire_valid;
    logic [TAG_W-1:0] retire_group_tag;
    logic [SEQ_W-1:0] retire_sequence;
    logic [ISSUE_W-1:0] retire_term_issue_seq;
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

    logic [TAG_W-1:0] exp_tag [0:2][0:DEPTH-1];
    logic [SEQ_W-1:0] exp_seq [0:2][0:DEPTH-1];
    logic [ISSUE_W-1:0] exp_issue [0:2][0:DEPTH-1];
    logic exp_first [0:2][0:DEPTH-1];
    logic exp_last [0:2][0:DEPTH-1];
    logic exp_head_last [0:2][0:DEPTH-1];
    logic [INPUT_CH_W-1:0] exp_channel [0:2][0:DEPTH-1];
    logic [SUPERTILE_W-1:0] exp_supertile [0:2][0:DEPTH-1];
    logic [GATE_W-1:0] exp_gate [0:2][0:DEPTH-1];
    logic [LANE_W-1:0] exp_lane [0:2][0:DEPTH-1];
    logic [1:0] exp_dest_valid [0:2][0:DEPTH-1];
    logic [(2*TOKEN_W)-1:0] exp_dest [0:2][0:DEPTH-1];
    integer exp_head [0:2];
    integer exp_tail [0:2];
    integer cycles;
    integer accepted;
    integer consumed [0:2];
    integer flushes;
    integer paired;
    integer split_ready;
    integer next_seq;
    logic [31:0] lfsr_q;

    gatestack_ppdi_dctf_term_fabric #(
        .Q(Q), .GROUP_TAG_W(TAG_W), .SEQUENCE_W(SEQ_W),
        .TERM_ISSUE_SEQ_W(ISSUE_W), .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(SUPERTILE_W), .GATE_CODE_W(GATE_W),
        .LANE_ID_W(LANE_W), .DEST_TOKEN_W(TOKEN_W),
        .COUNTER_W(COUNTER_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    always @(negedge clk_core) begin
        if (rst_core) begin
            flush = 1'b0;
            cmd_valid = 1'b0;
            bank_ready = '0;
        end else begin
            flush = (cycles == 53);
            bank_ready[0] = (cycles % 5) != 0;
            bank_ready[1] = lfsr_q[0] | lfsr_q[4];
            bank_ready[2] = ((cycles < 25) || (cycles > 42)) &&
                            (lfsr_q[1] | lfsr_q[7]);
            if (accepted >= TARGET)
                bank_ready = 3'b111;
            if (flush) begin
                cmd_valid = 1'b0;
            end else if (!(cmd_valid && !cmd_ready)) begin
                cmd_valid = (accepted < TARGET) && ((cycles % 9) != 0);
                cmd_group_tag = TAG_W'((next_seq < 1000) ?
                                      12'h123 : 12'h456);
                cmd_sequence = SEQ_W'(next_seq);
                cmd_term_issue_seq = ISSUE_W'(next_seq / 5);
                cmd_term_first = (next_seq % 5) == 0;
                cmd_term_last = (next_seq % 5) == 4;
                cmd_head_last = cmd_term_last && ((next_seq % 15) == 14);
                cmd_input_channel = INPUT_CH_W'((next_seq * 7) % 251);
                cmd_logical_supertile =
                    SUPERTILE_W'((next_seq * 3) % 29);
                cmd_gate_code = GATE_W'((next_seq * 11) % 257);
                cmd_lane_id = LANE_W'(next_seq % 32);
                case (next_seq % 3)
                    0: cmd_destination_valid = 2'b11;
                    1: cmd_destination_valid = 2'b01;
                    default: cmd_destination_valid = 2'b10;
                endcase
                cmd_destination_tokens[0 +: TOKEN_W] =
                    TOKEN_W'((next_seq * 4) % 64);
                cmd_destination_tokens[TOKEN_W +: TOKEN_W] =
                    TOKEN_W'(((next_seq * 6) % 64) | 1);
            end
        end
    end

    always @(posedge clk_core) begin : p_scoreboard
        integer bank;
        integer slot;
        if (rst_core) begin
            cycles = 0;
            accepted = 0;
            consumed[0] = 0;
            consumed[1] = 0;
            consumed[2] = 0;
            flushes = 0;
            paired = 0;
            split_ready = 0;
            next_seq = 0;
            lfsr_q = 32'hc001_d00d;
            for (bank = 0; bank < 3; bank = bank + 1) begin
                exp_head[bank] = 0;
                exp_tail[bank] = 0;
            end
        end else begin
            cycles = cycles + 1;
            lfsr_q = {lfsr_q[30:0],
                      lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            if (flush) begin
                flushes = flushes + 1;
                next_seq = 1000;
                for (bank = 0; bank < 3; bank = bank + 1) begin
                    exp_head[bank] = 0;
                    exp_tail[bank] = 0;
                end
            end else begin
                if ((bank_ready != 3'b000) && (bank_ready != 3'b111))
                    split_ready = split_ready + 1;
                if (cmd_valid && cmd_ready) begin
                    accepted = accepted + 1;
                    if (cmd_destination_valid == 2'b11)
                        paired = paired + 1;
                    for (bank = 0; bank < 3; bank = bank + 1) begin
                        slot = exp_tail[bank];
                        exp_tag[bank][slot] = cmd_group_tag;
                        exp_seq[bank][slot] = cmd_sequence;
                        exp_issue[bank][slot] = cmd_term_issue_seq;
                        exp_first[bank][slot] = cmd_term_first;
                        exp_last[bank][slot] = cmd_term_last;
                        exp_head_last[bank][slot] = cmd_head_last;
                        exp_channel[bank][slot] = cmd_input_channel;
                        exp_supertile[bank][slot] = cmd_logical_supertile;
                        exp_gate[bank][slot] = cmd_gate_code;
                        exp_lane[bank][slot] = cmd_lane_id;
                        exp_dest_valid[bank][slot] = cmd_destination_valid;
                        exp_dest[bank][slot] = cmd_destination_tokens;
                        exp_tail[bank] = exp_tail[bank] + 1;
                    end
                    next_seq = next_seq + 1;
                end
                for (bank = 0; bank < 3; bank = bank + 1) begin
                    if (bank_valid[bank] && bank_ready[bank]) begin
                        if (exp_head[bank] >= exp_tail[bank])
                            $fatal(1, "bank %0d consumed without expected", bank);
                        slot = exp_head[bank];
                        if (bank_group_tags[(bank*TAG_W) +: TAG_W] !==
                            exp_tag[bank][slot] ||
                            bank_sequences[(bank*SEQ_W) +: SEQ_W] !==
                            exp_seq[bank][slot] ||
                            bank_term_issue_seqs[(bank*ISSUE_W) +: ISSUE_W] !==
                            exp_issue[bank][slot] ||
                            bank_term_first[bank] !== exp_first[bank][slot] ||
                            bank_term_last[bank] !== exp_last[bank][slot] ||
                            bank_head_last[bank] !== exp_head_last[bank][slot] ||
                            bank_input_channels[(bank*INPUT_CH_W) +: INPUT_CH_W] !==
                            exp_channel[bank][slot] ||
                            bank_logical_supertiles[(bank*SUPERTILE_W) +:
                                                    SUPERTILE_W] !==
                            exp_supertile[bank][slot] ||
                            bank_gate_codes[(bank*GATE_W) +: GATE_W] !==
                            exp_gate[bank][slot] ||
                            bank_lane_ids[(bank*LANE_W) +: LANE_W] !==
                            exp_lane[bank][slot] ||
                            bank_destination_valid[(bank*2) +: 2] !==
                            exp_dest_valid[bank][slot] ||
                            bank_destination_tokens[(bank*2*TOKEN_W) +:
                                                    (2*TOKEN_W)] !==
                            exp_dest[bank][slot])
                            $fatal(1, "bank %0d payload mismatch", bank);
                        exp_head[bank] = exp_head[bank] + 1;
                        consumed[bank] = consumed[bank] + 1;
                    end
                end
            end

            if (cycles > 5000)
                $fatal(1, "timeout accepted=%0d occupancy=%0d", accepted,
                       occupancy);
            if ((accepted >= TARGET) && (occupancy == 0) &&
                (exp_head[0] == exp_tail[0]) &&
                (exp_head[1] == exp_tail[1]) &&
                (exp_head[2] == exp_tail[2])) begin
                if ((flushes != 1) || (paired == 0) || (split_ready == 0))
                    $fatal(1, "coverage missing");
                if ((count_accepted != accepted) ||
                    (count_retired > count_accepted))
                    $fatal(1, "counter mismatch accepted=%0d rtl=%0d retired=%0d",
                           accepted, count_accepted, count_retired);
                $display("PASS PPDI DCTF FABRIC Q=%0d cycles=%0d accepted=%0d paired=%0d consumed={%0d,%0d,%0d} skew=%0d",
                         Q, cycles, accepted, paired, consumed[0], consumed[1],
                         consumed[2], count_skew_cycles);
                $finish;
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        cmd_valid = 1'b0;
        bank_ready = '0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
    end
endmodule

`default_nettype wire
