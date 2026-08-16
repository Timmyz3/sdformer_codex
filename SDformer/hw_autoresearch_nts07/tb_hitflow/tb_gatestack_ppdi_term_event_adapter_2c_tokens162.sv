`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
module tb_gatestack_ppdi_term_event_adapter_2c_tokens162;
    localparam int TOKENS = 162;
    localparam int EVENT_WAYS = 4;
    localparam int TAG_W = 32;
    localparam int GATE_W = 9;
    localparam int LANE_W = 5;
    localparam int INPUT_CH_W = 10;
    localparam int SUPERTILE_W = 8;
    localparam int TOKEN_W = 8;
    localparam int ISSUE_W = 13;
    localparam int CMD_W = 16;
    localparam int COUNT_W = 3;
    localparam int PARITY_CAP = 81;
    localparam int EXPECTED_COMMANDS = 243;
    localparam int EXPECTED_DESTINATIONS = 324;
    localparam int PAYLOAD_W = TAG_W + CMD_W + GATE_W + LANE_W + 2 +
        (2 * TOKEN_W) + ISSUE_W + 3 + INPUT_CH_W + SUPERTILE_W;
    localparam logic [TAG_W-1:0] FULL_TAG = 32'h1620_0001;
    localparam logic [TAG_W-1:0] EVEN_TAG = 32'h1620_0002;
    localparam logic [TAG_W-1:0] ODD_TAG = 32'h1620_0003;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic clear_error;

    logic term_valid;
    logic term_ready;
    logic [TAG_W-1:0] term_tag;
    logic [GATE_W-1:0] term_gate_code;
    logic [LANE_W-1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [ISSUE_W-1:0] term_issue_seq;
    logic term_head_last;
    logic [INPUT_CH_W-1:0] term_input_channel_base;
    logic [SUPERTILE_W-1:0] term_logical_supertile;

    logic event_valid;
    logic event_ready;
    logic [GATE_W-1:0] event_gate_code;
    logic [LANE_W-1:0] event_lane_id;
    logic [EVENT_WAYS-1:0] event_token_valid;
    logic [(EVENT_WAYS*TOKEN_W)-1:0] event_token_ids;
    logic [COUNT_W-1:0] event_count;
    logic [ISSUE_W-1:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;

    logic cmd_valid;
    logic cmd_ready;
    logic [TAG_W-1:0] cmd_group_tag;
    logic [CMD_W-1:0] cmd_sequence;
    logic [GATE_W-1:0] cmd_gate_code;
    logic [LANE_W-1:0] cmd_lane_id;
    logic [1:0] cmd_destination_valid;
    logic [(2*TOKEN_W)-1:0] cmd_destination_tokens;
    logic [ISSUE_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel_base;
    logic [SUPERTILE_W-1:0] cmd_logical_supertile;
    logic idle;
    logic protocol_error;

    logic [TAG_W-1:0] exp_tag [0:EXPECTED_COMMANDS-1];
    logic [GATE_W-1:0] exp_gate [0:EXPECTED_COMMANDS-1];
    logic [LANE_W-1:0] exp_lane [0:EXPECTED_COMMANDS-1];
    logic [1:0] exp_mask [0:EXPECTED_COMMANDS-1];
    logic [(2*TOKEN_W)-1:0] exp_tokens [0:EXPECTED_COMMANDS-1];
    logic [ISSUE_W-1:0] exp_issue [0:EXPECTED_COMMANDS-1];
    logic exp_first [0:EXPECTED_COMMANDS-1];
    logic exp_last [0:EXPECTED_COMMANDS-1];
    logic exp_head_last [0:EXPECTED_COMMANDS-1];
    logic [INPUT_CH_W-1:0] exp_base [0:EXPECTED_COMMANDS-1];
    logic [SUPERTILE_W-1:0] exp_supertile [0:EXPECTED_COMMANDS-1];
    logic [TOKEN_W-1:0] source_tokens [0:TOKENS-1];

    integer exp_head;
    integer exp_tail;
    integer mismatch_count;
    integer observed_commands;
    integer observed_destinations;
    integer paired_commands;
    integer only_even_commands;
    integer only_odd_commands;
    integer backpressure_cycles;
    integer full_stall_cycles;
    integer stable_payload_checks;
    integer flush_count;
    integer full_compact_commands;
    integer even_compact_commands;
    integer odd_compact_commands;
    logic held_valid_q;
    logic [PAYLOAD_W-1:0] held_payload_q;
    logic reset_checked;

    gatestack_ppdi_term_event_adapter_2c #(
        .TOKENS(TOKENS),
        .EVENT_WAYS(EVENT_WAYS),
        .TAG_W(TAG_W),
        .GATE_CODE_W(GATE_W),
        .LANE_ID_W(LANE_W),
        .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(SUPERTILE_W),
        .TOKEN_ID_W(TOKEN_W),
        .ISSUE_SEQ_W(ISSUE_W),
        .CMD_SEQUENCE_W(CMD_W),
        .WAY_COUNT_W(COUNT_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic add_expected(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input logic [1:0] mask_value,
        input logic [(2*TOKEN_W)-1:0] tokens_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        begin
            if (exp_tail >= EXPECTED_COMMANDS)
                $fatal(1, "TOKENS162 expected command queue overflow");
            exp_tag[exp_tail] = tag_value;
            exp_gate[exp_tail] = gate_value;
            exp_lane[exp_tail] = lane_value;
            exp_mask[exp_tail] = mask_value;
            exp_tokens[exp_tail] = tokens_value;
            exp_issue[exp_tail] = issue_value;
            exp_first[exp_tail] = first_value;
            exp_last[exp_tail] = last_value;
            exp_head_last[exp_tail] = head_last_value;
            exp_base[exp_tail] = base_value;
            exp_supertile[exp_tail] = supertile_value;
            exp_tail = exp_tail + 1;
        end
    endtask

    task automatic queue_source_term(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input integer destination_count_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        logic [TOKEN_W-1:0] even_tokens [0:PARITY_CAP-1];
        logic [TOKEN_W-1:0] odd_tokens [0:PARITY_CAP-1];
        logic [1:0] mask_value;
        logic [(2*TOKEN_W)-1:0] tokens_value;
        integer even_count;
        integer odd_count;
        integer command_count;
        begin
            even_count = 0;
            odd_count = 0;
            for (int index = 0; index < destination_count_value;
                 index = index + 1) begin
                if (source_tokens[index][0]) begin
                    odd_tokens[odd_count] = source_tokens[index];
                    odd_count = odd_count + 1;
                end else begin
                    even_tokens[even_count] = source_tokens[index];
                    even_count = even_count + 1;
                end
            end
            command_count = (even_count >= odd_count) ?
                even_count : odd_count;
            for (int index = 0; index < command_count;
                 index = index + 1) begin
                mask_value = '0;
                tokens_value = '0;
                if (index < even_count) begin
                    mask_value[0] = 1'b1;
                    tokens_value[0 +: TOKEN_W] = even_tokens[index];
                end
                if (index < odd_count) begin
                    mask_value[1] = 1'b1;
                    tokens_value[TOKEN_W +: TOKEN_W] = odd_tokens[index];
                end
                add_expected(tag_value, gate_value, lane_value, mask_value,
                             tokens_value, issue_value, index == 0,
                             index + 1 == command_count,
                             head_last_value &&
                                 (index + 1 == command_count),
                             base_value, supertile_value);
            end
        end
    endtask

    task automatic send_term(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input logic [7:0] count_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        begin
            @(negedge clk_core);
            term_tag = tag_value;
            term_gate_code = gate_value;
            term_lane_id = lane_value;
            term_destination_count = count_value;
            term_issue_seq = issue_value;
            term_head_last = head_last_value;
            term_input_channel_base = base_value;
            term_logical_supertile = supertile_value;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;
        end
    endtask

    task automatic send_event(
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input logic [EVENT_WAYS-1:0] valid_value,
        input logic [(EVENT_WAYS*TOKEN_W)-1:0] token_value,
        input logic [COUNT_W-1:0] count_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            event_gate_code = gate_value;
            event_lane_id = lane_value;
            event_token_valid = valid_value;
            event_token_ids = token_value;
            event_count = count_value;
            event_issue_seq = issue_value;
            event_term_first = first_value;
            event_term_last = last_value;
            event_head_last = head_last_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic send_source_term(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input integer destination_count_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        logic [EVENT_WAYS-1:0] valid_mask;
        logic [(EVENT_WAYS*TOKEN_W)-1:0] packed_tokens;
        integer beat_index;
        integer beat_count;
        integer offset;
        integer assigned;
        begin
            queue_source_term(tag_value, gate_value, lane_value,
                              destination_count_value, issue_value,
                              head_last_value, base_value, supertile_value);
            send_term(tag_value, gate_value, lane_value,
                      8'(destination_count_value), issue_value,
                      head_last_value, base_value, supertile_value);
            beat_index = 0;
            offset = 0;
            while (offset < destination_count_value) begin
                case (beat_index % 4)
                    0: begin
                        valid_mask = 4'b1111;
                        beat_count = 4;
                    end
                    1: begin
                        valid_mask = 4'b1011;
                        beat_count = 3;
                    end
                    2: begin
                        valid_mask = 4'b0101;
                        beat_count = 2;
                    end
                    default: begin
                        valid_mask = 4'b1000;
                        beat_count = 1;
                    end
                endcase
                if (beat_count > destination_count_value - offset)
                    beat_count = destination_count_value - offset;
                packed_tokens = '0;
                assigned = 0;
                for (int way = 0; way < EVENT_WAYS; way = way + 1) begin
                    if (valid_mask[way] && assigned < beat_count) begin
                        packed_tokens[(way*TOKEN_W) +: TOKEN_W] =
                            source_tokens[offset + assigned];
                        assigned = assigned + 1;
                    end else begin
                        valid_mask[way] = 1'b0;
                    end
                end
                send_event(gate_value, lane_value, valid_mask, packed_tokens,
                           COUNT_W'(beat_count), issue_value, offset == 0,
                           offset + beat_count == destination_count_value,
                           head_last_value &&
                               (offset + beat_count ==
                                destination_count_value));
                offset = offset + beat_count;
                beat_index = beat_index + 1;
            end
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        logic [PAYLOAD_W-1:0] payload;
        payload = {cmd_group_tag, cmd_sequence, cmd_gate_code, cmd_lane_id,
                   cmd_destination_valid, cmd_destination_tokens,
                   cmd_term_issue_seq, cmd_term_first, cmd_term_last,
                   cmd_head_last, cmd_input_channel_base,
                   cmd_logical_supertile};
        if (rst_core || flush) begin
            held_valid_q = 1'b0;
        end else begin
            if (held_valid_q) begin
                stable_payload_checks = stable_payload_checks + 1;
                if (!cmd_valid || payload !== held_payload_q) begin
                    $error("TOKENS162 command payload changed under backpressure");
                    mismatch_count = mismatch_count + 1;
                end
            end
            held_valid_q = cmd_valid && !cmd_ready;
            if (held_valid_q)
                held_payload_q = payload;

            if (cmd_valid && !cmd_ready) begin
                backpressure_cycles = backpressure_cycles + 1;
                if (cmd_group_tag == FULL_TAG)
                    full_stall_cycles = full_stall_cycles + 1;
            end

            if (cmd_valid && cmd_ready) begin
                if (exp_head >= exp_tail) begin
                    $error("TOKENS162 unexpected command seq=%0d", cmd_sequence);
                    mismatch_count = mismatch_count + 1;
                end else begin
                    if (cmd_group_tag !== exp_tag[exp_head] ||
                        cmd_sequence !== CMD_W'(exp_head) ||
                        cmd_gate_code !== exp_gate[exp_head] ||
                        cmd_lane_id !== exp_lane[exp_head] ||
                        cmd_destination_valid !== exp_mask[exp_head] ||
                        cmd_destination_tokens !== exp_tokens[exp_head] ||
                        cmd_term_issue_seq !== exp_issue[exp_head] ||
                        cmd_term_first !== exp_first[exp_head] ||
                        cmd_term_last !== exp_last[exp_head] ||
                        cmd_head_last !== exp_head_last[exp_head] ||
                        cmd_input_channel_base !== exp_base[exp_head] ||
                        cmd_logical_supertile !== exp_supertile[exp_head]) begin
                        $error("TOKENS162 command mismatch index=%0d got={tag=%h seq=%0d mask=%b tokens=%h first=%b last=%b head_last=%b} exp={tag=%h seq=%0d mask=%b tokens=%h first=%b last=%b head_last=%b}",
                               exp_head, cmd_group_tag, cmd_sequence,
                               cmd_destination_valid, cmd_destination_tokens,
                               cmd_term_first, cmd_term_last, cmd_head_last,
                               exp_tag[exp_head], exp_head,
                               exp_mask[exp_head], exp_tokens[exp_head],
                               exp_first[exp_head], exp_last[exp_head],
                               exp_head_last[exp_head]);
                        mismatch_count = mismatch_count + 1;
                    end

                    case (cmd_group_tag)
                        FULL_TAG: begin
                            if (full_compact_commands == 80 &&
                                (cmd_destination_valid !== 2'b11 ||
                                 cmd_destination_tokens[0 +: TOKEN_W] !==
                                     8'd160 ||
                                 cmd_destination_tokens[
                                     TOKEN_W +: TOKEN_W] !== 8'd161 ||
                                 !cmd_term_last)) begin
                                $error("TOKENS162 full compact index 80 mismatch");
                                mismatch_count = mismatch_count + 1;
                            end
                            full_compact_commands =
                                full_compact_commands + 1;
                        end
                        EVEN_TAG: begin
                            if (even_compact_commands == 80 &&
                                (cmd_destination_valid !== 2'b01 ||
                                 cmd_destination_tokens[0 +: TOKEN_W] !==
                                     8'd160 || !cmd_term_last)) begin
                                $error("TOKENS162 even compact index 80 mismatch");
                                mismatch_count = mismatch_count + 1;
                            end
                            even_compact_commands =
                                even_compact_commands + 1;
                        end
                        ODD_TAG: begin
                            if (odd_compact_commands == 80 &&
                                (cmd_destination_valid !== 2'b10 ||
                                 cmd_destination_tokens[
                                     TOKEN_W +: TOKEN_W] !== 8'd161 ||
                                 !cmd_term_last)) begin
                                $error("TOKENS162 odd compact index 80 mismatch");
                                mismatch_count = mismatch_count + 1;
                            end
                            odd_compact_commands = odd_compact_commands + 1;
                        end
                        default: begin
                        end
                    endcase
                    exp_head = exp_head + 1;
                end

                observed_commands = observed_commands + 1;
                observed_destinations = observed_destinations +
                    32'(cmd_destination_valid[0]) +
                    32'(cmd_destination_valid[1]);
                case (cmd_destination_valid)
                    2'b11: paired_commands = paired_commands + 1;
                    2'b01: only_even_commands = only_even_commands + 1;
                    2'b10: only_odd_commands = only_odd_commands + 1;
                    default: begin
                        $error("TOKENS162 empty/invalid destination mask");
                        mismatch_count = mismatch_count + 1;
                    end
                endcase
            end
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        term_valid = 1'b0;
        event_valid = 1'b0;
        cmd_ready = 1'b0;
        term_tag = '0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        term_input_channel_base = '0;
        term_logical_supertile = '0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_issue_seq = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        exp_head = 0;
        exp_tail = 0;
        mismatch_count = 0;
        observed_commands = 0;
        observed_destinations = 0;
        paired_commands = 0;
        only_even_commands = 0;
        only_odd_commands = 0;
        backpressure_cycles = 0;
        full_stall_cycles = 0;
        stable_payload_checks = 0;
        flush_count = 0;
        full_compact_commands = 0;
        even_compact_commands = 0;
        odd_compact_commands = 0;
        held_valid_q = 1'b0;
        held_payload_q = '0;
        reset_checked = 1'b0;
        for (int index = 0; index < TOKENS; index = index + 1)
            source_tokens[index] = '0;

        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        if (!idle || cmd_valid || event_ready || !term_ready ||
            protocol_error || cmd_sequence !== '0)
            $fatal(1, "TOKENS162 reset state mismatch");
        reset_checked = 1'b1;
        rst_core = 1'b0;

        // Commit a command, then prove flush masks and discards it.
        send_term(32'hf100_0001, 9'h011, 5'd1, 8'd2, 13'd1,
                  1'b0, 10'd1, 8'd1);
        send_event(9'h011, 5'd1, 4'b0101,
                   {8'd0, 8'd1, 8'd0, 8'd0}, 3'd2, 13'd1,
                   1'b1, 1'b1, 1'b0);
        wait (cmd_valid);
        @(negedge clk_core);
        flush = 1'b1;
        #1;
        if (cmd_valid || term_ready || event_ready)
            $fatal(1, "TOKENS162 flush did not mask handshakes");
        @(negedge clk_core);
        flush = 1'b0;
        flush_count = flush_count + 1;
        repeat (2) @(posedge clk_core);
        if (!idle || cmd_valid || protocol_error || cmd_sequence !== '0)
            $fatal(1, "TOKENS162 flush state mismatch");

        // Full capacity. Each four-token group swaps its even arrival order.
        for (int base = 0; base < 160; base = base + 4) begin
            source_tokens[base] = TOKEN_W'(base + 2);
            source_tokens[base + 1] = TOKEN_W'(base + 1);
            source_tokens[base + 2] = TOKEN_W'(base);
            source_tokens[base + 3] = TOKEN_W'(base + 3);
        end
        source_tokens[160] = 8'd160;
        source_tokens[161] = 8'd161;
        cmd_ready = 1'b0;
        send_source_term(FULL_TAG, 9'h155, 5'd7, 162, 13'd101,
                         1'b0, 10'd321, 8'd21);
        wait (cmd_valid);
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        cmd_ready = 1'b1;
        wait (exp_head == exp_tail && idle);

        // Complete compact even capacity: tokens 0,2,...,160.
        for (int index = 0; index < PARITY_CAP; index = index + 1)
            source_tokens[index] = TOKEN_W'(index * 2);
        send_source_term(EVEN_TAG, 9'h0a6, 5'd13, PARITY_CAP, 13'd202,
                         1'b1, 10'd654, 8'd42);
        wait (exp_head == exp_tail && idle);

        // Complete compact odd capacity: tokens 1,3,...,161.
        for (int index = 0; index < PARITY_CAP; index = index + 1)
            source_tokens[index] = TOKEN_W'((index * 2) + 1);
        send_source_term(ODD_TAG, 9'h1c3, 5'd19, PARITY_CAP, 13'd303,
                         1'b1, 10'd777, 8'd63);
        wait (exp_head == exp_tail && idle);
        repeat (2) @(posedge clk_core);

        if (mismatch_count != 0 || exp_head != EXPECTED_COMMANDS ||
            exp_tail != EXPECTED_COMMANDS)
            $fatal(1, "TOKENS162 scoreboard mismatch=%0d head=%0d tail=%0d",
                   mismatch_count, exp_head, exp_tail);
        if (observed_commands != EXPECTED_COMMANDS ||
            observed_destinations != EXPECTED_DESTINATIONS)
            $fatal(1, "TOKENS162 totals mismatch commands=%0d destinations=%0d",
                   observed_commands, observed_destinations);
        if (paired_commands != PARITY_CAP ||
            only_even_commands != PARITY_CAP ||
            only_odd_commands != PARITY_CAP)
            $fatal(1, "TOKENS162 parity command totals mismatch");
        if (full_stall_cycles != 5 || stable_payload_checks < 5)
            $fatal(1, "TOKENS162 deterministic backpressure mismatch stall=%0d stable=%0d",
                   full_stall_cycles, stable_payload_checks);
        if (!reset_checked || flush_count != 1 || protocol_error)
            $fatal(1, "TOKENS162 reset/flush coverage mismatch");
        if (full_compact_commands != PARITY_CAP ||
            even_compact_commands != PARITY_CAP ||
            odd_compact_commands != PARITY_CAP)
            $fatal(1, "TOKENS162 compact index coverage mismatch full=%0d even=%0d odd=%0d",
                   full_compact_commands, even_compact_commands,
                   odd_compact_commands);

        $display("PASS PPDI ADAPTER TOKENS162 commands=%0d destinations=%0d paired=%0d only_even=%0d only_odd=%0d backpressure=%0d stable_checks=%0d flush=%0d",
                 observed_commands, observed_destinations, paired_commands,
                 only_even_commands, only_odd_commands,
                 backpressure_cycles, stable_payload_checks, flush_count);
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $fatal(1, "TOKENS162 PPDI adapter TB timeout");
    end
endmodule

`default_nettype wire
