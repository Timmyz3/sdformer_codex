`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
module tb_gatestack_ppdi_term_event_adapter_2c;
    localparam int TOKENS = 11;
    localparam int EVENT_WAYS = 4;
    localparam int TAG_W = 12;
    localparam int GATE_W = 9;
    localparam int LANE_W = 3;
    localparam int INPUT_CH_W = 6;
    localparam int SUPERTILE_W = 3;
    localparam int TOKEN_W = 4;
    localparam int ISSUE_W = 5;
    localparam int CMD_W = 5;
    localparam int COUNT_W = 3;
    localparam int EVEN_CAP = (TOKENS + 1) / 2;
    localparam int ODD_CAP = TOKENS / 2;
    localparam int MAX_EXPECTED = 1024;
    localparam int STRESS_TERMS = 48;
    localparam int PAYLOAD_W = TAG_W + CMD_W + GATE_W + LANE_W + 2 +
        (2 * TOKEN_W) + ISSUE_W + 3 + INPUT_CH_W + SUPERTILE_W;

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

    logic [TAG_W-1:0] exp_tag [0:MAX_EXPECTED-1];
    logic [CMD_W-1:0] exp_sequence [0:MAX_EXPECTED-1];
    logic [GATE_W-1:0] exp_gate [0:MAX_EXPECTED-1];
    logic [LANE_W-1:0] exp_lane [0:MAX_EXPECTED-1];
    logic [1:0] exp_destination_valid [0:MAX_EXPECTED-1];
    logic [(2*TOKEN_W)-1:0] exp_destination_tokens
        [0:MAX_EXPECTED-1];
    logic [ISSUE_W-1:0] exp_issue [0:MAX_EXPECTED-1];
    logic exp_first [0:MAX_EXPECTED-1];
    logic exp_last [0:MAX_EXPECTED-1];
    logic exp_head_last [0:MAX_EXPECTED-1];
    logic [INPUT_CH_W-1:0] exp_base [0:MAX_EXPECTED-1];
    logic [SUPERTILE_W-1:0] exp_supertile [0:MAX_EXPECTED-1];
    logic [TOKEN_W-1:0] test_tokens [0:TOKENS-1];

    integer exp_head;
    integer exp_tail;
    integer mismatch_count;
    integer expected_destination_count;
    integer observed_destination_count;
    integer overlap_cycles;
    integer full_context_cycles;
    integer backpressure_cycles;
    integer paired_commands;
    integer only_even_commands;
    integer only_odd_commands;
    integer stress_term_count;
    integer malformed_term_count;
    integer flush_count;
    integer sequence_wrap_count;
    integer output_count_before_bad;
    logic random_ready_enable;
    logic held_valid_q;
    logic [PAYLOAD_W-1:0] held_payload_q;
    logic command_seen_q;
    logic [CMD_W-1:0] last_sequence_q;
    logic sequence_wrap_seen;
    logic gate_zero_legal_seen;

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

    task automatic add_expected_command(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input logic [1:0] destination_valid_value,
        input logic [(2*TOKEN_W)-1:0] destination_tokens_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        begin
            if (exp_tail >= MAX_EXPECTED)
                $fatal(1, "PPDI expected queue overflow");
            exp_tag[exp_tail] = tag_value;
            exp_sequence[exp_tail] = CMD_W'(exp_tail);
            exp_gate[exp_tail] = gate_value;
            exp_lane[exp_tail] = lane_value;
            exp_destination_valid[exp_tail] = destination_valid_value;
            exp_destination_tokens[exp_tail] = destination_tokens_value;
            exp_issue[exp_tail] = issue_value;
            exp_first[exp_tail] = first_value;
            exp_last[exp_tail] = last_value;
            exp_head_last[exp_tail] = head_last_value;
            exp_base[exp_tail] = base_value;
            exp_supertile[exp_tail] = supertile_value;
            exp_tail = exp_tail + 1;
            expected_destination_count = expected_destination_count +
                32'(destination_valid_value[0]) +
                32'(destination_valid_value[1]);
        end
    endtask

    task automatic queue_current_term(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input integer destination_count_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        logic [TOKEN_W-1:0] even_tokens [0:EVEN_CAP-1];
        logic [TOKEN_W-1:0] odd_tokens [0:ODD_CAP-1];
        logic [1:0] valid_value;
        logic [(2*TOKEN_W)-1:0] packed_value;
        integer even_count;
        integer odd_count;
        integer command_count;
        begin
            even_count = 0;
            odd_count = 0;
            for (int index = 0; index < destination_count_value;
                 index = index + 1) begin
                if (test_tokens[index][0]) begin
                    odd_tokens[odd_count] = test_tokens[index];
                    odd_count = odd_count + 1;
                end else begin
                    even_tokens[even_count] = test_tokens[index];
                    even_count = even_count + 1;
                end
            end
            command_count = (even_count >= odd_count) ? even_count : odd_count;
            for (int index = 0; index < command_count;
                 index = index + 1) begin
                valid_value = '0;
                packed_value = '0;
                if (index < even_count) begin
                    valid_value[0] = 1'b1;
                    packed_value[0 +: TOKEN_W] = even_tokens[index];
                end
                if (index < odd_count) begin
                    valid_value[1] = 1'b1;
                    packed_value[TOKEN_W +: TOKEN_W] = odd_tokens[index];
                end
                add_expected_command(
                    tag_value, gate_value, lane_value, valid_value,
                    packed_value, issue_value, index == 0,
                    index + 1 == command_count,
                    head_last_value && (index + 1 == command_count),
                    base_value, supertile_value
                );
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

    task automatic send_current_term(
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
        integer beat_count;
        begin
            queue_current_term(tag_value, gate_value, lane_value,
                               destination_count_value, issue_value,
                               head_last_value, base_value, supertile_value);
            send_term(tag_value, gate_value, lane_value,
                      8'(destination_count_value), issue_value,
                      head_last_value, base_value, supertile_value);
            for (int offset = 0; offset < destination_count_value;
                 offset = offset + EVENT_WAYS) begin
                beat_count = destination_count_value - offset;
                if (beat_count > EVENT_WAYS)
                    beat_count = EVENT_WAYS;
                valid_mask = '0;
                packed_tokens = '0;
                for (int way = 0; way < EVENT_WAYS; way = way + 1) begin
                    if (way < beat_count) begin
                        valid_mask[way] = 1'b1;
                        packed_tokens[(way*TOKEN_W) +: TOKEN_W] =
                            test_tokens[offset + way];
                    end
                end
                send_event(gate_value, lane_value, valid_mask, packed_tokens,
                           COUNT_W'(beat_count), issue_value, offset == 0,
                           offset + beat_count == destination_count_value,
                           head_last_value &&
                               (offset + beat_count ==
                                destination_count_value));
            end
        end
    endtask

    task automatic pulse_clear_error;
        begin
            @(negedge clk_core);
            clear_error = 1'b1;
            @(negedge clk_core);
            clear_error = 1'b0;
        end
    endtask

    task automatic pulse_flush;
        begin
            @(negedge clk_core);
            flush = 1'b1;
            @(negedge clk_core);
            flush = 1'b0;
            flush_count = flush_count + 1;
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        logic [PAYLOAD_W-1:0] payload;
        payload = {cmd_group_tag, cmd_sequence, cmd_gate_code, cmd_lane_id,
                   cmd_destination_valid, cmd_destination_tokens,
                   cmd_term_issue_seq, cmd_term_first, cmd_term_last,
                   cmd_head_last, cmd_input_channel_base,
                   cmd_logical_supertile};
        if (rst_core) begin
            held_valid_q = 1'b0;
            command_seen_q = 1'b0;
            last_sequence_q = '0;
        end else begin
            if (held_valid_q && !flush &&
                (!cmd_valid || payload !== held_payload_q)) begin
                $error("PPDI command changed under backpressure");
                mismatch_count = mismatch_count + 1;
            end
            held_valid_q = cmd_valid && !cmd_ready && !flush;
            if (held_valid_q)
                held_payload_q = payload;
            if (cmd_valid && !cmd_ready && !flush)
                backpressure_cycles = backpressure_cycles + 1;
            if (event_valid && event_ready && cmd_valid)
                overlap_cycles = overlap_cycles + 1;
            if (dut.context_valid_q == 2'b11)
                full_context_cycles = full_context_cycles + 1;
            if (flush && cmd_valid) begin
                $error("PPDI flush did not mask command valid");
                mismatch_count = mismatch_count + 1;
            end
            if (cmd_valid) begin
                if (cmd_destination_valid == 2'b00) begin
                    $error("PPDI empty destination mask");
                    mismatch_count = mismatch_count + 1;
                end
                if (cmd_destination_valid[0] && cmd_destination_tokens[0]) begin
                    $error("PPDI even port carried odd token");
                    mismatch_count = mismatch_count + 1;
                end
                if (cmd_destination_valid[1] &&
                    !cmd_destination_tokens[TOKEN_W]) begin
                    $error("PPDI odd port carried even token");
                    mismatch_count = mismatch_count + 1;
                end
            end
            if (cmd_valid && cmd_ready) begin
                if (exp_head >= exp_tail) begin
                    $error("PPDI unexpected command");
                    mismatch_count = mismatch_count + 1;
                end else begin
                    if (cmd_group_tag !== exp_tag[exp_head] ||
                        cmd_sequence !== exp_sequence[exp_head] ||
                        cmd_gate_code !== exp_gate[exp_head] ||
                        cmd_lane_id !== exp_lane[exp_head] ||
                        cmd_destination_valid !==
                            exp_destination_valid[exp_head] ||
                        cmd_destination_tokens !==
                            exp_destination_tokens[exp_head] ||
                        cmd_term_issue_seq !== exp_issue[exp_head] ||
                        cmd_term_first !== exp_first[exp_head] ||
                        cmd_term_last !== exp_last[exp_head] ||
                        cmd_head_last !== exp_head_last[exp_head] ||
                        cmd_input_channel_base !== exp_base[exp_head] ||
                        cmd_logical_supertile !== exp_supertile[exp_head]) begin
                        $error("PPDI command mismatch index=%0d got={tag=%h seq=%h mask=%b tok=%h first=%b last=%b} exp={tag=%h seq=%h mask=%b tok=%h first=%b last=%b}",
                               exp_head, cmd_group_tag, cmd_sequence,
                               cmd_destination_valid, cmd_destination_tokens,
                               cmd_term_first, cmd_term_last,
                               exp_tag[exp_head], exp_sequence[exp_head],
                               exp_destination_valid[exp_head],
                               exp_destination_tokens[exp_head],
                               exp_first[exp_head], exp_last[exp_head]);
                        mismatch_count = mismatch_count + 1;
                    end
                    exp_head = exp_head + 1;
                end
                observed_destination_count = observed_destination_count +
                    32'(cmd_destination_valid[0]) +
                    32'(cmd_destination_valid[1]);
                case (cmd_destination_valid)
                    2'b11: paired_commands = paired_commands + 1;
                    2'b01: only_even_commands = only_even_commands + 1;
                    2'b10: only_odd_commands = only_odd_commands + 1;
                    default: mismatch_count = mismatch_count + 1;
                endcase
                if (command_seen_q && last_sequence_q == {CMD_W{1'b1}} &&
                    cmd_sequence == '0) begin
                    sequence_wrap_seen = 1'b1;
                    sequence_wrap_count = sequence_wrap_count + 1;
                end
                command_seen_q = 1'b1;
                last_sequence_q = cmd_sequence;
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
        expected_destination_count = 0;
        observed_destination_count = 0;
        overlap_cycles = 0;
        full_context_cycles = 0;
        backpressure_cycles = 0;
        paired_commands = 0;
        only_even_commands = 0;
        only_odd_commands = 0;
        stress_term_count = 0;
        malformed_term_count = 0;
        flush_count = 0;
        sequence_wrap_count = 0;
        output_count_before_bad = 0;
        random_ready_enable = 1'b0;
        held_valid_q = 1'b0;
        held_payload_q = '0;
        command_seen_q = 1'b0;
        last_sequence_q = '0;
        sequence_wrap_seen = 1'b0;
        gate_zero_legal_seen = 1'b0;
        for (int index = 0; index < TOKENS; index = index + 1)
            test_tokens[index] = '0;
        repeat (4) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        cmd_ready = 1'b1;

        // Balanced and mixed arrival order: parity order must stay stable.
        test_tokens[0] = 4'd0;
        test_tokens[1] = 4'd1;
        test_tokens[2] = 4'd4;
        test_tokens[3] = 4'd3;
        send_current_term(12'h101, 9'h055, 3'd1, 4, 5'd3, 1'b0,
                          6'd5, 3'd1);

        // Fill the complete compact even array over multiple event beats.
        for (int index = 0; index < EVEN_CAP; index = index + 1)
            test_tokens[index] = TOKEN_W'(index * 2);
        send_current_term(12'h202, 9'h066, 3'd2, EVEN_CAP, 5'd4, 1'b0,
                          6'd9, 3'd2);

        // Fill the complete compact odd array over multiple event beats.
        for (int index = 0; index < ODD_CAP; index = index + 1)
            test_tokens[index] = TOKEN_W'((index * 2) + 1);
        send_current_term(12'h303, 9'h077, 3'd3, ODD_CAP, 5'd5, 1'b1,
                          6'd13, 3'd3);

        test_tokens[0] = 4'd2;
        test_tokens[1] = 4'd1;
        test_tokens[2] = 4'd4;
        test_tokens[3] = 4'd6;
        test_tokens[4] = 4'd8;
        send_current_term(12'h404, 9'h088, 3'd4, 5, 5'd6, 1'b0,
                          6'd17, 3'd4);
        wait (exp_head == exp_tail && idle);

        // Hold context 0 at the output while context 1 collects and commits.
        cmd_ready = 1'b0;
        test_tokens[0] = 4'd0;
        test_tokens[1] = 4'd3;
        test_tokens[2] = 4'd2;
        test_tokens[3] = 4'd5;
        send_current_term(12'h505, 9'h099, 3'd5, 4, 5'd7, 1'b0,
                          6'd21, 3'd5);
        wait (cmd_valid);
        test_tokens[0] = 4'd7;
        test_tokens[1] = 4'd4;
        test_tokens[2] = 4'd9;
        send_current_term(12'h606, 9'h0aa, 3'd6, 3, 5'd8, 1'b1,
                          6'd25, 3'd6);
        repeat (2) @(posedge clk_core);
        if (dut.context_valid_q != 2'b11)
            $fatal(1, "PPDI two-context overlap was not established");
        @(negedge clk_core);
        cmd_ready = 1'b1;
        wait (exp_head == exp_tail && idle);

        // A late duplicate invalidates the entire term before any command.
        output_count_before_bad = exp_head;
        send_term(12'h707, 9'h0bb, 3'd7, 8'd4, 5'd9, 1'b0,
                  6'd29, 3'd7);
        send_event(9'h0bb, 3'd7, 4'b0011,
                   {4'd0, 4'd0, 4'd1, 4'd0}, 3'd2, 5'd9,
                   1'b1, 1'b0, 1'b0);
        send_event(9'h0bb, 3'd7, 4'b0011,
                   {4'd0, 4'd0, 4'd2, 4'd2}, 3'd2, 5'd9,
                   1'b0, 1'b1, 1'b0);
        wait (idle);
        repeat (3) @(posedge clk_core);
        if (!protocol_error || exp_head != output_count_before_bad)
            $fatal(1, "PPDI malformed term escaped atomic rejection");
        malformed_term_count = malformed_term_count + 1;
        pulse_clear_error();
        if (protocol_error)
            $fatal(1, "PPDI protocol error did not clear");

        // An out-of-range token invalidates the complete term atomically.
        output_count_before_bad = exp_head;
        send_term(12'h708, 9'h0bc, 3'd6, 8'd1, 5'd14, 1'b0,
                  6'd30, 3'd6);
        send_event(9'h0bc, 3'd6, 4'b0001,
                   {4'd0, 4'd0, 4'd0, 4'd15}, 3'd1, 5'd14,
                   1'b1, 1'b1, 1'b0);
        wait (idle);
        repeat (2) @(posedge clk_core);
        if (!protocol_error || exp_head != output_count_before_bad)
            $fatal(1, "PPDI out-of-range term escaped atomic rejection");
        malformed_term_count = malformed_term_count + 1;
        pulse_clear_error();
        if (protocol_error)
            $fatal(1, "PPDI out-of-range sticky error did not clear");

        // A new bad term wins over a simultaneous clear of an old error.
        @(negedge clk_core);
        clear_error = 1'b1;
        term_tag = 12'h70A;
        term_gate_code = 9'h0bf;
        term_lane_id = 3'd4;
        term_destination_count = 8'd0;
        term_issue_seq = 5'd16;
        term_head_last = 1'b0;
        term_input_channel_base = 6'd30;
        term_logical_supertile = 3'd4;
        term_valid = 1'b1;
        do @(posedge clk_core); while (!term_ready);
        @(negedge clk_core);
        term_valid = 1'b0;
        clear_error = 1'b0;
        if (!protocol_error)
            $fatal(1, "PPDI simultaneous clear hid a bad term");
        malformed_term_count = malformed_term_count + 1;
        pulse_clear_error();

        // A new bad event also wins over clear_error in the same cycle.
        send_term(12'h70B, 9'h0c1, 3'd4, 8'd1, 5'd17, 1'b0,
                  6'd30, 3'd4);
        @(negedge clk_core);
        clear_error = 1'b1;
        event_gate_code = 9'h0c2;
        event_lane_id = 3'd4;
        event_token_valid = 4'b0001;
        event_token_ids = {4'd0, 4'd0, 4'd0, 4'd2};
        event_count = 3'd1;
        event_issue_seq = 5'd17;
        event_term_first = 1'b1;
        event_term_last = 1'b1;
        event_head_last = 1'b0;
        event_valid = 1'b1;
        do @(posedge clk_core); while (!event_ready);
        @(negedge clk_core);
        event_valid = 1'b0;
        clear_error = 1'b0;
        wait (idle);
        if (!protocol_error)
            $fatal(1, "PPDI simultaneous clear hid a bad event");
        malformed_term_count = malformed_term_count + 1;
        pulse_clear_error();

        // Metadata mismatch is sticky across flush and emits no command.
        output_count_before_bad = exp_head;
        send_term(12'h709, 9'h0bd, 3'd5, 8'd1, 5'd15, 1'b0,
                  6'd31, 3'd5);
        send_event(9'h0be, 3'd5, 4'b0001,
                   {4'd0, 4'd0, 4'd0, 4'd1}, 3'd1, 5'd15,
                   1'b1, 1'b1, 1'b0);
        wait (idle);
        repeat (3) @(posedge clk_core);
        if (!protocol_error || exp_head != output_count_before_bad)
            $fatal(1, "PPDI metadata term escaped atomic rejection");
        malformed_term_count = malformed_term_count + 1;
        pulse_flush();
        @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "PPDI flush unexpectedly cleared sticky error");
        pulse_clear_error();
        if (protocol_error)
            $fatal(1, "PPDI metadata sticky error did not clear");

        // Gate zero remains legal, matching the scalar adapter contract.
        test_tokens[0] = 4'd2;
        test_tokens[1] = 4'd3;
        test_tokens[2] = 4'd4;
        send_current_term(12'h7A7, 9'h000, 3'd2, 3, 5'd10, 1'b0,
                          6'd31, 3'd2);
        wait (exp_head == exp_tail && idle);
        if (protocol_error)
            $fatal(1, "PPDI gate-zero legal term raised protocol error");
        gate_zero_legal_seen = 1'b1;

        // Flush one committed context and one partially collected context.
        cmd_ready = 1'b0;
        send_term(12'h808, 9'h0cc, 3'd0, 8'd2, 5'd10, 1'b0,
                  6'd33, 3'd0);
        send_event(9'h0cc, 3'd0, 4'b0011,
                   {4'd0, 4'd0, 4'd3, 4'd2}, 3'd2, 5'd10,
                   1'b1, 1'b1, 1'b0);
        wait (cmd_valid);
        send_term(12'h909, 9'h0dd, 3'd1, 8'd4, 5'd11, 1'b0,
                  6'd37, 3'd1);
        send_event(9'h0dd, 3'd1, 4'b0011,
                   {4'd0, 4'd0, 4'd5, 4'd4}, 3'd2, 5'd11,
                   1'b1, 1'b0, 1'b0);
        pulse_flush();
        cmd_ready = 1'b1;
        repeat (3) @(posedge clk_core);
        if (!idle || cmd_valid || dut.context_valid_q != 2'b00)
            $fatal(1, "PPDI flush did not clear both contexts");

        random_ready_enable = 1'b1;
        fork
            begin : p_random_ready
                integer ready_state;
                ready_state = 32'h1357_2468;
                while (random_ready_enable) begin
                    @(negedge clk_core);
                    ready_state = (ready_state * 32'd1103515245) +
                                  32'd12345;
                    cmd_ready = ready_state[1] | ready_state[4];
                end
                cmd_ready = 1'b1;
            end
            begin : p_stress_driver
                integer destination_count;
                for (int term_index = 0; term_index < STRESS_TERMS;
                     term_index = term_index + 1) begin
                    destination_count = (term_index % TOKENS) + 1;
                    for (int destination = 0;
                         destination < destination_count;
                         destination = destination + 1) begin
                        test_tokens[destination] = TOKEN_W'(
                            ((term_index * 5) + (destination * 3)) % TOKENS);
                    end
                    send_current_term(
                        TAG_W'(32'h0000_0A00 + term_index),
                        GATE_W'(32'h0000_0100 + term_index),
                        LANE_W'(term_index), destination_count,
                        ISSUE_W'(term_index + 12),
                        (term_index % 9) == 8,
                        INPUT_CH_W'(term_index * 3),
                        SUPERTILE_W'(term_index)
                    );
                    stress_term_count = stress_term_count + 1;
                end
                wait (exp_head == exp_tail && idle);
                random_ready_enable = 1'b0;
            end
        join

        if (mismatch_count != 0 || exp_head != exp_tail)
            $fatal(1, "PPDI scoreboard mismatch=%0d head=%0d tail=%0d",
                   mismatch_count, exp_head, exp_tail);
        if (observed_destination_count != expected_destination_count)
            $fatal(1, "PPDI destination conservation mismatch exp=%0d got=%0d",
                   expected_destination_count, observed_destination_count);
        if (overlap_cycles == 0 || full_context_cycles == 0)
            $fatal(1, "PPDI context overlap coverage missing");
        if (backpressure_cycles == 0)
            $fatal(1, "PPDI random backpressure coverage missing");
        if (!sequence_wrap_seen)
            $fatal(1, "PPDI command sequence wrap coverage missing");
        if (malformed_term_count != 5 || flush_count != 2 ||
            sequence_wrap_count == 0)
            $fatal(1, "PPDI malformed/flush/wrap coverage count mismatch");
        if (!gate_zero_legal_seen)
            $fatal(1, "PPDI gate-zero scalar-contract coverage missing");
        if (paired_commands == 0 || only_even_commands < EVEN_CAP ||
            only_odd_commands < ODD_CAP)
            $fatal(1, "PPDI parity command coverage missing");
        if (stress_term_count != STRESS_TERMS)
            $fatal(1, "PPDI stress term count mismatch");
        $display("PASS PPDI ADAPTER 2C commands=%0d destinations=%0d overlap=%0d full=%0d backpressure=%0d wraps=%0d gate_zero_legal=%0d paired=%0d only_even=%0d only_odd=%0d malformed=%0d flush=%0d stress_terms=%0d",
                 exp_head, observed_destination_count, overlap_cycles,
                 full_context_cycles, backpressure_cycles,
                 sequence_wrap_count, gate_zero_legal_seen, paired_commands,
                 only_even_commands, only_odd_commands,
                 malformed_term_count, flush_count, stress_term_count);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "PPDI adapter TB timeout");
    end
endmodule

`default_nettype wire
