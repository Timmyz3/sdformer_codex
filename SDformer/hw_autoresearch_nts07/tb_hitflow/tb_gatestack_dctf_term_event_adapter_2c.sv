`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
module tb_gatestack_dctf_term_event_adapter_2c;
    localparam int TOKENS = 10;
    localparam int EVENT_WAYS = 4;
    localparam int TAG_W = 12;
    localparam int GATE_W = 9;
    localparam int LANE_W = 3;
    localparam int INPUT_CH_W = 6;
    localparam int SUPERTILE_W = 3;
    localparam int TOKEN_W = 4;
    localparam int ISSUE_W = 5;
    localparam int CMD_W = 8;
    localparam int COUNT_W = 3;
    localparam int MAX_EXPECTED = 2048;
    localparam int STRESS_TERMS = 128;

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
    logic [TOKEN_W-1:0] cmd_destination_token;
    logic [ISSUE_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic [INPUT_CH_W-1:0] cmd_input_channel_base;
    logic [SUPERTILE_W-1:0] cmd_logical_supertile;
    logic idle;
    logic protocol_error;

    logic [TAG_W-1:0] exp_tag [0:MAX_EXPECTED-1];
    logic [GATE_W-1:0] exp_gate [0:MAX_EXPECTED-1];
    logic [LANE_W-1:0] exp_lane [0:MAX_EXPECTED-1];
    logic [TOKEN_W-1:0] exp_token [0:MAX_EXPECTED-1];
    logic [ISSUE_W-1:0] exp_issue [0:MAX_EXPECTED-1];
    logic exp_first [0:MAX_EXPECTED-1];
    logic exp_last [0:MAX_EXPECTED-1];
    logic exp_head_last [0:MAX_EXPECTED-1];
    logic [INPUT_CH_W-1:0] exp_base [0:MAX_EXPECTED-1];
    logic [SUPERTILE_W-1:0] exp_supertile [0:MAX_EXPECTED-1];

    integer exp_head;
    integer exp_tail;
    integer mismatch_count;
    integer overlap_cycles;
    integer backpressure_cycles;
    integer output_count_before_bad;
    integer stress_destination_count;
    integer stress_term_count;
    logic random_ready_enable;
    logic held_valid_q;
    logic [TAG_W+CMD_W+GATE_W+LANE_W+TOKEN_W+ISSUE_W+3+
           INPUT_CH_W+SUPERTILE_W-1:0] held_payload_q;

    gatestack_dctf_term_event_adapter_2c #(
        .TOKENS(TOKENS), .EVENT_WAYS(EVENT_WAYS), .TAG_W(TAG_W),
        .GATE_CODE_W(GATE_W), .LANE_ID_W(LANE_W),
        .INPUT_CH_W(INPUT_CH_W),
        .LOGICAL_SUPERTILE_W(SUPERTILE_W), .TOKEN_ID_W(TOKEN_W),
        .ISSUE_SEQ_W(ISSUE_W), .CMD_SEQUENCE_W(CMD_W),
        .WAY_COUNT_W(COUNT_W)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic add_expected(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_W-1:0] gate_value,
        input logic [LANE_W-1:0] lane_value,
        input logic [TOKEN_W-1:0] token_value,
        input logic [ISSUE_W-1:0] issue_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value,
        input logic [INPUT_CH_W-1:0] base_value,
        input logic [SUPERTILE_W-1:0] supertile_value
    );
        begin
            exp_tag[exp_tail] = tag_value;
            exp_gate[exp_tail] = gate_value;
            exp_lane[exp_tail] = lane_value;
            exp_token[exp_tail] = token_value;
            exp_issue[exp_tail] = issue_value;
            exp_first[exp_tail] = first_value;
            exp_last[exp_tail] = last_value;
            exp_head_last[exp_tail] = head_last_value;
            exp_base[exp_tail] = base_value;
            exp_supertile[exp_tail] = supertile_value;
            exp_tail = exp_tail + 1;
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
        end
    endtask

    always @(posedge clk_core) begin : p_scoreboard
        logic [TAG_W+CMD_W+GATE_W+LANE_W+TOKEN_W+ISSUE_W+3+
               INPUT_CH_W+SUPERTILE_W-1:0] payload;
        if (!rst_core) begin
            payload = {cmd_group_tag, cmd_sequence, cmd_gate_code,
                       cmd_lane_id, cmd_destination_token,
                       cmd_term_issue_seq, cmd_term_first, cmd_term_last,
                       cmd_head_last, cmd_input_channel_base,
                       cmd_logical_supertile};
            if (cmd_valid && !cmd_ready && !flush) begin
                backpressure_cycles = backpressure_cycles + 1;
                if (held_valid_q && payload !== held_payload_q) begin
                    $error("2c command payload changed under backpressure");
                    mismatch_count = mismatch_count + 1;
                end
                held_valid_q = 1'b1;
                held_payload_q = payload;
            end else begin
                held_valid_q = 1'b0;
            end
            if (event_valid && event_ready && cmd_valid && cmd_ready)
                overlap_cycles = overlap_cycles + 1;
            if (cmd_valid && cmd_ready) begin
                if (exp_head >= exp_tail) begin
                    $error("2c unexpected command");
                    mismatch_count = mismatch_count + 1;
                end else begin
                    if (cmd_group_tag !== exp_tag[exp_head] ||
                        cmd_sequence !== CMD_W'(exp_head) ||
                        cmd_gate_code !== exp_gate[exp_head] ||
                        cmd_lane_id !== exp_lane[exp_head] ||
                        cmd_destination_token !== exp_token[exp_head] ||
                        cmd_term_issue_seq !== exp_issue[exp_head] ||
                        cmd_term_first !== exp_first[exp_head] ||
                        cmd_term_last !== exp_last[exp_head] ||
                        cmd_head_last !== exp_head_last[exp_head] ||
                        cmd_input_channel_base !== exp_base[exp_head] ||
                        cmd_logical_supertile !== exp_supertile[exp_head]) begin
                        $error("2c command mismatch index=%0d", exp_head);
                        mismatch_count = mismatch_count + 1;
                    end
                    exp_head = exp_head + 1;
                end
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
        overlap_cycles = 0;
        backpressure_cycles = 0;
        stress_destination_count = 0;
        stress_term_count = 0;
        random_ready_enable = 1'b0;
        held_valid_q = 1'b0;
        held_payload_q = '0;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        for (int token = 1; token <= 6; token = token + 1)
            add_expected(12'h101, 9'h055, 3'd1, TOKEN_W'(token),
                         5'd3, token == 1, token == 6, 1'b0,
                         6'd5, 3'd1);
        add_expected(12'h202, 9'h066, 3'd2, 4'd7, 5'd4,
                     1'b1, 1'b0, 1'b0, 6'd19, 3'd5);
        add_expected(12'h202, 9'h066, 3'd2, 4'd8, 5'd4,
                     1'b0, 1'b1, 1'b1, 6'd19, 3'd5);

        send_term(12'h101, 9'h055, 3'd1, 8'd6, 5'd3, 1'b0,
                  6'd5, 3'd1);
        send_event(9'h055, 3'd1, 4'b1111, {4'd4,4'd3,4'd2,4'd1},
                   3'd4, 5'd3, 1'b1, 1'b0, 1'b0);
        send_event(9'h055, 3'd1, 4'b0011, {4'd0,4'd0,4'd6,4'd5},
                   3'd2, 5'd3, 1'b0, 1'b1, 1'b0);
        wait (cmd_valid);
        send_term(12'h202, 9'h066, 3'd2, 8'd2, 5'd4, 1'b1,
                  6'd19, 3'd5);
        cmd_ready = 1'b1;
        send_event(9'h066, 3'd2, 4'b0011, {4'd0,4'd0,4'd8,4'd7},
                   3'd2, 5'd4, 1'b1, 1'b1, 1'b1);
        wait (exp_head == exp_tail && idle);
        if (overlap_cycles == 0)
            $fatal(1, "2c collect/emit overlap not observed");

        output_count_before_bad = exp_head;
        send_term(12'h303, 9'h077, 3'd3, 8'd2, 5'd5, 1'b0,
                  6'd31, 3'd6);
        send_event(9'h077, 3'd3, 4'b0011, {4'd0,4'd0,4'd4,4'd4},
                   3'd2, 5'd5, 1'b1, 1'b1, 1'b0);
        wait (idle);
        if (!protocol_error || exp_head != output_count_before_bad)
            $fatal(1, "2c malformed term was not atomically rejected");
        pulse_clear_error();
        if (protocol_error)
            $fatal(1, "2c protocol error did not clear");

        cmd_ready = 1'b0;
        send_term(12'h404, 9'h088, 3'd4, 8'd2, 5'd6, 1'b0,
                  6'd41, 3'd7);
        send_event(9'h088, 3'd4, 4'b0011, {4'd0,4'd0,4'd3,4'd2},
                   3'd2, 5'd6, 1'b1, 1'b1, 1'b0);
        wait (cmd_valid);
        pulse_flush();
        cmd_ready = 1'b1;
        repeat (4) @(posedge clk_core);
        if (!idle || cmd_valid)
            $fatal(1, "2c flush did not clear both contexts");

        random_ready_enable = 1'b1;
        fork
            begin : p_random_ready
                integer ready_state;
                ready_state = 32'h1357_2468;
                while (random_ready_enable) begin
                    @(negedge clk_core);
                    ready_state = (ready_state * 32'd1103515245) + 32'd12345;
                    cmd_ready = ready_state[2] | ready_state[5];
                end
                cmd_ready = 1'b1;
            end
            begin : p_stress_driver
                logic [EVENT_WAYS-1:0] valid_mask;
                logic [(EVENT_WAYS*TOKEN_W)-1:0] packed_tokens;
                integer destination_count;
                integer beat_count;
                logic [TOKEN_W-1:0] token_value;
                for (int term_index = 0; term_index < STRESS_TERMS;
                     term_index = term_index + 1) begin
                    destination_count = (term_index % TOKENS) + 1;
                    for (int destination = 0;
                         destination < destination_count;
                         destination = destination + 1) begin
                        token_value = TOKEN_W'(((term_index * 7) +
                                       (destination * 3)) % TOKENS);
                        add_expected(TAG_W'(32'h0000_0500 + term_index),
                                     GATE_W'(32'h0000_0100 + term_index),
                                     LANE_W'(term_index),
                                     TOKEN_W'(token_value),
                                     ISSUE_W'(term_index),
                                     destination == 0,
                                     destination + 1 == destination_count,
                                     (destination + 1 == destination_count) &&
                                         ((term_index % 7) == 6),
                                     INPUT_CH_W'(term_index * 3),
                                     SUPERTILE_W'(term_index));
                    end
                    send_term(TAG_W'(32'h0000_0500 + term_index),
                              GATE_W'(32'h0000_0100 + term_index),
                              LANE_W'(term_index),
                              8'(destination_count), ISSUE_W'(term_index),
                              (term_index % 7) == 6,
                              INPUT_CH_W'(term_index * 3),
                              SUPERTILE_W'(term_index));
                    for (int offset = 0; offset < destination_count;
                         offset = offset + EVENT_WAYS) begin
                        beat_count = destination_count - offset;
                        if (beat_count > EVENT_WAYS)
                            beat_count = EVENT_WAYS;
                        valid_mask = '0;
                        packed_tokens = '0;
                        for (int way = 0; way < EVENT_WAYS;
                             way = way + 1) begin
                            if (way < beat_count) begin
                                token_value = TOKEN_W'(((term_index * 7) +
                                    ((offset + way) * 3)) % TOKENS);
                                valid_mask[way] = 1'b1;
                                packed_tokens[(way*TOKEN_W) +: TOKEN_W] =
                                    TOKEN_W'(token_value);
                            end
                        end
                        send_event(GATE_W'(32'h0000_0100 + term_index),
                                   LANE_W'(term_index), valid_mask,
                                   packed_tokens, COUNT_W'(beat_count),
                                   ISSUE_W'(term_index), offset == 0,
                                   offset + beat_count == destination_count,
                                   (offset + beat_count == destination_count) &&
                                       ((term_index % 7) == 6));
                    end
                    stress_term_count = stress_term_count + 1;
                    stress_destination_count = stress_destination_count +
                                               destination_count;
                end
                wait (exp_head == exp_tail && idle);
                random_ready_enable = 1'b0;
            end
        join

        if (mismatch_count != 0 || exp_head != exp_tail)
            $fatal(1, "2c scoreboard mismatch=%0d head=%0d tail=%0d",
                   mismatch_count, exp_head, exp_tail);
        if (stress_term_count != STRESS_TERMS ||
            stress_destination_count != 696)
            $fatal(1, "2c stress accounting mismatch terms=%0d destinations=%0d",
                   stress_term_count, stress_destination_count);
        $display("PASS DCTF ADAPTER 2C outputs=%0d overlap=%0d backpressure=%0d stress_terms=%0d stress_destinations=%0d",
                 exp_head, overlap_cycles, backpressure_cycles,
                 stress_term_count, stress_destination_count);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "2c adapter timeout");
    end
endmodule

`default_nettype wire
