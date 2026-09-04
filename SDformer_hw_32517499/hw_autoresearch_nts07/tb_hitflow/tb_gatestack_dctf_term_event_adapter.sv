`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off BLKSEQ */
/* verilator lint_off UNUSEDSIGNAL */

module tb_gatestack_dctf_term_event_adapter;
    localparam int TOKENS = 10;
    localparam int EVENT_WAYS = 4;
    localparam int TAG_W = 12;
    localparam int GATE_CODE_W = 9;
    localparam int LANE_ID_W = 3;
    localparam int TOKEN_ID_W = 4;
    localparam int ISSUE_SEQ_W = 5;
    localparam int CMD_SEQUENCE_W = 8;
    localparam int WAY_COUNT_W = 3;
    localparam int MAX_EXPECTED = 64;

    logic clk_core;
    logic rst_core;
    logic flush;
    logic clear_error;
    logic term_valid;
    logic term_ready;
    logic [TAG_W-1:0] term_tag;
    logic [GATE_CODE_W-1:0] term_gate_code;
    logic [LANE_ID_W-1:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [ISSUE_SEQ_W-1:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [GATE_CODE_W-1:0] event_gate_code;
    logic [LANE_ID_W-1:0] event_lane_id;
    logic [EVENT_WAYS-1:0] event_token_valid;
    logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids;
    logic [WAY_COUNT_W-1:0] event_count;
    logic [ISSUE_SEQ_W-1:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic cmd_valid;
    logic cmd_ready;
    logic [TAG_W-1:0] cmd_group_tag;
    logic [CMD_SEQUENCE_W-1:0] cmd_sequence;
    logic [GATE_CODE_W-1:0] cmd_gate_code;
    logic [LANE_ID_W-1:0] cmd_lane_id;
    logic [TOKEN_ID_W-1:0] cmd_destination_token;
    logic [ISSUE_SEQ_W-1:0] cmd_term_issue_seq;
    logic cmd_term_first;
    logic cmd_term_last;
    logic cmd_head_last;
    logic protocol_error;

    logic [TAG_W-1:0] exp_tag [0:MAX_EXPECTED-1];
    logic [GATE_CODE_W-1:0] exp_gate [0:MAX_EXPECTED-1];
    logic [LANE_ID_W-1:0] exp_lane [0:MAX_EXPECTED-1];
    logic [TOKEN_ID_W-1:0] exp_token [0:MAX_EXPECTED-1];
    logic [ISSUE_SEQ_W-1:0] exp_issue_seq [0:MAX_EXPECTED-1];
    logic exp_first [0:MAX_EXPECTED-1];
    logic exp_last [0:MAX_EXPECTED-1];
    logic exp_head_last [0:MAX_EXPECTED-1];

    integer expected_head;
    integer expected_tail;
    integer output_seen;
    integer mismatch_count;
    integer cycle_count;
    integer error_cases_seen;
    integer multi_beat_seen;
    integer single_destination_seen;
    integer duplicate_seen;
    integer metadata_error_seen;
    integer count_error_seen;
    integer range_error_seen;
    integer drain_recovery_seen;
    integer collect_flush_seen;
    integer emit_flush_seen;
    integer backpressure_seen;
    logic force_output_stall;
    logic [31:0] lfsr_q;
    logic held_valid_q;
    logic [TAG_W+CMD_SEQUENCE_W+GATE_CODE_W+LANE_ID_W+
           TOKEN_ID_W+ISSUE_SEQ_W+3-1:0] held_payload_q;

    gatestack_dctf_term_event_adapter #(
        .TOKENS(TOKENS),
        .EVENT_WAYS(EVENT_WAYS),
        .TAG_W(TAG_W),
        .GATE_CODE_W(GATE_CODE_W),
        .LANE_ID_W(LANE_ID_W),
        .TOKEN_ID_W(TOKEN_ID_W),
        .ISSUE_SEQ_W(ISSUE_SEQ_W),
        .CMD_SEQUENCE_W(CMD_SEQUENCE_W),
        .WAY_COUNT_W(WAY_COUNT_W)
    ) dut (
        .clk_core,
        .rst_core,
        .flush,
        .clear_error,
        .term_valid,
        .term_ready,
        .term_tag,
        .term_gate_code,
        .term_lane_id,
        .term_destination_count,
        .term_issue_seq,
        .term_head_last,
        .event_valid,
        .event_ready,
        .event_gate_code,
        .event_lane_id,
        .event_token_valid,
        .event_token_ids,
        .event_count,
        .event_issue_seq,
        .event_term_first,
        .event_term_last,
        .event_head_last,
        .cmd_valid,
        .cmd_ready,
        .cmd_group_tag,
        .cmd_sequence,
        .cmd_gate_code,
        .cmd_lane_id,
        .cmd_destination_token,
        .cmd_term_issue_seq,
        .cmd_term_first,
        .cmd_term_last,
        .cmd_head_last,
        .protocol_error
    );

    always #5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            lfsr_q <= 32'h6d2b_79f5;
        end else begin
            cycle_count <= cycle_count + 1;
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
        end
    end

    always_comb begin
        if (rst_core || flush || force_output_stall)
            cmd_ready = 1'b0;
        else
            cmd_ready = lfsr_q[0] | lfsr_q[2] | lfsr_q[5];
    end

    task automatic drive_term(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_CODE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [7:0] destination_count_value,
        input logic [ISSUE_SEQ_W-1:0] issue_seq_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            term_tag = tag_value;
            term_gate_code = gate_value;
            term_lane_id = lane_value;
            term_destination_count = destination_count_value;
            term_issue_seq = issue_seq_value;
            term_head_last = head_last_value;
            term_valid = 1'b1;
            while (!term_ready)
                @(negedge clk_core);
            @(negedge clk_core);
            term_valid = 1'b0;
        end
    endtask

    task automatic drive_event(
        input logic [GATE_CODE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [EVENT_WAYS-1:0] valid_value,
        input logic [(EVENT_WAYS*TOKEN_ID_W)-1:0] ids_value,
        input logic [WAY_COUNT_W-1:0] count_value,
        input logic [ISSUE_SEQ_W-1:0] issue_seq_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value
    );
        begin
            @(negedge clk_core);
            event_gate_code = gate_value;
            event_lane_id = lane_value;
            event_token_valid = valid_value;
            event_token_ids = ids_value;
            event_count = count_value;
            event_issue_seq = issue_seq_value;
            event_term_first = first_value;
            event_term_last = last_value;
            event_head_last = head_last_value;
            event_valid = 1'b1;
            while (!event_ready)
                @(negedge clk_core);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic add_expected(
        input logic [TAG_W-1:0] tag_value,
        input logic [GATE_CODE_W-1:0] gate_value,
        input logic [LANE_ID_W-1:0] lane_value,
        input logic [TOKEN_ID_W-1:0] token_value,
        input logic [ISSUE_SEQ_W-1:0] issue_seq_value,
        input logic first_value,
        input logic last_value,
        input logic head_last_value
    );
        begin
            if (expected_tail >= MAX_EXPECTED)
                $fatal(1, "adapter scoreboard capacity exceeded");
            exp_tag[expected_tail] = tag_value;
            exp_gate[expected_tail] = gate_value;
            exp_lane[expected_tail] = lane_value;
            exp_token[expected_tail] = token_value;
            exp_issue_seq[expected_tail] = issue_seq_value;
            exp_first[expected_tail] = first_value;
            exp_last[expected_tail] = last_value;
            exp_head_last[expected_tail] = head_last_value;
            expected_tail = expected_tail + 1;
        end
    endtask

    task automatic wait_for_all_expected;
        begin
            while ((expected_head != expected_tail) || !term_ready)
                @(posedge clk_core);
            @(negedge clk_core);
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
        logic [TAG_W+CMD_SEQUENCE_W+GATE_CODE_W+LANE_ID_W+
               TOKEN_ID_W+ISSUE_SEQ_W+3-1:0] current_payload;
        if (!rst_core) begin
            current_payload = {cmd_group_tag, cmd_sequence, cmd_gate_code,
                               cmd_lane_id, cmd_destination_token,
                               cmd_term_issue_seq, cmd_term_first,
                               cmd_term_last, cmd_head_last};
            if (held_valid_q && !flush) begin
                if (!cmd_valid || current_payload !== held_payload_q) begin
                    $error("adapter output changed under backpressure");
                    mismatch_count = mismatch_count + 1;
                end
            end
            if (cmd_valid && !cmd_ready)
                backpressure_seen = 1;
            if (cmd_valid && cmd_ready) begin
                if (expected_head >= expected_tail) begin
                    $error("adapter emitted unexpected command seq=%0d",
                           cmd_sequence);
                    mismatch_count = mismatch_count + 1;
                end else begin
                    if (cmd_group_tag !== exp_tag[expected_head] ||
                        cmd_sequence !== CMD_SEQUENCE_W'(expected_head) ||
                        cmd_gate_code !== exp_gate[expected_head] ||
                        cmd_lane_id !== exp_lane[expected_head] ||
                        cmd_destination_token !== exp_token[expected_head] ||
                        cmd_term_issue_seq !== exp_issue_seq[expected_head] ||
                        cmd_term_first !== exp_first[expected_head] ||
                        cmd_term_last !== exp_last[expected_head] ||
                        cmd_head_last !== exp_head_last[expected_head]) begin
                        $error("adapter command mismatch slot=%0d seq=%0d token=%0d",
                               expected_head, cmd_sequence,
                               cmd_destination_token);
                        mismatch_count = mismatch_count + 1;
                    end
                    expected_head = expected_head + 1;
                end
                output_seen = output_seen + 1;
            end
            held_valid_q = cmd_valid && !cmd_ready && !flush;
            held_payload_q = current_payload;
        end
    end

    initial begin
        integer output_before_error;
        clk_core = 1'b0;
        rst_core = 1'b1;
        flush = 1'b0;
        clear_error = 1'b0;
        term_valid = 1'b0;
        term_tag = '0;
        term_gate_code = '0;
        term_lane_id = '0;
        term_destination_count = '0;
        term_issue_seq = '0;
        term_head_last = 1'b0;
        event_valid = 1'b0;
        event_gate_code = '0;
        event_lane_id = '0;
        event_token_valid = '0;
        event_token_ids = '0;
        event_count = '0;
        event_issue_seq = '0;
        event_term_first = 1'b0;
        event_term_last = 1'b0;
        event_head_last = 1'b0;
        force_output_stall = 1'b0;
        expected_head = 0;
        expected_tail = 0;
        output_seen = 0;
        mismatch_count = 0;
        cycle_count = 0;
        error_cases_seen = 0;
        multi_beat_seen = 0;
        single_destination_seen = 0;
        duplicate_seen = 0;
        metadata_error_seen = 0;
        count_error_seen = 0;
        range_error_seen = 0;
        drain_recovery_seen = 0;
        collect_flush_seen = 0;
        emit_flush_seen = 0;
        backpressure_seen = 0;
        lfsr_q = 32'h6d2b_79f5;
        held_valid_q = 1'b0;
        held_payload_q = '0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        add_expected(12'h121, 9'h12a, 3'd3, 4'd3, 5'd2,
                     1'b1, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd7, 5'd2,
                     1'b0, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd1, 5'd2,
                     1'b0, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd9, 5'd2,
                     1'b0, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd2, 5'd2,
                     1'b0, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd8, 5'd2,
                     1'b0, 1'b0, 1'b0);
        add_expected(12'h121, 9'h12a, 3'd3, 4'd5, 5'd2,
                     1'b0, 1'b1, 1'b0);
        drive_term(12'h121, 9'h12a, 3'd3, 8'd7, 5'd2, 1'b0);
        drive_event(9'h12a, 3'd3, 4'b1011, {4'd1, 4'd0, 4'd7, 4'd3},
                    3'd3, 5'd2, 1'b1, 1'b0, 1'b0);
        force_output_stall = 1'b1;
        drive_event(9'h12a, 3'd3, 4'b1111, {4'd5, 4'd8, 4'd2, 4'd9},
                    3'd4, 5'd2, 1'b0, 1'b1, 1'b0);
        repeat (2) begin
            @(posedge clk_core);
            if (!cmd_valid || cmd_ready)
                $fatal(1, "directed adapter backpressure was not applied");
            backpressure_seen = 1;
        end
        force_output_stall = 1'b0;
        multi_beat_seen = 1;
        wait_for_all_expected();

        add_expected(12'h222, 9'h055, 3'd1, 4'd4, 5'd3,
                     1'b1, 1'b1, 1'b1);
        drive_term(12'h222, 9'h055, 3'd1, 8'd1, 5'd3, 1'b1);
        drive_event(9'h055, 3'd1, 4'b0100, {4'd0, 4'd4, 4'd0, 4'd0},
                    3'd1, 5'd3, 1'b1, 1'b1, 1'b1);
        single_destination_seen = 1;
        wait_for_all_expected();

        drive_term(12'h301, 9'h101, 3'd2, 8'd2, 5'd4, 1'b0);
        drive_event(9'h101, 3'd2, 4'b0001, 16'h0001,
                    3'd1, 5'd4, 1'b1, 1'b0, 1'b0);
        pulse_flush();
        collect_flush_seen = 1;

        force_output_stall = 1'b1;
        drive_term(12'h302, 9'h102, 3'd2, 8'd3, 5'd5, 1'b0);
        drive_event(9'h102, 3'd2, 4'b0111, {4'd0, 4'd3, 4'd2, 4'd1},
                    3'd3, 5'd5, 1'b1, 1'b1, 1'b0);
        while (!cmd_valid)
            @(posedge clk_core);
        pulse_flush();
        force_output_stall = 1'b0;
        emit_flush_seen = 1;

        output_before_error = output_seen;
        drive_term(12'h401, 9'h110, 3'd2, 8'd3, 5'd6, 1'b0);
        drive_event(9'h110, 3'd2, 4'b0011, 16'h0021,
                    3'd2, 5'd6, 1'b1, 1'b0, 1'b0);
        drive_event(9'h110, 3'd2, 4'b0100, 16'h0200,
                    3'd1, 5'd6, 1'b0, 1'b1, 1'b0);
        duplicate_seen = 1;
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h402, 9'h111, 3'd2, 8'd1, 5'd7, 1'b1);
        drive_event(9'h111, 3'd3, 4'b0001, 16'h0004,
                    3'd1, 5'd7, 1'b1, 1'b1, 1'b1);
        metadata_error_seen = 1;
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h403, 9'h112, 3'd2, 8'd2, 5'd8, 1'b0);
        drive_event(9'h112, 3'd2, 4'b0011, 16'h0065,
                    3'd1, 5'd8, 1'b1, 1'b1, 1'b0);
        count_error_seen = 1;
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h404, 9'h113, 3'd2, 8'd1, 5'd9, 1'b0);
        drive_event(9'h113, 3'd2, 4'b0001, 16'h000a,
                    3'd1, 5'd9, 1'b1, 1'b1, 1'b0);
        range_error_seen = 1;
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h405, 9'h114, 3'd2, 8'd3, 5'd10, 1'b0);
        drive_event(9'h114, 3'd2, 4'b0011, 16'h0021,
                    3'd2, 5'd10, 1'b1, 1'b1, 1'b0);
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h406, 9'h115, 3'd2, 8'd2, 5'd11, 1'b0);
        drive_event(9'h115, 3'd2, 4'b0011, 16'h0033,
                    3'd2, 5'd11, 1'b1, 1'b1, 1'b0);
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h407, 9'h116, 3'd2, 8'd3, 5'd12, 1'b0);
        drive_event(9'h116, 3'd1, 4'b0001, 16'h0001,
                    3'd1, 5'd12, 1'b1, 1'b0, 1'b0);
        drive_event(9'h000, 3'd0, 4'b0011, 16'h0032,
                    3'd2, 5'd0, 1'b0, 1'b1, 1'b0);
        drain_recovery_seen = 1;
        error_cases_seen = error_cases_seen + 1;

        drive_term(12'h408, 9'h117, 3'd2, 8'd0, 5'd13, 1'b0);
        error_cases_seen = error_cases_seen + 1;

        add_expected(12'h501, 9'h120, 3'd4, 4'd6, 5'd14,
                     1'b1, 1'b1, 1'b0);
        drive_term(12'h501, 9'h120, 3'd4, 8'd1, 5'd14, 1'b0);
        drive_event(9'h120, 3'd4, 4'b1000, 16'h6000,
                    3'd1, 5'd14, 1'b1, 1'b1, 1'b0);
        wait_for_all_expected();
        repeat (8) @(posedge clk_core);

        if (output_seen != expected_tail ||
            output_before_error != 8 ||
            output_seen != 9 ||
            mismatch_count != 0 ||
            !protocol_error || error_cases_seen < 8 ||
            (multi_beat_seen == 0) || (single_destination_seen == 0) ||
            (duplicate_seen == 0) || (metadata_error_seen == 0) ||
            (count_error_seen == 0) || (range_error_seen == 0) ||
            (drain_recovery_seen == 0) || (collect_flush_seen == 0) ||
            (emit_flush_seen == 0) || (backpressure_seen == 0)) begin
            $fatal(1,
                "adapter coverage failure out=%0d expected=%0d mismatch=%0d error=%0d cases=%0d multi=%0d single=%0d dup=%0d meta=%0d count=%0d range=%0d drain=%0d flush={%0d,%0d} stall=%0d",
                output_seen, expected_tail, mismatch_count, protocol_error,
                error_cases_seen, multi_beat_seen, single_destination_seen,
                duplicate_seen, metadata_error_seen, count_error_seen,
                range_error_seen, drain_recovery_seen, collect_flush_seen,
                emit_flush_seen, backpressure_seen);
        end

        pulse_flush();
        if (!protocol_error)
            $fatal(1, "flush incorrectly cleared adapter sticky error");
        @(negedge clk_core);
        clear_error = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        clear_error = 1'b0;
        if (protocol_error)
            $fatal(1, "clear_error did not clear adapter sticky error");

        $display("PASS DCTF ADAPTER cycles=%0d commands=%0d error_cases=%0d multi_beat=%0d single=%0d duplicate=%0d metadata=%0d count=%0d range=%0d drain=%0d flush_collect=%0d flush_emit=%0d backpressure=%0d",
                 cycle_count, output_seen, error_cases_seen, multi_beat_seen,
                 single_destination_seen, duplicate_seen,
                 metadata_error_seen, count_error_seen, range_error_seen,
                 drain_recovery_seen, collect_flush_seen, emit_flush_seen,
                 backpressure_seen);
        $finish;
    end

    always @(posedge clk_core) begin
        if (!rst_core && cycle_count > 2000)
            $fatal(1, "adapter regression timeout");
    end
endmodule

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on BLKSEQ */
`default_nettype wire
