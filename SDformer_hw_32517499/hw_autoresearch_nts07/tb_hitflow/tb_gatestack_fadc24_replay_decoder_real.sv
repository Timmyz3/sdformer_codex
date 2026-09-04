`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_fadc24_replay_decoder_real;
    localparam int SLOT_WORDS = 104;
    localparam int PAYLOAD_WORDS = 102;
    localparam int EXPECTED_TERMS = 61;
    localparam int EXPECTED_EVENTS = 814;
    localparam int EXPECTED_BITMAP_TERMS = 15;
    localparam logic [31:0] EXPECTED_TAG = 32'hfa00_0304;

    logic clk_core = 1'b0;
    logic rst_core = 1'b1;
    always #5 clk_core <= ~clk_core;

    logic start_valid, start_ready;
    logic word_valid, word_ready, word_last;
    logic [63:0] word_data;
    logic [6:0] word_index;
    logic descriptor_begin_valid, descriptor_begin_ready;
    logic [31:0] descriptor_begin_tag;
    logic [7:0] descriptor_begin_term_count;
    logic term_valid, term_ready, term_head_last;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [6:0] term_index;
    logic event_valid, event_ready;
    logic [8:0] event_gate_code;
    logic [4:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [31:0] event_token_ids;
    logic [2:0] event_count;
    logic event_term_first, event_term_last, event_head_last;
    logic done_valid, done_ready, done_error, protocol_error;
    logic [31:0] done_tag;
    logic [31:0] count_heads, count_terms, count_events;
    logic [31:0] count_bitmap_terms;
    logic [31:0] count_input_stall_cycles;
    logic [31:0] count_term_stall_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [63:0] payload_words [0:SLOT_WORDS-1];
    logic [8:0] expected_term_gate [0:EXPECTED_TERMS-1];
    logic [4:0] expected_term_lane [0:EXPECTED_TERMS-1];
    logic [7:0] expected_term_count [0:EXPECTED_TERMS-1];
    logic [8:0] expected_event_gate [0:EXPECTED_EVENTS-1];
    logic [4:0] expected_event_lane [0:EXPECTED_EVENTS-1];
    logic [7:0] expected_event_token [0:EXPECTED_EVENTS-1];

    logic [31:0] lfsr_q;
    integer observed_terms;
    integer observed_events;
    integer descriptor_begins;
    integer cycles;
    integer current_term_events;
    logic expect_error;
    logic no_backpressure;

`ifdef FADC24_STREAMING
    gatestack_fadc24_streaming_replay_decoder #(
`else
    gatestack_fadc24_replay_decoder #(
`endif
        .TOKENS(162),
        .LANES(32),
        .MAX_TERMS(128),
        .EVENT_WAYS(4),
        .SLOT_WORDS(SLOT_WORDS)
    ) dut (
        .clk_core,
        .rst_core,
        .start_valid,
        .start_ready,
        .word_valid,
        .word_ready,
        .word_data,
        .word_index,
        .word_last,
        .descriptor_begin_valid,
        .descriptor_begin_ready,
        .descriptor_begin_tag,
        .descriptor_begin_term_count,
        .term_valid,
        .term_ready,
        .term_gate_code,
        .term_lane_id,
        .term_destination_count,
        .term_index,
        .term_head_last,
        .event_valid,
        .event_ready,
        .event_gate_code,
        .event_lane_id,
        .event_token_valid,
        .event_token_ids,
        .event_count,
        .event_term_first,
        .event_term_last,
        .event_head_last,
        .done_valid,
        .done_ready,
        .done_tag,
        .done_error,
        .protocol_error,
        .count_heads,
        .count_terms,
        .count_events,
        .count_bitmap_terms,
        .count_input_stall_cycles,
        .count_term_stall_cycles,
        .count_output_stall_cycles
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= 32'h1ace_b00c;
            descriptor_begin_ready <= 1'b0;
            term_ready <= 1'b0;
            event_ready <= 1'b0;
            done_ready <= 1'b0;
            cycles <= 0;
        end else begin
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            descriptor_begin_ready <= no_backpressure ?
                                      1'b1 : lfsr_q[0] | lfsr_q[5];
            term_ready <= no_backpressure ? 1'b1 : lfsr_q[2] | lfsr_q[9];
            event_ready <= no_backpressure ? 1'b1 : lfsr_q[4] | lfsr_q[13];
            done_ready <= no_backpressure ? 1'b1 : lfsr_q[6] | lfsr_q[17];
            cycles <= cycles + 1;
        end
    end

    always @(posedge clk_core) begin
        if (rst_core) begin
            observed_terms <= 0;
            observed_events <= 0;
            descriptor_begins <= 0;
            current_term_events <= 0;
        end else begin
            if (descriptor_begin_valid && descriptor_begin_ready) begin
                if (descriptor_begin_tag !== EXPECTED_TAG ||
                    descriptor_begin_term_count !== 8'(EXPECTED_TERMS)) begin
                    $fatal(1, "descriptor begin错误 tag=%h terms=%0d",
                           descriptor_begin_tag, descriptor_begin_term_count);
                end
                descriptor_begins <= descriptor_begins + 1;
            end

            if (term_valid && term_ready) begin
                if (observed_terms >= EXPECTED_TERMS)
                    $fatal(1, "term输出过多");
                if (term_index !== 7'(observed_terms) ||
                    term_gate_code !== expected_term_gate[observed_terms] ||
                    term_lane_id !== expected_term_lane[observed_terms] ||
                    term_destination_count !== expected_term_count[observed_terms] ||
                    term_head_last !== (observed_terms == EXPECTED_TERMS-1)) begin
                    $fatal(1,
                           "term[%0d]错误 idx=%0d gate=%0d lane=%0d count=%0d last=%0b",
                           observed_terms, term_index, term_gate_code,
                           term_lane_id, term_destination_count, term_head_last);
                end
                observed_terms <= observed_terms + 1;
                current_term_events <= 0;
            end

            if (event_valid && event_ready) begin
                integer local_events;
                local_events = 0;
                for (integer way = 0; way < 4; way = way + 1) begin
                    if (event_token_valid[way]) begin
                        if (observed_events + local_events >= EXPECTED_EVENTS)
                            $fatal(1, "event输出过多");
                        if (event_gate_code !==
                                expected_event_gate[observed_events + local_events] ||
                            event_lane_id !==
                                expected_event_lane[observed_events + local_events] ||
                            event_token_ids[(way*8) +: 8] !==
                                expected_event_token[observed_events + local_events]) begin
                            $fatal(1,
                                   "event[%0d]错误 gate=%0d lane=%0d token=%0d",
                                   observed_events + local_events,
                                   event_gate_code, event_lane_id,
                                   event_token_ids[(way*8) +: 8]);
                        end
                        local_events = local_events + 1;
                    end
                end
                if (local_events != 32'(event_count))
                    $fatal(1, "event_count错误 got=%0d valid=%0d",
                           event_count, local_events);
                if (event_term_first !== (current_term_events == 0) ||
                    event_term_last !==
                        (current_term_events + local_events ==
                         32'(expected_term_count[observed_terms-1])) ||
                    event_head_last !==
                        (event_term_last && observed_terms == EXPECTED_TERMS)) begin
                    $fatal(1,
                           "event边界错误 first=%0b last=%0b head_last=%0b term_events=%0d",
                           event_term_first, event_term_last, event_head_last,
                           current_term_events);
                end
                observed_events <= observed_events + local_events;
                current_term_events <= current_term_events + local_events;
            end

            if (done_valid && done_ready) begin
                if (expect_error) begin
                    if (!done_error || !protocol_error)
                        $fatal(1, "错误向量未触发done/protocol error");
                    $display("PASS: FADC24错误注入被检测 cycles=%0d terms=%0d events=%0d",
                             cycles, observed_terms, observed_events);
                    $finish;
                end else begin
                    if (done_tag !== EXPECTED_TAG || done_error || protocol_error)
                        $fatal(1, "done错误 tag=%h error=%0b protocol=%0b",
                               done_tag, done_error, protocol_error);
                    if (descriptor_begins != 1 ||
                        observed_terms != EXPECTED_TERMS ||
                        observed_events != EXPECTED_EVENTS)
                        $fatal(1, "输出计数错误 begin=%0d terms=%0d events=%0d",
                               descriptor_begins, observed_terms, observed_events);
                    if (count_heads != 1 || count_terms != EXPECTED_TERMS ||
                        count_events != EXPECTED_EVENTS ||
                        count_bitmap_terms != EXPECTED_BITMAP_TERMS)
                        $fatal(1, "内部计数错误 heads=%0d terms=%0d events=%0d bitmaps=%0d",
                               count_heads, count_terms, count_events,
                               count_bitmap_terms);
                    if (!no_backpressure &&
                        (count_term_stall_cycles == 0 ||
                         count_output_stall_cycles == 0))
                        $fatal(1, "随机反压未覆盖term/event stall");
                    $display("PASS: FADC24真实S3/head4 leaf decode terms=%0d events=%0d bitmaps=%0d cycles=%0d term_stall=%0d event_stall=%0d",
                             observed_terms, observed_events, count_bitmap_terms,
                             cycles, count_term_stall_cycles,
                             count_output_stall_cycles);
                    $display("FADC24 INPUT STALL=%0d", count_input_stall_cycles);
                    $finish;
                end
            end

            if (cycles > 20000)
                $fatal(1, "FADC24 decoder超时");
        end
    end

    task automatic send_word(input integer index);
        begin
            if (!no_backpressure)
                repeat ((index * 7) % 3) @(posedge clk_core);
            @(negedge clk_core);
            word_valid = 1'b1;
            word_data = payload_words[index];
            word_index = index[6:0];
            word_last = index == PAYLOAD_WORDS-1;
            do @(posedge clk_core); while (!word_ready);
            @(negedge clk_core);
            word_valid = 1'b0;
            word_data = '0;
            word_index = '0;
            word_last = 1'b0;
        end
    endtask

    initial begin
        string vector_dir;
        string payload_file;
        if (!$value$plusargs("vector_dir=%s", vector_dir))
            vector_dir = "tb_hitflow/vectors/fadc24_real_sample0_s3_b0_h4";
        if (!$value$plusargs("payload_file=%s", payload_file))
            payload_file = "payload_words.memh";
        if (!$value$plusargs("expect_error=%d", expect_error))
            expect_error = 0;
        if (!$value$plusargs("no_backpressure=%d", no_backpressure))
            no_backpressure = 0;
        $readmemh({vector_dir, "/", payload_file}, payload_words);
        $readmemh({vector_dir, "/term_gate.memh"}, expected_term_gate);
        $readmemh({vector_dir, "/term_lane.memh"}, expected_term_lane);
        $readmemh({vector_dir, "/term_destination_count.memh"},
                  expected_term_count);
        $readmemh({vector_dir, "/event_gate.memh"}, expected_event_gate);
        $readmemh({vector_dir, "/event_lane.memh"}, expected_event_lane);
        $readmemh({vector_dir, "/event_token.memh"}, expected_event_token);

        start_valid = 1'b0;
        word_valid = 1'b0;
        word_data = '0;
        word_index = '0;
        word_last = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        repeat (2) @(posedge clk_core);
        @(negedge clk_core);
        start_valid = 1'b1;
        do @(posedge clk_core); while (!start_ready);
        @(negedge clk_core);
        start_valid = 1'b0;
        for (integer index = 0; index < PAYLOAD_WORDS; index = index + 1)
            send_word(index);
    end
endmodule

`default_nettype wire
