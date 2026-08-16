`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_ipd32w_replay_decoder;
    logic clk_core;
    logic rst_core;
    logic start_valid;
    logic start_ready;
    logic word_valid;
    logic word_ready;
    logic [63:0] word_data;
    logic [6:0] word_index;
    logic word_last;
    logic descriptor_begin_valid, descriptor_begin_ready;
    logic [31:0] descriptor_begin_tag;
    logic [7:0] descriptor_begin_term_count;
    logic term_valid;
    logic term_ready;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [6:0] term_index;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [8:0] event_gate_code;
    logic [4:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [31:0] event_token_ids;
    logic [2:0] event_count;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic done_error;
    logic protocol_error;
    logic [31:0] count_heads;
    logic [31:0] count_terms;
    logic [31:0] count_events;
    logic [31:0] count_input_stall_cycles;
    logic [31:0] count_term_stall_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [63:0] stream_words [0:7];
    logic [8:0] expected_gate [0:8];
    logic [4:0] expected_lane [0:8];
    logic [7:0] expected_token [0:8];
    logic expected_first [0:8];
    logic expected_last [0:8];
    integer descriptor_begin_count;

    gatestack_ipd32w_replay_decoder dut (.*);
    always #5 clk_core <= ~clk_core;
    always @(posedge clk_core) begin
        if (rst_core) begin
            descriptor_begin_count <= 0;
        end else if (descriptor_begin_valid && descriptor_begin_ready) begin
            if (descriptor_begin_tag != done_tag && done_tag != 0)
                $fatal(1, "descriptor begin tag mismatch");
            if ((descriptor_begin_tag == 32'h1234_5678 &&
                 descriptor_begin_term_count != 3) ||
                (descriptor_begin_tag == 32'h0000_ee00 &&
                 descriptor_begin_term_count != 0) ||
                (descriptor_begin_tag == 32'hdead_0001 &&
                 descriptor_begin_term_count != 1))
                $fatal(1, "descriptor begin count mismatch");
            descriptor_begin_count <= descriptor_begin_count + 1;
        end
    end

    function automatic logic [31:0] descriptor(
        input logic [8:0] gate,
        input logic [4:0] lane,
        input logic [7:0] count
    );
        descriptor = 32'(gate) | (32'(lane) << 9) | (32'(count) << 14);
    endfunction

    function automatic logic [63:0] header0(input logic [31:0] tag);
        header0 = (64'(tag) << 32) | (64'(1) << 20) |
                  (64'(1) << 16) | 64'h4753;
    endfunction

    function automatic logic [63:0] header1(
        input int payload_bits,
        input int terms,
        input int events,
        input int classes,
        input int active_tokens,
        input int token_offset
    );
        header1 = 64'(payload_bits) |
                  (64'(terms) << 13) |
                  (64'(events) << 21) |
                  (64'(classes) << 34) |
                  (64'(active_tokens) << 37) |
                  (64'(token_offset) << 45);
    endfunction

    task automatic start_decoder;
        begin
            @(negedge clk_core);
            start_valid = 1'b1;
            do @(posedge clk_core); while (!start_ready);
            @(negedge clk_core);
            start_valid = 1'b0;
        end
    endtask

    task automatic run_stream(
        input int word_count,
        input int expected_event_count,
        input logic [31:0] expected_tag,
        input logic expected_error
    );
        int word_pointer;
        int event_pointer;
        int term_pointer;
        int cycles;
        int batch_count;
        begin
            word_pointer = 0;
            event_pointer = 0;
            term_pointer = 0;
            cycles = 0;
            start_decoder();
            while (!done_valid) begin
                @(negedge clk_core);
                word_valid = word_pointer < word_count && ((cycles % 5) != 1);
                word_data = stream_words[word_pointer];
                word_index = 7'(word_pointer);
                word_last = word_pointer == word_count - 1;
                event_ready = (cycles % 3) != 1;
                term_ready = (cycles % 4) != 2;
                @(posedge clk_core);
                if (word_valid && word_ready) word_pointer = word_pointer + 1;
                if (term_valid && term_ready) begin
                    if (term_pointer >= 3 ||
                        term_gate_code != expected_gate[term_pointer == 0 ? 0 :
                            (term_pointer == 1 ? 3 : 4)] ||
                        term_lane_id != expected_lane[term_pointer == 0 ? 0 :
                            (term_pointer == 1 ? 3 : 4)] ||
                        term_destination_count !=
                            (term_pointer == 0 ? 3 : (term_pointer == 1 ? 1 : 5)) ||
                        term_index != 7'(term_pointer) ||
                        term_head_last != (term_pointer == 2)) begin
                        $fatal(1, "term command mismatch index=%0d", term_pointer);
                    end
                    term_pointer = term_pointer + 1;
                end
                if (event_valid && event_ready) begin
                    batch_count = 0;
                    for (int way = 0; way < 4; way = way + 1) begin
                        if (event_token_valid[way]) begin
                            if (event_pointer + batch_count >= expected_event_count ||
                                event_gate_code != expected_gate[event_pointer + batch_count] ||
                                event_lane_id != expected_lane[event_pointer + batch_count] ||
                                event_token_ids[(way*8) +: 8] !=
                                    expected_token[event_pointer + batch_count]) begin
                                $fatal(1, "event mismatch index=%0d way=%0d",
                                       event_pointer, way);
                            end
                            batch_count = batch_count + 1;
                        end
                    end
                    if (batch_count != 32'(event_count) ||
                        event_term_first != expected_first[event_pointer] ||
                        event_term_last !=
                            expected_last[event_pointer + batch_count - 1] ||
                        event_head_last !=
                            (event_pointer + batch_count == expected_event_count)) begin
                        $fatal(1, "event flags mismatch index=%0d", event_pointer);
                    end
                    event_pointer = event_pointer + batch_count;
                end
                cycles = cycles + 1;
                if (cycles > 2000) begin
                    $display("DEBUG state=%0d word_ptr=%0d event_ptr=%0d expected_word=%0d bytes=%0d recv=%0d emit=%0d term=%0d remain=%0d last_seen=%0b",
                             dut.state_q, word_pointer, event_pointer,
                             dut.expected_word_index_q, dut.token_bytes_q,
                             dut.tokens_received_q, dut.tokens_emitted_q,
                             dut.term_index_q, dut.current_term_remaining_q,
                             dut.input_last_seen_q);
                    $fatal(1, "decoder stream timeout");
                end
            end
            if (word_pointer != word_count || event_pointer != expected_event_count ||
                term_pointer != (expected_event_count == 0 ? 0 : 3) ||
                done_tag != expected_tag || done_error != expected_error) begin
                $fatal(1, "done mismatch words=%0d events=%0d error=%0b",
                       word_pointer, event_pointer, done_error);
            end
            @(negedge clk_core);
            word_valid = 1'b0;
            word_last = 1'b0;
            event_ready = 1'b0;
            term_ready = 1'b0;
            repeat (3) @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        start_valid = 1'b0;
        word_valid = 1'b0;
        word_data = '0;
        word_index = '0;
        word_last = 1'b0;
        event_ready = 1'b0;
        term_ready = 1'b0;
        descriptor_begin_ready = 1'b1;
        done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        // Three terms, nine events, and an odd descriptor padding slot.
        stream_words[0] = header0(32'h1234_5678);
        stream_words[1] = header1(328, 3, 9, 3, 9, 32);
        stream_words[2] = {descriptor(9'd256, 5'd31, 8'd1),
                           descriptor(9'd3, 5'd1, 8'd3)};
        stream_words[3] = {32'd0, descriptor(9'd7, 5'd4, 8'd5)};
        stream_words[4] = 64'h0804_0301_0209_0500;
        stream_words[5] = 64'h0000_0000_0000_000a;

        expected_gate[0] = 3; expected_lane[0] = 1; expected_token[0] = 0;
        expected_gate[1] = 3; expected_lane[1] = 1; expected_token[1] = 5;
        expected_gate[2] = 3; expected_lane[2] = 1; expected_token[2] = 9;
        expected_gate[3] = 256; expected_lane[3] = 31; expected_token[3] = 2;
        expected_gate[4] = 7; expected_lane[4] = 4; expected_token[4] = 1;
        expected_gate[5] = 7; expected_lane[5] = 4; expected_token[5] = 3;
        expected_gate[6] = 7; expected_lane[6] = 4; expected_token[6] = 4;
        expected_gate[7] = 7; expected_lane[7] = 4; expected_token[7] = 8;
        expected_gate[8] = 7; expected_lane[8] = 4; expected_token[8] = 10;
        expected_first[0] = 1; expected_first[1] = 0; expected_first[2] = 0;
        expected_first[3] = 1;
        expected_first[4] = 1; expected_first[5] = 0; expected_first[6] = 0;
        expected_first[7] = 0; expected_first[8] = 0;
        expected_last[0] = 0; expected_last[1] = 0; expected_last[2] = 1;
        expected_last[3] = 1;
        expected_last[4] = 0; expected_last[5] = 0; expected_last[6] = 0;
        expected_last[7] = 0; expected_last[8] = 1;
        run_stream(6, 9, 32'h1234_5678, 1'b0);

        // Empty CSR head completes after the second header word.
        stream_words[0] = header0(32'h0000_ee00);
        stream_words[1] = header1(128, 0, 0, 0, 0, 16);
        run_stream(2, 0, 32'h0000_ee00, 1'b0);

        // Header says one event while descriptor says two; drain to last word.
        stream_words[0] = header0(32'hdead_0001);
        stream_words[1] = header1(200, 1, 1, 1, 1, 24);
        stream_words[2] = {32'd0, descriptor(9'd1, 5'd0, 8'd2)};
        stream_words[3] = 64'h1;
        run_stream(4, 0, 32'hdead_0001, 1'b1);
        if (!protocol_error) $fatal(1, "protocol_error did not stick");

        if (count_heads != 2 || count_terms != 3 || count_events != 9 ||
            descriptor_begin_count != 3 ||
            count_input_stall_cycles == 0 || count_term_stall_cycles == 0 ||
            count_output_stall_cycles == 0) begin
            $fatal(1, "decoder counters mismatch");
        end
        $display("PASS: IPD32W decoder heads=%0d terms=%0d events=%0d in_stall=%0d term_stall=%0d out_stall=%0d",
                 count_heads, count_terms, count_events,
                 count_input_stall_cycles, count_term_stall_cycles,
                 count_output_stall_cycles);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "IPD32W decoder TB timeout");
    end

endmodule

`default_nettype wire
