`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_raw41_replay_decoder;
    logic clk_core;
    logic rst_core;
    logic start_valid;
    logic start_ready;
    logic [31:0] start_tag;
    logic word_valid;
    logic word_ready;
    logic [63:0] word_data;
    logic [6:0] word_index;
    logic word_last;
    logic direct_valid;
    logic direct_ready;
    logic [8:0] direct_gate_code;
    logic [4:0] direct_lane_id;
    logic [7:0] direct_token_id;
    logic direct_head_last;
    logic done_valid;
    logic done_ready;
    logic [31:0] done_tag;
    logic done_error;
    logic protocol_error;
    logic [31:0] count_heads;
    logic [31:0] count_records;
    logic [31:0] count_kzero_records;
    logic [31:0] count_direct_events;
    logic [31:0] count_input_stall_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [6655:0] packed_bits;
    logic [63:0] stream_words [0:103];
    logic [8:0] expected_gate [0:4];
    logic [4:0] expected_lane [0:4];
    logic [7:0] expected_token [0:4];

    gatestack_raw41_replay_decoder dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic set_record(
        input int token,
        input logic [8:0] gate,
        input logic [31:0] k_bits
    );
        begin
            packed_bits[token*41 +: 32] = k_bits;
            packed_bits[token*41+32 +: 9] = gate;
        end
    endtask

    initial begin
        int word_pointer;
        int event_pointer;
        int cycles;
        int last_stalled_event;
        clk_core = 1'b0;
        rst_core = 1'b1;
        start_valid = 1'b0;
        start_tag = 32'hface_0041;
        word_valid = 1'b0;
        word_data = '0;
        word_index = '0;
        word_last = 1'b0;
        direct_ready = 1'b0;
        done_ready = 1'b0;
        packed_bits = '0;
        set_record(0, 9'd3, 32'h0000_0021);
        set_record(63, 9'd256, 32'h8000_0000);
        set_record(161, 9'd7, 32'h0000_0006);
        for (int index = 0; index < 104; index = index + 1) begin
            stream_words[index] = packed_bits[index*64 +: 64];
        end
        expected_gate[0] = 3; expected_lane[0] = 0; expected_token[0] = 0;
        expected_gate[1] = 3; expected_lane[1] = 5; expected_token[1] = 0;
        expected_gate[2] = 256; expected_lane[2] = 31; expected_token[2] = 63;
        expected_gate[3] = 7; expected_lane[3] = 1; expected_token[3] = 161;
        expected_gate[4] = 7; expected_lane[4] = 2; expected_token[4] = 161;

        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start_valid = 1'b1;
        do @(posedge clk_core); while (!start_ready);
        @(negedge clk_core);
        start_valid = 1'b0;

        word_pointer = 0;
        event_pointer = 0;
        cycles = 0;
        last_stalled_event = -1;
        while (!done_valid) begin
            @(negedge clk_core);
            word_valid = word_pointer < 104 && ((cycles % 7) != 2);
            word_data = stream_words[word_pointer];
            word_index = 7'(word_pointer);
            word_last = word_pointer == 103;
            if (direct_valid && last_stalled_event != event_pointer) begin
                direct_ready = 1'b0;
                last_stalled_event = event_pointer;
            end else begin
                direct_ready = 1'b1;
            end
            @(posedge clk_core);
            if (word_valid && word_ready) word_pointer = word_pointer + 1;
            if (direct_valid && direct_ready) begin
                if (event_pointer >= 5 ||
                    direct_gate_code != expected_gate[event_pointer] ||
                    direct_lane_id != expected_lane[event_pointer] ||
                    direct_token_id != expected_token[event_pointer] ||
                    direct_head_last != (event_pointer == 4)) begin
                    $fatal(1, "RAW direct event mismatch index=%0d", event_pointer);
                end
                event_pointer = event_pointer + 1;
            end
            cycles = cycles + 1;
            if (cycles > 10000) begin
                $display("DEBUG words=%0d events=%0d records=%0d bits=%0d pending=%h",
                         word_pointer, event_pointer, dut.records_consumed_q,
                         dut.reservoir_bits_q, dut.pending_k_q);
                $fatal(1, "RAW decoder timeout");
            end
        end

        if (word_pointer != 104 || event_pointer != 5 ||
            done_tag != 32'hface_0041 || done_error || protocol_error ||
            count_heads != 1 || count_records != 162 ||
            count_kzero_records != 159 || count_direct_events != 5 ||
            count_input_stall_cycles == 0 || count_output_stall_cycles == 0) begin
            $display("DEBUG done words=%0d events=%0d tag=%h err=%0b protocol=%0b heads=%0d records=%0d kzero=%0d direct=%0d in_stall=%0d out_stall=%0d bits=%0d",
                     word_pointer, event_pointer, done_tag, done_error,
                     protocol_error, count_heads, count_records,
                     count_kzero_records, count_direct_events,
                     count_input_stall_cycles, count_output_stall_cycles,
                     dut.reservoir_bits_q);
            $fatal(1, "RAW decoder counters/done mismatch");
        end
        @(negedge clk_core);
        word_valid = 1'b0;
        direct_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: RAW41 decoder records=%0d kzero=%0d events=%0d in_stall=%0d out_stall=%0d",
                 count_records, count_kzero_records, count_direct_events,
                 count_input_stall_cycles, count_output_stall_cycles);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "RAW41 decoder TB timeout");
    end

endmodule

`default_nettype wire
