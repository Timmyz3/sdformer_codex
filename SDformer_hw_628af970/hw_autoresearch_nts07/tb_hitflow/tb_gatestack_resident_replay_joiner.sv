`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_resident_replay_joiner;
    localparam int TOKENS = 162;
    localparam int LANES = 32;
    localparam int MAX_TERMS = 8;
    localparam int EVENT_WAYS = 4;

    logic clk_core;
    logic rst_core;
    logic start_valid;
    logic start_ready;
    logic [31:0] start_tag;
    logic [7:0] start_term_count;
    logic [12:0] start_event_count;
    logic descriptor_valid;
    logic descriptor_ready;
    logic [8:0] descriptor_gate_code;
    logic [4:0] descriptor_lane_id;
    logic [7:0] descriptor_destination_count;
    logic [2:0] descriptor_term_index;
    logic descriptor_last;
    logic word_valid;
    logic word_ready;
    logic [63:0] word_data;
    logic [6:0] word_index;
    logic word_last;
    logic term_valid;
    logic term_ready;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [2:0] term_index;
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
    logic [31:0] count_descriptor_stall_cycles;
    logic [31:0] count_input_stall_cycles;
    logic [31:0] count_term_stall_cycles;
    logic [31:0] count_output_stall_cycles;

    logic [8:0] expected_gate [0:2];
    logic [4:0] expected_lane [0:2];
    int expected_count [0:2];
    int expected_tokens [0:10];

    gatestack_resident_replay_joiner #(
        .TOKENS(TOKENS),
        .LANES(LANES),
        .MAX_TERMS(MAX_TERMS),
        .EVENT_WAYS(EVENT_WAYS)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic launch_head(
        input logic [31:0] tag,
        input logic [7:0] terms,
        input logic [12:0] events
    );
        begin
            @(negedge clk_core);
            start_tag = tag;
            start_term_count = terms;
            start_event_count = events;
            start_valid = 1'b1;
            do @(posedge clk_core); while (!start_ready);
            @(negedge clk_core);
            start_valid = 1'b0;
        end
    endtask

    task automatic drive_descriptors;
        begin
            for (int index = 0; index < 3; index = index + 1) begin
                if (index == 2) @(posedge clk_core);
                @(negedge clk_core);
                descriptor_gate_code = expected_gate[index];
                descriptor_lane_id = expected_lane[index];
                descriptor_destination_count = 8'(expected_count[index]);
                descriptor_term_index = 3'(index);
                descriptor_last = index == 2;
                descriptor_valid = 1'b1;
                do @(posedge clk_core); while (!descriptor_ready);
                @(negedge clk_core);
                descriptor_valid = 1'b0;
            end
        end
    endtask

    task automatic drive_token_words;
        logic [63:0] packed_word;
        int base;
        int valid_bytes;
        begin
            for (int index = 0; index < 2; index = index + 1) begin
                packed_word = '0;
                base = index * 8;
                valid_bytes = (11 - base >= 8) ? 8 : 11 - base;
                for (int byte_index = 0; byte_index < valid_bytes;
                     byte_index = byte_index + 1) begin
                    packed_word[(byte_index*8) +: 8] =
                        8'(expected_tokens[base + byte_index]);
                end
                @(negedge clk_core);
                word_data = packed_word;
                word_index = 7'(index);
                word_last = index == 1;
                word_valid = 1'b1;
                do @(posedge clk_core); while (!word_ready);
                @(negedge clk_core);
                word_valid = 1'b0;
                word_last = 1'b0;
            end
        end
    endtask

    task automatic check_terms;
        int received;
        int cycles;
        begin
            received = 0;
            cycles = 0;
            while (received < 3) begin
                @(negedge clk_core);
                term_ready = (cycles % 4) != 1;
                @(posedge clk_core);
                if (term_valid && term_ready) begin
                    if (term_gate_code != expected_gate[received] ||
                        term_lane_id != expected_lane[received] ||
                        term_destination_count != 8'(expected_count[received]) ||
                        term_index != 3'(received) ||
                        term_head_last != (received == 2)) begin
                        $fatal(1, "term mismatch index=%0d", received);
                    end
                    received = received + 1;
                end
                cycles = cycles + 1;
            end
            @(negedge clk_core);
            term_ready = 1'b0;
        end
    endtask

    task automatic check_events;
        int token_offset;
        int term_seen;
        int term_offset;
        int cycles;
        logic forced_stall;
        begin
            token_offset = 0;
            term_seen = 0;
            term_offset = 0;
            cycles = 0;
            forced_stall = 1'b0;
            while (token_offset < 11) begin
                @(negedge clk_core);
                if (!forced_stall && event_valid) begin
                    event_ready = 1'b0;
                    forced_stall = 1'b1;
                end else begin
                    event_ready = (cycles % 5) != 2;
                end
                @(posedge clk_core);
                if (event_valid && event_ready) begin
                    if (event_gate_code != expected_gate[term_seen] ||
                        event_lane_id != expected_lane[term_seen] ||
                        event_term_first != (term_offset == 0) ||
                        event_term_last !=
                            (term_offset + 32'(event_count) ==
                             expected_count[term_seen]) ||
                        event_head_last !=
                            (term_seen == 2 && event_term_last)) begin
                        $fatal(1, "event metadata mismatch term=%0d",
                               term_seen);
                    end
                    for (int way = 0; way < EVENT_WAYS; way = way + 1) begin
                        if (way < event_count) begin
                            if (!event_token_valid[way] ||
                                event_token_ids[(way*8) +: 8] !=
                                8'(expected_tokens[token_offset + way])) begin
                                $fatal(1, "token mismatch offset=%0d way=%0d",
                                       token_offset, way);
                            end
                        end else if (event_token_valid[way]) begin
                            $fatal(1, "unexpected token-valid way=%0d", way);
                        end
                    end
                    token_offset = token_offset + 32'(event_count);
                    term_offset = term_offset + 32'(event_count);
                    if (event_term_last) begin
                        term_seen = term_seen + 1;
                        term_offset = 0;
                    end
                end
                cycles = cycles + 1;
            end
            @(negedge clk_core);
            event_ready = 1'b0;
        end
    endtask

    task automatic accept_done(
        input logic [31:0] expected_tag
    );
        begin
            wait (done_valid);
            if (done_tag != expected_tag || done_error) begin
                $fatal(1, "done mismatch tag=%h error=%b", done_tag,
                       done_error);
            end
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        expected_gate[0] = 9'h012;
        expected_gate[1] = 9'h155;
        expected_gate[2] = 9'h0a3;
        expected_lane[0] = 5'd2;
        expected_lane[1] = 5'd17;
        expected_lane[2] = 5'd31;
        expected_count[0] = 5;
        expected_count[1] = 2;
        expected_count[2] = 4;
        expected_tokens[0] = 1;
        expected_tokens[1] = 4;
        expected_tokens[2] = 7;
        expected_tokens[3] = 9;
        expected_tokens[4] = 12;
        expected_tokens[5] = 3;
        expected_tokens[6] = 8;
        expected_tokens[7] = 0;
        expected_tokens[8] = 2;
        expected_tokens[9] = 5;
        expected_tokens[10] = 6;

        clk_core = 1'b0;
        rst_core = 1'b1;
        start_valid = 1'b0;
        start_tag = '0;
        start_term_count = '0;
        start_event_count = '0;
        descriptor_valid = 1'b0;
        descriptor_gate_code = '0;
        descriptor_lane_id = '0;
        descriptor_destination_count = '0;
        descriptor_term_index = '0;
        descriptor_last = 1'b0;
        word_valid = 1'b0;
        word_data = '0;
        word_index = '0;
        word_last = 1'b0;
        term_ready = 1'b0;
        event_ready = 1'b0;
        done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        launch_head(32'hbd80_0001, 3, 11);
        fork
            drive_descriptors();
            drive_token_words();
            check_terms();
            check_events();
            accept_done(32'hbd80_0001);
        join

        launch_head(32'hbd80_0002, 0, 0);
        accept_done(32'hbd80_0002);

        if (protocol_error || count_heads != 2 || count_terms != 3 ||
            count_events != 11 || count_descriptor_stall_cycles == 0 ||
            count_input_stall_cycles == 0 || count_term_stall_cycles == 0 ||
            count_output_stall_cycles == 0) begin
            $fatal(1, "counter/error mismatch");
        end
        $display("PASS: resident replay joiner heads=%0d terms=%0d events=%0d desc_stall=%0d input_stall=%0d term_stall=%0d output_stall=%0d",
                 count_heads, count_terms, count_events,
                 count_descriptor_stall_cycles, count_input_stall_cycles,
                 count_term_stall_cycles, count_output_stall_cycles);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "resident replay joiner TB timeout");
    end

endmodule

`default_nettype wire
