`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_destination_bitmap_assembler;
    logic clk_core;
    logic rst_core;
    logic term_valid;
    logic term_ready;
    logic [31:0] term_tag;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic event_valid;
    logic event_ready;
    logic [8:0] event_gate_code;
    logic [4:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [31:0] event_token_ids;
    logic [2:0] event_count;
    logic [12:0] event_issue_seq;
    logic event_term_first;
    logic event_term_last;
    logic event_head_last;
    logic bitmap_valid;
    logic bitmap_ready;
    logic [31:0] bitmap_tag;
    logic [8:0] bitmap_gate_code;
    logic [4:0] bitmap_lane_id;
    logic [12:0] bitmap_issue_seq;
    logic bitmap_head_last;
    logic [161:0] bitmap_destinations;
    logic protocol_error;
    logic [31:0] count_terms;
    logic [31:0] count_events;
    logic [31:0] count_bitmaps;
    logic [31:0] count_term_stall_cycles;
    logic [31:0] count_event_stall_cycles;
    logic [31:0] count_bitmap_stall_cycles;

    gatestack_destination_bitmap_assembler dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_term(
        input int index,
        input logic [7:0] destinations
    );
        begin
            @(negedge clk_core);
            term_gate_code = 9'(index + 11);
            term_tag = 32'hba00_0000 + 32'(index);
            term_lane_id = 5'(index + 3);
            term_destination_count = destinations;
            term_issue_seq = 13'(index);
            term_head_last = index == 1;
            term_valid = 1'b1;
            do @(posedge clk_core); while (!term_ready);
            @(negedge clk_core);
            term_valid = 1'b0;
        end
    endtask

    task automatic send_event(
        input int index,
        input logic [3:0] token_valid,
        input logic [31:0] token_ids,
        input logic [2:0] count,
        input logic first,
        input logic last
    );
        begin
            @(negedge clk_core);
            event_gate_code = 9'(index + 11);
            event_lane_id = 5'(index + 3);
            event_token_valid = token_valid;
            event_token_ids = token_ids;
            event_count = count;
            event_issue_seq = 13'(index);
            event_term_first = first;
            event_term_last = last;
            event_head_last = index == 1 && last;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_ready);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic check_bitmaps;
        logic [161:0] expected;
        begin
            for (int index = 0; index < 2; index = index + 1) begin
                wait (bitmap_valid);
                expected = '0;
                if (index == 0) begin
                    expected[1] = 1'b1;
                    expected[4] = 1'b1;
                    expected[7] = 1'b1;
                    expected[9] = 1'b1;
                    expected[12] = 1'b1;
                end else begin
                    expected[3] = 1'b1;
                    expected[8] = 1'b1;
                end
                if (bitmap_tag != 32'hba00_0000 + 32'(index) ||
                    bitmap_gate_code != 9'(index + 11) ||
                    bitmap_lane_id != 5'(index + 3) ||
                    bitmap_issue_seq != 13'(index) ||
                    bitmap_head_last != (index == 1) ||
                    bitmap_destinations != expected) begin
                    $fatal(1, "bitmap mismatch index=%0d", index);
                end
                @(negedge clk_core);
                bitmap_ready = 1'b0;
                @(posedge clk_core);
                @(negedge clk_core);
                bitmap_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                bitmap_ready = 1'b0;
            end
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
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
        bitmap_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        fork
            begin
                send_term(0, 5);
                send_term(1, 2);
            end
            begin
                send_event(0, 4'b1111, 32'h0907_0401, 4, 1'b1, 1'b0);
                send_event(0, 4'b0001, 32'h0000_000c, 1, 1'b0, 1'b1);
                send_event(1, 4'b0011, 32'h0000_0803, 2, 1'b1, 1'b1);
            end
            check_bitmaps();
        join

        if (protocol_error || count_terms != 2 || count_events != 7 ||
            count_bitmaps != 2 || count_bitmap_stall_cycles == 0 ||
            count_event_stall_cycles == 0) begin
            $fatal(1, "bitmap assembler counters/error mismatch");
        end
        $display("PASS: bitmap assembler terms=%0d events=%0d bitmaps=%0d term_stall=%0d event_stall=%0d bitmap_stall=%0d",
                 count_terms, count_events, count_bitmaps,
                 count_term_stall_cycles, count_event_stall_cycles,
                 count_bitmap_stall_cycles);
        $finish;
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "bitmap assembler TB timeout");
    end
endmodule

`default_nettype wire
