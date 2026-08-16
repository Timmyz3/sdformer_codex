`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_raw_issue_adapter;
    logic clk_core;
    logic rst_core;
    logic direct_valid;
    logic direct_ready;
    logic [8:0] direct_gate_code;
    logic [4:0] direct_lane_id;
    logic [7:0] direct_token_id;
    logic direct_head_last;
    logic term_valid;
    logic term_ready;
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
    logic [31:0] count_direct_inputs;
    logic [31:0] count_term_stall_cycles;
    logic [31:0] count_event_stall_cycles;

    gatestack_raw_issue_adapter dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_direct(
        input int index
    );
        begin
            @(negedge clk_core);
            direct_gate_code = 9'(index + 3);
            direct_lane_id = 5'(index * 7);
            direct_token_id = 8'(index * 11);
            direct_head_last = index == 2;
            direct_valid = 1'b1;
            do @(posedge clk_core); while (!direct_ready);
            @(negedge clk_core);
            direct_valid = 1'b0;
        end
    endtask

    task automatic check_outputs;
        int terms;
        int events;
        int cycles;
        begin
            terms = 0;
            events = 0;
            cycles = 0;
            while (events < 3) begin
                @(negedge clk_core);
                term_ready = (cycles % 3) != 0;
                event_ready = (cycles % 4) != 2;
                @(posedge clk_core);
                if (term_valid && term_ready) begin
                    if (term_gate_code != 9'(terms + 3) ||
                        term_lane_id != 5'(terms * 7) ||
                        term_destination_count != 1 ||
                        term_issue_seq != 13'(terms) ||
                        term_head_last != (terms == 2)) begin
                        $fatal(1, "RAW term mismatch %0d", terms);
                    end
                    terms = terms + 1;
                end
                if (event_valid && event_ready) begin
                    if (events >= terms ||
                        event_gate_code != 9'(events + 3) ||
                        event_lane_id != 5'(events * 7) ||
                        event_token_valid != 4'b0001 ||
                        event_token_ids[7:0] != 8'(events * 11) ||
                        event_token_ids[31:8] != 0 || event_count != 1 ||
                        event_issue_seq != 13'(events) ||
                        !event_term_first || !event_term_last ||
                        event_head_last != (events == 2)) begin
                        $fatal(1, "RAW event mismatch %0d", events);
                    end
                    events = events + 1;
                end
                cycles = cycles + 1;
            end
            if (terms != 3) $fatal(1, "RAW term count mismatch");
            @(negedge clk_core);
            term_ready = 1'b0;
            event_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        direct_valid = 1'b0;
        direct_gate_code = '0;
        direct_lane_id = '0;
        direct_token_id = '0;
        direct_head_last = 1'b0;
        term_ready = 1'b0;
        event_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        fork
            begin
                for (int index = 0; index < 3; index = index + 1) begin
                    send_direct(index);
                end
            end
            check_outputs();
        join
        if (count_direct_inputs != 3 || count_term_stall_cycles == 0 ||
            count_event_stall_cycles == 0) begin
            $fatal(1, "RAW adapter counters mismatch");
        end
        $display("PASS: RAW issue adapter inputs=%0d term_stall=%0d event_stall=%0d",
                 count_direct_inputs, count_term_stall_cycles,
                 count_event_stall_cycles);
        $finish;
    end

    initial begin
        repeat (1000) @(posedge clk_core);
        $fatal(1, "RAW issue adapter TB timeout");
    end
endmodule

`default_nettype wire
