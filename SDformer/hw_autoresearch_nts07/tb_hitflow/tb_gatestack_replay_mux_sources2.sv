`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_mux_sources2;
    localparam int SOURCES = 2;
    logic clk_core, rst_core;
    logic route_start_valid, route_start_ready;
    logic route_start_select, route_active, route_active_select;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [SOURCES-1:0] source_term_valid, source_term_ready;
    logic [(SOURCES*9)-1:0] source_term_gate_code;
    logic [(SOURCES*5)-1:0] source_term_lane_id;
    logic [(SOURCES*8)-1:0] source_term_destination_count;
    logic [SOURCES-1:0] source_term_head_last;
    logic [SOURCES-1:0] source_event_valid, source_event_ready;
    logic [(SOURCES*9)-1:0] source_event_gate_code;
    logic [(SOURCES*5)-1:0] source_event_lane_id;
    logic [(SOURCES*4)-1:0] source_event_token_valid;
    logic [(SOURCES*32)-1:0] source_event_token_ids;
    logic [(SOURCES*3)-1:0] source_event_count;
    logic [SOURCES-1:0] source_event_term_first;
    logic [SOURCES-1:0] source_event_term_last;
    logic [SOURCES-1:0] source_event_head_last;
    logic [SOURCES-1:0] source_done_valid, source_done_ready;
    logic [(SOURCES*32)-1:0] source_done_tag;
    logic [SOURCES-1:0] source_done_error;
    logic term_valid, term_ready;
    logic [8:0] term_gate_code;
    logic [4:0] term_lane_id;
    logic [7:0] term_destination_count;
    logic [12:0] term_issue_seq;
    logic term_head_last;
    logic event_valid, event_ready;
    logic [8:0] event_gate_code;
    logic [4:0] event_lane_id;
    logic [3:0] event_token_valid;
    logic [31:0] event_token_ids;
    logic [2:0] event_count;
    logic [12:0] event_issue_seq;
    logic event_term_first, event_term_last, event_head_last;
    /* verilator lint_on UNUSEDSIGNAL */
    logic done_valid, done_ready;
    logic [31:0] done_tag;
    logic done_error, protocol_error;
    logic [31:0] count_completed_heads;
    logic [(SOURCES*32)-1:0] count_route_heads;

    gatestack_replay_mux #(.SOURCES(SOURCES), .ROUTE_W(1)) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic run_empty_route(input integer route_value);
        begin
            @(negedge clk_core);
            route_start_select = 1'(route_value);
            route_start_valid = 1'b1;
            do @(posedge clk_core); while (!route_start_ready);
            @(negedge clk_core);
            route_start_valid = 1'b0;
            if (!route_active || route_active_select != 1'(route_value))
                $fatal(1, "sources2 route lock mismatch");
            source_done_tag[(route_value*32) +: 32] = 32'hA200_0000 +
                                                          32'(route_value);
            source_done_valid = '1;
            done_ready = 1'b0;
            @(posedge clk_core);
            if (!done_valid || source_done_ready != 0 ||
                done_tag != 32'hA200_0000 + 32'(route_value))
                $fatal(1, "sources2 done stall mismatch");
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            if (!source_done_ready[route_value] ||
                (source_done_ready & ~(2'b01 << route_value)) != 0)
                $fatal(1, "sources2 selected ready mismatch");
            @(negedge clk_core);
            source_done_valid = '0;
            done_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        route_start_valid = 1'b0;
        route_start_select = '0;
        source_term_valid = '0;
        source_term_gate_code = '0;
        source_term_lane_id = '0;
        source_term_destination_count = '0;
        source_term_head_last = '0;
        source_event_valid = '0;
        source_event_gate_code = '0;
        source_event_lane_id = '0;
        source_event_token_valid = '0;
        source_event_token_ids = '0;
        source_event_count = '0;
        source_event_term_first = '0;
        source_event_term_last = '0;
        source_event_head_last = '0;
        source_done_valid = '0;
        source_done_tag = '0;
        source_done_error = '0;
        term_ready = 1'b0;
        event_ready = 1'b0;
        done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;
        run_empty_route(0);
        run_empty_route(1);
        if (protocol_error || term_valid || event_valid || done_error ||
            count_completed_heads != 2 || count_route_heads[31:0] != 1 ||
            count_route_heads[63:32] != 1)
            $fatal(1, "sources2 counters mismatch");
        $display("PASS: replay mux SOURCES=2 parameterization");
        $finish;
    end

    initial begin
        repeat (500) @(posedge clk_core);
        $fatal(1, "sources2 replay mux timeout");
    end
endmodule

`default_nettype wire
