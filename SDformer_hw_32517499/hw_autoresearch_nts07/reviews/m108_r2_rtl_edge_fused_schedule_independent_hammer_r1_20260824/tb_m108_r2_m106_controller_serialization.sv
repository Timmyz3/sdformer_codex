`timescale 1ns/1ps
`default_nettype none

// Small commercial trace for the exact M108-r2 recurrence corner: descriptor 1
// closes while descriptor 0 is draining, and descriptor 1 is empty/PWP=0.
// Frozen M106 must release descriptor 1 only after descriptor 0's four service
// accepts and one subsequent in-order dispatch edge.
module tb_m108_r2_m106_controller_serialization;
    localparam int WIN_ROWS = 64;
    localparam int ROW_W = 6;
    localparam int BASE_W = 12;
    localparam int CONTEXT_W = 16;

    logic clk_core, rst_core;
    logic event_valid, event_ready;
    logic [3:0] event_source;
    logic [2:0] event_block;
    logic [ROW_W-1:0] event_row_offset;
    logic event_negate;
    logic [BASE_W-1:0] window_base_row;
    logic [CONTEXT_W-1:0] window_context;
    logic event_accept;
    logic window_close_valid, window_close_ready, window_close_accept;
    logic service_valid, service_ready, service_is_event;
    logic [3:0] service_source;
    logic [2:0] service_block;
    logic [1:0] service_load_beat;
    logic [ROW_W-1:0] service_row_offset;
    logic [BASE_W-1:0] service_destination_row;
    logic service_negate, service_last_for_key;
    logic [CONTEXT_W-1:0] service_context;
    logic service_accept;
    logic fill_bank, drain_bank;
    logic [1:0] bank_ready;
    logic protocol_error, busy;

    integer cycle_count;
    integer close_count;
    integer first_close_cycle;
    integer empty_close_cycle;
    integer empty_release_cycle;
    integer service_count;
    integer first_service_cycle;
    integer last_service_cycle;

    m106_bounded_bitmap_transpose_scheduler dut (.*);

    always #1.5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count = cycle_count + 1;
            if (protocol_error)
                $fatal(1, "protocol_error cycle=%0d", cycle_count);
            if (window_close_accept) begin
                close_count = close_count + 1;
                if (close_count == 1)
                    first_close_cycle = cycle_count;
                if (close_count == 2)
                    empty_close_cycle = cycle_count;
            end
            if (service_accept) begin
                service_count = service_count + 1;
                if (service_count == 1)
                    first_service_cycle = cycle_count;
                last_service_cycle = cycle_count;
                if (service_context != 16'h0101)
                    $fatal(1, "out-of-order context=%h cycle=%0d",
                           service_context, cycle_count);
            end
        end
    end

    // Sample after the sequential nonblocking updates from the preceding edge.
    always @(negedge clk_core) begin
        if (!rst_core && empty_close_cycle >= 0 && empty_release_cycle < 0
                && dut.bank_state_q[1] == 2'd0)
            empty_release_cycle = cycle_count;
    end

    task automatic send_event(
        input logic [3:0] source,
        input logic [2:0] block,
        input logic [ROW_W-1:0] row,
        input logic [BASE_W-1:0] base,
        input logic [CONTEXT_W-1:0] context_value
    );
        begin
            @(negedge clk_core);
            event_source = source;
            event_block = block;
            event_row_offset = row;
            event_negate = 1'b0;
            window_base_row = base;
            window_context = context_value;
            event_valid = 1'b1;
            do @(posedge clk_core); while (!event_accept);
            @(negedge clk_core);
            event_valid = 1'b0;
        end
    endtask

    task automatic send_close(
        input logic [BASE_W-1:0] base,
        input logic [CONTEXT_W-1:0] context_value
    );
        begin
            window_base_row = base;
            window_context = context_value;
            window_close_valid = 1'b1;
            do @(posedge clk_core); while (!window_close_accept);
            @(negedge clk_core);
            window_close_valid = 1'b0;
        end
    endtask

    initial begin
        #10000;
        $fatal(1, "watchdog close=%0d service=%0d release=%0d",
               close_count, service_count, empty_release_cycle);
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        event_valid = 1'b0;
        event_source = '0;
        event_block = '0;
        event_row_offset = '0;
        event_negate = 1'b0;
        window_base_row = '0;
        window_context = '0;
        window_close_valid = 1'b0;
        service_ready = 1'b1;
        cycle_count = 0;
        close_count = 0;
        first_close_cycle = -1;
        empty_close_cycle = -1;
        empty_release_cycle = -1;
        service_count = 0;
        first_service_cycle = -1;
        last_service_cycle = -1;

        repeat (3) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        // Descriptor 0: one key/row -> three load tokens plus one event.
        send_event(4'd0, 3'd0, 6'd0, 12'd0, 16'h0101);
        send_close(12'd0, 16'h0101);

        // Descriptor 1: empty and no modeled PWP tokens.  It closes on the
        // following edge while descriptor 0 is selected for drain.
        send_close(12'd64, 16'h0202);

        wait (empty_release_cycle >= 0);
        @(negedge clk_core);
        if (service_count != 4)
            $fatal(1, "expected four descriptor-0 service tokens, got %0d",
                   service_count);
        if (empty_close_cycle != first_close_cycle + 1)
            $fatal(1, "empty descriptor did not close on adjacent edge");
        if (first_service_cycle != empty_close_cycle + 1)
            $fatal(1, "first service timing drift first=%0d empty_close=%0d",
                   first_service_cycle, empty_close_cycle);
        if (last_service_cycle != first_service_cycle + 3)
            $fatal(1, "four-token contiguous drain timing drift");
        if (empty_release_cycle != last_service_cycle + 1)
            $fatal(1, "empty release lacks subsequent dispatch edge release=%0d last=%0d",
                   empty_release_cycle, last_service_cycle);
        if (empty_release_cycle - empty_close_cycle != 5)
            $fatal(1, "fill-only dispatch would predict 1, actual delta=%0d",
                   empty_release_cycle - empty_close_cycle);
        $display("PASS M108 r2 independent M106 serialization VCS first_close=%0d empty_close=%0d first_service=%0d last_service=%0d empty_release=%0d close_to_release=%0d prior_tokens=4 dispatch_edges=2 pwp_tokens=0 system_speedup=false headline=false physical=false",
                 first_close_cycle, empty_close_cycle, first_service_cycle,
                 last_service_cycle, empty_release_cycle,
                 empty_release_cycle - empty_close_cycle);
        $finish;
    end
endmodule

`default_nettype wire
