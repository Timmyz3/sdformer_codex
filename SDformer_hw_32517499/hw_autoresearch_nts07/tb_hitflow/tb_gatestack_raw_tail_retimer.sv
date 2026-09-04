`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_raw_tail_retimer;
    logic clk_core, rst_core;
    logic input_valid, input_ready;
    logic [8:0] input_gate_code;
    logic [4:0] input_lane_id;
    logic [7:0] input_token_id;
    logic input_done_valid, input_done_ready;
    logic [15:0] input_done_tag;
    logic input_done_error;
    logic output_valid, output_ready;
    logic [8:0] output_gate_code;
    logic [4:0] output_lane_id;
    logic [7:0] output_token_id;
    logic output_head_last;
    logic output_done_valid, output_done_ready;
    logic [15:0] output_done_tag;
    logic output_done_error;
    logic protocol_error;
    logic [31:0] count_inputs, count_outputs, count_empty_sessions;
    integer output_index;

    gatestack_raw_tail_retimer #(.TAG_W(16)) dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic send_event(
        input logic [8:0] gate_value,
        input logic [4:0] lane_value,
        input logic [7:0] token_value
    );
        begin
            @(negedge clk_core);
            input_gate_code = gate_value;
            input_lane_id = lane_value;
            input_token_id = token_value;
            input_valid = 1'b1;
            do @(posedge clk_core); while (!input_ready);
            @(negedge clk_core);
            input_valid = 1'b0;
        end
    endtask

    task automatic finish_session(input logic [15:0] tag_value);
        begin
            @(negedge clk_core);
            input_done_tag = tag_value;
            input_done_valid = 1'b1;
            do @(posedge clk_core); while (!input_done_ready);
            @(negedge clk_core);
            input_done_valid = 1'b0;
        end
    endtask

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            output_index <= 0;
        end else if (output_valid && output_ready) begin
            if (output_index == 0) begin
                if (output_gate_code != 9'd3 || output_lane_id != 5'd0 ||
                    output_token_id != 8'd0 || output_head_last)
                    $fatal(1, "first retimed event mismatch");
            end else if (output_index == 1) begin
                if (output_gate_code != 9'd7 || output_lane_id != 5'd2 ||
                    output_token_id != 8'd5 || !output_head_last)
                    $fatal(1, "last retimed event mismatch");
            end else begin
                $fatal(1, "unexpected retimed event");
            end
            output_index <= output_index + 1;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        input_valid = 1'b0;
        input_gate_code = '0;
        input_lane_id = '0;
        input_token_id = '0;
        input_done_valid = 1'b0;
        input_done_tag = '0;
        input_done_error = 1'b0;
        output_ready = 1'b0;
        output_done_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        send_event(9'd3, 5'd0, 8'd0);
        #1;
        if (output_valid) $fatal(1, "first event was not retained");
        fork
            send_event(9'd7, 5'd2, 8'd5);
            begin
                wait (output_valid);
                repeat (2) @(posedge clk_core);
                @(negedge clk_core);
                output_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                output_ready = 1'b0;
            end
        join
        fork
            finish_session(16'h7401);
            begin
                wait (output_valid && output_head_last);
                @(negedge clk_core);
                output_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                output_ready = 1'b0;
                wait (output_done_valid);
                if (output_done_tag != 16'h7401 || output_done_error)
                    $fatal(1, "retimed done mismatch");
                repeat (2) @(posedge clk_core);
                @(negedge clk_core);
                output_done_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                output_done_ready = 1'b0;
            end
        join

        // A session with no direct events passes done without inventing data.
        wait (!output_done_valid);
        @(posedge clk_core);
        fork
            finish_session(16'h7402);
            begin
                wait (output_done_valid);
                #1;
                if (output_valid || output_done_tag != 16'h7402) begin
                    $display("empty output_valid=%b tag=%h", output_valid,
                             output_done_tag);
                    $fatal(1, "empty retimed session mismatch");
                end
                @(negedge clk_core);
                output_done_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                output_done_ready = 1'b0;
            end
        join
        if (protocol_error || output_index != 2 || count_inputs != 2 ||
            count_outputs != 2 || count_empty_sessions != 1)
            $fatal(1, "retimer counters/error mismatch");
        $display("PASS: RAW tail retimer inputs=%0d outputs=%0d empty=%0d",
                 count_inputs, count_outputs, count_empty_sessions);
        $finish;
    end

    initial begin
        repeat (3000) @(posedge clk_core);
        $fatal(1, "RAW tail retimer TB timeout");
    end
endmodule

`default_nettype wire
