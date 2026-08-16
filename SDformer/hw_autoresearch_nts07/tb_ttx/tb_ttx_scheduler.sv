`timescale 1ns/1ps
`default_nettype none

module tb_ttx_scheduler;
    logic clk;
    logic rst_n;
    logic start_frame;
    logic row_req_valid;
    logic row_req_ready;
    logic [1:0] row_stage;
    logic [2:0] row_block;
    logic [4:0] row_head;
    logic [9:0] row_window;
    logic [7:0] row_n_tokens;
    logic row_done;
    logic busy;
    logic done;
    logic [15:0] perf_rows_issued;

    integer accepted_rows;
    integer errors;
    logic completion_pending;

    ttx_descriptor_scheduler dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_frame(start_frame),
        .row_req_valid(row_req_valid),
        .row_req_ready(row_req_ready),
        .row_stage(row_stage),
        .row_block(row_block),
        .row_head(row_head),
        .row_window(row_window),
        .row_n_tokens(row_n_tokens),
        .row_done(row_done),
        .busy(busy),
        .done(done),
        .perf_rows_issued(perf_rows_issued)
    );

    always #5 clk = ~clk;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            row_done <= 1'b0;
            completion_pending <= 1'b0;
            accepted_rows <= 0;
        end else begin
            row_done <= completion_pending;
            completion_pending <= 1'b0;
            if (row_req_valid && row_req_ready) begin
                accepted_rows <= accepted_rows + 1;
                completion_pending <= 1'b1;
                if (row_n_tokens != 8'd162) begin
                    errors <= errors + 1;
                end
            end
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start_frame = 1'b0;
        row_req_ready = 1'b1;
        row_done = 1'b0;
        completion_pending = 1'b0;
        accepted_rows = 0;
        errors = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        start_frame = 1'b1;
        @(posedge clk);
        @(negedge clk);
        start_frame = 1'b0;

        wait (done);
        if (accepted_rows != 6720) begin
            $display("ERROR: scheduler rows got=%0d expected=6720", accepted_rows);
            errors = errors + 1;
        end
        if (perf_rows_issued != 16'd6720) begin
            $display("ERROR: scheduler perf count got=%0d expected=6720", perf_rows_issued);
            errors = errors + 1;
        end
        if (row_stage != 2'd3 || row_block != 3'd1 || row_head != 5'd23 || row_window != 10'd9) begin
            $display("ERROR: final descriptor mismatch S%0d B%0d H%0d W%0d", row_stage, row_block, row_head, row_window);
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("PASS: TTX 12-block descriptor scheduler issued 6720 rows");
        end else begin
            $display("FAIL: %0d scheduler error(s)", errors);
            $fatal(1);
        end
        $finish;
    end
endmodule

`default_nettype wire
