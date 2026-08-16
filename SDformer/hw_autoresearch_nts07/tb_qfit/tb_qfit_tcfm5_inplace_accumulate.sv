`timescale 1ns/1ps
`default_nettype none

module tb_qfit_tcfm5_inplace_accumulate #(
    parameter int ACC_BACKEND_KIND = 0
);
    localparam int HEIGHT = 3;
    localparam int WIDTH = 6;
    localparam int HEAD_DIM = 4;
    localparam int OUT_DIM = 2;

    logic clk_core, rst_core;
    logic weight_valid, weight_ready, weight_last;
    logic [1:0] weight_lane;
    logic weight_out;
    logic signed [7:0] weight_data;
    logic weight_context_release, weight_context_release_ready;
    logic run_start, run_accumulate, run_busy, run_done;
    logic term_valid, term_ready;
    logic term_source_plane;
    logic [1:0] term_source_y;
    logic [2:0] term_source_x;
    logic [1:0] term_lane;
    logic [8:0] term_gate;
    logic [4:0] term_destination_mask;
    logic [OUT_DIM*32-1:0] term_product;
    logic term_window_last, window_close, window_close_ready;
    logic read_valid, read_ready, read_plane;
    logic [1:0] read_y;
    logic [2:0] read_x;
    logic read_out, read_data_valid;
    logic signed [31:0] read_data;
    logic protocol_error;
    logic [31:0] perf_product_terms, perf_destination_updates;
    integer mode;

    qfit_tcfm5_projection_top #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(1),
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    task automatic load_context(input integer base0, input integer base1);
        begin
            for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                for (int out = 0; out < OUT_DIM; out = out + 1) begin
                    do @(negedge clk_core); while (!weight_ready);
                    weight_valid = 1'b1;
                    weight_lane = 2'(lane);
                    weight_out = 1'(out);
                    weight_data = 8'(
                        out == 0 ? base0 + lane : base1 - lane
                    );
                    weight_last = lane == HEAD_DIM - 1
                               && out == OUT_DIM - 1;
                    @(negedge clk_core);
                    weight_valid = 1'b0;
                    weight_last = 1'b0;
                end
            end
        end
    endtask

    task automatic start_run(input bit accumulate);
        begin
            @(negedge clk_core);
            run_accumulate = accumulate;
            run_start = 1'b1;
            @(negedge clk_core);
            run_start = 1'b0;
            run_accumulate = 1'b0;
        end
    endtask

    task automatic send_self_term;
        begin
            do @(negedge clk_core); while (!term_ready);
            term_valid = 1'b1;
            term_source_y = 2'd1;
            term_source_x = 3'd2;
            term_lane = 2'd0;
            term_gate = 9'd1;
            term_destination_mask = 5'b00001;
            term_window_last = 1'b1;
            @(negedge clk_core);
            term_valid = 1'b0;
            term_window_last = 1'b0;
        end
    endtask

    task automatic release_weights;
        begin
            do @(negedge clk_core); while (!weight_context_release_ready);
            weight_context_release = 1'b1;
            @(negedge clk_core);
            weight_context_release = 1'b0;
        end
    endtask

    task automatic check_result(input integer out, input integer expected);
        begin
            do @(negedge clk_core); while (!read_ready);
            read_valid = 1'b1;
            read_y = 2'd1;
            read_x = 3'd2;
            read_out = 1'(out);
            @(negedge clk_core);
            read_valid = 1'b0;
            while (!read_data_valid)
                @(negedge clk_core);
            if ($signed(read_data) != expected)
                $fatal(1, "in-place result out=%0d got=%0d exp=%0d",
                       out, read_data, expected);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        weight_valid = 1'b0;
        weight_lane = '0;
        weight_out = '0;
        weight_data = '0;
        weight_last = 1'b0;
        weight_context_release = 1'b0;
        run_start = 1'b0;
        run_accumulate = 1'b0;
        term_valid = 1'b0;
        term_source_plane = 1'b0;
        term_source_y = '0;
        term_source_x = '0;
        term_lane = '0;
        term_gate = '0;
        term_destination_mask = '0;
        term_window_last = 1'b0;
        window_close = 1'b0;
        read_valid = 1'b0;
        read_plane = 1'b0;
        read_y = '0;
        read_x = '0;
        read_out = '0;
        mode = 0;
        void'($value$plusargs("MODE=%d", mode));
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        load_context(3, 4);
        if (mode == 1) begin
            start_run(1'b1);
            repeat (3) @(negedge clk_core);
            if (!protocol_error || run_busy)
                $fatal(1, "uninitialized accumulate did not fail closed");
            $display("PASS TCFM5 uninitialized accumulate fail-closed");
            $finish;
        end

        start_run(1'b0);
        send_self_term();
        wait (run_done);
        release_weights();
        load_context(5, -2);
        start_run(1'b1);
        send_self_term();
        wait (run_done);
        check_result(0, 8);
        check_result(1, 2);
        if (protocol_error || perf_product_terms != 1
            || perf_destination_updates != 1)
            $fatal(1, "in-place accumulator ledger mismatch");
        $display("PASS TCFM5 in-place cross-context accumulation");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $display(
            "DEBUG state=%0d loaded=%0d acc_init=%0d run_done=%0d release_ready=%0d read_ready=%0d read_data_valid=%0d error=%0d",
            dut.state_q, dut.weights_loaded_q,
            dut.accumulator_initialized_q, run_done,
            weight_context_release_ready, read_ready,
            read_data_valid, protocol_error
        );
        $fatal(1, "TCFM5 in-place accumulation timeout");
    end
endmodule

`default_nettype wire
