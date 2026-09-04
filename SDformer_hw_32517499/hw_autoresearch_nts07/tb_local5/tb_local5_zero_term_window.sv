`timescale 1ns/1ps
`default_nettype none

module tb_local5_zero_term_window;
    localparam int OUT_DIM = 4;
`ifdef LOCAL5_TARE_MODE
    localparam bit USE_TARE = 1'b1;
`else
    localparam bit USE_TARE = 1'b0;
`endif

    logic clk_core, rst_core;
    logic w_load_valid, w_load_ready, w_load_last;
    logic [4:0] w_load_lane;
    logic [1:0] w_load_out;
    logic signed [7:0] w_load_data;
    logic run_start, run_busy, run_done;
    logic dest_valid, dest_ready;
    logic [15:0] dest_tag;
    logic [7:0] dest_id;
    logic [31:0] dest_q, dest_k_self, dest_k_n, dest_k_s, dest_k_e, dest_k_w;
    logic [4:0] dest_valid_mask;
    logic dest_last_in_window;
    logic acc_read_valid, acc_read_ready, acc_data_valid;
    logic [7:0] acc_read_dest;
    logic [1:0] acc_read_out;
    logic signed [31:0] acc_data;
    logic protocol_error;
    logic [31:0] perf_dest_count, perf_cmd_count, perf_cycle_count;

    local5_window_attention_top #(
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(4),
        .EXPLODE_MULT(1'b0),
        .USE_TARE(USE_TARE)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    initial begin
        clk_core = 0;
        rst_core = 1;
        w_load_valid = 0;
        w_load_last = 0;
        run_start = 0;
        dest_valid = 0;
        acc_read_valid = 0;
        dest_tag = 16'h5a00;
        dest_id = 8'd1;
        dest_q = 32'hffff_ffff;
        dest_k_self = '0;
        dest_k_n = '0;
        dest_k_s = '0;
        dest_k_e = '0;
        dest_k_w = '0;
        dest_valid_mask = 5'b1_1111;
        dest_last_in_window = 1'b1;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        for (int lane = 0; lane < 32; lane++) begin
            for (int out = 0; out < OUT_DIM; out++) begin
                @(posedge clk_core);
                w_load_valid = 1;
                w_load_lane = 5'(lane);
                w_load_out = 2'(out);
                w_load_data = 8'(lane + out + 1);
                w_load_last = (lane == 31 && out == OUT_DIM - 1);
                while (!w_load_ready) @(posedge clk_core);
            end
        end
        @(posedge clk_core);
        w_load_valid = 0;
        w_load_last = 0;

        run_start = 1;
        @(posedge clk_core);
        run_start = 0;
        while (!dest_ready) @(posedge clk_core);
        dest_valid = 1;
        @(posedge clk_core);
        dest_valid = 0;

        begin
            int guard;
            guard = 0;
            while (!run_done) begin
                @(posedge clk_core);
                guard++;
                if (guard > 2000)
                    $fatal(1, "zero-term window failed to close");
            end
        end

        if (protocol_error)
            $fatal(1, "unexpected protocol error");
        if (perf_dest_count != 1 || perf_cmd_count != 0)
            $fatal(
                1, "bad counters dest=%0d cmd=%0d",
                perf_dest_count, perf_cmd_count
            );

        for (int out = 0; out < OUT_DIM; out++) begin
            acc_read_valid = 1;
            acc_read_dest = 8'd1;
            acc_read_out = 2'(out);
            #1;
            if (!acc_data_valid || acc_data != 0)
                $fatal(1, "zero-term accumulator changed out=%0d val=%0d", out, acc_data);
        end
        acc_read_valid = 0;

        $display(
            "PASS tb_local5_zero_term_window mode=%0s cycles=%0d",
            USE_TARE ? "TARE" : "DIRECT", perf_cycle_count
        );
        $finish;
    end
endmodule

`default_nettype wire
