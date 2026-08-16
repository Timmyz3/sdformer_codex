`timescale 1ns/1ps
`default_nettype none

// 16-destination direct window (no linebuf) for equal-lane cycle scaling.
module tb_local5_window16;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM  = 4;
    localparam int N_DEST   = 16;
    localparam int MAX_DEST = 32;
`ifdef LOCAL5_DIRECT_BASELINE
    localparam bit USE_TARE = 1'b0;
`else
    localparam bit USE_TARE = 1'b1;
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

    // Use original window top (1-bank path)
    local5_window_attention_top #(
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .EXPLODE_MULT(1'b0),
        .USE_TARE(USE_TARE)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    logic [31:0] prng;
    function automatic logic [31:0] npr(input logic [31:0] s);
        logic [31:0] v; v = s;
        v ^= v << 13; v ^= v >> 17; v ^= v << 5;
        return v;
    endfunction

    int errors;

    initial begin
        clk_core = 0;
        rst_core = 1;
        w_load_valid = 0;
        run_start = 0;
        dest_valid = 0;
        acc_read_valid = 0;
        prng = 32'hC0DEC0DE;
        errors = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        for (int l = 0; l < 32; l++)
            for (int o = 0; o < OUT_DIM; o++) begin
                prng = npr(prng);
                @(posedge clk_core);
                w_load_valid = 1;
                w_load_lane = 5'(l);
                w_load_out = 2'(o);
                w_load_data = signed'(prng[7:0]);
                w_load_last = (l == 31 && o == OUT_DIM-1);
                while (!(w_load_valid && w_load_ready)) @(posedge clk_core);
            end
        @(posedge clk_core); w_load_valid = 0;

        @(posedge clk_core); run_start = 1;
        @(posedge clk_core); run_start = 0;

        for (int d = 0; d < N_DEST; d++) begin
            while (!dest_ready) @(posedge clk_core);
            @(posedge clk_core);
            dest_valid = 1;
            dest_tag = 16'(d);
            dest_id = 8'(d);
            prng = npr(prng); dest_q = prng;
            prng = npr(prng); dest_k_self = prng;
            prng = npr(prng); dest_k_n = prng;
            prng = npr(prng); dest_k_s = prng;
            prng = npr(prng); dest_k_e = prng;
            prng = npr(prng); dest_k_w = prng;
            dest_valid_mask = (d == 0 || d == N_DEST-1) ? 5'b10111 : 5'b11111;
            dest_valid_mask[0] = 1'b1;
            dest_last_in_window = (d == N_DEST - 1);
            while (!(dest_valid && dest_ready)) @(posedge clk_core);
            @(posedge clk_core);
            dest_valid = 0;
        end

        begin
            int guard; guard = 0;
            while (!run_done) begin
                @(posedge clk_core);
                guard++;
                if (guard > 1000000) $fatal(1, "TIMEOUT");
            end
        end

        if (protocol_error) errors++;
        if (perf_dest_count !== 32'(N_DEST)) errors++;
        if (errors) $fatal(1, "FAIL");
        $display("PASS tb_local5_window16 mode=%0s dests=%0d cmds=%0d cycles=%0d",
                 USE_TARE ? "TARE" : "DIRECT",
                 perf_dest_count, perf_cmd_count, perf_cycle_count);
        $display("CYCLES %0d CMDS %0d DESTS %0d",
                 perf_cycle_count, perf_cmd_count, perf_dest_count);
        $finish;
    end
endmodule

`default_nettype wire
