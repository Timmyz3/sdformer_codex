`timescale 1ns/1ps
`default_nettype none

// Multi-window line-buffer → stencil → score/MFEP → 3-bank proj cycle TB.
module tb_local5_linebuf_window;
    localparam int HEAD_DIM   = 32;
    localparam int ROW_TOKENS = 8;
    localparam int OUT_DIM    = 4;
    localparam int MAX_DEST   = 16;
    localparam int N_WINDOWS  = 3;
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
    logic row_push_valid, row_push_ready;
    logic [15:0] row_push_tag;
    logic [31:0] row_push_q [0:ROW_TOKENS-1];
    logic [31:0] row_push_k [0:ROW_TOKENS-1];
    logic [ROW_TOKENS-1:0] row_push_valid_mask;
    logic run_start, run_busy, run_done;
    logic acc_read_valid, acc_read_ready, acc_data_valid;
    logic [7:0] acc_read_dest;
    logic [1:0] acc_read_out;
    logic signed [31:0] acc_data;
    logic protocol_error;
    logic [31:0] perf_dest_count, perf_cmd_count, perf_cycle_count, perf_bank_conflict_count;

    local5_linebuf_window_top #(
        .HEAD_DIM(HEAD_DIM),
        .ROW_TOKENS(ROW_TOKENS),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .NUM_BANKS(3),
        .USE_TARE(USE_TARE)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    logic [31:0] prng;
    function automatic logic [31:0] npr(input logic [31:0] s);
        logic [31:0] v; v = s;
        v ^= (v << 13); v ^= (v >> 17); v ^= (v << 5);
        return v;
    endfunction

    int win_cycles [0:N_WINDOWS-1];
    int win_cmds [0:N_WINDOWS-1];
    int win_conflicts [0:N_WINDOWS-1];
    int errors;

    task automatic load_weights;
        for (int l = 0; l < 32; l++) begin
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
        end
        @(posedge clk_core);
        w_load_valid = 0;
    endtask

    task automatic push_row(input int tag, input int seed_off);
        @(posedge clk_core);
        row_push_valid = 1;
        row_push_tag = 16'(tag);
        row_push_valid_mask = {ROW_TOKENS{1'b1}};
        for (int i = 0; i < ROW_TOKENS; i++) begin
            prng = npr(prng);
            row_push_q[i] = prng ^ 32'(seed_off + i);
            prng = npr(prng);
            row_push_k[i] = prng ^ 32'(seed_off * 3 + i * 7);
        end
        while (!(row_push_valid && row_push_ready)) @(posedge clk_core);
        @(posedge clk_core);
        row_push_valid = 0;
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        w_load_valid = 0;
        row_push_valid = 0;
        run_start = 0;
        acc_read_valid = 0;
        prng = 32'hA11C_E026;
        errors = 0;
        for (int i = 0; i < N_WINDOWS; i++) begin
            win_cycles[i] = 0;
            win_cmds[i] = 0;
            win_conflicts[i] = 0;
        end
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        load_weights();

        for (int w = 0; w < N_WINDOWS; w++) begin
            // three rows for stencil
            push_row(100 + w * 3 + 0, w * 100 + 0);
            push_row(100 + w * 3 + 1, w * 100 + 1);
            push_row(100 + w * 3 + 2, w * 100 + 2);

            @(posedge clk_core);
            run_start = 1;
            @(posedge clk_core);
            run_start = 0;

            begin
                int guard;
                guard = 0;
                while (!run_done) begin
                    @(posedge clk_core);
                    guard = guard + 1;
                    if (guard > 500000)
                        $fatal(1, "TIMEOUT window %0d", w);
                end
            end

            if (protocol_error) begin
                $error("protocol_error window %0d", w);
                errors++;
            end
            if (perf_dest_count !== 32'(ROW_TOKENS)) begin
                $error("dest_count window %0d got %0d", w, perf_dest_count);
                errors++;
            end
            win_cycles[w] = int'(perf_cycle_count);
            win_cmds[w] = int'(perf_cmd_count);
            win_conflicts[w] = int'(perf_bank_conflict_count);
            $display("WINDOW %0d CYCLES %0d CMDS %0d DESTS %0d CONFLICTS %0d",
                     w, perf_cycle_count, perf_cmd_count, perf_dest_count,
                     perf_bank_conflict_count);
            // Next iteration pushes new rows then run_start (re-enters from FINISH)
            repeat (2) @(posedge clk_core);
        end

        if (errors) $fatal(1, "FAIL errors=%0d", errors);

        // summary stats
        begin
            int sum_c, min_c, max_c;
            sum_c = 0;
            min_c = win_cycles[0];
            max_c = win_cycles[0];
            for (int w = 0; w < N_WINDOWS; w++) begin
                sum_c += win_cycles[w];
                if (win_cycles[w] < min_c) min_c = win_cycles[w];
                if (win_cycles[w] > max_c) max_c = win_cycles[w];
            end
            $display("PASS tb_local5_linebuf_window mode=%0s windows=%0d mean_cycles=%0d min=%0d max=%0d",
                     USE_TARE ? "TARE" : "DIRECT",
                     N_WINDOWS, sum_c / N_WINDOWS, min_c, max_c);
            $display("SUMMARY mean_cycles=%0d min_cycles=%0d max_cycles=%0d row_tokens=%0d",
                     sum_c / N_WINDOWS, min_c, max_c, ROW_TOKENS);
        end
        $finish;
    end
endmodule

`default_nettype wire
