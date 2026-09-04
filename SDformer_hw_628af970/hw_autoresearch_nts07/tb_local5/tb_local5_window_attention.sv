`timescale 1ns/1ps
`default_nettype none

// Cycle-counting multi-dest Local5 window TB (score→MFEP→bridge→banklocal).
module tb_local5_window_attention;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM  = 4;
    localparam int N_DEST   = 4;
    localparam int MAX_DEST = 16;
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

    // Golden soft: for each dest, stencil scores/gates then MFEP multiset
    // Acc[d][o] += mult * gate * W[lane][o]
    logic signed [7:0] W [0:31][0:3];
    logic signed [31:0] golden_acc [0:MAX_DEST-1][0:3];

    // reuse comb golden via stencil leaf for gates
    logic [31:0] gq;
    logic [31:0] gk [0:4];
    logic [4:0] gv;
    logic [5*16-1:0] gs;
    logic [5*9-1:0] gg;
    local5_stencil_token u_g (
        .q_bits(gq), .k_bits(gk), .valid(gv),
        .score_q7(gs), .gate_q17(gg)
    );

    task automatic clear_golden;
        for (int d = 0; d < MAX_DEST; d++)
            for (int o = 0; o < OUT_DIM; o++)
                golden_acc[d][o] = 0;
    endtask

    task automatic accumulate_dest(
        input int did,
        input logic [31:0] q,
        input logic [31:0] k0,k1,k2,k3,k4,
        input logic [4:0] mask
    );
        int mult_map [0:31][0:511];
        for (int l = 0; l < 32; l++)
            for (int g = 0; g < 512; g++)
                mult_map[l][g] = 0;
        gq = q;
        gk[0]=k0; gk[1]=k1; gk[2]=k2; gk[3]=k3; gk[4]=k4;
        for (int i = 0; i < 5; i++) gv[i] = mask[i];
        #1;
        for (int i = 0; i < 5; i++) begin
            if (!mask[i] || gg[i*9 +: 9] == 0) continue;
            for (int l = 0; l < 32; l++) begin
                if (gk[i][l]) mult_map[l][gg[i*9 +: 9]] += 1;
            end
        end
        for (int l = 0; l < 32; l++)
            for (int g = 0; g < 512; g++)
                if (mult_map[l][g] > 0)
                    for (int o = 0; o < OUT_DIM; o++)
                        golden_acc[did][o] +=
                            mult_map[l][g] * g * int'(W[l][o]);
    endtask

    int errors;

    initial begin
        clk_core = 0;
        rst_core = 1;
        w_load_valid = 0;
        run_start = 0;
        dest_valid = 0;
        acc_read_valid = 0;
        prng = 32'hBEEF_0066;
        errors = 0;
        clear_golden();
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        // Load weights
        for (int l = 0; l < 32; l++) begin
            for (int o = 0; o < OUT_DIM; o++) begin
                prng = npr(prng);
                W[l][o] = signed'(prng[7:0]);
                @(posedge clk_core);
                w_load_valid = 1;
                w_load_lane = 5'(l);
                w_load_out = 2'(o);
                w_load_data = W[l][o];
                w_load_last = (l == 31 && o == OUT_DIM-1);
                while (!(w_load_valid && w_load_ready)) @(posedge clk_core);
            end
        end
        @(posedge clk_core);
        w_load_valid = 0;

        // Prepare dest vectors + golden
        begin
            logic [31:0] qs [0:N_DEST-1];
            logic [31:0] kself [0:N_DEST-1];
            logic [31:0] kn [0:N_DEST-1];
            logic [31:0] ks [0:N_DEST-1];
            logic [31:0] ke [0:N_DEST-1];
            logic [31:0] kw [0:N_DEST-1];
            logic [4:0] masks [0:N_DEST-1];
            for (int d = 0; d < N_DEST; d++) begin
                prng = npr(prng); qs[d] = prng;
                prng = npr(prng); kself[d] = prng;
                prng = npr(prng); kn[d] = prng;
                prng = npr(prng); ks[d] = prng;
                prng = npr(prng); ke[d] = prng;
                prng = npr(prng); kw[d] = prng;
                masks[d] = (d == 0) ? 5'b11111 : 5'b10111;
                masks[d][0] = 1'b1;
                accumulate_dest(d+1, qs[d], kself[d], kn[d], ks[d], ke[d], kw[d], masks[d]);
            end

            @(posedge clk_core);
            run_start = 1;
            @(posedge clk_core);
            run_start = 0;

            for (int d = 0; d < N_DEST; d++) begin
                while (!dest_ready) @(posedge clk_core);
                @(posedge clk_core);
                dest_valid = 1;
                dest_tag = 16'(d);
                dest_id = 8'(d + 1);
                dest_q = qs[d];
                dest_k_self = kself[d];
                dest_k_n = kn[d];
                dest_k_s = ks[d];
                dest_k_e = ke[d];
                dest_k_w = kw[d];
                dest_valid_mask = masks[d];
                dest_last_in_window = (d == N_DEST - 1);
                while (!(dest_valid && dest_ready)) @(posedge clk_core);
                @(posedge clk_core);
                dest_valid = 0;
            end

            // Wait done (no fork — Verilator LIFETIME friendly)
            begin
                int guard;
                guard = 0;
                while (!run_done) begin
                    @(posedge clk_core);
                    guard = guard + 1;
                    if (guard > 200000)
                        $fatal(1, "TIMEOUT waiting run_done cycles=%0d",
                               perf_cycle_count);
                end
            end
        end

        if (protocol_error) begin
            $error("protocol_error");
            errors++;
        end
        if (perf_dest_count !== 32'(N_DEST)) begin
            $error("dest count %0d", perf_dest_count);
            errors++;
        end
        // Check Acc
        for (int d = 1; d <= N_DEST; d++) begin
            for (int o = 0; o < OUT_DIM; o++) begin
                @(posedge clk_core);
                acc_read_valid = 1;
                acc_read_dest = 8'(d);
                acc_read_out = 2'(o);
                @(posedge clk_core);
                if (acc_data !== golden_acc[d][o]) begin
                    $error("acc mismatch d=%0d o=%0d got=%0d exp=%0d",
                           d, o, acc_data, golden_acc[d][o]);
                    errors++;
                end
            end
        end
        acc_read_valid = 0;

        if (errors != 0)
            $fatal(1, "FAIL errors=%0d", errors);

        $display("PASS tb_local5_window_attention mode=%0s dests=%0d cmds=%0d cycles=%0d",
                 USE_TARE ? "TARE" : "DIRECT",
                 perf_dest_count, perf_cmd_count, perf_cycle_count);
        // machine-readable cycle line for ledger scripts
        $display("CYCLES %0d CMDS %0d DESTS %0d",
                 perf_cycle_count, perf_cmd_count, perf_dest_count);
        $finish;
    end
endmodule

`default_nettype wire
