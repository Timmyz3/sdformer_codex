`timescale 1ns/1ps
`default_nettype none

// End-to-end Local5 score->gate->MFEP->cmd smoke with Python-aligned golden.
module tb_local5_score_gate_term_top;
    localparam int HEAD_DIM = 32;
    localparam int N_CAND   = 5;
    localparam int N_VEC    = 9;
`ifdef LOCAL5_DEST_W9
    localparam int DEST_W = 9;
`else
    localparam int DEST_W = 8;
`endif

    logic clk_core, rst_core;
    logic anchor_valid, anchor_ready;
    logic [15:0] anchor_tag;
    logic [DEST_W-1:0] anchor_dest_id;
    logic [31:0] anchor_q_bits, anchor_k_bits;
    logic [4:0] anchor_valid_mask;
    logic probe_valid, probe_ready;
    logic [2:0] probe_dir;
    logic [31:0] probe_k_bits;
    logic probe_last;
    logic cmd_valid, cmd_ready;
    logic [15:0] cmd_group_tag, cmd_sequence;
    logic [8:0] cmd_gate_code;
    logic [4:0] cmd_lane_id;
    logic [DEST_W-1:0] cmd_destination_token;
    logic [2:0] cmd_multiplicity;
    logic cmd_term_first, cmd_term_last, cmd_head_last;
    logic stencil_done_valid, stencil_done_ready;
    logic [15:0] stencil_done_tag;
    logic protocol_error;
    logic [31:0] perf_edges, perf_terms, perf_naive_products;
    logic [15:0] perf_tare_issues, perf_tare_zero;
    logic [15:0] perf_tare_sparse, perf_tare_dense;

`ifdef LOCAL5_DIRECT_BASELINE
    localparam bit USE_TARE = 1'b0;
`else
    localparam bit USE_TARE = 1'b1;
`endif

    local5_score_gate_term_top #(
        .DEST_W(DEST_W),
        .USE_TARE(USE_TARE)
    ) dut (.*);

    // Golden via stencil leaf + software multiset
    logic [31:0] g_q;
    logic [31:0] g_k [0:4];
    logic [4:0] g_valid;
    logic [5*16-1:0] g_score;
    logic [5*9-1:0] g_gate;
    local5_stencil_token u_golden (
        .q_bits(g_q), .k_bits(g_k), .valid(g_valid),
        .score_q7(g_score), .gate_q17(g_gate)
    );

    always #5 clk_core = ~clk_core;

    logic [31:0] prng;
    logic [7:0] backpressure_lfsr;
    function automatic logic [31:0] next_prng(input logic [31:0] s);
        logic [31:0] v; v = s;
        v = v ^ (v << 13); v = v ^ (v >> 17); v = v ^ (v << 5);
        return v;
    endfunction

    int errors;
    int cmds_seen;
    int golden_mult [0:31][0:511];
    int remaining_terms;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            backpressure_lfsr <= 8'hA7;
            cmd_ready <= 1'b0;
        end else begin
            backpressure_lfsr <= {
                backpressure_lfsr[6:0],
                backpressure_lfsr[7] ^ backpressure_lfsr[5] ^
                backpressure_lfsr[4] ^ backpressure_lfsr[3]
            };
            // Deterministic 75%-ready sink stresses command hold/retry.
            cmd_ready <= backpressure_lfsr[0] | backpressure_lfsr[3];
        end
    end

    task automatic build_mfep_golden;
        remaining_terms = 0;
        for (int l = 0; l < 32; l++)
            for (int g = 0; g < 512; g++)
                golden_mult[l][g] = 0;
        for (int d = 0; d < 5; d++) begin
            if (g_valid[d] && g_gate[d*9 +: 9] != 0) begin
                for (int l = 0; l < 32; l++) begin
                    if (g_k[d][l]) begin
                        if (golden_mult[l][g_gate[d*9 +: 9]] == 0)
                            remaining_terms++;
                        golden_mult[l][g_gate[d*9 +: 9]]++;
                    end
                end
            end
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        anchor_valid = 0;
        probe_valid = 0;
        stencil_done_ready = 0;
        prng = 32'hC0FF_EE66;
        errors = 0;
        cmds_seen = 0;
        repeat (5) @(posedge clk_core);
        rst_core = 0;

        for (int v = 0; v < N_VEC; v++) begin
            logic [4:0] vmask;
            logic [31:0] qv, ks;
            logic [31:0] kn [1:4];
            logic [DEST_W-1:0] expected_dest;
            int nprobe, p;

            prng = next_prng(prng); qv = prng;
            prng = next_prng(prng); ks = prng;
            for (int d = 1; d <= 4; d++) begin
                prng = next_prng(prng); kn[d] = prng;
            end
            if (v == 0) begin
                // ZERO: all valid targets equal the self anchor.
                for (int d = 1; d <= 4; d++) kn[d] = ks;
            end else if (v == 1) begin
                // SPARSE: directions 1..4 flip exactly 1..4 lanes.
                for (int d = 1; d <= 4; d++)
                    kn[d] = ks ^ (32'hffff_ffff >> (32 - d));
            end
            if (v == N_VEC - 1) begin
                vmask = 5'b00000;
            end else begin
                vmask = (v % 2 == 0) ? 5'b11111 : 5'b10111;
                vmask[0] = 1'b1;
            end
            expected_dest = (DEST_W == 9 && v == N_VEC - 2) ?
                            DEST_W'(449) : DEST_W'(v + 1);

            g_q = qv; g_k[0] = ks; g_valid[0] = vmask[0];
            for (int d = 1; d <= 4; d++) begin
                g_k[d] = kn[d]; g_valid[d] = vmask[d];
            end
            // allow comb settle
            #1;
            build_mfep_golden();

            @(negedge clk_core);
            anchor_valid = 1;
            anchor_tag = 16'(v);
            anchor_dest_id = expected_dest;
            anchor_q_bits = qv;
            anchor_k_bits = ks;
            anchor_valid_mask = vmask;
            while (!(anchor_valid && anchor_ready)) @(posedge clk_core);
            @(negedge clk_core); anchor_valid = 0;

            nprobe = 0;
            for (int d = 1; d <= 4; d++) if (vmask[d]) nprobe++;
            p = 0;
            for (int d = 1; d <= 4; d++) begin
                if (vmask[d]) begin
                    p++;
                    @(negedge clk_core);
                    probe_valid = 1;
                    probe_dir = 3'(d);
                    probe_k_bits = kn[d];
                    probe_last = (p == nprobe);
                    while (!(probe_valid && probe_ready)) @(posedge clk_core);
                    @(negedge clk_core); probe_valid = 0;
                end
            end

            // Drain cmds until stencil done
            while (!stencil_done_valid) begin
                @(posedge clk_core);
                if (cmd_valid && cmd_ready) begin
                    if (cmd_destination_token !== expected_dest) begin
                        $error("dest mismatch");
                        errors++;
                    end
                    if (cmd_group_tag !== 16'(v)) begin
                        $error("tag mismatch");
                        errors++;
                    end
                    if (golden_mult[cmd_lane_id][cmd_gate_code] != int'(cmd_multiplicity)) begin
                        $error("cmd mult lane=%0d gate=%0d got=%0d exp=%0d",
                               cmd_lane_id, cmd_gate_code, cmd_multiplicity,
                               golden_mult[cmd_lane_id][cmd_gate_code]);
                        errors++;
                    end
                    if (golden_mult[cmd_lane_id][cmd_gate_code] > 0) begin
                        golden_mult[cmd_lane_id][cmd_gate_code] = -1;
                        remaining_terms--;
                    end
                    cmds_seen++;
                end
            end
            if (remaining_terms != 0) begin
                $error("v=%0d remaining golden terms=%0d", v, remaining_terms);
                errors++;
            end
            if (protocol_error) begin
                $error("protocol_error v=%0d", v);
                errors++;
            end
            if (stencil_done_tag !== 16'(v)) begin
                $error("done tag mismatch v=%0d got=%0d", v, stencil_done_tag);
                errors++;
            end
            // Hold completion under backpressure for 1..3 cycles. The DUT
            // must keep its tag stable and must not expose anchor_ready.
            repeat ((v % 3) + 1) begin
                @(posedge clk_core);
                if (!stencil_done_valid || anchor_ready ||
                    stencil_done_tag !== 16'(v)) begin
                    $error("done stall violation v=%0d valid=%0b ready=%0b tag=%0d",
                           v, stencil_done_valid, anchor_ready, stencil_done_tag);
                    errors++;
                end
            end
            @(negedge clk_core);
            stencil_done_ready = 1;
            while (!(stencil_done_valid && stencil_done_ready))
                @(posedge clk_core);
            @(negedge clk_core);
            stencil_done_ready = 0;
        end

        if (USE_TARE) begin
            if (perf_tare_issues !=
                perf_tare_zero + perf_tare_sparse + perf_tare_dense) begin
                $fatal(1, "TARE分类计数不守恒 issues=%0d kinds=%0d/%0d/%0d",
                       perf_tare_issues, perf_tare_zero,
                       perf_tare_sparse, perf_tare_dense);
            end
            if (perf_tare_zero == 0 || perf_tare_sparse == 0 ||
                perf_tare_dense == 0) begin
                $fatal(1, "TARE ZERO/SPARSE/DENSE覆盖不完整");
            end
        end else if (
            perf_tare_issues != 0 || perf_tare_zero != 0 ||
            perf_tare_sparse != 0 || perf_tare_dense != 0
        ) begin
            $fatal(1, "Direct基线出现TARE计数");
        end
        if (errors != 0) $fatal(1, "FAIL errors=%0d cmds=%0d", errors, cmds_seen);
        $display("PASS tb_local5_score_gate_term_top mode=%0s cmds=%0d vectors=%0d tare=%0d kinds=%0d/%0d/%0d",
                 USE_TARE ? "TARE" : "DIRECT", cmds_seen, N_VEC,
                 perf_tare_issues, perf_tare_zero,
                 perf_tare_sparse, perf_tare_dense);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
