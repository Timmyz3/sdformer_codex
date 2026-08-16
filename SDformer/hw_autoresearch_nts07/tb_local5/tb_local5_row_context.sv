`timescale 1ns/1ps
`default_nettype none

// Self-checking TB for local5_row_context_engine.
// Compares emitted gates against combinational stencil golden.
module tb_local5_row_context;
    localparam int HEAD_DIM = 32;
    localparam int N_CAND   = 5;
    localparam int N_VEC    = 24;

    logic clk_core;
    logic rst_core;
    logic anchor_valid, anchor_ready;
    logic [15:0] anchor_tag;
    logic [7:0]  anchor_dest_id;
    logic [31:0] anchor_q_bits;
    logic [31:0] anchor_k_bits;
    logic [4:0]  anchor_valid_mask;
    logic probe_valid, probe_ready;
    logic [2:0]  probe_dir;
    logic [31:0] probe_k_bits;
    logic probe_last;
    logic edge_valid, edge_ready;
    logic [15:0] edge_tag;
    logic [7:0]  edge_dest_id;
    logic [2:0]  edge_dir;
    logic [31:0] edge_k_bits;
    logic [8:0]  edge_gate_q17;
    logic signed [15:0] edge_score_q7;
    logic edge_last;
    logic row_done_valid, row_done_ready;
    logic [15:0] row_done_tag;
    logic [2:0]  row_done_degree;
    logic protocol_error;
    logic [15:0] perf_probe_count, perf_edge_emit_count;

    local5_row_context_engine dut (.*);

    // Golden stencil leaf
    logic [31:0] g_q;
    logic [31:0] g_k [0:4];
    logic [4:0] g_valid;
    logic [5*16-1:0] g_score;
    logic [5*9-1:0] g_gate;

    local5_stencil_token u_golden (
        .q_bits(g_q),
        .k_bits(g_k),
        .valid(g_valid),
        .score_q7(g_score),
        .gate_q17(g_gate)
    );

    always #5 clk_core = ~clk_core;

    logic [31:0] prng;
    function automatic logic [31:0] next_prng(input logic [31:0] s);
        logic [31:0] v;
        v = s;
        v = v ^ (v << 13);
        v = v ^ (v >> 17);
        v = v ^ (v << 5);
        return v;
    endfunction

    int edges_checked;
    int rows_done;
    int errors;

    initial begin
        clk_core = 0;
        rst_core = 1;
        anchor_valid = 0;
        probe_valid = 0;
        edge_ready = 1;
        row_done_ready = 1;
        prng = 32'hA5A5_66D5;
        edges_checked = 0;
        rows_done = 0;
        errors = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        for (int v = 0; v < N_VEC; v = v + 1) begin
            logic [4:0] vmask;
            logic [31:0] qv, ks;
            logic [31:0] kn [1:4];
            int nprobe;
            int p;

            prng = next_prng(prng); qv = prng;
            prng = next_prng(prng); ks = prng;
            for (int d = 1; d <= 4; d = d + 1) begin
                prng = next_prng(prng);
                kn[d] = prng;
            end
            // interior-like: all valid; boundary-like every 4th
            if (v % 4 == 0) vmask = 5'b10101; // self,S,W
            else if (v % 4 == 1) vmask = 5'b11111;
            else if (v % 4 == 2) vmask = 5'b11011; // no E
            else vmask = 5'b01111; // no self invalid? keep self
            vmask[0] = 1'b1; // always valid self for this TB

            // Program golden
            g_q = qv;
            g_k[0] = ks;
            g_valid[0] = vmask[0];
            for (int d = 1; d <= 4; d = d + 1) begin
                g_k[d] = kn[d];
                g_valid[d] = vmask[d];
            end
            #1;

            // ANCHOR_LOAD
            @(posedge clk_core);
            anchor_valid = 1;
            anchor_tag = 16'(v);
            anchor_dest_id = 8'(v);
            anchor_q_bits = qv;
            anchor_k_bits = ks;
            anchor_valid_mask = vmask;
            while (!(anchor_valid && anchor_ready)) @(posedge clk_core);
            @(posedge clk_core);
            anchor_valid = 0;

            // PROBEs
            nprobe = 0;
            for (int d = 1; d <= 4; d = d + 1) if (vmask[d]) nprobe++;
            p = 0;
            for (int d = 1; d <= 4; d = d + 1) begin
                if (!vmask[d]) continue;
                p++;
                @(posedge clk_core);
                probe_valid = 1;
                probe_dir = 3'(d);
                probe_k_bits = kn[d];
                probe_last = (p == nprobe);
                while (!(probe_valid && probe_ready)) @(posedge clk_core);
                @(posedge clk_core);
                probe_valid = 0;
            end

            // Collect edges
            begin
                int got;
                logic seen [0:4];
                got = 0;
                for (int i = 0; i < 5; i++) seen[i] = 0;
                while (got < $countones(vmask)) begin
                    @(posedge clk_core);
                    if (edge_valid && edge_ready) begin
                        if (edge_tag !== 16'(v) || edge_dest_id !== 8'(v)) begin
                            $error("tag/dest mismatch v=%0d", v);
                            errors++;
                        end
                        if (seen[edge_dir]) begin
                            $error("duplicate dir %0d", edge_dir);
                            errors++;
                        end
                        seen[edge_dir] = 1;
                        if (edge_gate_q17 !== g_gate[edge_dir*9 +: 9]) begin
                            $error("gate mismatch v=%0d dir=%0d got=%0d exp=%0d",
                                   v, edge_dir, edge_gate_q17,
                                   g_gate[edge_dir*9 +: 9]);
                            errors++;
                        end
                        if (edge_score_q7 !== g_score[edge_dir*16 +: 16]) begin
                            $error("score mismatch v=%0d dir=%0d got=%0d exp=%0d",
                                   v, edge_dir, edge_score_q7,
                                   g_score[edge_dir*16 +: 16]);
                            errors++;
                        end
                        if (edge_k_bits !== g_k[edge_dir]) begin
                            $error("k mismatch v=%0d dir=%0d", v, edge_dir);
                            errors++;
                        end
                        got++;
                        edges_checked++;
                    end
                    if (rows_done > v) break;
                end
            end

            // Wait row done
            while (!row_done_valid) @(posedge clk_core);
            if (row_done_tag !== 16'(v)) begin
                $error("row tag mismatch");
                errors++;
            end
            if (protocol_error) begin
                $error("protocol error on v=%0d", v);
                errors++;
            end
            rows_done++;
            @(posedge clk_core);
        end

        if (errors != 0) begin
            $fatal(1, "FAIL errors=%0d edges=%0d rows=%0d",
                   errors, edges_checked, rows_done);
        end
        $display("PASS tb_local5_row_context edges=%0d rows=%0d",
                 edges_checked, rows_done);
        $finish;
    end

    // timeout
    initial begin
        #500000;
        $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
