`timescale 1ns/1ps
`default_nettype none

// Compare TARE row-context neighbor scores vs direct axnor golden (Local5 mode).
module tb_local5_row_context_tare;
    localparam int N_VEC = 32;

    logic clk_core, rst_core;
    logic anchor_valid, anchor_ready;
    logic [15:0] anchor_tag;
    logic [7:0] anchor_dest_id;
    logic [31:0] anchor_q_bits, anchor_k_bits;
    logic [4:0] anchor_valid_mask;
    logic probe_valid, probe_ready;
    logic [2:0] probe_dir;
    logic [31:0] probe_k_bits;
    logic probe_last;
    logic edge_valid, edge_ready;
    logic [15:0] edge_tag;
    logic [7:0] edge_dest_id;
    logic [2:0] edge_dir;
    logic [31:0] edge_k_bits;
    logic [8:0] edge_gate_q17;
    logic signed [15:0] edge_score_q7;
    logic edge_last;
    logic row_done_valid, row_done_ready;
    logic [15:0] row_done_tag;
    logic [2:0] row_done_degree;
    logic protocol_error;
    logic [15:0] perf_probe_count, perf_tare_issue_count, perf_edge_emit_count;
    logic [15:0] perf_tare_zero_count, perf_tare_sparse_count, perf_tare_dense_count;

    local5_row_context_tare_engine dut (.*);

    // Golden scores use the same raw16+RNE contract as both Local5 direct and TARE.
    function automatic int lane_score(input logic qb, input logic kb);
        if (qb && kb) return 64;
        if (!qb && !kb) return 1;
        return 0;
    endfunction
    function automatic int axnor_raw16(input logic [31:0] q, input logic [31:0] k);
        int r; r = 0;
        for (int lane = 0; lane < 32; lane++) r += lane_score(q[lane], k[lane]);
        return r;
    endfunction
    function automatic int rne_div16(input int raw);
        int q, rem;
        q = raw / 16; rem = raw % 16;
        if (rem > 8 || (rem == 8 && ((q & 1) != 0))) q += 1;
        return q;
    endfunction

    function automatic int exp2_lut_ref(input int index);
        case (index)
            0:  return 256;
            1:  return 245;
            2:  return 234;
            3:  return 224;
            4:  return 215;
            5:  return 205;
            6:  return 196;
            7:  return 188;
            8:  return 181;
            9:  return 173;
            10: return 165;
            11: return 158;
            12: return 152;
            13: return 145;
            14: return 139;
            default: return 133;
        endcase
    endfunction

    always #5 clk_core = ~clk_core;

    logic [31:0] prng;
    function automatic logic [31:0] npr(input logic [31:0] s);
        logic [31:0] v; v = s;
        v ^= v << 13; v ^= v >> 17; v ^= v << 5;
        return v;
    endfunction

    int errors, edges_ok;

    initial begin
        clk_core = 0;
        rst_core = 1;
        anchor_valid = 0;
        probe_valid = 0;
        edge_ready = 1;
        row_done_ready = 1;
        prng = 32'h7A7E_4C0D;
        errors = 0;
        edges_ok = 0;
        repeat (4) @(posedge clk_core);
        rst_core = 0;

        for (int v = 0; v < N_VEC; v++) begin
            logic [4:0] vmask;
            logic [31:0] qv, ks;
            logic [31:0] kn [1:4];
            int nprobe, p, row_max, row_sum, den_shift, probe;
            int expected_score [0:4];
            int expected_gate [0:4];
            int exp_q8 [0:4];

            prng = npr(prng); qv = prng;
            prng = npr(prng); ks = prng;
            for (int d = 1; d <= 4; d++) begin
                prng = npr(prng); kn[d] = prng;
            end
            if ((v % 3) == 0) begin
                // Topology ZERO: every neighbor equals the self anchor.
                for (int d = 1; d <= 4; d++)
                    kn[d] = ks;
            end else if ((v % 3) == 1) begin
                // Topology SPARSE: directions 1..4 flip exactly 1..4 lanes.
                for (int d = 1; d <= 4; d++)
                    kn[d] = ks ^ (32'hffff_ffff >> (32 - d));
            end
            // Exhaust all five-candidate validity masks, including empty.
            vmask = 5'(v);

            expected_score[0] = rne_div16(axnor_raw16(qv, ks));
            for (int d = 1; d <= 4; d++) begin
                if (vmask[d])
                    expected_score[d] = rne_div16(axnor_raw16(qv, kn[d]));
                else
                    expected_score[d] = -256;
            end

            row_max = -256;
            for (int d = 0; d < 5; d++)
                if (vmask[d] && expected_score[d] > row_max)
                    row_max = expected_score[d];
            row_sum = 0;
            for (int d = 0; d < 5; d++) begin
                int delta, int_shift, frac_index;
                if (!vmask[d]) begin
                    exp_q8[d] = 0;
                end else begin
                    delta = row_max - expected_score[d];
                    int_shift = (delta >> 7) > 8 ? 8 : (delta >> 7);
                    frac_index = (((delta & 127) + 7) >> 3);
                    if (frac_index > 15) frac_index = 15;
                    exp_q8[d] = exp2_lut_ref(frac_index) >> int_shift;
                end
                row_sum += exp_q8[d];
            end
            probe = row_sum - 1;
            den_shift = 0;
            while (probe > 0) begin
                den_shift++;
                probe >>= 1;
            end
            for (int d = 0; d < 5; d++) begin
                int scaled, quotient, remainder, half;
                if (!vmask[d]) begin
                    expected_gate[d] = 0;
                end else begin
                    scaled = exp_q8[d] * 128;
                    quotient = scaled >> den_shift;
                    remainder = scaled - (quotient << den_shift);
                    half = 1 << (den_shift - 1);
                    expected_gate[d] = quotient;
                    if (remainder > half
                        || (remainder == half && (quotient & 1)))
                        expected_gate[d]++;
                    if (expected_gate[d] > 256)
                        expected_gate[d] = 256;
                end
            end

            @(negedge clk_core);
            anchor_valid = 1;
            anchor_tag = 16'(v);
            anchor_dest_id = 8'(v);
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

            begin
                int got; logic seen [0:4];
                got = 0;
                for (int i = 0; i < 5; i++) seen[i] = 0;
                while (got < $countones(vmask)) begin
                    @(posedge clk_core);
                    if (edge_valid && edge_ready) begin
                        if (edge_score_q7 !== 16'(expected_score[edge_dir])) begin
                            $error("score mismatch v=%0d dir=%0d got=%0d exp=%0d",
                                   v, edge_dir, edge_score_q7,
                                   expected_score[edge_dir]);
                            errors++;
                        end
                        if (edge_gate_q17 > 9'd256) begin
                            $error("gate overflow");
                            errors++;
                        end
                        if (edge_gate_q17 !== 9'(expected_gate[edge_dir])) begin
                            $error(
                                "gate mismatch v=%0d dir=%0d got=%0d exp=%0d",
                                v, edge_dir, edge_gate_q17,
                                expected_gate[edge_dir]
                            );
                            errors++;
                        end
                        seen[edge_dir] = 1;
                        got++;
                        edges_ok++;
                    end
                end
            end

            while (!row_done_valid) @(posedge clk_core);
            if (protocol_error) begin
                $error("protocol v=%0d", v);
                errors++;
            end
            @(posedge clk_core);
        end

        if (perf_tare_issue_count != edges_ok)
            $fatal(1, "empty-mask或分类issue不守恒 issues=%0d edges=%0d",
                   perf_tare_issue_count, edges_ok);
        if (errors) $fatal(1, "FAIL errors=%0d", errors);
        $display("PASS tb_local5_row_context_tare edges=%0d tare_issues=%0d zero=%0d sparse=%0d dense=%0d",
                 edges_ok, perf_tare_issue_count, perf_tare_zero_count,
                 perf_tare_sparse_count, perf_tare_dense_count);
        $finish;
    end

    initial begin
        #500000; $fatal(1, "TIMEOUT");
    end
endmodule

`default_nettype wire
