`timescale 1ns/1ps
`default_nettype none

module tb_local5_score_shiftmax_vectors;
    localparam int N_CAND = 5;

    logic [31:0] q_bits;
    logic [31:0] k_bits [0:N_CAND-1];
    logic [4:0] valid_mask;
    logic [N_CAND-1:0] valid;
    logic [5:0] overlap [0:N_CAND-1];
    logic [5:0] same_zero [0:N_CAND-1];
    logic [N_CAND*16-1:0] score_q7;
    logic [N_CAND*9-1:0]  gate_q17;

    generate
        for (genvar index = 0; index < N_CAND; index++) begin : g_score
            local5_axnor_score_q7 u_score (
                .q_bits(q_bits),
                .k_bits(k_bits[index]),
                .overlap(overlap[index]),
                .same_zero(same_zero[index]),
                .score_q7(score_q7[index*16 +: 16])
            );
            always_comb valid[index] = valid_mask[index];
        end
    endgenerate

    local5_shiftmax5_q17 u_shiftmax (
        .score_q7(score_q7),
        .valid(valid),
        .gate_q17(gate_q17)
    );

    integer fd;
    integer rc;
    integer vectors;
    integer errors;
    integer expected_vectors;
    string vector_path;
    logic [31:0] read_q;
    logic [31:0] read_k0, read_k1, read_k2, read_k3, read_k4;
    logic [4:0] read_valid_mask;
    logic [15:0] read_s0, read_s1, read_s2, read_s3, read_s4;
    logic [8:0] read_g0, read_g1, read_g2, read_g3, read_g4;
    logic [15:0] expected_score [0:N_CAND-1];
    logic [8:0] expected_gate [0:N_CAND-1];

    initial begin
        vectors = 0;
        errors = 0;
        if (!$value$plusargs("VECTORS=%s", vector_path))
            vector_path = "build_local5/parity/local5_score_shiftmax_vectors.txt";
        if (!$value$plusargs("EXPECTED=%d", expected_vectors))
            expected_vectors = 256;
        fd = $fopen(vector_path, "r");
        if (fd == 0)
            $fatal(1, "cannot open integer reference vectors");

        while (!$feof(fd)) begin
            rc = $fscanf(
                fd,
                "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
                read_q,
                read_k0, read_k1, read_k2, read_k3, read_k4,
                read_valid_mask,
                read_s0, read_s1, read_s2, read_s3, read_s4,
                read_g0, read_g1, read_g2, read_g3, read_g4
            );
            if (rc == 17) begin
                q_bits = read_q;
                k_bits[0] = read_k0;
                k_bits[1] = read_k1;
                k_bits[2] = read_k2;
                k_bits[3] = read_k3;
                k_bits[4] = read_k4;
                valid_mask = read_valid_mask;
                expected_score[0] = read_s0;
                expected_score[1] = read_s1;
                expected_score[2] = read_s2;
                expected_score[3] = read_s3;
                expected_score[4] = read_s4;
                expected_gate[0] = read_g0;
                expected_gate[1] = read_g1;
                expected_gate[2] = read_g2;
                expected_gate[3] = read_g3;
                expected_gate[4] = read_g4;
                #1;
                for (int index = 0; index < N_CAND; index++) begin
                    if (score_q7[index*16 +: 16] !== signed'(expected_score[index])) begin
                        $display(
                            "  q=%h k=%h overlap=%0d same_zero=%0d qc=%0d kc=%0d num=%0d",
                            q_bits, k_bits[index],
                            overlap[index], same_zero[index],
                            g_score[0].u_score.q_count,
                            g_score[0].u_score.k_count,
                            g_score[0].u_score.tx_num_q8
                        );
                        $error(
                            "score mismatch vector=%0d cand=%0d got=%0d exp=%0d",
                            vectors, index, score_q7[index*16 +: 16],
                            signed'(expected_score[index])
                        );
                        errors++;
                    end
                    if (gate_q17[index*9 +: 9] !== expected_gate[index]) begin
                        $error(
                            "gate mismatch vector=%0d cand=%0d got=%0d exp=%0d",
                            vectors, index, gate_q17[index*9 +: 9],
                            expected_gate[index]
                        );
                        errors++;
                    end
                end
                vectors++;
            end
        end
        $fclose(fd);

        if (vectors != expected_vectors)
            $fatal(1, "wrong vector count %0d expected=%0d",
                   vectors, expected_vectors);
        if (errors != 0)
            $fatal(1, "FAIL errors=%0d", errors);
        $display(
            "PASS tb_local5_score_shiftmax_vectors vectors=%0d", vectors
        );
        $finish;
    end
endmodule

`default_nettype wire
