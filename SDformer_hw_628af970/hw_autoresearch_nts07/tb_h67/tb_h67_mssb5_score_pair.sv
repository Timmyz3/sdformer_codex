`timescale 1ns/1ps
`default_nettype none

module tb_h67_mssb5_score_pair;
    localparam int RANDOM_VECTORS = 20000;

    logic [63:0] q_pair;
    logic [63:0] k_pair;

    logic [5:0] base_o0;
    logic [5:0] base_z0;
    logic [5:0] base_o1;
    logic [5:0] base_z1;
    logic [5:0] base_m;
    logic signed [15:0] base_s0;
    logic signed [15:0] base_s1;

    logic [5:0] dut_o0;
    logic [5:0] dut_z0;
    logic [5:0] dut_o1;
    logic [5:0] dut_z1;
    logic [5:0] dut_m;
    logic signed [15:0] dut_s0;
    logic signed [15:0] dut_s1;

    logic [5:0] strong_o0;
    logic [5:0] strong_z0;
    logic [5:0] strong_o1;
    logic [5:0] strong_z1;
    logic [5:0] strong_m;
    logic signed [15:0] strong_s0;
    logic signed [15:0] strong_s1;

    logic [5:0] cse_o0;
    logic [5:0] cse_z0;
    logic [5:0] cse_o1;
    logic [5:0] cse_z1;
    logic [5:0] cse_m;
    logic signed [15:0] cse_s0;
    logic signed [15:0] cse_s1;

    integer vectors;
    integer errors;

    h67_direct_score_pair baseline (
        .q_pair(q_pair), .k_pair(k_pair),
        .overlap0(base_o0), .same_zero0(base_z0),
        .overlap1(base_o1), .same_zero1(base_z1),
        .motion(base_m), .score0_q7(base_s0), .score1_q7(base_s1)
    );

    h67_mssb5_score_pair dut (
        .q_pair(q_pair), .k_pair(k_pair),
        .overlap0(dut_o0), .same_zero0(dut_z0),
        .overlap1(dut_o1), .same_zero1(dut_z1),
        .motion(dut_m), .score0_q7(dut_s0), .score1_q7(dut_s1)
    );

    h67_ssr5_score_pair strong_baseline (
        .q_pair(q_pair), .k_pair(k_pair),
        .overlap0(strong_o0), .same_zero0(strong_z0),
        .overlap1(strong_o1), .same_zero1(strong_z1),
        .motion(strong_m), .score0_q7(strong_s0), .score1_q7(strong_s1)
    );

    h67_cse7_score_pair conventional_baseline (
        .q_pair(q_pair), .k_pair(k_pair),
        .overlap0(cse_o0), .same_zero0(cse_z0),
        .overlap1(cse_o1), .same_zero1(cse_z1),
        .motion(cse_m), .score0_q7(cse_s0), .score1_q7(cse_s1)
    );

    function automatic integer ref_score(
        input integer overlap,
        input integer same_zero,
        input integer motion
    );
        integer integer_part;
        integer remainder;
        integer increment;
        begin
            integer_part = 4 * overlap + motion + same_zero / 16;
            remainder = same_zero % 16;
            increment = (remainder > 8) || ((remainder == 8) && (integer_part & 1));
            ref_score = integer_part + increment;
        end
    endfunction

    task automatic check_vector(
        input logic [31:0] q0,
        input logic [31:0] k0,
        input logic [31:0] q1,
        input logic [31:0] k1
    );
        integer o0;
        integer z0;
        integer o1;
        integer z1;
        integer m;
        begin
            q_pair = {q1, q0};
            k_pair = {k1, k0};
            #1;
            o0 = 0;
            z0 = 0;
            o1 = 0;
            z1 = 0;
            m = 0;
            for (int lane = 0; lane < 32; lane = lane + 1) begin
                o0 = o0 + (q0[lane] & k0[lane]);
                z0 = z0 + !(q0[lane] | k0[lane]);
                o1 = o1 + (q1[lane] & k1[lane]);
                z1 = z1 + !(q1[lane] | k1[lane]);
                m = m + (k0[lane] ^ k1[lane]);
            end
            if (
                base_o0 !== o0[5:0] || base_z0 !== z0[5:0]
                || base_o1 !== o1[5:0] || base_z1 !== z1[5:0]
                || base_m !== m[5:0]
                || base_s0 !== ref_score(o0, z0, m)
                || base_s1 !== ref_score(o1, z1, m)
                || dut_o0 !== base_o0 || dut_z0 !== base_z0
                || dut_o1 !== base_o1 || dut_z1 !== base_z1
                || dut_m !== base_m || dut_s0 !== base_s0 || dut_s1 !== base_s1
                || strong_o0 !== base_o0 || strong_z0 !== base_z0
                || strong_o1 !== base_o1 || strong_z1 !== base_z1
                || strong_m !== base_m
                || strong_s0 !== base_s0 || strong_s1 !== base_s1
                || cse_o0 !== base_o0 || cse_z0 !== base_z0
                || cse_o1 !== base_o1 || cse_z1 !== base_z1
                || cse_m !== base_m || cse_s0 !== base_s0 || cse_s1 !== base_s1
            ) begin
                $display("ERROR vector=%0d q=%h k=%h base=%0d/%0d dut=%0d/%0d counts=%0d,%0d,%0d,%0d,%0d",
                         vectors, q_pair, k_pair, base_s0, base_s1, dut_s0, dut_s1,
                         dut_o0, dut_z0, dut_o1, dut_z1, dut_m);
                errors = errors + 1;
            end
            vectors = vectors + 1;
        end
    endtask

    initial begin
        q_pair = '0;
        k_pair = '0;
        vectors = 0;
        errors = 0;

        check_vector(32'h0, 32'h0, 32'h0, 32'h0);
        check_vector(32'hffff_ffff, 32'hffff_ffff,
                     32'hffff_ffff, 32'hffff_ffff);
        check_vector(32'haaaa_aaaa, 32'h5555_5555,
                     32'h5555_5555, 32'haaaa_aaaa);
        check_vector(32'hffff_0000, 32'h00ff_00ff,
                     32'h0f0f_f0f0, 32'h3333_cccc);

        for (int lane = 0; lane < 32; lane = lane + 1) begin
            for (int pattern = 0; pattern < 16; pattern = pattern + 1) begin
                logic [31:0] q0;
                logic [31:0] k0;
                logic [31:0] q1;
                logic [31:0] k1;
                q0 = '0;
                k0 = '0;
                q1 = '0;
                k1 = '0;
                q0[lane] = pattern[0];
                k0[lane] = pattern[1];
                q1[lane] = pattern[2];
                k1[lane] = pattern[3];
                check_vector(q0, k0, q1, k1);
            end
        end

        for (int index = 0; index < RANDOM_VECTORS; index = index + 1)
            check_vector($urandom, $urandom, $urandom, $urandom);

        if (errors != 0) begin
            $display("FAIL tb_h67_mssb5_score_pair vectors=%0d errors=%0d", vectors, errors);
            $fatal(1);
        end
        $display("PASS tb_h67_mssb5_score_pair vectors=%0d errors=0", vectors);
        $finish;
    end
endmodule

`default_nettype wire
