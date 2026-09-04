`timescale 1ns/1ps
`default_nettype none

module tb_h67_motionxor_score #(
    parameter bit ENABLE_MOTION_XOR = 1'b1
);
    localparam int HEAD_DIM = 32;
    localparam int COUNT_W = $clog2(HEAD_DIM + 1);

    logic [HEAD_DIM-1:0] q_bits;
    logic [HEAD_DIM-1:0] k_current_bits;
    logic [HEAD_DIM-1:0] k_peer_bits;
    logic [COUNT_W-1:0] overlap;
    logic [COUNT_W-1:0] same_zero;
    logic [COUNT_W-1:0] motion_xor;
    logic signed [15:0] score_q7;

    integer errors;
    integer trial;
    integer expected_overlap;
    integer expected_same_zero;
    integer expected_motion;
    integer expected_silence;
    integer expected_score;

    h67_motionxor_score_q7 #(
        .ENABLE_MOTION_XOR(ENABLE_MOTION_XOR)
    ) dut (
        .q_bits(q_bits),
        .k_current_bits(k_current_bits),
        .k_peer_bits(k_peer_bits),
        .overlap(overlap),
        .same_zero(same_zero),
        .motion_xor(motion_xor),
        .score_q7(score_q7)
    );

    function automatic integer popcount32(input logic [31:0] value);
        integer idx;
        begin
            popcount32 = 0;
            for (idx = 0; idx < 32; idx = idx + 1) begin
                popcount32 = popcount32 + value[idx];
            end
        end
    endfunction

    function automatic integer round_even_silence(
        input integer count,
        input integer integer_base
    );
        integer quotient;
        integer remainder;
        begin
            quotient = count >> 4;
            remainder = count & 15;
            if ((remainder > 8) || ((remainder == 8) && ((integer_base + quotient) & 1))) begin
                quotient = quotient + 1;
            end
            round_even_silence = quotient;
        end
    endfunction

    task automatic check_score;
        begin
            #1;
            expected_overlap = popcount32(q_bits & k_current_bits);
            expected_same_zero = popcount32(~q_bits & ~k_current_bits);
            expected_motion = ENABLE_MOTION_XOR ? popcount32(k_current_bits ^ k_peer_bits) : 0;
            expected_silence = round_even_silence(expected_same_zero, 4 * expected_overlap + expected_motion);
            expected_score = 4 * expected_overlap + expected_motion + expected_silence;
            if (overlap !== expected_overlap[COUNT_W-1:0]
                || same_zero !== expected_same_zero[COUNT_W-1:0]
                || motion_xor !== expected_motion[COUNT_W-1:0]
                || score_q7 !== expected_score) begin
                $display("ERROR score q=%h k=%h peer=%h got=(%0d,%0d,%0d,%0d) expected=(%0d,%0d,%0d,%0d)",
                         q_bits, k_current_bits, k_peer_bits,
                         overlap, same_zero, motion_xor, score_q7,
                         expected_overlap, expected_same_zero, expected_motion, expected_score);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;

        q_bits = 32'h00000000;
        k_current_bits = 32'h00000000;
        k_peer_bits = 32'h00000000;
        check_score();

        k_peer_bits = 32'hffffffff;
        check_score();

        // same_zero=8: 0.5 must tie to even zero, not round half up to one.
        q_bits = 32'hffffff00;
        k_current_bits = 32'h00000000;
        k_peer_bits = 32'h00000000;
        check_score();

        // The same 0.5 tie rounds up when the motion term makes the integer odd.
        k_peer_bits = 32'h00000001;
        check_score();

        // same_zero=24: 1.5 must tie to even two.
        q_bits = 32'h000000ff;
        k_peer_bits = 32'h00000000;
        check_score();

        // 1 + 1.5 = 2.5 ties to the even integer two.
        k_peer_bits = 32'h00000001;
        check_score();

        q_bits = 32'hffffffff;
        k_current_bits = 32'hffffffff;
        k_peer_bits = 32'h00000000;
        check_score();

        for (trial = 0; trial < 1000; trial = trial + 1) begin
            q_bits = $urandom;
            k_current_bits = $urandom;
            k_peer_bits = $urandom;
            check_score();
        end

        if (errors == 0) begin
            $display("PASS: dyadic score matches frozen formula, motion=%0d", ENABLE_MOTION_XOR);
        end else begin
            $fatal(1, "FAIL: H67 Motion-XOR score errors=%0d", errors);
        end
        $finish;
    end
endmodule

`default_nettype wire
