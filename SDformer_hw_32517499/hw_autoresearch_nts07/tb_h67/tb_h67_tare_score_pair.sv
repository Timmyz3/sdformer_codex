`timescale 1ns/1ps
`default_nettype none

module tb_h67_tare_score_pair #(
    parameter int RESIDUAL_W = 16
);
    localparam int HEAD_DIM = 32;
    localparam int SCORE_W = 16;

    logic clk;
    logic rst;
    logic window_start;
    logic in_valid;
    logic in_enable;
    logic in_ready;
    logic [7:0] in_tag;
    logic [63:0] in_q_pair;
    logic [63:0] in_k_pair;
    logic out_valid;
    logic out_ready;
    logic [7:0] out_tag;
    logic signed [15:0] out_score0_q7;
    logic signed [15:0] out_score1_q7;
    logic [1:0] out_k_active;
    logic [5:0] out_update_count;
    logic out_dense_fallback;
    logic signed [12:0] out_delta_raw16;
    logic protocol_error;

    logic [5:0] unused_overlap0;
    logic [5:0] unused_same_zero0;
    logic [5:0] unused_motion0;
    logic [5:0] unused_overlap1;
    logic [5:0] unused_same_zero1;
    logic [5:0] unused_motion1;
    logic signed [15:0] ref_score0;
    logic signed [15:0] ref_score1;

    integer expected_score0;
    integer expected_score1;
    integer expected_count;
    integer expected_dense;
    integer expected_delta;
    integer received;
    integer errors;
    logic [31:0] lfsr;
    logic random_ready_enable;

    h67_tare_score_pair #(
        .RESIDUAL_W(RESIDUAL_W),
        .TAG_W(8)
    ) dut (
        .clk_core(clk),
        .rst_core(rst),
        .*
    );

    h67_motionxor_score_q7 ref0 (
        .q_bits(in_q_pair[31:0]),
        .k_current_bits(in_k_pair[31:0]),
        .k_peer_bits(in_k_pair[63:32]),
        .overlap(unused_overlap0),
        .same_zero(unused_same_zero0),
        .motion_xor(unused_motion0),
        .score_q7(ref_score0)
    );

    h67_motionxor_score_q7 ref1 (
        .q_bits(in_q_pair[63:32]),
        .k_current_bits(in_k_pair[63:32]),
        .k_peer_bits(in_k_pair[31:0]),
        .overlap(unused_overlap1),
        .same_zero(unused_same_zero1),
        .motion_xor(unused_motion1),
        .score_q7(ref_score1)
    );

    always #5 clk = ~clk;

    always @(negedge clk) begin
        if (rst) begin
            lfsr <= 32'h1ace_b00c;
            out_ready <= 1'b1;
        end else begin
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
            if (random_ready_enable)
                out_ready <= lfsr[0] | lfsr[3];
            else
                out_ready <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (!rst && out_valid && out_ready) begin
            if (out_score0_q7 !== expected_score0
                || out_score1_q7 !== expected_score1
                || out_update_count !== expected_count
                || out_dense_fallback !== expected_dense) begin
                $display("ERROR W=%0d tag=%0d score=%0d/%0d exp=%0d/%0d count=%0d exp=%0d dense=%0d exp=%0d",
                         RESIDUAL_W, out_tag, out_score0_q7, out_score1_q7,
                         expected_score0, expected_score1, out_update_count,
                         expected_count, out_dense_fallback, expected_dense);
                errors = errors + 1;
            end
            if (!expected_dense && out_delta_raw16 !== expected_delta) begin
                $display("ERROR W=%0d delta=%0d expected=%0d", RESIDUAL_W,
                         out_delta_raw16, expected_delta);
                errors = errors + 1;
            end
            received = received + 1;
        end
    end

    task automatic send_vector(
        input integer tag,
        input logic [31:0] q0,
        input logic [31:0] k0,
        input logic [31:0] q1,
        input logic [31:0] k1,
        input integer count,
        input integer delta
    );
        integer received_before;
        begin
            @(negedge clk);
            in_tag = tag[7:0];
            in_q_pair = {q1, q0};
            in_k_pair = {k1, k0};
            #1;
            expected_score0 = ref_score0;
            expected_score1 = ref_score1;
            expected_count = count;
            expected_dense = count > RESIDUAL_W;
            expected_delta = delta;
            received_before = received;
            in_valid = 1'b1;
            do @(posedge clk); while (!in_ready);
            @(negedge clk);
            in_valid = 1'b0;
            while (received == received_before)
                @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        window_start = 1'b0;
        in_valid = 1'b0;
        in_enable = 1'b1;
        in_tag = '0;
        in_q_pair = '0;
        in_k_pair = '0;
        out_ready = 1'b1;
        random_ready_enable = 1'b0;
        lfsr = 32'h1ace_b00c;
        expected_score0 = 0;
        expected_score1 = 0;
        expected_count = 0;
        expected_dense = 0;
        expected_delta = 0;
        received = 0;
        errors = 0;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        window_start = 1'b1;
        @(negedge clk);
        window_start = 1'b0;

        random_ready_enable = 1'b1;
        for (int count = 0; count <= 32; count = count + 1) begin
            logic [31:0] mask;
            mask = count == 32 ? 32'hffff_ffff : ((32'h1 << count) - 1'b1);
            send_vector(count, 32'h0000_0000, 32'hffff_ffff,
                        mask, 32'hffff_ffff, count, 64 * count);
        end

        if (RESIDUAL_W == 16) begin
            send_vector(100, 32'h0000_0000, 32'h0000_ffff,
                        32'h0000_ffff, 32'h0000_ffff, 16, 1024);
            send_vector(101, 32'h0000_ffff, 32'h0000_ffff,
                        32'h0000_0000, 32'h0000_ffff, 16, -1024);
        end

        random_ready_enable = 1'b0;
        repeat (5) @(posedge clk);
        if (protocol_error || errors != 0) begin
            $display("FAIL tb_h67_tare_score_pair W=%0d received=%0d errors=%0d protocol=%0d",
                     RESIDUAL_W, received, errors, protocol_error);
            $fatal(1);
        end
        $display("PASS tb_h67_tare_score_pair W=%0d received=%0d", RESIDUAL_W, received);
        $finish;
    end
endmodule

`default_nettype wire
