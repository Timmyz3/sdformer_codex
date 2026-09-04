`default_nettype none

module h67_motionxor_score_q7 #(
    parameter int HEAD_DIM = 32,
    parameter int SCORE_W  = 16,
    parameter int COUNT_W  = $clog2(HEAD_DIM + 1),
    parameter bit ENABLE_MOTION_XOR = 1'b1
)(
    input  logic [HEAD_DIM-1:0] q_bits,
    input  logic [HEAD_DIM-1:0] k_current_bits,
    input  logic [HEAD_DIM-1:0] k_peer_bits,
    output logic [COUNT_W-1:0]  overlap,
    output logic [COUNT_W-1:0]  same_zero,
    output logic [COUNT_W-1:0]  motion_xor,
    output logic signed [SCORE_W-1:0] score_q7
);
    // Frozen H67 deployment formula:
    // round_even(128 * (overlap + same_zero/64 + motion_xor/4) / 32)
    // Round-to-even is applied after adding the integer overlap/motion terms,
    // because motion parity changes the half-way tie decision.
    localparam int RAW_W = COUNT_W + 3;

    integer bit_idx;
    logic [COUNT_W-1:0] q_count;
    logic [COUNT_W-1:0] k_count;
    logic [COUNT_W-1:0] overlap_count;
    logic [COUNT_W-1:0] motion_count;
    logic [COUNT_W-1:0] same_zero_count;
    logic [COUNT_W-1:0] silence_integer;
    logic [3:0] silence_remainder;
    logic silence_increment;
    logic [RAW_W-1:0] score_integer;
    logic [RAW_W-1:0] score_unsigned;

    always_comb begin
        q_count = '0;
        k_count = '0;
        overlap_count = '0;
        motion_count = '0;
        for (bit_idx = 0; bit_idx < HEAD_DIM; bit_idx = bit_idx + 1) begin
            q_count = q_count + COUNT_W'(q_bits[bit_idx]);
            k_count = k_count + COUNT_W'(k_current_bits[bit_idx]);
            overlap_count = overlap_count
                          + {{(COUNT_W-1){1'b0}}, (q_bits[bit_idx] && k_current_bits[bit_idx])};
            if (ENABLE_MOTION_XOR) begin
                motion_count = motion_count
                             + {{(COUNT_W-1){1'b0}}, (k_current_bits[bit_idx] ^ k_peer_bits[bit_idx])};
            end
        end

        same_zero_count = COUNT_W'(HEAD_DIM) - q_count - k_count + overlap_count;
        silence_integer = same_zero_count >> 4;
        silence_remainder = same_zero_count[3:0];
        score_integer = (RAW_W'(overlap_count) << 2)
                      + RAW_W'(motion_count)
                      + RAW_W'(silence_integer);
        silence_increment = (silence_remainder > 4'd8)
                         || ((silence_remainder == 4'd8) && score_integer[0]);
        score_unsigned = score_integer + RAW_W'(silence_increment);
        overlap = overlap_count;
        same_zero = same_zero_count;
        motion_xor = motion_count;
        score_q7 = SCORE_W'(score_unsigned);
    end
endmodule

`default_nettype wire
