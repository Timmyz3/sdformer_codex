`include "nts07_pkg.vh"

// Per-channel contribution for TX alpha-XNOR (NTS-07b: alpha0=0, beta=0, gamma=0)
// and SC signed product, then accumulate over HEAD_DIM.
module tx_sc_score_unit #(
    parameter integer HEAD_DIM = 32
)(
    input  wire [1:0] q_ternary,
    input  wire [1:0] k_ternary,
    output wire signed [`NTS07_SCORE_W-1:0] tx_contrib,
    output wire signed [`NTS07_SCORE_W-1:0] sc_contrib
);
    wire q_pos = (q_ternary == `TERN_POS);
    wire q_neg = (q_ternary == `TERN_NEG);
    wire q_act = q_pos | q_neg;
    wire k_pos = (k_ternary == `TERN_POS);
    wire k_neg = (k_ternary == `TERN_NEG);
    wire k_act = k_pos | k_neg;

    wire same_nonzero = q_act & k_act & (
        (q_pos & k_pos) | (q_neg & k_neg)
    );
    wire opposite = q_act & k_act & (
        (q_pos & k_neg) | (q_neg & k_pos)
    );

    // NTS-07b: only strong match (+1) contributes in TX
    assign tx_contrib = same_nonzero ? 6'sd1 : 6'sd0;

    // SC: signed product q*k in {-1,0,+1}
    assign sc_contrib = same_nonzero ? 6'sd1 :
                        opposite     ? -6'sd1 : 6'sd0;
endmodule


module tx_sc_score_tree #(
    parameter integer HEAD_DIM = 32,
    parameter integer SCORE_W = 6
)(
    input  wire [1:0] q_ternary [0:HEAD_DIM-1],
    input  wire [1:0] k_ternary [0:HEAD_DIM-1],
    output wire signed [SCORE_W-1:0] tx_score,
    output wire signed [SCORE_W-1:0] sc_score
);
    genvar i;
    wire signed [SCORE_W-1:0] tx_part [0:HEAD_DIM-1];
    wire signed [SCORE_W-1:0] sc_part [0:HEAD_DIM-1];

    generate
        for (i = 0; i < HEAD_DIM; i = i + 1) begin : gen_ch
            tx_sc_score_unit #(.HEAD_DIM(HEAD_DIM)) u_ch (
                .q_ternary(q_ternary[i]),
                .k_ternary(k_ternary[i]),
                .tx_contrib(tx_part[i]),
                .sc_contrib(sc_part[i])
            );
        end
    endgenerate

    // Simple sequential accumulator placeholder; synthesis will flatten.
    integer j;
    reg signed [SCORE_W-1:0] tx_acc, sc_acc;
    always @* begin
        tx_acc = 0;
        sc_acc = 0;
        for (j = 0; j < HEAD_DIM; j = j + 1) begin
            tx_acc = tx_acc + tx_part[j];
            sc_acc = sc_acc + sc_part[j];
        end
    end
    assign tx_score = tx_acc;
    assign sc_score = sc_acc / $signed(32'sd32);  // head_dim normalize
endmodule