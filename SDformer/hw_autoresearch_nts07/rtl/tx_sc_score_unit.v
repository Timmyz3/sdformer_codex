`include "nts07_pkg.vh"

// ============================================================
// TX (alpha-XNOR) + SC (signed consensus) per-channel contribution
// Computes 4 counters per channel pair, fused at tree output.
//
// Channel match categories (mutually exclusive):
//   SAME_NONZERO = (q+ & k+) | (q- & k-)           → TX += +1,  SC += +1
//   SAME_ZERO    = (~qact) & (~kact)               → TX += +α₀, SC += 0
//   OPPOSITE     = (q+ & k-) | (q- & k+)           → TX += -β,  SC += -1
//   SINGLE       = qact ^ kact (exactly one fires) → TX += -γ,  SC += 0
//
// Coefficients are Q0.8 fixed-point (checkpoint-frozen):
//   α₀ ≈ 0.02 → 5/256, β = 0.25 → 64/256, γ ≈ 0.15 → 38/256, μ ≈ 0.05 → 13/256
// ============================================================

module tx_sc_per_channel (
    input  wire [1:0] q_ternary,
    input  wire [1:0] k_ternary,
    output wire       same_nonzero,
    output wire       same_zero,
    output wire       opposite,
    output wire       single_active,
    output wire       sc_sign     // +1=same_nonzero, -1=opposite, 0=otherwise
);
    wire q_pos = (q_ternary == `TERN_POS);
    wire q_neg = (q_ternary == `TERN_NEG);
    wire q_act = q_pos | q_neg;
    wire k_pos = (k_ternary == `TERN_POS);
    wire k_neg = (k_ternary == `TERN_NEG);
    wire k_act = k_pos | k_neg;

    assign same_nonzero  = (q_pos & k_pos) | (q_neg & k_neg);
    assign same_zero     = (~q_act) & (~k_act);
    assign opposite      = (q_pos & k_neg) | (q_neg & k_pos);
    assign single_active = q_act ^ k_act;
    assign sc_sign       = same_nonzero;     // SC uses +1 for same, handled by subtracting opposite
endmodule


// ============================================================
// Pipelined popcount adder tree with 5 stages (for 32-dim)
// Input: 32 bits of 1/0 for a match category
// Output: 6-bit count, fully pipelined
// ============================================================
module popcount_pipelined #(
    parameter integer N = 32,
    parameter integer W = 6    // ceil(log2(N))
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    input  wire [N-1:0]     bits_in,
    output reg  [W-1:0]     count_out
);
    localparam STAGES = $clog2(N);  // 5 for N=32

    // Stage registers
    reg [W-1:0] stage [0:STAGES-1][0:N-1];
    integer s, i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (s = 0; s < STAGES; s = s + 1)
                for (i = 0; i < N; i = i + 1)
                    stage[s][i] <= {W{1'b0}};
            count_out <= {W{1'b0}};
        end else if (en) begin
            // Stage 0: register inputs as 0/1
            for (i = 0; i < N; i = i + 1)
                stage[0][i] <= {{(W-1){1'b0}}, bits_in[i]};

            // Stages 1 to STAGES-1: pairwise add
            for (s = 1; s < STAGES; s = s + 1) begin
                for (i = 0; i < (N >> s); i = i + 1) begin
                    stage[s][i] <= stage[s-1][2*i] + stage[s-1][2*i+1];
                end
            end

            count_out <= stage[STAGES-1][0];
        end
    end
endmodule


// ============================================================
// Full TX+SC fused score computation for one (q_token, k_token) pair
// over all HEAD_DIM channels.
//
// Latency: 5 cycles (popcount tree) + 1 cycle (coefficient fusion) = 6 cycles
// Throughput: one (i,j) pair per cycle after pipeline fill
//
// Output scores:
//   tx_score_q43: Q4.3 signed, range approx [-8, +32] normalized to /HEAD_DIM
//   sc_score_q43: Q4.3 signed, range [-4, +4] = (same_nonzero - opposite) / HEAD_DIM
// ============================================================
module tx_sc_pair_score #(
    parameter integer HEAD_DIM = `NTS07_HEAD_DIM,
    parameter integer SCORE_W  = `NTS07_SCORE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire [1:0]                   q_ternary [0:HEAD_DIM-1],
    input  wire [1:0]                   k_ternary [0:HEAD_DIM-1],
    input  wire [7:0]                   alpha0_q8,    // same-zero bonus
    input  wire [7:0]                   beta_q8,      // opposite penalty
    input  wire [7:0]                   gamma_q8,     // single-active penalty
    output wire signed [SCORE_W-1:0]    tx_score,
    output wire signed [SCORE_W-1:0]    sc_score,
    output wire                         valid_out
);
    localparam CNT_W = $clog2(HEAD_DIM) + 1;  // 6 bits for 32

    // Per-channel match bits
    wire same_nonzero [0:HEAD_DIM-1];
    wire same_zero    [0:HEAD_DIM-1];
    wire opposite     [0:HEAD_DIM-1];
    wire single_act   [0:HEAD_DIM-1];
    wire sc_pos       [0:HEAD_DIM-1];

    // Packed bit vectors for popcount
    wire [HEAD_DIM-1:0] v_same, v_zero, v_opp, v_single;

    genvar ch;
    generate
        for (ch = 0; ch < HEAD_DIM; ch = ch + 1) begin : gen_ch
            tx_sc_per_channel u_pc (
                .q_ternary(q_ternary[ch]), .k_ternary(k_ternary[ch]),
                .same_nonzero(same_nonzero[ch]), .same_zero(same_zero[ch]),
                .opposite(opposite[ch]), .single_active(single_act[ch]),
                .sc_sign(sc_pos[ch])
            );
            assign v_same[ch]   = same_nonzero[ch];
            assign v_zero[ch]   = same_zero[ch];
            assign v_opp[ch]    = opposite[ch];
            assign v_single[ch] = single_act[ch];
        end
    endgenerate

    // Pipelined popcounts for all four categories (5 cycles latency)
    wire [CNT_W-1:0] cnt_same, cnt_zero, cnt_opp, cnt_single;
    popcount_pipelined #(.N(HEAD_DIM), .W(CNT_W)) u_pop_same
        (.clk(clk), .rst_n(rst_n), .en(en), .bits_in(v_same), .count_out(cnt_same));
    popcount_pipelined #(.N(HEAD_DIM), .W(CNT_W)) u_pop_zero
        (.clk(clk), .rst_n(rst_n), .en(en), .bits_in(v_zero), .count_out(cnt_zero));
    popcount_pipelined #(.N(HEAD_DIM), .W(CNT_W)) u_pop_opp
        (.clk(clk), .rst_n(rst_n), .en(en), .bits_in(v_opp),  .count_out(cnt_opp));
    popcount_pipelined #(.N(HEAD_DIM), .W(CNT_W)) u_pop_single
        (.clk(clk), .rst_n(rst_n), .en(en), .bits_in(v_single), .count_out(cnt_single));

    // Align valid: 5 stage delay from popcount
    reg [5:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= 6'd0;
        else
            valid_pipe <= {valid_pipe[4:0], en};
    end
    assign valid_out = valid_pipe[5];

    // Final fused score (one more cycle = total 6)
    // TX = (cnt_same*256 + cnt_zero*α₀ - cnt_opp*β - cnt_single*γ) >>> (8 + log2(HEAD_DIM))
    // HEAD_DIM = 32 = 2^5 → shift right total 13, then repack to Q4.3
    // SC = (cnt_same - cnt_opp) >>> log2(HEAD_DIM) = >>>5 → Q4.3
    reg signed [20:0] tx_full;
    reg signed [CNT_W:0] sc_full;
    reg signed [SCORE_W-1:0] tx_r, sc_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_r <= {SCORE_W{1'b0}};
            sc_r <= {SCORE_W{1'b0}};
        end else if (valid_pipe[4]) begin
            // Q(5+8).8 intermediate
            tx_full = (  $signed({1'b0, cnt_same})   * $signed(8'd256)
                       + $signed({1'b0, cnt_zero})   * $signed(alpha0_q8)
                       - $signed({1'b0, cnt_opp})    * $signed(beta_q8)
                       - $signed({1'b0, cnt_single}) * $signed(gamma_q8) );
            // Normalize by HEAD_DIM=32 → >>>5, and Q8 fractional → >>>8 → total >>>13
            // Result in Q4.3: we want 3 fractional bits, so shift right 10 (13-3=10)
            tx_r <= tx_full >>> 10;

            // SC: (same - opposite) / 32 → Q4.3
            sc_full = $signed({1'b0, cnt_same}) - $signed({1'b0, cnt_opp});
            sc_r <= sc_full >>> 2;  // divide by 4 gives Q4.3 from 6-bit signed
        end
    end

    assign tx_score = tx_r;
    assign sc_score = sc_r;

endmodule


// ============================================================
// Score fusion: tx + mu*SC, with optional row centering (subtract mean)
// This is the final fused score feeding into Shiftmax.
// ============================================================
module score_fuse_unit #(
    parameter integer SCORE_W = `NTS07_SCORE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire signed [SCORE_W-1:0]    tx_in,
    input  wire signed [SCORE_W-1:0]    sc_in,
    input  wire [7:0]                   mu_q8,
    input  wire signed [SCORE_W-1:0]    row_mean,   // precomputed mean for centering
    input  wire                         center_en,
    output reg signed [SCORE_W-1:0]     score_out,
    output reg                          valid_out
);
    reg signed [SCORE_W+8-1:0] mu_sc_full;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            score_out <= {SCORE_W{1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= en;
            if (en) begin
                mu_sc_full = $signed(sc_in) * $signed({1'b0, mu_q8});
                score_out  = tx_in + (mu_sc_full >>> 8);
                if (center_en)
                    score_out = score_out - row_mean;
            end
        end
    end
endmodule
