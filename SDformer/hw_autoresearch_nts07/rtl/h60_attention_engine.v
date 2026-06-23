`include "nts07_pkg.vh"

// ============================================================
// H60 Attention Engine: one window × one head
// Implements full N×N TX+μSC fused attention with Shiftmax gating.
//
// Algorithm per head per window:
//   For each query token i (0..N-1):
//     1. Score row i: for all key tokens j (0..N-1):
//          s[i][j] = TX(q[i],k[j]) + μ·SC(q[i],k[j])
//        where TX = α-XNOR popcount, SC = signed consensus
//     2. Optional center: s[i][j] -= mean(s[i][*])
//     3. Shiftmax: gate[i][j] = 2^(s[i][j]-max_i) / Σ 2^(s[i][j]-max_i)
//        (no division: denominator is power-of-two, done by barrel shift)
//     4. Weighted sum (value = K_orig, binary/ternary spikes × INT8 act):
//          attn_out[i][d] = Σ_j gate[i][j] · K_orig[j][d]  (all channels d)
//
// Hardware features (DATE 2027 paper innovations):
//   - Single-ISA H60 fused attention: TX+SC on same popcount datapath
//   - Shiftmax normalization: no dividers, no exp(), LUT-only
//   - Zero-skip at token level: silent Q row → skip entire score pipeline
//   - Zero-skip at channel level inside Sparse MAC (see sparse_mac_pe.v)
//   - All activations binary/ternary; weights INT8; no floating point anywhere
//   - Fully pipelined score path (1 pair/cycle throughput, 6-cycle latency)
//
// Latency estimate per window-head (N=98, D=32):
//   Load Q/K: ~300 cycles
//   Score rows: 98 × (98 + 6) ≈ 10,200 cycles
//   Shiftmax (18 cyc) + accumulate (98 cyc) per row: 98 × 116 ≈ 11,400 cycles
//   Total: ~22k cycles per window-head @500MHz ≈ 44μs
// ============================================================
module h60_attention_engine #(
    parameter integer HEAD_DIM    = `NTS07_HEAD_DIM,
    parameter integer MAX_TOKENS  = `NTS07_MAX_TOKENS,
    parameter integer ACT_W       = `NTS07_ACT_W,
    parameter integer SCORE_W     = `NTS07_SCORE_W,
    parameter integer GATE_W      = `NTS07_GATE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    // Control
    input  wire                         start,
    output reg                          done,
    // Config (checkpoint-frozen, per-stage from LUT)
    input  wire [7:0]                   mu_q8,
    input  wire [7:0]                   alpha0_q8,
    input  wire [7:0]                   beta_q8,
    input  wire [7:0]                   gamma_q8,
    input  wire                         center_scores,
    input  wire                         preserve_mean,
    input  wire [6:0]                   n_tokens,
    // Load port: write Q/K/K_orig before asserting start
    input  wire                         load_en,
    input  wire [1:0]                   load_qkv_sel,   // 0=Q, 1=K, 2=K_orig
    input  wire [6:0]                   load_idx,
    input  wire [1:0]                   q_ternary  [0:HEAD_DIM-1],
    input  wire [1:0]                   k_ternary  [0:HEAD_DIM-1],
    input  wire signed [ACT_W-1:0]      k_orig     [0:HEAD_DIM-1],
    // Output: one token per cycle
    output reg                          out_valid,
    output reg  [6:0]                   out_idx,
    output reg  signed [ACT_W-1:0]      attn_out   [0:HEAD_DIM-1]
);
    // --- Local register files ---
    reg [1:0]              Q_mem [0:MAX_TOKENS-1][0:HEAD_DIM-1];
    reg [1:0]              K_mem [0:MAX_TOKENS-1][0:HEAD_DIM-1];
    reg signed [ACT_W-1:0] V_mem [0:MAX_TOKENS-1][0:HEAD_DIM-1];

    // --- Score pipeline feed ---
    reg                      score_en;
    reg [6:0]                score_i, score_j;
    wire [1:0]               q_vec [0:HEAD_DIM-1];
    wire [1:0]               k_vec [0:HEAD_DIM-1];
    wire signed [SCORE_W-1:0] tx_raw, sc_raw;
    wire                     pair_valid;
    wire signed [SCORE_W-1:0] fused_score;
    wire                     fused_valid;

    // --- Score row buffer for current query i ---
    reg [6:0] cur_i;
    reg signed [SCORE_W-1:0] score_buf [0:MAX_TOKENS-1];
    reg [6:0]                wb_idx;
    reg                      all_scores_written;

    // --- Shiftmax ---
    reg                      shift_start;
    wire [GATE_W-1:0]        gate_vec [0:MAX_TOKENS-1];
    wire                     shift_done;

    // --- Accumulation ---
    reg [6:0]                acc_j;
    reg signed [ACT_W+GATE_W-1:0] acc [0:HEAD_DIM-1];
    reg signed [ACT_W+GATE_W-1:0] prod;

    // --- Q active mask: precomputed during load ---
    reg q_active [0:MAX_TOKENS-1];

    // --- FSM ---
    localparam [2:0]
        S_IDLE   = 3'd0,
        S_SCORE  = 3'd1,
        S_DRAIN  = 3'd2,
        S_CENTER = 3'd3,
        S_SHIFT  = 3'd4,
        S_ACCUM  = 3'd5,
        S_NEXT   = 3'd6,
        S_DONE   = 3'd7;

    reg [2:0] state;
    reg [4:0] drain_cnt;
    integer d, j;
    reg signed [SCORE_W+6:0] row_sum;
    reg signed [SCORE_W-1:0]  row_mean;

    // ============================================================
    // Datapath instantiation
    // ============================================================

    // Feed Q[cur_i] and K[score_j] into score pipeline
    genvar fd;
    generate
        for (fd = 0; fd < HEAD_DIM; fd = fd + 1) begin : gen_feed
            assign q_vec[fd] = Q_mem[score_i][fd];
            assign k_vec[fd] = K_mem[score_j][fd];
        end
    endgenerate

    // TX+SC pair score (6-stage pipeline)
    tx_sc_pair_score #(.HEAD_DIM(HEAD_DIM), .SCORE_W(SCORE_W)) u_pair (
        .clk(clk), .rst_n(rst_n), .en(score_en),
        .q_ternary(q_vec), .k_ternary(k_vec),
        .alpha0_q8(alpha0_q8), .beta_q8(beta_q8), .gamma_q8(gamma_q8),
        .tx_score(tx_raw), .sc_score(sc_raw), .valid_out(pair_valid)
    );

    // Fuse: s = tx + μ·sc (1 additional cycle → total 7)
    score_fuse_unit #(.SCORE_W(SCORE_W)) u_fuse (
        .clk(clk), .rst_n(rst_n),
        .en(pair_valid),
        .tx_in(tx_raw), .sc_in(sc_raw), .mu_q8(mu_q8),
        .row_mean({SCORE_W{1'b0}}), .center_en(1'b0),
        .score_out(fused_score), .valid_out(fused_valid)
    );

    // Shiftmax normalizer (17-stage pipeline)
    shiftmax_unit #(.MAX_TOKENS(MAX_TOKENS), .SCORE_W(SCORE_W), .GATE_W(GATE_W))
        u_shiftmax (
            .clk(clk), .rst_n(rst_n),
            .start(shift_start), .n_tokens(n_tokens),
            .preserve_mean(preserve_mean),
            .scores(score_buf), .gates(gate_vec), .done(shift_done)
        );

    // ============================================================
    // Load port + Q-active precompute
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < MAX_TOKENS; j = j + 1)
                q_active[j] <= 1'b0;
        end else if (load_en) begin
            case (load_qkv_sel)
                2'd0: begin
                    q_active[load_idx] <= 1'b0;
                    for (d = 0; d < HEAD_DIM; d = d + 1) begin
                        Q_mem[load_idx][d] <= q_ternary[d];
                        if (q_ternary[d] != `TERN_SILENT)
                            q_active[load_idx] <= 1'b1;
                    end
                end
                2'd1: begin
                    for (d = 0; d < HEAD_DIM; d = d + 1)
                        K_mem[load_idx][d] <= k_ternary[d];
                end
                2'd2: begin
                    for (d = 0; d < HEAD_DIM; d = d + 1)
                        V_mem[load_idx][d] <= k_orig[d];
                end
            endcase
        end
    end

    // ============================================================
    // Score writeback: wb_idx increments with fused_valid, writes to score_buf
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_idx <= 0;
            for (j = 0; j < MAX_TOKENS; j = j + 1)
                score_buf[j] <= {SCORE_W{1'b0}};
        end else begin
            if (state == S_SCORE && score_j == 0) begin
                wb_idx <= 0;
            end
            if (fused_valid && wb_idx < n_tokens) begin
                score_buf[wb_idx] <= fused_score;
                wb_idx <= wb_idx + 1;
            end
        end
    end

    // ============================================================
    // Main FSM
    // ============================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            score_en <= 1'b0;
            shift_start <= 1'b0;
            score_i <= 0;
            score_j <= 0;
            cur_i <= 0;
            drain_cnt <= 0;
            acc_j <= 0;
            out_idx <= 0;
            all_scores_written <= 1'b0;
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                acc[d] <= 0;
                attn_out[d] <= 0;
            end
        end else begin
            score_en <= 1'b0;
            shift_start <= 1'b0;
            out_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                // -------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        cur_i <= 0;
                        score_i <= 0;
                        score_j <= 0;
                        wb_idx <= 0;
                        state <= S_SCORE;
                    end
                end

                // -------------------------------------------
                // Feed j=0..n_tokens-1 into score pipeline (one per cycle)
                S_SCORE: begin
                    if (!q_active[cur_i]) begin
                        // Silent query → zero score row, go directly to Shiftmax
                        for (j = 0; j < MAX_TOKENS; j = j + 1)
                            score_buf[j] <= {SCORE_W{1'b0}};
                        shift_start <= 1'b1;
                        state <= S_SHIFT;
                    end else if (score_j < n_tokens) begin
                        score_en <= 1'b1;
                        score_j <= score_j + 1;
                    end else begin
                        // Last j fed; wait for pipeline to drain (7 cycles)
                        drain_cnt <= 0;
                        state <= S_DRAIN;
                    end
                end

                // -------------------------------------------
                S_DRAIN: begin
                    if (fused_valid) begin
                        if (drain_cnt == 6) begin
                            // All N scores have arrived (wb_idx should == n_tokens)
                            state <= S_CENTER;
                        end else begin
                            drain_cnt <= drain_cnt + 1;
                        end
                    end
                end

                // -------------------------------------------
                // Optional centering: subtract row mean
                S_CENTER: begin
                    if (center_scores) begin
                        row_sum = 0;
                        for (j = 0; j < MAX_TOKENS; j = j + 1)
                            if (j < n_tokens) row_sum = row_sum + score_buf[j];
                        row_mean = row_sum / $signed({1'b0, n_tokens});
                        for (j = 0; j < MAX_TOKENS; j = j + 1)
                            if (j < n_tokens)
                                score_buf[j] <= score_buf[j] - row_mean;
                            else
                                score_buf[j] <= {1'b1, {SCORE_W-1{1'b1}}};
                    end else begin
                        for (j = 0; j < MAX_TOKENS; j = j + 1)
                            if (j >= n_tokens)
                                score_buf[j] <= {1'b1, {SCORE_W-1{1'b1}}};
                    end
                    shift_start <= 1'b1;
                    state <= S_SHIFT;
                end

                // -------------------------------------------
                // Wait for Shiftmax to complete
                S_SHIFT: begin
                    if (shift_done) begin
                        acc_j <= 0;
                        for (d = 0; d < HEAD_DIM; d = d + 1)
                            acc[d] <= 0;
                        state <= S_ACCUM;
                    end
                end

                // -------------------------------------------
                // Weighted sum: out[i][d] = Σ_j gate[j] * V[j][d] >>> GATE_W
                S_ACCUM: begin
                    if (acc_j < n_tokens) begin
                        for (d = 0; d < HEAD_DIM; d = d + 1) begin
                            prod = $signed({1'b0, gate_vec[acc_j]}) * $signed(V_mem[acc_j][d]);
                            acc[d] <= acc[d] + prod;
                        end
                        acc_j <= acc_j + 1;
                    end else begin
                        for (d = 0; d < HEAD_DIM; d = d + 1)
                            attn_out[d] <= acc[d] >>> GATE_W;
                        out_idx <= cur_i;
                        out_valid <= 1'b1;

                        if (cur_i == n_tokens - 1) begin
                            state <= S_DONE;
                        end else begin
                            cur_i <= cur_i + 1;
                            score_i <= cur_i + 1;
                            score_j <= 0;
                            wb_idx <= 0;
                            state <= S_SCORE;
                        end
                    end
                end

                // -------------------------------------------
                S_DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
