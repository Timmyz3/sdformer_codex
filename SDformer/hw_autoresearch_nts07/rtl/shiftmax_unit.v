`include "nts07_pkg.vh"

// ============================================================
// Pipelined Shiftmax normalization unit (DATE 2027)
// Computes: gate[i] = (2^(s[i] - max_s) * scale) >>> denom_pow
// where denom_pow = ceil(log2(Σ 2^(s[i] - max_s)))
//
// No division: denominator is a power-of-two, implemented as barrel shift.
// 2^x implemented as a small hardwired LUT (16 entries for x=-15..0).
// Fully pipelined; latency = ~10 cycles, throughput = 1 row/cycle.
//
// Inputs:  N fused scores (Q4.3 signed)
// Outputs: N gates (Q0.8 unsigned, 0..255)
// ============================================================

// Hardwired LUT for 2^x in Q0.8, x ∈ [-15, 0]
//  x = 0  → 2^0 = 1.000 → 255 (clipped from 256)
//  x = -1 → 0.500 → 128
//  x = -2 → 0.250 → 64
//  ...
//  x = -8 → 1/256 → 1
//  x ≤ -9 → <1/256 → 0
module pow2_lut #(
    parameter integer OUT_W = `NTS07_GATE_W
)(
    input  wire signed [`NTS07_SCORE_W-1:0] x,  // shifted score = s[i]-max_s (≤ 0)
    output reg  [OUT_W-1:0]  y                    // 2^x in Q0.8
);
    always @* begin
        case (x)
            8'sd0:  y = 8'd255;
            -8'sd1: y = 8'd128;
            -8'sd2: y = 8'd64;
            -8'sd3: y = 8'd32;
            -8'sd4: y = 8'd16;
            -8'sd5: y = 8'd8;
            -8'sd6: y = 8'd4;
            -8'sd7: y = 8'd2;
            -8'sd8: y = 8'd1;
            default: y = 8'd0;  // x ≤ -9 → 2^x < 1/256 = 0 in Q0.8
        endcase
    end
endmodule


// Pipelined tree maximum finder for N tokens
module pipelined_max #(
    parameter integer N = `NTS07_MAX_TOKENS,
    parameter integer W = `NTS07_SCORE_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,
    input  wire signed [W-1:0]      data_in  [0:N-1],
    output reg  signed [W-1:0]      data_out,
    output reg                      valid_out
);
    localparam STAGES = $clog2(N + 1);  // 7 stages for 98 tokens
    reg signed [W-1:0] stage [0:STAGES-1][0:N-1];
    reg [STAGES:0] valid_pipe;
    integer s, i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (s = 0; s < STAGES; s = s + 1)
                for (i = 0; i < N; i = i + 1)
                    stage[s][i] <= {W{1'b0}};
            data_out <= {W{1'b0}};
            valid_pipe <= 0;
            valid_out <= 0;
        end else if (en) begin
            // Stage 0: register inputs
            for (i = 0; i < N; i = i + 1)
                stage[0][i] <= data_in[i];

            // Pairwise max stages
            for (s = 1; s < STAGES; s = s + 1) begin
                for (i = 0; i < (N >> s) + 1; i = i + 1) begin
                    if (2*i+1 < N)
                        stage[s][i] <= (stage[s-1][2*i] >= stage[s-1][2*i+1]) ?
                                       stage[s-1][2*i] : stage[s-1][2*i+1];
                    else if (2*i < N)
                        stage[s][i] <= stage[s-1][2*i];
                    else
                        stage[s][i] <= {W{1'b0}};
                end
            end

            data_out <= stage[STAGES-1][0];
            valid_pipe <= {valid_pipe[STAGES-1:0], 1'b1};
            valid_out <= valid_pipe[STAGES];
        end else begin
            valid_pipe <= {valid_pipe[STAGES-1:0], 1'b0};
            valid_out <= valid_pipe[STAGES];
        end
    end
endmodule


// Pipelined adder tree for N numbers
module pipelined_sum_tree #(
    parameter integer N = `NTS07_MAX_TOKENS,
    parameter integer W = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,
    input  wire [W-1:0]             data_in  [0:N-1],
    output reg  [W-1:0]             data_out,
    output reg                      valid_out
);
    localparam STAGES = $clog2(N + 1);
    reg [W-1:0] stage [0:STAGES-1][0:N-1];
    reg [STAGES:0] valid_pipe;
    integer s, i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (s = 0; s < STAGES; s = s + 1)
                for (i = 0; i < N; i = i + 1)
                    stage[s][i] <= {W{1'b0}};
            data_out <= {W{1'b0}};
            valid_pipe <= 0;
            valid_out <= 0;
        end else if (en) begin
            for (i = 0; i < N; i = i + 1)
                stage[0][i] <= (i < N) ? data_in[i] : {W{1'b0}};

            for (s = 1; s < STAGES; s = s + 1) begin
                for (i = 0; i < (N >> s) + 1; i = i + 1) begin
                    if (2*i+1 < N)
                        stage[s][i] <= stage[s-1][2*i] + stage[s-1][2*i+1];
                    else if (2*i < N)
                        stage[s][i] <= stage[s-1][2*i];
                    else
                        stage[s][i] <= {W{1'b0}};
                end
            end

            data_out <= stage[STAGES-1][0];
            valid_pipe <= {valid_pipe[STAGES-1:0], 1'b1};
            valid_out <= valid_pipe[STAGES];
        end else begin
            valid_pipe <= {valid_pipe[STAGES-1:0], 1'b0};
            valid_out <= valid_pipe[STAGES];
        end
    end
endmodule


// Count leading zeros (CLZ) for ceil(log2)
// Returns position of highest set bit + 1 = ceil(log2(x)) for x>0
module clz_ceil_log2 #(
    parameter integer W = 16
)(
    input  wire [W-1:0] x,
    output reg  [4:0]   pow_out
);
    integer b;
    reg [4:0] b_plus_1;
    always @* begin
        pow_out = 0;
        if (x != 0) begin
            for (b = W-1; b >= 0; b = b - 1) begin
                if (x[b]) begin
                    b_plus_1 = b[4:0] + 5'd1;
                    if (x == (1 << b))
                        pow_out = b[4:0];
                    else
                        pow_out = b_plus_1;
                    b = -1;
                end
            end
        end
    end
endmodule


// ============================================================
// Top-level pipelined Shiftmax unit
// Latency: ~12 cycles (max_tree 7 + lut 1 + sum_tree 7 + clz 1 + gate 1, overlapped)
// ============================================================
module shiftmax_unit #(
    parameter integer MAX_TOKENS = `NTS07_MAX_TOKENS,
    parameter integer SCORE_W    = `NTS07_SCORE_W,
    parameter integer GATE_W     = `NTS07_GATE_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [6:0]               n_tokens,
    input  wire                     preserve_mean,
    input  wire signed [SCORE_W-1:0] scores [0:MAX_TOKENS-1],
    output reg  [GATE_W-1:0]        gates  [0:MAX_TOKENS-1],
    output reg                      done
);
    localparam SUM_W = 16;  // enough for 98 tokens × 255 = ~25k

    // --- Stage 1-7: find row max (pipelined tree) ---
    wire signed [SCORE_W-1:0] row_max;
    wire max_valid;
    pipelined_max #(.N(MAX_TOKENS), .W(SCORE_W)) u_max (
        .clk(clk), .rst_n(rst_n), .en(start),
        .data_in(scores), .data_out(row_max), .valid_out(max_valid)
    );

    // Delay scores to match max latency (7 stages)
    reg signed [SCORE_W-1:0] scores_d [6:0][0:MAX_TOKENS-1];
    reg [7:0] start_d;
    integer d, i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_d <= 0;
        end else begin
            start_d <= {start_d[6:0], start};
        end
    end

    always @(posedge clk) begin
        for (i = 0; i < MAX_TOKENS; i = i + 1)
            scores_d[0][i] <= scores[i];
        for (d = 1; d < 7; d = d + 1)
            for (i = 0; i < MAX_TOKENS; i = i + 1)
                scores_d[d][i] <= scores_d[d-1][i];
    end

    // --- Stage 8: subtract max and LUT lookup ---
    reg signed [SCORE_W-1:0] shifted   [0:MAX_TOKENS-1];
    wire [GATE_W-1:0]        pow2_val  [0:MAX_TOKENS-1];
    reg                      lut_valid;
    reg [6:0]                n_tokens_d0;
    reg                      preserve_mean_d0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_valid <= 0;
            n_tokens_d0 <= 0;
            preserve_mean_d0 <= 0;
        end else begin
            lut_valid <= max_valid;
            n_tokens_d0 <= n_tokens;
            preserve_mean_d0 <= preserve_mean;
            if (max_valid) begin
                for (i = 0; i < MAX_TOKENS; i = i + 1) begin
                    if (i < n_tokens)
                        shifted[i] <= scores_d[6][i] - row_max;
                    else
                        shifted[i] <= {SCORE_W{1'b1}};  // very negative → LUT 0
                end
            end
        end
    end

    // LUT instantiation in generate block (not allowed inside always)
    genvar lut_i;
    generate
        for (lut_i = 0; lut_i < MAX_TOKENS; lut_i = lut_i + 1) begin : gen_pow2_lut
            pow2_lut #(.OUT_W(GATE_W)) u_lut (.x(shifted[lut_i]), .y(pow2_val[lut_i]));
        end
    endgenerate

    // --- Stage 9-15: sum tree for row_sum ---
    // Zero-extend pow2_val from GATE_W to SUM_W (8→16)
    reg [SUM_W-1:0] pow2_ext [0:MAX_TOKENS-1];
    always @* begin
        for (i = 0; i < MAX_TOKENS; i = i + 1)
            pow2_ext[i] = {{(SUM_W-GATE_W){1'b0}}, pow2_val[i]};
    end

    wire [SUM_W-1:0] row_sum;
    wire sum_valid;
    pipelined_sum_tree #(.N(MAX_TOKENS), .W(SUM_W)) u_sum (
        .clk(clk), .rst_n(rst_n), .en(lut_valid),
        .data_in(pow2_ext), .data_out(row_sum), .valid_out(sum_valid)
    );

    // Delay pow2_val to match sum latency (7 stages)
    reg [GATE_W-1:0] pow2_d [6:0][0:MAX_TOKENS-1];
    reg [7:0] lut_valid_d;
    reg [6:0] n_tokens_d1;
    reg preserve_mean_d1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_valid_d <= 0;
            n_tokens_d1 <= 0;
            preserve_mean_d1 <= 0;
        end else begin
            lut_valid_d <= {lut_valid_d[6:0], lut_valid};
            n_tokens_d1 <= n_tokens_d0;
            preserve_mean_d1 <= preserve_mean_d0;
            for (i = 0; i < MAX_TOKENS; i = i + 1)
                pow2_d[0][i] <= pow2_val[i];
            for (d = 1; d < 7; d = d + 1)
                for (i = 0; i < MAX_TOKENS; i = i + 1)
                    pow2_d[d][i] <= pow2_d[d-1][i];
        end
    end

    // --- Stage 16: CLZ for denom_pow ---
    wire [4:0] denom_pow;
    clz_ceil_log2 #(.W(SUM_W)) u_clz (.x(row_sum), .pow_out(denom_pow));

    reg [4:0] denom_pow_r;
    reg [GATE_W-1:0] pow2_final [0:MAX_TOKENS-1];
    reg [6:0] n_tokens_final;
    reg preserve_mean_final;
    reg clz_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clz_valid <= 0;
            n_tokens_final <= 0;
            preserve_mean_final <= 0;
            denom_pow_r <= 0;
        end else begin
            clz_valid <= sum_valid;
            n_tokens_final <= n_tokens_d1;
            preserve_mean_final <= preserve_mean_d1;
            denom_pow_r <= denom_pow;
            for (i = 0; i < MAX_TOKENS; i = i + 1)
                pow2_final[i] <= pow2_d[6][i];
        end
    end

    // --- Stage 17: final gate computation (shift only, no division) ---
    // gate[i] = (pow2_final[i] * scale) >>> denom_pow_r
    // scale = n_tokens_final when preserve_mean, else 1
    // Since pow2_val is Q0.8 (0..255), and row_sum is sum (Q8.0 approx),
    // dividing by 2^denom_pow gives Q0.8 normalized gate (0..255).
    reg [SUM_W+GATE_W-1:0] gate_num;
    reg [4:0] shift_amt;
    reg [SUM_W+GATE_W-1:0] gate_shifted;
    reg [SUM_W+GATE_W-1:0] gate_max;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 0;
            for (i = 0; i < MAX_TOKENS; i = i + 1)
                gates[i] <= {GATE_W{1'b0}};
        end else if (clz_valid) begin
            done <= 1'b1;
            gate_max = (1 << GATE_W) - 1;
            for (i = 0; i < MAX_TOKENS; i = i + 1) begin
                if (i < n_tokens_final) begin
                    gate_num = {{(SUM_W){1'b0}}, pow2_final[i]};
                    if (preserve_mean_final)
                        gate_num = gate_num * {{(SUM_W-7){1'b0}}, n_tokens_final};
                    shift_amt = denom_pow_r;
                    if (shift_amt > 0)
                        gate_shifted = gate_num >> shift_amt;
                    else
                        gate_shifted = gate_num;
                    if (gate_shifted > gate_max)
                        gates[i] <= {GATE_W{1'b1}};
                    else
                        gates[i] <= gate_shifted[GATE_W-1:0];
                end else begin
                    gates[i] <= {GATE_W{1'b0}};
                end
            end
        end else begin
            done <= 0;
        end
    end

endmodule
