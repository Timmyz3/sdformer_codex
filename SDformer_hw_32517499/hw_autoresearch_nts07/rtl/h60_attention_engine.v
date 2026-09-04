`include "nts07_pkg.vh"

// H60 no-carrier attention engine for one head × one window.
// scores = TX + mu*SC; gate = Shiftmax; attn = K_orig * gate
module h60_attention_engine #(
    parameter integer HEAD_DIM = 32,
    parameter integer MAX_TOKENS = 98,
    parameter integer DATA_W = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    output reg                      done,
    input  wire [7:0]               mu_q8,
    input  wire                     center_scores,
    input  wire                     preserve_mean,
    input  wire [6:0]               n_tokens,
    input  wire [1:0]               q_ternary [0:MAX_TOKENS-1][0:HEAD_DIM-1],
    input  wire [1:0]               k_ternary [0:MAX_TOKENS-1][0:HEAD_DIM-1],
    input  wire signed [DATA_W-1:0] k_orig    [0:MAX_TOKENS-1][0:HEAD_DIM-1],
    output reg  signed [DATA_W-1:0] attn_out  [0:MAX_TOKENS-1][0:HEAD_DIM-1]
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_SCORE = 2'd1;
    localparam [1:0] ST_SHIFT = 2'd2;
    localparam [1:0] ST_GATE = 2'd3;

    reg [1:0] state;
    reg [6:0] tok_idx;
    reg [6:0] ch_idx;

    wire signed [`NTS07_SCORE_W-1:0] tx_s, sc_s;
    reg signed [`NTS07_SCORE_W-1:0] fused_scores [0:MAX_TOKENS-1];
    reg signed [`NTS07_SCORE_W-1:0] score_mean;
    wire [`NTS07_GATE_W-1:0] gates [0:MAX_TOKENS-1];

    tx_sc_score_tree #(.HEAD_DIM(HEAD_DIM)) u_score (
        .q_ternary(q_ternary[tok_idx]),
        .k_ternary(k_ternary[tok_idx]),
        .tx_score(tx_s),
        .sc_score(sc_s)
    );

    shiftmax_unit #(.MAX_TOKENS(MAX_TOKENS)) u_shiftmax (
        .scores(fused_scores),
        .n_tokens(n_tokens),
        .preserve_mean(preserve_mean),
        .gates(gates)
    );

    integer i, j;
    reg signed [15:0] mu_sc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            tok_idx <= 0;
            ch_idx <= 0;
            score_mean <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        // Pre-compute row mean if center_scores enabled
                        if (center_scores) begin
                            score_mean = 0;
                            for (i = 0; i < MAX_TOKENS; i = i + 1) begin
                                if (i < n_tokens)
                                    score_mean = score_mean + fused_scores[i];
                            end
                            score_mean = score_mean / $signed({1'b0, n_tokens});
                        end
                        tok_idx <= 0;
                        state <= ST_SCORE;
                    end
                end

                ST_SCORE: begin
                    mu_sc = ($signed(sc_s) * $signed({8'b0, mu_q8})) >>> 8;
                    fused_scores[tok_idx] <= tx_s + mu_sc;
                    if (center_scores)
                        fused_scores[tok_idx] <= fused_scores[tok_idx] - score_mean;

                    if (tok_idx == n_tokens - 1) begin
                        tok_idx <= 0;
                        state <= ST_SHIFT;
                    end else begin
                        tok_idx <= tok_idx + 1;
                    end
                end

                ST_SHIFT: begin
                    // shiftmax_unit is combinational; proceed to gating
                    tok_idx <= 0;
                    ch_idx <= 0;
                    state <= ST_GATE;
                end

                ST_GATE: begin
                    attn_out[tok_idx][ch_idx] <= (
                        k_orig[tok_idx][ch_idx] * $signed({8'b0, gates[tok_idx]})
                    ) >>> 8;

                    if (ch_idx == HEAD_DIM - 1) begin
                        ch_idx <= 0;
                        if (tok_idx == n_tokens - 1) begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            tok_idx <= tok_idx + 1;
                        end
                    end else begin
                        ch_idx <= ch_idx + 1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule