// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_nl_stmfa — nonlinear STMFA-style residual wake (Card I)
// INNOVATION: Beyond linear MW/TMA — multi-scale previous-feature stub + digital
//             nonlinear residual mask → wake / MW path.
// Honest: algo-inspired digital sketch (E-NMSTFlow / STMFA ICRA'25), NOT ICRA AEE claim.
// Nonlinear sketch: r = |f_cur - f_prev| + (|f_cur - f_prev| >> 1)  (≈1.5× abs; soft saturating)
// Multi-scale: OR of per-scale residual>TH into nl_wake; amfe_w = scale hit one-hot-ish.
// NOT claimed: unsupervised loss; FPGA AEE; CIM.

`timescale 1ns/1ps
`default_nettype none

module c1s_nl_stmfa #(
  parameter int N_TILE  = 8,
  parameter int N_SCALE = 4,
  parameter int FEAT_W  = 8,
  parameter int RES_W   = 8,
  parameter logic [RES_W-1:0] TH_WAKE = 8'd12,
  parameter bit ABLATE_LINEAR_ONLY = 1'b0  // 1 = use |diff| only (no nonlinear boost)
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  // Packed: scale s, tile t at [(s*N_TILE + t)*FEAT_W +: FEAT_W]
  input  logic [N_SCALE*N_TILE*FEAT_W-1:0] f_prev_bus,
  input  logic [N_TILE*FEAT_W-1:0]     f_cur_bus,
  // Optional linear seed (TMA/MW) — used as floor for residual compare
  input  logic [N_TILE*FEAT_W-1:0]     f_lin_bus,
  output logic [N_TILE-1:0]            nl_wake,
  output logic [N_TILE*RES_W-1:0]      r_nl_bus,
  output logic [N_SCALE-1:0]           amfe_w,      // which scales contributed (frame-agg)
  output logic                         nl_valid
);

  logic [N_TILE-1:0] wake_c;
  logic [N_TILE*RES_W-1:0] r_c;
  logic [N_SCALE-1:0] amfe_c;
  integer t, s;
  logic signed [FEAT_W-1:0] cur_i, prev_i, lin_i;
  logic signed [FEAT_W:0]   diff_x, diff_lin;
  logic [FEAT_W:0]          abs_x, abs_lin;
  logic [RES_W:0]           r_boost;
  logic [RES_W-1:0]         r_sat, r_best;
  logic                     hit_s;

  always @(*) begin
    wake_c = {N_TILE{1'b0}};
    r_c    = {N_TILE*RES_W{1'b0}};
    amfe_c = {N_SCALE{1'b0}};
    for (t = 0; t < N_TILE; t = t + 1) begin
      cur_i = f_cur_bus[t*FEAT_W +: FEAT_W];
      lin_i = f_lin_bus[t*FEAT_W +: FEAT_W];
      r_best = {RES_W{1'b0}};
      for (s = 0; s < N_SCALE; s = s + 1) begin
        prev_i = f_prev_bus[(s*N_TILE + t)*FEAT_W +: FEAT_W];
        diff_x = $signed({cur_i[FEAT_W-1], cur_i}) - $signed({prev_i[FEAT_W-1], prev_i});
        if (diff_x[FEAT_W])
          abs_x = (~diff_x) + {{FEAT_W{1'b0}}, 1'b1};
        else
          abs_x = {1'b0, diff_x[FEAT_W-1:0]};

        // Nonlinear residual sketch vs linear floor
        if (ABLATE_LINEAR_ONLY)
          r_boost = {1'b0, abs_x[RES_W-1:0]};
        else
          r_boost = {1'b0, abs_x[RES_W-1:0]} + {2'b0, abs_x[RES_W-1:1]}; // ≈1.5|d|

        // Soft add distance from linear seed (motion-guided)
        diff_lin = $signed({cur_i[FEAT_W-1], cur_i}) - $signed({lin_i[FEAT_W-1], lin_i});
        if (diff_lin[FEAT_W])
          abs_lin = (~diff_lin) + {{FEAT_W{1'b0}}, 1'b1};
        else
          abs_lin = {1'b0, diff_lin[FEAT_W-1:0]};
        if (!ABLATE_LINEAR_ONLY)
          r_boost = r_boost + {2'b0, abs_lin[RES_W-1:2]}; // +|cur-lin|/4

        if (r_boost > {1'b0, {RES_W{1'b1}}})
          r_sat = {RES_W{1'b1}};
        else
          r_sat = r_boost[RES_W-1:0];

        hit_s = (r_sat >= TH_WAKE);
        if (hit_s)
          amfe_c[s] = 1'b1;
        if (r_sat > r_best)
          r_best = r_sat;
      end
      r_c[t*RES_W +: RES_W] = r_best;
      wake_c[t] = (r_best >= TH_WAKE);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      nl_valid <= 1'b0;
      nl_wake  <= '0;
      r_nl_bus <= '0;
      amfe_w   <= '0;
    end else begin
      nl_valid <= valid_i;
      if (valid_i) begin
        nl_wake  <= wake_c;
        r_nl_bus <= r_c;
        amfe_w   <= amfe_c;
      end
    end
  end

endmodule

`default_nettype wire
