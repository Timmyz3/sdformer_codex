// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c1s_edc_delta_fuse — EDC-ΔFuse (Card I2)
// INNOVATION: Multi-scale temporal Δ-feat fused with low-res corr → detail_hit /
//             exact_boost before exact_capture. Orthogonal to TID (corr-free).
// Cite: EDCFlow arXiv:2506.03512 (2025). Enhance CFP/OGEC — do NOT replace.
// Ablation: ABLATE_CORR_ONLY / ABLATE_DIFF_ONLY.
// NOT claimed: beat EDCFlow AEE on ASIC; high-res full-pair corr volume.

`timescale 1ns/1ps
`default_nettype none

module c1s_edc_delta_fuse #(
  parameter int N_TILE   = 8,
  parameter int FEAT_W   = 8,
  parameter int CORR_W   = 16,
  parameter int MOT_W    = 8,
  parameter int N_SCALE  = 3,
  parameter logic [MOT_W-1:0] TH_DETAIL = 8'd32,
  parameter bit ABLATE_CORR_ONLY = 1'b0,
  parameter bit ABLATE_DIFF_ONLY = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [N_TILE*FEAT_W-1:0]     feat_t0_bus,
  input  logic [N_TILE*FEAT_W-1:0]     feat_tn_bus,
  input  logic [N_TILE*CORR_W-1:0]     corr_lo_bus,
  input  logic [N_TILE-1:0]            match_ok,
  output logic [N_TILE*MOT_W-1:0]      mot_diff_bus,
  output logic [N_TILE*MOT_W-1:0]      mot_fuse_bus,
  output logic [N_TILE-1:0]            detail_hit,
  output logic [N_TILE-1:0]            exact_boost,
  output logic                         fuse_valid
);

  logic [N_TILE*MOT_W-1:0] diff_c, fuse_c;
  logic [N_TILE-1:0] hit_c, boost_c;
  integer t, s, nbr;
  logic signed [FEAT_W-1:0] f0, fn, f0n, fnn;
  logic signed [FEAT_W:0]   dx;
  logic [FEAT_W:0]          abs_d;
  logic [MOT_W-1:0]         d_i, d_s, d_acc, corr_q, fuse_i;
  logic [CORR_W-1:0]        corr_i;

  always @(*) begin
    diff_c  = {N_TILE*MOT_W{1'b0}};
    fuse_c  = {N_TILE*MOT_W{1'b0}};
    hit_c   = {N_TILE{1'b0}};
    boost_c = {N_TILE{1'b0}};
    for (t = 0; t < N_TILE; t = t + 1) begin
      f0 = feat_t0_bus[t*FEAT_W +: FEAT_W];
      fn = feat_tn_bus[t*FEAT_W +: FEAT_W];
      dx = $signed({f0[FEAT_W-1], f0}) - $signed({fn[FEAT_W-1], fn});
      if (dx[FEAT_W])
        abs_d = (~dx) + {{FEAT_W{1'b0}}, 1'b1};
      else
        abs_d = {1'b0, dx[FEAT_W-1:0]};
      d_i = (abs_d > {{(FEAT_W+1-MOT_W){1'b0}}, {MOT_W{1'b1}}})
              ? {MOT_W{1'b1}} : abs_d[MOT_W-1:0];

      // Multi-scale Δ: average |Δ| with neighbor tiles at stride≈{1,2,≈N/2}
      d_acc = d_i;
      for (s = 1; s < N_SCALE; s = s + 1) begin
        nbr = t + (s == 1 ? 1 : (s == 2 ? 2 : (N_TILE/2)));
        if (nbr >= N_TILE) nbr = nbr - N_TILE;
        f0n = feat_t0_bus[nbr*FEAT_W +: FEAT_W];
        fnn = feat_tn_bus[nbr*FEAT_W +: FEAT_W];
        dx = $signed({f0n[FEAT_W-1], f0n}) - $signed({fnn[FEAT_W-1], fnn});
        if (dx[FEAT_W])
          abs_d = (~dx) + {{FEAT_W{1'b0}}, 1'b1};
        else
          abs_d = {1'b0, dx[FEAT_W-1:0]};
        d_s = (abs_d > {{(FEAT_W+1-MOT_W){1'b0}}, {MOT_W{1'b1}}})
                ? {MOT_W{1'b1}} : abs_d[MOT_W-1:0];
        d_acc = (d_acc + d_s) >> 1;
      end

      corr_i = corr_lo_bus[t*CORR_W +: CORR_W];
      // Compress corr_lo into MOT_W (low-res corr cue)
      if (corr_i > {{(CORR_W-MOT_W){1'b0}}, {MOT_W{1'b1}}})
        corr_q = {MOT_W{1'b1}};
      else
        corr_q = corr_i[MOT_W-1:0];

      if (ABLATE_CORR_ONLY)
        fuse_i = corr_q;
      else if (ABLATE_DIFF_ONLY)
        fuse_i = d_acc;
      else
        // Adaptive fuse: (3*Δ + corr)/4 — Δ-feat heavy, corr assist
        fuse_i = (({2'b0, d_acc} + {2'b0, d_acc} + {2'b0, d_acc} + {2'b0, corr_q}) >> 2);

      diff_c[t*MOT_W +: MOT_W] = d_acc;
      fuse_c[t*MOT_W +: MOT_W] = fuse_i;
      hit_c[t] = (fuse_i >= TH_DETAIL);
      // Boost exact on detail tiles that matched (refine-before-exact assist)
      // unmatched detail also boosts (uncertain texture → exact)
      boost_c[t] = hit_c[t] & (match_ok[t] | (fuse_i >= (TH_DETAIL + (TH_DETAIL >> 2))));
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fuse_valid   <= 1'b0;
      mot_diff_bus <= '0;
      mot_fuse_bus <= '0;
      detail_hit   <= '0;
      exact_boost  <= '0;
    end else begin
      fuse_valid <= valid_i;
      if (valid_i) begin
        mot_diff_bus <= diff_c;
        mot_fuse_bus <= fuse_c;
        detail_hit   <= hit_c;
        exact_boost  <= boost_c; // detail_hit → exact_boost (combine with CFP in glue/TB)
      end
    end
  end

endmodule

`default_nettype wire
