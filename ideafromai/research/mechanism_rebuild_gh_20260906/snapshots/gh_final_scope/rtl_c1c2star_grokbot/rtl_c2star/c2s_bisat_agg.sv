// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c2s_bisat_agg — BiSAT-Agg bidirectional temporal aggregation (Card H1)
// INNOVATION: Forward TMA seed + backward feat consistency fuse (SATMA-style)
//             under spike/OF schedule — upgrades c2s_tma_agg, does NOT replace it.
// Policy:
//   bwd_en_i=0 | ABLATE_FWD_ONLY → fuse_flow≈tma_agg_flow, fuse_hit≈0
//   agree (fwd==bwd≠0) → hit++, fuse=fwd (or blend with tma)
//   disagree (both nz, unequal) → suppress fuse toward 0 / tma>>1; edge → occ_exit
// Ports PACKED (iverilog/yosys-friendly): tile i at [i*W +: W].
// NOT claimed: first temporal OF HW; BAT GPU AEE; DualRail-CIM.

`timescale 1ns/1ps
`default_nettype none

module c2s_bisat_agg #(
  parameter int N_TILE   = 8,
  parameter int N_SLICE  = 4,
  parameter int FEAT_W   = 4,
  parameter int DIR_W    = 2,
  parameter int HIT_W    = 4,
  parameter logic [HIT_W-1:0] TH_FUSE = 4'd3,
  parameter bit  ABLATE_FWD_ONLY = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic                         bwd_en_i,
  input  logic [$clog2(N_SLICE)-1:0]   slice_idx,
  input  logic [N_TILE*FEAT_W-1:0]     feat_fwd_bus,
  input  logic [N_TILE*FEAT_W-1:0]     feat_bwd_bus,
  input  logic [N_TILE*DIR_W-1:0]      dir_code_bus,
  input  logic [N_TILE*FEAT_W-1:0]     tma_agg_flow_bus,
  output logic [N_TILE*FEAT_W-1:0]     fuse_flow_bus,
  output logic [N_TILE-1:0]            fuse_hit,
  output logic                         occ_exit_hint,
  output logic [1:0]                   hyp_hint_o,
  output logic                         fuse_valid
);

  logic [HIT_W-1:0]  fuse_cnt [N_TILE];
  logic [HIT_W-1:0]  fuse_cnt_n [N_TILE];
  logic [N_TILE*FEAT_W-1:0] fuse_c;
  logic [N_TILE-1:0]        hit_c;
  logic                     occ_c;
  logic [1:0]               hyp_c;
  logic [HIT_W-1:0]         hit_pop, vote_l, vote_r, conflict_edge;
  logic                     use_bwd;
  integer                   ti;
  logic [FEAT_W-1:0]        fwd_i, bwd_i, tma_i, fuse_i;
  logic [DIR_W-1:0]         dir_i;
  logic                     agree_i, disagree_i, edge_i;
  logic [$clog2(N_SLICE)-1:0] slice_unused;

  always @(*) begin
    use_bwd = bwd_en_i && !ABLATE_FWD_ONLY;
    hit_pop = {HIT_W{1'b0}};
    vote_l  = {HIT_W{1'b0}};
    vote_r  = {HIT_W{1'b0}};
    conflict_edge = {HIT_W{1'b0}};
    fuse_c  = {N_TILE*FEAT_W{1'b0}};
    hit_c   = {N_TILE{1'b0}};
    occ_c   = 1'b0;
    slice_unused = slice_idx;

    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      fwd_i = feat_fwd_bus[ti*FEAT_W +: FEAT_W];
      bwd_i = feat_bwd_bus[ti*FEAT_W +: FEAT_W];
      tma_i = tma_agg_flow_bus[ti*FEAT_W +: FEAT_W];
      dir_i = dir_code_bus[ti*DIR_W +: DIR_W];
      edge_i = (ti == 0) || (ti == (N_TILE-1));

      if (!use_bwd) begin
        // FWD_ONLY / TMA-compat path: pass tma seed, no fuse hits
        fuse_i = tma_i;
        fuse_cnt_n[ti] = {HIT_W{1'b0}};
        hit_c[ti] = 1'b0;
      end else begin
        agree_i    = (fwd_i != {FEAT_W{1'b0}}) && (fwd_i == bwd_i);
        disagree_i = (fwd_i != {FEAT_W{1'b0}}) && (bwd_i != {FEAT_W{1'b0}}) && (fwd_i != bwd_i);

        if (agree_i) begin
          if (fuse_cnt[ti] == {HIT_W{1'b1}})
            fuse_cnt_n[ti] = fuse_cnt[ti];
          else
            fuse_cnt_n[ti] = fuse_cnt[ti] + {{(HIT_W-1){1'b0}}, 1'b1};
          // Blend: prefer agreed fwd; mild average with tma when tma nonzero
          if (tma_i != {FEAT_W{1'b0}})
            fuse_i = (fwd_i + tma_i) >> 1;
          else
            fuse_i = fwd_i;
        end else if (disagree_i) begin
          // Suppress inconsistent motion
          if (fuse_cnt[ti] != {HIT_W{1'b0}})
            fuse_cnt_n[ti] = fuse_cnt[ti] - {{(HIT_W-1){1'b0}}, 1'b1};
          else
            fuse_cnt_n[ti] = {HIT_W{1'b0}};
          fuse_i = tma_i >> 1;
          if (edge_i)
            conflict_edge = conflict_edge + {{(HIT_W-1){1'b0}}, 1'b1};
        end else begin
          // One-sided: keep tma / fwd seed
          fuse_cnt_n[ti] = fuse_cnt[ti];
          if (fwd_i != {FEAT_W{1'b0}})
            fuse_i = fwd_i;
          else
            fuse_i = tma_i;
        end

        hit_c[ti] = (fuse_cnt_n[ti] >= TH_FUSE);
      end

      fuse_c[ti*FEAT_W +: FEAT_W] = fuse_i;
      if (hit_c[ti]) begin
        hit_pop = hit_pop + {{(HIT_W-1){1'b0}}, 1'b1};
        if (dir_i == 2'd1)
          vote_l = vote_l + {{(HIT_W-1){1'b0}}, 1'b1};
        else if (dir_i == 2'd2)
          vote_r = vote_r + {{(HIT_W-1){1'b0}}, 1'b1};
      end
    end

    // Strong edge conflict → occlusion / exit hint for OGEC/SCI side-channel
    occ_c = use_bwd && (conflict_edge >= 2'd2);

    if (vote_r > vote_l)
      hyp_c = 2'd2;
    else if (vote_l > vote_r)
      hyp_c = 2'd1;
    else if (hit_pop != {HIT_W{1'b0}})
      hyp_c = 2'd0;
    else
      hyp_c = 2'd0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fuse_valid     <= 1'b0;
      occ_exit_hint  <= 1'b0;
      hyp_hint_o     <= 2'd0;
      fuse_hit       <= '0;
      fuse_flow_bus  <= '0;
      for (ti = 0; ti < N_TILE; ti = ti + 1)
        fuse_cnt[ti] <= '0;
    end else begin
      fuse_valid <= valid_i;
      if (valid_i) begin
        fuse_hit      <= hit_c;
        occ_exit_hint <= occ_c;
        hyp_hint_o    <= hyp_c;
        fuse_flow_bus <= fuse_c;
        for (ti = 0; ti < N_TILE; ti = ti + 1)
          fuse_cnt[ti] <= fuse_cnt_n[ti];
      end
    end
  end

endmodule

`default_nettype wire
