// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-06
// Module: c2s_tma_agg — TMA-Agg temporal motion aggregation (C2*)
// INNOVATION: First-HW sketch of TMA-style split + lookup-align + aggregate
//             under spike/OF schedule — digital, NOT ICCV GPU accuracy /
//             DualRail-CIM physical arrays.
// Ports PACKED (iverilog/yosys-friendly): tile i at [i*W +: W].

`timescale 1ns/1ps
`default_nettype none

module c2s_tma_agg #(
  parameter int N_TILE  = 8,
  parameter int N_SLICE = 4,
  parameter int FEAT_W  = 4,
  parameter int DIR_W   = 2,
  parameter int CNT_W   = 4,
  parameter int HIT_W   = 4,
  parameter logic [CNT_W-1:0] TH_CONS  = 4'd2,
  parameter logic [HIT_W-1:0] TH_EARLY = 4'd4
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [$clog2(N_SLICE)-1:0]   slice_idx,
  input  logic [N_TILE*FEAT_W-1:0]     feat_cur_bus,
  input  logic [N_TILE*DIR_W-1:0]      dir_code_bus,
  output logic [N_TILE*FEAT_W-1:0]     agg_flow_bus,
  output logic [N_TILE-1:0]            pattern_hit,
  output logic                         early_exit,
  output logic [1:0]                   hyp_hint,
  output logic                         agg_valid
);

  logic [FEAT_W-1:0] prev_feat [N_TILE];
  logic [CNT_W-1:0]  cons_cnt  [N_TILE];
  logic [$clog2(N_SLICE)-1:0] last_slice;

  logic [N_TILE*FEAT_W-1:0] agg_c;
  logic [N_TILE-1:0]        hit_c;
  logic [HIT_W-1:0]         hit_pop;
  logic [CNT_W-1:0]         cons_n [N_TILE];
  logic                     early_c;
  logic [1:0]               hyp_c;
  logic [HIT_W-1:0]         vote_l, vote_r;
  integer                   ti, src;
  logic [FEAT_W-1:0]        feat_i, aligned_i, prev_src;

  always @(*) begin
    hit_pop = {HIT_W{1'b0}};
    vote_l  = {HIT_W{1'b0}};
    vote_r  = {HIT_W{1'b0}};
    agg_c   = {N_TILE*FEAT_W{1'b0}};
    hit_c   = {N_TILE{1'b0}};
    for (ti = 0; ti < N_TILE; ti = ti + 1) begin
      feat_i = feat_cur_bus[ti*FEAT_W +: FEAT_W];
      // Lookup-align previous feature by dir
      case (dir_code_bus[ti*DIR_W +: DIR_W])
        2'd1: begin
          src = ti + 1;
          if (src < N_TILE)
            prev_src = prev_feat[src];
          else
            prev_src = prev_feat[ti];
        end
        2'd2: begin
          if (ti > 0)
            prev_src = prev_feat[ti-1];
          else
            prev_src = prev_feat[ti];
        end
        default: prev_src = prev_feat[ti];
      endcase
      aligned_i = prev_src;

      if ((feat_i != {FEAT_W{1'b0}}) && (feat_i == aligned_i)) begin
        if (cons_cnt[ti] == {CNT_W{1'b1}})
          cons_n[ti] = cons_cnt[ti];
        else
          cons_n[ti] = cons_cnt[ti] + {{(CNT_W-1){1'b0}}, 1'b1};
      end else if (feat_i != {FEAT_W{1'b0}}) begin
        if (cons_cnt[ti] != {CNT_W{1'b0}})
          cons_n[ti] = cons_cnt[ti] - {{(CNT_W-1){1'b0}}, 1'b1};
        else
          cons_n[ti] = {CNT_W{1'b0}};
      end else begin
        cons_n[ti] = cons_cnt[ti];
      end

      hit_c[ti] = (cons_n[ti] >= TH_CONS);
      if (hit_c[ti]) begin
        hit_pop = hit_pop + {{(HIT_W-1){1'b0}}, 1'b1};
        if (dir_code_bus[ti*DIR_W +: DIR_W] == 2'd1)
          vote_l = vote_l + {{(HIT_W-1){1'b0}}, 1'b1};
        else if (dir_code_bus[ti*DIR_W +: DIR_W] == 2'd2)
          vote_r = vote_r + {{(HIT_W-1){1'b0}}, 1'b1};
      end

      if (hit_c[ti])
        agg_c[ti*FEAT_W +: FEAT_W] = feat_i;
      else if (feat_i != {FEAT_W{1'b0}})
        agg_c[ti*FEAT_W +: FEAT_W] = feat_i;
      else
        agg_c[ti*FEAT_W +: FEAT_W] = aligned_i;
    end
    early_c = (hit_pop >= TH_EARLY);
    if (vote_r > vote_l)
      hyp_c = 2'd2;
    else if (vote_l > vote_r)
      hyp_c = 2'd1;
    else
      hyp_c = 2'd0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      agg_valid    <= 1'b0;
      early_exit   <= 1'b0;
      hyp_hint     <= 2'd0;
      pattern_hit  <= '0;
      agg_flow_bus <= '0;
      last_slice   <= '0;
      for (ti = 0; ti < N_TILE; ti = ti + 1) begin
        prev_feat[ti] <= '0;
        cons_cnt[ti]  <= '0;
      end
    end else begin
      agg_valid <= valid_i;
      if (valid_i) begin
        pattern_hit  <= hit_c;
        early_exit   <= early_c;
        hyp_hint     <= hyp_c;
        agg_flow_bus <= agg_c;
        last_slice   <= slice_idx;
        for (ti = 0; ti < N_TILE; ti = ti + 1) begin
          cons_cnt[ti]  <= cons_n[ti];
          prev_feat[ti] <= feat_cur_bus[ti*FEAT_W +: FEAT_W];
        end
      end
    end
  end

endmodule

`default_nettype wire
