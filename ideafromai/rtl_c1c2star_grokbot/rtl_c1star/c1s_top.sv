// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Thin integration: OP-STW + ECP-QKV + MW-ΔBuf + OGEC + PRRC +
// exact_capture_wrap + c1s_stats (parallel side ports).

`timescale 1ns/1ps
`default_nettype none

module c1s_top #(
  parameter int N_TILE  = 64,
  parameter int FLOW_W  = 8,
  parameter int EVT_W   = 8,
  parameter int SCORE_W = 8,
  parameter int SAMP_W  = 8,
  parameter int N_LEVEL = 3,
  parameter int BUDGET_W = 8,
  parameter logic [BUDGET_W-1:0] INIT_BUDGET = 8'd16,
  // Exact capture
  parameter int CAP_CNT_W   = 16,
  parameter int CAPACITY    = 16,
  // Stats
  parameter int STAT_CNT_W  = 16
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic signed [FLOW_W-1:0]     flow_cur  [N_TILE],
  input  logic signed [FLOW_W-1:0]     flow_prev [N_TILE],
  input  logic        [EVT_W-1:0]      event_cnt [N_TILE],
  input  logic        [SCORE_W-1:0]    corr_score [N_TILE],
  input  logic                         use_wake,
  input  logic signed [SAMP_W-1:0]     ref_samp [N_TILE],
  input  logic signed [SAMP_W-1:0]     cur_samp [N_TILE],
  input  logic        [N_TILE-1:0]     match_ok,
  output logic        [N_TILE-1:0]     wake_bitmap,
  output logic                         wake_valid,
  output logic        [N_TILE-1:0]     proj_en_bitmap,
  output logic        [N_TILE-1:0]     skip_bitmap,
  output logic                         proj_valid,
  output logic signed [N_TILE*(SAMP_W+1)-1:0] delta,
  output logic        [N_TILE-1:0]     delta_nz,
  output logic                         delta_valid,
  output logic        [N_TILE-1:0]     exact_en,
  output logic        [N_TILE-1:0]     prop_en,
  output logic                         ogec_valid,
  // PRRC ledger (parallel side ports)
  input  logic                         prrc_valid_i,
  input  logic [N_LEVEL-1:0]           prrc_level_sel,
  input  logic [$clog2(N_LEVEL)-1:0]   prrc_level_idx,
  input  logic                         prrc_spend_i,
  input  logic                         prrc_refill_i,
  output logic [N_LEVEL*BUDGET_W-1:0]  prrc_budget,
  output logic                         prrc_allow_exact,
  output logic                         prrc_ledger_valid,
  // Exact-capture wrap (OGEC exact_en × PRRC allow_exact)
  input  logic                         cap_valid_i,
  input  logic                         capture_en,
  output logic [N_TILE-1:0]            cap_hit_bitmap,
  output logic [CAP_CNT_W-1:0]         capture_cnt,
  output logic                         cap_busy,
  output logic                         cap_done,
  // C1* stats window
  input  logic                         stats_valid_i,
  input  logic                         stats_window_en,
  input  logic                         stats_clear_i,
  output logic [STAT_CNT_W-1:0]        wake_pop_cnt,
  output logic [STAT_CNT_W-1:0]        proj_skip_cnt,
  output logic [STAT_CNT_W-1:0]        delta_nz_cnt,
  output logic                         stats_out_valid
);

  c1s_op_stw_predictor #(
    .N_TILE(N_TILE),
    .FLOW_W(FLOW_W),
    .EVT_W (EVT_W)
  ) u_op_stw (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_i    (valid_i),
    .flow_cur   (flow_cur),
    .flow_prev  (flow_prev),
    .event_cnt  (event_cnt),
    .wake_bitmap(wake_bitmap),
    .wake_valid (wake_valid)
  );

  c1s_ecp_qkv_predictor #(
    .N_TILE (N_TILE),
    .SCORE_W(SCORE_W)
  ) u_ecp_qkv (
    .clk           (clk),
    .rst_n         (rst_n),
    .valid_i       (valid_i),
    .corr_score    (corr_score),
    .wake_bitmap   (wake_bitmap),
    .use_wake      (use_wake),
    .proj_en_bitmap(proj_en_bitmap),
    .skip_bitmap   (skip_bitmap),
    .proj_valid    (proj_valid)
  );

  c1s_mw_delta_buf #(
    .N_TILE(N_TILE),
    .SAMP_W(SAMP_W)
  ) u_mw_delta (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_i    (valid_i),
    .ref_samp   (ref_samp),
    .cur_samp   (cur_samp),
    .delta      (delta),
    .delta_nz   (delta_nz),
    .delta_valid(delta_valid)
  );

  c1s_ogec_gate #(
    .N_TILE(N_TILE)
  ) u_ogec (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_i   (valid_i),
    .match_ok  (match_ok),
    .exact_en  (exact_en),
    .prop_en   (prop_en),
    .ogec_valid(ogec_valid)
  );

  c1s_prrc_ledger #(
    .N_LEVEL    (N_LEVEL),
    .BUDGET_W   (BUDGET_W),
    .INIT_BUDGET(INIT_BUDGET)
  ) u_prrc (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (prrc_valid_i),
    .level_sel   (prrc_level_sel),
    .level_idx   (prrc_level_idx),
    .spend_i     (prrc_spend_i),
    .refill_i    (prrc_refill_i),
    .budget      (prrc_budget),
    .allow_exact (prrc_allow_exact),
    .ledger_valid(prrc_ledger_valid)
  );

  c1s_exact_capture_wrap #(
    .N_TILE  (N_TILE),
    .CNT_W   (CAP_CNT_W),
    .CAPACITY(CAPACITY)
  ) u_exact_cap (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_i    (cap_valid_i),
    .capture_en (capture_en),
    .exact_en   (exact_en),
    .allow_exact(prrc_allow_exact),
    .hit_bitmap (cap_hit_bitmap),
    .capture_cnt(capture_cnt),
    .busy       (cap_busy),
    .done       (cap_done)
  );

  c1s_stats #(
    .N_TILE(N_TILE),
    .CNT_W (STAT_CNT_W)
  ) u_stats (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (stats_valid_i),
    .window_en   (stats_window_en),
    .clear_i     (stats_clear_i),
    .wake_bitmap (wake_bitmap),
    .skip_bitmap (skip_bitmap),
    .delta_nz    (delta_nz),
    .wake_pop_cnt(wake_pop_cnt),
    .proj_skip_cnt(proj_skip_cnt),
    .delta_nz_cnt(delta_nz_cnt),
    .stats_valid (stats_out_valid)
  );

  // Serial front-end chain available as c1s_front_pipe

endmodule

`default_nettype wire
