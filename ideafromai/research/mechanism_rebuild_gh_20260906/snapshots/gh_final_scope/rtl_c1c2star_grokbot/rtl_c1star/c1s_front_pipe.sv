// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c1s_front_pipe — thin C1* front-end chain
// OP-STW → ECP-QKV (use_wake=1, wake|delta_nz) ; MW-ΔBuf parallel w/ OP-STW
// tile_active = wake | proj_en | delta_nz (valid with proj_valid).

`timescale 1ns/1ps
`default_nettype none

module c1s_front_pipe #(
  parameter int N_TILE  = 8,
  parameter int FLOW_W  = 8,
  parameter int EVT_W   = 8,
  parameter int SCORE_W = 8,
  parameter int SAMP_W  = 8
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic signed [FLOW_W-1:0]     flow_cur  [N_TILE],
  input  logic signed [FLOW_W-1:0]     flow_prev [N_TILE],
  input  logic        [EVT_W-1:0]      event_cnt [N_TILE],
  input  logic        [SCORE_W-1:0]    corr_score [N_TILE],
  input  logic signed [SAMP_W-1:0]     ref_samp [N_TILE],
  input  logic signed [SAMP_W-1:0]     cur_samp [N_TILE],
  // Stage taps
  output logic        [N_TILE-1:0]     wake_bitmap,
  output logic                         wake_valid,
  output logic        [N_TILE-1:0]     proj_en_bitmap,
  output logic        [N_TILE-1:0]     skip_bitmap,
  output logic                         proj_valid,
  output logic signed [N_TILE*(SAMP_W+1)-1:0] delta,
  output logic        [N_TILE-1:0]     delta_nz,
  output logic                         delta_valid,
  // Combined activity for downstream (PE wake / exact schedule)
  output logic        [N_TILE-1:0]     tile_active,
  output logic                         tile_active_valid
);

  // Hold corr_score so ECP (1 cycle later) still sees the sample
  logic [SCORE_W-1:0] corr_held [N_TILE];
  integer ci;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (ci = 0; ci < N_TILE; ci++)
        corr_held[ci] <= '0;
    end else if (valid_i) begin
      for (ci = 0; ci < N_TILE; ci++)
        corr_held[ci] <= corr_score[ci];
    end
  end

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

  // Residual nz ORs into wake for ECP (same cycle as wake_valid)
  logic [N_TILE-1:0] wake_for_ecp;
  assign wake_for_ecp = wake_bitmap | delta_nz;

  c1s_ecp_qkv_predictor #(
    .N_TILE (N_TILE),
    .SCORE_W(SCORE_W)
  ) u_ecp_qkv (
    .clk           (clk),
    .rst_n         (rst_n),
    .valid_i       (wake_valid),
    .corr_score    (corr_held),
    .wake_bitmap   (wake_for_ecp),
    .use_wake      (1'b1),
    .proj_en_bitmap(proj_en_bitmap),
    .skip_bitmap   (skip_bitmap),
    .proj_valid    (proj_valid)
  );

  // Hold wake|delta_nz from stage-1; OR with proj_en when stage-2 lands
  logic [N_TILE-1:0] held_wake_or_nz;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      held_wake_or_nz <= '0;
    else if (wake_valid)
      held_wake_or_nz <= wake_for_ecp;
  end

  // Combinational OR of registered stage outs (proj_en updates with proj_valid)
  assign tile_active       = held_wake_or_nz | proj_en_bitmap;
  assign tile_active_valid = proj_valid;

endmodule

`default_nettype wire
