// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Thin integration: c2s_back_pipe (HBG→SMAM + STH parallel) + Motion-TTB side
// + ADP-MAC / ARM-Acc / MFBD / SP-Gate / c2s_stats parallel side ports
// (not inside back_pipe — keeps back_pipe TB green).

`timescale 1ns/1ps
`default_nettype none

module c2s_top #(
  parameter int                 AMP_W       = 8,
  parameter int                 ACC_W       = 16,
  parameter int                 N_TILE      = 8,
  parameter int                 DT_W        = 3,
  parameter int                 HYP_W       = 2,
  parameter int                 MAX_BUNDLES = 8,
  parameter int                 TILE_W      = (N_TILE <= 1) ? 1 : $clog2(N_TILE),
  parameter int                 CNT_W       = $clog2(MAX_BUNDLES + 1),
  parameter int                 N_HEAD      = 8,
  parameter int                 SCORE_W     = 8,
  parameter logic [SCORE_W-1:0] TH_SP       = 8'd2,
  parameter logic [SCORE_W-1:0] TH_TP       = 8'd2,
  parameter int                 W_A         = 8,
  parameter int                 W_B         = 8,
  parameter int                 W_ACC       = 16,
  // ARM-Acc
  parameter int                 ARM_N_HYP   = 4,
  parameter int                 ARM_ACC_W   = 16,
  parameter int                 ARM_DATA_W  = 8,
  parameter int                 ARM_SEL_W   = (ARM_N_HYP <= 1) ? 1 : $clog2(ARM_N_HYP),
  // MFBD
  parameter int                 MFBD_MAX_B  = 4,
  parameter int                 MFBD_TILE_W = 4,
  parameter int                 MFBD_PAY_W  = 8,
  parameter int                 MFBD_N_HYP  = (1 << HYP_W),
  parameter int                 MFBD_CNT_W  = $clog2(MFBD_MAX_B + 1),
  // SP-Gate
  parameter int                 SP_N        = 8,
  parameter int                 SP_MASS_W   = 8,
  parameter logic [SP_MASS_W-1:0] SP_TH_M   = 8'd2,
  // Stats
  parameter int                 STAT_CNT_W  = 16
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              amp_valid,
  input  logic signed [AMP_W-1:0]           amp,
  output logic                              g,
  output logic signed [AMP_W-1:0]           p,
  output logic                              gp_valid,
  output logic                              pe_clk_en,
  output logic                              mac_en,
  output logic                              mask_add_en,
  output logic signed [AMP_W-1:0]           payload_q,
  output logic signed [ACC_W-1:0]           mask_add_result,
  output logic                              smam_out_valid,
  // Motion-TTB midend (side port — not inside back_pipe)
  input  logic                              ttb_valid_i,
  input  logic [N_TILE-1:0]                 ttb_wake_bitmap,
  input  logic [DT_W-1:0]                   ttb_dt_bin  [N_TILE],
  input  logic [HYP_W-1:0]                  ttb_hyp_id  [N_TILE],
  output logic [MAX_BUNDLES*TILE_W-1:0]     ttb_bundle_tile,
  output logic [MAX_BUNDLES*DT_W-1:0]       ttb_bundle_dt,
  output logic [MAX_BUNDLES*HYP_W-1:0]      ttb_bundle_hyp,
  output logic [CNT_W-1:0]                  ttb_bundle_count,
  output logic                              ttb_bundle_valid,
  // STH-Gate (via back_pipe)
  input  logic                              sth_valid_i,
  input  logic [SCORE_W-1:0]                sth_spat_score [N_HEAD],
  input  logic [SCORE_W-1:0]                sth_temp_score [N_HEAD],
  output logic [N_HEAD-1:0]                 sth_spat_en,
  output logic [N_HEAD-1:0]                 sth_temp_en,
  output logic [N_HEAD-1:0]                 sth_dual_en,
  output logic [N_HEAD-1:0]                 sth_skip_en,
  output logic                              sth_gate_valid,
  // ADP-MAC (parallel side ports — not inside back_pipe)
  input  logic                              adp_valid_i,
  input  logic signed [W_A-1:0]             adp_a,
  input  logic signed [W_B-1:0]             adp_b,
  input  logic                              adp_skip_a,
  input  logic                              adp_skip_b,
  input  logic                              adp_mac_en,
  input  logic                              adp_clear_i,
  output logic signed [W_ACC-1:0]           adp_acc,
  output logic                              adp_skipped,
  output logic                              adp_out_valid,
  // ARM-Acc side ports
  input  logic                              arm_valid_i,
  input  logic                              arm_clear_i,
  input  logic [ARM_SEL_W-1:0]              arm_hyp_sel,
  input  logic signed [ARM_DATA_W-1:0]      arm_data_i,
  input  logic                              arm_add_en,
  output logic [ARM_N_HYP*ARM_ACC_W-1:0]    arm_acc_bus,
  output logic                              arm_out_valid,
  // MFBD side ports
  input  logic                              mfbd_valid_i,
  input  logic [MFBD_CNT_W-1:0]             mfbd_bundle_count,
  input  logic [MFBD_MAX_B*MFBD_TILE_W-1:0] mfbd_bundle_tile,
  input  logic [MFBD_MAX_B*DT_W-1:0]        mfbd_bundle_dt,
  input  logic [MFBD_MAX_B*HYP_W-1:0]       mfbd_bundle_hyp,
  input  logic signed [MFBD_PAY_W-1:0]      mfbd_payload,
  output logic [MFBD_N_HYP-1:0]             mfbd_lane_valid,
  output logic signed [MFBD_PAY_W-1:0]      mfbd_lane_payload,
  output logic                              mfbd_deliver_valid,
  // SP-Gate side ports
  input  logic                              sp_valid_i,
  input  logic [SP_MASS_W-1:0]              sp_mass [SP_N],
  input  logic [SP_N-1:0]                   sp_force_bitmap,
  output logic [SP_N-1:0]                   sp_run_en,
  output logic [SP_N-1:0]                   sp_drop_en,
  output logic                              sp_gate_valid,
  // C2* stats window
  input  logic                              stats_valid_i,
  input  logic                              stats_window_en,
  input  logic                              stats_clear_i,
  input  logic                              stats_mac_en,
  input  logic                              stats_skipped,
  input  logic                              stats_gate_fire,
  output logic [STAT_CNT_W-1:0]             mac_en_cnt,
  output logic [STAT_CNT_W-1:0]             skip_cnt,
  output logic [STAT_CNT_W-1:0]             gate_fire_cnt,
  output logic                              stats_out_valid
);

  c2s_back_pipe #(
    .AMP_W  (AMP_W),
    .ACC_W  (ACC_W),
    .N_HEAD (N_HEAD),
    .SCORE_W(SCORE_W),
    .TH_SP  (TH_SP),
    .TH_TP  (TH_TP)
  ) u_back_pipe (
    .clk            (clk),
    .rst_n          (rst_n),
    .amp_valid      (amp_valid),
    .amp            (amp),
    .g              (g),
    .p              (p),
    .gp_valid       (gp_valid),
    .pe_clk_en      (pe_clk_en),
    .mac_en         (mac_en),
    .mask_add_en    (mask_add_en),
    .mac_payload    (payload_q),
    .mask_add_result(mask_add_result),
    .smam_out_valid (smam_out_valid),
    .sth_valid_i    (sth_valid_i),
    .sth_spat_score (sth_spat_score),
    .sth_temp_score (sth_temp_score),
    .sth_spat_en    (sth_spat_en),
    .sth_temp_en    (sth_temp_en),
    .sth_dual_en    (sth_dual_en),
    .sth_skip_en    (sth_skip_en),
    .sth_gate_valid (sth_gate_valid)
  );

  c2s_motion_ttb_packer #(
    .N_TILE     (N_TILE),
    .DT_W       (DT_W),
    .HYP_W      (HYP_W),
    .MAX_BUNDLES(MAX_BUNDLES)
  ) u_motion_ttb (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (ttb_valid_i),
    .wake_bitmap (ttb_wake_bitmap),
    .dt_bin      (ttb_dt_bin),
    .hyp_id      (ttb_hyp_id),
    .bundle_tile (ttb_bundle_tile),
    .bundle_dt   (ttb_bundle_dt),
    .bundle_hyp  (ttb_bundle_hyp),
    .bundle_count(ttb_bundle_count),
    .bundle_valid(ttb_bundle_valid)
  );

  c2s_adp_mac #(
    .W_A  (W_A),
    .W_B  (W_B),
    .W_ACC(W_ACC)
  ) u_adp_mac (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_i  (adp_valid_i),
    .a        (adp_a),
    .b        (adp_b),
    .skip_a   (adp_skip_a),
    .skip_b   (adp_skip_b),
    .mac_en   (adp_mac_en),
    .clear_i  (adp_clear_i),
    .acc      (adp_acc),
    .skipped  (adp_skipped),
    .out_valid(adp_out_valid)
  );

  c2s_arm_acc #(
    .N_HYP (ARM_N_HYP),
    .ACC_W (ARM_ACC_W),
    .DATA_W(ARM_DATA_W)
  ) u_arm_acc (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_i  (arm_valid_i),
    .clear_i  (arm_clear_i),
    .hyp_sel  (arm_hyp_sel),
    .data_i   (arm_data_i),
    .add_en   (arm_add_en),
    .acc_bus  (arm_acc_bus),
    .out_valid(arm_out_valid)
  );

  c2s_mfbd #(
    .MAX_B (MFBD_MAX_B),
    .TILE_W(MFBD_TILE_W),
    .DT_W  (DT_W),
    .HYP_W (HYP_W),
    .PAY_W (MFBD_PAY_W)
  ) u_mfbd (
    .clk          (clk),
    .rst_n        (rst_n),
    .valid_i      (mfbd_valid_i),
    .bundle_count (mfbd_bundle_count),
    .bundle_tile  (mfbd_bundle_tile),
    .bundle_dt    (mfbd_bundle_dt),
    .bundle_hyp   (mfbd_bundle_hyp),
    .payload      (mfbd_payload),
    .lane_valid   (mfbd_lane_valid),
    .lane_payload (mfbd_lane_payload),
    .deliver_valid(mfbd_deliver_valid)
  );

  c2s_sp_gate #(
    .N     (SP_N),
    .MASS_W(SP_MASS_W),
    .TH_M  (SP_TH_M)
  ) u_sp_gate (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (sp_valid_i),
    .mass        (sp_mass),
    .force_bitmap(sp_force_bitmap),
    .run_en      (sp_run_en),
    .drop_en     (sp_drop_en),
    .gate_valid  (sp_gate_valid)
  );

  c2s_stats #(
    .CNT_W(STAT_CNT_W)
  ) u_stats (
    .clk          (clk),
    .rst_n        (rst_n),
    .valid_i      (stats_valid_i),
    .window_en    (stats_window_en),
    .clear_i      (stats_clear_i),
    .mac_en       (stats_mac_en),
    .skipped      (stats_skipped),
    .gate_fire    (stats_gate_fire),
    .mac_en_cnt   (mac_en_cnt),
    .skip_cnt     (skip_cnt),
    .gate_fire_cnt(gate_fire_cnt),
    .stats_valid  (stats_out_valid)
  );

endmodule

`default_nettype wire
