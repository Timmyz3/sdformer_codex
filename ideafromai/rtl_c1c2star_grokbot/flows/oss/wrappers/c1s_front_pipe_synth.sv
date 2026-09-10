// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened front_pipe @ N_TILE=8 (Yosys + unpacked-array children).
// Instantiates OP-STW / ECP / MW flat synth wrappers + tile_active glue.

`timescale 1ns/1ps
`default_nettype none

module c1s_front_pipe_synth #(
  parameter int N_TILE  = 8,
  parameter int FLOW_W  = 8,
  parameter int EVT_W   = 8,
  parameter int SCORE_W = 8,
  parameter int SAMP_W  = 8
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic signed [N_TILE*FLOW_W-1:0]   flow_cur,
  input  logic signed [N_TILE*FLOW_W-1:0]   flow_prev,
  input  logic        [N_TILE*EVT_W-1:0]    event_cnt,
  input  logic        [N_TILE*SCORE_W-1:0]  corr_score,
  input  logic signed [N_TILE*SAMP_W-1:0]   ref_samp,
  input  logic signed [N_TILE*SAMP_W-1:0]   cur_samp,
  output logic        [N_TILE-1:0]          wake_bitmap,
  output logic                              wake_valid,
  output logic        [N_TILE-1:0]          proj_en_bitmap,
  output logic        [N_TILE-1:0]          skip_bitmap,
  output logic                              proj_valid,
  output logic signed [N_TILE*(SAMP_W+1)-1:0] delta,
  output logic        [N_TILE-1:0]          delta_nz,
  output logic                              delta_valid,
  output logic        [N_TILE-1:0]          tile_active,
  output logic                              tile_active_valid
);

  logic [N_TILE*SCORE_W-1:0] corr_held;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      corr_held <= '0;
    else if (valid_i)
      corr_held <= corr_score;
  end

  c1s_op_stw_predictor_synth #(
    .N_TILE(N_TILE), .FLOW_W(FLOW_W), .EVT_W(EVT_W)
  ) u_op_stw (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .flow_cur(flow_cur), .flow_prev(flow_prev), .event_cnt(event_cnt),
    .wake_bitmap(wake_bitmap), .wake_valid(wake_valid)
  );

  c1s_mw_delta_buf_synth #(
    .N_TILE(N_TILE), .SAMP_W(SAMP_W)
  ) u_mw_delta (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .ref_samp(ref_samp), .cur_samp(cur_samp),
    .delta(delta), .delta_nz(delta_nz), .delta_valid(delta_valid)
  );

  logic [N_TILE-1:0] wake_for_ecp;
  assign wake_for_ecp = wake_bitmap | delta_nz;

  c1s_ecp_qkv_predictor_synth #(
    .N_TILE(N_TILE), .SCORE_W(SCORE_W)
  ) u_ecp_qkv (
    .clk(clk), .rst_n(rst_n), .valid_i(wake_valid),
    .corr_score(corr_held), .wake_bitmap(wake_for_ecp), .use_wake(1'b1),
    .proj_en_bitmap(proj_en_bitmap), .skip_bitmap(skip_bitmap),
    .proj_valid(proj_valid)
  );

  logic [N_TILE-1:0] held_wake_or_nz;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      held_wake_or_nz <= '0;
    else if (wake_valid)
      held_wake_or_nz <= wake_for_ecp;
  end

  assign tile_active       = held_wake_or_nz | proj_en_bitmap;
  assign tile_active_valid = proj_valid;

endmodule

`default_nettype wire
