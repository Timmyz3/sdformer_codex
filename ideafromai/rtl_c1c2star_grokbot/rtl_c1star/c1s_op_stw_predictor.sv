// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh) / for Codex fill-in
// Date: 2026-09-05
// Module: c1s_op_stw_predictor — OP-STW optical-flow predictive spike-tile wake
// INNOVATION: PE/tile wake from flow-Δ + event density — NOT flat quota / zero-skip.
// Claim sentence: "Optical-flow predictive spike-tile wake schedules exact work."

`timescale 1ns/1ps
`default_nettype none

module c1s_op_stw_predictor #(
  parameter int N_TILE = 64,
  parameter int FLOW_W = 8,
  parameter int EVT_W  = 8,
  parameter logic signed [FLOW_W-1:0] TH_W = 8'sd2,
  parameter logic [EVT_W-1:0] TH_E = 8'd1
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic signed [FLOW_W-1:0]     flow_cur  [N_TILE],
  input  logic signed [FLOW_W-1:0]     flow_prev [N_TILE],
  input  logic        [EVT_W-1:0]      event_cnt [N_TILE],
  output logic        [N_TILE-1:0]     wake_bitmap,
  output logic                         wake_valid
);

  logic [N_TILE-1:0] wake_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      // Wider signed diff avoids FLOW_W overflow; abs handles most-negative
      wire signed [FLOW_W:0] cur_sx  = $signed({flow_cur[gi][FLOW_W-1],  flow_cur[gi]});
      wire signed [FLOW_W:0] prev_sx = $signed({flow_prev[gi][FLOW_W-1], flow_prev[gi]});
      wire signed [FLOW_W:0] diff_sx = cur_sx - prev_sx;
      wire        [FLOW_W:0] abs_diff =
          diff_sx[FLOW_W] ? -$unsigned(diff_sx) : $unsigned(diff_sx);
      // TH_W as non-negative magnitude; strict >
      wire flow_wake_i = (abs_diff > $unsigned({{1'b0}, TH_W}));
      wire evt_wake_i  = (event_cnt[gi] > TH_E);
      assign wake_comb[gi] = flow_wake_i || evt_wake_i;
    end
  endgenerate

  // Registered: wake_valid high one cycle after valid_i; reset clears wake
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wake_bitmap <= '0;
      wake_valid  <= 1'b0;
    end else begin
      wake_valid <= valid_i;
      if (valid_i) begin
        wake_bitmap <= wake_comb;
      end
    end
  end

endmodule

`default_nettype wire
