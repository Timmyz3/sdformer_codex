// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Synth-only flattened OP-STW: Yosys 0.52 cannot parse unpacked array ports.
// Functionally equivalent to c1s_op_stw_predictor with packed buses.
// Used ONLY by flows/oss synth; functional TB still targets the unpacked RTL.

`timescale 1ns/1ps
`default_nettype none

module c1s_op_stw_predictor_synth #(
  parameter int N_TILE = 64,
  parameter int FLOW_W = 8,
  parameter int EVT_W  = 8,
  parameter logic signed [FLOW_W-1:0] TH_W = 8'sd2,
  parameter logic [EVT_W-1:0] TH_E = 8'd1
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic signed [N_TILE*FLOW_W-1:0]   flow_cur,
  input  logic signed [N_TILE*FLOW_W-1:0]   flow_prev,
  input  logic        [N_TILE*EVT_W-1:0]    event_cnt,
  output logic        [N_TILE-1:0]          wake_bitmap,
  output logic                              wake_valid
);

  logic [N_TILE-1:0] wake_comb;

  genvar gi;
  generate
    for (gi = 0; gi < N_TILE; gi++) begin : g_tile
      wire signed [FLOW_W-1:0] fc = flow_cur[gi*FLOW_W +: FLOW_W];
      wire signed [FLOW_W-1:0] fp = flow_prev[gi*FLOW_W +: FLOW_W];
      wire        [EVT_W-1:0]  ec = event_cnt[gi*EVT_W +: EVT_W];
      wire signed [FLOW_W:0]   cur_sx  = $signed({fc[FLOW_W-1], fc});
      wire signed [FLOW_W:0]   prev_sx = $signed({fp[FLOW_W-1], fp});
      wire signed [FLOW_W:0]   diff_sx = cur_sx - prev_sx;
      wire        [FLOW_W:0]   abs_diff =
          diff_sx[FLOW_W] ? -$unsigned(diff_sx) : $unsigned(diff_sx);
      wire flow_wake_i = (abs_diff > $unsigned({{1'b0}, TH_W}));
      wire evt_wake_i  = (ec > TH_E);
      assign wake_comb[gi] = flow_wake_i || evt_wake_i;
    end
  endgenerate

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
