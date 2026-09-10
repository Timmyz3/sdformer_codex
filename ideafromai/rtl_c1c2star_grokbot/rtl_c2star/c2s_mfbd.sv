// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_mfbd — Motion-bundle delivery (steer payload to hyp destination lane)
// Accepts Motion-TTB-style packed bundle descriptors; on valid, routes payload
// using bundle[0]'s hyp_id to a one-hot lane enable. Synthesizable; packed I/O.
// bundle_tile / bundle_dt are accepted for TTB interface compatibility (hyp steers).

`timescale 1ns/1ps
`default_nettype none

module c2s_mfbd #(
  parameter int MAX_B  = 4,
  parameter int TILE_W = 4,
  parameter int DT_W   = 3,
  parameter int HYP_W  = 2,
  parameter int PAY_W  = 8,
  parameter int N_HYP  = (1 << HYP_W),
  parameter int CNT_W  = $clog2(MAX_B + 1)
) (
  input  logic                              clk,
  input  logic                              rst_n,
  input  logic                              valid_i,
  input  logic [CNT_W-1:0]                  bundle_count,
  input  logic [MAX_B*TILE_W-1:0]           bundle_tile,
  input  logic [MAX_B*DT_W-1:0]             bundle_dt,
  input  logic [MAX_B*HYP_W-1:0]            bundle_hyp,
  input  logic signed [PAY_W-1:0]           payload,
  output logic [N_HYP-1:0]                  lane_valid,
  output logic signed [PAY_W-1:0]           lane_payload,
  output logic                              deliver_valid
);

  logic [HYP_W-1:0] hyp0;
  logic [N_HYP-1:0] lane_oh;
  logic             have_bundle;

  // Simple delivery: bundle[0] hyp_id selects destination lane
  assign hyp0        = bundle_hyp[0 +: HYP_W];
  assign have_bundle = (bundle_count != '0);
  assign lane_oh     = have_bundle ? ({{(N_HYP-1){1'b0}}, 1'b1} << hyp0) : '0;

  // Suppress unused-input warnings while keeping TTB-compatible ports
  /* verilator lint_off UNUSED */
  wire _unused_tile = |bundle_tile;
  wire _unused_dt   = |bundle_dt;
  /* verilator lint_on UNUSED */

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lane_valid    <= '0;
      lane_payload  <= '0;
      deliver_valid <= 1'b0;
    end else begin
      deliver_valid <= valid_i;
      if (valid_i) begin
        lane_valid   <= lane_oh;
        lane_payload <= payload;
      end else begin
        lane_valid <= '0;
      end
    end
  end

endmodule

`default_nettype wire
