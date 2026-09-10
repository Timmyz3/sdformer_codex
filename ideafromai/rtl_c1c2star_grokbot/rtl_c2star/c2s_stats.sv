// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_stats — C2* window counters (mac_en beats / skip / gate fires)
// INNOVATION: measures HBG/SMAM mac_en density, ADP skip, and SP/STH gate fires —
//             dual-rail gate+payload activity, NOT multiply-reorder stats.
// Over a window (window_en): accumulate event counts; clear_i / !window_en resets.

`timescale 1ns/1ps
`default_nettype none

module c2s_stats #(
  parameter int CNT_W = 16
) (
  input  logic             clk,
  input  logic             rst_n,
  input  logic             valid_i,
  input  logic             window_en,
  input  logic             clear_i,
  input  logic             mac_en,      // SMAM/HBG beat enable
  input  logic             skipped,     // ADP-MAC skip (or !do_mac)
  input  logic             gate_fire,   // SP-Gate / STH any-run / dual fire
  output logic [CNT_W-1:0] mac_en_cnt,
  output logic [CNT_W-1:0] skip_cnt,
  output logic [CNT_W-1:0] gate_fire_cnt,
  output logic             stats_valid
);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mac_en_cnt    <= {CNT_W{1'b0}};
      skip_cnt      <= {CNT_W{1'b0}};
      gate_fire_cnt <= {CNT_W{1'b0}};
      stats_valid   <= 1'b0;
    end else if (clear_i || !window_en) begin
      mac_en_cnt    <= {CNT_W{1'b0}};
      skip_cnt      <= {CNT_W{1'b0}};
      gate_fire_cnt <= {CNT_W{1'b0}};
      stats_valid   <= 1'b0;
    end else begin
      stats_valid <= valid_i;
      if (valid_i) begin
        if (mac_en)
          mac_en_cnt <= mac_en_cnt + {{(CNT_W-1){1'b0}}, 1'b1};
        if (skipped)
          skip_cnt <= skip_cnt + {{(CNT_W-1){1'b0}}, 1'b1};
        if (gate_fire)
          gate_fire_cnt <= gate_fire_cnt + {{(CNT_W-1){1'b0}}, 1'b1};
      end
    end
  end

endmodule

`default_nettype wire
