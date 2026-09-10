// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_sth_gate — Spatial–Temporal Head Gate (C2*)
// Idea 08 Sparse VideoGen remake: classify each head/token lane as
// spatial vs temporal and emit enable masks for SDSA scheduling.
// Note: enable outs are packed [N_HEAD-1:0] (iverilog cannot reliably
// drive multi-bit unpacked array *output* ports). Score inputs stay unpacked.
// Yosys: use flows/oss/wrappers/c2s_sth_gate_synth.sv (flat score ports too).

`timescale 1ns/1ps
`default_nettype none

module c2s_sth_gate #(
  parameter int                 N_HEAD  = 8,
  parameter int                 SCORE_W = 8,
  parameter logic [SCORE_W-1:0] TH_SP   = 8'd2,
  parameter logic [SCORE_W-1:0] TH_TP   = 8'd2
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    valid_i,
  input  logic [SCORE_W-1:0]      spat_score [N_HEAD],
  input  logic [SCORE_W-1:0]      temp_score [N_HEAD],
  output logic [N_HEAD-1:0]       spat_en,
  output logic [N_HEAD-1:0]       temp_en,
  output logic [N_HEAD-1:0]       dual_en,
  output logic [N_HEAD-1:0]       skip_en,
  output logic                    gate_valid
);

  logic [N_HEAD-1:0] spat_en_c;
  logic [N_HEAD-1:0] temp_en_c;
  logic [N_HEAD-1:0] dual_en_c;
  logic [N_HEAD-1:0] skip_en_c;

  integer hi;

  always_comb begin
    for (hi = 0; hi < N_HEAD; hi = hi + 1) begin
      spat_en_c[hi] = (spat_score[hi] > TH_SP);
      temp_en_c[hi] = (temp_score[hi] > TH_TP);
      dual_en_c[hi] = spat_en_c[hi] & temp_en_c[hi];
      skip_en_c[hi] = ~(spat_en_c[hi] | temp_en_c[hi]);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      spat_en    <= '0;
      temp_en    <= '0;
      dual_en    <= '0;
      skip_en    <= '0;
      gate_valid <= 1'b0;
    end else begin
      gate_valid <= valid_i;
      if (valid_i) begin
        spat_en <= spat_en_c;
        temp_en <= temp_en_c;
        dual_en <= dual_en_c;
        skip_en <= skip_en_c;
      end
    end
  end

endmodule

`default_nettype wire
