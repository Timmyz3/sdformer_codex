// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened STH-Gate (Yosys cannot parse unpacked array ports).

`timescale 1ns/1ps
`default_nettype none

module c2s_sth_gate_synth #(
  parameter int                 N_HEAD  = 8,
  parameter int                 SCORE_W = 8,
  parameter logic [SCORE_W-1:0] TH_SP   = 8'd2,
  parameter logic [SCORE_W-1:0] TH_TP   = 8'd2
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [N_HEAD*SCORE_W-1:0]    spat_score_flat,
  input  logic [N_HEAD*SCORE_W-1:0]    temp_score_flat,
  output logic [N_HEAD-1:0]            spat_en,
  output logic [N_HEAD-1:0]            temp_en,
  output logic [N_HEAD-1:0]            dual_en,
  output logic [N_HEAD-1:0]            skip_en,
  output logic                         gate_valid
);

  logic [N_HEAD-1:0] spat_en_c;
  logic [N_HEAD-1:0] temp_en_c;
  logic [N_HEAD-1:0] dual_en_c;
  logic [N_HEAD-1:0] skip_en_c;

  integer hi;

  always_comb begin
    for (hi = 0; hi < N_HEAD; hi = hi + 1) begin
      spat_en_c[hi] = (spat_score_flat[hi*SCORE_W +: SCORE_W] > TH_SP);
      temp_en_c[hi] = (temp_score_flat[hi*SCORE_W +: SCORE_W] > TH_TP);
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
