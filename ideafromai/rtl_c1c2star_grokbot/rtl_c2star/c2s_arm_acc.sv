// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_arm_acc — Aperture / multi-hypothesis accumulator (hARMS-style)
// Acc contexts = N_HYP direction hypotheses; select via hyp_sel and accumulate
// signed data. Packed acc_bus avoids unpacked array *output* (iverilog).

`timescale 1ns/1ps
`default_nettype none

module c2s_arm_acc #(
  parameter int N_HYP  = 4,
  parameter int ACC_W  = 16,
  parameter int DATA_W = 8,
  parameter int SEL_W  = (N_HYP <= 1) ? 1 : $clog2(N_HYP)
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        valid_i,
  input  logic                        clear_i,
  input  logic [SEL_W-1:0]            hyp_sel,
  input  logic signed [DATA_W-1:0]    data_i,
  input  logic                        add_en,
  output logic [N_HYP*ACC_W-1:0]      acc_bus,
  output logic                        out_valid
);

  // Internal unpacked regs; packed for output only
  logic signed [ACC_W-1:0] acc [N_HYP];
  logic signed [ACC_W-1:0] data_sext;
  integer hi;
  integer pj;

  assign data_sext = {{(ACC_W-DATA_W){data_i[DATA_W-1]}}, data_i};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (hi = 0; hi < N_HYP; hi = hi + 1)
        acc[hi] <= '0;
      out_valid <= 1'b0;
    end else begin
      out_valid <= valid_i;
      if (clear_i) begin
        for (hi = 0; hi < N_HYP; hi = hi + 1)
          acc[hi] <= '0;
      end else if (valid_i && add_en) begin
        for (hi = 0; hi < N_HYP; hi = hi + 1) begin
          if (hi == int'(hyp_sel))
            acc[hi] <= acc[hi] + data_sext;
        end
      end
    end
  end

  always_comb begin
    for (pj = 0; pj < N_HYP; pj = pj + 1)
      acc_bus[pj*ACC_W +: ACC_W] = acc[pj];
  end

endmodule

`default_nettype wire
