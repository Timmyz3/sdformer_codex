// GROKBOT NEW FILE -- iscas_ssh
// Written by: Grok Bot (iscas_ssh)
// Date: 2026-09-05
// Module: c2s_sp_gate — Attention-mass schedule gate (SpAtten-style)
// run_en[i] = (mass[i] > TH_M) | force_bitmap[i]; drop_en = ~run_en.
// Registered 1-cycle. Does NOT multiply-reorder. Yosys: flat wrapper
// flows/oss/wrappers/c2s_sp_gate_synth.sv (unpacked mass[]).

`timescale 1ns/1ps
`default_nettype none

module c2s_sp_gate #(
  parameter int                 N      = 8,
  parameter int                 MASS_W = 8,
  parameter logic [MASS_W-1:0]  TH_M   = 8'd2
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    valid_i,
  input  logic [MASS_W-1:0]       mass [N],
  input  logic [N-1:0]            force_bitmap,
  output logic [N-1:0]            run_en,
  output logic [N-1:0]            drop_en,
  output logic                    gate_valid
);

  logic [N-1:0] run_c;
  logic [N-1:0] drop_c;
  integer i;

  always_comb begin
    for (i = 0; i < N; i = i + 1) begin
      run_c[i]  = (mass[i] > TH_M) | force_bitmap[i];
      drop_c[i] = ~run_c[i];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      run_en     <= '0;
      drop_en    <= '0;
      gate_valid <= 1'b0;
    end else begin
      gate_valid <= valid_i;
      if (valid_i) begin
        run_en  <= run_c;
        drop_en <= drop_c;
      end
    end
  end

endmodule

`default_nettype wire
