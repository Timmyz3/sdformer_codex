// GROKBOT NEW FILE -- iscas_ssh
// Synth wrapper: OGEC with N_TILE=8 (ports already packed — instantiates RTL).

`timescale 1ns/1ps
`default_nettype none

module c1s_ogec_gate_synth #(
  parameter int N_TILE = 8
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic                 valid_i,
  input  logic [N_TILE-1:0]    match_ok,
  output logic [N_TILE-1:0]    exact_en,
  output logic [N_TILE-1:0]    prop_en,
  output logic                 ogec_valid
);

  c1s_ogec_gate #(.N_TILE(N_TILE)) u_ogec (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_i   (valid_i),
    .match_ok  (match_ok),
    .exact_en  (exact_en),
    .prop_en   (prop_en),
    .ogec_valid(ogec_valid)
  );

endmodule

`default_nettype wire
