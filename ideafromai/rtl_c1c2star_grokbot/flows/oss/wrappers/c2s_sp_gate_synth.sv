// GROKBOT NEW FILE -- iscas_ssh
// Synth-only flattened SP-Gate (Yosys cannot parse unpacked array ports).

`timescale 1ns/1ps
`default_nettype none

module c2s_sp_gate_synth #(
  parameter int                 N      = 8,
  parameter int                 MASS_W = 8,
  parameter logic [MASS_W-1:0]  TH_M   = 8'd2
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    valid_i,
  input  logic [N*MASS_W-1:0]     mass_flat,
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
      run_c[i]  = (mass_flat[i*MASS_W +: MASS_W] > TH_M) | force_bitmap[i];
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
