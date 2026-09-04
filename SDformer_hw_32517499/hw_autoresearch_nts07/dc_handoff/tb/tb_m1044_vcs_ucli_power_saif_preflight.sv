`timescale 1ns/1ps

module m1044_tiny_ucli_power_dut (
    input  logic       clk,
    input  logic       rst_n,
    input  logic [7:0] stimulus,
    output logic [7:0] state
);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= 8'h00;
    else state <= {state[6:0], state[7] ^ stimulus[0]} + stimulus;
  end
endmodule

module tb_m1044_vcs_ucli_power_saif_preflight;
  logic clk;
  logic rst_n;
  logic [7:0] stimulus;
  logic [7:0] state;

  m1044_tiny_ucli_power_dut dut (
      .clk(clk), .rst_n(rst_n), .stimulus(stimulus), .state(state));

  initial begin
    clk = 1'b0;
    forever #1 clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    stimulus = 8'h01;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    $stop;
    repeat (12) begin
      @(negedge clk);
      stimulus <= {stimulus[6:0], ~stimulus[7]};
    end
    @(posedge clk);
    $stop;
    #2 $finish;
  end
endmodule
