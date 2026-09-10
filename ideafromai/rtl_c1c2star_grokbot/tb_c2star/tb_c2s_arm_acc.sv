// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_arm_acc — add to hyp0/hyp1; clear; PASS

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_arm_acc;
  localparam int N_HYP  = 4;
  localparam int ACC_W  = 16;
  localparam int DATA_W = 8;
  localparam int SEL_W  = $clog2(N_HYP);

  logic                     clk;
  logic                     rst_n;
  logic                     valid_i;
  logic                     clear_i;
  logic [SEL_W-1:0]         hyp_sel;
  logic signed [DATA_W-1:0] data_i;
  logic                     add_en;
  logic [N_HYP*ACC_W-1:0]   acc_bus;
  logic                     out_valid;

  function automatic logic signed [ACC_W-1:0] slice(input int h);
    return acc_bus[h*ACC_W +: ACC_W];
  endfunction

  c2s_arm_acc #(
    .N_HYP (N_HYP),
    .ACC_W (ACC_W),
    .DATA_W(DATA_W)
  ) dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_i  (valid_i),
    .clear_i  (clear_i),
    .hyp_sel  (hyp_sel),
    .data_i   (data_i),
    .add_en   (add_en),
    .acc_bus  (acc_bus),
    .out_valid(out_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_arm_acc.vcd");
      $dumpvars(0, tb_c2s_arm_acc);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    rst_n   = 1'b0;
    valid_i = 1'b0;
    clear_i = 1'b0;
    hyp_sel = '0;
    data_i  = '0;
    add_en  = 1'b0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;  // leave clock edge before driving (avoid TB/NBA race)

    // Case1: add to hyp0 then hyp1
    valid_i = 1'b1;
    add_en  = 1'b1;
    hyp_sel = 2'd0;
    data_i  = 8'sd5;
    @(posedge clk);
    #1;
    if (slice(0) !== 16'sd5 || slice(1) !== 16'sd0) begin
      $display("FAIL Case1a: hyp0=%0d hyp1=%0d", slice(0), slice(1));
      $fatal(1);
    end
    if (out_valid !== 1'b1) begin
      $display("FAIL Case1a: out_valid");
      $fatal(1);
    end

    hyp_sel = 2'd1;
    data_i  = 8'sd7;
    @(posedge clk);
    #1;
    if (slice(0) !== 16'sd5 || slice(1) !== 16'sd7) begin
      $display("FAIL Case1b: hyp0=%0d hyp1=%0d", slice(0), slice(1));
      $fatal(1);
    end
    // Accrue hyp0 again
    hyp_sel = 2'd0;
    data_i  = -8'sd2;
    @(posedge clk);
    #1;
    if (slice(0) !== 16'sd3 || slice(1) !== 16'sd7) begin
      $display("FAIL Case1c: hyp0=%0d hyp1=%0d", slice(0), slice(1));
      $fatal(1);
    end
    $display("PASS Case1 add hyp0/hyp1");

    // Case2: !add_en holds
    add_en = 1'b0;
    data_i = 8'sd99;
    @(posedge clk);
    #1;
    if (slice(0) !== 16'sd3 || slice(1) !== 16'sd7) begin
      $display("FAIL Case2 hold: hyp0=%0d hyp1=%0d", slice(0), slice(1));
      $fatal(1);
    end
    $display("PASS Case2 !add_en hold");

    // Case3: clear zeros all
    clear_i = 1'b1;
    valid_i = 1'b0;
    add_en  = 1'b0;
    @(posedge clk);
    #1;
    if (slice(0) !== '0 || slice(1) !== '0 || slice(2) !== '0 || slice(3) !== '0) begin
      $display("FAIL Case3 clear");
      $fatal(1);
    end
    $display("PASS Case3 clear_i");

    $display("PASS: all ARM-Acc cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
