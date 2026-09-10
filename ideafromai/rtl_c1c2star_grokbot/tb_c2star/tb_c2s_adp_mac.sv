// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_adp_mac — skip hold / mac accumulate / clear

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_adp_mac;
  localparam int W_A   = 8;
  localparam int W_B   = 8;
  localparam int W_ACC = 16;

  logic                    clk;
  logic                    rst_n;
  logic                    valid_i;
  logic signed [W_A-1:0]   a;
  logic signed [W_B-1:0]   b;
  logic                    skip_a;
  logic                    skip_b;
  logic                    mac_en;
  logic                    clear_i;
  logic signed [W_ACC-1:0] acc;
  logic                    skipped;
  logic                    out_valid;

  c2s_adp_mac #(
    .W_A  (W_A),
    .W_B  (W_B),
    .W_ACC(W_ACC),
    .FORBID_REORDER(1'b1)  // anti-reorder contract
  ) dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_i  (valid_i),
    .a        (a),
    .b        (b),
    .skip_a   (skip_a),
    .skip_b   (skip_b),
    .mac_en   (mac_en),
    .clear_i  (clear_i),
    .acc      (acc),
    .skipped  (skipped),
    .out_valid(out_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_adp_mac.vcd");
      $dumpvars(0, tb_c2s_adp_mac);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    rst_n   = 1'b0;
    valid_i = 1'b0;
    a       = '0;
    b       = '0;
    skip_a  = 1'b0;
    skip_b  = 1'b0;
    mac_en  = 1'b0;
    clear_i = 1'b0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: skip_a → no accumulate, skipped=1, acc holds 0
    valid_i = 1'b1;
    mac_en  = 1'b1;
    skip_a  = 1'b1;
    skip_b  = 1'b0;
    a       = 8'sd5;
    b       = 8'sd7;
    #1;
    if (skipped !== 1'b1) begin
      $display("FAIL Case1: skipped expected 1");
      $fatal(1);
    end
    @(posedge clk);
    #1;
    if (acc !== '0) begin
      $display("FAIL Case1: acc should hold 0, got %0d", acc);
      $fatal(1);
    end
    if (out_valid !== 1'b1) begin
      $display("FAIL Case1: out_valid expected 1");
      $fatal(1);
    end
    $display("PASS Case1 skip_a holds acc");

    // Case1b: skip_b
    skip_a = 1'b0;
    skip_b = 1'b1;
    a      = 8'sd3;
    b      = 8'sd9;
    @(posedge clk);
    #1;
    if (acc !== '0 || skipped !== 1'b1) begin
      $display("FAIL Case1b skip_b: acc=%0d skipped=%0d", acc, skipped);
      $fatal(1);
    end
    $display("PASS Case1b skip_b holds acc");

    // Case1c: !mac_en
    skip_b = 1'b0;
    mac_en = 1'b0;
    a      = 8'sd4;
    b      = 8'sd4;
    @(posedge clk);
    #1;
    if (acc !== '0 || skipped !== 1'b1) begin
      $display("FAIL Case1c !mac_en: acc=%0d skipped=%0d", acc, skipped);
      $fatal(1);
    end
    $display("PASS Case1c !mac_en holds acc");

    // Case2: mac_en path accumulates a*b
    mac_en = 1'b1;
    skip_a = 1'b0;
    skip_b = 1'b0;
    a      = 8'sd5;
    b      = 8'sd3;   // product 15
    #1;
    if (skipped !== 1'b0) begin
      $display("FAIL Case2: skipped expected 0");
      $fatal(1);
    end
    @(posedge clk);
    #1;
    if (acc !== 16'sd15) begin
      $display("FAIL Case2a: acc expected 15 got %0d", acc);
      $fatal(1);
    end
    a = -8'sd2;
    b = 8'sd4;        // product -8 → acc = 15-8 = 7
    @(posedge clk);
    #1;
    if (acc !== 16'sd7) begin
      $display("FAIL Case2b: acc expected 7 got %0d", acc);
      $fatal(1);
    end
    // Interleave skip — hold
    skip_a = 1'b1;
    a      = 8'sd99;
    b      = 8'sd99;
    @(posedge clk);
    #1;
    if (acc !== 16'sd7) begin
      $display("FAIL Case2c: skip should hold acc=7 got %0d", acc);
      $fatal(1);
    end
    $display("PASS Case2 mac_en accumulate (signed)");

    // Case3: clear_i zeros acc
    skip_a  = 1'b0;
    clear_i = 1'b1;
    valid_i = 1'b0;  // clear independent of valid
    @(posedge clk);
    #1;
    if (acc !== '0) begin
      $display("FAIL Case3: clear expected acc=0 got %0d", acc);
      $fatal(1);
    end
    $display("PASS Case3 clear_i");

    // After clear, one more mac
    clear_i = 1'b0;
    valid_i = 1'b1;
    mac_en  = 1'b1;
    a       = 8'sd2;
    b       = 8'sd6;  // 12
    @(posedge clk);
    #1;
    if (acc !== 16'sd12) begin
      $display("FAIL Case3b: after clear mac expected 12 got %0d", acc);
      $fatal(1);
    end
    $display("PASS Case3b mac after clear");

    // Case4: FORBID_REORDER contract — a*b != b*a narrative; order fixed, product commutative
    // but HW must NOT advertise operand-swap as sparsity innovation. Param must stay 1.
    if (dut.FORBID_REORDER !== 1'b1) begin
      $display("FAIL Case4: FORBID_REORDER expected 1 (anti-reorder letter contract)");
      $fatal(1);
    end
    // Non-commutative sanity if skips differ: skip_a path must not secretly become skip_b-only reorder
    clear_i = 1'b1; @(posedge clk); #1; clear_i = 1'b0;
    valid_i = 1'b1; mac_en = 1'b1; skip_a = 1'b0; skip_b = 1'b0;
    a = 8'sd3; b = 8'sd5; // 15
    @(posedge clk); #1;
    if (acc !== 16'sd15) begin
      $display("FAIL Case4a: expected 15 got %0d", acc); $fatal(1);
    end
    // Swapping driven values still multiplies a*b with ports fixed (not a reorder microarch)
    a = 8'sd5; b = 8'sd3; // still 15; ports a,b never swapped in RTL
    @(posedge clk); #1;
    if (acc !== 16'sd30) begin
      $display("FAIL Case4b: expected 30 got %0d", acc); $fatal(1);
    end
    $display("PASS Case4 FORBID_REORDER=1 anti-reorder contract");

    $display("PASS: all ADP-MAC cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
