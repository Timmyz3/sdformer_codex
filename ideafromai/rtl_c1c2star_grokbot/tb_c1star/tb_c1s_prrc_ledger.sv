// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_prrc_ledger — init allow / spend-to-zero / refill (+ one-hot)

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_prrc_ledger;
  localparam int N_LEVEL  = 3;
  localparam int BUDGET_W = 8;
  localparam logic [BUDGET_W-1:0] INIT_BUDGET = 8'd4;

  logic                        clk;
  logic                        rst_n;
  logic                        valid_i;
  logic [N_LEVEL-1:0]          level_sel;
  logic [$clog2(N_LEVEL)-1:0]  level_idx;
  logic                        spend_i;
  logic                        refill_i;
  logic [N_LEVEL*BUDGET_W-1:0] budget;
  logic                        allow_exact;
  logic                        ledger_valid;

  function automatic [BUDGET_W-1:0] bud;
    input integer lvl;
    begin
      bud = budget[lvl*BUDGET_W +: BUDGET_W];
    end
  endfunction

  c1s_prrc_ledger #(
    .N_LEVEL    (N_LEVEL),
    .BUDGET_W   (BUDGET_W),
    .INIT_BUDGET(INIT_BUDGET)
  ) dut (
    .clk(clk), .rst_n(rst_n), .valid_i(valid_i),
    .level_sel(level_sel), .level_idx(level_idx),
    .spend_i(spend_i), .refill_i(refill_i),
    .budget(budget), .allow_exact(allow_exact), .ledger_valid(ledger_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_prrc.vcd");
      $dumpvars(0, tb_c1s_prrc_ledger);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer i;

  initial begin
    rst_n = 0; valid_i = 0; level_sel = 0; level_idx = 0; spend_i = 0; refill_i = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk); #1;

    // Case1
    for (i = 0; i < N_LEVEL; i = i + 1)
      if (bud(i) !== INIT_BUDGET) begin
        $display("FAIL Case1 init: budget[%0d]=%0d", i, bud(i));
        $fatal(1);
      end
    level_idx = 2'd0; #1;
    if (allow_exact !== 1'b1) begin $display("FAIL Case1 allow"); $fatal(1); end
    $display("PASS Case1 init allow (budget=%0d)", bud(0));

    // Case2 spend to zero
    level_sel = 0; level_idx = 2'd0; spend_i = 1; refill_i = 0;
    for (i = 0; i < INIT_BUDGET; i = i + 1) begin
      valid_i = 1; @(posedge clk); #1;
      if (!ledger_valid) begin $display("FAIL Case2 valid"); $fatal(1); end
    end
    if (bud(0) !== 0 || allow_exact !== 0) begin
      $display("FAIL Case2 zero budget=%0d allow=%0d", bud(0), allow_exact); $fatal(1);
    end
    valid_i = 1; @(posedge clk); #1;
    if (bud(0) !== 0) begin $display("FAIL Case2 underflow"); $fatal(1); end
    if (bud(1) !== INIT_BUDGET || bud(2) !== INIT_BUDGET) begin
      $display("FAIL Case2 others %0d %0d", bud(1), bud(2)); $fatal(1);
    end
    valid_i = 0; spend_i = 0; @(posedge clk); #1;
    $display("PASS Case2 spend to zero → allow=0");

    // Case3 refill
    level_idx = 0; level_sel = 0; refill_i = 1; spend_i = 0; valid_i = 1;
    @(posedge clk); #1;
    if (bud(0) !== INIT_BUDGET || allow_exact !== 1 || ledger_valid !== 1) begin
      $display("FAIL Case3"); $fatal(1);
    end
    valid_i = 0; refill_i = 0; @(posedge clk); #1;
    $display("PASS Case3 refill restores allow");

    // Case4 one-hot → level 2
    level_idx = 0; level_sel = 3'b100; spend_i = 1; refill_i = 0;
    #1;
    valid_i = 1;
    @(posedge clk); #1;
    if (bud(2) !== INIT_BUDGET - 1) begin
      $display("FAIL Case4 b2=%0d", bud(2)); $fatal(1);
    end
    if (bud(0) !== INIT_BUDGET) begin
      $display("FAIL Case4 b0=%0d", bud(0)); $fatal(1);
    end
    $display("PASS Case4 one-hot level_sel");

    $display("PASS: all PRRC cases");
    $finish;
  end

  initial begin
    #100000; $display("FAIL: timeout"); $fatal(1);
  end
endmodule

`default_nettype wire
