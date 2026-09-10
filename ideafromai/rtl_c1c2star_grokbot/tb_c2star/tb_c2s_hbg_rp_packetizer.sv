// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_hbg_rp_packetizer — 4 directed cases from CARD_B_HBG_RP

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_hbg_rp_packetizer;
  localparam int AMP_W = 8;

  logic                      clk;
  logic                      rst_n;
  logic                      amp_valid;
  logic signed [AMP_W-1:0]   amp;
  logic                      g;
  logic signed [AMP_W-1:0]   p;
  logic                      gp_valid;
  logic                      pe_clk_en;

  c2s_hbg_rp_packetizer #(
    .AMP_W(AMP_W),
    .EPS  (8'sd1)
  ) dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .amp_valid(amp_valid),
    .amp      (amp),
    .g        (g),
    .p        (p),
    .gp_valid (gp_valid),
    .pe_clk_en(pe_clk_en)
  );


  // VCD dump when +DUMP_VCD=1 (OSS flow); keeps self-check PASS otherwise
  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_hbg_rp.vcd");
      $dumpvars(0, tb_c2s_hbg_rp_packetizer);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic check(
    input string name,
    input logic exp_g,
    input logic signed [AMP_W-1:0] exp_p,
    input logic exp_pe
  );
    begin
      #1;
      if (g !== exp_g || p !== exp_p || pe_clk_en !== exp_pe) begin
        $display("FAIL %s: g=%0d p=%0d pe_clk_en=%0d (exp g=%0d p=%0d pe=%0d)",
                 name, g, p, pe_clk_en, exp_g, exp_p, exp_pe);
        $fatal(1);
      end else begin
        $display("PASS %s: g=%0d p=%0d pe_clk_en=%0d", name, g, p, pe_clk_en);
      end
    end
  endtask

  initial begin
    rst_n = 1'b0;
    amp_valid = 1'b0;
    amp = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: amp=0 → g=0, p=0
    amp_valid = 1'b1;
    amp = 8'sd0;
    check("Case1 amp=0", 1'b0, 8'sd0, 1'b0);
    @(posedge clk);

    // Case2: amp=±1 with EPS=1 → g=0 (strict >)
    amp = 8'sd1;
    check("Case2a amp=+1", 1'b0, 8'sd0, 1'b0);
    amp = -8'sd1;
    check("Case2b amp=-1", 1'b0, 8'sd0, 1'b0);
    @(posedge clk);

    // Case3: amp=±2 → g=1, p=amp
    amp = 8'sd2;
    check("Case3a amp=+2", 1'b1, 8'sd2, 1'b1);
    amp = -8'sd2;
    check("Case3b amp=-2", 1'b1, -8'sd2, 1'b1);
    @(posedge clk);

    // Most-negative abs sanity (optional extra): -128 abs=128 > EPS → g=1
    amp = -8'sd128;
    check("Case3c amp=-128", 1'b1, -8'sd128, 1'b1);
    @(posedge clk);

    // Case4: amp_valid=0 → g=0, pe_clk_en=0
    amp = 8'sd5;
    amp_valid = 1'b0;
    check("Case4 amp_valid=0", 1'b0, 8'sd0, 1'b0);

    // gp_valid registered: after amp_valid went 0, next cycle gp_valid=0
    @(posedge clk);
    #1;
    if (gp_valid !== 1'b0) begin
      $display("FAIL: gp_valid expected 0 after amp_valid=0");
      $fatal(1);
    end

    // Pulse amp_valid and check gp_valid one cycle later
    amp_valid = 1'b1;
    amp = 8'sd3;
    @(posedge clk);
    #1;
    if (gp_valid !== 1'b1) begin
      $display("FAIL: gp_valid expected 1 one cycle after amp_valid");
      $fatal(1);
    end

    // ATLIF eps contract (docs/ATLIF_contract_r1_grokbot.md): EPS=1 → |amp|==1 does NOT gate;
    // |amp|>1 gates; payload non-absorbable (p==amp when g).
    if (dut.EPS !== 8'sd1) begin
      $display("FAIL eps_contract: dut.EPS expected 1 got %0d", dut.EPS);
      $fatal(1);
    end
    amp_valid = 1'b1;
    amp = 8'sd1; #1;
    if (g !== 1'b0 || p !== 8'sd0) begin
      $display("FAIL eps_contract: |amp|==EPS must not gate"); $fatal(1);
    end
    amp = 8'sd2; #1;
    if (g !== 1'b1 || p !== 8'sd2) begin
      $display("FAIL eps_contract: |amp|>EPS must gate with p=amp"); $fatal(1);
    end
    $display("PASS eps_contract ATLIF EPS=1 strict-gt + non-absorbable p");

    $display("PASS: all 4 HBG-RP cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end

endmodule

`default_nettype wire
