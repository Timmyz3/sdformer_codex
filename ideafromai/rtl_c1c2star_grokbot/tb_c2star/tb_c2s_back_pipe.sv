// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_back_pipe — HBG→SMAM chain + parallel STH; N_HEAD=8

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_back_pipe;
  localparam int                 AMP_W   = 8;
  localparam int                 ACC_W   = 16;
  localparam int                 N_HEAD  = 8;
  localparam int                 SCORE_W = 8;
  localparam logic [SCORE_W-1:0] TH_SP   = 8'd2;
  localparam logic [SCORE_W-1:0] TH_TP   = 8'd2;

  logic                      clk;
  logic                      rst_n;
  logic                      amp_valid;
  logic signed [AMP_W-1:0]   amp;
  logic                      g;
  logic signed [AMP_W-1:0]   p;
  logic                      gp_valid;
  logic                      pe_clk_en;
  logic                      mac_en;
  logic                      mask_add_en;
  logic signed [AMP_W-1:0]   mac_payload;
  logic signed [ACC_W-1:0]   mask_add_result;
  logic                      smam_out_valid;
  logic                      sth_valid_i;
  logic [SCORE_W-1:0]        sth_spat_score [N_HEAD];
  logic [SCORE_W-1:0]        sth_temp_score [N_HEAD];
  logic [N_HEAD-1:0]         sth_spat_en;
  logic [N_HEAD-1:0]         sth_temp_en;
  logic [N_HEAD-1:0]         sth_dual_en;
  logic [N_HEAD-1:0]         sth_skip_en;
  logic                      sth_gate_valid;

  c2s_back_pipe #(
    .AMP_W  (AMP_W),
    .ACC_W  (ACC_W),
    .N_HEAD (N_HEAD),
    .SCORE_W(SCORE_W),
    .TH_SP  (TH_SP),
    .TH_TP  (TH_TP)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .amp_valid      (amp_valid),
    .amp            (amp),
    .g              (g),
    .p              (p),
    .gp_valid       (gp_valid),
    .pe_clk_en      (pe_clk_en),
    .mac_en         (mac_en),
    .mask_add_en    (mask_add_en),
    .mac_payload    (mac_payload),
    .mask_add_result(mask_add_result),
    .smam_out_valid (smam_out_valid),
    .sth_valid_i    (sth_valid_i),
    .sth_spat_score (sth_spat_score),
    .sth_temp_score (sth_temp_score),
    .sth_spat_en    (sth_spat_en),
    .sth_temp_en    (sth_temp_en),
    .sth_dual_en    (sth_dual_en),
    .sth_skip_en    (sth_skip_en),
    .sth_gate_valid (sth_gate_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_back_pipe.vcd");
      $dumpvars(0, tb_c2s_back_pipe);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer hi;

  task automatic clear_sth;
    begin
      sth_valid_i = 1'b0;
      for (hi = 0; hi < N_HEAD; hi = hi + 1) begin
        sth_spat_score[hi] = '0;
        sth_temp_score[hi] = '0;
      end
    end
  endtask

  // Pulse amp one cycle; wait gp_valid (T+1) then smam_out_valid (T+2)
  task automatic pulse_amp_and_wait_smam;
    begin
      amp_valid = 1'b1;
      @(posedge clk); // T+1: gp_valid + held g/p → mac_en combo
      #1;
      if (!gp_valid) begin
        $display("FAIL: expected gp_valid after 1 cycle");
        $fatal(1);
      end
      amp_valid = 1'b0;
      @(posedge clk); // T+2: smam_out_valid / acc update
      #1;
      if (!smam_out_valid) begin
        $display("FAIL: expected smam_out_valid after 2 cycles");
        $fatal(1);
      end
    end
  endtask

  initial begin
    rst_n     = 1'b0;
    amp_valid = 1'b0;
    amp       = '0;
    clear_sth();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: amp=0 → g=0; after pipe no mac / acc stays 0
    amp = 8'sd0;
    #1;
    if (g !== 1'b0 || p !== '0 || pe_clk_en !== 1'b0) begin
      $display("FAIL Case1 combo: g=%0d p=%0d pe=%0d", g, p, pe_clk_en);
      $fatal(1);
    end
    pulse_amp_and_wait_smam();
    if (mask_add_result !== '0) begin
      $display("FAIL Case1: acc expected 0 got %0d", mask_add_result);
      $fatal(1);
    end
    $display("PASS Case1 amp=0 no mac");

    // Case2: amp=+2 → g=1; held into SMAM → mac once, acc=1, payload=+2
    amp = 8'sd2;
    amp_valid = 1'b1;
    #1;
    if (g !== 1'b1 || p !== 8'sd2) begin
      $display("FAIL Case2 combo: g=%0d p=%0d", g, p);
      $fatal(1);
    end
    @(posedge clk); // T+1
    #1;
    if (!gp_valid || mac_en !== 1'b1 || mask_add_en !== 1'b1 ||
        mac_payload !== 8'sd2) begin
      $display("FAIL Case2 T+1: gp=%0d mac_en=%0d payload=%0d",
               gp_valid, mac_en, mac_payload);
      $fatal(1);
    end
    amp_valid = 1'b0;
    @(posedge clk); // T+2
    #1;
    if (mask_add_result !== 16'sd1 || !smam_out_valid) begin
      $display("FAIL Case2 T+2: acc=%0d out_valid=%0d",
               mask_add_result, smam_out_valid);
      $fatal(1);
    end
    $display("PASS Case2 amp=+2 → mac acc=1");

    // Case3: second gated beat accumulates to 2
    amp = -8'sd5;
    amp_valid = 1'b1;
    @(posedge clk); #1;
    if (mac_en !== 1'b1 || mac_payload !== -8'sd5) begin
      $display("FAIL Case3 T+1: mac_en=%0d payload=%0d", mac_en, mac_payload);
      $fatal(1);
    end
    amp_valid = 1'b0;
    @(posedge clk); #1;
    if (mask_add_result !== 16'sd2) begin
      $display("FAIL Case3: acc expected 2 got %0d", mask_add_result);
      $fatal(1);
    end
    $display("PASS Case3 second beat acc=2");

    // Case4: STH parallel — spat head0, temp head1, both head2, skip rest
    clear_sth();
    sth_spat_score[0] = 8'd5;
    sth_temp_score[1] = 8'd4;
    sth_spat_score[2] = 8'd3;
    sth_temp_score[2] = 8'd3;
    sth_valid_i = 1'b1;
    @(posedge clk); #1;
    if (!sth_gate_valid) begin
      $display("FAIL Case4: sth_gate_valid");
      $fatal(1);
    end
    if (sth_spat_en !== 8'b0000_0101 || sth_temp_en !== 8'b0000_0110 ||
        sth_dual_en !== 8'b0000_0100 ||
        sth_skip_en !== ~(sth_spat_en | sth_temp_en)) begin
      $display("FAIL Case4: spat=%b temp=%b dual=%b skip=%b",
               sth_spat_en, sth_temp_en, sth_dual_en, sth_skip_en);
      $fatal(1);
    end
    sth_valid_i = 1'b0;
    $display("PASS Case4 STH parallel spat/temp/dual");

    // Case5: reset clears SMAM acc + STH
    amp = 8'sd9;
    amp_valid = 1'b1;
    sth_valid_i = 1'b1;
    sth_spat_score[0] = 8'd9;
    @(posedge clk); #1;
    amp_valid = 1'b0;
    sth_valid_i = 1'b0;
    @(posedge clk); #1;
    if (mask_add_result === '0) begin
      $display("FAIL Case5 pre-reset: expected nonzero acc");
      $fatal(1);
    end
    rst_n = 1'b0;
    @(posedge clk); #1;
    if (mask_add_result !== '0 || smam_out_valid !== 1'b0 ||
        sth_gate_valid !== 1'b0 || sth_spat_en !== '0) begin
      $display("FAIL Case5: reset clear acc=%0d sov=%0d sgv=%0d spat=%b",
               mask_add_result, smam_out_valid, sth_gate_valid, sth_spat_en);
      $fatal(1);
    end
    $display("PASS Case5 reset clears");

    $display("PASS: all back_pipe cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
