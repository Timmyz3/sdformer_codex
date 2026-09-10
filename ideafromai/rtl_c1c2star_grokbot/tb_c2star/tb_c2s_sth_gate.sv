// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_sth_gate — all-low skip / spat only / temp only / both / reset

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_sth_gate;
  localparam int                 N_HEAD  = 8;
  localparam int                 SCORE_W = 8;
  localparam logic [SCORE_W-1:0] TH_SP   = 8'd2;
  localparam logic [SCORE_W-1:0] TH_TP   = 8'd2;

  logic               clk;
  logic               rst_n;
  logic               valid_i;
  logic [SCORE_W-1:0] spat_score [N_HEAD];
  logic [SCORE_W-1:0] temp_score [N_HEAD];
  logic [N_HEAD-1:0]  spat_en;
  logic [N_HEAD-1:0]  temp_en;
  logic [N_HEAD-1:0]  dual_en;
  logic [N_HEAD-1:0]  skip_en;
  logic               gate_valid;

  c2s_sth_gate #(
    .N_HEAD (N_HEAD),
    .SCORE_W(SCORE_W),
    .TH_SP  (TH_SP),
    .TH_TP  (TH_TP)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_i   (valid_i),
    .spat_score(spat_score),
    .temp_score(temp_score),
    .spat_en   (spat_en),
    .temp_en   (temp_en),
    .dual_en   (dual_en),
    .skip_en   (skip_en),
    .gate_valid(gate_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_sth_gate.vcd");
      $dumpvars(0, tb_c2s_sth_gate);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer i;

  task automatic clear_scores;
    begin
      for (i = 0; i < N_HEAD; i = i + 1) begin
        spat_score[i] = '0;
        temp_score[i] = '0;
      end
    end
  endtask

  initial begin
    rst_n   = 1'b0;
    valid_i = 1'b0;
    clear_scores();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: all low (≤ TH) → skip
    clear_scores();
    for (i = 0; i < N_HEAD; i = i + 1) begin
      spat_score[i] = 8'd2; // == TH → not >
      temp_score[i] = 8'd1;
    end
    valid_i = 1'b1;
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1 || spat_en !== '0 || temp_en !== '0 ||
        dual_en !== '0 || skip_en !== {N_HEAD{1'b1}}) begin
      $display("FAIL Case1 all-low→skip: g=%0d spat=%b temp=%b dual=%b skip=%b",
               gate_valid, spat_en, temp_en, dual_en, skip_en);
      $fatal(1);
    end
    $display("PASS Case1 all low → skip");

    // Case2: spat only
    for (i = 0; i < N_HEAD; i = i + 1) begin
      spat_score[i] = 8'd5;
      temp_score[i] = 8'd0;
    end
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1 || spat_en !== {N_HEAD{1'b1}} || temp_en !== '0 ||
        dual_en !== '0 || skip_en !== '0) begin
      $display("FAIL Case2 spat only: spat=%b temp=%b dual=%b skip=%b",
               spat_en, temp_en, dual_en, skip_en);
      $fatal(1);
    end
    $display("PASS Case2 spat only");

    // Case3: temp only
    for (i = 0; i < N_HEAD; i = i + 1) begin
      spat_score[i] = 8'd0;
      temp_score[i] = 8'd7;
    end
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1 || spat_en !== '0 || temp_en !== {N_HEAD{1'b1}} ||
        dual_en !== '0 || skip_en !== '0) begin
      $display("FAIL Case3 temp only: spat=%b temp=%b dual=%b skip=%b",
               spat_en, temp_en, dual_en, skip_en);
      $fatal(1);
    end
    $display("PASS Case3 temp only");

    // Case4: both strong → dual
    for (i = 0; i < N_HEAD; i = i + 1) begin
      spat_score[i] = 8'd3;
      temp_score[i] = 8'd4;
    end
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1 || spat_en !== {N_HEAD{1'b1}} ||
        temp_en !== {N_HEAD{1'b1}} || dual_en !== {N_HEAD{1'b1}} ||
        skip_en !== '0) begin
      $display("FAIL Case4 both/dual: spat=%b temp=%b dual=%b skip=%b",
               spat_en, temp_en, dual_en, skip_en);
      $fatal(1);
    end
    // Mixed: head0 dual, head1 spat, head2 temp, head3 skip, rest skip
    for (i = 0; i < N_HEAD; i = i + 1) begin
      spat_score[i] = 8'd0;
      temp_score[i] = 8'd0;
    end
    spat_score[0] = 8'd9; temp_score[0] = 8'd9;
    spat_score[1] = 8'd9; temp_score[1] = 8'd0;
    spat_score[2] = 8'd0; temp_score[2] = 8'd9;
    spat_score[3] = 8'd2; temp_score[3] = 8'd2;
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1 ||
        spat_en[0] !== 1'b1 || temp_en[0] !== 1'b1 || dual_en[0] !== 1'b1 || skip_en[0] !== 1'b0 ||
        spat_en[1] !== 1'b1 || temp_en[1] !== 1'b0 || dual_en[1] !== 1'b0 || skip_en[1] !== 1'b0 ||
        spat_en[2] !== 1'b0 || temp_en[2] !== 1'b1 || dual_en[2] !== 1'b0 || skip_en[2] !== 1'b0 ||
        spat_en[3] !== 1'b0 || temp_en[3] !== 1'b0 || dual_en[3] !== 1'b0 || skip_en[3] !== 1'b1 ||
        spat_en[7:4] !== 4'b0000 || skip_en[7:4] !== 4'b1111) begin
      $display("FAIL Case4b mixed: spat=%b temp=%b dual=%b skip=%b",
               spat_en, temp_en, dual_en, skip_en);
      $fatal(1);
    end
    $display("PASS Case4 both/dual (+ mixed)");

    // Case5: reset clears
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b0 || spat_en !== '0 || temp_en !== '0 ||
        dual_en !== '0 || skip_en !== '0) begin
      $display("FAIL Case5 reset clear");
      $fatal(1);
    end
    $display("PASS Case5 reset clears");

    $display("PASS: all STH-Gate cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
