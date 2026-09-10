// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_ecp_qkv_predictor — directed cases (Card C ECP-QKV)

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_ecp_qkv_predictor;
  localparam int N_TILE  = 8;
  localparam int SCORE_W = 8;

  logic                      clk;
  logic                      rst_n;
  logic                      valid_i;
  logic [SCORE_W-1:0]        corr_score [N_TILE];
  logic [N_TILE-1:0]         wake_bitmap;
  logic                      use_wake;
  logic [N_TILE-1:0]         proj_en_bitmap;
  logic [N_TILE-1:0]         skip_bitmap;
  logic                      proj_valid;

  c1s_ecp_qkv_predictor #(
    .N_TILE (N_TILE),
    .SCORE_W(SCORE_W),
    .TH_S   (8'd2)
  ) dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .valid_i       (valid_i),
    .corr_score    (corr_score),
    .wake_bitmap   (wake_bitmap),
    .use_wake      (use_wake),
    .proj_en_bitmap(proj_en_bitmap),
    .skip_bitmap   (skip_bitmap),
    .proj_valid    (proj_valid)
  );

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_inputs;
    int i;
    begin
      valid_i = 1'b0;
      use_wake = 1'b0;
      wake_bitmap = '0;
      for (i = 0; i < N_TILE; i++) corr_score[i] = '0;
    end
  endtask

  task automatic apply_and_check(
    input string name,
    input logic [N_TILE-1:0] exp_proj
  );
    begin
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      if (!proj_valid) begin
        $display("FAIL %s: expected proj_valid=1", name);
        $fatal(1);
      end
      if (proj_en_bitmap !== exp_proj) begin
        $display("FAIL %s: proj_en=%b expected %b", name, proj_en_bitmap, exp_proj);
        $fatal(1);
      end
      if (skip_bitmap !== ~exp_proj) begin
        $display("FAIL %s: skip=%b expected %b", name, skip_bitmap, ~exp_proj);
        $fatal(1);
      end
      $display("PASS %s: proj_en=%b", name, proj_en_bitmap);
      valid_i = 1'b0;
      @(posedge clk);
      #1;
    end
  endtask

  initial begin
    rst_n = 1'b0;
    clear_inputs();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: all scores 0, no wake → all skip
    clear_inputs();
    apply_and_check("Case1 all0", {N_TILE{1'b0}});

    // Case2: one tile score > TH_S → only that proj_en
    clear_inputs();
    corr_score[2] = 8'd5;
    apply_and_check("Case2 score tile2", (8'b1 << 2));

    // Case3: score == TH_S (strict >) → still skip that tile
    clear_inputs();
    corr_score[1] = 8'd2;
    apply_and_check("Case3 score==TH skip", {N_TILE{1'b0}});

    // Case4: use_wake + wake bit forces proj even if score low
    clear_inputs();
    use_wake = 1'b1;
    wake_bitmap = (8'b1 << 6);
    apply_and_check("Case4 wake force", (8'b1 << 6));

    // Reset clears
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (proj_en_bitmap !== '0 || proj_valid !== 1'b0) begin
      $display("FAIL reset clear");
      $fatal(1);
    end
    $display("PASS reset clears");

    $display("PASS: all ECP-QKV cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
