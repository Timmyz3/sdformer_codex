// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_motion_ttb_packer — no wake / two wakes order / reset

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_motion_ttb_packer;
  localparam int N_TILE      = 16;
  localparam int DT_W        = 3;
  localparam int HYP_W       = 2;
  localparam int MAX_BUNDLES = 8;
  localparam int TILE_W      = $clog2(N_TILE);
  localparam int CNT_W       = $clog2(MAX_BUNDLES + 1);

  logic                              clk;
  logic                              rst_n;
  logic                              valid_i;
  logic [N_TILE-1:0]                 wake_bitmap;
  logic [DT_W-1:0]                   dt_bin  [N_TILE];
  logic [HYP_W-1:0]                  hyp_id  [N_TILE];
  logic [MAX_BUNDLES*TILE_W-1:0]     bundle_tile;
  logic [MAX_BUNDLES*DT_W-1:0]       bundle_dt;
  logic [MAX_BUNDLES*HYP_W-1:0]      bundle_hyp;
  logic [CNT_W-1:0]                  bundle_count;
  logic                              bundle_valid;

  // Unpack helpers for checks
  wire [TILE_W-1:0] btile0 = bundle_tile[0*TILE_W +: TILE_W];
  wire [TILE_W-1:0] btile1 = bundle_tile[1*TILE_W +: TILE_W];
  wire [DT_W-1:0]   bdt0   = bundle_dt  [0*DT_W   +: DT_W];
  wire [DT_W-1:0]   bdt1   = bundle_dt  [1*DT_W   +: DT_W];
  wire [HYP_W-1:0]  bhyp0  = bundle_hyp [0*HYP_W  +: HYP_W];
  wire [HYP_W-1:0]  bhyp1  = bundle_hyp [1*HYP_W  +: HYP_W];

  c2s_motion_ttb_packer #(
    .N_TILE     (N_TILE),
    .DT_W       (DT_W),
    .HYP_W      (HYP_W),
    .MAX_BUNDLES(MAX_BUNDLES)
  ) dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (valid_i),
    .wake_bitmap (wake_bitmap),
    .dt_bin      (dt_bin),
    .hyp_id      (hyp_id),
    .bundle_tile (bundle_tile),
    .bundle_dt   (bundle_dt),
    .bundle_hyp  (bundle_hyp),
    .bundle_count(bundle_count),
    .bundle_valid(bundle_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_motion_ttb.vcd");
      $dumpvars(0, tb_c2s_motion_ttb_packer);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  integer i;

  initial begin
    rst_n = 1'b0;
    valid_i = 1'b0;
    wake_bitmap = '0;
    for (i = 0; i < N_TILE; i = i + 1) begin
      dt_bin[i] = '0;
      hyp_id[i] = '0;
    end
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: no wake → count 0
    wake_bitmap = '0;
    valid_i = 1'b1;
    @(posedge clk);
    #1;
    if (bundle_valid !== 1'b1 || bundle_count !== '0) begin
      $display("FAIL Case1 no-wake: valid=%0d count=%0d", bundle_valid, bundle_count);
      $fatal(1);
    end
    $display("PASS Case1 no wake → count0");

    // Case2: two wakes (tiles 3 and 7) → count2 ascending order
    for (i = 0; i < N_TILE; i = i + 1) begin
      dt_bin[i] = i[DT_W-1:0];
      hyp_id[i] = i[HYP_W-1:0];
    end
    wake_bitmap = '0;
    wake_bitmap[3] = 1'b1;
    wake_bitmap[7] = 1'b1;
    dt_bin[3] = 3'd5;
    hyp_id[3] = 2'd2;
    dt_bin[7] = 3'd1;
    hyp_id[7] = 2'd3;
    valid_i = 1'b1;
    @(posedge clk);
    #1;
    if (bundle_valid !== 1'b1 || bundle_count != 2) begin
      $display("FAIL Case2 count: valid=%0d count=%0d (exp 2)", bundle_valid, bundle_count);
      $fatal(1);
    end
    if (btile0 !== 4'd3 || bdt0 !== 3'd5 || bhyp0 !== 2'd2) begin
      $display("FAIL Case2 order[0]: tile=%0d dt=%0d hyp=%0d", btile0, bdt0, bhyp0);
      $fatal(1);
    end
    if (btile1 !== 4'd7 || bdt1 !== 3'd1 || bhyp1 !== 2'd3) begin
      $display("FAIL Case2 order[1]: tile=%0d dt=%0d hyp=%0d", btile1, bdt1, bhyp1);
      $fatal(1);
    end
    $display("PASS Case2 two wakes → count2 order tile3 then tile7");

    // Drop valid → bundle_valid clears; count held
    valid_i = 1'b0;
    @(posedge clk);
    #1;
    if (bundle_valid !== 1'b0) begin
      $display("FAIL Case2b: bundle_valid expected 0");
      $fatal(1);
    end
    if (bundle_count != 2) begin
      $display("FAIL Case2b: count should hold 2, got %0d", bundle_count);
      $fatal(1);
    end

    // Case3: reset clears
    valid_i = 1'b1;
    wake_bitmap[1] = 1'b1;
    @(posedge clk);
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (bundle_valid !== 1'b0 || bundle_count !== '0 || bundle_tile !== '0 ||
        bundle_dt !== '0 || bundle_hyp !== '0) begin
      $display("FAIL Case3 reset: valid=%0d count=%0d tile=%h",
               bundle_valid, bundle_count, bundle_tile);
      $fatal(1);
    end
    $display("PASS Case3 reset clears");

    $display("PASS: all Motion-TTB packer cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
