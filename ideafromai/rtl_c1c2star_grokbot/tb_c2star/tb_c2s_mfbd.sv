// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_mfbd — ≥2 cases: route to hyp0 / hyp2; empty count

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_mfbd;
  localparam int MAX_B  = 4;
  localparam int TILE_W = 4;
  localparam int DT_W   = 3;
  localparam int HYP_W  = 2;
  localparam int PAY_W  = 8;
  localparam int N_HYP  = (1 << HYP_W);
  localparam int CNT_W  = $clog2(MAX_B + 1);

  logic                       clk;
  logic                       rst_n;
  logic                       valid_i;
  logic [CNT_W-1:0]           bundle_count;
  logic [MAX_B*TILE_W-1:0]    bundle_tile;
  logic [MAX_B*DT_W-1:0]      bundle_dt;
  logic [MAX_B*HYP_W-1:0]     bundle_hyp;
  logic signed [PAY_W-1:0]    payload;
  logic [N_HYP-1:0]           lane_valid;
  logic signed [PAY_W-1:0]    lane_payload;
  logic                       deliver_valid;

  c2s_mfbd #(
    .MAX_B (MAX_B),
    .TILE_W(TILE_W),
    .DT_W  (DT_W),
    .HYP_W (HYP_W),
    .PAY_W (PAY_W)
  ) dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .valid_i      (valid_i),
    .bundle_count (bundle_count),
    .bundle_tile  (bundle_tile),
    .bundle_dt    (bundle_dt),
    .bundle_hyp   (bundle_hyp),
    .payload      (payload),
    .lane_valid   (lane_valid),
    .lane_payload (lane_payload),
    .deliver_valid(deliver_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_mfbd.vcd");
      $dumpvars(0, tb_c2s_mfbd);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    rst_n        = 1'b0;
    valid_i      = 1'b0;
    bundle_count = '0;
    bundle_tile  = '0;
    bundle_dt    = '0;
    bundle_hyp   = '0;
    payload      = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;  // leave clock edge before driving (avoid TB/NBA race)

    // Case1: bundle[0] hyp=0 → lane0, payload=11
    valid_i      = 1'b1;
    bundle_count = 3'd2;
    bundle_tile  = {4'd3, 4'd1, 4'd0, 4'd5}; // entry0 tile=5
    bundle_dt    = {3'd0, 3'd0, 3'd1, 3'd2};
    bundle_hyp   = {2'd0, 2'd0, 2'd1, 2'd0}; // entry0 hyp=0
    payload      = 8'sd11;
    @(posedge clk);
    #1;
    if (deliver_valid !== 1'b1) begin
      $display("FAIL Case1: deliver_valid");
      $fatal(1);
    end
    if (lane_valid !== 4'b0001) begin
      $display("FAIL Case1: lane_valid=%b expected 0001", lane_valid);
      $fatal(1);
    end
    if (lane_payload !== 8'sd11) begin
      $display("FAIL Case1: payload=%0d", lane_payload);
      $fatal(1);
    end
    $display("PASS Case1 route hyp0");

    // Case2: bundle[0] hyp=2 → lane2, payload=-4
    bundle_count = 3'd1;
    bundle_tile  = {4'd0, 4'd0, 4'd0, 4'd7};
    bundle_dt    = {3'd0, 3'd0, 3'd0, 3'd4};
    bundle_hyp   = {2'd0, 2'd0, 2'd0, 2'd2}; // entry0 hyp=2
    payload      = -8'sd4;
    @(posedge clk);
    #1;
    if (lane_valid !== 4'b0100) begin
      $display("FAIL Case2: lane_valid=%b expected 0100", lane_valid);
      $fatal(1);
    end
    if (lane_payload !== -8'sd4) begin
      $display("FAIL Case2: payload=%0d", lane_payload);
      $fatal(1);
    end
    $display("PASS Case2 route hyp2");

    // Case3: count=0 → no lane
    bundle_count = '0;
    payload      = 8'sd9;
    @(posedge clk);
    #1;
    if (lane_valid !== 4'b0000) begin
      $display("FAIL Case3 empty: lane_valid=%b", lane_valid);
      $fatal(1);
    end
    $display("PASS Case3 empty count");

    $display("PASS: all MFBD cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
