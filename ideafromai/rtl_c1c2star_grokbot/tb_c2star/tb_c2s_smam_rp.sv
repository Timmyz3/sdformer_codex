// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_smam_rp — dual-rail SMAM-RP directed cases

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_smam_rp;
  localparam int AMP_W = 8;
  localparam int ACC_W = 16;

  logic                      clk;
  logic                      rst_n;
  logic                      valid;
  logic                      spike_gate;
  logic signed [AMP_W-1:0]   payload;
  logic                      mac_en;
  logic                      mask_add_en;
  logic signed [AMP_W-1:0]   mac_payload;
  logic signed [ACC_W-1:0]   mask_add_result;
  logic                      out_valid;

  c2s_smam_rp #(
    .AMP_W(AMP_W),
    .ACC_W(ACC_W)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .valid          (valid),
    .spike_gate     (spike_gate),
    .payload        (payload),
    .mac_en         (mac_en),
    .mask_add_en    (mask_add_en),
    .mac_payload    (mac_payload),
    .mask_add_result(mask_add_result),
    .out_valid      (out_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_smam_rp.vcd");
      $dumpvars(0, tb_c2s_smam_rp);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    rst_n = 1'b0;
    valid = 1'b0;
    spike_gate = 1'b0;
    payload = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: gate=0 → no mac, payload held at 0, acc unchanged
    valid = 1'b1;
    spike_gate = 1'b0;
    payload = 8'sd42;
    #1;
    if (mac_en !== 1'b0 || mask_add_en !== 1'b0 || mac_payload !== '0) begin
      $display("FAIL Case1 gate=0: mac_en=%0d mask_add_en=%0d mac_payload=%0d",
               mac_en, mask_add_en, mac_payload);
      $fatal(1);
    end
    @(posedge clk);
    #1;
    if (mask_add_result !== '0) begin
      $display("FAIL Case1: acc should stay 0, got %0d", mask_add_result);
      $fatal(1);
    end
    if (out_valid !== 1'b1) begin
      $display("FAIL Case1: out_valid expected 1");
      $fatal(1);
    end
    $display("PASS Case1 gate=0 no mac");

    // Case2: gate=1 → payload passes, mac_en=1
    spike_gate = 1'b1;
    payload = -8'sd7;
    #1;
    if (mac_en !== 1'b1 || mask_add_en !== 1'b1 || mac_payload !== -8'sd7) begin
      $display("FAIL Case2: mac_en=%0d payload=%0d", mac_en, mac_payload);
      $fatal(1);
    end
    @(posedge clk);
    #1;
    if (mask_add_result !== 16'sd1) begin
      $display("FAIL Case2: acc expected 1 got %0d", mask_add_result);
      $fatal(1);
    end
    $display("PASS Case2 gate=1 payload passes acc=1");

    // Case3: several beats accumulate mask-add count
    payload = 8'sd3;
    @(posedge clk); #1; // beat 2 → acc=2
    if (mask_add_result !== 16'sd2) begin
      $display("FAIL Case3a: acc expected 2 got %0d", mask_add_result);
      $fatal(1);
    end
    @(posedge clk); #1; // beat 3 → acc=3
    if (mask_add_result !== 16'sd3) begin
      $display("FAIL Case3b: acc expected 3 got %0d", mask_add_result);
      $fatal(1);
    end
    // Interleave gate=0 — should not increment
    spike_gate = 1'b0;
    payload = 8'sd99;
    @(posedge clk); #1;
    if (mask_add_result !== 16'sd3 || mac_payload !== '0) begin
      $display("FAIL Case3c: gate=0 should hold acc=3, got %0d payload=%0d",
               mask_add_result, mac_payload);
      $fatal(1);
    end
    spike_gate = 1'b1;
    payload = 8'sd1;
    @(posedge clk); #1;
    if (mask_add_result !== 16'sd4) begin
      $display("FAIL Case3d: acc expected 4 got %0d", mask_add_result);
      $fatal(1);
    end
    $display("PASS Case3 multi-beat accumulate");

    // Reset clears acc
    rst_n = 1'b0;
    @(posedge clk); #1;
    if (mask_add_result !== '0 || out_valid !== 1'b0) begin
      $display("FAIL reset clear");
      $fatal(1);
    end
    $display("PASS reset clears");

    $display("PASS: all SMAM-RP cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
