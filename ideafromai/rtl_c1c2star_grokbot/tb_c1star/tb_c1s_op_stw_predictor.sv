// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_op_stw_predictor — 3 directed cases from CARD_A_OP_STW

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_op_stw_predictor;
  localparam int N_TILE = 8;
  localparam int FLOW_W = 8;
  localparam int EVT_W  = 8;

  logic                         clk;
  logic                         rst_n;
  logic                         valid_i;
  logic signed [FLOW_W-1:0]     flow_cur  [N_TILE];
  logic signed [FLOW_W-1:0]     flow_prev [N_TILE];
  logic        [EVT_W-1:0]      event_cnt [N_TILE];
  logic        [N_TILE-1:0]     wake_bitmap;
  logic                         wake_valid;

  c1s_op_stw_predictor #(
    .N_TILE(N_TILE),
    .FLOW_W(FLOW_W),
    .EVT_W (EVT_W),
    .TH_W  (8'sd2),
    .TH_E  (8'd1)
  ) dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_i    (valid_i),
    .flow_cur   (flow_cur),
    .flow_prev  (flow_prev),
    .event_cnt  (event_cnt),
    .wake_bitmap(wake_bitmap),
    .wake_valid (wake_valid)
  );


  // VCD dump when +DUMP_VCD=1 (OSS flow); keeps self-check PASS otherwise
  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_op_stw.vcd");
      $dumpvars(0, tb_c1s_op_stw_predictor);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_inputs;
    int i;
    begin
      valid_i = 1'b0;
      for (i = 0; i < N_TILE; i++) begin
        flow_cur[i]  = '0;
        flow_prev[i] = '0;
        event_cnt[i] = '0;
      end
    end
  endtask

  // Drive valid_i=1 through one posedge; outputs update after that edge
  task automatic apply_and_check(
    input string name,
    input logic [N_TILE-1:0] exp_wake
  );
    begin
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      if (!wake_valid) begin
        $display("FAIL %s: expected wake_valid=1 one cycle after valid_i", name);
        $fatal(1);
      end
      if (wake_bitmap !== exp_wake) begin
        $display("FAIL %s: wake_bitmap=%b expected %b", name, wake_bitmap, exp_wake);
        $fatal(1);
      end
      $display("PASS %s: wake_bitmap=%b", name, wake_bitmap);
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

    // Case1: all deltas 0, events 0 → wake all 0
    clear_inputs();
    apply_and_check("Case1 all0", {N_TILE{1'b0}});

    // Case2: one tile large delta → only that bit 1
    clear_inputs();
    flow_cur[3]  = 8'sd10;
    flow_prev[3] = 8'sd0;
    apply_and_check("Case2 tile3 delta", (8'b1 << 3));

    // Case3: one tile high event_cnt → that bit 1
    clear_inputs();
    event_cnt[5] = 8'd5;
    apply_and_check("Case3 tile5 events", (8'b1 << 5));

    // Reset clears
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (wake_bitmap !== '0 || wake_valid !== 1'b0) begin
      $display("FAIL reset: wake not cleared (bitmap=%b valid=%0d)", wake_bitmap, wake_valid);
      $fatal(1);
    end
    $display("PASS reset clears wake");

    $display("PASS: all 3 OP-STW cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end

endmodule

`default_nettype wire
