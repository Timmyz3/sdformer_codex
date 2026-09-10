// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_front_pipe — thin OP-STW→ECP→MW chain; check tile_active

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_front_pipe;
  localparam int N_TILE  = 8;
  localparam int FLOW_W  = 8;
  localparam int EVT_W   = 8;
  localparam int SCORE_W = 8;
  localparam int SAMP_W  = 8;
  localparam int DW      = SAMP_W + 1;

  logic                      clk;
  logic                      rst_n;
  logic                      valid_i;
  logic signed [FLOW_W-1:0]  flow_cur  [N_TILE];
  logic signed [FLOW_W-1:0]  flow_prev [N_TILE];
  logic        [EVT_W-1:0]   event_cnt [N_TILE];
  logic        [SCORE_W-1:0] corr_score [N_TILE];
  logic signed [SAMP_W-1:0]  ref_samp [N_TILE];
  logic signed [SAMP_W-1:0]  cur_samp [N_TILE];

  logic [N_TILE-1:0] wake_bitmap;
  logic              wake_valid;
  logic [N_TILE-1:0] proj_en_bitmap;
  logic [N_TILE-1:0] skip_bitmap;
  logic              proj_valid;
  logic signed [N_TILE*DW-1:0] delta;
  logic [N_TILE-1:0] delta_nz;
  logic              delta_valid;
  logic [N_TILE-1:0] tile_active;
  logic              tile_active_valid;

  c1s_front_pipe #(
    .N_TILE (N_TILE),
    .FLOW_W (FLOW_W),
    .EVT_W  (EVT_W),
    .SCORE_W(SCORE_W),
    .SAMP_W (SAMP_W)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .valid_i          (valid_i),
    .flow_cur         (flow_cur),
    .flow_prev        (flow_prev),
    .event_cnt        (event_cnt),
    .corr_score       (corr_score),
    .ref_samp         (ref_samp),
    .cur_samp         (cur_samp),
    .wake_bitmap      (wake_bitmap),
    .wake_valid       (wake_valid),
    .proj_en_bitmap   (proj_en_bitmap),
    .skip_bitmap      (skip_bitmap),
    .proj_valid       (proj_valid),
    .delta            (delta),
    .delta_nz         (delta_nz),
    .delta_valid      (delta_valid),
    .tile_active      (tile_active),
    .tile_active_valid(tile_active_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_front_pipe.vcd");
      $dumpvars(0, tb_c1s_front_pipe);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_inputs;
    int i;
    begin
      valid_i = 1'b0;
      for (i = 0; i < N_TILE; i++) begin
        flow_cur[i]   = '0;
        flow_prev[i]  = '0;
        event_cnt[i]  = '0;
        corr_score[i] = '0;
        ref_samp[i]   = '0;
        cur_samp[i]   = '0;
      end
    end
  endtask

  // Drive one frame; wait for tile_active_valid; check reacts
  // T+1: wake/delta; T+2: proj / tile_active
  task automatic pulse_and_wait_active;
    begin
      valid_i = 1'b1;
      @(posedge clk); // T+1 capture OP-STW / MW / corr_held
      #1;
      if (!wake_valid || !delta_valid) begin
        $display("FAIL: expected wake_valid & delta_valid after 1 cycle (wake=%0d delta=%0d)",
                 wake_valid, delta_valid);
        $fatal(1);
      end
      valid_i = 1'b0;
      @(posedge clk); // T+2 ECP / tile_active
      #1;
      if (!tile_active_valid || !proj_valid) begin
        $display("FAIL: expected tile_active_valid after 2 cycles (tav=%0d pv=%0d)",
                 tile_active_valid, proj_valid);
        $fatal(1);
      end
    end
  endtask

  initial begin
    rst_n = 1'b0;
    clear_inputs();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: idle / all-below-threshold → tile_active=0
    clear_inputs();
    pulse_and_wait_active();
    if (tile_active !== '0) begin
      $display("FAIL Case1 idle: tile_active=%b expected 0", tile_active);
      $fatal(1);
    end
    $display("PASS Case1 idle tile_active=0");

    // Case2: flow wake on tile 2 only (abs(diff)>TH_W=2)
    clear_inputs();
    flow_prev[2] = 8'sd0;
    flow_cur[2]  = 8'sd5;  // abs=5 > 2
    pulse_and_wait_active();
    if (tile_active !== (8'b1 << 2)) begin
      $display("FAIL Case2: tile_active=%b expected only bit2", tile_active);
      $fatal(1);
    end
    $display("PASS Case2 flow-wake → tile_active=%b", tile_active);

    // Case3: delta_nz only → OR into wake → ECP proj → tile_active
    clear_inputs();
    ref_samp[5] = 8'sd0;
    cur_samp[5] = 8'sd4;
    pulse_and_wait_active();
    if (tile_active[5] !== 1'b1) begin
      $display("FAIL Case3: tile_active=%b expected bit5", tile_active);
      $fatal(1);
    end
    $display("PASS Case3 delta_nz → tile_active=%b", tile_active);

    // Case4: corr_score alone (TH_S=2 strict >; score=3 on tile0)
    clear_inputs();
    corr_score[0] = 8'd3;
    pulse_and_wait_active();
    if (tile_active[0] !== 1'b1) begin
      $display("FAIL Case4: tile_active=%b expected bit0", tile_active);
      $fatal(1);
    end
    $display("PASS Case4 corr_score → tile_active=%b", tile_active);

    // Case5: reset clears
    clear_inputs();
    flow_cur[1] = 8'sd9;
    pulse_and_wait_active();
    if (tile_active === '0) begin
      $display("FAIL Case5 pre-reset: expected nonzero tile_active");
      $fatal(1);
    end
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (tile_active_valid !== 1'b0 || tile_active !== '0) begin
      $display("FAIL Case5: after reset tav=%0d active=%b", tile_active_valid, tile_active);
      $fatal(1);
    end
    $display("PASS Case5 reset clears");

    $display("PASS: all front_pipe cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
