// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_mw_delta_buf — directed cases (MW-ΔBuf C1*)

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_mw_delta_buf;
  localparam int N_TILE = 8;
  localparam int SAMP_W = 8;
  localparam int DW     = SAMP_W + 1;

  logic                      clk;
  logic                      rst_n;
  logic                      valid_i;
  logic signed [SAMP_W-1:0]  ref_samp [N_TILE];
  logic signed [SAMP_W-1:0]  cur_samp [N_TILE];
  logic signed [N_TILE*DW-1:0] delta;
  logic [N_TILE-1:0]         delta_nz;
  logic                      delta_valid;

  function automatic logic signed [DW-1:0] tile_delta(input int i);
    tile_delta = delta[i*DW +: DW];
  endfunction

  c1s_mw_delta_buf #(
    .N_TILE(N_TILE),
    .SAMP_W(SAMP_W),
    .TH_D  (8'd0)
  ) dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .valid_i    (valid_i),
    .ref_samp   (ref_samp),
    .cur_samp   (cur_samp),
    .delta      (delta),
    .delta_nz   (delta_nz),
    .delta_valid(delta_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_mw_delta.vcd");
      $dumpvars(0, tb_c1s_mw_delta_buf);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic clear_inputs;
    int i;
    begin
      valid_i = 1'b0;
      for (i = 0; i < N_TILE; i++) begin
        ref_samp[i] = '0;
        cur_samp[i] = '0;
      end
    end
  endtask

  task automatic apply_and_check(
    input string name,
    input logic [N_TILE-1:0] exp_nz
  );
    int i;
    logic signed [DW-1:0] exp_d [N_TILE];
    begin
      for (i = 0; i < N_TILE; i++)
        exp_d[i] = $signed({cur_samp[i][SAMP_W-1], cur_samp[i]}) -
                   $signed({ref_samp[i][SAMP_W-1], ref_samp[i]});
      valid_i = 1'b1;
      @(posedge clk);
      #1;
      if (!delta_valid) begin
        $display("FAIL %s: expected delta_valid=1", name);
        $fatal(1);
      end
      if (delta_nz !== exp_nz) begin
        $display("FAIL %s: delta_nz=%b expected %b", name, delta_nz, exp_nz);
        $fatal(1);
      end
      for (i = 0; i < N_TILE; i++) begin
        if (tile_delta(i) !== exp_d[i]) begin
          $display("FAIL %s: delta[%0d]=%0d expected %0d", name, i, tile_delta(i), exp_d[i]);
          $fatal(1);
        end
      end
      $display("PASS %s: delta_nz=%b", name, delta_nz);
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

    // Case1: all equal → nz=0
    clear_inputs();
    begin
      int i;
      for (i = 0; i < N_TILE; i++) begin
        ref_samp[i] = 8'sd10;
        cur_samp[i] = 8'sd10;
      end
    end
    apply_and_check("Case1 all equal nz=0", {N_TILE{1'b0}});

    // Case2: one tile jump → that bit only
    clear_inputs();
    begin
      int i;
      for (i = 0; i < N_TILE; i++) begin
        ref_samp[i] = 8'sd0;
        cur_samp[i] = 8'sd0;
      end
      cur_samp[3] = 8'sd7;
    end
    apply_and_check("Case2 tile3 jump", (8'b1 << 3));

    // Case3: reset clears
    clear_inputs();
    cur_samp[0] = 8'sd5;
    valid_i = 1'b1;
    @(posedge clk);
    #1;
    if (delta_nz === '0) begin
      $display("FAIL pre-reset: expected nonzero delta_nz");
      $fatal(1);
    end
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (delta_nz !== '0 || delta_valid !== 1'b0 || delta !== '0) begin
      $display("FAIL reset clear: nz=%b valid=%0d delta=%0h", delta_nz, delta_valid, delta);
      $fatal(1);
    end
    $display("PASS Case3 reset clears");

    $display("PASS: all MW-DeltaBuf cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
