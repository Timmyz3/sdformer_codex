// GROKBOT NEW FILE -- iscas_ssh
// TB: c1s_ogec_gate — directed cases (OGEC)

`timescale 1ns/1ps
`default_nettype none

module tb_c1s_ogec_gate;
  localparam int N_TILE = 8;

  logic              clk;
  logic              rst_n;
  logic              valid_i;
  logic [N_TILE-1:0] match_ok;
  logic [N_TILE-1:0] exact_en;
  logic [N_TILE-1:0] prop_en;
  logic              ogec_valid;

  c1s_ogec_gate #(.N_TILE(N_TILE)) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_i   (valid_i),
    .match_ok  (match_ok),
    .exact_en  (exact_en),
    .prop_en   (prop_en),
    .ogec_valid(ogec_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c1s_ogec.vcd");
      $dumpvars(0, tb_c1s_ogec_gate);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  task automatic apply_and_check(
    input string name,
    input logic [N_TILE-1:0] mok
  );
    begin
      match_ok = mok;
      valid_i  = 1'b1;
      @(posedge clk);
      #1;
      if (!ogec_valid) begin
        $display("FAIL %s: expected ogec_valid=1", name);
        $fatal(1);
      end
      if (exact_en !== mok) begin
        $display("FAIL %s: exact_en=%b expected %b", name, exact_en, mok);
        $fatal(1);
      end
      if (prop_en !== ~mok) begin
        $display("FAIL %s: prop_en=%b expected %b", name, prop_en, ~mok);
        $fatal(1);
      end
      $display("PASS %s: exact=%b prop=%b", name, exact_en, prop_en);
      valid_i = 1'b0;
      @(posedge clk);
      #1;
    end
  endtask

  initial begin
    rst_n    = 1'b0;
    valid_i  = 1'b0;
    match_ok = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: all matched → all exact, no prop
    apply_and_check("Case1 all match", {N_TILE{1'b1}});

    // Case2: mixed — tile 1,3 unmatched
    apply_and_check("Case2 mixed", 8'b1111_0101);

    // Case3: all unmatched → all prop
    apply_and_check("Case3 all unmatched", {N_TILE{1'b0}});

    // Case4: reset clears
    match_ok = 8'hA5;
    valid_i  = 1'b1;
    @(posedge clk);
    #1;
    if (exact_en === '0 && prop_en === '0) begin
      $display("FAIL pre-reset: expected nonzero enables");
      $fatal(1);
    end
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (exact_en !== '0 || prop_en !== '0 || ogec_valid !== 1'b0) begin
      $display("FAIL reset: exact=%b prop=%b valid=%0d", exact_en, prop_en, ogec_valid);
      $fatal(1);
    end
    $display("PASS Case4 reset clears");

    $display("PASS: all OGEC cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
