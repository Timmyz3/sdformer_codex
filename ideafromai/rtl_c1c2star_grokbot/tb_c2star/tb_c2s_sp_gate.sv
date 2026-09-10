// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_sp_gate — all low drop; one high run; force; PASS

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_sp_gate;
  localparam int N      = 8;
  localparam int MASS_W = 8;
  localparam logic [MASS_W-1:0] TH_M = 8'd2;

  logic               clk;
  logic               rst_n;
  logic               valid_i;
  logic [MASS_W-1:0]  mass [N];
  logic [N-1:0]       force_bitmap;
  logic [N-1:0]       run_en;
  logic [N-1:0]       drop_en;
  logic               gate_valid;

  integer i;

  c2s_sp_gate #(
    .N     (N),
    .MASS_W(MASS_W),
    .TH_M  (TH_M)
  ) dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .valid_i     (valid_i),
    .mass        (mass),
    .force_bitmap(force_bitmap),
    .run_en      (run_en),
    .drop_en     (drop_en),
    .gate_valid  (gate_valid)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_sp_gate.vcd");
      $dumpvars(0, tb_c2s_sp_gate);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  initial begin
    rst_n        = 1'b0;
    valid_i      = 1'b0;
    force_bitmap = '0;
    for (i = 0; i < N; i = i + 1) mass[i] = '0;
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;  // leave clock edge before driving (avoid TB/NBA race)

    // Case1: all mass <= TH_M → all drop
    valid_i = 1'b1;
    for (i = 0; i < N; i = i + 1) mass[i] = 8'd1; // 1 <= 2 → drop
    mass[3] = 8'd2; // equal → not > → drop
    force_bitmap = '0;
    @(posedge clk);
    #1;
    if (gate_valid !== 1'b1) begin
      $display("FAIL Case1: gate_valid");
      $fatal(1);
    end
    if (run_en !== 8'b0 || drop_en !== 8'hFF) begin
      $display("FAIL Case1: run=%b drop=%b", run_en, drop_en);
      $fatal(1);
    end
    $display("PASS Case1 all low drop");

    // Case2: one high mass → that lane runs
    for (i = 0; i < N; i = i + 1) mass[i] = 8'd0;
    mass[5] = 8'd7; // > 2
    @(posedge clk);
    #1;
    if (run_en !== 8'b0010_0000 || drop_en !== 8'b1101_1111) begin
      $display("FAIL Case2: run=%b drop=%b", run_en, drop_en);
      $fatal(1);
    end
    $display("PASS Case2 one high run");

    // Case3: force_bitmap overrides low mass
    for (i = 0; i < N; i = i + 1) mass[i] = 8'd0;
    force_bitmap = 8'b0000_0100; // force lane 2
    @(posedge clk);
    #1;
    if (run_en !== 8'b0000_0100 || drop_en !== 8'b1111_1011) begin
      $display("FAIL Case3 force: run=%b drop=%b", run_en, drop_en);
      $fatal(1);
    end
    $display("PASS Case3 force_bitmap");

    $display("PASS: all SP-Gate cases");
    $finish;
  end

  initial begin
    #10000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
