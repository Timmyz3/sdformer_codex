// GROKBOT NEW FILE -- iscas_ssh
// TB: c2s_bui_guard_sdsa — N_TOKEN=8; KEEP_ALL vs BUI ablation + prove-by skip counts

`timescale 1ns/1ps
`default_nettype none

module tb_c2s_bui_guard_sdsa;
  localparam int N_TOKEN = 8;
  localparam int SCORE_W = 8;
  localparam int BOUND_W = 8;

  logic clk, rst_n, bit_round_valid_i, hbg_g_i;
  logic [N_TOKEN*SCORE_W-1:0] score_msb_bus;
  logic [N_TOKEN*BOUND_W-1:0] ub_bus, lb_bus;

  // DUT: guarded
  logic [N_TOKEN-1:0] token_keep, payload_en;
  logic               ooo_ready, guard_valid;

  // Ablation: KEEP_ALL
  logic [N_TOKEN-1:0] token_keep_a, payload_en_a;
  logic               ooo_a, guard_valid_a;

  integer i, k;
  int n_pass;
  longint cnt_keep, cnt_drop, cnt_payload_fire, cnt_ooo;
  longint cnt_keep_a, cnt_payload_a;
  int pop_k, pop_d, pop_p;

  c2s_bui_guard_sdsa #(
    .N_TOKEN(N_TOKEN), .SCORE_W(SCORE_W), .BOUND_W(BOUND_W),
    .TH_DROP(8'd16), .ABLATE_KEEP_ALL(1'b0)
  ) dut (
    .clk(clk), .rst_n(rst_n), .bit_round_valid_i(bit_round_valid_i),
    .score_msb_bus(score_msb_bus), .ub_bus(ub_bus), .lb_bus(lb_bus),
    .hbg_g_i(hbg_g_i),
    .token_keep(token_keep), .payload_en(payload_en),
    .ooo_ready(ooo_ready), .guard_valid(guard_valid)
  );

  c2s_bui_guard_sdsa #(
    .N_TOKEN(N_TOKEN), .SCORE_W(SCORE_W), .BOUND_W(BOUND_W),
    .TH_DROP(8'd16), .ABLATE_KEEP_ALL(1'b1)
  ) dut_ablate (
    .clk(clk), .rst_n(rst_n), .bit_round_valid_i(bit_round_valid_i),
    .score_msb_bus(score_msb_bus), .ub_bus(ub_bus), .lb_bus(lb_bus),
    .hbg_g_i(hbg_g_i),
    .token_keep(token_keep_a), .payload_en(payload_en_a),
    .ooo_ready(ooo_a), .guard_valid(guard_valid_a)
  );

  initial begin
    if ($test$plusargs("DUMP_VCD")) begin
      $dumpfile("out/waves/c2s_bui_guard_sdsa.vcd");
      $dumpvars(0, tb_c2s_bui_guard_sdsa);
    end
  end

  initial clk = 1'b0;
  always #5 clk = ~clk;

  function automatic int popc(input logic [N_TOKEN-1:0] b);
    int kk, c;
    begin
      c = 0;
      for (kk = 0; kk < N_TOKEN; kk = kk + 1)
        if (b[kk]) c = c + 1;
      popc = c;
    end
  endfunction

  task automatic clear_in;
    begin
      bit_round_valid_i = 1'b0;
      hbg_g_i = 1'b1;
      score_msb_bus = '0;
      ub_bus = '0;
      lb_bus = '0;
    end
  endtask

  task automatic beat;
    begin
      @(negedge clk);
      bit_round_valid_i = 1'b1;
      @(posedge clk);
      #1;
      bit_round_valid_i = 1'b0;
    end
  endtask

  task automatic accum;
    begin
      pop_k = popc(token_keep);
      pop_d = N_TOKEN - pop_k;
      pop_p = popc(payload_en);
      cnt_keep = cnt_keep + pop_k;
      cnt_drop = cnt_drop + pop_d;
      cnt_payload_fire = cnt_payload_fire + pop_p;
      if (ooo_ready) cnt_ooo = cnt_ooo + 1;
      cnt_keep_a = cnt_keep_a + popc(token_keep_a);
      cnt_payload_a = cnt_payload_a + popc(payload_en_a);
    end
  endtask

  initial begin
    n_pass = 0;
    cnt_keep = 0; cnt_drop = 0; cnt_payload_fire = 0; cnt_ooo = 0;
    cnt_keep_a = 0; cnt_payload_a = 0;
    rst_n = 1'b0;
    clear_in();
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Case1: all ub < TH_DROP → drop all; KEEP_ALL keeps all
    clear_in();
    hbg_g_i = 1'b1;
    for (i = 0; i < N_TOKEN; i = i + 1) begin
      score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd4;
      ub_bus[i*BOUND_W +: BOUND_W] = 8'd8;   // < 16
      lb_bus[i*BOUND_W +: BOUND_W] = 8'd0;
    end
    beat();
    accum();
    if (!guard_valid) begin $display("FAIL Case1: guard_valid"); $fatal(1); end
    if (token_keep !== {N_TOKEN{1'b0}}) begin
      $display("FAIL Case1: token_keep=%b expect 0", token_keep); $fatal(1);
    end
    if (payload_en !== {N_TOKEN{1'b0}}) begin
      $display("FAIL Case1: payload_en=%b", payload_en); $fatal(1);
    end
    if (token_keep_a !== {N_TOKEN{1'b1}}) begin
      $display("FAIL Case1: KEEP_ALL keep=%b", token_keep_a); $fatal(1);
    end
    if (!ooo_ready) begin $display("FAIL Case1: expected ooo"); $fatal(1); end
    $display("PASS Case1: ub<TH → drop-all; KEEP_ALL=all; ooo=1");
    n_pass = n_pass + 1;

    // Case2: tight high bounds + strong score → keep; hbg_g=0 → payload_en=0
    clear_in();
    hbg_g_i = 1'b0;
    for (i = 0; i < N_TOKEN; i = i + 1) begin
      score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd32;
      ub_bus[i*BOUND_W +: BOUND_W] = 8'd40;
      lb_bus[i*BOUND_W +: BOUND_W] = 8'd28;  // width=12 <= 16
    end
    beat();
    accum();
    if (token_keep !== {N_TOKEN{1'b1}}) begin
      $display("FAIL Case2: token_keep=%b expect all", token_keep); $fatal(1);
    end
    if (payload_en !== {N_TOKEN{1'b0}}) begin
      $display("FAIL Case2: payload must AND hbg_g=0 → 0 got %b", payload_en);
      $fatal(1);
    end
    $display("PASS Case2: keep-all but hbg_g=0 → payload_en=0");
    n_pass = n_pass + 1;

    // Case3: split — low-ub drop, tight strong keep
    clear_in();
    hbg_g_i = 1'b1;
    for (i = 0; i < N_TOKEN; i = i + 1) begin
      if (i < 4) begin
        score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd2;
        ub_bus[i*BOUND_W +: BOUND_W] = 8'd10;
        lb_bus[i*BOUND_W +: BOUND_W] = 8'd0;
      end else begin
        score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd40;
        ub_bus[i*BOUND_W +: BOUND_W] = 8'd48;
        lb_bus[i*BOUND_W +: BOUND_W] = 8'd36;
      end
    end
    beat();
    accum();
    if (token_keep !== 8'b11110000) begin
      $display("FAIL Case3: token_keep=%b expect 11110000", token_keep); $fatal(1);
    end
    if (payload_en !== 8'b11110000) begin
      $display("FAIL Case3: payload_en=%b", payload_en); $fatal(1);
    end
    if (token_keep_a !== {N_TOKEN{1'b1}}) begin
      $display("FAIL Case3: KEEP_ALL"); $fatal(1);
    end
    $display("PASS Case3: split keep=%b payload=%b", token_keep, payload_en);
    n_pass = n_pass + 1;

    // Case4: wide uncertainty weak mag → drop; strong mag → keep; ablation fire↓
    clear_in();
    hbg_g_i = 1'b1;
    for (i = 0; i < N_TOKEN; i = i + 1) begin
      if (i < 6) begin
        score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd2;   // weak
        ub_bus[i*BOUND_W +: BOUND_W] = 8'd80;
        lb_bus[i*BOUND_W +: BOUND_W] = 8'd0;           // width=80 > 16
      end else begin
        score_msb_bus[i*SCORE_W +: SCORE_W] = 8'd64;  // strong mag
        ub_bus[i*BOUND_W +: BOUND_W] = 8'd80;
        lb_bus[i*BOUND_W +: BOUND_W] = 8'd0;
      end
    end
    beat();
    accum();
    if (token_keep !== 8'b11000000) begin
      $display("FAIL Case4: token_keep=%b expect 11000000", token_keep); $fatal(1);
    end
    $display("COUNTERS cnt_keep=%0d cnt_drop=%0d cnt_payload_fire=%0d cnt_ooo=%0d",
             cnt_keep, cnt_drop, cnt_payload_fire, cnt_ooo);
    $display("ABLATION BUI_payload=%0d KEEP_ALL_payload=%0d BUI_keep=%0d KEEP_ALL_keep=%0d",
             cnt_payload_fire, cnt_payload_a, cnt_keep, cnt_keep_a);
    if (cnt_payload_fire >= cnt_payload_a) begin
      $display("FAIL Case4: BUI payload_fire should be < KEEP_ALL");
      $fatal(1);
    end
    if (cnt_drop == 0) begin
      $display("FAIL Case4: expected drops"); $fatal(1);
    end
    $display("PASS Case4: wide+weak drop; ablation payload↓; counters OK");
    n_pass = n_pass + 1;

    $display("PASS: all BUI-GuardSDSA cases n_pass=%0d", n_pass);
    $finish;
  end

  initial begin
    #200000;
    $display("FAIL: timeout");
    $fatal(1);
  end
endmodule

`default_nettype wire
