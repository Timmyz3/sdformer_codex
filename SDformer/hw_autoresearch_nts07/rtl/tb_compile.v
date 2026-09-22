`include "nts07_pkg.vh"
`include "atlif_unified_encode_unit.v"
`include "tx_sc_score_unit.v"
`include "shiftmax_unit.v"
`include "sparse_mac_pe.v"
`include "h60_attention_engine.v"

module tb_compile;
    reg clk, rst_n;
    always #5 clk = ~clk;
    initial begin clk=0; rst_n=0; #20 rst_n=1; end

    // Instantiate all modules to check compilation
    atlif_unified_encode_unit u_atlif (
        .clk(clk), .rst_n(rst_n), .en(1'b0), .acc_clear(1'b0),
        .ternary_en(1'b1), .input_acc(16'd0),
        .pos_thresh(16'd100), .neg_thresh(-16'd100),
        .spike_out(), .binary_out(), .pos_fire(), .neg_fire()
    );

    wire [1:0] q_vec[0:31], k_vec[0:31];
    genvar i;
    generate for(i=0;i<32;i=i+1) begin assign q_vec[i]=2'b00; assign k_vec[i]=2'b00; end endgenerate

    wire signed [7:0] tx_s, sc_s, fuse_s;
    wire pair_valid, fuse_valid;
    tx_sc_pair_score #(.HEAD_DIM(32), .SCORE_W(8)) u_pair (
        .clk(clk), .rst_n(rst_n), .en(1'b0),
        .q_ternary(q_vec), .k_ternary(k_vec),
        .alpha0_q8(8'd5), .beta_q8(8'd64), .gamma_q8(8'd38),
        .tx_score(tx_s), .sc_score(sc_s), .valid_out(pair_valid)
    );
    score_fuse_unit #(.SCORE_W(8)) u_fuse (
        .clk(clk), .rst_n(rst_n), .en(pair_valid),
        .tx_in(tx_s), .sc_in(sc_s), .mu_q8(8'd13),
        .row_mean(8'd0), .center_en(1'b0),
        .score_out(fuse_s), .valid_out(fuse_valid)
    );

    wire signed [7:0] s_in[0:97];
    wire [7:0] g_out[0:97];
    wire shift_done;
    generate for(i=0;i<98;i=i+1) assign s_in[i] = 8'd0; endgenerate
    shiftmax_unit #(.MAX_TOKENS(98), .SCORE_W(8), .GATE_W(8)) u_shift (
        .clk(clk), .rst_n(rst_n), .start(1'b0), .n_tokens(7'd98),
        .preserve_mean(1'b0), .scores(s_in), .gates(g_out), .done(shift_done)
    );

    sparse_mac_pe #(.WGT_W(8), .ACC_W(24)) u_pe (
        .clk(clk), .rst_n(rst_n), .en(1'b0),
        .spike_in(1'b0), .neg_spike(1'b0), .weight(8'd0), .acc_clear(1'b0),
        .acc_out()
    );

    wire [1:0] q_load[0:31], k_load[0:31];
    wire signed [15:0] kv_load[0:31];
    wire out_valid;
    wire [6:0] out_idx;
    wire signed [15:0] attn_out[0:31];
    generate for(i=0;i<32;i=i+1) begin assign q_load[i]=2'b00; assign k_load[i]=2'b00; assign kv_load[i]=16'd0; end endgenerate

    h60_attention_engine #(.HEAD_DIM(32), .MAX_TOKENS(98), .ACT_W(16), .SCORE_W(8), .GATE_W(8))
        u_attn (
            .clk(clk), .rst_n(rst_n),
            .start(1'b0), .done(),
            .mu_q8(8'd13), .alpha0_q8(8'd5), .beta_q8(8'd64), .gamma_q8(8'd38),
            .center_scores(1'b0), .preserve_mean(1'b0), .n_tokens(7'd98),
            .load_en(1'b0), .load_qkv_sel(2'd0), .load_idx(7'd0),
            .q_ternary(q_load), .k_ternary(k_load), .k_orig(kv_load),
            .out_valid(out_valid), .out_idx(out_idx), .attn_out(attn_out)
        );

    initial begin
        #100;
        $display("Compile OK");
        $finish;
    end
endmodule
