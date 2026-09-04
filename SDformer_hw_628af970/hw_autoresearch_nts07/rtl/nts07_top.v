`include "nts07_pkg.vh"

// NTS-11bc+ unified H60 top: controller + H60 engine + sparse MAC (no Legacy attn).
module nts07_top #(
    parameter integer LANES = 8,
    parameter integer HEAD_DIM = 32,
    parameter integer MAX_TOKENS = 98
)(
    input  wire clk,
    input  wire rst_n,
    input  wire frame_start,
    output wire frame_done,
    input  wire [7:0] mu_q8,
    // Sparse MAC streaming interface
    input  wire in_valid,
    output wire in_ready,
    input  wire [LANES-1:0] spike_in,
    input  wire signed [LANES*8-1:0] weight_in,
    output wire signed [23:0] mac_acc [0:LANES-1]
);
    wire h60_start, h60_done;
    wire [1:0] engine_id;
    wire window_enable;
    wire [31:0] perf_cycles;

    reg h60_done_r;
    reg h60_busy;

    nts07_controller u_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .frame_done(frame_done),
        .stage_id(),
        .block_id(),
        .engine_id(engine_id),
        .window_enable(window_enable),
        .h60_start(h60_start),
        .h60_done(h60_done_r),
        .perf_cycles(perf_cycles)
    );

    // H60 engine hookup (ternary/k arrays stubbed to zero in top integration test)
    wire [1:0] q_t [0:MAX_TOKENS-1][0:HEAD_DIM-1];
    wire [1:0] k_t [0:MAX_TOKENS-1][0:HEAD_DIM-1];
    wire signed [15:0] k_o [0:MAX_TOKENS-1][0:HEAD_DIM-1];
    wire signed [15:0] attn [0:MAX_TOKENS-1][0:HEAD_DIM-1];

    h60_attention_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS)
    ) u_h60 (
        .clk(clk),
        .rst_n(rst_n),
        .start(h60_start),
        .done(h60_done),
        .mu_q8(mu_q8),
        .center_scores(1'b1),
        .preserve_mean(1'b1),
        .n_tokens(7'd98),
        .q_ternary(q_t),
        .k_ternary(k_t),
        .k_orig(k_o),
        .attn_out(attn)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h60_done_r <= 1'b0;
            h60_busy <= 1'b0;
        end else begin
            if (h60_start)
                h60_busy <= 1'b1;
            if (h60_done) begin
                h60_done_r <= 1'b1;
                h60_busy <= 1'b0;
            end else if (!h60_start) begin
                h60_done_r <= 1'b0;
            end
        end
    end

    sparse_mac_lane #(.LANES(LANES)) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .fire(in_valid & in_ready & (engine_id == `ENG_SPARSE_MAC)),
        .spike_vec(spike_in),
        .weight_vec(weight_in),
        .acc_clear(1'b0),
        .acc_lane(mac_acc)
    );

    assign in_ready = 1'b1;
endmodule