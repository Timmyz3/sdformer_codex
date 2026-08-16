`default_nettype none

// Batched numeric checker for real checkpoint projection weights. This is a
// verification sidecar, not the throughput-oriented projection backend.
module h67_gated_k_projection16_acc #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int OUT_CHANNELS = 16
) (
    input  logic                                  clk,
    input  logic                                  rst_core,
    input  logic                                  row_start,
    input  logic                                  row_done,
    input  logic                                  in_fire,
    input  logic                                  in_last,
    input  logic [HEAD_DIM-1:0]                   in_k_bits,
    input  logic [GATE_W-1:0]                     in_gate_q17,
    input  logic [OUT_CHANNELS*HEAD_DIM*8-1:0]    weight_flat,
    output logic                                  result_valid,
    output logic [OUT_CHANNELS*32-1:0]            result_acc32_flat
);
    logic signed [31:0] acc_q [0:OUT_CHANNELS-1];
    logic signed [31:0] contribution_w [0:OUT_CHANNELS-1];
    logic signed [31:0] next_acc_w [0:OUT_CHANNELS-1];
    logic saw_input_q;
    logic result_emitted_q;
    integer comb_channel;
    integer comb_lane;
    integer seq_channel;

    always_comb begin
        for (comb_channel = 0; comb_channel < OUT_CHANNELS;
             comb_channel = comb_channel + 1) begin
            contribution_w[comb_channel] = '0;
            for (comb_lane = 0; comb_lane < HEAD_DIM;
                 comb_lane = comb_lane + 1) begin
                if (in_k_bits[comb_lane]) begin
                    contribution_w[comb_channel] = contribution_w[comb_channel]
                        + $signed(weight_flat[
                            (comb_channel*HEAD_DIM + comb_lane)*8 +: 8
                        ]) * $signed({1'b0, in_gate_q17});
                end
            end
            next_acc_w[comb_channel] = acc_q[comb_channel]
                + contribution_w[comb_channel];
        end
    end

    always_ff @(posedge clk) begin
        if (rst_core) begin
            for (seq_channel = 0; seq_channel < OUT_CHANNELS;
                 seq_channel = seq_channel + 1) begin
                acc_q[seq_channel] <= '0;
                result_acc32_flat[seq_channel*32 +: 32] <= '0;
            end
            saw_input_q <= 1'b0;
            result_emitted_q <= 1'b0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= 1'b0;
            if (row_start) begin
                for (seq_channel = 0; seq_channel < OUT_CHANNELS;
                     seq_channel = seq_channel + 1)
                    acc_q[seq_channel] <= '0;
                saw_input_q <= 1'b0;
                result_emitted_q <= 1'b0;
            end else if (in_fire) begin
                for (seq_channel = 0; seq_channel < OUT_CHANNELS;
                     seq_channel = seq_channel + 1)
                    acc_q[seq_channel] <= next_acc_w[seq_channel];
                saw_input_q <= 1'b1;
                if (in_last) begin
                    result_valid <= 1'b1;
                    result_emitted_q <= 1'b1;
                    for (seq_channel = 0; seq_channel < OUT_CHANNELS;
                         seq_channel = seq_channel + 1)
                        result_acc32_flat[seq_channel*32 +: 32]
                            <= next_acc_w[seq_channel];
                end
            end else if (row_done && !saw_input_q && !result_emitted_q) begin
                result_valid <= 1'b1;
                result_emitted_q <= 1'b1;
                for (seq_channel = 0; seq_channel < OUT_CHANNELS;
                     seq_channel = seq_channel + 1)
                    result_acc32_flat[seq_channel*32 +: 32]
                        <= acc_q[seq_channel];
            end
        end
    end
endmodule

`default_nettype wire
