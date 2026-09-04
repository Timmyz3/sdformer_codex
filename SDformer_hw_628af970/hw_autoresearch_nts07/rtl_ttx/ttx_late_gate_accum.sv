`default_nettype none

module ttx_late_gate_accum #(
    parameter int HEAD_DIM    = 32,
    parameter int WEIGHT_W    = 8,
    parameter int GATE_W      = 9,
    parameter int THRESHOLD_W = 8,
    parameter int SUM_W       = WEIGHT_W + $clog2(HEAD_DIM) + 1,
    parameter int OUT_W       = SUM_W + GATE_W + THRESHOLD_W + 1
)(
    input  logic [HEAD_DIM-1:0]              k_bits,
    input  logic signed [HEAD_DIM*WEIGHT_W-1:0] weights_flat,
    input  logic [GATE_W-1:0]                gate_q8,
    input  logic [THRESHOLD_W-1:0]           threshold_q8,
    output logic signed [SUM_W-1:0]          active_weight_sum,
    output logic signed [OUT_W-1:0]          scaled_accum
);
    integer channel_idx;
    logic signed [WEIGHT_W-1:0] weight_value;
    logic [GATE_W+THRESHOLD_W-1:0] shared_scale;

    always_comb begin
        active_weight_sum = '0;
        for (channel_idx = 0; channel_idx < HEAD_DIM; channel_idx = channel_idx + 1) begin
            weight_value = weights_flat[channel_idx*WEIGHT_W +: WEIGHT_W];
            if (k_bits[channel_idx]) begin
                active_weight_sum = active_weight_sum + SUM_W'(weight_value);
            end
        end
        shared_scale = gate_q8 * threshold_q8;
        scaled_accum = OUT_W'(active_weight_sum * $signed({1'b0, shared_scale}));
    end
endmodule

`default_nettype wire
