`default_nettype none

module h67_h68_protocol_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int TOKEN_W = 8,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int SCORE_CLASS_DEPTH = 35,
    parameter int CLASS_COUNT_W = 6
)(
    input logic clk,
    input logic rst_n,
    input logic in_valid,
    input logic in_ready,
    input logic in_last,
    input logic out_valid,
    input logic out_ready,
    input logic out_last,
    input logic [TOKEN_W-1:0] out_token_idx,
    input logic [HEAD_DIM-1:0] out_k_bits,
    input logic [GATE_W-1:0] out_gate_q8,
    input logic [THRESHOLD_W-1:0] out_threshold_q8,
    input logic busy,
    input logic done,
    input logic perf_score_range_error,
    input logic [SCORE_CLASS_DEPTH-1:0] class_present,
    input logic [CLASS_COUNT_W-1:0] classes_remaining,
    input logic class_inflight
);
    function automatic integer popcount_classes(input logic [SCORE_CLASS_DEPTH-1:0] value);
        integer idx;
        begin
            popcount_classes = 0;
            for (idx = 0; idx < SCORE_CLASS_DEPTH; idx = idx + 1) begin
                popcount_classes = popcount_classes + value[idx];
            end
        end
    endfunction
    property p_output_stable_under_backpressure;
        @(posedge clk) disable iff (!rst_n)
        out_valid && !out_ready |=> out_valid
            && $stable({out_last, out_token_idx, out_k_bits, out_gate_q8, out_threshold_q8});
    endproperty

    property p_done_implies_busy;
        @(posedge clk) disable iff (!rst_n)
        done |-> busy;
    endproperty

    property p_last_requires_valid;
        @(posedge clk) disable iff (!rst_n)
        out_last |-> out_valid;
    endproperty

    property p_no_frozen_score_overflow;
        @(posedge clk) disable iff (!rst_n)
        !perf_score_range_error;
    endproperty

    property p_class_count_matches_bitmap;
        @(posedge clk) disable iff (!rst_n)
        classes_remaining == CLASS_COUNT_W'(popcount_classes(class_present) + class_inflight);
    endproperty

    a_output_stable_under_backpressure: assert property (p_output_stable_under_backpressure);
    a_done_implies_busy: assert property (p_done_implies_busy);
    a_last_requires_valid: assert property (p_last_requires_valid);
    a_no_frozen_score_overflow: assert property (p_no_frozen_score_overflow);
    a_class_count_matches_bitmap: assert property (p_class_count_matches_bitmap);

    c_input_backpressure: cover property (@(posedge clk) disable iff (!rst_n) in_valid && !in_ready);
    c_output_backpressure: cover property (@(posedge clk) disable iff (!rst_n) out_valid && !out_ready);
    c_last_handshake: cover property (@(posedge clk) disable iff (!rst_n) out_valid && out_ready && out_last);
    c_input_last: cover property (@(posedge clk) disable iff (!rst_n) in_valid && in_ready && in_last);
endmodule

`default_nettype wire
