`default_nettype none

bind h67_score_class_row_engine h67_h68_protocol_assertions #(
    .HEAD_DIM(HEAD_DIM),
    .TOKEN_W(TOKEN_W),
    .GATE_W(GATE_W),
    .THRESHOLD_W(THRESHOLD_W),
    .SCORE_CLASS_DEPTH(SCORE_CLASS_DEPTH),
    .CLASS_COUNT_W(CLASS_COUNT_W)
) u_protocol_assertions (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .in_last(in_last),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_last(out_last),
    .out_token_idx(out_token_idx),
    .out_k_bits(out_k_bits),
    .out_gate_q8(out_gate_q8),
    .out_threshold_q8(out_threshold_q8),
    .busy(busy),
    .done(done),
    .perf_score_range_error(perf_score_range_error),
    .class_present(class_present_q),
    .classes_remaining(classes_remaining_q),
    .class_inflight(state_q == ST_SUM_FOLD)
);

`default_nettype wire
