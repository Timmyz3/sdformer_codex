`timescale 1ns/1ps
`default_nettype none

module h67_denominator_certificate_assertions #(
    parameter int PAIRS = 225,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int QCOUNT_W = 6,
    parameter int QCOUNT_LIMIT = 15,
    parameter int CERTIFIED_SHIFT = 17
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         row_load_start,
    input logic                         load_accept,
    input logic [PAIR_ID_W-1:0]         load_pair_id,
    input logic                         certificate_valid,
    input logic                         certificate_pass,
    input logic [5:0]                   denominator_shift,
    input logic [QCOUNT_W-1:0]          row_qcount_max,
    input logic [$clog2(PAIRS+1)-1:0]   accepted_pairs,
    input logic                         protocol_error
);
    property p_pass_is_sound;
        @(posedge clk_core) disable iff (rst_core)
        certificate_pass |-> certificate_valid
                         && row_qcount_max <= QCOUNT_W'(QCOUNT_LIMIT)
                         && denominator_shift == 6'(CERTIFIED_SHIFT);
    endproperty
    assert property (p_pass_is_sound);

    property p_nonpass_shift_is_zero;
        @(posedge clk_core) disable iff (rst_core)
        !certificate_pass |-> denominator_shift == 0;
    endproperty
    assert property (p_nonpass_shift_is_zero);

    property p_valid_only_after_complete_row;
        @(posedge clk_core) disable iff (rst_core)
        certificate_valid |-> 32'(accepted_pairs) == 32'(PAIRS);
    endproperty
    assert property (p_valid_only_after_complete_row);

    property p_legal_last_pair_certifies;
        @(posedge clk_core) disable iff (rst_core)
        load_accept && !row_load_start && !protocol_error
            && 32'(accepted_pairs) == 32'(PAIRS - 1)
            && 32'(load_pair_id) == 32'(PAIRS - 1)
        |=> certificate_valid;
    endproperty
    assert property (p_legal_last_pair_certifies);

    property p_certificate_holds_until_new_row;
        @(posedge clk_core) disable iff (rst_core)
        certificate_valid && !row_load_start
        |=> certificate_valid || row_load_start;
    endproperty
    assert property (p_certificate_holds_until_new_row);
endmodule

bind h67_row_qmax_denominator_certificate_core
    h67_denominator_certificate_assertions #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .QCOUNT_W(QCOUNT_W),
        .QCOUNT_LIMIT(QCOUNT_LIMIT),
        .CERTIFIED_SHIFT(CERTIFIED_SHIFT)
    ) u_h67_denominator_certificate_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_load_start(row_load_start),
        .load_accept(load_accept),
        .load_pair_id(load_pair_id),
        .certificate_valid(certificate_valid),
        .certificate_pass(certificate_pass),
        .denominator_shift(denominator_shift),
        .row_qcount_max(row_qcount_max),
        .accepted_pairs(accepted_pairs),
        .protocol_error(protocol_error)
    );

`default_nettype wire
