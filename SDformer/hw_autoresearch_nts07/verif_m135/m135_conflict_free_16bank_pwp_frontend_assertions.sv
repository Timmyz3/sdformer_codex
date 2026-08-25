`timescale 1ns/1ps
`default_nettype none

module m135_conflict_free_16bank_pwp_frontend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic beat_valid,
    input logic beat_ready,
    input logic beat_start,
    input logic [3:0] beat_width,
    input logic [11:0] logical_base_word,
    input logic [127:0] bank_row_addresses,
    input logic [15:0] bank_use_mask,
    input logic bank_conflict_free,
    input logic beat_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_beat_accept_definition:
        assert property (beat_accept == (beat_valid && beat_ready));
    ap_output_accept_definition:
        assert property (output_accept == (output_valid && output_ready));
    ap_fault_quarantines_all_traffic:
        assert property (protocol_error
                         |-> !(beat_ready || beat_accept
                              || output_valid || output_accept));
    ap_stall_stable_or_fault:
        assert property (output_valid && !output_ready
                         |=> protocol_error
                             || (output_valid
                                 && $stable({output_tag, output_width,
                                             output_escape,
                                             output_values})));
    ap_escape_has_no_bank_read:
        assert property (beat_valid && beat_start && beat_width == 12
                         |-> bank_row_addresses == 0);
    ap_accepted_data_base_in_range:
        assert property (beat_accept
                         && !(beat_start && beat_width == 12)
                         |-> ({1'b0, logical_base_word} + 13'd15
                              < 13'd3680));
    ap_accepted_data_uses_all_banks:
        assert property (beat_accept
                         && !(beat_start && beat_width == 12)
                         |-> bank_use_mask == 16'hffff
                             && bank_conflict_free);

    cp_cross_row_bank_mapping:
        cover property (beat_accept && logical_base_word[3:0] != 0);
    cp_last_legal_bank_window:
        cover property (beat_accept && logical_base_word == 12'd3664);
    cp_output_stall_release:
        cover property (output_valid && !output_ready
                        ##1 output_valid && output_ready);
    cp_invalid_base_quarantine:
        cover property (protocol_error && logical_base_word == 12'd3665
                        && !beat_accept && !output_valid);
endmodule

`default_nettype wire
