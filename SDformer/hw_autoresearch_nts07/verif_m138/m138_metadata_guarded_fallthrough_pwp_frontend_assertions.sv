`timescale 1ns/1ps
`default_nettype none

module m138_metadata_guarded_fallthrough_pwp_frontend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic beat_valid,
    input logic beat_ready,
    input logic beat_start,
    input logic beat_last,
    input logic [3:0] beat_width,
    input logic [31:0] beat_tag,
    input logic beat_accept,
    input logic macro_request_valid,
    input logic [15:0] macro_bank_read_enable,
    input logic macro_bank_conflict_free,
    input logic [127:0] macro_bank_row_addresses,
    input logic [15:0] macro_request_token,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic metadata_fault,
    input logic collecting,
    input logic downstream_fault_q
);
`ifdef SVA_RUNTIME_ENABLED
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_accept_requires_handshake: assert property (
        beat_accept |-> beat_valid && beat_ready);
    ap_macro_request_is_accepted_data_beat: assert property (
        macro_request_valid |-> beat_accept && beat_width != 4'd12);
    ap_macro_boundary_complete: assert property (
        macro_request_valid |-> macro_bank_read_enable == 16'hffff
                             && macro_bank_conflict_free);
    ap_macro_boundary_quiet_without_request: assert property (
        !macro_request_valid |-> macro_bank_read_enable == 0
                              && !macro_bank_conflict_free
                              && macro_bank_row_addresses == 0
                              && macro_request_token == 0);
    ap_metadata_fault_suppresses_sram: assert property (
        metadata_fault |-> protocol_error && !beat_ready && !beat_accept
                       && !macro_request_valid && macro_bank_read_enable == 0
                       && !macro_bank_conflict_free
                       && macro_bank_row_addresses == 0
                       && macro_request_token == 0);
    ap_idle_continuation_suppressed: assert property (
        beat_valid && !collecting && !beat_start
        |-> metadata_fault && !macro_request_valid);
    ap_idle_unsupported_width_suppressed: assert property (
        beat_valid && !collecting && beat_start
        && !(beat_width inside {4'd8,4'd9,4'd10,4'd11,4'd12})
        |-> metadata_fault && !macro_request_valid);
    ap_idle_premature_last_suppressed: assert property (
        beat_valid && !collecting && beat_start
        && beat_width inside {4'd8,4'd9,4'd10,4'd11} && beat_last
        |-> metadata_fault && !macro_request_valid);
    ap_collecting_restart_suppressed: assert property (
        beat_valid && collecting && beat_start
        |-> metadata_fault && !macro_request_valid);
    ap_no_output_on_error: assert property (
        protocol_error |-> !output_valid && !output_accept);
    ap_output_accept_definition: assert property (
        output_accept |-> output_valid && output_ready);
    ap_stalled_output_stable: assert property (
        output_valid && !output_ready && !protocol_error
        |=> protocol_error || (output_valid
            && $stable({output_tag, output_width, output_escape,
                        output_values})));
    ap_error_sticky: assert property (protocol_error |=> protocol_error);
    ap_registered_downstream_fault_quiet: assert property (
        downstream_fault_q |-> protocol_error && !beat_ready && !beat_accept
                            && !macro_request_valid
                            && macro_bank_read_enable == 0);
    ap_reset_quiet: assert property (@(posedge clk_core) disable iff (1'b0)
        rst_core |-> !beat_accept && !macro_request_valid
                    && macro_bank_read_enable == 0
                    && macro_bank_row_addresses == 0
                    && !output_valid && !protocol_error);

    cp_macro_request_cross_row: cover property (
        macro_request_valid && |macro_bank_row_addresses);
    cp_escape_without_macro: cover property (
        beat_accept && beat_width == 4'd12 && !macro_request_valid);
    cp_output_stall_release: cover property (
        output_valid && !output_ready ##1 output_valid && output_ready);
    cp_metadata_suppressed_read: cover property (
        metadata_fault && !macro_request_valid
        && macro_bank_read_enable == 0 && macro_bank_row_addresses == 0);
    cp_restart_suppressed: cover property (
        beat_valid && collecting && beat_start ##0 metadata_fault);
    cp_data_fault_registered: cover property (
        protocol_error && !metadata_fault ##1 downstream_fault_q
        && protocol_error && !beat_ready);
`endif
endmodule

bind m138_metadata_guarded_fallthrough_pwp_frontend
    m138_metadata_guarded_fallthrough_pwp_frontend_assertions m138_sva (.*);

`default_nettype wire
