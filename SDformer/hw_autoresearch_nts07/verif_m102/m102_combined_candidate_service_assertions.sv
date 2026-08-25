`timescale 1ns/1ps
`default_nettype none

module m102_combined_candidate_service_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic phase_load_valid,
    input logic phase_load_ready,
    input logic phase_loaded,
    input logic metadata_error,
    input logic service_valid,
    input logic service_ready,
    input logic [1:0] service_kind,
    input logic [2:0] service_beat,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [1:0] output_kind,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic parse_active,
    input logic [6:0] parse_index,
    input logic transaction_active,
    input logic [2:0] expected_beat,
    input logic request_fault,
    input logic m82_beat_accept,
    input logic m82_output_valid,
    input logic output_negate,
    input logic request_last,
    input logic request_semantically_valid,
    input logic request_violation,
    input logic accepted_grace_match
);
    ap_load_service_mutual_exclusion: assert property (@(posedge clk_core)
        disable iff (rst_core) service_valid |-> !phase_load_ready);
    ap_no_double_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) phase_load_valid && service_valid
        |-> !(phase_load_ready && service_ready));
    ap_parser_starts: assert property (@(posedge clk_core)
        disable iff (rst_core) phase_load_valid && phase_load_ready
        |=> parse_active && parse_index == 0 && !phase_loaded);
    ap_parser_progresses: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && parse_index < 127
        |=> parse_active && parse_index == $past(parse_index) + 1'b1);
    ap_parser_finishes: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && parse_index == 127
        |=> !parse_active && phase_loaded);
    ap_parse_blocks_service: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active |-> !service_ready && !m82_beat_accept);
    ap_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core) m82_beat_accept == (service_valid && service_ready));
    ap_first_beat_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) m82_beat_accept && !transaction_active
        |-> service_beat == 0);
    ap_continuation_monotonic: assert property (@(posedge clk_core)
        disable iff (rst_core) m82_beat_accept && transaction_active
        |-> service_beat == expected_beat);
    ap_fault_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core)
        request_fault |=> request_fault);
    ap_fault_blocks_phase_reload: assert property (@(posedge clk_core)
        disable iff (rst_core)
        request_fault |-> !phase_load_ready);
    ap_bad_metadata_blocks_service: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> !service_ready);
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        && !(service_valid && !request_semantically_valid)
        |=> protocol_error
            || (output_valid
                && $stable({output_tag, output_kind, output_width,
                            output_escape, output_values})));
    ap_no_zero_escape_token: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid |-> !output_escape);
    ap_weight_width8: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_kind inside {2'd1, 2'd2}
        |-> output_width == 8);
    ap_pwp_width_range: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_kind == 0
        |-> output_width inside {4'd8, 4'd9, 4'd10, 4'd11});
    ap_protocol_reflects_fault: assert property (@(posedge clk_core)
        disable iff (rst_core) request_fault |-> protocol_error);
    ap_same_cycle_bad_request_quarantines_output: assert property (
        @(posedge clk_core) disable iff (rst_core)
        request_violation |-> protocol_error && !service_ready
            && !phase_load_ready && !output_valid && !output_accept);
    ap_accepted_request_grace_is_not_a_fault: assert property (
        @(posedge clk_core) disable iff (rst_core || request_fault
                                         || metadata_error)
        service_valid && accepted_grace_match
        |-> !request_violation && !protocol_error && !service_ready);
    ap_fault_quarantines_output: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error
        |-> !output_valid && !output_accept && output_tag == 0
            && output_kind == 0 && output_width == 0 && output_values == 0);

    cp_pwp: cover property (@(posedge clk_core)
        output_valid && output_kind == 0);
    cp_positive_correction: cover property (@(posedge clk_core)
        output_valid && output_kind == 1 && !output_negate);
    cp_negative_correction: cover property (@(posedge clk_core)
        output_valid && output_kind == 1 && output_negate);
    cp_fallback: cover property (@(posedge clk_core)
        output_valid && output_kind == 2);
    cp_stall: cover property (@(posedge clk_core)
        output_valid && !output_ready);
    cp_protocol_fault: cover property (@(posedge clk_core)
        protocol_error && !service_ready);
    cp_fault_quarantines_buffered_output: cover property (@(posedge clk_core)
        protocol_error && m82_output_valid && !output_valid && !output_accept);
    cp_same_cycle_release_quarantine: cover property (@(posedge clk_core)
        request_violation && output_ready && m82_output_valid
        && !output_valid && !output_accept);
    cp_accepted_request_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        service_valid && accepted_grace_match && output_valid);
    cp_fault_blocks_phase_reload: cover property (@(posedge clk_core)
        request_fault && phase_load_valid && !phase_load_ready);
    cp_metadata_error: cover property (@(posedge clk_core)
        metadata_error && !service_ready && !output_valid);
    cp_pwp_to_correction_seam: cover property (@(posedge clk_core)
        disable iff (rst_core)
        m82_beat_accept && service_beat == 0 && service_kind == 1
        && $past(m82_beat_accept && request_last && service_kind == 0));
endmodule

`default_nettype wire
