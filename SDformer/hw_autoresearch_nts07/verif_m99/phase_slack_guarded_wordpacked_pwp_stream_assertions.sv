`timescale 1ns/1ps
`default_nettype none

module phase_slack_guarded_wordpacked_pwp_stream_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic phase_load_valid,
    input logic phase_load_ready,
    input logic phase_loaded,
    input logic metadata_error,
    input logic lookup_valid,
    input logic lookup_ready,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_width,
    input logic output_escape,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic parse_active,
    input logic [6:0] parse_index,
    input logic [13:0] parse_cursor,
    input logic [2:0] parse_code,
    input logic parse_poison,
    input logic [591:0] captured_metadata,
    input logic lookup_error,
    input logic m82_beat_accept
);
    function automatic logic [13:0] used_words(input logic [2:0] code);
        case (code)
            3'd0: used_words = 14'd24;
            3'd1: used_words = 14'd27;
            3'd2: used_words = 14'd30;
            3'd3: used_words = 14'd33;
            default: used_words = 14'd0;
        endcase
    endfunction

    ap_load_lookup_mutual_exclusion: assert property (@(posedge clk_core)
        disable iff (rst_core) lookup_valid |-> !phase_load_ready);
    ap_simultaneous_request_never_double_accepts: assert property (@(posedge clk_core)
        disable iff (rst_core) phase_load_valid && lookup_valid
        |-> !(phase_load_ready && lookup_ready));
    ap_unloaded_simultaneous_request_accepts_neither: assert property (
        @(posedge clk_core) disable iff (rst_core)
        phase_load_valid && lookup_valid && !phase_loaded
        |-> !phase_load_ready && !lookup_ready);
    ap_parser_starts_at_entry_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) phase_load_valid && phase_load_ready
        |=> parse_active && parse_index == 0 && parse_cursor == 0
            && !phase_loaded);
    ap_parser_index_in_range: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active |-> parse_index <= 127);
    ap_parser_progresses_one_entry: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && parse_index < 127
        |=> parse_active && parse_index == $past(parse_index) + 1'b1);
    ap_parser_finishes_after_entry_127: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && parse_index == 127
        |=> !parse_active && phase_loaded);
    ap_parser_cursor_delta: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active
        |=> parse_cursor == $past(parse_cursor) + used_words($past(parse_code)));
    ap_captured_metadata_stable_during_parse: assert property (
        @(posedge clk_core) disable iff (rst_core)
        parse_active |=> $stable(captured_metadata));
    ap_parser_poison_monotonic: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && parse_poison |=> parse_poison);
    ap_parser_blocks_datapath_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active |-> !lookup_ready && !m82_beat_accept);
    ap_early_lookup_sets_sticky_error: assert property (@(posedge clk_core)
        disable iff (rst_core) parse_active && lookup_valid |=> lookup_error);
    ap_lookup_error_sticky_until_load: assert property (@(posedge clk_core)
        disable iff (rst_core)
        lookup_error && !(phase_load_valid && phase_load_ready)
        |=> lookup_error);
    ap_loaded_stable_until_load: assert property (@(posedge clk_core)
        disable iff (rst_core)
        phase_loaded && !(phase_load_valid && phase_load_ready)
        |=> phase_loaded);
    ap_bad_metadata_blocks_lookup: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> !lookup_ready);
    ap_unloaded_phase_blocks_lookup: assert property (@(posedge clk_core)
        disable iff (rst_core) !phase_loaded |-> !lookup_ready);
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready
        |=> output_valid && $stable({output_tag, output_width,
                                     output_escape, output_values}));
    ap_escape_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) output_valid && output_escape
        |-> output_width == 12 && output_values == '0);
    ap_protocol_reflects_metadata: assert property (@(posedge clk_core)
        disable iff (rst_core) metadata_error |-> protocol_error);

    cp_phase_load: cover property (@(posedge clk_core)
        phase_load_valid && phase_load_ready);
    cp_simultaneous_load_lookup: cover property (@(posedge clk_core)
        !rst_core && phase_load_valid && lookup_valid
        && !phase_load_ready && !lookup_ready);
    cp_loaded_lookup_priority: cover property (@(posedge clk_core)
        !rst_core && phase_loaded && phase_load_valid && lookup_valid
        && !phase_load_ready && lookup_ready);
    cp_parser_first_entry: cover property (@(posedge clk_core)
        parse_active && parse_index == 0);
    cp_parser_middle_entry: cover property (@(posedge clk_core)
        parse_active && parse_index == 63);
    cp_parser_final_entry: cover property (@(posedge clk_core)
        parse_active && parse_index == 127);
    cp_lookup_stall: cover property (@(posedge clk_core)
        lookup_valid && !lookup_ready && busy);
    cp_escape: cover property (@(posedge clk_core)
        output_valid && output_escape);
    cp_width9: cover property (@(posedge clk_core)
        output_valid && output_width == 9);
    cp_width10: cover property (@(posedge clk_core)
        output_valid && output_width == 10);
    cp_width11: cover property (@(posedge clk_core)
        output_valid && output_width == 11);
    cp_metadata_error: cover property (@(posedge clk_core)
        metadata_error && protocol_error && !lookup_ready);
endmodule

`default_nettype wire
