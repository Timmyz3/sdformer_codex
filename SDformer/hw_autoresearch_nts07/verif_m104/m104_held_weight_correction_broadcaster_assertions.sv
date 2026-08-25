`timescale 1ns/1ps
`default_nettype none

module m104_held_weight_correction_broadcaster_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic load_valid,
    input logic load_ready,
    input logic [1:0] load_beat,
    input logic load_accept,
    input logic event_valid,
    input logic event_ready,
    input logic event_negate,
    input logic event_last_for_key,
    input logic event_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_source,
    input logic [2:0] output_block,
    input logic output_negate,
    input logic [96*12-1:0] output_values,
    input logic output_accept,
    input logic held_valid,
    input logic collecting,
    input logic [1:0] expected_load_beat,
    input logic protocol_error,
    input logic illegal_request,
    input logic accepted_event_grace_match,
    input logic request_fault,
    input logic output_valid_q
);
    ap_no_double_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) !(load_accept && event_accept));
    ap_load_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core) load_accept == (load_valid && load_ready));
    ap_event_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept == (event_valid && event_ready));
    ap_accepted_event_grace_is_not_a_fault: assert property (
        @(posedge clk_core) disable iff (rst_core || request_fault)
        event_valid && accepted_event_grace_match
        |-> !illegal_request && !protocol_error && !event_ready
            && !event_accept);
    ap_first_load_starts_collection: assert property (@(posedge clk_core)
        disable iff (rst_core) load_accept && load_beat == 0 && !collecting
        |=> collecting && expected_load_beat == 1);
    ap_middle_load_advances: assert property (@(posedge clk_core)
        disable iff (rst_core) load_accept && load_beat == 1
        |=> collecting && expected_load_beat == 2);
    ap_final_load_exposes_held_vector: assert property (@(posedge clk_core)
        disable iff (rst_core) load_accept && load_beat == 2
        |=> held_valid && !collecting);
    ap_event_requires_held_vector: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept |-> held_valid);
    ap_nonlast_event_keeps_weight: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept && !event_last_for_key
        |=> held_valid);
    ap_last_event_releases_weight: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept && event_last_for_key
        |=> !held_valid);
    ap_event_buffers_result: assert property (@(posedge clk_core)
        disable iff (rst_core) event_accept |=> output_valid_q);
    ap_output_accept_equivalence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_accept == (output_valid && output_ready));
    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && !output_ready && !illegal_request
        |=> protocol_error
            || (output_valid
                && $stable({output_tag, output_source, output_block,
                            output_negate, output_values})));
    ap_fault_reflected: assert property (@(posedge clk_core)
        disable iff (rst_core) request_fault |-> protocol_error);
    ap_fault_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) request_fault |=> request_fault);
    ap_fault_quarantines_interfaces: assert property (@(posedge clk_core)
        disable iff (rst_core) protocol_error
        |-> !load_ready && !event_ready && !output_valid && !output_accept);

    cp_three_load_beats: cover property (@(posedge clk_core)
        disable iff (rst_core)
        load_accept && load_beat == 0 ##1
        load_accept && load_beat == 1 ##1
        load_accept && load_beat == 2);
    cp_positive_event: cover property (@(posedge clk_core)
        event_accept && !event_negate);
    cp_negative_event: cover property (@(posedge clk_core)
        event_accept && event_negate);
    cp_consecutive_events: cover property (@(posedge clk_core)
        disable iff (rst_core) event_accept ##1 event_accept);
    cp_output_stall: cover property (@(posedge clk_core)
        output_valid && !output_ready);
    cp_last_releases_key: cover property (@(posedge clk_core)
        event_accept && event_last_for_key);
    cp_protocol_fault: cover property (@(posedge clk_core)
        protocol_error && !load_ready && !event_ready);
    cp_fault_quarantines_buffered_output: cover property (@(posedge clk_core)
        protocol_error && output_valid_q && !output_valid && !output_accept);
    cp_accepted_event_grace: cover property (@(posedge clk_core)
        disable iff (rst_core)
        event_valid && accepted_event_grace_match && output_valid
            && !event_ready && !event_accept);
endmodule

`default_nettype wire
