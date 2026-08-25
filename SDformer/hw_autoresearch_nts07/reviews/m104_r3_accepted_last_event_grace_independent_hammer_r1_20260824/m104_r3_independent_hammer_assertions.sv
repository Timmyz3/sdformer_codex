`timescale 1ns/1ps
`default_nettype none

module m104_r3_independent_hammer_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic event_valid,
    input logic event_ready,
    input logic [3:0] event_source,
    input logic [2:0] event_block,
    input logic event_negate,
    input logic event_last_for_key,
    input logic [31:0] event_tag,
    input logic event_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [31:0] output_tag,
    input logic [3:0] output_source,
    input logic [2:0] output_block,
    input logic output_negate,
    input logic [1151:0] output_values,
    input logic output_accept,
    input logic protocol_error,
    input logic illegal_request,
    input logic accepted_event_grace_match,
    input logic request_fault,
    input logic output_valid_q
);
    ap_last_exact_linger_is_grace: assert property (@(posedge clk_core)
        disable iff (rst_core)
        $past(event_accept && event_last_for_key) && event_valid
        && {event_source, event_block, event_negate,
            event_last_for_key, event_tag}
           == $past({event_source, event_block, event_negate,
                     event_last_for_key, event_tag})
        |-> accepted_event_grace_match && !illegal_request
            && !protocol_error && !event_ready && !event_accept);

    ap_nonlast_exact_linger_is_grace: assert property (@(posedge clk_core)
        disable iff (rst_core)
        $past(event_accept && !event_last_for_key) && event_valid
        && {event_source, event_block, event_negate,
            event_last_for_key, event_tag}
           == $past({event_source, event_block, event_negate,
                     event_last_for_key, event_tag})
        |-> accepted_event_grace_match && !illegal_request
            && !protocol_error && !event_ready && !event_accept);

    ap_stalled_output_stable: assert property (@(posedge clk_core)
        disable iff (rst_core)
        $past(output_valid && !output_ready && !protocol_error)
        && !protocol_error
        |-> output_valid
            && $stable({output_tag, output_source, output_block,
                        output_negate, output_values}));

    ap_same_cycle_fault_quarantine: assert property (@(posedge clk_core)
        disable iff (rst_core)
        illegal_request
        |-> protocol_error && !event_ready && !event_accept
            && !output_valid && !output_accept);

    ap_fault_preserves_older_buffer: assert property (@(posedge clk_core)
        disable iff (rst_core)
        illegal_request && output_valid_q
        |=> request_fault && output_valid_q && !output_valid
            && !output_accept);

    ap_sticky_fault: assert property (@(posedge clk_core)
        disable iff (rst_core) request_fault |=> request_fault);

    cp_last_exact_linger: cover property (@(posedge clk_core)
        disable iff (rst_core)
        $past(event_accept && event_last_for_key) && event_valid
        && accepted_event_grace_match && !event_accept);

    cp_nonlast_exact_linger: cover property (@(posedge clk_core)
        disable iff (rst_core)
        $past(event_accept && !event_last_for_key) && event_valid
        && accepted_event_grace_match && !event_accept);

    cp_same_cycle_buffer_quarantine: cover property (@(posedge clk_core)
        disable iff (rst_core)
        illegal_request && output_valid_q && !output_valid && !output_accept);
endmodule

`default_nettype wire
