`timescale 1ns/1ps
`default_nettype none

module qfit_parent_delta_p8_l96_multicontext_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic command_valid,
    input logic command_ready,
    input logic command_accept,
    input logic weight_request_valid,
    input logic weight_request_ready,
    input logic [7:0] weight_request_bank_valid,
    input logic [39:0] weight_request_bank_addr,
    input logic [7:0] weight_request_bank_subtract,
    input logic weight_request_last,
    input logic [1:0] weight_request_context,
    input logic request_accept,
    input logic weight_response_valid,
    input logic weight_response_ready,
    input logic response_accept,
    input logic output_valid,
    input logic output_ready,
    input logic [47:0] output_tag,
    input logic [8:0] output_source_count,
    input logic [1823:0] output_acc,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic [2:0] context_occupancy,
    input logic [4:0] response_metadata_occupancy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_command_accept_definition: assert property (
        command_accept == (command_valid && command_ready));
    ap_request_accept_definition: assert property (
        request_accept == (weight_request_valid && weight_request_ready));
    ap_response_accept_definition: assert property (
        response_accept == (weight_response_valid && weight_response_ready));
    ap_output_accept_definition: assert property (
        output_accept == (output_valid && output_ready));
    ap_context_bound: assert property (context_occupancy <= 4);
    ap_metadata_bound: assert property (response_metadata_occupancy <= 16);
    ap_response_requires_metadata: assert property (
        response_accept |-> response_metadata_occupancy != 0);
    ap_request_has_source: assert property (
        weight_request_valid |-> |weight_request_bank_valid);
    ap_output_stable_under_stall: assert property (
        output_valid && !output_ready && !protocol_error
        |=> protocol_error || (output_valid
            && $stable({output_tag, output_source_count, output_acc})));
    ap_request_stable_under_stall: assert property (
        weight_request_valid && !weight_request_ready && !protocol_error
        |=> protocol_error || (weight_request_valid
            && $stable({weight_request_bank_valid, weight_request_bank_addr,
                        weight_request_bank_subtract, weight_request_last,
                        weight_request_context})));
    ap_fault_is_sticky: assert property (
        protocol_error |=> protocol_error);
    ap_fault_stops_new_work: assert property (
        protocol_error |-> (!command_ready && !weight_request_valid
                            && !weight_response_ready && !output_valid));
    ap_busy_definition: assert property (
        busy == ((context_occupancy != 0)
                 || (response_metadata_occupancy != 0)));

    cp_four_contexts: cover property (context_occupancy == 4);
    cp_metadata_full: cover property (response_metadata_occupancy == 16);
    cp_request_stall: cover property (
        weight_request_valid && !weight_request_ready);
    cp_output_stall: cover property (output_valid && !output_ready);
    cp_request_response_overlap: cover property (
        request_accept && response_accept);
    cp_fault: cover property (protocol_error);

    initial $display("M43_ASSERTION_MODULE_ACTIVE=1");
endmodule

bind qfit_parent_delta_p8_l96_multicontext
    qfit_parent_delta_p8_l96_multicontext_assertions m43_assertions (.*);

`default_nettype wire
