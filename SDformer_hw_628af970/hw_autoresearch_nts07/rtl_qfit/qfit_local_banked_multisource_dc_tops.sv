`timescale 1ns/1ps
`default_nettype none

`define QFIT_LOCAL_BANKED_DC_TOP(MODULE_NAME, P) \
module MODULE_NAME ( \
    input  logic clk_core, input logic rst_core, \
    input  logic command_valid, output logic command_ready, \
    input  logic [31:0] command_tag, input logic [255:0] command_current_bits, \
    input  logic [511:0] command_seed_acc, \
    output logic weight_request_valid, input logic weight_request_ready, \
    output logic [(P)-1:0] weight_request_bank_valid, \
    output logic [(P)*(8-$clog2(P))-1:0] weight_request_bank_addr, \
    output logic weight_request_last, \
    input  logic weight_response_valid, output logic weight_response_ready, \
    input  logic [(P)-1:0] weight_response_bank_valid, \
    input  logic [(P)*128-1:0] weight_response_data, \
    output logic output_valid, input logic output_ready, output logic [31:0] output_tag, \
    output logic [8:0] output_source_count, output logic [511:0] output_acc, \
    output logic protocol_error \
); \
    qfit_local_banked_multisource_engine #( \
        .TILE_BITS(256), .WORD_BITS(32), .ISSUE_WIDTH(P), .OUT_LANES(16), \
        .TAG_W(32), .W_W(8), .ACC_W(32) \
    ) u_engine (.*); \
endmodule

`QFIT_LOCAL_BANKED_DC_TOP(qfit_local_banked_multisource_p1_top, 1)
`QFIT_LOCAL_BANKED_DC_TOP(qfit_local_banked_multisource_p2_top, 2)
`QFIT_LOCAL_BANKED_DC_TOP(qfit_local_banked_multisource_p4_top, 4)
`QFIT_LOCAL_BANKED_DC_TOP(qfit_local_banked_multisource_p8_top, 8)

`undef QFIT_LOCAL_BANKED_DC_TOP
`default_nettype wire
