`timescale 1ns/1ps
`default_nettype none

// Locks one replay route for the complete slot-word session. The persistent
// payload tag is checked before each word is admitted to a decoder.
module gatestack_slot_replay_word_router #(
    parameter int TAG_W = 32,
    parameter int WORD_INDEX_W = 7,
    parameter int FORMAT_W = 2,
    parameter int ROUTE_W = 2
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         session_start_valid,
    output logic                         session_start_ready,
    input  logic [ROUTE_W-1:0]           session_route,
    input  logic [FORMAT_W-1:0]          session_format,
    input  logic [TAG_W-1:0]             session_payload_tag,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [63:0]                  input_data,
    input  logic [WORD_INDEX_W-1:0]      input_index,
    input  logic                         input_last,
    input  logic [TAG_W-1:0]             input_payload_tag,
    input  logic                         input_mode_is_csr,
    input  logic [FORMAT_W-1:0]          input_format,

    output logic                         resident_valid,
    input  logic                         resident_ready,
    output logic [63:0]                  resident_data,
    output logic [WORD_INDEX_W-1:0]      resident_index,
    output logic                         resident_last,
    output logic                         ipd_valid,
    input  logic                         ipd_ready,
    output logic [63:0]                  ipd_data,
    output logic [WORD_INDEX_W-1:0]      ipd_index,
    output logic                         ipd_last,
    output logic                         raw_valid,
    input  logic                         raw_ready,
    output logic [63:0]                  raw_data,
    output logic [WORD_INDEX_W-1:0]      raw_index,
    output logic                         raw_last,
    output logic                         session_active,
    output logic                         protocol_error
);
    localparam logic [ROUTE_W-1:0] ROUTE_RESIDENT = ROUTE_W'(0);
    localparam logic [ROUTE_W-1:0] ROUTE_IPD = ROUTE_W'(1);
    localparam logic [ROUTE_W-1:0] ROUTE_RAW = ROUTE_W'(2);
    localparam logic [FORMAT_W-1:0] FORMAT_RAW = FORMAT_W'(0);
    localparam logic [FORMAT_W-1:0] FORMAT_IPD32W = FORMAT_W'(1);
    localparam logic [FORMAT_W-1:0] FORMAT_FADC24 = FORMAT_W'(2);
    logic [ROUTE_W-1:0] route_q;
    logic [FORMAT_W-1:0] format_q;
    logic [TAG_W-1:0] payload_tag_q;
    logic route_legal, format_legal, route_format_legal;
    logic mode_legal, metadata_legal;
    logic start_fire, input_fire;

    assign route_legal = session_route == ROUTE_RESIDENT ||
                         session_route == ROUTE_IPD ||
                         session_route == ROUTE_RAW;
    assign format_legal = session_format == FORMAT_RAW ||
                          session_format == FORMAT_IPD32W ||
                          session_format == FORMAT_FADC24;
    assign route_format_legal =
        (session_route == ROUTE_RESIDENT &&
         session_format == FORMAT_IPD32W) ||
        (session_route == ROUTE_IPD &&
         (session_format == FORMAT_IPD32W ||
          session_format == FORMAT_FADC24)) ||
        (session_route == ROUTE_RAW && session_format == FORMAT_RAW);
    assign session_start_ready = !session_active && route_legal &&
                                 format_legal && route_format_legal;
    assign start_fire = session_start_valid && session_start_ready;
    assign mode_legal = route_q == ROUTE_RAW ? !input_mode_is_csr :
                                               input_mode_is_csr;
    assign metadata_legal = input_payload_tag == payload_tag_q && mode_legal &&
                            input_format == format_q;

    always_comb begin
        input_ready = 1'b0;
        resident_valid = 1'b0;
        ipd_valid = 1'b0;
        raw_valid = 1'b0;
        if (session_active && metadata_legal) begin
            if (route_q == ROUTE_RESIDENT) begin
                resident_valid = input_valid;
                input_ready = resident_ready;
            end else if (route_q == ROUTE_IPD) begin
                ipd_valid = input_valid;
                input_ready = ipd_ready;
            end else if (route_q == ROUTE_RAW) begin
                raw_valid = input_valid;
                input_ready = raw_ready;
            end
        end
    end

    assign resident_data = input_data;
    assign resident_index = input_index;
    assign resident_last = input_last;
    assign ipd_data = input_data;
    assign ipd_index = input_index;
    assign ipd_last = input_last;
    assign raw_data = input_data;
    assign raw_index = input_index;
    assign raw_last = input_last;
    assign input_fire = input_valid && input_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            route_q <= ROUTE_RESIDENT;
            format_q <= FORMAT_RAW;
            payload_tag_q <= '0;
            session_active <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (session_start_valid &&
                (!route_legal || !format_legal || !route_format_legal))
                protocol_error <= 1'b1;
            if (start_fire) begin
                route_q <= session_route;
                format_q <= session_format;
                payload_tag_q <= session_payload_tag;
                session_active <= 1'b1;
            end
            if (session_active && input_valid && !metadata_legal)
                protocol_error <= 1'b1;
            if (input_fire && input_last)
                session_active <= 1'b0;
        end
    end
endmodule

`default_nettype wire
