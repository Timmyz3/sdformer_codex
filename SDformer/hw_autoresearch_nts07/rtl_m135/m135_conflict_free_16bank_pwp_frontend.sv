`timescale 1ns/1ps
`default_nettype none

// M135 composes the exhaustive M134 16-bank mapper with the unchanged M133
// elastic signed-vector assembler.  The input is one word from each of sixteen
// banks plus a logical base word; bank macros and their read latency remain a
// port cut.
module m135_conflict_free_16bank_pwp_frontend (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         beat_valid,
    output logic                         beat_ready,
    input  logic                         beat_start,
    input  logic                         beat_last,
    input  logic [3:0]                   beat_width,
    input  logic [31:0]                  beat_tag,
    input  logic [11:0]                  logical_base_word,
    input  logic [511:0]                 bank_words,
    output logic [127:0]                 bank_row_addresses,
    output logic [15:0]                  bank_use_mask,
    output logic                         bank_conflict_free,
    output logic                         beat_accept,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [31:0]                  output_tag,
    output logic [3:0]                   output_width,
    output logic                         output_escape,
    output logic [96*12-1:0]             output_values,
    output logic                         output_accept,
    output logic                         protocol_error,
    output logic                         collecting,
    output logic                         busy
);
    logic mapper_request_valid;
    logic mapper_request_legal;
    logic [511:0] mapper_logical_words;
    logic mapper_violation;
    logic mapper_fault_q;
    logic quarantine;

    logic m133_beat_valid;
    logic m133_beat_ready;
    logic m133_beat_accept;
    logic m133_output_valid;
    logic m133_output_accept;
    logic m133_protocol_error;
    logic m133_busy;

    assign mapper_request_valid = beat_valid
                                && !(beat_start && beat_width == 4'd12);

    m134_conflict_free_16bank_dualrow_mapper mapper (
        .request_valid(mapper_request_valid),
        .logical_base_word(logical_base_word),
        .bank_words(bank_words),
        .request_legal(mapper_request_legal),
        .bank_row_addresses(bank_row_addresses),
        .logical_words(mapper_logical_words),
        .bank_use_mask(bank_use_mask),
        .conflict_free(bank_conflict_free)
    );

    assign mapper_violation = mapper_request_valid
                            && (!mapper_request_legal
                                || bank_use_mask != 16'hffff
                                || !bank_conflict_free);
    assign quarantine = mapper_fault_q || mapper_violation;
    assign m133_beat_valid = beat_valid && !quarantine
                           && (!mapper_request_valid || mapper_request_legal);

    m133_dualrow512_elastic_pwp_stream assembler (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .beat_valid(m133_beat_valid),
        .beat_ready(m133_beat_ready),
        .beat_start(beat_start),
        .beat_last(beat_last),
        .beat_width(beat_width),
        .beat_tag(beat_tag),
        .beat_data(mapper_request_valid ? mapper_logical_words : 512'b0),
        .beat_accept(m133_beat_accept),
        .output_valid(m133_output_valid),
        .output_ready(output_ready),
        .output_tag(output_tag),
        .output_width(output_width),
        .output_escape(output_escape),
        .output_values(output_values),
        .output_accept(m133_output_accept),
        .protocol_error(m133_protocol_error),
        .collecting(collecting),
        .busy(m133_busy)
    );

    assign beat_ready = !rst_core && !quarantine && m133_beat_ready;
    assign beat_accept = !quarantine && m133_beat_accept;
    assign output_valid = !rst_core && !quarantine && m133_output_valid;
    assign output_accept = !quarantine && m133_output_accept;
    assign protocol_error = !rst_core
                          && (quarantine || m133_protocol_error);
    assign busy = m133_busy;

    always_ff @(posedge clk_core) begin
        if (rst_core)
            mapper_fault_q <= 1'b0;
        else if (mapper_violation)
            mapper_fault_q <= 1'b1;
    end
endmodule

`default_nettype wire
