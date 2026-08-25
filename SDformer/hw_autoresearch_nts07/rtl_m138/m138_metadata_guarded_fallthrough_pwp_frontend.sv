`timescale 1ns/1ps
`default_nettype none

// Minimal executable macro-aligned PWP frontend.  A state-aware metadata guard
// suppresses every SRAM read whose illegality is decidable before data access;
// valid data beats pass through M137's one-cycle tagged fall-through bridge and
// then into the unchanged M133 signed assembler.  Final-padding violations are
// intentionally detected after the returned data because they are not
// metadata-decidable.
module m138_metadata_guarded_fallthrough_pwp_frontend (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         beat_valid,
    output logic                         beat_ready,
    input  logic                         beat_start,
    input  logic                         beat_last,
    input  logic [3:0]                   beat_width,
    input  logic [31:0]                  beat_tag,
    input  logic [11:0]                  logical_base_word,
    output logic                         beat_accept,

    output logic                         macro_request_valid,
    output logic [15:0]                  macro_bank_read_enable,
    output logic                         macro_bank_conflict_free,
    output logic [127:0]                 macro_bank_row_addresses,
    output logic [15:0]                  macro_request_token,
    input  logic                         macro_response_valid,
    input  logic [15:0]                  macro_response_token,
    input  logic [511:0]                 macro_bank_words,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [31:0]                  output_tag,
    output logic [3:0]                   output_width,
    output logic                         output_escape,
    output logic [96*12-1:0]             output_values,
    output logic                         output_accept,

    output logic                         protocol_error,
    output logic                         metadata_fault,
    output logic                         collecting,
    output logic                         busy
);
    logic guard_collecting_q;
    logic guard_fault_q;
    logic [1:0] guard_accepted_beats_q;
    logic [1:0] guard_beats_needed_q;
    logic guard_expected_last;
    logic metadata_violation;
    logic metadata_legal;
    logic escape_candidate;
    logic data_candidate;

    logic bridge_request_valid;
    logic bridge_request_ready;
    logic bridge_request_accept;
    logic bridge_macro_request_valid;
    logic [127:0] bridge_macro_addresses;
    logic [15:0] bridge_macro_token;
    logic bridge_response_valid;
    logic bridge_response_ready;
    logic [511:0] bridge_response_words;
    logic bridge_response_start;
    logic bridge_response_last;
    logic [3:0] bridge_response_width;
    logic [31:0] bridge_response_tag;
    logic [15:0] bridge_response_token;
    logic bridge_response_accept;
    logic bridge_protocol_error;
    logic bridge_pending;
    logic [1:0] bridge_buffered;
    logic bridge_busy;

    logic assembler_beat_valid;
    logic assembler_beat_ready;
    logic assembler_beat_accept;
    logic assembler_protocol_error;
    logic assembler_collecting;
    logic assembler_busy;
    logic assembler_output_valid;
    logic assembler_output_accept;
    logic quarantine;
    logic pre_assembler_quarantine;
    logic downstream_fault_q;
    logic escape_to_assembler;

    assign guard_expected_last = guard_accepted_beats_q + 1'b1
                               == guard_beats_needed_q;

    always_comb begin : metadata_precheck
        metadata_violation = 1'b0;
        escape_candidate = 1'b0;
        data_candidate = 1'b0;
        if (beat_valid) begin
            if (!guard_collecting_q) begin
                if (!beat_start) begin
                    metadata_violation = 1'b1;
                end else if (beat_width == 4'd12) begin
                    if (!beat_last)
                        metadata_violation = 1'b1;
                    else
                        escape_candidate = 1'b1;
                end else if (beat_width inside {4'd8,4'd9,4'd10,4'd11}) begin
                    if (beat_last)
                        metadata_violation = 1'b1;
                    else
                        data_candidate = 1'b1;
                end else begin
                    metadata_violation = 1'b1;
                end
            end else begin
                if (beat_start || beat_width != 0 || beat_tag != 0
                        || beat_last != guard_expected_last)
                    metadata_violation = 1'b1;
                else
                    data_candidate = 1'b1;
            end
        end
    end

    assign metadata_legal = beat_valid && !metadata_violation;
    // Do not feed the assembler's combinational request check back into its
    // own input valid/ready cone.  A data-dependent final-padding violation is
    // only knowable when the macro response arrives, so latch it at this
    // boundary and stop all following traffic.  This keeps the ready/valid
    // graph acyclic while allowing at most the already-overlapped request in
    // the detecting cycle; metadata-decidable faults remain zero-read.
    assign pre_assembler_quarantine = guard_fault_q || metadata_violation
                                    || downstream_fault_q
                                    || bridge_protocol_error;
    assign quarantine = pre_assembler_quarantine || assembler_protocol_error;

    assign bridge_request_valid = metadata_legal && data_candidate
                                && !guard_fault_q && !downstream_fault_q;

    m137_fallthrough_tagged_16bank_response_bridge bridge (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .request_valid(bridge_request_valid),
        .request_ready(bridge_request_ready),
        .logical_base_word(logical_base_word),
        .request_start(beat_start),
        .request_last(beat_last),
        .request_width(beat_width),
        .request_tag(beat_tag),
        .request_accept(bridge_request_accept),
        .macro_request_valid(bridge_macro_request_valid),
        .macro_bank_row_addresses(bridge_macro_addresses),
        .macro_request_token(bridge_macro_token),
        .macro_response_valid(macro_response_valid),
        .macro_response_token(macro_response_token),
        .macro_bank_words(macro_bank_words),
        .response_valid(bridge_response_valid),
        .response_ready(bridge_response_ready),
        .response_logical_words(bridge_response_words),
        .response_start(bridge_response_start),
        .response_last(bridge_response_last),
        .response_width(bridge_response_width),
        .response_tag(bridge_response_tag),
        .response_token(bridge_response_token),
        .response_accept(bridge_response_accept),
        .protocol_error(bridge_protocol_error),
        .pending_response(bridge_pending),
        .buffered_responses(bridge_buffered),
        .busy(bridge_busy)
    );

    // Escape traffic is data-free and may bypass the macro only after every
    // older bank response has drained, preventing reordering.
    assign escape_to_assembler = metadata_legal && escape_candidate
                               && !bridge_busy && !pre_assembler_quarantine;
    assign assembler_beat_valid = !pre_assembler_quarantine
                                && (bridge_response_valid
                                    || escape_to_assembler);
    assign bridge_response_ready = assembler_beat_ready
                                 && !pre_assembler_quarantine;

    m133_dualrow512_elastic_pwp_stream assembler (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .beat_valid(assembler_beat_valid),
        .beat_ready(assembler_beat_ready),
        .beat_start(escape_to_assembler ? beat_start : bridge_response_start),
        .beat_last(escape_to_assembler ? beat_last : bridge_response_last),
        .beat_width(escape_to_assembler ? beat_width : bridge_response_width),
        .beat_tag(escape_to_assembler ? beat_tag : bridge_response_tag),
        .beat_data(escape_to_assembler ? 512'b0 : bridge_response_words),
        .beat_accept(assembler_beat_accept),
        .output_valid(assembler_output_valid),
        .output_ready(output_ready),
        .output_tag(output_tag),
        .output_width(output_width),
        .output_escape(output_escape),
        .output_values(output_values),
        .output_accept(assembler_output_accept),
        .protocol_error(assembler_protocol_error),
        .collecting(assembler_collecting),
        .busy(assembler_busy)
    );

    assign beat_ready = !rst_core && !pre_assembler_quarantine
                      && (data_candidate ? bridge_request_ready
                          : escape_candidate ? (!bridge_busy
                                                && assembler_beat_ready)
                          : !beat_valid);
    assign beat_accept = bridge_request_accept
                       || (escape_to_assembler && assembler_beat_accept);

    // Only a real accepted macro transaction may toggle the SRAM boundary.
    assign macro_request_valid = bridge_macro_request_valid
                               && !pre_assembler_quarantine;
    assign macro_bank_read_enable = macro_request_valid ? 16'hffff : 16'h0000;
    assign macro_bank_conflict_free = macro_request_valid;
    assign macro_bank_row_addresses = macro_request_valid
                                    ? bridge_macro_addresses : '0;
    assign macro_request_token = macro_request_valid ? bridge_macro_token : '0;

    assign output_valid = !rst_core && !quarantine && assembler_output_valid;
    assign output_accept = !quarantine && assembler_output_accept;
    assign protocol_error = !rst_core && quarantine;
    assign metadata_fault = !rst_core && (guard_fault_q || metadata_violation);
    assign collecting = guard_collecting_q;
    assign busy = bridge_busy || assembler_busy || guard_collecting_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            guard_collecting_q <= 1'b0;
            guard_fault_q <= 1'b0;
            downstream_fault_q <= 1'b0;
            guard_accepted_beats_q <= '0;
            guard_beats_needed_q <= '0;
        end else begin
            if (metadata_violation)
                guard_fault_q <= 1'b1;
            if (assembler_protocol_error)
                downstream_fault_q <= 1'b1;

            if (!pre_assembler_quarantine && beat_accept) begin
                if (!guard_collecting_q) begin
                    if (escape_candidate) begin
                        guard_collecting_q <= 1'b0;
                        guard_accepted_beats_q <= '0;
                        guard_beats_needed_q <= '0;
                    end else begin
                        guard_collecting_q <= 1'b1;
                        guard_accepted_beats_q <= 2'd1;
                        if (beat_width == 4'd11)
                            guard_beats_needed_q <= 2'd3;
                        else
                            guard_beats_needed_q <= 2'd2;
                    end
                end else if (guard_expected_last) begin
                    guard_collecting_q <= 1'b0;
                    guard_accepted_beats_q <= '0;
                    guard_beats_needed_q <= '0;
                end else begin
                    guard_accepted_beats_q <= guard_accepted_beats_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
