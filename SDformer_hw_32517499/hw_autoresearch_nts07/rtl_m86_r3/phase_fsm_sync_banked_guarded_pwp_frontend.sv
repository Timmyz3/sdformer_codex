`timescale 1ns/1ps
`default_nettype none

// M86-R3 makes the phase protocol explicit around the unchanged M86-R1 data
// path.  External requests may overlap, but exactly one class is forwarded in
// each state.  A phase contains exactly 460 unique payload rows followed by
// one metadata commit and exactly 128 descriptors before drain.
module phase_fsm_sync_banked_guarded_pwp_frontend (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     payload_load_valid,
    output logic                     payload_load_ready,
    input  logic [9:0]               payload_load_row,
    input  logic [255:0]             payload_load_words,
    output logic                     payload_load_accept,
    input  logic                     phase_load_valid,
    output logic                     phase_load_ready,
    input  logic [591:0]             phase_metadata,
    output logic                     phase_load_accept,
    output logic                     phase_loaded,
    output logic                     metadata_error,
    input  logic                     descriptor_valid,
    output logic                     descriptor_ready,
    input  logic [3:0]               descriptor_pattern,
    input  logic [2:0]               descriptor_block,
    input  logic [31:0]              descriptor_tag,
    output logic                     descriptor_accept,
    output logic                     output_valid,
    input  logic                     output_ready,
    output logic [31:0]              output_tag,
    output logic [3:0]               output_width,
    output logic                     output_escape,
    output logic [96*12-1:0]         output_values,
    output logic                     output_accept,
    output logic                     protocol_error,
    output logic                     busy,
    output logic                     bank_read_issue,
    output logic [2:0]               bank_read_beat,
    output logic                     bank_response_enqueue,
    output logic [2:0]               response_fifo_level,
    output logic                     payload_selected,
    output logic                     phase_selected,
    output logic                     descriptor_selected,
    output logic [2:0]               fsm_state,
    output logic [8:0]               accepted_rows,
    output logic [8:0]               accepted_descriptors
);
    localparam logic [2:0] ST_LOAD    = 3'd0;
    localparam logic [2:0] ST_COMMIT  = 3'd1;
    localparam logic [2:0] ST_EXECUTE = 3'd2;
    localparam logic [2:0] ST_DRAIN   = 3'd3;
    localparam logic [2:0] ST_FAULT   = 3'd4;

    logic [2:0] state_q;
    logic [459:0] row_seen_q;
    logic [8:0] row_count_q;
    logic [8:0] descriptor_count_q;
    logic r1_payload_valid, r1_payload_ready, r1_payload_accept;
    logic r1_phase_valid, r1_phase_ready, r1_phase_loaded;
    logic r1_descriptor_valid, r1_descriptor_ready, r1_descriptor_accept;
    logic r1_metadata_error, r1_protocol_error, r1_busy;

    always_comb begin
        payload_selected = state_q == ST_LOAD && payload_load_valid;
        phase_selected = state_q == ST_COMMIT && phase_load_valid;
        descriptor_selected = state_q == ST_EXECUTE && descriptor_valid;
        r1_payload_valid = payload_selected;
        r1_phase_valid = phase_selected;
        r1_descriptor_valid = descriptor_selected;
        payload_load_ready = payload_selected && r1_payload_ready;
        phase_load_ready = phase_selected && r1_phase_ready;
        descriptor_ready = descriptor_selected && r1_descriptor_ready;
        payload_load_accept = r1_payload_accept;
        phase_load_accept = r1_phase_valid && r1_phase_ready;
        descriptor_accept = r1_descriptor_accept;
        phase_loaded = r1_phase_loaded
                     && (state_q == ST_EXECUTE || state_q == ST_DRAIN);
        metadata_error = r1_metadata_error;
        protocol_error = state_q == ST_FAULT || r1_protocol_error;
        busy = r1_busy || state_q != ST_LOAD || row_count_q != 0;
        fsm_state = state_q;
        accepted_rows = row_count_q;
        accepted_descriptors = descriptor_count_q;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_LOAD;
            row_seen_q <= '0;
            row_count_q <= '0;
            descriptor_count_q <= '0;
        end else if (state_q == ST_FAULT) begin
            state_q <= ST_FAULT;
        end else if (r1_protocol_error || r1_metadata_error) begin
            state_q <= ST_FAULT;
        end else begin
            case (state_q)
                ST_LOAD: begin
                    if (payload_load_accept) begin
                        if (payload_load_row >= 460
                                || row_seen_q[payload_load_row]) begin
                            state_q <= ST_FAULT;
                        end else begin
                            row_seen_q[payload_load_row] <= 1'b1;
                            row_count_q <= row_count_q + 1'b1;
                            if (row_count_q == 9'd459)
                                state_q <= ST_COMMIT;
                        end
                    end
                end
                ST_COMMIT: begin
                    if (phase_load_accept) begin
                        state_q <= ST_EXECUTE;
                        row_seen_q <= '0;
                        row_count_q <= '0;
                        descriptor_count_q <= '0;
                    end
                end
                ST_EXECUTE: begin
                    if (descriptor_accept) begin
                        if (descriptor_count_q == 9'd127) begin
                            descriptor_count_q <= 9'd128;
                            state_q <= ST_DRAIN;
                        end else begin
                            descriptor_count_q <= descriptor_count_q + 1'b1;
                        end
                    end
                end
                ST_DRAIN: begin
                    if (!r1_busy) begin
                        state_q <= ST_LOAD;
                        descriptor_count_q <= '0;
                    end
                end
                default: state_q <= ST_FAULT;
            endcase
        end
    end

    sync_banked_guarded_pwp_frontend r1_frontend (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .payload_load_valid(r1_payload_valid),
        .payload_load_ready(r1_payload_ready),
        .payload_load_row(payload_load_row),
        .payload_load_words(payload_load_words),
        .payload_load_accept(r1_payload_accept),
        .phase_load_valid(r1_phase_valid),
        .phase_load_ready(r1_phase_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(r1_phase_loaded),
        .metadata_error(r1_metadata_error),
        .descriptor_valid(r1_descriptor_valid),
        .descriptor_ready(r1_descriptor_ready),
        .descriptor_pattern(descriptor_pattern),
        .descriptor_block(descriptor_block),
        .descriptor_tag(descriptor_tag),
        .descriptor_accept(r1_descriptor_accept),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .output_tag(output_tag),
        .output_width(output_width),
        .output_escape(output_escape),
        .output_values(output_values),
        .output_accept(output_accept),
        .protocol_error(r1_protocol_error),
        .busy(r1_busy),
        .bank_read_issue(bank_read_issue),
        .bank_read_beat(bank_read_beat),
        .bank_response_enqueue(bank_response_enqueue),
        .response_fifo_level(response_fifo_level)
    );
endmodule

`default_nettype wire
