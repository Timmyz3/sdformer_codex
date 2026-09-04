`timescale 1ns/1ps
`default_nettype none

// M86-R2 closes the independent-review deadlock without changing the sealed
// M86-R1 data path.  Before a phase is committed the loader wins contention;
// after a legal phase is committed the descriptor stream wins contention.
// Exactly one request channel is ever presented to the R1 instance.
module arbitrated_sync_banked_guarded_pwp_frontend (
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
    output logic                     descriptor_selected
);
    logic r1_payload_valid, r1_payload_ready, r1_payload_accept;
    logic r1_descriptor_valid, r1_descriptor_ready, r1_descriptor_accept;
    logic r1_phase_loaded;

    always_comb begin
        payload_selected = payload_load_valid
                         && (!r1_phase_loaded || !descriptor_valid);
        descriptor_selected = descriptor_valid
                            && (r1_phase_loaded || !payload_load_valid);
        r1_payload_valid = payload_selected;
        r1_descriptor_valid = descriptor_selected;
        payload_load_ready = payload_selected && r1_payload_ready;
        descriptor_ready = descriptor_selected && r1_descriptor_ready;
        payload_load_accept = r1_payload_accept;
        descriptor_accept = r1_descriptor_accept;
        phase_loaded = r1_phase_loaded;
    end

    sync_banked_guarded_pwp_frontend r1_frontend (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .payload_load_valid(r1_payload_valid),
        .payload_load_ready(r1_payload_ready),
        .payload_load_row(payload_load_row),
        .payload_load_words(payload_load_words),
        .payload_load_accept(r1_payload_accept),
        .phase_load_valid(phase_load_valid),
        .phase_load_ready(phase_load_ready),
        .phase_metadata(phase_metadata),
        .phase_loaded(r1_phase_loaded),
        .metadata_error(metadata_error),
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
        .protocol_error(protocol_error),
        .busy(busy),
        .bank_read_issue(bank_read_issue),
        .bank_read_beat(bank_read_beat),
        .bank_response_enqueue(bank_response_enqueue),
        .response_fifo_level(response_fifo_level)
    );
endmodule

`default_nettype wire
