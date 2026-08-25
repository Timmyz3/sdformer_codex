`timescale 1ns/1ps
`default_nettype none

// M186: flat-composable K8 FC2 request/response issue island.
//
// M184 emits structural bank masks and addresses.  A one-outstanding in-order
// request slot backpressures that frontend, accepts a later banked weight
// response plus its external Acc24 context, and drives M185 without bank IDs,
// prefix packing or a lane crossbar.  Token completion is withheld until both
// the outstanding response and the arithmetic result have drained.
//
// The actual SRAM macros, descriptor producer, accumulator-context store,
// BN2 and residual commit remain outside this island.
module m186_fc2_k8_fixed_bank_issue_island #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         header_valid,
    output logic                         header_ready,
    input  logic [TAG_BITS-1:0]          header_tag,
    input  logic [3:0]                   header_output_blocks,
    input  logic [5:0]                   header_descriptor_count,
    output logic                         header_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [4:0]                   descriptor_beat_index,
    input  logic [95:0]                  descriptor_bitmap,
    output logic                         descriptor_accept,

    output logic                         weight_request_valid,
    input  logic                         weight_request_ready,
    output logic [TAG_BITS-1:0]          weight_request_tag,
    output logic [2:0]                   weight_request_output_block,
    output logic [3:0]                   weight_request_source_count,
    output logic [7:0]                   weight_request_bank_valid,
    output logic [CHANNEL_BITS-1:0]      weight_request_source_channel [0:7],
    output logic                         weight_request_accept,

    input  logic                         weight_response_valid,
    output logic                         weight_response_ready,
    input  logic signed [7:0]            weight_response [0:7][0:LANES-1],
    input  logic signed [23:0]           accumulator_context [0:LANES-1],
    output logic                         weight_response_accept,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_token_tag,
    output logic [2:0]                   result_output_block,
    output logic [3:0]                   result_source_count,
    output logic [7:0]                   result_bank_mask,
    output logic signed [23:0]           result_accumulator [0:LANES-1],
    output logic                         result_accept,

    output logic                         token_done_valid,
    input  logic                         token_done_ready,
    output logic [TAG_BITS-1:0]          token_done_tag,
    output logic                         token_done_had_event,
    output logic                         token_done_accept,

    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         busy
);
    logic local_fault_q;
    logic new_admission_open;
    logic illegal_response;

    logic m184_header_valid;
    logic m184_header_ready;
    logic m184_header_accept;
    logic m184_descriptor_valid;
    logic m184_descriptor_ready;
    logic m184_descriptor_accept;
    logic m184_group_valid;
    logic m184_group_ready;
    logic [TAG_BITS-1:0] m184_group_tag;
    logic [2:0] m184_group_output_block;
    logic [3:0] m184_group_source_count;
    logic [7:0] m184_group_bank_valid;
    logic [CHANNEL_BITS-1:0] m184_group_source_channel [0:7];
    logic m184_group_accept;
    logic m184_done_valid;
    logic m184_done_ready;
    logic [TAG_BITS-1:0] m184_done_tag;
    logic m184_done_had_event;
    logic m184_done_accept;
    logic m184_protocol_error;
    logic m184_busy;

    logic pending_valid_q;
    logic [TAG_BITS-1:0] pending_tag_q;
    logic [2:0] pending_output_block_q;
    logic [7:0] pending_bank_valid_q;
    logic request_slot_open;

    logic m185_issue_valid;
    logic m185_issue_ready;
    logic m185_issue_accept;
    logic [TAG_BITS+2:0] m185_issue_tag;
    logic m185_result_valid;
    logic [TAG_BITS+2:0] m185_result_tag;
    logic m185_result_last;
    logic m185_result_accept;
    logic [8*LANES-1:0] m185_activity_unused;
    logic m185_protocol_error;
    logic m185_overflow;
    logic m185_busy;

    assign protocol_error = local_fault_q || m184_protocol_error
        || m185_protocol_error;
    assign numeric_overflow = m185_overflow;
    // M184 must see a malformed header/descriptor so it can raise its own
    // fail-closed fault; excluding its combinational protocol_error here also
    // avoids a legality/admission combinational loop.
    assign new_admission_open = !local_fault_q
        && !m185_protocol_error && !numeric_overflow;

    assign m184_header_valid = header_valid && new_admission_open;
    assign header_ready = m184_header_ready && new_admission_open;
    assign header_accept = m184_header_accept;
    assign m184_descriptor_valid = descriptor_valid && new_admission_open;
    assign descriptor_ready = m184_descriptor_ready && new_admission_open;
    assign descriptor_accept = m184_descriptor_accept;

    assign request_slot_open = !pending_valid_q || weight_response_accept;
    assign weight_request_valid = m184_group_valid
        && request_slot_open && new_admission_open && !m184_protocol_error;
    assign weight_request_accept = weight_request_valid
        && weight_request_ready;
    assign m184_group_ready = request_slot_open && weight_request_ready
        && new_admission_open && !m184_protocol_error;
    assign weight_request_tag = m184_group_tag;
    assign weight_request_output_block = m184_group_output_block;
    assign weight_request_source_count = m184_group_source_count;
    assign weight_request_bank_valid = m184_group_bank_valid;
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_request_channel
            assign weight_request_source_channel[bank]
                = m184_group_source_channel[bank];
        end
    endgenerate

    // Responses are in order and must arrive at least one cycle after their
    // accepted request.  An unsolicited response is a sticky local fault.
    assign illegal_response = weight_response_valid && !pending_valid_q;
    assign weight_response_ready = pending_valid_q
        && m185_issue_ready && !local_fault_q;
    assign weight_response_accept = weight_response_valid
        && weight_response_ready;
    assign m185_issue_valid = weight_response_valid
        && pending_valid_q && !local_fault_q;
    assign m185_issue_tag = {pending_tag_q, pending_output_block_q};

    // A frontend token is not architecturally complete until the outstanding
    // weight request and arithmetic result register have both drained.
    assign token_done_valid = m184_done_valid
        && !pending_valid_q && !m185_busy;
    assign m184_done_ready = token_done_ready
        && !pending_valid_q && !m185_busy;
    assign token_done_accept = m184_done_accept;
    assign token_done_tag = m184_done_tag;
    assign token_done_had_event = m184_done_had_event;

    assign result_valid = m185_result_valid;
    assign result_token_tag = m185_result_tag[TAG_BITS+2:3];
    assign result_output_block = m185_result_tag[2:0];
    assign result_accept = m185_result_accept;
    assign busy = m184_busy || pending_valid_q || m185_busy;

    always_ff @(posedge clk_core) begin : pending_response_slot
        if (rst_core) begin
            local_fault_q <= 1'b0;
            pending_valid_q <= 1'b0;
            pending_tag_q <= '0;
            pending_output_block_q <= '0;
            pending_bank_valid_q <= '0;
        end else begin
            if (illegal_response)
                local_fault_q <= 1'b1;
            if (weight_response_accept && !weight_request_accept)
                pending_valid_q <= 1'b0;
            if (weight_request_accept) begin
                pending_valid_q <= 1'b1;
                pending_tag_q <= weight_request_tag;
                pending_output_block_q <= weight_request_output_block;
                pending_bank_valid_q <= weight_request_bank_valid;
            end
        end
    end

    m184_fc2_dual_window_k8_fixed_bank_frontend #(
        .TAG_BITS(TAG_BITS),
        .CHANNEL_BITS(CHANNEL_BITS),
        .MAX_WINDOW_DESCRIPTORS(8)
    ) frontend (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .header_valid(m184_header_valid),
        .header_ready(m184_header_ready),
        .header_tag(header_tag),
        .header_output_blocks(header_output_blocks),
        .header_descriptor_count(header_descriptor_count),
        .header_accept(m184_header_accept),
        .descriptor_valid(m184_descriptor_valid),
        .descriptor_ready(m184_descriptor_ready),
        .descriptor_beat_index(descriptor_beat_index),
        .descriptor_bitmap(descriptor_bitmap),
        .descriptor_accept(m184_descriptor_accept),
        .group_valid(m184_group_valid),
        .group_ready(m184_group_ready),
        .group_tag(m184_group_tag),
        .group_output_block(m184_group_output_block),
        .group_source_count(m184_group_source_count),
        .group_bank_valid(m184_group_bank_valid),
        .group_source_channel(m184_group_source_channel),
        .group_accept(m184_group_accept),
        .token_done_valid(m184_done_valid),
        .token_done_ready(m184_done_ready),
        .token_done_tag(m184_done_tag),
        .token_done_had_event(m184_done_had_event),
        .token_done_accept(m184_done_accept),
        .protocol_error(m184_protocol_error),
        .busy(m184_busy)
    );

    m185_fc2_k8_fixed_bank_accumulator #(
        .LANES(LANES),
        .TAG_BITS(TAG_BITS + 3)
    ) accumulator (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .issue_valid(m185_issue_valid),
        .issue_ready(m185_issue_ready),
        .issue_tag(m185_issue_tag),
        .issue_last(1'b0),
        .issue_bank_valid(pending_bank_valid_q),
        .issue_weight(weight_response),
        .issue_accumulator(accumulator_context),
        .issue_accept(m185_issue_accept),
        .result_valid(m185_result_valid),
        .result_ready(result_ready),
        .result_tag(m185_result_tag),
        .result_last(m185_result_last),
        .result_source_count(result_source_count),
        .result_bank_mask(result_bank_mask),
        .result_accumulator(result_accumulator),
        .result_accept(m185_result_accept),
        .accepted_weight_active_mask(m185_activity_unused),
        .protocol_error(m185_protocol_error),
        .numeric_overflow(m185_overflow),
        .busy(m185_busy)
    );
endmodule

`default_nettype wire
