`timescale 1ns/1ps
`default_nettype none

// Block-phased lossless K4 source fold.
//
// Sixteen 96-lane signed-INT8 weight vectors (1536 logical bytes) are resident
// for exactly one output block. Each accepted row mask is partitioned in
// canonical lowest-set-bit groups of at most four. The selected weights are
// optionally negated per source, added lane-wise in signed11 and sign-extended
// into one signed19 accumulator delta. No saturation or rounding is used.
module m125_block_phased_k4_row_fold #(
    parameter int SOURCES = 16,
    parameter int LANES = 96,
    parameter int WEIGHT_BITS = 8,
    parameter int WEIGHT_VECTOR_BITS = LANES * WEIGHT_BITS,
    parameter int BEAT_BITS = 256,
    parameter int ACC_BITS = 19,
    parameter int UPDATE_BITS = LANES * ACC_BITS
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         weight_fill_valid,
    output logic                         weight_fill_ready,
    input  logic [2:0]                   weight_fill_block,
    input  logic [3:0]                   weight_fill_source,
    input  logic [1:0]                   weight_fill_beat,
    input  logic [BEAT_BITS-1:0]         weight_fill_data,
    output logic                         weight_fill_accept,

    input  logic                         row_valid,
    output logic                         row_ready,
    input  logic [2:0]                   row_block,
    input  logic [8:0]                   row_offset,
    input  logic [SOURCES-1:0]           row_source_mask,
    input  logic [SOURCES-1:0]           row_negate_mask,
    output logic                         row_accept,

    output logic                         update_valid,
    input  logic                         update_ready,
    output logic [2:0]                   update_block,
    output logic [8:0]                   update_row,
    output logic [UPDATE_BITS-1:0]       update_delta,
    output logic [SOURCES-1:0]           update_selected_mask,
    output logic                         update_accept,
    output logic                         row_done,

    output logic [SOURCES-1:0]           observed_remaining_mask,
    output logic [SOURCES-1:0]           observed_cache_valid,
    output logic                         observed_resident_block_valid,
    output logic [2:0]                   observed_resident_block,
    output logic                         protocol_error,
    output logic                         busy
);
    logic request_fault_q;
    logic resident_block_valid_q;
    logic [2:0] resident_block_q;
    logic [SOURCES-1:0] cache_valid_q;
    logic [WEIGHT_VECTOR_BITS-1:0] weight_cache_q [0:SOURCES-1];

    logic fill_active_q;
    logic [2:0] fill_block_q;
    logic [3:0] fill_source_q;
    logic [1:0] expected_fill_beat_q;
    logic [WEIGHT_VECTOR_BITS-1:0] fill_payload_q;

    logic row_active_q;
    logic [2:0] row_block_q;
    logic [8:0] row_offset_q;
    logic [SOURCES-1:0] row_mask_q;
    logic [SOURCES-1:0] row_negate_q;

    logic request_collision;
    logic fill_semantically_valid;
    logic row_semantically_valid;
    logic illegal_request;
    logic [SOURCES-1:0] selected_mask;
    logic [SOURCES-1:0] remainder_mask;
    logic [3:0] selected_source [0:3];
    logic [3:0] selected_valid;
    logic signed [WEIGHT_BITS-1:0] selected_weight [0:3][0:LANES-1];
    logic signed [10:0] selected_contribution [0:3][0:LANES-1];
    logic signed [10:0] lane_fold_sum [0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (SOURCES != 16 || LANES != 96 || WEIGHT_BITS != 8
                || WEIGHT_VECTOR_BITS != 768 || BEAT_BITS != 256
                || ACC_BITS != 19 || UPDATE_BITS != 1824)
            $fatal(1, "M125 production geometry drift");
    end
`endif

    always_comb begin : request_audit
        request_collision = weight_fill_valid && row_valid;
        if (weight_fill_beat == 0) begin
            fill_semantically_valid = !fill_active_q && !row_active_q;
        end else begin
            fill_semantically_valid = fill_active_q && !row_active_q
                && weight_fill_block == fill_block_q
                && weight_fill_source == fill_source_q
                && weight_fill_beat == expected_fill_beat_q;
        end
        row_semantically_valid = !fill_active_q && !row_active_q
            && resident_block_valid_q
            && row_block == resident_block_q
            && !(|(row_source_mask & ~cache_valid_q))
            && !(|(row_negate_mask & ~row_source_mask));
        illegal_request = request_collision
                        || (weight_fill_valid && !fill_semantically_valid)
                        || (row_valid && !row_semantically_valid);
    end

    assign protocol_error = request_fault_q || illegal_request;
    assign weight_fill_ready = !protocol_error && fill_semantically_valid
                             && !row_valid;
    assign row_ready = !protocol_error && row_semantically_valid
                     && !weight_fill_valid;
    assign weight_fill_accept = weight_fill_valid && weight_fill_ready;
    assign row_accept = row_valid && row_ready;
    assign update_valid = row_active_q && |selected_mask && !protocol_error;
    assign update_accept = update_valid && update_ready;
    assign update_block = row_block_q;
    assign update_row = row_offset_q;
    assign update_selected_mask = selected_mask;
    assign observed_remaining_mask = row_mask_q;
    assign observed_cache_valid = cache_valid_q;
    assign observed_resident_block_valid = resident_block_valid_q;
    assign observed_resident_block = resident_block_q;
    assign busy = fill_active_q || row_active_q || update_valid;

    always_comb begin : canonical_select_and_fold
        logic found;
        logic signed [10:0] weight_ext;
        selected_mask = '0;
        remainder_mask = row_mask_q;
        selected_valid = '0;
        for (int pick = 0; pick < 4; pick++) begin
            selected_source[pick] = '0;
            found = 1'b0;
            for (int source = 0; source < SOURCES; source++) begin
                if (!found && remainder_mask[source]) begin
                    selected_source[pick] = source[3:0];
                    selected_valid[pick] = 1'b1;
                    selected_mask[source] = 1'b1;
                    remainder_mask[source] = 1'b0;
                    found = 1'b1;
                end
            end
        end

        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            lane_fold_sum[lane] = '0;
            for (int pick = 0; pick < 4; pick++) begin
                selected_weight[pick][lane] = '0;
                selected_contribution[pick][lane] = '0;
                if (selected_valid[pick]) begin
                    selected_weight[pick][lane]
                        = weight_cache_q[selected_source[pick]]
                          [lane * WEIGHT_BITS +: WEIGHT_BITS];
                    weight_ext = {{(11-WEIGHT_BITS)
                                   {selected_weight[pick][lane]
                                    [WEIGHT_BITS-1]}},
                                  selected_weight[pick][lane]};
                    selected_contribution[pick][lane]
                        = row_negate_q[selected_source[pick]]
                        ? -weight_ext : weight_ext;
                end
                lane_fold_sum[lane]
                    = lane_fold_sum[lane]
                    + selected_contribution[pick][lane];
            end
            update_delta[lane * ACC_BITS +: ACC_BITS]
                = {{(ACC_BITS-11){lane_fold_sum[lane][10]}},
                   lane_fold_sum[lane]};
        end
    end

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            request_fault_q <= 1'b0;
            resident_block_valid_q <= 1'b0;
            resident_block_q <= '0;
            cache_valid_q <= '0;
            fill_active_q <= 1'b0;
            fill_block_q <= '0;
            fill_source_q <= '0;
            expected_fill_beat_q <= '0;
            fill_payload_q <= '0;
            row_active_q <= 1'b0;
            row_block_q <= '0;
            row_offset_q <= '0;
            row_mask_q <= '0;
            row_negate_q <= '0;
            row_done <= 1'b0;
            for (int source = 0; source < SOURCES; source++)
                weight_cache_q[source] <= '0;
        end else begin
            row_done <= 1'b0;
            if (illegal_request)
                request_fault_q <= 1'b1;

            if (!request_fault_q && !illegal_request && weight_fill_accept) begin
                if (weight_fill_beat == 0) begin
                    fill_active_q <= 1'b1;
                    fill_block_q <= weight_fill_block;
                    fill_source_q <= weight_fill_source;
                    expected_fill_beat_q <= 1;
                    fill_payload_q <= '0;
                    fill_payload_q[0 +: BEAT_BITS] <= weight_fill_data;
                    if (!resident_block_valid_q
                            || weight_fill_block != resident_block_q) begin
                        resident_block_valid_q <= 1'b1;
                        resident_block_q <= weight_fill_block;
                        cache_valid_q <= '0;
                    end
                    cache_valid_q[weight_fill_source] <= 1'b0;
                end else if (weight_fill_beat == 1) begin
                    expected_fill_beat_q <= 2;
                    fill_payload_q[BEAT_BITS +: BEAT_BITS]
                        <= weight_fill_data;
                end else begin
                    fill_active_q <= 1'b0;
                    expected_fill_beat_q <= '0;
                    weight_cache_q[fill_source_q]
                        <= {weight_fill_data,
                            fill_payload_q[2 * BEAT_BITS-1:0]};
                    cache_valid_q[fill_source_q] <= 1'b1;
                end
            end

            if (!request_fault_q && !illegal_request && row_accept) begin
                row_block_q <= row_block;
                row_offset_q <= row_offset;
                row_mask_q <= row_source_mask;
                row_negate_q <= row_negate_mask;
                if (row_source_mask == 0) begin
                    row_active_q <= 1'b0;
                    row_done <= 1'b1;
                end else begin
                    row_active_q <= 1'b1;
                end
            end

            if (!request_fault_q && !illegal_request && update_accept) begin
                row_mask_q <= remainder_mask;
                if (remainder_mask == 0) begin
                    row_active_q <= 1'b0;
                    row_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
