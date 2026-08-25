`timescale 1ns/1ps
`default_nettype none

// Compact canonical signed K4 row fold.
//
// The descriptor payload is 35 bits:
//   block3,row9,count_minus_one2,source_ids16,negate4,last1.
// Validity and the selected-source mask are derived internally.  Source IDs
// must be strictly increasing inside a descriptor and across descriptors of
// one open row.  The output carries last and emits a same-cycle tagged done
// token only when the last update is accepted.
module m130_compact_canonical_k4_row_fold #(
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

    input  logic                         group_valid,
    output logic                         group_ready,
    input  logic [2:0]                   group_block,
    input  logic [8:0]                   group_row,
    input  logic [1:0]                   group_source_count_m1,
    input  logic [3:0]                   group_source [0:3],
    input  logic [3:0]                   group_negate,
    input  logic                         group_last,
    output logic                         group_accept,

    output logic                         update_valid,
    input  logic                         update_ready,
    output logic [2:0]                   update_block,
    output logic [8:0]                   update_row,
    output logic [UPDATE_BITS-1:0]       update_delta,
    output logic [SOURCES-1:0]           update_selected_mask,
    output logic                         update_last,
    output logic                         update_accept,

    output logic                         done_valid,
    output logic [2:0]                   done_block,
    output logic [8:0]                   done_row,

    output logic [SOURCES-1:0]           observed_cache_valid,
    output logic                         observed_resident_block_valid,
    output logic [2:0]                   observed_resident_block,
    output logic                         observed_pair_pipeline_valid,
    output logic                         observed_row_stream_open,
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

    logic row_stream_open_q;
    logic [2:0] row_stream_block_q;
    logic [8:0] row_stream_row_q;
    logic [3:0] row_stream_last_source_q;

    logic pipe_valid_q;
    logic [2:0] pipe_block_q;
    logic [8:0] pipe_row_q;
    logic [SOURCES-1:0] pipe_selected_mask_q;
    logic pipe_last_q;
    logic signed [9:0] pipe_pair_sum_q [0:1][0:LANES-1];

    logic request_collision;
    logic raw_pipe_can_advance;
    logic fill_semantically_valid;
    logic group_semantically_valid;
    logic descriptor_order_error;
    logic descriptor_cache_miss;
    logic descriptor_padding_dirty;
    logic descriptor_row_error;
    logic descriptor_nonlast_exhausted;
    logic [2:0] descriptor_source_count;
    logic [3:0] descriptor_last_source;
    logic [SOURCES-1:0] descriptor_derived_mask;
    logic illegal_request;
    logic quarantine;
    logic group_capacity;
    logic fill_capacity;
    logic signed [8:0] group_contribution [0:3][0:LANES-1];
    logic signed [9:0] group_pair_sum [0:1][0:LANES-1];
    logic signed [10:0] output_fold_sum [0:LANES-1];

`ifndef SYNTHESIS
    initial begin
        if (SOURCES != 16 || LANES != 96 || WEIGHT_BITS != 8
                || WEIGHT_VECTOR_BITS != 768 || BEAT_BITS != 256
                || ACC_BITS != 19 || UPDATE_BITS != 1824)
            $fatal(1, "M130 production geometry drift");
    end
`endif

    assign raw_pipe_can_advance = !pipe_valid_q || update_ready;
    assign descriptor_source_count = {1'b0, group_source_count_m1} + 3'd1;

    always_comb begin : descriptor_audit
        descriptor_derived_mask = '0;
        descriptor_order_error = 1'b0;
        descriptor_cache_miss = 1'b0;
        descriptor_padding_dirty = 1'b0;
        descriptor_last_source = group_source[0];
        for (int pick = 0; pick < 4; pick++) begin
            if (pick < descriptor_source_count) begin
                if (pick != 0
                        && group_source[pick] <= group_source[pick-1])
                    descriptor_order_error = 1'b1;
                descriptor_derived_mask[group_source[pick]] = 1'b1;
                if (!cache_valid_q[group_source[pick]])
                    descriptor_cache_miss = 1'b1;
                descriptor_last_source = group_source[pick];
            end else if (group_source[pick] != 0 || group_negate[pick]) begin
                descriptor_padding_dirty = 1'b1;
            end
        end
        descriptor_row_error = row_stream_open_q
            && (group_block != row_stream_block_q
                || group_row != row_stream_row_q
                || group_source[0] <= row_stream_last_source_q);
        descriptor_nonlast_exhausted = !group_last
            && descriptor_last_source == 4'd15;
    end

    always_comb begin : request_audit
        request_collision = weight_fill_valid && group_valid;
        if (weight_fill_beat == 0) begin
            fill_semantically_valid = !fill_active_q && !row_stream_open_q;
        end else begin
            fill_semantically_valid = fill_active_q
                && weight_fill_block == fill_block_q
                && weight_fill_source == fill_source_q
                && weight_fill_beat == expected_fill_beat_q;
        end
        group_semantically_valid = resident_block_valid_q
            && group_block == resident_block_q
            && !descriptor_order_error
            && !descriptor_cache_miss
            && !descriptor_padding_dirty
            && !descriptor_row_error
            && !descriptor_nonlast_exhausted;
        illegal_request = request_collision
                        || (weight_fill_valid && !fill_semantically_valid)
                        || (group_valid && !group_semantically_valid);
        quarantine = request_fault_q || illegal_request;
    end

    assign group_capacity = !rst_core && !request_fault_q
                          && !weight_fill_valid && !fill_active_q
                          && raw_pipe_can_advance;
    assign fill_capacity = !rst_core && !request_fault_q
                         && !group_valid && !pipe_valid_q;
    // With valid low, ready is capacity-only and independent of payload.
    assign group_ready = group_capacity
                       && (!group_valid || group_semantically_valid);
    assign weight_fill_ready = fill_capacity
                             && (!weight_fill_valid
                                 || fill_semantically_valid);
    assign weight_fill_accept = weight_fill_valid && weight_fill_ready;
    assign group_accept = group_valid && group_ready;

    assign protocol_error = !rst_core && quarantine;
    assign update_valid = !rst_core && pipe_valid_q && !quarantine;
    assign update_accept = update_valid && update_ready;
    assign update_block = pipe_block_q;
    assign update_row = pipe_row_q;
    assign update_selected_mask = pipe_selected_mask_q;
    assign update_last = pipe_last_q;
    assign done_valid = update_accept && update_last;
    assign done_block = update_block;
    assign done_row = update_row;
    assign observed_cache_valid = cache_valid_q;
    assign observed_resident_block_valid = resident_block_valid_q;
    assign observed_resident_block = resident_block_q;
    assign observed_pair_pipeline_valid = pipe_valid_q;
    assign observed_row_stream_open = row_stream_open_q;
    assign busy = fill_active_q || pipe_valid_q || row_stream_open_q;

    always_comb begin : descriptor_pair_reduce
        logic signed [7:0] selected_weight;
        logic signed [8:0] weight_ext;
        for (int lane = 0; lane < LANES; lane++) begin
            for (int pick = 0; pick < 4; pick++) begin
                group_contribution[pick][lane] = '0;
                if (group_accept && pick < descriptor_source_count) begin
                    selected_weight = weight_cache_q[group_source[pick]]
                                      [lane * WEIGHT_BITS +: WEIGHT_BITS];
                    weight_ext = {selected_weight[WEIGHT_BITS-1],
                                  selected_weight};
                    group_contribution[pick][lane]
                        = group_negate[pick] ? -weight_ext : weight_ext;
                end
            end
            group_pair_sum[0][lane]
                = {{1{group_contribution[0][lane][8]}},
                   group_contribution[0][lane]}
                + {{1{group_contribution[1][lane][8]}},
                   group_contribution[1][lane]};
            group_pair_sum[1][lane]
                = {{1{group_contribution[2][lane][8]}},
                   group_contribution[2][lane]}
                + {{1{group_contribution[3][lane][8]}},
                   group_contribution[3][lane]};
        end
    end

    always_comb begin : final_pair_add
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            output_fold_sum[lane]
                = {pipe_pair_sum_q[0][lane][9], pipe_pair_sum_q[0][lane]}
                + {pipe_pair_sum_q[1][lane][9], pipe_pair_sum_q[1][lane]};
            update_delta[lane * ACC_BITS +: ACC_BITS]
                = {{(ACC_BITS-11){output_fold_sum[lane][10]}},
                   output_fold_sum[lane]};
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
            row_stream_open_q <= 1'b0;
            row_stream_block_q <= '0;
            row_stream_row_q <= '0;
            row_stream_last_source_q <= '0;
            pipe_valid_q <= 1'b0;
            pipe_block_q <= '0;
            pipe_row_q <= '0;
            pipe_selected_mask_q <= '0;
            pipe_last_q <= 1'b0;
            for (int source = 0; source < SOURCES; source++)
                weight_cache_q[source] <= '0;
            for (int pair = 0; pair < 2; pair++)
                for (int lane = 0; lane < LANES; lane++)
                    pipe_pair_sum_q[pair][lane] <= '0;
        end else begin
            if (illegal_request)
                request_fault_q <= 1'b1;

            if (!quarantine && weight_fill_accept) begin
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

            if (!quarantine && group_accept) begin
                if (group_last) begin
                    row_stream_open_q <= 1'b0;
                end else begin
                    row_stream_open_q <= 1'b1;
                    row_stream_block_q <= group_block;
                    row_stream_row_q <= group_row;
                    row_stream_last_source_q <= descriptor_last_source;
                end
            end

            if (!quarantine && raw_pipe_can_advance) begin
                pipe_valid_q <= group_accept;
                if (group_accept) begin
                    pipe_block_q <= group_block;
                    pipe_row_q <= group_row;
                    pipe_selected_mask_q <= descriptor_derived_mask;
                    pipe_last_q <= group_last;
                    for (int pair = 0; pair < 2; pair++)
                        for (int lane = 0; lane < LANES; lane++)
                            pipe_pair_sum_q[pair][lane]
                                <= group_pair_sum[pair][lane];
                end
            end
        end
    end
endmodule

`default_nettype wire
