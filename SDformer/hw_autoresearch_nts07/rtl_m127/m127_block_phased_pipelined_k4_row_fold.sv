`timescale 1ns/1ps
`default_nettype none

// Timing-oriented K4 row fold.
//
// The row mask is canonically predecoded once into as many as four groups.
// Four selected signed-INT8 vectors are reduced into two registered signed10
// pair sums; the output stage performs only one signed10+signed10 add.  The
// one-entry elastic pair-sum register retains one accepted K4 update per cycle
// under no backpressure, with no extra cycle versus M125 for the first group.
module m127_block_phased_pipelined_k4_row_fold #(
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
    output logic                         observed_pair_pipeline_valid,
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
    logic [SOURCES-1:0] row_negate_q;
    logic [SOURCES-1:0] remaining_mask_q;
    logic [15:0] group_mask_q [0:3];
    logic [3:0] group_source_q [0:3][0:3];
    logic [3:0] group_source_valid_q [0:3];
    logic [2:0] group_count_q;
    logic [2:0] issue_group_q;

    logic pipe_valid_q;
    logic [2:0] pipe_block_q;
    logic [8:0] pipe_row_q;
    logic [15:0] pipe_selected_mask_q;
    logic pipe_last_q;
    logic signed [9:0] pipe_pair_sum_q [0:1][0:LANES-1];

    logic request_collision;
    logic fill_semantically_valid;
    logic row_semantically_valid;
    logic illegal_request;
    logic pipe_can_advance;
    logic launch_valid;
    logic launch_from_new_row;
    logic [2:0] launch_block;
    logic [8:0] launch_row;
    logic [15:0] launch_selected_mask;
    logic [15:0] launch_negate_mask;
    logic [3:0] launch_source [0:3];
    logic [3:0] launch_source_valid;
    logic launch_last;
    logic signed [8:0] launch_contribution [0:3][0:LANES-1];
    logic signed [9:0] launch_pair_sum [0:1][0:LANES-1];
    logic signed [10:0] output_fold_sum [0:LANES-1];

    logic [15:0] pre_group_mask [0:3];
    logic [3:0] pre_group_source [0:3][0:3];
    logic [3:0] pre_group_source_valid [0:3];
    logic [2:0] pre_group_count;

`ifndef SYNTHESIS
    initial begin
        if (SOURCES != 16 || LANES != 96 || WEIGHT_BITS != 8
                || WEIGHT_VECTOR_BITS != 768 || BEAT_BITS != 256
                || ACC_BITS != 19 || UPDATE_BITS != 1824)
            $fatal(1, "M127 production geometry drift");
    end
`endif

    always_comb begin : request_audit
        request_collision = weight_fill_valid && row_valid;
        if (weight_fill_beat == 0) begin
            fill_semantically_valid = !fill_active_q && !row_active_q
                                    && !pipe_valid_q;
        end else begin
            fill_semantically_valid = fill_active_q && !row_active_q
                && !pipe_valid_q
                && weight_fill_block == fill_block_q
                && weight_fill_source == fill_source_q
                && weight_fill_beat == expected_fill_beat_q;
        end
        row_semantically_valid = !fill_active_q && !row_active_q
            && !pipe_valid_q && resident_block_valid_q
            && row_block == resident_block_q
            && !(|(row_source_mask & ~cache_valid_q))
            && !(|(row_negate_mask & ~row_source_mask));
        illegal_request = request_collision
                        || (weight_fill_valid && !fill_semantically_valid)
                        || (row_valid && !row_semantically_valid);
    end

    assign protocol_error = !rst_core
                          && (request_fault_q || illegal_request);
    assign weight_fill_ready = !rst_core && !protocol_error
                             && fill_semantically_valid && !row_valid;
    assign row_ready = !rst_core && !protocol_error
                     && row_semantically_valid && !weight_fill_valid;
    assign weight_fill_accept = weight_fill_valid && weight_fill_ready;
    assign row_accept = row_valid && row_ready;
    assign update_valid = !rst_core && pipe_valid_q && !protocol_error;
    assign update_accept = update_valid && update_ready;
    assign update_block = pipe_block_q;
    assign update_row = pipe_row_q;
    assign update_selected_mask = pipe_selected_mask_q;
    assign observed_remaining_mask = remaining_mask_q;
    assign observed_cache_valid = cache_valid_q;
    assign observed_resident_block_valid = resident_block_valid_q;
    assign observed_resident_block = resident_block_q;
    assign observed_pair_pipeline_valid = pipe_valid_q;
    assign busy = fill_active_q || row_active_q || pipe_valid_q;
    assign pipe_can_advance = !pipe_valid_q
                            || (update_ready && !protocol_error);

    always_comb begin : canonical_group_predecode
        logic [15:0] remaining;
        logic found;
        remaining = row_source_mask;
        pre_group_count = 0;
        for (int group = 0; group < 4; group++) begin
            pre_group_mask[group] = '0;
            pre_group_source_valid[group] = '0;
            for (int pick = 0; pick < 4; pick++) begin
                pre_group_source[group][pick] = '0;
                found = 1'b0;
                for (int source = 0; source < SOURCES; source++) begin
                    if (!found && remaining[source]) begin
                        pre_group_mask[group][source] = 1'b1;
                        pre_group_source[group][pick] = source[3:0];
                        pre_group_source_valid[group][pick] = 1'b1;
                        remaining[source] = 1'b0;
                        found = 1'b1;
                    end
                end
            end
            if (pre_group_mask[group] != 0)
                pre_group_count = group + 1;
        end
    end

    always_comb begin : launch_select_and_pair_reduce
        logic signed [7:0] selected_weight;
        logic signed [8:0] weight_ext;
        launch_valid = 1'b0;
        launch_from_new_row = 1'b0;
        launch_block = '0;
        launch_row = '0;
        launch_selected_mask = '0;
        launch_negate_mask = '0;
        launch_source_valid = '0;
        launch_last = 1'b0;
        for (int pick = 0; pick < 4; pick++)
            launch_source[pick] = '0;

        if (pipe_can_advance && row_accept && pre_group_count != 0) begin
            launch_valid = 1'b1;
            launch_from_new_row = 1'b1;
            launch_block = row_block;
            launch_row = row_offset;
            launch_selected_mask = pre_group_mask[0];
            launch_negate_mask = row_negate_mask;
            launch_source_valid = pre_group_source_valid[0];
            launch_last = pre_group_count == 1;
            for (int pick = 0; pick < 4; pick++)
                launch_source[pick] = pre_group_source[0][pick];
        end else if (pipe_can_advance && row_active_q
                     && issue_group_q < group_count_q) begin
            launch_valid = 1'b1;
            launch_block = pipe_block_q;
            launch_row = pipe_row_q;
            launch_selected_mask = group_mask_q[issue_group_q];
            launch_negate_mask = row_negate_q;
            launch_source_valid = group_source_valid_q[issue_group_q];
            launch_last = issue_group_q == group_count_q - 1'b1;
            for (int pick = 0; pick < 4; pick++)
                launch_source[pick]
                    = group_source_q[issue_group_q][pick];
        end

        for (int lane = 0; lane < LANES; lane++) begin
            for (int pick = 0; pick < 4; pick++) begin
                launch_contribution[pick][lane] = '0;
                if (launch_valid && launch_source_valid[pick]) begin
                    selected_weight
                        = weight_cache_q[launch_source[pick]]
                          [lane * WEIGHT_BITS +: WEIGHT_BITS];
                    weight_ext = {selected_weight[WEIGHT_BITS-1],
                                  selected_weight};
                    launch_contribution[pick][lane]
                        = launch_negate_mask[launch_source[pick]]
                        ? -weight_ext : weight_ext;
                end
            end
            launch_pair_sum[0][lane]
                = {{1{launch_contribution[0][lane][8]}},
                   launch_contribution[0][lane]}
                + {{1{launch_contribution[1][lane][8]}},
                   launch_contribution[1][lane]};
            launch_pair_sum[1][lane]
                = {{1{launch_contribution[2][lane][8]}},
                   launch_contribution[2][lane]}
                + {{1{launch_contribution[3][lane][8]}},
                   launch_contribution[3][lane]};
        end
    end

    always_comb begin : final_pair_add
        update_delta = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            output_fold_sum[lane]
                = {pipe_pair_sum_q[0][lane][9],
                   pipe_pair_sum_q[0][lane]}
                + {pipe_pair_sum_q[1][lane][9],
                   pipe_pair_sum_q[1][lane]};
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
            row_active_q <= 1'b0;
            row_negate_q <= '0;
            remaining_mask_q <= '0;
            group_count_q <= '0;
            issue_group_q <= '0;
            pipe_valid_q <= 1'b0;
            pipe_block_q <= '0;
            pipe_row_q <= '0;
            pipe_selected_mask_q <= '0;
            pipe_last_q <= 1'b0;
            row_done <= 1'b0;
            for (int group = 0; group < 4; group++) begin
                group_mask_q[group] <= '0;
                group_source_valid_q[group] <= '0;
                for (int pick = 0; pick < 4; pick++)
                    group_source_q[group][pick] <= '0;
            end
            for (int source = 0; source < SOURCES; source++)
                weight_cache_q[source] <= '0;
            for (int pair = 0; pair < 2; pair++)
                for (int lane = 0; lane < LANES; lane++)
                    pipe_pair_sum_q[pair][lane] <= '0;
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
                row_negate_q <= row_negate_mask;
                remaining_mask_q <= row_source_mask;
                group_count_q <= pre_group_count;
                issue_group_q <= pre_group_count == 0 ? 0 : 1;
                for (int group = 0; group < 4; group++) begin
                    group_mask_q[group] <= pre_group_mask[group];
                    group_source_valid_q[group]
                        <= pre_group_source_valid[group];
                    for (int pick = 0; pick < 4; pick++)
                        group_source_q[group][pick]
                            <= pre_group_source[group][pick];
                end
                if (row_source_mask == 0) begin
                    row_active_q <= 1'b0;
                    row_done <= 1'b1;
                end else begin
                    row_active_q <= 1'b1;
                end
            end

            if (!request_fault_q && !illegal_request && update_accept) begin
                remaining_mask_q
                    <= remaining_mask_q & ~pipe_selected_mask_q;
                if (pipe_last_q) begin
                    row_active_q <= 1'b0;
                    row_done <= 1'b1;
                end
            end

            if (!request_fault_q && !illegal_request && pipe_can_advance) begin
                pipe_valid_q <= launch_valid;
                if (launch_valid) begin
                    pipe_block_q <= launch_block;
                    pipe_row_q <= launch_row;
                    pipe_selected_mask_q <= launch_selected_mask;
                    pipe_last_q <= launch_last;
                    for (int pair = 0; pair < 2; pair++)
                        for (int lane = 0; lane < LANES; lane++)
                            pipe_pair_sum_q[pair][lane]
                                <= launch_pair_sum[pair][lane];
                    if (!launch_from_new_row)
                        issue_group_q <= issue_group_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
