`timescale 1ns/1ps
`default_nettype none

// M2067 is an additive source-only wrapper around the frozen physical-G48
// M2018/M803 engine.  It does not change the scheduler, cache, signed source,
// or Acc24 arithmetic inside that engine.  A logical G96/G192 FC2 tile is
// supplied as two/four contiguous G48 chunks.  Explicit header fields prevent
// local-group aliasing; global_group_base is added before weight SRAM address
// generation.  Intermediate per-chunk commits are accumulated into a retained
// 4-context x 6-slice x 16-lane Acc24 array and are never externally visible.
// Only the final chunk may commit or retire the logical output tile.
//
// Each chunk starts from a flushed M2018 cache.  The two wrapper reset states
// make that cost explicit and identical for ordinary and TSBG schedule modes;
// no cross-chunk cache reuse is credited.  This source contains no RTL-cycle,
// CPU-ratio, energy, full-network, or paper-admission claim.
module m2067_fc2_exact_continuation_wrapper #(
    parameter int SCHEDULE_MODE = 1,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int LANES = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         chunk_valid,
    output logic                         chunk_ready,
    input  logic [TAG_BITS-1:0]          logical_tag,
    input  logic [7:0]                   source_group_count,
    input  logic [2:0]                   output_tile_id,
    input  logic [1:0]                   chunk_index,
    input  logic [2:0]                   chunk_count,
    input  logic [7:0]                   global_group_base,
    input  logic                         chunk_first,
    input  logic                         chunk_intermediate,
    input  logic                         chunk_final,
    output logic                         chunk_accept,

    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [2:0]                   load_context,
    input  logic [5:0]                   load_group,
    input  logic [15:0]                  load_source_active,
    input  logic [15:0]                  load_source_sign,
    input  logic                         load_last,
    output logic                         load_accept,

    output logic [7:0]                   mem_req_valid,
    input  logic [7:0]                   mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch [0:7],
    output logic [2:0]                   mem_req_slot [0:7],
    output logic [GENERATION_BITS-1:0]   mem_req_generation [0:7],
    output logic [TAG_BITS-1:0]          mem_req_tag [0:7],
    output logic [2:0]                   mem_req_output_block [0:7],
    output logic [2:0]                   mem_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel [0:7],
    output logic [7:0]                   mem_req_global_group [0:7],
    output logic [11:0]                  mem_req_weight_row_index [0:7],
    output logic [7:0]                   mem_req_accept,

    input  logic [7:0]                   mem_rsp_valid,
    output logic [7:0]                   mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch [0:7],
    input  logic [2:0]                   mem_rsp_slot [0:7],
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation [0:7],
    input  logic [TAG_BITS-1:0]          mem_rsp_tag [0:7],
    input  logic signed [7:0]            mem_rsp_weight [0:7][0:LANES-1],
    output logic [7:0]                   mem_rsp_accept,

    output logic                         bridge_valid,
    input  logic                         bridge_ready,
    output logic [2:0]                   bridge_context,
    output logic [7:0]                   bridge_global_group,
    output logic                         bridge_half,
    output logic [2:0]                   bridge_slice,
    output logic [7:0]                   bridge_bank_valid,
    output logic [CHANNEL_BITS-1:0]      bridge_source_channel [0:7],
    output logic signed [1:0]            bridge_source_value [0:7],
    output logic signed [8:0]            bridge_effective_weight
                                                   [0:7][0:LANES-1],
    output logic                         bridge_accept,

    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_context,
    output logic [TAG_BITS-1:0]          commit_tag,
    output logic [2:0]                   commit_slice,
    output logic signed [23:0]           commit_accumulator [0:LANES-1],
    output logic                         commit_terminal,
    output logic                         commit_accept,
    output logic                         bundle_done_valid,
    input  logic                         bundle_done_ready,

    output logic                         protocol_error,
    output logic                         stale_response_seen,
    output logic                         numeric_overflow,
    output logic                         busy,
    output logic [31:0]                  debug_logical_cycle_count,
    output logic [31:0]                  debug_descriptor_preload_cycles,
    output logic [31:0]                  debug_continuation_cycles,
    output logic [31:0]                  debug_chunk_count,
    output logic [31:0]                  debug_intermediate_chunk_count,
    output logic [31:0]                  debug_final_chunk_count,
    output logic [31:0]                  debug_final_commit_count,
    output logic [31:0]                  debug_row_access_count,
    output logic [31:0]                  debug_cache_hit_count,
    output logic [31:0]                  debug_cache_miss_count,
    output logic [31:0]                  debug_cache_eviction_count,
    output logic [31:0]                  debug_weight_bundle_beat_count,
    output logic [31:0]                  debug_scalar_bank_request_count,
    output logic [31:0]                  debug_scalar_bank_response_count,
    output logic [31:0]                  debug_issue_count,
    output logic [31:0]                  debug_signed_product_count,
    output logic [31:0]                  debug_alias_reject_count,
    output logic [7:0]                   debug_global_group_base,
    output logic [1:0]                   debug_chunk_index,
    output logic                         debug_chunk_first,
    output logic                         debug_chunk_intermediate,
    output logic                         debug_chunk_final,
    output logic                         debug_retained_acc_valid
);
    localparam int BUNDLE = 4;
    localparam int SOURCE_GROUPS = 48;
    localparam int OUTPUT_SLICES = 6;
    localparam int SOURCES_PER_GROUP = 16;
    localparam int CACHE_ROWS = 4;
    localparam int ACC24_MAX = (1 << 23) - 1;
    localparam int ACC24_MIN = -(1 << 23);
    localparam bit PARAMETERS_LEGAL =
        (SCHEDULE_MODE == 0 || SCHEDULE_MODE == 1)
        && TAG_BITS == 24 && CHANNEL_BITS >= 12 && LANES == 16;

    typedef enum logic [2:0] {
        W_HEADER, W_RESET_ASSERT, W_RESET_RELEASE, W_LOAD, W_RUN, W_FAULT
    } wrapper_state_t;
    wrapper_state_t wrapper_state_q;

    logic fault_q, continuation_active_q, wrapper_overflow_q;
    logic [TAG_BITS-1:0] logical_tag_q;
    logic [7:0] source_group_count_q, global_group_base_q;
    logic [2:0] output_tile_id_q, chunk_count_q;
    logic [1:0] chunk_index_q, expected_chunk_index_q;
    logic chunk_first_q, chunk_intermediate_q, chunk_final_q;
    logic header_geometry_legal, header_sequence_legal, header_flags_legal;
    logic inner_rst_core;
    logic signed [23:0] retained_acc_q
        [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1];
    logic include_inner_counters_q;

    logic inner_load_valid, inner_load_ready, inner_load_accept;
    logic [7:0] inner_mem_req_valid, inner_mem_req_accept;
    logic [EPOCH_BITS-1:0] inner_mem_req_epoch [0:7];
    logic [2:0] inner_mem_req_slot [0:7];
    logic [GENERATION_BITS-1:0] inner_mem_req_generation [0:7];
    logic [TAG_BITS-1:0] inner_mem_req_tag [0:7];
    logic [2:0] inner_mem_req_output_block [0:7];
    logic [2:0] inner_mem_req_slice [0:7];
    logic [CHANNEL_BITS-1:0] inner_mem_req_source_channel [0:7];
    logic [7:0] inner_mem_rsp_ready, inner_mem_rsp_accept;
    logic inner_bridge_valid, inner_bridge_accept;
    logic [2:0] inner_bridge_context;
    logic [5:0] inner_bridge_group;
    logic inner_bridge_half;
    logic [2:0] inner_bridge_slice;
    logic [7:0] inner_bridge_bank_valid;
    logic [CHANNEL_BITS-1:0] inner_bridge_source_channel [0:7];
    logic signed [1:0] inner_bridge_source_value [0:7];
    logic signed [8:0] inner_bridge_effective_weight [0:7][0:LANES-1];
    logic inner_commit_valid, inner_commit_ready, inner_commit_accept;
    logic [2:0] inner_commit_context, inner_commit_slice;
    logic [TAG_BITS-1:0] inner_commit_tag;
    logic signed [23:0] inner_commit_accumulator [0:LANES-1];
    logic inner_commit_terminal, inner_bundle_done_valid;
    logic inner_bundle_done_ready;
    logic inner_protocol_error, inner_stale_response_seen;
    logic inner_numeric_overflow, inner_busy;
    logic [31:0] inner_cycle_count, inner_row_access_count;
    logic [31:0] inner_cache_hit_count, inner_cache_miss_count;
    logic [31:0] inner_cache_eviction_count;
    logic [31:0] inner_weight_bundle_beat_count;
    logic [31:0] inner_scalar_bank_request_count;
    logic [31:0] inner_scalar_bank_response_count;
    logic [31:0] inner_issue_count, inner_signed_product_count;
    logic [31:0] inner_commit_count;

    logic [31:0] logical_cycle_count_q, descriptor_preload_cycles_q;
    logic [31:0] continuation_cycles_q, chunk_counter_q;
    logic [31:0] intermediate_chunk_counter_q, final_chunk_counter_q;
    logic [31:0] final_commit_counter_q, alias_reject_counter_q;
    logic [31:0] prior_row_access_count_q, prior_cache_hit_count_q;
    logic [31:0] prior_cache_miss_count_q, prior_cache_eviction_count_q;
    logic [31:0] prior_weight_bundle_beat_count_q;
    logic [31:0] prior_scalar_bank_request_count_q;
    logic [31:0] prior_scalar_bank_response_count_q;
    logic [31:0] prior_issue_count_q, prior_signed_product_count_q;
    logic final_sum_overflow [0:LANES-1];
    logic final_sum_overflow_any;

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M2067 legal point is mode0/1, tag24, channel>=12, lane16");
        end
    endgenerate

    always_comb begin : header_checks
        logic expected_first, expected_intermediate, expected_final;
        logic [2:0] derived_count;
        derived_count = source_group_count == 8'd96 ? 3'd2 :
                        source_group_count == 8'd192 ? 3'd4 : 3'd0;
        expected_first = chunk_index == 0;
        expected_final = derived_count != 0 && chunk_index == derived_count - 1;
        expected_intermediate = derived_count != 0 && chunk_index != 0
            && chunk_index < derived_count - 1;
        header_geometry_legal = derived_count != 0
            && chunk_count == derived_count
            && chunk_index < derived_count
            && global_group_base == 8'(int'(chunk_index) * SOURCE_GROUPS);
        header_flags_legal = chunk_first == expected_first
            && chunk_intermediate == expected_intermediate
            && chunk_final == expected_final
            && ({chunk_first, chunk_intermediate, chunk_final} == 3'b100
                || {chunk_first, chunk_intermediate, chunk_final} == 3'b010
                || {chunk_first, chunk_intermediate, chunk_final} == 3'b001);
        if (!continuation_active_q) begin
            header_sequence_legal = expected_first && chunk_first;
        end else begin
            header_sequence_legal = !chunk_first
                && chunk_index == expected_chunk_index_q
                && logical_tag == logical_tag_q
                && source_group_count == source_group_count_q
                && output_tile_id == output_tile_id_q
                && chunk_count == chunk_count_q;
        end
    end

    assign chunk_ready = wrapper_state_q == W_HEADER && !fault_q;
    assign chunk_accept = chunk_valid && chunk_ready
        && header_geometry_legal && header_flags_legal && header_sequence_legal;
    assign inner_rst_core = rst_core || wrapper_state_q == W_RESET_ASSERT;
    assign inner_load_valid = load_valid && wrapper_state_q == W_LOAD && !fault_q;
    assign load_ready = inner_load_ready && wrapper_state_q == W_LOAD && !fault_q;
    assign load_accept = inner_load_accept && wrapper_state_q == W_LOAD;

    assign mem_req_valid = inner_mem_req_valid;
    assign mem_req_accept = inner_mem_req_accept;
    assign mem_rsp_ready = inner_mem_rsp_ready;
    assign mem_rsp_accept = inner_mem_rsp_accept;

    always_comb begin : global_weight_address_translation
        logic [7:0] translated_group;
        for (int bank = 0; bank < 8; bank++) begin
            mem_req_epoch[bank] = inner_mem_req_epoch[bank];
            mem_req_slot[bank] = inner_mem_req_slot[bank];
            mem_req_generation[bank] = inner_mem_req_generation[bank];
            mem_req_tag[bank] = inner_mem_req_tag[bank];
            mem_req_output_block[bank] = inner_mem_req_output_block[bank];
            mem_req_slice[bank] = inner_mem_req_slice[bank];
            translated_group = global_group_base_q
                + 8'(inner_mem_req_source_channel[bank] >> 4);
            mem_req_global_group[bank] = translated_group;
            mem_req_source_channel[bank] = CHANNEL_BITS'(
                int'(global_group_base_q) * SOURCES_PER_GROUP
                + int'(inner_mem_req_source_channel[bank]));
            mem_req_weight_row_index[bank] = 12'(
                int'(output_tile_id_q) * int'(source_group_count_q)
                + int'(translated_group));
        end
    end

    assign bridge_valid = inner_bridge_valid;
    assign bridge_context = inner_bridge_context;
    assign bridge_global_group = global_group_base_q + inner_bridge_group;
    assign bridge_half = inner_bridge_half;
    assign bridge_slice = inner_bridge_slice;
    assign bridge_bank_valid = inner_bridge_bank_valid;
    assign bridge_source_value = inner_bridge_source_value;
    assign bridge_effective_weight = inner_bridge_effective_weight;
    assign bridge_accept = inner_bridge_accept;
    always_comb begin : global_bridge_channel_translation
        for (int bank = 0; bank < 8; bank++)
            bridge_source_channel[bank] = CHANNEL_BITS'(
                int'(global_group_base_q) * SOURCES_PER_GROUP
                + int'(inner_bridge_source_channel[bank]));
    end

    assign inner_commit_ready = chunk_final_q ? commit_ready : 1'b1;
    assign commit_valid = wrapper_state_q == W_RUN && chunk_final_q
        && inner_commit_valid && !fault_q;
    assign commit_context = inner_commit_context;
    assign commit_tag = logical_tag_q;
    assign commit_slice = inner_commit_slice;
    assign commit_terminal = commit_valid && inner_commit_terminal;
    assign commit_accept = commit_valid && commit_ready;
    always_comb begin : final_accumulator_view
        logic signed [24:0] sum;
        final_sum_overflow_any = 1'b0;
        for (int lane = 0; lane < LANES; lane++) begin
            sum = {retained_acc_q[inner_commit_context]
                  [inner_commit_slice][lane][23],
                  retained_acc_q[inner_commit_context]
                  [inner_commit_slice][lane]}
                + {inner_commit_accumulator[lane][23],
                   inner_commit_accumulator[lane]};
            commit_accumulator[lane] = sum[23:0];
            final_sum_overflow[lane] = sum > ACC24_MAX || sum < ACC24_MIN;
            final_sum_overflow_any |= final_sum_overflow[lane];
        end
    end
    assign inner_bundle_done_ready = chunk_final_q ? bundle_done_ready : 1'b1;
    assign bundle_done_valid = wrapper_state_q == W_RUN && chunk_final_q
        && inner_bundle_done_valid && !fault_q;

    assign protocol_error = fault_q || inner_protocol_error;
    assign stale_response_seen = inner_stale_response_seen;
    assign numeric_overflow = wrapper_overflow_q || inner_numeric_overflow
        || (commit_valid && final_sum_overflow_any);
    assign busy = wrapper_state_q != W_HEADER || continuation_active_q
        || inner_busy;
    assign debug_logical_cycle_count = logical_cycle_count_q;
    assign debug_descriptor_preload_cycles = descriptor_preload_cycles_q;
    assign debug_continuation_cycles = continuation_cycles_q;
    assign debug_chunk_count = chunk_counter_q;
    assign debug_intermediate_chunk_count = intermediate_chunk_counter_q;
    assign debug_final_chunk_count = final_chunk_counter_q;
    assign debug_final_commit_count = final_commit_counter_q;
    assign debug_alias_reject_count = alias_reject_counter_q;
    assign debug_global_group_base = global_group_base_q;
    assign debug_chunk_index = chunk_index_q;
    assign debug_chunk_first = chunk_first_q;
    assign debug_chunk_intermediate = chunk_intermediate_q;
    assign debug_chunk_final = chunk_final_q;
    assign debug_retained_acc_valid = continuation_active_q;
    assign debug_row_access_count = prior_row_access_count_q
        + (include_inner_counters_q ? inner_row_access_count : 0);
    assign debug_cache_hit_count = prior_cache_hit_count_q
        + (include_inner_counters_q ? inner_cache_hit_count : 0);
    assign debug_cache_miss_count = prior_cache_miss_count_q
        + (include_inner_counters_q ? inner_cache_miss_count : 0);
    assign debug_cache_eviction_count = prior_cache_eviction_count_q
        + (include_inner_counters_q ? inner_cache_eviction_count : 0);
    assign debug_weight_bundle_beat_count = prior_weight_bundle_beat_count_q
        + (include_inner_counters_q ? inner_weight_bundle_beat_count : 0);
    assign debug_scalar_bank_request_count = prior_scalar_bank_request_count_q
        + (include_inner_counters_q ? inner_scalar_bank_request_count : 0);
    assign debug_scalar_bank_response_count = prior_scalar_bank_response_count_q
        + (include_inner_counters_q ? inner_scalar_bank_response_count : 0);
    assign debug_issue_count = prior_issue_count_q
        + (include_inner_counters_q ? inner_issue_count : 0);
    assign debug_signed_product_count = prior_signed_product_count_q
        + (include_inner_counters_q ? inner_signed_product_count : 0);

    m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend #(
        .SCHEDULE_MODE(SCHEDULE_MODE), .BUNDLE(BUNDLE),
        .SOURCE_GROUPS(SOURCE_GROUPS),
        .SOURCES_PER_GROUP(SOURCES_PER_GROUP),
        .OUTPUT_SLICES(OUTPUT_SLICES), .CACHE_ROWS(CACHE_ROWS),
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .LANES(LANES)
    ) engine (
        .clk_core(clk_core), .rst_core(inner_rst_core),
        .load_valid(inner_load_valid), .load_ready(inner_load_ready),
        .load_context(load_context), .load_tag(logical_tag_q),
        .load_group(load_group), .load_source_active(load_source_active),
        .load_source_sign(load_source_sign), .load_last(load_last),
        .load_accept(inner_load_accept),
        .mem_req_valid(inner_mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_epoch(inner_mem_req_epoch), .mem_req_slot(inner_mem_req_slot),
        .mem_req_generation(inner_mem_req_generation),
        .mem_req_tag(inner_mem_req_tag),
        .mem_req_output_block(inner_mem_req_output_block),
        .mem_req_slice(inner_mem_req_slice),
        .mem_req_source_channel(inner_mem_req_source_channel),
        .mem_req_accept(inner_mem_req_accept),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(inner_mem_rsp_ready),
        .mem_rsp_epoch(mem_rsp_epoch), .mem_rsp_slot(mem_rsp_slot),
        .mem_rsp_generation(mem_rsp_generation), .mem_rsp_tag(mem_rsp_tag),
        .mem_rsp_weight(mem_rsp_weight), .mem_rsp_accept(inner_mem_rsp_accept),
        .bridge_valid(inner_bridge_valid), .bridge_ready(bridge_ready),
        .bridge_context(inner_bridge_context),
        .bridge_group(inner_bridge_group), .bridge_half(inner_bridge_half),
        .bridge_slice(inner_bridge_slice),
        .bridge_bank_valid(inner_bridge_bank_valid),
        .bridge_source_channel(inner_bridge_source_channel),
        .bridge_source_value(inner_bridge_source_value),
        .bridge_effective_weight(inner_bridge_effective_weight),
        .bridge_accept(inner_bridge_accept),
        .commit_valid(inner_commit_valid), .commit_ready(inner_commit_ready),
        .commit_context(inner_commit_context), .commit_tag(inner_commit_tag),
        .commit_slice(inner_commit_slice),
        .commit_accumulator(inner_commit_accumulator),
        .commit_terminal(inner_commit_terminal),
        .commit_accept(inner_commit_accept),
        .bundle_done_valid(inner_bundle_done_valid),
        .bundle_done_ready(inner_bundle_done_ready),
        .protocol_error(inner_protocol_error),
        .stale_response_seen(inner_stale_response_seen),
        .numeric_overflow(inner_numeric_overflow), .busy(inner_busy),
        .debug_cycle_count(inner_cycle_count),
        .debug_row_access_count(inner_row_access_count),
        .debug_cache_hit_count(inner_cache_hit_count),
        .debug_cache_miss_count(inner_cache_miss_count),
        .debug_cache_eviction_count(inner_cache_eviction_count),
        .debug_weight_bundle_beat_count(inner_weight_bundle_beat_count),
        .debug_scalar_bank_request_count(inner_scalar_bank_request_count),
        .debug_scalar_bank_response_count(inner_scalar_bank_response_count),
        .debug_issue_count(inner_issue_count),
        .debug_signed_product_count(inner_signed_product_count),
        .debug_commit_count(inner_commit_count)
    );

    always_ff @(posedge clk_core) begin : wrapper_state_and_retention
        if (rst_core) begin
            wrapper_state_q <= W_HEADER;
            fault_q <= 0;
            continuation_active_q <= 0;
            wrapper_overflow_q <= 0;
            logical_tag_q <= 0;
            source_group_count_q <= 0;
            output_tile_id_q <= 0;
            chunk_count_q <= 0;
            chunk_index_q <= 0;
            expected_chunk_index_q <= 0;
            global_group_base_q <= 0;
            chunk_first_q <= 0;
            chunk_intermediate_q <= 0;
            chunk_final_q <= 0;
            include_inner_counters_q <= 0;
            logical_cycle_count_q <= 0;
            descriptor_preload_cycles_q <= 0;
            continuation_cycles_q <= 0;
            chunk_counter_q <= 0;
            intermediate_chunk_counter_q <= 0;
            final_chunk_counter_q <= 0;
            final_commit_counter_q <= 0;
            alias_reject_counter_q <= 0;
            prior_row_access_count_q <= 0;
            prior_cache_hit_count_q <= 0;
            prior_cache_miss_count_q <= 0;
            prior_cache_eviction_count_q <= 0;
            prior_weight_bundle_beat_count_q <= 0;
            prior_scalar_bank_request_count_q <= 0;
            prior_scalar_bank_response_count_q <= 0;
            prior_issue_count_q <= 0;
            prior_signed_product_count_q <= 0;
            for (int context = 0; context < BUNDLE; context++)
                for (int slice = 0; slice < OUTPUT_SLICES; slice++)
                    for (int lane = 0; lane < LANES; lane++)
                        retained_acc_q[context][slice][lane] <= 0;
        end else begin
            // Count only this axis's intrinsic service.  W_HEADER may wait for
            // a testbench peer or an upstream producer after this axis has
            // completed an intermediate chunk; that external synchronization
            // wait must not flatten ordinary and TSBG cycle observations.
            if (chunk_accept || wrapper_state_q == W_RESET_ASSERT
                    || wrapper_state_q == W_RESET_RELEASE
                    || wrapper_state_q == W_LOAD
                    || wrapper_state_q == W_RUN)
                logical_cycle_count_q <= logical_cycle_count_q + 1'b1;
            if (wrapper_state_q == W_LOAD)
                descriptor_preload_cycles_q <=
                    descriptor_preload_cycles_q + 1'b1;
            if ((wrapper_state_q == W_RESET_ASSERT
                    || wrapper_state_q == W_RESET_RELEASE)
                    && !chunk_first_q)
                continuation_cycles_q <= continuation_cycles_q + 1'b1;

            if (chunk_valid && chunk_ready && !chunk_accept) begin
                fault_q <= 1;
                wrapper_state_q <= W_FAULT;
                alias_reject_counter_q <= alias_reject_counter_q + 1'b1;
            end else if (chunk_accept) begin
                logical_tag_q <= logical_tag;
                source_group_count_q <= source_group_count;
                output_tile_id_q <= output_tile_id;
                chunk_count_q <= chunk_count;
                chunk_index_q <= chunk_index;
                expected_chunk_index_q <= chunk_index + 1'b1;
                global_group_base_q <= global_group_base;
                chunk_first_q <= chunk_first;
                chunk_intermediate_q <= chunk_intermediate;
                chunk_final_q <= chunk_final;
                chunk_counter_q <= chunk_first ? 1 : chunk_counter_q + 1'b1;
                if (chunk_first) begin
                    continuation_active_q <= 1;
                    wrapper_overflow_q <= 0;
                    logical_cycle_count_q <= 1;
                    descriptor_preload_cycles_q <= 0;
                    continuation_cycles_q <= 0;
                    intermediate_chunk_counter_q <= 0;
                    final_chunk_counter_q <= 0;
                    final_commit_counter_q <= 0;
                    prior_row_access_count_q <= 0;
                    prior_cache_hit_count_q <= 0;
                    prior_cache_miss_count_q <= 0;
                    prior_cache_eviction_count_q <= 0;
                    prior_weight_bundle_beat_count_q <= 0;
                    prior_scalar_bank_request_count_q <= 0;
                    prior_scalar_bank_response_count_q <= 0;
                    prior_issue_count_q <= 0;
                    prior_signed_product_count_q <= 0;
                    for (int context = 0; context < BUNDLE; context++)
                        for (int slice = 0; slice < OUTPUT_SLICES; slice++)
                            for (int lane = 0; lane < LANES; lane++)
                                retained_acc_q[context][slice][lane] <= 0;
                end
                if (chunk_intermediate)
                    intermediate_chunk_counter_q <=
                        intermediate_chunk_counter_q + 1'b1;
                if (chunk_final)
                    final_chunk_counter_q <= final_chunk_counter_q + 1'b1;
                include_inner_counters_q <= 0;
                wrapper_state_q <= W_RESET_ASSERT;
            end

            if (wrapper_state_q == W_RESET_ASSERT)
                wrapper_state_q <= W_RESET_RELEASE;
            if (wrapper_state_q == W_RESET_RELEASE) begin
                include_inner_counters_q <= 1;
                wrapper_state_q <= W_LOAD;
            end
            if (wrapper_state_q == W_LOAD && inner_load_accept
                    && load_context == BUNDLE - 1 && load_last)
                wrapper_state_q <= W_RUN;

            if (wrapper_state_q == W_RUN && inner_commit_accept
                    && !chunk_final_q) begin
                logic signed [24:0] retained_sum;
                for (int lane = 0; lane < LANES; lane++) begin
                    retained_sum = {retained_acc_q[inner_commit_context]
                                    [inner_commit_slice][lane][23],
                                    retained_acc_q[inner_commit_context]
                                    [inner_commit_slice][lane]}
                        + {inner_commit_accumulator[lane][23],
                           inner_commit_accumulator[lane]};
                    retained_acc_q[inner_commit_context]
                                  [inner_commit_slice][lane]
                        <= retained_sum[23:0];
                    if (retained_sum > ACC24_MAX || retained_sum < ACC24_MIN)
                        wrapper_overflow_q <= 1;
                end
            end
            if (commit_accept) begin
                final_commit_counter_q <= final_commit_counter_q + 1'b1;
                if (final_sum_overflow_any) wrapper_overflow_q <= 1;
            end

            if (wrapper_state_q == W_RUN && inner_bundle_done_valid
                    && inner_bundle_done_ready) begin
                if (!chunk_final_q) begin
                    prior_row_access_count_q <= prior_row_access_count_q
                        + inner_row_access_count;
                    prior_cache_hit_count_q <= prior_cache_hit_count_q
                        + inner_cache_hit_count;
                    prior_cache_miss_count_q <= prior_cache_miss_count_q
                        + inner_cache_miss_count;
                    prior_cache_eviction_count_q <= prior_cache_eviction_count_q
                        + inner_cache_eviction_count;
                    prior_weight_bundle_beat_count_q <=
                        prior_weight_bundle_beat_count_q
                        + inner_weight_bundle_beat_count;
                    prior_scalar_bank_request_count_q <=
                        prior_scalar_bank_request_count_q
                        + inner_scalar_bank_request_count;
                    prior_scalar_bank_response_count_q <=
                        prior_scalar_bank_response_count_q
                        + inner_scalar_bank_response_count;
                    prior_issue_count_q <= prior_issue_count_q
                        + inner_issue_count;
                    prior_signed_product_count_q <= prior_signed_product_count_q
                        + inner_signed_product_count;
                    include_inner_counters_q <= 0;
                    wrapper_state_q <= W_HEADER;
                end else begin
                    continuation_active_q <= 0;
                    wrapper_state_q <= W_HEADER;
                end
            end

            if (inner_protocol_error) begin
                fault_q <= 1;
                wrapper_state_q <= W_FAULT;
            end
        end
    end
endmodule

`default_nettype wire
