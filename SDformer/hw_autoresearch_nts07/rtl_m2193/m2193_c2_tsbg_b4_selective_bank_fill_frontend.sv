`timescale 1ns/1ps
`default_nettype none

// M2193 is an additive source-only clone of M2018; M2018/M803 are immutable.
// It adds exact B4-union selective bank fill with per-bank cache validity while
// retaining the real M803 protocol, typed bridge, private Acc24 state, and the
// already-reviewed ordinary/TSBG scheduler representations.
//
// Both modes expose one common 192-bit row-live bitmap to the same 12-by-16
// hierarchical priority encoder.  Generate-time wiring orders that bitmap as
// token-major in MODE=0 and source-group-major in MODE=1.  Empty descriptors
// never set row_live_q, so ST_FIND skips them combinationally without a bubble.
// A selected 16-bit active/sign row is latched before the existing fetch,
// bridge, commit, and debug paths.  The synthesizable scheduler contains no
// runtime division, remainder, or two-dimensional active-array read.
//
// The pre-existing binary source is load_source_active with sign=0 (+1).
// load_source_sign is the additive typed-signed bridge: sign=1 means -1.
// Before Acc24, -1 uses exact nine-bit two's-complement negation, so INT8 -128
// maps to +128 without the invalid eight-bit wraparound.  Inactive sources do
// not issue and never update an accumulator.
module m2193_c2_tsbg_b4_selective_bank_fill_frontend #(
    parameter int SCHEDULE_MODE = 1,
    parameter int BUNDLE = 4,
    parameter int SOURCE_GROUPS = 48,
    parameter int SOURCES_PER_GROUP = 16,
    parameter int OUTPUT_SLICES = 6,
    parameter int CACHE_ROWS = 4,
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int LANES = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [2:0]                   load_context,
    input  logic [TAG_BITS-1:0]          load_tag,
    input  logic [5:0]                   load_group,
    input  logic [SOURCES_PER_GROUP-1:0] load_source_active,
    input  logic [SOURCES_PER_GROUP-1:0] load_source_sign,
    input  logic                         load_last,
    output logic                         load_accept,

    // Frozen M803 eight-channel SRAM protocol.  Ready and responses are
    // independent per bank; response order need not match request order.
    output logic [7:0]                   mem_req_valid,
    input  logic [7:0]                   mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch [0:7],
    output logic [2:0]                   mem_req_slot [0:7],
    output logic [GENERATION_BITS-1:0]   mem_req_generation [0:7],
    output logic [TAG_BITS-1:0]          mem_req_tag [0:7],
    output logic [2:0]                   mem_req_output_block [0:7],
    output logic [2:0]                   mem_req_slice [0:7],
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel [0:7],
    output logic [7:0]                   mem_req_accept,

    input  logic [7:0]                   mem_rsp_valid,
    output logic [7:0]                   mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch [0:7],
    input  logic [2:0]                   mem_rsp_slot [0:7],
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation [0:7],
    input  logic [TAG_BITS-1:0]          mem_rsp_tag [0:7],
    input  logic signed [7:0]            mem_rsp_weight [0:7][0:LANES-1],
    output logic [7:0]                   mem_rsp_accept,

    // Observable typed-signed bridge immediately before the shared Acc24.
    output logic                         bridge_valid,
    input  logic                         bridge_ready,
    output logic [2:0]                   bridge_context,
    output logic [5:0]                   bridge_group,
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
    output logic [31:0]                  debug_cycle_count,
    output logic [31:0]                  debug_row_access_count,
    output logic [31:0]                  debug_cache_hit_count,
    output logic [31:0]                  debug_cache_miss_count,
    output logic [31:0]                  debug_cache_eviction_count,
    output logic [31:0]                  debug_weight_bundle_beat_count,
    output logic [31:0]                  debug_scalar_bank_request_count,
    output logic [31:0]                  debug_scalar_bank_response_count,
    output logic [31:0]                  debug_issue_count,
    output logic [31:0]                  debug_signed_product_count,
    output logic [31:0]                  debug_commit_count,
    output logic [31:0]                  debug_partial_hit_count,
    output logic [31:0]                  debug_refill_bank_request_count,
    output logic [31:0]                  debug_zero_descriptor_skip_count
);
    localparam int ORDERED_ROWS = 192;
    localparam int PRIORITY_BLOCKS = 12;
    localparam int PRIORITY_LANES = 16;
    // Production proof is frozen at the maximum legal H67 geometry.  The
    // elaborated bound may be smaller for a directed source-only regression,
    // but it must never exceed the fixed production proof.
    localparam int PRODUCTION_SOURCE_GROUPS = 48;
    localparam int PRODUCTION_ACC24_ABS_BOUND =
        PRODUCTION_SOURCE_GROUPS * 16 * 128;
    localparam int ELABORATED_ACC24_ABS_BOUND = SOURCE_GROUPS * 16 * 128;
    localparam bit PARAMETERS_LEGAL =
        (SCHEDULE_MODE == 0 || SCHEDULE_MODE == 1)
        && BUNDLE == 4 && SOURCES_PER_GROUP == 16
        && OUTPUT_SLICES == 6 && CACHE_ROWS == 4 && LANES == 16
        && SOURCE_GROUPS >= 1 && SOURCE_GROUPS <= PRODUCTION_SOURCE_GROUPS
        && PRODUCTION_ACC24_ABS_BOUND == 98304
        && ELABORATED_ACC24_ABS_BOUND <= PRODUCTION_ACC24_ABS_BOUND
        && PRODUCTION_ACC24_ABS_BOUND < (1 << 23);

    typedef enum logic [3:0] {
        ST_LOAD, ST_FIND, ST_FETCH_REQ, ST_FETCH_RSP,
        ST_BRIDGE, ST_COMMIT, ST_DONE, ST_FAULT
    } state_t;
    state_t state_q;

    logic fault_q, overflow_q;
    logic [2:0] expected_context_q;
    logic first_descriptor_q;
    logic [5:0] last_group_q;
    logic [TAG_BITS-1:0] context_tag_q [0:BUNDLE-1];
    logic [SOURCES_PER_GROUP-1:0] active_row_q
        [0:BUNDLE-1][0:PRODUCTION_SOURCE_GROUPS-1];
    logic [SOURCES_PER_GROUP-1:0] sign_row_q
        [0:BUNDLE-1][0:PRODUCTION_SOURCE_GROUPS-1];
    logic row_live_q [0:BUNDLE-1][0:PRODUCTION_SOURCE_GROUPS-1];
    logic signed [23:0] acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1];

    logic cache_valid_q [0:CACHE_ROWS-1];
    logic [5:0] cache_group_q [0:CACHE_ROWS-1];
    // A tag may be resident while only a subset of source banks is resident.
    // Validity is per half/per bank and covers all six output slices only
    // after the final slice response for that refill has been accepted.
    logic [7:0] cache_bank_valid_q [0:CACHE_ROWS-1][0:1];
    logic [31:0] cache_age_q [0:CACHE_ROWS-1];
    logic signed [7:0] cache_weight_q
        [0:CACHE_ROWS-1][0:1][0:OUTPUT_SLICES-1][0:7][0:LANES-1];
    logic [31:0] access_clock_q;

    logic [2:0] current_context_q, current_cache_q;
    logic [5:0] current_group_q;
    logic [SOURCES_PER_GROUP-1:0] current_active_row_q;
    logic [SOURCES_PER_GROUP-1:0] current_sign_row_q;
    logic current_half_q;
    logic [2:0] current_slice_q;
    logic [2:0] fill_cache_q;
    logic fill_half_q;
    logic [2:0] fill_slice_q;
    logic [7:0] fill_missing_mask_q [0:1];
    logic [2:0] commit_context_q, commit_slice_q;

    logic [EPOCH_BITS-1:0] transaction_epoch_q;
    logic [GENERATION_BITS-1:0] transaction_generation_q;
    logic core_req_valid, core_req_ready, core_req_accept;
    logic [EPOCH_BITS-1:0] core_req_epoch;
    logic [2:0] core_req_slot;
    logic [GENERATION_BITS-1:0] core_req_generation;
    logic [TAG_BITS-1:0] core_req_tag;
    logic [2:0] core_req_output_block, core_req_slice;
    logic [3:0] core_req_source_count;
    logic [7:0] core_req_bank_valid;
    logic [CHANNEL_BITS-1:0] core_req_source_channel [0:7];
    logic core_rsp_valid, core_rsp_ready, core_rsp_accept;
    logic [EPOCH_BITS-1:0] core_rsp_epoch;
    logic [2:0] core_rsp_slot;
    logic [GENERATION_BITS-1:0] core_rsp_generation;
    logic [TAG_BITS-1:0] core_rsp_tag;
    logic [7:0] core_rsp_bank_valid;
    logic signed [7:0] core_rsp_weight [0:7][0:LANES-1];
    logic adapter_protocol_error, adapter_stale_response_seen, adapter_busy;
    logic [3:0] adapter_live_slots;
    logic [31:0] adapter_bundle_req_count, adapter_bank_req_count;
    logic [31:0] adapter_bank_rsp_count, adapter_bundle_rsp_count;

    logic find_valid;
    logic [2:0] find_context;
    logic [5:0] find_group;
    logic [SOURCES_PER_GROUP-1:0] find_active_row;
    logic [SOURCES_PER_GROUP-1:0] find_sign_row;
    logic [ORDERED_ROWS-1:0] ordered_row_live;
    logic [SOURCES_PER_GROUP-1:0] ordered_active_row [0:ORDERED_ROWS-1];
    logic [SOURCES_PER_GROUP-1:0] ordered_sign_row [0:ORDERED_ROWS-1];
    logic [2:0] ordered_context [0:ORDERED_ROWS-1];
    logic [5:0] ordered_group [0:ORDERED_ROWS-1];
    logic [PRIORITY_BLOCKS-1:0] priority_block_live;
    logic [PRIORITY_LANES-1:0] priority_lane_onehot
        [0:PRIORITY_BLOCKS-1];
    logic [PRIORITY_BLOCKS-1:0] priority_block_onehot;
    logic [ORDERED_ROWS-1:0] find_onehot;
    logic find_cache_hit, find_cache_tag_match, find_cache_has_invalid;
    logic [2:0] find_cache_index, find_cache_tag_index, find_victim_index;
    logic [7:0] find_needed_mask [0:1];
    logic [7:0] find_missing_mask [0:1];
    logic current_half_active, other_half_active;
    logic core_rsp_identity_legal;
    logic signed [24:0] bridge_delta [0:LANES-1];
    logic signed [24:0] bridge_next_acc [0:LANES-1];
    logic bridge_overflow [0:LANES-1];
    logic [4:0] bridge_source_count;

    logic [31:0] cycle_count_q, row_access_count_q;
    logic [31:0] cache_hit_count_q, cache_miss_count_q;
    logic [31:0] cache_eviction_count_q, weight_bundle_beat_count_q;
    logic [31:0] issue_count_q, signed_product_count_q, commit_count_q;
    logic [31:0] partial_hit_count_q, refill_bank_request_count_q;
    logic [31:0] zero_descriptor_skip_count_q;

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M2193 legal point is B4/G1..48/S16/O6/LRU4/L16/Acc24");
        end

        // The two modes differ only in static wiring into the same physical
        // 192-bit bitmap and priority encoder.  Unelaborated group positions
        // are hard zero, preserving the directed G1..48 legal parameter range.
        if (SCHEDULE_MODE == 0) begin : g_token_major_order
            for (genvar map_ctx = 0; map_ctx < BUNDLE; map_ctx++) begin : g_ctx
                for (genvar map_group = 0;
                     map_group < PRODUCTION_SOURCE_GROUPS;
                     map_group++) begin : g_group
                    localparam int ORDER_INDEX =
                        map_ctx * PRODUCTION_SOURCE_GROUPS + map_group;
                    if (map_group < SOURCE_GROUPS) begin : g_present
                        assign ordered_row_live[ORDER_INDEX] =
                            row_live_q[map_ctx][map_group];
                        assign ordered_active_row[ORDER_INDEX] =
                            active_row_q[map_ctx][map_group];
                        assign ordered_sign_row[ORDER_INDEX] =
                            sign_row_q[map_ctx][map_group];
                        assign ordered_context[ORDER_INDEX] = 3'(map_ctx);
                        assign ordered_group[ORDER_INDEX] = 6'(map_group);
                    end else begin : g_absent
                        assign ordered_row_live[ORDER_INDEX] = 1'b0;
                        assign ordered_active_row[ORDER_INDEX] = '0;
                        assign ordered_sign_row[ORDER_INDEX] = '0;
                        assign ordered_context[ORDER_INDEX] = 3'(map_ctx);
                        assign ordered_group[ORDER_INDEX] = 6'(map_group);
                    end
                end
            end
        end else begin : g_group_major_order
            for (genvar map_group = 0;
                 map_group < PRODUCTION_SOURCE_GROUPS;
                 map_group++) begin : g_group
                for (genvar map_ctx = 0; map_ctx < BUNDLE; map_ctx++) begin : g_ctx
                    localparam int ORDER_INDEX = map_group * BUNDLE + map_ctx;
                    if (map_group < SOURCE_GROUPS) begin : g_present
                        assign ordered_row_live[ORDER_INDEX] =
                            row_live_q[map_ctx][map_group];
                        assign ordered_active_row[ORDER_INDEX] =
                            active_row_q[map_ctx][map_group];
                        assign ordered_sign_row[ORDER_INDEX] =
                            sign_row_q[map_ctx][map_group];
                        assign ordered_context[ORDER_INDEX] = 3'(map_ctx);
                        assign ordered_group[ORDER_INDEX] = 6'(map_group);
                    end else begin : g_absent
                        assign ordered_row_live[ORDER_INDEX] = 1'b0;
                        assign ordered_active_row[ORDER_INDEX] = '0;
                        assign ordered_sign_row[ORDER_INDEX] = '0;
                        assign ordered_context[ORDER_INDEX] = 3'(map_ctx);
                        assign ordered_group[ORDER_INDEX] = 6'(map_group);
                    end
                end
            end
        end
    endgenerate

    function automatic logic half_has_source(
        input logic [SOURCES_PER_GROUP-1:0] active_row,
        input logic half);
        begin
            if (half)
                return |active_row[15:8];
            return |active_row[7:0];
        end
    endfunction

    function automatic logic [TAG_BITS-1:0] fetch_tag(
        input logic [5:0] group,
        input logic half,
        input logic [2:0] output_slice,
        input logic [5:0] generation_low);
        fetch_tag = {8'h87, group, half, output_slice, generation_low};
    endfunction

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bank = 0; bank < 8; bank++)
                count = count + value[bank];
            return count;
        end
    endfunction

    assign load_ready = state_q == ST_LOAD && !fault_q
        && !adapter_protocol_error;
    assign load_accept = load_valid && load_ready
        && load_context == expected_context_q
        && load_group < SOURCE_GROUPS
        && (first_descriptor_q || load_group > last_group_q)
        && (first_descriptor_q || load_tag == context_tag_q[expected_context_q]);

    // Twelve local 16-way encoders feed one 12-way block encoder.  This is a
    // fixed-depth scheduler for both modes; empty rows never enter row_live_q.
    always_comb begin : hierarchical_row_priority
        logic seen_block;
        logic seen_lane;
        find_valid = 0;
        find_context = 0;
        find_group = 0;
        find_active_row = '0;
        find_sign_row = '0;
        find_onehot = '0;
        priority_block_live = '0;
        priority_block_onehot = '0;

        for (int block = 0; block < PRIORITY_BLOCKS; block++) begin
            seen_lane = 1'b0;
            for (int lane = 0; lane < PRIORITY_LANES; lane++) begin
                priority_block_live[block] |=
                    ordered_row_live[block * PRIORITY_LANES + lane];
                priority_lane_onehot[block][lane] =
                    ordered_row_live[block * PRIORITY_LANES + lane]
                    && !seen_lane;
                seen_lane |= ordered_row_live[block * PRIORITY_LANES + lane];
            end
        end

        seen_block = 1'b0;
        for (int block = 0; block < PRIORITY_BLOCKS; block++) begin
            priority_block_onehot[block] =
                priority_block_live[block] && !seen_block;
            seen_block |= priority_block_live[block];
        end

        for (int block = 0; block < PRIORITY_BLOCKS; block++) begin
            for (int lane = 0; lane < PRIORITY_LANES; lane++) begin
                find_onehot[block * PRIORITY_LANES + lane] =
                    priority_block_onehot[block]
                    && priority_lane_onehot[block][lane];
                if (find_onehot[block * PRIORITY_LANES + lane]) begin
                    find_valid = 1'b1;
                    find_context =
                        ordered_context[block * PRIORITY_LANES + lane];
                    find_group =
                        ordered_group[block * PRIORITY_LANES + lane];
                    find_active_row =
                        ordered_active_row[block * PRIORITY_LANES + lane];
                    find_sign_row =
                        ordered_sign_row[block * PRIORITY_LANES + lane];
                end
            end
        end
    end

    always_comb begin : cache_lookup
        logic invalid_found;
        logic [31:0] oldest_age;
        find_needed_mask[0] = 0;
        find_needed_mask[1] = 0;
        for (int ctx = 0; ctx < BUNDLE; ctx++) begin
            find_needed_mask[0] |= active_row_q[ctx][find_group][7:0];
            find_needed_mask[1] |= active_row_q[ctx][find_group][15:8];
        end
        find_cache_hit = 0;
        find_cache_tag_match = 0;
        find_cache_has_invalid = 0;
        find_cache_index = 0;
        find_cache_tag_index = 0;
        find_victim_index = 0;
        find_missing_mask[0] = find_needed_mask[0];
        find_missing_mask[1] = find_needed_mask[1];
        invalid_found = 0;
        oldest_age = 32'hffff_ffff;
        for (int entry = 0; entry < CACHE_ROWS; entry++) begin
            if (cache_valid_q[entry] && cache_group_q[entry] == find_group
                    && !find_cache_tag_match) begin
                find_cache_tag_match = 1;
                find_cache_tag_index = entry[2:0];
                find_missing_mask[0] = find_needed_mask[0]
                    & ~cache_bank_valid_q[entry][0];
                find_missing_mask[1] = find_needed_mask[1]
                    & ~cache_bank_valid_q[entry][1];
                if (((find_needed_mask[0] & ~cache_bank_valid_q[entry][0]) == 0)
                        && ((find_needed_mask[1]
                             & ~cache_bank_valid_q[entry][1]) == 0)) begin
                    find_cache_hit = 1;
                    find_cache_index = entry[2:0];
                end
            end
            if (!cache_valid_q[entry] && !invalid_found) begin
                invalid_found = 1;
                find_cache_has_invalid = 1;
                find_victim_index = entry[2:0];
            end
        end
        if (find_cache_tag_match) begin
            find_victim_index = find_cache_tag_index;
        end else if (!invalid_found) begin
            for (int entry = 0; entry < CACHE_ROWS; entry++) begin
                if (cache_age_q[entry] < oldest_age) begin
                    oldest_age = cache_age_q[entry];
                    find_victim_index = entry[2:0];
                end
            end
        end
    end

    assign core_req_valid = state_q == ST_FETCH_REQ && !fault_q;
    assign core_req_epoch = transaction_epoch_q;
    assign core_req_slot = transaction_generation_q[2:0];
    assign core_req_generation = transaction_generation_q;
    assign core_req_tag = fetch_tag(current_group_q, fill_half_q,
                                    fill_slice_q, transaction_generation_q[5:0]);
    assign core_req_output_block = {2'b00, fill_half_q};
    assign core_req_slice = fill_slice_q;
    assign core_req_bank_valid = fill_missing_mask_q[fill_half_q];
    assign core_req_source_count = popcount8(core_req_bank_valid);
    always_comb begin : fetch_channels
        for (int bank = 0; bank < 8; bank++)
            core_req_source_channel[bank] =
                CHANNEL_BITS'(int'(current_group_q) * 16
                              + int'(fill_half_q) * 8 + bank);
    end

    assign core_rsp_ready = state_q == ST_FETCH_RSP && !fault_q;
    assign core_rsp_identity_legal = core_rsp_epoch == transaction_epoch_q
        && core_rsp_slot == transaction_generation_q[2:0]
        && core_rsp_generation == transaction_generation_q
        && core_rsp_tag == fetch_tag(current_group_q, fill_half_q,
                                     fill_slice_q, transaction_generation_q[5:0])
        && core_rsp_bank_valid == fill_missing_mask_q[fill_half_q];

    m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter (
        .clk_core(clk_core), .rst_core(rst_core),
        .core_req_valid(core_req_valid), .core_req_ready(core_req_ready),
        .core_req_epoch(core_req_epoch), .core_req_slot(core_req_slot),
        .core_req_generation(core_req_generation), .core_req_tag(core_req_tag),
        .core_req_output_block(core_req_output_block),
        .core_req_slice(core_req_slice),
        .core_req_source_count(core_req_source_count),
        .core_req_bank_valid(core_req_bank_valid),
        .core_req_source_channel(core_req_source_channel),
        .core_req_accept(core_req_accept),
        .bank_req_valid(mem_req_valid), .bank_req_ready(mem_req_ready),
        .bank_req_epoch(mem_req_epoch), .bank_req_slot(mem_req_slot),
        .bank_req_generation(mem_req_generation), .bank_req_tag(mem_req_tag),
        .bank_req_output_block(mem_req_output_block),
        .bank_req_slice(mem_req_slice),
        .bank_req_source_channel(mem_req_source_channel),
        .bank_req_accept(mem_req_accept),
        .bank_rsp_valid(mem_rsp_valid), .bank_rsp_ready(mem_rsp_ready),
        .bank_rsp_epoch(mem_rsp_epoch), .bank_rsp_slot(mem_rsp_slot),
        .bank_rsp_generation(mem_rsp_generation), .bank_rsp_tag(mem_rsp_tag),
        .bank_rsp_weight(mem_rsp_weight), .bank_rsp_accept(mem_rsp_accept),
        .core_rsp_valid(core_rsp_valid), .core_rsp_ready(core_rsp_ready),
        .core_rsp_epoch(core_rsp_epoch), .core_rsp_slot(core_rsp_slot),
        .core_rsp_generation(core_rsp_generation), .core_rsp_tag(core_rsp_tag),
        .core_rsp_bank_valid(core_rsp_bank_valid),
        .core_rsp_weight(core_rsp_weight), .core_rsp_accept(core_rsp_accept),
        .protocol_error(adapter_protocol_error),
        .stale_response_seen(adapter_stale_response_seen),
        .busy(adapter_busy), .debug_live_slots(adapter_live_slots),
        .debug_bundle_request_count(adapter_bundle_req_count),
        .debug_bank_request_count(adapter_bank_req_count),
        .debug_bank_response_count(adapter_bank_rsp_count),
        .debug_bundle_response_count(adapter_bundle_rsp_count));

    always_comb begin : signed_bridge
        logic signed [8:0] widened_weight;
        logic signed [24:0] delta;
        bridge_valid = state_q == ST_BRIDGE && !fault_q;
        bridge_context = current_context_q;
        bridge_group = current_group_q;
        bridge_half = current_half_q;
        bridge_slice = current_slice_q;
        bridge_bank_valid = 0;
        bridge_source_count = 0;
        for (int bank = 0; bank < 8; bank++) begin
            bridge_source_channel[bank] = CHANNEL_BITS'(
                int'(current_group_q) * 16 + int'(current_half_q) * 8 + bank);
            if (current_half_q)
                bridge_bank_valid[bank] = current_active_row_q[bank + 8];
            else
                bridge_bank_valid[bank] = current_active_row_q[bank];
            bridge_source_value[bank] = 0;
            if (bridge_bank_valid[bank]) begin
                bridge_source_count = bridge_source_count + 1'b1;
                if (current_half_q)
                    bridge_source_value[bank] = current_sign_row_q[bank + 8]
                        ? -2'sd1 : 2'sd1;
                else
                    bridge_source_value[bank] = current_sign_row_q[bank]
                        ? -2'sd1 : 2'sd1;
            end
            for (int lane = 0; lane < LANES; lane++) begin
                widened_weight = {cache_weight_q[current_cache_q][current_half_q]
                                  [current_slice_q][bank][lane][7],
                                  cache_weight_q[current_cache_q][current_half_q]
                                  [current_slice_q][bank][lane]};
                bridge_effective_weight[bank][lane] = 0;
                if (bridge_bank_valid[bank]) begin
                    if (bridge_source_value[bank] == -2'sd1)
                        bridge_effective_weight[bank][lane] = -widened_weight;
                    else
                        bridge_effective_weight[bank][lane] = widened_weight;
                end
            end
        end
        bridge_accept = bridge_valid && bridge_ready;
        for (int lane = 0; lane < LANES; lane++) begin
            delta = 0;
            for (int bank = 0; bank < 8; bank++)
                delta = delta + {{16{bridge_effective_weight[bank][lane][8]}},
                                  bridge_effective_weight[bank][lane]};
            bridge_delta[lane] = delta;
            bridge_next_acc[lane] =
                {acc_q[current_context_q][current_slice_q][lane][23],
                 acc_q[current_context_q][current_slice_q][lane]} + delta;
            bridge_overflow[lane] = bridge_next_acc[lane][24]
                != bridge_next_acc[lane][23];
        end
        current_half_active = half_has_source(current_active_row_q,
                                              current_half_q);
        other_half_active = half_has_source(current_active_row_q,
                                            !current_half_q);
    end

    always_comb begin : commit_view
        commit_valid = state_q == ST_COMMIT && !fault_q;
        commit_context = commit_context_q;
        commit_tag = context_tag_q[commit_context_q];
        commit_slice = commit_slice_q;
        for (int lane = 0; lane < LANES; lane++)
            commit_accumulator[lane] = acc_q[commit_context_q][commit_slice_q][lane];
        commit_terminal = commit_slice_q == OUTPUT_SLICES - 1;
        commit_accept = commit_valid && commit_ready;
        bundle_done_valid = state_q == ST_DONE && !fault_q;
    end

    assign protocol_error = fault_q || adapter_protocol_error;
    assign stale_response_seen = adapter_stale_response_seen;
    assign numeric_overflow = overflow_q;
    assign busy = state_q != ST_LOAD || expected_context_q != 0 || adapter_busy;
    assign debug_cycle_count = cycle_count_q;
    assign debug_row_access_count = row_access_count_q;
    assign debug_cache_hit_count = cache_hit_count_q;
    assign debug_cache_miss_count = cache_miss_count_q;
    assign debug_cache_eviction_count = cache_eviction_count_q;
    assign debug_weight_bundle_beat_count = weight_bundle_beat_count_q;
    assign debug_scalar_bank_request_count = adapter_bank_req_count;
    assign debug_scalar_bank_response_count = adapter_bank_rsp_count;
    assign debug_issue_count = issue_count_q;
    assign debug_signed_product_count = signed_product_count_q;
    assign debug_commit_count = commit_count_q;
    assign debug_partial_hit_count = partial_hit_count_q;
    assign debug_refill_bank_request_count = refill_bank_request_count_q;
    assign debug_zero_descriptor_skip_count = zero_descriptor_skip_count_q;

    always_ff @(posedge clk_core) begin : state_and_ledgers
        if (rst_core) begin
            state_q <= ST_LOAD;
            fault_q <= 0;
            overflow_q <= 0;
            expected_context_q <= 0;
            first_descriptor_q <= 1;
            last_group_q <= 0;
            current_context_q <= 0;
            current_group_q <= 0;
            current_active_row_q <= 0;
            current_sign_row_q <= 0;
            current_cache_q <= 0;
            current_half_q <= 0;
            current_slice_q <= 0;
            fill_cache_q <= 0;
            fill_half_q <= 0;
            fill_slice_q <= 0;
            fill_missing_mask_q[0] <= 0;
            fill_missing_mask_q[1] <= 0;
            commit_context_q <= 0;
            commit_slice_q <= 0;
            transaction_epoch_q <= 16'h1787;
            transaction_generation_q <= 1;
            access_clock_q <= 1;
            cycle_count_q <= 0;
            row_access_count_q <= 0;
            cache_hit_count_q <= 0;
            cache_miss_count_q <= 0;
            cache_eviction_count_q <= 0;
            weight_bundle_beat_count_q <= 0;
            issue_count_q <= 0;
            signed_product_count_q <= 0;
            commit_count_q <= 0;
            partial_hit_count_q <= 0;
            refill_bank_request_count_q <= 0;
            zero_descriptor_skip_count_q <= 0;
            for (int ctx = 0; ctx < BUNDLE; ctx++) begin
                context_tag_q[ctx] <= 0;
                for (int group = 0; group < PRODUCTION_SOURCE_GROUPS;
                     group++) begin
                    active_row_q[ctx][group] <= 0;
                    sign_row_q[ctx][group] <= 0;
                    row_live_q[ctx][group] <= 0;
                end
                for (int output_slice = 0; output_slice < OUTPUT_SLICES;
                     output_slice++)
                    for (int lane = 0; lane < LANES; lane++)
                        acc_q[ctx][output_slice][lane] <= 0;
            end
            for (int entry = 0; entry < CACHE_ROWS; entry++) begin
                cache_valid_q[entry] <= 0;
                cache_group_q[entry] <= 0;
                cache_age_q[entry] <= 0;
                cache_bank_valid_q[entry][0] <= 0;
                cache_bank_valid_q[entry][1] <= 0;
            end
        end else begin
            if (state_q != ST_LOAD && state_q != ST_DONE && state_q != ST_FAULT)
                cycle_count_q <= cycle_count_q + 1'b1;

            if ((load_valid && load_ready && !load_accept)
                    || (core_rsp_valid && core_rsp_ready
                        && !core_rsp_identity_legal)
                    || adapter_protocol_error) begin
                fault_q <= 1;
                state_q <= ST_FAULT;
            end

            if (load_accept) begin
                if (load_source_active == 0)
                    zero_descriptor_skip_count_q
                        <= zero_descriptor_skip_count_q + 1'b1;
                if (first_descriptor_q) context_tag_q[load_context] <= load_tag;
                active_row_q[load_context][load_group] <= load_source_active;
                sign_row_q[load_context][load_group] <=
                    load_source_active & load_source_sign;
                row_live_q[load_context][load_group] <= |load_source_active;
                last_group_q <= load_group;
                first_descriptor_q <= 0;
                if (load_last) begin
                    first_descriptor_q <= 1;
                    last_group_q <= 0;
                    if (expected_context_q == BUNDLE - 1) begin
                        expected_context_q <= 0;
                        state_q <= ST_FIND;
                    end else begin
                        expected_context_q <= expected_context_q + 1'b1;
                    end
                end
            end

            if (state_q == ST_FIND) begin
                if (!find_valid) begin
                    commit_context_q <= 0;
                    commit_slice_q <= 0;
                    state_q <= ST_COMMIT;
                end else begin
                    current_context_q <= find_context;
                    current_group_q <= find_group;
                    current_active_row_q <= find_active_row;
                    current_sign_row_q <= find_sign_row;
                    // Consume exactly the statically selected row.  Clearing
                    // here preserves one-cycle ST_FIND selection and prevents
                    // any row replay after cache/fetch/bridge backpressure.
                    for (int ctx = 0; ctx < BUNDLE; ctx++) begin
                        for (int group = 0;
                             group < PRODUCTION_SOURCE_GROUPS; group++) begin
                            if (SCHEDULE_MODE == 0) begin
                                if (find_onehot[
                                        ctx * PRODUCTION_SOURCE_GROUPS + group])
                                    row_live_q[ctx][group] <= 1'b0;
                            end else begin
                                if (find_onehot[group * BUNDLE + ctx])
                                    row_live_q[ctx][group] <= 1'b0;
                            end
                        end
                    end
                    row_access_count_q <= row_access_count_q + 1'b1;
                    access_clock_q <= access_clock_q + 1'b1;
                    if (find_cache_hit) begin
                        current_cache_q <= find_cache_index;
                        cache_age_q[find_cache_index] <= access_clock_q;
                        cache_hit_count_q <= cache_hit_count_q + 1'b1;
                        current_half_q <= half_has_source(find_active_row, 0)
                            ? 1'b0 : 1'b1;
                        current_slice_q <= 0;
                        state_q <= ST_BRIDGE;
                    end else begin
                        current_cache_q <= find_victim_index;
                        fill_cache_q <= find_victim_index;
                        fill_missing_mask_q[0] <= find_missing_mask[0];
                        fill_missing_mask_q[1] <= find_missing_mask[1];
                        fill_half_q <= find_missing_mask[0] != 0 ? 1'b0 : 1'b1;
                        fill_slice_q <= 0;
                        cache_miss_count_q <= cache_miss_count_q + 1'b1;
                        if (find_cache_tag_match) begin
                            partial_hit_count_q <= partial_hit_count_q + 1'b1;
                        end else begin
                            cache_valid_q[find_victim_index] <= 1;
                            cache_group_q[find_victim_index] <= find_group;
                            cache_bank_valid_q[find_victim_index][0] <= 0;
                            cache_bank_valid_q[find_victim_index][1] <= 0;
                        end
                        if (!find_cache_tag_match && !find_cache_has_invalid)
                            cache_eviction_count_q <= cache_eviction_count_q + 1'b1;
                        state_q <= ST_FETCH_REQ;
                    end
                end
            end

            if (state_q == ST_FETCH_REQ && core_req_accept) begin
                refill_bank_request_count_q <= refill_bank_request_count_q
                    + popcount8(core_req_bank_valid);
                state_q <= ST_FETCH_RSP;
            end

            if (state_q == ST_FETCH_RSP && core_rsp_accept
                    && core_rsp_identity_legal) begin
                for (int bank = 0; bank < 8; bank++) begin
                    if (core_rsp_bank_valid[bank]) begin
                        for (int lane = 0; lane < LANES; lane++)
                            cache_weight_q[fill_cache_q][fill_half_q][fill_slice_q]
                                          [bank][lane] <= core_rsp_weight[bank][lane];
                    end
                end
                weight_bundle_beat_count_q <= weight_bundle_beat_count_q + 1'b1;
                transaction_generation_q <= transaction_generation_q + 1'b1;
                if (fill_slice_q == OUTPUT_SLICES - 1) begin
                    cache_bank_valid_q[fill_cache_q][fill_half_q]
                        <= cache_bank_valid_q[fill_cache_q][fill_half_q]
                           | fill_missing_mask_q[fill_half_q];
                end
                if (fill_slice_q == OUTPUT_SLICES - 1
                        && (fill_half_q
                            || fill_missing_mask_q[1] == 0)) begin
                    cache_age_q[fill_cache_q] <= access_clock_q;
                    current_half_q <= half_has_source(current_active_row_q, 0)
                        ? 1'b0 : 1'b1;
                    current_slice_q <= 0;
                    state_q <= ST_BRIDGE;
                end else begin
                    if (fill_slice_q == OUTPUT_SLICES - 1) begin
                        fill_slice_q <= 0;
                        fill_half_q <= 1;
                    end else begin
                        fill_slice_q <= fill_slice_q + 1'b1;
                    end
                    state_q <= ST_FETCH_REQ;
                end
            end

            if (state_q == ST_BRIDGE && bridge_accept) begin
                issue_count_q <= issue_count_q + 1'b1;
                signed_product_count_q <= signed_product_count_q
                    + bridge_source_count * LANES;
                for (int lane = 0; lane < LANES; lane++) begin
                    acc_q[current_context_q][current_slice_q][lane]
                        <= bridge_next_acc[lane][23:0];
                    if (bridge_overflow[lane]) overflow_q <= 1;
                end
                if (current_slice_q == OUTPUT_SLICES - 1) begin
                    if (!current_half_q && other_half_active) begin
                        current_half_q <= 1;
                        current_slice_q <= 0;
                    end else begin
                        current_slice_q <= 0;
                        state_q <= ST_FIND;
                    end
                end else begin
                    current_slice_q <= current_slice_q + 1'b1;
                end
            end

            if (state_q == ST_COMMIT && commit_accept) begin
                commit_count_q <= commit_count_q + 1'b1;
                if (commit_slice_q == OUTPUT_SLICES - 1) begin
                    commit_slice_q <= 0;
                    if (commit_context_q == BUNDLE - 1)
                        state_q <= ST_DONE;
                    else
                        commit_context_q <= commit_context_q + 1'b1;
                end else begin
                    commit_slice_q <= commit_slice_q + 1'b1;
                end
            end

            if (state_q == ST_DONE && bundle_done_ready) begin
                state_q <= ST_LOAD;
                expected_context_q <= 0;
                first_descriptor_q <= 1;
                last_group_q <= 0;
                current_active_row_q <= 0;
                current_sign_row_q <= 0;
                transaction_epoch_q <= transaction_epoch_q + 1'b1;
                for (int ctx = 0; ctx < BUNDLE; ctx++) begin
                    context_tag_q[ctx] <= 0;
                    for (int group = 0;
                         group < PRODUCTION_SOURCE_GROUPS; group++) begin
                        active_row_q[ctx][group] <= 0;
                        sign_row_q[ctx][group] <= 0;
                        row_live_q[ctx][group] <= 0;
                    end
                    for (int output_slice = 0;
                         output_slice < OUTPUT_SLICES; output_slice++)
                        for (int lane = 0; lane < LANES; lane++)
                            acc_q[ctx][output_slice][lane] <= 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
