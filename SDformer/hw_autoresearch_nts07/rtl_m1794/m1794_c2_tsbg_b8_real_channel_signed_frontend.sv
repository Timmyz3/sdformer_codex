`timescale 1ns/1ps
`default_nettype none

// M1794 is the additive successor to M1787 after the M1788 source hammer.
// M1787 is intentionally not modified.  Its real M803 protocol and typed
// bridge are retained; the production Acc24 proof is separated from a legal
// reduced directed geometry so SOURCE_GROUPS=12 cannot fatal at time zero.
//
// This successor instantiates the frozen M803 channel-split adapter and exposes
// its eight independently handshaken SRAM banks verbatim.  Both schedule modes
// own exactly the same adapter, ordinary LRU8 row store, B8 source/sign FIFO,
// eight independent 96-value Acc24 contexts, and commit machinery.  MODE=0 is
// token-major; MODE=1 is source-group-major.  Only a fetched weight row is
// shared.  A product and accumulator update always retain token identity.
//
// The pre-existing binary source is load_source_active with sign=0 (+1).
// load_source_sign is the additive typed-signed bridge: sign=1 means -1.
// Before Acc24, -1 uses exact nine-bit two's-complement negation, so INT8 -128
// maps to +128 without the invalid eight-bit wraparound.  Inactive sources do
// not issue and never update an accumulator.
module m1794_c2_tsbg_b8_real_channel_signed_frontend #(
    parameter int SCHEDULE_MODE = 1,
    parameter int BUNDLE = 8,
    parameter int SOURCE_GROUPS = 48,
    parameter int SOURCES_PER_GROUP = 16,
    parameter int OUTPUT_SLICES = 6,
    parameter int CACHE_ROWS = 8,
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
    output logic [31:0]                  debug_commit_count
);
    localparam int TOTAL_ITEMS = BUNDLE * SOURCE_GROUPS;
    // Production proof is frozen at the maximum legal H67 geometry.  The
    // elaborated bound may be smaller for a directed source-only regression,
    // but it must never exceed the fixed production proof.
    localparam int PRODUCTION_SOURCE_GROUPS = 48;
    localparam int PRODUCTION_ACC24_ABS_BOUND =
        PRODUCTION_SOURCE_GROUPS * 16 * 128;
    localparam int ELABORATED_ACC24_ABS_BOUND = SOURCE_GROUPS * 16 * 128;
    localparam bit PARAMETERS_LEGAL =
        (SCHEDULE_MODE == 0 || SCHEDULE_MODE == 1)
        && BUNDLE == 8 && SOURCES_PER_GROUP == 16
        && OUTPUT_SLICES == 6 && CACHE_ROWS == 8 && LANES == 16
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
    logic active_q [0:BUNDLE-1][0:SOURCE_GROUPS-1][0:SOURCES_PER_GROUP-1];
    logic sign_q [0:BUNDLE-1][0:SOURCE_GROUPS-1][0:SOURCES_PER_GROUP-1];
    logic signed [23:0] acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1];

    logic cache_valid_q [0:CACHE_ROWS-1];
    logic [5:0] cache_group_q [0:CACHE_ROWS-1];
    logic [31:0] cache_age_q [0:CACHE_ROWS-1];
    logic signed [7:0] cache_weight_q
        [0:CACHE_ROWS-1][0:1][0:OUTPUT_SLICES-1][0:7][0:LANES-1];
    logic [31:0] access_clock_q;

    logic [$clog2(TOTAL_ITEMS+1)-1:0] scan_linear_q;
    logic [2:0] current_context_q, current_cache_q;
    logic [5:0] current_group_q;
    logic current_half_q;
    logic [2:0] current_slice_q;
    logic [2:0] fill_cache_q;
    logic fill_half_q;
    logic [2:0] fill_slice_q;
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
    logic [$clog2(TOTAL_ITEMS+1)-1:0] find_linear;
    logic find_cache_hit, find_cache_has_invalid;
    logic [2:0] find_cache_index, find_victim_index;
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

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M1794 legal point is B8/G1..48/S16/O6/LRU8/L16/Acc24");
        end
    endgenerate

    function automatic logic half_has_source(
        input logic [2:0] context,
        input logic [5:0] group,
        input logic half);
        logic any_source;
        begin
            any_source = 0;
            for (int bank = 0; bank < 8; bank++)
                any_source |= active_q[context][group][bank + (half ? 8 : 0)];
            return any_source;
        end
    endfunction

    function automatic logic [TAG_BITS-1:0] fetch_tag(
        input logic [5:0] group,
        input logic half,
        input logic [2:0] output_slice,
        input logic [5:0] generation_low);
        fetch_tag = {8'h87, group, half, output_slice, generation_low};
    endfunction

    assign load_ready = state_q == ST_LOAD && !fault_q
        && !adapter_protocol_error;
    assign load_accept = load_valid && load_ready
        && load_context == expected_context_q
        && load_group < SOURCE_GROUPS
        && (first_descriptor_q || load_group > last_group_q)
        && (first_descriptor_q || load_tag == context_tag_q[expected_context_q]);

    always_comb begin : next_active
        int raw, context_index, group_index;
        find_valid = 0;
        find_context = 0;
        find_group = 0;
        find_linear = scan_linear_q;
        for (int offset = 0; offset < TOTAL_ITEMS; offset++) begin
            raw = int'(scan_linear_q) + offset;
            if (!find_valid && raw < TOTAL_ITEMS) begin
                if (SCHEDULE_MODE == 0) begin
                    context_index = raw / SOURCE_GROUPS;
                    group_index = raw % SOURCE_GROUPS;
                end else begin
                    group_index = raw / BUNDLE;
                    context_index = raw % BUNDLE;
                end
                for (int source = 0; source < SOURCES_PER_GROUP; source++) begin
                    if (active_q[context_index][group_index][source]) begin
                        find_valid = 1;
                        find_context = context_index[2:0];
                        find_group = group_index[5:0];
                        find_linear = raw + 1;
                    end
                end
            end
        end
    end

    always_comb begin : cache_lookup
        logic invalid_found;
        logic [31:0] oldest_age;
        find_cache_hit = 0;
        find_cache_has_invalid = 0;
        find_cache_index = 0;
        find_victim_index = 0;
        invalid_found = 0;
        oldest_age = 32'hffff_ffff;
        for (int entry = 0; entry < CACHE_ROWS; entry++) begin
            if (cache_valid_q[entry] && cache_group_q[entry] == find_group
                    && !find_cache_hit) begin
                find_cache_hit = 1;
                find_cache_index = entry[2:0];
            end
            if (!cache_valid_q[entry] && !invalid_found) begin
                invalid_found = 1;
                find_cache_has_invalid = 1;
                find_victim_index = entry[2:0];
            end
        end
        if (!invalid_found) begin
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
    assign core_req_source_count = 4'd8;
    assign core_req_bank_valid = 8'hff;
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
        && core_rsp_bank_valid == 8'hff;

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
            bridge_bank_valid[bank] = active_q[current_context_q]
                [current_group_q][bank + (current_half_q ? 8 : 0)];
            bridge_source_value[bank] = 0;
            if (bridge_bank_valid[bank]) begin
                bridge_source_count = bridge_source_count + 1'b1;
                bridge_source_value[bank] = sign_q[current_context_q]
                    [current_group_q][bank + (current_half_q ? 8 : 0)]
                    ? -2'sd1 : 2'sd1;
            end
            for (int lane = 0; lane < LANES; lane++) begin
                widened_weight = {cache_weight_q[current_cache_q][current_half_q]
                                  [current_slice_q][bank][lane][7],
                                  cache_weight_q[current_cache_q][current_half_q]
                                  [current_slice_q][bank][lane]};
                bridge_effective_weight[bank][lane] = 0;
                if (bridge_bank_valid[bank]) begin
                    if (sign_q[current_context_q][current_group_q]
                              [bank + (current_half_q ? 8 : 0)])
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
        current_half_active = half_has_source(current_context_q,
                                              current_group_q, current_half_q);
        other_half_active = half_has_source(current_context_q,
                                            current_group_q, !current_half_q);
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

    always_ff @(posedge clk_core) begin : state_and_ledgers
        if (rst_core) begin
            state_q <= ST_LOAD;
            fault_q <= 0;
            overflow_q <= 0;
            expected_context_q <= 0;
            first_descriptor_q <= 1;
            last_group_q <= 0;
            scan_linear_q <= 0;
            current_context_q <= 0;
            current_group_q <= 0;
            current_cache_q <= 0;
            current_half_q <= 0;
            current_slice_q <= 0;
            fill_cache_q <= 0;
            fill_half_q <= 0;
            fill_slice_q <= 0;
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
            for (int context = 0; context < BUNDLE; context++) begin
                context_tag_q[context] <= 0;
                for (int group = 0; group < SOURCE_GROUPS; group++) begin
                    for (int source = 0; source < SOURCES_PER_GROUP; source++) begin
                        active_q[context][group][source] <= 0;
                        sign_q[context][group][source] <= 0;
                    end
                end
                for (int output_slice = 0; output_slice < OUTPUT_SLICES;
                     output_slice++)
                    for (int lane = 0; lane < LANES; lane++)
                        acc_q[context][output_slice][lane] <= 0;
            end
            for (int entry = 0; entry < CACHE_ROWS; entry++) begin
                cache_valid_q[entry] <= 0;
                cache_group_q[entry] <= 0;
                cache_age_q[entry] <= 0;
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
                if (first_descriptor_q) context_tag_q[load_context] <= load_tag;
                for (int source = 0; source < SOURCES_PER_GROUP; source++) begin
                    active_q[load_context][load_group][source]
                        <= load_source_active[source];
                    sign_q[load_context][load_group][source]
                        <= load_source_active[source] && load_source_sign[source];
                end
                last_group_q <= load_group;
                first_descriptor_q <= 0;
                if (load_last) begin
                    first_descriptor_q <= 1;
                    last_group_q <= 0;
                    if (expected_context_q == BUNDLE - 1) begin
                        expected_context_q <= 0;
                        scan_linear_q <= 0;
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
                    scan_linear_q <= find_linear;
                    row_access_count_q <= row_access_count_q + 1'b1;
                    access_clock_q <= access_clock_q + 1'b1;
                    if (find_cache_hit) begin
                        current_cache_q <= find_cache_index;
                        cache_age_q[find_cache_index] <= access_clock_q;
                        cache_hit_count_q <= cache_hit_count_q + 1'b1;
                        current_half_q <= half_has_source(find_context, find_group, 0)
                            ? 1'b0 : 1'b1;
                        current_slice_q <= 0;
                        state_q <= ST_BRIDGE;
                    end else begin
                        current_cache_q <= find_victim_index;
                        fill_cache_q <= find_victim_index;
                        cache_valid_q[find_victim_index] <= 0;
                        cache_group_q[find_victim_index] <= find_group;
                        fill_half_q <= 0;
                        fill_slice_q <= 0;
                        cache_miss_count_q <= cache_miss_count_q + 1'b1;
                        if (!find_cache_has_invalid)
                            cache_eviction_count_q <= cache_eviction_count_q + 1'b1;
                        state_q <= ST_FETCH_REQ;
                    end
                end
            end

            if (state_q == ST_FETCH_REQ && core_req_accept)
                state_q <= ST_FETCH_RSP;

            if (state_q == ST_FETCH_RSP && core_rsp_accept
                    && core_rsp_identity_legal) begin
                for (int bank = 0; bank < 8; bank++)
                    for (int lane = 0; lane < LANES; lane++)
                        cache_weight_q[fill_cache_q][fill_half_q][fill_slice_q]
                                      [bank][lane] <= core_rsp_weight[bank][lane];
                weight_bundle_beat_count_q <= weight_bundle_beat_count_q + 1'b1;
                transaction_generation_q <= transaction_generation_q + 1'b1;
                if (fill_half_q && fill_slice_q == OUTPUT_SLICES - 1) begin
                    cache_valid_q[fill_cache_q] <= 1;
                    cache_age_q[fill_cache_q] <= access_clock_q;
                    current_half_q <= half_has_source(current_context_q,
                                                      current_group_q, 0)
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
                scan_linear_q <= 0;
                transaction_epoch_q <= transaction_epoch_q + 1'b1;
                for (int context = 0; context < BUNDLE; context++) begin
                    context_tag_q[context] <= 0;
                    for (int group = 0; group < SOURCE_GROUPS; group++) begin
                        for (int source = 0; source < SOURCES_PER_GROUP;
                             source++) begin
                            active_q[context][group][source] <= 0;
                            sign_q[context][group][source] <= 0;
                        end
                    end
                    for (int output_slice = 0;
                         output_slice < OUTPUT_SLICES; output_slice++)
                        for (int lane = 0; lane < LANES; lane++)
                            acc_q[context][output_slice][lane] <= 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
