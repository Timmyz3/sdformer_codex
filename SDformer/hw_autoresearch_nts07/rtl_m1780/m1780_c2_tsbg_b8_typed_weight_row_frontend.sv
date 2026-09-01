`timescale 1ns/1ps
`default_nettype none

// M1780 is the source-only TSBG specialization at the existing C2 boundary.
// It does not create a second sparse executor.  Both elaboration modes own the
// same eight-bank weight port, ordinary LRU8 row store, B8 signed-code FIFO,
// eight independent Acc24 contexts and commit port.  MODE=0 schedules the
// buffered tokens in ordinary token-major order; MODE=1 changes only the
// order to source-group-major, so one fetched row can feed every live token.
// A weight is never shared as a product: each accepted issue retains its own
// signed {-1,0,+1} source value and updates exactly one Acc24 context.
module m1780_c2_tsbg_b8_typed_weight_row_frontend #(
    parameter int SCHEDULE_MODE = 1,       // 0 token-major, 1 TSBG row-major
    parameter int BUNDLE = 8,
    parameter int SOURCE_GROUPS = 48,
    parameter int SOURCES_PER_GROUP = 16,
    parameter int OUTPUT_SLICES = 6,
    parameter int BANKS = 8,
    parameter int LANES = 16,
    parameter int CACHE_ROWS = 8,
    parameter int TAG_BITS = 24,
    parameter int GROUP_BITS = 6
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    // One context is loaded at a time.  Only active groups need a descriptor;
    // an all-zero final descriptor is the legal empty-token representation.
    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [2:0]                   load_context,
    input  logic [TAG_BITS-1:0]          load_tag,
    input  logic [GROUP_BITS-1:0]        load_group,
    input  logic signed [7:0]            load_source_value
                                              [0:SOURCES_PER_GROUP-1],
    input  logic                         load_last,
    output logic                         load_accept,

    // One row is 16 sources x 96 outputs.  The existing eight INT8 banks
    // return one source half x one 16-lane slice per response beat.
    output logic                         mem_req_valid,
    input  logic                         mem_req_ready,
    output logic [GROUP_BITS-1:0]        mem_req_group,
    output logic                         mem_req_half,
    output logic [2:0]                   mem_req_slice,
    output logic                         mem_req_accept,
    input  logic                         mem_rsp_valid,
    output logic                         mem_rsp_ready,
    input  logic [GROUP_BITS-1:0]        mem_rsp_group,
    input  logic                         mem_rsp_half,
    input  logic [2:0]                   mem_rsp_slice,
    input  logic signed [7:0]            mem_rsp_weight [0:BANKS-1][0:LANES-1],
    output logic                         mem_rsp_accept,

    // This is the typed K8 handoff shape used by C2.  Values and weights are
    // both visible so a later composition can replace the local Acc24 ledger
    // with the admitted M803 service without changing TSBG ownership.
    output logic                         issue_valid,
    input  logic                         issue_ready,
    output logic [2:0]                   issue_context,
    output logic [GROUP_BITS-1:0]        issue_group,
    output logic                         issue_half,
    output logic [2:0]                   issue_slice,
    output logic [BANKS-1:0]             issue_bank_valid,
    output logic [GROUP_BITS+3:0]        issue_source_channel [0:BANKS-1],
    output logic signed [7:0]            issue_source_value [0:BANKS-1],
    output logic signed [7:0]            issue_weight [0:BANKS-1][0:LANES-1],
    output logic                         issue_accept,

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
    output logic                         numeric_overflow,
    output logic                         busy,
    output logic [31:0]                  debug_cycle_count,
    output logic [31:0]                  debug_row_access_count,
    output logic [31:0]                  debug_cache_hit_count,
    output logic [31:0]                  debug_cache_miss_count,
    output logic [31:0]                  debug_weight_beat_count,
    output logic [31:0]                  debug_issue_count,
    output logic [31:0]                  debug_signed_product_count,
    output logic [31:0]                  debug_commit_count
);
    localparam int TOTAL_ITEMS = BUNDLE * SOURCE_GROUPS;
    localparam int ACC24_CONTEXT_BYTES = BUNDLE * OUTPUT_SLICES * LANES * 3;
    localparam int SOURCE_FIFO_BYTES = BUNDLE * SOURCE_GROUPS * SOURCES_PER_GROUP;
    localparam int ROW_CACHE_BYTES = CACHE_ROWS * SOURCES_PER_GROUP
        * OUTPUT_SLICES * LANES;
    localparam int M1763_B8_INCREMENTAL_STATE_LOWER_BOUND_BYTES = 2128;
    localparam bit PARAMETERS_LEGAL = (SCHEDULE_MODE == 0 || SCHEDULE_MODE == 1)
        && BUNDLE == 8 && SOURCES_PER_GROUP == 16 && OUTPUT_SLICES <= 6
        && BANKS == 8 && LANES == 16 && CACHE_ROWS == 8
        && SOURCE_GROUPS > CACHE_ROWS && SOURCE_GROUPS <= (1 << GROUP_BITS)
        && ACC24_CONTEXT_BYTES >= 2304 && SOURCE_FIFO_BYTES >= 128
        && ROW_CACHE_BYTES >= 12288;

    typedef enum logic [3:0] {
        ST_LOAD, ST_FIND, ST_MEM_REQ, ST_MEM_RSP, ST_ISSUE,
        ST_COMMIT, ST_DONE, ST_FAULT
    } state_t;
    state_t state_q;

    logic fault_q, overflow_q;
    logic [2:0] expected_context_q;
    logic first_descriptor_q;
    logic [GROUP_BITS-1:0] last_group_q;
    logic [TAG_BITS-1:0] context_tag_q [0:BUNDLE-1];
    logic active_q [0:BUNDLE-1][0:SOURCE_GROUPS-1];
    logic signed [7:0] source_value_q
        [0:BUNDLE-1][0:SOURCE_GROUPS-1][0:SOURCES_PER_GROUP-1];
    logic signed [23:0] acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1];

    // The data store is deliberately identical in both modes.  Only valid,
    // tag and age require reset; invalid row data is never observed.
    logic cache_valid_q [0:CACHE_ROWS-1];
    logic [GROUP_BITS-1:0] cache_group_q [0:CACHE_ROWS-1];
    logic [31:0] cache_age_q [0:CACHE_ROWS-1];
    logic [31:0] access_clock_q;
    logic signed [7:0] cache_weight_q
        [0:CACHE_ROWS-1][0:1][0:OUTPUT_SLICES-1][0:BANKS-1][0:LANES-1];

    logic [$clog2(TOTAL_ITEMS+1)-1:0] scan_linear_q;
    logic [2:0] current_context_q;
    logic [GROUP_BITS-1:0] current_group_q;
    logic [2:0] current_cache_q;
    logic current_half_q;
    logic [2:0] current_slice_q;
    logic [2:0] fill_cache_q;
    logic fill_half_q;
    logic [2:0] fill_slice_q;
    logic [2:0] commit_context_q;
    logic [2:0] commit_slice_q;

    logic [31:0] cycle_count_q, row_access_count_q;
    logic [31:0] cache_hit_count_q, cache_miss_count_q;
    logic [31:0] weight_beat_count_q, issue_count_q;
    logic [31:0] signed_product_count_q, commit_count_q;

    logic load_payload_legal, load_any_nonzero;
    logic find_valid;
    logic [2:0] find_context;
    logic [GROUP_BITS-1:0] find_group;
    logic [$clog2(TOTAL_ITEMS+1)-1:0] find_linear;
    logic find_cache_hit;
    logic [2:0] find_cache_index, find_victim_index;
    logic signed [24:0] issue_next_acc [0:LANES-1];
    logic issue_overflow [0:LANES-1];
    logic [3:0] issue_source_count;
    logic current_half_active, other_half_active;
    logic illegal_rsp;

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M1780 legal point is B8/G<=64/S16/O<=6/K8/L16/LRU8");
        end
    endgenerate

    function automatic logic source_value_legal(input logic signed [7:0] value);
        source_value_legal = (value == -8'sd1) || (value == 8'sd0)
            || (value == 8'sd1);
    endfunction

    function automatic logic half_has_source(
        input logic [2:0] context,
        input logic [GROUP_BITS-1:0] group,
        input logic half);
        logic found;
        begin
            found = 0;
            for (int bank = 0; bank < BANKS; bank++)
                if (source_value_q[context][group][bank + (half ? 8 : 0)] != 0)
                    found = 1;
            half_has_source = found;
        end
    endfunction

    always_comb begin : load_checks
        load_any_nonzero = 0;
        load_payload_legal = state_q == ST_LOAD && PARAMETERS_LEGAL
            && load_context == expected_context_q
            && load_context < BUNDLE && load_group < SOURCE_GROUPS;
        if (!first_descriptor_q && load_group <= last_group_q)
            load_payload_legal = 0;
        for (int source = 0; source < SOURCES_PER_GROUP; source++) begin
            if (!source_value_legal(load_source_value[source]))
                load_payload_legal = 0;
            if (load_source_value[source] != 0) load_any_nonzero = 1;
        end
        if (!first_descriptor_q
                && load_tag != context_tag_q[expected_context_q])
            load_payload_legal = 0;
        load_ready = state_q == ST_LOAD && !fault_q;
        load_accept = load_valid && load_ready && load_payload_legal;
    end

    // Find the next active (context,group) in one of two orders.  Both modes
    // retain the same B8 buffers; this loop is the only ordering difference.
    always_comb begin : next_active
        int raw;
        int context_index;
        int group_index;
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
                if (active_q[context_index][group_index]) begin
                    find_valid = 1;
                    find_context = context_index[2:0];
                    find_group = group_index[GROUP_BITS-1:0];
                    find_linear = raw + 1;
                end
            end
        end
    end

    always_comb begin : cache_lookup_and_victim
        logic invalid_found;
        logic [31:0] oldest_age;
        find_cache_hit = 0;
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

    assign mem_req_valid = state_q == ST_MEM_REQ && !fault_q;
    assign mem_req_group = current_group_q;
    assign mem_req_half = fill_half_q;
    assign mem_req_slice = fill_slice_q;
    assign mem_req_accept = mem_req_valid && mem_req_ready;
    assign mem_rsp_ready = state_q == ST_MEM_RSP && !fault_q;
    assign mem_rsp_accept = mem_rsp_valid && mem_rsp_ready && !illegal_rsp;
    assign illegal_rsp = mem_rsp_valid && mem_rsp_ready
        && (mem_rsp_group != current_group_q || mem_rsp_half != fill_half_q
            || mem_rsp_slice != fill_slice_q);

    always_comb begin : issue_view
        logic signed [24:0] delta;
        logic signed [24:0] extended_acc;
        issue_valid = state_q == ST_ISSUE && !fault_q;
        issue_context = current_context_q;
        issue_group = current_group_q;
        issue_half = current_half_q;
        issue_slice = current_slice_q;
        issue_bank_valid = 0;
        issue_source_count = 0;
        for (int bank = 0; bank < BANKS; bank++) begin
            issue_source_channel[bank] = {current_group_q, 4'b0000}
                + (current_half_q ? 8 : 0) + bank;
            issue_source_value[bank] = source_value_q[current_context_q]
                [current_group_q][bank + (current_half_q ? 8 : 0)];
            issue_bank_valid[bank] = issue_source_value[bank] != 0;
            if (issue_source_value[bank] != 0)
                issue_source_count = issue_source_count + 1'b1;
            for (int lane = 0; lane < LANES; lane++)
                issue_weight[bank][lane] = cache_weight_q[current_cache_q]
                    [current_half_q][current_slice_q][bank][lane];
        end
        issue_accept = issue_valid && issue_ready;
        for (int lane = 0; lane < LANES; lane++) begin
            delta = 0;
            for (int bank = 0; bank < BANKS; bank++) begin
                if (issue_source_value[bank] == 8'sd1)
                    delta = delta + {{17{issue_weight[bank][lane][7]}},
                                     issue_weight[bank][lane]};
                else if (issue_source_value[bank] == -8'sd1)
                    delta = delta - {{17{issue_weight[bank][lane][7]}},
                                     issue_weight[bank][lane]};
            end
            extended_acc = {acc_q[current_context_q][current_slice_q][lane][23],
                            acc_q[current_context_q][current_slice_q][lane]};
            issue_next_acc[lane] = extended_acc + delta;
            issue_overflow[lane] = issue_next_acc[lane][24]
                != issue_next_acc[lane][23];
        end
        current_half_active = half_has_source(
            current_context_q, current_group_q, current_half_q);
        other_half_active = half_has_source(
            current_context_q, current_group_q, !current_half_q);
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

    assign protocol_error = fault_q;
    assign numeric_overflow = overflow_q;
    assign busy = state_q != ST_LOAD || expected_context_q != 0;
    assign debug_cycle_count = cycle_count_q;
    assign debug_row_access_count = row_access_count_q;
    assign debug_cache_hit_count = cache_hit_count_q;
    assign debug_cache_miss_count = cache_miss_count_q;
    assign debug_weight_beat_count = weight_beat_count_q;
    assign debug_issue_count = issue_count_q;
    assign debug_signed_product_count = signed_product_count_q;
    assign debug_commit_count = commit_count_q;

    always_ff @(posedge clk_core) begin : state_and_ledger
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
            access_clock_q <= 1;
            cycle_count_q <= 0;
            row_access_count_q <= 0;
            cache_hit_count_q <= 0;
            cache_miss_count_q <= 0;
            weight_beat_count_q <= 0;
            issue_count_q <= 0;
            signed_product_count_q <= 0;
            commit_count_q <= 0;
            for (int context = 0; context < BUNDLE; context++) begin
                context_tag_q[context] <= 0;
                for (int group = 0; group < SOURCE_GROUPS; group++) begin
                    active_q[context][group] <= 0;
                    for (int source = 0; source < SOURCES_PER_GROUP; source++)
                        source_value_q[context][group][source] <= 0;
                end
                for (int slice = 0; slice < OUTPUT_SLICES; slice++)
                    for (int lane = 0; lane < LANES; lane++)
                        acc_q[context][slice][lane] <= 0;
            end
            for (int entry = 0; entry < CACHE_ROWS; entry++) begin
                cache_valid_q[entry] <= 0;
                cache_group_q[entry] <= 0;
                cache_age_q[entry] <= 0;
            end
        end else begin
            if (state_q != ST_LOAD && state_q != ST_DONE && state_q != ST_FAULT)
                cycle_count_q <= cycle_count_q + 1'b1;
            if ((load_valid && load_ready && !load_payload_legal) || illegal_rsp) begin
                fault_q <= 1;
                state_q <= ST_FAULT;
            end

            if (load_accept) begin
                if (first_descriptor_q) context_tag_q[load_context] <= load_tag;
                if (load_any_nonzero) begin
                    active_q[load_context][load_group] <= 1;
                    for (int source = 0; source < SOURCES_PER_GROUP; source++)
                        source_value_q[load_context][load_group][source]
                            <= load_source_value[source];
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
                        state_q <= ST_ISSUE;
                    end else begin
                        fill_cache_q <= find_victim_index;
                        current_cache_q <= find_victim_index;
                        cache_valid_q[find_victim_index] <= 0;
                        cache_group_q[find_victim_index] <= find_group;
                        fill_half_q <= 0;
                        fill_slice_q <= 0;
                        cache_miss_count_q <= cache_miss_count_q + 1'b1;
                        state_q <= ST_MEM_REQ;
                    end
                end
            end

            if (state_q == ST_MEM_REQ && mem_req_accept)
                state_q <= ST_MEM_RSP;

            if (state_q == ST_MEM_RSP && mem_rsp_accept) begin
                for (int bank = 0; bank < BANKS; bank++)
                    for (int lane = 0; lane < LANES; lane++)
                        cache_weight_q[fill_cache_q][fill_half_q][fill_slice_q]
                            [bank][lane] <= mem_rsp_weight[bank][lane];
                weight_beat_count_q <= weight_beat_count_q + 1'b1;
                if (fill_half_q && fill_slice_q == OUTPUT_SLICES - 1) begin
                    cache_valid_q[fill_cache_q] <= 1;
                    cache_age_q[fill_cache_q] <= access_clock_q;
                    current_half_q <= half_has_source(
                        current_context_q, current_group_q, 0) ? 1'b0 : 1'b1;
                    current_slice_q <= 0;
                    state_q <= ST_ISSUE;
                end else begin
                    if (fill_slice_q == OUTPUT_SLICES - 1) begin
                        fill_slice_q <= 0;
                        fill_half_q <= 1;
                    end else begin
                        fill_slice_q <= fill_slice_q + 1'b1;
                    end
                    state_q <= ST_MEM_REQ;
                end
            end

            if (state_q == ST_ISSUE && issue_accept) begin
                issue_count_q <= issue_count_q + 1'b1;
                signed_product_count_q <= signed_product_count_q
                    + issue_source_count * LANES;
                for (int lane = 0; lane < LANES; lane++) begin
                    acc_q[current_context_q][current_slice_q][lane]
                        <= issue_next_acc[lane][23:0];
                    if (issue_overflow[lane]) overflow_q <= 1;
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
                    if (commit_context_q == BUNDLE - 1) begin
                        state_q <= ST_DONE;
                    end else begin
                        commit_context_q <= commit_context_q + 1'b1;
                    end
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
                for (int context = 0; context < BUNDLE; context++) begin
                    context_tag_q[context] <= 0;
                    for (int group = 0; group < SOURCE_GROUPS; group++) begin
                        active_q[context][group] <= 0;
                        for (int source = 0; source < SOURCES_PER_GROUP; source++)
                            source_value_q[context][group][source] <= 0;
                    end
                    for (int slice = 0; slice < OUTPUT_SLICES; slice++)
                        for (int lane = 0; lane < LANES; lane++)
                            acc_q[context][slice][lane] <= 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
