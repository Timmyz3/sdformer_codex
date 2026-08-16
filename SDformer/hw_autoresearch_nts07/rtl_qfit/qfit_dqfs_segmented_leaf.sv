`timescale 1ns/1ps
`default_nettype none

module qfit_dqfs_segmented_leaf #(
    parameter int CONTEXTS = 2,
    parameter int LANES = 4,
    parameter int WAYS = 2,
    parameter int TERM_CAPACITY = 16,
    parameter int ROW_ID_W = 4,
    parameter int EPOCH_W = 4,
    parameter int TILE_W = 4,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int GATE_W = 9,
    parameter int DEST_MASK_W = 5
) (
    input  logic clk_core,
    input  logic rst_core,

    input  logic txn_start_valid,
    output logic txn_start_ready,
    input  logic [EPOCH_W-1:0] txn_epoch,
    input  logic [TILE_W-1:0] txn_output_tile,
    input  logic txn_close_valid,
    output logic txn_close_ready,
    output logic txn_done,

    input  logic in_valid,
    output logic in_ready,
    input  logic [ROW_ID_W-1:0] in_row_id,
    input  logic in_row_last,
    input  logic in_window_last,
    input  logic [$clog2(LANES)-1:0] in_lane,
    input  logic [GATE_W-1:0] in_gate,
    input  logic [PLANE_W-1:0] in_source_plane,
    input  logic [Y_W-1:0] in_source_y,
    input  logic [X_W-1:0] in_source_x,
    input  logic [DEST_MASK_W-1:0] in_destination_mask,

    output logic group_valid,
    input  logic group_ready,
    output logic [$clog2(LANES)-1:0] group_lane,
    output logic [GATE_W-1:0] group_gate,
    output logic [EPOCH_W-1:0] group_epoch,
    output logic [TILE_W-1:0] group_output_tile,
    output logic [$clog2(TERM_CAPACITY+1)-1:0] group_member_count,

    output logic member_valid,
    input  logic member_ready,
    output logic [PLANE_W-1:0] member_source_plane,
    output logic [Y_W-1:0] member_source_y,
    output logic [X_W-1:0] member_source_x,
    output logic [DEST_MASK_W-1:0] member_destination_mask,
    output logic member_group_last,
    output logic member_row_last,
    output logic member_window_last,

    output logic protocol_error,
    output logic [31:0] perf_accepted_terms,
    output logic [31:0] perf_emitted_members,
    output logic [31:0] perf_emitted_groups,
    output logic [31:0] perf_capacity_seals,
    output logic [31:0] perf_way_seals,
    output logic [31:0] perf_input_stalls
);
    localparam int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS);
    localparam int LANE_W = (LANES <= 1) ? 1 : $clog2(LANES);
    localparam int WAY_W = (WAYS <= 1) ? 1 : $clog2(WAYS);
    localparam int PTR_W = (TERM_CAPACITY <= 1)
        ? 1
        : $clog2(TERM_CAPACITY);
    localparam int COUNT_W = $clog2(TERM_CAPACITY + 1);

    typedef enum logic [2:0] {
        CTX_FREE = 3'd0,
        CTX_COLLECT = 3'd1,
        CTX_SEALED = 3'd2,
        CTX_EMIT = 3'd3
    } ctx_state_t;

    typedef enum logic [1:0] {
        EMIT_IDLE = 2'd0,
        EMIT_GROUP = 2'd1,
        EMIT_WAIT = 2'd2,
        EMIT_MEMBER = 2'd3
    } emit_state_t;

    ctx_state_t ctx_state_q [0:CONTEXTS-1];
    logic [ROW_ID_W-1:0] ctx_row_q [0:CONTEXTS-1];
    logic [EPOCH_W-1:0] ctx_epoch_q [0:CONTEXTS-1];
    logic [TILE_W-1:0] ctx_tile_q [0:CONTEXTS-1];
    logic ctx_row_last_q [0:CONTEXTS-1];
    logic ctx_window_last_q [0:CONTEXTS-1];
    logic [COUNT_W-1:0] ctx_term_count_q [0:CONTEXTS-1];
    logic [COUNT_W-1:0] ctx_group_count_q [0:CONTEXTS-1];
    logic [31:0] ctx_seal_seq_q [0:CONTEXTS-1];

    logic dir_valid_q [0:CONTEXTS-1][0:LANES-1][0:WAYS-1];
    logic [GATE_W-1:0] dir_gate_q
        [0:CONTEXTS-1][0:LANES-1][0:WAYS-1];
    logic [PTR_W-1:0] dir_tail_q
        [0:CONTEXTS-1][0:LANES-1][0:WAYS-1];
    logic [COUNT_W-1:0] dir_count_q
        [0:CONTEXTS-1][0:LANES-1][0:WAYS-1];

    logic [PLANE_W-1:0] term_plane_q
        [0:CONTEXTS-1][0:TERM_CAPACITY-1];
    logic [Y_W-1:0] term_y_q
        [0:CONTEXTS-1][0:TERM_CAPACITY-1];
    logic [X_W-1:0] term_x_q
        [0:CONTEXTS-1][0:TERM_CAPACITY-1];
    logic [DEST_MASK_W-1:0] term_mask_q
        [0:CONTEXTS-1][0:TERM_CAPACITY-1];
    logic [PTR_W-1:0] term_prev_q
        [0:CONTEXTS-1][0:TERM_CAPACITY-1];
    logic term_prev_valid_q [0:CONTEXTS-1][0:TERM_CAPACITY-1];

    logic txn_active_q;
    logic closing_q;
    logic [EPOCH_W-1:0] active_epoch_q;
    logic [TILE_W-1:0] active_tile_q;
    logic protocol_error_q;
    logic [31:0] seal_seq_q;

    logic [31:0] perf_accepted_q;
    logic [31:0] perf_members_q;
    logic [31:0] perf_groups_q;
    logic [31:0] perf_capacity_q;
    logic [31:0] perf_way_q;
    logic [31:0] perf_stalls_q;

    logic input_contract_valid;
    logic match_ctx_found;
    logic free_ctx_found;
    logic selected_ctx_found;
    logic [CTX_W-1:0] match_ctx;
    logic [CTX_W-1:0] free_ctx;
    logic [CTX_W-1:0] selected_ctx;
    logic selected_is_free;
    logic dir_match_found;
    logic dir_free_found;
    logic [WAY_W-1:0] dir_match_way;
    logic [WAY_W-1:0] dir_free_way;
    logic selected_capacity_full;
    logic selected_way_full;
    logic seal_request;
    logic seal_due_capacity;
    logic seal_due_way;
    logic [CTX_W-1:0] seal_ctx;
    logic in_fire;
    logic txn_start_fire;
    logic txn_close_fire;

    emit_state_t emit_state_q;
    logic [CTX_W-1:0] emit_ctx_q;
    logic [LANE_W-1:0] active_lane_q;
    logic [WAY_W-1:0] active_way_q;
    logic [COUNT_W-1:0] members_left_q;
    logic [PTR_W-1:0] read_addr_q;
    logic read_pending_q;
    logic member_valid_q;
    logic [PLANE_W-1:0] member_plane_q;
    logic [Y_W-1:0] member_y_q;
    logic [X_W-1:0] member_x_q;
    logic [DEST_MASK_W-1:0] member_mask_q;
    logic [PTR_W-1:0] member_prev_q;
    logic member_prev_valid_q;

    logic sealed_found;
    logic [CTX_W-1:0] sealed_ctx;
    logic [31:0] sealed_seq_best;
    logic scan_group_found;
    logic [LANE_W-1:0] scan_lane;
    logic [WAY_W-1:0] scan_way;
    logic all_contexts_free;
    logic group_fire;
    logic member_fire;

    always_comb begin
        input_contract_valid =
            32'(in_lane) < LANES
            && in_gate != '0
            && in_destination_mask != '0;

        match_ctx_found = 1'b0;
        free_ctx_found = 1'b0;
        match_ctx = '0;
        free_ctx = '0;
        for (integer ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            if (
                !match_ctx_found
                && ctx_state_q[ctx] == CTX_COLLECT
                && ctx_row_q[ctx] == in_row_id
            ) begin
                match_ctx_found = 1'b1;
                match_ctx = CTX_W'(ctx);
            end
            if (
                !free_ctx_found
                && ctx_state_q[ctx] == CTX_FREE
            ) begin
                free_ctx_found = 1'b1;
                free_ctx = CTX_W'(ctx);
            end
        end
        selected_ctx_found = match_ctx_found || free_ctx_found;
        selected_is_free = !match_ctx_found && free_ctx_found;
        selected_ctx = match_ctx_found ? match_ctx : free_ctx;

        dir_match_found = 1'b0;
        dir_free_found = 1'b0;
        dir_match_way = '0;
        dir_free_way = '0;
        if (
            selected_ctx_found
            && !selected_is_free
            && 32'(in_lane) < LANES
        ) begin
            for (integer way = 0; way < WAYS; way = way + 1) begin
                if (
                    !dir_match_found
                    && dir_valid_q[selected_ctx][in_lane][way]
                    && dir_gate_q[selected_ctx][in_lane][way] == in_gate
                ) begin
                    dir_match_found = 1'b1;
                    dir_match_way = WAY_W'(way);
                end
                if (
                    !dir_free_found
                    && !dir_valid_q[selected_ctx][in_lane][way]
                ) begin
                    dir_free_found = 1'b1;
                    dir_free_way = WAY_W'(way);
                end
            end
        end else if (selected_ctx_found) begin
            dir_free_found = 1'b1;
            dir_free_way = '0;
        end

        selected_capacity_full =
            selected_ctx_found
            && !selected_is_free
            && ctx_term_count_q[selected_ctx] == COUNT_W'(TERM_CAPACITY);
        selected_way_full =
            selected_ctx_found
            && !selected_is_free
            && !dir_match_found
            && !dir_free_found;

        in_ready = 1'b0;
        if (txn_active_q && !closing_q) begin
            if (!input_contract_valid)
                in_ready = 1'b1;
            else if (
                selected_ctx_found
                && !selected_capacity_full
                && !selected_way_full
            )
                in_ready = 1'b1;
        end

        seal_request = 1'b0;
        seal_due_capacity = 1'b0;
        seal_due_way = 1'b0;
        seal_ctx = '0;
        if (
            in_valid
            && txn_active_q
            && !closing_q
            && input_contract_valid
            && !in_ready
        ) begin
            if (selected_ctx_found && selected_capacity_full) begin
                seal_request = 1'b1;
                seal_due_capacity = 1'b1;
                seal_ctx = selected_ctx;
            end else if (selected_ctx_found && selected_way_full) begin
                seal_request = 1'b1;
                seal_due_way = 1'b1;
                seal_ctx = selected_ctx;
            end else begin
                for (
                    integer ctx = 0;
                    ctx < CONTEXTS;
                    ctx = ctx + 1
                ) begin
                    if (
                        !seal_request
                        && ctx_state_q[ctx] == CTX_COLLECT
                    ) begin
                        seal_request = 1'b1;
                        seal_ctx = CTX_W'(ctx);
                    end
                end
            end
        end
    end

    always_comb begin
        sealed_found = 1'b0;
        sealed_ctx = '0;
        sealed_seq_best = '1;
        for (integer ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
            if (
                ctx_state_q[ctx] == CTX_SEALED
                && !ctx_window_last_q[ctx]
                && (
                    !sealed_found
                    || ctx_seal_seq_q[ctx] < sealed_seq_best
                )
            ) begin
                sealed_found = 1'b1;
                sealed_ctx = CTX_W'(ctx);
                sealed_seq_best = ctx_seal_seq_q[ctx];
            end
        end
        if (!sealed_found) begin
            sealed_seq_best = '1;
            for (
                integer ctx = 0;
                ctx < CONTEXTS;
                ctx = ctx + 1
            ) begin
                if (
                    ctx_state_q[ctx] == CTX_SEALED
                    && (
                        !sealed_found
                        || ctx_seal_seq_q[ctx] < sealed_seq_best
                    )
                ) begin
                    sealed_found = 1'b1;
                    sealed_ctx = CTX_W'(ctx);
                    sealed_seq_best = ctx_seal_seq_q[ctx];
                end
            end
        end

        scan_group_found = 1'b0;
        scan_lane = '0;
        scan_way = '0;
        if (emit_state_q == EMIT_GROUP) begin
            for (integer lane = 0; lane < LANES; lane = lane + 1) begin
                for (integer way = 0; way < WAYS; way = way + 1) begin
                    if (
                        !scan_group_found
                        && dir_valid_q[emit_ctx_q][lane][way]
                    ) begin
                        scan_group_found = 1'b1;
                        scan_lane = LANE_W'(lane);
                        scan_way = WAY_W'(way);
                    end
                end
            end
        end
    end

    always_comb begin
        all_contexts_free = 1'b1;
        for (integer ctx = 0; ctx < CONTEXTS; ctx = ctx + 1)
            if (ctx_state_q[ctx] != CTX_FREE)
                all_contexts_free = 1'b0;
    end

    assign txn_start_ready =
        !txn_active_q
        && all_contexts_free
        && emit_state_q == EMIT_IDLE
        && !member_valid_q
        && !read_pending_q;
    assign txn_start_fire = txn_start_valid && txn_start_ready;
    assign txn_close_ready = txn_active_q && !closing_q && !in_valid;
    assign txn_close_fire = txn_close_valid && txn_close_ready;
    assign in_fire = in_valid && in_ready;

    assign group_valid =
        emit_state_q == EMIT_GROUP && scan_group_found;
    assign group_lane = scan_lane;
    assign group_gate =
        dir_gate_q[emit_ctx_q][scan_lane][scan_way];
    assign group_epoch = ctx_epoch_q[emit_ctx_q];
    assign group_output_tile = ctx_tile_q[emit_ctx_q];
    assign group_member_count =
        dir_count_q[emit_ctx_q][scan_lane][scan_way];
    assign group_fire = group_valid && group_ready;

    assign member_valid = member_valid_q;
    assign member_source_plane = member_plane_q;
    assign member_source_y = member_y_q;
    assign member_source_x = member_x_q;
    assign member_destination_mask = member_mask_q;
    assign member_group_last = members_left_q == COUNT_W'(1);
    assign member_row_last =
        member_group_last
        && ctx_group_count_q[emit_ctx_q] == COUNT_W'(1)
        && ctx_row_last_q[emit_ctx_q];
    assign member_window_last =
        member_group_last
        && ctx_group_count_q[emit_ctx_q] == COUNT_W'(1)
        && ctx_window_last_q[emit_ctx_q];
    assign member_fire = member_valid && member_ready;

    assign protocol_error = protocol_error_q;
    assign perf_accepted_terms = perf_accepted_q;
    assign perf_emitted_members = perf_members_q;
    assign perf_emitted_groups = perf_groups_q;
    assign perf_capacity_seals = perf_capacity_q;
    assign perf_way_seals = perf_way_q;
    assign perf_input_stalls = perf_stalls_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            txn_active_q <= 1'b0;
            closing_q <= 1'b0;
            active_epoch_q <= '0;
            active_tile_q <= '0;
            protocol_error_q <= 1'b0;
            seal_seq_q <= '0;
            txn_done <= 1'b0;
            perf_accepted_q <= '0;
            perf_members_q <= '0;
            perf_groups_q <= '0;
            perf_capacity_q <= '0;
            perf_way_q <= '0;
            perf_stalls_q <= '0;
            emit_state_q <= EMIT_IDLE;
            emit_ctx_q <= '0;
            active_lane_q <= '0;
            active_way_q <= '0;
            members_left_q <= '0;
            read_addr_q <= '0;
            read_pending_q <= 1'b0;
            member_valid_q <= 1'b0;
            member_plane_q <= '0;
            member_y_q <= '0;
            member_x_q <= '0;
            member_mask_q <= '0;
            member_prev_q <= '0;
            member_prev_valid_q <= 1'b0;
            for (
                integer ctx = 0;
                ctx < CONTEXTS;
                ctx = ctx + 1
            ) begin
                ctx_state_q[ctx] <= CTX_FREE;
                ctx_row_q[ctx] <= '0;
                ctx_epoch_q[ctx] <= '0;
                ctx_tile_q[ctx] <= '0;
                ctx_row_last_q[ctx] <= 1'b0;
                ctx_window_last_q[ctx] <= 1'b0;
                ctx_term_count_q[ctx] <= '0;
                ctx_group_count_q[ctx] <= '0;
                ctx_seal_seq_q[ctx] <= '0;
                for (integer lane = 0; lane < LANES; lane = lane + 1) begin
                    for (integer way = 0; way < WAYS; way = way + 1) begin
                        dir_valid_q[ctx][lane][way] <= 1'b0;
                        dir_gate_q[ctx][lane][way] <= '0;
                        dir_tail_q[ctx][lane][way] <= '0;
                        dir_count_q[ctx][lane][way] <= '0;
                    end
                end
            end
        end else begin
            txn_done <= 1'b0;

            if (txn_start_fire) begin
                txn_active_q <= 1'b1;
                closing_q <= 1'b0;
                active_epoch_q <= txn_epoch;
                active_tile_q <= txn_output_tile;
                protocol_error_q <= 1'b0;
                seal_seq_q <= '0;
                perf_accepted_q <= '0;
                perf_members_q <= '0;
                perf_groups_q <= '0;
                perf_capacity_q <= '0;
                perf_way_q <= '0;
                perf_stalls_q <= '0;
            end

            if (in_valid && txn_active_q && !closing_q && !in_ready)
                perf_stalls_q <= perf_stalls_q + 1'b1;

            if (in_fire && !input_contract_valid) begin
                protocol_error_q <= 1'b1;
                closing_q <= 1'b1;
                seal_seq_q <= seal_seq_q + 32'(CONTEXTS);
                for (
                    integer ctx = 0;
                    ctx < CONTEXTS;
                    ctx = ctx + 1
                ) begin
                    if (ctx_state_q[ctx] == CTX_COLLECT) begin
                        ctx_state_q[ctx] <= CTX_SEALED;
                        ctx_seal_seq_q[ctx] <=
                            seal_seq_q + 32'(ctx);
                    end
                end
            end else if (in_fire) begin
                logic [WAY_W-1:0] append_way;
                logic [PTR_W-1:0] append_addr;
                append_way = dir_match_found
                    ? dir_match_way
                    : dir_free_way;
                append_addr = selected_is_free
                    ? '0
                    : PTR_W'(ctx_term_count_q[selected_ctx]);

                if (selected_is_free) begin
                    ctx_state_q[selected_ctx] <= CTX_COLLECT;
                    ctx_row_q[selected_ctx] <= in_row_id;
                    ctx_epoch_q[selected_ctx] <= active_epoch_q;
                    ctx_tile_q[selected_ctx] <= active_tile_q;
                    ctx_row_last_q[selected_ctx] <= 1'b0;
                    ctx_window_last_q[selected_ctx] <= 1'b0;
                    ctx_term_count_q[selected_ctx] <= COUNT_W'(1);
                    ctx_group_count_q[selected_ctx] <= COUNT_W'(1);
                    for (
                        integer lane = 0;
                        lane < LANES;
                        lane = lane + 1
                    ) begin
                        for (
                            integer way = 0;
                            way < WAYS;
                            way = way + 1
                        ) begin
                            dir_valid_q[selected_ctx][lane][way] <= 1'b0;
                            dir_count_q[selected_ctx][lane][way] <= '0;
                        end
                    end
                end else begin
                    ctx_term_count_q[selected_ctx] <=
                        ctx_term_count_q[selected_ctx] + 1'b1;
                    if (!dir_match_found)
                        ctx_group_count_q[selected_ctx] <=
                            ctx_group_count_q[selected_ctx] + 1'b1;
                end

                term_plane_q[selected_ctx][append_addr] <= in_source_plane;
                term_y_q[selected_ctx][append_addr] <= in_source_y;
                term_x_q[selected_ctx][append_addr] <= in_source_x;
                term_mask_q[selected_ctx][append_addr] <=
                    in_destination_mask;
                term_prev_valid_q[selected_ctx][append_addr] <=
                    !selected_is_free
                    && dir_match_found;
                term_prev_q[selected_ctx][append_addr] <=
                    dir_tail_q[selected_ctx][in_lane][append_way];

                dir_valid_q[selected_ctx][in_lane][append_way] <= 1'b1;
                dir_gate_q[selected_ctx][in_lane][append_way] <= in_gate;
                dir_tail_q[selected_ctx][in_lane][append_way] <= append_addr;
                dir_count_q[selected_ctx][in_lane][append_way] <=
                    (selected_is_free || !dir_match_found)
                    ? COUNT_W'(1)
                    : dir_count_q[selected_ctx][in_lane][append_way] + 1'b1;
                perf_accepted_q <= perf_accepted_q + 1'b1;

                if (in_row_last) begin
                    ctx_row_last_q[selected_ctx] <= 1'b1;
                    ctx_state_q[selected_ctx] <= CTX_SEALED;
                    ctx_seal_seq_q[selected_ctx] <= seal_seq_q;
                    seal_seq_q <= seal_seq_q + 1'b1;
                end
                if (in_window_last) begin
                    ctx_window_last_q[selected_ctx] <= 1'b1;
                    ctx_row_last_q[selected_ctx] <= 1'b1;
                    ctx_state_q[selected_ctx] <= CTX_SEALED;
                    ctx_seal_seq_q[selected_ctx] <= seal_seq_q;
                    seal_seq_q <= seal_seq_q + 1'b1;
                    closing_q <= 1'b1;
                    for (
                        integer ctx = 0;
                        ctx < CONTEXTS;
                        ctx = ctx + 1
                    ) begin
                        if (
                            CTX_W'(ctx) != selected_ctx
                            && ctx_state_q[ctx] == CTX_COLLECT
                        ) begin
                            ctx_state_q[ctx] <= CTX_SEALED;
                            ctx_seal_seq_q[ctx] <= seal_seq_q;
                        end
                    end
                end
            end

            if (seal_request) begin
                ctx_state_q[seal_ctx] <= CTX_SEALED;
                ctx_seal_seq_q[seal_ctx] <= seal_seq_q;
                seal_seq_q <= seal_seq_q + 1'b1;
                if (seal_due_capacity)
                    perf_capacity_q <= perf_capacity_q + 1'b1;
                if (seal_due_way)
                    perf_way_q <= perf_way_q + 1'b1;
            end

            if (txn_close_fire) begin
                closing_q <= 1'b1;
                for (
                    integer ctx = 0;
                    ctx < CONTEXTS;
                    ctx = ctx + 1
                ) begin
                    if (ctx_state_q[ctx] == CTX_COLLECT) begin
                        ctx_state_q[ctx] <= CTX_SEALED;
                        ctx_seal_seq_q[ctx] <= seal_seq_q;
                    end
                end
            end

            case (emit_state_q)
                EMIT_IDLE: begin
                    if (sealed_found) begin
                        emit_ctx_q <= sealed_ctx;
                        ctx_state_q[sealed_ctx] <= CTX_EMIT;
                        emit_state_q <= EMIT_GROUP;
                    end
                end

                EMIT_GROUP: begin
                    if (!scan_group_found) begin
                        protocol_error_q <= 1'b1;
                    end else if (group_fire) begin
                        active_lane_q <= scan_lane;
                        active_way_q <= scan_way;
                        members_left_q <=
                            dir_count_q[emit_ctx_q][scan_lane][scan_way];
                        read_addr_q <=
                            dir_tail_q[emit_ctx_q][scan_lane][scan_way];
                        read_pending_q <= 1'b1;
                        perf_groups_q <= perf_groups_q + 1'b1;
                        emit_state_q <= EMIT_WAIT;
                    end
                end

                EMIT_WAIT: begin
                    if (read_pending_q && !member_valid_q) begin
                        member_plane_q <=
                            term_plane_q[emit_ctx_q][read_addr_q];
                        member_y_q <= term_y_q[emit_ctx_q][read_addr_q];
                        member_x_q <= term_x_q[emit_ctx_q][read_addr_q];
                        member_mask_q <=
                            term_mask_q[emit_ctx_q][read_addr_q];
                        member_prev_q <=
                            term_prev_q[emit_ctx_q][read_addr_q];
                        member_prev_valid_q <=
                            term_prev_valid_q[emit_ctx_q][read_addr_q];
                        read_pending_q <= 1'b0;
                        member_valid_q <= 1'b1;
                        emit_state_q <= EMIT_MEMBER;
                    end
                end

                EMIT_MEMBER: begin
                    if (member_fire) begin
                        member_valid_q <= 1'b0;
                        perf_members_q <= perf_members_q + 1'b1;
                        if (members_left_q > COUNT_W'(1)) begin
                            if (!member_prev_valid_q) begin
                                protocol_error_q <= 1'b1;
                            end else begin
                                members_left_q <= members_left_q - 1'b1;
                                read_addr_q <= member_prev_q;
                                read_pending_q <= 1'b1;
                                emit_state_q <= EMIT_WAIT;
                            end
                        end else begin
                            if (member_prev_valid_q)
                                protocol_error_q <= 1'b1;
                            dir_valid_q[
                                emit_ctx_q
                            ][active_lane_q][active_way_q] <= 1'b0;
                            dir_count_q[
                                emit_ctx_q
                            ][active_lane_q][active_way_q] <= '0;
                            if (
                                ctx_group_count_q[emit_ctx_q]
                                == COUNT_W'(1)
                            ) begin
                                ctx_group_count_q[emit_ctx_q] <= '0;
                                ctx_term_count_q[emit_ctx_q] <= '0;
                                ctx_row_last_q[emit_ctx_q] <= 1'b0;
                                ctx_window_last_q[emit_ctx_q] <= 1'b0;
                                ctx_state_q[emit_ctx_q] <= CTX_FREE;
                                emit_state_q <= EMIT_IDLE;
                            end else begin
                                ctx_group_count_q[emit_ctx_q] <=
                                    ctx_group_count_q[emit_ctx_q] - 1'b1;
                                emit_state_q <= EMIT_GROUP;
                            end
                        end
                    end
                end

                default: emit_state_q <= EMIT_IDLE;
            endcase

            if (
                txn_active_q
                && closing_q
                && all_contexts_free
                && emit_state_q == EMIT_IDLE
                && !member_valid_q
                && !read_pending_q
            ) begin
                if (perf_accepted_q != perf_members_q)
                    protocol_error_q <= 1'b1;
                txn_active_q <= 1'b0;
                closing_q <= 1'b0;
                txn_done <= 1'b1;
            end
        end
    end

    initial begin
        if (CONTEXTS != 2)
            $fatal(1, "DQFS proof currently requires CONTEXTS=2");
        if (LANES < 1 || WAYS < 1 || TERM_CAPACITY < 2)
            $fatal(1, "DQFS parameters must be positive");
    end
endmodule

`default_nettype wire
