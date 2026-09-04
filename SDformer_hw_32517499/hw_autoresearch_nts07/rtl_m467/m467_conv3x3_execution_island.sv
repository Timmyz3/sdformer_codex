`timescale 1ns/1ps
`default_nettype none

// Standalone one-context Conv3x3 execution island.  External memories are
// explicit port cuts: descriptor SRAM, PWP/weight payload store and the
// 8*3000*96*19b accumulator.  The eight phase-cache entries are forwarding
// registers, not a replacement for persistent old_psum.
module m467_conv3x3_execution_island #(
    parameter int TAG_BITS = 24,
    parameter int ROWS_PER_PHASE = 3000
) (
    input  logic clk_core,
    input  logic reset_n,

    input  logic config_valid,
    output logic config_ready,
    output logic config_accept,
    input  logic [1:0] config_beat_index,
    input  logic config_commit,
    input  logic [TAG_BITS-1:0] config_tag,
    input  logic [255:0] config_data,
    input  logic config_operator_last_phase,

    input  logic row_valid,
    output logic row_ready,
    output logic row_accept,
    input  logic [11:0] row_id,
    input  logic [15:0] row_original,
    input  logic row_last,

    output logic descriptor_write_valid,
    input  logic descriptor_write_ready,
    output logic [TAG_BITS-1:0] descriptor_write_tag,
    output logic [11:0] descriptor_write_address,
    output logic [47:0] descriptor_write_data,
    output logic descriptor_read_valid,
    input  logic descriptor_read_ready,
    output logic [TAG_BITS-1:0] descriptor_read_tag,
    output logic [11:0] descriptor_read_address,
    input  logic descriptor_response_valid,
    output logic descriptor_response_ready,
    input  logic [TAG_BITS-1:0] descriptor_response_tag,
    input  logic [11:0] descriptor_response_address,
    input  logic [47:0] descriptor_response_data,

    output logic payload_request_valid,
    input  logic payload_request_ready,
    output logic payload_request_pwp,
    output logic [TAG_BITS-1:0] payload_request_tag,
    output logic payload_request_tile,
    output logic [2:0] payload_request_block,
    output logic [4:0] payload_request_center,
    output logic [3:0] payload_request_source,
    output logic payload_request_narrow,
    input  logic payload_response_valid,
    output logic payload_response_ready,
    input  logic payload_response_pwp,
    input  logic [TAG_BITS-1:0] payload_response_tag,
    input  logic payload_response_tile,
    input  logic [2:0] payload_response_block,
    input  logic [4:0] payload_response_center,
    input  logic [3:0] payload_response_source,
    input  logic payload_response_narrow,
    input  logic [767:0] payload_response_low_or_weight,
    input  logic [511:0] payload_response_high,

    output logic accumulator_read_valid,
    input  logic accumulator_read_ready,
    output logic [14:0] accumulator_read_address,
    input  logic accumulator_response_valid,
    output logic accumulator_response_ready,
    input  logic [14:0] accumulator_response_address,
    input  logic [1823:0] accumulator_response_data,
    output logic accumulator_write_valid,
    input  logic accumulator_write_ready,
    output logic [14:0] accumulator_write_address,
    output logic [1823:0] accumulator_write_data,

    output logic commit_valid,
    input  logic commit_ready,
    output logic [14:0] commit_address,
    output logic [1823:0] commit_data,
    output logic commit_last,
    output logic phase_done_valid,
    input  logic phase_done_ready,
    output logic [TAG_BITS-1:0] phase_done_tag,
    output logic [11:0] phase_done_active_rows,

    output logic protocol_error,
    output logic busy,
    output logic [4:0] debug_state,
    output logic [31:0] debug_descriptor_writes,
    output logic [31:0] debug_descriptor_reads,
    output logic [31:0] debug_pwp_requests,
    output logic [31:0] debug_weight_requests,
    output logic [31:0] debug_forward_hits,
    output logic [31:0] debug_accumulator_reads,
    output logic [31:0] debug_commits,
    output logic [31:0] debug_zero_initializations,
    output logic [31:0] debug_zero_commits,
    output logic [31:0] debug_row_live_sets,
    output logic [31:0] debug_row_live_clears,
    output logic [7:0] debug_zero_init_slot_mask,
    output logic debug_row_live_set_event,
    output logic debug_row_live_clear_event,
    output logic debug_forward_event,
    output logic debug_operator_boundary_pending
);
    localparam logic [4:0] ST_CONFIG = 0, ST_ROWS = 1, ST_DWRITE = 2,
        ST_DREQ = 3, ST_DRSP = 4, ST_PWP_REQ = 5, ST_PWP_RSP = 6,
        ST_PWP_OUT = 7, ST_WREQ = 8, ST_WRSP = 9, ST_ACC = 10,
        ST_ARREQ = 11, ST_ARRSP = 12, ST_NEXT = 13, ST_COMMIT = 14,
        ST_RELEASE = 15, ST_DONE = 16, ST_AWRITE = 17,
        ST_CREAD = 18, ST_CRSP = 19, ST_FAULT = 31;

    logic [4:0] state_q;
    logic fault_q;
    logic [TAG_BITS-1:0] tag_q;
    logic [11:0] active_q, desc_index_q, commit_row_q;
    logic operator_last_q;
    logic tile_q;
    logic [2:0] block_q, commit_index_q;
    logic [511:0] centers_w;
    logic [255:0] narrow_bitmap_w;
    logic matcher_result_valid, matcher_result_ready, matcher_result_accept;
    logic [TAG_BITS-1:0] matcher_result_tag;
    logic [11:0] matcher_result_row;
    logic [15:0] matcher_result_original;
    logic [4:0] matcher_result_center, matcher_result_distance;
    logic matcher_result_use, matcher_result_last;
    logic matcher_release_valid, matcher_release_ready, matcher_release_accept;
    logic matcher_protocol_error, matcher_busy;
    logic [11:0] match_row_q;
    logic [15:0] original_q, center_pattern_w;
    logic [4:0] center_q, distance_q;
    logic use_pwp_q;
    logic match_last_q;
    logic [15:0] plus_q, minus_q, correction_mask_q;
    logic [3:0] source_q;
    logic correction_minus_q;
    logic narrow_q;
    logic [1823:0] delta_q;
    logic [1823:0] write_data_q, commit_data_q;
    logic [1823:0] last_data_q [0:7];
    logic [11:0] last_row_q [0:7];
    logic [7:0] last_valid_q;
    // Every admitted descriptor visits all eight slots in the fixed order
    // tile0/block0..3, tile1/block0..3.  Therefore one row-live bit covers all
    // slots, but it is set only after slot7 writes; setting it at slot0 would
    // expose stale external SRAM to slots1..7.  Commit clears only at slot7.
    logic row_live_q [0:ROWS_PER_PHASE-1];
    logic [14:0] current_address_w;
    logic [2:0] current_slot_w;
    logic current_live_w, commit_live_w;
    logic [31:0] dwrites_q, dreads_q, preqs_q, wreqs_q;
    logic [31:0] forward_q, areads_q, commits_q;
    logic [31:0] zero_initializations_q, zero_commits_q;
    logic [31:0] row_live_sets_q, row_live_clears_q;
    logic [7:0] zero_init_slot_mask_q;
    logic operator_boundary_pending_q;

    logic m433_request_valid, m433_request_ready, m433_request_accept;
    logic m433_contribution_valid, m433_contribution_ready;
    logic m433_contribution_accept, m433_protocol_error;
    logic [1151:0] m433_contribution_data;
    logic unused_m433_busy, unused_m433_full;
    logic [31:0] unused32a, unused32b, unused32c, unused32d, unused32e;
    logic unused_match_config_accept, unused_match_row_accept;
    logic unused_match_busybit;
    logic [31:0] unused_match32a, unused_match32b, unused_match32c;
    logic [31:0] unused_match32d, unused_match32e;

    assign current_slot_w = {tile_q, block_q[1:0]};
    assign current_address_w = {current_slot_w, match_row_q};
    assign center_pattern_w = centers_w[center_q*16 +: 16];
    assign current_live_w = row_live_q[match_row_q];
    assign commit_live_w = row_live_q[commit_row_q];

    m414_q32_balanced16_zero_stop_controller #(
        .TAG_BITS(TAG_BITS), .ROWS_PER_PHASE(ROWS_PER_PHASE)
    ) u_matcher (
        .clk_core(clk_core), .reset_n(reset_n),
        .config_valid(config_valid), .config_ready(config_ready),
        .config_accept(unused_match_config_accept),
        .config_beat_index(config_beat_index), .config_commit(config_commit),
        .config_tag(config_tag), .config_data(config_data),
        .phase_release_valid(matcher_release_valid),
        .phase_release_ready(matcher_release_ready),
        .phase_release_accept(matcher_release_accept),
        .row_valid(row_valid), .row_ready(row_ready),
        .row_accept(unused_match_row_accept), .row_id(row_id),
        .row_original(row_original), .row_last(row_last),
        .result_valid(matcher_result_valid),
        .result_ready(matcher_result_ready),
        .result_accept(matcher_result_accept),
        .result_tag(matcher_result_tag), .result_row_id(matcher_result_row),
        .result_original(matcher_result_original),
        .result_center_id(matcher_result_center),
        .result_distance(matcher_result_distance),
        .result_use_pwp(matcher_result_use),
        .result_last(matcher_result_last),
        .configured_centers_q32(centers_w),
        .configured_narrow_bitmap(narrow_bitmap_w),
        .configured_tag(), .configuration_live(),
        .protocol_error(matcher_protocol_error), .busy(matcher_busy),
        .debug_pass1_pending(unused_match_busybit),
        .debug_source_rows(unused_match32a),
        .debug_pass0_tasks(unused_match32b),
        .debug_pass1_tasks(unused_match32c),
        .debug_early_stops(unused_match32d),
        .debug_results(unused_match32e)
    );

    // M433 is the only PWP signed12 decoder.  Its high-side metadata is
    // generated from the checked response; narrow mode zeros that side.
    m433_exact_dualbank_coread_pwp_adapter #(.TAG_BITS(TAG_BITS)) u_pwp (
        .clk_core(clk_core), .reset_n(reset_n),
        .config_reload(matcher_release_valid),
        .request_valid(m433_request_valid), .request_ready(m433_request_ready),
        .request_accept(m433_request_accept),
        .low_tag(tag_q), .low_tile(tile_q), .low_center_id(center_q),
        .low_output_block(block_q), .request_narrow(narrow_q),
        .low_data(payload_response_low_or_weight),
        .high_tag(narrow_q ? '0 : tag_q),
        .high_tile(narrow_q ? 1'b0 : tile_q),
        .high_center_id(narrow_q ? '0 : center_q),
        .high_output_block(narrow_q ? '0 : block_q),
        .high_data(payload_response_high),
        .contribution_valid(m433_contribution_valid),
        .contribution_ready(m433_contribution_ready),
        .contribution_accept(m433_contribution_accept),
        .contribution_tag(), .contribution_tile(),
        .contribution_center_id(), .contribution_output_block(),
        .contribution_narrow(), .contribution_data(m433_contribution_data),
        .protocol_error(m433_protocol_error), .busy(unused_m433_busy),
        .debug_output_full(unused_m433_full),
        .debug_request_accepts(unused32a), .debug_narrow_accepts(unused32b),
        .debug_wide_accepts(unused32c), .debug_contributions(unused32d),
        .debug_protocol_faults(unused32e)
    );

    assign config_accept = config_valid && config_ready;
    assign row_accept = row_valid && row_ready;
    assign matcher_result_ready = (state_q == ST_ROWS &&
        matcher_result_original == 0) || state_q == ST_DWRITE;
    assign matcher_release_valid = state_q == ST_RELEASE;

    assign descriptor_write_valid = !fault_q && state_q == ST_DWRITE;
    assign descriptor_write_tag = tag_q;
    assign descriptor_write_address = active_q;
    assign descriptor_write_data = {7'b0, use_pwp_q, distance_q, 2'b0, center_q,
        original_q, match_row_q};
    assign descriptor_read_valid = !fault_q && state_q == ST_DREQ;
    assign descriptor_read_tag = tag_q;
    assign descriptor_read_address = desc_index_q;
    assign descriptor_response_ready = !fault_q && state_q == ST_DRSP;

    assign payload_request_valid = !fault_q &&
        (state_q == ST_PWP_REQ || state_q == ST_WREQ);
    assign payload_request_pwp = state_q == ST_PWP_REQ;
    assign payload_request_tag = tag_q;
    assign payload_request_tile = tile_q;
    assign payload_request_block = block_q;
    assign payload_request_center = center_q;
    assign payload_request_source = first_set(correction_mask_q);
    assign payload_request_narrow = narrow_q;
    assign payload_response_ready = !fault_q &&
        (state_q == ST_PWP_RSP || state_q == ST_WRSP);
    assign m433_request_valid = state_q == ST_PWP_RSP &&
        payload_response_valid;
    assign m433_contribution_ready = state_q == ST_PWP_OUT;

    assign accumulator_read_valid = !fault_q &&
        ((state_q == ST_ARREQ && current_live_w) ||
         (state_q == ST_CREAD && commit_live_w));
    assign accumulator_read_address = state_q == ST_CREAD ?
        {commit_index_q,commit_row_q} : current_address_w;
    assign accumulator_response_ready = !fault_q && (state_q == ST_ARRSP || state_q == ST_CRSP);
    assign accumulator_write_valid = !fault_q && state_q == ST_AWRITE;
    assign accumulator_write_address = current_address_w;
    assign accumulator_write_data = write_data_q;
    assign commit_valid = !fault_q && state_q == ST_COMMIT;
    assign commit_address = {commit_index_q, commit_row_q};
    assign commit_data = commit_data_q;
    assign commit_last = commit_index_q == 7 && commit_row_q == ROWS_PER_PHASE-1;
    assign phase_done_valid = !fault_q && state_q == ST_DONE;
    assign phase_done_tag = tag_q;
    assign phase_done_active_rows = active_q;
    assign protocol_error = fault_q;
    assign busy = state_q != ST_CONFIG;
    assign debug_state = state_q;
    assign debug_descriptor_writes = dwrites_q;
    assign debug_descriptor_reads = dreads_q;
    assign debug_pwp_requests = preqs_q;
    assign debug_weight_requests = wreqs_q;
    assign debug_forward_hits = forward_q;
    assign debug_accumulator_reads = areads_q;
    assign debug_commits = commits_q;
    assign debug_zero_initializations = zero_initializations_q;
    assign debug_zero_commits = zero_commits_q;
    assign debug_row_live_sets = row_live_sets_q;
    assign debug_row_live_clears = row_live_clears_q;
    assign debug_zero_init_slot_mask = zero_init_slot_mask_q;
    assign debug_row_live_set_event = accumulator_write_valid &&
        accumulator_write_ready && current_slot_w == 7 && !current_live_w;
    assign debug_row_live_clear_event = commit_valid && commit_ready &&
        commit_index_q == 7 && commit_live_w;
    assign debug_forward_event = state_q == ST_ACC && last_valid_q[current_slot_w] &&
        last_row_q[current_slot_w] == match_row_q;
    assign debug_operator_boundary_pending = operator_boundary_pending_q;

    function automatic logic [3:0] first_set(input logic [15:0] mask);
        logic found;
        begin
            first_set = 0; found = 0;
            for (integer k = 0; k < 16; k = k + 1)
                if (!found && mask[k]) begin first_set = k[3:0]; found = 1; end
        end
    endfunction

    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            state_q <= ST_CONFIG; fault_q <= 0; tag_q <= 0;
            active_q <= 0; desc_index_q <= 0; commit_row_q <= 0;
            operator_last_q <= 0;
            tile_q <= 0; block_q <= 0; commit_index_q <= 0;
            match_row_q <= 0; original_q <= 0; center_q <= 0;
            distance_q <= 0; use_pwp_q <= 0; plus_q <= 0; minus_q <= 0;
            match_last_q <= 0;
            correction_mask_q <= 0; source_q <= 0;
            correction_minus_q <= 0; narrow_q <= 0; delta_q <= 0;
            last_valid_q <= 0; write_data_q <= 0; commit_data_q <= 0;
            dwrites_q <= 0; dreads_q <= 0; preqs_q <= 0; wreqs_q <= 0;
            forward_q <= 0; areads_q <= 0; commits_q <= 0;
            zero_initializations_q <= 0; zero_commits_q <= 0;
            row_live_sets_q <= 0; row_live_clears_q <= 0;
            zero_init_slot_mask_q <= 0; operator_boundary_pending_q <= 1;
            for (integer c = 0; c < 8; c = c + 1) begin last_data_q[c] <= 0; last_row_q[c] <= 0; end
            for (integer r = 0; r < ROWS_PER_PHASE; r = r + 1)
                row_live_q[r] <= 0;
        end else begin
            if ((descriptor_response_valid && state_q != ST_DRSP) ||
                (payload_response_valid && state_q != ST_PWP_RSP && state_q != ST_WRSP) ||
                (accumulator_response_valid && state_q != ST_ARRSP && state_q != ST_CRSP)) begin
                fault_q <= 1; state_q <= ST_FAULT;
            end
            if (matcher_protocol_error || m433_protocol_error) begin
                fault_q <= 1; state_q <= ST_FAULT;
            end
            case (state_q)
                ST_CONFIG: begin
                    if (config_accept && config_beat_index == 0) begin
                        tag_q <= config_tag; operator_last_q <= config_operator_last_phase;
                    end
                    if (config_accept && config_beat_index == 2) begin
                        if (config_operator_last_phase != operator_last_q || config_tag != tag_q) begin
                            fault_q <= 1; state_q <= ST_FAULT;
                        end else begin
                            state_q <= ST_ROWS; active_q <= 0;
                        end
                    end
                end
                ST_ROWS: if (matcher_result_valid) begin
                    if (matcher_result_tag != tag_q) begin fault_q <= 1; state_q <= ST_FAULT; end
                    else if (matcher_result_original == 0) begin
                        if (matcher_result_accept && matcher_result_last) begin
                            desc_index_q <= 0; tile_q <= 0; block_q <= 0;
                            state_q <= active_q == 0 ? ST_RELEASE : ST_DREQ;
                            commit_index_q <= 0;
                        end
                    end else begin
                        match_row_q <= matcher_result_row;
                        original_q <= matcher_result_original;
                        center_q <= matcher_result_center;
                        distance_q <= matcher_result_distance;
                        use_pwp_q <= matcher_result_use;
                        match_last_q <= matcher_result_last;
                        state_q <= ST_DWRITE;
                    end
                end
                ST_DWRITE: if (descriptor_write_valid && descriptor_write_ready) begin
                    dwrites_q <= dwrites_q + 1; active_q <= active_q + 1;
                    if (match_last_q) begin
                        desc_index_q <= 0; tile_q <= 0; block_q <= 0;
                        commit_index_q <= 0; state_q <= ST_DREQ;
                    end else state_q <= ST_ROWS;
                end
                ST_DREQ: if (descriptor_read_valid && descriptor_read_ready) begin
                    dreads_q <= dreads_q + 1; state_q <= ST_DRSP;
                end
                ST_DRSP: if (descriptor_response_valid) begin
                    if (descriptor_response_tag != tag_q ||
                        descriptor_response_address != desc_index_q ||
                        descriptor_response_data[47:41] != 0 ||
                        descriptor_response_data[27:12] == 0) begin
                        fault_q <= 1; state_q <= ST_FAULT;
                    end else begin
                        match_row_q <= descriptor_response_data[11:0];
                        original_q <= descriptor_response_data[27:12];
                        center_q <= descriptor_response_data[32:28];
                        distance_q <= descriptor_response_data[39:35];
                        use_pwp_q <= descriptor_response_data[40];
                        narrow_q <= narrow_bitmap_w[descriptor_response_data[32:28]];
                        plus_q <= descriptor_response_data[27:12] &
                            ~centers_w[descriptor_response_data[32:28]*16 +: 16];
                        minus_q <= centers_w[descriptor_response_data[32:28]*16 +: 16] &
                            ~descriptor_response_data[27:12];
                        correction_mask_q <= descriptor_response_data[40] ?
                            (descriptor_response_data[27:12] ^ centers_w[descriptor_response_data[32:28]*16 +: 16]) :
                            descriptor_response_data[27:12];
                        state_q <= descriptor_response_data[40] ? ST_PWP_REQ :
                            ((descriptor_response_data[27:12] == 0) ? ST_ACC : ST_WREQ);
                        delta_q <= 0;
                    end
                end
                ST_PWP_REQ: if (payload_request_valid && payload_request_ready) begin
                    preqs_q <= preqs_q + 1; state_q <= ST_PWP_RSP;
                end
                ST_PWP_RSP: if (payload_response_valid) begin
                    if (!payload_response_pwp || payload_response_tag != tag_q ||
                        payload_response_tile != tile_q || payload_response_block != block_q ||
                        payload_response_center != center_q ||
                        payload_response_narrow != narrow_q ||
                        (narrow_q && payload_response_high != 0)) begin
                        fault_q <= 1; state_q <= ST_FAULT;
                    end else if (m433_request_ready) state_q <= ST_PWP_OUT;
                end
                ST_PWP_OUT: if (m433_contribution_valid) begin
                    for (integer l = 0; l < 96; l = l + 1)
                        delta_q[l*19 +: 19] <= {{7{m433_contribution_data[l*12+11]}},m433_contribution_data[l*12 +: 12]};
                    if (correction_mask_q == 0) state_q <= ST_ACC;
                    else begin source_q <= first_set(correction_mask_q);
                        correction_minus_q <= minus_q[first_set(correction_mask_q)];
                        state_q <= ST_WREQ; end
                end
                ST_WREQ: begin
                    source_q <= first_set(correction_mask_q);
                    correction_minus_q <= minus_q[first_set(correction_mask_q)];
                    if (payload_request_valid && payload_request_ready) begin
                        wreqs_q <= wreqs_q + 1; state_q <= ST_WRSP;
                    end
                end
                ST_WRSP: if (payload_response_valid) begin
                    if (payload_response_pwp || payload_response_tag != tag_q ||
                        payload_response_tile != tile_q || payload_response_block != block_q ||
                        payload_response_source != first_set(correction_mask_q) || payload_response_high != 0) begin
                        fault_q <= 1; state_q <= ST_FAULT;
                    end else begin
                        for (integer l = 0; l < 96; l = l + 1)
                            if (minus_q[first_set(correction_mask_q)])
                                delta_q[l*19 +: 19] <= $signed(delta_q[l*19 +: 19]) -
                                    $signed(payload_response_low_or_weight[l*8 +: 8]);
                            else delta_q[l*19 +: 19] <= $signed(delta_q[l*19 +: 19]) +
                                    $signed(payload_response_low_or_weight[l*8 +: 8]);
                        correction_mask_q[first_set(correction_mask_q)] <= 0;
                        state_q <= (correction_mask_q & ~(16'b1 << first_set(correction_mask_q))) == 0 ? ST_ACC : ST_WREQ;
                    end
                end
                ST_ACC: if (last_valid_q[current_slot_w] && last_row_q[current_slot_w] == match_row_q) begin
                    for (integer l = 0; l < 96; l = l + 1)
                        write_data_q[l*19 +: 19] <=
                            $signed(last_data_q[current_slot_w][l*19 +: 19]) +
                            $signed(delta_q[l*19 +: 19]);
                    forward_q <= forward_q + 1; state_q <= ST_AWRITE;
                end else if (!current_live_w) begin
                    write_data_q <= delta_q;
                    zero_initializations_q <= zero_initializations_q + 1;
                    zero_init_slot_mask_q <= zero_init_slot_mask_q | (8'b1 << current_slot_w);
                    state_q <= ST_AWRITE;
                end else state_q <= ST_ARREQ;
                ST_ARREQ: if (accumulator_read_valid && accumulator_read_ready) begin
                    areads_q <= areads_q + 1; state_q <= ST_ARRSP;
                end
                ST_ARRSP: if (accumulator_response_valid) begin
                    if (accumulator_response_address != current_address_w) begin
                        fault_q <= 1; state_q <= ST_FAULT;
                    end else begin
                        for (integer l = 0; l < 96; l = l + 1)
                            write_data_q[l*19 +: 19] <=
                                $signed(accumulator_response_data[l*19 +: 19]) +
                                $signed(delta_q[l*19 +: 19]);
                        state_q <= ST_AWRITE;
                    end
                end
                ST_AWRITE: if (accumulator_write_valid && accumulator_write_ready) begin
                    if (current_slot_w == 7 && !current_live_w) begin
                        row_live_q[match_row_q] <= 1;
                        row_live_sets_q <= row_live_sets_q + 1;
                    end
                    operator_boundary_pending_q <= 0;
                    last_valid_q[current_slot_w] <= 1;
                    last_row_q[current_slot_w] <= match_row_q;
                    last_data_q[current_slot_w] <= write_data_q;
                    state_q <= ST_NEXT;
                end
                ST_NEXT: begin
                    if (block_q != 3) begin
                        block_q <= block_q + 1;
                        delta_q <= 0;
                        correction_mask_q <= use_pwp_q ?
                            (original_q ^ center_pattern_w) : original_q;
                        state_q <= use_pwp_q ? ST_PWP_REQ : ST_WREQ;
                    end
                    else if (desc_index_q + 1 < active_q) begin
                        block_q <= 0; desc_index_q <= desc_index_q + 1; state_q <= ST_DREQ;
                    end else if (!tile_q) begin
                        tile_q <= 1; block_q <= 0; desc_index_q <= 0; state_q <= ST_DREQ;
                    end else begin state_q <= ST_RELEASE; end
                end
                ST_RELEASE: if (matcher_release_accept) begin
                    if (operator_last_q) begin commit_index_q <= 0; commit_row_q <= 0; state_q <= ST_CREAD; end
                    else state_q <= ST_DONE;
                end
                ST_CREAD: if (!commit_live_w) begin
                    commit_data_q <= 0;
                    zero_commits_q <= zero_commits_q + 1;
                    state_q <= ST_COMMIT;
                end else if (accumulator_read_valid && accumulator_read_ready) begin
                    areads_q <= areads_q + 1; state_q <= ST_CRSP;
                end
                ST_CRSP: if (accumulator_response_valid) begin
                    if (accumulator_response_address != {commit_index_q,commit_row_q}) begin fault_q<=1; state_q<=ST_FAULT; end
                    else begin commit_data_q <= accumulator_response_data; state_q <= ST_COMMIT; end
                end
                ST_COMMIT: if (commit_valid && commit_ready) begin
                    commits_q <= commits_q + 1;
                    if (commit_index_q == 7 && commit_live_w) begin
                        row_live_q[commit_row_q] <= 0;
                        row_live_clears_q <= row_live_clears_q + 1;
                    end
                    if (commit_row_q == ROWS_PER_PHASE-1) begin
                        commit_row_q <= 0;
                        if (commit_index_q == 7) begin
                            last_valid_q <= 0;
                            operator_boundary_pending_q <= 1;
                            state_q <= ST_DONE;
                        end
                        else begin commit_index_q <= commit_index_q + 1; state_q <= ST_CREAD; end
                    end else begin commit_row_q <= commit_row_q + 1; state_q <= ST_CREAD; end
                end
                ST_DONE: if (phase_done_valid && phase_done_ready) state_q <= ST_CONFIG;
                default: begin fault_q <= 1; state_q <= ST_FAULT; end
            endcase
        end
    end
endmodule

`default_nettype wire
