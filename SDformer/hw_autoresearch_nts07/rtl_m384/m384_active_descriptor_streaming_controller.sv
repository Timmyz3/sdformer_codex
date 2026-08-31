`timescale 1ns/1ps
`default_nettype none

// M384 is deliberately a bounded control-plane cut.  The q32 matcher and the
// Conv backend remain outside.  This q32-layout revision compacts only original16!=0 rows,
// seals the active count/used-center bitmap, and replays the external 48-bit
// descriptor SRAM twice with an eight-credit, in-order, II=1 interface.
module m384_active_descriptor_streaming_controller #(
    parameter int TAG_BITS = 24,
    parameter int FIFO_DEPTH = 8
) (
    input  logic                    clk_core,
    input  logic                    reset_n,

    input  logic                    config_reload,
    output logic                    config_reload_accept,

    input  logic                    phase_valid,
    output logic                    phase_ready,
    output logic                    phase_accept,
    input  logic [TAG_BITS-1:0]     phase_tag,
    input  logic                    phase_bank,
    input  logic [511:0]            phase_centers_q16,

    input  logic                    row_valid,
    output logic                    row_ready,
    output logic                    row_accept,
    input  logic [11:0]             row_id,
    input  logic [15:0]             row_original,
    input  logic [6:0]              row_center_id,
    input  logic [4:0]              row_distance,
    input  logic                    row_use_pwp,
    input  logic                    row_last,

    output logic                    descriptor_write_valid,
    input  logic                    descriptor_write_ready,
    output logic                    descriptor_write_accept,
    output logic [TAG_BITS-1:0]     descriptor_write_tag,
    output logic                    descriptor_write_bank,
    output logic [11:0]             descriptor_write_address,
    output logic [47:0]             descriptor_write_data,

    output logic                    phase_seal_valid,
    input  logic                    phase_seal_ready,
    output logic                    phase_seal_accept,
    output logic [TAG_BITS-1:0]     phase_seal_tag,
    output logic                    phase_seal_bank,
    output logic [11:0]             phase_seal_active_count,
    output logic [31:0]             phase_seal_used_center_bitmap,
    output logic                    phase_seal_empty,

    output logic                    pwp_run_valid,
    input  logic                    pwp_run_ready,
    output logic                    pwp_run_accept,
    output logic [4:0]              pwp_run_start_center,
    output logic [5:0]              pwp_run_length_centers,
    output logic [15:0]             pwp_run_tile0_address,
    output logic [15:0]             pwp_run_tile1_address,
    output logic [15:0]             pwp_run_bytes,
    output logic                    pwp_run_last,

    output logic                    tile1_prefetch_valid,
    input  logic                    tile1_prefetch_ready,
    output logic                    tile1_prefetch_accept,
    output logic [TAG_BITS-1:0]     tile1_prefetch_tag,
    output logic                    tile1_prefetch_bank,
    output logic [15:0]             tile1_prefetch_weight_address,
    output logic [15:0]             tile1_prefetch_pwp_base_address,
    output logic [31:0]             tile1_prefetch_used_center_bitmap,
    input  logic                    tile1_prefetch_done_valid,
    output logic                    tile1_prefetch_done_ready,
    output logic                    tile1_prefetch_done_accept,
    input  logic [TAG_BITS-1:0]     tile1_prefetch_done_tag,
    input  logic                    tile1_prefetch_done_bank,

    input  logic                    replay_start_valid,
    output logic                    replay_start_ready,
    output logic                    replay_start_accept,
    input  logic                    replay_start_tile,

    output logic                    descriptor_read_req_valid,
    input  logic                    descriptor_read_req_ready,
    output logic                    descriptor_read_req_accept,
    output logic [TAG_BITS-1:0]     descriptor_read_req_tag,
    output logic                    descriptor_read_req_bank,
    output logic [11:0]             descriptor_read_req_address,

    input  logic                    descriptor_read_rsp_valid,
    output logic                    descriptor_read_rsp_ready,
    output logic                    descriptor_read_rsp_accept,
    input  logic [TAG_BITS-1:0]     descriptor_read_rsp_tag,
    input  logic                    descriptor_read_rsp_bank,
    input  logic [11:0]             descriptor_read_rsp_address,
    input  logic [47:0]             descriptor_read_rsp_data,

    output logic                    bundle_valid,
    input  logic                    bundle_ready,
    output logic                    bundle_accept,
    output logic [TAG_BITS-1:0]     bundle_tag,
    output logic                    bundle_tile,
    output logic [11:0]             bundle_row_id,
    output logic [15:0]             bundle_original,
    output logic [6:0]              bundle_center_id,
    output logic [15:0]             bundle_center,
    output logic [4:0]              bundle_distance,
    output logic                    bundle_use_pwp,
    output logic                    bundle_fallback_bit_sparse,
    output logic [15:0]             bundle_plus_mask,
    output logic [15:0]             bundle_minus_mask,

    output logic                    replay_done_valid,
    input  logic                    replay_done_ready,
    output logic                    replay_done_accept,
    output logic [TAG_BITS-1:0]     replay_done_tag,
    output logic                    replay_done_tile,
    output logic [11:0]             replay_done_count,

    output logic                    phase_done_valid,
    input  logic                    phase_done_ready,
    output logic                    phase_done_accept,
    output logic [TAG_BITS-1:0]     phase_done_tag,
    output logic [11:0]             phase_done_active_count,
    output logic [31:0]             phase_done_used_center_bitmap,
    output logic                    phase_done_empty,

    output logic                    protocol_error,
    output logic                    busy,
    output logic [3:0]              debug_state,
    output logic [11:0]             debug_rows_accepted,
    output logic [11:0]             debug_active_count,
    output logic [3:0]              debug_fifo_occupancy,
    output logic [3:0]              debug_outstanding_reads,
    output logic [3:0]              debug_credit_used,
    output logic [1:0]              debug_replays_completed,
    output logic [31:0]             debug_descriptor_writes,
    output logic [31:0]             debug_descriptor_requests,
    output logic [31:0]             debug_descriptor_responses,
    output logic [31:0]             debug_bundle_accepts,
    output logic [31:0]             debug_pwp_runs_issued
);
    localparam int ROWS_PER_PHASE = 3000;
    localparam int FIFO_PTR_BITS = 3;
    localparam logic [3:0] ST_IDLE       = 4'd0;
    localparam logic [3:0] ST_MATCH      = 4'd1;
    localparam logic [3:0] ST_SEAL       = 4'd2;
    localparam logic [3:0] ST_RUNS       = 4'd3;
    localparam logic [3:0] ST_WAIT0      = 4'd4;
    localparam logic [3:0] ST_REPLAY0    = 4'd5;
    localparam logic [3:0] ST_DONE0      = 4'd6;
    localparam logic [3:0] ST_WAIT1      = 4'd7;
    localparam logic [3:0] ST_REPLAY1    = 4'd8;
    localparam logic [3:0] ST_DONE1      = 4'd9;
    localparam logic [3:0] ST_PHASE_DONE = 4'd10;
    localparam logic [3:0] ST_ERROR      = 4'd15;

    logic [3:0] state_q;
    logic fault_q;
    logic [TAG_BITS-1:0] tag_q;
    logic bank_q, replay_tile_q;
    logic tile1_prefetch_started_q, tile1_prefetch_done_q;
    logic [511:0] centers_q;
    logic [11:0] row_count_q, active_count_q;
    logic [31:0] used_center_bitmap_q;

    logic [11:0] request_count_q, response_count_q, consume_count_q;
    logic [3:0] outstanding_q, fifo_count_q;
    logic [FIFO_PTR_BITS-1:0] fifo_write_ptr_q, fifo_read_ptr_q;
    logic [47:0] fifo_mem [0:FIFO_DEPTH-1];
    logic [11:0] last_response_row_q;
    logic last_response_row_valid_q;
    logic [1:0] replays_completed_q;

    logic [31:0] descriptor_writes_q, descriptor_requests_q;
    logic [31:0] descriptor_responses_q, bundle_accepts_q;
    logic [31:0] pwp_runs_issued_q, run_remaining_q, run_clear_mask_w;
    logic [4:0] run_start_w;
    logic [5:0] run_length_w;
    logic run_found_w, run_open_w;

    logic [15:0] row_center_w, response_center_w, bundle_center_w;
    logic [4:0] row_population_w, row_distance_recomputed_w;
    logic [4:0] response_population_w, response_distance_recomputed_w;
    logic row_expected_use_w, response_expected_use_w;
    logic row_shape_legal_w, response_shape_legal_w;
    logic response_identity_legal_w, response_order_legal_w;
    logic response_payload_legal_w;
    logic [47:0] fifo_head_w;
    logic [11:0] response_row_w, bundle_row_w;
    logic [15:0] response_original_w, bundle_original_w;
    logic [6:0] response_center_id_w, bundle_center_id_w;
    logic [4:0] response_distance_w, bundle_distance_w;
    logic response_use_w, bundle_use_w;
    logic [6:0] response_flags_w;
    logic replay_state_w, credit_available_w, fifo_space_w;
    logic illegal_phase_w, illegal_reload_w, illegal_row_w;
    logic illegal_replay_start_w, illegal_response_w;
    logic illegal_prefetch_done_w, internal_fault_w;
    logic fault_event_w, safe_w;

    function automatic logic [4:0] popcount16(input logic [15:0] value);
        integer bit_index;
        logic [4:0] count;
        begin
            count = '0;
            for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
                count = count + value[bit_index];
            popcount16 = count;
        end
    endfunction

    generate
        if (FIFO_DEPTH != 8) begin : g_bad_fifo_depth
            initial $fatal(1, "M384 frozen FIFO depth must equal eight");
        end
    endgenerate

    always_comb begin : row_validation
        row_center_w = centers_q[row_center_id[4:0]*16 +: 16];
        row_population_w = popcount16(row_original);
        row_distance_recomputed_w = popcount16(row_original ^ row_center_w);
        row_expected_use_w = ({1'b0,row_distance_recomputed_w} + 6'd1)
            < {1'b0,row_population_w};
        row_shape_legal_w = row_id == row_count_q
            && row_id < ROWS_PER_PHASE
            && row_center_id < 7'd32
            && row_distance == row_distance_recomputed_w
            && row_use_pwp == row_expected_use_w
            && row_last == (row_count_q == ROWS_PER_PHASE-1);
    end

    always_comb begin : response_validation
        response_row_w = descriptor_read_rsp_data[11:0];
        response_original_w = descriptor_read_rsp_data[27:12];
        response_center_id_w = descriptor_read_rsp_data[34:28];
        response_distance_w = descriptor_read_rsp_data[39:35];
        response_use_w = descriptor_read_rsp_data[40];
        response_flags_w = descriptor_read_rsp_data[47:41];
        response_center_w = centers_q[response_center_id_w[4:0]*16 +: 16];
        response_population_w = popcount16(response_original_w);
        response_distance_recomputed_w = popcount16(
            response_original_w ^ response_center_w);
        response_expected_use_w =
            ({1'b0,response_distance_recomputed_w} + 6'd1)
            < {1'b0,response_population_w};
        response_identity_legal_w = descriptor_read_rsp_tag == tag_q
            && descriptor_read_rsp_bank == bank_q
            && descriptor_read_rsp_address == response_count_q;
        response_order_legal_w = response_row_w < ROWS_PER_PHASE
            && (!last_response_row_valid_q
                || response_row_w > last_response_row_q);
        response_shape_legal_w = response_original_w != 0
            && response_center_id_w < 7'd32
            && response_distance_w == response_distance_recomputed_w
            && response_use_w == response_expected_use_w
            && response_flags_w == 0;
        response_payload_legal_w = response_identity_legal_w
            && response_order_legal_w && response_shape_legal_w;
    end

    always_comb begin : bundle_decode
        fifo_head_w = fifo_mem[fifo_read_ptr_q];
        bundle_row_w = fifo_head_w[11:0];
        bundle_original_w = fifo_head_w[27:12];
        bundle_center_id_w = fifo_head_w[34:28];
        bundle_distance_w = fifo_head_w[39:35];
        bundle_use_w = fifo_head_w[40];
        bundle_center_w = centers_q[bundle_center_id_w[4:0]*16 +: 16];
    end

    always_comb begin : pwp_run_decode
        run_start_w = '0;
        run_length_w = '0;
        run_found_w = 1'b0;
        run_open_w = 1'b0;
        run_clear_mask_w = '0;
        for (integer center_index=0; center_index<32;
             center_index=center_index+1) begin
            if (!run_found_w && run_remaining_q[center_index]) begin
                run_start_w = center_index[4:0];
                run_found_w = 1'b1;
                run_open_w = 1'b1;
            end
            if (run_found_w && run_open_w
                && center_index >= run_start_w) begin
                if (run_remaining_q[center_index]) begin
                    run_length_w = run_length_w + 1'b1;
                    run_clear_mask_w[center_index] = 1'b1;
                end else begin
                    run_open_w = 1'b0;
                end
            end
        end
    end

    always_comb begin : protocol_faults
        replay_state_w = state_q == ST_REPLAY0 || state_q == ST_REPLAY1;
        illegal_phase_w = phase_valid && state_q != ST_IDLE;
        illegal_reload_w = config_reload && state_q != ST_IDLE;
        illegal_row_w = row_valid
            && (state_q != ST_MATCH || !row_shape_legal_w);
        illegal_replay_start_w = replay_start_valid
            && !((state_q == ST_WAIT0 && replay_start_tile == 1'b0)
                 || (state_q == ST_WAIT1 && replay_start_tile == 1'b1));
        illegal_response_w = descriptor_read_rsp_valid
            && (!replay_state_w || outstanding_q == 0
                || !response_payload_legal_w);
        illegal_prefetch_done_w = tile1_prefetch_done_valid
            && (!tile1_prefetch_started_q || tile1_prefetch_done_q
                || tile1_prefetch_done_tag != tag_q
                || tile1_prefetch_done_bank != bank_q);
        internal_fault_w = fifo_count_q > FIFO_DEPTH
            || outstanding_q > FIFO_DEPTH
            || ({1'b0,fifo_count_q} + {1'b0,outstanding_q}) > FIFO_DEPTH
            || request_count_q > active_count_q
            || response_count_q > request_count_q
            || consume_count_q > response_count_q
            || replays_completed_q > 2;
        fault_event_w = !fault_q && (illegal_phase_w || illegal_reload_w
            || illegal_row_w || illegal_replay_start_w
            || illegal_response_w || illegal_prefetch_done_w
            || internal_fault_w);
        safe_w = !fault_q && !fault_event_w;
    end

    always_comb begin : interfaces
        config_reload_accept = config_reload && state_q == ST_IDLE
            && !phase_valid && safe_w;
        phase_ready = state_q == ST_IDLE && !config_reload && safe_w;
        phase_accept = phase_valid && phase_ready;

        descriptor_write_valid = state_q == ST_MATCH && row_valid
            && row_shape_legal_w && row_original != 0 && safe_w;
        descriptor_write_tag = tag_q;
        descriptor_write_bank = bank_q;
        descriptor_write_address = active_count_q;
        descriptor_write_data = {7'b0,row_use_pwp,row_distance,
                                 row_center_id,row_original,row_count_q};
        descriptor_write_accept = descriptor_write_valid
            && descriptor_write_ready;
        row_ready = state_q == ST_MATCH && row_shape_legal_w && safe_w
            && (row_original == 0 || descriptor_write_ready);
        row_accept = row_valid && row_ready;

        phase_seal_valid = state_q == ST_SEAL && safe_w;
        phase_seal_tag = tag_q;
        phase_seal_bank = bank_q;
        phase_seal_active_count = active_count_q;
        phase_seal_used_center_bitmap = used_center_bitmap_q;
        phase_seal_empty = active_count_q == 0;
        phase_seal_accept = phase_seal_valid && phase_seal_ready;

        pwp_run_valid = state_q == ST_RUNS && run_found_w && safe_w;
        pwp_run_accept = pwp_run_valid && pwp_run_ready;
        pwp_run_start_center = run_start_w;
        pwp_run_length_centers = run_length_w;
        pwp_run_tile0_address = 16'd6240 + run_start_w * 16'd640;
        pwp_run_tile1_address = 16'd38912 + run_start_w * 16'd640;
        pwp_run_bytes = run_length_w * 16'd640;
        pwp_run_last = (run_remaining_q & ~run_clear_mask_w) == 0;

        tile1_prefetch_valid = state_q == ST_WAIT0 && replay_start_valid
            && replay_start_tile == 1'b0 && safe_w;
        tile1_prefetch_accept = tile1_prefetch_valid
            && tile1_prefetch_ready;
        tile1_prefetch_tag = tag_q;
        tile1_prefetch_bank = bank_q;
        tile1_prefetch_weight_address = 16'd32768;
        tile1_prefetch_pwp_base_address = 16'd38912;
        tile1_prefetch_used_center_bitmap = used_center_bitmap_q;
        tile1_prefetch_done_ready = tile1_prefetch_started_q
            && !tile1_prefetch_done_q
            && (state_q == ST_REPLAY0 || state_q == ST_DONE0
                || state_q == ST_WAIT1) && safe_w;
        tile1_prefetch_done_accept = tile1_prefetch_done_valid
            && tile1_prefetch_done_ready;

        replay_start_ready = ((state_q == ST_WAIT0
                               && tile1_prefetch_ready)
                              || (state_q == ST_WAIT1
                                  && tile1_prefetch_done_q)) && safe_w;
        replay_start_accept = replay_start_valid && replay_start_ready;

        bundle_valid = replay_state_w && fifo_count_q != 0 && safe_w;
        bundle_accept = bundle_valid && bundle_ready;
        credit_available_w =
            ({1'b0,fifo_count_q} + {1'b0,outstanding_q}) < FIFO_DEPTH
            || bundle_accept;
        fifo_space_w = fifo_count_q < FIFO_DEPTH || bundle_accept;

        descriptor_read_req_valid = replay_state_w
            && request_count_q < active_count_q
            && credit_available_w && safe_w;
        descriptor_read_req_tag = tag_q;
        descriptor_read_req_bank = bank_q;
        descriptor_read_req_address = request_count_q;
        descriptor_read_req_accept = descriptor_read_req_valid
            && descriptor_read_req_ready;

        descriptor_read_rsp_ready = replay_state_w && outstanding_q != 0
            && fifo_space_w && response_payload_legal_w && safe_w;
        descriptor_read_rsp_accept = descriptor_read_rsp_valid
            && descriptor_read_rsp_ready;

        bundle_tag = tag_q;
        bundle_tile = replay_tile_q;
        bundle_row_id = bundle_row_w;
        bundle_original = bundle_original_w;
        bundle_center_id = bundle_center_id_w;
        bundle_center = bundle_center_w;
        bundle_distance = bundle_distance_w;
        bundle_use_pwp = bundle_use_w;
        bundle_fallback_bit_sparse = !bundle_use_w;
        bundle_plus_mask = bundle_use_w
            ? (bundle_original_w & ~bundle_center_w) : bundle_original_w;
        bundle_minus_mask = bundle_use_w
            ? (bundle_center_w & ~bundle_original_w) : 16'b0;

        replay_done_valid = (state_q == ST_DONE0 || state_q == ST_DONE1)
            && safe_w;
        replay_done_accept = replay_done_valid && replay_done_ready;
        replay_done_tag = tag_q;
        replay_done_tile = replay_tile_q;
        replay_done_count = consume_count_q;

        phase_done_valid = state_q == ST_PHASE_DONE && safe_w;
        phase_done_accept = phase_done_valid && phase_done_ready;
        phase_done_tag = tag_q;
        phase_done_active_count = active_count_q;
        phase_done_used_center_bitmap = used_center_bitmap_q;
        phase_done_empty = active_count_q == 0;

        protocol_error = fault_q || fault_event_w;
        busy = state_q != ST_IDLE;
        debug_state = state_q;
        debug_rows_accepted = row_count_q;
        debug_active_count = active_count_q;
        debug_fifo_occupancy = fifo_count_q;
        debug_outstanding_reads = outstanding_q;
        debug_credit_used = fifo_count_q + outstanding_q;
        debug_replays_completed = replays_completed_q;
        debug_descriptor_writes = descriptor_writes_q;
        debug_descriptor_requests = descriptor_requests_q;
        debug_descriptor_responses = descriptor_responses_q;
        debug_bundle_accepts = bundle_accepts_q;
        debug_pwp_runs_issued = pwp_runs_issued_q;
    end

    always_ff @(posedge clk_core or negedge reset_n) begin : state_and_data
        if (!reset_n) begin
            state_q <= ST_IDLE;
            fault_q <= 1'b0;
            tag_q <= '0;
            bank_q <= 1'b0;
            replay_tile_q <= 1'b0;
            tile1_prefetch_started_q <= 1'b0;
            tile1_prefetch_done_q <= 1'b0;
            centers_q <= '0;
            row_count_q <= '0;
            active_count_q <= '0;
            used_center_bitmap_q <= '0;
            request_count_q <= '0;
            response_count_q <= '0;
            consume_count_q <= '0;
            outstanding_q <= '0;
            fifo_count_q <= '0;
            fifo_write_ptr_q <= '0;
            fifo_read_ptr_q <= '0;
            last_response_row_q <= '0;
            last_response_row_valid_q <= 1'b0;
            replays_completed_q <= '0;
            descriptor_writes_q <= '0;
            descriptor_requests_q <= '0;
            descriptor_responses_q <= '0;
            bundle_accepts_q <= '0;
            pwp_runs_issued_q <= '0;
            run_remaining_q <= '0;
        end else if (fault_event_w) begin
            state_q <= ST_ERROR;
            fault_q <= 1'b1;
            outstanding_q <= '0;
            fifo_count_q <= '0;
            fifo_write_ptr_q <= '0;
            fifo_read_ptr_q <= '0;
        end else begin
            if (config_reload_accept) begin
                centers_q <= '0;
                tag_q <= '0;
                bank_q <= 1'b0;
                tile1_prefetch_started_q <= 1'b0;
                tile1_prefetch_done_q <= 1'b0;
            end
            if (phase_accept) begin
                state_q <= ST_MATCH;
                tag_q <= phase_tag;
                bank_q <= phase_bank;
                tile1_prefetch_started_q <= 1'b0;
                tile1_prefetch_done_q <= 1'b0;
                centers_q <= phase_centers_q16;
                row_count_q <= '0;
                active_count_q <= '0;
                used_center_bitmap_q <= '0;
                request_count_q <= '0;
                response_count_q <= '0;
                consume_count_q <= '0;
                outstanding_q <= '0;
                fifo_count_q <= '0;
                fifo_write_ptr_q <= '0;
                fifo_read_ptr_q <= '0;
                last_response_row_valid_q <= 1'b0;
                replays_completed_q <= '0;
                descriptor_writes_q <= '0;
                descriptor_requests_q <= '0;
                descriptor_responses_q <= '0;
                bundle_accepts_q <= '0;
                pwp_runs_issued_q <= '0;
                run_remaining_q <= '0;
            end

            if (row_accept) begin
                if (row_original != 0) begin
                    active_count_q <= active_count_q + 1'b1;
                    descriptor_writes_q <= descriptor_writes_q + 1'b1;
                    if (row_use_pwp)
                        used_center_bitmap_q[row_center_id[4:0]] <= 1'b1;
                end
                if (row_last)
                    state_q <= ST_SEAL;
                else
                    row_count_q <= row_count_q + 1'b1;
            end

            if (phase_seal_accept) begin
                run_remaining_q <= used_center_bitmap_q;
                if (phase_seal_empty)
                    state_q <= ST_PHASE_DONE;
                else if (used_center_bitmap_q != 0)
                    state_q <= ST_RUNS;
                else
                    state_q <= ST_WAIT0;
            end
            if (pwp_run_accept) begin
                run_remaining_q <= run_remaining_q & ~run_clear_mask_w;
                pwp_runs_issued_q <= pwp_runs_issued_q + 1'b1;
                if (pwp_run_last)
                    state_q <= ST_WAIT0;
            end
            if (tile1_prefetch_accept)
                tile1_prefetch_started_q <= 1'b1;
            if (tile1_prefetch_done_accept)
                tile1_prefetch_done_q <= 1'b1;

            if (replay_start_accept) begin
                replay_tile_q <= replay_start_tile;
                request_count_q <= '0;
                response_count_q <= '0;
                consume_count_q <= '0;
                outstanding_q <= '0;
                fifo_count_q <= '0;
                fifo_write_ptr_q <= '0;
                fifo_read_ptr_q <= '0;
                last_response_row_q <= '0;
                last_response_row_valid_q <= 1'b0;
                state_q <= replay_start_tile ? ST_REPLAY1 : ST_REPLAY0;
            end

            if (descriptor_read_req_accept) begin
                request_count_q <= request_count_q + 1'b1;
                descriptor_requests_q <= descriptor_requests_q + 1'b1;
            end
            if (descriptor_read_rsp_accept) begin
                fifo_mem[fifo_write_ptr_q] <= descriptor_read_rsp_data;
                fifo_write_ptr_q <= fifo_write_ptr_q + 1'b1;
                response_count_q <= response_count_q + 1'b1;
                last_response_row_q <= response_row_w;
                last_response_row_valid_q <= 1'b1;
                descriptor_responses_q <= descriptor_responses_q + 1'b1;
            end
            if (bundle_accept) begin
                fifo_read_ptr_q <= fifo_read_ptr_q + 1'b1;
                consume_count_q <= consume_count_q + 1'b1;
                bundle_accepts_q <= bundle_accepts_q + 1'b1;
                if (consume_count_q + 1'b1 == active_count_q) begin
                    state_q <= replay_tile_q ? ST_DONE1 : ST_DONE0;
                    replays_completed_q <= replays_completed_q + 1'b1;
                end
            end

            case ({descriptor_read_req_accept,
                   descriptor_read_rsp_accept})
                2'b10: outstanding_q <= outstanding_q + 1'b1;
                2'b01: outstanding_q <= outstanding_q - 1'b1;
                default: begin end
            endcase
            case ({descriptor_read_rsp_accept,bundle_accept})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: begin end
            endcase

            if (replay_done_accept) begin
                state_q <= replay_done_tile ? ST_PHASE_DONE : ST_WAIT1;
            end
            if (phase_done_accept)
                state_q <= ST_IDLE;
        end
    end
endmodule

`default_nettype wire
