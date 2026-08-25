`timescale 1ns/1ps
`default_nettype none

// M219: native-cropped K1 tagged, lane-sliced FC2 update service.
//
// One accepted single-source group is expanded into six 16-lane requests.
// Eight equal-capacity physical banks remain outside this island, but each
// request selects exactly one 128-bit bank word.  Echoed
// epoch/slot/generation/bank identity owns an out-of-order response;
// per-(output block,slice) busy bits preserve exact accumulator update order.
// A one-entry 128-bit elastic response skid cuts the external memory-to-Acc24
// path without charging K1 for M218's 1024-bit K8 response.
//
// Common POR is assumed to reset both this island and the external memory.
// Local soft flushes use an explicit epoch-tagged request/acknowledgement and
// block all new tokens until the acknowledgement is accepted.
module m219_fc2_k1_cropped_tagged_slice_service_island #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int OUTSTANDING = 8,
    parameter int GROUP_FIFO_DEPTH = 4,
    parameter int SLICE_LANES = 16,
    parameter int FLUSH_ACK_TIMEOUT_CYCLES = 1024
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         soft_flush,
    output logic                         mem_flush_valid,
    input  logic                         mem_flush_ready,
    output logic [EPOCH_BITS-1:0]        mem_flush_epoch,
    input  logic                         mem_flush_ack_valid,
    output logic                         mem_flush_ack_ready,
    input  logic [EPOCH_BITS-1:0]        mem_flush_ack_epoch,

    input  logic                         header_valid,
    output logic                         header_ready,
    input  logic [TAG_BITS-1:0]          header_tag,
    input  logic [3:0]                   header_output_blocks,
    output logic                         header_accept,

    input  logic                         group_valid,
    output logic                         group_ready,
    input  logic [TAG_BITS-1:0]          group_tag,
    input  logic [2:0]                   group_output_block,
    input  logic [2:0]                   group_bank_id,
    input  logic [CHANNEL_BITS-1:0]      group_source_channel,
    output logic                         group_accept,

    input  logic                         frontend_done_valid,
    output logic                         frontend_done_ready,
    input  logic [TAG_BITS-1:0]          frontend_done_tag,
    input  logic                         frontend_done_had_event,
    output logic                         frontend_done_accept,

    output logic                         mem_req_valid,
    input  logic                         mem_req_ready,
    output logic [EPOCH_BITS-1:0]        mem_req_epoch,
    output logic [2:0]                   mem_req_slot,
    output logic [GENERATION_BITS-1:0]   mem_req_generation,
    output logic [TAG_BITS-1:0]          mem_req_tag,
    output logic [2:0]                   mem_req_output_block,
    output logic [2:0]                   mem_req_slice,
    output logic [2:0]                   mem_req_bank_id,
    output logic [CHANNEL_BITS-1:0]      mem_req_source_channel,
    output logic                         mem_req_accept,

    input  logic                         mem_rsp_valid,
    output logic                         mem_rsp_ready,
    input  logic [EPOCH_BITS-1:0]        mem_rsp_epoch,
    input  logic [2:0]                   mem_rsp_slot,
    input  logic [GENERATION_BITS-1:0]   mem_rsp_generation,
    input  logic [TAG_BITS-1:0]          mem_rsp_tag,
    input  logic [2:0]                   mem_rsp_bank_id,
    input  logic signed [7:0]            mem_rsp_weight [0:SLICE_LANES-1],
    output logic                         mem_rsp_accept,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic [2:0]                   result_output_block,
    output logic [2:0]                   result_slice,
    output logic signed [23:0]           result_accumulator [0:SLICE_LANES-1],
    output logic                         result_last,
    output logic                         result_accept,

    output logic                         token_done_valid,
    input  logic                         token_done_ready,
    output logic [TAG_BITS-1:0]          token_done_tag,
    output logic                         token_done_had_event,
    output logic                         token_done_accept,

    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         stale_response_seen,
    output logic                         busy,

    output logic [2:0]                   debug_fifo_count,
    output logic [3:0]                   debug_outstanding_count,
    output logic [31:0]                  debug_group_accept_count,
    output logic [31:0]                  debug_request_accept_count,
    output logic [31:0]                  debug_response_accept_count,
    output logic [31:0]                  debug_context_write_count,
    output logic [31:0]                  debug_result_accept_count,
    output logic [31:0]                  debug_active_bank_read_count
);
    localparam int FIFO_PTR_BITS = 2;
    localparam int SLOT_BITS = 3;
    localparam int SLICES = 6;
    localparam bit PARAMETERS_LEGAL = OUTSTANDING == 8
        && GROUP_FIFO_DEPTH == 4 && SLICE_LANES == 16
        && FLUSH_ACK_TIMEOUT_CYCLES == 1024;

    generate
        if (!PARAMETERS_LEGAL) begin : g_illegal_parameters
            initial $fatal(1, "M219 supports only O8/FIFO4/SLICE16");
        end
    endgenerate

    logic fault_q, overflow_q, stale_seen_q;
    logic [7:0] fault_cause_q;
    logic [EPOCH_BITS-1:0] epoch_q;
    logic [GENERATION_BITS-1:0] generation_q;
    logic flush_active_q, flush_req_pending_q, flush_ack_wait_q;
    logic [10:0] flush_ack_timer_q;

    logic token_active_q, frontend_done_seen_q, token_had_event_q;
    logic [TAG_BITS-1:0] token_tag_q;
    logic [3:0] output_blocks_q;

    logic [TAG_BITS-1:0] fifo_tag_q [0:GROUP_FIFO_DEPTH-1];
    logic [2:0] fifo_block_q [0:GROUP_FIFO_DEPTH-1];
    logic [2:0] fifo_bank_id_q [0:GROUP_FIFO_DEPTH-1];
    logic [CHANNEL_BITS-1:0] fifo_channel_q [0:GROUP_FIFO_DEPTH-1];
    logic [FIFO_PTR_BITS-1:0] fifo_read_q, fifo_write_q;
    logic [2:0] fifo_count_q;
    logic [2:0] head_slice_q;

    logic sb_valid_q [0:OUTSTANDING-1];
    logic [EPOCH_BITS-1:0] sb_epoch_q [0:OUTSTANDING-1];
    logic [GENERATION_BITS-1:0] sb_generation_q [0:OUTSTANDING-1];
    logic [TAG_BITS-1:0] sb_tag_q [0:OUTSTANDING-1];
    logic [2:0] sb_block_q [0:OUTSTANDING-1];
    logic [2:0] sb_slice_q [0:OUTSTANDING-1];
    logic [2:0] sb_bank_id_q [0:OUTSTANDING-1];
    logic [3:0] outstanding_count_q;

    logic ctx_busy_q [0:7][0:SLICES-1];
    logic ctx_valid_q [0:7][0:SLICES-1];
    logic signed [23:0] ctx_q [0:7][0:SLICES-1][0:SLICE_LANES-1];

    logic rsp_skid_valid_q;
    logic [2:0] rsp_skid_block_q, rsp_skid_slice_q;
    logic signed [7:0] rsp_skid_weight_q [0:SLICE_LANES-1];

    logic emit_q, done_pending_q;
    logic [2:0] emit_block_q, emit_slice_q;

    logic [31:0] group_accept_count_q, request_accept_count_q;
    logic [31:0] response_accept_count_q, context_write_count_q;
    logic [31:0] result_accept_count_q, active_bank_read_count_q;

    logic header_shape_legal, header_admission_open;
    logic group_shape_legal, frontend_done_shape_legal;
    logic illegal_header, illegal_group, illegal_frontend_done;
    logic illegal_flush, illegal_flush_ack, generation_exhausted;
    logic flush_ack_timeout;
    logic response_slot_in_range, response_identity_legal;
    logic legal_response_accept, response_skid_commit;
    logic free_slot_found;
    logic [SLOT_BITS-1:0] free_slot;
    logic held_slot_valid_q;
    logic [SLOT_BITS-1:0] held_slot_q, issue_slot;
    logic head_context_open, response_releases_head_context;
    logic fifo_pop;
    logic start_emit;

    always_comb begin : protocol_analysis
        header_shape_legal = PARAMETERS_LEGAL && (header_output_blocks == 1
            || header_output_blocks == 2
            || header_output_blocks == 4
            || header_output_blocks == 8);
        header_admission_open = !token_active_q && !emit_q
            && !done_pending_q && fifo_count_q == 0
            && outstanding_count_q == 0 && !rsp_skid_valid_q
            && !flush_active_q && !fault_q && !overflow_q;
        header_ready = header_shape_legal && header_admission_open;
        header_accept = header_valid && header_ready;
        illegal_header = header_valid && !header_shape_legal;

        group_shape_legal = token_active_q && !frontend_done_seen_q
            && group_tag == token_tag_q
            && group_output_block < output_blocks_q
            && group_source_channel[2:0] == group_bank_id;
        group_ready = group_shape_legal
            && fifo_count_q < GROUP_FIFO_DEPTH
            && !flush_active_q && !fault_q && !overflow_q;
        group_accept = group_valid && group_ready;
        illegal_group = group_valid && !group_shape_legal;

        frontend_done_shape_legal = token_active_q
            && !frontend_done_seen_q && frontend_done_tag == token_tag_q;
        frontend_done_ready = frontend_done_shape_legal
            && !flush_active_q && !fault_q && !overflow_q;
        frontend_done_accept = frontend_done_valid
            && frontend_done_ready;
        illegal_frontend_done = frontend_done_valid
            && !frontend_done_shape_legal;

        response_slot_in_range = mem_rsp_slot < OUTSTANDING;
        response_identity_legal = token_active_q && response_slot_in_range;
        if (response_slot_in_range) begin
            response_identity_legal = response_identity_legal
                && sb_valid_q[mem_rsp_slot]
                && mem_rsp_epoch == sb_epoch_q[mem_rsp_slot]
                && mem_rsp_generation == sb_generation_q[mem_rsp_slot]
                && mem_rsp_tag == sb_tag_q[mem_rsp_slot]
                && mem_rsp_bank_id == sb_bank_id_q[mem_rsp_slot];
        end
        response_skid_commit = rsp_skid_valid_q
            && !flush_active_q && !soft_flush && !fault_q && !overflow_q;
        mem_rsp_ready = flush_active_q || soft_flush || fault_q || overflow_q
            || !rsp_skid_valid_q || response_skid_commit;
        mem_rsp_accept = mem_rsp_valid && mem_rsp_ready;
        legal_response_accept = mem_rsp_accept && response_identity_legal
            && !flush_active_q && !soft_flush && !fault_q && !overflow_q;

        free_slot_found = 0;
        free_slot = 0;
        for (int slot = 0; slot < OUTSTANDING; slot++) begin
            if (!free_slot_found && (!sb_valid_q[slot]
                    || (legal_response_accept && mem_rsp_slot == slot))) begin
                free_slot_found = 1;
                free_slot = slot[SLOT_BITS-1:0];
            end
        end
        response_releases_head_context = legal_response_accept
            && fifo_count_q != 0
            && sb_block_q[mem_rsp_slot] == fifo_block_q[fifo_read_q]
            && sb_slice_q[mem_rsp_slot] == head_slice_q;
        head_context_open = fifo_count_q != 0
            && (!ctx_busy_q[fifo_block_q[fifo_read_q]][head_slice_q]
                || response_releases_head_context);
        generation_exhausted = fifo_count_q != 0
            && generation_q == {GENERATION_BITS{1'b1}};

        issue_slot = held_slot_valid_q ? held_slot_q : free_slot;
        mem_req_valid = fifo_count_q != 0
            && (held_slot_valid_q || free_slot_found)
            && head_context_open && !generation_exhausted
            && token_active_q && !emit_q && !done_pending_q
            && !flush_active_q && !protocol_error && !numeric_overflow;
        mem_req_accept = mem_req_valid && mem_req_ready;
        fifo_pop = mem_req_accept && head_slice_q == SLICES-1;

        mem_req_epoch = epoch_q;
        mem_req_slot = issue_slot;
        mem_req_generation = generation_q;
        mem_req_tag = fifo_count_q == 0 ? 0 : fifo_tag_q[fifo_read_q];
        mem_req_output_block = fifo_count_q == 0
            ? 0 : fifo_block_q[fifo_read_q];
        mem_req_slice = head_slice_q;
        mem_req_bank_id = fifo_count_q == 0
            ? 0 : fifo_bank_id_q[fifo_read_q];
        mem_req_source_channel = fifo_count_q == 0
            ? 0 : fifo_channel_q[fifo_read_q];

        mem_flush_valid = flush_active_q && flush_req_pending_q;
        mem_flush_epoch = epoch_q;
        mem_flush_ack_ready = flush_active_q && flush_ack_wait_q;
        illegal_flush = soft_flush && (flush_active_q
            || epoch_q == {EPOCH_BITS{1'b1}});
        illegal_flush_ack = mem_flush_ack_valid
            && (!mem_flush_ack_ready || mem_flush_ack_epoch != epoch_q);
        flush_ack_timeout = flush_active_q && flush_ack_wait_q
            && flush_ack_timer_q == FLUSH_ACK_TIMEOUT_CYCLES-1
            && !(mem_flush_ack_valid && mem_flush_ack_ready
                 && mem_flush_ack_epoch == epoch_q);

        start_emit = token_active_q && frontend_done_seen_q
            && fifo_count_q == 0 && outstanding_count_q == 0
            && !rsp_skid_valid_q && !emit_q && !done_pending_q
            && !flush_active_q && !fault_q && !overflow_q;
    end

    assign protocol_error = fault_q || illegal_header || illegal_group
        || illegal_frontend_done || illegal_flush || illegal_flush_ack
        || flush_ack_timeout
        || (mem_rsp_valid && !flush_active_q && !soft_flush
            && !response_identity_legal)
        || generation_exhausted;
    assign numeric_overflow = overflow_q;
    assign stale_response_seen = stale_seen_q;
    assign busy = token_active_q || fifo_count_q != 0
        || outstanding_count_q != 0 || rsp_skid_valid_q || emit_q
        || done_pending_q || flush_active_q;

    assign result_valid = emit_q && !protocol_error && !numeric_overflow;
    assign result_accept = result_valid && result_ready;
    assign result_tag = token_tag_q;
    assign result_output_block = emit_block_q;
    assign result_slice = emit_slice_q;
    assign result_last = emit_q
        && emit_block_q + 1 == output_blocks_q
        && emit_slice_q == SLICES-1;
    generate
        for (genvar lane = 0; lane < SLICE_LANES; lane++) begin : g_result
            assign result_accumulator[lane]
                = ctx_valid_q[emit_block_q][emit_slice_q]
                ? ctx_q[emit_block_q][emit_slice_q][lane] : 24'sd0;
        end
    endgenerate

    assign token_done_valid = done_pending_q
        && !protocol_error && !numeric_overflow;
    assign token_done_accept = token_done_valid && token_done_ready;
    assign token_done_tag = token_tag_q;
    assign token_done_had_event = token_had_event_q;

    assign debug_fifo_count = fifo_count_q;
    assign debug_outstanding_count = outstanding_count_q;
    assign debug_group_accept_count = group_accept_count_q;
    assign debug_request_accept_count = request_accept_count_q;
    assign debug_response_accept_count = response_accept_count_q;
    assign debug_context_write_count = context_write_count_q;
    assign debug_result_accept_count = result_accept_count_q;
    assign debug_active_bank_read_count = active_bank_read_count_q;

    always_ff @(posedge clk_core) begin : service_state
        logic overflow_any;
        logic signed [23:0] accumulator_value;
        logic signed [24:0] extended_sum;
        if (rst_core) begin
            fault_q <= 0;
            fault_cause_q <= 0;
            overflow_q <= 0;
            stale_seen_q <= 0;
            epoch_q <= 0;
            generation_q <= 0;
            flush_active_q <= 0;
            flush_req_pending_q <= 0;
            flush_ack_wait_q <= 0;
            flush_ack_timer_q <= 0;
            token_active_q <= 0;
            frontend_done_seen_q <= 0;
            token_had_event_q <= 0;
            token_tag_q <= 0;
            output_blocks_q <= 0;
            fifo_read_q <= 0;
            fifo_write_q <= 0;
            fifo_count_q <= 0;
            head_slice_q <= 0;
            held_slot_valid_q <= 0;
            held_slot_q <= 0;
            outstanding_count_q <= 0;
            rsp_skid_valid_q <= 0;
            rsp_skid_block_q <= 0;
            rsp_skid_slice_q <= 0;
            emit_q <= 0;
            done_pending_q <= 0;
            emit_block_q <= 0;
            emit_slice_q <= 0;
            group_accept_count_q <= 0;
            request_accept_count_q <= 0;
            response_accept_count_q <= 0;
            context_write_count_q <= 0;
            result_accept_count_q <= 0;
            active_bank_read_count_q <= 0;
            for (int slot = 0; slot < OUTSTANDING; slot++) begin
                sb_valid_q[slot] <= 0;
                sb_epoch_q[slot] <= 0;
                sb_generation_q[slot] <= 0;
                sb_tag_q[slot] <= 0;
                sb_block_q[slot] <= 0;
                sb_slice_q[slot] <= 0;
                sb_bank_id_q[slot] <= 0;
            end
            for (int block = 0; block < 8; block++) begin
                for (int slice = 0; slice < SLICES; slice++) begin
                    ctx_busy_q[block][slice] <= 0;
                    ctx_valid_q[block][slice] <= 0;
                    for (int lane = 0; lane < SLICE_LANES; lane++)
                        ctx_q[block][slice][lane] <= 0;
                end
            end
            for (int lane = 0; lane < SLICE_LANES; lane++)
                rsp_skid_weight_q[lane] <= 0;
        end else begin
            if (illegal_header || illegal_group || illegal_frontend_done
                    || illegal_flush || illegal_flush_ack
                    || (mem_rsp_valid && !flush_active_q && !soft_flush
                        && !response_identity_legal)
                    || generation_exhausted)
                fault_q <= 1;
            if (illegal_header) fault_cause_q[0] <= 1;
            if (illegal_group) fault_cause_q[1] <= 1;
            if (illegal_frontend_done) fault_cause_q[2] <= 1;
            if (illegal_flush) fault_cause_q[3] <= 1;
            if (illegal_flush_ack) fault_cause_q[4] <= 1;
            if (mem_rsp_valid && !flush_active_q && !soft_flush
                    && !response_identity_legal) fault_cause_q[5] <= 1;
            if (generation_exhausted) fault_cause_q[6] <= 1;
            if (flush_ack_timeout) begin
                fault_q <= 1;
                fault_cause_q[7] <= 1;
            end

            if (soft_flush && !illegal_flush) begin
                epoch_q <= epoch_q + 1'b1;
                flush_active_q <= 1;
                flush_req_pending_q <= 1;
                flush_ack_wait_q <= 0;
                flush_ack_timer_q <= 0;
                token_active_q <= 0;
                frontend_done_seen_q <= 0;
                token_had_event_q <= 0;
                fifo_read_q <= 0;
                fifo_write_q <= 0;
                fifo_count_q <= 0;
                head_slice_q <= 0;
                held_slot_valid_q <= 0;
                outstanding_count_q <= 0;
                rsp_skid_valid_q <= 0;
                emit_q <= 0;
                done_pending_q <= 0;
                for (int slot = 0; slot < OUTSTANDING; slot++)
                    sb_valid_q[slot] <= 0;
                for (int block = 0; block < 8; block++) begin
                    for (int slice = 0; slice < SLICES; slice++) begin
                        ctx_busy_q[block][slice] <= 0;
                        ctx_valid_q[block][slice] <= 0;
                    end
                end
                if (mem_rsp_valid)
                    stale_seen_q <= 1;
            end else begin
                if (flush_active_q) begin
                    if (mem_flush_valid && mem_flush_ready) begin
                        flush_req_pending_q <= 0;
                        flush_ack_wait_q <= 1;
                        flush_ack_timer_q <= 0;
                    end
                    if (flush_ack_wait_q && !flush_ack_timeout)
                        flush_ack_timer_q <= flush_ack_timer_q + 1'b1;
                    if (mem_flush_ack_valid && mem_flush_ack_ready
                            && mem_flush_ack_epoch == epoch_q) begin
                        flush_active_q <= 0;
                        flush_ack_wait_q <= 0;
                        flush_ack_timer_q <= 0;
                    end
                    if (mem_rsp_accept)
                        stale_seen_q <= 1;
                end

                if (header_accept) begin
                    token_active_q <= 1;
                    frontend_done_seen_q <= 0;
                    token_had_event_q <= 0;
                    token_tag_q <= header_tag;
                    output_blocks_q <= header_output_blocks;
                    fifo_read_q <= 0;
                    fifo_write_q <= 0;
                    fifo_count_q <= 0;
                    head_slice_q <= 0;
                    held_slot_valid_q <= 0;
                    outstanding_count_q <= 0;
                    rsp_skid_valid_q <= 0;
                    emit_q <= 0;
                    done_pending_q <= 0;
                    emit_block_q <= 0;
                    emit_slice_q <= 0;
                    group_accept_count_q <= 0;
                    request_accept_count_q <= 0;
                    response_accept_count_q <= 0;
                    context_write_count_q <= 0;
                    result_accept_count_q <= 0;
                    active_bank_read_count_q <= 0;
                    for (int slot = 0; slot < OUTSTANDING; slot++)
                        sb_valid_q[slot] <= 0;
                    for (int block = 0; block < 8; block++) begin
                        for (int slice = 0; slice < SLICES; slice++) begin
                            ctx_busy_q[block][slice] <= 0;
                            ctx_valid_q[block][slice] <= 0;
                        end
                    end
                end

                if (group_accept) begin
                    fifo_tag_q[fifo_write_q] <= group_tag;
                    fifo_block_q[fifo_write_q] <= group_output_block;
                    fifo_bank_id_q[fifo_write_q] <= group_bank_id;
                    fifo_channel_q[fifo_write_q]
                        <= group_source_channel;
                    fifo_write_q <= fifo_write_q + 1'b1;
                    group_accept_count_q <= group_accept_count_q + 1'b1;
                end
                if (fifo_pop)
                    fifo_read_q <= fifo_read_q + 1'b1;
                case ({group_accept, fifo_pop})
                    2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                    2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                    default: fifo_count_q <= fifo_count_q;
                endcase
                if (mem_req_accept) begin
                    if (head_slice_q == SLICES-1)
                        head_slice_q <= 0;
                    else
                        head_slice_q <= head_slice_q + 1'b1;
                end

                if (mem_req_valid && !mem_req_ready
                        && !held_slot_valid_q) begin
                    held_slot_valid_q <= 1;
                    held_slot_q <= free_slot;
                end else if (mem_req_accept) begin
                    held_slot_valid_q <= 0;
                end

                if (legal_response_accept) begin
                    sb_valid_q[mem_rsp_slot] <= 0;
                    ctx_busy_q[sb_block_q[mem_rsp_slot]]
                        [sb_slice_q[mem_rsp_slot]] <= 0;
                end
                if (mem_req_accept) begin
                    sb_valid_q[mem_req_slot] <= 1;
                    sb_epoch_q[mem_req_slot] <= epoch_q;
                    sb_generation_q[mem_req_slot] <= generation_q;
                    sb_tag_q[mem_req_slot] <= mem_req_tag;
                    sb_block_q[mem_req_slot] <= mem_req_output_block;
                    sb_slice_q[mem_req_slot] <= mem_req_slice;
                    sb_bank_id_q[mem_req_slot] <= mem_req_bank_id;
                    ctx_busy_q[mem_req_output_block][mem_req_slice] <= 1;
                    generation_q <= generation_q + 1'b1;
                    request_accept_count_q <= request_accept_count_q + 1'b1;
                    active_bank_read_count_q <= active_bank_read_count_q
                        + 1'b1;
                end
                case ({mem_req_accept, legal_response_accept})
                    2'b10: outstanding_count_q
                        <= outstanding_count_q + 1'b1;
                    2'b01: outstanding_count_q
                        <= outstanding_count_q - 1'b1;
                    default: outstanding_count_q <= outstanding_count_q;
                endcase

                if (response_skid_commit && !legal_response_accept)
                    rsp_skid_valid_q <= 0;
                if (legal_response_accept) begin
                    rsp_skid_valid_q <= 1;
                    rsp_skid_block_q <= sb_block_q[mem_rsp_slot];
                    rsp_skid_slice_q <= sb_slice_q[mem_rsp_slot];
                    for (int lane = 0; lane < SLICE_LANES; lane++)
                        rsp_skid_weight_q[lane] <= mem_rsp_weight[lane];
                    response_accept_count_q <= response_accept_count_q + 1'b1;
                end

                if (response_skid_commit) begin
                    overflow_any = 0;
                    for (int lane = 0; lane < SLICE_LANES; lane++) begin
                        accumulator_value = ctx_valid_q[rsp_skid_block_q]
                            [rsp_skid_slice_q]
                            ? ctx_q[rsp_skid_block_q][rsp_skid_slice_q][lane]
                            : 24'sd0;
                        extended_sum = $signed({accumulator_value[23],
                                                accumulator_value})
                            + $signed({{17{rsp_skid_weight_q[lane][7]}},
                                      rsp_skid_weight_q[lane]});
                        ctx_q[rsp_skid_block_q][rsp_skid_slice_q][lane]
                            <= extended_sum[23:0];
                        if (extended_sum[24] != extended_sum[23])
                            overflow_any = 1;
                    end
                    ctx_valid_q[rsp_skid_block_q][rsp_skid_slice_q] <= 1;
                    context_write_count_q <= context_write_count_q + 1'b1;
                    if (overflow_any)
                        overflow_q <= 1;
                end

                if (frontend_done_accept) begin
                    frontend_done_seen_q <= 1;
                    token_had_event_q <= frontend_done_had_event;
                end
                if (start_emit) begin
                    emit_q <= 1;
                    emit_block_q <= 0;
                    emit_slice_q <= 0;
                end
                if (result_accept) begin
                    result_accept_count_q <= result_accept_count_q + 1'b1;
                    if (result_last) begin
                        emit_q <= 0;
                        done_pending_q <= 1;
                    end else if (emit_slice_q == SLICES-1) begin
                        emit_slice_q <= 0;
                        emit_block_q <= emit_block_q + 1'b1;
                    end else begin
                        emit_slice_q <= emit_slice_q + 1'b1;
                    end
                end
                if (token_done_accept) begin
                    done_pending_q <= 0;
                    token_active_q <= 0;
                    frontend_done_seen_q <= 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
