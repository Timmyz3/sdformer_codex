`timescale 1ns/1ps
`default_nettype none

// M4 descriptor-resident execution with its Local/Motion destination state
// closed through the shared synchronous-bank fabric.  This is an integration
// revision of the M4 paper object, not a second accelerator mechanism.
module qfit_dual_line_descriptor_stateful_engine #(
    parameter int TILE_BITS = 256,
    parameter int MAX_CHUNKS = 12,
    parameter int MAX_LANE_TILES = 32,
    parameter int ISSUE_WIDTH = 16,
    parameter int CONTEXTS = 4,
    parameter int REDUCE_SLOTS = 4,
    parameter int OUT_LANES = 96,
    parameter int TAG_W = 32,
    parameter int OBJECT_W = 64,
    parameter int W_W = 8,
    parameter int ACC_W = 32,
    parameter int STATE_CONTEXTS = 4,
    parameter int STATE_BASE_TILES = 32,
    parameter int STATE_BANKS = 6,
    parameter int STATE_LANES_PER_BANK = 16,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 32,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    // M4 only issues coherent 96-lane wide transactions.  Share temporal
    // metadata across the six data banks by default; retain the legacy
    // dual-granularity state engine as an explicit equivalence ablation.
    parameter bit USE_SHARED_WIDE_METADATA = 1'b1,
    // A short transaction queue lets the single-buffer M4 controller begin
    // the next batch while a prior Motion output completes its synchronous
    // state read-modify-write.  Every entry carries the full persistent-state
    // identity, so descriptor metadata may be safely reused by the next batch.
    parameter int STATE_QUEUE_DEPTH = 4,
    parameter int INDEX_W = (TILE_BITS <= 1) ? 1 : $clog2(TILE_BITS),
    parameter int BANK_BITS = (ISSUE_WIDTH <= 1) ? 0 : $clog2(ISSUE_WIDTH),
    parameter int BANK_ADDR_W = INDEX_W - BANK_BITS,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int CTX_COUNT_W = $clog2(CONTEXTS + 1),
    parameter int CHUNK_W = (MAX_CHUNKS <= 1) ? 1 : $clog2(MAX_CHUNKS),
    parameter int CHUNK_COUNT_W = $clog2(MAX_CHUNKS + 1),
    parameter int LANE_TILE_W = (MAX_LANE_TILES <= 1) ? 1 : $clog2(MAX_LANE_TILES),
    parameter int LANE_COUNT_W = $clog2(MAX_LANE_TILES + 1),
    parameter int SLOT_W = (REDUCE_SLOTS <= 1) ? 1 : $clog2(REDUCE_SLOTS),
    parameter int SOURCE_COUNT_W = $clog2(MAX_CHUNKS*TILE_BITS + 1),
    parameter int STATE_CTX_W = (STATE_CONTEXTS <= 1) ? 1 : $clog2(STATE_CONTEXTS),
    parameter int STATE_BASE_TILE_W = (STATE_BASE_TILES <= 1) ? 1 :
        $clog2(STATE_BASE_TILES),
    parameter int STATE_BANK_W = (STATE_BANKS <= 1) ? 1 : $clog2(STATE_BANKS),
    parameter int STATE_BANK_ACC_BITS = STATE_LANES_PER_BANK * ACC_W,
    parameter int STATE_QUEUE_PTR_W = (STATE_QUEUE_DEPTH <= 1) ? 1 :
        $clog2(STATE_QUEUE_DEPTH),
    parameter int STATE_QUEUE_COUNT_W = $clog2(STATE_QUEUE_DEPTH + 1)
) (
    input  logic                                      clk_core,
    input  logic                                      por_core,
    input  logic                                      rst_core,
    input  logic [DOMAIN_W-1:0]                       active_domain,

    input  logic                                      descriptor_valid,
    output logic                                      descriptor_ready,
    input  logic                                      descriptor_row_first,
    input  logic                                      descriptor_row_last,
    input  logic                                      descriptor_batch_last,
    input  logic [TAG_W-1:0]                          descriptor_tag,
    input  logic [OBJECT_W-1:0]                       descriptor_object_tag,
    input  logic [CHUNK_W-1:0]                        descriptor_chunk_index,
    input  logic [CHUNK_COUNT_W-1:0]                  descriptor_chunk_count,
    input  logic [LANE_COUNT_W-1:0]                   descriptor_lane_tile_count,
    input  logic                                      descriptor_use_motion,
    input  logic [TILE_BITS-1:0]                      descriptor_source_bits,
    input  logic [TILE_BITS-1:0]                      descriptor_negative_bits,
    input  logic [STATE_CTX_W-1:0]                    descriptor_state_context,
    input  logic [STATE_BASE_TILE_W-1:0]              descriptor_state_base_tile,
    input  logic [EPOCH_W-1:0]                        descriptor_epoch,
    input  logic [DOMAIN_W-1:0]                       descriptor_domain,
    input  logic [STEP_W-1:0]                         descriptor_temporal_step,
    input  logic [LEN_W-1:0]                          descriptor_temporal_length,
    input  logic                                      descriptor_temporal_first,
    input  logic                                      descriptor_temporal_last,

    output logic                                      weight_request_valid,
    input  logic                                      weight_request_ready,
    output logic [OBJECT_W-1:0]                       weight_request_object_tag,
    output logic [CHUNK_W-1:0]                        weight_request_chunk_index,
    output logic [LANE_TILE_W-1:0]                    weight_request_lane_tile,
    output logic [ISSUE_WIDTH-1:0]                    weight_request_bank_valid,
    output logic [ISSUE_WIDTH*BANK_ADDR_W-1:0]        weight_request_bank_addr,
    output logic [ISSUE_WIDTH*CTX_W-1:0]              weight_request_bank_context,
    output logic [ISSUE_WIDTH*SLOT_W-1:0]             weight_request_bank_slot,
    output logic [ISSUE_WIDTH-1:0]                    weight_request_bank_negative,
    input  logic                                      weight_response_valid,
    output logic                                      weight_response_ready,
    input  logic [ISSUE_WIDTH-1:0]                    weight_response_bank_valid,
    input  logic [ISSUE_WIDTH*OUT_LANES*W_W-1:0]     weight_response_data,

    output logic                                      output_valid,
    input  logic                                      output_ready,
    output logic [STATE_CTX_W-1:0]                    output_state_context,
    output logic [STATE_BASE_TILE_W-1:0]              output_state_base_tile,
    output logic [EPOCH_W-1:0]                        output_epoch,
    output logic [DOMAIN_W-1:0]                       output_domain,
    output logic [STEP_W-1:0]                         output_temporal_step,
    output logic [LEN_W-1:0]                          output_temporal_length,
    output logic                                      output_temporal_first,
    output logic                                      output_temporal_last,
    output logic                                      output_used_motion,
    output logic [TAG_W-1:0]                          output_tag,
    output logic [OUT_LANES*ACC_W-1:0]                output_current_acc,

    output logic [2:0]                                controller_state,
    output logic [CTX_COUNT_W-1:0]                    resident_contexts,
    output logic                                      state_rmw_busy,
    output logic                                      domain_fence_ready,
    output logic                                      domain_fence_error,
    output logic                                      protocol_error
);
    logic m4_descriptor_valid;
    logic m4_descriptor_ready;
    logic m4_reset;
    logic m4_output_valid;
    logic m4_output_ready;
    logic [TAG_W-1:0] m4_output_tag;
    logic [OBJECT_W-1:0] m4_output_object_tag;
    logic [LANE_TILE_W-1:0] m4_output_lane_tile;
    logic m4_output_use_motion;
    logic [SOURCE_COUNT_W-1:0] m4_output_source_count;
    logic [OUT_LANES*ACC_W-1:0] m4_output_acc;
    logic m4_protocol_error;

    logic [CTX_W-1:0] load_slot_q;
    logic [CTX_W-1:0] emit_slot_q;
    logic adapter_fault_q;
    logic descriptor_contract_valid;
    logic descriptor_fire;
    logic metadata_index_valid;
    logic metadata_time_valid;
    logic metadata_nonoverlap_valid;
    logic load_slot_valid;
    logic [STATE_BASE_TILE_W:0] metadata_tile_start;
    logic [STATE_BASE_TILE_W:0] metadata_tile_limit;

    logic [STATE_CTX_W-1:0] meta_context_q [0:CONTEXTS-1];
    logic [STATE_BASE_TILE_W-1:0] meta_base_tile_q [0:CONTEXTS-1];
    logic [EPOCH_W-1:0] meta_epoch_q [0:CONTEXTS-1];
    logic [DOMAIN_W-1:0] meta_domain_q [0:CONTEXTS-1];
    logic [STEP_W-1:0] meta_step_q [0:CONTEXTS-1];
    logic [LEN_W-1:0] meta_length_q [0:CONTEXTS-1];
    logic meta_first_q [0:CONTEXTS-1];
    logic meta_last_q [0:CONTEXTS-1];
    logic [LANE_COUNT_W-1:0] meta_lane_tiles_q [0:CONTEXTS-1];

    logic [STATE_BASE_TILE_W:0] m9_base_tile_sum;
    logic m9_base_tile_valid;
    logic m9_wide_valid;
    logic m9_wide_ready;
    logic m9_wide_protocol_error;
    logic m9_narrow_ready;
    logic m9_abort_ready;
    logic m9_abort_error;
    logic m9_output_is_wide;
    logic [STATE_BANKS-1:0] m9_output_bank_mask;
    logic m9_output_used_motion;
    logic m9_narrow_protocol_error;

    logic [STATE_QUEUE_PTR_W-1:0] state_queue_head_q;
    logic [STATE_QUEUE_PTR_W-1:0] state_queue_tail_q;
    logic [STATE_QUEUE_COUNT_W-1:0] state_queue_count_q;
    logic state_queue_full;
    logic state_queue_empty;
    logic state_queue_push_ready;
    logic state_queue_push;
    logic state_queue_pop;
    logic [STATE_CTX_W-1:0] state_queue_context_q
        [0:STATE_QUEUE_DEPTH-1];
    logic [STATE_BASE_TILE_W-1:0] state_queue_base_tile_q
        [0:STATE_QUEUE_DEPTH-1];
    logic [EPOCH_W-1:0] state_queue_epoch_q
        [0:STATE_QUEUE_DEPTH-1];
    logic [DOMAIN_W-1:0] state_queue_domain_q
        [0:STATE_QUEUE_DEPTH-1];
    logic [STEP_W-1:0] state_queue_step_q
        [0:STATE_QUEUE_DEPTH-1];
    logic [LEN_W-1:0] state_queue_length_q
        [0:STATE_QUEUE_DEPTH-1];
    logic state_queue_first_q [0:STATE_QUEUE_DEPTH-1];
    logic state_queue_last_q [0:STATE_QUEUE_DEPTH-1];
    logic state_queue_use_motion_q [0:STATE_QUEUE_DEPTH-1];
    logic [TAG_W-1:0] state_queue_tag_q [0:STATE_QUEUE_DEPTH-1];
    logic [OUT_LANES*ACC_W-1:0] state_queue_acc_q
        [0:STATE_QUEUE_DEPTH-1];

    logic [STATE_CTX_W-1:0] state_request_context;
    logic [STATE_BASE_TILE_W-1:0] state_request_base_tile;
    logic [EPOCH_W-1:0] state_request_epoch;
    logic [DOMAIN_W-1:0] state_request_domain;
    logic [STEP_W-1:0] state_request_step;
    logic [LEN_W-1:0] state_request_length;
    logic state_request_first;
    logic state_request_last;
    logic state_request_use_motion;
    logic [TAG_W-1:0] state_request_tag;
    logic [OUT_LANES*ACC_W-1:0] state_request_acc;

    function automatic logic [STATE_QUEUE_PTR_W-1:0] state_queue_next_ptr(
        input logic [STATE_QUEUE_PTR_W-1:0] pointer
    );
        if ($unsigned(pointer) == STATE_QUEUE_DEPTH-1)
            state_queue_next_ptr = '0;
        else
            state_queue_next_ptr = pointer + 1'b1;
    endfunction

    initial begin
        if (OUT_LANES != STATE_BANKS * STATE_LANES_PER_BANK)
            $error("M4 output lanes must exactly cover the state-bank vector");
        if (CONTEXTS > STATE_CONTEXTS)
            $error("transient M4 contexts exceed persistent state contexts");
        if (STATE_QUEUE_DEPTH < 1)
            $error("state transaction queue depth must be positive");
    end

    assign metadata_tile_limit =
        (STATE_BASE_TILE_W+1)'($unsigned(descriptor_state_base_tile)) +
        (STATE_BASE_TILE_W+1)'($unsigned(descriptor_lane_tile_count));
    assign metadata_tile_start =
        (STATE_BASE_TILE_W+1)'($unsigned(descriptor_state_base_tile));
    assign metadata_index_valid =
        load_slot_valid &&
        ($unsigned(descriptor_state_context) < STATE_CONTEXTS) &&
        (descriptor_lane_tile_count != '0) &&
        (metadata_tile_limit <= (STATE_BASE_TILE_W+1)'(STATE_BASE_TILES));
    assign metadata_time_valid = (descriptor_domain == active_domain) &&
        ((descriptor_temporal_length == LEN_W'(2)) ||
         (descriptor_temporal_length == LEN_W'(10))) &&
        (descriptor_temporal_last ==
         (descriptor_temporal_step == descriptor_temporal_length - 1'b1)) &&
        (!descriptor_temporal_first ||
         (!descriptor_use_motion && descriptor_temporal_step == '0 &&
          !descriptor_temporal_last));
    assign load_slot_valid = $unsigned(load_slot_q) < CONTEXTS;

    always_comb begin
        metadata_nonoverlap_valid = 1'b1;
        for (int prior = 0; prior < CONTEXTS; prior = prior + 1) begin
            logic [STATE_BASE_TILE_W:0] prior_tile_start;
            logic [STATE_BASE_TILE_W:0] prior_tile_limit;
            prior_tile_start = (STATE_BASE_TILE_W+1)'(
                $unsigned(meta_base_tile_q[prior]));
            prior_tile_limit = prior_tile_start +
                (STATE_BASE_TILE_W+1)'($unsigned(meta_lane_tiles_q[prior]));
            if (CTX_W'(prior) < load_slot_q &&
                    descriptor_state_context == meta_context_q[prior] &&
                    metadata_tile_limit > prior_tile_start &&
                    prior_tile_limit > metadata_tile_start)
                metadata_nonoverlap_valid = 1'b0;
        end
    end

    always_comb begin
        if (descriptor_row_first) begin
            descriptor_contract_valid = metadata_index_valid &&
                metadata_time_valid && metadata_nonoverlap_valid;
        end else begin
            descriptor_contract_valid =
                descriptor_state_context == meta_context_q[load_slot_q] &&
                descriptor_state_base_tile == meta_base_tile_q[load_slot_q] &&
                descriptor_epoch == meta_epoch_q[load_slot_q] &&
                descriptor_domain == meta_domain_q[load_slot_q] &&
                descriptor_temporal_step == meta_step_q[load_slot_q] &&
                descriptor_temporal_length == meta_length_q[load_slot_q] &&
                descriptor_temporal_first == meta_first_q[load_slot_q] &&
                descriptor_temporal_last == meta_last_q[load_slot_q] &&
                descriptor_lane_tile_count == meta_lane_tiles_q[load_slot_q];
        end
    end

    assign m4_descriptor_valid = descriptor_valid &&
                                 descriptor_contract_valid &&
                                 !adapter_fault_q;
    assign descriptor_ready = m4_descriptor_ready &&
                              descriptor_contract_valid &&
                              !adapter_fault_q;
    assign descriptor_fire = descriptor_valid && descriptor_ready;

    assign m4_reset = por_core || rst_core;

    always_ff @(posedge clk_core) begin
        if (m4_reset) begin
            load_slot_q <= '0;
            emit_slot_q <= '0;
            adapter_fault_q <= 1'b0;
            state_queue_head_q <= '0;
            state_queue_tail_q <= '0;
            state_queue_count_q <= '0;
            for (int ctx = 0; ctx < CONTEXTS; ctx = ctx + 1) begin
                meta_context_q[ctx] <= '0;
                meta_base_tile_q[ctx] <= '0;
                meta_epoch_q[ctx] <= '0;
                meta_domain_q[ctx] <= '0;
                meta_step_q[ctx] <= '0;
                meta_length_q[ctx] <= '0;
                meta_first_q[ctx] <= 1'b0;
                meta_last_q[ctx] <= 1'b0;
                meta_lane_tiles_q[ctx] <= '0;
            end
        end else begin
            if (descriptor_valid && m4_descriptor_ready &&
                    !descriptor_contract_valid)
                adapter_fault_q <= 1'b1;
            if (m4_output_valid && !m9_base_tile_valid)
                adapter_fault_q <= 1'b1;
            if (descriptor_fire && descriptor_row_first) begin
                meta_context_q[load_slot_q] <= descriptor_state_context;
                meta_base_tile_q[load_slot_q] <= descriptor_state_base_tile;
                meta_epoch_q[load_slot_q] <= descriptor_epoch;
                meta_domain_q[load_slot_q] <= descriptor_domain;
                meta_step_q[load_slot_q] <= descriptor_temporal_step;
                meta_length_q[load_slot_q] <= descriptor_temporal_length;
                meta_first_q[load_slot_q] <= descriptor_temporal_first;
                meta_last_q[load_slot_q] <= descriptor_temporal_last;
                meta_lane_tiles_q[load_slot_q] <=
                    descriptor_lane_tile_count;
            end
            if (descriptor_fire && descriptor_row_last) begin
                if (descriptor_batch_last)
                    load_slot_q <= '0;
                else if (CTX_COUNT_W'(load_slot_q) + CTX_COUNT_W'(1) <
                        CTX_COUNT_W'(CONTEXTS))
                    load_slot_q <= load_slot_q + 1'b1;
                else begin
                    load_slot_q <= load_slot_q;
                    adapter_fault_q <= 1'b1;
                end
            end
            if (state_queue_push) begin
                if (CTX_COUNT_W'(emit_slot_q) + CTX_COUNT_W'(1) <
                        resident_contexts)
                    emit_slot_q <= emit_slot_q + 1'b1;
                else
                    emit_slot_q <= '0;
            end

            case ({state_queue_push, state_queue_pop})
                2'b10: state_queue_count_q <= state_queue_count_q + 1'b1;
                2'b01: state_queue_count_q <= state_queue_count_q - 1'b1;
                default: state_queue_count_q <= state_queue_count_q;
            endcase
            if (state_queue_push) begin
                state_queue_context_q[state_queue_tail_q] <=
                    meta_context_q[emit_slot_q];
                state_queue_base_tile_q[state_queue_tail_q] <=
                    m9_base_tile_sum[STATE_BASE_TILE_W-1:0];
                state_queue_epoch_q[state_queue_tail_q] <=
                    meta_epoch_q[emit_slot_q];
                state_queue_domain_q[state_queue_tail_q] <=
                    meta_domain_q[emit_slot_q];
                state_queue_step_q[state_queue_tail_q] <=
                    meta_step_q[emit_slot_q];
                state_queue_length_q[state_queue_tail_q] <=
                    meta_length_q[emit_slot_q];
                state_queue_first_q[state_queue_tail_q] <=
                    meta_first_q[emit_slot_q];
                state_queue_last_q[state_queue_tail_q] <=
                    meta_last_q[emit_slot_q];
                state_queue_use_motion_q[state_queue_tail_q] <=
                    m4_output_use_motion;
                state_queue_tag_q[state_queue_tail_q] <= m4_output_tag;
                state_queue_acc_q[state_queue_tail_q] <= m4_output_acc;
                state_queue_tail_q <= state_queue_next_ptr(state_queue_tail_q);
            end
            if (state_queue_pop)
                state_queue_head_q <= state_queue_next_ptr(state_queue_head_q);
        end
    end

    qfit_dual_line_descriptor_resident_engine #(
        .TILE_BITS(TILE_BITS), .MAX_CHUNKS(MAX_CHUNKS),
        .MAX_LANE_TILES(MAX_LANE_TILES), .ISSUE_WIDTH(ISSUE_WIDTH),
        .CONTEXTS(CONTEXTS), .REDUCE_SLOTS(REDUCE_SLOTS),
        .OUT_LANES(OUT_LANES), .TAG_W(TAG_W), .OBJECT_W(OBJECT_W),
        .W_W(W_W), .ACC_W(ACC_W)
    ) u_m4 (
        .clk_core, .rst_core(m4_reset),
        .descriptor_valid(m4_descriptor_valid),
        .descriptor_ready(m4_descriptor_ready),
        .descriptor_row_first, .descriptor_row_last, .descriptor_batch_last,
        .descriptor_tag, .descriptor_object_tag, .descriptor_chunk_index,
        .descriptor_chunk_count, .descriptor_lane_tile_count,
        .descriptor_use_motion, .descriptor_source_bits,
        .descriptor_negative_bits,
        .weight_request_valid, .weight_request_ready,
        .weight_request_object_tag, .weight_request_chunk_index,
        .weight_request_lane_tile, .weight_request_bank_valid,
        .weight_request_bank_addr, .weight_request_bank_context,
        .weight_request_bank_slot, .weight_request_bank_negative,
        .weight_response_valid, .weight_response_ready,
        .weight_response_bank_valid, .weight_response_data,
        .output_valid(m4_output_valid), .output_ready(m4_output_ready),
        .output_tag(m4_output_tag),
        .output_object_tag(m4_output_object_tag),
        .output_lane_tile(m4_output_lane_tile),
        .output_use_motion(m4_output_use_motion),
        .output_source_count(m4_output_source_count),
        .output_acc(m4_output_acc), .controller_state, .resident_contexts,
        .protocol_error(m4_protocol_error)
    );

    assign m9_base_tile_sum =
        (STATE_BASE_TILE_W+1)'($unsigned(meta_base_tile_q[emit_slot_q])) +
        (STATE_BASE_TILE_W+1)'($unsigned(m4_output_lane_tile));
    assign m9_base_tile_valid =
        m9_base_tile_sum < (STATE_BASE_TILE_W+1)'(STATE_BASE_TILES);
    assign state_queue_full =
        state_queue_count_q == STATE_QUEUE_COUNT_W'(STATE_QUEUE_DEPTH);
    assign state_queue_empty = state_queue_count_q == '0;
    assign state_queue_push_ready = !state_queue_full || state_queue_pop;
    assign state_queue_push = m4_output_valid && m4_output_ready;
    assign state_queue_pop = m9_wide_valid && m9_wide_ready;
    assign m4_output_ready = !adapter_fault_q && m9_base_tile_valid &&
                             state_queue_push_ready;

    assign m9_wide_valid = !state_queue_empty && !adapter_fault_q;
    assign state_request_context = state_queue_context_q[state_queue_head_q];
    assign state_request_base_tile =
        state_queue_base_tile_q[state_queue_head_q];
    assign state_request_epoch = state_queue_epoch_q[state_queue_head_q];
    assign state_request_domain = state_queue_domain_q[state_queue_head_q];
    assign state_request_step = state_queue_step_q[state_queue_head_q];
    assign state_request_length = state_queue_length_q[state_queue_head_q];
    assign state_request_first = state_queue_first_q[state_queue_head_q];
    assign state_request_last = state_queue_last_q[state_queue_head_q];
    assign state_request_use_motion =
        state_queue_use_motion_q[state_queue_head_q];
    assign state_request_tag = state_queue_tag_q[state_queue_head_q];
    assign state_request_acc = state_queue_acc_q[state_queue_head_q];

    generate
        if (USE_SHARED_WIDE_METADATA) begin : g_shared_wide_state
            qfit_wide_temporal_state_engine #(
                .CONTEXTS(STATE_CONTEXTS), .BASE_TILES(STATE_BASE_TILES),
                .BANKS(STATE_BANKS),
                .LANES_PER_BANK(STATE_LANES_PER_BANK), .ACC_W(ACC_W),
                .TAG_W(TAG_W), .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W),
                .STEP_W(STEP_W), .LEN_W(LEN_W)
            ) u_state (
                .clk_core, .por_core, .rst_core, .active_domain,
                .domain_fence_ready, .domain_fence_error,
                .request_valid(m9_wide_valid),
                .request_ready(m9_wide_ready),
                .request_context(state_request_context),
                .request_base_tile(state_request_base_tile),
                .request_epoch(state_request_epoch),
                .request_domain(state_request_domain),
                .request_temporal_step(state_request_step),
                .request_temporal_length(state_request_length),
                .request_temporal_first(state_request_first),
                .request_temporal_last(state_request_last),
                .request_use_motion(state_request_use_motion),
                .request_tag(state_request_tag), .request_acc(state_request_acc),
                .output_valid, .output_ready,
                .output_context(output_state_context),
                .output_base_tile(output_state_base_tile), .output_epoch,
                .output_domain, .output_temporal_step,
                .output_temporal_length, .output_temporal_first,
                .output_temporal_last,
                .output_used_motion(m9_output_used_motion), .output_tag,
                .output_current_acc, .rmw_busy(state_rmw_busy),
                .protocol_error(m9_wide_protocol_error)
            );
            assign m9_narrow_ready = 1'b0;
            assign m9_abort_ready = 1'b0;
            assign m9_abort_error = 1'b0;
            assign m9_narrow_protocol_error = 1'b0;
            assign m9_output_is_wide = 1'b1;
            assign m9_output_bank_mask = {STATE_BANKS{1'b1}};
        end else begin : g_legacy_dual_state
            qfit_dual_granularity_temporal_state_engine #(
                .CONTEXTS(STATE_CONTEXTS), .BASE_TILES(STATE_BASE_TILES),
                .BANKS(STATE_BANKS),
                .LANES_PER_BANK(STATE_LANES_PER_BANK), .ACC_W(ACC_W),
                .TAG_W(TAG_W), .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W),
                .STEP_W(STEP_W), .LEN_W(LEN_W)
            ) u_state (
                .clk_core, .por_core, .rst_core, .active_domain,
                .domain_fence_ready, .domain_fence_error,
                .wide_valid(m9_wide_valid), .wide_ready(m9_wide_ready),
                .wide_context(state_request_context),
                .wide_base_tile(state_request_base_tile),
                .wide_epoch(state_request_epoch),
                .wide_domain(state_request_domain),
                .wide_temporal_step(state_request_step),
                .wide_temporal_length(state_request_length),
                .wide_temporal_first(state_request_first),
                .wide_temporal_last(state_request_last),
                .wide_use_motion(state_request_use_motion),
                .wide_tag(state_request_tag), .wide_acc(state_request_acc),
                .narrow_valid(1'b0), .narrow_ready(m9_narrow_ready),
                .narrow_context('0), .narrow_base_tile('0), .narrow_bank('0),
                .narrow_epoch('0), .narrow_domain('0),
                .narrow_temporal_step('0), .narrow_temporal_length('0),
                .narrow_temporal_first(1'b0), .narrow_temporal_last(1'b0),
                .narrow_use_motion(1'b0), .narrow_tag('0), .narrow_acc('0),
                .abort_valid(1'b0), .abort_ready(m9_abort_ready),
                .abort_context('0), .abort_base_tile('0),
                .abort_bank_mask('0), .abort_epoch('0), .abort_domain('0),
                .abort_tag('0), .abort_error(m9_abort_error),
                .output_valid, .output_ready,
                .output_is_wide(m9_output_is_wide),
                .output_context(output_state_context),
                .output_base_tile(output_state_base_tile),
                .output_bank_mask(m9_output_bank_mask), .output_epoch,
                .output_domain, .output_temporal_step,
                .output_temporal_length, .output_temporal_first,
                .output_temporal_last,
                .output_used_motion(m9_output_used_motion), .output_tag,
                .output_current_acc, .rmw_busy(state_rmw_busy),
                .wide_protocol_error(m9_wide_protocol_error),
                .narrow_protocol_error(m9_narrow_protocol_error)
            );
        end
    endgenerate

    assign output_used_motion = m9_output_used_motion;
    assign protocol_error = adapter_fault_q || m4_protocol_error ||
        m9_wide_protocol_error || m9_narrow_protocol_error ||
        m9_abort_error || domain_fence_error ||
        (output_valid && (!m9_output_is_wide || !(&m9_output_bank_mask)));
endmodule

`default_nettype wire
