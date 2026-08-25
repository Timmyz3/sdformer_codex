`timescale 1ns/1ps
`default_nettype none

module qfit_dual_line_descriptor_stateful_engine_assertions #(
    parameter int STATE_CONTEXTS = 4,
    parameter int STATE_BASE_TILES = 32,
    parameter int STATE_BANKS = 6,
    parameter int STATE_LANES_PER_BANK = 16,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 32,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    parameter int STATE_QUEUE_DEPTH = 4,
    parameter int CTX_W = 2,
    parameter int STATE_CTX_W = (STATE_CONTEXTS <= 1) ? 1 :
        $clog2(STATE_CONTEXTS),
    parameter int STATE_BASE_TILE_W = (STATE_BASE_TILES <= 1) ? 1 :
        $clog2(STATE_BASE_TILES),
    parameter int STATE_QUEUE_PTR_W = (STATE_QUEUE_DEPTH <= 1) ? 1 :
        $clog2(STATE_QUEUE_DEPTH),
    parameter int STATE_QUEUE_COUNT_W = $clog2(STATE_QUEUE_DEPTH + 1),
    parameter int WIDE_ACC_BITS = STATE_BANKS*STATE_LANES_PER_BANK*ACC_W
) (
    input logic clk_core,
    input logic por_core,
    input logic rst_core,
    input logic descriptor_fire,
    input logic descriptor_row_first,
    input logic m4_output_valid,
    input logic m4_output_ready,
    input logic m4_output_use_motion,
    input logic [TAG_W-1:0] m4_output_tag,
    input logic [WIDE_ACC_BITS-1:0] m4_output_acc,
    input logic [STATE_CTX_W-1:0] enqueue_state_context,
    input logic [STATE_BASE_TILE_W-1:0] enqueue_state_base_tile,
    input logic [EPOCH_W-1:0] enqueue_epoch,
    input logic [DOMAIN_W-1:0] enqueue_domain,
    input logic [STEP_W-1:0] enqueue_step,
    input logic [LEN_W-1:0] enqueue_length,
    input logic enqueue_first,
    input logic enqueue_last,
    input logic [CTX_W-1:0] m4_output_context_q,
    input logic [CTX_W-1:0] emit_slot_q,
    input logic state_queue_push,
    input logic state_queue_pop,
    input logic [STATE_QUEUE_PTR_W-1:0] state_queue_head_q,
    input logic [STATE_QUEUE_PTR_W-1:0] state_queue_tail_q,
    input logic [STATE_QUEUE_COUNT_W-1:0] state_queue_count_q,
    input logic m9_wide_valid,
    input logic m9_wide_ready,
    input logic [STATE_CTX_W-1:0] state_request_context,
    input logic [STATE_BASE_TILE_W-1:0] state_request_base_tile,
    input logic [EPOCH_W-1:0] state_request_epoch,
    input logic [DOMAIN_W-1:0] state_request_domain,
    input logic [STEP_W-1:0] state_request_step,
    input logic [LEN_W-1:0] state_request_length,
    input logic state_request_first,
    input logic state_request_last,
    input logic state_request_use_motion,
    input logic [TAG_W-1:0] state_request_tag,
    input logic [WIDE_ACC_BITS-1:0] state_request_acc,
    input logic state_rmw_busy,
    input logic output_valid,
    input logic output_ready,
    input logic output_used_motion,
    input logic [STATE_CTX_W-1:0] output_state_context,
    input logic [STATE_BASE_TILE_W-1:0] output_state_base_tile,
    input logic [EPOCH_W-1:0] output_epoch,
    input logic [DOMAIN_W-1:0] output_domain,
    input logic [STEP_W-1:0] output_temporal_step,
    input logic [LEN_W-1:0] output_temporal_length,
    input logic output_temporal_first,
    input logic output_temporal_last,
    input logic [TAG_W-1:0] output_tag,
    input logic [WIDE_ACC_BITS-1:0] output_current_acc
);
    function automatic logic [STATE_QUEUE_PTR_W-1:0] next_queue_ptr(
        input logic [STATE_QUEUE_PTR_W-1:0] pointer
    );
        if ($unsigned(pointer) == STATE_QUEUE_DEPTH-1)
            next_queue_ptr = '0;
        else
            next_queue_ptr = pointer + 1'b1;
    endfunction

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (por_core || rst_core);

    ap_m4_accept_is_queue_push: assert property (
        (m4_output_valid && m4_output_ready) == state_queue_push)
        else $error("M4-state queue push diverged from M4 output handshake");

    ap_state_accept_is_queue_pop: assert property (
        (m9_wide_valid && m9_wide_ready) == state_queue_pop)
        else $error("M4-state queue pop diverged from state input handshake");

    ap_queue_count_bounded: assert property (
        state_queue_count_q <= STATE_QUEUE_DEPTH)
        else $error("M4-state transaction queue overflowed");

    ap_queue_head_in_range: assert property (
        $unsigned(state_queue_head_q) < STATE_QUEUE_DEPTH)
        else $error("M4-state queue head pointer left its parameterized range");

    ap_queue_tail_in_range: assert property (
        $unsigned(state_queue_tail_q) < STATE_QUEUE_DEPTH)
        else $error("M4-state queue tail pointer left its parameterized range");

    ap_no_empty_pop: assert property (
        state_queue_pop |-> state_queue_count_q != '0)
        else $error("M4-state transaction queue popped while empty");

    ap_no_full_push_without_pop: assert property (
        state_queue_count_q == STATE_QUEUE_COUNT_W'(STATE_QUEUE_DEPTH) &&
        !state_queue_pop |-> !state_queue_push)
        else $error("M4-state transaction queue pushed while full");

    ap_count_increments_on_push: assert property (
        state_queue_push && !state_queue_pop |=>
        state_queue_count_q == $past(state_queue_count_q) + 1'b1)
        else $error("M4-state queue count failed to increment");

    ap_count_decrements_on_pop: assert property (
        !state_queue_push && state_queue_pop |=>
        state_queue_count_q == $past(state_queue_count_q) - 1'b1)
        else $error("M4-state queue count failed to decrement");

    ap_count_stable_on_equal_flow: assert property (
        state_queue_push == state_queue_pop |=>
        state_queue_count_q == $past(state_queue_count_q))
        else $error("M4-state queue count changed on equal push/pop flow");

    ap_tail_advances_on_push: assert property (
        state_queue_push |=>
        state_queue_tail_q == next_queue_ptr($past(state_queue_tail_q)))
        else $error("M4-state queue tail pointer failed to advance/wrap");

    ap_head_advances_on_pop: assert property (
        state_queue_pop |=>
        state_queue_head_q == next_queue_ptr($past(state_queue_head_q)))
        else $error("M4-state queue head pointer failed to advance/wrap");

    ap_tail_stable_without_push: assert property (
        !state_queue_push |=> $stable(state_queue_tail_q))
        else $error("M4-state queue tail moved without push");

    ap_head_stable_without_pop: assert property (
        !state_queue_pop |=> $stable(state_queue_head_q))
        else $error("M4-state queue head moved without pop");

    ap_emit_slot_tracks_m4: assert property (
        m4_output_valid |-> emit_slot_q == m4_output_context_q)
        else $error("M4-state adapter slot diverged from M4 output context");

    ap_enqueue_identity_stable: assert property (
        m4_output_valid && !m4_output_ready |=>
        $stable({emit_slot_q, enqueue_state_context,
                 enqueue_state_base_tile, enqueue_epoch, enqueue_domain,
                 enqueue_step, enqueue_length, enqueue_first, enqueue_last,
                 m4_output_use_motion,
                 m4_output_tag, m4_output_acc}))
        else $error("M4-state enqueue payload changed under backpressure");

    ap_state_request_identity_stable: assert property (
        m9_wide_valid && !m9_wide_ready |=>
        m9_wide_valid &&
        $stable({state_request_context, state_request_base_tile,
                 state_request_epoch, state_request_domain,
                 state_request_step, state_request_length,
                 state_request_first, state_request_last,
                 state_request_use_motion, state_request_tag,
                 state_request_acc}))
        else $error("queued state request identity changed under backpressure");

    ap_integrated_output_stable: assert property (
        output_valid && !output_ready |=>
        output_valid && $stable({output_state_context,
            output_state_base_tile, output_epoch, output_domain,
            output_temporal_step, output_temporal_length,
            output_temporal_first, output_temporal_last,
            output_used_motion, output_tag, output_current_acc}))
        else $error("M4-state integrated output changed under backpressure");

    cp_local_commit: cover property (
        m4_output_valid && m4_output_ready && !m4_output_use_motion);
    cp_motion_commit: cover property (
        m4_output_valid && m4_output_ready && m4_output_use_motion);
    cp_queue_decouples_rmw: cover property (
        state_queue_push && !state_queue_pop && state_rmw_busy);
    cp_queue_push_pop_overlap: cover property (
        state_queue_push && state_queue_pop);
    cp_full_queue_push_pop_overlap: cover property (
        state_queue_count_q == STATE_QUEUE_COUNT_W'(STATE_QUEUE_DEPTH) &&
        state_queue_push && state_queue_pop);
    cp_tail_pointer_wrap: cover property (
        state_queue_push &&
        $unsigned(state_queue_tail_q) == STATE_QUEUE_DEPTH-1);
    cp_head_pointer_wrap: cover property (
        state_queue_pop &&
        $unsigned(state_queue_head_q) == STATE_QUEUE_DEPTH-1);
    cp_next_batch_accepts_with_state_pending: cover property (
        descriptor_fire && descriptor_row_first &&
        state_queue_count_q != '0);

endmodule

bind qfit_dual_line_descriptor_stateful_engine
    qfit_dual_line_descriptor_stateful_engine_assertions #(
        .STATE_CONTEXTS(STATE_CONTEXTS), .STATE_BASE_TILES(STATE_BASE_TILES),
        .STATE_BANKS(STATE_BANKS),
        .STATE_LANES_PER_BANK(STATE_LANES_PER_BANK), .ACC_W(ACC_W),
        .TAG_W(TAG_W), .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W),
        .STEP_W(STEP_W), .LEN_W(LEN_W),
        .STATE_QUEUE_DEPTH(STATE_QUEUE_DEPTH), .CTX_W(CTX_W),
        .STATE_CTX_W(STATE_CTX_W),
        .STATE_BASE_TILE_W(STATE_BASE_TILE_W),
        .STATE_QUEUE_PTR_W(STATE_QUEUE_PTR_W),
        .STATE_QUEUE_COUNT_W(STATE_QUEUE_COUNT_W),
        .WIDE_ACC_BITS(OUT_LANES*ACC_W)
    ) u_m4_state_assertions (
        .clk_core, .por_core, .rst_core,
        .descriptor_fire, .descriptor_row_first,
        .m4_output_valid, .m4_output_ready, .m4_output_use_motion,
        .m4_output_tag, .m4_output_acc,
        .enqueue_state_context(meta_context_q[emit_slot_q]),
        .enqueue_state_base_tile(
            m9_base_tile_sum[STATE_BASE_TILE_W-1:0]),
        .enqueue_epoch(meta_epoch_q[emit_slot_q]),
        .enqueue_domain(meta_domain_q[emit_slot_q]),
        .enqueue_step(meta_step_q[emit_slot_q]),
        .enqueue_length(meta_length_q[emit_slot_q]),
        .enqueue_first(meta_first_q[emit_slot_q]),
        .enqueue_last(meta_last_q[emit_slot_q]),
        .m4_output_context_q(u_m4.output_context_q), .emit_slot_q,
        .state_queue_push, .state_queue_pop,
        .state_queue_head_q, .state_queue_tail_q, .state_queue_count_q,
        .m9_wide_valid, .m9_wide_ready,
        .state_request_context, .state_request_base_tile,
        .state_request_epoch, .state_request_domain,
        .state_request_step, .state_request_length,
        .state_request_first, .state_request_last,
        .state_request_use_motion, .state_request_tag,
        .state_request_acc, .state_rmw_busy,
        .output_valid, .output_ready, .output_used_motion,
        .output_state_context, .output_state_base_tile, .output_epoch,
        .output_domain, .output_temporal_step, .output_temporal_length,
        .output_temporal_first, .output_temporal_last, .output_tag,
        .output_current_acc
    );

`default_nettype wire
