`timescale 1ns/1ps
`default_nettype none

module qfit_dynamic_bn_banked_moment_scheduler_assertions #(
    parameter int IN_W = 32,
    parameter int TAG_W = 48,
    parameter int MAX_REDUCTION_POPULATION = 4194304,
    parameter int MAX_LANE_TILES = 16,
    localparam int COUNT_W = $clog2(MAX_REDUCTION_POPULATION + 1),
    localparam int LANE_TILE_W = (MAX_LANE_TILES <= 2) ? 1 : $clog2(MAX_LANE_TILES),
    localparam int ACTIVE_TILES_W = $clog2(MAX_LANE_TILES + 1),
    localparam int POP_GROWTH_W =
        (MAX_REDUCTION_POPULATION <= 1) ? 0 : $clog2(MAX_REDUCTION_POPULATION),
    localparam int SUM_W = IN_W + POP_GROWTH_W,
    localparam int SUMSQ_W = (2 * IN_W) - 1 + POP_GROWTH_W,
    localparam int RESULT_TARGET_W = $clog2((MAX_LANE_TILES*6) + 1)
) (
    input logic clk_core,
    input logic rst_core,
    input logic operator_start_valid,
    input logic operator_start_ready,
    input logic [COUNT_W-1:0] operator_reduction_population,
    input logic [ACTIVE_TILES_W-1:0] operator_active_lane_tiles,
    input logic operator_start_legal,
    input logic operator_active,
    input logic [COUNT_W-1:0] active_reduction_population,
    input logic packet_valid,
    input logic packet_ready,
    input logic packet_first,
    input logic packet_last,
    input logic packet_legal,
    input logic [COUNT_W-1:0] packet_accepted_count,
    input logic result_valid,
    input logic result_ready,
    input logic [TAG_W-1:0] result_tag,
    input logic [LANE_TILE_W-1:0] result_lane_tile_id,
    input logic [2:0] result_slice_id,
    input logic [COUNT_W-1:0] result_count,
    input logic [(16*SUM_W)-1:0] result_sum,
    input logic [(16*SUMSQ_W)-1:0] result_sumsq,
    input logic operator_done,
    input logic protocol_error,
    input logic [2:0] fifo_level,
    input logic [2:0] serializer_slice,
    input logic arithmetic_update,
    input logic dequeue_candidate,
    input logic dequeue_fire,
    input logic enqueue_fire,
    input logic illegal_packet_fire,
    input logic result_valid_internal,
    input logic result_fire_internal,
    input logic [RESULT_TARGET_W-1:0] results_retired,
    input logic [1:0] fifo_read_ptr,
    input logic [1:0] fifo_write_ptr
);
    integer legal_packets_seen = 0;
    integer illegal_packets_seen = 0;
    integer results_seen = 0;
    integer result_stalls_seen = 0;
    integer fifo_full_seen = 0;
    integer done_seen = 0;
    integer full_swap_seen = 0;
    integer pending_result_cancel_seen = 0;

    wire start_fire = operator_start_valid && operator_start_ready;
    wire packet_fire = packet_valid && packet_ready;

    property p_start_legality_exact;
        @(posedge clk_core)
            operator_start_legal ==
                (operator_start_ready
                 && operator_reduction_population != 0
                 && operator_reduction_population <= MAX_REDUCTION_POPULATION
                 && operator_active_lane_tiles != 0
                 && operator_active_lane_tiles <= MAX_LANE_TILES);
    endproperty

    property p_legal_packet_first_exact;
        @(posedge clk_core) disable iff (rst_core)
            packet_fire && packet_legal |->
                packet_first == (packet_accepted_count == 0);
    endproperty

    property p_legal_packet_last_exact;
        @(posedge clk_core) disable iff (rst_core)
            packet_fire && packet_legal |->
                packet_last ==
                    ((packet_accepted_count + 1'b1)
                     == active_reduction_population);
    endproperty

    property p_illegal_start_sets_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            start_fire && !operator_start_legal |=> protocol_error;
    endproperty

    property p_illegal_packet_sets_sticky_error;
        @(posedge clk_core) disable iff (rst_core)
            packet_fire && !packet_legal |=> protocol_error;
    endproperty

    property p_error_is_sticky;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |=> protocol_error;
    endproperty

    property p_error_blocks_new_work;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> !operator_start_ready && !packet_ready;
    endproperty

    property p_result_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core)
            result_valid && !result_ready |=>
                illegal_packet_fire || protocol_error
                || (result_valid && $stable(result_tag)
                    && $stable(result_lane_tile_id) && $stable(result_slice_id)
                    && $stable(result_count) && $stable(result_sum)
                    && $stable(result_sumsq) && $stable(serializer_slice));
    endproperty

    property p_exported_result_valid_is_fail_closed;
        @(posedge clk_core) disable iff (rst_core)
            result_valid == (result_valid_internal
                             && !protocol_error && !illegal_packet_fire);
    endproperty

    property p_result_fire_matches_exported_handshake;
        @(posedge clk_core) disable iff (rst_core)
            result_fire_internal == (result_valid && result_ready);
    endproperty

    property p_illegal_packet_suppresses_pending_result;
        @(posedge clk_core) disable iff (rst_core)
            illegal_packet_fire |-> !result_valid && !result_fire_internal;
    endproperty

    property p_sticky_error_suppresses_all_results;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> !result_valid && !result_fire_internal;
    endproperty

    property p_pending_result_illegal_collision_preserves_retirement;
        @(posedge clk_core) disable iff (rst_core)
            result_valid_internal && result_ready && illegal_packet_fire
            |=> protocol_error && result_valid_internal
                && $stable(results_retired)
                && $stable(fifo_read_ptr) && $stable(fifo_write_ptr)
                && $stable(serializer_slice);
    endproperty

    property p_backpressure_blocks_arithmetic_update;
        @(posedge clk_core) disable iff (rst_core)
            result_valid && !result_ready |-> !arithmetic_update;
    endproperty

    property p_illegal_packet_blocks_arithmetic_update;
        @(posedge clk_core) disable iff (rst_core)
            packet_fire && !packet_legal |-> !arithmetic_update;
    endproperty

    property p_full_dequeue_candidate_exposes_ready;
        @(posedge clk_core) disable iff (rst_core)
            fifo_level == 4 && dequeue_candidate |-> packet_ready;
    endproperty

    property p_full_legal_swap_fires_atomically;
        @(posedge clk_core) disable iff (rst_core)
            fifo_level == 4 && dequeue_candidate
                && packet_valid && packet_legal
            |-> enqueue_fire && dequeue_fire;
    endproperty

    property p_full_swap_preserves_level_and_advances_pointers;
        @(posedge clk_core) disable iff (rst_core)
            fifo_level == 4 && enqueue_fire && dequeue_fire
            |=> fifo_level == 4
                && fifo_read_ptr == ($past(fifo_read_ptr) + 1'b1)
                && fifo_write_ptr == ($past(fifo_write_ptr) + 1'b1);
    endproperty

    property p_illegal_full_offer_cancels_dequeue;
        @(posedge clk_core) disable iff (rst_core)
            fifo_level == 4 && dequeue_candidate
                && packet_valid && packet_ready && !packet_legal
            |-> !dequeue_fire && !arithmetic_update;
    endproperty

    property p_illegal_full_offer_preserves_fifo_state;
        @(posedge clk_core) disable iff (rst_core)
            fifo_level == 4 && dequeue_candidate
                && packet_valid && packet_ready && !packet_legal
            |=> protocol_error && fifo_level == 4
                && $stable(fifo_read_ptr) && $stable(fifo_write_ptr)
                && $stable(serializer_slice);
    endproperty

    property p_result_metadata_is_bounded;
        @(posedge clk_core) disable iff (rst_core)
            result_valid |-> result_slice_id < 6 && result_count != 0;
    endproperty

    property p_done_is_one_cycle;
        @(posedge clk_core) disable iff (rst_core)
            operator_done |=> !operator_done;
    endproperty

    property p_done_only_follows_final_slice_retirement;
        @(posedge clk_core) disable iff (rst_core)
            $rose(operator_done) |->
                $past(result_valid && result_ready && result_slice_id == 5);
    endproperty

    assert property (p_start_legality_exact);
    assert property (p_legal_packet_first_exact);
    assert property (p_legal_packet_last_exact);
    assert property (p_illegal_start_sets_sticky_error);
    assert property (p_illegal_packet_sets_sticky_error);
    assert property (p_error_is_sticky);
    assert property (p_error_blocks_new_work);
    assert property (p_result_stable_under_backpressure);
    assert property (p_exported_result_valid_is_fail_closed);
    assert property (p_result_fire_matches_exported_handshake);
    assert property (p_illegal_packet_suppresses_pending_result);
    assert property (p_sticky_error_suppresses_all_results);
    assert property (p_pending_result_illegal_collision_preserves_retirement);
    assert property (p_backpressure_blocks_arithmetic_update);
    assert property (p_illegal_packet_blocks_arithmetic_update);
    assert property (p_full_dequeue_candidate_exposes_ready);
    assert property (p_full_legal_swap_fires_atomically);
    assert property (p_full_swap_preserves_level_and_advances_pointers);
    assert property (p_illegal_full_offer_cancels_dequeue);
    assert property (p_illegal_full_offer_preserves_fifo_state);
    assert property (p_result_metadata_is_bounded);
    assert property (p_done_is_one_cycle);
    assert property (p_done_only_follows_final_slice_retirement);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            if (packet_fire && packet_legal)
                legal_packets_seen <= legal_packets_seen + 1;
            if (packet_fire && !packet_legal)
                illegal_packets_seen <= illegal_packets_seen + 1;
            if (result_valid && result_ready)
                results_seen <= results_seen + 1;
            if (result_valid && !result_ready)
                result_stalls_seen <= result_stalls_seen + 1;
            if (fifo_level == 4)
                fifo_full_seen <= fifo_full_seen + 1;
            if (operator_done)
                done_seen <= done_seen + 1;
            if (fifo_level == 4 && enqueue_fire && dequeue_fire)
                full_swap_seen <= full_swap_seen + 1;
            if (result_valid_internal && result_ready && illegal_packet_fire)
                pending_result_cancel_seen <= pending_result_cancel_seen + 1;
        end
    end

    final begin
        $display("M21_SVA_COVERAGE legal_packets=%0d illegal_packets=%0d results=%0d result_stalls=%0d fifo_full=%0d done=%0d full_swaps=%0d pending_result_cancels=%0d",
                 legal_packets_seen, illegal_packets_seen, results_seen,
                 result_stalls_seen, fifo_full_seen, done_seen, full_swap_seen,
                 pending_result_cancel_seen);
        if (legal_packets_seen <= 0 || illegal_packets_seen <= 0
            || results_seen <= 0 || result_stalls_seen <= 0
            || fifo_full_seen <= 0 || done_seen <= 0 || full_swap_seen <= 0
            || pending_result_cancel_seen <= 0)
            $error("M21 bound-SVA runtime coverage is incomplete");
    end
endmodule

bind qfit_dynamic_bn_banked_moment_scheduler
qfit_dynamic_bn_banked_moment_scheduler_assertions #(
    .IN_W(IN_W), .TAG_W(TAG_W),
    .MAX_REDUCTION_POPULATION(MAX_REDUCTION_POPULATION),
    .MAX_LANE_TILES(MAX_LANE_TILES)
) u_qfit_dynamic_bn_banked_moment_scheduler_assertions (
    .clk_core, .rst_core,
    .operator_start_valid, .operator_start_ready,
    .operator_reduction_population, .operator_active_lane_tiles,
    .operator_start_legal, .operator_active, .active_reduction_population,
    .packet_valid, .packet_ready, .packet_first, .packet_last,
    .packet_legal, .packet_accepted_count,
    .result_valid, .result_ready, .result_tag, .result_lane_tile_id,
    .result_slice_id, .result_count, .result_sum, .result_sumsq,
    .operator_done, .protocol_error, .fifo_level, .serializer_slice,
    .arithmetic_update(process_slice),
    .dequeue_candidate, .dequeue_fire, .enqueue_fire,
    .illegal_packet_fire, .result_valid_internal(result_valid_q),
    .result_fire_internal(result_fire), .results_retired(results_retired_q),
    .fifo_read_ptr(fifo_read_ptr_q), .fifo_write_ptr(fifo_write_ptr_q)
);

`default_nettype wire
