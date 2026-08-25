`timescale 1ns/1ps
`default_nettype none

module qfit_wide_temporal_state_engine_assertions #(
    parameter int CONTEXTS = 4,
    parameter int BASE_TILES = 32,
    parameter int BANKS = 6,
    parameter int LANES_PER_BANK = 16,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int EPOCH_W = 16,
    parameter int DOMAIN_W = 32,
    parameter int STEP_W = 4,
    parameter int LEN_W = 4,
    parameter int CTX_W = (CONTEXTS <= 1) ? 1 : $clog2(CONTEXTS),
    parameter int BASE_TILE_W = (BASE_TILES <= 1) ? 1 : $clog2(BASE_TILES),
    parameter int WIDE_ACC_BITS = BANKS*LANES_PER_BANK*ACC_W
) (
    input logic clk_core,
    input logic por_core,
    input logic rst_core,
    input logic request_valid,
    input logic request_ready,
    input logic request_admitted,
    input logic request_use_motion,
    input logic [CTX_W-1:0] request_context,
    input logic [BASE_TILE_W-1:0] request_base_tile,
    input logic [EPOCH_W-1:0] request_epoch,
    input logic [DOMAIN_W-1:0] request_domain,
    input logic [STEP_W-1:0] request_temporal_step,
    input logic [LEN_W-1:0] request_temporal_length,
    input logic request_temporal_first,
    input logic request_temporal_last,
    input logic [TAG_W-1:0] request_tag,
    input logic [WIDE_ACC_BITS-1:0] request_acc,
    input logic rmw_pending_q,
    input logic rmw_commit,
    input logic [CTX_W-1:0] rmw_context_q,
    input logic [BASE_TILE_W-1:0] rmw_base_tile_q,
    input logic [EPOCH_W-1:0] rmw_epoch_q,
    input logic [DOMAIN_W-1:0] rmw_domain_q,
    input logic [STEP_W-1:0] rmw_step_q,
    input logic [LEN_W-1:0] rmw_length_q,
    input logic rmw_first_q,
    input logic rmw_last_q,
    input logic [TAG_W-1:0] rmw_tag_q,
    input logic [WIDE_ACC_BITS-1:0] rmw_delta_q,
    input logic output_valid,
    input logic output_ready,
    input logic [CTX_W-1:0] output_context,
    input logic [BASE_TILE_W-1:0] output_base_tile,
    input logic [EPOCH_W-1:0] output_epoch,
    input logic [DOMAIN_W-1:0] output_domain,
    input logic [STEP_W-1:0] output_temporal_step,
    input logic [LEN_W-1:0] output_temporal_length,
    input logic output_temporal_first,
    input logic output_temporal_last,
    input logic output_used_motion,
    input logic [TAG_W-1:0] output_tag,
    input logic [WIDE_ACC_BITS-1:0] output_current_acc,
    input logic protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (por_core || rst_core);

    ap_ready_only_for_admitted_request: assert property (
        request_ready |-> request_valid && request_admitted)
        else $error("wide state accepted an inadmissible request");

    ap_protocol_error_matches_rejection: assert property (
        protocol_error == (request_valid && !request_admitted))
        else $error("wide state protocol error diverged from admission");

    ap_first_request_is_local: assert property (
        request_valid && request_ready && request_temporal_first |->
        !request_use_motion)
        else $error("wide state admitted Motion as a temporal first request");

    ap_request_stable_under_backpressure: assert property (
        request_valid && !request_ready |=> request_valid &&
        $stable({request_context, request_base_tile, request_epoch,
                 request_domain, request_temporal_step,
                 request_temporal_length, request_temporal_first,
                 request_temporal_last, request_use_motion, request_tag,
                 request_acc}))
        else $error("wide state request changed under backpressure");

    ap_rmw_blocks_new_request: assert property (
        rmw_pending_q |-> !request_ready)
        else $error("wide state accepted a request while RMW was pending");

    ap_rmw_snapshot_stable_until_commit: assert property (
        rmw_pending_q && !rmw_commit |=> rmw_pending_q &&
        $stable({rmw_context_q, rmw_base_tile_q, rmw_epoch_q,
                 rmw_domain_q, rmw_step_q, rmw_length_q, rmw_first_q,
                 rmw_last_q, rmw_tag_q, rmw_delta_q}))
        else $error("wide state RMW snapshot changed before commit");

    ap_output_stable_under_backpressure: assert property (
        output_valid && !output_ready |=> output_valid &&
        $stable({output_context, output_base_tile, output_epoch,
                 output_domain, output_temporal_step,
                 output_temporal_length, output_temporal_first,
                 output_temporal_last, output_used_motion, output_tag,
                 output_current_acc}))
        else $error("wide state output changed under backpressure");

    cp_wide_local: cover property (
        request_valid && request_ready && !request_use_motion);
    cp_wide_motion: cover property (
        request_valid && request_ready && request_use_motion);
    cp_wide_first: cover property (
        request_valid && request_ready && request_temporal_first);
    cp_wide_last: cover property (
        request_valid && request_ready && request_temporal_last);
    cp_local_refresh: cover property (
        request_valid && request_ready && !request_temporal_first &&
        !request_use_motion);
    cp_rmw_backpressure: cover property (
        rmw_pending_q && output_valid && !output_ready);
endmodule

bind qfit_wide_temporal_state_engine
    qfit_wide_temporal_state_engine_assertions #(
        .CONTEXTS(CONTEXTS), .BASE_TILES(BASE_TILES), .BANKS(BANKS),
        .LANES_PER_BANK(LANES_PER_BANK), .ACC_W(ACC_W), .TAG_W(TAG_W),
        .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W), .STEP_W(STEP_W),
        .LEN_W(LEN_W), .CTX_W(CTX_W), .BASE_TILE_W(BASE_TILE_W),
        .WIDE_ACC_BITS(WIDE_ACC_BITS)
    ) u_wide_state_assertions (
        .clk_core, .por_core, .rst_core, .request_valid, .request_ready,
        .request_admitted, .request_use_motion, .request_context,
        .request_base_tile, .request_epoch, .request_domain,
        .request_temporal_step, .request_temporal_length,
        .request_temporal_first, .request_temporal_last, .request_tag,
        .request_acc, .rmw_pending_q, .rmw_commit, .rmw_context_q,
        .rmw_base_tile_q, .rmw_epoch_q, .rmw_domain_q, .rmw_step_q,
        .rmw_length_q, .rmw_first_q, .rmw_last_q, .rmw_tag_q,
        .rmw_delta_q, .output_valid, .output_ready, .output_context,
        .output_base_tile, .output_epoch, .output_domain,
        .output_temporal_step, .output_temporal_length,
        .output_temporal_first, .output_temporal_last,
        .output_used_motion, .output_tag, .output_current_acc,
        .protocol_error
    );

`default_nettype wire
