`timescale 1ns/1ps
`default_nettype none

module qfit_dual_granularity_temporal_state_engine_assertions #(
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
    parameter int BANK_W = (BANKS <= 1) ? 1 : $clog2(BANKS),
    parameter int ROWS = CONTEXTS * BASE_TILES,
    parameter int ROW_ADDR_W = (ROWS <= 1) ? 1 : $clog2(ROWS),
    parameter int BANK_ACC_BITS = LANES_PER_BANK * ACC_W,
    parameter int WIDE_ACC_BITS = BANKS * BANK_ACC_BITS
) (
    input logic clk_core,
    input logic por_core,
    input logic rst_core,
    input logic domain_fence_ready,
    input logic wide_valid,
    input logic wide_ready,
    input logic wide_admitted,
    input logic wide_eligible,
    input logic wide_use_motion,
    input logic [WIDE_ACC_BITS-1:0] wide_acc,
    input logic narrow_valid,
    input logic narrow_ready,
    input logic narrow_admitted,
    input logic narrow_eligible,
    input logic [BANK_W-1:0] narrow_bank,
    input logic narrow_use_motion,
    input logic [BANK_ACC_BITS-1:0] narrow_acc,
    input logic abort_valid,
    input logic abort_ready,
    input logic abort_admitted,
    input logic abort_eligible,
    input logic abort_error,
    input logic last_grant_wide_q,
    input logic rmw_pending_q,
    input logic rmw_commit,
    input logic rmw_is_wide_q,
    input logic [CTX_W-1:0] rmw_context_q,
    input logic [BASE_TILE_W-1:0] rmw_base_tile_q,
    input logic [BANKS-1:0] rmw_bank_mask_q,
    input logic [EPOCH_W-1:0] rmw_epoch_q,
    input logic [DOMAIN_W-1:0] rmw_domain_q,
    input logic [STEP_W-1:0] rmw_step_q,
    input logic [LEN_W-1:0] rmw_length_q,
    input logic rmw_first_q,
    input logic rmw_last_q,
    input logic [TAG_W-1:0] rmw_tag_q,
    input logic [WIDE_ACC_BITS-1:0] rmw_result,
    input logic [BANKS-1:0] bank_enable,
    input logic [BANKS-1:0] bank_write_enable,
    input logic [ROW_ADDR_W-1:0] bank_address [0:BANKS-1],
    input logic [BANK_ACC_BITS-1:0] bank_write_data [0:BANKS-1],
    input logic output_valid,
    input logic output_ready,
    input logic output_is_wide,
    input logic [CTX_W-1:0] output_context,
    input logic [BASE_TILE_W-1:0] output_base_tile,
    input logic [BANKS-1:0] output_bank_mask,
    input logic [EPOCH_W-1:0] output_epoch,
    input logic [DOMAIN_W-1:0] output_domain,
    input logic [STEP_W-1:0] output_temporal_step,
    input logic [LEN_W-1:0] output_temporal_length,
    input logic output_temporal_first,
    input logic output_temporal_last,
    input logic output_used_motion,
    input logic [TAG_W-1:0] output_tag,
    input logic [WIDE_ACC_BITS-1:0] output_current_acc,
    input logic wide_protocol_error,
    input logic narrow_protocol_error
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (por_core || rst_core);

    ap_output_stable: assert property (
        output_valid && !output_ready |=>
        output_valid && $stable({output_is_wide, output_context,
            output_base_tile, output_bank_mask, output_epoch, output_domain,
            output_temporal_step, output_temporal_length,
            output_temporal_first, output_temporal_last, output_used_motion,
            output_tag, output_current_acc}))
        else $error("M9.1 output changed while stalled");

    ap_reset_blocks_accept: assert property (
        disable iff (1'b0) (por_core || rst_core) |->
        !domain_fence_ready && !wide_ready && !narrow_ready && !abort_ready)
        else $error("M9.1 request accepted while reset was asserted");

    ap_reset_blocks_sram_write: assert property (
        disable iff (1'b0) (por_core || rst_core) |->
        bank_write_enable == '0 && !rmw_commit)
        else $error("M9.1 SRAM write escaped reset cancellation");

    ap_single_accept: assert property (
        $onehot0({wide_ready, narrow_ready, abort_ready}))
        else $error("M9.1 accepted multiple request ports");

    ap_motion_reserves_rmw: assert property (
        (wide_valid && wide_ready && wide_use_motion) ||
        (narrow_valid && narrow_ready && narrow_use_motion) |=> rmw_pending_q)
        else $error("M9.1 Motion request did not enter RMW phase");

    ap_rmw_waits_for_output: assert property (
        rmw_pending_q && output_valid && !output_ready |=> rmw_pending_q)
        else $error("M9.1 RMW retired through output backpressure");

    ap_rmw_output: assert property (
        rmw_commit |=> output_valid && output_used_motion &&
        (output_is_wide == $past(rmw_is_wide_q)) &&
        (output_context == $past(rmw_context_q)) &&
        (output_base_tile == $past(rmw_base_tile_q)) &&
        (output_bank_mask == $past(rmw_bank_mask_q)) &&
        (output_epoch == $past(rmw_epoch_q)) &&
        (output_domain == $past(rmw_domain_q)) &&
        (output_temporal_step == $past(rmw_step_q)) &&
        (output_temporal_length == $past(rmw_length_q)) &&
        (output_temporal_first == $past(rmw_first_q)) &&
        (output_temporal_last == $past(rmw_last_q)) &&
        (output_tag == $past(rmw_tag_q)))
        else $error("M9.1 RMW output identity mismatch");

    ap_wide_local_exact: assert property (
        wide_valid && wide_ready && !wide_use_motion |=>
        output_valid && output_is_wide && (&output_bank_mask) &&
        !output_used_motion && output_current_acc == $past(wide_acc))
        else $error("M9.1 wide Local direct write mismatch");

    ap_narrow_local_shape: assert property (
        narrow_valid && narrow_ready && !narrow_use_motion |=>
        output_valid && !output_is_wide && $onehot(output_bank_mask) &&
        output_bank_mask[$past(narrow_bank)] &&
        output_current_acc[($past(narrow_bank)*BANK_ACC_BITS) +:
            BANK_ACC_BITS] == $past(narrow_acc))
        else $error("M9.1 narrow Local direct write mismatch");

    ap_wide_sram_atomic: assert property (
        wide_valid && wide_ready |-> bank_enable == {BANKS{1'b1}})
        else $error("M9.1 wide request did not reserve every bank");

    ap_wide_local_writes_all: assert property (
        wide_valid && wide_ready && !wide_use_motion |->
        bank_write_enable == {BANKS{1'b1}})
        else $error("M9.1 wide Local write was not all-bank atomic");

    ap_wide_motion_reads_all: assert property (
        wide_valid && wide_ready && wide_use_motion |->
        bank_write_enable == '0)
        else $error("M9.1 wide Motion skipped synchronous read phase");

    ap_wide_motion_commit_atomic: assert property (
        rmw_commit && rmw_is_wide_q |->
        bank_enable == {BANKS{1'b1}} &&
        bank_write_enable == {BANKS{1'b1}})
        else $error("M9.1 wide Motion commit was not all-bank atomic");

    ap_narrow_sram_onehot: assert property (
        narrow_valid && narrow_ready |-> $onehot(bank_enable))
        else $error("M9.1 narrow request touched multiple SRAM banks");

    ap_round_robin_after_wide: assert property (
        wide_eligible && narrow_eligible && !abort_eligible &&
        last_grant_wide_q |-> narrow_ready)
        else $error("M9.1 legal wide traffic starved legal narrow traffic");

    ap_round_robin_after_narrow: assert property (
        wide_eligible && narrow_eligible && !abort_eligible &&
        !last_grant_wide_q |-> wide_ready)
        else $error("M9.1 legal narrow traffic starved legal wide traffic");

    ap_illegal_wide_visible: assert property (
        wide_valid && !wide_admitted |-> wide_protocol_error)
        else $error("M9.1 illegal wide request hidden");

    ap_illegal_narrow_visible: assert property (
        narrow_valid && !narrow_admitted |-> narrow_protocol_error)
        else $error("M9.1 illegal narrow request hidden");

    ap_illegal_abort_visible: assert property (
        abort_valid && !abort_admitted |-> abort_error)
        else $error("M9.1 illegal abort hidden");

    ap_reset_disarms_domain: assert property (rst_core |=> !domain_fence_ready)
        else $error("M9.1 functional reset did not disarm replay fence");

    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_rmw_bank
        ap_rmw_sram_command_exact: assert property (
            rmw_commit && rmw_bank_mask_q[bank] |->
            bank_enable[bank] && bank_write_enable[bank] &&
            bank_address[bank] == ROW_ADDR_W'(
                $unsigned(rmw_context_q) * BASE_TILES +
                $unsigned(rmw_base_tile_q)) &&
            bank_write_data[bank] ==
                rmw_result[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS])
            else $error("M9.1 RMW SRAM command mismatch bank=%0d", bank);

        ap_rmw_unselected_bank_idle: assert property (
            rmw_commit && !rmw_bank_mask_q[bank] |-> !bank_enable[bank])
            else $error("M9.1 RMW touched an unselected bank=%0d", bank);

        ap_rmw_value_exact: assert property (
            rmw_commit && rmw_bank_mask_q[bank] |=>
            output_current_acc[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS] ==
                $past(rmw_result[(bank*BANK_ACC_BITS) +: BANK_ACC_BITS]))
            else $error("M9.1 RMW value mismatch bank=%0d", bank);
    end

    cp_wide_local: cover property (wide_valid && wide_ready && !wide_use_motion);
    cp_wide_motion: cover property (wide_valid && wide_ready && wide_use_motion);
    cp_narrow_local: cover property (narrow_valid && narrow_ready && !narrow_use_motion);
    cp_narrow_motion: cover property (narrow_valid && narrow_ready && narrow_use_motion);
    cp_abort: cover property (abort_valid && abort_ready);
    cp_rmw_backpressure: cover property (
        rmw_pending_q && output_valid && !output_ready);
    cp_round_robin_wide: cover property (
        wide_eligible && narrow_eligible && wide_ready);
    cp_round_robin_narrow: cover property (
        wide_eligible && narrow_eligible && narrow_ready);
endmodule

bind qfit_dual_granularity_temporal_state_engine
    qfit_dual_granularity_temporal_state_engine_assertions #(
        .CONTEXTS(CONTEXTS), .BASE_TILES(BASE_TILES), .BANKS(BANKS),
        .LANES_PER_BANK(LANES_PER_BANK), .ACC_W(ACC_W),
        .TAG_W(TAG_W), .EPOCH_W(EPOCH_W), .DOMAIN_W(DOMAIN_W),
        .STEP_W(STEP_W), .LEN_W(LEN_W), .CTX_W(CTX_W),
        .BASE_TILE_W(BASE_TILE_W), .ROWS(ROWS), .ROW_ADDR_W(ROW_ADDR_W),
        .BANK_W(BANK_W), .BANK_ACC_BITS(BANK_ACC_BITS),
        .WIDE_ACC_BITS(WIDE_ACC_BITS)
    ) u_m9_assertions (
        .clk_core, .por_core, .rst_core, .domain_fence_ready,
        .wide_valid, .wide_ready, .wide_admitted, .wide_eligible,
        .wide_use_motion, .wide_acc,
        .narrow_valid, .narrow_ready, .narrow_admitted, .narrow_eligible,
        .narrow_bank, .narrow_use_motion, .narrow_acc,
        .abort_valid, .abort_ready, .abort_admitted, .abort_eligible,
        .abort_error, .last_grant_wide_q, .rmw_pending_q, .rmw_commit,
        .rmw_is_wide_q, .rmw_context_q, .rmw_base_tile_q,
        .rmw_bank_mask_q, .rmw_epoch_q, .rmw_domain_q, .rmw_step_q,
        .rmw_length_q, .rmw_first_q, .rmw_last_q, .rmw_tag_q,
        .rmw_result, .bank_enable, .bank_write_enable,
        .bank_address, .bank_write_data,
        .output_valid, .output_ready, .output_is_wide, .output_context,
        .output_base_tile, .output_bank_mask, .output_epoch, .output_domain,
        .output_temporal_step, .output_temporal_length,
        .output_temporal_first, .output_temporal_last,
        .output_used_motion, .output_tag, .output_current_acc,
        .wide_protocol_error, .narrow_protocol_error
    );

`default_nettype wire
