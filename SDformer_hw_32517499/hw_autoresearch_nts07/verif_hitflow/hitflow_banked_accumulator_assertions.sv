`timescale 1ns/1ps
`default_nettype none

module hitflow_banked_accumulator_assertions #(
    parameter int TOKENS       = 162,
    parameter int BANKS        = 2,
    parameter int ACC_W        = 32,
    parameter int OUT_TILE     = 8,
    parameter int TAG_W        = 32,
    parameter int COUNTER_W    = 32,
    parameter int TOKEN_ID_W   = (TOKENS <= 1) ? 1 : $clog2(TOKENS)
) (
    input logic                              clk_core,
    input logic                              rst_core,
    input logic                              flush,
    input logic                              protocol_error,
    input logic                              group_start_ready,
    input logic [BANKS-1:0]                  update_valid,
    input logic [BANKS-1:0]                  update_ready,
    input logic [BANKS-1:0]                  final_valid,
    input logic [BANKS-1:0]                  final_ready,
    input logic [(BANKS*TOKEN_ID_W)-1:0]     final_token_ids,
    input logic [TAG_W-1:0]                  final_tag,
    input logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values,
    input logic                              group_finish_ready,
    input logic [COUNTER_W-1:0]              count_updates,
    input logic [COUNTER_W-1:0]              count_writes,
    input logic [COUNTER_W-1:0]              count_bias_commits,
    input logic [COUNTER_W-1:0]              count_bank_stall_cycles,
    input logic [COUNTER_W-1:0]              count_final_stall_cycles
);

    for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : g_bank_assertions
        property p_final_stable_under_backpressure;
            @(posedge clk_core) disable iff (rst_core)
                !flush && final_valid[bank] && !final_ready[bank] |=>
                flush || (final_valid[bank] && $stable(final_tag) &&
                $stable(final_token_ids[(bank*TOKEN_ID_W) +: TOKEN_ID_W]) &&
                $stable(final_values[(bank*OUT_TILE*ACC_W) +:
                                     (OUT_TILE*ACC_W)]));
        endproperty

        assert property (p_final_stable_under_backpressure);
    end

    property p_invalid_update_is_rejected;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error |-> ((update_valid & update_ready) == '0);
    endproperty

    property p_finish_has_no_pending_final;
        @(posedge clk_core) disable iff (rst_core)
            group_finish_ready |-> (final_valid == '0);
    endproperty

    property p_flush_masks_interfaces;
        @(posedge clk_core) disable iff (rst_core)
            flush |-> !group_start_ready && (update_ready == '0) &&
                      (final_valid == '0) && !group_finish_ready &&
                      !protocol_error;
    endproperty

    property p_flush_returns_idle;
        @(posedge clk_core) disable iff (rst_core)
            flush |=> flush || group_start_ready;
    endproperty

    property p_flush_drops_stale_final;
        @(posedge clk_core) disable iff (rst_core)
            flush |=> (final_valid == '0);
    endproperty

    property p_counters_monotonic;
        @(posedge clk_core) disable iff (rst_core)
            count_updates >= $past(count_updates) &&
            count_writes >= $past(count_writes) &&
            count_bias_commits >= $past(count_bias_commits) &&
            count_bank_stall_cycles >= $past(count_bank_stall_cycles) &&
            count_final_stall_cycles >= $past(count_final_stall_cycles);
    endproperty

    property p_flush_preserves_counters;
        @(posedge clk_core) disable iff (rst_core)
            flush |=> count_updates == $past(count_updates) &&
                       count_writes == $past(count_writes) &&
                       count_bias_commits == $past(count_bias_commits) &&
                       count_bank_stall_cycles ==
                           $past(count_bank_stall_cycles) &&
                       count_final_stall_cycles ==
                           $past(count_final_stall_cycles);
    endproperty

    assert property (p_invalid_update_is_rejected);
    assert property (p_finish_has_no_pending_final);
    assert property (p_flush_masks_interfaces);
    assert property (p_flush_returns_idle);
    assert property (p_flush_drops_stale_final);
    assert property (p_counters_monotonic);
    assert property (p_flush_preserves_counters);

endmodule

`default_nettype wire
