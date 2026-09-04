`timescale 1ns/1ps
`default_nettype none

module gatestack_adaptive_csr_selector_assertions #(
    parameter int WORD_INDEX_W = 7
) (
    input logic clk_core,
    input logic rst_core,
    input logic [1:0] state_q,
    input logic select_fadc_q,
    input logic first_word_pending_q,
    input logic [63:0] first_word_data_q,
    input logic [WORD_INDEX_W-1:0] first_word_index_q,
    input logic first_word_last_q,
    input logic start_ready,
    input logic start_valid,
    input logic start_format_valid,
    input logic start_select_fadc,
    input logic word_valid,
    input logic word_ready,
    input logic [15:0] word_magic,
    input logic [1:0] child_start_valid,
    input logic [1:0] child_word_valid,
    input logic [1:0] child_word_ready,
    input logic selected_done_fire
);
    localparam logic [1:0] ST_IDLE = 2'd0;
    localparam logic [1:0] ST_PEEK = 2'd1;
    localparam logic [1:0] ST_START = 2'd2;
    localparam logic [1:0] ST_RUN = 2'd3;

    property p_start_ready_only_idle;
        @(posedge clk_core) disable iff (rst_core)
        start_ready |-> state_q == ST_IDLE;
    endproperty

    property p_magic_selects_child;
        @(posedge clk_core) disable iff (rst_core)
        state_q == ST_PEEK && word_valid && word_ready |=>
            select_fadc_q == ($past(word_magic) == 16'h4641);
    endproperty

    property p_metadata_bypasses_peek;
        @(posedge clk_core) disable iff (rst_core)
        state_q == ST_IDLE && start_valid && start_ready &&
        start_format_valid |=> state_q == ST_START &&
            select_fadc_q == $past(start_select_fadc) &&
            !first_word_pending_q;
    endproperty

    property p_legacy_path_keeps_peek;
        @(posedge clk_core) disable iff (rst_core)
        state_q == ST_IDLE && start_valid && start_ready &&
        !start_format_valid |=> state_q == ST_PEEK;
    endproperty

    property p_selection_stable_after_peek;
        @(posedge clk_core) disable iff (rst_core)
        (state_q == ST_START || state_q == ST_RUN) &&
        ($past(state_q) == ST_START || $past(state_q) == ST_RUN) |->
            $stable(select_fadc_q);
    endproperty

    property p_single_selected_child_start;
        @(posedge clk_core) disable iff (rst_core)
        child_start_valid != 0 |->
            state_q == ST_START && $onehot(child_start_valid) &&
            child_start_valid == (select_fadc_q ? 2'b10 : 2'b01);
    endproperty

    property p_single_selected_child_word;
        @(posedge clk_core) disable iff (rst_core)
        child_word_valid != 0 |->
            state_q == ST_RUN && $onehot(child_word_valid) &&
            child_word_valid == (select_fadc_q ? 2'b10 : 2'b01);
    endproperty

    property p_cached_first_word_blocks_upstream;
        @(posedge clk_core) disable iff (rst_core)
        first_word_pending_q |-> !word_ready;
    endproperty

    property p_cached_first_word_stable_until_child_accepts;
        @(posedge clk_core) disable iff (rst_core)
        first_word_pending_q && state_q == ST_RUN &&
        !child_word_ready[select_fadc_q] |=>
            first_word_pending_q &&
            $stable({first_word_data_q, first_word_index_q, first_word_last_q});
    endproperty

    property p_cached_first_word_clears_on_accept;
        @(posedge clk_core) disable iff (rst_core)
        first_word_pending_q && state_q == ST_RUN &&
        child_word_ready[select_fadc_q] |=> !first_word_pending_q;
    endproperty

    property p_done_only_while_running;
        @(posedge clk_core) disable iff (rst_core)
        selected_done_fire |-> state_q == ST_RUN;
    endproperty

    assert property (p_start_ready_only_idle);
    assert property (p_magic_selects_child);
    assert property (p_metadata_bypasses_peek);
    assert property (p_legacy_path_keeps_peek);
    assert property (p_selection_stable_after_peek);
    assert property (p_single_selected_child_start);
    assert property (p_single_selected_child_word);
    assert property (p_cached_first_word_blocks_upstream);
    assert property (p_cached_first_word_stable_until_child_accepts);
    assert property (p_cached_first_word_clears_on_accept);
    assert property (p_done_only_while_running);
endmodule

`default_nettype wire
