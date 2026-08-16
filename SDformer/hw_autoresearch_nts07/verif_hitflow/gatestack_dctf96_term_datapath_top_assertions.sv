`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf96_term_datapath_top_assertions #(
    parameter int Q = 2,
    parameter int OUT_TILE = 32,
    parameter int PRODUCT_W = 17,
    parameter int GROUP_TAG_W = 32,
    parameter int ISSUE_SEQ_W = 13,
    parameter int INPUT_CH_W = 10,
    parameter int TOKEN_ID_W = 8,
    parameter int OUTPUT_TILE_W = 8,
    parameter int EPOCH_W = 4,
    parameter int COUNTER_W = 32
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic clear_error,
    input logic term_valid,
    input logic term_ready,
    input logic adapter_term_ready,
    input logic adapter_idle,
    input logic event_ready,
    input logic term_metadata_legal,
    input logic adapter_term_valid,
    input logic adapter_event_valid,
    input logic legal_term_fire,
    input logic illegal_term_fire,
    input logic illegal_event_fire,
    input logic illegal_drop_active_q,
    input logic [7:0] term_destination_count,
    input logic event_term_last,
    input logic issue_term_fire,
    input logic [2:0] weight_req_valid,
    input logic [2:0] weight_req_ready,
    input logic [(3*GROUP_TAG_W)-1:0] weight_req_tags,
    input logic [(3*INPUT_CH_W)-1:0] weight_req_input_channels,
    input logic [(3*OUTPUT_TILE_W)-1:0] weight_req_output_tiles,
    input logic [(3*EPOCH_W)-1:0] weight_req_epochs,
    input logic [2:0] weight_rsp_ready,
    input logic [5:0] acc_update_valid,
    input logic [5:0] acc_update_ready,
    input logic [(6*TOKEN_ID_W)-1:0] acc_update_token_ids,
    input logic [(3*GROUP_TAG_W)-1:0] acc_update_tags,
    input logic [(3*OUT_TILE*PRODUCT_W)-1:0] acc_update_values,
    input logic [2:0] bank_term_done,
    input logic [2:0] bank_term_done_match,
    input logic head_compute_done,
    input logic [GROUP_TAG_W-1:0] head_compute_done_group_tag,
    input logic [ISSUE_SEQ_W-1:0] head_compute_done_issue_seq,
    input logic dispatch_retire_valid,
    input logic track_pop,
    input logic tracked_head_last,
    input logic [2:0] head_done_mask_after,
    input logic [GROUP_TAG_W-1:0] tracked_head_tag,
    input logic [ISSUE_SEQ_W-1:0] tracked_head_issue_seq,
    input logic [(Q+4)-1:0] track_valid_q,
    input logic [((Q+4 < 2) ? 1 : $clog2(Q+4)):0] track_count_q,
    input logic [((Q < 2) ? 1 : $clog2(Q+1))-1:0] fabric_occupancy,
    input logic [COUNTER_W-1:0] issued_terms,
    input logic [(3*COUNTER_W)-1:0] completed_terms,
    input logic datapath_idle,
    input logic protocol_error
);
    property p_illegal_term_consumed_outside_adapter;
        @(posedge clk_core) disable iff (rst_core || flush)
        illegal_term_fire |-> term_valid && term_ready &&
                               !term_metadata_legal && !adapter_term_valid;
    endproperty

    property p_illegal_ready_unlock_direct;
        @(posedge clk_core) disable iff (rst_core || flush)
        adapter_term_ready && !illegal_drop_active_q && term_valid &&
        !term_metadata_legal |-> term_ready && !adapter_term_valid;
    endproperty

    property p_legal_term_enters_adapter;
        @(posedge clk_core) disable iff (rst_core || flush)
        legal_term_fire |-> term_metadata_legal && adapter_term_valid;
    endproperty

    property p_illegal_nonempty_enters_drain;
        @(posedge clk_core) disable iff (rst_core || flush)
        illegal_term_fire && (term_destination_count != 0) |=>
            illegal_drop_active_q;
    endproperty

    property p_drop_isolated_from_adapter;
        @(posedge clk_core) disable iff (rst_core || flush)
        illegal_drop_active_q |-> !term_ready && event_ready &&
                                  !adapter_term_valid &&
                                  !adapter_event_valid && !datapath_idle;
    endproperty

    property p_drop_last_event_releases;
        @(posedge clk_core) disable iff (rst_core || flush)
        illegal_event_fire && event_term_last |=> !illegal_drop_active_q;
    endproperty

    property p_head_done_condition;
        @(posedge clk_core) disable iff (rst_core || flush)
        head_compute_done |-> track_pop && tracked_head_last &&
                              (&head_done_mask_after);
    endproperty

    property p_head_done_metadata;
        @(posedge clk_core) disable iff (rst_core || flush)
        head_compute_done |->
            (head_compute_done_group_tag == tracked_head_tag) &&
            (head_compute_done_issue_seq == tracked_head_issue_seq);
    endproperty

    property p_head_done_one_cycle;
        @(posedge clk_core) disable iff (rst_core || flush)
        head_compute_done |=> !head_compute_done;
    endproperty

    property p_flush_masks_interfaces;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> !term_ready && !event_ready &&
                  (weight_req_valid == '0) && (weight_rsp_ready == '0) &&
                  (acc_update_valid == '0) && (bank_term_done == '0) &&
                  !head_compute_done && !dispatch_retire_valid;
    endproperty

    property p_flush_clears_tracking;
        @(posedge clk_core) disable iff (rst_core)
        flush |=> (track_valid_q == '0) && (fabric_occupancy == '0);
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !clear_error |=> clear_error || protocol_error;
    endproperty

    property p_clear_error_clears_sticky;
        @(posedge clk_core) disable iff (rst_core)
        clear_error && !illegal_term_fire |=> !protocol_error;
    endproperty

    property p_new_illegal_error_wins_clear;
        @(posedge clk_core) disable iff (rst_core || flush)
        clear_error && illegal_term_fire |=> protocol_error;
    endproperty

    property p_datapath_idle_exact;
        @(posedge clk_core) disable iff (rst_core)
        datapath_idle == (adapter_idle && !illegal_drop_active_q &&
                          (fabric_occupancy == '0) &&
                          (track_count_q == '0));
    endproperty

    property p_issued_term_count;
        @(posedge clk_core) disable iff (rst_core || flush)
        issue_term_fire |=> issued_terms ==
                            ($past(issued_terms) + 1'b1);
    endproperty

    assert property (p_illegal_term_consumed_outside_adapter);
    assert property (p_illegal_ready_unlock_direct);
    assert property (p_legal_term_enters_adapter);
    assert property (p_illegal_nonempty_enters_drain);
    assert property (p_drop_isolated_from_adapter);
    assert property (p_drop_last_event_releases);
    assert property (p_head_done_condition);
    assert property (p_head_done_metadata);
    assert property (p_head_done_one_cycle);
    assert property (p_flush_masks_interfaces);
    assert property (p_flush_clears_tracking);
    assert property (p_protocol_error_sticky);
    assert property (p_clear_error_clears_sticky);
    assert property (p_new_illegal_error_wins_clear);
    assert property (p_datapath_idle_exact);
    assert property (p_issued_term_count);

    generate
        for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
            property p_weight_request_stable;
                @(posedge clk_core) disable iff (rst_core)
                weight_req_valid[bank] && !weight_req_ready[bank] && !flush |=>
                    flush || (weight_req_valid[bank] &&
                    $stable({weight_req_tags[
                                 (bank*GROUP_TAG_W) +: GROUP_TAG_W],
                             weight_req_input_channels[
                                 (bank*INPUT_CH_W) +: INPUT_CH_W],
                             weight_req_output_tiles[
                                 (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W],
                             weight_req_epochs[
                                 (bank*EPOCH_W) +: EPOCH_W]}));
            endproperty

            property p_bank_done_matches_issued_record;
                @(posedge clk_core) disable iff (rst_core || flush)
                bank_term_done[bank] |-> bank_term_done_match[bank];
            endproperty

            property p_completed_not_ahead_of_issued;
                @(posedge clk_core) disable iff (rst_core)
                completed_terms[(bank*COUNTER_W) +: COUNTER_W] <=
                    issued_terms + COUNTER_W'(issue_term_fire);
            endproperty

            assert property (p_weight_request_stable);
            assert property (p_bank_done_matches_issued_record);
            assert property (p_completed_not_ahead_of_issued);
        end

        for (genvar channel = 0; channel < 6;
             channel = channel + 1) begin : g_acc_channel
            localparam int BANK = channel / 2;
            property p_acc_update_stable;
                @(posedge clk_core) disable iff (rst_core)
                acc_update_valid[channel] &&
                !acc_update_ready[channel] && !flush |=>
                    flush || (acc_update_valid[channel] &&
                    $stable({acc_update_token_ids[
                                 (channel*TOKEN_ID_W) +: TOKEN_ID_W],
                             acc_update_tags[
                                 (BANK*GROUP_TAG_W) +: GROUP_TAG_W],
                             acc_update_values[
                                 (BANK*OUT_TILE*PRODUCT_W) +:
                                 (OUT_TILE*PRODUCT_W)]}));
            endproperty

            assert property (p_acc_update_stable);
        end
    endgenerate

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    issue_term_fire && (fabric_occupancy != '0));
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    head_compute_done && (bank_term_done != '0));
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    adapter_term_ready && !illegal_drop_active_q &&
                    term_valid && !term_metadata_legal && term_ready);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    illegal_drop_active_q);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    illegal_event_fire && event_term_last);
endmodule

`default_nettype wire
