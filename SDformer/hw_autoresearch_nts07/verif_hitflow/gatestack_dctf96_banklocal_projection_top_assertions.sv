`timescale 1ns/1ps
`default_nettype none

module gatestack_dctf96_banklocal_projection_top_assertions #(
    parameter int OUT_TILE = 32,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int TOKEN_ID_W = 8,
    parameter int OUTPUT_TILE_W = 8,
    parameter int LOGICAL_SUPERTILE_W = OUTPUT_TILE_W,
    parameter int HEAD_COUNT_W = 6,
    parameter int EPOCH_W = 4
) (
    input logic clk_core,
    input logic rst_core,
    input logic flush,
    input logic tile_start_valid,
    input logic tile_start_ready,
    input logic [TAG_W-1:0] tile_start_tag,
    input logic [LOGICAL_SUPERTILE_W-1:0] tile_start_logical_supertile,
    input logic [HEAD_COUNT_W-1:0] tile_start_head_count,
    input logic head_start_valid,
    input logic head_start_ready,
    input logic [TAG_W-1:0] head_start_tag,
    input logic [HEAD_COUNT_W-1:0] head_start_index,
    input logic [INPUT_CH_W-1:0] head_start_input_channel_base,
    input logic head_start_last,
    input logic term_ready,
    input logic event_ready,
    input logic source_done_ready,
    input logic head_done_valid,
    input logic head_done_ready,
    input logic [TAG_W-1:0] head_done_tag,
    input logic [HEAD_COUNT_W-1:0] head_done_index,
    input logic head_done_last,
    input logic head_done_error,
    input logic [2:0] weight_req_valid,
    input logic [2:0] weight_req_ready,
    input logic [(3*TAG_W)-1:0] weight_req_tags,
    input logic [(3*INPUT_CH_W)-1:0] weight_req_input_channels,
    input logic [(3*OUTPUT_TILE_W)-1:0] weight_req_output_tiles,
    input logic [(3*EPOCH_W)-1:0] weight_req_epochs,
    input logic [2:0] weight_rsp_ready,
    input logic [2:0] bias_req_valid,
    input logic [2:0] bias_req_ready,
    input logic [(3*TAG_W)-1:0] bias_req_tags,
    input logic [(3*OUTPUT_TILE_W)-1:0] bias_req_output_tiles,
    input logic [(3*TOKEN_ID_W)-1:0] bias_req_token_ids,
    input logic [(3*EPOCH_W)-1:0] bias_req_epochs,
    input logic [2:0] bias_rsp_valid,
    input logic [2:0] bias_rsp_ready,
    input logic [(3*EPOCH_W)-1:0] bias_rsp_epochs,
    input logic [5:0] final_valid,
    input logic [5:0] final_ready,
    input logic [(6*TOKEN_ID_W)-1:0] final_token_ids,
    input logic [(3*TAG_W)-1:0] final_tags,
    input logic [(6*OUT_TILE*ACC_W)-1:0] final_values,
    input logic tile_done_valid,
    input logic tile_done_ready,
    input logic [TAG_W-1:0] tile_done_tag,
    input logic tile_done_error,
    input logic protocol_error,
    input logic [2:0] state_q,
    input logic [TAG_W-1:0] tile_tag_q,
    input logic source_done_seen_q,
    input logic source_done_fire,
    input logic event_input_fire,
    input logic flush_q,
    input logic [EPOCH_W-1:0] bias_epoch_q,
    input logic [2:0] bias_outstanding_q,
    input logic [2:0] bias_rsp_stale,
    input logic [2:0] bias_rsp_match,
    input logic [2:0] bias_rsp_wrong_current,
    input logic [2:0] bias_commit_fire
);
    property p_flush_masks_external_interfaces;
        @(posedge clk_core) disable iff (rst_core)
        flush |-> !tile_start_ready && !head_start_ready && !term_ready &&
                  !event_ready && !source_done_ready && !head_done_valid &&
                  (weight_req_valid == '0) && (weight_rsp_ready == '0) &&
                  (bias_req_valid == '0) && (bias_rsp_ready == '0) &&
                  (final_valid == '0) && !tile_done_valid;
    endproperty

    property p_flush_returns_idle;
        @(posedge clk_core) disable iff (rst_core)
        flush |=> (state_q == 3'd0) && (bias_outstanding_q == '0);
    endproperty

    property p_tile_start_stable;
        @(posedge clk_core) disable iff (rst_core)
        tile_start_valid && !tile_start_ready && !flush |=>
            flush || (tile_start_valid &&
            $stable({tile_start_tag, tile_start_logical_supertile,
                     tile_start_head_count}));
    endproperty

    property p_head_start_stable;
        @(posedge clk_core) disable iff (rst_core)
        head_start_valid && !head_start_ready && !flush |=>
            flush || (head_start_valid &&
            $stable({head_start_tag, head_start_index,
                     head_start_input_channel_base, head_start_last}));
    endproperty

    property p_head_done_stable;
        @(posedge clk_core) disable iff (rst_core)
        head_done_valid && !head_done_ready && !flush |=>
            flush || (head_done_valid &&
            $stable({head_done_tag, head_done_index, head_done_last,
                     head_done_error}));
    endproperty

    property p_tile_done_stable;
        @(posedge clk_core) disable iff (rst_core)
        tile_done_valid && !tile_done_ready && !flush |=>
            flush || (tile_done_valid &&
            $stable({tile_done_tag, tile_done_error}));
    endproperty

    property p_done_tag_is_active;
        @(posedge clk_core) disable iff (rst_core || flush)
        (head_done_valid || tile_done_valid) |->
            ((head_done_valid ? head_done_tag : tile_done_tag) == tile_tag_q);
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
        protocol_error && !flush |=> flush || protocol_error;
    endproperty

    property p_concurrent_event_done_is_retained;
        @(posedge clk_core) disable iff (rst_core || flush)
        event_input_fire && source_done_fire |=> source_done_seen_q;
    endproperty

    property p_long_flush_epoch_stable;
        @(posedge clk_core) disable iff (rst_core)
        flush && flush_q |=> $stable(bias_epoch_q);
    endproperty

    assert property (p_flush_masks_external_interfaces);
    assert property (p_flush_returns_idle);
    assert property (p_tile_start_stable);
    assert property (p_head_start_stable);
    assert property (p_head_done_stable);
    assert property (p_tile_done_stable);
    assert property (p_done_tag_is_active);
    assert property (p_protocol_error_sticky);
    assert property (p_concurrent_event_done_is_retained);
    assert property (p_long_flush_epoch_stable);

    generate
        for (genvar bank = 0; bank < 3; bank = bank + 1) begin : g_bank
            property p_weight_request_stable;
                @(posedge clk_core) disable iff (rst_core)
                weight_req_valid[bank] && !weight_req_ready[bank] && !flush |=>
                    flush || (weight_req_valid[bank] &&
                    $stable({weight_req_tags[(bank*TAG_W) +: TAG_W],
                             weight_req_input_channels[
                                 (bank*INPUT_CH_W) +: INPUT_CH_W],
                             weight_req_output_tiles[
                                 (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W],
                             weight_req_epochs[(bank*EPOCH_W) +: EPOCH_W]}));
            endproperty

            property p_bias_request_stable;
                @(posedge clk_core) disable iff (rst_core)
                bias_req_valid[bank] && !bias_req_ready[bank] && !flush |=>
                    flush || (bias_req_valid[bank] &&
                    $stable({bias_req_tags[(bank*TAG_W) +: TAG_W],
                             bias_req_output_tiles[
                                 (bank*OUTPUT_TILE_W) +: OUTPUT_TILE_W],
                             bias_req_token_ids[
                                 (bank*TOKEN_ID_W) +: TOKEN_ID_W],
                             bias_req_epochs[(bank*EPOCH_W) +: EPOCH_W]}));
            endproperty

            property p_wrong_current_bias_rejected;
                @(posedge clk_core) disable iff (rst_core || flush)
                bias_rsp_valid[bank] && bias_rsp_wrong_current[bank] |->
                    bias_rsp_ready[bank] && protocol_error &&
                    !bias_commit_fire[bank] && bias_outstanding_q[bank];
            endproperty

            property p_wrong_current_preserves_outstanding;
                @(posedge clk_core) disable iff (rst_core || flush)
                bias_rsp_valid[bank] && bias_rsp_wrong_current[bank] |=>
                    flush || bias_outstanding_q[bank];
            endproperty

            property p_stale_bias_dropped;
                @(posedge clk_core) disable iff (rst_core || flush)
                bias_rsp_valid[bank] && bias_rsp_stale[bank] |->
                    bias_rsp_ready[bank] && !bias_commit_fire[bank] &&
                    (bias_rsp_epochs[(bank*EPOCH_W) +: EPOCH_W] !=
                     bias_epoch_q);
            endproperty

            property p_bias_commit_requires_match;
                @(posedge clk_core) disable iff (rst_core || flush)
                bias_commit_fire[bank] |-> bias_rsp_match[bank] &&
                                          bias_outstanding_q[bank];
            endproperty

            assert property (p_weight_request_stable);
            assert property (p_bias_request_stable);
            assert property (p_wrong_current_bias_rejected);
            assert property (p_wrong_current_preserves_outstanding);
            assert property (p_stale_bias_dropped);
            assert property (p_bias_commit_requires_match);
        end

        for (genvar port = 0; port < 6; port = port + 1) begin : g_final
            localparam int BANK = port / 2;
            property p_final_stable;
                @(posedge clk_core) disable iff (rst_core)
                final_valid[port] && !final_ready[port] && !flush |=>
                    flush || (final_valid[port] &&
                    $stable({final_token_ids[
                                 (port*TOKEN_ID_W) +: TOKEN_ID_W],
                             final_tags[(BANK*TAG_W) +: TAG_W],
                             final_values[(port*OUT_TILE*ACC_W) +:
                                          (OUT_TILE*ACC_W)]}));
            endproperty

            assert property (p_final_stable);
        end
    endgenerate

    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    state_q == 3'd4 && bias_rsp_wrong_current != '0);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    bias_rsp_valid != '0 && bias_rsp_stale != '0);
    cover property (@(posedge clk_core) disable iff (rst_core || flush)
                    final_valid != '0 && (final_valid & ~final_ready) != '0);
endmodule

`default_nettype wire
