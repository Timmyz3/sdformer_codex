`timescale 1ns/1ps
`default_nettype none

module m163_q8_dynamic_bn_rank3_frontend_assertions #(
    parameter int TAG_BITS = 16
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         config_valid,
    input logic                         config_ready,
    input logic                         config_accept,
    input logic                         tile_valid,
    input logic                         tile_ready,
    input logic [TAG_BITS-1:0]          tile_tag,
    input logic [2:0]                   tile_beat,
    input logic                         tile_channel_start,
    input logic                         tile_channel_last,
    input logic signed [7:0]            tile_data [0:1][0:15],
    input logic                         tile_accept,
    input logic                         rank_valid,
    input logic                         rank_ready,
    input logic [TAG_BITS-1:0]          rank_tag,
    input logic                         rank_channel_last,
    input logic signed [7:0]            rank_data [0:2][0:15],
    input logic signed [11:0]           rank_factor_sum [0:2],
    input logic                         rank_accept,
    input logic                         moment_valid,
    input logic                         moment_ready,
    input logic [TAG_BITS-1:0]          moment_tag,
    input logic [31:0]                  moment_count,
    input logic signed [47:0]           moment_sum [0:15],
    input logic [55:0]                  moment_sumsq [0:15],
    input logic                         moment_accept,
    input logic                         configured,
    input logic                         channel_active,
    input logic                         protocol_error,
    input logic                         busy
);
    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_config_accept_definition:
        assert property (config_accept == (config_valid && config_ready));
    ap_tile_accept_definition:
        assert property (tile_accept == (tile_valid && tile_ready));
    ap_rank_accept_definition:
        assert property (rank_accept == (rank_valid && rank_ready));
    ap_moment_accept_definition:
        assert property (moment_accept == (moment_valid && moment_ready));
    ap_fault_is_sticky:
        assert property (protocol_error |=> protocol_error);
    ap_fault_closes_inputs:
        assert property (protocol_error |=> !config_ready && !tile_ready);
    ap_rank_metadata_stable_under_stall:
        assert property (rank_valid && !rank_ready
            |=> rank_valid && $stable({rank_tag, rank_channel_last,
                rank_factor_sum[0], rank_factor_sum[1],
                rank_factor_sum[2]}));
    ap_moment_stable_under_stall:
        assert property (moment_valid && !moment_ready
            |=> moment_valid && $stable({moment_tag, moment_count}));
    ap_busy_for_channel:
        assert property (channel_active |-> busy);
    ap_accepted_nonzero_beat_has_no_metadata:
        assert property (tile_accept && tile_beat != 0
            |-> !tile_channel_start && !tile_channel_last);
    ap_channel_start_is_beat_zero:
        assert property (tile_accept && tile_channel_start
            |-> tile_beat == 0);

    generate
        for (genvar rank = 0; rank < 3; rank++) begin : g_rank
            for (genvar lane = 0; lane < 16; lane++) begin : g_lane
                ap_rank_data_stable_under_stall:
                    assert property (rank_valid && !rank_ready
                        |=> $stable(rank_data[rank][lane]));
            end
        end
        for (genvar moment_lane = 0; moment_lane < 16;
                moment_lane++) begin : g_moment_lane
            ap_moment_lane_stable_under_stall:
                assert property (moment_valid && !moment_ready
                    |=> $stable({moment_sum[moment_lane],
                                 moment_sumsq[moment_lane]}));
        end
    endgenerate

    cp_five_beat_tile:
        cover property (tile_accept && tile_beat == 0
            ##1 tile_accept && tile_beat == 1
            ##1 tile_accept && tile_beat == 2
            ##1 tile_accept && tile_beat == 3
            ##1 tile_accept && tile_beat == 4);
    cp_rank_stall_then_accept:
        cover property (rank_valid && !rank_ready
            ##1 rank_valid && rank_ready);
    cp_moment_stall_then_accept:
        cover property (moment_valid && !moment_ready
            ##1 moment_valid && moment_ready);
    cp_negative_128_input:
        cover property (tile_accept && tile_data[0][0] == -8'sd128);
    cp_positive_127_input:
        cover property (tile_accept && tile_data[1][15] == 8'sd127);
    cp_channel_last_tile:
        cover property (tile_accept && tile_beat == 0 && tile_channel_last);
    cp_distinct_hidden_lane_moments:
        cover property (moment_valid
            && (moment_sum[0] != moment_sum[1]
                || moment_sumsq[0] != moment_sumsq[1]));
    cp_fault_with_pending_outputs:
        cover property (protocol_error && rank_valid && moment_valid);
endmodule

`default_nettype wire
