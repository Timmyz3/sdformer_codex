`timescale 1ns/1ps
`default_nettype none

module m165_q8_owned_raw_bank_dynamic_bn_rank3_frontend_assertions #(
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
    input logic [17:0]                  moment_count,
    input logic signed [25:0]           moment_sum [0:15],
    input logic [31:0]                  moment_sumsq [0:15],
    input logic                         moment_accept,
    input logic                         configured,
    input logic                         channel_active,
    input logic                         protocol_error,
    input logic                         busy,
    input logic                         raw_push_internal,
    input logic                         raw_start_internal,
    input logic                         raw_release_internal,
    input logic                         quant_busy_internal,
    input logic [1:0]                   raw_count_internal,
    input logic                         raw_rd_ptr_internal
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
    ap_moment_count_within_frozen_h67_bound:
        assert property (moment_valid |-> moment_count <= 18'd192000);
    ap_raw_count_bounded:
        assert property (raw_count_internal <= 2);
    ap_raw_start_requires_owned_entry:
        assert property (raw_start_internal
            |-> raw_count_internal != 0 && !quant_busy_internal);
    ap_raw_read_pointer_stable_while_owned:
        assert property (quant_busy_internal && !raw_release_internal
            |=> $stable(raw_rd_ptr_internal));
    ap_raw_release_advances_read_pointer:
        assert property (raw_release_internal
            |=> $changed(raw_rd_ptr_internal));
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
    cp_positive_and_negative_half_ties:
        cover property (rank_valid && rank_tag == 16'h6314
            && rank_data[0][0] == 8'sd0
            && rank_data[0][1] == 8'sd2
            && rank_data[0][2] == 8'sd0
            && rank_data[0][3] == -8'sd2);
    cp_positive_and_negative_saturation:
        cover property (rank_valid && rank_tag == 16'h6315
            && rank_data[0][0] == 8'sd127
            && rank_data[0][1] == -8'sd128);
    cp_shift23_rounds_to_zero:
        cover property (rank_valid && rank_tag == 16'h6316
            && rank_data[0][0] == 8'sd0);
    cp_exact_h67_max_population_and_worst_q8_moments:
        cover property (moment_valid && moment_tag == 16'h6318
            && moment_count == 18'd192000
            && moment_sum[0] == -26'sd24576000
            && moment_sumsq[0] == 32'd3145728000);
    cp_raw_push_release_same_cycle:
        cover property (raw_push_internal && raw_release_internal);
    cp_raw_fifo_full_during_owned_service:
        cover property (quant_busy_internal && raw_count_internal == 2);
    cp_fault_with_pending_outputs:
        cover property (protocol_error && rank_valid && moment_valid);
endmodule

`default_nettype wire
