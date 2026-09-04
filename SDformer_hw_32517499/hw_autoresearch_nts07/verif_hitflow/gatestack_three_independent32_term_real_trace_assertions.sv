`timescale 1ns/1ps
`default_nettype none

module gatestack_three_independent32_term_real_trace_assertions #(
    parameter int EVENT_WAYS = 4,
    parameter int BANKS = 2,
    parameter int GATE_W = 9,
    parameter int ACC_W = 32,
    parameter int TAG_W = 32,
    parameter int INPUT_CH_W = 10,
    parameter int OUTPUT_TILE_W = 8,
    parameter int ISSUE_SEQ_W = 13,
    parameter int TOKEN_ID_W = 8,
    parameter int LANE_ID_W = 5,
    parameter int WAY_COUNT_W = 3
) (
    input logic clk_core,
    input logic rst_core,
    input logic [2:0] term_valid,
    input logic [2:0] term_ready,
    input logic [(3*GATE_W)-1:0] term_gate_codes,
    input logic [(3*LANE_ID_W)-1:0] term_lane_ids,
    input logic [(3*8)-1:0] term_destination_counts,
    input logic [(3*ISSUE_SEQ_W)-1:0] term_issue_seqs,
    input logic [2:0] term_head_last,
    input logic [2:0] event_valid,
    input logic [2:0] event_ready,
    input logic [(3*GATE_W)-1:0] event_gate_codes,
    input logic [(3*LANE_ID_W)-1:0] event_lane_ids,
    input logic [(3*EVENT_WAYS)-1:0] event_token_valids,
    input logic [(3*EVENT_WAYS*TOKEN_ID_W)-1:0] event_token_ids,
    input logic [(3*WAY_COUNT_W)-1:0] event_counts,
    input logic [(3*ISSUE_SEQ_W)-1:0] event_issue_seqs,
    input logic [2:0] event_term_first,
    input logic [2:0] event_term_last,
    input logic [2:0] event_head_last,
    input logic [2:0] source_done_valid,
    input logic [2:0] source_done_ready,
    input logic [(3*TAG_W)-1:0] source_done_tags,
    input logic [2:0] source_done_error,
    input logic [2:0] weight_req_valid,
    input logic [2:0] weight_req_ready,
    input logic [(3*TAG_W)-1:0] weight_req_tags,
    input logic [(3*INPUT_CH_W)-1:0] weight_req_input_channels,
    input logic [(3*OUTPUT_TILE_W)-1:0] weight_req_output_tiles,
    input logic [2:0] weight_rsp_valid,
    input logic [2:0] bias_req_valid,
    input logic [2:0] bias_req_ready,
    input logic [(3*TAG_W)-1:0] bias_req_tags,
    input logic [(3*OUTPUT_TILE_W)-1:0] bias_req_output_tiles,
    input logic [(3*TOKEN_ID_W)-1:0] bias_req_token_ids,
    input logic [2:0] bias_rsp_valid,
    input logic [(3*BANKS)-1:0] final_valid,
    input logic [(3*BANKS)-1:0] final_ready,
    input logic [(3*BANKS*TOKEN_ID_W)-1:0] final_token_ids,
    input logic [(3*TAG_W)-1:0] final_tags,
    input logic [(3*BANKS*32*ACC_W)-1:0] final_values
);
    generate
        for (genvar engine = 0; engine < 3; engine = engine + 1) begin : g_engine
            property p_term_stable;
                @(posedge clk_core) disable iff (rst_core)
                term_valid[engine] && !term_ready[engine] |=>
                    term_valid[engine] &&
                    $stable({term_gate_codes[(engine*GATE_W) +: GATE_W],
                             term_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W],
                             term_destination_counts[(engine*8) +: 8],
                             term_issue_seqs[
                                 (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W],
                             term_head_last[engine]});
            endproperty

            property p_event_stable;
                @(posedge clk_core) disable iff (rst_core)
                event_valid[engine] && !event_ready[engine] |=>
                    event_valid[engine] &&
                    $stable({event_gate_codes[(engine*GATE_W) +: GATE_W],
                             event_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W],
                             event_token_valids[
                                 (engine*EVENT_WAYS) +: EVENT_WAYS],
                             event_token_ids[
                                 (engine*EVENT_WAYS*TOKEN_ID_W) +:
                                 (EVENT_WAYS*TOKEN_ID_W)],
                             event_counts[
                                 (engine*WAY_COUNT_W) +: WAY_COUNT_W],
                             event_issue_seqs[
                                 (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W],
                             event_term_first[engine],
                             event_term_last[engine],
                             event_head_last[engine]});
            endproperty

            property p_source_done_stable;
                @(posedge clk_core) disable iff (rst_core)
                source_done_valid[engine] && !source_done_ready[engine] |=>
                    source_done_valid[engine] &&
                    $stable({source_done_tags[(engine*TAG_W) +: TAG_W],
                             source_done_error[engine]});
            endproperty

            property p_weight_request_stable;
                @(posedge clk_core) disable iff (rst_core)
                weight_req_valid[engine] && !weight_req_ready[engine] |=>
                    weight_req_valid[engine] &&
                    $stable({weight_req_tags[(engine*TAG_W) +: TAG_W],
                             weight_req_input_channels[
                                 (engine*INPUT_CH_W) +: INPUT_CH_W],
                             weight_req_output_tiles[
                                 (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W]});
            endproperty

            property p_bias_request_stable;
                @(posedge clk_core) disable iff (rst_core)
                bias_req_valid[engine] && !bias_req_ready[engine] |=>
                    bias_req_valid[engine] &&
                    $stable({bias_req_tags[(engine*TAG_W) +: TAG_W],
                             bias_req_output_tiles[
                                 (engine*OUTPUT_TILE_W) +: OUTPUT_TILE_W],
                             bias_req_token_ids[
                                 (engine*TOKEN_ID_W) +: TOKEN_ID_W]});
            endproperty

            property p_weight_fixed_one_cycle;
                @(posedge clk_core) disable iff (rst_core)
                weight_req_valid[engine] && weight_req_ready[engine] |=>
                    weight_rsp_valid[engine];
            endproperty

            property p_bias_fixed_one_cycle;
                @(posedge clk_core) disable iff (rst_core)
                bias_req_valid[engine] && bias_req_ready[engine] |=>
                    bias_rsp_valid[engine];
            endproperty

            assert property (p_term_stable);
            assert property (p_event_stable);
            assert property (p_source_done_stable);
            assert property (p_weight_request_stable);
            assert property (p_bias_request_stable);
            assert property (p_weight_fixed_one_cycle);
            assert property (p_bias_fixed_one_cycle);

            if (engine > 0) begin : g_same_stream
                property p_term_payload_matches_engine0;
                    @(posedge clk_core) disable iff (rst_core)
                    term_valid[0] && term_valid[engine] |->
                        term_gate_codes[0 +: GATE_W] ==
                            term_gate_codes[(engine*GATE_W) +: GATE_W] &&
                        term_lane_ids[0 +: LANE_ID_W] ==
                            term_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W] &&
                        term_destination_counts[0 +: 8] ==
                            term_destination_counts[(engine*8) +: 8] &&
                        term_issue_seqs[0 +: ISSUE_SEQ_W] ==
                            term_issue_seqs[
                                (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W] &&
                        term_head_last[0] == term_head_last[engine];
                endproperty

                property p_event_payload_matches_engine0;
                    @(posedge clk_core) disable iff (rst_core)
                    event_valid[0] && event_valid[engine] |->
                        event_gate_codes[0 +: GATE_W] ==
                            event_gate_codes[(engine*GATE_W) +: GATE_W] &&
                        event_lane_ids[0 +: LANE_ID_W] ==
                            event_lane_ids[(engine*LANE_ID_W) +: LANE_ID_W] &&
                        event_token_valids[0 +: EVENT_WAYS] ==
                            event_token_valids[
                                (engine*EVENT_WAYS) +: EVENT_WAYS] &&
                        event_token_ids[0 +: (EVENT_WAYS*TOKEN_ID_W)] ==
                            event_token_ids[
                                (engine*EVENT_WAYS*TOKEN_ID_W) +:
                                (EVENT_WAYS*TOKEN_ID_W)] &&
                        event_counts[0 +: WAY_COUNT_W] ==
                            event_counts[
                                (engine*WAY_COUNT_W) +: WAY_COUNT_W] &&
                        event_issue_seqs[0 +: ISSUE_SEQ_W] ==
                            event_issue_seqs[
                                (engine*ISSUE_SEQ_W) +: ISSUE_SEQ_W] &&
                        event_term_first[0] == event_term_first[engine] &&
                        event_term_last[0] == event_term_last[engine] &&
                        event_head_last[0] == event_head_last[engine];
                endproperty

                assert property (p_term_payload_matches_engine0);
                assert property (p_event_payload_matches_engine0);
            end
        end

        for (genvar port = 0; port < 3*BANKS; port = port + 1) begin : g_final
            localparam int ENGINE = port / BANKS;
            property p_final_stable;
                @(posedge clk_core) disable iff (rst_core)
                final_valid[port] && !final_ready[port] |=>
                    final_valid[port] &&
                    $stable({final_token_ids[
                                 (port*TOKEN_ID_W) +: TOKEN_ID_W],
                             final_tags[(ENGINE*TAG_W) +: TAG_W],
                             final_values[
                                 (port*32*ACC_W) +: (32*ACC_W)]});
            endproperty
            assert property (p_final_stable);
        end
    endgenerate
endmodule

`default_nettype wire
