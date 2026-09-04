`timescale 1ns/1ps
`default_nettype none

module qfit_local5_cross_head_tile_executor_assertions #(
    parameter int HEIGHT = 15,
    parameter int WIDTH = 15,
    parameter int TIME_PLANES = 2,
    parameter int OUT_DIM = 32,
    parameter int TAG_W = 24,
    parameter int HEAD_W = 5,
    parameter int OUTPUT_TILE_W = 5,
    parameter int ACC_W = 32,
    parameter bit USE_INPLACE_CROSS_HEAD_ACC = 1'b0,
    parameter bit VECTOR_RESULT_MODE = 1'b0,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int OUT_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic head_done_valid,
    input logic head_done_ready,
    input logic [TAG_W-1:0] head_done_tag,
    input logic [HEAD_W-1:0] head_done_input_head,
    input logic head_done_error,
    input logic tile_done_valid,
    input logic tile_done_ready,
    input logic [TAG_W-1:0] tile_done_tag,
    input logic tile_done_error,
    input logic tile_result_valid,
    input logic tile_result_ready,
    input logic [TAG_W-1:0] tile_result_tag,
    input logic [OUTPUT_TILE_W-1:0] tile_result_output_tile,
    input logic [PLANE_W-1:0] tile_result_plane,
    input logic [Y_W-1:0] tile_result_y,
    input logic [X_W-1:0] tile_result_x,
    input logic [OUT_W-1:0] tile_result_out,
    input logic signed [ACC_W-1:0] tile_result_data,
    input logic tile_result_last,
    input logic token_req_valid,
    input logic weight_req_valid,
    input logic protocol_error,
    input logic [31:0] perf_tiles,
    input logic [31:0] perf_heads,
    input logic [31:0] perf_partial_results,
    input logic [31:0] perf_accumulator_writes,
    input logic [31:0] perf_final_results,
    input logic child_result_fire,
    input logic child_result_matches,
    input logic memory_command_valid,
    input logic child_protocol_error,
    input logic in_error_state
);
    localparam int TOTAL_RESULTS = HEIGHT * WIDTH * TIME_PLANES * OUT_DIM;
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;

    property p_head_done_stable;
        @(posedge clk_core) disable iff (rst_core)
            head_done_valid && !head_done_ready
            |=> head_done_valid
                && $stable({head_done_tag, head_done_input_head,
                            head_done_error});
    endproperty

    property p_tile_done_stable;
        @(posedge clk_core) disable iff (rst_core)
            tile_done_valid && !tile_done_ready
            |=> tile_done_valid
                && $stable({tile_done_tag, tile_done_error});
    endproperty

    property p_tile_result_stable;
        @(posedge clk_core) disable iff (rst_core)
            tile_result_valid && !tile_result_ready
            |=> tile_result_valid
                && $stable({tile_result_tag, tile_result_output_tile,
                            tile_result_plane, tile_result_y, tile_result_x,
                            tile_result_out, tile_result_data,
                            tile_result_last});
    endproperty

    property p_tile_result_geometry;
        @(posedge clk_core) disable iff (rst_core)
            tile_result_valid
            |-> 32'(tile_result_plane) < TIME_PLANES
                && 32'(tile_result_y) < HEIGHT
                && 32'(tile_result_x) < WIDTH
                && 32'(tile_result_out) < OUT_DIM;
    endproperty

    property p_clean_head_has_complete_partial_ledger;
        @(posedge clk_core) disable iff (rst_core)
            head_done_valid && !head_done_error
            |-> (USE_INPLACE_CROSS_HEAD_ACC
                 ? (perf_partial_results == 0
                    && perf_accumulator_writes == 0)
                 : (perf_partial_results == perf_heads
                        * (VECTOR_RESULT_MODE ? TOTAL_TOKENS : TOTAL_RESULTS)
                    && perf_accumulator_writes == perf_partial_results));
    endproperty

    property p_clean_tile_has_complete_final_ledger;
        @(posedge clk_core) disable iff (rst_core)
            tile_done_valid && !tile_done_error
            |-> perf_final_results == perf_tiles * TOTAL_RESULTS;
    endproperty

    property p_tile_done_follows_last_result;
        @(posedge clk_core) disable iff (rst_core)
            $rose(tile_done_valid)
            |-> $past(tile_result_valid && tile_result_ready
                      && tile_result_last);
    endproperty

    property p_accumulator_writes_never_lead;
        @(posedge clk_core) disable iff (rst_core)
            perf_accumulator_writes <= perf_partial_results;
    endproperty

    property p_protocol_error_sticky;
        @(posedge clk_core) disable iff (rst_core)
            $past(protocol_error) |-> protocol_error;
    endproperty

    property p_error_stops_external_work;
        @(posedge clk_core) disable iff (rst_core)
            protocol_error
            |-> !token_req_valid && !weight_req_valid && !tile_result_valid;
    endproperty

    property p_bad_partial_has_no_memory_command;
        @(posedge clk_core) disable iff (rst_core)
            child_result_fire && !child_result_matches
            |-> !memory_command_valid;
    endproperty

    property p_child_error_reaches_terminal_state;
        @(posedge clk_core) disable iff (rst_core)
            child_protocol_error |=> protocol_error && in_error_state;
    endproperty

    assert property (p_head_done_stable);
    assert property (p_tile_done_stable);
    assert property (p_tile_result_stable);
    assert property (p_tile_result_geometry);
    assert property (p_clean_head_has_complete_partial_ledger);
    assert property (p_clean_tile_has_complete_final_ledger);
    generate
        if (!USE_INPLACE_CROSS_HEAD_ACC) begin : g_scalar_last_result_order
            assert property (p_tile_done_follows_last_result);
        end
    endgenerate
    assert property (p_accumulator_writes_never_lead);
    assert property (p_protocol_error_sticky);
    assert property (p_error_stops_external_work);
    assert property (p_bad_partial_has_no_memory_command);
    assert property (p_child_error_reaches_terminal_state);
endmodule

bind qfit_local5_cross_head_tile_executor
    qfit_local5_cross_head_tile_executor_assertions #(
        .HEIGHT(HEIGHT), .WIDTH(WIDTH), .TIME_PLANES(TIME_PLANES),
        .OUT_DIM(OUT_DIM), .TAG_W(TAG_W), .HEAD_W(HEAD_W),
        .OUTPUT_TILE_W(OUTPUT_TILE_W), .ACC_W(ACC_W),
        .USE_INPLACE_CROSS_HEAD_ACC(USE_INPLACE_CROSS_HEAD_ACC),
        .VECTOR_RESULT_MODE(VECTOR_RESULT_MODE),
        .PLANE_W(PLANE_W), .Y_W(Y_W), .X_W(X_W), .OUT_W(OUT_W)
    ) u_qfit_local5_cross_head_tile_executor_assertions (
        .clk_core(clk_core), .rst_core(rst_core),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_input_head(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag), .tile_done_error(tile_done_error),
        .tile_result_valid(tile_result_valid),
        .tile_result_ready(tile_result_ready),
        .tile_result_tag(tile_result_tag),
        .tile_result_output_tile(tile_result_output_tile),
        .tile_result_plane(tile_result_plane), .tile_result_y(tile_result_y),
        .tile_result_x(tile_result_x), .tile_result_out(tile_result_out),
        .tile_result_data(tile_result_data),
        .tile_result_last(tile_result_last),
        .token_req_valid(token_req_valid),
        .weight_req_valid(weight_req_valid),
        .protocol_error(protocol_error), .perf_tiles(perf_tiles),
        .perf_heads(perf_heads),
        .perf_partial_results(perf_partial_results),
        .perf_accumulator_writes(perf_accumulator_writes),
        .perf_final_results(perf_final_results),
        .child_result_fire(child_result_fire),
        .child_result_matches(child_result_matches),
        .memory_command_valid(memory_command_valid),
        .child_protocol_error(child_protocol_error),
        .in_error_state(tx_state_q == TX_ERROR)
    );

`default_nettype wire
