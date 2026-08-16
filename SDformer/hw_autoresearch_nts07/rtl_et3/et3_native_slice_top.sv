`timescale 1ns/1ps
`default_nettype none

// End-to-end ET3 prototype:
// SET/MULTISET source items -> bounded directory -> segmented typed terms
// -> native multiplicity-aware executor -> accumulator.
module et3_native_slice_top #(
    parameter int KEY_CAP = 4,
    parameter int SEG_DEPTH = 4,
    parameter int FALLBACK_DEPTH = 8,
    parameter int HEAD_DIM = 4,
    parameter int OUT_DIM = 4,
    parameter int MAX_DEST = 16,
    parameter int TAG_W = 16,
    parameter int GATE_W = 9,
    parameter int LANE_W = 5,
    parameter int MULT_W = 3,
    parameter int DEST_W = 8,
    parameter int WEIGHT_W = 8,
    parameter int ACC_W = 32,
    parameter int COUNTER_W = 32,
    parameter int OUT_ID_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM)
) (
    input  logic                             clk_core,
    input  logic                             rst_core,
    input  logic                             flush,

    input  logic                             weight_load_valid,
    output logic                             weight_load_ready,
    input  logic [LANE_W-1:0]                weight_load_lane,
    input  logic [OUT_ID_W-1:0]              weight_load_output,
    input  logic signed [WEIGHT_W-1:0]       weight_load_value,
    input  logic                             weight_load_last,
    input  logic                             run_start,
    output logic                             run_active,

    input  logic                             source_valid,
    output logic                             source_ready,
    input  logic [TAG_W-1:0]                 source_group_tag,
    input  logic                             source_mode_multiset,
    input  logic [GATE_W-1:0]                source_gate_code,
    input  logic [LANE_W-1:0]                source_lane_id,
    input  logic [MULT_W-1:0]                source_multiplicity,
    input  logic [DEST_W-1:0]                source_destination,
    input  logic                             source_head_last,
    input  logic                             group_close_valid,
    output logic                             group_close_ready,
    input  logic [TAG_W-1:0]                 group_close_tag,

    input  logic                             acc_write_ready,
    input  logic                             acc_read_valid,
    input  logic [DEST_W-1:0]                acc_read_destination,
    input  logic [OUT_ID_W-1:0]              acc_read_output,
    output logic                             acc_read_data_valid,
    output logic signed [ACC_W-1:0]          acc_read_data,

    output logic                             group_done,
    output logic                             protocol_error,

    output logic                             trace_cmd_valid,
    output logic                             trace_cmd_ready,
    output logic [TAG_W-1:0]                 trace_cmd_group_tag,
    output logic                             trace_cmd_mode_multiset,
    output logic [GATE_W-1:0]                trace_cmd_gate_code,
    output logic [LANE_W-1:0]                trace_cmd_lane_id,
    output logic [MULT_W-1:0]                trace_cmd_multiplicity,
    output logic [DEST_W-1:0]                trace_cmd_destination,
    output logic                             trace_cmd_term_first,
    output logic                             trace_cmd_term_last,
    output logic                             trace_cmd_head_last,
    output logic                             trace_cmd_fallback,

    output logic [COUNTER_W-1:0]             count_source_items,
    output logic [COUNTER_W-1:0]             count_directory_entries,
    output logic [COUNTER_W-1:0]             count_fallback_items,
    output logic [COUNTER_W-1:0]             count_typed_terms,
    output logic [COUNTER_W-1:0]             count_destination_beats,
    output logic [COUNTER_W-1:0]             count_partial_drains,
    output logic [COUNTER_W-1:0]             count_product_computes,
    output logic [COUNTER_W-1:0]             count_native_commands,
    output logic [COUNTER_W-1:0]             count_explode_baseline_commands,
    output logic [COUNTER_W-1:0]             count_fallback_terms,
    output logic [COUNTER_W-1:0]             count_set_terms,
    output logic [COUNTER_W-1:0]             count_multiset_terms
);
    logic directory_error;
    logic executor_error;
    logic directory_done;
    logic executor_done;
    logic directory_source_ready;
    logic directory_group_close_ready;
    logic empty_group_commit;

    assign source_ready = run_active && !directory_done &&
                          directory_source_ready;
    assign group_close_ready = run_active && !directory_done &&
                               directory_group_close_ready;
    assign empty_group_commit = directory_done && !executor_done;
    assign group_done = executor_done;

    et3_bounded_term_directory #(
        .KEY_CAP(KEY_CAP),
        .SEG_DEPTH(SEG_DEPTH),
        .FALLBACK_DEPTH(FALLBACK_DEPTH),
        .TAG_W(TAG_W),
        .GATE_W(GATE_W),
        .LANE_W(LANE_W),
        .MULT_W(MULT_W),
        .DEST_W(DEST_W),
        .COUNTER_W(COUNTER_W)
    ) u_directory (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .source_valid(source_valid && run_active),
        .source_ready(directory_source_ready),
        .source_group_tag(source_group_tag),
        .source_mode_multiset(source_mode_multiset),
        .source_gate_code(source_gate_code),
        .source_lane_id(source_lane_id),
        .source_multiplicity(source_multiplicity),
        .source_destination(source_destination),
        .source_head_last(source_head_last),
        .group_close_valid(group_close_valid && run_active),
        .group_close_ready(directory_group_close_ready),
        .group_close_tag(group_close_tag),
        .cmd_valid(trace_cmd_valid),
        .cmd_ready(trace_cmd_ready),
        .cmd_group_tag(trace_cmd_group_tag),
        .cmd_mode_multiset(trace_cmd_mode_multiset),
        .cmd_gate_code(trace_cmd_gate_code),
        .cmd_lane_id(trace_cmd_lane_id),
        .cmd_multiplicity(trace_cmd_multiplicity),
        .cmd_destination(trace_cmd_destination),
        .cmd_term_first(trace_cmd_term_first),
        .cmd_term_last(trace_cmd_term_last),
        .cmd_head_last(trace_cmd_head_last),
        .cmd_fallback(trace_cmd_fallback),
        .group_emit_done(directory_done),
        .protocol_error(directory_error),
        .count_source_items(count_source_items),
        .count_directory_entries(count_directory_entries),
        .count_fallback_items(count_fallback_items),
        .count_typed_terms(count_typed_terms),
        .count_destination_beats(count_destination_beats),
        .count_partial_drains(count_partial_drains)
    );

    et3_native_multiset_executor #(
        .HEAD_DIM(HEAD_DIM),
        .OUT_DIM(OUT_DIM),
        .MAX_DEST(MAX_DEST),
        .TAG_W(TAG_W),
        .GATE_W(GATE_W),
        .LANE_W(LANE_W),
        .MULT_W(MULT_W),
        .DEST_W(DEST_W),
        .WEIGHT_W(WEIGHT_W),
        .ACC_W(ACC_W),
        .COUNTER_W(COUNTER_W)
    ) u_executor (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .flush(flush),
        .weight_load_valid(weight_load_valid),
        .weight_load_ready(weight_load_ready),
        .weight_load_lane(weight_load_lane),
        .weight_load_output(weight_load_output),
        .weight_load_value(weight_load_value),
        .weight_load_last(weight_load_last),
        .run_start(run_start),
        .run_active(run_active),
        .empty_group_commit(empty_group_commit),
        .cmd_valid(trace_cmd_valid),
        .cmd_ready(trace_cmd_ready),
        .cmd_group_tag(trace_cmd_group_tag),
        .cmd_mode_multiset(trace_cmd_mode_multiset),
        .cmd_gate_code(trace_cmd_gate_code),
        .cmd_lane_id(trace_cmd_lane_id),
        .cmd_multiplicity(trace_cmd_multiplicity),
        .cmd_destination(trace_cmd_destination),
        .cmd_term_first(trace_cmd_term_first),
        .cmd_term_last(trace_cmd_term_last),
        .cmd_head_last(trace_cmd_head_last),
        .cmd_fallback(trace_cmd_fallback),
        .acc_write_ready(acc_write_ready),
        .acc_read_valid(acc_read_valid),
        .acc_read_destination(acc_read_destination),
        .acc_read_output(acc_read_output),
        .acc_read_data_valid(acc_read_data_valid),
        .acc_read_data(acc_read_data),
        .group_done(executor_done),
        .protocol_error(executor_error),
        .count_product_computes(count_product_computes),
        .count_native_commands(count_native_commands),
        .count_explode_baseline_commands(
            count_explode_baseline_commands
        ),
        .count_fallback_terms(count_fallback_terms),
        .count_set_terms(count_set_terms),
        .count_multiset_terms(count_multiset_terms)
    );

    assign protocol_error = directory_error || executor_error;

endmodule

`default_nettype wire
