`timescale 1ns/1ps
`default_nettype none

// Fair per-destination native-m baseline.
// It shares the same multiplicity-aware executor as ET3, but every accepted
// source item is a single-destination term. No cross-destination aggregation
// or product reuse is performed.
module et3_native_m_queue_baseline #(
    parameter int FIFO_DEPTH = 8,
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
    parameter int OUT_ID_W = (OUT_DIM <= 1) ? 1 : $clog2(OUT_DIM),
    parameter int FIFO_ADDR_W = (FIFO_DEPTH <= 1) ? 1 :
                                $clog2(FIFO_DEPTH),
    parameter int FIFO_COUNT_W = (FIFO_DEPTH <= 1) ? 1 :
                                 $clog2(FIFO_DEPTH + 1)
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
    output logic                             trace_cmd_head_last,
    output logic [COUNTER_W-1:0]             count_source_items,
    output logic [COUNTER_W-1:0]             count_queue_commands,
    output logic [COUNTER_W-1:0]             count_close_markers,
    output logic [COUNTER_W-1:0]             count_product_computes,
    output logic [COUNTER_W-1:0]             count_native_commands,
    output logic [COUNTER_W-1:0]             count_explode_commands,
    output logic [COUNTER_W-1:0]             count_fallback_terms,
    output logic [COUNTER_W-1:0]             count_set_terms,
    output logic [COUNTER_W-1:0]             count_multiset_terms,
    output logic [FIFO_COUNT_W-1:0]          max_fifo_occupancy
);
    logic fifo_marker_q [0:FIFO_DEPTH-1];
    logic [TAG_W-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic fifo_mode_q [0:FIFO_DEPTH-1];
    logic [GATE_W-1:0] fifo_gate_q [0:FIFO_DEPTH-1];
    logic [LANE_W-1:0] fifo_lane_q [0:FIFO_DEPTH-1];
    logic [MULT_W-1:0] fifo_mult_q [0:FIFO_DEPTH-1];
    logic [DEST_W-1:0] fifo_dest_q [0:FIFO_DEPTH-1];
    logic fifo_head_last_q [0:FIFO_DEPTH-1];

    logic [FIFO_ADDR_W-1:0] write_ptr_q;
    logic [FIFO_ADDR_W-1:0] read_ptr_q;
    logic [FIFO_COUNT_W-1:0] fifo_count_q;
    logic group_active_q;
    logic group_sealed_q;
    logic [TAG_W-1:0] group_tag_q;
    logic local_error_q;
    logic executor_error;
    logic executor_done;

    logic source_contract_ok;
    logic source_fire;
    logic close_only_fire;
    logic queue_push;
    logic queue_pop;
    logic queue_head_valid;
    logic queue_head_marker;
    logic marker_commit;
    logic executor_cmd_valid;
    logic executor_cmd_ready;

    assign source_contract_ok =
        (source_gate_code != '0) &&
        (source_multiplicity != '0) &&
        (32'(source_multiplicity) <= 5) &&
        (source_mode_multiset ||
         (source_multiplicity == MULT_W'(1))) &&
        (!group_active_q || (source_group_tag == group_tag_q));
    assign source_ready = run_active && !group_sealed_q &&
                          (
                              (32'(fifo_count_q) < FIFO_DEPTH) ||
                              queue_pop
                          ) &&
                          source_contract_ok;
    assign source_fire = source_valid && source_ready;

    assign group_close_ready = run_active && !group_sealed_q &&
        (
            (
                !source_valid &&
                (
                    (32'(fifo_count_q) < FIFO_DEPTH) ||
                    queue_pop
                ) &&
                (!group_active_q || (group_close_tag == group_tag_q))
            ) ||
            (
                source_valid && source_ready && source_head_last &&
                (group_close_tag == source_group_tag)
            )
        );
    assign close_only_fire = group_close_valid && group_close_ready &&
                             !source_valid;
    assign queue_push = source_fire || close_only_fire;

    assign queue_head_valid = fifo_count_q != '0;
    assign queue_head_marker = queue_head_valid &&
                               fifo_marker_q[read_ptr_q];
    assign executor_cmd_valid = queue_head_valid && !queue_head_marker;
    assign marker_commit = queue_head_marker && run_active;
    assign queue_pop = (executor_cmd_valid && executor_cmd_ready) ||
                       marker_commit;

    assign trace_cmd_valid = executor_cmd_valid;
    assign trace_cmd_ready = executor_cmd_ready;
    assign trace_cmd_group_tag = fifo_tag_q[read_ptr_q];
    assign trace_cmd_mode_multiset = fifo_mode_q[read_ptr_q];
    assign trace_cmd_gate_code = fifo_gate_q[read_ptr_q];
    assign trace_cmd_lane_id = fifo_lane_q[read_ptr_q];
    assign trace_cmd_multiplicity = fifo_mult_q[read_ptr_q];
    assign trace_cmd_destination = fifo_dest_q[read_ptr_q];
    assign trace_cmd_head_last = fifo_head_last_q[read_ptr_q];

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
        .empty_group_commit(marker_commit),
        .cmd_valid(executor_cmd_valid),
        .cmd_ready(executor_cmd_ready),
        .cmd_group_tag(fifo_tag_q[read_ptr_q]),
        .cmd_mode_multiset(fifo_mode_q[read_ptr_q]),
        .cmd_gate_code(fifo_gate_q[read_ptr_q]),
        .cmd_lane_id(fifo_lane_q[read_ptr_q]),
        .cmd_multiplicity(fifo_mult_q[read_ptr_q]),
        .cmd_destination(fifo_dest_q[read_ptr_q]),
        .cmd_term_first(1'b1),
        .cmd_term_last(1'b1),
        .cmd_head_last(fifo_head_last_q[read_ptr_q]),
        .cmd_fallback(1'b0),
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
        .count_explode_baseline_commands(count_explode_commands),
        .count_fallback_terms(count_fallback_terms),
        .count_set_terms(count_set_terms),
        .count_multiset_terms(count_multiset_terms)
    );

    assign group_done = executor_done;
    assign protocol_error = local_error_q || executor_error;

    always_ff @(posedge clk_core) begin
        if (rst_core || flush) begin
            write_ptr_q <= '0;
            read_ptr_q <= '0;
            fifo_count_q <= '0;
            group_active_q <= 1'b0;
            group_sealed_q <= 1'b0;
            group_tag_q <= '0;
            local_error_q <= 1'b0;
            count_source_items <= '0;
            count_queue_commands <= '0;
            count_close_markers <= '0;
            max_fifo_occupancy <= '0;
            for (int entry = 0; entry < FIFO_DEPTH; entry++) begin
                fifo_marker_q[entry] <= 1'b0;
                fifo_tag_q[entry] <= '0;
                fifo_mode_q[entry] <= 1'b0;
                fifo_gate_q[entry] <= '0;
                fifo_lane_q[entry] <= '0;
                fifo_mult_q[entry] <= '0;
                fifo_dest_q[entry] <= '0;
                fifo_head_last_q[entry] <= 1'b0;
            end
        end else begin
            if (run_start) begin
                if (run_active || (fifo_count_q != '0)) begin
                    local_error_q <= 1'b1;
                end else begin
                    group_active_q <= 1'b0;
                    group_sealed_q <= 1'b0;
                    group_tag_q <= '0;
                    count_source_items <= '0;
                    count_queue_commands <= '0;
                    count_close_markers <= '0;
                    max_fifo_occupancy <= '0;
                end
            end

            if (source_valid && run_active && !source_contract_ok) begin
                local_error_q <= 1'b1;
            end

            if (queue_push) begin
                if (source_fire) begin
                    fifo_marker_q[write_ptr_q] <= 1'b0;
                    fifo_tag_q[write_ptr_q] <= source_group_tag;
                    fifo_mode_q[write_ptr_q] <= source_mode_multiset;
                    fifo_gate_q[write_ptr_q] <= source_gate_code;
                    fifo_lane_q[write_ptr_q] <= source_lane_id;
                    fifo_mult_q[write_ptr_q] <= source_multiplicity;
                    fifo_dest_q[write_ptr_q] <= source_destination;
                    fifo_head_last_q[write_ptr_q] <= source_head_last;
                    count_source_items <= count_source_items + 1'b1;
                    if (!group_active_q) begin
                        group_active_q <= 1'b1;
                        group_tag_q <= source_group_tag;
                    end
                    if (source_head_last) begin
                        group_sealed_q <= 1'b1;
                    end
                end else begin
                    fifo_marker_q[write_ptr_q] <= 1'b1;
                    fifo_tag_q[write_ptr_q] <= group_close_tag;
                    fifo_mode_q[write_ptr_q] <= 1'b0;
                    fifo_gate_q[write_ptr_q] <= '0;
                    fifo_lane_q[write_ptr_q] <= '0;
                    fifo_mult_q[write_ptr_q] <= '0;
                    fifo_dest_q[write_ptr_q] <= '0;
                    fifo_head_last_q[write_ptr_q] <= 1'b0;
                    count_close_markers <= count_close_markers + 1'b1;
                    group_sealed_q <= 1'b1;
                    if (!group_active_q) begin
                        group_active_q <= 1'b1;
                        group_tag_q <= group_close_tag;
                    end
                end
                if (write_ptr_q == FIFO_ADDR_W'(FIFO_DEPTH - 1)) begin
                    write_ptr_q <= '0;
                end else begin
                    write_ptr_q <= write_ptr_q + 1'b1;
                end
            end

            if (queue_pop) begin
                if (!queue_head_marker) begin
                    count_queue_commands <= count_queue_commands + 1'b1;
                end
                if (read_ptr_q == FIFO_ADDR_W'(FIFO_DEPTH - 1)) begin
                    read_ptr_q <= '0;
                end else begin
                    read_ptr_q <= read_ptr_q + 1'b1;
                end
            end

            case ({queue_push, queue_pop})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase
            if (queue_push && !queue_pop &&
                (fifo_count_q + 1'b1 > max_fifo_occupancy)) begin
                max_fifo_occupancy <= fifo_count_q + 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
