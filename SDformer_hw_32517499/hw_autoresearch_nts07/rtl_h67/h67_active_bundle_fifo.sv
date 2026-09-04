`timescale 1ns/1ps
`default_nettype none

// 只保存bundle id与active mask；按bundle序、lane序逐pair消费。
module h67_active_bundle_fifo #(
    parameter int PAIRS = 225,
    parameter int BUNDLE_SIZE = 8,
    parameter int DEPTH = 32,
    parameter int BUNDLE_COUNT = (PAIRS + BUNDLE_SIZE - 1) / BUNDLE_SIZE,
    parameter int BUNDLE_ID_W = (BUNDLE_COUNT <= 1) ? 1 : $clog2(BUNDLE_COUNT),
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int OCC_W = $clog2(DEPTH + 1),
    parameter int LANE_W = (BUNDLE_SIZE <= 1) ? 1 : $clog2(BUNDLE_SIZE)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,

    input  logic                       enq_valid,
    output logic                       enq_ready,
    input  logic [BUNDLE_ID_W-1:0]     enq_bundle_id,
    input  logic [BUNDLE_SIZE-1:0]     enq_active_mask,

    output logic                       pair_valid,
    input  logic                       pair_ready,
    output logic [PAIR_ID_W-1:0]       pair_id,
    output logic [BUNDLE_ID_W-1:0]     pair_bundle_id,
    output logic [LANE_W-1:0]          pair_lane,

    output logic [OCC_W-1:0]           occupancy,
    output logic [OCC_W-1:0]           max_occupancy,
    output logic                       empty,
    output logic                       protocol_error
);
    logic [BUNDLE_ID_W-1:0] bundle_mem [0:DEPTH-1];
    logic [BUNDLE_SIZE-1:0] mask_mem [0:DEPTH-1];
    logic [PTR_W-1:0] rd_ptr_q;
    logic [PTR_W-1:0] wr_ptr_q;
    logic [OCC_W-1:0] count_q;
    logic [OCC_W-1:0] max_count_q;
    logic protocol_error_q;
    logic [LANE_W-1:0] selected_lane;
    logic selected_valid;
    logic [BUNDLE_SIZE-1:0] head_mask;
    logic [BUNDLE_SIZE-1:0] head_after_pop;
    logic pair_fire;
    logic pop_descriptor;
    logic enq_fire;

    always_comb begin
        selected_lane = '0;
        selected_valid = 1'b0;
        for (integer lane = 0; lane < BUNDLE_SIZE; lane = lane + 1)
            if (!selected_valid && head_mask[lane]) begin
                selected_lane = LANE_W'(lane);
                selected_valid = 1'b1;
            end
    end

    assign head_mask = count_q == 0 ? '0 : mask_mem[rd_ptr_q];
    assign head_after_pop = head_mask & ~(BUNDLE_SIZE'(1) << selected_lane);
    assign pair_valid = count_q != 0 && selected_valid;
    assign pair_bundle_id = count_q == 0 ? '0 : bundle_mem[rd_ptr_q];
    assign pair_lane = selected_lane;
    assign pair_id = PAIR_ID_W'(32'(pair_bundle_id) * BUNDLE_SIZE
                              + 32'(selected_lane));
    assign pair_fire = pair_valid && pair_ready;
    assign pop_descriptor = pair_fire && head_after_pop == 0;
    assign enq_ready = count_q < OCC_W'(DEPTH) || pop_descriptor;
    assign enq_fire = enq_valid && enq_ready;
    assign occupancy = count_q;
    assign max_occupancy = max_count_q;
    assign empty = count_q == 0;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core || window_start) begin
            rd_ptr_q <= '0;
            wr_ptr_q <= '0;
            count_q <= '0;
            max_count_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (enq_valid && enq_active_mask == 0)
                protocol_error_q <= 1'b1;
            if (enq_fire) begin
                bundle_mem[wr_ptr_q] <= enq_bundle_id;
                mask_mem[wr_ptr_q] <= enq_active_mask;
                wr_ptr_q <= wr_ptr_q == PTR_W'(DEPTH - 1)
                          ? '0 : wr_ptr_q + 1'b1;
            end

            if (pair_fire) begin
                if (32'(pair_id) >= 32'(PAIRS)) begin
                    protocol_error_q <= 1'b1;
                end else if (pop_descriptor) begin
                    rd_ptr_q <= rd_ptr_q == PTR_W'(DEPTH - 1)
                              ? '0 : rd_ptr_q + 1'b1;
                end else begin
                    mask_mem[rd_ptr_q] <= head_after_pop;
                end
            end

            case ({enq_fire, pop_descriptor})
                2'b10: count_q <= count_q + 1'b1;
                2'b01: count_q <= count_q - 1'b1;
                default: count_q <= count_q;
            endcase

            if (enq_fire && !pop_descriptor
                && count_q + 1'b1 > max_count_q)
                max_count_q <= count_q + 1'b1;
            if (count_q != 0 && !selected_valid)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
