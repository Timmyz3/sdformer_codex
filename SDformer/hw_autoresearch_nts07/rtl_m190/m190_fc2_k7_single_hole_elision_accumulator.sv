`timescale 1ns/1ps
`default_nettype none

// M190: K7 single-hole-elision weight steering plus signed-INT8 Acc24.
//
// Every legal K7 mask has at least one empty structural bank.  M189 paid for
// stable prefix compaction of every active bank, even though the arithmetic
// only needs seven lanes whose invalid lanes contribute zero.  M190 selects
// the lowest empty bank as a hole and removes only that bank.  Packed lane s
// therefore chooses only structural bank s or s+1, reducing the steering
// network from a general prefix crossbar to seven adjacent 2:1 choices.  Any
// additional empty banks remain zero lanes.  This preserves the exact sum for
// every mask with popcount 1..7 and keeps the routing cost inside this module.
module m190_fc2_k7_single_hole_elision_accumulator #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         issue_valid,
    output logic                         issue_ready,
    input  logic [TAG_BITS-1:0]          issue_tag,
    input  logic                         issue_last,
    input  logic [7:0]                   issue_bank_valid,
    input  logic signed [7:0]            issue_weight_bank [0:7][0:LANES-1],
    input  logic signed [23:0]           issue_accumulator [0:LANES-1],

    output logic                         issue_accept,
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic                         result_last,
    output logic [3:0]                   result_source_count,
    output logic [7:0]                   result_bank_mask,
    output logic signed [23:0]           result_accumulator [0:LANES-1],
    output logic                         result_accept,

    output logic [8*LANES-1:0]           accepted_weight_bank_active_mask,
    output logic [7*LANES-1:0]           accepted_elided_lane_active_mask,
    output logic [2:0]                   accepted_hole_bank,
    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         busy
);
    logic fault_q;
    logic overflow_q;
    logic result_valid_q;
    logic [TAG_BITS-1:0] result_tag_q;
    logic result_last_q;
    logic [3:0] result_source_count_q;
    logic [7:0] result_bank_mask_q;
    logic signed [23:0] result_accumulator_q [0:LANES-1];

    logic [3:0] issue_source_count;
    logic [2:0] hole_bank;
    logic [6:0] elided_lane_valid;
    logic signed [7:0] elided_weight [0:6][0:LANES-1];
    logic legal_issue;
    logic illegal_request;

    always_comb begin : count_and_find_lowest_hole
        issue_source_count = '0;
        for (int bank = 0; bank < 8; bank++)
            issue_source_count = issue_source_count + issue_bank_valid[bank];
        if (!issue_bank_valid[0])
            hole_bank = 3'd0;
        else if (!issue_bank_valid[1])
            hole_bank = 3'd1;
        else if (!issue_bank_valid[2])
            hole_bank = 3'd2;
        else if (!issue_bank_valid[3])
            hole_bank = 3'd3;
        else if (!issue_bank_valid[4])
            hole_bank = 3'd4;
        else if (!issue_bank_valid[5])
            hole_bank = 3'd5;
        else if (!issue_bank_valid[6])
            hole_bank = 3'd6;
        else
            hole_bank = 3'd7;
    end

    generate
        for (genvar slot = 0; slot < 7; slot++) begin : g_adjacent_elision
            wire use_upper_bank = hole_bank <= slot;
            assign elided_lane_valid[slot] = use_upper_bank
                ? issue_bank_valid[slot+1] : issue_bank_valid[slot];
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                assign elided_weight[slot][lane] = !elided_lane_valid[slot]
                    ? 8'sd0
                    : (use_upper_bank
                        ? issue_weight_bank[slot+1][lane]
                        : issue_weight_bank[slot][lane]);
                assign accepted_elided_lane_active_mask[(slot*LANES)+lane]
                    = issue_accept && elided_lane_valid[slot];
            end
        end
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_activity
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                assign accepted_weight_bank_active_mask[(bank*LANES)+lane]
                    = issue_accept && issue_bank_valid[bank];
            end
        end
        for (genvar lane = 0; lane < LANES; lane++) begin : g_result
            assign result_accumulator[lane] = result_accumulator_q[lane];
        end
    endgenerate

    assign legal_issue = issue_source_count != 0
        && issue_source_count <= 7;
    assign illegal_request = issue_valid && !legal_issue;
    assign issue_ready = !fault_q && !overflow_q && legal_issue
        && (!result_valid_q || result_ready);
    assign issue_accept = issue_valid && issue_ready;
    assign result_valid = result_valid_q;
    assign result_accept = result_valid_q && result_ready;
    assign result_tag = result_tag_q;
    assign result_last = result_last_q;
    assign result_source_count = result_source_count_q;
    assign result_bank_mask = result_bank_mask_q;
    assign accepted_hole_bank = issue_accept ? hole_bank : 3'd0;
    assign protocol_error = fault_q || illegal_request;
    assign numeric_overflow = overflow_q;
    assign busy = result_valid_q;

    always_ff @(posedge clk_core) begin : elastic_accumulator_update
        logic overflow_any;
        logic signed [8:0] pair_01;
        logic signed [8:0] pair_23;
        logic signed [8:0] pair_45;
        logic signed [8:0] lone_6;
        logic signed [9:0] quad_03;
        logic signed [9:0] tri_46;
        logic signed [10:0] weight_sum;
        logic signed [24:0] extended_sum;
        if (rst_core) begin
            fault_q <= 1'b0;
            overflow_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_tag_q <= '0;
            result_last_q <= 1'b0;
            result_source_count_q <= '0;
            result_bank_mask_q <= '0;
            for (int lane = 0; lane < LANES; lane++)
                result_accumulator_q[lane] <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;
            if (result_accept && !issue_accept)
                result_valid_q <= 1'b0;
            if (issue_accept) begin
                result_valid_q <= 1'b1;
                result_tag_q <= issue_tag;
                result_last_q <= issue_last;
                result_source_count_q <= issue_source_count;
                result_bank_mask_q <= issue_bank_valid;
                overflow_any = 1'b0;
                for (int lane = 0; lane < LANES; lane++) begin
                    pair_01 = $signed({elided_weight[0][lane][7],
                                       elided_weight[0][lane]})
                        + $signed({elided_weight[1][lane][7],
                                   elided_weight[1][lane]});
                    pair_23 = $signed({elided_weight[2][lane][7],
                                       elided_weight[2][lane]})
                        + $signed({elided_weight[3][lane][7],
                                   elided_weight[3][lane]});
                    pair_45 = $signed({elided_weight[4][lane][7],
                                       elided_weight[4][lane]})
                        + $signed({elided_weight[5][lane][7],
                                   elided_weight[5][lane]});
                    lone_6 = $signed({elided_weight[6][lane][7],
                                      elided_weight[6][lane]});
                    quad_03 = $signed({pair_01[8], pair_01})
                        + $signed({pair_23[8], pair_23});
                    tri_46 = $signed({pair_45[8], pair_45})
                        + $signed({lone_6[8], lone_6});
                    weight_sum = $signed({quad_03[9], quad_03})
                        + $signed({tri_46[9], tri_46});
                    extended_sum = $signed({issue_accumulator[lane][23],
                                            issue_accumulator[lane]})
                        + $signed({{14{weight_sum[10]}}, weight_sum});
                    result_accumulator_q[lane] <= extended_sum[23:0];
                    if (extended_sum[24] != extended_sum[23])
                        overflow_any = 1'b1;
                end
                if (overflow_any)
                    overflow_q <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
