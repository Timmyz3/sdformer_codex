`timescale 1ns/1ps
`default_nettype none

// M189: honest 8-structural-bank to 7-lane compactor plus multiplier-free
// FC2 accumulator.
//
// M188 never selects more than seven structural banks.  This module measures
// the logic cost that is otherwise hidden by saying that the response width
// shrinks from eight lanes to seven: arbitrary nonempty structural masks are
// compacted, in increasing bank order, before the seven-input adder tree.
// Empty and eight-source requests fail closed.  The external accumulator is
// intentionally retained so this remains a module boundary, not complete FC2.
module m189_fc2_k7_bank_compacting_accumulator #(
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
    output logic [7*LANES-1:0]           accepted_compacted_lane_active_mask,
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
    logic legal_issue;
    logic illegal_request;
    logic signed [7:0] compacted_weight [0:6][0:LANES-1];

    always_comb begin : count_and_compact
        integer compact_slot;
        issue_source_count = '0;
        for (int bank = 0; bank < 8; bank++)
            issue_source_count = issue_source_count + issue_bank_valid[bank];

        for (int slot = 0; slot < 7; slot++) begin
            for (int lane = 0; lane < LANES; lane++)
                compacted_weight[slot][lane] = 8'sd0;
        end
        compact_slot = 0;
        for (int bank = 0; bank < 8; bank++) begin
            if (issue_bank_valid[bank] && compact_slot < 7) begin
                for (int lane = 0; lane < LANES; lane++)
                    compacted_weight[compact_slot][lane]
                        = issue_weight_bank[bank][lane];
                compact_slot = compact_slot + 1;
            end
        end
    end

    assign legal_issue = (issue_source_count != 0)
        && (issue_source_count <= 7);
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
    assign protocol_error = fault_q || illegal_request;
    assign numeric_overflow = overflow_q;
    assign busy = result_valid_q;

    generate
        for (genvar lane = 0; lane < LANES; lane++) begin : g_result
            assign result_accumulator[lane] = result_accumulator_q[lane];
        end
        for (genvar bank = 0; bank < 8; bank++) begin : g_bank_activity
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                assign accepted_weight_bank_active_mask[(bank*LANES)+lane]
                    = issue_accept && issue_bank_valid[bank];
            end
        end
        for (genvar slot = 0; slot < 7; slot++) begin : g_compacted_activity
            for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
                assign accepted_compacted_lane_active_mask[(slot*LANES)+lane]
                    = issue_accept && (slot < issue_source_count);
            end
        end
    endgenerate

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
                    pair_01 = $signed({compacted_weight[0][lane][7],
                                       compacted_weight[0][lane]})
                        + $signed({compacted_weight[1][lane][7],
                                   compacted_weight[1][lane]});
                    pair_23 = $signed({compacted_weight[2][lane][7],
                                       compacted_weight[2][lane]})
                        + $signed({compacted_weight[3][lane][7],
                                   compacted_weight[3][lane]});
                    pair_45 = $signed({compacted_weight[4][lane][7],
                                       compacted_weight[4][lane]})
                        + $signed({compacted_weight[5][lane][7],
                                   compacted_weight[5][lane]});
                    lone_6 = $signed({compacted_weight[6][lane][7],
                                      compacted_weight[6][lane]});
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
