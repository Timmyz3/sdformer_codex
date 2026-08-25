`timescale 1ns/1ps
`default_nettype none

// M185: fixed-bank specialization of the multiplier-free K8 FC2 accumulator.
//
// Structural slot b is permanently owned by weight bank b.  The arbitrary
// bank-valid mask connects directly to M184 and removes M183's prefix packing,
// bank-ID payload, 28 pairwise uniqueness checks and external packing crossbar.
// The frozen-checkpoint threshold-one identity and all module-only exclusions
// remain unchanged.
module m185_fc2_k8_fixed_bank_accumulator #(
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
    input  logic signed [7:0]            issue_weight [0:7][0:LANES-1],
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

    output logic [8*LANES-1:0]           accepted_weight_active_mask,
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

    logic legal_issue;
    logic illegal_request;

    assign legal_issue = |issue_bank_valid;
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
        for (genvar bank = 0; bank < 8; bank++) begin : g_activity_bank
            for (genvar lane = 0; lane < LANES; lane++) begin : g_activity_lane
                assign accepted_weight_active_mask[(bank*LANES)+lane]
                    = issue_accept && issue_bank_valid[bank];
            end
        end
    endgenerate

    always_ff @(posedge clk_core) begin : elastic_accumulator_update
        logic [3:0] source_count;
        logic overflow_any;
        logic signed [8:0] pair_01;
        logic signed [8:0] pair_23;
        logic signed [8:0] pair_45;
        logic signed [8:0] pair_67;
        logic signed [9:0] quad_03;
        logic signed [9:0] quad_47;
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
                source_count = '0;
                for (int bank = 0; bank < 8; bank++)
                    source_count = source_count + issue_bank_valid[bank];
                result_valid_q <= 1'b1;
                result_tag_q <= issue_tag;
                result_last_q <= issue_last;
                result_source_count_q <= source_count;
                result_bank_mask_q <= issue_bank_valid;
                overflow_any = 1'b0;
                for (int lane = 0; lane < LANES; lane++) begin
                    pair_01 = (issue_bank_valid[0]
                            ? $signed({issue_weight[0][lane][7],
                                      issue_weight[0][lane]}) : 9'sd0)
                        + (issue_bank_valid[1]
                            ? $signed({issue_weight[1][lane][7],
                                      issue_weight[1][lane]}) : 9'sd0);
                    pair_23 = (issue_bank_valid[2]
                            ? $signed({issue_weight[2][lane][7],
                                      issue_weight[2][lane]}) : 9'sd0)
                        + (issue_bank_valid[3]
                            ? $signed({issue_weight[3][lane][7],
                                      issue_weight[3][lane]}) : 9'sd0);
                    pair_45 = (issue_bank_valid[4]
                            ? $signed({issue_weight[4][lane][7],
                                      issue_weight[4][lane]}) : 9'sd0)
                        + (issue_bank_valid[5]
                            ? $signed({issue_weight[5][lane][7],
                                      issue_weight[5][lane]}) : 9'sd0);
                    pair_67 = (issue_bank_valid[6]
                            ? $signed({issue_weight[6][lane][7],
                                      issue_weight[6][lane]}) : 9'sd0)
                        + (issue_bank_valid[7]
                            ? $signed({issue_weight[7][lane][7],
                                      issue_weight[7][lane]}) : 9'sd0);
                    quad_03 = $signed({pair_01[8], pair_01})
                        + $signed({pair_23[8], pair_23});
                    quad_47 = $signed({pair_45[8], pair_45})
                        + $signed({pair_67[8], pair_67});
                    weight_sum = $signed({quad_03[9], quad_03})
                        + $signed({quad_47[9], quad_47});
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
