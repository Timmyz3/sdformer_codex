`timescale 1ns/1ps
`default_nettype none

// M169: multiplier-free K4 FC2 event update island.
//
// Each accepted issue carries one to four prefix-packed events from distinct
// input_channel-modulo-8 weight banks.  The frozen H67 checkpoint has an exact
// sn2 ATLIF nonzero amplitude of one, so each event contributes one signed
// INT8 FC2 weight row.  PAFT must keep that threshold frozen or regenerate an
// exact folded-weight quantization contract before this identity is reused.
//
// Per output lane the datapath is a balanced four-weight reduction followed by
// one signed24 accumulator update.  It contains no multiplier and no hidden
// accumulator context: the external owner supplies and receives the Acc24
// vector.  A one-entry elastic result register sustains issue II=1 across
// independent contexts when the consumer accepts the preceding result.
module m169_fc2_k4_unique_bank_accumulator #(
    parameter int LANES = 96,
    parameter int TAG_BITS = 24
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         issue_valid,
    output logic                         issue_ready,
    input  logic [TAG_BITS-1:0]          issue_tag,
    input  logic                         issue_last,
    input  logic [3:0]                   issue_slot_valid,
    input  logic [2:0]                   issue_bank_id [0:3],
    input  logic signed [7:0]            issue_weight [0:3][0:LANES-1],
    input  logic signed [23:0]           issue_accumulator [0:LANES-1],

    output logic                         issue_accept,
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [TAG_BITS-1:0]          result_tag,
    output logic                         result_last,
    output logic [2:0]                   result_source_count,
    output logic [7:0]                   result_bank_mask,
    output logic signed [23:0]           result_accumulator [0:LANES-1],
    output logic                         result_accept,

    output logic [4*LANES-1:0]           accepted_weight_active_mask,
    output logic                         protocol_error,
    output logic                         numeric_overflow,
    output logic                         busy
);
    logic fault_q;
    logic overflow_q;
    logic result_valid_q;
    logic [TAG_BITS-1:0] result_tag_q;
    logic result_last_q;
    logic [2:0] result_source_count_q;
    logic [7:0] result_bank_mask_q;
    logic signed [23:0] result_accumulator_q [0:LANES-1];

    logic prefix_packed;
    logic unique_banks;
    logic legal_issue;
    logic illegal_request;

    always_comb begin : legality
        prefix_packed = (!issue_slot_valid[1] || issue_slot_valid[0])
            && (!issue_slot_valid[2] || issue_slot_valid[1])
            && (!issue_slot_valid[3] || issue_slot_valid[2]);
        unique_banks = 1'b1;
        for (int left = 0; left < 4; left++) begin
            for (int right = left + 1; right < 4; right++) begin
                if (issue_slot_valid[left] && issue_slot_valid[right]
                        && issue_bank_id[left] == issue_bank_id[right])
                    unique_banks = 1'b0;
            end
        end
        legal_issue = (|issue_slot_valid) && prefix_packed && unique_banks;
    end

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
        for (genvar slot = 0; slot < 4; slot++) begin : g_activity_slot
            for (genvar lane = 0; lane < LANES; lane++) begin : g_activity_lane
                assign accepted_weight_active_mask[(slot*LANES)+lane]
                    = issue_accept && issue_slot_valid[slot];
            end
        end
    endgenerate

    always_ff @(posedge clk_core) begin : elastic_accumulator_update
        logic [2:0] source_count;
        logic [7:0] bank_mask;
        logic overflow_any;
        logic signed [8:0] pair_01;
        logic signed [8:0] pair_23;
        logic signed [9:0] weight_sum;
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
                bank_mask = '0;
                for (int slot = 0; slot < 4; slot++) begin
                    if (issue_slot_valid[slot]) begin
                        source_count = source_count + 1'b1;
                        bank_mask[issue_bank_id[slot]] = 1'b1;
                    end
                end
                result_valid_q <= 1'b1;
                result_tag_q <= issue_tag;
                result_last_q <= issue_last;
                result_source_count_q <= source_count;
                result_bank_mask_q <= bank_mask;
                overflow_any = 1'b0;
                for (int lane = 0; lane < LANES; lane++) begin
                    pair_01 = $signed({issue_weight[0][lane][7],
                                      issue_weight[0][lane]})
                        + (issue_slot_valid[1]
                            ? $signed({issue_weight[1][lane][7],
                                      issue_weight[1][lane]})
                            : 9'sd0);
                    pair_23 = (issue_slot_valid[2]
                            ? $signed({issue_weight[2][lane][7],
                                      issue_weight[2][lane]})
                            : 9'sd0)
                        + (issue_slot_valid[3]
                            ? $signed({issue_weight[3][lane][7],
                                      issue_weight[3][lane]})
                            : 9'sd0);
                    weight_sum = $signed({pair_01[8], pair_01})
                        + $signed({pair_23[8], pair_23});
                    extended_sum = $signed({issue_accumulator[lane][23],
                                            issue_accumulator[lane]})
                        + $signed({{15{weight_sum[9]}}, weight_sum});
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
