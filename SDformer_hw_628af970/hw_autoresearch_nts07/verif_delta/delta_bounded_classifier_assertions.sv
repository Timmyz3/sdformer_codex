`timescale 1ns/1ps
`default_nettype none

module delta_bounded_classifier_assertions #(
    parameter int TAG_W = 16,
    parameter int PAYLOAD_W = 128
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [TAG_W-1:0] out_tag,
    input logic [1:0] out_kind,
    input logic [31:0] out_delta_mask,
    input logic [PAYLOAD_W-1:0] out_payload,
    input logic [5:0] out_count,
    input logic [3:0] out_lane_valid,
    input logic [19:0] out_lane_ids
);

    property p_output_stable_under_stall;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=> out_valid &&
            $stable({out_tag, out_kind, out_delta_mask, out_payload,
                     out_count, out_lane_valid, out_lane_ids});
    endproperty

    property p_count_matches_mask;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> (out_count == $countones(out_delta_mask));
    endproperty

    property p_zero_contract;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd0 |->
            out_count == 0 && out_lane_valid == '0;
    endproperty

    property p_sparse_contract;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 |->
            out_count > 0 && 32'(out_count) <= 4 &&
            $countones(out_lane_valid) == out_count;
    endproperty

    property p_dense_contract;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd2 |->
            32'(out_count) > 4 && out_lane_valid == '0;
    endproperty

    property p_kind_legal;
        @(posedge clk_core) disable iff (rst_core)
        out_valid |-> out_kind inside {2'd0, 2'd1, 2'd2};
    endproperty

    property p_sparse_lane_valid_is_prefix;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 |->
            out_lane_valid inside {
                4'b0001, 4'b0011, 4'b0111, 4'b1111
            };
    endproperty

    property p_sparse_lane_ids_ordered;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 |->
            (!out_lane_valid[1] ||
             out_lane_ids[9:5] > out_lane_ids[4:0]) &&
            (!out_lane_valid[2] ||
             out_lane_ids[14:10] > out_lane_ids[9:5]) &&
            (!out_lane_valid[3] ||
             out_lane_ids[19:15] > out_lane_ids[14:10]);
    endproperty

    property p_lane0_belongs_to_mask;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 && out_lane_valid[0] |->
            out_delta_mask[out_lane_ids[4:0]];
    endproperty

    property p_lane1_belongs_to_mask;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 && out_lane_valid[1] |->
            out_delta_mask[out_lane_ids[9:5]];
    endproperty

    property p_lane2_belongs_to_mask;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 && out_lane_valid[2] |->
            out_delta_mask[out_lane_ids[14:10]];
    endproperty

    property p_lane3_belongs_to_mask;
        @(posedge clk_core) disable iff (rst_core)
        out_valid && out_kind == 2'd1 && out_lane_valid[3] |->
            out_delta_mask[out_lane_ids[19:15]];
    endproperty

    assert property (p_output_stable_under_stall);
    assert property (p_count_matches_mask);
    assert property (p_zero_contract);
    assert property (p_sparse_contract);
    assert property (p_dense_contract);
    assert property (p_kind_legal);
    assert property (p_sparse_lane_valid_is_prefix);
    assert property (p_sparse_lane_ids_ordered);
    assert property (p_lane0_belongs_to_mask);
    assert property (p_lane1_belongs_to_mask);
    assert property (p_lane2_belongs_to_mask);
    assert property (p_lane3_belongs_to_mask);

endmodule

`default_nettype wire
