`timescale 1ns/1ps
`default_nettype none

// Shared H67/Local5 control primitive. It preserves every input transaction and
// classifies the delta mask as zero, bounded sparse, or exact dense fallback.
module delta_bounded_classifier #(
    parameter int TAG_W = 16,
    parameter int PAYLOAD_W = 128
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [TAG_W-1:0]             in_tag,
    input  logic [31:0]                  in_delta_mask,
    input  logic [PAYLOAD_W-1:0]         in_payload,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [TAG_W-1:0]             out_tag,
    output logic [1:0]                   out_kind,
    output logic [31:0]                  out_delta_mask,
    output logic [PAYLOAD_W-1:0]         out_payload,
    output logic [5:0]                   out_count,
    output logic [3:0]                   out_lane_valid,
    output logic [19:0]                  out_lane_ids
);

    localparam int LANES = 32'd32;
    localparam int WAYS = 32'd4;
    localparam int LANE_ID_W = 32'd5;
    localparam int COUNT_W = 32'd6;
    localparam logic [1:0] KIND_ZERO   = 2'd0;
    localparam logic [1:0] KIND_SPARSE = 2'd1;
    localparam logic [1:0] KIND_DENSE  = 2'd2;

    logic [COUNT_W-1:0] count_comb;
    logic [1:0] kind_comb;
    logic [WAYS-1:0] lane_valid_comb;
    logic [(WAYS*LANE_ID_W)-1:0] lane_ids_comb;
    logic [LANES-1:0] remaining_mask;
    logic [WAYS-1:0] way_found;

    assign in_ready = !out_valid || out_ready;

    always_comb begin
        count_comb = '0;
        for (int lane = 32'd0; lane < 32; lane = lane + 32'd1) begin
            count_comb = count_comb + COUNT_W'(in_delta_mask[lane]);
        end

        remaining_mask = in_delta_mask;
        way_found = '0;
        lane_valid_comb = '0;
        lane_ids_comb = '0;
        for (int way = 32'd0; way < 4; way = way + 32'd1) begin
            for (int lane = 32'd0; lane < 32; lane = lane + 32'd1) begin
                if (!way_found[way] && remaining_mask[lane]) begin
                    way_found[way] = 1'b1;
                    lane_valid_comb[way] = 1'b1;
                    lane_ids_comb[(way*LANE_ID_W) +: LANE_ID_W] =
                        LANE_ID_W'(lane);
                    remaining_mask[lane] = 1'b0;
                end
            end
        end

        if (count_comb == '0) begin
            kind_comb = KIND_ZERO;
            lane_valid_comb = '0;
            lane_ids_comb = '0;
        end else if (32'(count_comb) <= WAYS) begin
            kind_comb = KIND_SPARSE;
        end else begin
            kind_comb = KIND_DENSE;
            lane_valid_comb = '0;
            lane_ids_comb = '0;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            out_valid <= 1'b0;
            out_tag <= '0;
            out_kind <= KIND_ZERO;
            out_delta_mask <= '0;
            out_payload <= '0;
            out_count <= '0;
            out_lane_valid <= '0;
            out_lane_ids <= '0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid) begin
                out_tag <= in_tag;
                out_kind <= kind_comb;
                out_delta_mask <= in_delta_mask;
                out_payload <= in_payload;
                out_count <= count_comb;
                out_lane_valid <= lane_valid_comb;
                out_lane_ids <= lane_ids_comb;
            end
        end
    end

endmodule

`default_nettype wire
