`timescale 1ns/1ps
`default_nettype none

module h87_t5_open_run_file_assertions #(
    parameter int POSITIONS = 225,
    parameter int SCORE_W = 8,
    parameter int POSITION_W = (POSITIONS <= 1) ? 1 : $clog2(POSITIONS)
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic packet_valid,
    input logic packet_ready,
    input logic [1:0] packet_desc_count,
    input logic [4:0] packet_desc0_temporal_mask,
    input logic [4:0] packet_desc0_active_mask,
    input logic [4:0] packet_desc1_temporal_mask,
    input logic [4:0] packet_desc1_active_mask,
    input logic packet_desc0_row_last,
    input logic packet_desc1_row_last,
    input logic row_done,
    input logic protocol_error
);
    function automatic logic mask_contiguous(input logic [4:0] mask);
        logic seen_one;
        logic seen_gap;
        integer i;
        begin
            seen_one = 1'b0;
            seen_gap = 1'b0;
            mask_contiguous = |mask;
            for (i = 0; i < 5; i = i + 1) begin
                if (mask[i]) begin
                    if (seen_gap)
                        mask_contiguous = 1'b0;
                    seen_one = 1'b1;
                end else if (seen_one) begin
                    seen_gap = 1'b1;
                end
            end
        end
    endfunction

    property p_packet_count;
        @(posedge clk_core) disable iff (rst_core)
        packet_valid |-> packet_desc_count inside {2'd1, 2'd2};
    endproperty

    property p_desc0_membership;
        @(posedge clk_core) disable iff (rst_core)
        packet_valid |-> mask_contiguous(packet_desc0_temporal_mask)
                     && (packet_desc0_active_mask
                         & ~packet_desc0_temporal_mask) == 5'b0;
    endproperty

    property p_desc1_membership;
        @(posedge clk_core) disable iff (rst_core)
        packet_valid && packet_desc_count == 2'd2
        |-> mask_contiguous(packet_desc1_temporal_mask)
         && (packet_desc1_active_mask & ~packet_desc1_temporal_mask) == 5'b0;
    endproperty

    property p_output_stable;
        @(posedge clk_core) disable iff (rst_core)
        packet_valid && !packet_ready |=> packet_valid
            && $stable({packet_desc_count,
                        packet_desc0_temporal_mask,
                        packet_desc0_active_mask,
                        packet_desc1_temporal_mask,
                        packet_desc1_active_mask,
                        packet_desc0_row_last,
                        packet_desc1_row_last});
    endproperty

    property p_row_done_is_retire;
        @(posedge clk_core) disable iff (rst_core)
        row_done |-> packet_valid && packet_ready
                  && (packet_desc0_row_last || packet_desc1_row_last);
    endproperty

    assert property (p_packet_count);
    assert property (p_desc0_membership);
    assert property (p_desc1_membership);
    assert property (p_output_stable);
    assert property (p_row_done_is_retire);

    cover property (@(posedge clk_core) disable iff (rst_core)
                    in_valid && !in_ready);
    cover property (@(posedge clk_core) disable iff (rst_core)
                    packet_valid && packet_desc_count == 2'd2);
    cover property (@(posedge clk_core) disable iff (rst_core) row_done);
endmodule

bind h87_t5_open_run_file h87_t5_open_run_file_assertions #(
    .POSITIONS(POSITIONS),
    .SCORE_W(SCORE_W),
    .POSITION_W(POSITION_W)
) u_h87_t5_open_run_file_assertions (.*);

`default_nettype wire
