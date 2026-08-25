`timescale 1ns/1ps
`default_nettype none
module m231_atlif32_to_fc2_raw4_pingpong_bridge_assertions #(
    parameter int INPUT_WIDTH=384,
    parameter int TAG_BITS=24
) (
    input logic clk_core,input logic rst_core,
    input logic pair_header_valid,input logic pair_header_ready,
    input logic pair_header_accept,input logic event_valid,
    input logic event_ready,input logic event_accept,
    input logic header_valid,input logic header_ready,
    input logic[TAG_BITS-1:0]header_tag,input logic[5:0]header_raw_beat_count,
    input logic[3:0]header_window_depth,input logic[3:0]header_output_blocks,
    input logic header_accept,input logic raw_valid,input logic raw_ready,
    input logic[3:0]raw_lane_valid,input logic[4:0]raw_beat_index[0:3],
    input logic[95:0]raw_bitmap[0:3],input logic raw_last,
    input logic raw_accept,input logic protocol_error,input logic busy,
    input logic[1:0]debug_full_slots,input logic[31:0]debug_pair_count,
    input logic[31:0]debug_token_count,
    input logic[31:0]debug_raw_packet_count
);
    localparam int RAW_BEATS=INPUT_WIDTH/96;
    localparam int OUTPUT_BLOCKS=INPUT_WIDTH/384;
    logic[4*(5+96)-1:0]raw_flat;
    always_comb for(int lane=0;lane<4;lane++)begin
        raw_flat[lane*101+:5]=raw_beat_index[lane];
        raw_flat[lane*101+5+:96]=raw_bitmap[lane];
    end
    ap_pair_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        pair_header_accept==(pair_header_valid&&pair_header_ready));
    ap_event_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        event_accept==(event_valid&&event_ready));
    ap_header_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept==(header_valid&&header_ready));
    ap_raw_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_accept==(raw_valid&&raw_ready));
    ap_header_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        header_valid|->header_raw_beat_count==RAW_BEATS
        &&header_output_blocks==OUTPUT_BLOCKS);
    ap_header_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        header_valid&&!header_ready|=>protocol_error||(header_valid
        &&$stable({header_tag,header_raw_beat_count,header_window_depth,
            header_output_blocks})));
    ap_raw_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_valid&&!raw_ready|=>protocol_error||(raw_valid
        &&$stable({raw_lane_valid,raw_flat,raw_last})));
    ap_raw_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_valid|->raw_lane_valid==4'hf);
    ap_token_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_token_count<=debug_pair_count*2);
    ap_packet_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_raw_packet_count<=debug_pair_count*2*(RAW_BEATS/4));
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    ap_fault_cycle_accept_atomic:assert property(@(posedge clk_core)
        disable iff(rst_core)protocol_error|->!(pair_header_accept
        ||event_accept||header_accept||raw_accept));
    ap_fault_cycle_interface_quarantine:assert property(@(posedge clk_core)
        disable iff(rst_core)protocol_error|->!pair_header_ready&&!event_ready
        &&!header_valid&&!raw_valid);
    cp_pingpong_full:cover property(@(posedge clk_core)debug_full_slots==2'b11);
    cp_header_stall:cover property(@(posedge clk_core)header_valid&&!header_ready);
    cp_raw_stall:cover property(@(posedge clk_core)raw_valid&&!raw_ready);
    cp_fault:cover property(@(posedge clk_core)protocol_error);
    cp_fault_while_raw_would_accept:cover property(@(posedge clk_core)
        protocol_error&&raw_ready);
    cp_complete_pair:cover property(@(posedge clk_core)
        debug_pair_count>=2&&debug_token_count>=4);
endmodule
`default_nettype wire
