`timescale 1ns/1ps
`default_nettype none

module m219_fc2_k1_cropped_tagged_slice_service_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SLICE_LANES = 16
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic [TAG_BITS-1:0] header_tag,
    input logic [3:0] header_output_blocks,
    input logic group_valid, input logic group_ready,
    input logic group_accept, input logic [TAG_BITS-1:0] group_tag,
    input logic [2:0] group_output_block,
    input logic [2:0] group_bank_id,
    input logic [CHANNEL_BITS-1:0] group_source_channel,
    input logic frontend_done_valid, input logic frontend_done_ready,
    input logic frontend_done_accept,
    input logic [TAG_BITS-1:0] frontend_done_tag,
    input logic frontend_done_had_event,
    input logic mem_req_valid, input logic mem_req_ready,
    input logic mem_req_accept,
    input logic [EPOCH_BITS-1:0] mem_req_epoch,
    input logic [2:0] mem_req_slot,
    input logic [GENERATION_BITS-1:0] mem_req_generation,
    input logic [TAG_BITS-1:0] mem_req_tag,
    input logic [2:0] mem_req_output_block,
    input logic [2:0] mem_req_slice,
    input logic [2:0] mem_req_bank_id,
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel,
    input logic mem_rsp_valid, input logic mem_rsp_ready,
    input logic mem_rsp_accept,
    input logic [EPOCH_BITS-1:0] mem_rsp_epoch,
    input logic [2:0] mem_rsp_slot,
    input logic [GENERATION_BITS-1:0] mem_rsp_generation,
    input logic [TAG_BITS-1:0] mem_rsp_tag,
    input logic [2:0] mem_rsp_bank_id,
    input logic result_valid, input logic result_ready,
    input logic result_accept, input logic [TAG_BITS-1:0] result_tag,
    input logic [2:0] result_output_block,
    input logic [2:0] result_slice,
    input logic signed [23:0] result_accumulator [0:SLICE_LANES-1],
    input logic result_last,
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept,
    input logic [TAG_BITS-1:0] token_done_tag,
    input logic token_done_had_event,
    input logic mem_flush_valid, input logic mem_flush_ready,
    input logic [EPOCH_BITS-1:0] mem_flush_epoch,
    input logic mem_flush_ack_valid, input logic mem_flush_ack_ready,
    input logic [EPOCH_BITS-1:0] mem_flush_ack_epoch,
    input logic protocol_error, input logic numeric_overflow,
    input logic stale_response_seen,
    input logic [2:0] debug_fifo_count,
    input logic [3:0] debug_outstanding_count,
    input logic [31:0] debug_group_accept_count,
    input logic [31:0] debug_request_accept_count,
    input logic [31:0] debug_response_accept_count,
    input logic [31:0] debug_context_write_count,
    input logic [31:0] debug_result_accept_count,
    input logic [31:0] debug_active_bank_read_count
);
    logic [SLICE_LANES*24-1:0] result_flat;
    logic [3:0] accepted_output_blocks_q;

    always_comb begin
        for (int lane = 0; lane < SLICE_LANES; lane++)
            result_flat[lane*24 +: 24] = result_accumulator[lane];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core)
            accepted_output_blocks_q <= 0;
        else if (header_accept)
            accepted_output_blocks_q <= header_output_blocks;
    end

    ap_header_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) header_accept == (header_valid && header_ready));
    ap_group_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) group_accept == (group_valid && group_ready));
    ap_frontend_done_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) frontend_done_accept
        == (frontend_done_valid && frontend_done_ready));
    ap_request_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) mem_req_accept
        == (mem_req_valid && mem_req_ready));
    ap_response_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) mem_rsp_accept
        == (mem_rsp_valid && mem_rsp_ready));
    ap_result_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) result_accept
        == (result_valid && result_ready));
    ap_done_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) token_done_accept
        == (token_done_valid && token_done_ready));

    ap_group_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) group_valid && !group_ready |=>
        $stable({group_tag,group_output_block,group_bank_id,
                 group_source_channel}));
    ap_request_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) mem_req_valid && !mem_req_ready |=>
        $stable({mem_req_epoch,mem_req_slot,mem_req_generation,mem_req_tag,
                 mem_req_output_block,mem_req_slice,mem_req_bank_id,
                 mem_req_source_channel}));
    ap_result_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) result_valid && !result_ready |=>
        $stable({result_tag,result_output_block,result_slice,result_flat,
                 result_last}));
    ap_done_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) token_done_valid && !token_done_ready |=>
        $stable({token_done_tag,token_done_had_event}));
    ap_flush_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) mem_flush_valid && !mem_flush_ready |=>
        $stable(mem_flush_epoch));

    ap_request_identity_shape: assert property (@(posedge clk_core)
        disable iff (rst_core) mem_req_valid |->
        mem_req_source_channel[2:0] == mem_req_bank_id
        && mem_req_slice < 6 && mem_req_output_block < 8);
    ap_fifo_bound: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_fifo_count <= 4);
    ap_outstanding_bound: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_outstanding_count <= 8);
    ap_response_le_request: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_response_accept_count
        <= debug_request_accept_count);
    ap_context_le_response: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_context_write_count
        <= debug_response_accept_count);
    ap_request_le_six_groups: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_request_accept_count
        <= debug_group_accept_count * 6);
    ap_fault_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) $past(protocol_error) |-> protocol_error);
    ap_overflow_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) $past(numeric_overflow) |-> numeric_overflow);
    ap_stale_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) $past(stale_response_seen)
        |-> stale_response_seen);
    ap_done_empty: assert property (@(posedge clk_core)
        disable iff (rst_core) token_done_valid |->
        debug_fifo_count == 0 && debug_outstanding_count == 0
        && debug_request_accept_count == debug_response_accept_count
        && debug_response_accept_count == debug_context_write_count
        && debug_request_accept_count == debug_group_accept_count * 6
        && debug_result_accept_count == accepted_output_blocks_q * 6);
    ap_done_tag: assert property (@(posedge clk_core)
        disable iff (rst_core) token_done_valid |->
        token_done_tag == result_tag);

    cp_k1_request: cover property (@(posedge clk_core)
        disable iff (rst_core) mem_req_accept);
    cp_same_cycle_replace: cover property (@(posedge clk_core)
        disable iff (rst_core) mem_req_accept && mem_rsp_accept);
    cp_result_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) result_valid && !result_ready);
    cp_flush: cover property (@(posedge clk_core)
        disable iff (rst_core) mem_flush_valid && mem_flush_ready ##[1:20]
        mem_flush_ack_valid && mem_flush_ack_ready);
    cp_stale_seen: cover property (@(posedge clk_core)
        disable iff (rst_core) stale_response_seen);
    cp_protocol_fault_rise: cover property (@(posedge clk_core)
        disable iff (rst_core) !$past(protocol_error) && protocol_error);
    cp_done: cover property (@(posedge clk_core)
        disable iff (rst_core) token_done_accept);
endmodule

`default_nettype wire

