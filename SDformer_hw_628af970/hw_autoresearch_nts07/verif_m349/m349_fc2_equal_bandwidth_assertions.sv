`timescale 1ns/1ps
`default_nettype none

module m349_fc2_k1x8_raw4_acc24_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int SLICE_LANES = 16
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic integration_header_legal,
    input logic [3:0] header_output_blocks,
    input logic adapter_fault_q,
    input logic fe_header_valid, input logic fe_header_ready,
    input logic fe_header_accept,
    input logic [7:0] lane_header_valid, lane_header_ready,
    input logic [7:0] lane_header_accept,
    input logic raw_valid, input logic raw_ready, input logic raw_accept,
    input logic fe_group_valid, input logic fe_group_ready,
    input logic fe_group_accept,
    input logic [3:0] fe_group_source_count,
    input logic [7:0] fe_group_bank_valid,
    input logic [CHANNEL_BITS-1:0] fe_group_source_channel [0:7],
    input logic [7:0] lane_group_valid, lane_group_ready,
    input logic [7:0] lane_group_accept,
    input logic fe_done_valid, input logic fe_done_ready,
    input logic fe_done_accept,
    input logic [7:0] lane_frontend_done_valid,
    input logic [7:0] lane_frontend_done_ready,
    input logic [7:0] lane_frontend_done_accept,
    input logic [7:0] mem_req_valid, input logic [7:0] mem_req_ready,
    input logic [7:0] mem_req_accept,
    input logic [2:0] mem_req_slice [0:7],
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    input logic [7:0] mem_rsp_valid, input logic [7:0] mem_rsp_ready,
    input logic [7:0] mem_rsp_accept,
    input logic result_valid, input logic result_ready,
    input logic result_accept,
    input logic [TAG_BITS-1:0] result_tag,
    input logic [2:0] result_output_block, input logic [2:0] result_slice,
    input logic signed [23:0] result_accumulator [0:SLICE_LANES-1],
    input logic result_last,
    input logic [7:0] lane_result_valid, lane_result_ready,
    input logic [7:0] lane_result_accept,
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept,
    input logic [TAG_BITS-1:0] token_done_tag,
    input logic token_done_had_event,
    input logic [7:0] lane_done_valid, lane_done_ready, lane_done_accept,
    input logic protocol_error, input logic numeric_overflow,
    input logic [5:0] debug_fifo_count,
    input logic [6:0] debug_outstanding_count,
    input logic [31:0] debug_request_accept_count,
    input logic [31:0] debug_response_accept_count,
    input logic [31:0] debug_context_write_count
);
`ifdef SVA_RUNTIME_ENABLED
    logic [SLICE_LANES*24-1:0] result_flat;
    always_comb begin
        for (int lane = 0; lane < SLICE_LANES; lane++)
            result_flat[lane*24 +: 24] = result_accumulator[lane];
    end

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_accept_definition:
        assert property (header_accept == (header_valid && header_ready));
    ap_header_atomic_frontend:
        assert property (header_accept == fe_header_accept);
    ap_header_atomic_lanes:
        assert property (header_accept |-> lane_header_accept == 8'hff);
    ap_no_partial_lane_header:
        assert property (!header_accept |-> lane_header_accept == 0);
    ap_header_valid_gating:
        assert property (fe_header_valid == (header_valid
            && integration_header_legal && (&lane_header_ready)
            && !adapter_fault_q));
    ap_raw_accept_definition:
        assert property (raw_accept == (raw_valid && raw_ready));
    ap_group_mask_count:
        assert property (fe_group_valid |-> fe_group_source_count >= 1
            && fe_group_source_count <= 8
            && fe_group_source_count == $countones(fe_group_bank_valid));
    ap_group_atomic_active_lanes:
        assert property (fe_group_accept
            |-> lane_group_accept == fe_group_bank_valid);
    ap_no_orphan_lane_group:
        assert property (!fe_group_accept |-> lane_group_accept == 0);
    ap_group_source_mapping:
        assert property (fe_group_valid
            |-> ((!fe_group_bank_valid[0]
                    || fe_group_source_channel[0][2:0] == 0)
                && (!fe_group_bank_valid[1]
                    || fe_group_source_channel[1][2:0] == 1)
                && (!fe_group_bank_valid[2]
                    || fe_group_source_channel[2][2:0] == 2)
                && (!fe_group_bank_valid[3]
                    || fe_group_source_channel[3][2:0] == 3)
                && (!fe_group_bank_valid[4]
                    || fe_group_source_channel[4][2:0] == 4)
                && (!fe_group_bank_valid[5]
                    || fe_group_source_channel[5][2:0] == 5)
                && (!fe_group_bank_valid[6]
                    || fe_group_source_channel[6][2:0] == 6)
                && (!fe_group_bank_valid[7]
                    || fe_group_source_channel[7][2:0] == 7)));
    ap_done_bridge_atomic:
        assert property (fe_done_accept
            |-> lane_frontend_done_accept == 8'hff);
    ap_no_orphan_lane_frontend_done:
        assert property (!fe_done_accept
            |-> lane_frontend_done_accept == 0);
    ap_request_accept_definition:
        assert property (mem_req_accept == (mem_req_valid & mem_req_ready));
    for (genvar bank = 0; bank < 8; bank++) begin : g_bank_sva
        ap_request_bank_mapping:
            assert property (mem_req_valid[bank]
                |-> mem_req_source_channel[bank][2:0] == bank[2:0]
                    && mem_req_slice[bank] < 6);
        ap_response_accept_definition:
            assert property (mem_rsp_accept[bank]
                == (mem_rsp_valid[bank] && mem_rsp_ready[bank]));
    end
    ap_result_accept_definition:
        assert property (result_accept == (result_valid && result_ready));
    ap_result_join_atomic:
        assert property (result_accept |-> lane_result_accept == 8'hff);
    ap_no_partial_lane_result:
        assert property (!result_accept |-> lane_result_accept == 0);
    ap_result_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid && $stable({result_tag,result_output_block,
                result_slice,result_flat,result_last}));
    ap_done_accept_definition:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_done_join_atomic:
        assert property (token_done_accept |-> lane_done_accept == 8'hff);
    ap_no_partial_lane_done:
        assert property (!token_done_accept |-> lane_done_accept == 0);
    ap_done_after_results:
        assert property (token_done_valid |-> !result_valid);
    ap_fifo_aggregate_bound:
        assert property (debug_fifo_count <= 32);
    ap_outstanding_aggregate_bound:
        assert property (debug_outstanding_count <= 64);
    ap_response_le_request:
        assert property (debug_response_accept_count
            <= debug_request_accept_count);
    ap_context_le_response:
        assert property (debug_context_write_count
            <= debug_response_accept_count);
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);
    ap_overflow_sticky:
        assert property ($past(numeric_overflow) |-> numeric_overflow);

    cp_b1: cover property (header_accept && header_output_blocks == 1);
    cp_b2: cover property (header_accept && header_output_blocks == 2);
    cp_b4: cover property (header_accept && header_output_blocks == 4);
    cp_b8: cover property (header_accept && header_output_blocks == 8);
    cp_all_eight_lane_group: cover property (fe_group_accept
        && fe_group_bank_valid == 8'hff);
    cp_eight_requests_same_cycle: cover property ($countones(mem_req_accept) == 8);
    cp_request_backpressure: cover property (|(mem_req_valid & ~mem_req_ready));
    cp_result_stall: cover property (result_valid && !result_ready);
    cp_done: cover property (token_done_accept);
    cp_protocol_fault: cover property (protocol_error);
`endif
endmodule

// These two binds close the M345 finding that the frozen service SVA modules
// were parsed but never elaborated.  They apply to the M342 candidate M218 and
// to every M349 K1x8 M219 lane.
bind m218_fc2_tagged_slice_service_island
    m218_fc2_tagged_slice_service_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SLICE_LANES(SLICE_LANES)
    ) m349_bound_service_sva (.*);

bind m219_fc2_k1_cropped_tagged_slice_service_island
    m219_fc2_k1_cropped_tagged_slice_service_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SLICE_LANES(SLICE_LANES)
    ) m349_bound_service_sva (.*);

bind m349_fc2_k1x8_raw4_acc24
    m349_fc2_k1x8_raw4_acc24_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .SLICE_LANES(SLICE_LANES)
    ) m349_top_sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_accept(header_accept),
        .integration_header_legal(integration_header_legal),
        .header_output_blocks(header_output_blocks),
        .adapter_fault_q(adapter_fault_q),
        .fe_header_valid(fe_header_valid),
        .fe_header_ready(fe_header_ready),
        .fe_header_accept(fe_header_accept),
        .lane_header_valid(lane_header_valid),
        .lane_header_ready(lane_header_ready),
        .lane_header_accept(lane_header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_accept(raw_accept),
        .fe_group_valid(fe_group_valid), .fe_group_ready(fe_group_ready),
        .fe_group_accept(fe_group_accept),
        .fe_group_source_count(fe_group_source_count),
        .fe_group_bank_valid(fe_group_bank_valid),
        .fe_group_source_channel(fe_group_source_channel),
        .lane_group_valid(lane_group_valid),
        .lane_group_ready(lane_group_ready),
        .lane_group_accept(lane_group_accept),
        .fe_done_valid(fe_done_valid), .fe_done_ready(fe_done_ready),
        .fe_done_accept(fe_done_accept),
        .lane_frontend_done_valid(lane_frontend_done_valid),
        .lane_frontend_done_ready(lane_frontend_done_ready),
        .lane_frontend_done_accept(lane_frontend_done_accept),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_accept(mem_req_accept), .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_accept(mem_rsp_accept),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_accept(result_accept), .result_tag(result_tag),
        .result_output_block(result_output_block),
        .result_slice(result_slice),
        .result_accumulator(result_accumulator),
        .result_last(result_last),
        .lane_result_valid(lane_result_valid),
        .lane_result_ready(lane_result_ready),
        .lane_result_accept(lane_result_accept),
        .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),
        .token_done_accept(token_done_accept),
        .token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .lane_done_valid(lane_done_valid),
        .lane_done_ready(lane_done_ready),
        .lane_done_accept(lane_done_accept),
        .protocol_error(protocol_error),
        .numeric_overflow(numeric_overflow),
        .debug_fifo_count(debug_fifo_count),
        .debug_outstanding_count(debug_outstanding_count),
        .debug_request_accept_count(debug_request_accept_count),
        .debug_response_accept_count(debug_response_accept_count),
        .debug_context_write_count(debug_context_write_count));

`default_nettype wire
