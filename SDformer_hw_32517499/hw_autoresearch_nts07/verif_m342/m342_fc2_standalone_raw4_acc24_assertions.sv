`timescale 1ns/1ps
`default_nettype none

module m342_fc2_standalone_raw4_acc24_assertions #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int EPOCH_BITS = 16,
    parameter int GENERATION_BITS = 32,
    parameter int SOURCE_CAP = 8,
    parameter int SLICE_LANES = 16
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, input logic header_ready,
    input logic header_accept, input logic integration_header_legal,
    input logic adapter_fault_q,
    input logic [3:0] header_output_blocks,
    input logic fe_header_valid, input logic fe_header_ready,
    input logic fe_header_accept, input logic svc_header_valid,
    input logic svc_header_ready, input logic svc_header_accept,
    input logic raw_valid, input logic raw_ready, input logic raw_accept,
    input logic fe_group_valid, input logic fe_group_ready,
    input logic fe_group_accept, input logic svc_group_valid,
    input logic svc_group_ready, input logic svc_group_accept,
    input logic [3:0] fe_group_source_count,
    input logic [7:0] fe_group_bank_valid,
    input logic [CHANNEL_BITS-1:0] fe_group_source_channel [0:7],
    input logic fe_done_valid, input logic fe_done_ready,
    input logic fe_done_accept, input logic svc_frontend_done_valid,
    input logic svc_frontend_done_ready,
    input logic svc_frontend_done_accept,
    input logic svc_soft_flush,
    input logic mem_req_valid, input logic mem_req_ready,
    input logic mem_req_accept,
    input logic [2:0] mem_req_output_block,
    input logic [2:0] mem_req_slice,
    input logic [3:0] mem_req_source_count,
    input logic [7:0] mem_req_bank_valid,
    input logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7],
    input logic mem_rsp_valid, input logic mem_rsp_ready,
    input logic mem_rsp_accept,
    input logic result_valid, input logic result_ready,
    input logic result_accept, input logic [TAG_BITS-1:0] result_tag,
    input logic [2:0] result_output_block, input logic [2:0] result_slice,
    input logic signed [23:0] result_accumulator [0:SLICE_LANES-1],
    input logic result_last,
    input logic token_done_valid, input logic token_done_ready,
    input logic token_done_accept, input logic protocol_error
);
`ifdef SVA_RUNTIME_ENABLED
    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 8; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction

    default clocking cb @(posedge clk_core); endclocking
    default disable iff (rst_core);

    ap_header_accept_definition:
        assert property (header_accept == (header_valid && header_ready));
    ap_atomic_child_header_accept:
        assert property (header_accept == fe_header_accept
            && header_accept == svc_header_accept);
    ap_atomic_header_valid_gating:
        assert property (fe_header_valid
            == (header_valid && integration_header_legal
                && svc_header_ready && !adapter_fault_q));
    ap_atomic_service_valid_gating:
        assert property (svc_header_valid
            == (header_valid && integration_header_legal
                && fe_header_ready && !adapter_fault_q));
    ap_raw_accept_definition:
        assert property (raw_accept == (raw_valid && raw_ready));
    ap_group_accept_lockstep:
        assert property (fe_group_accept == svc_group_accept);
    ap_group_payload_mask_count:
        assert property (fe_group_valid
            |-> fe_group_source_count == popcount8(fe_group_bank_valid));
    ap_group_source_bank_mapping:
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
    ap_k1_adapter_onehot:
        assert property (SOURCE_CAP == 1 && fe_group_valid
            |-> $onehot(fe_group_bank_valid)
                && fe_group_source_count == 1);
    ap_frontend_done_bridge_valid:
        assert property (svc_frontend_done_valid == fe_done_valid);
    ap_frontend_done_bridge_ready:
        assert property (fe_done_ready == svc_frontend_done_ready);
    ap_frontend_done_accept_lockstep:
        assert property (fe_done_accept == svc_frontend_done_accept);
    ap_soft_flush_tied_low: assert property (!svc_soft_flush);
    ap_mem_req_accept_definition:
        assert property (mem_req_accept == (mem_req_valid && mem_req_ready));
    ap_mem_req_legal_slice:
        assert property (mem_req_valid |-> mem_req_slice < 6);
    ap_mem_req_mask_count:
        assert property (mem_req_valid
            |-> mem_req_source_count == popcount8(mem_req_bank_valid));
    ap_mem_req_source_mapping:
        assert property (mem_req_valid
            |-> ((!mem_req_bank_valid[0]
                    || mem_req_source_channel[0][2:0] == 0)
                && (!mem_req_bank_valid[1]
                    || mem_req_source_channel[1][2:0] == 1)
                && (!mem_req_bank_valid[2]
                    || mem_req_source_channel[2][2:0] == 2)
                && (!mem_req_bank_valid[3]
                    || mem_req_source_channel[3][2:0] == 3)
                && (!mem_req_bank_valid[4]
                    || mem_req_source_channel[4][2:0] == 4)
                && (!mem_req_bank_valid[5]
                    || mem_req_source_channel[5][2:0] == 5)
                && (!mem_req_bank_valid[6]
                    || mem_req_source_channel[6][2:0] == 6)
                && (!mem_req_bank_valid[7]
                    || mem_req_source_channel[7][2:0] == 7)));
    ap_mem_req_stable_under_stall:
        assert property (mem_req_valid && !mem_req_ready
            |=> mem_req_valid && $stable(mem_req_output_block)
                && $stable(mem_req_slice) && $stable(mem_req_source_count)
                && $stable(mem_req_bank_valid)
                && $stable(mem_req_source_channel));
    ap_mem_rsp_accept_definition:
        assert property (mem_rsp_accept == (mem_rsp_valid && mem_rsp_ready));
    ap_result_accept_definition:
        assert property (result_accept == (result_valid && result_ready));
    ap_result_stable_under_stall:
        assert property (result_valid && !result_ready
            |=> result_valid && $stable(result_tag)
                && $stable(result_output_block) && $stable(result_slice)
                && $stable(result_accumulator) && $stable(result_last));
    ap_done_accept_definition:
        assert property (token_done_accept
            == (token_done_valid && token_done_ready));
    ap_done_follows_results:
        assert property (token_done_valid |-> !result_valid);
    ap_fault_sticky:
        assert property ($past(protocol_error) |-> protocol_error);

    cp_b1: cover property (header_accept && header_output_blocks == 1);
    cp_b2: cover property (header_accept && header_output_blocks == 2);
    cp_b4: cover property (header_accept && header_output_blocks == 4);
    cp_b8: cover property (header_accept && header_output_blocks == 8);
    cp_group_stall: cover property (fe_group_valid && !fe_group_ready);
    cp_memory_request_stall: cover property (mem_req_valid && !mem_req_ready);
    cp_full_eight_source_request: cover property (mem_req_accept
        && mem_req_source_count == 8);
    cp_single_source_request: cover property (mem_req_accept
        && mem_req_source_count == 1);
    cp_result_stall: cover property (result_valid && !result_ready);
    cp_final_done: cover property (token_done_accept);
    cp_protocol_attack: cover property (protocol_error);
`endif
endmodule

bind m342_fc2_standalone_raw4_acc24
    m342_fc2_standalone_raw4_acc24_assertions #(
        .TAG_BITS(TAG_BITS), .CHANNEL_BITS(CHANNEL_BITS),
        .EPOCH_BITS(EPOCH_BITS), .GENERATION_BITS(GENERATION_BITS),
        .SOURCE_CAP(SOURCE_CAP), .SLICE_LANES(SLICE_LANES)
    ) sva (
        .clk_core(clk_core), .rst_core(rst_core),
        .header_valid(header_valid), .header_ready(header_ready),
        .header_accept(header_accept),
        .integration_header_legal(integration_header_legal),
        .adapter_fault_q(adapter_fault_q),
        .header_output_blocks(header_output_blocks),
        .fe_header_valid(fe_header_valid),
        .fe_header_ready(fe_header_ready),
        .fe_header_accept(fe_header_accept),
        .svc_header_valid(svc_header_valid),
        .svc_header_ready(svc_header_ready),
        .svc_header_accept(svc_header_accept),
        .raw_valid(raw_valid), .raw_ready(raw_ready),
        .raw_accept(raw_accept), .fe_group_valid(fe_group_valid),
        .fe_group_ready(fe_group_ready),
        .fe_group_accept(fe_group_accept),
        .svc_group_valid(svc_group_valid),
        .svc_group_ready(svc_group_ready),
        .svc_group_accept(svc_group_accept),
        .fe_group_source_count(fe_group_source_count),
        .fe_group_bank_valid(fe_group_bank_valid),
        .fe_group_source_channel(fe_group_source_channel),
        .fe_done_valid(fe_done_valid), .fe_done_ready(fe_done_ready),
        .fe_done_accept(fe_done_accept),
        .svc_frontend_done_valid(svc_frontend_done_valid),
        .svc_frontend_done_ready(svc_frontend_done_ready),
        .svc_frontend_done_accept(svc_frontend_done_accept),
        .svc_soft_flush(svc_soft_flush),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_accept(mem_req_accept),
        .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_count(mem_req_source_count),
        .mem_req_bank_valid(mem_req_bank_valid),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_rsp_valid(mem_rsp_valid), .mem_rsp_ready(mem_rsp_ready),
        .mem_rsp_accept(mem_rsp_accept), .result_valid(result_valid),
        .result_ready(result_ready), .result_accept(result_accept),
        .result_tag(result_tag),
        .result_output_block(result_output_block),
        .result_slice(result_slice),
        .result_accumulator(result_accumulator),
        .result_last(result_last),
        .token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),
        .token_done_accept(token_done_accept),
        .protocol_error(protocol_error));

`default_nettype wire
