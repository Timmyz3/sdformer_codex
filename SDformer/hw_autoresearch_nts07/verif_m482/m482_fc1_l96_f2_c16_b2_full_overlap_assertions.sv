`timescale 1ns/1ps
`default_nettype none
module m482_fc1_l96_f2_c16_b2_full_overlap_assertions #(
    parameter int TAG_BITS=24,EPOCH_BITS=16,DESC_BITS=12,
    parameter int LANES=96,ACC_BITS=19
)(
    input logic clk_core,input logic rst_core,
    input logic header_valid,input logic header_ready,input logic header_accept,
    input logic[TAG_BITS-1:0]header_tag,
    input logic[EPOCH_BITS-1:0]header_epoch,input logic header_factorized,
    input logic[4:0]header_chunk_count,
    input logic[DESC_BITS-1:0]header_descriptor_count,
    input logic directory_valid,input logic directory_ready,
    input logic directory_accept,input logic[4:0]directory_chunk,
    input logic[7:0]directory_descriptor_count,
    input logic factor_req_valid,input logic factor_req_ready,
    input logic factor_req_accept,input logic[1:0]factor_req_slot,
    input logic[TAG_BITS-1:0]factor_req_tag,
    input logic[EPOCH_BITS-1:0]factor_req_epoch,
    input logic[DESC_BITS-1:0]factor_req_descriptor,
    input logic[4:0]factor_req_chunk,input logic[6:0]factor_req_ordinal,
    input logic factor_rsp_valid,input logic factor_rsp_ready,
    input logic factor_rsp_accept,input logic[1:0]factor_rsp_slot,
    input logic[TAG_BITS-1:0]factor_rsp_tag,
    input logic[EPOCH_BITS-1:0]factor_rsp_epoch,
    input logic[DESC_BITS-1:0]factor_rsp_descriptor,
    input logic[4:0]factor_rsp_chunk,input logic[6:0]factor_rsp_ordinal,
    input logic[3:0]factor_rsp_source_offset,
    input logic[7:0]factor_rsp_context_mask,
    input logic[7:0]factor_rsp_sign_mask,
    input logic weight_req_valid,input logic weight_req_ready,
    input logic weight_req_accept,input logic[1:0]weight_req_slot,
    input logic[TAG_BITS-1:0]weight_req_tag,
    input logic[EPOCH_BITS-1:0]weight_req_epoch,
    input logic[8:0]weight_req_source,
    input logic weight_rsp_valid,input logic weight_rsp_ready,
    input logic weight_rsp_accept,input logic[1:0]weight_rsp_slot,
    input logic[TAG_BITS-1:0]weight_rsp_tag,
    input logic[EPOCH_BITS-1:0]weight_rsp_epoch,
    input logic[8:0]weight_rsp_source,
    input logic[LANES*8-1:0]weight_rsp_data,
    input logic[1:0]acc_bank_read_valid,input logic[1:0]acc_bank_read_ready,
    input logic[1:0][1:0]acc_bank_read_row,
    input logic[1:0][2:0]acc_bank_read_context,
    input logic acc_issue_accept,input logic commit_valid,
    input logic commit_ready,input logic commit_accept,
    input logic[TAG_BITS-1:0]commit_tag,
    input logic[EPOCH_BITS-1:0]commit_epoch,input logic[2:0]commit_context,
    input logic commit_last,input logic[LANES*ACC_BITS-1:0]commit_data,
    input logic done_valid,input logic done_ready,input logic done_accept,
    input logic[TAG_BITS-1:0]done_tag,
    input logic[EPOCH_BITS-1:0]done_epoch,input logic done_empty_bypass,
    input logic protocol_error,input logic numeric_overflow,input logic busy,
    input logic[31:0]debug_tile_cycles,
    input logic[31:0]debug_factor_requests,
    input logic[31:0]debug_weight_requests,
    input logic[31:0]debug_issue_rounds,
    input logic[31:0]debug_context_updates,
    input logic[31:0]debug_bank_conflict_extra_rounds,
    input logic[31:0]debug_factor_weight_overlap,
    input logic[31:0]debug_weight_update_overlap,
    input logic[31:0]debug_triple_overlap,
    input logic[31:0]debug_same_bank_rdw,
    input logic[31:0]debug_same_address_forward,
    input logic[2:0]debug_credit_count
);
    ap_header:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept==(header_valid&&header_ready));
    ap_directory:assert property(@(posedge clk_core)disable iff(rst_core)
        directory_accept==(directory_valid&&directory_ready));
    ap_factor_req:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_req_accept==(factor_req_valid&&factor_req_ready));
    ap_factor_rsp:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_rsp_accept==(factor_rsp_valid&&factor_rsp_ready));
    ap_weight_req:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_accept==(weight_req_valid&&weight_req_ready));
    ap_weight_rsp:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_rsp_accept==(weight_rsp_valid&&weight_rsp_ready));
    ap_commit:assert property(@(posedge clk_core)disable iff(rst_core)
        commit_accept==(commit_valid&&commit_ready));
    ap_done:assert property(@(posedge clk_core)disable iff(rst_core)
        done_accept==(done_valid&&done_ready));
    ap_factor_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_rsp_accept|->factor_rsp_context_mask!=0
        &&!(|(factor_rsp_sign_mask&~factor_rsp_context_mask)));
    ap_header_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        header_valid&&!header_ready|=>protocol_error||(header_valid
        &&$stable({header_tag,header_epoch,header_factorized,
            header_chunk_count,header_descriptor_count})));
    ap_directory_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        directory_valid&&!directory_ready|=>protocol_error||(directory_valid
        &&$stable({directory_chunk,directory_descriptor_count})));
    ap_factor_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_req_valid&&!factor_req_ready|=>protocol_error
        ||(factor_req_valid&&$stable({factor_req_slot,factor_req_tag,
            factor_req_epoch,factor_req_descriptor,factor_req_chunk,
            factor_req_ordinal})));
    ap_weight_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_valid&&!weight_req_ready|=>protocol_error
        ||(weight_req_valid&&$stable({weight_req_slot,weight_req_tag,
            weight_req_epoch,weight_req_source})));
    ap_factor_rsp_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_rsp_valid&&!factor_rsp_ready|=>protocol_error
        ||(factor_rsp_valid&&$stable({factor_rsp_slot,factor_rsp_tag,
            factor_rsp_epoch,factor_rsp_descriptor,factor_rsp_chunk,
            factor_rsp_ordinal,factor_rsp_source_offset,
            factor_rsp_context_mask,factor_rsp_sign_mask})));
    ap_weight_rsp_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_rsp_valid&&!weight_rsp_ready|=>protocol_error
        ||(weight_rsp_valid&&$stable({weight_rsp_slot,weight_rsp_tag,
            weight_rsp_epoch,weight_rsp_source,weight_rsp_data})));
    ap_commit_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        commit_valid&&!commit_ready|=>protocol_error
        ||(commit_valid&&$stable({commit_tag,commit_epoch,commit_context,
            commit_last,commit_data})));
    ap_credit:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_credit_count<=4);
    ap_weight_le_factor:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_weight_requests<=debug_factor_requests);
    ap_update_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_context_updates<=debug_issue_rounds*2);
    ap_bank0_even:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_bank_read_valid[0]|->acc_bank_read_context[0][0]==0);
    ap_bank1_odd:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_bank_read_valid[1]|->acc_bank_read_context[1][0]==1);
    ap_row_map0:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_bank_read_valid[0]|->acc_bank_read_row[0]
            ==acc_bank_read_context[0][2:1]);
    ap_row_map1:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_bank_read_valid[1]|->acc_bank_read_row[1]
            ==acc_bank_read_context[1][2:1]);
    ap_issue_atomic:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_issue_accept|->|(acc_bank_read_valid&acc_bank_read_ready));
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    cp_full_credit:cover property(@(posedge clk_core)debug_credit_count==4);
    cp_dual_bank:cover property(@(posedge clk_core)acc_issue_accept
        &&acc_bank_read_valid==2'b11);
    cp_factor_weight_overlap:cover property(@(posedge clk_core)
        factor_req_accept&&weight_req_accept);
    cp_weight_update_overlap:cover property(@(posedge clk_core)
        weight_req_accept&&acc_issue_accept);
    cp_triple_overlap:cover property(@(posedge clk_core)
        factor_req_accept&&weight_req_accept&&acc_issue_accept);
    cp_factor_stall:cover property(@(posedge clk_core)
        factor_req_valid&&!factor_req_ready);
    cp_weight_stall:cover property(@(posedge clk_core)
        weight_req_valid&&!weight_req_ready);
    cp_bank_stall:cover property(@(posedge clk_core)
        (|acc_bank_read_valid)&&!acc_issue_accept);
    cp_commit_stall:cover property(@(posedge clk_core)
        commit_valid&&!commit_ready);
    cp_same_bank_rdw:cover property(@(posedge clk_core)
        debug_same_bank_rdw>0);
    cp_same_address_forward:cover property(@(posedge clk_core)
        debug_same_address_forward>0);
    cp_conflict:cover property(@(posedge clk_core)
        debug_bank_conflict_extra_rounds>0);
    cp_fault:cover property(@(posedge clk_core)protocol_error);
    cp_done:cover property(@(posedge clk_core)done_accept&&!done_empty_bypass);
    cp_empty:cover property(@(posedge clk_core)done_accept&&done_empty_bypass);
endmodule
`default_nettype wire
