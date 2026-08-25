`timescale 1ns/1ps
`default_nettype none
module m270_fc1_descriptor_lifecycle_assertions #(
    parameter int LANES=8,CONTEXTS=8,ACC_BITS=19,TAG_BITS=24,
    parameter int EPOCH_BITS=16,DESC_BITS=12,FACTOR_ADDR_BITS=20
)(
    input logic clk_core,input logic rst_core,
    input logic header_valid,input logic header_ready,input logic header_accept,
    input logic[1:0]header_mode,
    input logic[DESC_BITS-1:0]header_descriptor_count,
    input logic factor_req_valid,input logic factor_req_ready,
    input logic factor_req_accept,input logic[TAG_BITS-1:0]factor_req_tag,
    input logic[EPOCH_BITS-1:0]factor_req_epoch,
    input logic[DESC_BITS-1:0]factor_req_descriptor,
    input logic[FACTOR_ADDR_BITS-1:0]factor_req_addr,
    input logic factor_rsp_valid,input logic factor_rsp_ready,
    input logic factor_rsp_accept,
    input logic weight_req_valid,input logic weight_req_ready,
    input logic weight_req_accept,input logic[TAG_BITS-1:0]weight_req_tag,
    input logic[EPOCH_BITS-1:0]weight_req_epoch,
    input logic[DESC_BITS-1:0]weight_req_descriptor,
    input logic[8:0]weight_req_source,input logic weight_rsp_valid,
    input logic weight_rsp_ready,input logic weight_rsp_accept,
    input logic acc_read_req_valid,input logic acc_read_req_ready,
    input logic acc_read_req_accept,input logic[TAG_BITS-1:0]acc_read_req_tag,
    input logic[EPOCH_BITS-1:0]acc_read_req_epoch,
    input logic[DESC_BITS-1:0]acc_read_req_descriptor,
    input logic[$clog2(CONTEXTS)-1:0]acc_read_req_context,
    input logic acc_read_req_commit,input logic acc_read_rsp_valid,
    input logic acc_read_rsp_ready,input logic acc_read_rsp_accept,
    input logic acc_write_valid,input logic acc_write_ready,
    input logic acc_write_accept,input logic[TAG_BITS-1:0]acc_write_tag,
    input logic[EPOCH_BITS-1:0]acc_write_epoch,
    input logic[DESC_BITS-1:0]acc_write_descriptor,
    input logic[$clog2(CONTEXTS)-1:0]acc_write_context,
    input logic acc_write_update,
    input logic[LANES*ACC_BITS-1:0]acc_write_data,
    input logic commit_valid,input logic commit_ready,input logic commit_accept,
    input logic[TAG_BITS-1:0]commit_tag,
    input logic[EPOCH_BITS-1:0]commit_epoch,
    input logic[$clog2(CONTEXTS)-1:0]commit_context,
    input logic commit_last,input logic[LANES*ACC_BITS-1:0]commit_data,
    input logic abort_valid,input logic abort_ready,input logic abort_accept,
    input logic[TAG_BITS-1:0]abort_tag,
    input logic[EPOCH_BITS-1:0]abort_epoch,input logic[3:0]abort_reason,
    input logic done_valid,input logic done_ready,input logic done_accept,
    input logic[TAG_BITS-1:0]done_tag,input logic[EPOCH_BITS-1:0]done_epoch,
    input logic done_empty_bypass,input logic descriptor_retire_valid,
    input logic[DESC_BITS-1:0]descriptor_retire_index,
    input logic[15:0]descriptor_retire_cycles,
    input logic protocol_error,input logic numeric_overflow,input logic busy
);
    ap_header_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept==(header_valid&&header_ready));
    ap_factor_req_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_req_accept==(factor_req_valid&&factor_req_ready));
    ap_factor_rsp_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_rsp_accept==(factor_rsp_valid&&factor_rsp_ready));
    ap_weight_req_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_accept==(weight_req_valid&&weight_req_ready));
    ap_weight_rsp_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_rsp_accept==(weight_rsp_valid&&weight_rsp_ready));
    ap_acc_read_req_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_read_req_accept==(acc_read_req_valid&&acc_read_req_ready));
    ap_acc_read_rsp_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_read_rsp_accept==(acc_read_rsp_valid&&acc_read_rsp_ready));
    ap_acc_write_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_write_accept==(acc_write_valid&&acc_write_ready));
    ap_commit_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        commit_accept==(commit_valid&&commit_ready));
    ap_abort_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        abort_accept==(abort_valid&&abort_ready));
    ap_done_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        done_accept==(done_valid&&done_ready));

    ap_factor_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        factor_req_valid&&!factor_req_ready|=>protocol_error
        ||(factor_req_valid&&$stable({factor_req_tag,factor_req_epoch,
            factor_req_descriptor,factor_req_addr})));
    ap_weight_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_valid&&!weight_req_ready|=>protocol_error
        ||(weight_req_valid&&$stable({weight_req_tag,weight_req_epoch,
            weight_req_descriptor,weight_req_source})));
    ap_acc_read_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_read_req_valid&&!acc_read_req_ready|=>protocol_error
        ||(acc_read_req_valid&&$stable({acc_read_req_tag,acc_read_req_epoch,
            acc_read_req_descriptor,acc_read_req_context,acc_read_req_commit})));
    ap_acc_write_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_write_valid&&!acc_write_ready|=>protocol_error||numeric_overflow
        ||(acc_write_valid&&$stable({acc_write_tag,acc_write_epoch,
            acc_write_descriptor,acc_write_context,acc_write_update,
            acc_write_data})));
    ap_commit_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        commit_valid&&!commit_ready|=>protocol_error
        ||(commit_valid&&$stable({commit_tag,commit_epoch,commit_context,
            commit_last,commit_data})));
    ap_abort_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        abort_valid&&!abort_ready|=>(abort_valid&&$stable({abort_tag,
            abort_epoch,abort_reason})));
    ap_done_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        done_valid&&!done_ready|=>protocol_error
        ||(done_valid&&$stable({done_tag,done_epoch,done_empty_bypass})));

    ap_empty_atomic:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept&&header_descriptor_count==0|->done_accept
        &&done_empty_bypass&&!busy);
    ap_mode_legal:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept|->header_mode<=2);
    ap_retire_minimum:assert property(@(posedge clk_core)disable iff(rst_core)
        descriptor_retire_valid|->descriptor_retire_cycles>=9);
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    ap_overflow_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(numeric_overflow)|->numeric_overflow);
    ap_fail_closed_side_effects:assert property(
        @(posedge clk_core)disable iff(rst_core)protocol_error
        |->!(factor_req_accept||weight_req_accept||acc_read_req_accept
             ||acc_write_accept||commit_accept||done_accept));
    ap_abort_reason:assert property(@(posedge clk_core)disable iff(rst_core)
        abort_valid|->abort_reason inside {[1:5]});

    cp_empty:cover property(@(posedge clk_core)header_accept
        &&header_descriptor_count==0&&done_accept);
    cp_dense:cover property(@(posedge clk_core)header_accept&&header_mode==0
        &&header_descriptor_count!=0);
    cp_bit_sparse:cover property(@(posedge clk_core)header_accept&&header_mode==1);
    cp_factorized:cover property(@(posedge clk_core)header_accept&&header_mode==2);
    cp_factor_stall:cover property(@(posedge clk_core)
        factor_req_valid&&!factor_req_ready);
    cp_weight_stall:cover property(@(posedge clk_core)
        weight_req_valid&&!weight_req_ready);
    cp_acc_read_stall:cover property(@(posedge clk_core)
        acc_read_req_valid&&!acc_read_req_ready);
    cp_acc_write_stall:cover property(@(posedge clk_core)
        acc_write_valid&&!acc_write_ready);
    cp_commit_stall:cover property(@(posedge clk_core)commit_valid&&!commit_ready);
    cp_commit_last:cover property(@(posedge clk_core)commit_accept&&commit_last);
    cp_abort_stall:cover property(@(posedge clk_core)abort_valid&&!abort_ready);
    cp_factor_fault:cover property(@(posedge clk_core)abort_valid&&abort_reason==2);
    cp_weight_fault:cover property(@(posedge clk_core)abort_valid&&abort_reason==3);
    cp_acc_fault:cover property(@(posedge clk_core)abort_valid&&abort_reason==4);
    cp_overflow:cover property(@(posedge clk_core)abort_valid&&abort_reason==5);
    cp_protocol_fault:cover property(@(posedge clk_core)
        abort_valid&&abort_reason==1);
endmodule
`default_nettype wire
