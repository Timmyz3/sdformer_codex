`timescale 1ns/1ps
`default_nettype none
module m229_fc1_dual_held_prefetch_replay_assertions#(
    parameter int FANOUT=1,TAG_BITS=24,EPOCH_BITS=16,LANES=96,ACC_BITS=19
)(
    input logic clk_core,input logic rst_core,
    input logic header_valid,input logic header_ready,input logic header_accept,
    input logic descriptor_valid,input logic descriptor_ready,
    input logic[8:0]descriptor_source,input logic[7:0]descriptor_context_mask,
    input logic[7:0]descriptor_sign_mask,input logic descriptor_last,
    input logic descriptor_accept,input logic weight_req_valid,
    input logic weight_req_ready,input logic[1:0]weight_req_slot,
    input logic[TAG_BITS-1:0]weight_req_tag,
    input logic[EPOCH_BITS-1:0]weight_req_epoch,
    input logic[8:0]weight_req_source,input logic weight_req_accept,
    input logic weight_rsp_valid,input logic weight_rsp_ready,
    input logic weight_rsp_accept,input logic[FANOUT-1:0]acc_update_valid,
    input logic acc_update_ready,input logic[2:0]acc_update_context[0:FANOUT-1],
    input logic[FANOUT*LANES*ACC_BITS-1:0]acc_read_data,
    input logic[FANOUT*LANES*ACC_BITS-1:0]acc_write_data,
    input logic acc_update_accept,input logic done_valid,input logic done_ready,
    input logic done_accept,input logic protocol_error,input logic numeric_overflow,
    input logic[2:0]debug_credit_count,
    input logic[31:0]debug_descriptor_count,
    input logic[31:0]debug_weight_request_count,
    input logic[31:0]debug_weight_response_count,
    input logic[31:0]debug_context_update_count,
    input logic[31:0]debug_overlap_count
);
    logic[FANOUT*(3+LANES*ACC_BITS)-1:0]update_flat;
    always_comb for(int slot=0;slot<FANOUT;slot++)begin
        update_flat[slot*(3+LANES*ACC_BITS)+:3]=acc_update_context[slot];
        update_flat[slot*(3+LANES*ACC_BITS)+3+:LANES*ACC_BITS]
            =acc_write_data[slot*LANES*ACC_BITS+:LANES*ACC_BITS];
    end
    ap_header:assert property(@(posedge clk_core)disable iff(rst_core)
        header_accept==(header_valid&&header_ready));
    ap_desc:assert property(@(posedge clk_core)disable iff(rst_core)
        descriptor_accept==(descriptor_valid&&descriptor_ready));
    ap_req:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_accept==(weight_req_valid&&weight_req_ready));
    ap_rsp:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_rsp_accept==(weight_rsp_valid&&weight_rsp_ready));
    ap_update:assert property(@(posedge clk_core)disable iff(rst_core)
        acc_update_accept==((|acc_update_valid)&&acc_update_ready&&!numeric_overflow));
    ap_done:assert property(@(posedge clk_core)disable iff(rst_core)
        done_accept==(done_valid&&done_ready));
    ap_desc_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        descriptor_accept|->descriptor_source<384&&descriptor_context_mask!=0
        &&!(|(descriptor_sign_mask&~descriptor_context_mask)));
    ap_desc_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        descriptor_valid&&!descriptor_ready|=>protocol_error||(descriptor_valid
        &&$stable({descriptor_source,descriptor_context_mask,
            descriptor_sign_mask,descriptor_last})));
    ap_req_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        weight_req_valid&&!weight_req_ready|=>protocol_error||(weight_req_valid
        &&$stable({weight_req_slot,weight_req_tag,weight_req_epoch,
            weight_req_source})));
    ap_update_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        (|acc_update_valid)&&!acc_update_ready|=>protocol_error||numeric_overflow
        ||((|acc_update_valid)&&$stable({acc_update_valid,update_flat,acc_read_data})));
    ap_credit:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_credit_count<=4);
    ap_req_le_desc:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_weight_request_count<=debug_descriptor_count);
    ap_rsp_le_req:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_weight_response_count<=debug_weight_request_count);
    ap_width:assert property(@(posedge clk_core)disable iff(rst_core)
        $countones(acc_update_valid)<=FANOUT);
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    cp_full_credit:cover property(@(posedge clk_core)debug_credit_count==4);
    cp_overlap:cover property(@(posedge clk_core)
        weight_req_accept&&acc_update_accept);
    cp_fanout:cover property(@(posedge clk_core)
        $countones(acc_update_valid)==FANOUT);
    cp_req_stall:cover property(@(posedge clk_core)weight_req_valid&&!weight_req_ready);
    cp_update_stall:cover property(@(posedge clk_core)(|acc_update_valid)&&!acc_update_ready);
    cp_fault:cover property(@(posedge clk_core)protocol_error);
    cp_done:cover property(@(posedge clk_core)done_accept&&debug_overlap_count>0);
endmodule
`default_nettype wire
