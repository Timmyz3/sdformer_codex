`timescale 1ns/1ps
`default_nettype none

// M229 decouples descriptor intake, weight request/response and Acc19 replay.
// Four physical credits contain at least a current and next held vector; the
// external 768-bit SRAM may accept one request per cycle while another slot is
// updating up to FANOUT context banks.  Acc19 storage is an explicit port cut:
// each context is a separate bank, so selected contexts are conflict-free.
module m229_fc1_dual_held_prefetch_replay_island #(
    parameter int FANOUT=1,
    parameter int TAG_BITS=24,
    parameter int EPOCH_BITS=16,
    parameter int LANES=96,
    parameter int ACC_BITS=19,
    parameter int DEPTH=4
)(
    input logic clk_core,input logic rst_core,
    input logic header_valid,output logic header_ready,
    input logic[TAG_BITS-1:0]header_tag,
    input logic[EPOCH_BITS-1:0]header_epoch,
    output logic header_accept,

    input logic descriptor_valid,output logic descriptor_ready,
    input logic[8:0]descriptor_source,
    input logic[7:0]descriptor_context_mask,
    input logic[7:0]descriptor_sign_mask,
    input logic descriptor_last,
    output logic descriptor_accept,

    output logic weight_req_valid,input logic weight_req_ready,
    output logic[1:0]weight_req_slot,
    output logic[TAG_BITS-1:0]weight_req_tag,
    output logic[EPOCH_BITS-1:0]weight_req_epoch,
    output logic[8:0]weight_req_source,
    output logic weight_req_accept,
    input logic weight_rsp_valid,output logic weight_rsp_ready,
    input logic[1:0]weight_rsp_slot,
    input logic[TAG_BITS-1:0]weight_rsp_tag,
    input logic[EPOCH_BITS-1:0]weight_rsp_epoch,
    input logic[8:0]weight_rsp_source,
    input logic[LANES*8-1:0]weight_rsp_data,
    output logic weight_rsp_accept,

    output logic[FANOUT-1:0]acc_update_valid,input logic acc_update_ready,
    output logic[2:0]acc_update_context[0:FANOUT-1],
    input logic[FANOUT*LANES*ACC_BITS-1:0]acc_read_data,
    output logic[FANOUT*LANES*ACC_BITS-1:0]acc_write_data,
    output logic acc_update_accept,

    output logic done_valid,input logic done_ready,
    output logic[TAG_BITS-1:0]done_tag,
    output logic[EPOCH_BITS-1:0]done_epoch,
    output logic done_accept,
    output logic protocol_error,output logic numeric_overflow,
    output logic busy,output logic[2:0]debug_credit_count,
    output logic[31:0]debug_descriptor_count,
    output logic[31:0]debug_weight_request_count,
    output logic[31:0]debug_weight_response_count,
    output logic[31:0]debug_context_update_count,
    output logic[31:0]debug_overlap_count
);
    localparam bit PARAMETERS_LEGAL=(FANOUT==1||FANOUT==2||FANOUT==4)
        &&TAG_BITS==24&&EPOCH_BITS==16&&LANES==96&&ACC_BITS==19&&DEPTH==4;
    logic active_q,done_q,fault_q,last_seen_q;
    logic[TAG_BITS-1:0]tag_q;logic[EPOCH_BITS-1:0]epoch_q;
    logic[1:0]head_q,tail_q;logic[2:0]count_q;
    logic slot_valid_q[0:3],slot_requested_q[0:3],slot_weight_valid_q[0:3];
    logic[8:0]slot_source_q[0:3];logic[7:0]slot_context_q[0:3];
    logic[7:0]slot_sign_q[0:3];logic slot_last_q[0:3];
    logic[LANES*8-1:0]slot_weight_q[0:3];
    logic request_found;logic[1:0]request_slot;
    logic request_hold_q;logic[1:0]request_hold_slot_q;
    logic[7:0]head_pending;logic[7:0]pending_after;
    logic[3:0]selected_valid;logic[2:0]selected_context[0:3];
    logic[2:0]selected_count;
    logic pop_head,push_descriptor;
    logic descriptor_shape_legal;
    logic illegal_header,illegal_descriptor,illegal_response;
    logic overflow_this_cycle;
    logic[31:0]descriptor_count_q,request_count_q,response_count_q;
    logic[31:0]context_update_count_q,overlap_count_q;

    generate if(!PARAMETERS_LEGAL)begin:g_bad
        initial $fatal(1,"M229 frozen geometry/FANOUT/depth drift");
    end endgenerate

    always_comb begin:request_select
        request_found=0;request_slot=0;
        for(int slot=0;slot<4;slot++)if(!request_found&&slot_valid_q[slot]
                &&!slot_requested_q[slot])begin
            request_found=1;request_slot=slot[1:0];
        end
    end

    always_comb begin:replay_select
        logic[7:0]work;
        head_pending=slot_valid_q[head_q]?slot_context_q[head_q]:'0;
        work=head_pending;selected_valid='0;selected_count='0;
        for(int slot=0;slot<4;slot++)selected_context[slot]='0;
        for(int lane_slot=0;lane_slot<FANOUT;lane_slot++)begin
            logic found;found=0;
            for(int ctx=0;ctx<8;ctx++)if(!found&&work[ctx])begin
                found=1;selected_valid[lane_slot]=1;
                selected_context[lane_slot]=ctx[2:0];work[ctx]=0;
                selected_count=selected_count+1'b1;
            end
        end
        pending_after=work;
    end

    always_comb begin:datapath
        acc_update_valid='0;acc_write_data='0;overflow_this_cycle=0;
        for(int slot=0;slot<FANOUT;slot++)acc_update_context[slot]='0;
        if(active_q&&count_q!=0&&slot_weight_valid_q[head_q]&&!fault_q)begin
            for(int lane_slot=0;lane_slot<FANOUT;lane_slot++)
                if(selected_valid[lane_slot])begin
                    acc_update_valid[lane_slot]=1;
                    acc_update_context[lane_slot]=selected_context[lane_slot];
                    for(int lane=0;lane<LANES;lane++)begin
                        logic signed[ACC_BITS:0]sum_ext;
                        logic signed[ACC_BITS-1:0]current_value,weight_value;
                        current_value=acc_read_data[(lane_slot*LANES+lane)
                            *ACC_BITS+:ACC_BITS];
                        weight_value={{(ACC_BITS-8){slot_weight_q[head_q]
                            [lane*8+7]}},slot_weight_q[head_q][lane*8+:8]};
                        if(slot_sign_q[head_q][selected_context[lane_slot]])
                            sum_ext=$signed(current_value)-$signed(weight_value);
                        else sum_ext=$signed(current_value)+$signed(weight_value);
                        acc_write_data[(lane_slot*LANES+lane)*ACC_BITS+:ACC_BITS]
                            =sum_ext[ACC_BITS-1:0];
                        if(sum_ext[ACC_BITS]!=sum_ext[ACC_BITS-1])
                            overflow_this_cycle=1;
                    end
                end
        end
        acc_update_accept=(|acc_update_valid)&&acc_update_ready
            &&!overflow_this_cycle;
        pop_head=acc_update_accept&&pending_after==0;
    end

    always_comb begin:interfaces
        illegal_header=header_valid&&(active_q||done_q);
        descriptor_shape_legal=active_q&&!done_q&&!last_seen_q
            &&descriptor_source<384&&descriptor_context_mask!=0
            &&!(|(descriptor_sign_mask&~descriptor_context_mask));
        illegal_descriptor=descriptor_valid&&!descriptor_shape_legal;
        illegal_response=weight_rsp_valid&&!(active_q&&!done_q
            &&weight_rsp_slot<4&&slot_valid_q[weight_rsp_slot]
            &&slot_requested_q[weight_rsp_slot]
            &&!slot_weight_valid_q[weight_rsp_slot]
            &&weight_rsp_tag==tag_q&&weight_rsp_epoch==epoch_q
            &&weight_rsp_source==slot_source_q[weight_rsp_slot]);
        header_ready=!active_q&&!done_q&&!fault_q;header_accept=header_valid&&header_ready;
        descriptor_ready=descriptor_shape_legal&&!fault_q
            &&(count_q<DEPTH||pop_head);
        descriptor_accept=descriptor_valid&&descriptor_ready;
        push_descriptor=descriptor_accept;
        weight_req_valid=active_q&&!done_q&&(request_hold_q||request_found)
            &&!fault_q;
        weight_req_slot=weight_req_valid
            ?(request_hold_q?request_hold_slot_q:request_slot):'0;
        weight_req_tag=weight_req_valid?tag_q:'0;
        weight_req_epoch=weight_req_valid?epoch_q:'0;
        weight_req_source=weight_req_valid?slot_source_q[weight_req_slot]:'0;
        weight_req_accept=weight_req_valid&&weight_req_ready;
        weight_rsp_ready=active_q&&!done_q&&!fault_q&&!illegal_response;
        weight_rsp_accept=weight_rsp_valid&&weight_rsp_ready;
        done_valid=done_q&&!fault_q;done_tag=done_valid?tag_q:'0;
        done_epoch=done_valid?epoch_q:'0;done_accept=done_valid&&done_ready;
        protocol_error=fault_q||illegal_header||illegal_descriptor||illegal_response;
        numeric_overflow=overflow_this_cycle;
        busy=active_q||done_q;debug_credit_count=count_q;
        debug_descriptor_count=descriptor_count_q;
        debug_weight_request_count=request_count_q;
        debug_weight_response_count=response_count_q;
        debug_context_update_count=context_update_count_q;
        debug_overlap_count=overlap_count_q;
    end

    always_ff @(posedge clk_core)begin
        if(rst_core)begin
            active_q<=0;done_q<=0;fault_q<=0;last_seen_q<=0;
            request_hold_q<=0;request_hold_slot_q<=0;
            tag_q<='0;epoch_q<='0;head_q<=0;tail_q<=0;count_q<=0;
            descriptor_count_q<=0;request_count_q<=0;response_count_q<=0;
            context_update_count_q<=0;overlap_count_q<=0;
            for(int slot=0;slot<4;slot++)begin
                slot_valid_q[slot]<=0;slot_requested_q[slot]<=0;
                slot_weight_valid_q[slot]<=0;slot_source_q[slot]<='0;
                slot_context_q[slot]<='0;slot_sign_q[slot]<='0;
                slot_last_q[slot]<=0;slot_weight_q[slot]<='0;
            end
        end else begin
            if(illegal_header||illegal_descriptor||illegal_response
                    ||overflow_this_cycle)fault_q<=1;
            if(!protocol_error&&!overflow_this_cycle)begin
                if(header_accept)begin
                    active_q<=1;done_q<=0;last_seen_q<=0;tag_q<=header_tag;
                    request_hold_q<=0;request_hold_slot_q<=0;
                    epoch_q<=header_epoch;head_q<=0;tail_q<=0;count_q<=0;
                    descriptor_count_q<=0;request_count_q<=0;response_count_q<=0;
                    context_update_count_q<=0;overlap_count_q<=0;
                    for(int slot=0;slot<4;slot++)begin slot_valid_q[slot]<=0;
                        slot_requested_q[slot]<=0;slot_weight_valid_q[slot]<=0;
                    end
                end
                if(weight_req_valid&&!weight_req_ready&&!request_hold_q)begin
                    request_hold_q<=1;request_hold_slot_q<=weight_req_slot;
                end
                if(weight_req_accept)begin
                    request_hold_q<=0;
                    slot_requested_q[weight_req_slot]<=1;
                    request_count_q<=request_count_q+1'b1;
                end
                if(weight_rsp_accept)begin
                    slot_weight_valid_q[weight_rsp_slot]<=1;
                    slot_weight_q[weight_rsp_slot]<=weight_rsp_data;
                    response_count_q<=response_count_q+1'b1;
                end
                if(acc_update_accept)begin
                    slot_context_q[head_q]<=pending_after;
                    context_update_count_q<=context_update_count_q+selected_count;
                end
                if(weight_req_accept&&acc_update_accept)
                    overlap_count_q<=overlap_count_q+1'b1;
                if(pop_head)begin
                    slot_valid_q[head_q]<=0;slot_requested_q[head_q]<=0;
                    slot_weight_valid_q[head_q]<=0;head_q<=head_q+1'b1;
                    if(slot_last_q[head_q])begin active_q<=0;done_q<=1;end
                end
                if(push_descriptor)begin
                    slot_valid_q[tail_q]<=1;slot_requested_q[tail_q]<=0;
                    slot_weight_valid_q[tail_q]<=0;
                    slot_source_q[tail_q]<=descriptor_source;
                    slot_context_q[tail_q]<=descriptor_context_mask;
                    slot_sign_q[tail_q]<=descriptor_sign_mask;
                    slot_last_q[tail_q]<=descriptor_last;tail_q<=tail_q+1'b1;
                    descriptor_count_q<=descriptor_count_q+1'b1;
                    if(descriptor_last)last_seen_q<=1;
                end
                case({push_descriptor,pop_head})
                    2'b10:count_q<=count_q+1'b1;
                    2'b01:count_q<=count_q-1'b1;
                    default:count_q<=count_q;
                endcase
                if(done_accept)begin done_q<=0;last_seen_q<=0;end
            end
        end
    end
endmodule
`default_nettype wire
