`timescale 1ns/1ps
`default_nettype none
`ifndef M229_FANOUT
`define M229_FANOUT 1
`endif
module tb_m229_fc1_dual_held_prefetch_replay_island;
    localparam int FANOUT=`M229_FANOUT,TAG_BITS=24,EPOCH_BITS=16;
    localparam int LANES=96,ACC_BITS=19;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic header_valid,header_ready,header_accept;logic[23:0]header_tag;
    logic[15:0]header_epoch;logic descriptor_valid,descriptor_ready;
    logic[8:0]descriptor_source;logic[7:0]descriptor_context_mask;
    logic[7:0]descriptor_sign_mask;logic descriptor_last,descriptor_accept;
    logic weight_req_valid,weight_req_ready;logic[1:0]weight_req_slot;
    logic[23:0]weight_req_tag;logic[15:0]weight_req_epoch;
    logic[8:0]weight_req_source;logic weight_req_accept;
    logic weight_rsp_valid,weight_rsp_ready;logic[1:0]weight_rsp_slot;
    logic[23:0]weight_rsp_tag;logic[15:0]weight_rsp_epoch;
    logic[8:0]weight_rsp_source;logic[LANES*8-1:0]weight_rsp_data;
    logic weight_rsp_accept;logic[FANOUT-1:0]acc_update_valid;
    logic acc_update_ready;logic[2:0]acc_update_context[0:FANOUT-1];
    logic[FANOUT*LANES*ACC_BITS-1:0]acc_read_data,acc_write_data;
    logic acc_update_accept,done_valid,done_ready,done_accept;
    logic[23:0]done_tag;logic[15:0]done_epoch;
    logic protocol_error,numeric_overflow,busy;logic[2:0]debug_credit_count;
    logic[31:0]debug_descriptor_count,debug_weight_request_count;
    logic[31:0]debug_weight_response_count,debug_context_update_count;
    logic[31:0]debug_overlap_count;
    integer signed bank[0:7][0:LANES-1],expected[0:7][0:LANES-1];
    logic[7:0]mask_by_source[0:383],sign_by_source[0:383];
    integer cycles,groups,total_desc,total_updates,total_overlaps;
    integer attacks,req_stalls,upd_stalls;
    integer group_cycles[0:2],clean_group_mode;
    typedef struct packed{logic[1:0]slot;logic[23:0]tag;logic[15:0]epoch;
        logic[8:0]source;}req_t;
    req_t response_queue[$];logic response_consumed,server_enabled;

    m229_fc1_dual_held_prefetch_replay_island#(.FANOUT(FANOUT))dut(.*);
    m229_fc1_dual_held_prefetch_replay_assertions#(.FANOUT(FANOUT))sva(.*);
    function automatic integer signed wval(input integer source,input integer lane);
        integer value;begin if(lane%11==0)value=-128;else if(lane%11==1)value=127;
        else begin value=(source*17+lane*29+13)%255;value-=127;end return value;end
    endfunction
    function automatic logic[LANES*8-1:0]wvec(input integer source);
        logic[LANES*8-1:0]value;begin value='0;
        for(integer lane=0;lane<LANES;lane++)value[lane*8+:8]=wval(source,lane);
        return value;end
    endfunction
    always_comb begin acc_read_data='0;
        for(int slot=0;slot<FANOUT;slot++)for(int lane=0;lane<LANES;lane++)
            acc_read_data[(slot*LANES+lane)*ACC_BITS+:ACC_BITS]
                =bank[acc_update_context[slot]][lane];
    end
    always @(posedge clk_core)begin
        if(rst_core)cycles=0;else begin cycles++;
            if(weight_req_valid&&!weight_req_ready)req_stalls++;
            if((|acc_update_valid)&&!acc_update_ready)upd_stalls++;
            if(weight_req_accept&&server_enabled)begin req_t item;
                item.slot=weight_req_slot;item.tag=weight_req_tag;
                item.epoch=weight_req_epoch;item.source=weight_req_source;
                response_queue.push_back(item);end
            if(weight_rsp_accept)response_consumed=1;
            if(acc_update_accept)for(integer slot=0;slot<FANOUT;slot++)
                if(acc_update_valid[slot])begin
                    integer ctx;ctx=acc_update_context[slot];
                    for(integer lane=0;lane<LANES;lane++)begin
                        logic signed[ACC_BITS-1:0]observed;
                        integer signed reference;
                        observed=acc_write_data[(slot*LANES+lane)*ACC_BITS+:ACC_BITS];
                        reference=bank[ctx][lane]
                            +(sign_by_source[dut.slot_source_q[dut.head_q]][ctx]
                              ?-wval(dut.slot_source_q[dut.head_q],lane)
                              : wval(dut.slot_source_q[dut.head_q],lane));
                        if($signed(observed)!==reference)
                            $fatal(1,"M229 update mismatch F=%0d src=%0d ctx=%0d lane=%0d e=%0d o=%0d",
                                FANOUT,dut.slot_source_q[dut.head_q],ctx,lane,reference,$signed(observed));
                        bank[ctx][lane]=reference;
                    end
                end
        end
    end
    always @(negedge clk_core)begin
        if(rst_core)begin weight_req_ready=0;acc_update_ready=0;end else begin
            weight_req_ready=clean_group_mode?1:(cycles%5)!=2;
            acc_update_ready=clean_group_mode?1:(cycles%7)!=3;
        end
        if(response_consumed)begin weight_rsp_valid=0;response_consumed=0;end
        if(!weight_rsp_valid&&response_queue.size()>0)begin req_t item;
            item=response_queue.pop_front();weight_rsp_valid=1;
            weight_rsp_slot=item.slot;weight_rsp_tag=item.tag;
            weight_rsp_epoch=item.epoch;weight_rsp_source=item.source;
            weight_rsp_data=wvec(item.source);
        end
    end
    task automatic clear;begin header_valid=0;header_tag=0;header_epoch=0;
        descriptor_valid=0;descriptor_source=0;descriptor_context_mask=0;
        descriptor_sign_mask=0;descriptor_last=0;weight_rsp_valid=0;
        weight_rsp_slot=0;weight_rsp_tag=0;weight_rsp_epoch=0;
        weight_rsp_source=0;weight_rsp_data=0;done_ready=1;response_consumed=0;
        response_queue.delete();end endtask
    task automatic reset;begin @(negedge clk_core);clear();rst_core=1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;end endtask
    task automatic build(input integer index,input integer count);begin
        for(integer source=0;source<384;source++)begin mask_by_source[source]=0;
            sign_by_source[source]=0;end
        for(integer ctx=0;ctx<8;ctx++)for(integer lane=0;lane<LANES;lane++)begin
            bank[ctx][lane]=0;expected[ctx][lane]=0;end
        for(integer item=0;item<count;item++)begin integer source,fan;
            source=(item*11+index*3)%384;fan=(item%8)+1;
            mask_by_source[source]=(8'h1<<fan)-1;
            for(integer ctx=0;ctx<fan;ctx++)begin
                sign_by_source[source][ctx]=(item+ctx)%3==0;
                for(integer lane=0;lane<LANES;lane++)expected[ctx][lane]
                    +=sign_by_source[source][ctx]?-wval(source,lane):wval(source,lane);
            end
        end
    end endtask
    task automatic start(input integer index);begin @(negedge clk_core);
        header_valid=1;header_tag=24'h229000+index;header_epoch=16'h2290+index;
        do @(posedge clk_core);while(!header_ready);@(negedge clk_core);header_valid=0;
    end endtask
    task automatic run(input integer index,input integer count,input logic clean);
        integer start_cycle;begin build(index,count);clean_group_mode=clean;
        start(index);start_cycle=cycles;
        for(integer item=0;item<count;item++)begin integer source;
            source=(item*11+index*3)%384;@(negedge clk_core);
            descriptor_valid=1;descriptor_source=source;
            descriptor_context_mask=mask_by_source[source];
            descriptor_sign_mask=sign_by_source[source];descriptor_last=item==count-1;
            do @(posedge clk_core);while(!descriptor_ready);
        end
        @(negedge clk_core);descriptor_valid=0;descriptor_last=0;
        do @(posedge clk_core);while(!done_accept);
        for(integer ctx=0;ctx<8;ctx++)for(integer lane=0;lane<LANES;lane++)
            if(bank[ctx][lane]!==expected[ctx][lane])
                $fatal(1,"M229 final mismatch F=%0d g=%0d ctx=%0d lane=%0d",
                    FANOUT,index,ctx,lane);
        if(debug_descriptor_count!=count||debug_weight_request_count!=count
                ||debug_weight_response_count!=count)
            $fatal(1,"M229 transaction conservation F=%0d g=%0d",FANOUT,index);
        group_cycles[index]=cycles-start_cycle;groups++;total_desc+=count;
        total_updates+=debug_context_update_count;
        total_overlaps+=debug_overlap_count;end endtask
    task automatic bad_descriptor;begin reset();clean_group_mode=0;start(7);
        @(negedge clk_core);descriptor_valid=1;descriptor_source=1;
        descriptor_context_mask=8'h01;descriptor_sign_mask=8'h02;#0.1;
        if(descriptor_ready)$fatal(1,"M229 bad descriptor ready");
        @(posedge clk_core);#0.2;if(!protocol_error)$fatal(1,"M229 bad descriptor no fault");
        attacks++;@(negedge clk_core);descriptor_valid=0;end endtask
    task automatic bad_response;logic[1:0]slot;logic[8:0]source;begin
        reset();clean_group_mode=1;server_enabled=0;start(8);
        @(negedge clk_core);descriptor_valid=1;descriptor_source=9;
        descriptor_context_mask=8'h03;descriptor_sign_mask=8'h01;
        descriptor_last=1;do @(posedge clk_core);while(!descriptor_ready);
        @(negedge clk_core);descriptor_valid=0;
        do begin @(posedge clk_core);slot=weight_req_slot;source=weight_req_source;
        end while(!weight_req_accept);
        @(negedge clk_core);weight_rsp_valid=1;weight_rsp_slot=slot;
        weight_rsp_tag=24'h229008;weight_rsp_epoch=16'h2298;
        weight_rsp_source=source+1'b1;weight_rsp_data=wvec(source);#0.1;
        if(weight_rsp_ready)$fatal(1,"M229 wrong response ready");
        @(posedge clk_core);#0.2;
        if(!protocol_error)$fatal(1,"M229 wrong response no fault");
        attacks++;@(negedge clk_core);weight_rsp_valid=0;server_enabled=1;
    end endtask
    task automatic overflow_attack;begin
        reset();clean_group_mode=1;server_enabled=1;
        for(integer ctx=0;ctx<8;ctx++)for(integer lane=0;lane<LANES;lane++)
            bank[ctx][lane]=0;
        for(integer lane=0;lane<LANES;lane++)bank[0][lane]=(1<<(ACC_BITS-1))-1;
        mask_by_source[1]=8'h01;sign_by_source[1]=0;start(9);
        @(negedge clk_core);descriptor_valid=1;descriptor_source=1;
        descriptor_context_mask=1;descriptor_sign_mask=0;descriptor_last=1;
        do @(posedge clk_core);while(!descriptor_ready);
        @(negedge clk_core);descriptor_valid=0;
        do @(negedge clk_core);while(!numeric_overflow);
        @(posedge clk_core);#0.2;
        if(!protocol_error||acc_update_accept)
            $fatal(1,"M229 overflow not quarantined");
        attacks++;reset();
    end endtask
    initial begin #2000000;$fatal(1,"M229 watchdog F=%0d",FANOUT);end
    initial begin rst_core=1;cycles=0;groups=0;total_desc=0;total_updates=0;
        total_overlaps=0;attacks=0;req_stalls=0;upd_stalls=0;
        clean_group_mode=1;server_enabled=1;
        clear();repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        run(0,8,1);run(1,18,1);run(2,38,0);bad_descriptor();
        bad_response();overflow_attack();
        if(numeric_overflow)$fatal(1,"M229 unexpected overflow");
        $display("PASS M229 F=%0d groups=%0d desc=%0d updates=%0d attacks=%0d overlaps=%0d req_stalls=%0d upd_stalls=%0d cycles=%0d,%0d,%0d",
            FANOUT,groups,total_desc,total_updates,attacks,total_overlaps,
            req_stalls,upd_stalls,group_cycles[0],group_cycles[1],group_cycles[2]);
        $finish;end
endmodule
`default_nettype wire
