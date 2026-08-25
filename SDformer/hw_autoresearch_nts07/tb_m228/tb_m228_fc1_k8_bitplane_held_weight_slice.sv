`timescale 1ns/1ps
`default_nettype none
`ifndef M228_FANOUT
`define M228_FANOUT 1
`endif
module tb_m228_fc1_k8_bitplane_held_weight_slice;
    localparam int FANOUT=`M228_FANOUT,TAG_BITS=24,EPOCH_BITS=16;
    localparam int LANES=96,ACC_BITS=19;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic begin_valid,begin_ready,begin_accept;logic[TAG_BITS-1:0]begin_tag;
    logic[EPOCH_BITS-1:0]begin_epoch;logic scan_valid,scan_ready,scan_accept;
    logic[3:0]scan_chunk;logic[255:0]scan_presence,scan_sign;
    logic scan_done_valid,scan_done_ready,scan_done_accept;
    logic weight_req_valid,weight_req_ready,weight_req_accept;
    logic[TAG_BITS-1:0]weight_req_tag;logic[EPOCH_BITS-1:0]weight_req_epoch;
    logic[8:0]weight_req_source;logic weight_rsp_valid,weight_rsp_ready;
    logic weight_rsp_accept;logic[TAG_BITS-1:0]weight_rsp_tag;
    logic[EPOCH_BITS-1:0]weight_rsp_epoch;logic[8:0]weight_rsp_source;
    logic[LANES*8-1:0]weight_rsp_data;logic result_valid,result_ready;
    logic[TAG_BITS-1:0]result_tag;logic[EPOCH_BITS-1:0]result_epoch;
    logic[2:0]result_context;logic[LANES*ACC_BITS-1:0]result_accumulator;
    logic result_last,result_accept,done_valid,done_ready,done_accept;
    logic[TAG_BITS-1:0]done_tag;logic[EPOCH_BITS-1:0]done_epoch;
    logic protocol_error,numeric_overflow,busy;logic[3:0]debug_scan_count;
    logic[9:0]debug_unique_sources,debug_weight_reads;
    logic[11:0]debug_context_updates;logic[2:0]debug_replay_width;
    logic[383:0]model_presence[0:7],model_sign[0:7];
    integer signed expected[0:7][0:LANES-1];
    integer cycles,clean_groups,clean_sources,clean_updates,clean_results;
    integer attacks,req_stalls,out_stalls,group_cycles[0:2];

    m228_fc1_k8_bitplane_held_weight_slice#(.FANOUT(FANOUT))dut(.*);
    m228_fc1_k8_bitplane_held_weight_slice_assertions#(.FANOUT(FANOUT))sva(.*);
    function automatic integer signed wval(input integer source,input integer lane);
        integer value;begin
            if(lane%11==0)value=-128;else if(lane%11==1)value=127;
            else begin value=(source*17+lane*29+13)%255;value-=127;end
            return value;
        end
    endfunction
    function automatic logic[LANES*8-1:0]wvec(input integer source);
        logic[LANES*8-1:0]value;begin value='0;
            for(integer lane=0;lane<LANES;lane++)value[lane*8+:8]=wval(source,lane);
            return value;
        end
    endfunction
    always @(posedge clk_core)if(rst_core)cycles=0;else begin
        cycles++;if(weight_req_valid&&!weight_req_ready)req_stalls++;
        if(result_valid&&!result_ready)out_stalls++;
    end
    always @(negedge clk_core)if(rst_core)begin weight_req_ready=0;result_ready=0;end
        else begin weight_req_ready=(cycles%5)!=2;
            result_ready=(cycles%7)!=3&&(cycles%7)!=4;end
    task automatic server;logic[23:0]tag;logic[15:0]epoch;logic[8:0]source;
        begin forever begin @(posedge clk_core);if(!rst_core&&weight_req_accept)begin
            tag=weight_req_tag;epoch=weight_req_epoch;source=weight_req_source;
            repeat((source%3)+1)@(posedge clk_core);@(negedge clk_core);
            weight_rsp_valid=1;weight_rsp_tag=tag;weight_rsp_epoch=epoch;
            weight_rsp_source=source;weight_rsp_data=wvec(source);
            do @(posedge clk_core);while(!weight_rsp_ready);
            @(negedge clk_core);weight_rsp_valid=0;
        end end end
    endtask
    task automatic clear;begin begin_valid=0;begin_tag=0;begin_epoch=0;
        scan_valid=0;scan_chunk=0;scan_presence=0;scan_sign=0;scan_done_valid=0;
        weight_rsp_valid=0;weight_rsp_tag=0;weight_rsp_epoch=0;
        weight_rsp_source=0;weight_rsp_data=0;done_ready=1;end endtask
    task automatic reset;begin @(negedge clk_core);clear();rst_core=1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;end endtask
    task automatic build(input integer case_id);integer count;begin
        for(integer ctx=0;ctx<8;ctx++)begin model_presence[ctx]='0;model_sign[ctx]='0;end
        if(case_id==1)for(integer ctx=0;ctx<8;ctx++)begin
            model_presence[ctx][0]=1;model_sign[ctx][0]=ctx[0];
            model_presence[ctx][ctx+1]=1;model_sign[ctx][ctx+1]=ctx%3==0;
            model_presence[ctx][32+ctx*2]=1;model_sign[ctx][32+ctx*2]=ctx%2==0;
            model_presence[ctx][255]=1;model_sign[ctx][255]=ctx[0];
            if(ctx<4)begin model_presence[ctx][383]=1;model_sign[ctx][383]=ctx[0];end
        end else if(case_id==2)begin
            for(integer source=0;source<18;source++)begin count=source%8+1;
                for(integer ctx=0;ctx<count;ctx++)begin
                    model_presence[ctx][source*7]=1;
                    model_sign[ctx][source*7]=(source+ctx)%3==0;
                end
            end
            for(integer ctx=0;ctx<8;ctx++)begin model_presence[ctx][383]=1;
                model_sign[ctx][383]=ctx%2==0;end
        end
        for(integer ctx=0;ctx<8;ctx++)for(integer lane=0;lane<LANES;lane++)begin
            expected[ctx][lane]=0;for(integer source=0;source<384;source++)
                if(model_presence[ctx][source])expected[ctx][lane]
                    +=model_sign[ctx][source]?-wval(source,lane):wval(source,lane);
        end
    end endtask
    task automatic start(input integer index);begin @(negedge clk_core);
        begin_valid=1;begin_tag=24'h228000+index;begin_epoch=16'h2280+index;
        do @(posedge clk_core);while(!begin_ready);@(negedge clk_core);begin_valid=0;
    end endtask
    task automatic scan_one(input integer chunk);begin @(negedge clk_core);
        scan_valid=1;scan_chunk=chunk;scan_presence='0;scan_sign='0;
        for(integer ctx=0;ctx<8;ctx++)begin
            scan_presence[ctx*32+:32]=model_presence[ctx][chunk*32+:32];
            scan_sign[ctx*32+:32]=model_sign[ctx][chunk*32+:32];
        end
        do @(posedge clk_core);while(!scan_ready);@(negedge clk_core);scan_valid=0;
    end endtask
    task automatic run(input integer index);integer start_cycle,results;
        integer exp_sources,exp_updates;logic[383:0]union_bits;begin
        build(index);start(index);start_cycle=cycles;
        for(integer chunk=0;chunk<12;chunk++)scan_one(chunk);
        @(negedge clk_core);scan_done_valid=1;
        do @(posedge clk_core);while(!scan_done_ready);
        @(negedge clk_core);scan_done_valid=0;results=0;
        while(results<8)begin @(posedge clk_core);if(result_accept)begin
            if(result_context!==results[2:0]||result_tag!==24'h228000+index)
                $fatal(1,"M228 result identity F=%0d g=%0d ctx=%0d",FANOUT,index,results);
            for(integer lane=0;lane<LANES;lane++)begin
                logic signed[ACC_BITS-1:0]observed;
                observed=result_accumulator[lane*ACC_BITS+:ACC_BITS];
                if($signed(observed)!==expected[results][lane])
                    $fatal(1,"M228 numeric F=%0d g=%0d ctx=%0d lane=%0d e=%0d o=%0d",
                        FANOUT,index,results,lane,expected[results][lane],$signed(observed));
            end results++;end end
        do @(posedge clk_core);while(!done_accept);
        union_bits='0;exp_updates=0;for(integer ctx=0;ctx<8;ctx++)begin
            union_bits|=model_presence[ctx];for(integer s=0;s<384;s++)
                exp_updates+=model_presence[ctx][s];end
        exp_sources=0;for(integer s=0;s<384;s++)exp_sources+=union_bits[s];
        if(debug_unique_sources!=exp_sources||debug_weight_reads!=exp_sources
                ||debug_context_updates!=exp_updates)
            $fatal(1,"M228 conservation F=%0d g=%0d s=%0d/%0d u=%0d/%0d",
                FANOUT,index,debug_unique_sources,exp_sources,
                debug_context_updates,exp_updates);
        group_cycles[index]=cycles-start_cycle;clean_groups++;clean_sources+=exp_sources;
        clean_updates+=exp_updates;clean_results+=results;
    end endtask
    task automatic attack_duplicate;begin reset();build(1);start(7);scan_one(0);
        @(negedge clk_core);scan_valid=1;scan_chunk=0;scan_presence='0;scan_sign='0;#0.1;
        if(scan_ready)$fatal(1,"M228 duplicate ready");@(posedge clk_core);#0.2;
        if(!protocol_error)$fatal(1,"M228 duplicate not fail closed");attacks++;
        @(negedge clk_core);scan_valid=0;end endtask
    task automatic attack_sign;begin reset();start(8);@(negedge clk_core);
        scan_valid=1;scan_chunk=0;scan_presence='0;scan_sign='0;scan_sign[3]=1;#0.1;
        if(scan_ready)$fatal(1,"M228 orphan sign ready");@(posedge clk_core);#0.2;
        if(!protocol_error)$fatal(1,"M228 orphan sign not fail closed");attacks++;
        @(negedge clk_core);scan_valid=0;end endtask
    initial begin #2000000;$fatal(1,"M228 watchdog F=%0d",FANOUT);end
    initial begin rst_core=1;cycles=0;clean_groups=0;clean_sources=0;clean_updates=0;
        clean_results=0;attacks=0;req_stalls=0;out_stalls=0;clear();fork server();join_none
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        run(0);run(1);run(2);attack_duplicate();attack_sign();
        if(numeric_overflow)$fatal(1,"M228 impossible overflow");
        $display("PASS M228 F=%0d groups=%0d sources=%0d updates=%0d results=%0d attacks=%0d req_stalls=%0d out_stalls=%0d cycles=%0d,%0d,%0d",
            FANOUT,clean_groups,clean_sources,clean_updates,clean_results,attacks,
            req_stalls,out_stalls,group_cycles[0],group_cycles[1],group_cycles[2]);
        $finish;end
endmodule
`default_nettype wire
