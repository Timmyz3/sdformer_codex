`timescale 1ns/1ps
`default_nettype none

`ifndef M227_FANOUT
`define M227_FANOUT 1
`endif

module tb_m227_fc1_k8_masked_held_weight_slice;
    localparam int FANOUT=`M227_FANOUT, TAG_BITS=24, EPOCH_BITS=16;
    localparam int LANES=96, ACC_BITS=19, CHANNELS=384, CONTEXTS=8;
    logic clk_core=0, rst_core;
    always #1.5 clk_core=~clk_core;

    logic begin_valid,begin_ready,begin_accept;
    logic[TAG_BITS-1:0]begin_tag;logic[EPOCH_BITS-1:0]begin_epoch;
    logic scan_valid,scan_ready,scan_beat,scan_accept;
    logic[2:0]scan_context;logic[255:0]scan_presence,scan_sign;
    logic scan_done_valid,scan_done_ready,scan_done_accept;
    logic weight_req_valid,weight_req_ready,weight_req_accept;
    logic[TAG_BITS-1:0]weight_req_tag;logic[EPOCH_BITS-1:0]weight_req_epoch;
    logic[8:0]weight_req_source;
    logic weight_rsp_valid,weight_rsp_ready,weight_rsp_accept;
    logic[TAG_BITS-1:0]weight_rsp_tag;logic[EPOCH_BITS-1:0]weight_rsp_epoch;
    logic[8:0]weight_rsp_source;logic[LANES*8-1:0]weight_rsp_data;
    logic result_valid,result_ready,result_last,result_accept;
    logic[TAG_BITS-1:0]result_tag;logic[EPOCH_BITS-1:0]result_epoch;
    logic[2:0]result_context;
    logic[LANES*ACC_BITS-1:0]result_accumulator;
    logic done_valid,done_ready,done_accept;
    logic[TAG_BITS-1:0]done_tag;logic[EPOCH_BITS-1:0]done_epoch;
    logic protocol_error,numeric_overflow,busy;
    logic[4:0]debug_scan_count;logic[9:0]debug_unique_sources;
    logic[11:0]debug_context_updates;logic[9:0]debug_weight_reads;
    logic[2:0]replay_width;

    logic[CHANNELS-1:0]model_presence[0:CONTEXTS-1];
    logic[CHANNELS-1:0]model_sign[0:CONTEXTS-1];
    integer signed expected[0:CONTEXTS-1][0:LANES-1];
    integer cycle_count,clean_groups,clean_sources,clean_updates;
    integer clean_results,protocol_attacks,request_stalls,result_stalls;
    integer signed_groups,tail_groups,empty_groups,full8_sources;
    integer group_start_cycle,group_cycles[0:2];
    logic automatic_response;

    assign replay_width = dut.selected_context_valid[0]
        + dut.selected_context_valid[1] + dut.selected_context_valid[2]
        + dut.selected_context_valid[3];

    m227_fc1_k8_masked_held_weight_slice #(.FANOUT(FANOUT)) dut(.*);
    m227_fc1_k8_masked_held_weight_slice_assertions #(.FANOUT(FANOUT)) sva(.*);

    function automatic integer signed weight_value(
        input integer source,input integer lane);
        integer value;
        begin
            case(lane%11)
                0:value=-128;
                1:value=127;
                default:begin
                    value=(source*17+lane*29+13)%255;
                    value=value-127;
                end
            endcase
            return value;
        end
    endfunction

    function automatic logic[LANES*8-1:0] weight_vector(
        input integer source);
        logic[LANES*8-1:0] value;
        begin
            value='0;
            for(integer lane=0;lane<LANES;lane++)
                value[lane*8 +: 8]=weight_value(source,lane);
            return value;
        end
    endfunction

    always @(posedge clk_core) begin
        if(rst_core)cycle_count=0;
        else begin
            cycle_count++;
            if(weight_req_valid&&!weight_req_ready)request_stalls++;
            if(result_valid&&!result_ready)result_stalls++;
        end
    end

    always @(negedge clk_core) begin
        if(rst_core)begin
            weight_req_ready=0;result_ready=0;
        end else begin
            weight_req_ready=(cycle_count%5)!=2;
            result_ready=(cycle_count%7)!=3 && (cycle_count%7)!=4;
        end
    end

    task automatic memory_server;
        logic[TAG_BITS-1:0]saved_tag;
        logic[EPOCH_BITS-1:0]saved_epoch;
        logic[8:0]saved_source;
        begin
            forever begin
                @(posedge clk_core);
                if(!rst_core&&weight_req_accept&&automatic_response)begin
                    saved_tag=weight_req_tag;saved_epoch=weight_req_epoch;
                    saved_source=weight_req_source;
                    repeat((saved_source%3)+1)@(posedge clk_core);
                    @(negedge clk_core);
                    weight_rsp_valid=1;
                    weight_rsp_tag=saved_tag;weight_rsp_epoch=saved_epoch;
                    weight_rsp_source=saved_source;
                    weight_rsp_data=weight_vector(saved_source);
                    do @(posedge clk_core); while(!weight_rsp_ready);
                    @(negedge clk_core);weight_rsp_valid=0;
                end
            end
        end
    endtask

    task automatic clear_drives;
        begin
            begin_valid=0;begin_tag=0;begin_epoch=0;
            scan_valid=0;scan_context=0;scan_beat=0;
            scan_presence=0;scan_sign=0;scan_done_valid=0;
            weight_rsp_valid=0;weight_rsp_tag=0;weight_rsp_epoch=0;
            weight_rsp_source=0;weight_rsp_data=0;done_ready=1;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);clear_drives();rst_core=1;
            repeat(4)@(posedge clk_core);
            @(negedge clk_core);rst_core=0;
        end
    endtask

    task automatic build_case(input integer case_id);
        integer source_count;
        begin
            for(integer ctx=0;ctx<CONTEXTS;ctx++)begin
                model_presence[ctx]='0;model_sign[ctx]='0;
            end
            if(case_id==1)begin
                for(integer ctx=0;ctx<CONTEXTS;ctx++)begin
                    model_presence[ctx][0]=1;
                    model_sign[ctx][0]=ctx[0];
                    model_presence[ctx][ctx+1]=1;
                    model_sign[ctx][ctx+1]=(ctx%3)==0;
                    model_presence[ctx][32+ctx*2]=1;
                    model_sign[ctx][32+ctx*2]=(ctx%2)==0;
                    model_presence[ctx][255]=1;
                    model_sign[ctx][255]=ctx[0];
                    if(ctx<4)begin
                        model_presence[ctx][383]=1;
                        model_sign[ctx][383]=ctx[0];
                    end
                end
            end else if(case_id==2)begin
                for(integer source=0;source<18;source++)begin
                    source_count=(source%8)+1;
                    for(integer ctx=0;ctx<source_count;ctx++)begin
                        model_presence[ctx][source*7]=1;
                        model_sign[ctx][source*7]=(source+ctx)%3==0;
                    end
                end
                for(integer ctx=0;ctx<CONTEXTS;ctx++)begin
                    model_presence[ctx][383]=1;
                    model_sign[ctx][383]=(ctx%2)==0;
                end
            end
            for(integer ctx=0;ctx<CONTEXTS;ctx++)
                for(integer lane=0;lane<LANES;lane++)begin
                    expected[ctx][lane]=0;
                    for(integer source=0;source<CHANNELS;source++)
                        if(model_presence[ctx][source])begin
                            if(model_sign[ctx][source])
                                expected[ctx][lane]-=weight_value(source,lane);
                            else expected[ctx][lane]+=weight_value(source,lane);
                        end
                end
        end
    endtask

    task automatic send_begin(input integer group_index);
        begin
            @(negedge clk_core);begin_valid=1;
            begin_tag=24'h227000+group_index;
            begin_epoch=16'h2200+group_index;
            do @(posedge clk_core);while(!begin_ready);
            @(negedge clk_core);begin_valid=0;
            group_start_cycle=cycle_count;
        end
    endtask

    task automatic send_scan(input integer ctx,input integer beat);
        begin
            @(negedge clk_core);scan_valid=1;scan_context=ctx;
            scan_beat=beat;
            if(beat==0)begin
                scan_presence=model_presence[ctx][255:0];
                scan_sign=model_sign[ctx][255:0];
            end else begin
                scan_presence='0;scan_sign='0;
                scan_presence[127:0]=model_presence[ctx][383:256];
                scan_sign[127:0]=model_sign[ctx][383:256];
            end
            do @(posedge clk_core);while(!scan_ready);
            @(negedge clk_core);scan_valid=0;
        end
    endtask

    task automatic run_group(input integer group_index);
        integer expected_sources,expected_updates,result_count;
        logic[CHANNELS-1:0]union_sources;
        begin
            build_case(group_index);send_begin(group_index);
            for(integer ctx=0;ctx<CONTEXTS;ctx++)begin
                send_scan(ctx,0);send_scan(ctx,1);
            end
            @(negedge clk_core);scan_done_valid=1;
            do @(posedge clk_core);while(!scan_done_ready);
            @(negedge clk_core);scan_done_valid=0;
            result_count=0;
            while(result_count<CONTEXTS)begin
                @(posedge clk_core);
                if(result_accept)begin
                    if(result_tag!==24'h227000+group_index
                            ||result_epoch!==16'h2200+group_index
                            ||result_context!==result_count[2:0])
                        $fatal(1,"M227 result identity mismatch group=%0d ctx=%0d",
                            group_index,result_count);
                    for(integer lane=0;lane<LANES;lane++)begin
                        logic signed[ACC_BITS-1:0]observed;
                        observed=result_accumulator[lane*ACC_BITS +: ACC_BITS];
                        if($signed(observed)!==expected[result_count][lane])
                            $fatal(1,"M227 numeric mismatch F=%0d group=%0d ctx=%0d lane=%0d exp=%0d got=%0d",
                                FANOUT,group_index,result_count,lane,
                                expected[result_count][lane],$signed(observed));
                    end
                    result_count++;
                end
            end
            do @(posedge clk_core);while(!done_accept);
            if(done_tag!==24'h227000+group_index
                    ||done_epoch!==16'h2200+group_index)
                $fatal(1,"M227 done identity mismatch");
            union_sources='0;expected_updates=0;
            for(integer ctx=0;ctx<CONTEXTS;ctx++)begin
                union_sources|=model_presence[ctx];
                for(integer source=0;source<CHANNELS;source++)
                    expected_updates+=model_presence[ctx][source];
            end
            expected_sources=0;
            for(integer source=0;source<CHANNELS;source++)
                expected_sources+=union_sources[source];
            if(debug_unique_sources!=expected_sources
                    ||debug_weight_reads!=expected_sources
                    ||debug_context_updates!=expected_updates)
                $fatal(1,"M227 conservation mismatch F=%0d group=%0d src exp/got=%0d/%0d upd=%0d/%0d",
                    FANOUT,group_index,expected_sources,debug_unique_sources,
                    expected_updates,debug_context_updates);
            group_cycles[group_index]=cycle_count-group_start_cycle;
            clean_groups++;clean_sources+=expected_sources;
            clean_updates+=expected_updates;clean_results+=result_count;
            if(group_index==0)empty_groups++;
            else begin signed_groups++;tail_groups++;end
            if(group_index>0)full8_sources++;
        end
    endtask

    task automatic duplicate_scan_attack;
        begin
            reset_dut();build_case(1);send_begin(7);send_scan(0,0);
            @(negedge clk_core);scan_valid=1;scan_context=0;scan_beat=0;
            scan_presence=model_presence[0][255:0];
            scan_sign=model_sign[0][255:0];#0.1;
            if(scan_ready)$fatal(1,"M227 duplicate scan became ready");
            @(posedge clk_core);#0.2;
            if(!protocol_error)$fatal(1,"M227 duplicate scan not fail-closed");
            protocol_attacks++;@(negedge clk_core);scan_valid=0;
        end
    endtask

    task automatic tail_attack;
        begin
            reset_dut();send_begin(8);
            @(negedge clk_core);scan_valid=1;scan_context=0;scan_beat=1;
            scan_presence='0;scan_sign='0;scan_presence[200]=1;#0.1;
            if(scan_ready)$fatal(1,"M227 illegal tail became ready");
            @(posedge clk_core);#0.2;
            if(!protocol_error)$fatal(1,"M227 illegal tail not fail-closed");
            protocol_attacks++;@(negedge clk_core);scan_valid=0;
        end
    endtask

    initial begin
        #2000000;$fatal(1,"M227 watchdog F=%0d cycle=%0d",FANOUT,cycle_count);
    end

    initial begin
        rst_core=1;cycle_count=0;clean_groups=0;clean_sources=0;
        clean_updates=0;clean_results=0;protocol_attacks=0;
        request_stalls=0;result_stalls=0;signed_groups=0;tail_groups=0;
        empty_groups=0;full8_sources=0;automatic_response=1;clear_drives();
        fork memory_server(); join_none
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        run_group(0);run_group(1);run_group(2);
        duplicate_scan_attack();tail_attack();
        if(numeric_overflow)$fatal(1,"M227 unexpected numeric overflow");
        $display("PASS M227 F=%0d clean_groups=%0d sources=%0d updates=%0d results=%0d protocol_attacks=%0d empty=%0d signed=%0d tail=%0d full8=%0d request_stalls=%0d result_stalls=%0d cycles=%0d,%0d,%0d",
            FANOUT,clean_groups,clean_sources,clean_updates,clean_results,
            protocol_attacks,empty_groups,signed_groups,tail_groups,
            full8_sources,request_stalls,result_stalls,
            group_cycles[0],group_cycles[1],group_cycles[2]);
        $finish;
    end
endmodule

`default_nettype wire
