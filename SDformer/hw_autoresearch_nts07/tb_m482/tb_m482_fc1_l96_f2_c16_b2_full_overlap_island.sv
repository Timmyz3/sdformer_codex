`timescale 1ns/1ps
`default_nettype none
module tb_m482_fc1_l96_f2_c16_b2_full_overlap_island;
    localparam int TAG_BITS=24,EPOCH_BITS=16,DESC_BITS=12;
    localparam int LANES=96,ACC_BITS=19,CHUNKS=24;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic header_valid,header_ready,header_accept;logic[23:0]header_tag;
    logic[15:0]header_epoch;logic header_factorized;
    logic[4:0]header_chunk_count;logic[DESC_BITS-1:0]header_descriptor_count;
    logic directory_valid,directory_ready,directory_accept;
    logic[4:0]directory_chunk;logic[7:0]directory_descriptor_count;
    logic factor_req_valid,factor_req_ready,factor_req_accept;
    logic[1:0]factor_req_slot;logic[23:0]factor_req_tag;
    logic[15:0]factor_req_epoch;logic[DESC_BITS-1:0]factor_req_descriptor;
    logic[4:0]factor_req_chunk;logic[6:0]factor_req_ordinal;
    logic factor_rsp_valid,factor_rsp_ready,factor_rsp_accept;
    logic[1:0]factor_rsp_slot;logic[23:0]factor_rsp_tag;
    logic[15:0]factor_rsp_epoch;logic[DESC_BITS-1:0]factor_rsp_descriptor;
    logic[4:0]factor_rsp_chunk;logic[6:0]factor_rsp_ordinal;
    logic[3:0]factor_rsp_source_offset;logic[7:0]factor_rsp_context_mask;
    logic[7:0]factor_rsp_sign_mask;
    logic weight_req_valid,weight_req_ready,weight_req_accept;
    logic[1:0]weight_req_slot;logic[23:0]weight_req_tag;
    logic[15:0]weight_req_epoch;logic[8:0]weight_req_source;
    logic weight_rsp_valid,weight_rsp_ready,weight_rsp_accept;
    logic[1:0]weight_rsp_slot;logic[23:0]weight_rsp_tag;
    logic[15:0]weight_rsp_epoch;logic[8:0]weight_rsp_source;
    logic[LANES*8-1:0]weight_rsp_data;
    logic[1:0]acc_bank_read_valid,acc_bank_read_ready;
    logic[1:0][1:0]acc_bank_read_row;
    logic[1:0][2:0]acc_bank_read_context;logic acc_issue_accept;
    logic commit_valid,commit_ready,commit_accept;logic[23:0]commit_tag;
    logic[15:0]commit_epoch;logic[2:0]commit_context;logic commit_last;
    logic[LANES*ACC_BITS-1:0]commit_data;
    logic done_valid,done_ready,done_accept;logic[23:0]done_tag;
    logic[15:0]done_epoch;logic done_empty_bypass;
    logic protocol_error,numeric_overflow,busy;logic[31:0]debug_tile_cycles;
    logic[31:0]debug_factor_requests,debug_weight_requests;
    logic[31:0]debug_issue_rounds,debug_context_updates;
    logic[31:0]debug_bank_conflict_extra_rounds;
    logic[31:0]debug_factor_weight_overlap,debug_weight_update_overlap;
    logic[31:0]debug_triple_overlap,debug_same_bank_rdw;
    logic[31:0]debug_same_address_forward;logic[2:0]debug_credit_count;

    logic[7:0]dir_count[0:CHUNKS-1];
    logic[3:0]factor_offset[0:CHUNKS-1][0:127];
    logic[7:0]factor_mask[0:CHUNKS-1][0:127];
    logic[7:0]factor_sign[0:CHUNKS-1][0:127];
    integer signed expected[0:7][0:LANES-1];
    integer cycles,groups,total_attacks,total_empty,total_commits;
    integer factor_stalls,weight_stalls,bank_stalls,commit_stalls;
    integer latency_checks,allmask_factor_cycles,allmask_sparse_cycles;
    integer allmask_factor_rounds,allmask_sparse_rounds;
    integer clean_cycle_rows;
    integer factor_queue_latency_accepted,weight_queue_latency_accepted;
    logic stall_enable,auto_factor,auto_weight;
    logic factor_consumed,weight_consumed;

    typedef struct packed {integer due;integer accepted;
        logic[1:0]slot;logic[23:0]tag;logic[15:0]epoch;
        logic[DESC_BITS-1:0]descriptor;logic[4:0]chunk;
        logic[6:0]ordinal;} factor_item_t;
    typedef struct packed {integer due;integer accepted;
        logic[1:0]slot;logic[23:0]tag;logic[15:0]epoch;
        logic[8:0]source;} weight_item_t;
    factor_item_t factor_queue[$];weight_item_t weight_queue[$];

    m482_fc1_l96_f2_c16_b2_full_overlap_island dut(.*);
    m482_fc1_l96_f2_c16_b2_full_overlap_assertions sva(.*);

    function automatic integer signed wval(input integer source,input integer lane);
        integer value;begin
            value=(source*17+lane*29+11)%63;value-=31;return value;
        end
    endfunction
    function automatic logic[LANES*8-1:0]wvec(input integer source);
        logic[LANES*8-1:0]value;begin value='0;
            for(integer lane=0;lane<LANES;lane++)
                value[lane*8+:8]=wval(source,lane);return value;end
    endfunction
    function automatic integer popcount8(input logic[7:0]value);
        integer count;begin count=0;
            for(integer bit_index=0;bit_index<8;bit_index++)
                count+=value[bit_index];return count;end
    endfunction
    function automatic integer bank_rounds(input logic[7:0]value);
        integer even_count,odd_count;begin even_count=0;odd_count=0;
            for(integer ctx=0;ctx<8;ctx++)if(value[ctx])begin
                if(ctx%2)odd_count++;else even_count++;end
            return even_count>odd_count?even_count:odd_count;end
    endfunction

    always @(posedge clk_core)begin
        if(rst_core)cycles=0;
        else begin
            cycles++;
            if(protocol_error&&auto_factor)
                $fatal(1,"M482 unexpected fault state=%0d ih=%0d id=%0d if=%0d iw=%0d dir=%0d/%0d sum=%0d/%0d frsp=%0d slot=%0d d=%0d/%0d c=%0d/%0d o=%0d/%0d sv=%0d fv=%0d tag=%h/%h ep=%h/%h off=%0d mask=%h sign=%h fact=%0d",
                    dut.state_q,dut.illegal_header,dut.illegal_directory,
                    dut.illegal_factor_response,dut.illegal_weight_response,
                    dut.directory_index_q,dut.chunk_count_q,
                    dut.directory_sum_q,dut.descriptor_count_q,
                    factor_rsp_valid,factor_rsp_slot,factor_rsp_descriptor,
                    dut.slot_descriptor_q[factor_rsp_slot],factor_rsp_chunk,
                    dut.slot_chunk_q[factor_rsp_slot],factor_rsp_ordinal,
                    dut.slot_ordinal_q[factor_rsp_slot],
                    dut.slot_valid_q[factor_rsp_slot],
                    dut.slot_factor_valid_q[factor_rsp_slot],factor_rsp_tag,
                    dut.tag_q,factor_rsp_epoch,dut.epoch_q,
                    factor_rsp_source_offset,factor_rsp_context_mask,
                    factor_rsp_sign_mask,dut.factorized_q);
            if(factor_req_valid&&!factor_req_ready)factor_stalls++;
            if(weight_req_valid&&!weight_req_ready)weight_stalls++;
            if((|acc_bank_read_valid)&&!acc_issue_accept)bank_stalls++;
            if(commit_valid&&!commit_ready)commit_stalls++;
            if(factor_req_accept&&auto_factor)begin factor_item_t item;
                item.due=cycles+1;item.accepted=cycles;
                item.slot=factor_req_slot;item.tag=factor_req_tag;
                item.epoch=factor_req_epoch;item.descriptor=factor_req_descriptor;
                item.chunk=factor_req_chunk;item.ordinal=factor_req_ordinal;
                factor_queue.push_back(item);
            end
            if(weight_req_accept&&auto_weight)begin weight_item_t item;
                item.due=cycles+1;item.accepted=cycles;
                item.slot=weight_req_slot;item.tag=weight_req_tag;
                item.epoch=weight_req_epoch;item.source=weight_req_source;
                weight_queue.push_back(item);
            end
            if(factor_rsp_accept)begin
                if(cycles-factor_queue_latency_accepted!=2)
                    $fatal(1,"M482 factor latency expected2 observed=%0d",
                        cycles-factor_queue_latency_accepted);
                latency_checks++;factor_consumed=1;
            end
            if(weight_rsp_accept)begin
                if(cycles-weight_queue_latency_accepted!=2)
                    $fatal(1,"M482 weight latency expected2 observed=%0d",
                        cycles-weight_queue_latency_accepted);
                latency_checks++;weight_consumed=1;
            end
            if(commit_accept)begin
                for(integer lane=0;lane<LANES;lane++)begin
                    logic signed[ACC_BITS-1:0]observed;
                    observed=commit_data[lane*ACC_BITS+:ACC_BITS];
                    if($signed(observed)!==expected[commit_context][lane])
                        $fatal(1,"M482 commit mismatch c=%0d l=%0d e=%0d o=%0d",
                            commit_context,lane,expected[commit_context][lane],
                            $signed(observed));
                end
                if(commit_last!=(commit_context==7))
                    $fatal(1,"M482 commit last mismatch");
                total_commits++;
            end
        end
    end

    always @(negedge clk_core)begin
        if(rst_core)begin
            factor_req_ready=0;weight_req_ready=0;acc_bank_read_ready=0;
            commit_ready=0;
        end else begin
            factor_req_ready=!stall_enable||(cycles%7)!=2;
            weight_req_ready=!stall_enable||(cycles%5)!=1;
            acc_bank_read_ready[0]=!stall_enable||(cycles%6)!=3;
            acc_bank_read_ready[1]=!stall_enable||(cycles%8)!=4;
            commit_ready=!stall_enable||(cycles%4)!=1;
        end
        if(factor_consumed)begin factor_rsp_valid=0;factor_consumed=0;end
        if(weight_consumed)begin weight_rsp_valid=0;weight_consumed=0;end
        if(auto_factor&&!factor_rsp_valid&&factor_queue.size()>0
                &&factor_queue[0].due<=cycles)begin factor_item_t item;
            item=factor_queue.pop_front();factor_rsp_valid=1;
            factor_rsp_slot=item.slot;factor_rsp_tag=item.tag;
            factor_rsp_epoch=item.epoch;factor_rsp_descriptor=item.descriptor;
            factor_rsp_chunk=item.chunk;factor_rsp_ordinal=item.ordinal;
            factor_rsp_source_offset=factor_offset[item.chunk][item.ordinal];
            factor_rsp_context_mask=factor_mask[item.chunk][item.ordinal];
            factor_rsp_sign_mask=factor_sign[item.chunk][item.ordinal];
            factor_queue_latency_accepted=item.accepted;
        end
        if(auto_weight&&!weight_rsp_valid&&weight_queue.size()>0
                &&weight_queue[0].due<=cycles)begin weight_item_t item;
            item=weight_queue.pop_front();weight_rsp_valid=1;
            weight_rsp_slot=item.slot;weight_rsp_tag=item.tag;
            weight_rsp_epoch=item.epoch;weight_rsp_source=item.source;
            weight_rsp_data=wvec(item.source);
            weight_queue_latency_accepted=item.accepted;
        end
    end

    task automatic clear_inputs;begin
        header_valid=0;header_tag=0;header_epoch=0;header_factorized=0;
        header_chunk_count=0;header_descriptor_count=0;
        directory_valid=0;directory_chunk=0;directory_descriptor_count=0;
        factor_rsp_valid=0;factor_rsp_slot=0;factor_rsp_tag=0;
        factor_rsp_epoch=0;factor_rsp_descriptor=0;factor_rsp_chunk=0;
        factor_rsp_ordinal=0;factor_rsp_source_offset=0;
        factor_rsp_context_mask=0;factor_rsp_sign_mask=0;
        weight_rsp_valid=0;weight_rsp_slot=0;weight_rsp_tag=0;
        weight_rsp_epoch=0;weight_rsp_source=0;weight_rsp_data=0;
        done_ready=1;factor_consumed=0;weight_consumed=0;
        factor_queue.delete();weight_queue.delete();stall_enable=0;
        auto_factor=1;auto_weight=1;
    end endtask
    task automatic reset;begin
        @(negedge clk_core);clear_inputs();rst_core=1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
    end endtask

    task automatic clear_workload;begin
        for(integer chunk=0;chunk<CHUNKS;chunk++)begin dir_count[chunk]=0;
            for(integer ordinal=0;ordinal<128;ordinal++)begin
                factor_offset[chunk][ordinal]=0;
                factor_mask[chunk][ordinal]=0;
                factor_sign[chunk][ordinal]=0;
            end
        end
        for(integer ctx=0;ctx<8;ctx++)for(integer lane=0;lane<LANES;lane++)
            expected[ctx][lane]=0;
    end endtask

    task automatic append_factor(input integer source,input logic[7:0]mask,
            input logic[7:0]sign);integer chunk,ordinal;begin
        chunk=source/16;ordinal=dir_count[chunk];
        if(ordinal>=128)$fatal(1,"M482 directory overflow chunk=%0d",chunk);
        factor_offset[chunk][ordinal]=source%16;
        factor_mask[chunk][ordinal]=mask;factor_sign[chunk][ordinal]=sign;
        dir_count[chunk]++;
    end endtask

    task automatic build_workload(input logic factorized,input integer sources,
            input integer pattern,output integer descriptors,
            output integer rounds,output integer updates);begin
        logic[7:0]mask,sign;clear_workload();descriptors=0;rounds=0;updates=0;
        for(integer source=0;source<sources;source++)begin
            case(pattern)
                1:mask=(source+1)&8'hff;
                2:mask=8'h01;
                3:mask=((source*73+151)%255)+1;
                default:begin mask=(source*37+8'h35)&8'hff;
                    if(mask==0)mask=8'h81;end
            endcase
            sign=(source*19+8'h29)&mask;
            for(integer ctx=0;ctx<8;ctx++)if(mask[ctx])begin
                updates++;
                for(integer lane=0;lane<LANES;lane++)
                    expected[ctx][lane]+=sign[ctx]
                        ?-wval(source,lane):wval(source,lane);
                if(!factorized)begin
                    append_factor(source,8'h1<<ctx,sign[ctx]?(8'h1<<ctx):0);
                    descriptors++;rounds++;
                end
            end
            if(factorized)begin append_factor(source,mask,sign);
                descriptors++;rounds+=bank_rounds(mask);end
        end
    end endtask

    task automatic send_header(input logic factorized,input integer chunks,
            input integer descriptors,input integer identity);begin
        @(negedge clk_core);header_valid=1;header_factorized=factorized;
        header_chunk_count=chunks;header_descriptor_count=descriptors;
        header_tag=24'h482000+identity;header_epoch=16'h4820+identity;
        do @(posedge clk_core);while(!header_accept);
        @(negedge clk_core);header_valid=0;
    end endtask

    task automatic send_directory(input integer chunks);begin
        for(integer chunk=0;chunk<chunks;chunk++)begin
            @(negedge clk_core);directory_valid=1;directory_chunk=chunk;
            directory_descriptor_count=dir_count[chunk];
            do @(posedge clk_core);while(!directory_accept);
        end
        @(negedge clk_core);directory_valid=0;
    end endtask

    task automatic run_group(input logic factorized,input integer sources,
            input integer pattern,input logic stalls,input integer identity,
            output integer tile_cycles,output integer expected_rounds);integer descriptors;
        integer updates,chunks,commit_before;begin
        build_workload(factorized,sources,pattern,descriptors,expected_rounds,updates);
        chunks=(sources+15)/16;stall_enable=stalls;commit_before=total_commits;
        send_header(factorized,chunks,descriptors,identity);send_directory(chunks);
        do @(posedge clk_core);while(!done_accept);#0.2;
        tile_cycles=debug_tile_cycles;
        if(debug_factor_requests!=descriptors
                ||debug_weight_requests!=descriptors)
            $fatal(1,"M482 descriptor conservation f=%0d d=%0d fr=%0d wr=%0d",
                factorized,descriptors,debug_factor_requests,debug_weight_requests);
        if(debug_issue_rounds!=expected_rounds||debug_context_updates!=updates)
            $fatal(1,"M482 update conservation f=%0d r=%0d/%0d u=%0d/%0d",
                factorized,debug_issue_rounds,expected_rounds,
                debug_context_updates,updates);
        if(!stalls&&tile_cycles!=expected_rounds+chunks+39)
            $fatal(1,"M482 clean recurrence f=%0d cycles=%0d expected=%0d rounds=%0d chunks=%0d",
                factorized,tile_cycles,expected_rounds+chunks+39,
                expected_rounds,chunks);
        if(total_commits-commit_before!=8)$fatal(1,"M482 commit count");
        if(numeric_overflow||protocol_error)$fatal(1,"M482 clean fault");
        if(!stalls&&descriptors>16&&debug_factor_weight_overlap==0)
            $fatal(1,"M482 no factor/weight overlap");
        if(!stalls&&descriptors>16&&debug_weight_update_overlap==0)
            $fatal(1,"M482 no weight/update overlap");
        if(!stalls&&descriptors>16&&debug_triple_overlap==0)
            $fatal(1,"M482 no triple overlap");
        groups++;clean_cycle_rows+=!stalls;
        @(negedge clk_core);stall_enable=0;
    end endtask

    task automatic run_empty;begin
        clear_workload();send_header(1,0,0,90);
        do @(posedge clk_core);while(!done_accept);#0.2;
        if(!done_empty_bypass||debug_tile_cycles!=2)
            $fatal(1,"M482 empty bypass mismatch cycles=%0d",debug_tile_cycles);
        total_empty++;
    end endtask

    task automatic protocol_attack;integer descriptors,rounds,updates;begin
        reset();build_workload(1,4,0,descriptors,rounds,updates);
        auto_factor=0;send_header(1,1,descriptors,99);send_directory(1);
        do @(posedge clk_core);while(!factor_req_accept);
        @(negedge clk_core);factor_rsp_valid=1;
        factor_rsp_slot=factor_req_slot;factor_rsp_tag=24'hbadbad;
        factor_rsp_epoch=factor_req_epoch;
        factor_rsp_descriptor=factor_req_descriptor;
        factor_rsp_chunk=factor_req_chunk;factor_rsp_ordinal=factor_req_ordinal;
        factor_rsp_source_offset=factor_offset[factor_req_chunk][factor_req_ordinal];
        factor_rsp_context_mask=factor_mask[factor_req_chunk][factor_req_ordinal];
        factor_rsp_sign_mask=factor_sign[factor_req_chunk][factor_req_ordinal];
        @(posedge clk_core);#0.2;
        if(!protocol_error||factor_rsp_accept)
            $fatal(1,"M482 wrong factor response not quarantined");
        total_attacks++;@(negedge clk_core);factor_rsp_valid=0;
    end endtask

    initial begin #10000000;
        $fatal(1,"M482 watchdog state=%0d issued=%0d rsp=%0d inflight=%0d fq=%0d slots=%0d head=%0d tail=%0d hp=%h hv=%0d desc=%0d",
            dut.state_q,dut.descriptor_issue_q,dut.factor_response_q,
            dut.factor_inflight_q,dut.factor_fifo_count_q,dut.slot_count_q,
            dut.head_q,dut.tail_q,dut.head_pending,
            dut.slot_weight_valid_q[dut.head_q],dut.descriptor_count_q);
    end
    initial begin
        integer factor_cycles,sparse_cycles,rounds0,rounds1;
        rst_core=1;cycles=0;groups=0;total_attacks=0;total_empty=0;
        total_commits=0;factor_stalls=0;weight_stalls=0;bank_stalls=0;
        commit_stalls=0;latency_checks=0;clean_cycle_rows=0;
        allmask_factor_cycles=0;allmask_sparse_cycles=0;
        allmask_factor_rounds=0;allmask_sparse_rounds=0;clear_inputs();
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        run_group(1,64,0,0,0,factor_cycles,rounds0);
        run_group(0,64,0,0,1,sparse_cycles,rounds1);
        run_group(1,255,1,0,2,allmask_factor_cycles,allmask_factor_rounds);
        run_group(0,255,1,0,3,allmask_sparse_cycles,allmask_sparse_rounds);
        run_group(1,16,2,0,4,factor_cycles,rounds0);
        if(debug_same_address_forward==0||debug_same_bank_rdw==0)
            $fatal(1,"M482 RDW forwarding not covered");
        run_group(1,48,0,1,5,factor_cycles,rounds0);
        run_group(1,127,3,0,6,factor_cycles,rounds0);
        run_group(0,31,3,0,7,sparse_cycles,rounds1);
        run_empty();protocol_attack();
        $display("PASS M482 groups=%0d all255_factor_cycles=%0d all255_sparse_cycles=%0d all255_ratio=%0f factor_rounds=%0d sparse_rounds=%0d attacks=%0d empty=%0d latency_checks=%0d stalls=%0d,%0d,%0d,%0d commits=%0d",
            groups,allmask_factor_cycles,allmask_sparse_cycles,
            real'(allmask_sparse_cycles)/real'(allmask_factor_cycles),
            allmask_factor_rounds,allmask_sparse_rounds,total_attacks,total_empty,
            latency_checks,factor_stalls,weight_stalls,bank_stalls,commit_stalls,
            total_commits);
        $finish;
    end
endmodule
`default_nettype wire
