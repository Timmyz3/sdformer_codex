`timescale 1ns/1ps
`default_nettype none
module tb_m270_fc1_descriptor_lifecycle_wrapper;
    localparam int LANES=8,CONTEXTS=8,ACC_BITS=19,TAG_BITS=24;
    localparam int EPOCH_BITS=16,DESC_BITS=12,FACTOR_ADDR_BITS=20;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic header_valid,header_ready,header_accept;logic[23:0]header_tag;
    logic[15:0]header_epoch;logic[1:0]header_mode;
    logic[DESC_BITS-1:0]header_descriptor_count;
    logic[FACTOR_ADDR_BITS-1:0]header_factor_base;
    logic factor_req_valid,factor_req_ready,factor_req_accept;
    logic[23:0]factor_req_tag;logic[15:0]factor_req_epoch;
    logic[DESC_BITS-1:0]factor_req_descriptor;
    logic[FACTOR_ADDR_BITS-1:0]factor_req_addr;
    logic factor_rsp_valid,factor_rsp_ready,factor_rsp_accept;
    logic[23:0]factor_rsp_tag;logic[15:0]factor_rsp_epoch;
    logic[DESC_BITS-1:0]factor_rsp_descriptor;
    logic[FACTOR_ADDR_BITS-1:0]factor_rsp_addr;logic[8:0]factor_rsp_source;
    logic[7:0]factor_rsp_context_mask,factor_rsp_sign_mask;
    logic factor_rsp_zero,factor_rsp_last;
    logic weight_req_valid,weight_req_ready,weight_req_accept;
    logic[23:0]weight_req_tag;logic[15:0]weight_req_epoch;
    logic[DESC_BITS-1:0]weight_req_descriptor;logic[8:0]weight_req_source;
    logic weight_rsp_valid,weight_rsp_ready,weight_rsp_accept;
    logic[23:0]weight_rsp_tag;logic[15:0]weight_rsp_epoch;
    logic[DESC_BITS-1:0]weight_rsp_descriptor;logic[8:0]weight_rsp_source;
    logic[LANES*8-1:0]weight_rsp_data;
    logic acc_read_req_valid,acc_read_req_ready,acc_read_req_accept;
    logic[23:0]acc_read_req_tag;logic[15:0]acc_read_req_epoch;
    logic[DESC_BITS-1:0]acc_read_req_descriptor;logic[2:0]acc_read_req_context;
    logic acc_read_req_commit;logic acc_read_rsp_valid,acc_read_rsp_ready;
    logic acc_read_rsp_accept;logic[23:0]acc_read_rsp_tag;
    logic[15:0]acc_read_rsp_epoch;
    logic[DESC_BITS-1:0]acc_read_rsp_descriptor;
    logic[2:0]acc_read_rsp_context;logic acc_read_rsp_commit;
    logic[LANES*ACC_BITS-1:0]acc_read_rsp_data;
    logic acc_write_valid,acc_write_ready,acc_write_accept;
    logic[23:0]acc_write_tag;logic[15:0]acc_write_epoch;
    logic[DESC_BITS-1:0]acc_write_descriptor;logic[2:0]acc_write_context;
    logic acc_write_update;logic[LANES*ACC_BITS-1:0]acc_write_data;
    logic commit_valid,commit_ready,commit_accept;logic[23:0]commit_tag;
    logic[15:0]commit_epoch;logic[2:0]commit_context;logic commit_last;
    logic[LANES*ACC_BITS-1:0]commit_data;
    logic abort_valid,abort_ready,abort_accept;logic[23:0]abort_tag;
    logic[15:0]abort_epoch;logic[3:0]abort_reason;
    logic done_valid,done_ready,done_accept;logic[23:0]done_tag;
    logic[15:0]done_epoch;logic done_empty_bypass;
    logic descriptor_retire_valid;logic[DESC_BITS-1:0]descriptor_retire_index;
    logic[15:0]descriptor_retire_cycles;logic protocol_error,numeric_overflow,busy;
    logic[31:0]debug_tile_cycles,debug_factor_request_count;
    logic[31:0]debug_weight_request_count,debug_acc_read_count;
    logic[31:0]debug_acc_write_count,debug_commit_count;
    logic[31:0]debug_empty_bypass_count,debug_abort_count;

    logic[8:0]factor_source[0:63];logic[7:0]factor_mask[0:63];
    logic[7:0]factor_sign[0:63];logic factor_zero[0:63];
    logic signed[ACC_BITS-1:0]bank[0:CONTEXTS-1][0:LANES-1];
    logic signed[ACC_BITS-1:0]bit_sparse_reference[0:CONTEXTS-1][0:LANES-1];
    integer cycles,total_tiles,total_descriptors,total_commits,total_attacks;
    integer factor_stalls,weight_stalls,acc_read_stalls,acc_write_stalls;
    integer commit_stalls,abort_stalls,empty_tiles,clean_retire_checks;
    logic stall_enable,clean_cycle_check,auto_factor,auto_weight,auto_acc;
    logic force_overflow_rsp;
    logic factor_consumed,weight_consumed,acc_consumed;
    typedef struct packed{int due;logic[23:0]tag;logic[15:0]epoch;
        logic[DESC_BITS-1:0]descriptor;
        logic[FACTOR_ADDR_BITS-1:0]addr;}factor_item_t;
    typedef struct packed{int due;logic[23:0]tag;logic[15:0]epoch;
        logic[DESC_BITS-1:0]descriptor;logic[8:0]source;}weight_item_t;
    typedef struct packed{int due;logic[23:0]tag;logic[15:0]epoch;
        logic[DESC_BITS-1:0]descriptor;logic[2:0]ctx;
        logic commit;}acc_item_t;
    factor_item_t factor_queue[$];weight_item_t weight_queue[$];
    acc_item_t acc_queue[$];

    m270_fc1_descriptor_lifecycle_wrapper dut(.*);
    m270_fc1_descriptor_lifecycle_assertions sva(.*);

    function automatic integer signed wval(input integer source,input integer lane);
        integer value;begin
            if(source==383)value=127;
            else begin value=(source*19+lane*31+7)%255;value-=127;end
            return value;
        end
    endfunction
    function automatic logic[LANES*8-1:0]wvec(input integer source);
        logic[LANES*8-1:0]value;begin value='0;
            for(integer lane=0;lane<LANES;lane++)
                value[lane*8+:8]=wval(source,lane);
            return value;
        end
    endfunction
    function automatic integer popcount8(input logic[7:0]value);
        integer count;begin count=0;
            for(integer bit_index=0;bit_index<8;bit_index++)count+=value[bit_index];
            return count;
        end
    endfunction

    always @(posedge clk_core)begin
        if(rst_core)cycles=0;
        else begin
            cycles++;
            if(factor_req_valid&&!factor_req_ready)factor_stalls++;
            if(weight_req_valid&&!weight_req_ready)weight_stalls++;
            if(acc_read_req_valid&&!acc_read_req_ready)acc_read_stalls++;
            if(acc_write_valid&&!acc_write_ready)acc_write_stalls++;
            if(commit_valid&&!commit_ready)commit_stalls++;
            if(abort_valid&&!abort_ready)abort_stalls++;
            if(factor_req_accept&&auto_factor)begin factor_item_t item;
                item.due=cycles+1;item.tag=factor_req_tag;
                item.epoch=factor_req_epoch;item.descriptor=factor_req_descriptor;
                item.addr=factor_req_addr;factor_queue.push_back(item);end
            if(weight_req_accept&&auto_weight)begin weight_item_t item;
                item.due=cycles+1;item.tag=weight_req_tag;
                item.epoch=weight_req_epoch;item.descriptor=weight_req_descriptor;
                item.source=weight_req_source;weight_queue.push_back(item);end
            if(acc_read_req_accept&&auto_acc)begin acc_item_t item;
                item.due=cycles;item.tag=acc_read_req_tag;
                item.epoch=acc_read_req_epoch;
                item.descriptor=acc_read_req_descriptor;
                item.ctx=acc_read_req_context;
                item.commit=acc_read_req_commit;acc_queue.push_back(item);end
            if(factor_rsp_accept)factor_consumed=1;
            if(weight_rsp_accept)weight_consumed=1;
            if(acc_read_rsp_accept)acc_consumed=1;
            if(acc_write_accept)begin
                integer ctx;ctx=acc_write_context;
                for(integer lane=0;lane<LANES;lane++)begin
                    logic signed[ACC_BITS-1:0]observed;
                    integer signed reference;
                    observed=acc_write_data[lane*ACC_BITS+:ACC_BITS];
                    if(!acc_write_update)reference=0;
                    else begin
                        reference=$signed(bank[ctx][lane]);
                        if(!factor_zero[acc_write_descriptor])begin
                            if(factor_sign[acc_write_descriptor][ctx])
                                reference-=wval(factor_source[acc_write_descriptor],lane);
                            else reference+=wval(factor_source[acc_write_descriptor],lane);
                        end
                    end
                    if($signed(observed)!==reference)
                        $fatal(1,"M262 acc mismatch d=%0d c=%0d l=%0d e=%0d o=%0d",
                            acc_write_descriptor,ctx,lane,reference,$signed(observed));
                    bank[ctx][lane]=observed;
                end
            end
            if(commit_accept)begin
                for(integer lane=0;lane<LANES;lane++)begin
                    logic signed[ACC_BITS-1:0]observed;
                    observed=commit_data[lane*ACC_BITS+:ACC_BITS];
                    if($signed(observed)!==$signed(bank[commit_context][lane]))
                        $fatal(1,"M262 commit mismatch c=%0d l=%0d",commit_context,lane);
                end
                total_commits++;
            end
            if(descriptor_retire_valid)begin
                integer expected_cycles;
                expected_cycles=6+3*popcount8(factor_mask[descriptor_retire_index]);
                if(clean_cycle_check&&descriptor_retire_cycles!=expected_cycles)
                    $fatal(1,"M262 descriptor cycle mismatch d=%0d e=%0d o=%0d",
                        descriptor_retire_index,expected_cycles,descriptor_retire_cycles);
                if(clean_cycle_check)clean_retire_checks++;
                total_descriptors++;
            end
        end
    end

    always @(negedge clk_core)begin
        if(rst_core)begin
            factor_req_ready=0;weight_req_ready=0;acc_read_req_ready=0;
            acc_write_ready=0;commit_ready=0;
        end else begin
            factor_req_ready=!stall_enable||(cycles%7)!=2;
            weight_req_ready=!stall_enable||(cycles%4)==0;
            acc_read_req_ready=!stall_enable||(cycles%6)!=3;
            acc_write_ready=!stall_enable||(cycles%8)!=4;
            commit_ready=!stall_enable||(cycles%4)!=1;
        end
        if(factor_consumed)begin factor_rsp_valid=0;factor_consumed=0;end
        if(weight_consumed)begin weight_rsp_valid=0;weight_consumed=0;end
        if(acc_consumed)begin acc_read_rsp_valid=0;acc_consumed=0;end
        if(auto_factor&&!factor_rsp_valid&&factor_queue.size()>0
                &&factor_queue[0].due<=cycles)begin factor_item_t item;
            integer descriptor;item=factor_queue.pop_front();
            descriptor=item.descriptor;factor_rsp_valid=1;
            factor_rsp_tag=item.tag;factor_rsp_epoch=item.epoch;
            factor_rsp_descriptor=item.descriptor;factor_rsp_addr=item.addr;
            factor_rsp_source=factor_source[descriptor];
            factor_rsp_context_mask=factor_mask[descriptor];
            factor_rsp_sign_mask=factor_sign[descriptor];
            factor_rsp_zero=factor_zero[descriptor];
            factor_rsp_last=descriptor==header_descriptor_count-1;
        end
        if(auto_weight&&!weight_rsp_valid&&weight_queue.size()>0
                &&weight_queue[0].due<=cycles)begin weight_item_t item;
            item=weight_queue.pop_front();weight_rsp_valid=1;
            weight_rsp_tag=item.tag;weight_rsp_epoch=item.epoch;
            weight_rsp_descriptor=item.descriptor;
            weight_rsp_source=item.source;weight_rsp_data=wvec(item.source);
        end
        if(auto_acc&&!acc_read_rsp_valid&&acc_queue.size()>0
                &&acc_queue[0].due<=cycles)begin acc_item_t item;
            item=acc_queue.pop_front();acc_read_rsp_valid=1;
            acc_read_rsp_tag=item.tag;acc_read_rsp_epoch=item.epoch;
            acc_read_rsp_descriptor=item.descriptor;
            acc_read_rsp_context=item.ctx;
            acc_read_rsp_commit=item.commit;acc_read_rsp_data='0;
            for(integer lane=0;lane<LANES;lane++)begin
                if(force_overflow_rsp&&!item.commit)
                    acc_read_rsp_data[lane*ACC_BITS+:ACC_BITS]=(1<<(ACC_BITS-1))-1;
                else acc_read_rsp_data[lane*ACC_BITS+:ACC_BITS]
                    =bank[item.ctx][lane];
            end
            force_overflow_rsp=0;
        end
    end

    task automatic clear_inputs;begin
        header_valid=0;header_tag=0;header_epoch=0;header_mode=0;
        header_descriptor_count=0;header_factor_base=0;
        factor_rsp_valid=0;factor_rsp_tag=0;factor_rsp_epoch=0;
        factor_rsp_descriptor=0;factor_rsp_addr=0;factor_rsp_source=0;
        factor_rsp_context_mask=0;factor_rsp_sign_mask=0;
        factor_rsp_zero=0;factor_rsp_last=0;
        weight_rsp_valid=0;weight_rsp_tag=0;weight_rsp_epoch=0;
        weight_rsp_descriptor=0;weight_rsp_source=0;weight_rsp_data=0;
        acc_read_rsp_valid=0;acc_read_rsp_tag=0;acc_read_rsp_epoch=0;
        acc_read_rsp_descriptor=0;acc_read_rsp_context=0;
        acc_read_rsp_commit=0;acc_read_rsp_data=0;
        done_ready=1;abort_ready=1;factor_consumed=0;weight_consumed=0;
        acc_consumed=0;factor_queue.delete();weight_queue.delete();
        acc_queue.delete();stall_enable=0;clean_cycle_check=0;
        auto_factor=1;auto_weight=1;auto_acc=1;force_overflow_rsp=0;
    end endtask
    task automatic reset;begin
        @(negedge clk_core);clear_inputs();rst_core=1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
    end endtask
    task automatic clear_factors;begin
        for(integer descriptor=0;descriptor<64;descriptor++)begin
            factor_source[descriptor]=0;factor_mask[descriptor]=1;
            factor_sign[descriptor]=0;factor_zero[descriptor]=0;end
        for(integer ctx=0;ctx<CONTEXTS;ctx++)
            for(integer lane=0;lane<LANES;lane++)bank[ctx][lane]=0;
    end endtask
    task automatic start_header(input logic[1:0]mode,input integer count,
            input integer identity);begin
        @(negedge clk_core);header_valid=1;header_mode=mode;
        header_descriptor_count=count;header_factor_base=20'h26000+identity*64;
        header_tag=24'h262000+identity;header_epoch=16'h2620+identity;
        do @(posedge clk_core);while(!header_accept);
        @(negedge clk_core);header_valid=0;
    end endtask
    task automatic finish_tile(input integer count);begin
        do @(posedge clk_core);while(!done_accept);
        #0.2;
        if(debug_factor_request_count!=count||debug_weight_request_count!=count)
            $fatal(1,"M262 request conservation count=%0d factor=%0d weight=%0d",
                count,debug_factor_request_count,debug_weight_request_count);
        if(debug_commit_count!=8)$fatal(1,"M262 commit conservation");
        total_tiles++;
    end endtask
    task automatic run_empty;begin
        reset();clear_factors();
        @(negedge clk_core);header_valid=1;header_mode=2;
        header_descriptor_count=0;header_factor_base=20'h26000;
        header_tag=24'h270000;header_epoch=16'h2700;done_ready=0;
        repeat(2)begin
            @(posedge clk_core);#0.2;
            if(header_ready||header_accept||done_accept||!done_valid)
                $fatal(1,"M270 empty header bypass ignored done backpressure");
        end
        @(negedge clk_core);done_ready=1;
        do @(posedge clk_core);while(!header_accept);
        #0.2;
        if(!done_accept)$fatal(1,"M270 empty header/done not atomic");
        @(negedge clk_core);header_valid=0;
        if(!done_empty_bypass||debug_factor_request_count!=0
                ||debug_acc_write_count!=0)
            $fatal(1,"M270 empty bypass side effect");
        empty_tiles++;total_tiles++;
    end endtask
    task automatic run_dense;begin
        reset();clear_factors();clean_cycle_check=1;
        for(integer descriptor=0;descriptor<6;descriptor++)begin
            factor_source[descriptor]=descriptor+3;
            factor_mask[descriptor]=8'h1<<(descriptor%8);
            factor_sign[descriptor]=(descriptor%3==0)?factor_mask[descriptor]:0;
            factor_zero[descriptor]=descriptor==2||descriptor==5;
        end
        start_header(0,6,1);finish_tile(6);
    end endtask
    task automatic build_bit_sparse;begin clear_factors();
        factor_source[0]=9;factor_mask[0]=8'h01;factor_sign[0]=8'h00;
        factor_source[1]=9;factor_mask[1]=8'h04;factor_sign[1]=8'h04;
        factor_source[2]=21;factor_mask[2]=8'h02;factor_sign[2]=8'h00;
        factor_source[3]=21;factor_mask[3]=8'h10;factor_sign[3]=8'h10;
        factor_source[4]=31;factor_mask[4]=8'h80;factor_sign[4]=8'h00;
    end endtask
    task automatic run_bit_sparse;begin
        reset();build_bit_sparse();clean_cycle_check=1;
        start_header(1,5,2);finish_tile(5);
        for(integer ctx=0;ctx<8;ctx++)
            for(integer lane=0;lane<LANES;lane++)
                bit_sparse_reference[ctx][lane]=bank[ctx][lane];
    end endtask
    task automatic run_factorized;begin
        reset();clear_factors();clean_cycle_check=1;
        factor_source[0]=9;factor_mask[0]=8'h05;factor_sign[0]=8'h04;
        factor_source[1]=21;factor_mask[1]=8'h12;factor_sign[1]=8'h10;
        factor_source[2]=31;factor_mask[2]=8'h80;factor_sign[2]=8'h00;
        start_header(2,3,3);finish_tile(3);
        for(integer ctx=0;ctx<8;ctx++)
            for(integer lane=0;lane<LANES;lane++)
                if(bank[ctx][lane]!==bit_sparse_reference[ctx][lane])
                    $fatal(1,"M262 factorized equivalence mismatch c=%0d l=%0d",
                        ctx,lane);
    end endtask
    task automatic run_stalled;begin
        reset();clear_factors();stall_enable=1;clean_cycle_check=0;
        factor_source[0]=4;factor_mask[0]=8'hff;factor_sign[0]=8'h55;
        factor_source[1]=7;factor_mask[1]=8'h3c;factor_sign[1]=8'h18;
        factor_source[2]=12;factor_mask[2]=8'h81;factor_sign[2]=8'h80;
        factor_source[3]=18;factor_mask[3]=8'h22;factor_sign[3]=8'h02;
        start_header(2,4,4);finish_tile(4);
    end endtask
    task automatic run_popcount_sweep;begin
        reset();clear_factors();clean_cycle_check=1;
        for(integer descriptor=0;descriptor<8;descriptor++)begin
            factor_source[descriptor]=40+descriptor;
            factor_mask[descriptor]=(1<<(descriptor+1))-1;
            factor_sign[descriptor]=factor_mask[descriptor]&8'h55;
            factor_zero[descriptor]=0;
        end
        start_header(2,8,9);finish_tile(8);
    end endtask
    task automatic accept_abort(input integer reason);begin
        do @(negedge clk_core);while(!abort_valid);
        if(abort_reason!=reason)$fatal(1,"M262 abort reason e=%0d o=%0d",
                                     reason,abort_reason);
        abort_ready=0;repeat(2)@(posedge clk_core);@(negedge clk_core);
        abort_ready=1;do @(posedge clk_core);while(!abort_accept);
        #0.2;if(debug_abort_count!=1||abort_valid||!protocol_error||!busy)
            $fatal(1,"M262 abort did not retire into sticky fault");
        total_attacks++;
    end endtask
    task automatic stale_factor_attack;begin
        reset();clear_factors();auto_factor=0;factor_source[0]=3;
        start_header(2,1,5);do @(posedge clk_core);while(!factor_req_accept);
        @(negedge clk_core);factor_rsp_valid=1;factor_rsp_tag=factor_req_tag+1'b1;
        factor_rsp_epoch=factor_req_epoch;factor_rsp_descriptor=0;
        factor_rsp_addr=factor_req_addr;factor_rsp_source=3;
        factor_rsp_context_mask=1;factor_rsp_sign_mask=0;
        factor_rsp_zero=0;factor_rsp_last=1;accept_abort(2);
    end endtask
    task automatic stale_weight_attack;begin
        reset();clear_factors();auto_weight=0;factor_source[0]=5;
        start_header(2,1,6);do @(posedge clk_core);while(!weight_req_accept);
        @(negedge clk_core);weight_rsp_valid=1;weight_rsp_tag=weight_req_tag;
        weight_rsp_epoch=weight_req_epoch;
        weight_rsp_descriptor=weight_req_descriptor+1'b1;
        weight_rsp_source=weight_req_source;weight_rsp_data=wvec(weight_req_source);
        accept_abort(3);
    end endtask
    task automatic stale_acc_attack;begin
        reset();clear_factors();auto_acc=0;factor_source[0]=6;
        start_header(2,1,7);do @(posedge clk_core);while(!acc_read_req_accept);
        @(negedge clk_core);acc_read_rsp_valid=1;acc_read_rsp_tag=acc_read_req_tag;
        acc_read_rsp_epoch=acc_read_req_epoch;
        acc_read_rsp_descriptor=acc_read_req_descriptor;
        acc_read_rsp_context=acc_read_req_context+1'b1;
        acc_read_rsp_commit=0;acc_read_rsp_data=0;accept_abort(4);
    end endtask
    task automatic overflow_attack;begin
        reset();clear_factors();factor_source[0]=383;factor_mask[0]=1;
        factor_sign[0]=0;force_overflow_rsp=1;
        start_header(2,1,8);accept_abort(5);
        if(!numeric_overflow)$fatal(1,"M262 overflow not sticky");
    end endtask
    task automatic malformed_header_attack(input logic[1:0]mode,
            input integer count,input logic[FACTOR_ADDR_BITS-1:0]base,
            input integer identity);begin
        logic[23:0] expected_tag;logic[15:0] expected_epoch;
        reset();clear_factors();expected_tag=24'h270000+identity;
        expected_epoch=16'h2700+identity;
        @(negedge clk_core);abort_ready=0;header_valid=1;header_mode=mode;
        header_descriptor_count=count;header_factor_base=base;
        header_tag=expected_tag;header_epoch=expected_epoch;
        @(posedge clk_core);#0.2;
        if(header_ready||header_accept||!protocol_error||!abort_valid
                ||abort_reason!=1||abort_tag!=expected_tag
                ||abort_epoch!=expected_epoch)
            $fatal(1,"M270 malformed header did not fail closed");
        @(negedge clk_core);header_valid=0;accept_abort(1);
    end endtask

    initial begin #3000000;$fatal(1,"M262 watchdog");end
    initial begin
        rst_core=1;cycles=0;total_tiles=0;total_descriptors=0;
        total_commits=0;total_attacks=0;factor_stalls=0;weight_stalls=0;
        acc_read_stalls=0;acc_write_stalls=0;commit_stalls=0;abort_stalls=0;
        empty_tiles=0;clean_retire_checks=0;clear_inputs();
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        run_empty();run_dense();run_bit_sparse();run_factorized();run_stalled();
        run_popcount_sweep();
        stale_factor_attack();stale_weight_attack();stale_acc_attack();
        overflow_attack();
        malformed_header_attack(3,1,20'h10000,10);
        malformed_header_attack(2,3073,20'h10000,11);
        malformed_header_attack(2,32,20'hffff0,12);
        if(total_tiles!=6||empty_tiles!=1||total_descriptors!=26
                ||total_commits!=40||total_attacks!=7||clean_retire_checks!=22)
            $fatal(1,"M270 aggregate conservation");
        if(factor_stalls==0||weight_stalls==0||acc_read_stalls==0
                ||acc_write_stalls==0||commit_stalls==0||abort_stalls==0)
            $fatal(1,"M262 missing backpressure cover");
        $display("PASS M270 lanes=8 contexts=8 tiles=%0d empty=%0d desc=%0d clean_cycle_checks=%0d commits=%0d attacks=%0d stalls=%0d,%0d,%0d,%0d,%0d,%0d",
            total_tiles,empty_tiles,total_descriptors,clean_retire_checks,
            total_commits,total_attacks,factor_stalls,weight_stalls,
            acc_read_stalls,acc_write_stalls,commit_stalls,abort_stalls);
        $finish;
    end
endmodule
`default_nettype wire
