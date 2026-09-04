`timescale 1ns/1ps
`default_nettype none

module tb_m218_fc2_tagged_slice_service_island;
    localparam int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16;
    localparam int GENERATION_BITS=32, SLICE_LANES=16;
    logic clk_core=0, rst_core;
    always #5 clk_core = ~clk_core;

    logic soft_flush,mem_flush_valid,mem_flush_ready;
    logic[EPOCH_BITS-1:0]mem_flush_epoch;
    logic mem_flush_ack_valid,mem_flush_ack_ready;
    logic[EPOCH_BITS-1:0]mem_flush_ack_epoch;
    logic header_valid,header_ready,header_accept;
    logic[TAG_BITS-1:0]header_tag;logic[3:0]header_output_blocks;
    logic group_valid,group_ready,group_accept;
    logic[TAG_BITS-1:0]group_tag;logic[2:0]group_output_block;
    logic[3:0]group_source_count;logic[7:0]group_bank_valid;
    logic[CHANNEL_BITS-1:0]group_source_channel[0:7];
    logic frontend_done_valid,frontend_done_ready,frontend_done_accept;
    logic[TAG_BITS-1:0]frontend_done_tag;logic frontend_done_had_event;
    logic mem_req_valid,mem_req_ready,mem_req_accept;
    logic[EPOCH_BITS-1:0]mem_req_epoch;logic[2:0]mem_req_slot;
    logic[GENERATION_BITS-1:0]mem_req_generation;
    logic[TAG_BITS-1:0]mem_req_tag;logic[2:0]mem_req_output_block;
    logic[2:0]mem_req_slice;logic[3:0]mem_req_source_count;
    logic[7:0]mem_req_bank_valid;
    logic[CHANNEL_BITS-1:0]mem_req_source_channel[0:7];
    logic mem_rsp_valid,mem_rsp_ready,mem_rsp_accept;
    logic[EPOCH_BITS-1:0]mem_rsp_epoch;logic[2:0]mem_rsp_slot;
    logic[GENERATION_BITS-1:0]mem_rsp_generation;
    logic[TAG_BITS-1:0]mem_rsp_tag;logic[7:0]mem_rsp_bank_valid;
    logic signed[7:0]mem_rsp_weight[0:7][0:SLICE_LANES-1];
    logic result_valid,result_ready,result_accept;
    logic[TAG_BITS-1:0]result_tag;logic[2:0]result_output_block;
    logic[2:0]result_slice;logic signed[23:0]result_accumulator[0:SLICE_LANES-1];
    logic result_last,token_done_valid,token_done_ready,token_done_accept;
    logic[TAG_BITS-1:0]token_done_tag;logic token_done_had_event;
    logic protocol_error,numeric_overflow,stale_response_seen,busy;
    logic[2:0]debug_fifo_count;logic[3:0]debug_outstanding_count;
    logic[31:0]debug_group_accept_count,debug_request_accept_count;
    logic[31:0]debug_response_accept_count,debug_context_write_count;
    logic[31:0]debug_result_accept_count,debug_active_bank_read_count;

    integer cycle_count,done_count,result_count,error_count;
    integer capture_mode;
    integer total_groups,total_requests,total_responses,total_contexts;
    integer total_results,total_done,total_active_bank_reads;
    integer same_edge_slot_reuse,same_edge_context_reuse,ooo_retirements;
    integer request_stalls,result_stalls,max_model_outstanding;
    integer max_fifo_count,max_dut_outstanding,flush_count,timeout_count;
    integer flush_stale_drops,identity_attacks,delayed_duplicate_attacks;
    logic signed[23:0]expected_ctx[0:7][0:5][0:SLICE_LANES-1];
    logic signed[23:0]k8_snapshot[0:7][0:5][0:SLICE_LANES-1];

    logic model_valid[0:7];
    logic[EPOCH_BITS-1:0]model_epoch[0:7];
    logic[GENERATION_BITS-1:0]model_generation[0:7];
    logic[TAG_BITS-1:0]model_tag[0:7];
    logic[2:0]model_block[0:7],model_slice[0:7];
    logic[7:0]model_mask[0:7];
    logic[CHANNEL_BITS-1:0]model_channel[0:7][0:7];
    logic score_responses;

    logic[EPOCH_BITS-1:0]stale_epoch;
    logic[2:0]stale_slot,stale_block,stale_slice;
    logic[GENERATION_BITS-1:0]stale_generation;
    logic[TAG_BITS-1:0]stale_tag;
    logic[7:0]stale_mask;
    logic[CHANNEL_BITS-1:0]stale_channel[0:7];

    function automatic integer popcount8(input logic[7:0]value);
        integer count;
        begin count=0;for(integer bank=0;bank<8;bank++)count+=value[bank];return count;end
    endfunction
    function automatic integer signed weight_value(
        input integer bank,input integer lane,input integer channel,
        input integer block,input integer slice);
        integer value;
        begin
            value=(channel*3+bank*5+block*7+slice*11+lane*13)%31;
            return value-15;
        end
    endfunction

    m218_fc2_tagged_slice_service_island dut(.*);
    m218_fc2_tagged_slice_service_assertions sva(.*);

    always @(posedge clk_core) begin
        if(rst_core) begin
            cycle_count=0;done_count=0;result_count=0;
            for(integer slot=0;slot<8;slot++)model_valid[slot]=0;
        end else begin
            integer live_before;
            cycle_count++;
            if(group_accept)total_groups++;
            if(mem_req_valid&&!mem_req_ready)request_stalls++;
            if(result_valid&&!result_ready)result_stalls++;
            if(mem_flush_valid&&mem_flush_ready)flush_count++;
            if(mem_rsp_accept&&!score_responses
                    &&(dut.flush_active_q||soft_flush))flush_stale_drops++;
            if(debug_fifo_count>max_fifo_count)max_fifo_count=debug_fifo_count;
            if(debug_outstanding_count>max_dut_outstanding)
                max_dut_outstanding=debug_outstanding_count;
            if(dut.response_skid_commit)total_contexts++;
            if(mem_rsp_accept&&score_responses) begin
                total_responses++;
                if(!model_valid[mem_rsp_slot])begin
                    $error("response accepted for empty model slot %0d",mem_rsp_slot);error_count++;
                end else begin
                    if(mem_rsp_epoch!==model_epoch[mem_rsp_slot]
                            ||mem_rsp_generation!==model_generation[mem_rsp_slot]
                            ||mem_rsp_tag!==model_tag[mem_rsp_slot]
                            ||mem_rsp_bank_valid!==model_mask[mem_rsp_slot])begin
                        $error("response identity mismatch slot=%0d",mem_rsp_slot);error_count++;
                    end
                    for(integer lane=0;lane<SLICE_LANES;lane++)begin
                        integer signed delta;
                        delta=0;
                        for(integer bank=0;bank<8;bank++)if(model_mask[mem_rsp_slot][bank])
                            delta+=weight_value(bank,lane,model_channel[mem_rsp_slot][bank],
                                model_block[mem_rsp_slot],model_slice[mem_rsp_slot]);
                        expected_ctx[model_block[mem_rsp_slot]][model_slice[mem_rsp_slot]][lane]
                            =expected_ctx[model_block[mem_rsp_slot]][model_slice[mem_rsp_slot]][lane]+delta;
                    end
                    for(integer slot=0;slot<8;slot++)
                        if(model_valid[slot]&&slot!=mem_rsp_slot
                                &&model_generation[slot]<model_generation[mem_rsp_slot])
                            ooo_retirements++;
                    model_valid[mem_rsp_slot]=0;
                end
            end
            if(mem_req_accept)begin
                total_requests++;
                total_active_bank_reads+=popcount8(mem_req_bank_valid);
                if(model_valid[mem_req_slot])begin
                    $error("request reused nonretired slot %0d",mem_req_slot);error_count++;
                end
                if(mem_rsp_accept&&score_responses&&mem_rsp_slot==mem_req_slot)
                    same_edge_slot_reuse++;
                for(integer slot=0;slot<8;slot++)begin
                    if(model_valid[slot]
                            &&model_block[slot]==mem_req_output_block
                            &&model_slice[slot]==mem_req_slice)begin
                        $error("context hazard block=%0d slice=%0d prior_slot=%0d new_slot=%0d",
                            mem_req_output_block,mem_req_slice,slot,mem_req_slot);
                        error_count++;
                    end
                end
                if(mem_rsp_accept&&score_responses
                        &&model_block[mem_rsp_slot]==mem_req_output_block
                        &&model_slice[mem_rsp_slot]==mem_req_slice)
                    same_edge_context_reuse++;
                model_valid[mem_req_slot]=1;
                model_epoch[mem_req_slot]=mem_req_epoch;
                model_generation[mem_req_slot]=mem_req_generation;
                model_tag[mem_req_slot]=mem_req_tag;
                model_block[mem_req_slot]=mem_req_output_block;
                model_slice[mem_req_slot]=mem_req_slice;
                model_mask[mem_req_slot]=mem_req_bank_valid;
                for(integer bank=0;bank<8;bank++)
                    model_channel[mem_req_slot][bank]=mem_req_source_channel[bank];
            end
            live_before=0;
            for(integer slot=0;slot<8;slot++)if(model_valid[slot])live_before++;
            if(live_before>max_model_outstanding)max_model_outstanding=live_before;
            if(result_accept)begin
                total_results++;
                result_count++;
                for(integer lane=0;lane<SLICE_LANES;lane++)begin
                    if(result_accumulator[lane]!==expected_ctx[result_output_block][result_slice][lane])begin
                        $error("numeric mismatch mode=%0d block=%0d slice=%0d lane=%0d got=%0d exp=%0d",
                            capture_mode,result_output_block,result_slice,lane,
                            result_accumulator[lane],expected_ctx[result_output_block][result_slice][lane]);
                        error_count++;
                    end
                    if(capture_mode==1)
                        k8_snapshot[result_output_block][result_slice][lane]=result_accumulator[lane];
                    if(capture_mode==2&&result_accumulator[lane]!==
                            k8_snapshot[result_output_block][result_slice][lane])begin
                        $error("K1/K8 equivalence mismatch block=%0d slice=%0d lane=%0d",
                            result_output_block,result_slice,lane);error_count++;
                    end
                end
            end
            if(token_done_accept)begin done_count++;total_done++;end
        end
    end

    always @(negedge clk_core) begin
        if(rst_core)begin mem_req_ready<=0;result_ready<=0;token_done_ready<=0;end
        else begin
            mem_req_ready<=cycle_count%7!=0;
            result_ready<=cycle_count%4!=0;
            token_done_ready<=cycle_count%3!=0;
        end
    end

    task automatic clear_expected;
        for(integer block=0;block<8;block++)
            for(integer slice=0;slice<6;slice++)
                for(integer lane=0;lane<SLICE_LANES;lane++)expected_ctx[block][slice][lane]=0;
    endtask
    task automatic apply_reset;
        begin
            @(negedge clk_core);rst_core=1;header_valid=0;group_valid=0;
            frontend_done_valid=0;mem_rsp_valid=0;soft_flush=0;
            mem_flush_ack_valid=0;score_responses=1;
            repeat(3)@(negedge clk_core);rst_core=0;clear_expected();
            repeat(2)@(posedge clk_core);
        end
    endtask
    task automatic send_header(input logic[TAG_BITS-1:0]tag,input integer blocks);
        begin
            clear_expected();result_count=0;
            @(negedge clk_core);header_tag=tag;header_output_blocks=blocks;header_valid=1;
            while(!header_accept)@(posedge clk_core);
            @(negedge clk_core);header_valid=0;
        end
    endtask
    task automatic send_group(input logic[TAG_BITS-1:0]tag,input integer block,
        input logic[7:0]mask,input integer channel_base);
        begin
            @(negedge clk_core);group_tag=tag;group_output_block=block;
            group_bank_valid=mask;group_source_count=popcount8(mask);
            for(integer bank=0;bank<8;bank++)
                group_source_channel[bank]=channel_base+bank;
            group_valid=1;
            while(!group_accept)@(posedge clk_core);
            @(negedge clk_core);group_valid=0;
        end
    endtask
    task automatic send_frontend_done(input logic[TAG_BITS-1:0]tag,input logic had_event);
        begin
            @(negedge clk_core);frontend_done_tag=tag;
            frontend_done_had_event=had_event;frontend_done_valid=1;
            while(!frontend_done_accept)@(posedge clk_core);
            @(negedge clk_core);frontend_done_valid=0;
        end
    endtask
    task automatic drive_model_response(input integer slot);
        begin
            @(negedge clk_core);
            mem_rsp_epoch=model_epoch[slot];mem_rsp_slot=slot;
            mem_rsp_generation=model_generation[slot];mem_rsp_tag=model_tag[slot];
            mem_rsp_bank_valid=model_mask[slot];
            for(integer bank=0;bank<8;bank++)for(integer lane=0;lane<SLICE_LANES;lane++)
                mem_rsp_weight[bank][lane]=weight_value(bank,lane,model_channel[slot][bank],
                    model_block[slot],model_slice[slot]);
            mem_rsp_valid=1;
            while(!mem_rsp_accept)@(posedge clk_core);
            @(negedge clk_core);mem_rsp_valid=0;
        end
    endtask
    task automatic service_until_done(input integer starting_done);
        integer selected,guard;
        begin
            guard=0;repeat(10)@(posedge clk_core);
            while(done_count==starting_done)begin
                selected=-1;
                for(integer slot=0;slot<8;slot++)if(model_valid[slot]
                        &&(selected<0||model_generation[slot]>model_generation[selected]))selected=slot;
                if(selected>=0)drive_model_response(selected);else @(posedge clk_core);
                guard++;
                if(guard>20000)$fatal(1,"service watchdog");
            end
        end
    endtask
    task automatic capture_stale_request;
        integer selected;
        begin
            selected=-1;
            while(selected<0)begin
                for(integer slot=0;slot<8;slot++)if(model_valid[slot])selected=slot;
                if(selected<0)@(posedge clk_core);
            end
            stale_epoch=model_epoch[selected];stale_slot=selected;
            stale_generation=model_generation[selected];stale_tag=model_tag[selected];
            stale_block=model_block[selected];stale_slice=model_slice[selected];
            stale_mask=model_mask[selected];
            for(integer bank=0;bank<8;bank++)stale_channel[bank]=model_channel[selected][bank];
        end
    endtask
    task automatic drive_stale_response;
        begin
            @(negedge clk_core);mem_rsp_epoch=stale_epoch;mem_rsp_slot=stale_slot;
            mem_rsp_generation=stale_generation;mem_rsp_tag=stale_tag;
            mem_rsp_bank_valid=stale_mask;
            for(integer bank=0;bank<8;bank++)for(integer lane=0;lane<SLICE_LANES;lane++)
                mem_rsp_weight[bank][lane]=weight_value(bank,lane,stale_channel[bank],
                    stale_block,stale_slice);
            mem_rsp_valid=1;
            while(!mem_rsp_accept)@(posedge clk_core);
            @(negedge clk_core);mem_rsp_valid=0;
        end
    endtask
    task automatic perform_soft_flush_with_stale;
        begin
            score_responses=0;
            @(negedge clk_core);soft_flush=1;
            @(negedge clk_core);soft_flush=0;
            while(!mem_flush_valid)@(posedge clk_core);
            fork
                drive_stale_response();
                begin
                    while(!(mem_flush_valid&&mem_flush_ready))@(posedge clk_core);
                    repeat(3)@(posedge clk_core);
                    @(negedge clk_core);mem_flush_ack_epoch=mem_flush_epoch;
                    mem_flush_ack_valid=1;
                    while(!mem_flush_ack_ready)@(posedge clk_core);
                    @(negedge clk_core);mem_flush_ack_valid=0;
                end
            join
            wait(!busy);
            for(integer slot=0;slot<8;slot++)model_valid[slot]=0;
            score_responses=1;
        end
    endtask

    task automatic drive_corrupt_model_response(input integer slot,
        input integer attack_mode);
        logic [7:0] corrupt_mask;
        begin
            score_responses=0;
            @(negedge clk_core);
            mem_rsp_epoch=model_epoch[slot];mem_rsp_slot=slot;
            mem_rsp_generation=model_generation[slot];mem_rsp_tag=model_tag[slot];
            mem_rsp_bank_valid=model_mask[slot];
            case(attack_mode)
                0: mem_rsp_epoch=model_epoch[slot]+1'b1;
                1: mem_rsp_generation=model_generation[slot]+1'b1;
                2: mem_rsp_tag=model_tag[slot]^24'h000001;
                default: begin
                    corrupt_mask=model_mask[slot]^8'h03;
                    if(corrupt_mask==0)corrupt_mask=8'h80;
                    mem_rsp_bank_valid=corrupt_mask;
                end
            endcase
            for(integer bank=0;bank<8;bank++)for(integer lane=0;lane<SLICE_LANES;lane++)
                mem_rsp_weight[bank][lane]=weight_value(bank,lane,model_channel[slot][bank],
                    model_block[slot],model_slice[slot]);
            mem_rsp_valid=1;
            while(!mem_rsp_accept)@(posedge clk_core);
            @(negedge clk_core);mem_rsp_valid=0;
            repeat(2)@(posedge clk_core);
            identity_attacks++;
            if(!protocol_error||!dut.fault_cause_q[5])begin
                $error("identity attack mode=%0d did not fail closed fault=%0d cause=%b",
                    attack_mode,protocol_error,dut.fault_cause_q);error_count++;
            end
        end
    endtask

    task automatic run_identity_attack(input integer attack_mode,
        input logic[TAG_BITS-1:0]tag);
        integer selected;
        begin
            apply_reset();
            send_header(tag,1);send_group(tag,0,8'hff,0);
            selected=-1;
            while(selected<0)begin
                for(integer slot=0;slot<8;slot++)if(model_valid[slot])selected=slot;
                if(selected<0)@(posedge clk_core);
            end
            drive_corrupt_model_response(selected,attack_mode);
        end
    endtask

    task automatic run_delayed_duplicate_attack;
        integer selected;
        begin
            apply_reset();
            send_header(24'h218d00,1);send_group(24'h218d00,0,8'hff,0);
            selected=-1;
            while(selected<0)begin
                for(integer slot=0;slot<8;slot++)if(model_valid[slot])selected=slot;
                if(selected<0)@(posedge clk_core);
            end
            stale_epoch=model_epoch[selected];stale_slot=selected;
            stale_generation=model_generation[selected];stale_tag=model_tag[selected];
            stale_block=model_block[selected];stale_slice=model_slice[selected];
            stale_mask=model_mask[selected];
            for(integer bank=0;bank<8;bank++)stale_channel[bank]=model_channel[selected][bank];
            drive_model_response(selected);
            score_responses=0;drive_stale_response();
            repeat(2)@(posedge clk_core);
            delayed_duplicate_attacks++;
            if(!protocol_error||!dut.fault_cause_q[5])begin
                $error("delayed duplicate did not fail closed fault=%0d cause=%b",
                    protocol_error,dut.fault_cause_q);error_count++;
            end
        end
    endtask

    task automatic run_flush_ack_timeout_attack;
        integer guard;
        begin
            apply_reset();mem_flush_ready=1;
            @(negedge clk_core);soft_flush=1;
            @(negedge clk_core);soft_flush=0;#1;
            do @(posedge clk_core); while(!(mem_flush_valid&&mem_flush_ready));
            #1;
            guard=0;
            while(!protocol_error&&guard<1100)begin
                @(posedge clk_core);#1;guard++;
            end
            // The combinational timeout indication precedes the sticky cause
            // register by the accepting clock edge.
            if(protocol_error&&!dut.fault_cause_q[7])begin
                @(posedge clk_core);#1;
            end
            timeout_count++;
            if(!protocol_error||!dut.fault_cause_q[7])begin
                $error("flush ack timeout did not fail closed cycles=%0d cause=%b",
                    guard,dut.fault_cause_q);error_count++;
            end
        end
    endtask

    initial begin
        rst_core=1;soft_flush=0;mem_flush_ready=1;mem_flush_ack_valid=0;
        mem_flush_ack_epoch=0;header_valid=0;header_tag=0;header_output_blocks=0;
        group_valid=0;group_tag=0;group_output_block=0;group_source_count=0;
        group_bank_valid=0;frontend_done_valid=0;frontend_done_tag=0;
        frontend_done_had_event=0;mem_req_ready=0;mem_rsp_valid=0;
        mem_rsp_epoch=0;mem_rsp_slot=0;mem_rsp_generation=0;mem_rsp_tag=0;
        mem_rsp_bank_valid=1;result_ready=0;token_done_ready=0;
        cycle_count=0;done_count=0;result_count=0;error_count=0;capture_mode=0;
        total_groups=0;total_requests=0;total_responses=0;total_contexts=0;
        total_results=0;total_done=0;total_active_bank_reads=0;
        same_edge_slot_reuse=0;same_edge_context_reuse=0;ooo_retirements=0;
        request_stalls=0;result_stalls=0;max_model_outstanding=0;
        max_fifo_count=0;max_dut_outstanding=0;flush_count=0;timeout_count=0;
        flush_stale_drops=0;identity_attacks=0;delayed_duplicate_attacks=0;
        score_responses=1;clear_expected();
        for(integer bank=0;bank<8;bank++)begin group_source_channel[bank]=bank;
            for(integer lane=0;lane<SLICE_LANES;lane++)mem_rsp_weight[bank][lane]=0;end
        apply_reset();

        // One K8 group per output block, then the exact same eight source rows
        // serialized as K1 groups.  Results must be bit-identical.
        capture_mode=1;
        fork
            begin
                send_header(24'h218801,2);
                send_group(24'h218801,0,8'hff,0);
                send_group(24'h218801,1,8'hff,0);
                send_frontend_done(24'h218801,1);
            end
            service_until_done(done_count);
        join
        if(result_count!=12)begin $error("K8 result count %0d",result_count);error_count++;end

        capture_mode=2;
        fork
            begin
                send_header(24'h218101,2);
                for(integer bank=0;bank<8;bank++)begin
                    send_group(24'h218101,0,8'b1<<bank,0);
                    send_group(24'h218101,1,8'b1<<bank,0);
                end
                send_frontend_done(24'h218101,1);
            end
            service_until_done(done_count);
        join
        if(result_count!=12)begin $error("K1 result count %0d",result_count);error_count++;end

        // Zero-event tokens still drain every output slice before done.
        capture_mode=0;
        fork
            begin send_header(24'h218000,8);send_frontend_done(24'h218000,0);end
            service_until_done(done_count);
        join
        if(result_count!=48)begin $error("zero result count %0d",result_count);error_count++;end

        // Partial masks across all four legal stage widths exercise source
        // counts 1/3/4/8 without changing the fixed bank mapping.
        fork
            begin
                send_header(24'h218400,4);
                send_group(24'h218400,0,8'h01,0);
                send_group(24'h218400,1,8'h07,8);
                send_group(24'h218400,2,8'h55,16);
                send_group(24'h218400,3,8'hff,24);
                send_frontend_done(24'h218400,1);
            end
            service_until_done(done_count);
        join
        if(result_count!=24)begin $error("partial result count %0d",result_count);error_count++;end

        // Dense-bank96 stress: twelve K8 groups cover 96 distinct source
        // channels while retaining six exact slice requests per group.
        fork
            begin
                send_header(24'h218960,1);
                for(integer group_index=0;group_index<12;group_index++)
                    send_group(24'h218960,0,8'hff,group_index*8);
                send_frontend_done(24'h218960,1);
            end
            service_until_done(done_count);
        join
        if(result_count!=6)begin $error("dense96 result count %0d",result_count);error_count++;end

        if(total_groups!=34||total_requests!=204||total_responses!=204
                ||total_contexts!=204||total_results!=102||total_done!=5
                ||total_active_bank_reads!=864)begin
            $error("clean conservation groups=%0d req=%0d rsp=%0d ctx=%0d result=%0d done=%0d reads=%0d",
                total_groups,total_requests,total_responses,total_contexts,
                total_results,total_done,total_active_bank_reads);error_count++;
        end
        if(max_model_outstanding!=8||max_dut_outstanding!=8
                ||same_edge_slot_reuse==0||same_edge_context_reuse==0
                ||ooo_retirements==0||request_stalls==0||result_stalls==0)begin
            $error("coverage deficit model_o=%0d dut_o=%0d slot_reuse=%0d ctx_reuse=%0d ooo=%0d req_stall=%0d result_stall=%0d",
                max_model_outstanding,max_dut_outstanding,same_edge_slot_reuse,
                same_edge_context_reuse,ooo_retirements,request_stalls,result_stalls);
            error_count++;
        end

        // A/flush/B/stale-A: an old response during flush is dropped and
        // observed.  B first completes numerically with zero A pollution;
        // only then is the same stale response replayed and rejected.
        send_header(24'h2180a0,1);send_group(24'h2180a0,0,8'hff,0);
        capture_stale_request();
        perform_soft_flush_with_stale();
        #1;
        if(!stale_response_seen||protocol_error)begin
            $error("flush stale handling failed stale=%0d fault=%0d cause=%b ih=%0d ig=%0d id=%0d ifl=%0d ifa=%0d rv=%0d fl=%0d sf=%0d ril=%0d gen=%0d",
                stale_response_seen,protocol_error,dut.fault_cause_q,
                dut.illegal_header,dut.illegal_group,dut.illegal_frontend_done,
                dut.illegal_flush,dut.illegal_flush_ack,mem_rsp_valid,
                dut.flush_active_q,soft_flush,dut.response_identity_legal,
                dut.generation_exhausted);error_count++;
        end
        fork
            begin
                send_header(24'h2180b0,1);send_group(24'h2180b0,0,8'h01,0);
                send_frontend_done(24'h2180b0,1);
            end
            service_until_done(done_count);
        join
        if(result_count!=6||debug_group_accept_count!=1
                ||debug_request_accept_count!=6||debug_response_accept_count!=6
                ||debug_context_write_count!=6||debug_result_accept_count!=6
                ||debug_active_bank_read_count!=6)begin
            $error("B zero-pollution/conservation failed result=%0d g=%0d req=%0d rsp=%0d ctx=%0d out=%0d reads=%0d",
                result_count,debug_group_accept_count,debug_request_accept_count,
                debug_response_accept_count,debug_context_write_count,
                debug_result_accept_count,debug_active_bank_read_count);error_count++;
        end
        score_responses=0;drive_stale_response();
        repeat(2)@(posedge clk_core);
        if(!protocol_error)begin $error("post-ack stale response not rejected");error_count++;end

        // Independent fail-closed identity attacks, each isolated by POR.
        run_identity_attack(0,24'h218e00);
        run_identity_attack(1,24'h218e01);
        run_identity_attack(2,24'h218e02);
        run_identity_attack(3,24'h218e03);
        run_delayed_duplicate_attack();
        run_flush_ack_timeout_attack();

        if(error_count==0)$display("PASS M218 directed numeric/protocol clean_groups=34 clean_requests=204 clean_responses=204 clean_contexts=204 clean_results=102 clean_done=5 clean_bank_reads=864 total_groups=%0d total_requests=%0d total_responses=%0d total_contexts=%0d total_results=%0d total_done=%0d total_bank_reads=%0d max_fifo=%0d max_outstanding=%0d slot_reuse=%0d context_reuse=%0d ooo_retirements=%0d request_stalls=%0d result_stalls=%0d flushes=%0d flush_stale_drops=%0d identity_attacks=%0d duplicate_attacks=%0d timeouts=%0d",
            total_groups,total_requests,total_responses,total_contexts,total_results,
            total_done,total_active_bank_reads,max_fifo_count,max_dut_outstanding,
            same_edge_slot_reuse,same_edge_context_reuse,ooo_retirements,
            request_stalls,result_stalls,flush_count,flush_stale_drops,
            identity_attacks,delayed_duplicate_attacks,timeout_count);
        else $fatal(1,"M218 failures=%0d",error_count);
        $finish;
    end
endmodule

`default_nettype wire
