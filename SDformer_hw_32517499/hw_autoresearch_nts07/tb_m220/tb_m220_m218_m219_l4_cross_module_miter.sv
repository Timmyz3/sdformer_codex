`timescale 1ns/1ps
`default_nettype none

module tb_m220_m218_m219_l4_cross_module_miter;
    localparam int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16;
    localparam int GENERATION_BITS=32, LANES=16, L=4, O=8;
    logic clk_core=0, rst_core;
    always #5 clk_core=~clk_core;

    logic a_soft_flush,a_mem_flush_valid,a_mem_flush_ready;
    logic[EPOCH_BITS-1:0]a_mem_flush_epoch;
    logic a_mem_flush_ack_valid,a_mem_flush_ack_ready;
    logic[EPOCH_BITS-1:0]a_mem_flush_ack_epoch;
    logic a_header_valid,a_header_ready,a_header_accept;
    logic[TAG_BITS-1:0]a_header_tag;logic[3:0]a_header_output_blocks;
    logic a_group_valid,a_group_ready,a_group_accept;
    logic[TAG_BITS-1:0]a_group_tag;logic[2:0]a_group_output_block;
    logic[3:0]a_group_source_count;logic[7:0]a_group_bank_valid;
    logic[CHANNEL_BITS-1:0]a_group_source_channel[0:7];
    logic a_frontend_done_valid,a_frontend_done_ready,a_frontend_done_accept;
    logic[TAG_BITS-1:0]a_frontend_done_tag;logic a_frontend_done_had_event;
    logic a_mem_req_valid,a_mem_req_ready,a_mem_req_accept;
    logic[EPOCH_BITS-1:0]a_mem_req_epoch;logic[2:0]a_mem_req_slot;
    logic[GENERATION_BITS-1:0]a_mem_req_generation;
    logic[TAG_BITS-1:0]a_mem_req_tag;logic[2:0]a_mem_req_output_block;
    logic[2:0]a_mem_req_slice;logic[3:0]a_mem_req_source_count;
    logic[7:0]a_mem_req_bank_valid;
    logic[CHANNEL_BITS-1:0]a_mem_req_source_channel[0:7];
    logic a_mem_rsp_valid,a_mem_rsp_ready,a_mem_rsp_accept;
    logic[EPOCH_BITS-1:0]a_mem_rsp_epoch;logic[2:0]a_mem_rsp_slot;
    logic[GENERATION_BITS-1:0]a_mem_rsp_generation;
    logic[TAG_BITS-1:0]a_mem_rsp_tag;logic[7:0]a_mem_rsp_bank_valid;
    logic signed[7:0]a_mem_rsp_weight[0:7][0:LANES-1];
    logic a_result_valid,a_result_ready,a_result_accept;
    logic[TAG_BITS-1:0]a_result_tag;logic[2:0]a_result_output_block;
    logic[2:0]a_result_slice;logic signed[23:0]a_result_accumulator[0:LANES-1];
    logic a_result_last,a_token_done_valid,a_token_done_ready,a_token_done_accept;
    logic[TAG_BITS-1:0]a_token_done_tag;logic a_token_done_had_event;
    logic a_protocol_error,a_numeric_overflow,a_stale_response_seen,a_busy;
    logic[2:0]a_debug_fifo_count;logic[3:0]a_debug_outstanding_count;
    logic[31:0]a_debug_group_accept_count,a_debug_request_accept_count;
    logic[31:0]a_debug_response_accept_count,a_debug_context_write_count;
    logic[31:0]a_debug_result_accept_count,a_debug_active_bank_read_count;

    logic b_soft_flush,b_mem_flush_valid,b_mem_flush_ready;
    logic[EPOCH_BITS-1:0]b_mem_flush_epoch;
    logic b_mem_flush_ack_valid,b_mem_flush_ack_ready;
    logic[EPOCH_BITS-1:0]b_mem_flush_ack_epoch;
    logic b_header_valid,b_header_ready,b_header_accept;
    logic[TAG_BITS-1:0]b_header_tag;logic[3:0]b_header_output_blocks;
    logic b_group_valid,b_group_ready,b_group_accept;
    logic[TAG_BITS-1:0]b_group_tag;logic[2:0]b_group_output_block;
    logic[2:0]b_group_bank_id;logic[CHANNEL_BITS-1:0]b_group_source_channel;
    logic b_frontend_done_valid,b_frontend_done_ready,b_frontend_done_accept;
    logic[TAG_BITS-1:0]b_frontend_done_tag;logic b_frontend_done_had_event;
    logic b_mem_req_valid,b_mem_req_ready,b_mem_req_accept;
    logic[EPOCH_BITS-1:0]b_mem_req_epoch;logic[2:0]b_mem_req_slot;
    logic[GENERATION_BITS-1:0]b_mem_req_generation;
    logic[TAG_BITS-1:0]b_mem_req_tag;logic[2:0]b_mem_req_output_block;
    logic[2:0]b_mem_req_slice,b_mem_req_bank_id;
    logic[CHANNEL_BITS-1:0]b_mem_req_source_channel;
    logic b_mem_rsp_valid,b_mem_rsp_ready,b_mem_rsp_accept;
    logic[EPOCH_BITS-1:0]b_mem_rsp_epoch;logic[2:0]b_mem_rsp_slot;
    logic[GENERATION_BITS-1:0]b_mem_rsp_generation;
    logic[TAG_BITS-1:0]b_mem_rsp_tag;logic[2:0]b_mem_rsp_bank_id;
    logic signed[7:0]b_mem_rsp_weight[0:LANES-1];
    logic b_result_valid,b_result_ready,b_result_accept;
    logic[TAG_BITS-1:0]b_result_tag;logic[2:0]b_result_output_block;
    logic[2:0]b_result_slice;logic signed[23:0]b_result_accumulator[0:LANES-1];
    logic b_result_last,b_token_done_valid,b_token_done_ready,b_token_done_accept;
    logic[TAG_BITS-1:0]b_token_done_tag;logic b_token_done_had_event;
    logic b_protocol_error,b_numeric_overflow,b_stale_response_seen,b_busy;
    logic[2:0]b_debug_fifo_count;logic[3:0]b_debug_outstanding_count;
    logic[31:0]b_debug_group_accept_count,b_debug_request_accept_count;
    logic[31:0]b_debug_response_accept_count,b_debug_context_write_count;
    logic[31:0]b_debug_result_accept_count,b_debug_active_bank_read_count;

    m218_fc2_tagged_slice_service_island m218(
        .clk_core,.rst_core,.soft_flush(a_soft_flush),
        .mem_flush_valid(a_mem_flush_valid),.mem_flush_ready(a_mem_flush_ready),
        .mem_flush_epoch(a_mem_flush_epoch),.mem_flush_ack_valid(a_mem_flush_ack_valid),
        .mem_flush_ack_ready(a_mem_flush_ack_ready),.mem_flush_ack_epoch(a_mem_flush_ack_epoch),
        .header_valid(a_header_valid),.header_ready(a_header_ready),
        .header_tag(a_header_tag),.header_output_blocks(a_header_output_blocks),
        .header_accept(a_header_accept),.group_valid(a_group_valid),
        .group_ready(a_group_ready),.group_tag(a_group_tag),
        .group_output_block(a_group_output_block),.group_source_count(a_group_source_count),
        .group_bank_valid(a_group_bank_valid),.group_source_channel(a_group_source_channel),
        .group_accept(a_group_accept),.frontend_done_valid(a_frontend_done_valid),
        .frontend_done_ready(a_frontend_done_ready),.frontend_done_tag(a_frontend_done_tag),
        .frontend_done_had_event(a_frontend_done_had_event),
        .frontend_done_accept(a_frontend_done_accept),.mem_req_valid(a_mem_req_valid),
        .mem_req_ready(a_mem_req_ready),.mem_req_epoch(a_mem_req_epoch),
        .mem_req_slot(a_mem_req_slot),.mem_req_generation(a_mem_req_generation),
        .mem_req_tag(a_mem_req_tag),.mem_req_output_block(a_mem_req_output_block),
        .mem_req_slice(a_mem_req_slice),.mem_req_source_count(a_mem_req_source_count),
        .mem_req_bank_valid(a_mem_req_bank_valid),
        .mem_req_source_channel(a_mem_req_source_channel),.mem_req_accept(a_mem_req_accept),
        .mem_rsp_valid(a_mem_rsp_valid),.mem_rsp_ready(a_mem_rsp_ready),
        .mem_rsp_epoch(a_mem_rsp_epoch),.mem_rsp_slot(a_mem_rsp_slot),
        .mem_rsp_generation(a_mem_rsp_generation),.mem_rsp_tag(a_mem_rsp_tag),
        .mem_rsp_bank_valid(a_mem_rsp_bank_valid),.mem_rsp_weight(a_mem_rsp_weight),
        .mem_rsp_accept(a_mem_rsp_accept),.result_valid(a_result_valid),
        .result_ready(a_result_ready),.result_tag(a_result_tag),
        .result_output_block(a_result_output_block),.result_slice(a_result_slice),
        .result_accumulator(a_result_accumulator),.result_last(a_result_last),
        .result_accept(a_result_accept),.token_done_valid(a_token_done_valid),
        .token_done_ready(a_token_done_ready),.token_done_tag(a_token_done_tag),
        .token_done_had_event(a_token_done_had_event),.token_done_accept(a_token_done_accept),
        .protocol_error(a_protocol_error),.numeric_overflow(a_numeric_overflow),
        .stale_response_seen(a_stale_response_seen),.busy(a_busy),
        .debug_fifo_count(a_debug_fifo_count),.debug_outstanding_count(a_debug_outstanding_count),
        .debug_group_accept_count(a_debug_group_accept_count),
        .debug_request_accept_count(a_debug_request_accept_count),
        .debug_response_accept_count(a_debug_response_accept_count),
        .debug_context_write_count(a_debug_context_write_count),
        .debug_result_accept_count(a_debug_result_accept_count),
        .debug_active_bank_read_count(a_debug_active_bank_read_count));

    m219_fc2_k1_cropped_tagged_slice_service_island m219(
        .clk_core,.rst_core,.soft_flush(b_soft_flush),
        .mem_flush_valid(b_mem_flush_valid),.mem_flush_ready(b_mem_flush_ready),
        .mem_flush_epoch(b_mem_flush_epoch),.mem_flush_ack_valid(b_mem_flush_ack_valid),
        .mem_flush_ack_ready(b_mem_flush_ack_ready),.mem_flush_ack_epoch(b_mem_flush_ack_epoch),
        .header_valid(b_header_valid),.header_ready(b_header_ready),
        .header_tag(b_header_tag),.header_output_blocks(b_header_output_blocks),
        .header_accept(b_header_accept),.group_valid(b_group_valid),
        .group_ready(b_group_ready),.group_tag(b_group_tag),
        .group_output_block(b_group_output_block),.group_bank_id(b_group_bank_id),
        .group_source_channel(b_group_source_channel),.group_accept(b_group_accept),
        .frontend_done_valid(b_frontend_done_valid),
        .frontend_done_ready(b_frontend_done_ready),.frontend_done_tag(b_frontend_done_tag),
        .frontend_done_had_event(b_frontend_done_had_event),
        .frontend_done_accept(b_frontend_done_accept),.mem_req_valid(b_mem_req_valid),
        .mem_req_ready(b_mem_req_ready),.mem_req_epoch(b_mem_req_epoch),
        .mem_req_slot(b_mem_req_slot),.mem_req_generation(b_mem_req_generation),
        .mem_req_tag(b_mem_req_tag),.mem_req_output_block(b_mem_req_output_block),
        .mem_req_slice(b_mem_req_slice),.mem_req_bank_id(b_mem_req_bank_id),
        .mem_req_source_channel(b_mem_req_source_channel),.mem_req_accept(b_mem_req_accept),
        .mem_rsp_valid(b_mem_rsp_valid),.mem_rsp_ready(b_mem_rsp_ready),
        .mem_rsp_epoch(b_mem_rsp_epoch),.mem_rsp_slot(b_mem_rsp_slot),
        .mem_rsp_generation(b_mem_rsp_generation),.mem_rsp_tag(b_mem_rsp_tag),
        .mem_rsp_bank_id(b_mem_rsp_bank_id),.mem_rsp_weight(b_mem_rsp_weight),
        .mem_rsp_accept(b_mem_rsp_accept),.result_valid(b_result_valid),
        .result_ready(b_result_ready),.result_tag(b_result_tag),
        .result_output_block(b_result_output_block),.result_slice(b_result_slice),
        .result_accumulator(b_result_accumulator),.result_last(b_result_last),
        .result_accept(b_result_accept),.token_done_valid(b_token_done_valid),
        .token_done_ready(b_token_done_ready),.token_done_tag(b_token_done_tag),
        .token_done_had_event(b_token_done_had_event),.token_done_accept(b_token_done_accept),
        .protocol_error(b_protocol_error),.numeric_overflow(b_numeric_overflow),
        .stale_response_seen(b_stale_response_seen),.busy(b_busy),
        .debug_fifo_count(b_debug_fifo_count),.debug_outstanding_count(b_debug_outstanding_count),
        .debug_group_accept_count(b_debug_group_accept_count),
        .debug_request_accept_count(b_debug_request_accept_count),
        .debug_response_accept_count(b_debug_response_accept_count),
        .debug_context_write_count(b_debug_context_write_count),
        .debug_result_accept_count(b_debug_result_accept_count),
        .debug_active_bank_read_count(b_debug_active_bank_read_count));

    integer cycle_count,error_count,pair_count,numeric_cases,recurrence_checks;
    integer a_done_count,b_done_count,a_first_issue,b_first_issue;
    integer a_done_cycle,b_done_cycle,a_issue_count,b_issue_count;
    integer a_recurrence_h,b_recurrence_h,weight_mode;
    integer a_model_issue[0:4095],b_model_issue[0:4095];
    logic signed[23:0]reference_result[0:7][0:5][0:LANES-1];
    logic capture_a,compare_b;

    logic a_pending[0:7];integer a_due[0:7];
    logic[EPOCH_BITS-1:0]a_p_epoch[0:7];
    logic[GENERATION_BITS-1:0]a_p_gen[0:7];
    logic[TAG_BITS-1:0]a_p_tag[0:7];
    logic[2:0]a_p_block[0:7],a_p_slice[0:7];
    logic[7:0]a_p_mask[0:7];
    logic[CHANNEL_BITS-1:0]a_p_channel[0:7][0:7];
    integer a_rsp_sel;

    logic b_pending[0:7];integer b_due[0:7];
    logic[EPOCH_BITS-1:0]b_p_epoch[0:7];
    logic[GENERATION_BITS-1:0]b_p_gen[0:7];
    logic[TAG_BITS-1:0]b_p_tag[0:7];
    logic[2:0]b_p_block[0:7],b_p_slice[0:7],b_p_bank[0:7];
    logic[CHANNEL_BITS-1:0]b_p_channel[0:7];
    integer b_rsp_sel;

    function automatic integer max2(input integer x,input integer y);
        return x>y?x:y;
    endfunction
    function automatic integer signed weight_value(input integer bank,
        input integer lane,input integer channel,input integer block,input integer slice);
        integer value;
        begin
            if(weight_mode==1)begin
                if(bank==0)value=-128;
                else if(bank==1)value=127;
                else value=(bank[0]?-1:1);
            end else begin
                value=(channel*3+bank*5+block*7+slice*11+lane*13)%31;
                value=value-15;
            end
            return value;
        end
    endfunction

    always_comb begin
        a_rsp_sel=-1;
        for(integer slot=0;slot<8;slot++)if(a_pending[slot]
                &&a_due[slot]<=cycle_count+1
                &&(a_rsp_sel<0||a_p_gen[slot]<a_p_gen[a_rsp_sel]))a_rsp_sel=slot;
        a_mem_rsp_valid=a_rsp_sel>=0;
        a_mem_rsp_epoch=a_rsp_sel<0?0:a_p_epoch[a_rsp_sel];
        a_mem_rsp_slot=a_rsp_sel<0?0:a_rsp_sel;
        a_mem_rsp_generation=a_rsp_sel<0?0:a_p_gen[a_rsp_sel];
        a_mem_rsp_tag=a_rsp_sel<0?0:a_p_tag[a_rsp_sel];
        a_mem_rsp_bank_valid=a_rsp_sel<0?0:a_p_mask[a_rsp_sel];
        for(integer bank=0;bank<8;bank++)for(integer lane=0;lane<LANES;lane++)
            a_mem_rsp_weight[bank][lane]=(a_rsp_sel>=0&&a_p_mask[a_rsp_sel][bank])
                ?weight_value(bank,lane,a_p_channel[a_rsp_sel][bank],
                    a_p_block[a_rsp_sel],a_p_slice[a_rsp_sel]):0;

        b_rsp_sel=-1;
        for(integer slot=0;slot<8;slot++)if(b_pending[slot]
                &&b_due[slot]<=cycle_count+1
                &&(b_rsp_sel<0||b_p_gen[slot]<b_p_gen[b_rsp_sel]))b_rsp_sel=slot;
        b_mem_rsp_valid=b_rsp_sel>=0;
        b_mem_rsp_epoch=b_rsp_sel<0?0:b_p_epoch[b_rsp_sel];
        b_mem_rsp_slot=b_rsp_sel<0?0:b_rsp_sel;
        b_mem_rsp_generation=b_rsp_sel<0?0:b_p_gen[b_rsp_sel];
        b_mem_rsp_tag=b_rsp_sel<0?0:b_p_tag[b_rsp_sel];
        b_mem_rsp_bank_id=b_rsp_sel<0?0:b_p_bank[b_rsp_sel];
        for(integer lane=0;lane<LANES;lane++)
            b_mem_rsp_weight[lane]=b_rsp_sel>=0
                ?weight_value(b_p_bank[b_rsp_sel],lane,b_p_channel[b_rsp_sel],
                    b_p_block[b_rsp_sel],b_p_slice[b_rsp_sel]):0;
    end

    always @(posedge clk_core) begin
        integer expected,idx;
        if(rst_core)begin
            cycle_count=0;a_done_count=0;b_done_count=0;
            a_first_issue=-1;b_first_issue=-1;a_done_cycle=-1;b_done_cycle=-1;
            a_issue_count=0;b_issue_count=0;
            for(integer slot=0;slot<8;slot++)begin a_pending[slot]=0;b_pending[slot]=0;end
        end else begin
            cycle_count++;
            if(a_mem_rsp_accept)begin
                if(cycle_count!=a_due[a_mem_rsp_slot])begin
                    $error("M218 L4 response mismatch slot=%0d got=%0d due=%0d",
                        a_mem_rsp_slot,cycle_count,a_due[a_mem_rsp_slot]);error_count++;
                end
                a_pending[a_mem_rsp_slot]=0;
            end
            if(a_mem_req_accept)begin
                idx=a_issue_count;
                if(idx==0)begin a_first_issue=cycle_count;a_model_issue[idx]=0;end
                else begin
                    expected=a_model_issue[idx-1]+1;
                    if(idx>=O)expected=max2(expected,a_model_issue[idx-O]+L);
                    if(idx>=a_recurrence_h)expected=max2(expected,
                        a_model_issue[idx-a_recurrence_h]+L);
                    a_model_issue[idx]=expected;
                    if(cycle_count-a_first_issue!=expected)begin
                        $error("M218 recurrence idx=%0d got=%0d exp=%0d",
                            idx,cycle_count-a_first_issue,expected);error_count++;
                    end
                    recurrence_checks++;
                end
                if(a_pending[a_mem_req_slot]&&!(a_mem_rsp_accept
                        &&a_mem_rsp_slot==a_mem_req_slot))begin
                    $error("M218 live slot reuse %0d",a_mem_req_slot);error_count++;
                end
                a_pending[a_mem_req_slot]=1;a_due[a_mem_req_slot]=cycle_count+L;
                a_p_epoch[a_mem_req_slot]=a_mem_req_epoch;
                a_p_gen[a_mem_req_slot]=a_mem_req_generation;
                a_p_tag[a_mem_req_slot]=a_mem_req_tag;
                a_p_block[a_mem_req_slot]=a_mem_req_output_block;
                a_p_slice[a_mem_req_slot]=a_mem_req_slice;
                a_p_mask[a_mem_req_slot]=a_mem_req_bank_valid;
                for(integer bank=0;bank<8;bank++)
                    a_p_channel[a_mem_req_slot][bank]=a_mem_req_source_channel[bank];
                a_issue_count++;
            end
            if(b_mem_rsp_accept)begin
                if(cycle_count!=b_due[b_mem_rsp_slot])begin
                    $error("M219 L4 response mismatch slot=%0d got=%0d due=%0d",
                        b_mem_rsp_slot,cycle_count,b_due[b_mem_rsp_slot]);error_count++;
                end
                b_pending[b_mem_rsp_slot]=0;
            end
            if(b_mem_req_accept)begin
                idx=b_issue_count;
                if(idx==0)begin b_first_issue=cycle_count;b_model_issue[idx]=0;end
                else begin
                    expected=b_model_issue[idx-1]+1;
                    if(idx>=O)expected=max2(expected,b_model_issue[idx-O]+L);
                    if(idx>=b_recurrence_h)expected=max2(expected,
                        b_model_issue[idx-b_recurrence_h]+L);
                    b_model_issue[idx]=expected;
                    if(cycle_count-b_first_issue!=expected)begin
                        $error("M219 recurrence idx=%0d got=%0d exp=%0d h=%0d",
                            idx,cycle_count-b_first_issue,expected,b_recurrence_h);error_count++;
                    end
                    recurrence_checks++;
                end
                if(b_pending[b_mem_req_slot]&&!(b_mem_rsp_accept
                        &&b_mem_rsp_slot==b_mem_req_slot))begin
                    $error("M219 live slot reuse %0d",b_mem_req_slot);error_count++;
                end
                b_pending[b_mem_req_slot]=1;b_due[b_mem_req_slot]=cycle_count+L;
                b_p_epoch[b_mem_req_slot]=b_mem_req_epoch;
                b_p_gen[b_mem_req_slot]=b_mem_req_generation;
                b_p_tag[b_mem_req_slot]=b_mem_req_tag;
                b_p_block[b_mem_req_slot]=b_mem_req_output_block;
                b_p_slice[b_mem_req_slot]=b_mem_req_slice;
                b_p_bank[b_mem_req_slot]=b_mem_req_bank_id;
                b_p_channel[b_mem_req_slot]=b_mem_req_source_channel;
                b_issue_count++;
            end
            if(a_result_accept&&capture_a)for(integer lane=0;lane<LANES;lane++)
                reference_result[a_result_output_block][a_result_slice][lane]
                    =a_result_accumulator[lane];
            if(b_result_accept&&compare_b)for(integer lane=0;lane<LANES;lane++)
                if(b_result_accumulator[lane]!==
                        reference_result[b_result_output_block][b_result_slice][lane])begin
                    $error("cross-module mismatch b=%0d s=%0d l=%0d M218=%0d M219=%0d",
                        b_result_output_block,b_result_slice,lane,
                        reference_result[b_result_output_block][b_result_slice][lane],
                        b_result_accumulator[lane]);error_count++;
                end
            if(a_token_done_accept)begin a_done_count++;a_done_cycle=cycle_count;end
            if(b_token_done_accept)begin b_done_count++;b_done_cycle=cycle_count;end
        end
    end

    task automatic reset_pair;
        begin
            @(negedge clk_core);rst_core=1;
            a_header_valid=0;a_group_valid=0;a_frontend_done_valid=0;
            b_header_valid=0;b_group_valid=0;b_frontend_done_valid=0;
            repeat(3)@(negedge clk_core);rst_core=0;
            repeat(2)@(posedge clk_core);
        end
    endtask
    task automatic a_header(input logic[TAG_BITS-1:0]tag,input integer blocks);
        begin @(negedge clk_core);a_header_tag=tag;a_header_output_blocks=blocks;
            a_header_valid=1;while(!a_header_accept)@(posedge clk_core);
            @(negedge clk_core);a_header_valid=0;end
    endtask
    task automatic a_group(input logic[TAG_BITS-1:0]tag,input integer block,
        input integer sources);
        begin @(negedge clk_core);a_group_tag=tag;a_group_output_block=block;
            a_group_bank_valid=(1<<sources)-1;a_group_source_count=sources;
            for(integer bank=0;bank<8;bank++)a_group_source_channel[bank]=bank;
            a_group_valid=1;while(!a_group_accept)@(posedge clk_core);
            @(negedge clk_core);a_group_valid=0;end
    endtask
    task automatic a_done(input logic[TAG_BITS-1:0]tag);
        begin @(negedge clk_core);a_frontend_done_tag=tag;a_frontend_done_had_event=1;
            a_frontend_done_valid=1;while(!a_frontend_done_accept)@(posedge clk_core);
            @(negedge clk_core);a_frontend_done_valid=0;end
    endtask
    task automatic b_header(input logic[TAG_BITS-1:0]tag,input integer blocks);
        begin @(negedge clk_core);b_header_tag=tag;b_header_output_blocks=blocks;
            b_header_valid=1;while(!b_header_accept)@(posedge clk_core);
            @(negedge clk_core);b_header_valid=0;end
    endtask
    task automatic b_group(input logic[TAG_BITS-1:0]tag,input integer block,
        input integer bank);
        begin @(negedge clk_core);b_group_tag=tag;b_group_output_block=block;
            b_group_bank_id=bank;b_group_source_channel=bank;
            b_group_valid=1;while(!b_group_accept)@(posedge clk_core);
            @(negedge clk_core);b_group_valid=0;end
    endtask
    task automatic b_done(input logic[TAG_BITS-1:0]tag);
        begin @(negedge clk_core);b_frontend_done_tag=tag;b_frontend_done_had_event=1;
            b_frontend_done_valid=1;while(!b_frontend_done_accept)@(posedge clk_core);
            @(negedge clk_core);b_frontend_done_valid=0;end
    endtask

    task automatic run_pair(input integer blocks,input integer sources,
        input integer numeric_mode);
        logic[TAG_BITS-1:0]tag;integer a_req,a_read,a_res,b_req,b_read,b_res;
        integer a_cycles,b_cycles;
        begin
            tag=24'h220000|(blocks<<8)|(sources<<1)|numeric_mode;
            weight_mode=numeric_mode;capture_a=1;compare_b=0;
            reset_pair();a_recurrence_h=6*blocks;b_recurrence_h=6*blocks;
            fork
                begin a_header(tag,blocks);for(integer block=0;block<blocks;block++)
                    a_group(tag,block,sources);a_done(tag);end
                begin while(a_done_count==0)@(posedge clk_core);end
            join
            a_req=a_debug_request_accept_count;a_read=a_debug_active_bank_read_count;
            a_res=a_debug_result_accept_count;a_cycles=a_done_cycle-a_first_issue+1;
            if(a_debug_group_accept_count!=blocks||a_req!=6*blocks
                    ||a_read!=6*blocks*sources||a_res!=6*blocks
                    ||a_debug_response_accept_count!=a_req
                    ||a_debug_context_write_count!=a_req)begin
                $error("M218 conservation B=%0d N=%0d g=%0d req=%0d read=%0d res=%0d",
                    blocks,sources,a_debug_group_accept_count,a_req,a_read,a_res);error_count++;
            end

            capture_a=0;compare_b=1;reset_pair();
            a_recurrence_h=6*blocks;b_recurrence_h=6*blocks;
            fork
                begin b_header(tag,blocks);for(integer source=0;source<sources;source++)
                    for(integer block=0;block<blocks;block++)b_group(tag,block,source);
                    b_done(tag);end
                begin while(b_done_count==0)@(posedge clk_core);end
            join
            b_req=b_debug_request_accept_count;b_read=b_debug_active_bank_read_count;
            b_res=b_debug_result_accept_count;b_cycles=b_done_cycle-b_first_issue+1;
            if(b_debug_group_accept_count!=blocks*sources||b_req!=6*blocks*sources
                    ||b_read!=6*blocks*sources||b_res!=6*blocks
                    ||b_debug_response_accept_count!=b_req
                    ||b_debug_context_write_count!=b_req||a_read!=b_read||a_res!=b_res)begin
                $error("M219 conservation B=%0d N=%0d g=%0d req=%0d read=%0d res=%0d",
                    blocks,sources,b_debug_group_accept_count,b_req,b_read,b_res);error_count++;
            end
            if(a_protocol_error||a_numeric_overflow||b_protocol_error||b_numeric_overflow)begin
                $error("fault B=%0d N=%0d a=%0d/%0d b=%0d/%0d",blocks,sources,
                    a_protocol_error,a_numeric_overflow,b_protocol_error,b_numeric_overflow);error_count++;
            end
            pair_count++;if(numeric_mode==1)numeric_cases++;
            $display("M220 pair blocks=%0d sources=%0d mode=%0d M218_cycles=%0d M219_cycles=%0d reads=%0d",
                blocks,sources,numeric_mode,a_cycles,b_cycles,a_read);
        end
    endtask

    initial begin
        rst_core=1;error_count=0;pair_count=0;numeric_cases=0;recurrence_checks=0;
        capture_a=0;compare_b=0;weight_mode=0;
        a_soft_flush=0;a_mem_flush_ready=1;a_mem_flush_ack_valid=0;a_mem_flush_ack_epoch=0;
        b_soft_flush=0;b_mem_flush_ready=1;b_mem_flush_ack_valid=0;b_mem_flush_ack_epoch=0;
        a_header_valid=0;a_group_valid=0;a_frontend_done_valid=0;
        b_header_valid=0;b_group_valid=0;b_frontend_done_valid=0;
        a_mem_req_ready=1;b_mem_req_ready=1;a_result_ready=1;b_result_ready=1;
        a_token_done_ready=1;b_token_done_ready=1;
        for(integer slot=0;slot<8;slot++)begin a_pending[slot]=0;b_pending[slot]=0;end
        for(integer bank=0;bank<8;bank++)a_group_source_channel[bank]=bank;
        repeat(3)@(negedge clk_core);rst_core=0;repeat(2)@(posedge clk_core);
        for(integer bi=0;bi<4;bi++)begin
            integer blocks;blocks=(1<<bi);
            for(integer sources=1;sources<=8;sources++)run_pair(blocks,sources,0);
        end
        run_pair(1,2,1);
        if(pair_count!=33||numeric_cases!=1||recurrence_checks<3000)
            begin $error("coverage pairs=%0d numeric=%0d recurrence=%0d",
                pair_count,numeric_cases,recurrence_checks);error_count++;end
        if(error_count==0)$display("PASS M220 cross-module L4 miter pairs=33 numeric_cases=1 recurrence_checks=%0d M218_M219_bit_exact=true work_conserved=true",
            recurrence_checks);
        else $fatal(1,"M220 failures=%0d",error_count);
        $finish;
    end
endmodule

`default_nettype wire
