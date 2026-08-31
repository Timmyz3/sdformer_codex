`timescale 1ns/1ps
`default_nettype none

// M482 turns the M229 F2 service cut into a full-width, bank-timed gate.
// Four ordered credits overlap chunk-factor fetch, weight fetch and replay.
// Contexts are mapped by ctx%2 into two 4x(96xAcc19) 1R1W banks.  A read may
// coincide with the preceding write; a same-address collision is explicitly
// forwarded (write-first) so the result is independent of macro RDW mode.
module m482_fc1_l96_f2_c16_b2_full_overlap_island #(
    parameter int TAG_BITS=24,
    parameter int EPOCH_BITS=16,
    parameter int DESC_BITS=12,
    parameter int LANES=96,
    parameter int ACC_BITS=19,
    parameter int CREDITS=4
)(
    input  logic clk_core,input logic rst_core,

    input  logic header_valid,output logic header_ready,
    output logic header_accept,input logic[TAG_BITS-1:0]header_tag,
    input  logic[EPOCH_BITS-1:0]header_epoch,input logic header_factorized,
    input  logic[4:0]header_chunk_count,
    input  logic[DESC_BITS-1:0]header_descriptor_count,

    input  logic directory_valid,output logic directory_ready,
    output logic directory_accept,input logic[4:0]directory_chunk,
    input  logic[7:0]directory_descriptor_count,

    output logic factor_req_valid,input logic factor_req_ready,
    output logic factor_req_accept,output logic[1:0]factor_req_slot,
    output logic[TAG_BITS-1:0]factor_req_tag,
    output logic[EPOCH_BITS-1:0]factor_req_epoch,
    output logic[DESC_BITS-1:0]factor_req_descriptor,
    output logic[4:0]factor_req_chunk,output logic[6:0]factor_req_ordinal,
    input  logic factor_rsp_valid,output logic factor_rsp_ready,
    output logic factor_rsp_accept,input logic[1:0]factor_rsp_slot,
    input  logic[TAG_BITS-1:0]factor_rsp_tag,
    input  logic[EPOCH_BITS-1:0]factor_rsp_epoch,
    input  logic[DESC_BITS-1:0]factor_rsp_descriptor,
    input  logic[4:0]factor_rsp_chunk,input logic[6:0]factor_rsp_ordinal,
    input  logic[3:0]factor_rsp_source_offset,
    input  logic[7:0]factor_rsp_context_mask,
    input  logic[7:0]factor_rsp_sign_mask,

    output logic weight_req_valid,input logic weight_req_ready,
    output logic weight_req_accept,output logic[1:0]weight_req_slot,
    output logic[TAG_BITS-1:0]weight_req_tag,
    output logic[EPOCH_BITS-1:0]weight_req_epoch,
    output logic[8:0]weight_req_source,
    input  logic weight_rsp_valid,output logic weight_rsp_ready,
    output logic weight_rsp_accept,input logic[1:0]weight_rsp_slot,
    input  logic[TAG_BITS-1:0]weight_rsp_tag,
    input  logic[EPOCH_BITS-1:0]weight_rsp_epoch,
    input  logic[8:0]weight_rsp_source,
    input  logic[LANES*8-1:0]weight_rsp_data,

    // Arbitration cut for the two physical bank read ports.  The write ports
    // are owned by this island and may operate concurrently with these reads.
    output logic[1:0]acc_bank_read_valid,input logic[1:0]acc_bank_read_ready,
    output logic[1:0][1:0]acc_bank_read_row,
    output logic[1:0][2:0]acc_bank_read_context,
    output logic acc_issue_accept,

    output logic commit_valid,input logic commit_ready,
    output logic commit_accept,output logic[TAG_BITS-1:0]commit_tag,
    output logic[EPOCH_BITS-1:0]commit_epoch,
    output logic[2:0]commit_context,output logic commit_last,
    output logic[LANES*ACC_BITS-1:0]commit_data,
    output logic done_valid,input logic done_ready,output logic done_accept,
    output logic[TAG_BITS-1:0]done_tag,
    output logic[EPOCH_BITS-1:0]done_epoch,output logic done_empty_bypass,

    output logic protocol_error,output logic numeric_overflow,
    output logic busy,output logic[31:0]debug_tile_cycles,
    output logic[31:0]debug_factor_requests,
    output logic[31:0]debug_weight_requests,
    output logic[31:0]debug_issue_rounds,
    output logic[31:0]debug_context_updates,
    output logic[31:0]debug_bank_conflict_extra_rounds,
    output logic[31:0]debug_factor_weight_overlap,
    output logic[31:0]debug_weight_update_overlap,
    output logic[31:0]debug_triple_overlap,
    output logic[31:0]debug_same_bank_rdw,
    output logic[31:0]debug_same_address_forward,
    output logic[2:0]debug_credit_count
);
    localparam int CONTEXTS=8,CHUNKS=24,BANKS=2;
    localparam bit PARAMETERS_LEGAL=TAG_BITS==24&&EPOCH_BITS==16
        &&DESC_BITS==12&&LANES==96&&ACC_BITS==19&&CREDITS==4;
    typedef enum logic[3:0]{ST_IDLE,ST_INIT,ST_DIRECTORY,ST_RUN,
        ST_COMMIT_REQ,ST_COMMIT_WAIT,ST_COMMIT_SEND,ST_DONE,ST_FAULT}state_t;
    state_t state_q;

    logic[TAG_BITS-1:0]tag_q;logic[EPOCH_BITS-1:0]epoch_q;
    logic factorized_q,empty_q,fault_q;
    logic[4:0]chunk_count_q,directory_index_q,issue_chunk_q;
    logic[6:0]issue_ordinal_q;
    logic[DESC_BITS-1:0]descriptor_count_q,descriptor_issue_q,
        factor_response_q;
    logic[7:0]directory_count_q[0:CHUNKS-1];
    logic[DESC_BITS:0]directory_sum_q;
    logic[1:0]init_row_q,head_q,tail_q;
    logic[2:0]slot_count_q;

    // M481 budgets 128 descriptor entries.  Keep factor latency out of the
    // four M229 held-weight credits so both modes use that resource fairly.
    logic[6:0]factor_fifo_head_q,factor_fifo_tail_q;
    logic[8:0]factor_fifo_count_q,factor_inflight_q;
    logic[DESC_BITS-1:0]factor_fifo_descriptor_q[0:127];
    logic[4:0]factor_fifo_chunk_q[0:127];
    logic[3:0]factor_fifo_offset_q[0:127];
    logic[7:0]factor_fifo_context_q[0:127],factor_fifo_sign_q[0:127];

    logic slot_valid_q[0:CREDITS-1],slot_factor_valid_q[0:CREDITS-1];
    logic slot_weight_requested_q[0:CREDITS-1];
    logic slot_weight_valid_q[0:CREDITS-1];
    logic[DESC_BITS-1:0]slot_descriptor_q[0:CREDITS-1];
    logic[4:0]slot_chunk_q[0:CREDITS-1];
    logic[6:0]slot_ordinal_q[0:CREDITS-1];
    logic[3:0]slot_offset_q[0:CREDITS-1];
    logic[7:0]slot_context_q[0:CREDITS-1],slot_sign_q[0:CREDITS-1];
    logic[LANES*8-1:0]slot_weight_q[0:CREDITS-1];

    logic request_found;logic[1:0]request_slot;
    logic[7:0]head_pending,pending_after;
    logic[1:0]selected_valid;logic[2:0]selected_context[0:1];
    logic[1:0]selected_count;
    logic update_issue_valid,update_issue_ready,pop_head,push_slot;
    logic allocate_slot,push_factor,pop_factor;

    logic signed[ACC_BITS-1:0]acc_bank_q[0:BANKS-1][0:3][0:LANES-1];
    logic[1:0]read_pipe_valid_q;logic[1:0][1:0]read_pipe_row_q;
    logic[1:0][2:0]read_pipe_context_q;logic[1:0]read_pipe_sign_q;
    logic signed[ACC_BITS-1:0]read_pipe_data_q[0:1][0:LANES-1];
    logic signed[7:0]read_pipe_weight_q[0:1][0:LANES-1];
    logic signed[ACC_BITS-1:0]write_value[0:1][0:LANES-1];
    logic overflow_this_cycle;

    logic[2:0]commit_context_q;logic[LANES*ACC_BITS-1:0]commit_data_q;
    logic[31:0]tile_cycles_q,factor_requests_q,weight_requests_q;
    logic[31:0]issue_rounds_q,context_updates_q,bank_conflict_q;
    logic[31:0]factor_weight_overlap_q,weight_update_overlap_q;
    logic[31:0]triple_overlap_q,same_bank_rdw_q,same_address_forward_q;
    logic illegal_header,illegal_directory,illegal_factor_response;
    logic illegal_weight_response,directory_complete_legal;
    logic all_work_done;

    function automatic logic[4:0]next_nonempty_chunk(
            input logic[4:0]start,input logic[4:0]limit);
        logic found;logic[4:0]value;
        begin found=0;value=limit;
            for(int chunk=0;chunk<CHUNKS;chunk++)
                if(!found&&chunk>=start&&chunk<limit
                        &&directory_count_q[chunk]!=0)begin
                    found=1;value=chunk[4:0];
                end
            return value;
        end
    endfunction

    function automatic logic[3:0]popcount8(input logic[7:0]value);
        logic[3:0]count;
        begin count=0;for(int bit_index=0;bit_index<8;bit_index++)
            count=count+value[bit_index];return count;end
    endfunction

    function automatic logic[2:0]bank_rounds(input logic[7:0]value);
        logic[2:0]even_count,odd_count;
        begin even_count=0;odd_count=0;
            for(int ctx=0;ctx<8;ctx++)if(value[ctx])begin
                if(ctx[0])odd_count=odd_count+1'b1;
                else even_count=even_count+1'b1;
            end
            return even_count>odd_count?even_count:odd_count;
        end
    endfunction

    generate if(!PARAMETERS_LEGAL)begin:g_bad
        initial $fatal(1,"M482 frozen L96_F2_C16_B2 geometry drift");
    end endgenerate

    always_comb begin:weight_request_select
        request_found=0;request_slot=0;
        for(int slot=0;slot<CREDITS;slot++)
            if(!request_found&&slot_valid_q[slot]&&slot_factor_valid_q[slot]
                    &&!slot_weight_requested_q[slot])begin
                request_found=1;request_slot=slot[1:0];
            end
    end

    // F2 is bank aware: slot zero selects the oldest even context and slot
    // one the oldest odd context.  Same-parity contexts are serialized.
    always_comb begin:bank_aware_select
        logic found_even,found_odd;
        head_pending=(slot_count_q!=0&&slot_valid_q[head_q])
            ?slot_context_q[head_q]:'0;
        pending_after=head_pending;selected_valid='0;selected_count=0;
        selected_context[0]='0;selected_context[1]='0;
        found_even=0;found_odd=0;
        for(int ctx=0;ctx<8;ctx++)if(head_pending[ctx])begin
            if(!ctx[0]&&!found_even)begin
                found_even=1;selected_valid[0]=1;
                selected_context[0]=ctx[2:0];pending_after[ctx]=0;
                selected_count=selected_count+1'b1;
            end else if(ctx[0]&&!found_odd)begin
                found_odd=1;selected_valid[1]=1;
                selected_context[1]=ctx[2:0];pending_after[ctx]=0;
                selected_count=selected_count+1'b1;
            end
        end
    end

    always_comb begin:numeric_write
        overflow_this_cycle=0;
        for(int bank=0;bank<BANKS;bank++)for(int lane=0;lane<LANES;lane++)begin
            logic signed[ACC_BITS:0]sum_ext;
            if(read_pipe_sign_q[bank])
                sum_ext=$signed(read_pipe_data_q[bank][lane])
                    -$signed(read_pipe_weight_q[bank][lane]);
            else sum_ext=$signed(read_pipe_data_q[bank][lane])
                    +$signed(read_pipe_weight_q[bank][lane]);
            write_value[bank][lane]=sum_ext[ACC_BITS-1:0];
            if(read_pipe_valid_q[bank]
                    &&sum_ext[ACC_BITS]!=sum_ext[ACC_BITS-1])
                overflow_this_cycle=1;
        end
    end

    always_comb begin:legality
        illegal_header=header_valid&&state_q!=ST_IDLE;
        // The producer may present the next directory beat early and hold it
        // until ready.  Shape/order is checked only while this phase owns it.
        illegal_directory=directory_valid&&state_q==ST_DIRECTORY
            &&!(directory_chunk==directory_index_q
            &&directory_chunk<chunk_count_q
            &&directory_descriptor_count<=128);
        illegal_factor_response=factor_rsp_valid&&!(state_q==ST_RUN
            &&factor_rsp_slot==factor_rsp_descriptor[1:0]
            &&factor_rsp_tag==tag_q&&factor_rsp_epoch==epoch_q
            &&factor_rsp_descriptor==factor_response_q
            &&factor_rsp_chunk<chunk_count_q
            &&factor_rsp_ordinal<directory_count_q[factor_rsp_chunk]
            &&factor_rsp_source_offset<16&&factor_rsp_context_mask!=0
            &&!(|(factor_rsp_sign_mask&~factor_rsp_context_mask))
            &&(factorized_q||$onehot(factor_rsp_context_mask)));
        illegal_weight_response=weight_rsp_valid&&!(state_q==ST_RUN
            &&weight_rsp_slot<CREDITS&&slot_valid_q[weight_rsp_slot]
            &&slot_factor_valid_q[weight_rsp_slot]
            &&slot_weight_requested_q[weight_rsp_slot]
            &&!slot_weight_valid_q[weight_rsp_slot]
            &&weight_rsp_tag==tag_q&&weight_rsp_epoch==epoch_q
            &&weight_rsp_source=={slot_chunk_q[weight_rsp_slot],
                                   slot_offset_q[weight_rsp_slot]});
        directory_complete_legal=(directory_sum_q
            +directory_descriptor_count)==descriptor_count_q;
    end

    always_comb begin:interfaces
        header_ready=state_q==ST_IDLE&&!fault_q
            &&(!header_valid||((header_descriptor_count==0
                    &&header_chunk_count==0)
                ||(header_descriptor_count!=0&&header_chunk_count>0
                    &&header_chunk_count<=CHUNKS)));
        header_accept=header_valid&&header_ready;
        directory_ready=state_q==ST_DIRECTORY&&!fault_q&&!illegal_directory;
        directory_accept=directory_valid&&directory_ready;

        update_issue_valid=state_q==ST_RUN&&slot_count_q!=0
            &&slot_valid_q[head_q]&&slot_weight_valid_q[head_q]
            &&head_pending!=0&&!fault_q;
        acc_bank_read_valid=update_issue_valid?selected_valid:'0;
        for(int bank=0;bank<BANKS;bank++)begin
            acc_bank_read_context[bank]=selected_context[bank];
            acc_bank_read_row[bank]=selected_context[bank][2:1];
        end
        update_issue_ready=(&(~selected_valid|acc_bank_read_ready));
        acc_issue_accept=update_issue_valid&&update_issue_ready;
        pop_head=acc_issue_accept&&pending_after==0;

        factor_req_valid=state_q==ST_RUN&&descriptor_issue_q<descriptor_count_q
            &&factor_fifo_count_q+factor_inflight_q<128&&!fault_q;
        factor_req_slot=descriptor_issue_q[1:0];factor_req_tag=tag_q;
        factor_req_epoch=epoch_q;
        factor_req_descriptor=descriptor_issue_q;
        factor_req_chunk=issue_chunk_q;factor_req_ordinal=issue_ordinal_q;
        factor_req_accept=factor_req_valid&&factor_req_ready;
        factor_rsp_ready=state_q==ST_RUN&&!fault_q&&!illegal_factor_response;
        factor_rsp_accept=factor_rsp_valid&&factor_rsp_ready;
        push_factor=factor_rsp_accept;

        allocate_slot=state_q==ST_RUN&&factor_fifo_count_q!=0
            &&(slot_count_q<CREDITS||pop_head)&&!fault_q;
        pop_factor=allocate_slot;push_slot=allocate_slot;

        weight_req_valid=state_q==ST_RUN&&request_found&&!fault_q;
        weight_req_slot=request_slot;weight_req_tag=tag_q;
        weight_req_epoch=epoch_q;
        weight_req_source=request_found
            ?{slot_chunk_q[request_slot],slot_offset_q[request_slot]}:'0;
        weight_req_accept=weight_req_valid&&weight_req_ready;
        weight_rsp_ready=state_q==ST_RUN&&!fault_q&&!illegal_weight_response;
        weight_rsp_accept=weight_rsp_valid&&weight_rsp_ready;

        commit_valid=state_q==ST_COMMIT_SEND&&!fault_q;
        commit_accept=commit_valid&&commit_ready;commit_tag=tag_q;
        commit_epoch=epoch_q;commit_context=commit_context_q;
        commit_last=commit_context_q==7;commit_data=commit_data_q;
        done_valid=state_q==ST_DONE&&!fault_q;done_accept=done_valid&&done_ready;
        done_tag=tag_q;done_epoch=epoch_q;done_empty_bypass=empty_q;
        protocol_error=fault_q||illegal_header||illegal_directory
            ||illegal_factor_response||illegal_weight_response
            ||overflow_this_cycle;
        numeric_overflow=overflow_this_cycle;
        busy=state_q!=ST_IDLE;
        all_work_done=descriptor_issue_q==descriptor_count_q
            &&factor_response_q==descriptor_count_q&&factor_inflight_q==0
            &&factor_fifo_count_q==0&&slot_count_q==0
            &&read_pipe_valid_q==0;
        debug_tile_cycles=tile_cycles_q;
        debug_factor_requests=factor_requests_q;
        debug_weight_requests=weight_requests_q;
        debug_issue_rounds=issue_rounds_q;
        debug_context_updates=context_updates_q;
        debug_bank_conflict_extra_rounds=bank_conflict_q;
        debug_factor_weight_overlap=factor_weight_overlap_q;
        debug_weight_update_overlap=weight_update_overlap_q;
        debug_triple_overlap=triple_overlap_q;
        debug_same_bank_rdw=same_bank_rdw_q;
        debug_same_address_forward=same_address_forward_q;
        debug_credit_count=slot_count_q;
    end

    always_ff @(posedge clk_core)begin:state_and_storage
        if(rst_core)begin
            state_q<=ST_IDLE;tag_q<='0;epoch_q<='0;factorized_q<=0;
            empty_q<=0;fault_q<=0;chunk_count_q<=0;directory_index_q<=0;
            directory_sum_q<=0;descriptor_count_q<=0;descriptor_issue_q<=0;
            factor_response_q<=0;issue_chunk_q<=0;issue_ordinal_q<=0;
            init_row_q<=0;head_q<=0;tail_q<=0;slot_count_q<=0;
            factor_fifo_head_q<=0;factor_fifo_tail_q<=0;
            factor_fifo_count_q<=0;factor_inflight_q<=0;
            read_pipe_valid_q<=0;commit_context_q<=0;commit_data_q<=0;
            tile_cycles_q<=0;factor_requests_q<=0;weight_requests_q<=0;
            issue_rounds_q<=0;context_updates_q<=0;bank_conflict_q<=0;
            factor_weight_overlap_q<=0;weight_update_overlap_q<=0;
            triple_overlap_q<=0;same_bank_rdw_q<=0;
            same_address_forward_q<=0;
            for(int chunk=0;chunk<CHUNKS;chunk++)directory_count_q[chunk]<=0;
            for(int slot=0;slot<CREDITS;slot++)begin
                slot_valid_q[slot]<=0;slot_factor_valid_q[slot]<=0;
                slot_weight_requested_q[slot]<=0;slot_weight_valid_q[slot]<=0;
                slot_descriptor_q[slot]<=0;slot_chunk_q[slot]<=0;
                slot_ordinal_q[slot]<=0;slot_offset_q[slot]<=0;
                slot_context_q[slot]<=0;slot_sign_q[slot]<=0;
                slot_weight_q[slot]<=0;
            end
        end else begin
            if(state_q!=ST_IDLE)tile_cycles_q<=tile_cycles_q+1'b1;
            if(illegal_header||illegal_directory||illegal_factor_response
                    ||illegal_weight_response||overflow_this_cycle)begin
                fault_q<=1;state_q<=ST_FAULT;
            end else begin
                // Acc19 writeback and a simultaneous next read are legal on
                // each independent 1R1W bank.  Same-address reads forward the
                // value being written instead of relying on macro semantics.
                for(int bank=0;bank<BANKS;bank++)begin
                    if(read_pipe_valid_q[bank])
                        for(int lane=0;lane<LANES;lane++)
                            acc_bank_q[bank][read_pipe_row_q[bank]][lane]
                                <=write_value[bank][lane];
                    if(acc_issue_accept&&selected_valid[bank])begin
                        for(int lane=0;lane<LANES;lane++)begin
                            if(read_pipe_valid_q[bank]
                                    &&read_pipe_row_q[bank]
                                      ==selected_context[bank][2:1])
                                read_pipe_data_q[bank][lane]
                                    <=write_value[bank][lane];
                            else read_pipe_data_q[bank][lane]
                                    <=acc_bank_q[bank]
                                        [selected_context[bank][2:1]][lane];
                            read_pipe_weight_q[bank][lane]
                                <=slot_weight_q[head_q][lane*8+:8];
                        end
                        read_pipe_row_q[bank]<=selected_context[bank][2:1];
                        read_pipe_context_q[bank]<=selected_context[bank];
                        read_pipe_sign_q[bank]
                            <=slot_sign_q[head_q][selected_context[bank]];
                    end
                    read_pipe_valid_q[bank]
                        <=acc_issue_accept&&selected_valid[bank];
                    if(read_pipe_valid_q[bank]&&acc_issue_accept
                            &&selected_valid[bank])begin
                        same_bank_rdw_q<=same_bank_rdw_q+1'b1;
                        if(read_pipe_row_q[bank]
                                ==selected_context[bank][2:1])
                            same_address_forward_q
                                <=same_address_forward_q+1'b1;
                    end
                end

                case(state_q)
                    ST_IDLE:if(header_accept)begin
                        tag_q<=header_tag;epoch_q<=header_epoch;
                        factorized_q<=header_factorized;
                        chunk_count_q<=header_chunk_count;
                        descriptor_count_q<=header_descriptor_count;
                        descriptor_issue_q<=0;factor_response_q<=0;
                        directory_index_q<=0;directory_sum_q<=0;
                        issue_chunk_q<=0;issue_ordinal_q<=0;init_row_q<=0;
                        head_q<=0;tail_q<=0;slot_count_q<=0;
                        factor_fifo_head_q<=0;factor_fifo_tail_q<=0;
                        factor_fifo_count_q<=0;factor_inflight_q<=0;
                        read_pipe_valid_q<=0;commit_context_q<=0;
                        tile_cycles_q<=1;factor_requests_q<=0;
                        weight_requests_q<=0;issue_rounds_q<=0;
                        context_updates_q<=0;bank_conflict_q<=0;
                        factor_weight_overlap_q<=0;weight_update_overlap_q<=0;
                        triple_overlap_q<=0;same_bank_rdw_q<=0;
                        same_address_forward_q<=0;fault_q<=0;
                        for(int chunk=0;chunk<CHUNKS;chunk++)
                            directory_count_q[chunk]<=0;
                        for(int slot=0;slot<CREDITS;slot++)begin
                            slot_valid_q[slot]<=0;slot_factor_valid_q[slot]<=0;
                            slot_weight_requested_q[slot]<=0;
                            slot_weight_valid_q[slot]<=0;
                        end
                        if(header_descriptor_count==0)begin
                            empty_q<=1;state_q<=ST_DONE;
                        end else begin empty_q<=0;state_q<=ST_INIT;end
                    end
                    ST_INIT:begin
                        for(int bank=0;bank<BANKS;bank++)
                            for(int lane=0;lane<LANES;lane++)
                                acc_bank_q[bank][init_row_q][lane]<=0;
                        if(init_row_q==3)begin
                            directory_index_q<=0;state_q<=ST_DIRECTORY;
                        end else init_row_q<=init_row_q+1'b1;
                    end
                    ST_DIRECTORY:if(directory_accept)begin
                        directory_count_q[directory_chunk]
                            <=directory_descriptor_count;
                        directory_sum_q<=directory_sum_q
                            +directory_descriptor_count;
                        if(directory_index_q==chunk_count_q-1'b1)begin
                            if(!directory_complete_legal)begin
                                fault_q<=1;state_q<=ST_FAULT;
                            end else begin
                                if(next_nonempty_chunk(0,chunk_count_q)
                                        ==chunk_count_q)
                                    issue_chunk_q<=directory_chunk;
                                else issue_chunk_q<=next_nonempty_chunk(
                                    0,chunk_count_q);
                                issue_ordinal_q<=0;state_q<=ST_RUN;
                            end
                        end else directory_index_q<=directory_index_q+1'b1;
                    end
                    ST_RUN:begin
                        if(factor_req_accept)begin
                            descriptor_issue_q<=descriptor_issue_q+1'b1;
                            factor_requests_q<=factor_requests_q+1'b1;
                            if(issue_ordinal_q+1'b1
                                    ==directory_count_q[issue_chunk_q])begin
                                issue_chunk_q<=next_nonempty_chunk(
                                    issue_chunk_q+1'b1,chunk_count_q);
                                issue_ordinal_q<=0;
                            end else issue_ordinal_q<=issue_ordinal_q+1'b1;
                        end
                        if(factor_rsp_accept)begin
                            factor_fifo_descriptor_q[factor_fifo_tail_q]
                                <=factor_rsp_descriptor;
                            factor_fifo_chunk_q[factor_fifo_tail_q]
                                <=factor_rsp_chunk;
                            factor_fifo_offset_q[factor_fifo_tail_q]
                                <=factor_rsp_source_offset;
                            factor_fifo_context_q[factor_fifo_tail_q]
                                <=factor_rsp_context_mask;
                            factor_fifo_sign_q[factor_fifo_tail_q]
                                <=factor_rsp_sign_mask;
                            factor_fifo_tail_q<=factor_fifo_tail_q+1'b1;
                            factor_response_q<=factor_response_q+1'b1;
                            bank_conflict_q<=bank_conflict_q
                                +bank_rounds(factor_rsp_context_mask)
                                -((popcount8(factor_rsp_context_mask)+1)>>1);
                        end
                        if(allocate_slot)begin
                            slot_valid_q[tail_q]<=1;
                            slot_factor_valid_q[tail_q]<=1;
                            slot_weight_requested_q[tail_q]<=0;
                            slot_weight_valid_q[tail_q]<=0;
                            slot_descriptor_q[tail_q]
                                <=factor_fifo_descriptor_q[factor_fifo_head_q];
                            slot_chunk_q[tail_q]
                                <=factor_fifo_chunk_q[factor_fifo_head_q];
                            slot_ordinal_q[tail_q]<=0;
                            slot_offset_q[tail_q]
                                <=factor_fifo_offset_q[factor_fifo_head_q];
                            slot_context_q[tail_q]
                                <=factor_fifo_context_q[factor_fifo_head_q];
                            slot_sign_q[tail_q]
                                <=factor_fifo_sign_q[factor_fifo_head_q];
                            factor_fifo_head_q<=factor_fifo_head_q+1'b1;
                            tail_q<=tail_q+1'b1;
                        end
                        case({factor_req_accept,factor_rsp_accept})
                            2'b10:factor_inflight_q<=factor_inflight_q+1'b1;
                            2'b01:factor_inflight_q<=factor_inflight_q-1'b1;
                            default:factor_inflight_q<=factor_inflight_q;
                        endcase
                        case({push_factor,pop_factor})
                            2'b10:factor_fifo_count_q
                                <=factor_fifo_count_q+1'b1;
                            2'b01:factor_fifo_count_q
                                <=factor_fifo_count_q-1'b1;
                            default:factor_fifo_count_q<=factor_fifo_count_q;
                        endcase
                        if(weight_req_accept)begin
                            slot_weight_requested_q[weight_req_slot]<=1;
                            weight_requests_q<=weight_requests_q+1'b1;
                        end
                        if(weight_rsp_accept)begin
                            slot_weight_valid_q[weight_rsp_slot]<=1;
                            slot_weight_q[weight_rsp_slot]<=weight_rsp_data;
                        end
                        if(acc_issue_accept&&!(pop_head&&allocate_slot
                                &&tail_q==head_q))begin
                            slot_context_q[head_q]<=pending_after;
                            issue_rounds_q<=issue_rounds_q+1'b1;
                        end else if(acc_issue_accept)
                            issue_rounds_q<=issue_rounds_q+1'b1;
                        if(|read_pipe_valid_q)
                            context_updates_q<=context_updates_q
                                +read_pipe_valid_q[0]+read_pipe_valid_q[1];
                        if(pop_head&&!(push_slot&&tail_q==head_q))begin
                            slot_valid_q[head_q]<=0;
                            slot_factor_valid_q[head_q]<=0;
                            slot_weight_requested_q[head_q]<=0;
                            slot_weight_valid_q[head_q]<=0;
                        end
                        if(pop_head)head_q<=head_q+1'b1;
                        case({push_slot,pop_head})
                            2'b10:slot_count_q<=slot_count_q+1'b1;
                            2'b01:slot_count_q<=slot_count_q-1'b1;
                            default:slot_count_q<=slot_count_q;
                        endcase
                        if(factor_req_accept&&weight_req_accept)
                            factor_weight_overlap_q
                                <=factor_weight_overlap_q+1'b1;
                        if(weight_req_accept&&acc_issue_accept)
                            weight_update_overlap_q
                                <=weight_update_overlap_q+1'b1;
                        if(factor_req_accept&&weight_req_accept
                                &&acc_issue_accept)
                            triple_overlap_q<=triple_overlap_q+1'b1;
                        if(all_work_done)begin
                            commit_context_q<=0;state_q<=ST_COMMIT_REQ;
                        end
                    end
                    ST_COMMIT_REQ:begin
                        for(int lane=0;lane<LANES;lane++)
                            commit_data_q[lane*ACC_BITS+:ACC_BITS]
                                <=acc_bank_q[commit_context_q[0]]
                                    [commit_context_q[2:1]][lane];
                        state_q<=ST_COMMIT_WAIT;
                    end
                    ST_COMMIT_WAIT:state_q<=ST_COMMIT_SEND;
                    ST_COMMIT_SEND:if(commit_accept)begin
                        if(commit_context_q==7)state_q<=ST_DONE;
                        else begin commit_context_q<=commit_context_q+1'b1;
                            state_q<=ST_COMMIT_REQ;end
                    end
                    ST_DONE:if(done_accept)state_q<=ST_IDLE;
                    default:state_q<=ST_FAULT;
                endcase
            end
        end
    end
endmodule
`default_nettype wire
