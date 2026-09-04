`timescale 1ns/1ps
`default_nettype none

// M228 replaces M227's flat 384-way source select with twelve 32-source
// bitplanes.  Each 256-bit scan beat carries 32 presence bits for all eight
// contexts, so the producer performs the K8 transpose while activity is born.
// A 12-way row summary and a 32-way in-row walk replace the flat priority/mux.
module m228_fc1_k8_bitplane_held_weight_slice #(
    parameter int FANOUT = 1,
    parameter int TAG_BITS = 24,
    parameter int EPOCH_BITS = 16,
    parameter int LANES = 96,
    parameter int ACC_BITS = 19
) (
    input logic clk_core, input logic rst_core,
    input logic begin_valid, output logic begin_ready,
    input logic [TAG_BITS-1:0] begin_tag,
    input logic [EPOCH_BITS-1:0] begin_epoch,
    output logic begin_accept,

    input logic scan_valid, output logic scan_ready,
    input logic [3:0] scan_chunk,
    input logic [255:0] scan_presence,
    input logic [255:0] scan_sign,
    output logic scan_accept,
    input logic scan_done_valid, output logic scan_done_ready,
    output logic scan_done_accept,

    output logic weight_req_valid, input logic weight_req_ready,
    output logic [TAG_BITS-1:0] weight_req_tag,
    output logic [EPOCH_BITS-1:0] weight_req_epoch,
    output logic [8:0] weight_req_source,
    output logic weight_req_accept,
    input logic weight_rsp_valid, output logic weight_rsp_ready,
    input logic [TAG_BITS-1:0] weight_rsp_tag,
    input logic [EPOCH_BITS-1:0] weight_rsp_epoch,
    input logic [8:0] weight_rsp_source,
    input logic [LANES*8-1:0] weight_rsp_data,
    output logic weight_rsp_accept,

    output logic result_valid, input logic result_ready,
    output logic [TAG_BITS-1:0] result_tag,
    output logic [EPOCH_BITS-1:0] result_epoch,
    output logic [2:0] result_context,
    output logic [LANES*ACC_BITS-1:0] result_accumulator,
    output logic result_last, output logic result_accept,
    output logic done_valid, input logic done_ready,
    output logic [TAG_BITS-1:0] done_tag,
    output logic [EPOCH_BITS-1:0] done_epoch,
    output logic done_accept,

    output logic protocol_error, output logic numeric_overflow,
    output logic busy, output logic [3:0] debug_scan_count,
    output logic [9:0] debug_unique_sources,
    output logic [11:0] debug_context_updates,
    output logic [9:0] debug_weight_reads,
    output logic [2:0] debug_replay_width
);
    localparam int CONTEXTS=8, CHUNKS=12, SOURCES_PER_CHUNK=32;
    localparam bit PARAMETERS_LEGAL=(FANOUT==1||FANOUT==2||FANOUT==4)
        && TAG_BITS==24 && EPOCH_BITS==16 && LANES==96 && ACC_BITS==19;
    typedef enum logic[2:0]{ST_IDLE,ST_SCAN,ST_REQUEST,ST_WAIT,
        ST_REPLAY,ST_DRAIN,ST_DONE}state_t;

    state_t state_q;
    logic fault_q;
    logic[TAG_BITS-1:0]tag_q;logic[EPOCH_BITS-1:0]epoch_q;
    logic[11:0]scan_seen_q,row_summary_q,remaining_rows_q;
    logic[3:0]scan_count_q,current_chunk_q;
    logic[4:0]current_bit_q;
    logic[31:0]row_remaining_q;
    logic[255:0]presence_row_q[0:CHUNKS-1];
    logic[255:0]sign_row_q[0:CHUNKS-1];
    logic[7:0]replay_pending_q;
    logic[LANES*8-1:0]held_weight_q;
    logic signed[ACC_BITS-1:0]accumulator_q[0:CONTEXTS-1][0:LANES-1];
    logic[2:0]drain_context_q;
    logic[9:0]unique_sources_q,weight_reads_q;
    logic[11:0]context_updates_q;
    logic[7:0]selected_valid;
    logic[2:0]selected_context[0:3];
    logic[3:0]selected_negate;
    logic[7:0]pending_after;
    logic[2:0]replay_width;
    logic illegal_begin,illegal_scan,illegal_scan_done,illegal_response;

    function automatic logic[31:0] source_union(input logic[255:0]row);
        logic[31:0]value;
        begin
            value='0;
            for(int bit_index=0;bit_index<32;bit_index++)
                for(int ctx=0;ctx<8;ctx++)
                    value[bit_index]|=row[ctx*32+bit_index];
            return value;
        end
    endfunction
    function automatic logic[3:0] first_chunk(input logic[11:0]rows);
        logic found;logic[3:0]value;
        begin
            found=0;value=0;
            for(int index=0;index<12;index++)if(!found&&rows[index])begin
                found=1;value=index[3:0];
            end
            return value;
        end
    endfunction
    function automatic logic[4:0] first_bit(input logic[31:0]bits);
        logic found;logic[4:0]value;
        begin
            found=0;value=0;
            for(int index=0;index<32;index++)if(!found&&bits[index])begin
                found=1;value=index[4:0];
            end
            return value;
        end
    endfunction

    generate if(!PARAMETERS_LEGAL)begin:g_bad
        initial $fatal(1,"M228 frozen geometry/FANOUT drift");
    end endgenerate

    always_comb begin:select_replay
        logic[7:0]work;
        work=replay_pending_q;selected_valid='0;selected_negate='0;
        replay_width='0;
        for(int slot=0;slot<4;slot++)selected_context[slot]='0;
        for(int slot=0;slot<FANOUT;slot++)begin
            logic found;found=0;
            for(int ctx=0;ctx<8;ctx++)if(!found&&work[ctx])begin
                found=1;selected_valid[slot]=1;selected_context[slot]=ctx[2:0];
                work[ctx]=0;replay_width=replay_width+1'b1;
            end
        end
        // Hoist this source/context lookup out of the 96-lane update loop.
        // Otherwise synthesis may replicate the same 256:1 sign mux per lane.
        for(int slot=0;slot<FANOUT;slot++)if(selected_valid[slot])
            selected_negate[slot]=sign_row_q[current_chunk_q]
                [selected_context[slot]*32+current_bit_q];
        pending_after=work;
    end

    always_comb begin:interfaces
        illegal_begin=begin_valid&&state_q!=ST_IDLE;
        illegal_scan=scan_valid&&!(state_q==ST_SCAN&&scan_chunk<12
            &&!scan_seen_q[scan_chunk]&&!(|(scan_sign&~scan_presence)));
        illegal_scan_done=scan_done_valid
            &&!(state_q==ST_SCAN&&scan_count_q==12);
        illegal_response=weight_rsp_valid&&!(state_q==ST_WAIT
            &&weight_rsp_tag==tag_q&&weight_rsp_epoch==epoch_q
            &&weight_rsp_source=={current_chunk_q,current_bit_q});
        begin_ready=state_q==ST_IDLE&&!fault_q;begin_accept=begin_valid&&begin_ready;
        scan_ready=state_q==ST_SCAN&&scan_chunk<12&&!scan_seen_q[scan_chunk]
            &&!(|(scan_sign&~scan_presence))&&!fault_q;
        scan_accept=scan_valid&&scan_ready;
        scan_done_ready=state_q==ST_SCAN&&scan_count_q==12&&!fault_q;
        scan_done_accept=scan_done_valid&&scan_done_ready;
        weight_req_valid=state_q==ST_REQUEST&&!fault_q;
        weight_req_tag=weight_req_valid?tag_q:'0;
        weight_req_epoch=weight_req_valid?epoch_q:'0;
        weight_req_source=weight_req_valid?{current_chunk_q,current_bit_q}:'0;
        weight_req_accept=weight_req_valid&&weight_req_ready;
        weight_rsp_ready=state_q==ST_WAIT&&!fault_q&&!illegal_response;
        weight_rsp_accept=weight_rsp_valid&&weight_rsp_ready;
        result_valid=state_q==ST_DRAIN&&!fault_q;
        result_tag=result_valid?tag_q:'0;result_epoch=result_valid?epoch_q:'0;
        result_context=result_valid?drain_context_q:'0;
        result_accumulator='0;
        if(result_valid)for(int lane=0;lane<LANES;lane++)
            result_accumulator[lane*ACC_BITS+:ACC_BITS]
                =accumulator_q[drain_context_q][lane];
        result_last=result_valid&&drain_context_q==7;
        result_accept=result_valid&&result_ready;
        done_valid=state_q==ST_DONE&&!fault_q;
        done_tag=done_valid?tag_q:'0;done_epoch=done_valid?epoch_q:'0;
        done_accept=done_valid&&done_ready;
        protocol_error=fault_q||illegal_begin||illegal_scan
            ||illegal_scan_done||illegal_response;
        // Static bound: at most 384 signed INT8 terms, magnitude <=49,152,
        // strictly below Acc19 range.  Runtime overflow is unreachable.
        numeric_overflow=1'b0;
        busy=state_q!=ST_IDLE;debug_scan_count=scan_count_q;
        debug_unique_sources=unique_sources_q;
        debug_context_updates=context_updates_q;debug_weight_reads=weight_reads_q;
        debug_replay_width=replay_width;
    end

    always_ff @(posedge clk_core)begin:state_update
        if(rst_core)begin
            state_q<=ST_IDLE;fault_q<=0;tag_q<='0;epoch_q<='0;
            scan_seen_q<='0;scan_count_q<='0;row_summary_q<='0;
            remaining_rows_q<='0;current_chunk_q<='0;current_bit_q<='0;
            row_remaining_q<='0;replay_pending_q<='0;held_weight_q<='0;
            drain_context_q<='0;unique_sources_q<='0;weight_reads_q<='0;
            context_updates_q<='0;
            for(int chunk=0;chunk<12;chunk++)begin
                presence_row_q[chunk]<='0;sign_row_q[chunk]<='0;
            end
            for(int ctx=0;ctx<8;ctx++)for(int lane=0;lane<LANES;lane++)
                accumulator_q[ctx][lane]<='0;
        end else begin
            if(illegal_begin||illegal_scan||illegal_scan_done||illegal_response)
                fault_q<=1;
            if(!protocol_error)case(state_q)
                ST_IDLE:if(begin_accept)begin
                    state_q<=ST_SCAN;tag_q<=begin_tag;epoch_q<=begin_epoch;
                    scan_seen_q<='0;scan_count_q<='0;row_summary_q<='0;
                    remaining_rows_q<='0;row_remaining_q<='0;
                    replay_pending_q<='0;drain_context_q<='0;
                    unique_sources_q<='0;weight_reads_q<='0;context_updates_q<='0;
                    for(int chunk=0;chunk<12;chunk++)begin
                        presence_row_q[chunk]<='0;sign_row_q[chunk]<='0;
                    end
                    for(int ctx=0;ctx<8;ctx++)for(int lane=0;lane<LANES;lane++)
                        accumulator_q[ctx][lane]<='0;
                end
                ST_SCAN:begin
                    if(scan_accept)begin
                        scan_seen_q[scan_chunk]<=1;scan_count_q<=scan_count_q+1'b1;
                        presence_row_q[scan_chunk]<=scan_presence;
                        sign_row_q[scan_chunk]<=scan_sign;
                        row_summary_q[scan_chunk]<=|scan_presence;
                    end
                    if(scan_done_accept)begin
                        logic[3:0]chunk_now;logic[31:0]bits_now;
                        remaining_rows_q<=row_summary_q;
                        if(|row_summary_q)begin
                            chunk_now=first_chunk(row_summary_q);
                            bits_now=source_union(presence_row_q[chunk_now]);
                            current_chunk_q<=chunk_now;row_remaining_q<=bits_now;
                            current_bit_q<=first_bit(bits_now);state_q<=ST_REQUEST;
                        end else begin drain_context_q<=0;state_q<=ST_DRAIN;end
                    end
                end
                ST_REQUEST:if(weight_req_accept)state_q<=ST_WAIT;
                ST_WAIT:if(weight_rsp_accept)begin
                    logic[7:0]context_mask;
                    for(int ctx=0;ctx<8;ctx++)context_mask[ctx]
                        =presence_row_q[current_chunk_q][ctx*32+current_bit_q];
                    held_weight_q<=weight_rsp_data;replay_pending_q<=context_mask;
                    unique_sources_q<=unique_sources_q+1'b1;
                    weight_reads_q<=weight_reads_q+1'b1;state_q<=ST_REPLAY;
                end
                ST_REPLAY:begin
                    for(int slot=0;slot<FANOUT;slot++)if(selected_valid[slot])
                        for(int lane=0;lane<LANES;lane++)begin
                            logic signed[ACC_BITS-1:0]weight_value;
                            weight_value={{(ACC_BITS-8){held_weight_q[lane*8+7]}},
                                held_weight_q[lane*8+:8]};
                            if(selected_negate[slot])
                                accumulator_q[selected_context[slot]][lane]
                                    <=$signed(accumulator_q[selected_context[slot]][lane])
                                      -$signed(weight_value);
                            else accumulator_q[selected_context[slot]][lane]
                                    <=$signed(accumulator_q[selected_context[slot]][lane])
                                      +$signed(weight_value);
                        end
                    context_updates_q<=context_updates_q+replay_width;
                    replay_pending_q<=pending_after;
                    if(pending_after==0)begin
                        logic[31:0]bits_after;logic[11:0]rows_after;
                        logic[3:0]next_chunk;logic[31:0]next_bits;
                        bits_after=row_remaining_q;bits_after[current_bit_q]=0;
                        if(|bits_after)begin
                            row_remaining_q<=bits_after;
                            current_bit_q<=first_bit(bits_after);state_q<=ST_REQUEST;
                        end else begin
                            rows_after=remaining_rows_q;rows_after[current_chunk_q]=0;
                            remaining_rows_q<=rows_after;
                            if(|rows_after)begin
                                next_chunk=first_chunk(rows_after);
                                next_bits=source_union(presence_row_q[next_chunk]);
                                current_chunk_q<=next_chunk;row_remaining_q<=next_bits;
                                current_bit_q<=first_bit(next_bits);state_q<=ST_REQUEST;
                            end else begin drain_context_q<=0;state_q<=ST_DRAIN;end
                        end
                    end
                end
                ST_DRAIN:if(result_accept)begin
                    if(drain_context_q==7)state_q<=ST_DONE;
                    else drain_context_q<=drain_context_q+1'b1;
                end
                ST_DONE:if(done_accept)state_q<=ST_IDLE;
                default:state_q<=ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
