`timescale 1ns/1ps
`default_nettype none

interface m342_fc2_bus_if #(
    parameter int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16,
    parameter int GENERATION_BITS=32, SLICE_LANES=16
);
    logic header_valid, header_ready, header_accept;
    logic [TAG_BITS-1:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth, header_output_blocks;
    logic raw_valid, raw_ready, raw_accept, raw_last;
    logic [3:0] raw_lane_valid;
    logic [4:0] raw_beat_index [0:3];
    logic [95:0] raw_bitmap [0:3];
    logic mem_req_valid, mem_req_ready, mem_req_accept;
    logic [EPOCH_BITS-1:0] mem_req_epoch;
    logic [2:0] mem_req_slot;
    logic [GENERATION_BITS-1:0] mem_req_generation;
    logic [TAG_BITS-1:0] mem_req_tag;
    logic [2:0] mem_req_output_block, mem_req_slice;
    logic [3:0] mem_req_source_count;
    logic [7:0] mem_req_bank_valid;
    logic [CHANNEL_BITS-1:0] mem_req_source_channel [0:7];
    logic mem_rsp_valid, mem_rsp_ready, mem_rsp_accept;
    logic [EPOCH_BITS-1:0] mem_rsp_epoch;
    logic [2:0] mem_rsp_slot;
    logic [GENERATION_BITS-1:0] mem_rsp_generation;
    logic [TAG_BITS-1:0] mem_rsp_tag;
    logic [7:0] mem_rsp_bank_valid;
    logic signed [7:0] mem_rsp_weight [0:7][0:SLICE_LANES-1];
    logic result_valid, result_ready, result_accept, result_last;
    logic [TAG_BITS-1:0] result_tag;
    logic [2:0] result_output_block, result_slice;
    logic signed [23:0] result_accumulator [0:SLICE_LANES-1];
    logic token_done_valid, token_done_ready, token_done_accept;
    logic [TAG_BITS-1:0] token_done_tag;
    logic token_done_had_event;
    logic protocol_error, numeric_overflow, stale_response_seen, busy;
    logic [2:0] debug_fifo_count;
    logic [3:0] debug_outstanding_count;
    logic [31:0] debug_group_accept_count;
    logic [31:0] debug_request_accept_count;
    logic [31:0] debug_response_accept_count;
    logic [31:0] debug_context_write_count;
    logic [31:0] debug_result_accept_count;
    logic [31:0] debug_active_bank_read_count;
    logic memory_enable, memory_stall, memory_newest_first;
    logic memory_spurious_valid;
    logic [31:0] memory_request_count, memory_response_count;
    logic [31:0] memory_active_bank_read_count;
    logic [3:0] memory_pending_count;
    logic memory_live_slot_reuse_error;
endinterface

`define M342_DUT_PORTS(P) \
    .clk_core(clk_core), .rst_core(rst_core), \
    .header_valid(P.header_valid), .header_ready(P.header_ready), \
    .header_tag(P.header_tag), \
    .header_raw_beat_count(P.header_raw_beat_count), \
    .header_window_depth(P.header_window_depth), \
    .header_output_blocks(P.header_output_blocks), \
    .header_accept(P.header_accept), \
    .raw_valid(P.raw_valid), .raw_ready(P.raw_ready), \
    .raw_lane_valid(P.raw_lane_valid), \
    .raw_beat_index(P.raw_beat_index), .raw_bitmap(P.raw_bitmap), \
    .raw_last(P.raw_last), .raw_accept(P.raw_accept), \
    .mem_req_valid(P.mem_req_valid), .mem_req_ready(P.mem_req_ready), \
    .mem_req_epoch(P.mem_req_epoch), .mem_req_slot(P.mem_req_slot), \
    .mem_req_generation(P.mem_req_generation), \
    .mem_req_tag(P.mem_req_tag), \
    .mem_req_output_block(P.mem_req_output_block), \
    .mem_req_slice(P.mem_req_slice), \
    .mem_req_source_count(P.mem_req_source_count), \
    .mem_req_bank_valid(P.mem_req_bank_valid), \
    .mem_req_source_channel(P.mem_req_source_channel), \
    .mem_req_accept(P.mem_req_accept), \
    .mem_rsp_valid(P.mem_rsp_valid), .mem_rsp_ready(P.mem_rsp_ready), \
    .mem_rsp_epoch(P.mem_rsp_epoch), .mem_rsp_slot(P.mem_rsp_slot), \
    .mem_rsp_generation(P.mem_rsp_generation), \
    .mem_rsp_tag(P.mem_rsp_tag), \
    .mem_rsp_bank_valid(P.mem_rsp_bank_valid), \
    .mem_rsp_weight(P.mem_rsp_weight), .mem_rsp_accept(P.mem_rsp_accept), \
    .result_valid(P.result_valid), .result_ready(P.result_ready), \
    .result_tag(P.result_tag), \
    .result_output_block(P.result_output_block), \
    .result_slice(P.result_slice), \
    .result_accumulator(P.result_accumulator), \
    .result_last(P.result_last), .result_accept(P.result_accept), \
    .token_done_valid(P.token_done_valid), \
    .token_done_ready(P.token_done_ready), \
    .token_done_tag(P.token_done_tag), \
    .token_done_had_event(P.token_done_had_event), \
    .token_done_accept(P.token_done_accept), \
    .protocol_error(P.protocol_error), \
    .numeric_overflow(P.numeric_overflow), \
    .stale_response_seen(P.stale_response_seen), .busy(P.busy), \
    .debug_fifo_count(P.debug_fifo_count), \
    .debug_outstanding_count(P.debug_outstanding_count), \
    .debug_group_accept_count(P.debug_group_accept_count), \
    .debug_request_accept_count(P.debug_request_accept_count), \
    .debug_response_accept_count(P.debug_response_accept_count), \
    .debug_context_write_count(P.debug_context_write_count), \
    .debug_result_accept_count(P.debug_result_accept_count), \
    .debug_active_bank_read_count(P.debug_active_bank_read_count)

`define M342_MEMORY_PORTS(P) \
    .clk_core(clk_core), .rst_core(rst_core), \
    .enable(P.memory_enable), .stall_enable(P.memory_stall), \
    .newest_first(P.memory_newest_first), \
    .spurious_valid(P.memory_spurious_valid), \
    .mem_req_valid(P.mem_req_valid), .mem_req_ready(P.mem_req_ready), \
    .mem_req_epoch(P.mem_req_epoch), .mem_req_slot(P.mem_req_slot), \
    .mem_req_generation(P.mem_req_generation), \
    .mem_req_tag(P.mem_req_tag), \
    .mem_req_output_block(P.mem_req_output_block), \
    .mem_req_slice(P.mem_req_slice), \
    .mem_req_source_count(P.mem_req_source_count), \
    .mem_req_bank_valid(P.mem_req_bank_valid), \
    .mem_req_source_channel(P.mem_req_source_channel), \
    .mem_req_accept(P.mem_req_accept), \
    .mem_rsp_valid(P.mem_rsp_valid), .mem_rsp_ready(P.mem_rsp_ready), \
    .mem_rsp_epoch(P.mem_rsp_epoch), .mem_rsp_slot(P.mem_rsp_slot), \
    .mem_rsp_generation(P.mem_rsp_generation), \
    .mem_rsp_tag(P.mem_rsp_tag), \
    .mem_rsp_bank_valid(P.mem_rsp_bank_valid), \
    .mem_rsp_weight(P.mem_rsp_weight), .mem_rsp_accept(P.mem_rsp_accept), \
    .request_count(P.memory_request_count), \
    .response_count(P.memory_response_count), \
    .active_bank_read_count(P.memory_active_bank_read_count), \
    .pending_count(P.memory_pending_count), \
    .live_slot_reuse_error(P.memory_live_slot_reuse_error)

module tb_m342_fc2_standalone_raw4_acc24;
    localparam int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16;
    localparam int GENERATION_BITS=32, LANES=16;
    logic clk_core=0, rst_core;
    always #1.5 clk_core=~clk_core;

    m342_fc2_bus_if a();
    m342_fc2_bus_if b();

    m342_fc2_standalone_raw4_acc24 #(.SOURCE_CAP(8)) candidate (
        `M342_DUT_PORTS(a));
    m342_fc2_standalone_raw4_acc24 #(.SOURCE_CAP(1)) baseline (
        `M342_DUT_PORTS(b));
    m342_fc2_eight_bank_memory_model memory_a (`M342_MEMORY_PORTS(a));
    m342_fc2_eight_bank_memory_model memory_b (`M342_MEMORY_PORTS(b));

    integer cycle_count, error_count, numeric_mismatch_count;
    integer clean_case_count, reset_case_count, protocol_attack_count;
    integer result_count, done_count, event_count, expected_blocks;
    integer expected_cap, start_cycle, done_cycle, watchdog_count;
    integer request_stall_count, result_stall_count, raw_stall_count;
    integer full8_request_count, ooo_response_count;
    integer k8_cycles [0:3], k1_cycles [0:3];
    integer signed reference_accum [0:7][0:5][0:LANES-1];
    logic [95:0] payload [0:31];
    logic [TAG_BITS-1:0] expected_tag;
    logic check_results;

    function automatic integer block_index(input integer blocks);
        case (blocks)
            1: return 0;
            2: return 1;
            4: return 2;
            default: return 3;
        endcase
    endfunction

    function automatic integer raw_count(input integer blocks);
        case (blocks)
            1: return 4;
            2: return 8;
            4: return 16;
            default: return 32;
        endcase
    endfunction

    function automatic integer window_depth(input integer blocks);
        case (blocks)
            1: return 2;
            2: return 4;
            default: return 8;
        endcase
    endfunction

    function automatic integer signed weight_value(
        input integer bank, input integer lane, input integer channel,
        input integer block, input integer slice);
        integer value;
        begin
            value = (channel*3 + bank*5 + block*7
                + slice*11 + lane*13) % 31;
            return value - 15;
        end
    endfunction

    task automatic build_payload_and_reference(
        input integer blocks, input integer mode);
        integer beats, row, bank, channel;
        begin
            beats=raw_count(blocks);
            event_count=0;
            for (integer beat=0;beat<32;beat++) payload[beat]=0;
            for (integer block=0;block<8;block++)
                for (integer slice=0;slice<6;slice++)
                    for (integer lane=0;lane<LANES;lane++)
                        reference_accum[block][slice][lane]=0;
            if (mode != 9) begin
                // Force one eight-bank opportunity while keeping row positions
                // nonuniform, then add deterministic noncontiguous activity.
                for (bank=0;bank<8;bank++)
                    payload[0][(bank%12)*8+bank]=1;
                for (integer beat=0;beat<beats;beat++) begin
                    for (integer item=0;item<3+(mode%3);item++) begin
                        row=(beat*3+item*5+mode)%12;
                        bank=(beat+item*3+mode)%8;
                        payload[beat][row*8+bank]=1;
                    end
                    if ((beat+mode)%4==0) begin
                        row=(beat+7)%12;
                        bank=(beat*5+2)%8;
                        payload[beat][row*8+bank]=1;
                    end
                end
            end
            for (integer beat=0;beat<beats;beat++) begin
                for (row=0;row<12;row++) begin
                    for (bank=0;bank<8;bank++) begin
                        if (payload[beat][row*8+bank]) begin
                            event_count++;
                            channel=(beat*12+row)*8+bank;
                            for (integer block=0;block<blocks;block++)
                                for (integer slice=0;slice<6;slice++)
                                    for (integer lane=0;lane<LANES;lane++)
                                        reference_accum[block][slice][lane]
                                            += weight_value(bank,lane,channel,
                                                block,slice);
                        end
                    end
                end
            end
        end
    endtask

    task automatic initialize_inputs;
        begin
            a.header_valid=0;a.raw_valid=0;a.raw_last=0;
            b.header_valid=0;b.raw_valid=0;b.raw_last=0;
            a.raw_lane_valid=0;b.raw_lane_valid=0;
            a.memory_enable=0;b.memory_enable=0;
            a.memory_stall=1;b.memory_stall=1;
            a.memory_newest_first=1;b.memory_newest_first=1;
            a.memory_spurious_valid=0;b.memory_spurious_valid=0;
            for(integer lane=0;lane<4;lane++)begin
                a.raw_beat_index[lane]=0;b.raw_beat_index[lane]=0;
                a.raw_bitmap[lane]=0;b.raw_bitmap[lane]=0;
            end
        end
    endtask

    task automatic reset_pair(input integer cap);
        begin
            @(negedge clk_core);rst_core=1;initialize_inputs();
            check_results=0;
            repeat(4)@(negedge clk_core);
            if(cap==8)a.memory_enable=1;
            else if(cap==1)b.memory_enable=1;
            rst_core=0;
            repeat(2)@(posedge clk_core);
        end
    endtask

    task automatic drive_header(input integer cap,
        input logic[TAG_BITS-1:0]tag,input integer blocks);
        begin
            @(negedge clk_core);
            if(cap==8)begin
                a.header_tag=tag;a.header_output_blocks=blocks;
                a.header_raw_beat_count=raw_count(blocks);
                a.header_window_depth=window_depth(blocks);a.header_valid=1;
                while(!a.header_accept)@(posedge clk_core);
                start_cycle=cycle_count;
                @(negedge clk_core);a.header_valid=0;
            end else begin
                b.header_tag=tag;b.header_output_blocks=blocks;
                b.header_raw_beat_count=raw_count(blocks);
                b.header_window_depth=window_depth(blocks);b.header_valid=1;
                while(!b.header_accept)@(posedge clk_core);
                start_cycle=cycle_count;
                @(negedge clk_core);b.header_valid=0;
            end
        end
    endtask

    task automatic drive_raw(input integer cap,input integer blocks,
        input integer packet_limit,input logic terminate_last);
        integer beats,packets,base;
        begin
            beats=raw_count(blocks);packets=beats/4;
            if(packet_limit<packets)packets=packet_limit;
            for(integer packet=0;packet<packets;packet++)begin
                base=packet*4;
                @(negedge clk_core);
                if(cap==8)begin
                    a.raw_lane_valid=4'b1111;
                    for(integer lane=0;lane<4;lane++)begin
                        a.raw_beat_index[lane]=base+lane;
                        a.raw_bitmap[lane]=payload[base+lane];
                    end
                    a.raw_last=terminate_last&&(packet+1==beats/4);
                    a.raw_valid=1;
                    while(!a.raw_accept)@(posedge clk_core);
                end else begin
                    b.raw_lane_valid=4'b1111;
                    for(integer lane=0;lane<4;lane++)begin
                        b.raw_beat_index[lane]=base+lane;
                        b.raw_bitmap[lane]=payload[base+lane];
                    end
                    b.raw_last=terminate_last&&(packet+1==beats/4);
                    b.raw_valid=1;
                    while(!b.raw_accept)@(posedge clk_core);
                end
            end
            @(negedge clk_core);
            if(cap==8)begin a.raw_valid=0;a.raw_last=0;a.raw_lane_valid=0;end
            else begin b.raw_valid=0;b.raw_last=0;b.raw_lane_valid=0;end
        end
    endtask

    task automatic run_clean(input integer cap,input integer blocks,
        input integer mode);
        integer expected_reads,expected_results,index;
        logic[TAG_BITS-1:0]tag;
        begin
            tag=24'h342000|(cap<<12)|(blocks<<4)|mode;
            build_payload_and_reference(blocks,mode);
            expected_cap=cap;expected_blocks=blocks;expected_tag=tag;
            reset_pair(cap);check_results=1;
            fork
                begin drive_header(cap,tag,blocks);drive_raw(cap,blocks,32,1);end
                begin
                    watchdog_count=0;
                    while(done_count==0&&watchdog_count<100000)begin
                        @(posedge clk_core);watchdog_count++;
                    end
                    if(done_count==0)begin
                        $error("M342 clean watchdog cap=%0d B=%0d",cap,blocks);
                        error_count++;
                    end
                end
            join
            check_results=0;
            expected_results=blocks*6;
            expected_reads=event_count*blocks*6;
            if(result_count!=expected_results)begin
                $error("result conservation cap=%0d B=%0d got=%0d exp=%0d",
                    cap,blocks,result_count,expected_results);error_count++;
            end
            if(cap==8)begin
                if(a.protocol_error||a.numeric_overflow
                        ||a.memory_live_slot_reuse_error)begin
                    $error("K8 clean fault B=%0d p=%0d n=%0d m=%0d",blocks,
                        a.protocol_error,a.numeric_overflow,
                        a.memory_live_slot_reuse_error);error_count++;
                end
                if(a.debug_request_accept_count
                        !=a.debug_group_accept_count*6
                        ||a.debug_response_accept_count
                            !=a.debug_request_accept_count
                        ||a.debug_context_write_count
                            !=a.debug_request_accept_count
                        ||a.debug_result_accept_count!=expected_results
                        ||a.debug_active_bank_read_count!=expected_reads
                        ||a.memory_request_count
                            !=a.debug_request_accept_count
                        ||a.memory_response_count
                            !=a.debug_response_accept_count
                        ||a.memory_active_bank_read_count!=expected_reads)begin
                    $error("K8 conservation B=%0d g=%0d q=%0d r=%0d out=%0d read=%0d expread=%0d",
                        blocks,a.debug_group_accept_count,
                        a.debug_request_accept_count,
                        a.debug_response_accept_count,
                        a.debug_result_accept_count,
                        a.debug_active_bank_read_count,expected_reads);
                    error_count++;
                end
                if(event_count==0&&a.debug_group_accept_count!=0)begin
                    $error("K8 zero token emitted groups");error_count++;
                end
                index=block_index(blocks);k8_cycles[index]=done_cycle-start_cycle+1;
            end else begin
                if(b.protocol_error||b.numeric_overflow
                        ||b.memory_live_slot_reuse_error)begin
                    $error("K1 clean fault B=%0d p=%0d n=%0d m=%0d",blocks,
                        b.protocol_error,b.numeric_overflow,
                        b.memory_live_slot_reuse_error);error_count++;
                end
                if(b.debug_group_accept_count!=event_count*blocks
                        ||b.debug_request_accept_count
                            !=b.debug_group_accept_count*6
                        ||b.debug_response_accept_count
                            !=b.debug_request_accept_count
                        ||b.debug_context_write_count
                            !=b.debug_request_accept_count
                        ||b.debug_result_accept_count!=expected_results
                        ||b.debug_active_bank_read_count!=expected_reads
                        ||b.memory_request_count
                            !=b.debug_request_accept_count
                        ||b.memory_response_count
                            !=b.debug_response_accept_count
                        ||b.memory_active_bank_read_count!=expected_reads)begin
                    $error("K1 conservation B=%0d events=%0d g=%0d q=%0d r=%0d out=%0d read=%0d expread=%0d",
                        blocks,event_count,b.debug_group_accept_count,
                        b.debug_request_accept_count,
                        b.debug_response_accept_count,
                        b.debug_result_accept_count,
                        b.debug_active_bank_read_count,expected_reads);
                    error_count++;
                end
                index=block_index(blocks);k1_cycles[index]=done_cycle-start_cycle+1;
            end
            clean_case_count++;
            $display("M342 clean cap=%0d B=%0d events=%0d cycles=%0d results=%0d mismatches=%0d",
                cap,blocks,event_count,done_cycle-start_cycle+1,
                result_count,numeric_mismatch_count);
        end
    endtask

    task automatic run_reset_attack(input integer cap);
        logic[TAG_BITS-1:0]tag;
        begin
            tag=24'h342a00|cap;build_payload_and_reference(8,2);
            reset_pair(cap);drive_header(cap,tag,8);drive_raw(cap,8,1,0);
            @(negedge clk_core);rst_core=1;
            repeat(3)@(negedge clk_core);
            if(cap==8)begin
                if(a.busy||a.protocol_error||a.result_valid
                        ||a.token_done_valid||a.memory_pending_count!=0)begin
                    $error("K8 common POR failed");error_count++;
                end
            end else begin
                if(b.busy||b.protocol_error||b.result_valid
                        ||b.token_done_valid||b.memory_pending_count!=0)begin
                    $error("K1 common POR failed");error_count++;
                end
            end
            rst_core=0;repeat(2)@(posedge clk_core);reset_case_count++;
        end
    endtask

    task automatic run_header_attack(input integer cap);
        begin
            reset_pair(cap);@(negedge clk_core);
            if(cap==8)begin
                a.header_tag=24'h342bad;a.header_output_blocks=3;
                a.header_raw_beat_count=7;a.header_window_depth=3;
                a.header_valid=1;@(posedge clk_core);@(negedge clk_core);
                a.header_valid=0;@(posedge clk_core);
                if(!a.protocol_error)begin
                    $error("K8 illegal header not quarantined");error_count++;
                end
            end else begin
                b.header_tag=24'h342bad;b.header_output_blocks=3;
                b.header_raw_beat_count=7;b.header_window_depth=3;
                b.header_valid=1;@(posedge clk_core);@(negedge clk_core);
                b.header_valid=0;@(posedge clk_core);
                if(!b.protocol_error)begin
                    $error("K1 illegal header not quarantined");error_count++;
                end
            end
            repeat(2)@(posedge clk_core);protocol_attack_count++;
        end
    endtask

    task automatic run_response_attack(input integer cap);
        logic[TAG_BITS-1:0]tag;
        begin
            tag=24'h342b00|cap;reset_pair(cap);drive_header(cap,tag,1);
            @(negedge clk_core);
            if(cap==8)a.memory_spurious_valid=1;
            else b.memory_spurious_valid=1;
            @(posedge clk_core);@(negedge clk_core);
            if(cap==8)a.memory_spurious_valid=0;
            else b.memory_spurious_valid=0;
            @(posedge clk_core);
            if(cap==8&&!a.protocol_error)begin
                $error("K8 spurious response not rejected");error_count++;
            end
            if(cap==1&&!b.protocol_error)begin
                $error("K1 spurious response not rejected");error_count++;
            end
            repeat(2)@(posedge clk_core);protocol_attack_count++;
        end
    endtask

    always_comb begin
        a.result_ready=!rst_core&&(cycle_count%5!=2);
        b.result_ready=!rst_core&&(cycle_count%5!=2);
        a.token_done_ready=!rst_core&&(cycle_count%4!=1);
        b.token_done_ready=!rst_core&&(cycle_count%4!=1);
    end

    always @(posedge clk_core) begin
        integer signed observed;
        if(rst_core)begin
            cycle_count=0;result_count=0;done_count=0;done_cycle=-1;
        end else begin
            cycle_count++;
            if(a.mem_req_valid&&!a.mem_req_ready)request_stall_count++;
            if(b.mem_req_valid&&!b.mem_req_ready)request_stall_count++;
            if(a.result_valid&&!a.result_ready)result_stall_count++;
            if(b.result_valid&&!b.result_ready)result_stall_count++;
            if(a.raw_valid&&!a.raw_ready)raw_stall_count++;
            if(b.raw_valid&&!b.raw_ready)raw_stall_count++;
            if(a.mem_req_accept&&a.mem_req_source_count==8)
                full8_request_count++;
            if(b.mem_req_accept&&b.mem_req_source_count==8)
                full8_request_count++;
            if(a.mem_rsp_accept&&a.mem_rsp_generation+1
                    <a.debug_request_accept_count)ooo_response_count++;
            if(b.mem_rsp_accept&&b.mem_rsp_generation+1
                    <b.debug_request_accept_count)ooo_response_count++;

            if(check_results&&expected_cap==8&&a.result_accept)begin
                if(a.result_tag!==expected_tag
                        ||a.result_output_block>=expected_blocks
                        ||a.result_slice>=6)begin
                    $error("K8 result identity mismatch");error_count++;
                end
                for(integer lane=0;lane<LANES;lane++)begin
                    observed=$signed(a.result_accumulator[lane]);
                    if(observed!==reference_accum[a.result_output_block]
                            [a.result_slice][lane])begin
                        $error("K8 numeric mismatch B=%0d S=%0d L=%0d got=%0d exp=%0d",
                            a.result_output_block,a.result_slice,lane,observed,
                            reference_accum[a.result_output_block]
                                [a.result_slice][lane]);
                        error_count++;numeric_mismatch_count++;
                    end
                end
                if(a.result_last!=(a.result_output_block+1==expected_blocks
                        &&a.result_slice==5))begin
                    $error("K8 result_last mismatch");error_count++;
                end
                result_count++;
            end
            if(check_results&&expected_cap==1&&b.result_accept)begin
                if(b.result_tag!==expected_tag
                        ||b.result_output_block>=expected_blocks
                        ||b.result_slice>=6)begin
                    $error("K1 result identity mismatch");error_count++;
                end
                for(integer lane=0;lane<LANES;lane++)begin
                    observed=$signed(b.result_accumulator[lane]);
                    if(observed!==reference_accum[b.result_output_block]
                            [b.result_slice][lane])begin
                        $error("K1 numeric mismatch B=%0d S=%0d L=%0d got=%0d exp=%0d",
                            b.result_output_block,b.result_slice,lane,observed,
                            reference_accum[b.result_output_block]
                                [b.result_slice][lane]);
                        error_count++;numeric_mismatch_count++;
                    end
                end
                if(b.result_last!=(b.result_output_block+1==expected_blocks
                        &&b.result_slice==5))begin
                    $error("K1 result_last mismatch");error_count++;
                end
                result_count++;
            end
            if(check_results&&expected_cap==8&&a.token_done_accept)begin
                if(a.token_done_tag!==expected_tag
                        ||a.token_done_had_event!=(event_count!=0))begin
                    $error("K8 final done mismatch");error_count++;
                end
                done_count++;done_cycle=cycle_count;
            end
            if(check_results&&expected_cap==1&&b.token_done_accept)begin
                if(b.token_done_tag!==expected_tag
                        ||b.token_done_had_event!=(event_count!=0))begin
                    $error("K1 final done mismatch");error_count++;
                end
                done_count++;done_cycle=cycle_count;
            end
        end
    end

    initial begin
        rst_core=1;error_count=0;numeric_mismatch_count=0;
        clean_case_count=0;reset_case_count=0;protocol_attack_count=0;
        request_stall_count=0;result_stall_count=0;raw_stall_count=0;
        full8_request_count=0;ooo_response_count=0;check_results=0;
        expected_cap=0;expected_blocks=0;expected_tag=0;initialize_inputs();
        repeat(4)@(negedge clk_core);

        run_reset_attack(8);run_reset_attack(1);
        run_header_attack(8);run_header_attack(1);
        run_response_attack(8);run_response_attack(1);

        run_clean(8,1,0);run_clean(1,1,0);
        run_clean(8,2,1);run_clean(1,2,1);
        run_clean(8,4,2);run_clean(1,4,2);
        run_clean(8,8,3);run_clean(1,8,3);
        run_clean(8,1,9);run_clean(1,1,9);

        if(clean_case_count!=10||reset_case_count!=2
                ||protocol_attack_count!=4||numeric_mismatch_count!=0
                ||request_stall_count==0||result_stall_count==0
                ||raw_stall_count==0||full8_request_count==0
                ||ooo_response_count==0)begin
            $error("coverage clean=%0d reset=%0d attacks=%0d numeric=%0d reqstall=%0d resultstall=%0d rawstall=%0d full8=%0d ooo=%0d",
                clean_case_count,reset_case_count,protocol_attack_count,
                numeric_mismatch_count,request_stall_count,result_stall_count,
                raw_stall_count,full8_request_count,ooo_response_count);
            error_count++;
        end
        if(error_count==0)$display("PASS M342 standalone raw4-to-Acc24 VCS clean_cases=10 B1_B2_B4_B8=true zero_tokens=2 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 K8_K1_reference_exact=true request_stalls=%0d result_stalls=%0d raw_stalls=%0d full8_requests=%0d ooo_responses=%0d",
            request_stall_count,result_stall_count,raw_stall_count,
            full8_request_count,ooo_response_count);
        else $fatal(1,"M342 failures=%0d numeric=%0d",error_count,
            numeric_mismatch_count);
        $finish;
    end
endmodule

`undef M342_DUT_PORTS
`undef M342_MEMORY_PORTS
`default_nettype wire
