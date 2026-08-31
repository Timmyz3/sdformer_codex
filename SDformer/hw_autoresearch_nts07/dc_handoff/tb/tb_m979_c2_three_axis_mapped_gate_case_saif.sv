`timescale 1ns/1ps
`default_nettype none

// M979 source-only mapped-gate replay. Exactly one M872 axis and one frozen
// M867 case are selected at compile/run time. UCLI starts at the accepted
// header stop and ends one clock after accepted token_done, so reset and
// inter-case idle are excluded and SAIF duration is cycles*3 ns.
`ifdef M979_AXIS_K1
  `define M979_DUT m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE0
  `define M979_AXIS_ID 0
  `define M979_AXIS_NAME "K1"
`elsif M979_AXIS_K8
  `define M979_DUT m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE1
  `define M979_AXIS_ID 1
  `define M979_AXIS_NAME "K8"
`elsif M979_AXIS_K1X8
  `define M979_DUT m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_ARCH_MODE2
  `define M979_AXIS_ID 2
  `define M979_AXIS_NAME "K1x8"
`else
  `error "M979 requires exactly one axis define"
`endif

module tb_m979_c2_three_axis_mapped_gate_case_saif;
    localparam int TAG_BITS=24, CHANNEL_BITS=12, EPOCH_BITS=16;
    localparam int GENERATION_BITS=32, LANES=16, MAX_CHANNELS=3072;
    localparam int AXIS_ID=`M979_AXIS_ID;

    logic clk_core=0, rst_core;
    always #1.5 clk_core=~clk_core;

    logic header_valid,header_ready,header_accept;
    logic [23:0] header_tag;
    logic [5:0] header_raw_beat_count;
    logic [3:0] header_window_depth,header_output_blocks;
    logic raw_valid,raw_ready,raw_accept,raw_last;
    logic [3:0] raw_lane_valid;
    logic [19:0] raw_beat_index;
    logic [383:0] raw_bitmap;
    logic [7:0] mem_req_valid,mem_req_ready,mem_req_accept;
    logic [127:0] mem_req_epoch;
    logic [23:0] mem_req_slot;
    logic [255:0] mem_req_generation;
    logic [191:0] mem_req_tag;
    logic [23:0] mem_req_output_block,mem_req_slice;
    logic [95:0] mem_req_source_channel;
    logic [7:0] mem_rsp_valid,mem_rsp_ready,mem_rsp_accept;
    logic [127:0] mem_rsp_epoch;
    logic [23:0] mem_rsp_slot;
    logic [255:0] mem_rsp_generation;
    logic [191:0] mem_rsp_tag;
    logic [1023:0] mem_rsp_weight;
    logic result_valid,result_ready,result_last,result_accept;
    logic [23:0] result_tag;
    logic [2:0] result_output_block,result_slice;
    logic [383:0] result_accumulator;
    logic token_done_valid,token_done_ready,token_done_had_event;
    logic token_done_accept;
    logic [23:0] token_done_tag;
    logic protocol_error,numeric_overflow,stale_response_seen,busy;

    `M979_DUT dut (
        .clk_core(clk_core),.rst_core(rst_core),
        .header_valid(header_valid),.header_ready(header_ready),
        .header_tag(header_tag),.header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),
        .header_accept(header_accept),.raw_valid(raw_valid),
        .raw_ready(raw_ready),.raw_lane_valid(raw_lane_valid),
        .raw_beat_index(raw_beat_index),.raw_bitmap(raw_bitmap),
        .raw_last(raw_last),.raw_accept(raw_accept),
        .mem_req_valid(mem_req_valid),.mem_req_ready(mem_req_ready),
        .mem_req_epoch(mem_req_epoch),.mem_req_slot(mem_req_slot),
        .mem_req_generation(mem_req_generation),.mem_req_tag(mem_req_tag),
        .mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),
        .mem_req_source_channel(mem_req_source_channel),
        .mem_req_accept(mem_req_accept),.mem_rsp_valid(mem_rsp_valid),
        .mem_rsp_ready(mem_rsp_ready),.mem_rsp_epoch(mem_rsp_epoch),
        .mem_rsp_slot(mem_rsp_slot),.mem_rsp_generation(mem_rsp_generation),
        .mem_rsp_tag(mem_rsp_tag),.mem_rsp_weight(mem_rsp_weight),
        .mem_rsp_accept(mem_rsp_accept),.result_valid(result_valid),
        .result_ready(result_ready),.result_tag(result_tag),
        .result_output_block(result_output_block),.result_slice(result_slice),
        .result_accumulator(result_accumulator),.result_last(result_last),
        .result_accept(result_accept),.token_done_valid(token_done_valid),
        .token_done_ready(token_done_ready),.token_done_tag(token_done_tag),
        .token_done_had_event(token_done_had_event),
        .token_done_accept(token_done_accept),.protocol_error(protocol_error),
        .numeric_overflow(numeric_overflow),
        .stale_response_seen(stale_response_seen),.busy(busy));

    logic request_allow,response_allow;
    logic [7:0] bank_rsp_valid;
    logic signed [7:0] bank_rsp_weight [0:7][0:LANES-1];
    logic [31:0] bank_requests[0:7],bank_responses[0:7];
    logic [3:0] bank_pending[0:7];
    logic bank_reuse_error[0:7];

    for(genvar bank=0;bank<8;bank++) begin:g_memory
        m349_fc2_scalar_bank_memory_model #(.BANK_ID(bank)) memory (
            .clk_core(clk_core),.rst_core(rst_core),.enable(1'b1),
            .request_allow(request_allow),.newest_first(1'b1),
            .spurious_valid(1'b0),.mem_req_valid(mem_req_valid[bank]),
            .mem_req_ready(mem_req_ready[bank]),
            .mem_req_epoch(mem_req_epoch[127-bank*16-:16]),
            .mem_req_slot(mem_req_slot[23-bank*3-:3]),
            .mem_req_generation(mem_req_generation[255-bank*32-:32]),
            .mem_req_tag(mem_req_tag[191-bank*24-:24]),
            .mem_req_output_block(mem_req_output_block[23-bank*3-:3]),
            .mem_req_slice(mem_req_slice[23-bank*3-:3]),
            .mem_req_source_channel(mem_req_source_channel[95-bank*12-:12]),
            .mem_req_accept(mem_req_accept[bank]),
            .mem_rsp_valid(bank_rsp_valid[bank]),
            .mem_rsp_ready(mem_rsp_ready[bank]),
            .mem_rsp_epoch(mem_rsp_epoch[127-bank*16-:16]),
            .mem_rsp_slot(mem_rsp_slot[23-bank*3-:3]),
            .mem_rsp_generation(mem_rsp_generation[255-bank*32-:32]),
            .mem_rsp_tag(mem_rsp_tag[191-bank*24-:24]),
            .mem_rsp_weight(bank_rsp_weight[bank]),
            .mem_rsp_accept(mem_rsp_accept[bank]),
            .request_count(bank_requests[bank]),
            .response_count(bank_responses[bank]),
            .pending_count(bank_pending[bank]),
            .live_slot_reuse_error(bank_reuse_error[bank]));
        for(genvar lane=0;lane<LANES;lane++) begin:g_weight_flatten
            always_comb mem_rsp_weight[1023-(bank*LANES+lane)*8-:8]
                =bank_rsp_weight[bank][lane];
        end
    end

    integer case_id,blocks,mode,events,beats;
    integer edge_ordinal,start_edge,measured_cycles,done_edge;
    integer errors,numeric_mismatches,tuple_mismatches,weight_mismatches;
    integer accepted_unknowns,protocol_errors,result_count,done_count;
    integer request_count,response_count;
    integer reference_accum[0:7][0:5][0:LANES-1];
    integer request_tuple[0:7][0:5][0:MAX_CHANNELS-1];
    integer response_tuple[0:7][0:5][0:MAX_CHANNELS-1];
    logic [95:0] payload[0:31];
    logic slot_valid[0:7][0:7];
    logic [31:0] slot_generation[0:7][0:7];
    logic [2:0] slot_block[0:7][0:7],slot_slice[0:7][0:7];
    logic [11:0] slot_channel[0:7][0:7];
    logic started,done_seen;

    function automatic integer raw_count(input integer b);
        case(b)1:return 4;2:return 8;4:return 16;default:return 32;endcase
    endfunction
    function automatic integer window_depth(input integer b);
        case(b)1:return 2;2:return 4;default:return 8;endcase
    endfunction
    function automatic integer signed weight_value(
        input integer bank,input integer lane,input integer channel,
        input integer block,input integer slice);
        integer value;
        begin value=(channel*3+bank*5+block*7+slice*11+lane*13)%31;
              return value-15; end
    endfunction
    function automatic integer expected_cycle(input integer axis,input integer c);
        if(axis==1)case(c)0:return 51;1:return 131;2:return 486;
            3:return 1231;default:return 14;endcase
        if(axis==2)case(c)0:return 53;1:return 133;2:return 499;
            3:return 1246;default:return 14;endcase
        return -1;
    endfunction

    task automatic select_case;
        begin
            if(!$value$plusargs("M979_CASE=%d",case_id))
                $fatal(1,"M979_CASE is required");
            case(case_id)
                0:begin blocks=1;mode=0;end 1:begin blocks=2;mode=1;end
                2:begin blocks=4;mode=2;end 3:begin blocks=8;mode=3;end
                4:begin blocks=1;mode=9;end
                default:$fatal(1,"M979_CASE outside 0..4");
            endcase
            beats=raw_count(blocks);
        end
    endtask

    task automatic build_payload_and_reference;
        integer row,bank,channel;
        begin
            events=0;
            for(integer beat=0;beat<32;beat++)payload[beat]=0;
            for(integer b=0;b<8;b++)for(integer s=0;s<6;s++)begin
                for(integer lane=0;lane<LANES;lane++)reference_accum[b][s][lane]=0;
                for(integer ch=0;ch<MAX_CHANNELS;ch++)begin
                    request_tuple[b][s][ch]=0;response_tuple[b][s][ch]=0;
                end
            end
            if(mode!=9)begin
                for(bank=0;bank<8;bank++)payload[0][(bank%12)*8+bank]=1;
                for(integer beat=0;beat<beats;beat++)begin
                    for(integer item=0;item<3+(mode%3);item++)begin
                        row=(beat*3+item*5+mode)%12;
                        bank=(beat+item*3+mode)%8;
                        payload[beat][row*8+bank]=1;
                    end
                    if((beat+mode)%4==0)begin
                        row=(beat+7)%12;bank=(beat*5+2)%8;
                        payload[beat][row*8+bank]=1;
                    end
                end
            end
            for(integer beat=0;beat<beats;beat++)for(row=0;row<12;row++)
                for(bank=0;bank<8;bank++)if(payload[beat][row*8+bank])begin
                    events++;channel=(beat*12+row)*8+bank;
                    for(integer b=0;b<blocks;b++)for(integer s=0;s<6;s++)
                        for(integer lane=0;lane<LANES;lane++)
                            reference_accum[b][s][lane]+=weight_value(
                                bank,lane,channel,b,s);
                end
        end
    endtask

    task automatic initialize_inputs;
        begin
            header_valid=0;header_tag=0;header_raw_beat_count=0;
            header_window_depth=0;header_output_blocks=0;
            raw_valid=0;raw_lane_valid=0;raw_beat_index=0;raw_bitmap=0;
            raw_last=0;
        end
    endtask

    task automatic drive_header_and_raw;
        integer base;
        begin
            @(negedge clk_core);header_tag=24'h979000|case_id;
            header_output_blocks=blocks;header_raw_beat_count=beats;
            header_window_depth=window_depth(blocks);header_valid=1;
            while(!header_accept)@(posedge clk_core);
            @(negedge clk_core);header_valid=0;
            for(integer packet=0;packet<beats/4;packet++)begin
                base=packet*4;raw_lane_valid=4'b1111;
                for(integer lane=0;lane<4;lane++)begin
                    raw_beat_index[19-lane*5-:5]=base+lane;
                    raw_bitmap[383-lane*96-:96]=payload[base+lane];
                end
                raw_last=(packet+1==beats/4);raw_valid=1;
                while(!raw_accept)@(posedge clk_core);
                @(negedge clk_core);
            end
            raw_valid=0;raw_lane_valid=0;raw_last=0;
        end
    endtask

    always_comb begin
        request_allow=!rst_core&&(edge_ordinal%7!=2);
        response_allow=!rst_core&&(edge_ordinal%17>=5);
        result_ready=!rst_core&&(edge_ordinal%5!=2);
        token_done_ready=!rst_core&&(edge_ordinal%4!=1);
        for(integer bank=0;bank<8;bank++)
            mem_rsp_valid[bank]=bank_rsp_valid[bank]&&response_allow;
    end

    always @(posedge clk_core)begin:monitor
        integer bank,slot,channel,block,slice;
        integer signed observed;
        if(rst_core)begin
            edge_ordinal=0;start_edge=0;measured_cycles=0;
            started=0;done_seen=0;result_count=0;done_count=0;
            request_count=0;response_count=0;
            for(bank=0;bank<8;bank++)for(slot=0;slot<8;slot++)slot_valid[bank][slot]=0;
        end else begin
            edge_ordinal++;
            if(header_accept)begin
                if(started)begin errors++;$error("duplicate header accept");end
                started=1;start_edge=edge_ordinal;
                $display("M979_SAIF_WINDOW_START axis=%s case=%0d edge=%0d",`M979_AXIS_NAME,case_id,edge_ordinal);
                if($test$plusargs("M979_UCLI_SAIF"))$stop;
            end
            for(bank=0;bank<8;bank++)begin
                if(mem_rsp_accept[bank])begin
                    if($isunknown({mem_rsp_epoch[127-bank*16-:16],
                            mem_rsp_slot[23-bank*3-:3],
                            mem_rsp_generation[255-bank*32-:32],
                            mem_rsp_tag[191-bank*24-:24],
                            mem_rsp_weight[1023-bank*128-:128]}))accepted_unknowns++;
                    slot=mem_rsp_slot[23-bank*3-:3];
                    if(!slot_valid[bank][slot]||slot_generation[bank][slot]
                            !=mem_rsp_generation[255-bank*32-:32])begin
                        errors++;tuple_mismatches++;
                    end else begin
                        block=slot_block[bank][slot];slice=slot_slice[bank][slot];
                        channel=slot_channel[bank][slot];
                        response_tuple[block][slice][channel]++;
                        for(integer lane=0;lane<LANES;lane++)if($signed(
                            mem_rsp_weight[1023-(bank*LANES+lane)*8-:8])
                            !==weight_value(bank,lane,channel,block,slice))
                            weight_mismatches++;
                    end
                    slot_valid[bank][slot]=0;response_count++;
                end
                if(mem_req_accept[bank])begin
                    if($isunknown({mem_req_epoch[127-bank*16-:16],
                            mem_req_slot[23-bank*3-:3],
                            mem_req_generation[255-bank*32-:32],
                            mem_req_tag[191-bank*24-:24],
                            mem_req_output_block[23-bank*3-:3],
                            mem_req_slice[23-bank*3-:3],
                            mem_req_source_channel[95-bank*12-:12]}))accepted_unknowns++;
                    slot=mem_req_slot[23-bank*3-:3];
                    if(slot_valid[bank][slot])begin errors++;tuple_mismatches++;end
                    slot_valid[bank][slot]=1;
                    slot_generation[bank][slot]=mem_req_generation[255-bank*32-:32];
                    slot_block[bank][slot]=mem_req_output_block[23-bank*3-:3];
                    slot_slice[bank][slot]=mem_req_slice[23-bank*3-:3];
                    slot_channel[bank][slot]=mem_req_source_channel[95-bank*12-:12];
                    request_tuple[mem_req_output_block[23-bank*3-:3]]
                        [mem_req_slice[23-bank*3-:3]]
                        [mem_req_source_channel[95-bank*12-:12]]++;
                    request_count++;
                end
            end
            if(result_accept)begin
                if($isunknown({result_tag,result_output_block,result_slice,
                               result_accumulator,result_last}))accepted_unknowns++;
                if(result_tag!==(24'h979000|case_id)||result_output_block>=blocks
                        ||result_slice>=6)errors++;
                for(integer lane=0;lane<LANES;lane++)begin
                    observed=$signed(result_accumulator[383-lane*24-:24]);
                    if(observed!==reference_accum[result_output_block]
                            [result_slice][lane])numeric_mismatches++;
                end
                result_count++;
            end
            if(token_done_accept)begin
                if(!started||token_done_tag!==(24'h979000|case_id))errors++;
                measured_cycles=edge_ordinal-start_edge+1;
                done_seen=1;done_edge=edge_ordinal;done_count++;
            end
            if(protocol_error||numeric_overflow||stale_response_seen)
                protocol_errors++;
        end
    end

    task automatic final_checks;
        integer pending_total,expected_reads,expected_results,anchor;
        begin
            pending_total=0;
            for(integer bank=0;bank<8;bank++)begin
                pending_total+=bank_pending[bank];
                if(bank_reuse_error[bank])errors++;
            end
            for(integer b=0;b<8;b++)for(integer s=0;s<6;s++)
                for(integer ch=0;ch<MAX_CHANNELS;ch++)
                    if(request_tuple[b][s][ch]!=response_tuple[b][s][ch])
                        tuple_mismatches++;
            expected_reads=events*blocks*6;expected_results=blocks*6;
            if(request_count!=expected_reads||response_count!=expected_reads
                    ||result_count!=expected_results||done_count!=1
                    ||pending_total!=0||busy)errors++;
            anchor=expected_cycle(AXIS_ID,case_id);
            if(anchor>=0&&measured_cycles!=anchor)begin
                errors++;$error("M979 M867 cycle mismatch axis=%s case=%0d got=%0d expected=%0d",`M979_AXIS_NAME,case_id,measured_cycles,anchor);
            end
            if(numeric_mismatches||tuple_mismatches||weight_mismatches
                    ||accepted_unknowns||protocol_errors)errors++;
        end
    endtask

    initial begin
        errors=0;numeric_mismatches=0;tuple_mismatches=0;weight_mismatches=0;
        accepted_unknowns=0;protocol_errors=0;initialize_inputs();select_case();
        build_payload_and_reference();rst_core=1;repeat(4)@(negedge clk_core);
        rst_core=0;repeat(2)@(posedge clk_core);
        fork
            drive_header_and_raw();
            begin integer watchdog;watchdog=0;
                while(!done_seen&&watchdog<100000)begin@(negedge clk_core);watchdog++;end
                if(!done_seen)$fatal(1,"M979 watchdog");
            end
        join
        @(negedge clk_core);final_checks();
        if(errors)$fatal(1,"M979 mapped replay failure axis=%s case=%0d errors=%0d numeric=%0d tuple=%0d weight=%0d unknown=%0d protocol=%0d",`M979_AXIS_NAME,case_id,errors,numeric_mismatches,tuple_mismatches,weight_mismatches,accepted_unknowns,protocol_errors);
        @(posedge clk_core);
        $display("PASS M979 mapped replay axis=%s case=%0d events=%0d cycles=%0d saif_duration_ns=%0d numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0",`M979_AXIS_NAME,case_id,events,measured_cycles,measured_cycles*3);
        $display("M979_SAIF_WINDOW_STOP axis=%s case=%0d edge=%0d",`M979_AXIS_NAME,case_id,edge_ordinal);
        if($test$plusargs("M979_UCLI_SAIF"))$stop;else $finish;
    end

    initial begin #1000000;$fatal(1,"M979 absolute watchdog");end
endmodule

`undef M979_DUT
`undef M979_AXIS_ID
`undef M979_AXIS_NAME
`default_nettype wire
