`timescale 1ns/1ps
`default_nettype none

module tb_m1090_c2_k1_observation_mapped_case0_short;
    localparam int LANES=16;
    logic clk_core=0,rst_core;
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
    logic obs_header_accept,obs_raw_accept,obs_busy,obs_protocol_error;
    logic obs_numeric_overflow,obs_stale_response,obs_fault;
    logic [7:0] obs_bank_request_accept,obs_bank_response_accept;
    logic [5:0] obs_service_fifo_count;
    logic [6:0] obs_service_outstanding_count;
    logic [31:0] obs_service_group_count,obs_service_request_count;
    logic [31:0] obs_service_response_count,obs_service_context_count;
    logic [31:0] obs_service_result_count,obs_service_active_bank_read_count;
    logic [3:0] obs_adapter_live_slots;
    logic [31:0] obs_adapter_bundle_request_count;
    logic [31:0] obs_adapter_bank_request_count;
    logic [31:0] obs_adapter_bank_response_count;
    logic [31:0] obs_adapter_bundle_response_count;

    m1090_c2_k1_observation_wrapper dut (
        .clk_core(clk_core),.rst_core(rst_core),.header_valid(header_valid),
        .header_ready(header_ready),.header_tag(header_tag),
        .header_raw_beat_count(header_raw_beat_count),
        .header_window_depth(header_window_depth),
        .header_output_blocks(header_output_blocks),.header_accept(header_accept),
        .raw_valid(raw_valid),.raw_ready(raw_ready),.raw_lane_valid(raw_lane_valid),
        .raw_beat_index(raw_beat_index),.raw_bitmap(raw_bitmap),.raw_last(raw_last),
        .raw_accept(raw_accept),.mem_req_valid(mem_req_valid),
        .mem_req_ready(mem_req_ready),.mem_req_epoch(mem_req_epoch),
        .mem_req_slot(mem_req_slot),.mem_req_generation(mem_req_generation),
        .mem_req_tag(mem_req_tag),.mem_req_output_block(mem_req_output_block),
        .mem_req_slice(mem_req_slice),.mem_req_source_channel(mem_req_source_channel),
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
        .numeric_overflow(numeric_overflow),.stale_response_seen(stale_response_seen),
        .busy(busy),.obs_header_accept(obs_header_accept),
        .obs_raw_accept(obs_raw_accept),.obs_busy(obs_busy),
        .obs_protocol_error(obs_protocol_error),
        .obs_numeric_overflow(obs_numeric_overflow),
        .obs_stale_response(obs_stale_response),.obs_fault(obs_fault),
        .obs_bank_request_accept(obs_bank_request_accept),
        .obs_bank_response_accept(obs_bank_response_accept),
        .obs_service_fifo_count(obs_service_fifo_count),
        .obs_service_outstanding_count(obs_service_outstanding_count),
        .obs_service_group_count(obs_service_group_count),
        .obs_service_request_count(obs_service_request_count),
        .obs_service_response_count(obs_service_response_count),
        .obs_service_context_count(obs_service_context_count),
        .obs_service_result_count(obs_service_result_count),
        .obs_service_active_bank_read_count(obs_service_active_bank_read_count),
        .obs_adapter_live_slots(obs_adapter_live_slots),
        .obs_adapter_bundle_request_count(obs_adapter_bundle_request_count),
        .obs_adapter_bank_request_count(obs_adapter_bank_request_count),
        .obs_adapter_bank_response_count(obs_adapter_bank_response_count),
        .obs_adapter_bundle_response_count(obs_adapter_bundle_response_count));

    logic [7:0] bank_rsp_valid;
    logic signed [7:0] bank_rsp_weight[0:7][0:LANES-1];
    logic [31:0] bank_requests[0:7],bank_responses[0:7];
    logic [3:0] bank_pending[0:7];
    logic bank_reuse_error[0:7];
    logic request_allow,response_allow;

    for(genvar bank=0;bank<8;bank++)begin:g_memory
        m349_fc2_scalar_bank_memory_model #(.BANK_ID(bank)) memory(
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
            .request_count(bank_requests[bank]),.response_count(bank_responses[bank]),
            .pending_count(bank_pending[bank]),
            .live_slot_reuse_error(bank_reuse_error[bank]));
        for(genvar lane=0;lane<LANES;lane++)begin:g_flatten
            always_comb mem_rsp_weight[1023-(bank*LANES+lane)*8-:8]
                =bank_rsp_weight[bank][lane];
        end
    end

    always_comb begin
        request_allow=!rst_core; response_allow=!rst_core;
        result_ready=!rst_core; token_done_ready=!rst_core;
        for(integer bank=0;bank<8;bank++)
            mem_rsp_valid[bank]=bank_rsp_valid[bank]&&response_allow;
    end

`define M1090_FAIL_X(signal_name) \
    if($isunknown(signal_name))begin \
        $display("M1090_FIRST_X cycle=%0d signal=%s value=%b",window_cycle,`"signal_name`",signal_name); \
        $fatal(1,"M1090 fail-closed on first unknown"); \
    end

    integer window_cycle;
    logic header_seen,raw_seen;
    always @(posedge clk_core)begin
        if(rst_core)begin window_cycle=0;header_seen=0;raw_seen=0;end
        else begin
            if(header_accept)header_seen=1;
            if(header_seen||header_accept)begin
                `M1090_FAIL_X(obs_header_accept)
                `M1090_FAIL_X(obs_raw_accept)
                `M1090_FAIL_X(obs_busy)
                `M1090_FAIL_X(obs_protocol_error)
                `M1090_FAIL_X(obs_numeric_overflow)
                `M1090_FAIL_X(obs_stale_response)
                `M1090_FAIL_X(obs_fault)
                `M1090_FAIL_X(obs_bank_request_accept)
                `M1090_FAIL_X(obs_bank_response_accept)
                `M1090_FAIL_X(obs_service_fifo_count)
                `M1090_FAIL_X(obs_service_outstanding_count)
                `M1090_FAIL_X(obs_service_group_count)
                `M1090_FAIL_X(obs_service_request_count)
                `M1090_FAIL_X(obs_service_response_count)
                `M1090_FAIL_X(obs_service_context_count)
                `M1090_FAIL_X(obs_service_result_count)
                `M1090_FAIL_X(obs_service_active_bank_read_count)
                `M1090_FAIL_X(obs_adapter_live_slots)
                `M1090_FAIL_X(obs_adapter_bundle_request_count)
                `M1090_FAIL_X(obs_adapter_bank_request_count)
                `M1090_FAIL_X(obs_adapter_bank_response_count)
                `M1090_FAIL_X(obs_adapter_bundle_response_count)
                $display("M1090_STAGE cycle=%0d h=%b raw=%b busy=%b fault=%b fifo=%0d out=%0d group=%0d svc_req=%0d svc_rsp=%0d ctx=%0d result=%0d active=%0d live=%0d bundle=%0d/%0d bank=%0d/%0d accept=%b/%b",
                    window_cycle,obs_header_accept,obs_raw_accept,obs_busy,obs_fault,
                    obs_service_fifo_count,obs_service_outstanding_count,
                    obs_service_group_count,obs_service_request_count,
                    obs_service_response_count,obs_service_context_count,
                    obs_service_result_count,obs_service_active_bank_read_count,
                    obs_adapter_live_slots,obs_adapter_bundle_request_count,
                    obs_adapter_bundle_response_count,obs_adapter_bank_request_count,
                    obs_adapter_bank_response_count,obs_bank_request_accept,
                    obs_bank_response_accept);
                if(raw_accept)raw_seen=1;
                window_cycle=window_cycle+1;
                if(window_cycle==128)begin
                    if(!raw_seen)$fatal(1,"M1090 no raw acceptance in short window");
                    $display("PASS_M1090_OBSERVATION_SHORT_WINDOW cycles=128 raw_seen=1 no_unknown=1 diagnostic_only=1");
                    $finish;
                end
            end
        end
    end
`undef M1090_FAIL_X

    initial begin:stimulus
        integer wait_cycles;
        rst_core=1;header_valid=0;header_tag=24'h109000;
        header_raw_beat_count=6'd4;header_window_depth=4'd2;
        header_output_blocks=4'd1;raw_valid=0;raw_lane_valid=0;
        raw_beat_index=0;raw_bitmap=0;raw_last=0;
        repeat(5)@(posedge clk_core);
        @(negedge clk_core);rst_core=0;header_valid=1;
        wait_cycles=0;
        while(!header_accept&&wait_cycles<16)begin@(posedge clk_core);wait_cycles++;end
        if(!header_accept)$fatal(1,"M1090 header not accepted within 16 cycles");
        @(negedge clk_core);header_valid=0;raw_lane_valid=4'b1111;
        for(integer lane=0;lane<4;lane++)begin
            raw_beat_index[19-lane*5-:5]=lane;
            raw_bitmap[383-lane*96-:96]=96'h000000000000000000000101<<(lane*8);
        end
        raw_last=1;raw_valid=1;wait_cycles=0;
        while(!raw_accept&&wait_cycles<32)begin@(posedge clk_core);wait_cycles++;end
        if(!raw_accept)$fatal(1,"M1090 raw not accepted within 32 cycles");
        @(negedge clk_core);raw_valid=0;raw_lane_valid=0;raw_last=0;
    end

    initial begin
        #1000 $fatal(1,"M1090 absolute watchdog");
    end
endmodule

`default_nettype wire
