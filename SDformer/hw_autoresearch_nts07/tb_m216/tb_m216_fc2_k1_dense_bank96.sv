`timescale 1ns/1ps
`default_nettype none

// Legal all-one stage-3 stress: every eight-descriptor window reaches 96
// events in every bank while the K1 sink must conserve all 3,072 sources and
// replay each across eight output blocks without overflow or deadlock.
module tb_m216_fc2_k1_dense_bank96;
    logic clk_core=0,rst_core; always #1.5 clk_core=~clk_core;
    logic header_valid,header_ready,header_accept; logic[23:0]header_tag;
    logic[5:0]header_raw_beat_count;logic[3:0]header_window_depth;
    logic[3:0]header_output_blocks;logic raw_valid,raw_ready,raw_accept;
    logic[3:0]raw_lane_valid;logic[4:0]raw_beat_index[0:3];
    logic[95:0]raw_bitmap[0:3];logic raw_last;
    logic group_valid,group_ready,group_accept;logic[23:0]group_tag;
    logic[2:0]group_output_block;logic[3:0]group_source_count;
    logic[7:0]group_bank_valid;logic[11:0]group_source_channel[0:7];
    logic token_done_valid,token_done_ready,token_done_accept;
    logic[23:0]token_done_tag;logic[5:0]token_done_descriptor_count;
    logic token_done_had_event,protocol_error,busy;
    int groups,done_count,dense_packet_accepts,cycle_count,start_cycle;
    int bank96_observations;

    m216_fc2_raw4_to_source_cap_frontend #(.SOURCE_CAP(1)) dut(.*);

    task automatic drive_packet(input int base,input logic last);
        begin
            raw_lane_valid=4'b1111;raw_last=last;raw_valid=1;
            for(int lane=0;lane<4;lane++)begin
                raw_beat_index[lane]=base+lane;raw_bitmap[lane]='1;
            end
        end
    endtask

    always @(posedge clk_core) begin
        if(!rst_core)begin
            cycle_count++;
            if(protocol_error)begin
                $display("M216DBG local_fault=%0b comp_fault=%0b sink_fault=%0b header_v=%0b header_legal=%0b raw_v=%0b raw_r=%0b desc_v=%0b desc_r=%0b desc_shape=%0b desc_store=%0b fill_blocked=%0b done_v=%0b done_r=%0b done_shape=%0b entries=%0d/%0d closed=%0b/%0b",
                    dut.local_fault_q,dut.m202_protocol_error,
                    dut.m204_protocol_error,header_valid,
                    dut.header_shape_legal,raw_valid,raw_ready,
                    dut.descriptor_valid,dut.descriptor_ready,
                    dut.paired_sink.descriptor_shape_legal,
                    dut.paired_sink.descriptor_storage_legal,
                    dut.paired_sink.descriptor_fill_blocked,
                    dut.compact_done_valid,dut.compact_done_ready,
                    dut.paired_sink.done_shape_legal,
                    dut.paired_sink.entry_count_q[0],
                    dut.paired_sink.entry_count_q[1],
                    dut.paired_sink.window_closed_q[0],
                    dut.paired_sink.window_closed_q[1]);
                $fatal(1,"M216 bank96 protocol error");
            end
            if(dut.descriptor_accept
                    && dut.paired_sink.descriptor_bank_sum[0]==48)
                dense_packet_accepts++;
            if(dut.paired_sink.bank_count_q[0][0]==96
                    ||dut.paired_sink.bank_count_q[1][0]==96)
                bank96_observations++;
            if(group_accept)begin
                if(group_tag!=24'h216096||group_source_count!=1
                        ||!$onehot(group_bank_valid)
                        ||group_output_block!=groups%8)
                    $fatal(1,"M216 bank96 group mismatch");
                groups++;
            end
            if(token_done_accept)begin
                if(token_done_tag!=24'h216096
                        ||token_done_descriptor_count!=32
                        ||!token_done_had_event)
                    $fatal(1,"M216 bank96 done mismatch");
                done_count++;
            end
        end
    end

    initial begin
        rst_core=1;header_valid=0;raw_valid=0;raw_lane_valid=0;raw_last=0;
        group_ready=1;token_done_ready=1;groups=0;done_count=0;
        dense_packet_accepts=0;bank96_observations=0;cycle_count=0;
        repeat(3)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        header_tag=24'h216096;header_raw_beat_count=32;
        header_window_depth=8;header_output_blocks=8;header_valid=1;
        do @(posedge clk_core);while(!header_accept);
        start_cycle=cycle_count;
        @(negedge clk_core);header_valid=0;
        for(int base=0;base<32;base+=4)begin
            drive_packet(base,base==28);
            do @(posedge clk_core);while(!raw_accept);
            @(negedge clk_core);
        end
        raw_valid=0;raw_last=0;
        do @(posedge clk_core);while(!token_done_accept);
        @(negedge clk_core);
        if(groups!=24576||done_count!=1||dense_packet_accepts!=8
                ||bank96_observations==0)
            $fatal(1,"M216 bank96 census mismatch");
        $display("PASS M216 K1 dense bank96 VCS events=3072 groups=%0d done=%0d dense_packet_accepts=%0d bank96_observations=%0d header_to_done_cycles=%0d source_conservation=true complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            groups,done_count,dense_packet_accepts,bank96_observations,
            cycle_count-start_cycle);
        $finish;
    end
    initial begin #10000000;$fatal(1,"M216 bank96 watchdog");end
endmodule

`default_nettype wire
