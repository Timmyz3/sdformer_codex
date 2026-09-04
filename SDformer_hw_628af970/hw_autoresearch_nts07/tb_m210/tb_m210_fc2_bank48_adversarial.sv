`timescale 1ns/1ps
`default_nettype none

// Legal worst-case per-packet/per-window bank population.  This is the P0
// regression that deadlocks M207's five-bit descriptor_bank_sum.
module tb_m210_fc2_bank48_adversarial;
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
    int groups,done_count,bank48_accepts,cycle_count,start_cycle;

    m210_fc2_raw4_to_paired_window_handoff_frontend dut(.*);

    task automatic drive_packet(input int base,input logic last);
        begin
            raw_lane_valid=4'b1111;raw_last=last;raw_valid=1;
            for(int lane=0;lane<4;lane++)begin
                raw_beat_index[lane]=base+lane;raw_bitmap[lane]=0;
                for(int row=0;row<12;row++)
                    raw_bitmap[lane][row*8]=1;
            end
        end
    endtask

    always @(posedge clk_core) begin
        if(!rst_core)begin
            cycle_count++;
            if(protocol_error)$fatal(1,"M210 bank48 protocol error");
            if(dut.descriptor_accept
                    && dut.paired_sink.descriptor_bank_sum[0]==48)
                bank48_accepts++;
            if(group_accept)begin
                if(group_tag!=24'h210048||group_source_count!=1
                        ||group_bank_valid!=8'b00000001)
                    $fatal(1,"M210 bank48 group mismatch");
                groups++;
            end
            if(token_done_accept)begin
                if(token_done_tag!=24'h210048
                        ||token_done_descriptor_count!=8
                        ||!token_done_had_event)
                    $fatal(1,"M210 bank48 done mismatch");
                done_count++;
            end
        end
    end

    initial begin
        rst_core=1;header_valid=0;raw_valid=0;raw_lane_valid=0;raw_last=0;
        group_ready=1;token_done_ready=1;groups=0;done_count=0;
        bank48_accepts=0;cycle_count=0;
        repeat(3)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        header_tag=24'h210048;header_raw_beat_count=8;
        header_window_depth=4;header_output_blocks=2;header_valid=1;
        do @(posedge clk_core);while(!header_accept);
        start_cycle=cycle_count;
        @(negedge clk_core);header_valid=0;drive_packet(0,0);
        do @(posedge clk_core);while(!raw_accept);
        @(negedge clk_core);drive_packet(4,1);
        do @(posedge clk_core);while(!raw_accept);
        @(negedge clk_core);raw_valid=0;raw_last=0;
        do @(posedge clk_core);while(!token_done_accept);
        @(negedge clk_core);
        if(groups!=192||done_count!=1||bank48_accepts!=2)
            $fatal(1,"M210 bank48 census mismatch");
        $display("PASS M210 bank48 adversarial VCS groups=%0d done=%0d bank48_accepts=%0d header_to_done_cycles=%0d m207_deadlocks=true complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            groups,done_count,bank48_accepts,cycle_count-start_cycle);
        $finish;
    end
    initial begin #10000000;$fatal(1,"M210 bank48 watchdog");end
endmodule

`default_nettype wire
