`timescale 1ns/1ps
`default_nettype none

module tb_m214_fc2_stage0_handoff_prefetch;
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
    int cycle_count,start_cycle,groups,done_count,handoffs,stalls;
    logic stall_injected;

    m214_fc2_raw4_to_same_done_load_frontend dut(.*);

    always @(posedge clk_core) begin
        if (!rst_core) begin
            cycle_count++;
            if (protocol_error) $fatal(1,"M214 handoff protocol error");
            if (dut.paired_sink.stage0_handoff_load) handoffs++;
            if (group_valid && !group_ready) stalls++;
            if (group_accept) begin
                if (group_tag != 24'h210001 || group_output_block != 0)
                    $fatal(1,"M214 handoff group identity mismatch");
                groups++;
            end
            if (token_done_accept) begin
                if (token_done_tag != 24'h210001
                        || token_done_descriptor_count != 4
                        || !token_done_had_event)
                    $fatal(1,"M214 handoff done mismatch");
                done_count++;
            end
        end
    end

    // Stall the prefetched second-window group for one cycle and require the
    // existing stable-under-stall assertion to hold its full payload.
    always @(negedge clk_core) begin
        if (!rst_core && handoffs == 1 && group_valid && !stall_injected) begin
            group_ready = 0; stall_injected = 1;
        end else if (!rst_core && !group_ready) begin
            group_ready = 1;
        end
    end

    initial begin
        rst_core=1;header_valid=0;raw_valid=0;raw_lane_valid=0;raw_last=0;
        group_ready=1;token_done_ready=1;cycle_count=0;groups=0;
        done_count=0;handoffs=0;stalls=0;stall_injected=0;
        repeat(3)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        header_tag=24'h210001;header_raw_beat_count=4;
        header_window_depth=2;header_output_blocks=1;header_valid=1;
        do @(posedge clk_core);while(!header_accept);
        start_cycle=cycle_count;
        @(negedge clk_core);header_valid=0;raw_lane_valid=4'b1111;
        for(int lane=0;lane<4;lane++)begin
            raw_beat_index[lane]=lane;raw_bitmap[lane]=0;
            raw_bitmap[lane][lane*8+lane]=1;
        end
        raw_last=1;raw_valid=1;
        do @(posedge clk_core);while(!raw_accept);
        @(negedge clk_core);raw_valid=0;raw_last=0;
        do @(posedge clk_core);while(!token_done_accept);
        @(negedge clk_core);
        if(groups!=2||done_count!=1||handoffs!=1||stalls!=1)
            $fatal(1,"M214 handoff coverage mismatch");
        $display("PASS M214 stage0 handoff prefetch groups=%0d done=%0d handoffs=%0d stalls=%0d header_to_done_cycles=%0d complete_fc2=false physical_speedup=false system_speedup=false headline=false",
            groups,done_count,handoffs,stalls,cycle_count-start_cycle);
        $finish;
    end
    initial begin #1000000;$fatal(1,"M214 handoff watchdog");end
endmodule

`default_nettype wire
