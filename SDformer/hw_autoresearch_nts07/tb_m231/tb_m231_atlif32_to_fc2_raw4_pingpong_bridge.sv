`timescale 1ns/1ps
`default_nettype none
`ifndef M231_INPUT_WIDTH
`define M231_INPUT_WIDTH 384
`endif
module tb_m231_atlif32_to_fc2_raw4_pingpong_bridge;
    localparam int INPUT_WIDTH=`M231_INPUT_WIDTH;
    localparam int TAG_BITS=24,GROUPS=INPUT_WIDTH/16;
    localparam int RAW_BEATS=INPUT_WIDTH/96,RAW_PACKETS=RAW_BEATS/4;
    logic clk_core=0,rst_core;always #1.5 clk_core=~clk_core;
    logic pair_header_valid,pair_header_ready,pair_header_accept;
    logic[TAG_BITS-2:0]pair_header_tag;
    logic event_valid,event_ready,event_accept,event_last_group;
    logic[TAG_BITS-2:0]event_pair_tag;logic[7:0]event_group_index;
    logic[31:0]event_bits;logic header_valid,header_ready,header_accept;
    logic[TAG_BITS-1:0]header_tag;logic[5:0]header_raw_beat_count;
    logic[3:0]header_window_depth,header_output_blocks;
    logic raw_valid,raw_ready,raw_last,raw_accept;logic[3:0]raw_lane_valid;
    logic[4:0]raw_beat_index[0:3];logic[95:0]raw_bitmap[0:3];
    logic protocol_error,busy;logic[1:0]debug_full_slots;
    logic[7:0]debug_fill_group;logic[31:0]debug_pair_count;
    logic[31:0]debug_token_count,debug_raw_packet_count;
    logic[INPUT_WIDTH-1:0]expected_row[0:15][0:1];
    logic[TAG_BITS-2:0]expected_tag[0:15];
    integer pairs_sent,pairs_checked,headers_checked,packets_checked;
    integer header_stalls,raw_stalls,full_hits,attacks,fault_atomic_attacks;
    integer cycle_count;
    integer expected_pair_index,expected_row_index,expected_packet_index;
    logic random_stall,force_output_stall;

    m231_atlif32_to_fc2_raw4_pingpong_bridge#(
        .INPUT_WIDTH(INPUT_WIDTH),.TAG_BITS(TAG_BITS))dut(.*);
    m231_atlif32_to_fc2_raw4_pingpong_bridge_assertions#(
        .INPUT_WIDTH(INPUT_WIDTH),.TAG_BITS(TAG_BITS))sva(.*);

    function automatic logic[15:0] pattern(input integer pair_index,
            input integer row,input integer group_index);
        logic[15:0]v;begin
            for(int bit_index=0;bit_index<16;bit_index++)
                v[bit_index]=((pair_index*37+row*19+group_index*11
                    +bit_index*7)%13)<5;
            return v;
        end
    endfunction

    always @(negedge clk_core)begin
        if(rst_core)begin header_ready=0;raw_ready=0;end
        else if(force_output_stall)begin header_ready=0;raw_ready=0;end
        else if(random_stall)begin
            header_ready=(cycle_count%5)!=1;
            raw_ready=(cycle_count%7)!=2&&(cycle_count%11)!=4;
        end else begin header_ready=1;raw_ready=1;end
    end
    always @(posedge clk_core)begin
        if(rst_core)begin cycle_count=0;header_stalls=0;raw_stalls=0;full_hits=0;end
        else begin
            cycle_count++;
            if(header_valid&&!header_ready)header_stalls++;
            if(raw_valid&&!raw_ready)raw_stalls++;
            if(debug_full_slots==2'b11)full_hits++;
            if(header_accept)begin
                if(header_tag!=={expected_tag[expected_pair_index],expected_row_index[0]})
                    $fatal(1,"M231 header tag mismatch W=%0d pair=%0d row=%0d",INPUT_WIDTH,expected_pair_index,expected_row_index);
                if(header_raw_beat_count!=RAW_BEATS
                        ||header_output_blocks!=INPUT_WIDTH/384)
                    $fatal(1,"M231 header geometry mismatch W=%0d",INPUT_WIDTH);
                headers_checked++;
            end
            if(raw_accept)begin
                for(int lane=0;lane<4;lane++)begin integer beat;
                    beat=expected_packet_index*4+lane;
                    if(raw_beat_index[lane]!==beat
                            ||raw_bitmap[lane]!==expected_row[expected_pair_index][expected_row_index][beat*96+:96])
                        $fatal(1,"M231 transpose mismatch W=%0d pair=%0d row=%0d packet=%0d lane=%0d",INPUT_WIDTH,expected_pair_index,expected_row_index,expected_packet_index,lane);
                end
                if(raw_last!==(expected_packet_index==RAW_PACKETS-1))
                    $fatal(1,"M231 raw_last mismatch W=%0d",INPUT_WIDTH);
                packets_checked++;
                if(expected_packet_index==RAW_PACKETS-1)begin
                    expected_packet_index=0;
                    if(expected_row_index==0)expected_row_index=1;
                    else begin expected_row_index=0;expected_pair_index++;
                        pairs_checked++;end
                end else expected_packet_index++;
            end
        end
    end

    task automatic clear_inputs;begin
        pair_header_valid=0;pair_header_tag=0;event_valid=0;event_pair_tag=0;
        event_group_index=0;event_bits=0;event_last_group=0;
    end endtask
    task automatic reset;begin @(negedge clk_core);clear_inputs();rst_core=1;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;end endtask
    task automatic send_pair(input integer pair_index);begin
        expected_tag[pair_index]=23'h23100+pair_index;
        expected_row[pair_index][0]='0;expected_row[pair_index][1]='0;
        @(negedge clk_core);pair_header_valid=1;
        pair_header_tag=expected_tag[pair_index];
        do @(posedge clk_core);while(!pair_header_accept);
        @(negedge clk_core);pair_header_valid=0;
        for(integer group_index=0;group_index<GROUPS;group_index++)begin
            logic[15:0]r0,r1;r0=pattern(pair_index,0,group_index);
            r1=pattern(pair_index,1,group_index);
            expected_row[pair_index][0][group_index*16+:16]=r0;
            expected_row[pair_index][1][group_index*16+:16]=r1;
            event_valid=1;event_pair_tag=expected_tag[pair_index];
            event_group_index=group_index;event_bits={r1,r0};
            event_last_group=group_index==GROUPS-1;
            do @(posedge clk_core);while(!event_accept);
            @(negedge clk_core);
        end
        event_valid=0;event_last_group=0;pairs_sent++;
    end endtask
    task automatic illegal_event_attack;begin
        reset();@(negedge clk_core);event_valid=1;event_pair_tag=23'h7;
        event_group_index=0;event_last_group=0;event_bits=32'hdeadbeef;#0.1;
        if(event_ready)$fatal(1,"M231 illegal event ready W=%0d",INPUT_WIDTH);
        @(posedge clk_core);#0.2;if(!protocol_error)
            $fatal(1,"M231 illegal event not quarantined W=%0d",INPUT_WIDTH);
        attacks++;@(negedge clk_core);event_valid=0;
    end endtask
    task automatic fault_cycle_raw_accept_attack;integer packets_before;begin
        reset();random_stall=0;force_output_stall=1;
        send_pair(0);wait(debug_full_slots!=0);
        @(negedge clk_core);force_output_stall=0;
        wait(raw_valid&&raw_ready);@(negedge clk_core);
        if(!raw_valid||!raw_ready)
            $fatal(1,"M231 fault attack missed legal raw beat W=%0d",INPUT_WIDTH);
        packets_before=debug_raw_packet_count;
        event_valid=1;event_pair_tag=expected_tag[0]^23'h1;
        event_group_index=0;event_last_group=0;event_bits=32'hc0decafe;
        #0.1;
        if(!protocol_error||raw_valid||raw_accept||header_valid||header_accept
                ||pair_header_ready||event_ready||event_accept)
            $fatal(1,"M231 same-cycle fault not atomic W=%0d pe=%0b rv=%0b ra=%0b",
                INPUT_WIDTH,protocol_error,raw_valid,raw_accept);
        @(posedge clk_core);#0.2;
        if(debug_raw_packet_count!=packets_before||!protocol_error)
            $fatal(1,"M231 fault cycle committed downstream state W=%0d",INPUT_WIDTH);
        attacks++;fault_atomic_attacks++;
        @(negedge clk_core);event_valid=0;force_output_stall=0;random_stall=1;
    end endtask

    initial begin #5000000;$fatal(1,"M231 watchdog W=%0d",INPUT_WIDTH);end
    initial begin rst_core=1;random_stall=1;force_output_stall=0;
        clear_inputs();pairs_sent=0;
        pairs_checked=0;headers_checked=0;packets_checked=0;attacks=0;
        fault_atomic_attacks=0;
        expected_pair_index=0;expected_row_index=0;expected_packet_index=0;
        repeat(4)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        illegal_event_attack();fault_cycle_raw_accept_attack();
        reset();pairs_sent=0;pairs_checked=0;headers_checked=0;
        packets_checked=0;expected_pair_index=0;expected_row_index=0;
        expected_packet_index=0;force_output_stall=1;
        send_pair(0);send_pair(1);wait(debug_full_slots==2'b11);
        @(negedge clk_core);force_output_stall=0;
        send_pair(2);
        wait(pairs_checked==3);repeat(3)@(posedge clk_core);
        if(debug_pair_count!=3||debug_token_count!=6
                ||debug_raw_packet_count!=6*RAW_PACKETS)
            $fatal(1,"M231 conservation mismatch W=%0d",INPUT_WIDTH);
        if(header_stalls==0||raw_stalls==0||full_hits==0)
            $fatal(1,"M231 coverage hole W=%0d h=%0d r=%0d full=%0d",INPUT_WIDTH,header_stalls,raw_stalls,full_hits);
        if(fault_atomic_attacks!=1||attacks!=2)
            $fatal(1,"M231 fault coverage mismatch W=%0d attacks=%0d atomic=%0d",
                INPUT_WIDTH,attacks,fault_atomic_attacks);
        $display("PASS M231r2 W=%0d pairs=3 tokens=6 packets=%0d header_stalls=%0d raw_stalls=%0d full_hits=%0d attacks=%0d fault_atomic=%0d cycles=%0d",INPUT_WIDTH,6*RAW_PACKETS,header_stalls,raw_stalls,full_hits,attacks,fault_atomic_attacks,cycle_count);
        $finish;end
endmodule
`default_nettype wire
