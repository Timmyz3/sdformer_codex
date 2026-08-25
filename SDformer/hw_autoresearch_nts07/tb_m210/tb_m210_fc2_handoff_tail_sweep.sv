`timescale 1ns/1ps
`default_nettype none

// Continuous-source controller sweep for calibrating the software recurrence
// against the exact M202 -> M210 ready/valid state machines.  This is a
// verification workload, not frozen-H67 performance evidence by itself.
module tb_m210_fc2_handoff_tail_sweep;
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
    logic[95:0]payload[0:31];int cycle_counter,total_cases;
    m210_fc2_raw4_to_paired_window_handoff_frontend dut(.*);

    function automatic logic[95:0]event_pattern(input int beat,input int seed);
        logic[95:0]v;int b0,b1,b2,r0,r1,r2;
        begin
            v=0;b0=(beat*5+seed)%8;r0=(beat*3+seed*2)%12;v[r0*8+b0]=1;
            b1=(beat*3+seed+1)%8;r1=(beat*7+seed+3)%12;
            if((beat+seed)%2==0)v[r1*8+b1]=1;
            b2=(beat+seed*3+2)%8;r2=(beat*11+seed+5)%12;
            if((beat+2*seed)%5==0)v[r2*8+b2]=1;
            return v;
        end
    endfunction
    task automatic shape(input int blocks,output int raw_count,output int depth);
        case(blocks)
            1:begin raw_count=4;depth=2;end
            2:begin raw_count=8;depth=4;end
            4:begin raw_count=16;depth=8;end
            8:begin raw_count=32;depth=8;end
            default:$fatal(1,"shape");
        endcase
    endtask
    task automatic run_case(input int blocks,input int mode,input int seed);
        int raw_count,depth,descriptors,base,start_cycle;logic accepted;
        begin
            shape(blocks,raw_count,depth);descriptors=0;
            for(int beat=0;beat<raw_count;beat++)begin
                case(mode)
                    0:payload[beat]=event_pattern(beat,seed);
                    1:payload[beat]=(((beat*7+seed)%5)<2)
                        ?event_pattern(beat,seed):0;
                    2:payload[beat]=(beat<(seed%(raw_count+1)))
                        ?event_pattern(beat,seed):0;
                    default:payload[beat]=0;
                endcase
                if(payload[beat]!=0)descriptors++;
            end
            @(negedge clk_core);header_tag=24'h520000+total_cases;
            header_raw_beat_count=raw_count;header_window_depth=depth;
            header_output_blocks=blocks;header_valid=1;
            do @(posedge clk_core);while(!header_accept);
            start_cycle=cycle_counter;
            @(negedge clk_core);header_valid=0;base=0;raw_lane_valid=4'b1111;
            for(int lane=0;lane<4;lane++)begin
                raw_beat_index[lane]=lane;raw_bitmap[lane]=payload[lane];
            end
            raw_last=(raw_count==4);raw_valid=1;
            while(base<raw_count)begin
                @(posedge clk_core);accepted=raw_accept;
                @(negedge clk_core);if(accepted)begin
                    base+=4;
                    if(base<raw_count)begin
                        for(int lane=0;lane<4;lane++)begin
                            raw_beat_index[lane]=base+lane;
                            raw_bitmap[lane]=payload[base+lane];
                        end
                        raw_last=(base+4==raw_count);
                    end else begin raw_valid=0;raw_last=0;end
                end
            end
            do @(posedge clk_core);while(!token_done_accept);
            $display("M210TAIL blocks=%0d mode=%0d seed=%0d descriptors=%0d measured=%0d",
                blocks,mode,seed,descriptors,cycle_counter-start_cycle);
            total_cases++;@(negedge clk_core);
        end
    endtask
    always @(posedge clk_core)begin
        if(!rst_core)begin cycle_counter++;
            if(protocol_error)$fatal(1,"M210 tail sweep protocol error");
        end
    end
    initial begin
        rst_core=1;header_valid=0;raw_valid=0;raw_lane_valid=0;raw_last=0;
        group_ready=1;token_done_ready=1;cycle_counter=0;total_cases=0;
        repeat(3)@(posedge clk_core);@(negedge clk_core);rst_core=0;
        for(int shape_code=0;shape_code<4;shape_code++)begin
            int blocks;case(shape_code)0:blocks=1;1:blocks=2;2:blocks=4;default:blocks=8;endcase
            for(int mode=0;mode<4;mode++)
                for(int seed=0;seed<16;seed++)run_case(blocks,mode,seed);
        end
        $display("PASS M210 terminal-collapse tail sweep cases=%0d",total_cases);$finish;
    end
    initial begin #100000000;$fatal(1,"M210 tail sweep watchdog");end
endmodule

`default_nettype wire
