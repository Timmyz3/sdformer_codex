`timescale 1ns/1ps
`default_nettype none
module tb_m273_integrated_rank3_atlif;
    localparam int TAG_W=48;
    localparam int FIFO_DEPTH=16;
    localparam int FIFO_COUNT_W=$clog2(FIFO_DEPTH+1);

    logic clk_core,rst_core;
    logic config_valid,config_ready,config_accept,config_last;
    logic[255:0]config_data;
    logic raw_valid,raw_ready,raw_accept,raw_last;
    logic[255:0]raw_data;
    logic[TAG_W-1:0]raw_tag;
    logic result_valid,result_ready,result_accept;
    logic[TAG_W-1:0]result_tag;
    logic[2:0]result_beat;
    logic[47:0]result_valid_bits,result_data;
    logic release_valid,release_ready,release_accept;
    logic tile_done_valid,context_retire_valid,config_loaded,protocol_error,busy;
    logic[TAG_W-1:0]tile_done_tag;
    logic[31:0]context_retire_cycles;
    logic stage1_issue,stage2_issue,product_push,product_replace,fifo_push,fifo_pop;
    logic[FIFO_COUNT_W-1:0]result_fifo_occupancy;
    logic[1:0]raw_bank_occupancy,intermediate_bank_occupancy;
    logic[31:0]debug_config_beats,debug_raw_beats,debug_tiles_loaded;
    logic[31:0]debug_stage1_issues,debug_stage1_done;
    logic[31:0]debug_stage2_issues,debug_stage2_done;
    logic[31:0]debug_product_pushes,debug_result_departures;
    logic[31:0]debug_product_replacements,debug_context_cycles;

    logic[1535:0]legal_config;
    logic[TAG_W-1:0]expected_tag[0:4095];
    logic[2:0]expected_beat[0:4095];
    logic[47:0]expected_data[0:4095];
    integer expected_read,expected_write,numerical_mismatches;
    integer ready_mode,ready_phase;
    integer observed_fifo_peak,observed_raw_stalls,observed_result_stalls;
    integer observed_overlap,observed_replacements,observed_full_pop_push;
    integer observed_tile_done;

    m273_integrated_rank3_atlif u_dut(.*);
    m273_integrated_rank3_atlif_assertions u_sva(.*);

    always #5 clk_core=~clk_core;

    function automatic integer right_coefficient(input integer rank,input integer time_index);
        begin
            case(rank)
                0:right_coefficient=1;
                1:right_coefficient=(time_index%2)==0?1:-1;
                default:right_coefficient=(time_index%3)==0?2:0;
            endcase
        end
    endfunction

    function automatic integer left_coefficient(input integer output_row,input integer rank);
        integer selector;
        begin
            selector=(output_row+rank)%5;
            case(selector)
                0:left_coefficient=0;
                1:left_coefficient=1;
                2:left_coefficient=-1;
                3:left_coefficient=2;
                default:left_coefficient=-2;
            endcase
        end
    endfunction

    function automatic integer raw_value(
        input integer tile_seed,input integer time_index,input integer lane);
        begin raw_value=((tile_seed*3+time_index*2+lane)%11)-5;end
    endfunction

    task automatic make_legal_config;
        integer coefficient,rank,time_index,output_row,value,term_shift;
        begin
            legal_config='0;
            for(rank=0;rank<3;rank=rank+1)
                for(time_index=0;time_index<10;time_index=time_index+1)begin
                    value=right_coefficient(rank,time_index);
                    legal_config[((rank*10+time_index)*8)+:8]=value[7:0];
                end
            legal_config[240+:5]=5'd0;
            for(output_row=0;output_row<10;output_row=output_row+1)begin
                for(rank=0;rank<3;rank=rank+1)begin
                    coefficient=output_row*3+rank;
                    value=left_coefficient(output_row,rank);
                    legal_config[245+(coefficient*8)+:8]=value[7:0];
                    if(value!=0)begin
                        term_shift=(value==2||value==-2)?1:0;
                        legal_config[485+(coefficient*4)]=1'b1;
                        legal_config[605+(coefficient*4)]=(value<0);
                        legal_config[725+(coefficient*12)+:3]=term_shift[2:0];
                    end
                end
                value=output_row-5;
                legal_config[1085+(output_row*24)+:24]=value[23:0];
            end
            legal_config[1325+:24]=24'sd0;
        end
    endtask

    task automatic enqueue_expected_tile(
        input integer tile_seed,input logic[TAG_W-1:0]tag_value);
        integer intermediate[0:47];
        integer rank,lane,time_index,row,beat,row_in_beat,total;
        logic[47:0]packed_result;
        begin
            for(rank=0;rank<3;rank=rank+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    total=0;
                    for(time_index=0;time_index<10;time_index=time_index+1)
                        total=total+raw_value(tile_seed,time_index,lane)
                            *right_coefficient(rank,time_index);
                    if(total>127)total=127;
                    else if(total< -128)total=-128;
                    intermediate[rank*16+lane]=total;
                end
            for(beat=0;beat<5;beat=beat+1)begin
                packed_result='0;
                for(row_in_beat=0;row_in_beat<2;row_in_beat=row_in_beat+1)begin
                    row=beat*2+row_in_beat;
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=row-5;
                        for(rank=0;rank<3;rank=rank+1)
                            total=total+intermediate[rank*16+lane]
                                *left_coefficient(row,rank);
                        packed_result[row_in_beat*16+lane]=(total>=0);
                    end
                end
                expected_tag[expected_write]=tag_value;
                expected_beat[expected_write]=beat[2:0];
                expected_data[expected_write]=packed_result;
                expected_write=expected_write+1;
            end
        end
    endtask

    task automatic clear_observation;
        begin
            expected_read=0;expected_write=0;numerical_mismatches=0;
            observed_fifo_peak=0;observed_raw_stalls=0;
            observed_result_stalls=0;observed_overlap=0;
            observed_replacements=0;observed_full_pop_push=0;
            observed_tile_done=0;ready_phase=0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core=1'b1;config_valid=1'b0;config_data='0;config_last=1'b0;
            raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
            release_valid=1'b0;ready_mode=0;
            repeat(4)@(posedge clk_core);
            @(negedge clk_core);rst_core=1'b0;
            clear_observation();
        end
    endtask

    task automatic send_config(
        input integer attack,input logic[1535:0]frame,input bit expect_fault,
        output time first_accept_time);
        integer beat;
        logic last_value;
        begin
            first_accept_time=0;config_valid=1'b1;
            for(beat=0;beat<6;beat=beat+1)begin
                config_data=frame[beat*256+:256];
                last_value=(beat==5);
                if(attack==1&&beat==2)last_value=1'b1;
                if(attack==2&&beat==5)last_value=1'b0;
                config_last=last_value;
                do @(posedge clk_core); while(!config_accept);
                if(beat==0)first_accept_time=$time;
                #1;
                if(u_dut.protocol_error_q)begin
                    if(!expect_fault)$fatal(1,
                        "unexpected config fault attack=%0d beat=%0d dutbeat=%0d padding=%0b requant=%0b descriptor=%0b",
                        attack,beat,u_dut.config_beat_q,u_dut.candidate_padding_legal,
                        u_dut.candidate_requant_legal,u_dut.candidate_descriptor_legal);
                    break;
                end
                if(beat!=5)@(negedge clk_core);
            end
            if(!u_dut.protocol_error_q&&expect_fault)
                $fatal(1,"config attack escaped %0d",attack);
            if(!expect_fault&&!config_loaded)$fatal(1,"legal config did not load");
            @(negedge clk_core);config_valid=1'b0;config_data='0;config_last=1'b0;
        end
    endtask

    task automatic send_tiles(
        input integer count,input integer seed_base,input logic[TAG_W-1:0]tag_base);
        integer tile,beat,time_index,lane,value;
        logic[1279:0]tile_payload;
        logic[TAG_W-1:0]tag_value;
        begin
            raw_valid=1'b1;
            for(tile=0;tile<count;tile=tile+1)begin
                tile_payload='0;tag_value=tag_base+tile;
                for(time_index=0;time_index<10;time_index=time_index+1)
                    for(lane=0;lane<16;lane=lane+1)begin
                        value=raw_value(seed_base+tile,time_index,lane);
                        tile_payload[((time_index*16+lane)*8)+:8]=value[7:0];
                    end
                enqueue_expected_tile(seed_base+tile,tag_value);
                for(beat=0;beat<5;beat=beat+1)begin
                    raw_data=tile_payload[beat*256+:256];
                    raw_last=(beat==4);raw_tag=tag_value;
                    do @(posedge clk_core); while(!raw_accept);
                    #1;
                    if(u_dut.protocol_error_q)
                        $fatal(1,"legal raw tile faulted tile=%0d beat=%0d",tile,beat);
                    if(!(tile==count-1&&beat==4))@(negedge clk_core);
                end
            end
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic send_raw_attack(input integer attack);
        integer beat,time_index,lane,value;
        logic[1279:0]tile_payload;
        logic[TAG_W-1:0]tag_value;
        begin
            tile_payload='0;tag_value=48'h2730_bad0_0000+attack;
            for(time_index=0;time_index<10;time_index=time_index+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    value=raw_value(90+attack,time_index,lane);
                    tile_payload[((time_index*16+lane)*8)+:8]=value[7:0];
                end
            raw_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                raw_data=tile_payload[beat*256+:256];
                raw_last=(beat==4);
                if(attack==1&&beat==2)raw_last=1'b1;
                if(attack==3&&beat==4)raw_last=1'b0;
                raw_tag=(attack==2&&beat==2)?tag_value+1'b1:tag_value;
                do @(posedge clk_core); while(!raw_accept);
                #1;
                if(u_dut.protocol_error_q)break;
                if(beat!=4)@(negedge clk_core);
            end
            if(!u_dut.protocol_error_q)$fatal(1,"raw attack escaped %0d",attack);
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic finish_context(
        input integer tiles,input time first_accept_time,input bit exact_cycle,
        output integer measured_cycles);
        time release_time;
        begin
            release_valid=1'b1;
            do @(posedge clk_core); while(!release_accept);
            release_time=$time;measured_cycles=((release_time-first_accept_time)/10)+1;
            #1;
            if(!context_retire_valid)$fatal(1,"missing registered context retire");
            if(context_retire_cycles!=measured_cycles)
                $fatal(1,"retire cycle mismatch got=%0d measured=%0d",
                    context_retire_cycles,measured_cycles);
            if(exact_cycle&&measured_cycles!=(5*tiles+19))
                $fatal(1,"clean cycle equality failed N=%0d got=%0d",tiles,measured_cycles);
            if(expected_read!=expected_write)
                $fatal(1,"context released with expected results pending %0d/%0d",
                    expected_read,expected_write);
            if(debug_config_beats!=6||debug_raw_beats!=5*tiles
                    ||debug_tiles_loaded!=tiles||debug_stage1_issues!=5*tiles
                    ||debug_stage1_done!=tiles||debug_stage2_issues!=5*tiles
                    ||debug_stage2_done!=tiles||debug_product_pushes!=5*tiles
                    ||debug_result_departures!=5*tiles)
                $fatal(1,"context conservation mismatch N=%0d",tiles);
            if(exact_cycle&&debug_product_replacements!=5*tiles-1)
                $fatal(1,"clean replacement mismatch N=%0d got=%0d",
                    tiles,debug_product_replacements);
            if(observed_tile_done!=tiles)
                $fatal(1,"tile_done count mismatch N=%0d got=%0d",tiles,observed_tile_done);
            @(negedge clk_core);release_valid=1'b0;
        end
    endtask

    task automatic run_clean_context(
        input integer tiles,input integer seed,input logic[TAG_W-1:0]tag_base,
        output integer cycles);
        time first_accept_time;
        begin
            clear_observation();ready_mode=0;
            send_config(0,legal_config,1'b0,first_accept_time);
            send_tiles(tiles,seed,tag_base);
            finish_context(tiles,first_accept_time,1'b1,cycles);
            if(numerical_mismatches!=0)$fatal(1,"clean numerical mismatch");
        end
    endtask

    task automatic run_pressure_context(output integer cycles);
        time first_accept_time;
        begin
            clear_observation();ready_mode=1;
            send_config(0,legal_config,1'b0,first_accept_time);
            send_tiles(40,40,48'h2730_3000_0000);
            finish_context(40,first_accept_time,1'b0,cycles);
            if(cycles<=219)$fatal(1,"pressure context did not extend clean bound");
            if(observed_fifo_peak!=16||observed_result_stalls==0
                    ||observed_raw_stalls==0||observed_full_pop_push==0)
                $fatal(1,"pressure coverage missing peak=%0d result_stall=%0d raw_stall=%0d full_pop_push=%0d",
                    observed_fifo_peak,observed_result_stalls,
                    observed_raw_stalls,observed_full_pop_push);
            ready_mode=0;
        end
    endtask

    task automatic check_sticky_fault(input bit expected_config_loaded);
        begin
            if(!protocol_error||config_loaded!=expected_config_loaded
                    ||result_valid||release_accept)
                $fatal(1,"fault not quarantined");
            repeat(2)begin
                @(posedge clk_core);#1;
                if(!protocol_error||config_accept||raw_accept||result_valid
                        ||stage1_issue||stage2_issue||product_push||fifo_pop)
                    $fatal(1,"fault not sticky/fail closed");
            end
        end
    endtask

    always @(negedge clk_core)begin
        if(rst_core)begin result_ready<=1'b1;ready_phase<=0;end
        else if(ready_mode==0)result_ready<=1'b1;
        else begin
            result_ready<=((ready_phase%8)==0);
            ready_phase<=ready_phase+1;
        end
    end

    always @(posedge clk_core)begin
        if(!rst_core)begin
            if(result_fifo_occupancy>observed_fifo_peak)
                observed_fifo_peak=result_fifo_occupancy;
            if(raw_valid&&!raw_ready)observed_raw_stalls=observed_raw_stalls+1;
            if(result_valid&&!result_ready)observed_result_stalls=observed_result_stalls+1;
            if(stage1_issue&&stage2_issue)observed_overlap=observed_overlap+1;
            if(product_replace)observed_replacements=observed_replacements+1;
            if(result_fifo_occupancy==16&&fifo_pop&&fifo_push)
                observed_full_pop_push=observed_full_pop_push+1;
            if(tile_done_valid)observed_tile_done=observed_tile_done+1;
            if(result_accept)begin
                if(expected_read>=expected_write)begin
                    numerical_mismatches=numerical_mismatches+1;
                    $fatal(1,"unexpected result tag=%h beat=%0d",result_tag,result_beat);
                end
                if(result_tag!==expected_tag[expected_read]
                        ||result_beat!==expected_beat[expected_read]
                        ||result_valid_bits!==48'h0000ffffffff
                        ||result_data!==expected_data[expected_read])begin
                    numerical_mismatches=numerical_mismatches+1;
                    $fatal(1,"numeric/order mismatch index=%0d tag=%h/%h beat=%0d/%0d data=%h/%h",
                        expected_read,result_tag,expected_tag[expected_read],
                        result_beat,expected_beat[expected_read],
                        result_data,expected_data[expected_read]);
                end
                expected_read=expected_read+1;
            end
        end
    end

    initial begin:directed_campaign
        integer clean1_cycles,clean4_cycles,pressure_cycles,attack;
        time ignored_time;
        logic[1535:0]attacked_config;
        clk_core=1'b0;rst_core=1'b1;config_valid=1'b0;config_data='0;
        config_last=1'b0;raw_valid=1'b0;raw_data='0;raw_last=1'b0;
        raw_tag='0;result_ready=1'b1;release_valid=1'b0;ready_mode=0;
        expected_read=0;expected_write=0;numerical_mismatches=0;
        ready_phase=0;make_legal_config();

        reset_dut();
        run_clean_context(1,1,48'h2730_1000_0000,clean1_cycles);
        run_clean_context(4,10,48'h2730_2000_0000,clean4_cycles);
        run_pressure_context(pressure_cycles);

        for(attack=1;attack<=4;attack=attack+1)begin
            reset_dut();attacked_config=legal_config;
            if(attack==3)attacked_config[1500]=1'b1;
            if(attack==4)attacked_config[606]=1'b1;
            send_config(attack<=2?attack:0,attacked_config,1'b1,ignored_time);
            check_sticky_fault(1'b0);
        end
        for(attack=1;attack<=3;attack=attack+1)begin
            reset_dut();
            send_config(0,legal_config,1'b0,ignored_time);
            send_raw_attack(attack);check_sticky_fault(1'b1);
        end

        $display("PASS M273 integrated rank3 ATLIF directed clean_contexts=2 pressure_contexts=1 attacks=7 numerical_mismatches=0 clean_cycles_N1=%0d clean_cycles_N4=%0d pressure_cycles=%0d fifo_peak=16 overlap=1 product_replace=1 full_pop_push=1",
            clean1_cycles,clean4_cycles,pressure_cycles);
        $finish;
    end

    initial begin
        #2000000;$fatal(1,"M273 directed timeout");
    end
endmodule
`default_nettype wire
