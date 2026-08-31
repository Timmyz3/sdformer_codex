`timescale 1ns/1ps
`default_nettype none

module tb_m518_matched_fixed_t10_atlif;
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
`ifdef M518_VCS_V06_HARNESS
    logic v06_hold_dense_issue,v06_first_empty_fill_bank1;
`endif

    logic[1279:0]legal_config;
    logic[TAG_W-1:0]expected_tag[0:16383];
    logic[2:0]expected_beat[0:16383];
    logic[47:0]expected_data[0:16383];
    integer expected_read,expected_write,numeric_mismatches;
    integer ready_mode,ready_phase,fifo_peak,result_stalls,raw_stalls;
    integer full_pop_push_cycles,tile_done_count;
    integer legal_halfcycle_checks,legal_protocol_pulses,legal_halfcycle_changes;
    integer active_profile;
    integer tuple_count,tile_issue_count,slot_ledger_tiles;
    bit tuple_seen[0:1599];
    logic legal_halfcycle_monitor_enable;

    integer total_config_attacks,total_padding_attacks,total_raw_attacks;
    integer total_zero_tile_attacks,total_fault_edge_attacks,total_reset_attacks;
    integer total_zero_tile_held_release_edges;
    integer total_zero_tile_fault_transitions;
    integer reset_partial_config_attacks,reset_partial_raw_attacks;
    integer reset_dense_c0_attacks,reset_dense_c11_attacks;
    integer reset_dense_c12_attacks,reset_dense_c15_attacks;
    integer reset_dense_c16_attacks,reset_fifo_full_close_attacks;
    integer reset_quarantine_attacks,total_clean_after_reset_probes;
    integer total_release_state_attacks,total_random_contexts;
    integer total_rail_boundary_contexts,total_rail_boundary_points;
    integer total_sequential_oldest_attacks,total_v06_harness_activations;
    integer phase12_stall_cycles,phase16_stall_cycles;
    integer result_beat_accepts[0:4];

    m518_matched_fixed_t10_atlif u_dut(.*);
    m518_matched_fixed_t10_atlif_assertions u_sva(
        .fault_event(u_dut.fault_event),
        .fifo_credit_internal(u_dut.fifo_credit),
        .config_frame_error_internal(u_dut.config_frame_error),
        .raw_frame_error_internal(u_dut.raw_frame_error),
        .zero_tile_release_error_internal(u_dut.zero_tile_release_error),
        .dense_active_internal(u_dut.dense_active_q),
        .dense_cycle_internal(u_dut.dense_selected_cycle),
        .dense_selected_raw_bank_internal(u_dut.dense_selected_raw_bank),
        .dense_raw_bank_internal(u_dut.dense_raw_bank_q),
        .dense_tag_internal(u_dut.dense_tag),
        .dense_push_beat_internal(u_dut.dense_result_beat_comb),
        .overflow_safe_internal(u_dut.overflow_safe_comb),
        .raw_owned_internal(u_dut.raw_owned_q),
        .raw_ready_internal(u_dut.raw_ready_q),
        .raw_order0_internal(u_dut.raw_order0_q),
        .raw_order1_internal(u_dut.raw_order1_q),
        .fill_active_internal(u_dut.fill_active_q),
        .fill_bank_internal(u_dut.fill_bank_q),
        .fill_beat_internal(u_dut.fill_beat_q),
        .fill_tag_internal(u_dut.fill_tag_q),
        .fifo_read_pointer_internal(u_dut.fifo_read_pointer_q),
        .fifo_write_pointer_internal(u_dut.fifo_write_pointer_q),
        .acc_state_internal(u_dut.acc_state_observe),
        .context_counting_internal(u_dut.context_counting_q),
        .multiplier_active_mask(u_dut.multiplier_active_mask),
        .issue_tuple_valid(u_dut.issue_tuple_valid),
        .issue_tuple_row(u_dut.issue_tuple_row),
        .issue_tuple_lane(u_dut.issue_tuple_lane),
        .issue_tuple_time(u_dut.issue_tuple_time),
        .*
    );

    always #5 clk_core=~clk_core;

    function automatic integer weight_value(
        input integer profile,input integer row,input integer time_index);
        integer selector;
        begin
            selector=(row*7+time_index*3)%11;
            case(profile)
                0:weight_value=0;
                1:weight_value=selector-5;
                2:weight_value=((row+time_index)&1)?127:-128;
                default:weight_value=0;
            endcase
        end
    endfunction

    function automatic integer bias_value(input integer profile,input integer row);
        begin
            case(profile)
                0:bias_value=row-5;
                1:bias_value=(row*37)-150;
                2:bias_value=row[0]?8388590:-8388590;
                default:bias_value=7;
            endcase
        end
    endfunction

    function automatic integer threshold_value(input integer profile);
        begin
            case(profile)
                0:threshold_value=0;
                1:threshold_value=13;
                2:threshold_value=0;
                default:threshold_value=7;
            endcase
        end
    endfunction

    function automatic integer raw_value(
        input integer profile,input integer tile_seed,
        input integer time_index,input integer lane);
        integer selector;
        begin
            selector=(tile_seed*5+time_index*3+lane)%17;
            if(profile==2)raw_value=((tile_seed+time_index+lane)&1)?127:-128;
            else begin
                case(selector)
                    0:raw_value=-128;
                    1:raw_value=127;
                    2:raw_value=-65;
                    3:raw_value=64;
                    4:raw_value=-31;
                    5:raw_value=29;
                    default:raw_value=selector-10;
                endcase
            end
        end
    endfunction

    function automatic longint signed sat_q24(input longint signed value);
        begin
            if(value>8388607)sat_q24=8388607;
            else if(value< -8388608)sat_q24=-8388608;
            else sat_q24=value;
        end
    endfunction

    function automatic integer decode_s8(input logic[7:0]encoded);
        begin decode_s8=$signed(encoded);end
    endfunction

    function automatic integer decode_s24(input logic[23:0]encoded);
        begin decode_s24=$signed(encoded);end
    endfunction

    task automatic build_payload(
        input integer profile,input integer tile_seed,
        output logic[1279:0]payload);
        integer time_index,lane,value;
        begin
            payload='0;
            for(time_index=0;time_index<10;time_index=time_index+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    value=raw_value(profile,tile_seed,time_index,lane);
                    payload[((time_index*16+lane)*8)+:8]=value[7:0];
                end
        end
    endtask

    task automatic enqueue_expected_frame(
        input logic[1279:0]frame,input logic[1279:0]payload,
        input logic[TAG_W-1:0]tag_value);
        integer beat,row_in_beat,row,lane,time_index;
        integer weight_decoded,input_decoded,bias_decoded,threshold_decoded;
        longint signed total;
        logic[47:0]packed_result;
        begin
            threshold_decoded=decode_s24(frame[1040+:24]);
            for(beat=0;beat<5;beat=beat+1)begin
                packed_result='0;
                for(row_in_beat=0;row_in_beat<2;row_in_beat=row_in_beat+1)begin
                    row=(beat*2)+row_in_beat;
                    bias_decoded=decode_s24(frame[800+(row*24)+:24]);
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=bias_decoded;
                        for(time_index=0;time_index<10;time_index=time_index+1)begin
                            weight_decoded=decode_s8(frame[
                                ((row*10+time_index)*8)+:8]);
                            input_decoded=decode_s8(payload[
                                ((time_index*16+lane)*8)+:8]);
                            total=total+(weight_decoded*input_decoded);
                        end
                        total=sat_q24(total);
                        packed_result[(row_in_beat*16)+lane]=
                            total>=threshold_decoded;
                    end
                end
                expected_tag[expected_write]=tag_value;
                expected_beat[expected_write]=beat[2:0];
                expected_data[expected_write]=packed_result;
                expected_write=expected_write+1;
            end
        end
    endtask

    task automatic build_random_case(
        input integer unsigned seed_value,
        output logic[1279:0]frame,output logic[1279:0]payload);
        integer unsigned state;
        integer coefficient,row,value_index;
        begin
            state=seed_value;frame='0;payload='0;
            for(coefficient=0;coefficient<100;coefficient=coefficient+1)begin
                state=(state*32'd1664525)+32'd1013904223;
                frame[(coefficient*8)+:8]=state[31:24];
            end
            for(row=0;row<10;row=row+1)begin
                state=(state*32'd1664525)+32'd1013904223;
                frame[800+(row*24)+:24]=state[23:0];
            end
            state=(state*32'd1664525)+32'd1013904223;
            frame[1040+:24]=state[23:0];
            for(value_index=0;value_index<160;value_index=value_index+1)begin
                state=(state*32'd1664525)+32'd1013904223;
                payload[(value_index*8)+:8]=state[31:24];
            end
        end
    endtask

    task automatic build_rail_case(
        input integer threshold_value_input,
        output logic[1279:0]frame,output logic[1279:0]payload);
        integer lane;
        begin
            frame='0;payload='0;
            // Rows0..5 produce max-1, max, max+1, min-1, min, min+1.
            frame[800+(0*24)+:24]=24'h7ffffe;
            frame[800+(1*24)+:24]=24'h7fffff;
            frame[800+(2*24)+:24]=24'h7fffff;
            frame[((2*10+0)*8)+:8]=8'h01;
            frame[800+(3*24)+:24]=24'h800000;
            frame[((3*10+0)*8)+:8]=8'hff;
            frame[800+(4*24)+:24]=24'h800000;
            frame[800+(5*24)+:24]=24'h800001;
            frame[1040+:24]=threshold_value_input[23:0];
            for(lane=0;lane<16;lane=lane+1)
                payload[((0*16+lane)*8)+:8]=8'h01;
        end
    endtask

    task automatic build_config(input integer profile);
        integer row,time_index,value;
        begin
            legal_config='0;
            for(row=0;row<10;row=row+1)begin
                for(time_index=0;time_index<10;time_index=time_index+1)begin
                    value=weight_value(profile,row,time_index);
                    legal_config[((row*10+time_index)*8)+:8]=value[7:0];
                end
                value=bias_value(profile,row);
                legal_config[800+(row*24)+:24]=value[23:0];
            end
            value=threshold_value(profile);
            legal_config[1040+:24]=value[23:0];
        end
    endtask

    task automatic enqueue_expected_tile(
        input integer profile,input integer tile_seed,
        input logic[TAG_W-1:0]tag_value);
        integer beat,row_in_beat,row,lane,time_index;
        longint signed total;
        logic[47:0]packed_result;
        begin
            for(beat=0;beat<5;beat=beat+1)begin
                packed_result='0;
                for(row_in_beat=0;row_in_beat<2;row_in_beat=row_in_beat+1)begin
                    row=(beat*2)+row_in_beat;
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=bias_value(profile,row);
                        for(time_index=0;time_index<10;time_index=time_index+1)
                            total=total+raw_value(profile,tile_seed,time_index,lane)
                                *weight_value(profile,row,time_index);
                        total=sat_q24(total);
                        packed_result[(row_in_beat*16)+lane]=
                            total>=threshold_value(profile);
                    end
                end
                expected_tag[expected_write]=tag_value;
                expected_beat[expected_write]=beat[2:0];
                expected_data[expected_write]=packed_result;
                expected_write=expected_write+1;
            end
        end
    endtask

    task automatic clear_context_observation;
        begin
            expected_read=0;expected_write=0;numeric_mismatches=0;
            fifo_peak=0;result_stalls=0;raw_stalls=0;
            full_pop_push_cycles=0;tile_done_count=0;
            tuple_count=0;tile_issue_count=0;slot_ledger_tiles=0;
            ready_phase=0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core=1'b1;config_valid=1'b0;config_data='0;config_last=1'b0;
            raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
            release_valid=1'b0;ready_mode=0;legal_halfcycle_monitor_enable=1'b0;
`ifdef M518_VCS_V06_HARNESS
            v06_hold_dense_issue=1'b0;v06_first_empty_fill_bank1=1'b0;
`endif
            repeat(4)@(posedge clk_core);
            @(negedge clk_core);rst_core=1'b0;
            clear_context_observation();
            #0.2;
            if(config_loaded||protocol_error||busy||result_valid
                    ||result_fifo_occupancy!=0||raw_bank_occupancy!=0
                    ||intermediate_bank_occupancy!=0||config_ready!==1'b1
                    ||debug_config_beats!=0||debug_raw_beats!=0
                    ||debug_tiles_loaded!=0||debug_stage1_issues!=0
                    ||debug_stage1_done!=0||debug_product_pushes!=0
                    ||debug_result_departures!=0)
                $fatal(1,"V18 reset did not produce deterministic empty state");
        end
    endtask

    task automatic send_config(
        input logic[1279:0]frame,input integer early_last_beat,
        input bit missing_last,input bit expect_fault,output time first_accept_time);
        integer beat;
        logic last_value;
        begin
            first_accept_time=0;config_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                config_data=frame[beat*256+:256];
                last_value=(beat==4)&&!missing_last;
                if(beat==early_last_beat)last_value=1'b1;
                config_last=last_value;
                do @(posedge clk_core); while(!config_accept);
                if(beat==0)first_accept_time=$time;
                #1;
                if(u_dut.protocol_error_q)break;
                if(beat!=4)@(negedge clk_core);
            end
            if(expect_fault&&!u_dut.protocol_error_q)
                $fatal(1,"V10/V11 config attack escaped early=%0d missing=%0b",
                    early_last_beat,missing_last);
            if(!expect_fault&&(!config_loaded||debug_config_beats!=5
                    ||protocol_error))
                $fatal(1,"legal five-beat config did not load");
            @(negedge clk_core);config_valid=1'b0;config_data='0;config_last=1'b0;
        end
    endtask

    task automatic send_tiles(
        input integer profile,input integer count,input integer seed_base,
        input logic[TAG_W-1:0]tag_base);
        integer tile,beat,time_index,lane,value;
        logic[1279:0]payload;
        logic[TAG_W-1:0]tag_value;
        begin
            raw_valid=1'b1;
            for(tile=0;tile<count;tile=tile+1)begin
                payload='0;tag_value=tag_base+tile;
                for(time_index=0;time_index<10;time_index=time_index+1)
                    for(lane=0;lane<16;lane=lane+1)begin
                        value=raw_value(profile,seed_base+tile,time_index,lane);
                        payload[((time_index*16+lane)*8)+:8]=value[7:0];
                    end
                enqueue_expected_tile(profile,seed_base+tile,tag_value);
                for(beat=0;beat<5;beat=beat+1)begin
                    raw_data=payload[beat*256+:256];
                    raw_last=(beat==4);raw_tag=tag_value;
                    do @(posedge clk_core); while(!raw_accept);
                    #1;
                    if(protocol_error)$fatal(1,"legal raw frame faulted");
                    if(!(tile==count-1&&beat==4))@(negedge clk_core);
                end
            end
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic send_frame_tile(
        input logic[1279:0]frame,input logic[1279:0]payload,
        input logic[TAG_W-1:0]tag_value);
        integer beat;
        begin
            enqueue_expected_frame(frame,payload,tag_value);
            raw_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                raw_data=payload[beat*256+:256];
                raw_last=(beat==4);raw_tag=tag_value;
                do @(posedge clk_core); while(!raw_accept);
                #1;
                if(protocol_error)$fatal(1,"legal explicit raw frame faulted");
                if(beat!=4)@(negedge clk_core);
            end
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;
            raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic finish_context(
        input integer tiles,input time first_accept_time,input bit exact_cycle,
        output integer measured_cycles);
        time release_time;
        begin
            release_valid=1'b1;
            do @(posedge clk_core); while(!release_accept);
            release_time=$time;
            measured_cycles=((release_time-first_accept_time)/10)+1;
            #1;
            if(!context_retire_valid||context_retire_cycles!=measured_cycles)
                $fatal(1,"retire cycle mismatch measured=%0d rtl=%0d",
                    measured_cycles,context_retire_cycles);
            if(exact_cycle&&measured_cycles!=(17*tiles+12))
                $fatal(1,"V01 cycle mismatch N=%0d got=%0d expected=%0d",
                    tiles,measured_cycles,17*tiles+12);
            if(expected_read!=expected_write||numeric_mismatches!=0)
                $fatal(1,"V02 scoreboard incomplete %0d/%0d mismatch=%0d",
                    expected_read,expected_write,numeric_mismatches);
            if(debug_config_beats!=5||debug_raw_beats!=5*tiles
                    ||debug_tiles_loaded!=tiles
                    ||debug_stage1_issues!=17*tiles
                    ||debug_stage1_done!=tiles
                    ||debug_stage2_issues!=0||debug_stage2_done!=0
                    ||debug_product_pushes!=5*tiles
                    ||debug_result_departures!=5*tiles
                    ||debug_product_replacements!=0)
                $fatal(1,"context conservation mismatch N=%0d",tiles);
            if(tile_done_count!=tiles||slot_ledger_tiles!=tiles)
                $fatal(1,"tile/slot ledger mismatch done=%0d ledger=%0d N=%0d",
                    tile_done_count,slot_ledger_tiles,tiles);
            @(negedge clk_core);release_valid=1'b0;
        end
    endtask

    task automatic run_context(
        input integer profile,input integer tiles,input integer seed,
        input logic[TAG_W-1:0]tag_base,input integer selected_ready_mode,
        input bit exact_cycle,output integer cycles);
        time first_accept_time;
        begin
            clear_context_observation();active_profile=profile;
            ready_mode=selected_ready_mode;legal_halfcycle_monitor_enable=1'b1;
            build_config(profile);
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            send_tiles(profile,tiles,seed,tag_base);
            finish_context(tiles,first_accept_time,exact_cycle,cycles);
            legal_halfcycle_monitor_enable=1'b0;ready_mode=0;
        end
    endtask

    task automatic run_frame_context(
        input logic[1279:0]frame,input logic[1279:0]payload,
        input logic[TAG_W-1:0]tag_value,output integer cycles);
        time first_accept_time;
        begin
            clear_context_observation();ready_mode=0;
            legal_halfcycle_monitor_enable=1'b1;
            send_config(frame,-1,1'b0,1'b0,first_accept_time);
            send_frame_tile(frame,payload,tag_value);
            finish_context(1,first_accept_time,1'b1,cycles);
            legal_halfcycle_monitor_enable=1'b0;
        end
    endtask

    task automatic check_quarantine(input bit expected_loaded);
        integer probe;
        begin
            config_valid=1'b1;raw_valid=1'b1;release_valid=1'b1;result_ready=1'b1;
            for(probe=0;probe<3;probe=probe+1)begin
                @(posedge clk_core);#1;
                if(!protocol_error||config_loaded!=expected_loaded||config_ready
                        ||raw_ready||release_ready||config_accept||raw_accept
                        ||result_valid||result_accept||release_accept||stage1_issue
                        ||product_push||fifo_push||fifo_pop)
                    $fatal(1,"registered sticky quarantine escaped");
            end
            @(negedge clk_core);config_valid=1'b0;raw_valid=1'b0;
            release_valid=1'b0;config_data='0;raw_data='0;
            config_last=1'b0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic send_raw_attack(input integer attack);
        integer beat,time_index,lane,value;
        logic[1279:0]payload;
        logic[TAG_W-1:0]tag_value;
        begin
            payload='0;tag_value=48'h5180_bad0_0000+attack;
            for(time_index=0;time_index<10;time_index=time_index+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    value=raw_value(1,90+attack,time_index,lane);
                    payload[((time_index*16+lane)*8)+:8]=value[7:0];
                end
            raw_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                raw_data=payload[beat*256+:256];raw_last=(beat==4);
                if(attack>=1&&attack<=4&&beat==attack-1)raw_last=1'b1;
                if(attack==5&&beat==4)raw_last=1'b0;
                raw_tag=((attack==6&&beat==1)||(attack==7&&beat==4))?
                    tag_value+1'b1:tag_value;
                do @(posedge clk_core); while(!raw_accept);
                #1;if(u_dut.protocol_error_q)break;
                if(beat!=4)@(negedge clk_core);
            end
            if(!u_dut.protocol_error_q)$fatal(1,"V12 raw attack escaped %0d",attack);
            total_raw_attacks=total_raw_attacks+1;
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic zero_tile_release_attack;
        time ignored;
        integer held_release_edges,quarantine_probe;
        logic prior_protocol_error;
        begin
            reset_dut();active_profile=0;build_config(0);
            send_config(legal_config,-1,1'b0,1'b0,ignored);
            @(negedge clk_core);release_valid=1'b1;#0.2;
            if(release_ready||release_accept||protocol_error)
                $fatal(1,"V15 zero-tile release was not edge registered");
            prior_protocol_error=protocol_error;
            for(held_release_edges=0;held_release_edges<8;
                    held_release_edges=held_release_edges+1)begin
                @(posedge clk_core);#1;
                if(!prior_protocol_error&&protocol_error)
                    total_zero_tile_fault_transitions=
                        total_zero_tile_fault_transitions+1;
                prior_protocol_error=protocol_error;
                if(!protocol_error||release_ready||release_accept
                        ||context_retire_valid||result_valid||result_accept
                        ||stage1_issue||product_push||fifo_push||fifo_pop)
                    $fatal(1,"V15 eight-edge held release escaped edge=%0d",
                        held_release_edges);
                total_zero_tile_held_release_edges=
                    total_zero_tile_held_release_edges+1;
            end
            if(total_zero_tile_fault_transitions!=1)
                $fatal(1,"V15 fault transition count drift count=%0d",
                    total_zero_tile_fault_transitions);
            @(negedge clk_core);release_valid=1'b0;
            config_valid=1'b1;raw_valid=1'b1;
            for(quarantine_probe=0;quarantine_probe<3;
                    quarantine_probe=quarantine_probe+1)begin
                @(posedge clk_core);#1;
                if(!protocol_error||!config_loaded||config_ready||raw_ready
                        ||release_ready||config_accept||raw_accept||result_valid
                        ||result_accept||release_accept||stage1_issue
                        ||product_push||fifo_push||fifo_pop)
                    $fatal(1,"V15 sticky quarantine escaped after eight-edge hold");
            end
            @(negedge clk_core);config_valid=1'b0;raw_valid=1'b0;
            config_data='0;raw_data='0;config_last=1'b0;raw_last=1'b0;raw_tag='0;
            total_zero_tile_attacks=total_zero_tile_attacks+1;
        end
    endtask

    task automatic raw_release_priority_attack;
        time ignored;
        begin
            reset_dut();active_profile=0;build_config(0);
            send_config(legal_config,-1,1'b0,1'b0,ignored);
            @(negedge clk_core);
            raw_valid=1'b1;raw_data=256'h518;raw_last=1'b0;
            raw_tag=48'h5180_1700_0000;release_valid=1'b1;
            @(posedge clk_core);#1;
            if(!raw_accept||release_accept||protocol_error)
                $fatal(1,"V17 raw did not win simultaneous zero-tile release");
            @(negedge clk_core);raw_valid=1'b0;release_valid=1'b0;
            raw_data='0;raw_tag='0;
        end
    endtask

    task automatic oldest_selection_sequential_attack;
        time first_accept_time;
        integer measured_cycles;
        logic[1279:0]payload_bank0,payload_bank1;
        logic[TAG_W-1:0]tag_bank0,tag_bank1;
        begin
            reset_dut();clear_context_observation();active_profile=1;
            build_config(1);ready_mode=0;
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            build_payload(1,600,payload_bank0);
            build_payload(1,601,payload_bank1);
            tag_bank0=48'h5180_0600_0000;
            tag_bank1=48'h5180_0600_0001;
`ifdef M518_VCS_V06_HARNESS
            // Hold issue, steer only the first empty fill to bank1, and use the
            // production five-beat raw path for both completed banks. No TB
            // process writes a DUT state variable.
            v06_hold_dense_issue=1'b1;
            v06_first_empty_fill_bank1=1'b1;
            send_frame_tile(legal_config,payload_bank1,tag_bank1);
            v06_first_empty_fill_bank1=1'b0;
            send_frame_tile(legal_config,payload_bank0,tag_bank0);
            if(u_dut.raw_ready_q!==2'b11||u_dut.raw_owned_q!==2'b11
                    ||u_dut.raw_order1_q>=u_dut.raw_order0_q
                    ||u_dut.raw_tag1_q!==tag_bank1
                    ||u_dut.raw_tag0_q!==tag_bank0
                    ||debug_raw_beats!=10||debug_tiles_loaded!=2
                    ||stage1_issue)
                $fatal(1,"V06 legal-fill harness failed to construct bank1-oldest dual-ready state");
            total_v06_harness_activations=total_v06_harness_activations+1;
            v06_hold_dense_issue=1'b0;
`else
            $fatal(1,"V06 requires M518_VCS_V06_HARNESS");
`endif
            #1;
            if(!stage1_issue||u_dut.dense_selected_raw_bank!==1'b1
                    ||u_dut.dense_tag!==tag_bank1)
                $fatal(1,"V06 oldest-ready bank1 pre-edge selection failed");
            @(posedge clk_core);#1;
            if(!u_dut.dense_active_q||u_dut.dense_cycle_q!=1
                    ||u_dut.dense_raw_bank_q!==1'b1
                    ||u_dut.raw_ready_q!==2'b01
                    ||u_dut.raw_owned_q!==2'b11
                    ||u_dut.dense_tag!==tag_bank1)
                $fatal(1,"V06 oldest-ready sequential ownership transition failed");
            total_sequential_oldest_attacks=total_sequential_oldest_attacks+1;
            finish_context(2,first_accept_time,1'b0,measured_cycles);
        end
    endtask

    task automatic release_partial_raw_attack;
        time first_accept_time;
        integer beat,measured_cycles;
        logic[1279:0]payload;
        logic[TAG_W-1:0]tag_value;
        begin
            reset_dut();clear_context_observation();active_profile=1;
            build_config(1);build_payload(1,700,payload);
            tag_value=48'h5180_1600_0000;ready_mode=0;
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            enqueue_expected_tile(1,700,tag_value);
            #0.2;release_valid=1'b1;raw_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                raw_data=payload[beat*256+:256];raw_last=(beat==4);raw_tag=tag_value;
                do @(posedge clk_core); while(!raw_accept);
                #1;
                if(protocol_error||release_accept||context_retire_valid)
                    $fatal(1,"V16 partial-raw release accepted or faulted beat=%0d",beat);
                if(beat!=4)@(negedge clk_core);
            end
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
            total_release_state_attacks=total_release_state_attacks+1;
            finish_context(1,first_accept_time,1'b1,measured_cycles);
        end
    endtask

    task automatic release_dense_phase_attack(input integer target_cycle,
        input integer seed_value,input logic[TAG_W-1:0]tag_value);
        time first_accept_time;
        integer measured_cycles,wait_cycles;
        logic[1279:0]payload;
        begin
            reset_dut();clear_context_observation();active_profile=1;
            build_config(1);build_payload(1,seed_value,payload);ready_mode=0;
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            send_frame_tile(legal_config,payload,tag_value);
            wait_cycles=0;
            while(!(u_dut.dense_source_valid
                    &&u_dut.dense_selected_cycle==target_cycle))begin
                @(negedge clk_core);wait_cycles=wait_cycles+1;
                if(wait_cycles>100)$fatal(1,"V16 target dense cycle not reached %0d",
                    target_cycle);
            end
            release_valid=1'b1;#0.2;
            if(release_ready||release_accept||context_retire_valid)
                $fatal(1,"V16 early release at dense cycle %0d",target_cycle);
            total_release_state_attacks=total_release_state_attacks+1;
            finish_context(1,first_accept_time,1'b1,measured_cycles);
        end
    endtask

    task automatic release_fifo_drain_attack;
        time first_accept_time;
        integer measured_cycles,hold_cycle;
        logic[1279:0]payload;
        begin
            reset_dut();clear_context_observation();active_profile=1;
            build_config(1);build_payload(1,704,payload);
            ready_mode=3;result_ready=1'b0;
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            send_frame_tile(legal_config,payload,48'h5180_1600_0004);
            while(u_dut.dense_active_q||result_fifo_occupancy!=5)@(negedge clk_core);
            release_valid=1'b1;
            for(hold_cycle=0;hold_cycle<3;hold_cycle=hold_cycle+1)begin
                @(posedge clk_core);#1;
                if(release_accept||release_ready||context_retire_valid)
                    $fatal(1,"V16 release accepted during FIFO-only drain");
            end
            @(negedge clk_core);result_ready=1'b1;
            total_release_state_attacks=total_release_state_attacks+1;
            finish_context(1,first_accept_time,1'b0,measured_cycles);
            ready_mode=0;
        end
    endtask

    task automatic targeted_phase12_phase16_stalls;
        time first_accept_time;
        integer measured_cycles,phase,wait_cycles;
        begin
            reset_dut();clear_context_observation();active_profile=1;
            build_config(1);ready_mode=3;result_ready=1'b0;
            send_config(legal_config,-1,1'b0,1'b0,first_accept_time);
            send_tiles(1,5,720,48'h5180_0800_0000);
            wait_cycles=0;
            while(!(result_fifo_occupancy==16&&u_dut.dense_active_q
                    &&u_dut.dense_selected_cycle==13&&!u_dut.fifo_credit))begin
                @(negedge clk_core);wait_cycles=wait_cycles+1;
                if(wait_cycles>500)$fatal(1,"V08 failed to reach initial full c13");
            end
            // Finish tile4 with simultaneous full pop/push, retaining count16.
            for(phase=13;phase<=16;phase=phase+1)begin
                if(u_dut.dense_selected_cycle!=phase)
                    $fatal(1,"V08 close phase drift got=%0d expected=%0d",
                        u_dut.dense_selected_cycle,phase);
                result_ready=1'b1;
                @(posedge clk_core);#1;
                @(negedge clk_core);
            end
            // Tile5 prologue is allowed to run against a full FIFO with no pops.
            result_ready=1'b0;
            while(!(u_dut.dense_active_q&&u_dut.dense_selected_cycle==12
                    &&result_fifo_occupancy==16&&!u_dut.fifo_credit))
                @(negedge clk_core);
            phase12_stall_cycles=phase12_stall_cycles+1;
            @(posedge clk_core);#1;
            if(u_dut.dense_cycle_q!=12||fifo_push)
                $fatal(1,"V08 phase12 close stall was not atomic");
            @(negedge clk_core);result_ready=1'b1;
            @(posedge clk_core);#1;
            // Keep full simultaneous pop/push for phases13..15.
            for(phase=13;phase<=15;phase=phase+1)begin
                @(negedge clk_core);
                if(u_dut.dense_selected_cycle!=phase)
                    $fatal(1,"V08 post-c12 phase drift %0d/%0d",
                        u_dut.dense_selected_cycle,phase);
                result_ready=1'b1;
                @(posedge clk_core);#1;
            end
            @(negedge clk_core);result_ready=1'b0;#0.2;
            if(!(u_dut.dense_active_q&&u_dut.dense_selected_cycle==16
                    &&result_fifo_occupancy==16&&!u_dut.fifo_credit))
                $fatal(1,"V08 phase16 targeted stall did not align");
            phase16_stall_cycles=phase16_stall_cycles+1;
            @(posedge clk_core);#1;
            if(u_dut.dense_cycle_q!=16||fifo_push)
                $fatal(1,"V08 phase16 close stall was not atomic");
            @(negedge clk_core);result_ready=1'b1;
            finish_context(5,first_accept_time,1'b0,measured_cycles);
            ready_mode=0;
        end
    endtask

    task automatic fault_edge_pop_push_attack;
        time ignored;
        integer cycles_waited,raw_before,push_before,pop_before;
        logic target_bank;
        logic[2:0]target_beat;
        logic[255:0]target_word_before;
        begin
            reset_dut();active_profile=1;build_config(1);
            send_config(legal_config,-1,1'b0,1'b0,ignored);
            ready_mode=2;result_ready=1'b0;
            send_tiles(1,3,200,48'h5180_1400_0000);
            cycles_waited=0;
            while(debug_product_pushes!=15)begin
                @(posedge clk_core);cycles_waited=cycles_waited+1;
                if(cycles_waited>500)$fatal(1,"V14 failed to build 15-beat FIFO");
            end
            send_tiles(1,1,203,48'h5180_1400_0003);

            @(negedge clk_core);
            raw_valid=1'b1;raw_data=256'h1111_518;raw_last=1'b0;
            raw_tag=48'h5180_1400_0004;
            do @(posedge clk_core); while(!raw_accept);
            @(negedge clk_core);raw_valid=1'b0;

            cycles_waited=0;
            while(!(result_fifo_occupancy==16&&u_dut.dense_active_q
                    &&u_dut.dense_selected_cycle==13&&u_dut.fill_active_q))begin
                @(negedge clk_core);cycles_waited=cycles_waited+1;
                if(cycles_waited>500)$fatal(1,"V14 failed to align close stall");
            end
            target_bank=u_dut.fill_bank_q;target_beat=u_dut.fill_beat_q;
            if(target_bank)
                target_word_before=u_dut.raw_bank1_q[target_beat*256+:256];
            else target_word_before=u_dut.raw_bank0_q[target_beat*256+:256];
            raw_before=debug_raw_beats;push_before=debug_product_pushes;
            pop_before=debug_result_departures;
            raw_valid=1'b1;raw_data=256'hbad0_0518;raw_last=1'b1;
            raw_tag=48'h5180_1400_0004;
            ready_mode=0;#0.1;result_ready=1'b1;#0.2;
            if(!raw_accept||!u_dut.raw_frame_error||!u_dut.fault_event
                    ||!result_accept||!fifo_pop||!fifo_push)
                $fatal(1,"V14 fault/pop/push edge did not align");
            @(posedge clk_core);#1;
            if(!protocol_error||debug_raw_beats!=raw_before
                    ||debug_product_pushes!=push_before+1
                    ||debug_result_departures!=pop_before+1
                    ||result_fifo_occupancy!=16)
                $fatal(1,"V14 fault-edge atomic commit mismatch");
            if(target_bank)begin
                if(u_dut.raw_bank1_q[target_beat*256+:256]!==target_word_before)
                    $fatal(1,"V14 bad bank1 payload committed");
            end else if(u_dut.raw_bank0_q[target_beat*256+:256]!==target_word_before)
                $fatal(1,"V14 bad bank0 payload committed");
            total_fault_edge_attacks=total_fault_edge_attacks+1;
            @(negedge clk_core);raw_valid=1'b0;raw_last=1'b0;raw_data='0;raw_tag='0;
            check_quarantine(1'b1);
        end
    endtask

    task automatic clean_after_reset_probe(
        input integer seed_value,input logic[TAG_W-1:0]tag_value);
        integer probe_cycles;
        begin
            run_context(1,1,seed_value,tag_value,0,1'b1,probe_cycles);
            if(probe_cycles!=29)$fatal(1,"V18 clean-after-reset N1 drift");
            total_clean_after_reset_probes=total_clean_after_reset_probes+1;
        end
    endtask

    task automatic dense_boundary_reset_attack(
        input integer target_cycle,input integer tile_seed,
        input integer clean_seed,input logic[TAG_W-1:0]tile_tag,
        input logic[TAG_W-1:0]clean_tag);
        time ignored;
        begin
            build_config(1);send_config(legal_config,-1,1'b0,1'b0,ignored);
            send_tiles(1,1,tile_seed,tile_tag);
            while(!(u_dut.dense_source_valid
                    &&u_dut.dense_selected_cycle==target_cycle))
                @(negedge clk_core);
            reset_dut();total_reset_attacks=total_reset_attacks+1;
            clean_after_reset_probe(clean_seed,clean_tag);
        end
    endtask

    task automatic reset_state_attacks;
        time ignored;
        integer wait_cycles;
        begin
            // V18a: partial configuration then a complete clean context.
            reset_dut();build_config(0);config_valid=1'b1;
            config_data=legal_config[0+:256];config_last=1'b0;
            do @(posedge clk_core); while(!config_accept);
            reset_dut();total_reset_attacks=total_reset_attacks+1;
            reset_partial_config_attacks=reset_partial_config_attacks+1;
            clean_after_reset_probe(810,48'h5180_1800_0010);

            // V18b: partial raw frame.
            build_config(0);send_config(legal_config,-1,1'b0,1'b0,ignored);
            @(negedge clk_core);raw_valid=1'b1;raw_data=256'h518;
            raw_last=1'b0;raw_tag=48'h5180_1800_0000;
            do @(posedge clk_core); while(!raw_accept);
            reset_dut();total_reset_attacks=total_reset_attacks+1;
            reset_partial_raw_attacks=reset_partial_raw_attacks+1;
            clean_after_reset_probe(811,48'h5180_1800_0011);

            // V18c-g: exact dense boundary matrix c0/c11/c12/c15/c16.
            dense_boundary_reset_attack(0,300,812,
                48'h5180_1800_0001,48'h5180_1800_0012);
            reset_dense_c0_attacks=reset_dense_c0_attacks+1;

            dense_boundary_reset_attack(11,301,813,
                48'h5180_1800_0002,48'h5180_1800_0013);
            reset_dense_c11_attacks=reset_dense_c11_attacks+1;

            dense_boundary_reset_attack(12,302,814,
                48'h5180_1800_0003,48'h5180_1800_0014);
            reset_dense_c12_attacks=reset_dense_c12_attacks+1;

            dense_boundary_reset_attack(15,303,815,
                48'h5180_1800_0004,48'h5180_1800_0015);
            reset_dense_c15_attacks=reset_dense_c15_attacks+1;

            dense_boundary_reset_attack(16,304,816,
                48'h5180_1800_0005,48'h5180_1800_0016);
            reset_dense_c16_attacks=reset_dense_c16_attacks+1;

            // V18h: FIFO-full close stall.
            build_config(1);ready_mode=3;result_ready=1'b0;
            send_config(legal_config,-1,1'b0,1'b0,ignored);
            send_tiles(1,4,305,48'h5180_1800_0020);
            wait_cycles=0;
            while(!(result_fifo_occupancy==16&&u_dut.dense_active_q
                    &&u_dut.dense_selected_cycle>=13&&!u_dut.fifo_credit))begin
                @(negedge clk_core);wait_cycles=wait_cycles+1;
                if(wait_cycles>500)$fatal(1,"V18 FIFO stall not reached");
            end
            reset_dut();total_reset_attacks=total_reset_attacks+1;
            reset_fifo_full_close_attacks=reset_fifo_full_close_attacks+1;
            clean_after_reset_probe(817,48'h5180_1800_0017);

            // V18i: sticky quarantine, then reset-only recovery and clean context.
            build_config(0);send_config(legal_config,-1,1'b0,1'b0,ignored);
            @(negedge clk_core);release_valid=1'b1;
            @(posedge clk_core);#1;
            if(!protocol_error)$fatal(1,"V18 quarantine setup failed");
            reset_dut();total_reset_attacks=total_reset_attacks+1;
            reset_quarantine_attacks=reset_quarantine_attacks+1;
            clean_after_reset_probe(818,48'h5180_1800_0018);
        end
    endtask

    always @(negedge clk_core)begin
        if(rst_core)begin result_ready<=1'b1;ready_phase<=0;end
        else if(ready_mode==0)result_ready<=1'b1;
        else if(ready_mode==2)result_ready<=1'b0;
        else if(ready_mode==3)begin end
        else begin
            result_ready<=((ready_phase%8)==0);
            ready_phase<=ready_phase+1;
        end
    end

    always @(posedge clk_core)begin:halfcycle_probe
        logic[4:0]control_sample;
        logic[147:0]result_sample;
        if(!rst_core&&legal_halfcycle_monitor_enable)begin
            #0.2;
            control_sample={protocol_error,result_valid,stage1_issue,
                product_push,fifo_push};
            result_sample={result_valid,result_tag,result_beat,
                result_valid_bits,result_data};
            if(protocol_error)legal_protocol_pulses=legal_protocol_pulses+1;
            #4.5;
            legal_halfcycle_checks=legal_halfcycle_checks+1;
            if({protocol_error,result_valid,stage1_issue,product_push,fifo_push}
                    !==control_sample
                    ||{result_valid,result_tag,result_beat,
                        result_valid_bits,result_data}!==result_sample)
                legal_halfcycle_changes=legal_halfcycle_changes+1;
        end
    end

    always @(posedge clk_core)begin:scoreboards
        if(rst_core)begin
            tuple_count=0;tile_issue_count=0;
            for(int tuple=0;tuple<1600;tuple++)tuple_seen[tuple]=1'b0;
        end else begin
            if(result_fifo_occupancy>fifo_peak)fifo_peak=result_fifo_occupancy;
            if(result_valid&&!result_ready)result_stalls=result_stalls+1;
            if(raw_valid&&!raw_ready)raw_stalls=raw_stalls+1;
            if(result_fifo_occupancy==16&&fifo_pop&&fifo_push)
                full_pop_push_cycles=full_pop_push_cycles+1;
            if(tile_done_valid)tile_done_count=tile_done_count+1;

            if(stage1_issue)begin
                if(u_dut.dense_selected_cycle==0)begin
                    tuple_count=0;tile_issue_count=0;
                    for(int tuple=0;tuple<1600;tuple++)tuple_seen[tuple]=1'b0;
                end
                tile_issue_count=tile_issue_count+1;
                for(int slot=0;slot<96;slot++)begin
                    if(u_dut.issue_tuple_valid[slot])begin
                        integer row,lane,time_index,tuple_index;
                        row=u_dut.issue_tuple_row[(slot*4)+:4];
                        lane=u_dut.issue_tuple_lane[(slot*4)+:4];
                        time_index=u_dut.issue_tuple_time[(slot*4)+:4];
                        tuple_index=((row*16)+lane)*10+time_index;
                        if(tuple_index<0||tuple_index>=1600||tuple_seen[tuple_index])
                            $fatal(1,"V04 duplicate/out-of-range tuple slot=%0d tuple=%0d",
                                slot,tuple_index);
                        tuple_seen[tuple_index]=1'b1;tuple_count=tuple_count+1;
                    end
                end
                if(u_dut.dense_selected_cycle==16)begin
                    if(tile_issue_count!=17||tuple_count!=1600)
                        $fatal(1,"V04 tile ledger issues=%0d tuples=%0d",
                            tile_issue_count,tuple_count);
                    slot_ledger_tiles=slot_ledger_tiles+1;
                    tile_issue_count=0;tuple_count=0;
                end
            end

            if(result_accept)begin
                if(result_beat<=4)
                    result_beat_accepts[result_beat]=
                        result_beat_accepts[result_beat]+1;
                if(expected_read>=expected_write)
                    $fatal(1,"unexpected result tag=%h beat=%0d",result_tag,result_beat);
                if(result_tag!==expected_tag[expected_read]
                        ||result_beat!==expected_beat[expected_read]
                        ||result_valid_bits!==48'h0000ffffffff
                        ||result_data!==expected_data[expected_read])begin
                    numeric_mismatches=numeric_mismatches+1;
                    $fatal(1,"V02 result mismatch index=%0d tag=%h/%h beat=%0d/%0d data=%h/%h",
                        expected_read,result_tag,expected_tag[expected_read],
                        result_beat,expected_beat[expected_read],
                        result_data,expected_data[expected_read]);
                end
                expected_read=expected_read+1;
            end
        end
    end

    initial begin:V01_to_V20_campaign
        integer cycles_n1,cycles_n4,cycles_extreme,cycles_equality;
        integer pressure_cycles,attack,padding_bit,random_index,explicit_cycles;
        time ignored;
        logic[1279:0]attacked_config,explicit_config,explicit_payload;
        clk_core=1'b0;rst_core=1'b1;config_valid=1'b0;config_data='0;
        config_last=1'b0;raw_valid=1'b0;raw_data='0;raw_last=1'b0;
        raw_tag='0;result_ready=1'b1;release_valid=1'b0;ready_mode=0;
`ifdef M518_VCS_V06_HARNESS
        v06_hold_dense_issue=1'b0;v06_first_empty_fill_bank1=1'b0;
`endif
        expected_read=0;expected_write=0;numeric_mismatches=0;
        legal_halfcycle_monitor_enable=1'b0;legal_halfcycle_checks=0;
        legal_protocol_pulses=0;legal_halfcycle_changes=0;
        total_config_attacks=0;total_padding_attacks=0;total_raw_attacks=0;
        total_zero_tile_attacks=0;total_fault_edge_attacks=0;total_reset_attacks=0;
        total_zero_tile_held_release_edges=0;
        total_zero_tile_fault_transitions=0;
        reset_partial_config_attacks=0;reset_partial_raw_attacks=0;
        reset_dense_c0_attacks=0;reset_dense_c11_attacks=0;
        reset_dense_c12_attacks=0;reset_dense_c15_attacks=0;
        reset_dense_c16_attacks=0;reset_fifo_full_close_attacks=0;
        reset_quarantine_attacks=0;total_clean_after_reset_probes=0;
        total_release_state_attacks=0;total_random_contexts=0;
        total_rail_boundary_contexts=0;total_rail_boundary_points=0;
        total_sequential_oldest_attacks=0;total_v06_harness_activations=0;
        phase12_stall_cycles=0;phase16_stall_cycles=0;
        for(int beat_counter=0;beat_counter<5;beat_counter=beat_counter+1)
            result_beat_accepts[beat_counter]=0;

        reset_dut();
        // V01/V02/V03/V04/V05/V09/V13/V19: four profiles without reset
        // between released contexts prove exact cycles, arithmetic rails,
        // slot uniqueness, bank flow, head stability and stale-state isolation.
        run_context(0,1,1,48'h5180_0100_0000,0,1'b1,cycles_n1);
        run_context(1,4,10,48'h5180_0100_1000,0,1'b1,cycles_n4);
        run_context(2,2,20,48'h5180_0300_0000,0,1'b1,cycles_extreme);
        run_context(3,1,30,48'h5180_0300_1000,0,1'b1,cycles_equality);

        // V02/V03: independent frame-decoding oracle, fixed-seed randomized
        // matrices, and all six Q24 just-below/at/just-above wide sums.
        build_rail_case(8388607,explicit_config,explicit_payload);
        run_frame_context(explicit_config,explicit_payload,
            48'h5180_0300_2000,explicit_cycles);
        total_rail_boundary_contexts=total_rail_boundary_contexts+1;
        build_rail_case(-8388608,explicit_config,explicit_payload);
        run_frame_context(explicit_config,explicit_payload,
            48'h5180_0300_2001,explicit_cycles);
        total_rail_boundary_contexts=total_rail_boundary_contexts+1;
        total_rail_boundary_points=6;
        for(random_index=0;random_index<4;random_index=random_index+1)begin
            build_random_case(32'h5182_0000+random_index,
                explicit_config,explicit_payload);
            run_frame_context(explicit_config,explicit_payload,
                48'h5180_0200_0000+random_index,explicit_cycles);
            total_random_contexts=total_random_contexts+1;
        end

        // V07/V08/V16: fixed one-in-eight pressure reaches close stalls and
        // exercises full FIFO same-cycle pop/push plus release wait.
        run_context(1,40,40,48'h5180_0700_0000,1,1'b0,pressure_cycles);
        if(fifo_peak!=16||result_stalls==0||raw_stalls==0
                ||full_pop_push_cycles==0)
            $fatal(1,"V07/V08 pressure coverage missing peak=%0d rs=%0d raw=%0d pp=%0d",
                fifo_peak,result_stalls,raw_stalls,full_pop_push_cycles);

        // V06/V07/V08: cross a real oldest-bank issue edge and then force
        // deterministic phase12/phase16 full-FIFO stalls.
        oldest_selection_sequential_attack();
        targeted_phase12_phase16_stalls();

        // V16: held release at partial raw, dense0/12/16 and FIFO-only drain.
        release_partial_raw_attack();
        release_dense_phase_attack(0,701,48'h5180_1600_0001);
        release_dense_phase_attack(12,702,48'h5180_1600_0002);
        release_dense_phase_attack(16,703,48'h5180_1600_0003);
        release_fifo_drain_attack();

        // V10: all four early-last positions plus missing final last.
        for(attack=0;attack<4;attack=attack+1)begin
            reset_dut();build_config(1);
            send_config(legal_config,attack,1'b0,1'b1,ignored);
            total_config_attacks=total_config_attacks+1;check_quarantine(1'b0);
        end
        reset_dut();build_config(1);
        send_config(legal_config,-1,1'b1,1'b1,ignored);
        total_config_attacks=total_config_attacks+1;check_quarantine(1'b0);

        // V11: every one of 216 required-zero padding bits is attacked.
        for(padding_bit=1064;padding_bit<1280;padding_bit=padding_bit+1)begin
            reset_dut();build_config(1);attacked_config=legal_config;
            attacked_config[padding_bit]=1'b1;
            send_config(attacked_config,-1,1'b0,1'b1,ignored);
            total_padding_attacks=total_padding_attacks+1;
            check_quarantine(1'b0);
        end

        // V12: early last beats0..3, missing beat4, tag drift beats1 and4.
        for(attack=1;attack<=7;attack=attack+1)begin
            reset_dut();build_config(1);
            send_config(legal_config,-1,1'b0,1'b0,ignored);
            send_raw_attack(attack);check_quarantine(1'b1);
        end

        // V14/V15/V17/V18/V20. V20 is launcher-level wrong-SHA preflight;
        // the remaining cases are dynamic fault-edge, N0, priority and reset.
        zero_tile_release_attack();
        raw_release_priority_attack();
        fault_edge_pop_push_attack();
        reset_state_attacks();

        if(cycles_n1!=29||cycles_n4!=80||cycles_extreme!=46
                ||cycles_equality!=29||pressure_cycles<=692
                ||total_config_attacks!=5||total_padding_attacks!=216
                ||total_raw_attacks!=7||total_zero_tile_attacks!=1
                ||total_fault_edge_attacks!=1||total_reset_attacks!=9
                ||total_zero_tile_held_release_edges!=8
                ||total_zero_tile_fault_transitions!=1
                ||reset_partial_config_attacks!=1
                ||reset_partial_raw_attacks!=1||reset_dense_c0_attacks!=1
                ||reset_dense_c11_attacks!=1||reset_dense_c12_attacks!=1
                ||reset_dense_c15_attacks!=1||reset_dense_c16_attacks!=1
                ||reset_fifo_full_close_attacks!=1
                ||reset_quarantine_attacks!=1
                ||total_clean_after_reset_probes!=9
                ||total_release_state_attacks!=5||total_random_contexts!=4
                ||total_rail_boundary_contexts!=2||total_rail_boundary_points!=6
                ||total_sequential_oldest_attacks!=1
                ||total_v06_harness_activations!=1
                ||phase12_stall_cycles==0||phase16_stall_cycles==0
                ||result_beat_accepts[0]==0||result_beat_accepts[1]==0
                ||result_beat_accepts[2]==0||result_beat_accepts[3]==0
                ||result_beat_accepts[4]==0
                ||legal_halfcycle_checks==0||legal_protocol_pulses!=0
                ||legal_halfcycle_changes!=0)
            $fatal(1,"M518 campaign closure mismatch N1=%0d N4=%0d ext=%0d eq=%0d pressure=%0d cfg=%0d pad=%0d raw=%0d n0=%0d edge=%0d reset=%0d release=%0d random=%0d railctx=%0d railpts=%0d oldest=%0d harness=%0d c12=%0d c16=%0d beats=%0d/%0d/%0d/%0d/%0d half=%0d pulse=%0d change=%0d",
                cycles_n1,cycles_n4,cycles_extreme,cycles_equality,
                pressure_cycles,total_config_attacks,total_padding_attacks,
                total_raw_attacks,total_zero_tile_attacks,total_fault_edge_attacks,
                total_reset_attacks,total_release_state_attacks,
                total_random_contexts,total_rail_boundary_contexts,
                total_rail_boundary_points,total_sequential_oldest_attacks,
                total_v06_harness_activations,
                phase12_stall_cycles,phase16_stall_cycles,
                result_beat_accepts[0],result_beat_accepts[1],
                result_beat_accepts[2],result_beat_accepts[3],
                result_beat_accepts[4],legal_halfcycle_checks,
                legal_protocol_pulses,legal_halfcycle_changes);

        $display("PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 clean_N4=80 random_contexts=4 rail_boundary_points=6 zero_tile_held_edges=8 zero_tile_fault_transitions=1 release_state_attacks=5 reset_attacks=9 reset_partial_config=1 reset_partial_raw=1 reset_dense_c0=1 reset_dense_c11=1 reset_dense_c12=1 reset_dense_c15=1 reset_dense_c16=1 reset_fifo_full_close=1 reset_quarantine=1 clean_after_reset_N1=9 sequential_oldest=1 v06_legal_fill_harness=1 phase12_stall=1 phase16_stall=1 padding_attacks=216 raw_attacks=7 config_attacks=5 fault_edge_pop_push=1 slot_tuples_per_tile=1600 multiplier_slots=96 issue_cycles=17 vcs_only=true dc=false formality=false ptpx=false speedup=false ppa=false headline=false");
        $finish;
    end

    initial begin
        #10000000;$fatal(1,"M518 V01-V20 directed timeout");
    end
endmodule
`default_nettype wire
