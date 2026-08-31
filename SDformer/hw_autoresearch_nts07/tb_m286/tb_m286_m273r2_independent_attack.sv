`timescale 1ns/1ps
`default_nettype none

module tb_m286_m273r2_independent_attack;
    localparam int TAG_W = 48;
    localparam int FIFO_DEPTH = 16;
    localparam int FIFO_COUNT_W = $clog2(FIFO_DEPTH+1);

    logic clk_core,rst_core;
    logic config_valid,config_ready,config_accept,config_last;
    logic [255:0] config_data;
    logic raw_valid,raw_ready,raw_accept,raw_last;
    logic [255:0] raw_data;
    logic [TAG_W-1:0] raw_tag;
    logic result_valid,result_ready,result_accept;
    logic [TAG_W-1:0] result_tag;
    logic [2:0] result_beat;
    logic [47:0] result_valid_bits,result_data;
    logic release_valid,release_ready,release_accept;
    logic tile_done_valid,context_retire_valid,config_loaded,protocol_error,busy;
    logic [TAG_W-1:0] tile_done_tag;
    logic [31:0] context_retire_cycles;
    logic stage1_issue,stage2_issue,product_push,product_replace,fifo_push,fifo_pop;
    logic [FIFO_COUNT_W-1:0] result_fifo_occupancy;
    logic [1:0] raw_bank_occupancy,intermediate_bank_occupancy;
    logic [31:0] debug_config_beats,debug_raw_beats,debug_tiles_loaded;
    logic [31:0] debug_stage1_issues,debug_stage1_done;
    logic [31:0] debug_stage2_issues,debug_stage2_done;
    logic [31:0] debug_product_pushes,debug_result_departures;
    logic [31:0] debug_product_replacements,debug_context_cycles;

    logic [1535:0] legal_config;
    logic [TAG_W-1:0] expected_tag [0:4095];
    logic [2:0] expected_beat [0:4095];
    logic [47:0] expected_data [0:4095];
    integer expected_read,expected_write,numeric_mismatches;
    integer ready_mode,ready_phase;
    integer fifo_peak,result_stalls,raw_stalls,release_waits;
    integer overlap_cycles,replace_cycles,full_pop_push_cycles;
    integer raw_dual_arb,inter_dual_arb,raw_order_errors,inter_order_errors;
    integer stage1_product_ops,stage2_product_ops,tile_done_count;
    integer no_fallthrough_checks,release_empty_checks;
    integer legal_protocol_error_glitches;
    integer legal_halfcycle_checks,legal_intra_half_changes;
    integer config_phase_accepts [0:5];
    integer raw_phase_accepts [0:4];
    integer result_phase_accepts [0:4];
    integer fault_edge_pop_push,fault_edge_result_order_checks;
    integer n0_held_cycles,quarantine_probe_cycles;
    logic legal_halfcycle_monitor_enable;
    integer active_profile,stage1_acc_reference_checks,stage2_event_reference_checks;
    integer rne_tie_even_cases,rne_tie_odd_cases,q8_saturation_cases;
    integer q24_saturation_cases,internal_reference_mismatches;

    m273_integrated_rank3_atlif u_dut(.*);

    m286_m273r2_independent_assertions u_m286_sva(
        .clk_core(clk_core),.rst_core(rst_core),
        .config_valid(config_valid),.config_ready(config_ready),
        .config_accept(config_accept),.raw_valid(raw_valid),
        .raw_ready(raw_ready),.raw_accept(raw_accept),
        .result_valid(result_valid),.result_ready(result_ready),
        .result_accept(result_accept),.result_tag(result_tag),
        .result_beat(result_beat),.result_valid_bits(result_valid_bits),
        .result_data(result_data),.release_valid(release_valid),
        .release_ready(release_ready),.release_accept(release_accept),
        .protocol_error(protocol_error),.fault_event(u_dut.fault_event),
        .product_push(product_push),.fifo_push(fifo_push),.fifo_pop(fifo_pop),
        .stage1_issue(stage1_issue),.stage2_issue(stage2_issue),
        .result_fifo_occupancy(result_fifo_occupancy),
        .debug_tiles_loaded(debug_tiles_loaded));

    always #5 clk_core = ~clk_core;

    function automatic integer right_coefficient(
        input integer profile,input integer rank,input integer time_index);
        integer selector;
        begin
            selector=(profile*7+rank*3+time_index)%9;
            case(selector)
                0:right_coefficient=7;
                1:right_coefficient=-5;
                2:right_coefficient=3;
                3:right_coefficient=-2;
                4:right_coefficient=1;
                5:right_coefficient=0;
                6:right_coefficient=11;
                7:right_coefficient=-9;
                default:right_coefficient=4;
            endcase
        end
    endfunction

    function automatic integer left_coefficient(
        input integer profile,input integer output_row,input integer rank);
        integer selector;
        begin
            selector=(profile*5+output_row*3+rank)%13;
            case(selector)
                0:left_coefficient=0;
                1:left_coefficient=1;
                2:left_coefficient=-1;
                3:left_coefficient=2;
                4:left_coefficient=-2;
                5:left_coefficient=4;
                6:left_coefficient=-4;
                7:left_coefficient=5;
                8:left_coefficient=-5;
                9:left_coefficient=9;
                10:left_coefficient=-9;
                11:left_coefficient=17;
                default:left_coefficient=-17;
            endcase
        end
    endfunction

    function automatic integer bias_value(input integer profile,input integer row);
        begin
            if(profile==1)bias_value=(row[0]?8388590:-8388590);
            else bias_value=(row*7)-25;
        end
    endfunction

    function automatic integer threshold_value(input integer profile);
        begin threshold_value=(profile==1)?8388600:3;end
    endfunction

    function automatic integer raw_value(
        input integer profile,input integer tile_seed,
        input integer time_index,input integer lane);
        integer selector;
        begin
            selector=(profile*11+tile_seed*5+time_index*3+lane)%17;
            case(selector)
                0:raw_value=-128;
                1:raw_value=127;
                2:raw_value=-65;
                3:raw_value=64;
                4:raw_value=-31;
                5:raw_value=29;
                6:raw_value=-13;
                7:raw_value=11;
                default:raw_value=selector-12;
            endcase
        end
    endfunction

    function automatic integer rne_sat_q8(input integer value,input integer shift);
        integer magnitude,quotient,remainder,half,rounded;
        begin
            magnitude=(value<0)?-value:value;
            if(shift==0)rounded=magnitude;
            else begin
                quotient=magnitude>>>shift;
                remainder=magnitude&((1<<shift)-1);
                half=1<<(shift-1);
                rounded=quotient+((remainder>half)||
                    ((remainder==half)&&((quotient&1)!=0)));
            end
            if(value<0)rounded=-rounded;
            if(rounded>127)rounded=127;
            if(rounded< -128)rounded=-128;
            rne_sat_q8=rounded;
        end
    endfunction

    function automatic integer sat_q24(input integer value);
        begin
            if(value>8388607)sat_q24=8388607;
            else if(value< -8388608)sat_q24=-8388608;
            else sat_q24=value;
        end
    endfunction

    task automatic encode_csd(
        inout logic [1535:0] frame,input integer coefficient,input integer value);
        integer magnitude,term;
        begin
            frame[245+(coefficient*8)+:8]=value[7:0];
            magnitude=(value<0)?-value:value;
            term=0;
            if((magnitude&1)!=0)begin
                frame[485+(coefficient*4)+term]=1'b1;
                frame[605+(coefficient*4)+term]=(value<0);
                frame[725+(coefficient*12)+(term*3)+:3]=3'd0;
                term=term+1;
            end
            if((magnitude&2)!=0)begin
                frame[485+(coefficient*4)+term]=1'b1;
                frame[605+(coefficient*4)+term]=(value<0);
                frame[725+(coefficient*12)+(term*3)+:3]=3'd1;
                term=term+1;
            end
            if((magnitude&4)!=0)begin
                frame[485+(coefficient*4)+term]=1'b1;
                frame[605+(coefficient*4)+term]=(value<0);
                frame[725+(coefficient*12)+(term*3)+:3]=3'd2;
                term=term+1;
            end
            if((magnitude&8)!=0)begin
                frame[485+(coefficient*4)+term]=1'b1;
                frame[605+(coefficient*4)+term]=(value<0);
                frame[725+(coefficient*12)+(term*3)+:3]=3'd3;
                term=term+1;
            end
            if((magnitude&16)!=0)begin
                frame[485+(coefficient*4)+term]=1'b1;
                frame[605+(coefficient*4)+term]=(value<0);
                frame[725+(coefficient*12)+(term*3)+:3]=3'd4;
            end
        end
    endtask

    task automatic build_config(input integer profile);
        integer rank,time_index,row,coefficient,value;
        begin
            legal_config='0;
            for(rank=0;rank<3;rank=rank+1)
                for(time_index=0;time_index<10;time_index=time_index+1)begin
                    value=right_coefficient(profile,rank,time_index);
                    legal_config[((rank*10+time_index)*8)+:8]=value[7:0];
                end
            legal_config[240+:5]=(profile==1)?5'd0:5'd3;
            for(row=0;row<10;row=row+1)begin
                for(rank=0;rank<3;rank=rank+1)begin
                    coefficient=row*3+rank;
                    encode_csd(legal_config,coefficient,
                        left_coefficient(profile,row,rank));
                end
                value=bias_value(profile,row);
                legal_config[1085+(row*24)+:24]=value[23:0];
            end
            value=threshold_value(profile);
            legal_config[1325+:24]=value[23:0];
        end
    endtask

    task automatic enqueue_expected_tile(
        input integer profile,input integer tile_seed,
        input logic [TAG_W-1:0] tag_value);
        integer intermediate [0:47];
        integer rank,lane,time_index,row,beat,row_in_beat,total,shift;
        logic [47:0] packed_result;
        begin
            shift=(profile==1)?0:3;
            for(rank=0;rank<3;rank=rank+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    total=0;
                    for(time_index=0;time_index<10;time_index=time_index+1)
                        total=total+raw_value(profile,tile_seed,time_index,lane)
                            *right_coefficient(profile,rank,time_index);
                    intermediate[rank*16+lane]=rne_sat_q8(total,shift);
                end
            for(beat=0;beat<5;beat=beat+1)begin
                packed_result='0;
                for(row_in_beat=0;row_in_beat<2;row_in_beat=row_in_beat+1)begin
                    row=beat*2+row_in_beat;
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=bias_value(profile,row);
                        for(rank=0;rank<3;rank=rank+1)
                            total=total+intermediate[rank*16+lane]
                                *left_coefficient(profile,row,rank);
                        total=sat_q24(total);
                        packed_result[row_in_beat*16+lane]=
                            (total>=threshold_value(profile));
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
        integer phase;
        begin
            expected_read=0;expected_write=0;numeric_mismatches=0;
            fifo_peak=0;result_stalls=0;raw_stalls=0;release_waits=0;
            overlap_cycles=0;replace_cycles=0;full_pop_push_cycles=0;
            raw_dual_arb=0;inter_dual_arb=0;
            raw_order_errors=0;inter_order_errors=0;
            stage1_product_ops=0;stage2_product_ops=0;tile_done_count=0;
            no_fallthrough_checks=0;release_empty_checks=0;
            legal_protocol_error_glitches=0;ready_phase=0;
            stage1_acc_reference_checks=0;stage2_event_reference_checks=0;
            rne_tie_even_cases=0;rne_tie_odd_cases=0;
            q8_saturation_cases=0;q24_saturation_cases=0;
            internal_reference_mismatches=0;
            legal_halfcycle_checks=0;legal_intra_half_changes=0;
            fault_edge_pop_push=0;fault_edge_result_order_checks=0;
            n0_held_cycles=0;quarantine_probe_cycles=0;
            for(phase=0;phase<6;phase=phase+1)config_phase_accepts[phase]=0;
            for(phase=0;phase<5;phase=phase+1)raw_phase_accepts[phase]=0;
            for(phase=0;phase<5;phase=phase+1)result_phase_accepts[phase]=0;
        end
    endtask

    task automatic reset_dut;
        begin
            @(negedge clk_core);
            rst_core=1'b1;config_valid=1'b0;config_data='0;config_last=1'b0;
            raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
            release_valid=1'b0;ready_mode=0;
            repeat(4)@(posedge clk_core);
            @(negedge clk_core);rst_core=1'b0;clear_observation();
            legal_halfcycle_monitor_enable=1'b0;
        end
    endtask

    task automatic send_config(
        input logic [1535:0] frame,input integer framing_attack,
        input integer bubbles_after_beat0,input bit expect_fault,
        output time first_accept_time);
        integer beat,bubble;
        logic last_value;
        begin
            if(!expect_fault&&frame[1535:1349]!=='0)
                $fatal(1,"independent config builder polluted padding value=%h",
                    frame[1535:1349]);
            first_accept_time=0;config_valid=1'b1;
            for(beat=0;beat<6;beat=beat+1)begin
                config_data=frame[beat*256+:256];
                last_value=(beat==5);
                if(framing_attack==1&&beat==0)last_value=1'b1;
                if(framing_attack==2&&beat==5)last_value=1'b0;
                config_last=last_value;
                if(beat==5&&!expect_fault)begin
                    #1;
                    if(u_dut.config_beat_q!=5||
                            u_dut.config_candidate[1535:1349]!=='0
                            ||!u_dut.candidate_padding_legal
                            ||!u_dut.candidate_requant_legal
                            ||!u_dut.candidate_descriptor_legal)
                        $fatal(1,"pre-final config assembly drift dutbeat=%0d padding=%h pad=%0b req=%0b desc=%0b",
                            u_dut.config_beat_q,
                            u_dut.config_candidate[1535:1349],
                            u_dut.candidate_padding_legal,
                            u_dut.candidate_requant_legal,
                            u_dut.candidate_descriptor_legal);
                end
                do @(posedge clk_core); while(!config_accept);
                if(beat==0)first_accept_time=$time;
                #1;
                if(!expect_fault&&protocol_error&&!u_dut.protocol_error_q)
                    legal_protocol_error_glitches=
                        legal_protocol_error_glitches+1;
                if(u_dut.protocol_error_q)break;
                if(beat!=5)begin
                    @(negedge clk_core);
                    if(beat==0&&bubbles_after_beat0!=0)begin
                        config_valid=1'b0;
                        for(bubble=0;bubble<bubbles_after_beat0;bubble=bubble+1)
                            @(posedge clk_core);
                        @(negedge clk_core);config_valid=1'b1;
                    end
                end
            end
            if(expect_fault&&!u_dut.protocol_error_q)
                $fatal(1,"config attack escaped");
            if(!expect_fault&&(!config_loaded||debug_config_beats!=6))
                $fatal(1,"legal 6x256 config did not load loaded=%0b beats=%0d protocol=%0b pad=%0b req=%0b desc=%0b",
                    config_loaded,debug_config_beats,protocol_error,
                    u_dut.candidate_padding_legal,u_dut.candidate_requant_legal,
                    u_dut.candidate_descriptor_legal);
            @(negedge clk_core);config_valid=1'b0;config_data='0;config_last=1'b0;
        end
    endtask

    task automatic send_tiles(
        input integer profile,input integer count,input integer seed_base,
        input logic [TAG_W-1:0] tag_base);
        integer tile,beat,time_index,lane,value;
        logic [1279:0] payload;
        logic [TAG_W-1:0] tag_value;
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
                    if(protocol_error&&!u_dut.protocol_error_q)
                        legal_protocol_error_glitches=
                            legal_protocol_error_glitches+1;
                    if(u_dut.protocol_error_q)
                        $fatal(1,"legal 5x256 raw frame faulted");
                    if(!(tile==count-1&&beat==4))@(negedge clk_core);
                end
            end
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic finish_context(
        input integer tiles,input integer expected_extra_cycles,
        input time first_accept_time,output integer measured_cycles);
        time release_time;
        begin
            release_valid=1'b1;
            do @(posedge clk_core); while(!release_accept);
            release_time=$time;
            measured_cycles=((release_time-first_accept_time)/10)+1;
            #1;
            if(!context_retire_valid||context_retire_cycles!=measured_cycles)
                $fatal(1,"retire count mismatch measured=%0d rtl=%0d",
                    measured_cycles,context_retire_cycles);
            if(tiles>0&&ready_mode==0
                    &&measured_cycles!=(5*tiles+19+expected_extra_cycles))
                $fatal(1,"nonempty cycle mismatch N=%0d got=%0d expected=%0d",
                    tiles,measured_cycles,5*tiles+19+expected_extra_cycles);
            if(tiles==0&&measured_cycles!=7)
                $fatal(1,"zero-tile boundary drift got=%0d expected=7",measured_cycles);
            if(expected_read!=expected_write||numeric_mismatches!=0)
                $fatal(1,"scoreboard incomplete %0d/%0d mismatch=%0d",
                    expected_read,expected_write,numeric_mismatches);
            if(debug_config_beats!=6||debug_raw_beats!=5*tiles
                    ||debug_tiles_loaded!=tiles||debug_stage1_issues!=5*tiles
                    ||debug_stage1_done!=tiles||debug_stage2_issues!=5*tiles
                    ||debug_stage2_done!=tiles||debug_product_pushes!=5*tiles
                    ||debug_result_departures!=5*tiles)
                $fatal(1,"conservation mismatch N=%0d",tiles);
            if(stage1_product_ops!=480*tiles||stage2_product_ops!=480*tiles)
                $fatal(1,"96-way product count mismatch N=%0d s1=%0d s2=%0d",
                    tiles,stage1_product_ops,stage2_product_ops);
            if(tile_done_count!=tiles)$fatal(1,"tile_done count mismatch");
            @(negedge clk_core);release_valid=1'b0;
        end
    endtask

    task automatic run_context(
        input integer profile,input integer tiles,input integer seed,
        input logic [TAG_W-1:0] tag_base,input integer ready_selection,
        input integer config_bubbles,output integer cycles);
        time first_accept_time;
        begin
            clear_observation();ready_mode=ready_selection;
            legal_halfcycle_monitor_enable=1'b1;
            active_profile=profile;build_config(profile);
            send_config(legal_config,0,config_bubbles,1'b0,first_accept_time);
            if(tiles!=0)send_tiles(profile,tiles,seed,tag_base);
            finish_context(tiles,config_bubbles,first_accept_time,cycles);
            legal_halfcycle_monitor_enable=1'b0;
            ready_mode=0;
        end
    endtask

    task automatic raw_attack(input integer attack);
        integer beat,time_index,lane,value;
        logic [1279:0] payload;
        logic [TAG_W-1:0] tag_value;
        begin
            payload='0;tag_value=48'h2760_bad0_0000+attack;
            for(time_index=0;time_index<10;time_index=time_index+1)
                for(lane=0;lane<16;lane=lane+1)begin
                    value=raw_value(0,80+attack,time_index,lane);
                    payload[((time_index*16+lane)*8)+:8]=value[7:0];
                end
            raw_valid=1'b1;
            for(beat=0;beat<5;beat=beat+1)begin
                raw_data=payload[beat*256+:256];raw_last=(beat==4);
                if(attack==1&&beat==0)raw_last=1'b1;
                if(attack==2&&beat==3)raw_last=1'b1;
                if(attack==3&&beat==4)raw_last=1'b0;
                raw_tag=((attack==4&&beat==1)||(attack==5&&beat==4))?
                    tag_value+1'b1:tag_value;
                do @(posedge clk_core); while(!raw_accept);
                #1;if(u_dut.protocol_error_q)break;
                if(beat!=4)@(negedge clk_core);
            end
            if(!u_dut.protocol_error_q)
                $fatal(1,"raw framing/tag attack escaped %0d",attack);
            @(negedge clk_core);raw_valid=1'b0;raw_data='0;raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic check_fail_closed(input bit expected_loaded);
        integer probe;
        begin
            config_valid=1'b1;raw_valid=1'b1;release_valid=1'b1;result_ready=1'b1;
            for(probe=0;probe<3;probe=probe+1)begin
                @(posedge clk_core);#1;
                if(!protocol_error||config_loaded!=expected_loaded||config_ready
                        ||raw_ready||release_ready||config_accept||raw_accept
                        ||result_valid||result_accept||release_accept||stage1_issue
                        ||stage2_issue||product_push||fifo_pop)
                    $fatal(1,"fail-closed quarantine escaped");
            end
            @(negedge clk_core);config_valid=1'b0;raw_valid=1'b0;release_valid=1'b0;
        end
    endtask

    task automatic held_zero_tile_release_attack;
        time ignored_time;
        integer hold_cycle;
        logic [31:0] cfg_before,raw_before,tiles_before,s1_before,s2_before;
        logic [31:0] push_before,pop_before,cycles_before;
        begin
            reset_dut();build_config(0);
            send_config(legal_config,0,0,1'b0,ignored_time);
            if(!config_loaded||debug_tiles_loaded!=0||busy)
                $fatal(1,"M286 N0 setup drift loaded=%0b tiles=%0d busy=%0b",
                    config_loaded,debug_tiles_loaded,busy);
            cfg_before=debug_config_beats;raw_before=debug_raw_beats;
            tiles_before=debug_tiles_loaded;s1_before=debug_stage1_issues;
            s2_before=debug_stage2_issues;push_before=debug_product_pushes;
            pop_before=debug_result_departures;cycles_before=debug_context_cycles;
            release_valid=1'b1;
            #0.2;
            if(release_ready||release_accept||protocol_error||context_retire_valid)
                $fatal(1,"M286 N0 attempt was not edge-registered fail closed");
            @(posedge clk_core);#1;
            if(!protocol_error||release_ready||release_accept
                    ||context_retire_valid||!config_loaded)
                $fatal(1,"M286 N0 first edge did not enter sticky quarantine");
            if(debug_context_cycles!=cycles_before+1)
                $fatal(1,"M286 N0 fault edge cycle accounting drift got=%0d before=%0d",
                    debug_context_cycles,cycles_before);
            cycles_before=debug_context_cycles;
            // Keep release asserted and drive every other producer after capture.
            @(negedge clk_core);
            config_valid=1'b1;config_data=256'h286;config_last=1'b1;
            raw_valid=1'b1;raw_data=256'hfeed_286;raw_last=1'b1;
            raw_tag=48'h2860_bad0_0000;result_ready=1'b1;
            for(hold_cycle=0;hold_cycle<8;hold_cycle=hold_cycle+1)begin
                @(posedge clk_core);#1;n0_held_cycles=n0_held_cycles+1;
                if(!protocol_error||config_ready||raw_ready||release_ready
                        ||config_accept||raw_accept||result_valid||result_accept
                        ||release_accept||stage1_issue||stage2_issue||product_push
                        ||fifo_push||fifo_pop||context_retire_valid)
                    $fatal(1,"M286 N0 held-release quarantine escaped cycle=%0d",hold_cycle);
                if(debug_config_beats!=cfg_before||debug_raw_beats!=raw_before
                        ||debug_tiles_loaded!=tiles_before
                        ||debug_stage1_issues!=s1_before||debug_stage2_issues!=s2_before
                        ||debug_product_pushes!=push_before
                        ||debug_result_departures!=pop_before
                        ||debug_context_cycles!=cycles_before)
                    $fatal(1,"M286 N0 quarantined state mutated cycle=%0d",hold_cycle);
            end
            @(negedge clk_core);release_valid=1'b0;config_valid=1'b0;
            raw_valid=1'b0;config_data='0;raw_data='0;config_last=1'b0;
            raw_last=1'b0;raw_tag='0;
        end
    endtask

    task automatic drive_legal_tiles_until_fault(
        input integer count,input integer seed_base,
        input logic [TAG_W-1:0] tag_base);
        integer tile,beat,time_index,lane,value;
        logic [1279:0] payload;
        logic [TAG_W-1:0] tag_value;
        begin:stream_body
            raw_valid=1'b1;
            for(tile=0;tile<count;tile=tile+1)begin
                payload='0;tag_value=tag_base+tile;
                for(time_index=0;time_index<10;time_index=time_index+1)
                    for(lane=0;lane<16;lane=lane+1)begin
                        value=raw_value(0,seed_base+tile,time_index,lane);
                        payload[((time_index*16+lane)*8)+:8]=value[7:0];
                    end
                enqueue_expected_tile(0,seed_base+tile,tag_value);
                for(beat=0;beat<5;beat=beat+1)begin
                    raw_data=payload[beat*256+:256];
                    raw_last=(beat==4);raw_tag=tag_value;
                    do @(posedge clk_core); while(!raw_accept&&!protocol_error);
                    #1;
                    if(protocol_error)disable stream_body;
                    if(!(tile==count-1&&beat==4))@(negedge clk_core);
                end
            end
        end
    endtask

    task automatic fault_edge_fifo_result_attack;
        time ignored_time;
        integer search_cycles,expected_index_before,target_beat;
        logic target_bank;
        logic [255:0] target_word_before;
        logic [31:0] raw_before,push_before,pop_before,cycles_after;
        logic [4:0] fifo_before;
        logic [3:0] read_ptr_before,write_ptr_before;
        logic [47:0] head_tag_before;
        logic [2:0] head_beat_before;
        logic [47:0] head_data_before;
        logic [31:0] cfg_after,raw_after,tiles_after,s1_after,s2_after;
        logic [31:0] push_after,pop_after,repl_after;
        logic [4:0] fifo_after;
        logic [3:0] read_ptr_after,write_ptr_after;
        begin
            reset_dut();active_profile=0;build_config(0);
            send_config(legal_config,0,0,1'b0,ignored_time);
            ready_mode=2;result_ready=1'b0;
            fork:edge_campaign
                drive_legal_tiles_until_fault(80,120,48'h2860_4000_0000);
                begin:edge_controller
                    search_cycles=0;
                    while(result_fifo_occupancy!=16||!u_dut.product_valid_q)begin
                        @(negedge clk_core);search_cycles=search_cycles+1;
                        if(search_cycles>1000)$fatal(1,"M286 failed to fill FIFO before fault edge");
                    end
                    ready_mode=0;#0.2;result_ready=1'b1;
                    search_cycles=0;
                    while(!(raw_valid&&raw_ready&&result_valid&&product_push))begin
                        @(negedge clk_core);search_cycles=search_cycles+1;
                        if(search_cycles>1000)$fatal(1,"M286 no concurrent raw/result/product window");
                    end
                    target_bank=u_dut.raw_target_bank;
                    target_beat=u_dut.fill_active_q?u_dut.fill_beat_q:0;
                    if(target_bank)
                        target_word_before=u_dut.raw_bank1_q[target_beat*256+:256];
                    else target_word_before=u_dut.raw_bank0_q[target_beat*256+:256];
                    raw_before=debug_raw_beats;push_before=debug_product_pushes;
                    pop_before=debug_result_departures;fifo_before=result_fifo_occupancy;
                    read_ptr_before=u_dut.fifo_read_pointer_q;
                    write_ptr_before=u_dut.fifo_write_pointer_q;
                    expected_index_before=expected_read;
                    head_tag_before=result_tag;head_beat_before=result_beat;
                    head_data_before=result_data;
                    if(expected_index_before>=expected_write
                            ||head_tag_before!==expected_tag[expected_index_before]
                            ||head_beat_before!==expected_beat[expected_index_before]
                            ||head_data_before!==expected_data[expected_index_before])
                        $fatal(1,"M286 fault-edge pre-head order mismatch index=%0d",
                            expected_index_before);
                    // Flip framing only after a stable legal beat has been held.
                    raw_last=!u_dut.raw_expected_last;
                    #0.2;
                    if(!u_dut.raw_frame_error||!u_dut.fault_event||!raw_accept
                            ||!result_accept||!fifo_pop||!product_push||!fifo_push)
                        $fatal(1,"M286 failed to align raw fault with FIFO pop+push");
                    @(posedge clk_core);#1;
                    fault_edge_pop_push=fault_edge_pop_push+1;
                    if(!protocol_error||result_valid||result_accept
                            ||debug_raw_beats!=raw_before
                            ||debug_product_pushes!=push_before+1
                            ||debug_result_departures!=pop_before+1
                            ||result_fifo_occupancy!=fifo_before
                            ||u_dut.fifo_read_pointer_q!=read_ptr_before+1'b1
                            ||u_dut.fifo_write_pointer_q!=write_ptr_before+1'b1
                            ||expected_read!=expected_index_before+1)
                        $fatal(1,"M286 fault-edge commit mismatch raw=%0d/%0d push=%0d/%0d pop=%0d/%0d fifo=%0d/%0d",
                            debug_raw_beats,raw_before,debug_product_pushes,push_before,
                            debug_result_departures,pop_before,
                            result_fifo_occupancy,fifo_before);
                    if(target_bank)begin
                        if(u_dut.raw_bank1_q[target_beat*256+:256]!==target_word_before)
                            $fatal(1,"M286 offending raw payload committed bank1 beat=%0d",target_beat);
                    end else if(u_dut.raw_bank0_q[target_beat*256+:256]!==target_word_before)
                        $fatal(1,"M286 offending raw payload committed bank0 beat=%0d",target_beat);
                    fault_edge_result_order_checks=fault_edge_result_order_checks+1;
                    cfg_after=debug_config_beats;raw_after=debug_raw_beats;
                    tiles_after=debug_tiles_loaded;s1_after=debug_stage1_issues;
                    s2_after=debug_stage2_issues;push_after=debug_product_pushes;
                    pop_after=debug_result_departures;repl_after=debug_product_replacements;
                    cycles_after=debug_context_cycles;fifo_after=result_fifo_occupancy;
                    read_ptr_after=u_dut.fifo_read_pointer_q;
                    write_ptr_after=u_dut.fifo_write_pointer_q;
                    @(negedge clk_core);config_valid=1'b1;config_last=1'b1;
                    release_valid=1'b1;raw_valid=1'b1;raw_last=1'b1;
                    result_ready=1'b1;
                    repeat(8)begin
                        @(posedge clk_core);#1;quarantine_probe_cycles=quarantine_probe_cycles+1;
                        if(!protocol_error||config_ready||raw_ready||release_ready
                                ||config_accept||raw_accept||result_valid||result_accept
                                ||release_accept||stage1_issue||stage2_issue
                                ||product_push||fifo_push||fifo_pop)
                            $fatal(1,"M286 fault-edge sticky quarantine escaped");
                        if(debug_config_beats!=cfg_after||debug_raw_beats!=raw_after
                                ||debug_tiles_loaded!=tiles_after
                                ||debug_stage1_issues!=s1_after
                                ||debug_stage2_issues!=s2_after
                                ||debug_product_pushes!=push_after
                                ||debug_result_departures!=pop_after
                                ||debug_product_replacements!=repl_after
                                ||debug_context_cycles!=cycles_after
                                ||result_fifo_occupancy!=fifo_after
                                ||u_dut.fifo_read_pointer_q!=read_ptr_after
                                ||u_dut.fifo_write_pointer_q!=write_ptr_after)
                            $fatal(1,"M286 sticky quarantine state mutation");
                    end
                    disable edge_campaign;
                end
            join
            @(negedge clk_core);config_valid=1'b0;raw_valid=1'b0;
            release_valid=1'b0;raw_last=1'b0;config_last=1'b0;
            raw_data='0;config_data='0;raw_tag='0;
        end
    endtask

    always @(negedge clk_core)begin
        if(rst_core)begin result_ready<=1'b1;ready_phase<=0;end
        else if(ready_mode==0)result_ready<=1'b1;
        else if(ready_mode==2)result_ready<=1'b0;
        else begin
            result_ready<=((ready_phase%8)==0);
            ready_phase<=ready_phase+1;
        end
    end

    always @(posedge clk_core)begin:independent_halfcycle_glitch_probe
        logic [4:0] control_sample;
        logic [147:0] result_sample;
        if(!rst_core&&legal_halfcycle_monitor_enable)begin
            #0.2;
            control_sample={protocol_error,result_valid,stage1_issue,
                stage2_issue,product_push};
            result_sample={result_valid,result_tag,result_beat,
                result_valid_bits,result_data};
            if(protocol_error)legal_protocol_error_glitches=
                legal_protocol_error_glitches+1;
            #4.5;
            legal_halfcycle_checks=legal_halfcycle_checks+1;
            if({protocol_error,result_valid,stage1_issue,stage2_issue,product_push}
                    !==control_sample
                    ||{result_valid,result_tag,result_beat,result_valid_bits,result_data}
                    !==result_sample)
                legal_intra_half_changes=legal_intra_half_changes+1;
        end
    end

    always @(posedge clk_core)begin
        if(!rst_core)begin
            if(legal_halfcycle_monitor_enable&&config_accept)
                config_phase_accepts[u_dut.config_beat_q]=
                    config_phase_accepts[u_dut.config_beat_q]+1;
            if(legal_halfcycle_monitor_enable&&raw_accept)begin
                integer raw_phase;
                raw_phase=u_dut.fill_active_q?u_dut.fill_beat_q:0;
                raw_phase_accepts[raw_phase]=raw_phase_accepts[raw_phase]+1;
            end
            if(legal_halfcycle_monitor_enable&&result_accept)
                result_phase_accepts[result_beat]=result_phase_accepts[result_beat]+1;
            if(result_fifo_occupancy>fifo_peak)fifo_peak=result_fifo_occupancy;
            if(result_valid&&!result_ready)result_stalls=result_stalls+1;
            if(raw_valid&&!raw_ready)raw_stalls=raw_stalls+1;
            if(release_valid&&!release_ready)release_waits=release_waits+1;
            if(stage1_issue&&stage2_issue)overlap_cycles=overlap_cycles+1;
            if(product_replace)replace_cycles=replace_cycles+1;
            if(result_fifo_occupancy==16&&fifo_pop&&fifo_push)
                full_pop_push_cycles=full_pop_push_cycles+1;
            if(stage1_issue)begin
                integer rank,lane,time_index,total,phase_limit,accumulator;
                integer shift,magnitude,quotient,remainder,half,rounded;
                stage1_product_ops=stage1_product_ops+96;
                phase_limit=(u_dut.stage1_selected_phase*2)+1;
                shift=(active_profile==1)?0:3;
                for(rank=0;rank<3;rank=rank+1)
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=0;accumulator=rank*16+lane;
                        for(time_index=0;time_index<=phase_limit;
                                time_index=time_index+1)
                            total=total+$signed(u_dut.stage1_raw_data[
                                ((time_index*16)+lane)*8+:8])
                                *right_coefficient(active_profile,rank,time_index);
                        stage1_acc_reference_checks=
                            stage1_acc_reference_checks+1;
                        if($signed(u_dut.stage1_sum_comb[accumulator])!=total)begin
                            internal_reference_mismatches=
                                internal_reference_mismatches+1;
                            $fatal(1,"stage1 accumulator reference mismatch phase=%0d rank=%0d lane=%0d rtl=%0d ref=%0d",
                                u_dut.stage1_selected_phase,rank,lane,
                                $signed(u_dut.stage1_sum_comb[accumulator]),total);
                        end
                        if(u_dut.stage1_selected_phase==4)begin
                            rounded=rne_sat_q8(total,shift);
                            if($signed(u_dut.stage1_requant_comb[
                                    accumulator*8+:8])!=rounded)begin
                                internal_reference_mismatches=
                                    internal_reference_mismatches+1;
                                $fatal(1,"stage1 requant reference mismatch rank=%0d lane=%0d",
                                    rank,lane);
                            end
                            magnitude=(total<0)?-total:total;
                            if(shift!=0)begin
                                quotient=magnitude>>>shift;
                                remainder=magnitude&((1<<shift)-1);
                                half=1<<(shift-1);
                                if(remainder==half)begin
                                    if((quotient&1)==0)
                                        rne_tie_even_cases=rne_tie_even_cases+1;
                                    else rne_tie_odd_cases=rne_tie_odd_cases+1;
                                end
                            end
                            if((total>>>shift)>127||(total>>>shift)< -128)
                                q8_saturation_cases=q8_saturation_cases+1;
                        end
                    end
            end
            if(stage2_issue)begin
                integer row_in_beat,row,lane,rank,total,intermediate;
                logic expected_event;
                stage2_product_ops=stage2_product_ops+96;
                for(row_in_beat=0;row_in_beat<2;row_in_beat=row_in_beat+1)begin
                    row=(u_dut.stage2_selected_phase*2)+row_in_beat;
                    for(lane=0;lane<16;lane=lane+1)begin
                        total=bias_value(active_profile,row);
                        for(rank=0;rank<3;rank=rank+1)begin
                            intermediate=$signed(u_dut.stage2_intermediate[
                                ((rank*16)+lane)*8+:8]);
                            total=total+intermediate
                                *left_coefficient(active_profile,row,rank);
                        end
                        if(total>8388607||total< -8388608)
                            q24_saturation_cases=q24_saturation_cases+1;
                        expected_event=(sat_q24(total)>=
                            threshold_value(active_profile));
                        stage2_event_reference_checks=
                            stage2_event_reference_checks+1;
                        if(u_dut.stage2_event_bits[
                                row_in_beat*16+lane]!==expected_event)begin
                            internal_reference_mismatches=
                                internal_reference_mismatches+1;
                            $fatal(1,"stage2 event reference mismatch phase=%0d row=%0d lane=%0d",
                                u_dut.stage2_selected_phase,row,lane);
                        end
                    end
                end
            end
            if(tile_done_valid)tile_done_count=tile_done_count+1;

            if(stage1_issue&&!u_dut.stage1_active_q
                    &&u_dut.raw_ready_q==2'b11)begin
                raw_dual_arb=raw_dual_arb+1;
                if((u_dut.raw_order0_q<u_dut.raw_order1_q
                        &&u_dut.stage1_selected_raw_bank!=0)
                    ||(u_dut.raw_order1_q<u_dut.raw_order0_q
                        &&u_dut.stage1_selected_raw_bank!=1))
                    raw_order_errors=raw_order_errors+1;
            end
            if(stage2_issue&&!u_dut.stage2_active_q
                    &&u_dut.inter_valid_q==2'b11)begin
                inter_dual_arb=inter_dual_arb+1;
                if((u_dut.inter_order0_q<u_dut.inter_order1_q
                        &&u_dut.stage2_selected_bank!=0)
                    ||(u_dut.inter_order1_q<u_dut.inter_order0_q
                        &&u_dut.stage2_selected_bank!=1))
                    inter_order_errors=inter_order_errors+1;
            end
            if(result_fifo_occupancy==0&&fifo_push)begin
                no_fallthrough_checks=no_fallthrough_checks+1;
                if(result_accept)$fatal(1,"registered FIFO illegally fell through");
            end
            if(release_accept)begin
                release_empty_checks=release_empty_checks+1;
                if(busy||result_fifo_occupancy!=0||raw_bank_occupancy!=0
                        ||intermediate_bank_occupancy!=0
                        ||u_dut.fill_active_q||u_dut.stage1_active_q
                        ||u_dut.stage2_active_q||u_dut.product_valid_q)
                    $fatal(1,"release accepted before complete drain");
            end
            if(result_accept)begin
                if(expected_read>=expected_write)$fatal(1,"unexpected result");
                if(result_tag!==expected_tag[expected_read]
                        ||result_beat!==expected_beat[expected_read]
                        ||result_valid_bits!==48'h0000ffffffff
                        ||result_data!==expected_data[expected_read])begin
                    numeric_mismatches=numeric_mismatches+1;
                    $fatal(1,"reference/order mismatch index=%0d tag=%h/%h beat=%0d/%0d data=%h/%h",
                        expected_read,result_tag,expected_tag[expected_read],
                        result_beat,expected_beat[expected_read],
                        result_data,expected_data[expected_read]);
                end
                expected_read=expected_read+1;
            end
        end
    end

    initial begin:independent_campaign
        integer n1_cycles,n4_cycles,gapped_cycles,pressure_cycles;
        integer pressure_legal_glitches,pressure_stage1_checks;
        integer pressure_stage2_checks,pressure_tie_even,pressure_tie_odd;
        integer pressure_q8_sat,profile1_q24_sat,pressure_half_checks;
        integer held_n0_observed;
        integer attack;
        time ignored_time;
        logic [1535:0] attacked;
        clk_core=0;rst_core=1;config_valid=0;config_data='0;config_last=0;
        raw_valid=0;raw_data='0;raw_last=0;raw_tag='0;
        result_ready=1;release_valid=0;ready_mode=0;ready_phase=0;
        legal_halfcycle_monitor_enable=1'b0;
        clear_observation();

        reset_dut();run_context(0,1,1,48'h2860_1000_0000,0,0,n1_cycles);
        reset_dut();run_context(1,4,10,48'h2860_2000_0000,0,0,n4_cycles);
        profile1_q24_sat=q24_saturation_cases;
        if(profile1_q24_sat==0)$fatal(1,"profile1 did not cover Q24 saturation");
        reset_dut();run_context(0,1,20,48'h2860_2500_0000,0,2,gapped_cycles);
        reset_dut();run_context(0,40,40,48'h2860_3000_0000,1,0,pressure_cycles);
        pressure_legal_glitches=legal_protocol_error_glitches;
        pressure_half_checks=legal_halfcycle_checks;
        pressure_stage1_checks=stage1_acc_reference_checks;
        pressure_stage2_checks=stage2_event_reference_checks;
        pressure_tie_even=rne_tie_even_cases;
        pressure_tie_odd=rne_tie_odd_cases;
        pressure_q8_sat=q8_saturation_cases;
        if(pressure_cycles!=1618)$fatal(1,"40-tile pressure drift got=%0d",pressure_cycles);
        if(pressure_legal_glitches!=0||legal_intra_half_changes!=0
                ||pressure_half_checks==0)
            $fatal(1,"M286 legal sustained-valid glitch probe failed pulse=%0d change=%0d checks=%0d",
                pressure_legal_glitches,legal_intra_half_changes,pressure_half_checks);
        if(config_phase_accepts[0]!=1||config_phase_accepts[1]!=1
                ||config_phase_accepts[2]!=1||config_phase_accepts[3]!=1
                ||config_phase_accepts[4]!=1||config_phase_accepts[5]!=1
                ||raw_phase_accepts[0]!=40||raw_phase_accepts[1]!=40
                ||raw_phase_accepts[2]!=40||raw_phase_accepts[3]!=40
                ||raw_phase_accepts[4]!=40||result_phase_accepts[0]!=40
                ||result_phase_accepts[1]!=40||result_phase_accepts[2]!=40
                ||result_phase_accepts[3]!=40||result_phase_accepts[4]!=40)
            $fatal(1,"M286 sustained-valid phase coverage drift cfg=%0d/%0d/%0d/%0d/%0d/%0d raw=%0d/%0d/%0d/%0d/%0d result=%0d/%0d/%0d/%0d/%0d",
                config_phase_accepts[0],config_phase_accepts[1],
                config_phase_accepts[2],config_phase_accepts[3],
                config_phase_accepts[4],config_phase_accepts[5],
                raw_phase_accepts[0],raw_phase_accepts[1],
                raw_phase_accepts[2],raw_phase_accepts[3],raw_phase_accepts[4],
                result_phase_accepts[0],result_phase_accepts[1],
                result_phase_accepts[2],result_phase_accepts[3],
                result_phase_accepts[4]);
        if(pressure_stage1_checks!=9600||pressure_stage2_checks!=6400
                ||pressure_tie_even==0||pressure_tie_odd==0
                ||pressure_q8_sat==0||internal_reference_mismatches!=0)
            $fatal(1,"internal numeric coverage missing s1=%0d s2=%0d tie=%0d/%0d q8sat=%0d mismatch=%0d",
                pressure_stage1_checks,pressure_stage2_checks,
                pressure_tie_even,pressure_tie_odd,pressure_q8_sat,
                internal_reference_mismatches);
        if(fifo_peak!=16||result_stalls==0||raw_stalls==0
                ||full_pop_push_cycles==0||overlap_cycles==0||replace_cycles==0
                ||raw_dual_arb==0||inter_dual_arb==0
                ||raw_order_errors!=0||inter_order_errors!=0
                ||release_waits==0||no_fallthrough_checks==0
                ||release_empty_checks!=1)
            $fatal(1,"pressure/lifecycle coverage missing fifo=%0d rstall=%0d rawstall=%0d full=%0d overlap=%0d replace=%0d rawarb=%0d interarb=%0d rawerr=%0d intererr=%0d relwait=%0d nofall=%0d relempty=%0d",
                fifo_peak,result_stalls,raw_stalls,full_pop_push_cycles,
                overlap_cycles,replace_cycles,raw_dual_arb,inter_dual_arb,
                raw_order_errors,inter_order_errors,release_waits,
                no_fallthrough_checks,release_empty_checks);

        for(attack=1;attack<=8;attack=attack+1)begin
            reset_dut();build_config(0);attacked=legal_config;
            if(attack==3)attacked[1500]=1'b1;
            if(attack==4)attacked[240+:5]=5'd24;
            if(attack==5)attacked[606]=1'b1;
            if(attack==6)begin
                attacked[485+2]=1'b1;attacked[725+(2*3)+:3]=3'd4;
                attacked[245+:8]=8'd17;
            end
            if(attack==7)begin
                attacked[245+:8]=8'd3;attacked[485+:4]=4'b0011;
                attacked[605+:4]=4'b0000;attacked[725+:12]='0;
                attacked[725+3+:3]=3'd1;
            end
            if(attack==8)attacked[245+:8]=attacked[245+:8]+1'b1;
            send_config(attacked,(attack<=2)?attack:0,0,1'b1,ignored_time);
            check_fail_closed(1'b0);
        end
        for(attack=1;attack<=5;attack=attack+1)begin
            reset_dut();build_config(0);
            send_config(legal_config,0,0,1'b0,ignored_time);
            raw_attack(attack);check_fail_closed(1'b1);
        end

        held_zero_tile_release_attack();held_n0_observed=n0_held_cycles;
        if(held_n0_observed!=8)$fatal(1,"M286 N0 held-release coverage drift");
        fault_edge_fifo_result_attack();
        if(fault_edge_pop_push!=1||fault_edge_result_order_checks!=1
                ||quarantine_probe_cycles!=8)
            $fatal(1,"M286 fault-edge coverage drift edge=%0d order=%0d quarantine=%0d",
                fault_edge_pop_push,fault_edge_result_order_checks,
                quarantine_probe_cycles);

        $display("PASS M286 independent M285/M273r2 hammer n1=%0d n4=%0d gapped_n1=%0d pressure_n40=%0d legal_protocol_glitches=%0d legal_intra_half_changes=0 halfcycle_checks=%0d config_phase_accepts=1/1/1/1/1/1 raw_phase_accepts=40/40/40/40/40 result_phase_accepts=40/40/40/40/40 stage1_checks=%0d stage2_checks=%0d rne_ties=%0d/%0d q8_sat=%0d q24_sat=%0d cfg_attacks=8 raw_attacks=5 n0_held_cycles=%0d fault_edge_fifo_pop_push=1 fault_edge_result_order=1 quarantine_cycles=8 reference_mismatches=0 new_speedup=false dc=false system_speedup=false headline=false",
            n1_cycles,n4_cycles,gapped_cycles,pressure_cycles,
            pressure_legal_glitches,pressure_half_checks,
            pressure_stage1_checks,
            pressure_stage2_checks,pressure_tie_even,pressure_tie_odd,
            pressure_q8_sat,profile1_q24_sat,held_n0_observed);
        $finish;
    end

    initial begin
        #4000000;$fatal(1,"M286 independent M273r2 timeout");
    end
endmodule

`default_nettype wire
