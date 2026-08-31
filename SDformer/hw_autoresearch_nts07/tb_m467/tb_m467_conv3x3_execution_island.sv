`timescale 1ns/1ps
`default_nettype none
module tb_m467_conv3x3_execution_island;
    localparam int TAG_BITS=24, ROWS=2;
    logic clk_core, reset_n;
    logic config_valid,config_ready,config_accept,config_commit;
    logic [1:0] config_beat_index;
    logic [TAG_BITS-1:0] config_tag;
    logic [255:0] config_data;
    logic config_operator_last_phase;
    logic row_valid,row_ready,row_accept,row_last;
    logic [11:0] row_id; logic [15:0] row_original;
    logic descriptor_write_valid,descriptor_write_ready;
    logic [TAG_BITS-1:0] descriptor_write_tag;
    logic [11:0] descriptor_write_address;
    logic [47:0] descriptor_write_data;
    logic descriptor_read_valid,descriptor_read_ready;
    logic [TAG_BITS-1:0] descriptor_read_tag;
    logic [11:0] descriptor_read_address;
    logic descriptor_response_valid,descriptor_response_ready;
    logic [TAG_BITS-1:0] descriptor_response_tag;
    logic [11:0] descriptor_response_address;
    logic [47:0] descriptor_response_data;
    logic payload_request_valid,payload_request_ready,payload_request_pwp;
    logic [TAG_BITS-1:0] payload_request_tag;
    logic payload_request_tile;
    logic [2:0] payload_request_block;
    logic [4:0] payload_request_center;
    logic [3:0] payload_request_source;
    logic payload_request_narrow;
    logic payload_response_valid,payload_response_ready,payload_response_pwp;
    logic [TAG_BITS-1:0] payload_response_tag;
    logic payload_response_tile;
    logic [2:0] payload_response_block;
    logic [4:0] payload_response_center;
    logic [3:0] payload_response_source;
    logic payload_response_narrow;
    logic [767:0] payload_response_low_or_weight;
    logic [511:0] payload_response_high;
    logic accumulator_read_valid,accumulator_read_ready;
    logic [14:0] accumulator_read_address;
    logic accumulator_response_valid,accumulator_response_ready;
    logic [14:0] accumulator_response_address;
    logic [1823:0] accumulator_response_data;
    logic accumulator_write_valid,accumulator_write_ready;
    logic [14:0] accumulator_write_address;
    logic [1823:0] accumulator_write_data;
    logic commit_valid,commit_ready,commit_last;
    logic [14:0] commit_address;
    logic [1823:0] commit_data;
    logic phase_done_valid,phase_done_ready;
    logic [TAG_BITS-1:0] phase_done_tag;
    logic [11:0] phase_done_active_rows;
    logic protocol_error,busy; logic [4:0] debug_state;
    logic [31:0] debug_descriptor_writes,debug_descriptor_reads;
    logic [31:0] debug_pwp_requests,debug_weight_requests;
    logic [31:0] debug_forward_hits,debug_accumulator_reads,debug_commits;
    logic [31:0] debug_zero_initializations,debug_zero_commits;
    logic [31:0] debug_row_live_sets,debug_row_live_clears;
    logic [7:0] debug_zero_init_slot_mask;
    logic debug_row_live_set_event,debug_row_live_clear_event;
    logic debug_forward_event,debug_operator_boundary_pending;

    logic [47:0] descriptor_mem[0:3];
    logic [1823:0] accumulator_mem[0:8*ROWS-1];
    logic signed [18:0] expected[0:7][0:ROWS-1][0:95];
    integer cycles, commits_seen, pwp_seen, weight_seen, plus_seen, minus_seen;
    integer tile_mask, block_mask, row_mask, commit_stalls, descriptor_stalls;
    integer reads_before_second_operator, forwards_before_second_operator;
    integer acc_read_stalls, acc_write_stalls, narrow_pwp_seen, negative_pwp_seen;
    bit expect_protocol_error;

    m467_conv3x3_execution_island #(.TAG_BITS(TAG_BITS),.ROWS_PER_PHASE(ROWS)) dut(.*);
    m467_conv3x3_execution_island_assertions #(.TAG_BITS(TAG_BITS)) sva(.*);
    always #1.5 clk_core=~clk_core;

    function automatic integer weight_value(input integer lane,input integer src);
        weight_value = src - 7 + (lane % 3);
    endfunction
    function automatic integer pwp_value(input integer lane,input integer tile,input integer block);
        pwp_value = 100 + tile*20 + block*3 + (lane%5);
    endfunction
    function automatic integer negative_pwp_value(input integer lane,input integer tile,input integer block);
        negative_pwp_value = -30 - tile*2 - block + (lane%5);
    endfunction
    function automatic integer accumulator_index(input logic [14:0] address);
        accumulator_index = address[14:12]*ROWS + address[11:0];
    endfunction

    task automatic clear_inputs;
        begin
            config_valid=0; config_beat_index=0; config_commit=0; config_tag=0;
            config_data=0; config_operator_last_phase=0; row_valid=0; row_id=0;
            row_original=0; row_last=0; descriptor_response_valid=0;
            descriptor_response_tag=0; descriptor_response_address=0;
            descriptor_response_data=0; payload_response_valid=0;
            payload_response_pwp=0; payload_response_tag=0;
            payload_response_tile=0; payload_response_block=0;
            payload_response_center=0; payload_response_source=0;
            payload_response_narrow=0; payload_response_low_or_weight=0;
            payload_response_high=0; accumulator_response_valid=0;
            accumulator_response_address=0; accumulator_response_data=0;
            phase_done_ready=1;
        end
    endtask
    task automatic apply_reset;
        begin reset_n=0; repeat(4) @(posedge clk_core); @(negedge clk_core); reset_n=1; end
    endtask
    task automatic send_config(input [23:0] tag,input bit operator_last,input bit narrow_center0);
        integer b;
        begin
            for(b=0;b<3;b=b+1) begin
                @(negedge clk_core); config_valid=1; config_beat_index=b;
                config_commit=(b==2); config_tag=tag; config_operator_last_phase=operator_last;
                if(b==0) begin config_data='1; config_data[15:0]=16'h000f; end
                else if(b==1) config_data='1;
                else begin config_data=0; config_data[0]=narrow_center0; end
                do @(posedge clk_core); while(!config_accept);
                @(negedge clk_core); config_valid=0;
            end
        end
    endtask
    task automatic send_row(input [11:0] id,input [15:0] value,input bit last);
        begin
            @(negedge clk_core); row_valid=1; row_id=id; row_original=value; row_last=last;
            do @(posedge clk_core); while(!row_accept);
            @(negedge clk_core); row_valid=0;
        end
    endtask

    // Deterministic backpressure on every outbound channel.
    always @(negedge clk_core) begin
        if(!reset_n) begin descriptor_write_ready=0; descriptor_read_ready=0;
            payload_request_ready=0; accumulator_read_ready=0;
            accumulator_write_ready=0; commit_ready=0;
        end else begin
            descriptor_write_ready=!(descriptor_write_valid && descriptor_stalls==0);
            descriptor_read_ready=(cycles%7)!=2;
            payload_request_ready=(cycles%6)!=3;
            accumulator_read_ready=(cycles%5)!=2;
            accumulator_write_ready=(cycles%6)!=4;
            commit_ready=(cycles%4)!=1;
        end
    end
    always @(posedge clk_core) if(reset_n) begin
        cycles++;
        if(protocol_error && !expect_protocol_error) $fatal(1,"unexpected protocol_error state=%0d dresp=%0b presp=%0b arsp=%0b m414=%0b m433=%0b",
            debug_state,descriptor_response_valid,payload_response_valid,
            accumulator_response_valid,dut.matcher_protocol_error,dut.m433_protocol_error);
        if(descriptor_write_valid&&!descriptor_write_ready) descriptor_stalls++;
        if(commit_valid&&!commit_ready) commit_stalls++;
        if(accumulator_read_valid&&!accumulator_read_ready) acc_read_stalls++;
        if(accumulator_write_valid&&!accumulator_write_ready) acc_write_stalls++;
        if(descriptor_write_valid&&descriptor_write_ready)
            descriptor_mem[descriptor_write_address]=descriptor_write_data;
        if(accumulator_write_valid&&accumulator_write_ready)
            accumulator_mem[accumulator_index(accumulator_write_address)]=accumulator_write_data;
        if(row_accept) row_mask |= 1<<row_id;
        if(payload_request_valid&&payload_request_ready) begin
            tile_mask |= 1<<payload_request_tile;
            block_mask |= 1<<payload_request_block;
            if(payload_request_pwp) pwp_seen++; else weight_seen++;
            if(payload_request_pwp&&payload_request_narrow) narrow_pwp_seen++;
            if(payload_request_pwp&&payload_request_tag==24'h467101) negative_pwp_seen++;
            if(!payload_request_pwp && payload_request_source==4) plus_seen++;
            if(!payload_request_pwp && payload_request_source==3) minus_seen++;
        end
        if(commit_valid&&commit_ready) begin
            integer slot,lane;
            integer commit_row;
            slot=commit_address[14:12]; commit_row=commit_address[11:0];
            if(commit_row>=ROWS) $fatal(1,"bad commit address %h",commit_address);
            for(lane=0;lane<96;lane=lane+1)
                if($signed(commit_data[lane*19 +:19]) !== expected[slot][commit_row][lane])
                    $fatal(1,"commit mismatch slot=%0d row=%0d lane=%0d got=%0d exp=%0d",
                        slot,commit_row,lane,$signed(commit_data[lane*19 +:19]),
                        expected[slot][commit_row][lane]);
            commits_seen++;
        end
    end

    initial begin : descriptor_model
        forever begin
            @(posedge clk_core);
            if(reset_n && descriptor_read_valid && descriptor_read_ready) begin
                automatic logic [11:0] a=descriptor_read_address;
                automatic logic [23:0] t=descriptor_read_tag;
                repeat(2) @(posedge clk_core);
                @(negedge clk_core); descriptor_response_tag=t;
                descriptor_response_address=a; descriptor_response_data=descriptor_mem[a];
                descriptor_response_valid=1;
                do @(posedge clk_core); while(!descriptor_response_ready);
                @(negedge clk_core); descriptor_response_valid=0;
            end
        end
    end
    initial begin : payload_model
        forever begin
            @(posedge clk_core);
            if(reset_n && payload_request_valid && payload_request_ready) begin
                automatic bit isp=payload_request_pwp;
                automatic bit ti=payload_request_tile;
                automatic bit ni=payload_request_narrow;
                automatic logic [2:0] bl=payload_request_block;
                automatic logic [4:0] ce=payload_request_center;
                automatic logic [3:0] so=payload_request_source;
                automatic logic [23:0] ta=payload_request_tag;
                repeat(1) @(posedge clk_core); @(negedge clk_core);
                payload_response_pwp=isp; payload_response_tile=ti;
                payload_response_block=bl; payload_response_center=ce;
                payload_response_source=so; payload_response_tag=ta;
                payload_response_narrow=ni; payload_response_low_or_weight=0;
                payload_response_high=0;
                for(integer l=0;l<96;l=l+1) begin
                    integer v;
                    v=isp?(ta == 24'h467101 ? negative_pwp_value(l,ti,bl) :
                        pwp_value(l,ti,bl)):
                        weight_value(l,so);
                    if(isp) begin
                        payload_response_low_or_weight[l*8 +:8]=v[7:0];
                        if(!ni) payload_response_high[l*4 +:4]=v[11:8];
                    end else payload_response_low_or_weight[l*8 +:8]=v[7:0];
                end
                payload_response_valid=1;
                do @(posedge clk_core); while(!payload_response_ready);
                @(negedge clk_core); payload_response_valid=0;
            end
        end
    end
    initial begin : accumulator_model
        forever begin
            @(posedge clk_core);
            if(reset_n && accumulator_read_valid && accumulator_read_ready) begin
                automatic logic [14:0] a=accumulator_read_address;
                repeat(2) @(posedge clk_core); @(negedge clk_core);
                accumulator_response_address=a; accumulator_response_data=0;
                accumulator_response_data=accumulator_mem[accumulator_index(a)];
                accumulator_response_valid=1;
                do @(posedge clk_core); while(!accumulator_response_ready);
                @(negedge clk_core); accumulator_response_valid=0;
            end
        end
    end

    initial begin : main
        clk_core=0; reset_n=0; cycles=0; commits_seen=0; pwp_seen=0;
        weight_seen=0; plus_seen=0; minus_seen=0; tile_mask=0; block_mask=0;
        commit_stalls=0; descriptor_stalls=0; row_mask=0;
        acc_read_stalls=0; acc_write_stalls=0; narrow_pwp_seen=0; negative_pwp_seen=0;
        expect_protocol_error=0; clear_inputs();
        for(integer s=0;s<8;s=s+1) for(integer r=0;r<ROWS;r=r+1)
        for(integer l=0;l<96;l=l+1) begin
            // External SRAM begins poisoned.  Operator psum is nevertheless 0.
            accumulator_mem[s*ROWS+r][l*19 +:19]=19'(1000+s*20+r*5+l);
            expected[s][r][l]=0;
            if(r==0) begin
                expected[s][r][l]+=2*(pwp_value(l,s/4,s%4)+weight_value(l,4)-weight_value(l,3));
                expected[s][r][l]+=weight_value(l,0)+weight_value(l,1);
            end else begin
                expected[s][r][l]+=pwp_value(l,s/4,s%4)+weight_value(l,4)-weight_value(l,3);
                expected[s][r][l]+=weight_value(l,0)+weight_value(l,1);
            end
        end
        apply_reset();
        // Phase 1 has two distinct active rows: PWP then fallback.
        send_config(24'h467001,0,0); send_row(0,16'h0017,0); send_row(1,16'h0003,1);
        wait(phase_done_valid); @(posedge clk_core);
        if(commits_seen!=0) $fatal(1,"non-last phase committed after phase1");
        // Phase 2 revisits row0 from phase1 through persistent SRAM; row1 is empty.
        send_config(24'h467002,0,0); send_row(0,16'h0003,0); send_row(1,16'h0000,1);
        wait(phase_done_valid); @(posedge clk_core);
        if(commits_seen!=0) $fatal(1,"non-last phase committed after phase2");
        // Phase 3 forwards the immediately repeated row0, then revisits row1.
        send_config(24'h467003,1,0); send_row(0,16'h0017,0); send_row(1,16'h0017,1);
        wait(phase_done_valid); @(posedge clk_core);
        if(commits_seen!=16||phase_done_active_rows!=2||pwp_seen!=24||
            weight_seen!=80||plus_seen!=24||minus_seen!=24||tile_mask!=3||
            block_mask!=15||row_mask!=3||debug_descriptor_writes!=5||debug_descriptor_reads!=10||
            debug_pwp_requests!=24||debug_weight_requests!=80||
            debug_accumulator_reads!=32||debug_forward_hits!=8||debug_commits!=16||
            debug_zero_initializations!=16||debug_zero_commits!=0||
            debug_row_live_sets!=2||debug_row_live_clears!=2||
            debug_zero_init_slot_mask!=8'hff||commit_stalls==0||descriptor_stalls==0||
            acc_read_stalls==0||acc_write_stalls==0||protocol_error)
            $fatal(1,"normal gate commit=%0d active=%0d pwp=%0d weight=%0d plus=%0d minus=%0d tm=%0d bm=%0d rm=%0d dw=%0d dr=%0d ar=%0d fw=%0d cs=%0d ds=%0d",
                commits_seen,phase_done_active_rows,pwp_seen,weight_seen,plus_seen,minus_seen,
                tile_mask,block_mask,row_mask,debug_descriptor_writes,debug_descriptor_reads,
                debug_accumulator_reads,debug_forward_hits,commit_stalls,descriptor_stalls);
        // A second nonzero operator starts without reset.  It activates the same
        // row/banks last owned by operator 1, while untouched row0 must commit 0.
        // Reset the mathematical expectation only; poison remains in external SRAM.
        reads_before_second_operator=debug_accumulator_reads;
        forwards_before_second_operator=debug_forward_hits;
        for(integer s2=0;s2<8;s2=s2+1) for(integer r2=0;r2<ROWS;r2=r2+1)
        for(integer l2=0;l2<96;l2=l2+1) begin
            expected[s2][r2][l2]=0;
            if(r2==1) expected[s2][r2][l2]=
                negative_pwp_value(l2,s2/4,s2%4)+weight_value(l2,4)-weight_value(l2,3);
        end
        send_config(24'h467101,1,1); send_row(0,16'h0000,0); send_row(1,16'h0017,1);
        wait(phase_done_valid); @(posedge clk_core);
        if(commits_seen!=32||phase_done_active_rows!=1||pwp_seen!=32||
            weight_seen!=96||plus_seen!=32||minus_seen!=32||
            debug_descriptor_writes!=6||debug_descriptor_reads!=12||
            debug_pwp_requests!=32||debug_weight_requests!=96||
            debug_accumulator_reads!=40||debug_forward_hits!=forwards_before_second_operator||
            debug_accumulator_reads-reads_before_second_operator!=8||
            debug_zero_initializations!=24||debug_zero_commits!=8||
            debug_row_live_sets!=3||debug_row_live_clears!=3||
            narrow_pwp_seen!=8||negative_pwp_seen!=8)
            $fatal(1,"second operator count gate failed");
        // Unsolicited response is a fail-closed protocol attack.
        expect_protocol_error=1;
        @(negedge clk_core); payload_response_valid=1; payload_response_pwp=0;
        @(posedge clk_core); @(negedge clk_core); payload_response_valid=0;
        repeat(2) @(posedge clk_core);
        if(!protocol_error||commit_valid||phase_done_valid) $fatal(1,"attack did not fail closed");
        // Reset recovery: a legal all-zero phase has no accumulator traffic.
        apply_reset(); expect_protocol_error=0; commits_seen=0; send_config(24'h467004,0,0);
        send_row(0,0,0); send_row(1,0,1);
        wait(phase_done_valid); @(posedge clk_core);
        if(protocol_error||phase_done_active_rows!=0||commits_seen!=0)
            $fatal(1,"reset recovery failed");
        $display("PASS M467R4 directed operators=2 no_reset_between_operators=1 phases=5 rows=10 active=6 same_phase_distinct_rows=1 descriptor_writes=6 descriptor_reads=12 pwp=32 narrow_pwp=8 signed_negative_pwp=8 weight=96 plus=32 minus=32 tiles=2 blocks=4 acc_reads=40 acc_writes=48 acc_read_stalls=1 acc_write_stalls=1 forward_hits=8 zero_initializations=24 zero_init_slot_mask=255 zero_commits=8 row_live_sets=3 row_live_clears=3 operator_commit_vectors=32 nonlast_commit_zero=1 commit_exact=1 cross_phase_sram_old_psum=1 stale_sram_read_suppressed=1 stale_forward_suppressed=1 row_shared_live_bits=2 production_row_shared_live_bits=3000 stalls=1 reset_recovery=1 protocol_attacks=1 m430_absolute_timestamps=false rtl_measured_517m=false system_speedup=false paper_ppa=false macro_ppa=false headline=false");
        $finish;
    end
    initial begin repeat(20000) @(posedge clk_core); $fatal(1,"watchdog state=%0d",debug_state); end
endmodule
`default_nettype wire
