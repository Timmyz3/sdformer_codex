`timescale 1ns/1ps
`default_nettype none

module m518_matched_fixed_t10_atlif_assertions #(
    parameter int TAG_W=48,
    parameter int FIFO_DEPTH=16,
    localparam int FIFO_COUNT_W=$clog2(FIFO_DEPTH+1),
    localparam int MULTIPLIERS=96
)(
    input logic clk_core,input logic rst_core,
    input logic config_valid,input logic config_ready,input logic config_accept,
    input logic[255:0]config_data,input logic config_last,
    input logic raw_valid,input logic raw_ready,input logic raw_accept,
    input logic[255:0]raw_data,input logic raw_last,input logic[TAG_W-1:0]raw_tag,
    input logic result_valid,input logic result_ready,input logic result_accept,
    input logic[TAG_W-1:0]result_tag,input logic[2:0]result_beat,
    input logic[47:0]result_valid_bits,input logic[47:0]result_data,
    input logic release_valid,input logic release_ready,input logic release_accept,
    input logic tile_done_valid,input logic[TAG_W-1:0]tile_done_tag,
    input logic context_retire_valid,input logic[31:0]context_retire_cycles,
    input logic config_loaded,input logic protocol_error,input logic busy,
    input logic stage1_issue,input logic stage2_issue,input logic product_push,
    input logic product_replace,input logic fifo_push,input logic fifo_pop,
    input logic[FIFO_COUNT_W-1:0]result_fifo_occupancy,
    input logic[1:0]raw_bank_occupancy,
    input logic[1:0]intermediate_bank_occupancy,
    input logic[31:0]debug_config_beats,input logic[31:0]debug_raw_beats,
    input logic[31:0]debug_tiles_loaded,input logic[31:0]debug_stage1_issues,
    input logic[31:0]debug_stage1_done,input logic[31:0]debug_stage2_issues,
    input logic[31:0]debug_stage2_done,input logic[31:0]debug_product_pushes,
    input logic[31:0]debug_result_departures,
    input logic[31:0]debug_product_replacements,
    input logic[31:0]debug_context_cycles,

    input logic fault_event,input logic fifo_credit_internal,
    input logic config_frame_error_internal,input logic raw_frame_error_internal,
    input logic zero_tile_release_error_internal,
    input logic dense_active_internal,input logic[4:0]dense_cycle_internal,
    input logic dense_selected_raw_bank_internal,
    input logic dense_raw_bank_internal,input logic[TAG_W-1:0]dense_tag_internal,
    input logic[2:0]dense_push_beat_internal,
    input logic overflow_safe_internal,
    input logic[1:0]raw_owned_internal,input logic[1:0]raw_ready_internal,
    input logic[31:0]raw_order0_internal,input logic[31:0]raw_order1_internal,
    input logic fill_active_internal,input logic fill_bank_internal,
    input logic[2:0]fill_beat_internal,input logic[TAG_W-1:0]fill_tag_internal,
    input logic[3:0]fifo_read_pointer_internal,
    input logic[3:0]fifo_write_pointer_internal,
    input logic[3999:0]acc_state_internal,
    input logic context_counting_internal,
    input logic[MULTIPLIERS-1:0]multiplier_active_mask,
    input logic[MULTIPLIERS-1:0]issue_tuple_valid,
    input logic[(MULTIPLIERS*4)-1:0]issue_tuple_row,
    input logic[(MULTIPLIERS*4)-1:0]issue_tuple_lane,
    input logic[(MULTIPLIERS*4)-1:0]issue_tuple_time
);
    ap_config_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        config_accept==(config_valid&&config_ready));
    ap_raw_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_accept==(raw_valid&&raw_ready));
    ap_result_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        result_accept==(result_valid&&result_ready));
    ap_release_accept:assert property(@(posedge clk_core)disable iff(rst_core)
        release_accept==(release_valid&&release_ready));
    ap_fifo_aliases:assert property(@(posedge clk_core)disable iff(rst_core)
        fifo_push==product_push&&fifo_pop==result_accept);
    ap_fixed_aliases:assert property(@(posedge clk_core)disable iff(rst_core)
        !stage2_issue&&!product_replace&&intermediate_bank_occupancy==0
            &&debug_stage2_issues==0&&debug_stage2_done==0
            &&debug_product_replacements==0);
    ap_config_accept_known:assert property(@(posedge clk_core)disable iff(rst_core)
        config_accept|->!$isunknown({config_valid,config_ready,config_last,
            config_data}));
    ap_raw_accept_known:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_accept|->!$isunknown({raw_valid,raw_ready,raw_last,raw_tag,raw_data}));
    ap_result_accept_known:assert property(@(posedge clk_core)disable iff(rst_core)
        result_accept|->!$isunknown({result_valid,result_ready,result_tag,
            result_beat,result_valid_bits,result_data}));
    ap_release_accept_known:assert property(@(posedge clk_core)disable iff(rst_core)
        release_accept|->!$isunknown({release_valid,release_ready}));

    ap_result_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        result_valid&&!result_ready|=>protocol_error||
        (result_valid&&$stable({result_tag,result_beat,result_valid_bits,result_data})));
    ap_result_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        result_valid|->result_beat<=4&&result_valid_bits==48'h0000ffffffff
            &&result_data[47:32]==0);
    ap_fault_registered:assert property(@(posedge clk_core)disable iff(rst_core)
        fault_event|=>protocol_error);
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    ap_fail_closed_after_fault:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->!(config_accept||raw_accept||result_valid
            ||release_accept||stage1_issue||product_push||fifo_pop));

    ap_release_requires_drain:assert property(@(posedge clk_core)disable iff(rst_core)
        release_ready|->config_loaded&&!busy&&result_fifo_occupancy==0
            &&raw_bank_occupancy==0&&debug_tiles_loaded>0&&!raw_valid);
    ap_busy_exact:assert property(@(posedge clk_core)disable iff(rst_core)
        busy==(fill_active_internal||raw_owned_internal!=0
            ||dense_active_internal||result_fifo_occupancy!=0));
    ap_zero_tile_release_never_accepts:assert property(
        @(posedge clk_core)disable iff(rst_core)
        config_loaded&&debug_tiles_loaded==0&&release_valid&&!raw_valid
            |->!release_ready&&!release_accept);
    ap_retire_minimum:assert property(@(posedge clk_core)disable iff(rst_core)
        context_retire_valid|->context_retire_cycles>=29);
    ap_context_cycle_progress:assert property(@(posedge clk_core)disable iff(rst_core)
        context_counting_internal&&!protocol_error&&!release_accept
            |=>debug_context_cycles==$past(debug_context_cycles)+1'b1);
    ap_context_retire_count:assert property(@(posedge clk_core)disable iff(rst_core)
        release_accept|=>context_retire_valid
            &&context_retire_cycles==$past(debug_context_cycles)+1'b1);

    ap_fifo_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        result_fifo_occupancy<=FIFO_DEPTH);
    ap_raw_bank_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_bank_occupancy<=2);
    ap_raw_occupancy_exact:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_bank_occupancy==$countones(raw_owned_internal));
    ap_ready_implies_owned:assert property(@(posedge clk_core)disable iff(rst_core)
        (raw_ready_internal&~raw_owned_internal)==0);
    ap_fill_implies_owned:assert property(@(posedge clk_core)disable iff(rst_core)
        fill_active_internal|->(fill_bank_internal?raw_owned_internal[1]:
            raw_owned_internal[0])&&fill_beat_internal>=1&&fill_beat_internal<=4);
    ap_dense_implies_owned:assert property(@(posedge clk_core)disable iff(rst_core)
        dense_active_internal|->(dense_raw_bank_internal?raw_owned_internal[1]:
            raw_owned_internal[0]));
    ap_oldest_bank1:assert property(@(posedge clk_core)disable iff(rst_core)
        !dense_active_internal&&raw_ready_internal==2'b11
            &&raw_order1_internal<raw_order0_internal&&stage1_issue
            |->dense_selected_raw_bank_internal);
    ap_oldest_bank0:assert property(@(posedge clk_core)disable iff(rst_core)
        !dense_active_internal&&raw_ready_internal==2'b11
            &&raw_order0_internal<raw_order1_internal&&stage1_issue
            |->!dense_selected_raw_bank_internal);
    ap_dense_start_ownership:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&!dense_active_internal|=>
            dense_active_internal&&dense_cycle_internal==1
                &&dense_raw_bank_internal==$past(dense_selected_raw_bank_internal)
                &&($past(dense_selected_raw_bank_internal)?
                    (raw_owned_internal[1]&&!raw_ready_internal[1]):
                    (raw_owned_internal[0]&&!raw_ready_internal[0])));
    ap_departure_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_result_departures<=debug_product_pushes);
    ap_fifo_conservation_exact:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_product_pushes-debug_result_departures==result_fifo_occupancy);
    ap_raw_beat_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_raw_beats<=5*debug_tiles_loaded+4);
    ap_dense_issue_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_stage1_issues>=17*debug_stage1_done
            &&debug_stage1_issues<=17*debug_stage1_done+16);
    ap_push_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_product_pushes>=5*debug_stage1_done
            &&debug_product_pushes<=5*debug_stage1_done+4);
    ap_tile_done_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        tile_done_valid|->debug_stage1_issues==17*debug_stage1_done
            &&debug_product_pushes==5*debug_stage1_done);

    ap_issue_phase_range:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue|->dense_cycle_internal<=16);
    ap_prologue_no_push:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal<12|->!fifo_push);
    ap_close_push:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal>=12|->fifo_push
            &&dense_push_beat_internal==dense_cycle_internal-12);
    ap_push_is_close:assert property(@(posedge clk_core)disable iff(rst_core)
        fifo_push|->stage1_issue&&dense_cycle_internal>=12
            &&dense_cycle_internal<=16);
    ap_close_requires_credit:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal>=12|->fifo_credit_internal);
    ap_close_stall_holds:assert property(@(posedge clk_core)disable iff(rst_core)
        dense_active_internal&&dense_cycle_internal>=12&&!fifo_credit_internal
            |=>$stable({dense_cycle_internal,
                dense_raw_bank_internal,dense_tag_internal,acc_state_internal,
                fifo_read_pointer_internal,fifo_write_pointer_internal,
                debug_stage1_issues,debug_product_pushes,
                debug_result_departures})
                &&(dense_raw_bank_internal?raw_owned_internal[1]:
                    raw_owned_internal[0]));
    ap_full_pop_push_atomic:assert property(@(posedge clk_core)disable iff(rst_core)
        result_fifo_occupancy==FIFO_DEPTH&&fifo_pop&&fifo_push
            |=>result_fifo_occupancy==FIFO_DEPTH
                &&fifo_read_pointer_internal==$past(fifo_read_pointer_internal)+1'b1
                &&fifo_write_pointer_internal==$past(fifo_write_pointer_internal)+1'b1);
    ap_phase_progress:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal<16
            |=>protocol_error||dense_cycle_internal==$past(dense_cycle_internal)+1'b1);
    ap_phase_close:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal==16
            |=>protocol_error||dense_cycle_internal==0);
    ap_accumulator_no_overflow:assert property(
        @(posedge clk_core)disable iff(rst_core)
        stage1_issue|->overflow_safe_internal);

    ap_mask_idle:assert property(@(posedge clk_core)disable iff(rst_core)
        !stage1_issue|->multiplier_active_mask==0&&issue_tuple_valid==0);
    ap_mask_active:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal<16
            |->$countones(multiplier_active_mask)==96
                &&issue_tuple_valid==multiplier_active_mask);
    ap_mask_tail:assert property(@(posedge clk_core)disable iff(rst_core)
        stage1_issue&&dense_cycle_internal==16
            |->$countones(multiplier_active_mask)==64
                &&issue_tuple_valid==multiplier_active_mask);
    ap_tile_done_exact:assert property(@(posedge clk_core)disable iff(rst_core)
        tile_done_valid|->$past(fifo_push&&dense_cycle_internal==16)
            &&tile_done_tag==$past(dense_tag_internal));

    generate
        for(genvar slot=0;slot<MULTIPLIERS;slot++)begin:tuple_bounds
            ap_tuple_bounds:assert property(@(posedge clk_core)disable iff(rst_core)
                issue_tuple_valid[slot]|->
                    issue_tuple_row[(slot*4)+:4]<=9
                    &&issue_tuple_lane[(slot*4)+:4]<=15
                    &&issue_tuple_time[(slot*4)+:4]<=9);
        end
    endgenerate

    cp_first_issue:cover property(@(posedge clk_core)
        stage1_issue&&dense_cycle_internal==0);
    cp_first_close:cover property(@(posedge clk_core)
        fifo_push&&dense_cycle_internal==12);
    cp_tail_close:cover property(@(posedge clk_core)
        fifo_push&&dense_cycle_internal==16);
    cp_close_stall:cover property(@(posedge clk_core)
        dense_active_internal&&dense_cycle_internal>=12&&!fifo_credit_internal);
    cp_phase12_stall:cover property(@(posedge clk_core)
        dense_active_internal&&dense_cycle_internal==12&&!fifo_credit_internal);
    cp_phase16_stall:cover property(@(posedge clk_core)
        dense_active_internal&&dense_cycle_internal==16&&!fifo_credit_internal);
    cp_result_stall:cover property(@(posedge clk_core)result_valid&&!result_ready);
    cp_fifo_full:cover property(@(posedge clk_core)
        result_fifo_occupancy==FIFO_DEPTH);
    cp_full_pop_push:cover property(@(posedge clk_core)
        result_fifo_occupancy==FIFO_DEPTH&&fifo_pop&&fifo_push);
    cp_raw_backpressure:cover property(@(posedge clk_core)raw_valid&&!raw_ready);
    cp_release_wait:cover property(@(posedge clk_core)release_valid&&!release_ready);
    cp_release:cover property(@(posedge clk_core)release_accept);
    cp_context_retire:cover property(@(posedge clk_core)context_retire_valid);
    cp_fault:cover property(@(posedge clk_core)$past(fault_event)&&protocol_error);
    cp_zero_tile_fault:cover property(@(posedge clk_core)
        $past(zero_tile_release_error_internal)&&protocol_error);
    cp_config_frame_fault:cover property(@(posedge clk_core)
        $past(config_frame_error_internal)&&protocol_error);
    cp_raw_frame_fault:cover property(@(posedge clk_core)
        $past(raw_frame_error_internal)&&protocol_error);
    cp_fault_with_pop_push:cover property(@(posedge clk_core)
        fault_event&&fifo_pop&&fifo_push);
    cp_dual_ready_oldest_bank1:cover property(@(posedge clk_core)
        !dense_active_internal&&raw_ready_internal==2'b11
            &&raw_order1_internal<raw_order0_internal&&stage1_issue
            &&dense_selected_raw_bank_internal);
    cp_beat0:cover property(@(posedge clk_core)result_accept&&result_beat==0);
    cp_beat1:cover property(@(posedge clk_core)result_accept&&result_beat==1);
    cp_beat2:cover property(@(posedge clk_core)result_accept&&result_beat==2);
    cp_beat3:cover property(@(posedge clk_core)result_accept&&result_beat==3);
    cp_beat4:cover property(@(posedge clk_core)result_accept&&result_beat==4);
    cp_reset_recovery:cover property(@(posedge clk_core)
        $past(rst_core)&&!rst_core&&!protocol_error&&!config_loaded
            &&result_fifo_occupancy==0&&raw_bank_occupancy==0&&!busy);
endmodule
`default_nettype wire
