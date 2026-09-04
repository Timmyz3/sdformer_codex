`timescale 1ns/1ps
`default_nettype none
module m273_integrated_rank3_atlif_assertions #(
    parameter int TAG_W=48,
    parameter int FIFO_DEPTH=16,
    localparam int FIFO_COUNT_W=$clog2(FIFO_DEPTH+1)
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
    input logic[31:0]debug_context_cycles
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
    ap_product_replace:assert property(@(posedge clk_core)disable iff(rst_core)
        product_replace==(product_push&&stage2_issue));

    ap_result_stable:assert property(@(posedge clk_core)disable iff(rst_core)
        result_valid&&!result_ready|=>protocol_error||
        (result_valid&&$stable({result_tag,result_beat,result_valid_bits,result_data})));
    ap_result_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        result_valid|->result_beat<=4&&result_valid_bits==48'h0000ffffffff);
    ap_tile_done_shape:assert property(@(posedge clk_core)disable iff(rst_core)
        tile_done_valid|->$past(product_push));
    ap_fault_sticky:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->protocol_error);
    ap_fail_closed_after_fault:assert property(@(posedge clk_core)disable iff(rst_core)
        $past(protocol_error)|->!(config_accept||raw_accept||result_valid||release_accept
            ||stage1_issue||stage2_issue||product_push||fifo_pop));
    ap_release_requires_drain:assert property(@(posedge clk_core)disable iff(rst_core)
        release_ready|->config_loaded&&!busy&&result_fifo_occupancy==0
            &&raw_bank_occupancy==0&&intermediate_bank_occupancy==0
            &&debug_tiles_loaded>0);
    ap_zero_tile_release_never_accepts:assert property(
        @(posedge clk_core)disable iff(rst_core)
        config_loaded&&debug_tiles_loaded==0&&release_valid
            |->!release_ready&&!release_accept);
    ap_zero_tile_release_faults_registered:assert property(
        @(posedge clk_core)disable iff(rst_core)
        config_loaded&&debug_tiles_loaded==0&&release_valid&&!busy
            |=>protocol_error);
    ap_retire_minimum:assert property(@(posedge clk_core)disable iff(rst_core)
        context_retire_valid|->context_retire_cycles>=24);
    ap_fifo_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        result_fifo_occupancy<=FIFO_DEPTH);
    ap_raw_bank_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        raw_bank_occupancy<=2);
    ap_intermediate_bank_bound:assert property(@(posedge clk_core)disable iff(rst_core)
        intermediate_bank_occupancy<=2);
    ap_departure_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_result_departures<=debug_product_pushes);
    ap_stage1_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_stage1_done<=debug_tiles_loaded&&debug_stage1_issues>=5*debug_stage1_done);
    ap_stage2_conservation:assert property(@(posedge clk_core)disable iff(rst_core)
        debug_stage2_done<=debug_stage1_done&&debug_stage2_issues>=5*debug_stage2_done);

    cp_clean_overlap:cover property(@(posedge clk_core)stage1_issue&&stage2_issue);
    cp_product_replace:cover property(@(posedge clk_core)product_replace);
    cp_result_stall:cover property(@(posedge clk_core)result_valid&&!result_ready);
    cp_fifo_full:cover property(@(posedge clk_core)
        result_fifo_occupancy==FIFO_DEPTH);
    cp_full_pop_push:cover property(@(posedge clk_core)
        result_fifo_occupancy==FIFO_DEPTH&&fifo_pop&&fifo_push);
    cp_raw_backpressure:cover property(@(posedge clk_core)raw_valid&&!raw_ready);
    cp_release_wait:cover property(@(posedge clk_core)release_valid&&!release_ready);
    cp_release:cover property(@(posedge clk_core)release_accept);
    cp_context_retire:cover property(@(posedge clk_core)context_retire_valid);
    cp_config_fault:cover property(@(posedge clk_core)
        $past(config_accept)&&protocol_error);
    cp_raw_fault:cover property(@(posedge clk_core)
        $past(raw_accept)&&protocol_error);
    cp_zero_tile_release_fault:cover property(@(posedge clk_core)
        $past(config_loaded&&debug_tiles_loaded==0&&release_valid&&!busy)
            &&protocol_error);
    cp_beat4:cover property(@(posedge clk_core)result_accept&&result_beat==4);
endmodule
`default_nettype wire
