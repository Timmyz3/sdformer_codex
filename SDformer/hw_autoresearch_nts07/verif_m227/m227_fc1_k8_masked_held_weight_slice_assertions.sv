`timescale 1ns/1ps
`default_nettype none

module m227_fc1_k8_masked_held_weight_slice_assertions #(
    parameter int FANOUT = 1,
    parameter int TAG_BITS = 24,
    parameter int EPOCH_BITS = 16,
    parameter int LANES = 96,
    parameter int ACC_BITS = 19
) (
    input logic clk_core, input logic rst_core,
    input logic begin_valid, input logic begin_ready,
    input logic begin_accept,
    input logic scan_valid, input logic scan_ready,
    input logic scan_beat, input logic [255:0] scan_presence,
    input logic [255:0] scan_sign, input logic scan_accept,
    input logic scan_done_valid, input logic scan_done_ready,
    input logic scan_done_accept,
    input logic weight_req_valid, input logic weight_req_ready,
    input logic [TAG_BITS-1:0] weight_req_tag,
    input logic [EPOCH_BITS-1:0] weight_req_epoch,
    input logic [8:0] weight_req_source,
    input logic weight_req_accept,
    input logic weight_rsp_valid, input logic weight_rsp_ready,
    input logic weight_rsp_accept,
    input logic result_valid, input logic result_ready,
    input logic [TAG_BITS-1:0] result_tag,
    input logic [EPOCH_BITS-1:0] result_epoch,
    input logic [2:0] result_context,
    input logic [LANES*ACC_BITS-1:0] result_accumulator,
    input logic result_last, input logic result_accept,
    input logic done_valid, input logic done_ready,
    input logic done_accept,
    input logic protocol_error, input logic numeric_overflow,
    input logic [4:0] debug_scan_count,
    input logic [9:0] debug_unique_sources,
    input logic [11:0] debug_context_updates,
    input logic [9:0] debug_weight_reads,
    input logic [2:0] replay_width
);
    ap_begin_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) begin_accept == (begin_valid && begin_ready));
    ap_scan_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) scan_accept == (scan_valid && scan_ready));
    ap_scan_done_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) scan_done_accept
        == (scan_done_valid && scan_done_ready));
    ap_weight_req_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) weight_req_accept
        == (weight_req_valid && weight_req_ready));
    ap_weight_rsp_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) weight_rsp_accept
        == (weight_rsp_valid && weight_rsp_ready));
    ap_result_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) result_accept
        == (result_valid && result_ready));
    ap_done_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) done_accept == (done_valid && done_ready));

    ap_scan_count_bound: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_scan_count <= 16);
    ap_done_scan_complete: assert property (@(posedge clk_core)
        disable iff (rst_core) scan_done_accept |-> debug_scan_count == 16);
    ap_sign_subset: assert property (@(posedge clk_core)
        disable iff (rst_core) scan_accept |-> !(|(scan_sign & ~scan_presence)));
    ap_tail_zero: assert property (@(posedge clk_core)
        disable iff (rst_core) scan_accept && scan_beat
        |-> !(|scan_presence[255:128]) && !(|scan_sign[255:128]));
    ap_request_source_bound: assert property (@(posedge clk_core)
        disable iff (rst_core) weight_req_valid |-> weight_req_source < 384);
    ap_request_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) weight_req_valid && !weight_req_ready
        |=> protocol_error || (weight_req_valid
            && $stable({weight_req_tag,weight_req_epoch,weight_req_source})));
    ap_result_stable: assert property (@(posedge clk_core)
        disable iff (rst_core) result_valid && !result_ready
        |=> protocol_error || numeric_overflow || (result_valid
            && $stable({result_tag,result_epoch,result_context,
                        result_accumulator,result_last})));
    ap_result_last_identity: assert property (@(posedge clk_core)
        disable iff (rst_core) result_valid
        |-> result_last == (result_context == 7));
    ap_read_conservation: assert property (@(posedge clk_core)
        disable iff (rst_core) debug_weight_reads == debug_unique_sources);
    ap_replay_width: assert property (@(posedge clk_core)
        disable iff (rst_core) replay_width <= FANOUT);
    ap_fault_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) $past(protocol_error) |-> protocol_error);
    ap_overflow_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) $past(numeric_overflow) |-> numeric_overflow);

    cp_signed_scan: cover property (@(posedge clk_core)
        disable iff (rst_core) scan_accept && |scan_sign);
    cp_tail_source383: cover property (@(posedge clk_core)
        disable iff (rst_core) scan_accept && scan_beat
        && scan_presence[127]);
    cp_request_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) weight_req_valid && !weight_req_ready);
    cp_result_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) result_valid && !result_ready);
    cp_full_fanout: cover property (@(posedge clk_core)
        disable iff (rst_core) replay_width == FANOUT);
    cp_empty_group: cover property (@(posedge clk_core)
        disable iff (rst_core) result_accept && debug_unique_sources == 0);
    cp_protocol_attack: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);
    cp_done: cover property (@(posedge clk_core)
        disable iff (rst_core) done_accept);
endmodule

`default_nettype wire
