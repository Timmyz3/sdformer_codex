`timescale 1ns/1ps
`default_nettype none

module h67_laws_shared_backend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic window_start,
    input logic row_k_present,
    input logic last_row_done,
    input logic out_valid,
    input logic [1:0] retire_count,
    input logic can_skip_retire,
    input logic can_emit_start,
    input logic [1:0] skip_ws,
    input logic [0:0] retire_head,
    input logic emit_active
);
    property p_retire_count_bound;
        @(posedge clk_core) disable iff (rst_core)
        retire_count <= 2'd2;
    endproperty
    assert property (p_retire_count_bound);

    property p_skip_emit_mutex;
        @(posedge clk_core) disable iff (rst_core)
        !(can_skip_retire && can_emit_start);
    endproperty
    assert property (p_skip_emit_mutex);

    property p_skip_no_tokens;
        @(posedge clk_core) disable iff (rst_core)
        can_skip_retire |-> !out_valid;
    endproperty
    assert property (p_skip_no_tokens);

    property p_done_had_inflight;
        @(posedge clk_core) disable iff (rst_core)
        last_row_done |-> $past(retire_count) != 2'd0;
    endproperty
    assert property (p_done_had_inflight);

    property p_skip_head_marked;
        @(posedge clk_core) disable iff (rst_core)
        can_skip_retire |-> skip_ws[retire_head];
    endproperty
    assert property (p_skip_head_marked);

    property p_emit_not_skip_head;
        @(posedge clk_core) disable iff (rst_core)
        can_emit_start |-> !skip_ws[retire_head];
    endproperty
    assert property (p_emit_not_skip_head);
endmodule

bind h67_laws_shared_backend_2s_top h67_laws_shared_backend_assertions
    u_h67_laws_shared_backend_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .row_k_present(row_k_present),
        .last_row_done(last_row_done),
        .out_valid(out_valid),
        .retire_count(retire_count_q),
        .can_skip_retire(can_skip_retire),
        .can_emit_start(can_emit_start),
        .skip_ws(skip_ws_q),
        .retire_head(retire_head),
        .emit_active(emit_active)
    );

`default_nettype wire
