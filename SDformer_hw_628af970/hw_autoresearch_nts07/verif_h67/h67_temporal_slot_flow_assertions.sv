`timescale 1ns/1ps
`default_nettype none

module h67_temporal_slot_flow_assertions #(
    parameter int HEAD_DIM = 32,
    parameter int TOKEN_W = 9,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int FIFO_OCC_W = 6,
    parameter int SLOT_FIFO_DEPTH = 32
) (
    input logic                       clk_core,
    input logic                       rst_core,
    input logic                       window_start,
    input logic                       pair_ready,
    input logic                       pair_commit,
    input logic                       packet_valid,
    input logic                       packet_ready,
    input logic [1:0]                 packet_slot_count,
    input logic [15:0]                packet_slot0,
    input logic [15:0]                packet_slot1,
    input logic                       fifo_valid,
    input logic                       fifo_ready,
    input logic [1:0]                 k_read_req_valid,
    input logic [1:0]                 k_read_resp_valid,
    input logic                       directory_in_valid,
    input logic                       directory_in_ready,
    input logic [1:0]                 slot_temporal_mask,
    input logic [31:0]                perf_original_tokens,
    input logic                       out_valid,
    input logic                       out_ready,
    input logic                       out_last,
    input logic [TOKEN_W-1:0]         out_token_id,
    input logic [HEAD_DIM-1:0]        out_k_bits,
    input logic [GATE_W-1:0]          out_gate_q17,
    input logic [THRESHOLD_W-1:0]     out_threshold_q8,
    input logic [FIFO_OCC_W-1:0]      perf_fifo_occupancy,
    input logic                       protocol_error
);
    property p_output_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || window_start)
        out_valid && !out_ready |=> out_valid
            && $stable({out_last, out_token_id, out_k_bits,
                        out_gate_q17, out_threshold_q8});
    endproperty

    property p_last_requires_valid;
        @(posedge clk_core) disable iff (rst_core)
        out_last |-> out_valid;
    endproperty

    property p_emitted_k_is_active;
        @(posedge clk_core) disable iff (rst_core || window_start)
        out_valid |-> out_k_bits != 0;
    endproperty

    property p_fifo_capacity;
        @(posedge clk_core) disable iff (rst_core)
        32'(perf_fifo_occupancy) <= 32'(SLOT_FIFO_DEPTH);
    endproperty

    property p_start_blocks_external_handshakes;
        @(posedge clk_core) disable iff (rst_core)
        window_start |-> !pair_ready && !pair_commit && !out_valid;
    endproperty

    property p_packet_shape;
        @(posedge clk_core) disable iff (rst_core || window_start)
        packet_valid |->
            (packet_slot_count == 1
             && packet_slot0[12] && packet_slot0[9:8] == 2'b11)
            || (packet_slot_count == 2
                && !packet_slot0[12] && packet_slot0[9:8] == 2'b01
                && packet_slot1[12] && packet_slot1[9:8] == 2'b10);
    endproperty

    property p_packet_stable_under_backpressure;
        @(posedge clk_core) disable iff (rst_core || window_start)
        packet_valid && !packet_ready |=> packet_valid
            && $stable({packet_slot_count, packet_slot0, packet_slot1});
    endproperty

    property p_k_read_is_one_cycle;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) |-> k_read_resp_valid == $past(k_read_req_valid);
    endproperty

    property p_multiplicity_accumulates;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && $past(directory_in_valid && directory_in_ready)
        |-> perf_original_tokens == $past(perf_original_tokens)
            + 32'($past(slot_temporal_mask[0]))
            + 32'($past(slot_temporal_mask[1]));
    endproperty

    property p_multiplicity_holds_without_input;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start) && !$past(directory_in_valid && directory_in_ready)
        |-> perf_original_tokens == $past(perf_original_tokens);
    endproperty

    property p_fifo_count_enq_only;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(packet_valid && packet_ready)
        && !$past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) == 32'($past(perf_fifo_occupancy))
            + 32'($past(packet_slot_count));
    endproperty

    property p_fifo_count_deq_only;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && !$past(packet_valid && packet_ready)
        && $past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) + 1 == 32'($past(perf_fifo_occupancy));
    endproperty

    property p_fifo_count_both;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && $past(packet_valid && packet_ready)
        && $past(fifo_valid && fifo_ready)
        |-> 32'(perf_fifo_occupancy) + 1 == 32'($past(perf_fifo_occupancy))
            + 32'($past(packet_slot_count));
    endproperty

    property p_fifo_count_idle;
        @(posedge clk_core) disable iff (rst_core || window_start)
        !$past(window_start)
        && !$past(packet_valid && packet_ready)
        && !$past(fifo_valid && fifo_ready)
        |-> perf_fifo_occupancy == $past(perf_fifo_occupancy);
    endproperty

    assert property (p_output_stable_under_backpressure)
        else $fatal(1, "slot flow output changed under backpressure");
    assert property (p_last_requires_valid)
        else $fatal(1, "slot flow last without valid");
    assert property (p_emitted_k_is_active)
        else $fatal(1, "slot flow emitted zero K");
    assert property (p_fifo_capacity)
        else $fatal(1, "slot FIFO occupancy exceeded capacity");
    assert property (p_start_blocks_external_handshakes)
        else $fatal(1, "window_start overlapped an external handshake");
    assert property (p_packet_shape)
        else $fatal(1, "slot packet shape is not atomic common/split form");
    assert property (p_packet_stable_under_backpressure)
        else $fatal(1, "slot packet changed under backpressure");
    assert property (p_k_read_is_one_cycle)
        else $fatal(1, "synchronous K response did not match prior request mask");
    assert property (p_multiplicity_accumulates)
        else $fatal(1, "weighted-SCS multiplicity increment mismatch");
    assert property (p_multiplicity_holds_without_input)
        else $fatal(1, "weighted-SCS multiplicity changed without input");
    assert property (p_fifo_count_enq_only)
        else $fatal(1, "slot FIFO enqueue-only conservation mismatch");
    assert property (p_fifo_count_deq_only)
        else $fatal(1, "slot FIFO dequeue-only conservation mismatch");
    assert property (p_fifo_count_both)
        else $fatal(1, "slot FIFO simultaneous conservation mismatch");
    assert property (p_fifo_count_idle)
        else $fatal(1, "slot FIFO occupancy changed while idle");

    cover property (@(posedge clk_core) disable iff (rst_core)
        out_valid && !out_ready |=> out_valid && out_ready);
    cover property (@(posedge clk_core) disable iff (rst_core)
        perf_fifo_occupancy == FIFO_OCC_W'(SLOT_FIFO_DEPTH));
    cover property (@(posedge clk_core) disable iff (rst_core)
        protocol_error);
endmodule

bind h67_temporal_slot_shiftmax_sync_k_top
    h67_temporal_slot_flow_assertions #(
        .HEAD_DIM(HEAD_DIM),
        .TOKEN_W(TOKEN_W),
        .GATE_W(GATE_W),
        .THRESHOLD_W(THRESHOLD_W),
        .FIFO_OCC_W(FIFO_OCC_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH)
    ) u_h67_temporal_slot_flow_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .pair_ready(pair_ready),
        .pair_commit(pair_commit),
        .packet_valid(packet_valid),
        .packet_ready(packet_ready),
        .packet_slot_count(packet_slot_count),
        .packet_slot0(packet_slot0),
        .packet_slot1(packet_slot1),
        .fifo_valid(fifo_valid),
        .fifo_ready(fifo_ready),
        .k_read_req_valid(k_read_req_valid),
        .k_read_resp_valid(k_read_resp_valid),
        .directory_in_valid(directory_in_valid),
        .directory_in_ready(directory_in_ready),
        .slot_temporal_mask(slot_temporal_mask),
        .perf_original_tokens(perf_original_tokens),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_id(out_token_id),
        .out_k_bits(out_k_bits),
        .out_gate_q17(out_gate_q17),
        .out_threshold_q8(out_threshold_q8),
        .perf_fifo_occupancy(perf_fifo_occupancy),
        .protocol_error(protocol_error)
    );

`default_nettype wire
