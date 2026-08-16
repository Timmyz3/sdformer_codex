`timescale 1ns/1ps
`default_nettype none

// 双slot公平基线FIFO：入队仍以pair packet原子提交，出队每拍最多两条slot。
module h67_temporal_slot_fifo_2s #(
    parameter int DEPTH = 32,
    parameter int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int OCC_W = $clog2(DEPTH + 1)
) (
    input  logic                 clk_core,
    input  logic                 rst_core,
    input  logic                 window_start,

    input  logic                 enq_valid,
    output logic                 enq_ready,
    input  logic [1:0]           enq_count,
    input  logic [15:0]          enq_slot0,
    input  logic [15:0]          enq_slot1,

    output logic                 deq_valid,
    input  logic                 deq_ready,
    output logic [1:0]           deq_count,
    output logic [15:0]          deq_slot0,
    output logic [15:0]          deq_slot1,

    output logic [OCC_W-1:0]     occupancy,
    output logic [OCC_W-1:0]     max_occupancy,
    output logic                 protocol_error
);
    logic [15:0] slot_mem [0:DEPTH-1];
    logic [PTR_W-1:0] write_ptr_q;
    logic [PTR_W-1:0] read_ptr_q;
    logic [OCC_W-1:0] count_q;
    logic [OCC_W-1:0] max_count_q;
    logic protocol_error_q;
    logic enq_fire;
    logic deq_fire;
    logic [OCC_W:0] available_after_deq;
    logic [OCC_W:0] next_count_w;

    function automatic logic [PTR_W-1:0] ptr_add(
        input logic [PTR_W-1:0] ptr,
        input int unsigned amount
    );
        int unsigned value;
        begin
            value = 32'(ptr) + amount;
            if (value >= DEPTH)
                value = value - DEPTH;
            ptr_add = PTR_W'(value);
        end
    endfunction

    assign deq_valid = !window_start && count_q != 0;
    assign deq_count = count_q >= OCC_W'(2) ? 2 : 1;
    assign deq_slot0 = slot_mem[read_ptr_q];
    assign deq_slot1 = slot_mem[ptr_add(read_ptr_q, 1)];
    assign deq_fire = deq_valid && deq_ready;

    assign available_after_deq = {1'b0, count_q}
        - (deq_fire ? (deq_count == 2 ? 2 : 1) : 0);
    assign enq_ready = !window_start
                    && (enq_count == 1 || enq_count == 2)
                    && 32'(available_after_deq) + 32'(enq_count)
                       <= 32'(DEPTH);
    assign enq_fire = enq_valid && enq_ready;
    assign occupancy = count_q;
    assign max_occupancy = max_count_q;
    assign protocol_error = protocol_error_q;

    always_comb begin
        next_count_w = {1'b0, count_q};
        if (enq_fire)
            next_count_w = next_count_w + (enq_count == 2 ? 2 : 1);
        if (deq_fire)
            next_count_w = next_count_w - (deq_count == 2 ? 2 : 1);
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            write_ptr_q <= '0;
            read_ptr_q <= '0;
            count_q <= '0;
            max_count_q <= '0;
            protocol_error_q <= 1'b0;
        end else if (window_start) begin
            protocol_error_q <= count_q != 0;
            write_ptr_q <= '0;
            read_ptr_q <= '0;
            count_q <= '0;
            max_count_q <= '0;
        end else begin
            if (enq_valid && enq_count != 1 && enq_count != 2)
                protocol_error_q <= 1'b1;
            if (enq_fire) begin
                slot_mem[write_ptr_q] <= enq_slot0;
                if (enq_count == 2)
                    slot_mem[ptr_add(write_ptr_q, 1)] <= enq_slot1;
                write_ptr_q <= ptr_add(write_ptr_q, enq_count == 2 ? 2 : 1);
            end
            if (deq_fire)
                read_ptr_q <= ptr_add(read_ptr_q, deq_count == 2 ? 2 : 1);
            count_q <= OCC_W'(next_count_w);
            if (next_count_w > {1'b0, max_count_q})
                max_count_q <= OCC_W'(next_count_w);
        end
    end
endmodule

`default_nettype wire
