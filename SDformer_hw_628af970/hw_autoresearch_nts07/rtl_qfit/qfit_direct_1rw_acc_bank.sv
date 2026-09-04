`timescale 1ns/1ps
`default_nettype none

// Fair single-port baseline. First touch writes delta directly; later updates
// use an exact read-modify-write sequence on the same 1RW memory contract.
module qfit_direct_1rw_acc_bank #(
    parameter int DEPTH = 90,
    parameter int OUT_DIM = 2,
    parameter int ACC_W = 32,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int VEC_W = OUT_DIM * ACC_W,
    parameter int MEMORY_IMPL = 0
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   run_start,
    input  logic                   run_accumulate,
    input  logic                   update_valid,
    output logic                   update_ready,
    input  logic [ADDR_W-1:0]      update_addr,
    input  logic [VEC_W-1:0]       update_delta,
    input  logic                   flush_valid,
    output logic                   flush_ready,
    output logic                   flush_done,
    input  logic                   read_valid,
    output logic                   read_ready,
    input  logic [ADDR_W-1:0]      read_addr,
    output logic                   read_data_valid,
    output logic [VEC_W-1:0]       read_data,
    output logic                   protocol_error,
    output logic [31:0]            perf_updates,
    output logic [31:0]            perf_sram_reads,
    output logic [31:0]            perf_sram_writes
);
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_RMW_WAIT = 2'd1,
        ST_READBACK_WAIT = 2'd2
    } state_t;

    state_t state_q;
    logic [DEPTH-1:0] backing_valid_q;
    logic [ADDR_W-1:0] pending_addr_q;
    logic [VEC_W-1:0] pending_delta_q;
    logic window_flushed_q;
    logic protocol_error_q;
    logic [31:0] updates_q;
    logic [31:0] reads_q;
    logic [31:0] writes_q;

    logic memory_command_valid;
    logic memory_command_write;
    logic [ADDR_W-1:0] memory_command_addr;
    logic [VEC_W-1:0] memory_command_write_data;
    logic memory_read_data_valid;
    logic [VEC_W-1:0] memory_read_data;
    logic update_fire;
    logic read_fire;
    logic [VEC_W-1:0] rmw_sum;
    logic update_addr_valid;
    logic read_addr_valid;

    function automatic logic [VEC_W-1:0] vector_add(
        input logic [VEC_W-1:0] lhs,
        input logic [VEC_W-1:0] rhs
    );
        logic [VEC_W-1:0] result;
        begin
            result = '0;
            for (integer out = 0; out < OUT_DIM; out++)
                result[out*ACC_W +: ACC_W] = ACC_W'(
                    signed'(lhs[out*ACC_W +: ACC_W])
                    + signed'(rhs[out*ACC_W +: ACC_W])
                );
            vector_add = result;
        end
    endfunction

    assign update_addr_valid = 32'(update_addr) < DEPTH;
    assign read_addr_valid = 32'(read_addr) < DEPTH;
    assign update_ready = state_q == ST_IDLE && !window_flushed_q
                        && !flush_valid && update_addr_valid;
    assign update_fire = update_valid && update_ready;
    assign flush_ready = state_q == ST_IDLE && !update_valid;
    assign read_ready = state_q == ST_IDLE && window_flushed_q
                     && !flush_valid && read_addr_valid;
    assign read_fire = read_valid && read_ready;
    assign protocol_error = protocol_error_q;
    assign perf_updates = updates_q;
    assign perf_sram_reads = reads_q;
    assign perf_sram_writes = writes_q;
    assign rmw_sum = vector_add(memory_read_data, pending_delta_q);

    always_comb begin
        memory_command_valid = 1'b0;
        memory_command_write = 1'b0;
        memory_command_addr = '0;
        memory_command_write_data = '0;
        if (state_q == ST_IDLE && update_fire) begin
            memory_command_valid = 1'b1;
            memory_command_addr = update_addr;
            memory_command_write = !backing_valid_q[update_addr];
            memory_command_write_data = update_delta;
        end else if (state_q == ST_RMW_WAIT && memory_read_data_valid) begin
            memory_command_valid = 1'b1;
            memory_command_write = 1'b1;
            memory_command_addr = pending_addr_q;
            memory_command_write_data = rmw_sum;
        end else if (state_q == ST_IDLE && read_fire && backing_valid_q[read_addr]) begin
            memory_command_valid = 1'b1;
            memory_command_addr = read_addr;
        end
    end

    qfit_single_port_acc_memory #(
        .DEPTH(DEPTH), .VEC_W(VEC_W), .ADDR_W(ADDR_W),
        .MEMORY_IMPL(MEMORY_IMPL)
    ) u_memory (
        .clk_core(clk_core), .rst_core(rst_core),
        .command_valid(memory_command_valid),
        .command_write(memory_command_write),
        .command_addr(memory_command_addr),
        .command_write_data(memory_command_write_data),
        .read_data_valid(memory_read_data_valid),
        .read_data(memory_read_data)
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            backing_valid_q <= '0;
            pending_addr_q <= '0;
            pending_delta_q <= '0;
            window_flushed_q <= 1'b0;
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            read_data <= '0;
            protocol_error_q <= 1'b0;
            updates_q <= '0;
            reads_q <= '0;
            writes_q <= '0;
        end else if (run_start) begin
            state_q <= ST_IDLE;
            if (!run_accumulate)
                backing_valid_q <= '0;
            window_flushed_q <= 1'b0;
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            protocol_error_q <= 1'b0;
            updates_q <= '0;
            reads_q <= '0;
            writes_q <= '0;
        end else begin
            flush_done <= 1'b0;
            read_data_valid <= 1'b0;
            if (update_valid && !update_addr_valid)
                protocol_error_q <= 1'b1;
            if (read_valid && !read_addr_valid)
                protocol_error_q <= 1'b1;

            if (state_q == ST_IDLE && flush_valid && flush_ready) begin
                window_flushed_q <= 1'b1;
                flush_done <= 1'b1;
            end else if (update_fire) begin
                updates_q <= updates_q + 1'b1;
                if (backing_valid_q[update_addr]) begin
                    pending_addr_q <= update_addr;
                    pending_delta_q <= update_delta;
                    reads_q <= reads_q + 1'b1;
                    state_q <= ST_RMW_WAIT;
                end else begin
                    backing_valid_q[update_addr] <= 1'b1;
                    writes_q <= writes_q + 1'b1;
                end
            end else if (state_q == ST_RMW_WAIT && memory_read_data_valid) begin
                writes_q <= writes_q + 1'b1;
                state_q <= ST_IDLE;
            end else if (state_q == ST_IDLE && read_fire) begin
                if (backing_valid_q[read_addr]) begin
                    pending_addr_q <= read_addr;
                    reads_q <= reads_q + 1'b1;
                    state_q <= ST_READBACK_WAIT;
                end else begin
                    read_data <= '0;
                    read_data_valid <= 1'b1;
                end
            end else if (
                state_q == ST_READBACK_WAIT && memory_read_data_valid
            ) begin
                read_data <= memory_read_data;
                read_data_valid <= 1'b1;
                state_q <= ST_IDLE;
            end
        end
    end
endmodule

`default_nettype wire
