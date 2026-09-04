`timescale 1ns/1ps
`default_nettype none

// One topology-colored accumulator bank. The packed OUT_DIM word uses one
// synchronous read port and one write port. Update and readback share the read
// port; clear and RMW writeback share the write port. A same-cycle RAW bypass
// makes the result independent of the SRAM macro's read-during-write mode.
module qfit_tcfm5_acc_bank #(
    parameter int DEPTH = 90,
    parameter int OUT_DIM = 4,
    parameter int ACC_W = 32,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int VEC_W = OUT_DIM * ACC_W
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   clear_valid,
    input  logic [ADDR_W-1:0]      clear_addr,
    input  logic                   update_valid,
    input  logic [ADDR_W-1:0]      update_addr,
    input  logic [VEC_W-1:0]       update_delta,
    output logic                   update_idle,
    input  logic                   read_valid,
    input  logic [ADDR_W-1:0]      read_addr,
    output logic                   read_data_valid,
    output logic [VEC_W-1:0]       read_data
);
    logic [VEC_W-1:0] memory_q [0:DEPTH-1];
    logic memory_read_enable;
    logic [ADDR_W-1:0] memory_read_addr;
    logic [VEC_W-1:0] memory_read_data_q;
    logic memory_write_enable;
    logic [ADDR_W-1:0] memory_write_addr;
    logic [VEC_W-1:0] memory_write_data;
    logic update_pipe_valid_q;
    logic [ADDR_W-1:0] update_pipe_addr_q;
    logic [VEC_W-1:0] update_delta_q;
    logic collision_forward_valid_q;
    logic [VEC_W-1:0] collision_forward_data_q;
    logic [VEC_W-1:0] update_base;
    logic [VEC_W-1:0] write_value;
    logic read_response_valid_q;

    function automatic logic [VEC_W-1:0] vector_add(
        input logic [VEC_W-1:0] lhs,
        input logic [VEC_W-1:0] rhs
    );
        logic [VEC_W-1:0] result;
        begin
            result = '0;
            for (integer out = 0; out < OUT_DIM; out = out + 1)
                result[out*ACC_W +: ACC_W] =
                    ACC_W'(
                        signed'(lhs[out*ACC_W +: ACC_W])
                        + signed'(rhs[out*ACC_W +: ACC_W])
                    );
            vector_add = result;
        end
    endfunction

    assign memory_read_enable = update_valid || read_valid;
    assign memory_read_addr = update_valid ? update_addr : read_addr;
    assign update_base = collision_forward_valid_q
        ? collision_forward_data_q
        : memory_read_data_q;
    assign write_value = vector_add(update_base, update_delta_q);
    assign memory_write_enable = clear_valid || update_pipe_valid_q;
    assign memory_write_addr = clear_valid
        ? clear_addr
        : update_pipe_addr_q;
    assign memory_write_data = clear_valid ? '0 : write_value;
    assign update_idle = !update_pipe_valid_q;
    assign read_data_valid = read_response_valid_q;
    assign read_data = memory_read_data_q;

    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            if (memory_write_enable)
                memory_q[memory_write_addr] <= memory_write_data;
            if (memory_read_enable)
                memory_read_data_q <= memory_q[memory_read_addr];
        end

        if (rst_core) begin
            update_pipe_valid_q <= 1'b0;
            update_pipe_addr_q <= '0;
            update_delta_q <= '0;
            collision_forward_valid_q <= 1'b0;
            collision_forward_data_q <= '0;
            read_response_valid_q <= 1'b0;
        end else begin
            read_response_valid_q <= read_valid;
            collision_forward_valid_q <=
                update_valid
                && update_pipe_valid_q
                && update_addr == update_pipe_addr_q;
            if (
                update_valid
                && update_pipe_valid_q
                && update_addr == update_pipe_addr_q
            )
                collision_forward_data_q <= write_value;
            if (update_valid) begin
                update_pipe_addr_q <= update_addr;
                update_delta_q <= update_delta;
            end
            update_pipe_valid_q <= update_valid;
        end
    end
endmodule

`default_nettype wire
