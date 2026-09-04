`timescale 1ns/1ps
`default_nettype none

// Physical-friendly lane striping for the M111 2304-bit logical vector port.
//
// Instead of requiring an implausibly wide monolithic SRAM, the accumulator is
// organized as 96 independent 3072x24 signed-lane banks.  Every update reads
// and writes one word in all lane banks at the same flattened (block,row)
// address, retaining vector II=1 while exposing conventional narrow macros.
module m112_w384_lane_sliced_accumulator_adapter #(
    parameter int WIN_ROWS = 384,
    parameter int BLOCKS = 8,
    parameter int LANES = 96,
    parameter int ACC_BITS = 24,
    parameter int VECTOR_BITS = LANES * ACC_BITS,
    parameter int DEPTH = BLOCKS * WIN_ROWS,
    parameter int ADDR_W = 12
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         window_start_valid,
    output logic                         window_start_ready,
    output logic                         window_start_accept,
    input  logic                         update_valid,
    output logic                         update_ready,
    input  logic [2:0]                   update_block,
    input  logic [8:0]                   update_row,
    input  logic [VECTOR_BITS-1:0]       update_delta,
    output logic                         update_accept,
    input  logic                         window_end_valid,
    output logic                         window_end_ready,
    output logic                         window_end_accept,
    output logic                         commit_valid,
    input  logic                         commit_ready,
    output logic [2:0]                   commit_block,
    output logic [8:0]                   commit_row,
    output logic [VECTOR_BITS-1:0]       commit_data,
    output logic                         commit_last,
    output logic                         window_done,

    output logic                         lane_mem_rd_en,
    output logic [ADDR_W-1:0]            lane_mem_rd_addr,
    input  logic [ACC_BITS-1:0]          lane_mem_rd_data [0:LANES-1],
    output logic                         lane_mem_wr_en,
    output logic [ADDR_W-1:0]            lane_mem_wr_addr,
    output logic [ACC_BITS-1:0]          lane_mem_wr_data [0:LANES-1],

    output logic                         protocol_error,
    output logic                         window_active,
    output logic                         busy
);
    logic [BLOCKS-1:0] core_mem_rd_en;
    logic [8:0] core_mem_rd_addr [0:BLOCKS-1];
    logic [VECTOR_BITS-1:0] core_mem_rd_data [0:BLOCKS-1];
    logic [BLOCKS-1:0] core_mem_wr_en;
    logic [8:0] core_mem_wr_addr [0:BLOCKS-1];
    logic [VECTOR_BITS-1:0] core_mem_wr_data [0:BLOCKS-1];
    logic [VECTOR_BITS-1:0] assembled_read_vector;
    logic [2:0] selected_read_block;
    logic [2:0] selected_write_block;

`ifndef SYNTHESIS
    initial begin
        if (WIN_ROWS != 384 || BLOCKS != 8 || LANES != 96
                || ACC_BITS != 24 || VECTOR_BITS != 2304
                || DEPTH != 3072 || ADDR_W != 12)
            $fatal(1, "M112 production lane-sliced geometry drift");
    end
`endif

    function automatic logic [ADDR_W-1:0] flatten_address(
        input logic [2:0] block,
        input logic [8:0] row
    );
        logic [ADDR_W-1:0] block_times_384;
        begin
            block_times_384 = ({9'b0, block} << 8)
                              + ({9'b0, block} << 7);
            flatten_address = block_times_384 + row;
        end
    endfunction

    always_comb begin : lane_macro_map
        assembled_read_vector = '0;
        for (int lane = 0; lane < LANES; lane++)
            assembled_read_vector[lane * ACC_BITS +: ACC_BITS]
                = lane_mem_rd_data[lane];
        for (int block = 0; block < BLOCKS; block++)
            core_mem_rd_data[block] = assembled_read_vector;

        selected_read_block = '0;
        selected_write_block = '0;
        for (int block = 0; block < BLOCKS; block++) begin
            if (core_mem_rd_en[block])
                selected_read_block = block[2:0];
            if (core_mem_wr_en[block])
                selected_write_block = block[2:0];
        end
        lane_mem_rd_en = |core_mem_rd_en;
        lane_mem_rd_addr = flatten_address(
            selected_read_block, core_mem_rd_addr[selected_read_block]);
        lane_mem_wr_en = |core_mem_wr_en;
        lane_mem_wr_addr = flatten_address(
            selected_write_block, core_mem_wr_addr[selected_write_block]);
        for (int lane = 0; lane < LANES; lane++)
            lane_mem_wr_data[lane]
                = core_mem_wr_data[selected_write_block]
                                  [lane * ACC_BITS +: ACC_BITS];
    end

    m111_w384_signed24_accumulator_frontend core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start_valid(window_start_valid),
        .window_start_ready(window_start_ready),
        .window_start_accept(window_start_accept),
        .update_valid(update_valid),
        .update_ready(update_ready),
        .update_block(update_block),
        .update_row(update_row),
        .update_delta(update_delta),
        .update_accept(update_accept),
        .window_end_valid(window_end_valid),
        .window_end_ready(window_end_ready),
        .window_end_accept(window_end_accept),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_block(commit_block),
        .commit_row(commit_row),
        .commit_data(commit_data),
        .commit_last(commit_last),
        .window_done(window_done),
        .mem_rd_en(core_mem_rd_en),
        .mem_rd_addr(core_mem_rd_addr),
        .mem_rd_data(core_mem_rd_data),
        .mem_wr_en(core_mem_wr_en),
        .mem_wr_addr(core_mem_wr_addr),
        .mem_wr_data(core_mem_wr_data),
        .protocol_error(protocol_error),
        .window_active(window_active),
        .busy(busy)
    );
endmodule

`default_nettype wire
