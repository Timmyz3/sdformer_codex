`timescale 1ns/1ps
`default_nettype none

// Relation memory with logical write/read channels and fixed synchronous read
// latency. The architecture makes the two phases mutually exclusive, allowing
// the implementation to bind this contract to a single-port 1RW SRAM macro.
module qfit_sync_relation_bank #(
    parameter int DEPTH = 450,
    parameter int DATA_W = 10,
    parameter int READ_LATENCY = 1,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   write_valid,
    input  logic [ADDR_W-1:0]      write_addr,
    input  logic [DATA_W-1:0]      write_data,
    input  logic                   read_valid,
    input  logic [ADDR_W-1:0]      read_addr,
    output logic                   read_data_valid,
    output logic [DATA_W-1:0]      read_data
);
    logic [DATA_W-1:0] memory [0:DEPTH-1];
    logic [READ_LATENCY-1:0] read_valid_pipe_q;
    logic [ADDR_W-1:0] read_addr_pipe_q [0:READ_LATENCY-1];

    initial begin
        if (DEPTH <= 0 || DATA_W <= 0)
            $error("qfit_sync_relation_bank parameters must be positive");
        if (READ_LATENCY < 1 || READ_LATENCY > 2)
            $error("qfit_sync_relation_bank supports READ_LATENCY=1 or 2");
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            read_valid_pipe_q <= '0;
            read_data_valid <= 1'b0;
            read_data <= '0;
            for (integer stage = 0; stage < READ_LATENCY; stage = stage + 1)
                read_addr_pipe_q[stage] <= '0;
        end else begin
            if (write_valid)
                memory[write_addr] <= write_data;
            read_valid_pipe_q[0] <= read_valid;
            read_addr_pipe_q[0] <= read_addr;
            for (integer stage = 1; stage < READ_LATENCY; stage = stage + 1) begin
                read_valid_pipe_q[stage] <= read_valid_pipe_q[stage-1];
                read_addr_pipe_q[stage] <= read_addr_pipe_q[stage-1];
            end
            read_data_valid <= read_valid_pipe_q[READ_LATENCY-1];
            if (read_valid_pipe_q[READ_LATENCY-1])
                read_data <= memory[read_addr_pipe_q[READ_LATENCY-1]];
        end
    end
endmodule

`default_nettype wire
