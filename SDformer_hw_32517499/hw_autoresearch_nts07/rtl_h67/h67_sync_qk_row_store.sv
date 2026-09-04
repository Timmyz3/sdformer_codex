`timescale 1ns/1ps
`default_nettype none

// Q/K行驻留存储。score和gated-K发射分时复用同一同步读口。
module h67_sync_qk_row_store #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int ADDR_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS)
) (
    input  logic                    clk_core,
    input  logic                    rst_core,
    input  logic                    row_reset,

    input  logic                    write_valid,
    output logic                    write_ready,
    input  logic [ADDR_W-1:0]       write_addr,
    input  logic [2*HEAD_DIM-1:0]   write_q_pair,
    input  logic [2*HEAD_DIM-1:0]   write_k_pair,

    input  logic                    read_req_valid,
    output logic                    read_req_ready,
    input  logic [ADDR_W-1:0]       read_req_addr,
    input  logic                    read_req_q,
    input  logic [1:0]              read_req_k_mask,
    input  logic                    read_req_score_tag,

    output logic                    read_resp_valid,
    input  logic                    read_resp_ready,
    output logic [ADDR_W-1:0]       read_resp_addr,
    output logic [2*HEAD_DIM-1:0]   read_resp_q_pair,
    output logic [2*HEAD_DIM-1:0]   read_resp_k_pair,
    output logic [1:0]              read_resp_k_mask,
    output logic                    read_resp_score_tag,

    output logic [31:0]             perf_read_transactions,
    output logic [31:0]             perf_read_bits,
    output logic                    protocol_error
);
    logic [2*HEAD_DIM-1:0] q_mem [0:PAIRS-1];
    logic [HEAD_DIM-1:0] k0_mem [0:PAIRS-1];
    logic [HEAD_DIM-1:0] k1_mem [0:PAIRS-1];
    logic [31:0] read_transactions_q;
    logic [31:0] read_bits_q;
    logic protocol_error_q;
    logic write_in_range;
    logic read_in_range;
    logic read_fire;

    assign write_in_range = 32'(write_addr) < 32'(PAIRS);
    assign read_in_range = 32'(read_req_addr) < 32'(PAIRS);
    assign write_ready = !row_reset;
    assign read_req_ready = !read_resp_valid || read_resp_ready;
    assign read_fire = read_req_valid && read_req_ready;
    assign perf_read_transactions = read_transactions_q;
    assign perf_read_bits = read_bits_q;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            read_resp_valid <= 1'b0;
            read_resp_addr <= '0;
            read_resp_q_pair <= '0;
            read_resp_k_pair <= '0;
            read_resp_k_mask <= '0;
            read_resp_score_tag <= 1'b0;
            read_transactions_q <= '0;
            read_bits_q <= '0;
            protocol_error_q <= 1'b0;
        end else if (row_reset) begin
            read_resp_valid <= 1'b0;
            read_transactions_q <= '0;
            read_bits_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (write_valid && write_ready) begin
                if (!write_in_range) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    q_mem[write_addr] <= write_q_pair;
                    k0_mem[write_addr] <= write_k_pair[HEAD_DIM-1:0];
                    k1_mem[write_addr] <= write_k_pair[2*HEAD_DIM-1:HEAD_DIM];
                end
            end

            if (read_fire) begin
                if (!read_in_range || (!read_req_q && read_req_k_mask == 0)) begin
                    protocol_error_q <= 1'b1;
                    read_resp_valid <= 1'b0;
                end else begin
                    read_resp_addr <= read_req_addr;
                    if (read_req_q)
                        read_resp_q_pair <= q_mem[read_req_addr];
                    if (read_req_k_mask[0])
                        read_resp_k_pair[HEAD_DIM-1:0] <= k0_mem[read_req_addr];
                    if (read_req_k_mask[1])
                        read_resp_k_pair[2*HEAD_DIM-1:HEAD_DIM] <= k1_mem[read_req_addr];
                    read_resp_k_mask <= read_req_k_mask;
                    read_resp_score_tag <= read_req_score_tag;
                    read_resp_valid <= 1'b1;
                    read_transactions_q <= read_transactions_q
                        + 32'(read_req_q)
                        + 32'(read_req_k_mask[0])
                        + 32'(read_req_k_mask[1]);
                    read_bits_q <= read_bits_q
                        + 32'(read_req_q) * (2 * HEAD_DIM)
                        + 32'(read_req_k_mask[0]) * HEAD_DIM
                        + 32'(read_req_k_mask[1]) * HEAD_DIM;
                end
            end else if (read_resp_valid && read_resp_ready) begin
                read_resp_valid <= 1'b0;
            end

            if (write_valid && !write_ready)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
