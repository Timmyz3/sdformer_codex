`timescale 1ns/1ps
`default_nettype none

// K0/K1分bank同步存储；read-enable直接由active mask控制。
module h67_sync_dual_bank_k_store #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int ADDR_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    // 0: flop array (sealed). 1: fakeram45_256x32 per bank, same 1-cycle read.
    parameter int MEMORY_IMPL = 0
) (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     window_start,

    input  logic                     write_valid,
    input  logic [ADDR_W-1:0]        write_addr,
    input  logic [2*HEAD_DIM-1:0]    write_k_pair,

    input  logic [1:0]               read_req_valid,
    input  logic [ADDR_W-1:0]        read_req_addr,
    output logic [1:0]               read_resp_valid,
    output logic [HEAD_DIM-1:0]      read_resp_k0,
    output logic [HEAD_DIM-1:0]      read_resp_k1,

    output logic [31:0]              perf_read_transactions,
    output logic [31:0]              perf_read_bits,
    output logic                     protocol_error
);
    logic [HEAD_DIM-1:0] bank0 [0:PAIRS-1];
    logic [HEAD_DIM-1:0] bank1 [0:PAIRS-1];
    logic [31:0] ram0_rd;
    logic [31:0] ram1_rd;
    logic [31:0] read_transactions_q;
    logic [31:0] read_bits_q;
    logic protocol_error_q;
    logic write_in_range;
    logic read_in_range;
    logic [1:0] read_hit;

    assign write_in_range = 32'(write_addr) < 32'(PAIRS);
    assign read_in_range = 32'(read_req_addr) < 32'(PAIRS);
    assign read_hit = read_req_valid & {2{read_in_range && !write_valid}};
    assign perf_read_transactions = read_transactions_q;
    assign perf_read_bits = read_bits_q;
    assign protocol_error = protocol_error_q;

    generate
        if (MEMORY_IMPL == 0) begin : g_flop
            always_ff @(posedge clk_core) begin
                if (rst_core || window_start) begin
                    read_resp_valid <= '0;
                    read_resp_k0 <= '0;
                    read_resp_k1 <= '0;
                end else begin
                    read_resp_valid <= '0;
                    if (write_valid && write_in_range) begin
                        bank0[write_addr] <= write_k_pair[HEAD_DIM-1:0];
                        bank1[write_addr] <= write_k_pair[2*HEAD_DIM-1:HEAD_DIM];
                    end
                    if (read_req_valid != 0 && read_in_range) begin
                        if (read_req_valid[0])
                            read_resp_k0 <= bank0[read_req_addr];
                        if (read_req_valid[1])
                            read_resp_k1 <= bank1[read_req_addr];
                        read_resp_valid <= read_req_valid;
                    end
                end
            end
        end else begin : g_fakeram
            fakeram45_256x32 u_bank0 (
                .clk(clk_core),
                .ce_in(write_valid || read_req_valid[0]),
                .we_in(write_valid),
                .w_mask_in(32'hffff_ffff),
                .addr_in(8'(write_valid ? write_addr : read_req_addr)),
                .wd_in(write_k_pair[HEAD_DIM-1:0]),
                .rd_out(ram0_rd)
            );
            fakeram45_256x32 u_bank1 (
                .clk(clk_core),
                .ce_in(write_valid || read_req_valid[1]),
                .we_in(write_valid),
                .w_mask_in(32'hffff_ffff),
                .addr_in(8'(write_valid ? write_addr : read_req_addr)),
                .wd_in(write_k_pair[2*HEAD_DIM-1:HEAD_DIM]),
                .rd_out(ram1_rd)
            );
            always_ff @(posedge clk_core) begin
                if (rst_core || window_start)
                    read_resp_valid <= '0;
                else
                    read_resp_valid <= read_hit;
            end
            assign read_resp_k0 = ram0_rd[HEAD_DIM-1:0];
            assign read_resp_k1 = ram1_rd[HEAD_DIM-1:0];
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            read_transactions_q <= '0;
            read_bits_q <= '0;
            protocol_error_q <= 1'b0;
        end else if (window_start) begin
            read_transactions_q <= '0;
            read_bits_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (write_valid && !write_in_range)
                protocol_error_q <= 1'b1;
            if (read_req_valid != 0 && !read_in_range)
                protocol_error_q <= 1'b1;
            if (write_valid && read_req_valid != 0)
                protocol_error_q <= 1'b1;
            if (read_req_valid != 0 && read_in_range && !write_valid) begin
                read_transactions_q <= read_transactions_q
                    + 32'(read_req_valid[0]) + 32'(read_req_valid[1]);
                read_bits_q <= read_bits_q
                    + HEAD_DIM * (32'(read_req_valid[0]) + 32'(read_req_valid[1]));
            end
        end
    end
endmodule

`default_nettype wire
