`timescale 1ns/1ps
`default_nettype none

// Nangate45开放SRAM宏绑定：四个256x32单口1RW宏保存225组Q/K pair。
// 写入与读取属于互斥相位；读通道保留一个响应槽并支持逐拍流水请求。
module h67_fakeram45_qk_row_store #(
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
    logic [31:0] macro_read_data [0:3];
    logic pending_q;
    logic [ADDR_W-1:0] pending_addr_q;
    logic pending_read_q_q;
    logic [1:0] pending_k_mask_q;
    logic pending_score_tag_q;
    logic [31:0] read_transactions_q;
    logic [31:0] read_bits_q;
    logic protocol_error_q;
    logic write_in_range;
    logic read_in_range;
    logic write_fire;
    logic read_fire;

    initial begin
        if (HEAD_DIM != 32 || PAIRS > 256 || ADDR_W != 8)
            $error("fakeram45 Q/K row store requires HEAD_DIM=32, PAIRS<=256, ADDR_W=8");
    end

    assign write_in_range = 32'(write_addr) < 32'(PAIRS);
    assign read_in_range = 32'(read_req_addr) < 32'(PAIRS);
    assign write_ready = !row_reset && !pending_q && !read_resp_valid;
    assign read_req_ready = !row_reset && (!read_resp_valid || read_resp_ready);
    assign write_fire = write_valid && write_ready;
    assign read_fire = read_req_valid && read_req_ready;
    assign perf_read_transactions = read_transactions_q;
    assign perf_read_bits = read_bits_q;
    assign protocol_error = protocol_error_q;

    fakeram45_256x32 u_q0 (
        .clk(clk_core),
        .ce_in(write_fire || (read_fire && read_req_q)),
        .we_in(write_fire),
        .w_mask_in({32{write_fire}}),
        .addr_in(write_fire ? write_addr : read_req_addr),
        .wd_in(write_q_pair[31:0]),
        .rd_out(macro_read_data[0])
    );
    fakeram45_256x32 u_q1 (
        .clk(clk_core),
        .ce_in(write_fire || (read_fire && read_req_q)),
        .we_in(write_fire),
        .w_mask_in({32{write_fire}}),
        .addr_in(write_fire ? write_addr : read_req_addr),
        .wd_in(write_q_pair[63:32]),
        .rd_out(macro_read_data[1])
    );
    fakeram45_256x32 u_k0 (
        .clk(clk_core),
        .ce_in(write_fire || (read_fire && read_req_k_mask[0])),
        .we_in(write_fire),
        .w_mask_in({32{write_fire}}),
        .addr_in(write_fire ? write_addr : read_req_addr),
        .wd_in(write_k_pair[31:0]),
        .rd_out(macro_read_data[2])
    );
    fakeram45_256x32 u_k1 (
        .clk(clk_core),
        .ce_in(write_fire || (read_fire && read_req_k_mask[1])),
        .we_in(write_fire),
        .w_mask_in({32{write_fire}}),
        .addr_in(write_fire ? write_addr : read_req_addr),
        .wd_in(write_k_pair[63:32]),
        .rd_out(macro_read_data[3])
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            pending_q <= 1'b0;
            pending_addr_q <= '0;
            pending_read_q_q <= 1'b0;
            pending_k_mask_q <= '0;
            pending_score_tag_q <= 1'b0;
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
            pending_q <= 1'b0;
            read_resp_valid <= 1'b0;
            read_transactions_q <= '0;
            read_bits_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            if (pending_q) begin
                read_resp_valid <= 1'b1;
                read_resp_addr <= pending_addr_q;
                if (pending_read_q_q)
                    read_resp_q_pair <= {macro_read_data[1], macro_read_data[0]};
                if (pending_k_mask_q[0])
                    read_resp_k_pair[31:0] <= macro_read_data[2];
                if (pending_k_mask_q[1])
                    read_resp_k_pair[63:32] <= macro_read_data[3];
                read_resp_k_mask <= pending_k_mask_q;
                read_resp_score_tag <= pending_score_tag_q;
            end else if (read_resp_valid && read_resp_ready) begin
                read_resp_valid <= 1'b0;
            end

            pending_q <= read_fire;
            if (read_fire) begin
                pending_addr_q <= read_req_addr;
                pending_read_q_q <= read_req_q;
                pending_k_mask_q <= read_req_k_mask;
                pending_score_tag_q <= read_req_score_tag;
                if (!read_in_range || (!read_req_q && read_req_k_mask == 0)) begin
                    protocol_error_q <= 1'b1;
                    pending_q <= 1'b0;
                end else begin
                    read_transactions_q <= read_transactions_q
                        + 32'(read_req_q)
                        + 32'(read_req_k_mask[0])
                        + 32'(read_req_k_mask[1]);
                    read_bits_q <= read_bits_q
                        + 32'(read_req_q) * (2 * HEAD_DIM)
                        + 32'(read_req_k_mask[0]) * HEAD_DIM
                        + 32'(read_req_k_mask[1]) * HEAD_DIM;
                end
            end

            if (write_valid && !write_ready)
                protocol_error_q <= 1'b1;
            if (write_fire && !write_in_range)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
