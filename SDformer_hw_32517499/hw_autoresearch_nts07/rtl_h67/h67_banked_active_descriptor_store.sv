`timescale 1ns/1ps
`default_nettype none

// active descriptor驻留存储。偶/奇逻辑地址分属两个bank，因此连续两项可同拍写入；
// build与emit相位互斥，使每个bank只需要单口1RW。MEMORY_IMPL=1绑定开放宏。
module h67_banked_active_descriptor_store #(
    parameter int DEPTH = 450,
    parameter int DATA_W = 28,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int MEMORY_IMPL = 0,
    parameter int BANK_DEPTH = (DEPTH + 1) / 2,
    parameter int BANK_ADDR_W = (BANK_DEPTH <= 1) ? 1 : $clog2(BANK_DEPTH)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   window_start,

    input  logic [1:0]             write_count,
    input  logic [ADDR_W-1:0]      write0_addr,
    input  logic [DATA_W-1:0]      write0_data,
    input  logic [ADDR_W-1:0]      write1_addr,
    input  logic [DATA_W-1:0]      write1_data,

    input  logic                   read_req_valid,
    output logic                   read_req_ready,
    input  logic [ADDR_W-1:0]      read_req_addr,
    output logic                   read_resp_valid,
    input  logic                   read_resp_ready,
    output logic [ADDR_W-1:0]      read_resp_addr,
    output logic [DATA_W-1:0]      read_resp_data,
    output logic                   protocol_error
);
    logic write0_valid;
    logic write1_valid;
    logic write0_legal;
    logic write1_legal;
    logic write_bank_conflict;
    logic bank0_write;
    logic bank1_write;
    logic [BANK_ADDR_W-1:0] bank0_write_addr;
    logic [BANK_ADDR_W-1:0] bank1_write_addr;
    logic [DATA_W-1:0] bank0_write_data;
    logic [DATA_W-1:0] bank1_write_data;
    logic read_fire;
    logic read_legal;
    logic read_bank_q;
    logic protocol_error_q;

    initial begin
        if (DEPTH <= 0 || DATA_W <= 0 || DATA_W > 32)
            $error("active descriptor store requires DEPTH>0 and 1<=DATA_W<=32");
        if (MEMORY_IMPL == 1 && (DEPTH > 512 || BANK_ADDR_W > 8))
            $error("fakeram45 descriptor store supports at most 512 logical entries");
    end

    assign write0_valid = write_count != 0;
    assign write1_valid = write_count == 2;
    assign write0_legal = 32'(write0_addr) < 32'(DEPTH);
    assign write1_legal = 32'(write1_addr) < 32'(DEPTH);
    assign write_bank_conflict = write1_valid
                               && write0_addr[0] == write1_addr[0];
    assign bank0_write = (write0_valid && !write0_addr[0])
                       || (write1_valid && !write1_addr[0]);
    assign bank1_write = (write0_valid && write0_addr[0])
                       || (write1_valid && write1_addr[0]);
    assign bank0_write_addr = write0_valid && !write0_addr[0]
                            ? write0_addr[ADDR_W-1:1]
                            : write1_addr[ADDR_W-1:1];
    assign bank1_write_addr = write0_valid && write0_addr[0]
                            ? write0_addr[ADDR_W-1:1]
                            : write1_addr[ADDR_W-1:1];
    assign bank0_write_data = write0_valid && !write0_addr[0]
                            ? write0_data : write1_data;
    assign bank1_write_data = write0_valid && write0_addr[0]
                            ? write0_data : write1_data;
    assign read_legal = 32'(read_req_addr) < 32'(DEPTH);
    assign read_req_ready = write_count == 0
                          && (!read_resp_valid || read_resp_ready);
    assign read_fire = read_req_valid && read_req_ready;
    assign protocol_error = protocol_error_q;

    generate
        if (MEMORY_IMPL == 0) begin : g_behavior
            logic [DATA_W-1:0] bank0_mem [0:BANK_DEPTH-1];
            logic [DATA_W-1:0] bank1_mem [0:BANK_DEPTH-1];
            logic [DATA_W-1:0] behavior_read_data_q;

            assign read_resp_data = behavior_read_data_q;
            always_ff @(posedge clk_core) begin
                if (bank0_write)
                    bank0_mem[bank0_write_addr] <= bank0_write_data;
                if (bank1_write)
                    bank1_mem[bank1_write_addr] <= bank1_write_data;
                if (read_fire && read_legal)
                    behavior_read_data_q <= read_req_addr[0]
                        ? bank1_mem[read_req_addr[ADDR_W-1:1]]
                        : bank0_mem[read_req_addr[ADDR_W-1:1]];
            end
        end else begin : g_fakeram45
            logic [31:0] bank0_read_data;
            logic [31:0] bank1_read_data;
            logic [31:0] bank0_write_data_padded;
            logic [31:0] bank1_write_data_padded;
            logic [31:0] bank0_write_mask;
            logic [31:0] bank1_write_mask;

            // 32-bit代理宏的padding位随低位数据写入；功能侧只读取DATA_W位。
            // 这会保守计入整字写活动，并避免用时序例外掩盖宏输入端点。
            assign bank0_write_data_padded =
                {{(32-DATA_W){bank0_write_data[0]}}, bank0_write_data};
            assign bank1_write_data_padded =
                {{(32-DATA_W){bank1_write_data[0]}}, bank1_write_data};
            assign bank0_write_mask = {32{bank0_write}};
            assign bank1_write_mask = {32{bank1_write}};
            assign read_resp_data = read_bank_q
                                  ? bank1_read_data[DATA_W-1:0]
                                  : bank0_read_data[DATA_W-1:0];

            fakeram45_256x32 u_bank0 (
                .clk(clk_core),
                .ce_in(bank0_write || (read_fire && !read_req_addr[0])),
                .we_in(bank0_write),
                .w_mask_in(bank0_write_mask),
                .addr_in(bank0_write ? 8'(bank0_write_addr)
                                     : 8'(read_req_addr[ADDR_W-1:1])),
                .wd_in(bank0_write_data_padded),
                .rd_out(bank0_read_data)
            );
            fakeram45_256x32 u_bank1 (
                .clk(clk_core),
                .ce_in(bank1_write || (read_fire && read_req_addr[0])),
                .we_in(bank1_write),
                .w_mask_in(bank1_write_mask),
                .addr_in(bank1_write ? 8'(bank1_write_addr)
                                     : 8'(read_req_addr[ADDR_W-1:1])),
                .wd_in(bank1_write_data_padded),
                .rd_out(bank1_read_data)
            );
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core || window_start) begin
            read_resp_valid <= 1'b0;
            read_resp_addr <= '0;
            read_bank_q <= 1'b0;
            protocol_error_q <= 1'b0;
        end else begin
            if (read_fire) begin
                read_resp_valid <= read_legal;
                read_resp_addr <= read_req_addr;
                read_bank_q <= read_req_addr[0];
                if (!read_legal)
                    protocol_error_q <= 1'b1;
            end else if (read_resp_valid && read_resp_ready) begin
                read_resp_valid <= 1'b0;
            end

            if (write_count > 2 || (write0_valid && !write0_legal)
                || (write1_valid && !write1_legal) || write_bank_conflict)
                protocol_error_q <= 1'b1;
            if (read_req_valid && write_count != 0)
                protocol_error_q <= 1'b1;
        end
    end
endmodule

`default_nettype wire
