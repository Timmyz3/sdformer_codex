`timescale 1ns/1ps
`default_nettype none

// Nangate45 open-macro binding for one 450-entry Local5 relation bank.
// Two 256-entry single-port 1RW macros cover the address range. DATA_W=32
// uses 256x32 macros; DATA_W=10 is zero-padded into 256x16 macros.
module qfit_fakeram45_relation_bank_450 #(
    parameter int DATA_W = 10,
    parameter int ADDR_W = 9
) (
    input  logic                  clk_core,
    input  logic                  rst_core,
    input  logic                  write_valid,
    input  logic [ADDR_W-1:0]     write_addr,
    input  logic [DATA_W-1:0]     write_data,
    input  logic                  read_valid,
    input  logic [ADDR_W-1:0]     read_addr,
    output logic                  read_data_valid,
    output logic [DATA_W-1:0]     read_data
);
    logic read_valid_q;
    logic read_bank_q;

    initial begin
        if (ADDR_W != 9)
            $error("fakeram45 relation bank requires ADDR_W=9");
        if (DATA_W != 10 && DATA_W != 32)
            $error("fakeram45 relation bank supports DATA_W=10 or 32");
    end

    generate
        if (DATA_W == 32) begin : g_w32
            logic [31:0] bank_read_data [0:1];
            logic [31:0] write_data_padded;

            assign write_data_padded = write_data;
            for (genvar bank = 0; bank < 2; bank = bank + 1) begin : g_bank
                localparam logic BANK_SELECT = bank;
                fakeram45_256x32 u_macro (
                    .clk(clk_core),
                    .ce_in(
                        (write_valid && write_addr[8] == BANK_SELECT)
                        || (read_valid && read_addr[8] == BANK_SELECT)
                    ),
                    .we_in(write_valid && write_addr[8] == BANK_SELECT),
                    .w_mask_in({32{write_valid}}),
                    .addr_in(write_valid ? write_addr[7:0] : read_addr[7:0]),
                    .wd_in(write_data_padded),
                    .rd_out(bank_read_data[bank])
                );
            end

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    read_valid_q <= 1'b0;
                    read_bank_q <= 1'b0;
                    read_data_valid <= 1'b0;
                    read_data <= '0;
                end else begin
                    read_valid_q <= read_valid;
                    read_data_valid <= read_valid_q;
                    if (read_valid)
                        read_bank_q <= read_addr[8];
                    if (read_valid_q)
                        read_data <= bank_read_data[read_bank_q];
                end
            end
        end else begin : g_w10
            logic [15:0] bank_read_data [0:1];
            logic [15:0] write_data_padded;

            assign write_data_padded = {{6{1'b0}}, write_data};
            for (genvar bank = 0; bank < 2; bank = bank + 1) begin : g_bank
                localparam logic BANK_SELECT = bank;
                fakeram45_256x16 u_macro (
                    .clk(clk_core),
                    .ce_in(
                        (write_valid && write_addr[8] == BANK_SELECT)
                        || (read_valid && read_addr[8] == BANK_SELECT)
                    ),
                    .we_in(write_valid && write_addr[8] == BANK_SELECT),
                    .w_mask_in({16{write_valid}}),
                    .addr_in(write_valid ? write_addr[7:0] : read_addr[7:0]),
                    .wd_in(write_data_padded),
                    .rd_out(bank_read_data[bank])
                );
            end

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    read_valid_q <= 1'b0;
                    read_bank_q <= 1'b0;
                    read_data_valid <= 1'b0;
                    read_data <= '0;
                end else begin
                    read_valid_q <= read_valid;
                    read_data_valid <= read_valid_q;
                    if (read_valid)
                        read_bank_q <= read_addr[8];
                    if (read_valid_q)
                        read_data <= bank_read_data[read_bank_q][DATA_W-1:0];
                end
            end
        end
    endgenerate
endmodule

`default_nettype wire
