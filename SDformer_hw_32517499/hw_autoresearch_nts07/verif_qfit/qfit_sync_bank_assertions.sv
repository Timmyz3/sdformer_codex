`timescale 1ns/1ps
`default_nettype none

module qfit_sync_bank_assertions #(
    parameter int DEPTH = 45,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input logic clk_core,
    input logic rst_core,
    input logic wr_en,
    input logic [ADDR_W-1:0] wr_addr,
    input logic rd_en,
    input logic [ADDR_W-1:0] rd_addr,
    input logic rd_valid
);
    property p_one_cycle_read_valid;
        @(posedge clk_core) disable iff (rst_core)
            rd_valid == $past(rd_en);
    endproperty

    property p_write_address_in_range;
        @(posedge clk_core) disable iff (rst_core)
            wr_en |-> 32'(wr_addr) < DEPTH;
    endproperty

    property p_read_address_in_range;
        @(posedge clk_core) disable iff (rst_core)
            rd_en |-> 32'(rd_addr) < DEPTH;
    endproperty

    assert property (p_one_cycle_read_valid);
    assert property (p_write_address_in_range);
    assert property (p_read_address_in_range);
endmodule

bind qfit_sync_1r1w_bank qfit_sync_bank_assertions #(
        .DEPTH(DEPTH),
        .ADDR_W(ADDR_W)
    )
    u_qfit_sync_bank_assertions (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .rd_en(rd_en),
        .rd_addr(rd_addr),
        .rd_valid(rd_valid)
    );

`default_nettype wire
