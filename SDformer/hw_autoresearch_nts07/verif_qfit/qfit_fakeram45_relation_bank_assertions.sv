`timescale 1ns/1ps
`default_nettype none

module qfit_fakeram45_relation_bank_assertions #(
    parameter int DATA_W = 10,
    parameter int ADDR_W = 9
) (
    input logic                  clk_core,
    input logic                  rst_core,
    input logic                  write_valid,
    input logic [ADDR_W-1:0]     write_addr,
    input logic                  read_valid,
    input logic [ADDR_W-1:0]     read_addr,
    input logic                  read_data_valid,
    input logic [DATA_W-1:0]     read_data,
    input logic                  read_valid_q
);
    assert property (@(posedge clk_core) disable iff (rst_core)
        !(write_valid && read_valid)
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        write_valid |-> 32'(write_addr) < 450
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        read_valid |-> 32'(read_addr) < 450
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        read_valid_q |=> read_data_valid
    );
    assert property (@(posedge clk_core) disable iff (rst_core)
        read_data_valid |-> !$isunknown(read_data)
    );
endmodule

bind qfit_fakeram45_relation_bank_450
    qfit_fakeram45_relation_bank_assertions #(
        .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) u_qfit_fakeram45_relation_bank_assertions (.*);

`default_nettype wire
