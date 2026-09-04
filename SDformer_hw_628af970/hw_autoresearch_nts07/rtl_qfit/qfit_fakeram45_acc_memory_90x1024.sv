`timescale 1ns/1ps
`default_nettype none

// Nangate45 open-macro binding for one Local5 accumulator bank. Four
// 128x256 single-port macros form one 128x1024 vector memory.
module qfit_fakeram45_acc_memory_90x1024 #(
    parameter int DEPTH = 90,
    parameter int VEC_W = 1024,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   command_valid,
    input  logic                   command_write,
    input  logic [ADDR_W-1:0]      command_addr,
    input  logic [VEC_W-1:0]       command_write_data,
    output logic                   read_data_valid,
    output logic [VEC_W-1:0]       read_data
);
    logic [255:0] macro_read_data [0:3];

    initial begin
        if (DEPTH > 128)
            $error("fakeram45 accumulator binding requires DEPTH<=128");
        if (VEC_W != 1024)
            $error("fakeram45 accumulator binding requires VEC_W=1024");
        if (ADDR_W != 7)
            $error("fakeram45 accumulator binding requires ADDR_W=7");
    end

    for (genvar slice = 0; slice < 4; slice = slice + 1) begin : g_slice
        fakeram45_128x256 u_macro (
            .clk(clk_core),
            .ce_in(command_valid),
            .we_in(command_write),
            .w_mask_in({256{command_write}}),
            .addr_in(command_addr),
            .wd_in(command_write_data[slice*256 +: 256]),
            .rd_out(macro_read_data[slice])
        );
        assign read_data[slice*256 +: 256] = macro_read_data[slice];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core)
            read_data_valid <= 1'b0;
        else
            read_data_valid <= command_valid && !command_write;
    end
endmodule

`default_nettype wire
