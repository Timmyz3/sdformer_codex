`timescale 1ns/1ps
`default_nettype none

// VCS resolves module references before it prunes a constant generate branch.
// Keep the unsupported geometry target defined, while making any real
// instantiation fail immediately and noisily.
module QFIT_TSMC28_SRAM_GEOMETRY_MUST_BE_128X512;
    initial $fatal(1, "TSMC28 SRAM adapter requires DEPTH=128 DATA_W=512 ADDR_W=7");
endmodule

// Macro-bound implementation of one logical 128x512 Acc32 bank.  The native
// TSMC28 compiler supports 128-bit words at this mux setting, so four identical
// 128x128 1RW macros share address/control and form one coherent wide bank.
// The proprietary macro views remain outside Git; only this pin-level adapter
// and their content-addressed manifest are source-controlled.
module qfit_sync_1rw_acc_bank #(
    parameter int DEPTH = 128,
    parameter int DATA_W = 512,
    parameter int ADDR_W = 7
) (
    input  logic                  clk_core,
    input  logic                  enable,
    input  logic                  write_enable,
    input  logic [ADDR_W-1:0]     address,
    input  logic [DATA_W-1:0]     write_data,
    output logic [DATA_W-1:0]     read_data
);
    if (DEPTH == 128 && DATA_W == 512 && ADDR_W == 7) begin : g_supported
        wire ceb = ~enable;
        wire web = ~write_enable;
        wire [DATA_W-1:0] macro_q;

        for (genvar slice = 0; slice < 4; slice = slice + 1) begin : g_slice
            TS1N28HPCPHVTB128X128M4S u_sram (
                .CLK (clk_core),
                .CEB (ceb),
                .WEB (web),
                .A   (address),
                .D   (write_data[slice*128 +: 128]),
                .Q   (macro_q[slice*128 +: 128])
            );
        end
        always_comb read_data = macro_q;
    end else begin : g_unsupported
        // An unsupported parameterization must fail at simulation time zero
        // instead of silently truncating storage or synthesizing a register array.
        QFIT_TSMC28_SRAM_GEOMETRY_MUST_BE_128X512 u_fail_closed (); // spyglass disable W505
        always_comb read_data = 'x;
    end
endmodule

`default_nettype wire
