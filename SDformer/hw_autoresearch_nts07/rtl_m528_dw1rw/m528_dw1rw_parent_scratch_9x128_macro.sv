`timescale 1ns/1ps
`default_nettype none

// Fixed pin-level adapter for the only admitted parent-scratch organization:
// nine coherent TS1N28HPCPHVTB128X128M4S single-port 1RW macros.  Logical
// addresses are six bits; A[6] is hard-wired low, so rows 64..127 cannot be
// selected.  There is intentionally no synthesizable register-array fallback.
//
// VCS must compile the checksum-verified slow foundry behavioral .v alongside
// this source.  DC must link the checksum-verified slow .db and must not read
// the behavioral .v.  Formality treats the nine cells as matched black boxes
// or cutpoints.  Those mutually exclusive bindings are frozen by the M529
// source-only contract and exact-SHA runner.
module m528_dw1rw_parent_scratch_9x128_macro (
    input  logic          clk_core,
    input  logic          enable,
    input  logic          write_enable,
    input  logic [5:0]    address,
    input  logic [1151:0] write_data,
    output logic [1151:0] read_data
);
    wire ceb = ~enable;
    wire web = ~write_enable;
    wire [6:0] macro_address = {1'b0, address};
    wire [1151:0] macro_q;

    for (genvar slice = 0; slice < 9; slice = slice + 1) begin : g_slice
        TS1N28HPCPHVTB128X128M4S u_parent_sram (
            .CLK (clk_core),
            .CEB (ceb),
            .WEB (web),
            .A   (macro_address),
            .D   (write_data[slice*128 +: 128]),
            .Q   (macro_q[slice*128 +: 128])
        );
    end

    always_comb read_data = macro_q;
endmodule

`default_nettype wire
