`timescale 1ns/1ps
`default_nettype none

// One shared 32-lane alpha-XNOR engine in raw16 units.
module alpha_xnor_raw32 (
    input  logic [31:0] q_bits,
    input  logic [31:0] k_bits,
    output logic [11:0] raw16
);

    always_comb begin
        raw16 = 12'd0;
        for (int lane = 32'd0; lane < 32; lane = lane + 32'd1) begin
            if (q_bits[lane] && k_bits[lane]) begin
                raw16 = raw16 + 12'd64;
            end else if (!q_bits[lane] && !k_bits[lane]) begin
                raw16 = raw16 + 12'd1;
            end
        end
    end

endmodule

`default_nettype wire
