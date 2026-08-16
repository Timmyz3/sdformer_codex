`default_nettype none

module ttx_ceil_log2_u32 (
    input  logic [31:0] value,
    output logic [5:0]  shift_amount
);
    integer bit_idx;
    logic [31:0] probe;

    always_comb begin
        shift_amount = 6'd0;
        probe = (value <= 1) ? 32'd1 : (value - 1'b1);
        for (bit_idx = 0; bit_idx < 32; bit_idx = bit_idx + 1) begin
            if (probe[bit_idx]) begin
                shift_amount = 6'(bit_idx + 1);
            end
        end
    end
endmodule

`default_nettype wire
