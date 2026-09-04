`include "unibin_h60_pkg.vh"

module ttb_skip_unit #(
    parameter integer BUNDLE_BITS = 64
)(
    input  wire [BUNDLE_BITS-1:0] q_bundle,
    input  wire [BUNDLE_BITS-1:0] k_bundle,
    output wire                   empty_bundle,
    output reg  [7:0]             active_count
);
    integer i;
    wire [BUNDLE_BITS-1:0] any_event = q_bundle | k_bundle;

    assign empty_bundle = (any_event == {BUNDLE_BITS{1'b0}});

    always @* begin
        active_count = 0;
        for (i = 0; i < BUNDLE_BITS; i = i + 1) begin
            active_count = active_count + {7'd0, any_event[i]};
        end
    end
endmodule
