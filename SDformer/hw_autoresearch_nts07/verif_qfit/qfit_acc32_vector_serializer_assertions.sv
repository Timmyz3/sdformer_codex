`timescale 1ns/1ps
`default_nettype none

module qfit_acc32_vector_serializer_assertions #(
    parameter int OUT_DIM = 32,
    parameter int ACC_W = 32,
    parameter int PLANE_W = 1,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    parameter int OUT_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic in_valid,
    input logic in_ready,
    input logic [PLANE_W-1:0] in_plane,
    input logic [Y_W-1:0] in_y,
    input logic [X_W-1:0] in_x,
    input logic [OUT_DIM*ACC_W-1:0] in_data,
    input logic in_last,
    input logic out_valid,
    input logic out_ready,
    input logic [PLANE_W-1:0] out_plane,
    input logic [Y_W-1:0] out_y,
    input logic [X_W-1:0] out_x,
    input logic [OUT_W-1:0] out_index,
    input logic signed [ACC_W-1:0] out_data,
    input logic out_last
);
    property p_input_stable;
        @(posedge clk_core) disable iff (rst_core)
            in_valid && !in_ready
            |=> in_valid && $stable({in_plane, in_y, in_x, in_data, in_last});
    endproperty

    property p_output_stable;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && !out_ready
            |=> out_valid
                && $stable({out_plane, out_y, out_x, out_index,
                            out_data, out_last});
    endproperty

    property p_last_only_on_final_lane;
        @(posedge clk_core) disable iff (rst_core)
            out_valid && out_last |-> 32'(out_index) + 1 == OUT_DIM;
    endproperty

    assert property (p_input_stable);
    assert property (p_output_stable);
    assert property (p_last_only_on_final_lane);
endmodule

bind qfit_acc32_vector_serializer
    qfit_acc32_vector_serializer_assertions #(
        .OUT_DIM(OUT_DIM), .ACC_W(ACC_W), .PLANE_W(PLANE_W),
        .Y_W(Y_W), .X_W(X_W), .OUT_W(OUT_W)
    ) u_qfit_acc32_vector_serializer_assertions (.*);

`default_nettype wire
