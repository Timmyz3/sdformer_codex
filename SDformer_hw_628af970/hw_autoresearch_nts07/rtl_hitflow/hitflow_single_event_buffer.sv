`timescale 1ns/1ps
`default_nettype none

module hitflow_single_event_buffer #(
    parameter int DATA_W = 32,
    parameter int TAG_W  = 48
) (
    input  logic              clk_core,
    input  logic              rst_core,
    input  logic              in_valid,
    output logic              in_ready,
    input  logic [DATA_W-1:0] in_data,
    input  logic [TAG_W-1:0]  in_tag,
    output logic              out_valid,
    input  logic              out_ready,
    output logic [DATA_W-1:0] out_data,
    output logic [TAG_W-1:0]  out_tag
);

    logic              valid_q;
    logic [DATA_W-1:0] data_q;
    logic [TAG_W-1:0]  tag_q;

    assign in_ready = ~valid_q | out_ready;
    assign out_valid = valid_q;
    assign out_data = data_q;
    assign out_tag = tag_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_q <= 1'b0;
            data_q  <= '0;
            tag_q   <= '0;
        end else if (in_ready) begin
            valid_q <= in_valid;
            if (in_valid) begin
                data_q <= in_data;
                tag_q  <= in_tag;
            end
        end
    end

endmodule

`default_nettype wire
