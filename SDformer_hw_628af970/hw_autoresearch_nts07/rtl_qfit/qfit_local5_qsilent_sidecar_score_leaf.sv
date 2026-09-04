`timescale 1ns/1ps
`default_nettype none

module qfit_local5_qsilent_sidecar_score_leaf #(
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16
) (
    input  logic                    clk_core,
    input  logic                    rst_core,
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [TAG_W-1:0]        in_tag,
    input  logic [5*6-1:0]          in_popcount,
    input  logic [4:0]              in_valid_mask,
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [TAG_W-1:0]        out_tag,
    output logic [5*SCORE_W-1:0]    out_score_q7,
    output logic [4:0]              out_valid_mask
);

    logic valid_q;
    logic [TAG_W-1:0] tag_q;
    logic [5*SCORE_W-1:0] score_q;
    logic [4:0] mask_q;
    logic signed [SCORE_W-1:0] score_w [0:4];

    function automatic logic signed [SCORE_W-1:0] score_from_popcount(
        input logic [5:0] popcount
    );
        logic [6:0] raw_value;
        logic [2:0] quotient;
        logic [3:0] remainder;
        logic increment;
        begin
            raw_value = 7'd32 - {1'b0, popcount};
            quotient = raw_value[6:4];
            remainder = raw_value[3:0];
            increment = (remainder > 4'd8)
                     || ((remainder == 4'd8) && quotient[0]);
            score_from_popcount = SCORE_W'({1'b0, quotient})
                                + SCORE_W'(increment);
        end
    endfunction

    always_comb begin
        for (int role = 0; role < 5; role = role + 1)
            score_w[role] = in_valid_mask[role]
                          ? score_from_popcount(in_popcount[role*6 +: 6])
                          : SCORE_W'(-256);
    end

    assign in_ready = !valid_q || out_ready;
    assign out_valid = valid_q;
    assign out_tag = tag_q;
    assign out_score_q7 = score_q;
    assign out_valid_mask = mask_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_q <= 1'b0;
            tag_q <= '0;
            score_q <= '0;
            mask_q <= '0;
        end else if (in_ready) begin
            valid_q <= in_valid;
            if (in_valid) begin
                tag_q <= in_tag;
                mask_q <= in_valid_mask;
                for (int role = 0; role < 5; role = role + 1)
                    score_q[role*SCORE_W +: SCORE_W] <= score_w[role];
            end
        end
    end
endmodule

`default_nettype wire
