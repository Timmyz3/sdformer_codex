`timescale 1ns/1ps
`default_nettype none

// Equal-throughput destination-owned reference for Q==0 Local5 rows.
// Five K words are read and reduced in parallel, so one destination can be
// accepted per cycle. This is intentionally stronger than the serial fast path.
module qfit_local5_qsilent_dest5_stat_leaf #(
    parameter int TAG_W = 16,
    parameter int SCORE_W = 16
) (
    input  logic                    clk_core,
    input  logic                    rst_core,
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [TAG_W-1:0]        in_tag,
    input  logic [5*32-1:0]         in_k,
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
    logic [5:0] pop_w [0:4];
    logic signed [12:0] raw_w [0:4];
    logic signed [SCORE_W-1:0] score_w [0:4];

    function automatic logic [5:0] popcount32(input logic [31:0] bits);
        logic [5:0] count;
        count = '0;
        for (int lane = 0; lane < 32; lane = lane + 1)
            count = count + 6'(bits[lane]);
        popcount32 = count;
    endfunction

    function automatic logic signed [SCORE_W-1:0] rne_q7(
        input logic signed [12:0] raw_value
    );
        logic [12:0] nonnegative;
        logic [8:0] quotient;
        logic [3:0] remainder;
        logic increment;
        nonnegative = raw_value[12] ? 13'd0 : raw_value;
        quotient = nonnegative[12:4];
        remainder = nonnegative[3:0];
        increment = (remainder > 4'd8)
                 || ((remainder == 4'd8) && quotient[0]);
        rne_q7 = $signed({7'b0, quotient}) + SCORE_W'(increment);
    endfunction

    always_comb begin
        for (int cand = 0; cand < 5; cand = cand + 1) begin
            pop_w[cand] = popcount32(in_k[cand*32 +: 32]);
            raw_w[cand] = 13'sd32 - $signed({7'b0, pop_w[cand]});
            score_w[cand] = in_valid_mask[cand]
                          ? rne_q7(raw_w[cand])
                          : SCORE_W'(-256);
        end
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
                for (int cand = 0; cand < 5; cand = cand + 1)
                    score_q[cand*SCORE_W +: SCORE_W] <= score_w[cand];
            end
        end
    end
endmodule

`default_nettype wire
