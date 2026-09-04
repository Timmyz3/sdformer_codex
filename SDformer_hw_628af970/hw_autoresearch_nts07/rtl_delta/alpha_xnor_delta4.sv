`timescale 1ns/1ps
`default_nettype none

// Exact shared residual arithmetic:
// H67 uses Q0/K0 -> Q1/K1; Local5 uses Q/Kself -> Q/Kneighbor.
module alpha_xnor_delta4 (
    input  logic [3:0]        lane_valid,
    input  logic [19:0]       lane_ids,
    input  logic [31:0]       q_old_bits,
    input  logic [31:0]       k_old_bits,
    input  logic [31:0]       q_new_bits,
    input  logic [31:0]       k_new_bits,
    output logic signed [9:0] delta_raw16
);

    logic [4:0] lane_id;
    logic [7:0] old_score;
    logic [7:0] new_score;

    always_comb begin
        delta_raw16 = 10'sd0;
        lane_id = 5'd0;
        old_score = 8'd0;
        new_score = 8'd0;
        for (int way = 32'd0; way < 4; way = way + 32'd1) begin
            lane_id = lane_ids[(way*5) +: 5];
            if (q_old_bits[lane_id] && k_old_bits[lane_id]) begin
                old_score = 8'd64;
            end else if (
                !q_old_bits[lane_id] && !k_old_bits[lane_id]
            ) begin
                old_score = 8'd1;
            end else begin
                old_score = 8'd0;
            end
            if (q_new_bits[lane_id] && k_new_bits[lane_id]) begin
                new_score = 8'd64;
            end else if (
                !q_new_bits[lane_id] && !k_new_bits[lane_id]
            ) begin
                new_score = 8'd1;
            end else begin
                new_score = 8'd0;
            end
            if (lane_valid[way]) begin
                delta_raw16 =
                    delta_raw16
                    + $signed({2'b00, new_score})
                    - $signed({2'b00, old_score});
            end
        end
    end

endmodule

`default_nettype wire
