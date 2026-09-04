`timescale 1ns/1ps
`default_nettype none

module qfit_local5_source_qsilent_stat_assertions #(
    parameter int TOKEN_W = 9,
    parameter int SCORE_W = 16
) (
    input logic clk_core,
    input logic rst_core,
    input logic out_valid,
    input logic out_ready,
    input logic [TOKEN_W-1:0] out_source_token,
    input logic signed [SCORE_W-1:0] out_score_q7,
    input logic [4:0] out_consumer_valid,
    input logic [5*TOKEN_W-1:0] out_destination,
    input logic [5*3-1:0] out_destination_bank
);

    logic stalled_q;
    logic [TOKEN_W+SCORE_W+5+5*TOKEN_W+15-1:0] stalled_payload_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            stalled_q <= 1'b0;
            stalled_payload_q <= '0;
        end else begin
            if (stalled_q) begin
                assert (out_valid);
                assert ({out_source_token, out_score_q7, out_consumer_valid,
                    out_destination, out_destination_bank}
                    == stalled_payload_q);
            end
            stalled_q <= out_valid && !out_ready;
            if (out_valid && !out_ready)
                stalled_payload_q <= {out_source_token, out_score_q7,
                    out_consumer_valid, out_destination,
                    out_destination_bank};
        end
    end

    always_ff @(posedge clk_core) begin
        if (!rst_core && out_valid) begin
            for (int lhs = 0; lhs < 5; lhs = lhs + 1) begin
                for (int rhs = lhs + 1; rhs < 5; rhs = rhs + 1) begin
                    if (out_consumer_valid[lhs] && out_consumer_valid[rhs]) begin
                        assert (out_destination[lhs*TOKEN_W +: TOKEN_W]
                                != out_destination[rhs*TOKEN_W +: TOKEN_W]);
                        assert (out_destination_bank[lhs*3 +: 3]
                                != out_destination_bank[rhs*3 +: 3]);
                    end
                end
            end
        end
    end
endmodule

bind qfit_local5_source_qsilent_stat_router
    qfit_local5_source_qsilent_stat_assertions #(
        .TOKEN_W(TOKEN_W),
        .SCORE_W(SCORE_W)
    ) u_qfit_local5_source_qsilent_stat_assertions (.*);

`default_nettype wire
