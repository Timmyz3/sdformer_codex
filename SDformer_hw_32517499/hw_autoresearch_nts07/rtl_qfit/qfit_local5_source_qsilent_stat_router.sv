`timescale 1ns/1ps
`default_nettype none

// Source-owned exact sufficient-statistic producer for Local5 Q==0 edges.
// One K word produces one score and up to five fixed-topology destinations.
// The five outputs are an atomic multicast descriptor, not five FIFO entries.
module qfit_local5_source_qsilent_stat_router #(
    parameter int TOKENS = 450,
    parameter int PLANE_TOKENS = 225,
    parameter int SIDE = 15,
    parameter int TOKEN_W = 9,
    parameter int SCORE_W = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,
    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic [TOKEN_W-1:0]           in_source_token,
    input  logic [3:0]                   in_source_y,
    input  logic [3:0]                   in_source_x,
    input  logic [31:0]                  in_source_k,
    input  logic [4:0]                   in_consumer_qzero_mask,
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [TOKEN_W-1:0]           out_source_token,
    output logic signed [SCORE_W-1:0]    out_score_q7,
    output logic [4:0]                   out_consumer_valid,
    output logic [5*TOKEN_W-1:0]         out_destination,
    output logic [5*3-1:0]               out_destination_bank
);

    logic valid_q;
    logic [TOKEN_W-1:0] source_q;
    logic signed [SCORE_W-1:0] score_q;
    logic [4:0] consumer_q;
    logic [5*TOKEN_W-1:0] destination_q;
    logic [5*3-1:0] bank_q;
    logic [5:0] pop_w;
    logic signed [12:0] raw_w;
    logic signed [SCORE_W-1:0] score_w;
    logic [4:0] geometry_valid_w;
    logic [5*TOKEN_W-1:0] destination_w;
    logic [5*3-1:0] bank_w;

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

    function automatic logic [2:0] mod5_small(
        input logic [5:0] value
    );
        logic [5:0] reduced;
        begin
            reduced = value;
            for (int step = 0; step < 8; step = step + 1)
                if (reduced >= 6'd5)
                    reduced = reduced - 6'd5;
            mod5_small = reduced[2:0];
        end
    endfunction

    always_comb begin
        logic [TOKEN_W-1:0] destination;
        logic [2:0] source_bank;
        logic [5:0] bank_sum;

        pop_w = popcount32(in_source_k);
        raw_w = 13'sd32 - $signed({7'b0, pop_w});
        score_w = rne_q7(raw_w);
        geometry_valid_w = 5'b00001;
        geometry_valid_w[1] = (in_source_y < SIDE-1);
        geometry_valid_w[2] = (in_source_y > 0);
        geometry_valid_w[3] = (in_source_x < SIDE-1);
        geometry_valid_w[4] = (in_source_x > 0);
        source_bank = mod5_small({2'b00, in_source_x}
                    + ({2'b00, in_source_y} << 1));

        for (int role = 0; role < 5; role = role + 1) begin
            destination = in_source_token;
            unique case (role)
                1: destination = in_source_token + TOKEN_W'(SIDE);
                2: destination = in_source_token - TOKEN_W'(SIDE);
                3: destination = in_source_token + TOKEN_W'(1);
                4: destination = in_source_token - TOKEN_W'(1);
                default: ;
            endcase
            destination_w[role*TOKEN_W +: TOKEN_W] = destination;
            unique case (role)
                1: bank_sum = {3'b000, source_bank} + 6'd2;
                2: bank_sum = {3'b000, source_bank} + 6'd3;
                3: bank_sum = {3'b000, source_bank} + 6'd1;
                4: bank_sum = {3'b000, source_bank} + 6'd4;
                default: bank_sum = {3'b000, source_bank};
            endcase
            bank_w[role*3 +: 3] = mod5_small(bank_sum);
        end
    end

    assign in_ready = !valid_q || out_ready;
    assign out_valid = valid_q;
    assign out_source_token = source_q;
    assign out_score_q7 = score_q;
    assign out_consumer_valid = consumer_q;
    assign out_destination = destination_q;
    assign out_destination_bank = bank_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            valid_q <= 1'b0;
            source_q <= '0;
            score_q <= '0;
            consumer_q <= '0;
            destination_q <= '0;
            bank_q <= '0;
        end else if (in_ready) begin
            valid_q <= in_valid;
            if (in_valid) begin
                source_q <= in_source_token;
                score_q <= score_w;
                consumer_q <= in_consumer_qzero_mask & geometry_valid_w;
                destination_q <= destination_w;
                bank_q <= bank_w;
            end
        end
    end
endmodule

`default_nettype wire
