`default_nettype none

// Motion-shared sufficient-statistic butterfly (MSSB5).
// Five exact lane predicates are reduced in a fixed radix-2 tree:
// {Q0&K0, ~Q0&~K0, Q1&K1, ~Q1&~K1, K0^K1}.
// Each level grows its per-field width by one bit, and the shared motion count
// is broadcast to both score finalizers. There is no dynamic lane compactor.
module h67_mssb5_score_pair #(
    parameter int HEAD_DIM = 32,
    parameter int SCORE_W = 16,
    parameter int COUNT_W = $clog2(HEAD_DIM + 1)
) (
    input  logic [2*HEAD_DIM-1:0] q_pair,
    input  logic [2*HEAD_DIM-1:0] k_pair,
    output logic [COUNT_W-1:0] overlap0,
    output logic [COUNT_W-1:0] same_zero0,
    output logic [COUNT_W-1:0] overlap1,
    output logic [COUNT_W-1:0] same_zero1,
    output logic [COUNT_W-1:0] motion,
    output logic signed [SCORE_W-1:0] score0_q7,
    output logic signed [SCORE_W-1:0] score1_q7
);
    localparam int FIELDS = 5;
    localparam int RAW_W = COUNT_W + 3;

    logic [FIELDS-1:0] lane_stat [0:31];
    logic [2*FIELDS-1:0] level1 [0:15];
    logic [3*FIELDS-1:0] level2 [0:7];
    logic [4*FIELDS-1:0] level3 [0:3];
    logic [5*FIELDS-1:0] level4 [0:1];
    logic [6*FIELDS-1:0] level5;

    initial begin
        if (HEAD_DIM != 32 || COUNT_W != 6)
            $error("h67_mssb5_score_pair requires HEAD_DIM=32 and COUNT_W=6");
    end

    generate
        for (genvar lane = 0; lane < 32; lane = lane + 1) begin : g_lane
            assign lane_stat[lane][0] = q_pair[lane] & k_pair[lane];
            assign lane_stat[lane][1] = ~(q_pair[lane] | k_pair[lane]);
            assign lane_stat[lane][2] = q_pair[32+lane] & k_pair[32+lane];
            assign lane_stat[lane][3] = ~(q_pair[32+lane] | k_pair[32+lane]);
            assign lane_stat[lane][4] = k_pair[lane] ^ k_pair[32+lane];
        end

        for (genvar node1 = 0; node1 < 16; node1 = node1 + 1) begin : g_level1
            logic [2*FIELDS-1:0] lhs;
            logic [2*FIELDS-1:0] rhs;
            for (genvar field1 = 0; field1 < FIELDS; field1 = field1 + 1) begin : g_field
                assign lhs[2*field1 +: 2] = {1'b0, lane_stat[2*node1][field1]};
                assign rhs[2*field1 +: 2] = {1'b0, lane_stat[2*node1+1][field1]};
            end
            assign level1[node1] = lhs + rhs;
        end

        for (genvar node2 = 0; node2 < 8; node2 = node2 + 1) begin : g_level2
            logic [3*FIELDS-1:0] lhs;
            logic [3*FIELDS-1:0] rhs;
            for (genvar field2 = 0; field2 < FIELDS; field2 = field2 + 1) begin : g_field
                assign lhs[3*field2 +: 3] = {1'b0, level1[2*node2][2*field2 +: 2]};
                assign rhs[3*field2 +: 3] = {1'b0, level1[2*node2+1][2*field2 +: 2]};
            end
            assign level2[node2] = lhs + rhs;
        end

        for (genvar node3 = 0; node3 < 4; node3 = node3 + 1) begin : g_level3
            logic [4*FIELDS-1:0] lhs;
            logic [4*FIELDS-1:0] rhs;
            for (genvar field3 = 0; field3 < FIELDS; field3 = field3 + 1) begin : g_field
                assign lhs[4*field3 +: 4] = {1'b0, level2[2*node3][3*field3 +: 3]};
                assign rhs[4*field3 +: 4] = {1'b0, level2[2*node3+1][3*field3 +: 3]};
            end
            assign level3[node3] = lhs + rhs;
        end

        for (genvar node4 = 0; node4 < 2; node4 = node4 + 1) begin : g_level4
            logic [5*FIELDS-1:0] lhs;
            logic [5*FIELDS-1:0] rhs;
            for (genvar field4 = 0; field4 < FIELDS; field4 = field4 + 1) begin : g_field
                assign lhs[5*field4 +: 5] = {1'b0, level3[2*node4][4*field4 +: 4]};
                assign rhs[5*field4 +: 5] = {1'b0, level3[2*node4+1][4*field4 +: 4]};
            end
            assign level4[node4] = lhs + rhs;
        end

        begin : g_level5
            logic [6*FIELDS-1:0] lhs;
            logic [6*FIELDS-1:0] rhs;
            for (genvar field5 = 0; field5 < FIELDS; field5 = field5 + 1) begin : g_field
                assign lhs[6*field5 +: 6] = {1'b0, level4[0][5*field5 +: 5]};
                assign rhs[6*field5 +: 6] = {1'b0, level4[1][5*field5 +: 5]};
            end
            assign level5 = lhs + rhs;
        end
    endgenerate

    assign overlap0 = level5[0*COUNT_W +: COUNT_W];
    assign same_zero0 = level5[1*COUNT_W +: COUNT_W];
    assign overlap1 = level5[2*COUNT_W +: COUNT_W];
    assign same_zero1 = level5[3*COUNT_W +: COUNT_W];
    assign motion = level5[4*COUNT_W +: COUNT_W];

    function automatic logic signed [SCORE_W-1:0] finalize_score(
        input logic [COUNT_W-1:0] overlap_count,
        input logic [COUNT_W-1:0] same_zero_count,
        input logic [COUNT_W-1:0] motion_count
    );
        logic [COUNT_W-1:0] silence_integer;
        logic [3:0] silence_remainder;
        logic [RAW_W-1:0] score_integer;
        logic silence_increment;
        logic [RAW_W-1:0] score_unsigned;
        begin
            silence_integer = same_zero_count >> 4;
            silence_remainder = same_zero_count[3:0];
            score_integer = (RAW_W'(overlap_count) << 2)
                          + RAW_W'(motion_count)
                          + RAW_W'(silence_integer);
            silence_increment = (silence_remainder > 4'd8)
                             || ((silence_remainder == 4'd8) && score_integer[0]);
            score_unsigned = score_integer + RAW_W'(silence_increment);
            finalize_score = SCORE_W'(score_unsigned);
        end
    endfunction

    assign score0_q7 = finalize_score(overlap0, same_zero0, motion);
    assign score1_q7 = finalize_score(overlap1, same_zero1, motion);
endmodule

`default_nettype wire
