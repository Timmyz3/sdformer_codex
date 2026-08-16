`timescale 1ns/1ps
`default_nettype none

module h67_score_finalize_pair_baseline (
    input  logic [5:0] overlap0,
    input  logic [5:0] same_zero0,
    input  logic [5:0] overlap1,
    input  logic [5:0] same_zero1,
    input  logic [5:0] motion,
    output logic [8:0] score0,
    output logic [8:0] score1,
    output logic       score_equal
);
    function automatic logic [8:0] finalize_score(
        input logic [5:0] overlap_count,
        input logic [5:0] same_zero_count,
        input logic [5:0] motion_count
    );
        logic [8:0] integer_part;
        logic increment;
        begin
            integer_part = ({3'b000, overlap_count} << 2)
                         + {3'b000, motion_count}
                         + {7'b0, same_zero_count[5:4]};
            increment = (same_zero_count[3:0] > 4'd8)
                     || (
                         same_zero_count[3:0] == 4'd8
                         && integer_part[0]
                     );
            finalize_score = integer_part + increment;
        end
    endfunction

    assign score0 = finalize_score(overlap0, same_zero0, motion);
    assign score1 = finalize_score(overlap1, same_zero1, motion);
    assign score_equal = score0 == score1;
endmodule

module h67_motion_cancelled_quotient_candidate (
    input  logic [5:0] overlap0,
    input  logic [5:0] same_zero0,
    input  logic [5:0] overlap1,
    input  logic [5:0] same_zero1,
    input  logic [5:0] motion,
    output logic [8:0] score0,
    output logic [8:0] score1,
    output logic       score_equal
);
    logic [8:0] reduced0;
    logic [8:0] reduced1;
    logic increment0;
    logic increment1;

    assign increment0 = (same_zero0[3:0] > 4'd8)
                     || (
                         same_zero0[3:0] == 4'd8
                         && (
                             same_zero0[4]
                             ^ motion[0]
                         )
                     );
    assign increment1 = (same_zero1[3:0] > 4'd8)
                     || (
                         same_zero1[3:0] == 4'd8
                         && (
                             same_zero1[4]
                             ^ motion[0]
                         )
                     );
    assign reduced0 = ({3'b000, overlap0} << 2)
                    + {7'b0, same_zero0[5:4]}
                    + increment0;
    assign reduced1 = ({3'b000, overlap1} << 2)
                    + {7'b0, same_zero1[5:4]}
                    + increment1;
    assign score_equal = reduced0 == reduced1;
    assign score0 = reduced0 + {3'b000, motion};
    assign score1 = reduced1 + {3'b000, motion};
endmodule

module h67_mssb5_quotient_baseline_openproxy (
    input  logic [63:0] q_pair,
    input  logic [63:0] k_pair,
    output logic [8:0]  score0,
    output logic [8:0]  score1,
    output logic        score_equal
);
    logic [5:0] overlap0;
    logic [5:0] same_zero0;
    logic [5:0] overlap1;
    logic [5:0] same_zero1;
    logic [5:0] motion;
    logic signed [15:0] score0_full;
    logic signed [15:0] score1_full;

    h67_mssb5_score_pair u_score (
        .q_pair,
        .k_pair,
        .overlap0,
        .same_zero0,
        .overlap1,
        .same_zero1,
        .motion,
        .score0_q7(score0_full),
        .score1_q7(score1_full)
    );
    assign score0 = score0_full[8:0];
    assign score1 = score1_full[8:0];
    assign score_equal = score0 == score1;
endmodule

module h67_mssb5_motion_cancelled_openproxy (
    input  logic [63:0] q_pair,
    input  logic [63:0] k_pair,
    output logic [8:0]  score0,
    output logic [8:0]  score1,
    output logic        score_equal
);
    logic [5:0] overlap0;
    logic [5:0] same_zero0;
    logic [5:0] overlap1;
    logic [5:0] same_zero1;
    logic [5:0] motion;
    logic signed [15:0] unused_score0;
    logic signed [15:0] unused_score1;

    h67_mssb5_score_pair u_stats (
        .q_pair,
        .k_pair,
        .overlap0,
        .same_zero0,
        .overlap1,
        .same_zero1,
        .motion,
        .score0_q7(unused_score0),
        .score1_q7(unused_score1)
    );
    h67_motion_cancelled_quotient_candidate u_finalize (
        .overlap0,
        .same_zero0,
        .overlap1,
        .same_zero1,
        .motion,
        .score0,
        .score1,
        .score_equal
    );
endmodule

`default_nettype wire
