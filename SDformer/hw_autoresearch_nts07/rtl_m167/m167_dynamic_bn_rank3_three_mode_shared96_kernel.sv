`timescale 1ns/1ps
`default_nettype none

// M167: one shared signed-INT8 96-product pool for the three mutually
// exclusive phases of the Q8 dynamic-BN/rank-3 path.
//
//   FRONT   : 2 time rows x 16 lanes x 3 right factors.  A separate set of
//             32 signed square lanes maintains the simultaneous moment rate.
//   BACK    : 2 reconstructed rows x 16 lanes x 3 folded-left factors,
//             followed by bias/center-correct threshold decisions.
//   PREFOLD : 96 generic coefficient products/cycle; 640 products per hidden
//             group therefore require seven issue cycles at this boundary.
//
// FRONT and BACK occur on opposite sides of the current-batch BN barrier, and
// PREFOLD occurs once between them.  They never require simultaneous access to
// the main product pool.  The one-entry elastic result register permits an
// accepted issue every cycle when the consumer is ready.
//
// BACK carries the learned ATLIF nonzero amplitude with the packed event bits.
// A downstream fc2 implementation must either reconstruct {0, amplitude} or
// prove an exact weight/scale fold including its RNE, saturation and bias order.
module m167_dynamic_bn_rank3_three_mode_shared96_kernel #(
    parameter int TAG_BITS = 16
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         issue_valid,
    output logic                         issue_ready,
    input  logic [1:0]                   issue_mode,
    input  logic [TAG_BITS-1:0]          issue_tag,

    input  logic signed [7:0]            front_data [0:1][0:15],
    input  logic signed [7:0]            front_right_factor [0:2][0:1],

    input  logic signed [7:0]            back_rank_data [0:2][0:15],
    input  logic signed [7:0]            back_folded_left [0:1][0:2][0:15],
    input  logic signed [23:0]           back_folded_bias [0:1][0:15],
    input  logic signed [23:0]           back_threshold,

    input  logic signed [7:0]            prefold_a [0:95],
    input  logic signed [7:0]            prefold_b [0:95],

    output logic                         issue_accept,
    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [1:0]                   result_mode,
    output logic [TAG_BITS-1:0]          result_tag,

    output logic signed [16:0]           front_projection_delta [0:2][0:15],
    output logic signed [8:0]            front_moment_sum_delta [0:15],
    output logic [16:0]                  front_moment_sumsq_delta [0:15],

    output logic [31:0]                  back_event_bits,
    output logic signed [23:0]           back_event_amplitude,

    output logic signed [15:0]           prefold_product [0:95],
    output logic                         result_accept,

    output logic [95:0]                  main_product_active_mask,
    output logic [31:0]                  square_product_active_mask,
    output logic                         protocol_error,
    output logic                         busy
);
    localparam logic [1:0] MODE_FRONT = 2'd0;
    localparam logic [1:0] MODE_BACK = 2'd1;
    localparam logic [1:0] MODE_PREFOLD = 2'd2;

    logic fault_q;
    logic result_valid_q;
    logic [1:0] result_mode_q;
    logic [TAG_BITS-1:0] result_tag_q;
    logic signed [16:0] projection_delta_q [0:2][0:15];
    logic signed [8:0] moment_sum_delta_q [0:15];
    logic [16:0] moment_sumsq_delta_q [0:15];
    logic [31:0] event_bits_q;
    logic signed [23:0] event_amplitude_q;
    logic signed [15:0] prefold_product_q [0:95];

    logic signed [7:0] main_a [0:95];
    logic signed [7:0] main_b [0:95];
    wire signed [15:0] main_product [0:95];
    logic signed [7:0] square_a [0:31];
    logic signed [7:0] square_b [0:31];
    wire signed [15:0] square_product [0:31];

    logic legal_mode;
    logic illegal_request;

    assign legal_mode = issue_mode != 2'd3;
    assign illegal_request = issue_valid && !legal_mode;
    assign issue_ready = !fault_q && legal_mode
        && (!result_valid_q || result_ready);
    assign issue_accept = issue_valid && issue_ready;
    assign result_valid = result_valid_q;
    assign result_accept = result_valid_q && result_ready;
    assign result_mode = result_mode_q;
    assign result_tag = result_tag_q;
    assign back_event_bits = event_bits_q;
    assign back_event_amplitude = event_amplitude_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = result_valid_q;
    assign main_product_active_mask
        = issue_valid && legal_mode ? {96{1'b1}} : '0;
    assign square_product_active_mask
        = issue_valid && issue_mode == MODE_FRONT ? {32{1'b1}} : '0;

    generate
        for (genvar rank = 0; rank < 3; rank++) begin : g_projection_rank
            for (genvar lane = 0; lane < 16; lane++) begin : g_projection_lane
                assign front_projection_delta[rank][lane]
                    = projection_delta_q[rank][lane];
            end
        end
        for (genvar lane = 0; lane < 16; lane++) begin : g_moment_lane
            assign front_moment_sum_delta[lane] = moment_sum_delta_q[lane];
            assign front_moment_sumsq_delta[lane] = moment_sumsq_delta_q[lane];
        end
        for (genvar slot = 0; slot < 96; slot++) begin : g_main_product
            assign main_product[slot] = main_a[slot] * main_b[slot];
            assign prefold_product[slot] = prefold_product_q[slot];
        end
        for (genvar square_slot = 0; square_slot < 32;
                square_slot++) begin : g_square_product
            assign square_product[square_slot]
                = square_a[square_slot] * square_b[square_slot];
        end
    endgenerate

    always_comb begin : operand_select
        for (int slot = 0; slot < 96; slot++) begin
            main_a[slot] = '0;
            main_b[slot] = '0;
        end
        for (int slot = 0; slot < 32; slot++) begin
            square_a[slot] = '0;
            square_b[slot] = '0;
        end
        case (issue_mode)
            MODE_FRONT: begin
                for (int rank = 0; rank < 3; rank++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        for (int row = 0; row < 2; row++) begin
                            main_a[((rank*16+lane)*2)+row]
                                = front_data[row][lane];
                            main_b[((rank*16+lane)*2)+row]
                                = front_right_factor[rank][row];
                        end
                    end
                end
                for (int row = 0; row < 2; row++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        square_a[(row*16)+lane] = front_data[row][lane];
                        square_b[(row*16)+lane] = front_data[row][lane];
                    end
                end
            end
            MODE_BACK: begin
                for (int row = 0; row < 2; row++) begin
                    for (int lane = 0; lane < 16; lane++) begin
                        for (int rank = 0; rank < 3; rank++) begin
                            main_a[((row*16+lane)*3)+rank]
                                = back_rank_data[rank][lane];
                            main_b[((row*16+lane)*3)+rank]
                                = back_folded_left[row][rank][lane];
                        end
                    end
                end
            end
            MODE_PREFOLD: begin
                for (int slot = 0; slot < 96; slot++) begin
                    main_a[slot] = prefold_a[slot];
                    main_b[slot] = prefold_b[slot];
                end
            end
            default: begin
                // Illegal mode is rejected before any state can commit.
            end
        endcase
    end

    always_ff @(posedge clk_core) begin : result_register
        logic signed [25:0] reconstructed;
        logic [31:0] packed_events;
        if (rst_core) begin
            fault_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_mode_q <= '0;
            result_tag_q <= '0;
            event_bits_q <= '0;
            event_amplitude_q <= '0;
            for (int rank = 0; rank < 3; rank++)
                for (int lane = 0; lane < 16; lane++)
                    projection_delta_q[rank][lane] <= '0;
            for (int lane = 0; lane < 16; lane++) begin
                moment_sum_delta_q[lane] <= '0;
                moment_sumsq_delta_q[lane] <= '0;
            end
            for (int slot = 0; slot < 96; slot++)
                prefold_product_q[slot] <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;
            if (result_accept && !issue_accept)
                result_valid_q <= 1'b0;
            if (issue_accept) begin
                result_valid_q <= 1'b1;
                result_mode_q <= issue_mode;
                result_tag_q <= issue_tag;
                event_bits_q <= '0;
                event_amplitude_q <= '0;
                for (int rank = 0; rank < 3; rank++)
                    for (int lane = 0; lane < 16; lane++)
                        projection_delta_q[rank][lane] <= '0;
                for (int lane = 0; lane < 16; lane++) begin
                    moment_sum_delta_q[lane] <= '0;
                    moment_sumsq_delta_q[lane] <= '0;
                end
                for (int slot = 0; slot < 96; slot++)
                    prefold_product_q[slot] <= '0;

                case (issue_mode)
                    MODE_FRONT: begin
                        for (int rank = 0; rank < 3; rank++) begin
                            for (int lane = 0; lane < 16; lane++) begin
                                projection_delta_q[rank][lane] <= $signed({
                                    main_product[((rank*16+lane)*2)][15],
                                    main_product[((rank*16+lane)*2)]
                                }) + $signed({
                                    main_product[((rank*16+lane)*2)+1][15],
                                    main_product[((rank*16+lane)*2)+1]
                                });
                            end
                        end
                        for (int lane = 0; lane < 16; lane++) begin
                            moment_sum_delta_q[lane]
                                <= $signed({front_data[0][lane][7],
                                           front_data[0][lane]})
                                + $signed({front_data[1][lane][7],
                                           front_data[1][lane]});
                            moment_sumsq_delta_q[lane]
                                <= {1'b0, $unsigned(square_product[lane])}
                                + {1'b0, $unsigned(square_product[16+lane])};
                        end
                    end
                    MODE_BACK: begin
                        packed_events = '0;
                        for (int row = 0; row < 2; row++) begin
                            for (int lane = 0; lane < 16; lane++) begin
                                reconstructed = $signed({
                                    {2{back_folded_bias[row][lane][23]}},
                                    back_folded_bias[row][lane]
                                }) + $signed({
                                    {10{main_product[((row*16+lane)*3)][15]}},
                                    main_product[((row*16+lane)*3)]
                                }) + $signed({
                                    {10{main_product[((row*16+lane)*3)+1][15]}},
                                    main_product[((row*16+lane)*3)+1]
                                }) + $signed({
                                    {10{main_product[((row*16+lane)*3)+2][15]}},
                                    main_product[((row*16+lane)*3)+2]
                                });
                                packed_events[(row*16)+lane]
                                    = reconstructed >= $signed({
                                        {2{back_threshold[23]}},
                                        back_threshold
                                    });
                            end
                        end
                        event_bits_q <= packed_events;
                        event_amplitude_q <= back_threshold;
                    end
                    MODE_PREFOLD: begin
                        for (int slot = 0; slot < 96; slot++)
                            prefold_product_q[slot] <= main_product[slot];
                    end
                    default: begin
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
