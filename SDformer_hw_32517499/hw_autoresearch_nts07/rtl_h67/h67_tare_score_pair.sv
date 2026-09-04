`timescale 1ns/1ps
`default_nettype none

// Exact temporal-anchor residual score pair. One 32-lane alpha-XNOR engine is
// reused for the anchor and dense target replay; sparse target work is handled
// by RESIDUAL_W selected update lanes. T0/T1 scores retire atomically.
module h67_tare_score_pair #(
    parameter int HEAD_DIM = 32,
    parameter int RESIDUAL_W = 16,
    parameter int TAG_W = 8,
    parameter int SCORE_W = 16
) (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     window_start,

    input  logic                     in_valid,
    input  logic                     in_enable,
    output logic                     in_ready,
    input  logic [TAG_W-1:0]         in_tag,
    input  logic [2*HEAD_DIM-1:0]    in_q_pair,
    input  logic [2*HEAD_DIM-1:0]    in_k_pair,

    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [TAG_W-1:0]         out_tag,
    output logic signed [SCORE_W-1:0] out_score0_q7,
    output logic signed [SCORE_W-1:0] out_score1_q7,
    output logic [1:0]               out_k_active,
    output logic [5:0]               out_update_count,
    output logic                     out_dense_fallback,
    output logic signed [12:0]       out_delta_raw16,
    output logic                     protocol_error
);
    localparam int LANE_ID_W = 5;

    logic [HEAD_DIM-1:0] q0;
    logic [HEAD_DIM-1:0] q1;
    logic [HEAD_DIM-1:0] k0;
    logic [HEAD_DIM-1:0] k1;
    logic [HEAD_DIM-1:0] update_mask;
    logic [HEAD_DIM-1:0] remaining_mask;
    logic [RESIDUAL_W-1:0] lane_valid;
    logic [(RESIDUAL_W*LANE_ID_W)-1:0] lane_ids;
    logic [RESIDUAL_W-1:0] way_found;
    logic [5:0] update_count;
    logic dense_comb;

    logic replay_q;
    logic [TAG_W-1:0] replay_tag_q;
    logic [HEAD_DIM-1:0] replay_q1_q;
    logic [HEAD_DIM-1:0] replay_k1_q;
    logic [9:0] replay_bias_raw16_q;
    logic [12:0] replay_raw0_q;
    logic [1:0] replay_k_active_q;
    logic [5:0] replay_update_count_q;

    logic [HEAD_DIM-1:0] engine_q;
    logic [HEAD_DIM-1:0] engine_k;
    logic [11:0] engine_alpha_raw16;
    logic [5:0] motion_count;
    logic [9:0] motion_bias_raw16;
    logic [12:0] anchor_raw16;
    logic [12:0] replay_target_raw16;
    logic signed [12:0] sparse_delta_raw16;
    logic signed [13:0] sparse_target_raw16;
    logic [7:0] old_lane_raw;
    logic [7:0] new_lane_raw;
    logic [LANE_ID_W-1:0] selected_lane;

    function automatic logic [11:0] alpha_xnor_raw16(
        input logic [HEAD_DIM-1:0] q_bits,
        input logic [HEAD_DIM-1:0] k_bits
    );
        logic [11:0] total;
        begin
            total = '0;
            for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                if (q_bits[lane] && k_bits[lane])
                    total = total + 12'd64;
                else if (!q_bits[lane] && !k_bits[lane])
                    total = total + 12'd1;
            end
            alpha_xnor_raw16 = total;
        end
    endfunction

    function automatic logic [8:0] rne_div16(input logic [12:0] raw16);
        logic [8:0] quotient;
        logic [3:0] remainder;
        logic increment;
        begin
            quotient = raw16[12:4];
            remainder = raw16[3:0];
            increment = remainder > 4'd8
                      || (remainder == 4'd8 && quotient[0]);
            rne_div16 = quotient + 9'(increment);
        end
    endfunction

    function automatic logic [7:0] lane_raw(
        input logic q_bit,
        input logic k_bit
    );
        begin
            if (q_bit && k_bit)
                lane_raw = 8'd64;
            else if (!q_bit && !k_bit)
                lane_raw = 8'd1;
            else
                lane_raw = 8'd0;
        end
    endfunction

    initial begin
        if (HEAD_DIM != 32)
            $error("h67_tare_score_pair currently requires HEAD_DIM=32");
        if (RESIDUAL_W != 8 && RESIDUAL_W != 16)
            $error("RESIDUAL_W must be 8 or 16");
    end

    assign q0 = in_q_pair[HEAD_DIM-1:0];
    assign q1 = in_q_pair[2*HEAD_DIM-1:HEAD_DIM];
    assign k0 = in_k_pair[HEAD_DIM-1:0];
    assign k1 = in_k_pair[2*HEAD_DIM-1:HEAD_DIM];
    assign update_mask = (q0 ^ q1) | (k0 ^ k1);

    always_comb begin
        update_count = '0;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
            update_count = update_count + 6'(update_mask[lane]);

        remaining_mask = update_mask;
        lane_valid = '0;
        lane_ids = '0;
        way_found = '0;
        for (int way = 0; way < RESIDUAL_W; way = way + 1) begin
            for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                if (!way_found[way] && remaining_mask[lane]) begin
                    way_found[way] = 1'b1;
                    lane_valid[way] = 1'b1;
                    lane_ids[(way*LANE_ID_W) +: LANE_ID_W] = LANE_ID_W'(lane);
                    remaining_mask[lane] = 1'b0;
                end
            end
        end
        dense_comb = 32'(update_count) > RESIDUAL_W;
    end

    always_comb begin
        motion_count = '0;
        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1)
            motion_count = motion_count + 6'(k0[lane] ^ k1[lane]);
        motion_bias_raw16 = {motion_count, 4'b0000};
    end

    assign engine_q = replay_q ? replay_q1_q : q0;
    assign engine_k = replay_q ? replay_k1_q : k0;
    assign engine_alpha_raw16 = alpha_xnor_raw16(engine_q, engine_k);
    assign anchor_raw16 = {1'b0, engine_alpha_raw16}
                        + {3'b000, motion_bias_raw16};
    assign replay_target_raw16 = {1'b0, engine_alpha_raw16}
                               + {3'b000, replay_bias_raw16_q};

    always_comb begin
        sparse_delta_raw16 = 13'sd0;
        selected_lane = '0;
        old_lane_raw = '0;
        new_lane_raw = '0;
        for (int way = 0; way < RESIDUAL_W; way = way + 1) begin
            selected_lane = lane_ids[(way*LANE_ID_W) +: LANE_ID_W];
            old_lane_raw = lane_raw(q0[selected_lane], k0[selected_lane]);
            new_lane_raw = lane_raw(q1[selected_lane], k1[selected_lane]);
            if (lane_valid[way]) begin
                sparse_delta_raw16 = sparse_delta_raw16
                    + $signed({5'b00000, new_lane_raw})
                    - $signed({5'b00000, old_lane_raw});
            end
        end
        sparse_target_raw16 = $signed({1'b0, anchor_raw16})
                            + sparse_delta_raw16;
    end

    // Sparse/zero packets use a combinational fall-through path. Only dense
    // packets occupy the replay register, avoiding one drain bubble per row.
    assign out_valid = replay_q || (in_valid && in_enable && !dense_comb);
    assign in_ready = in_enable && !replay_q && (dense_comb || out_ready);
    assign out_tag = replay_q ? replay_tag_q : in_tag;
    assign out_score0_q7 = replay_q
        ? SCORE_W'(rne_div16(replay_raw0_q))
        : SCORE_W'(rne_div16(anchor_raw16));
    assign out_score1_q7 = replay_q
        ? SCORE_W'(rne_div16(replay_target_raw16))
        : SCORE_W'(rne_div16(sparse_target_raw16[12:0]));
    assign out_k_active = replay_q ? replay_k_active_q : {|k1, |k0};
    assign out_update_count = replay_q ? replay_update_count_q : update_count;
    assign out_dense_fallback = replay_q;
    assign out_delta_raw16 = replay_q ? 13'sd0 : sparse_delta_raw16;

    always_ff @(posedge clk_core) begin
        if (rst_core || window_start) begin
            replay_q <= 1'b0;
            replay_tag_q <= '0;
            replay_q1_q <= '0;
            replay_k1_q <= '0;
            replay_bias_raw16_q <= '0;
            replay_raw0_q <= '0;
            replay_k_active_q <= '0;
            replay_update_count_q <= '0;
            protocol_error <= 1'b0;
        end else begin
            if (replay_q && out_ready) begin
                replay_q <= 1'b0;
            end
            if (in_valid && in_ready && dense_comb) begin
                replay_q <= 1'b1;
                replay_tag_q <= in_tag;
                replay_q1_q <= q1;
                replay_k1_q <= k1;
                replay_bias_raw16_q <= motion_bias_raw16;
                replay_raw0_q <= anchor_raw16;
                replay_k_active_q <= {|k1, |k0};
                replay_update_count_q <= update_count;
            end
            if (in_valid && in_ready && !dense_comb
                && (sparse_target_raw16 < 0 || sparse_target_raw16 > 14'd8191))
                protocol_error <= 1'b1;
        end
    end

endmodule

`default_nettype wire
