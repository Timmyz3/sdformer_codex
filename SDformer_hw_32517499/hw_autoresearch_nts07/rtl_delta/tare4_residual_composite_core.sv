`timescale 1ns/1ps
`default_nettype none

// Shared temporal/topology-anchor residual engine. Bias is expressed in raw16 units.
module tare4_residual_composite_core (
    input  logic          clk_core,
    input  logic          rst_core,

    input  logic          in_valid,
    output logic          in_ready,
    input  logic [15:0]   in_tag,
    input  logic [31:0]   in_q_anchor,
    input  logic [31:0]   in_k_anchor,
    input  logic [31:0]   in_q_target,
    input  logic [31:0]   in_k_target,
    input  logic [9:0]    in_bias_raw16,
    input  logic          in_mode_meta,

    output logic          out_valid,
    input  logic          out_ready,
    output logic [15:0]   out_tag,
    output logic          out_mode_meta,
    output logic [1:0]    out_kind,
    output logic [5:0]    out_update_count,
    output logic [12:0]   out_raw16,
    output logic [8:0]    out_score_q7
);

    localparam int CLASS_PAYLOAD_W = 32'd151;

    logic class_in_valid;
    logic class_in_ready;
    logic [150:0] class_in_payload;
    logic class_out_valid;
    logic class_out_ready;
    logic [15:0] class_out_tag;
    logic [1:0] class_out_kind;
    logic [31:0] class_out_mask;
    logic [150:0] class_out_payload;
    logic [5:0] class_out_count;
    logic [3:0] class_out_lane_valid;
    logic [19:0] class_out_lane_ids;

    logic [31:0] class_q_anchor;
    logic [31:0] class_k_anchor;
    logic [31:0] class_q_target;
    logic [31:0] class_k_target;
    logic [11:0] class_anchor_raw;
    logic [9:0] class_bias_raw;
    logic class_mode_meta;

    logic replay_cycle;
    logic result_slot_ready;
    logic [31:0] engine_q;
    logic [31:0] engine_k;
    logic [11:0] engine_raw;
    logic [11:0] input_anchor_raw;

    logic signed [9:0] sparse_delta_raw;
    logic signed [12:0] sparse_anchor_signed;
    logic signed [12:0] sparse_delta_extended;
    logic signed [12:0] sparse_sum_signed;
    logic [12:0] dense_sum;
    logic signed [12:0] selected_raw_signed;
    logic [8:0] selected_quotient;
    logic [3:0] selected_remainder;
    logic selected_increment;
    logic [8:0] selected_score_q7;

    assign class_q_anchor = class_out_payload[149:118];
    assign class_k_anchor = class_out_payload[117:86];
    assign class_q_target = class_out_payload[85:54];
    assign class_k_target = class_out_payload[53:22];
    assign class_anchor_raw = class_out_payload[21:10];
    assign class_bias_raw = class_out_payload[9:0];
    assign class_mode_meta = class_out_payload[150];

    assign result_slot_ready = !out_valid || out_ready;
    assign replay_cycle =
        class_out_valid &&
        class_out_kind == 2'd2 &&
        result_slot_ready;

    assign in_ready = class_in_ready && !replay_cycle;
    assign class_in_valid = in_valid && !replay_cycle;
    assign class_out_ready = result_slot_ready;

    assign engine_q = replay_cycle ? class_q_target : in_q_anchor;
    assign engine_k = replay_cycle ? class_k_target : in_k_anchor;

    alpha_xnor_raw32 raw_engine (
        .q_bits(engine_q),
        .k_bits(engine_k),
        .raw16(engine_raw)
    );

    assign input_anchor_raw =
        engine_raw + {2'b00, in_bias_raw16};
    assign class_in_payload = {
        in_mode_meta,
        in_q_anchor,
        in_k_anchor,
        in_q_target,
        in_k_target,
        input_anchor_raw,
        in_bias_raw16
    };

    delta_bounded_classifier #(
        .TAG_W(16),
        .PAYLOAD_W(CLASS_PAYLOAD_W)
    ) classifier (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(class_in_valid),
        .in_ready(class_in_ready),
        .in_tag(in_tag),
        .in_delta_mask(
            (in_q_anchor ^ in_q_target) |
            (in_k_anchor ^ in_k_target)
        ),
        .in_payload(class_in_payload),
        .out_valid(class_out_valid),
        .out_ready(class_out_ready),
        .out_tag(class_out_tag),
        .out_kind(class_out_kind),
        .out_delta_mask(class_out_mask),
        .out_payload(class_out_payload),
        .out_count(class_out_count),
        .out_lane_valid(class_out_lane_valid),
        .out_lane_ids(class_out_lane_ids)
    );

    alpha_xnor_delta4 residual_engine (
        .lane_valid(class_out_lane_valid),
        .lane_ids(class_out_lane_ids),
        .q_old_bits(class_q_anchor),
        .k_old_bits(class_k_anchor),
        .q_new_bits(class_q_target),
        .k_new_bits(class_k_target),
        .delta_raw16(sparse_delta_raw)
    );

    always_comb begin
        sparse_anchor_signed = $signed({1'b0, class_anchor_raw});
        sparse_delta_extended = {
            {3{sparse_delta_raw[9]}},
            sparse_delta_raw
        };
        sparse_sum_signed =
            sparse_anchor_signed + sparse_delta_extended;
        dense_sum =
            {1'b0, engine_raw} + {3'b000, class_bias_raw};

        if (class_out_kind == 2'd2) begin
            selected_raw_signed = $signed(dense_sum);
        end else if (class_out_kind == 2'd1) begin
            selected_raw_signed = sparse_sum_signed;
        end else begin
            selected_raw_signed = $signed({1'b0, class_anchor_raw});
        end

        selected_quotient = selected_raw_signed[12:4];
        selected_remainder = selected_raw_signed[3:0];
        selected_increment =
            selected_remainder > 4'd8 ||
            (
                selected_remainder == 4'd8 &&
                selected_quotient[0]
            );
        selected_score_q7 =
            selected_quotient + 9'(selected_increment);
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            out_valid <= 1'b0;
            out_tag <= '0;
            out_mode_meta <= 1'b0;
            out_kind <= 2'd0;
            out_update_count <= '0;
            out_raw16 <= '0;
            out_score_q7 <= '0;
        end else if (result_slot_ready) begin
            out_valid <= class_out_valid;
            if (class_out_valid) begin
                out_tag <= class_out_tag;
                out_mode_meta <= class_mode_meta;
                out_kind <= class_out_kind;
                out_update_count <= class_out_count;
                out_raw16 <= selected_raw_signed;
                out_score_q7 <= selected_score_q7;
            end
        end
    end

endmodule

`default_nettype wire
