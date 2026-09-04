`timescale 1ns/1ps
`default_nettype none

// One physical residual core selected by a descriptor mode bit.
module dual_mode_tare4_composite_top (
    input  logic          clk_core,
    input  logic          rst_core,

    input  logic          in_valid,
    output logic          in_ready,
    input  logic [15:0]   in_tag,
    input  logic          in_mode_local5,
    input  logic [31:0]   in_q_anchor,
    input  logic [31:0]   in_k_anchor,
    input  logic [31:0]   in_q_temporal_target,
    input  logic [31:0]   in_k_target,

    output logic          out_valid,
    input  logic          out_ready,
    output logic [15:0]   out_tag,
    output logic          out_mode_local5,
    output logic [1:0]    out_kind,
    output logic [5:0]    out_update_count,
    output logic [12:0]   out_raw16,
    output logic [8:0]    out_score_q7
);

    logic [5:0] motion_count;
    logic [9:0] selected_bias_raw16;
    logic [31:0] selected_q_target;

    always_comb begin
        motion_count = 6'd0;
        if (!in_mode_local5) begin
            for (int lane = 32'd0; lane < 32; lane = lane + 32'd1) begin
                if (in_k_anchor[lane] ^ in_k_target[lane]) begin
                    motion_count = motion_count + 6'd1;
                end
            end
        end
    end

    assign selected_q_target =
        in_mode_local5 ? in_q_anchor : in_q_temporal_target;
    assign selected_bias_raw16 =
        in_mode_local5 ? 10'd0 : {motion_count, 4'b0000};

    tare4_residual_composite_core core (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_tag(in_tag),
        .in_q_anchor(in_q_anchor),
        .in_k_anchor(in_k_anchor),
        .in_q_target(selected_q_target),
        .in_k_target(in_k_target),
        .in_bias_raw16(selected_bias_raw16),
        .in_mode_meta(in_mode_local5),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_tag(out_tag),
        .out_mode_meta(out_mode_local5),
        .out_kind(out_kind),
        .out_update_count(out_update_count),
        .out_raw16(out_raw16),
        .out_score_q7(out_score_q7)
    );

endmodule

`default_nettype wire
