`include "nts07_pkg.vh"

// ============================================================
// Unified ATLIF-PSN Inference Encoder (DATE 2027)
// Single comparator + membrane datapath for both binary/ternary modes.
//
// Inference-frozen dynamics:
//   1) Membrane leak:   u_leak = u * leak_q8 >>> 8   (leak ≈ 15/16)
//   2) Integrate:       u_new  = u_leak + input_acc  (input from Sparse MAC)
//   3) Fire check:
//        pos_fire = (u_new >= pos_thresh)
//        neg_fire = ternary_en & (u_new <= neg_thresh)
//   4) Reset:          u = pos_fire | neg_fire ? RESET_VAL : u_new
//
// ternary_en = 1 → 2-bit {-1,0,+1} for Q/K, downsample.sn
// ternary_en = 0 → 1-bit {0,+1} for MLP/proj/patch/decoder
// ============================================================
module atlif_unified_encode_unit #(
    parameter integer ACT_W      = `NTS07_ACT_W,
    parameter integer THRESH_W   = `NTS07_THRESH_W,
    parameter integer MEMBRANE_W = `NTS07_MEMBRANE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,             // Compute enable
    input  wire                         acc_clear,      // Clear membrane (timestep start / new neuron)
    input  wire                         ternary_en,     // 0=binary, 1=ternary
    input  wire signed [ACT_W-1:0]      input_acc,      // Input from MAC (pre-synaptic activation)
    input  wire signed [THRESH_W-1:0]   pos_thresh,     // Frozen positive threshold (APB/LUT)
    input  wire signed [THRESH_W-1:0]   neg_thresh,     // Frozen negative threshold (ternary only)
    output reg  [1:0]                   spike_out,      // Packed ternary: SILENT/POS/NEG
    output reg                          binary_out,     // Binary spike (pos_fire, for Sparse MAC)
    output wire                         pos_fire,
    output wire                         neg_fire
);
    reg signed [MEMBRANE_W-1:0] u;
    reg signed [MEMBRANE_W-1:0] u_leak;
    reg signed [MEMBRANE_W-1:0] u_integrated;

    wire signed [MEMBRANE_W+8-1:0] u_leak_full;

    // Leak multiplication (Q0.8 fixed-point): u * leak >>> 8
    // Using leak = 240/256 = 15/16 → approximation: u - (u >>> 4)
    // Synthesis can replace constant multiply with shift+sub for area efficiency
    assign u_leak_full = $signed(u) * $signed(`NTS07_LEAK_Q8);

    always @* begin
        u_leak = u_leak_full >>> 8;
        u_integrated = u_leak + {{(MEMBRANE_W-ACT_W){input_acc[ACT_W-1]}}, input_acc};
    end

    assign pos_fire = (u_integrated >= $signed({{(MEMBRANE_W-THRESH_W){pos_thresh[THRESH_W-1]}}, pos_thresh}));
    assign neg_fire = ternary_en & (u_integrated <= $signed({{(MEMBRANE_W-THRESH_W){neg_thresh[THRESH_W-1]}}, neg_thresh}));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            u <= {MEMBRANE_W{1'b0}};
            spike_out  <= `TERN_SILENT;
            binary_out <= 1'b0;
        end else if (acc_clear) begin
            u <= {MEMBRANE_W{1'b0}};
            spike_out  <= `TERN_SILENT;
            binary_out <= 1'b0;
        end else if (en) begin
            // Fire and reset
            if (pos_fire) begin
                u <= `NTS07_RESET_VAL;
                spike_out  <= `TERN_POS;
                binary_out <= 1'b1;
            end else if (neg_fire) begin
                u <= `NTS07_RESET_VAL;
                spike_out  <= `TERN_NEG;
                binary_out <= 1'b0;
            end else begin
                u <= u_integrated;
                spike_out  <= `TERN_SILENT;
                binary_out <= 1'b0;
            end
        end
    end

endmodule


// ============================================================
// Thin wrapper: ternary-only (ternary_en=1), for Q/K paths
// ============================================================
module ternary_encode_unit #(
    parameter integer ACT_W      = `NTS07_ACT_W,
    parameter integer THRESH_W   = `NTS07_THRESH_W,
    parameter integer MEMBRANE_W = `NTS07_MEMBRANE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire                         acc_clear,
    input  wire signed [ACT_W-1:0]      input_acc,
    input  wire signed [THRESH_W-1:0]   pos_thresh,
    input  wire signed [THRESH_W-1:0]   neg_thresh,
    output wire [1:0]                   ternary_out,
    output wire                         binary_unused
);
    atlif_unified_encode_unit #(
        .ACT_W(ACT_W), .THRESH_W(THRESH_W), .MEMBRANE_W(MEMBRANE_W)
    ) u_tern (
        .clk(clk), .rst_n(rst_n), .en(en), .acc_clear(acc_clear),
        .ternary_en(1'b1),
        .input_acc(input_acc),
        .pos_thresh(pos_thresh), .neg_thresh(neg_thresh),
        .spike_out(ternary_out), .binary_out(binary_unused),
        .pos_fire(), .neg_fire()
    );
endmodule


// ============================================================
// Thin wrapper: binary-only (ternary_en=0), for all_non_qk paths
// ============================================================
module binary_encode_unit #(
    parameter integer ACT_W      = `NTS07_ACT_W,
    parameter integer THRESH_W   = `NTS07_THRESH_W,
    parameter integer MEMBRANE_W = `NTS07_MEMBRANE_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire                         acc_clear,
    input  wire signed [ACT_W-1:0]      input_acc,
    input  wire signed [THRESH_W-1:0]   pos_thresh,
    output wire                         binary_out,
    output wire [1:0]                   ternary_unused
);
    wire neg_unused;
    atlif_unified_encode_unit #(
        .ACT_W(ACT_W), .THRESH_W(THRESH_W), .MEMBRANE_W(MEMBRANE_W)
    ) u_bin (
        .clk(clk), .rst_n(rst_n), .en(en), .acc_clear(acc_clear),
        .ternary_en(1'b0),
        .input_acc(input_acc),
        .pos_thresh(pos_thresh), .neg_thresh({THRESH_W{1'b0}}),
        .spike_out(ternary_unused), .binary_out(binary_out),
        .pos_fire(), .neg_fire(neg_unused)
    );
endmodule


// ============================================================
// Parallel encoding lane array: all channels of one neuron in parallel
// Typically 32 lanes for Q/K head_dim, or 96/192/384/768 for other layers
// ============================================================
module atlif_encode_lane_array #(
    parameter integer LANES    = `NTS07_HEAD_DIM,
    parameter integer ACT_W    = `NTS07_ACT_W,
    parameter integer THRESH_W = `NTS07_THRESH_W
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire                         acc_clear,
    input  wire                         ternary_en,
    input  wire signed [ACT_W-1:0]      act_in      [0:LANES-1],
    input  wire signed [THRESH_W-1:0]   pos_thresh,
    input  wire signed [THRESH_W-1:0]   neg_thresh,
    output wire [1:0]                   tern_out    [0:LANES-1],
    output wire                         bin_out     [0:LANES-1]
);
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_lane
            atlif_unified_encode_unit #(
                .ACT_W(ACT_W), .THRESH_W(THRESH_W)
            ) u_lane (
                .clk(clk), .rst_n(rst_n), .en(en), .acc_clear(acc_clear),
                .ternary_en(ternary_en),
                .input_acc(act_in[i]),
                .pos_thresh(pos_thresh), .neg_thresh(neg_thresh),
                .spike_out(tern_out[i]), .binary_out(bin_out[i]),
                .pos_fire(), .neg_fire()
            );
        end
    endgenerate
endmodule
