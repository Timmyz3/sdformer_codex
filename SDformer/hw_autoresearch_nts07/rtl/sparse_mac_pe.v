`include "nts07_pkg.vh"

// ============================================================
// Sparse MAC Processing Element (PE) for binary/ternary spikes
// Computes: acc += spike ? weight : 0
// Zero-skip: if spike is silent (0), no accumulation (saves power)
//
// For binary spikes: spike_in is 1-bit {0,1}
// For ternary spikes: pos spike → add weight, neg spike → sub weight
// The sign is handled at cluster level by inverting weight bits.
// ============================================================
module sparse_mac_pe #(
    parameter integer WGT_W = `NTS07_WGT_W,
    parameter integer ACC_W = `NTS07_ACC_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,             // PE enable (clock gate)
    input  wire                     spike_in,       // 1-bit: 1=fire (pos or neg per ctrl)
    input  wire                     neg_spike,      // 1=negative spike (sub weight)
    input  wire signed [WGT_W-1:0]  weight,
    input  wire                     acc_clear,      // Clear accumulator
    output reg  signed [ACC_W-1:0]  acc_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= {ACC_W{1'b0}};
        end else if (acc_clear) begin
            acc_out <= {ACC_W{1'b0}};
        end else if (en && spike_in) begin
            if (neg_spike)
                acc_out <= acc_out - {{(ACC_W-WGT_W){weight[WGT_W-1]}}, weight};
            else
                acc_out <= acc_out + {{(ACC_W-WGT_W){weight[WGT_W-1]}}, weight};
        end
    end
endmodule


// ============================================================
// Sparse MAC Lane: N PEs processing one output channel in parallel,
// each taking one input channel spike/weight pair.
// Zero-skips at PE level: silent spikes consume no switching power.
// ============================================================
module sparse_mac_lane #(
    parameter integer IN_DIM   = 32,     // Input channels per cycle
    parameter integer WGT_W    = `NTS07_WGT_W,
    parameter integer ACC_W    = `NTS07_ACC_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     fire,           // Compute enable
    input  wire [IN_DIM-1:0]        spike_vec,      // Binary spikes
    input  wire [IN_DIM-1:0]        neg_mask,       // Which spikes are negative
    input  wire signed [IN_DIM*WGT_W-1:0] weight_vec,
    input  wire                     acc_clear,
    output wire signed [ACC_W-1:0]  acc_out
);
    wire signed [ACC_W-1:0] pe_acc [0:IN_DIM-1];
    reg signed [ACC_W-1:0]  acc_sum;
    integer i;

    genvar g;
    generate
        for (g = 0; g < IN_DIM; g = g + 1) begin : gen_pe
            sparse_mac_pe #(.WGT_W(WGT_W), .ACC_W(ACC_W)) u_pe (
                .clk(clk), .rst_n(rst_n),
                .en(fire),
                .spike_in(spike_vec[g]),
                .neg_spike(neg_mask[g]),
                .weight(weight_vec[g*WGT_W +: WGT_W]),
                .acc_clear(acc_clear),
                .acc_out(pe_acc[g])
            );
        end
    endgenerate

    // Sum all PE partial sums via pipelined adder (simple combinational for paper;
    // synthesis will retime. In production this becomes an adder tree.)
    always @* begin
        acc_sum = {ACC_W{1'b0}};
        for (i = 0; i < IN_DIM; i = i + 1)
            acc_sum = acc_sum + pe_acc[i];
    end
    assign acc_out = acc_sum;

endmodule


// ============================================================
// Sparse MAC Cluster: OUT_DIM lanes processing OUT_DIM output channels
// in parallel (e.g., 32 outputs = 32 lanes).
// Supports:
//   - Binary ATLIF spikes (1-bit spike_vec)
//   - Ternary ATLIF spikes (spike_vec + neg_mask; NEG flips weight sign)
//   - Zero-skip per-PE (no switching when spike=0)
//   - Accumulator clear per new output neuron
// ============================================================
module sparse_mac_cluster #(
    parameter integer IN_DIM   = 32,
    parameter integer OUT_DIM  = 32,
    parameter integer WGT_W    = `NTS07_WGT_W,
    parameter integer ACC_W    = `NTS07_ACC_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    // Control
    input  wire                     start,
    output reg                      done,
    input  wire                     is_ternary,     // Ternary input (use neg_mask)
    input  wire [9:0]               n_inputs,       // Number of input channels to accumulate
    // Streaming input: IN_DIM spikes + weights per cycle
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [IN_DIM-1:0]        spike_vec,
    input  wire [IN_DIM-1:0]        neg_mask,       // Valid when is_ternary=1
    input  wire signed [OUT_DIM*IN_DIM*WGT_W-1:0] weight_block, // OUT×IN weights
    input  wire                     acc_clear,      // Clear all accumulators
    // Output: OUT_DIM accumulated values (one per output channel)
    output reg                      out_valid,
    output wire signed [ACC_W-1:0]  acc_out [0:OUT_DIM-1]
);
    localparam [1:0] ST_IDLE = 2'd0, ST_ACC = 2'd1, ST_DONE = 2'd2;
    reg [1:0] state;
    reg [9:0] cycle_cnt;
    integer oc, ic;

    // Per-output lane
    wire signed [ACC_W-1:0] lane_acc [0:OUT_DIM-1];
    reg lane_clear;

    assign in_ready = (state == ST_ACC);

    genvar o;
    generate
        for (o = 0; o < OUT_DIM; o = o + 1) begin : gen_lane
            sparse_mac_lane #(.IN_DIM(IN_DIM), .WGT_W(WGT_W), .ACC_W(ACC_W))
                u_lane (
                    .clk(clk), .rst_n(rst_n),
                    .fire(in_valid & in_ready),
                    .spike_vec(spike_vec),
                    .neg_mask(is_ternary ? neg_mask : {IN_DIM{1'b0}}),
                    .weight_vec(weight_block[o*IN_DIM*WGT_W +: IN_DIM*WGT_W]),
                    .acc_clear(lane_clear),
                    .acc_out(lane_acc[o])
                );
            assign acc_out[o] = lane_acc[o];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            out_valid <= 1'b0;
            lane_clear <= 1'b0;
            cycle_cnt <= 0;
        end else begin
            lane_clear <= 1'b0;
            out_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        lane_clear <= 1'b1;
                        cycle_cnt <= 0;
                        state <= ST_ACC;
                    end
                end

                ST_ACC: begin
                    if (in_valid) begin
                        if (cycle_cnt >= (n_inputs + IN_DIM - 1) / IN_DIM - 1) begin
                            state <= ST_DONE;
                            out_valid <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            cycle_cnt <= cycle_cnt + 1;
                        end
                    end
                end

                ST_DONE: begin
                    if (!start)
                        state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


// ============================================================
// K-Gate Unit: applies gate[i][j] * K[j][d] and accumulates for output
// This is the value weighting step inside H60, implemented as
// a gated-accumulator (multiply by Q0.8 gate, shift right 8).
// ============================================================
module k_gate_unit #(
    parameter integer HEAD_DIM = `NTS07_HEAD_DIM,
    parameter integer ACT_W    = `NTS07_ACT_W,
    parameter integer GATE_W   = `NTS07_GATE_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,
    input  wire                     acc_clear,
    input  wire [GATE_W-1:0]        gate,
    input  wire signed [ACT_W-1:0]  k_val [0:HEAD_DIM-1],
    output reg signed [ACT_W-1:0]   acc_out [0:HEAD_DIM-1]
);
    integer d;
    reg signed [ACT_W+GATE_W-1:0] product;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (d = 0; d < HEAD_DIM; d = d + 1)
                acc_out[d] <= {ACT_W{1'b0}};
        end else if (acc_clear) begin
            for (d = 0; d < HEAD_DIM; d = d + 1)
                acc_out[d] <= {ACT_W{1'b0}};
        end else if (en) begin
            for (d = 0; d < HEAD_DIM; d = d + 1) begin
                product = $signed({1'b0, gate}) * $signed(k_val[d]);
                acc_out[d] <= acc_out[d] + product[ACT_W+GATE_W-1:GATE_W];
            end
        end
    end
endmodule
