`timescale 1ns/1ps
`default_nettype none

// Transposed class/segment/lane bitmap storage.
// Capture writes one token column across active K lanes. Term replay reads all
// token segments for one {class,lane} row. The explicit banking contract maps
// one tiny bank to every {segment,lane}, with class as the bank address.
module gatestack_transposed_bitmap_bank #(
    parameter int TOKENS       = 162,
    parameter int LANES        = 32,
    parameter int CLASS_SLOTS  = 4,
    parameter int SEGMENT_W    = 16,
    parameter int CLASS_ID_W   =
        (CLASS_SLOTS <= 1) ? 1 : $clog2(CLASS_SLOTS),
    parameter int TOKEN_ID_W   = (TOKENS <= 1) ? 1 : $clog2(TOKENS),
    parameter int LANE_ID_W    = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input  logic                      clk_core,
    input  logic                      rst_core,
    input  logic                      clear_valid,

    input  logic                      write_valid,
    input  logic [CLASS_ID_W-1:0]     write_class_id,
    input  logic [TOKEN_ID_W-1:0]     write_token_id,
    input  logic [LANES-1:0]          write_lane_bits,

    input  logic [CLASS_ID_W-1:0]     read_class_id,
    input  logic [LANE_ID_W-1:0]      read_lane_id,
    output logic [TOKENS-1:0]         read_bitmap
);

    localparam int SEGMENTS = (TOKENS + SEGMENT_W - 1) / SEGMENT_W;
    localparam int SEGMENT_ID_W =
        (SEGMENTS <= 1) ? 1 : $clog2(SEGMENTS);
    localparam int BIT_ID_W = (SEGMENT_W <= 1) ? 1 : $clog2(SEGMENT_W);
    localparam LOOP_TOKENS = TOKENS;
    localparam GEN_SEGMENTS = SEGMENTS;
    localparam GEN_LANES = LANES;
    localparam GEN_CLASSES = CLASS_SLOTS;

    logic [SEGMENT_W-1:0] segment_bank_q
        [0:SEGMENTS-1][0:LANES-1][0:CLASS_SLOTS-1];
    logic segment_valid_q
        [0:SEGMENTS-1][0:LANES-1][0:CLASS_SLOTS-1];
    logic [SEGMENT_ID_W-1:0] write_segment;
    logic [BIT_ID_W-1:0] write_bit;
    logic [SEGMENT_W-1:0] write_onehot;

    always_comb begin
        write_segment = SEGMENT_ID_W'(32'(write_token_id) / SEGMENT_W);
        write_bit = BIT_ID_W'(32'(write_token_id) % SEGMENT_W);
        write_onehot = '0;
        write_onehot[write_bit] = 1'b1;
    end

    always_comb begin
        read_bitmap = '0;
        for (int token = 32'd0; token < LOOP_TOKENS;
             token = token + 32'd1) begin
            if (segment_valid_q[token / SEGMENT_W]
                               [read_lane_id][read_class_id])
                read_bitmap[token] =
                    segment_bank_q[token / SEGMENT_W]
                                  [read_lane_id][read_class_id]
                                  [token % SEGMENT_W];
        end
    end

    generate
        for (genvar segment = 32'd0; segment < GEN_SEGMENTS;
             segment = segment + 32'd1) begin : g_segment
            for (genvar lane = 32'd0; lane < GEN_LANES;
                 lane = lane + 32'd1) begin : g_lane
                for (genvar class_slot = 32'd0;
                     class_slot < GEN_CLASSES;
                     class_slot = class_slot + 32'd1) begin : g_class
                    always_ff @(posedge clk_core) begin
                        if (rst_core || clear_valid) begin
                            segment_valid_q[segment][lane][class_slot] <= 1'b0;
                        end else if (write_valid && write_lane_bits[lane] &&
                                     write_segment == SEGMENT_ID_W'(segment) &&
                                     write_class_id == CLASS_ID_W'(class_slot)) begin
                            if (segment_valid_q[segment][lane][class_slot])
                                segment_bank_q[segment][lane][class_slot] <=
                                    segment_bank_q[segment][lane][class_slot] |
                                    write_onehot;
                            else
                                segment_bank_q[segment][lane][class_slot] <=
                                    write_onehot;
                            segment_valid_q[segment][lane][class_slot] <= 1'b1;
                        end
                    end
                end
            end
        end
    endgenerate

endmodule

`default_nettype wire
