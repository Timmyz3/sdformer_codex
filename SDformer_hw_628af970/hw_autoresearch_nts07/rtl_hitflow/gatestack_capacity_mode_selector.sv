`timescale 1ns/1ps
`default_nettype none

module gatestack_capacity_mode_selector #(
    parameter int TOKENS          = 162,
    parameter int HEAD_DIM        = 32,
    parameter int GATE_W          = 9,
    parameter int CLASS_SLOTS     = 4,
    parameter int HEADER_BITS     = 128,
    parameter int TAG_W           = 32,
    parameter int CLASS_COUNT_W   = 4,
    parameter int TERM_COUNT_W    = 8,
    parameter int ACTIVE_COUNT_W  = 13,
    parameter int SIZE_W          = 16,
    parameter int COUNTER_W       = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [TAG_W-1:0]             request_tag,
    input  logic [CLASS_COUNT_W-1:0]     request_active_classes,
    input  logic [TERM_COUNT_W-1:0]      request_class_terms,
    input  logic [ACTIVE_COUNT_W-1:0]    request_active_lanes,

    output logic                         response_valid,
    input  logic                         response_ready,
    output logic [TAG_W-1:0]             response_tag,
    output logic                         response_is_csr,
    output logic [1:0]                   response_reason,
    output logic [SIZE_W-1:0]            response_csr_bits,

    output logic [COUNTER_W-1:0]         count_requests,
    output logic [COUNTER_W-1:0]         count_csr,
    output logic [COUNTER_W-1:0]         count_raw_class_overflow,
    output logic [COUNTER_W-1:0]         count_raw_capacity_overflow
);

    localparam int RAW_HEAD_BITS = TOKENS * (HEAD_DIM + GATE_W);
    localparam logic [1:0] REASON_CSR = 2'd0;
    localparam logic [1:0] REASON_CLASS = 2'd1;
    localparam logic [1:0] REASON_CAPACITY = 2'd2;

    logic response_valid_q;
    logic [TAG_W-1:0] response_tag_q;
    logic response_is_csr_q;
    logic [1:0] response_reason_q;
    logic [SIZE_W-1:0] response_csr_bits_q;

    logic [31:0] csr_bits_comb;
    logic is_csr_comb;
    logic [1:0] reason_comb;
    logic request_fire;
    logic response_fire;

    assign request_ready = !response_valid_q || response_ready;
    assign request_fire = request_valid && request_ready;
    assign response_valid = response_valid_q;
    assign response_fire = response_valid && response_ready;
    assign response_tag = response_tag_q;
    assign response_is_csr = response_is_csr_q;
    assign response_reason = response_reason_q;
    assign response_csr_bits = response_csr_bits_q;

    always_comb begin
        // IPD32W: two 32-bit descriptors per 64-bit word; odd term is padded.
        csr_bits_comb = HEADER_BITS
                      + (((32'(request_class_terms) + 1) >> 1) << 6)
                      + (32'(request_active_lanes) << 3);
        if (32'(request_active_classes) > CLASS_SLOTS) begin
            is_csr_comb = 1'b0;
            reason_comb = REASON_CLASS;
        end else if (csr_bits_comb > RAW_HEAD_BITS) begin
            is_csr_comb = 1'b0;
            reason_comb = REASON_CAPACITY;
        end else begin
            is_csr_comb = 1'b1;
            reason_comb = REASON_CSR;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            response_valid_q <= 1'b0;
            response_tag_q <= '0;
            response_is_csr_q <= 1'b0;
            response_reason_q <= REASON_CSR;
            response_csr_bits_q <= '0;
            count_requests <= '0;
            count_csr <= '0;
            count_raw_class_overflow <= '0;
            count_raw_capacity_overflow <= '0;
        end else begin
            if (request_fire) begin
                response_valid_q <= 1'b1;
                response_tag_q <= request_tag;
                response_is_csr_q <= is_csr_comb;
                response_reason_q <= reason_comb;
                response_csr_bits_q <= SIZE_W'(csr_bits_comb);
                count_requests <= count_requests + 1'b1;
                if (reason_comb == REASON_CSR) begin
                    count_csr <= count_csr + 1'b1;
                end else if (reason_comb == REASON_CLASS) begin
                    count_raw_class_overflow <= count_raw_class_overflow + 1'b1;
                end else begin
                    count_raw_capacity_overflow <=
                        count_raw_capacity_overflow + 1'b1;
                end
            end else if (response_fire) begin
                response_valid_q <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
