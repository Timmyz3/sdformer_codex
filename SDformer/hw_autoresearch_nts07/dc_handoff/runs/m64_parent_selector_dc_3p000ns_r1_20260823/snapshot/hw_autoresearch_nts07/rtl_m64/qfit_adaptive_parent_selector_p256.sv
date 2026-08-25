`timescale 1ns/1ps
`default_nettype none

// Two-stage elastic selector for the Local/Motion parent alternatives used by
// the K4 delta engine.  The selected parent ID is an address/control result;
// the wide parent accumulator remains in the seed SRAM and is not multiplexed
// through this block.
module qfit_adaptive_parent_selector_p256 #(
    parameter int TAG_W = 48,
    parameter int TILE_BITS = 256,
    parameter int COUNT_W = 9
) (
    input  logic                   clk_core,
    input  logic                   rst_core,
    input  logic                   in_valid,
    output logic                   in_ready,
    input  logic [TAG_W-1:0]       in_tag,
    input  logic [TILE_BITS-1:0]   in_target_bits,
    input  logic [TILE_BITS-1:0]   in_left_bits,
    input  logic [TILE_BITS-1:0]   in_up_bits,
    input  logic [TILE_BITS-1:0]   in_previous_bits,
    input  logic                   in_left_valid,
    input  logic                   in_up_valid,
    input  logic                   in_previous_valid,
    output logic                   out_valid,
    input  logic                   out_ready,
    output logic [TAG_W-1:0]       out_tag,
    output logic [1:0]             out_parent_id,
    output logic [TILE_BITS-1:0]   out_add_bits,
    output logic [TILE_BITS-1:0]   out_subtract_bits,
    output logic [COUNT_W-1:0]     out_source_count
);
    localparam logic [1:0] PARENT_ZERO = 2'd0;
    localparam logic [1:0] PARENT_LEFT = 2'd1;
    localparam logic [1:0] PARENT_UP = 2'd2;
    localparam logic [1:0] PARENT_PREVIOUS = 2'd3;
    localparam logic [COUNT_W-1:0] INVALID_COUNT = {COUNT_W{1'b1}};

    logic s0_valid_q;
    logic [TAG_W-1:0] s0_tag_q;
    logic [TILE_BITS-1:0] s0_target_q;
    logic [TILE_BITS-1:0] s0_left_q;
    logic [TILE_BITS-1:0] s0_up_q;
    logic [TILE_BITS-1:0] s0_previous_q;
    logic s0_left_valid_q, s0_up_valid_q, s0_previous_valid_q;
    logic [COUNT_W-1:0] s0_zero_count_q;
    logic [COUNT_W-1:0] s0_left_count_q;
    logic [COUNT_W-1:0] s0_up_count_q;
    logic [COUNT_W-1:0] s0_previous_count_q;

    logic s1_valid_q;
    logic [TAG_W-1:0] s1_tag_q;
    logic [1:0] s1_parent_id_q;
    logic [TILE_BITS-1:0] s1_add_bits_q;
    logic [TILE_BITS-1:0] s1_subtract_bits_q;
    logic [COUNT_W-1:0] s1_source_count_q;

    logic s1_ready;
    logic s0_ready;
    logic [1:0] selected_parent;
    logic [COUNT_W-1:0] selected_count;
    logic [TILE_BITS-1:0] selected_parent_bits;

    function automatic logic [COUNT_W-1:0] popcount_tile(
        input logic [TILE_BITS-1:0] value
    );
        logic [COUNT_W-1:0] count;
        begin
            count = '0;
            for (int bit_index = 0; bit_index < TILE_BITS; bit_index++)
                count = count + value[bit_index];
            popcount_tile = count;
        end
    endfunction

    assign s1_ready = !s1_valid_q || out_ready;
    assign s0_ready = !s0_valid_q || s1_ready;
    assign in_ready = s0_ready;

    always_comb begin
        // Deterministic tie order is zero, left, up, previous.  Invalid
        // spatial/temporal boundaries receive the unreachable sentinel count.
        selected_parent = PARENT_ZERO;
        selected_count = s0_zero_count_q;
        selected_parent_bits = '0;
        if (s0_left_valid_q && s0_left_count_q < selected_count) begin
            selected_parent = PARENT_LEFT;
            selected_count = s0_left_count_q;
            selected_parent_bits = s0_left_q;
        end
        if (s0_up_valid_q && s0_up_count_q < selected_count) begin
            selected_parent = PARENT_UP;
            selected_count = s0_up_count_q;
            selected_parent_bits = s0_up_q;
        end
        if (s0_previous_valid_q && s0_previous_count_q < selected_count) begin
            selected_parent = PARENT_PREVIOUS;
            selected_count = s0_previous_count_q;
            selected_parent_bits = s0_previous_q;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            s0_valid_q <= 1'b0;
            s1_valid_q <= 1'b0;
            s0_tag_q <= '0;
            s0_target_q <= '0;
            s0_left_q <= '0;
            s0_up_q <= '0;
            s0_previous_q <= '0;
            s0_left_valid_q <= 1'b0;
            s0_up_valid_q <= 1'b0;
            s0_previous_valid_q <= 1'b0;
            s0_zero_count_q <= '0;
            s0_left_count_q <= INVALID_COUNT;
            s0_up_count_q <= INVALID_COUNT;
            s0_previous_count_q <= INVALID_COUNT;
            s1_tag_q <= '0;
            s1_parent_id_q <= PARENT_ZERO;
            s1_add_bits_q <= '0;
            s1_subtract_bits_q <= '0;
            s1_source_count_q <= '0;
        end else begin
            if (s1_ready) begin
                s1_valid_q <= s0_valid_q;
                if (s0_valid_q) begin
                    s1_tag_q <= s0_tag_q;
                    s1_parent_id_q <= selected_parent;
                    s1_add_bits_q <= s0_target_q & ~selected_parent_bits;
                    s1_subtract_bits_q <= selected_parent_bits & ~s0_target_q;
                    s1_source_count_q <= selected_count;
                end
            end
            if (s0_ready) begin
                s0_valid_q <= in_valid;
                if (in_valid) begin
                    s0_tag_q <= in_tag;
                    s0_target_q <= in_target_bits;
                    s0_left_q <= in_left_bits;
                    s0_up_q <= in_up_bits;
                    s0_previous_q <= in_previous_bits;
                    s0_left_valid_q <= in_left_valid;
                    s0_up_valid_q <= in_up_valid;
                    s0_previous_valid_q <= in_previous_valid;
                    s0_zero_count_q <= popcount_tile(in_target_bits);
                    s0_left_count_q <= in_left_valid
                        ? popcount_tile(in_target_bits ^ in_left_bits)
                        : INVALID_COUNT;
                    s0_up_count_q <= in_up_valid
                        ? popcount_tile(in_target_bits ^ in_up_bits)
                        : INVALID_COUNT;
                    s0_previous_count_q <= in_previous_valid
                        ? popcount_tile(in_target_bits ^ in_previous_bits)
                        : INVALID_COUNT;
                end
            end
        end
    end

    assign out_valid = s1_valid_q;
    assign out_tag = s1_tag_q;
    assign out_parent_id = s1_parent_id_q;
    assign out_add_bits = s1_add_bits_q;
    assign out_subtract_bits = s1_subtract_bits_q;
    assign out_source_count = s1_source_count_q;
endmodule

`default_nettype wire
