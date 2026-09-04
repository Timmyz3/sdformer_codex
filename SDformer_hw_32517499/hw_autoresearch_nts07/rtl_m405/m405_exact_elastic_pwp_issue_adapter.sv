`timescale 1ns/1ps
`default_nettype none

// Exact M401 low8/high4 PWP decoder.  The two-entry completed-record FIFO is
// intentional: input-side validation overlaps output issue, while a wide
// block is never made visible before its matching high sidecar is checked.
module m405_exact_elastic_pwp_issue_adapter #(
    parameter int TAG_BITS = 24
) (
    input  logic                    clk_core,
    input  logic                    reset_n,

    input  logic                    config_reload,

    input  logic                    low_valid,
    output logic                    low_ready,
    output logic                    low_accept,
    input  logic [TAG_BITS-1:0]     low_tag,
    input  logic                    low_tile,
    input  logic [4:0]              low_center_id,
    input  logic [2:0]              low_output_block,
    input  logic                    low_narrow,
    input  logic [767:0]            low_data,

    input  logic                    high_valid,
    output logic                    high_ready,
    output logic                    high_accept,
    input  logic [TAG_BITS-1:0]     high_tag,
    input  logic                    high_tile,
    input  logic [4:0]              high_center_id,
    input  logic [2:0]              high_output_block,
    // Bytes 0..47 contain packed high nibbles. Bytes 48..63 are physical
    // padding and must be zero. The logical SHARED96 bytes 64..95 do not
    // exist on this 512-bit sidecar port and therefore cannot add traffic.
    input  logic [511:0]            high_data,

    output logic                    contribution_valid,
    input  logic                    contribution_ready,
    output logic                    contribution_accept,
    output logic [TAG_BITS-1:0]     contribution_tag,
    output logic                    contribution_tile,
    output logic [4:0]              contribution_center_id,
    output logic [2:0]              contribution_output_block,
    output logic                    contribution_narrow,
    output logic                    contribution_part_high,
    output logic                    contribution_last,
    output logic [1151:0]           contribution_data,

    output logic                    protocol_error,
    output logic                    busy,
    output logic [1:0]              debug_completed_fifo_count,
    output logic [31:0]             debug_low_accepts,
    output logic [31:0]             debug_high_accepts,
    output logic [31:0]             debug_narrow_blocks,
    output logic [31:0]             debug_wide_blocks,
    output logic [31:0]             debug_contributions
);
    localparam int FIFO_DEPTH = 2;

    logic fault_q;
    logic assembly_valid_q;
    logic [TAG_BITS-1:0] assembly_tag_q;
    logic assembly_tile_q;
    logic [4:0] assembly_center_q;
    logic [2:0] assembly_block_q;
    logic [767:0] assembly_low_q;

    logic [TAG_BITS-1:0] fifo_tag_q [0:FIFO_DEPTH-1];
    logic fifo_tile_q [0:FIFO_DEPTH-1];
    logic [4:0] fifo_center_q [0:FIFO_DEPTH-1];
    logic [2:0] fifo_block_q [0:FIFO_DEPTH-1];
    logic fifo_narrow_q [0:FIFO_DEPTH-1];
    logic [767:0] fifo_low_q [0:FIFO_DEPTH-1];
    logic [383:0] fifo_high_q [0:FIFO_DEPTH-1];
    logic fifo_head_q, fifo_tail_q;
    logic [1:0] fifo_count_q;
    logic emit_high_q;

    logic [31:0] low_accepts_q, high_accepts_q;
    logic [31:0] narrow_blocks_q, wide_blocks_q, contributions_q;

    logic safe_w, pop_w, fifo_space_w;
    logic push_narrow_w, push_wide_w, push_w;
    logic high_metadata_legal_w, high_padding_legal_w;
    logic illegal_high_w, illegal_reload_w;

    assign safe_w = !fault_q;
    assign contribution_valid = safe_w && (fifo_count_q != 0);
    assign contribution_accept = contribution_valid && contribution_ready;
    assign contribution_tag = contribution_valid ? fifo_tag_q[fifo_head_q] : '0;
    assign contribution_tile = contribution_valid && fifo_tile_q[fifo_head_q];
    assign contribution_center_id = contribution_valid ?
        fifo_center_q[fifo_head_q] : '0;
    assign contribution_output_block = contribution_valid ?
        fifo_block_q[fifo_head_q] : '0;
    assign contribution_narrow = contribution_valid &&
        fifo_narrow_q[fifo_head_q];
    assign contribution_part_high = contribution_valid && emit_high_q;
    assign contribution_last = contribution_valid &&
        (fifo_narrow_q[fifo_head_q] || emit_high_q);
    assign pop_w = contribution_accept && contribution_last;
    assign fifo_space_w = (fifo_count_q < FIFO_DEPTH) || pop_w;

    // A low beat cannot overwrite the one in assembly. Requiring FIFO space
    // even for a future wide completion bounds all accepted state explicitly.
    assign low_ready = safe_w && !assembly_valid_q && fifo_space_w;
    assign low_accept = low_valid && low_ready;
    assign high_ready = safe_w && assembly_valid_q && fifo_space_w;
    assign high_accept = high_valid && high_ready;

    assign high_metadata_legal_w = high_tag == assembly_tag_q
        && high_tile == assembly_tile_q
        && high_center_id == assembly_center_q
        && high_output_block == assembly_block_q;
    assign high_padding_legal_w = high_data[511:384] == 0;
    assign push_narrow_w = low_accept && low_narrow;
    assign push_wide_w = high_accept && high_metadata_legal_w
        && high_padding_legal_w;
    assign push_w = push_narrow_w || push_wide_w;
    assign illegal_high_w = high_valid && (!assembly_valid_q
        || (high_ready && (!high_metadata_legal_w || !high_padding_legal_w)));
    assign illegal_reload_w = config_reload &&
        (assembly_valid_q || fifo_count_q != 0);

    always_comb begin : contribution_decode
        contribution_data = '0;
        if (contribution_valid) begin
            for (integer lane = 0; lane < 96; lane = lane + 1) begin
                if (fifo_narrow_q[fifo_head_q]) begin
                    contribution_data[lane*12 +: 12] = {
                        {4{fifo_low_q[fifo_head_q][lane*8+7]}},
                        fifo_low_q[fifo_head_q][lane*8 +: 8]};
                end else if (!emit_high_q) begin
                    contribution_data[lane*12 +: 12] = {
                        4'b0000, fifo_low_q[fifo_head_q][lane*8 +: 8]};
                end else begin
                    contribution_data[lane*12 +: 12] = {
                        fifo_high_q[fifo_head_q][lane*4 +: 4], 8'b0};
                end
            end
        end
    end

    assign protocol_error = fault_q;
    assign busy = assembly_valid_q || fifo_count_q != 0;
    assign debug_completed_fifo_count = fifo_count_q;
    assign debug_low_accepts = low_accepts_q;
    assign debug_high_accepts = high_accepts_q;
    assign debug_narrow_blocks = narrow_blocks_q;
    assign debug_wide_blocks = wide_blocks_q;
    assign debug_contributions = contributions_q;

    integer reset_index;
    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            fault_q <= 1'b0;
            assembly_valid_q <= 1'b0;
            assembly_tag_q <= '0;
            assembly_tile_q <= 1'b0;
            assembly_center_q <= '0;
            assembly_block_q <= '0;
            assembly_low_q <= '0;
            fifo_head_q <= 1'b0;
            fifo_tail_q <= 1'b0;
            fifo_count_q <= '0;
            emit_high_q <= 1'b0;
            low_accepts_q <= '0;
            high_accepts_q <= '0;
            narrow_blocks_q <= '0;
            wide_blocks_q <= '0;
            contributions_q <= '0;
            for (reset_index = 0; reset_index < FIFO_DEPTH;
                 reset_index = reset_index + 1) begin
                fifo_tag_q[reset_index] <= '0;
                fifo_tile_q[reset_index] <= 1'b0;
                fifo_center_q[reset_index] <= '0;
                fifo_block_q[reset_index] <= '0;
                fifo_narrow_q[reset_index] <= 1'b0;
                fifo_low_q[reset_index] <= '0;
                fifo_high_q[reset_index] <= '0;
            end
        end else begin
            if (illegal_high_w || illegal_reload_w)
                fault_q <= 1'b1;

            if (low_accept) begin
                low_accepts_q <= low_accepts_q + 1'b1;
                if (low_narrow) begin
                    narrow_blocks_q <= narrow_blocks_q + 1'b1;
                end else begin
                    assembly_valid_q <= 1'b1;
                    assembly_tag_q <= low_tag;
                    assembly_tile_q <= low_tile;
                    assembly_center_q <= low_center_id;
                    assembly_block_q <= low_output_block;
                    assembly_low_q <= low_data;
                end
            end

            if (high_accept) begin
                high_accepts_q <= high_accepts_q + 1'b1;
                assembly_valid_q <= 1'b0;
                if (high_metadata_legal_w && high_padding_legal_w)
                    wide_blocks_q <= wide_blocks_q + 1'b1;
            end

            if (push_w) begin
                if (push_narrow_w) begin
                    fifo_tag_q[fifo_tail_q] <= low_tag;
                    fifo_tile_q[fifo_tail_q] <= low_tile;
                    fifo_center_q[fifo_tail_q] <= low_center_id;
                    fifo_block_q[fifo_tail_q] <= low_output_block;
                    fifo_narrow_q[fifo_tail_q] <= 1'b1;
                    fifo_low_q[fifo_tail_q] <= low_data;
                    fifo_high_q[fifo_tail_q] <= '0;
                end else begin
                    fifo_tag_q[fifo_tail_q] <= assembly_tag_q;
                    fifo_tile_q[fifo_tail_q] <= assembly_tile_q;
                    fifo_center_q[fifo_tail_q] <= assembly_center_q;
                    fifo_block_q[fifo_tail_q] <= assembly_block_q;
                    fifo_narrow_q[fifo_tail_q] <= 1'b0;
                    fifo_low_q[fifo_tail_q] <= assembly_low_q;
                    fifo_high_q[fifo_tail_q] <= high_data[383:0];
                end
                fifo_tail_q <= ~fifo_tail_q;
            end

            if (contribution_accept) begin
                contributions_q <= contributions_q + 1'b1;
                if (!fifo_narrow_q[fifo_head_q] && !emit_high_q) begin
                    emit_high_q <= 1'b1;
                end else begin
                    emit_high_q <= 1'b0;
                    fifo_head_q <= ~fifo_head_q;
                end
            end

            case ({push_w, pop_w})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: fifo_count_q <= fifo_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
