`timescale 1ns/1ps
`default_nettype none

// M171: 64-bit FC2 bitmap scanner with bank-unique K4 group retention.
//
// A scan beat contains eight consecutive channel rows; bit (row*8+bank)
// maps to channel=(scan_base_row+row)*8+bank.  At most one event from each
// modulo-8 bank is selected into a group, and at most four banks are selected.
// The group is then held and replayed across every Cout/96 output block before
// it retires.  While a group is replaying, one raw bitmap beat can be prefetched.
// On a final replay, extraction can replace the group in the same cycle, so
// dense stage-0 streams sustain one group/cycle after fill.
//
// This module makes scan/replay cycles explicit but intentionally excludes the
// weight SRAM response, M169 arithmetic, accumulator context and BN2/residual.
module m171_fc2_bitmap_k4_group_replay_frontend #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int BASE_ROW_BITS = 9
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         scan_valid,
    output logic                         scan_ready,
    input  logic [TAG_BITS-1:0]          scan_tag,
    input  logic [3:0]                   scan_output_blocks,
    input  logic [BASE_ROW_BITS-1:0]     scan_base_row,
    input  logic [63:0]                  scan_bitmap,
    input  logic                         scan_last,
    output logic                         scan_accept,

    output logic                         group_valid,
    input  logic                         group_ready,
    output logic [TAG_BITS-1:0]          group_tag,
    output logic [2:0]                   group_output_block,
    output logic [2:0]                   group_source_count,
    output logic [2:0]                   group_bank_id [0:3],
    output logic [CHANNEL_BITS-1:0]      group_source_channel [0:3],
    output logic                         group_accept,

    output logic                         token_done_valid,
    input  logic                         token_done_ready,
    output logic [TAG_BITS-1:0]          token_done_tag,
    output logic                         token_done_had_event,
    output logic                         token_done_accept,

    output logic                         protocol_error,
    output logic                         busy
);
    logic fault_q;
    logic token_active_q;
    logic token_last_seen_q;
    logic token_had_event_q;
    logic [TAG_BITS-1:0] token_tag_q;
    logic [3:0] token_output_blocks_q;
    logic [BASE_ROW_BITS-1:0] expected_base_row_q;

    logic residual_valid_q;
    logic [63:0] residual_bitmap_q;
    logic [BASE_ROW_BITS-1:0] residual_base_row_q;

    logic group_valid_q;
    logic [2:0] group_output_block_q;
    logic [2:0] group_source_count_q;
    logic [2:0] group_bank_id_q [0:3];
    logic [CHANNEL_BITS-1:0] group_source_channel_q [0:3];

    logic done_valid_q;
    logic [TAG_BITS-1:0] done_tag_q;
    logic done_had_event_q;

    logic scan_shape_legal;
    logic scan_identity_legal;
    logic illegal_request;
    logic group_final_accept;
    logic group_slot_open;

    logic [63:0] residual_remainder;
    logic [2:0] residual_select_count;
    logic [2:0] residual_select_bank [0:3];
    logic [CHANNEL_BITS-1:0] residual_select_channel [0:3];
    logic [63:0] scan_remainder;
    logic [2:0] scan_select_count;
    logic [2:0] scan_select_bank [0:3];
    logic [CHANNEL_BITS-1:0] scan_select_channel [0:3];
    logic residual_will_clear;
    logic group_load_from_residual;
    logic group_load_from_scan;

    always_comb begin : select_residual_group
        integer selected;
        logic found;
        residual_remainder = residual_bitmap_q;
        residual_select_count = '0;
        for (int slot = 0; slot < 4; slot++) begin
            residual_select_bank[slot] = '0;
            residual_select_channel[slot] = '0;
        end
        selected = 0;
        for (int bank = 0; bank < 8; bank++) begin
            found = 1'b0;
            for (int row = 0; row < 8; row++) begin
                if (!found && selected < 4
                        && residual_bitmap_q[(row*8)+bank]) begin
                    residual_select_bank[selected] = bank[2:0];
                    residual_select_channel[selected]
                        = ((residual_base_row_q + row) << 3) + bank;
                    residual_remainder[(row*8)+bank] = 1'b0;
                    selected = selected + 1;
                    found = 1'b1;
                end
            end
        end
        residual_select_count = selected[2:0];
    end

    always_comb begin : select_scan_group
        integer selected;
        logic found;
        scan_remainder = scan_bitmap;
        scan_select_count = '0;
        for (int slot = 0; slot < 4; slot++) begin
            scan_select_bank[slot] = '0;
            scan_select_channel[slot] = '0;
        end
        selected = 0;
        for (int bank = 0; bank < 8; bank++) begin
            found = 1'b0;
            for (int row = 0; row < 8; row++) begin
                if (!found && selected < 4
                        && scan_bitmap[(row*8)+bank]) begin
                    scan_select_bank[selected] = bank[2:0];
                    scan_select_channel[selected]
                        = ((scan_base_row + row) << 3) + bank;
                    scan_remainder[(row*8)+bank] = 1'b0;
                    selected = selected + 1;
                    found = 1'b1;
                end
            end
        end
        scan_select_count = selected[2:0];
    end

    always_comb begin : scan_legality
        scan_shape_legal = (scan_output_blocks == 4'd1)
            || (scan_output_blocks == 4'd2)
            || (scan_output_blocks == 4'd4)
            || (scan_output_blocks == 4'd8);
        if (token_active_q) begin
            scan_identity_legal = !token_last_seen_q
                && scan_tag == token_tag_q
                && scan_output_blocks == token_output_blocks_q
                && scan_base_row == expected_base_row_q;
        end else begin
            scan_identity_legal = scan_base_row == '0;
        end
        illegal_request = scan_valid
            && (!scan_shape_legal || !scan_identity_legal);
    end

    assign group_accept = group_valid_q && group_ready;
    assign group_final_accept = group_accept
        && ({1'b0, group_output_block_q} + 4'd1
            == token_output_blocks_q);
    assign group_slot_open = !group_valid_q || group_final_accept;
    assign residual_will_clear = residual_valid_q && group_slot_open
        && residual_select_count != 0 && residual_remainder == 64'b0;
    assign scan_ready = !fault_q && !done_valid_q && !token_last_seen_q
        && scan_shape_legal && scan_identity_legal
        && (!residual_valid_q || residual_will_clear);
    assign scan_accept = scan_valid && scan_ready;

    assign group_load_from_residual = group_slot_open && residual_valid_q
        && residual_select_count != 0;
    assign group_load_from_scan = group_slot_open && !residual_valid_q
        && scan_accept && scan_select_count != 0;

    assign group_valid = group_valid_q;
    assign group_tag = token_tag_q;
    assign group_output_block = group_output_block_q;
    assign group_source_count = group_source_count_q;
    assign token_done_valid = done_valid_q;
    assign token_done_tag = done_tag_q;
    assign token_done_had_event = done_had_event_q;
    assign token_done_accept = done_valid_q && token_done_ready;
    assign protocol_error = fault_q || illegal_request;
    assign busy = token_active_q || residual_valid_q || group_valid_q
        || done_valid_q;

    generate
        for (genvar slot = 0; slot < 4; slot++) begin : g_group_output
            assign group_bank_id[slot] = group_bank_id_q[slot];
            assign group_source_channel[slot]
                = group_source_channel_q[slot];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 1'b0;
            token_active_q <= 1'b0;
            token_last_seen_q <= 1'b0;
            token_had_event_q <= 1'b0;
            token_tag_q <= '0;
            token_output_blocks_q <= '0;
            expected_base_row_q <= '0;
            residual_valid_q <= 1'b0;
            residual_bitmap_q <= '0;
            residual_base_row_q <= '0;
            group_valid_q <= 1'b0;
            group_output_block_q <= '0;
            group_source_count_q <= '0;
            for (int slot = 0; slot < 4; slot++) begin
                group_bank_id_q[slot] <= '0;
                group_source_channel_q[slot] <= '0;
            end
            done_valid_q <= 1'b0;
            done_tag_q <= '0;
            done_had_event_q <= 1'b0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;

            if (token_done_accept) begin
                done_valid_q <= 1'b0;
                token_active_q <= 1'b0;
                token_last_seen_q <= 1'b0;
                token_had_event_q <= 1'b0;
                expected_base_row_q <= '0;
            end

            if (token_active_q && token_last_seen_q
                    && !residual_valid_q && !group_valid_q
                    && !done_valid_q) begin
                done_valid_q <= 1'b1;
                done_tag_q <= token_tag_q;
                done_had_event_q <= token_had_event_q;
            end

            if (scan_accept) begin
                if (!token_active_q) begin
                    token_active_q <= 1'b1;
                    token_tag_q <= scan_tag;
                    token_output_blocks_q <= scan_output_blocks;
                    token_had_event_q <= |scan_bitmap;
                end else if (|scan_bitmap) begin
                    token_had_event_q <= 1'b1;
                end
                expected_base_row_q <= scan_base_row + 8;
                if (scan_last)
                    token_last_seen_q <= 1'b1;
            end

            if (group_accept && !group_final_accept)
                group_output_block_q <= group_output_block_q + 1'b1;

            if (group_slot_open) begin
                if (group_load_from_residual) begin
                    group_valid_q <= 1'b1;
                    group_output_block_q <= '0;
                    group_source_count_q <= residual_select_count;
                    for (int slot = 0; slot < 4; slot++) begin
                        group_bank_id_q[slot]
                            <= residual_select_bank[slot];
                        group_source_channel_q[slot]
                            <= residual_select_channel[slot];
                    end
                end else if (group_load_from_scan) begin
                    group_valid_q <= 1'b1;
                    group_output_block_q <= '0;
                    group_source_count_q <= scan_select_count;
                    for (int slot = 0; slot < 4; slot++) begin
                        group_bank_id_q[slot] <= scan_select_bank[slot];
                        group_source_channel_q[slot]
                            <= scan_select_channel[slot];
                    end
                end else if (group_final_accept) begin
                    group_valid_q <= 1'b0;
                end
            end

            if (residual_valid_q) begin
                if (group_load_from_residual) begin
                    if (residual_remainder != 0) begin
                        residual_valid_q <= 1'b1;
                        residual_bitmap_q <= residual_remainder;
                    end else if (scan_accept) begin
                        residual_valid_q <= |scan_bitmap;
                        residual_bitmap_q <= scan_bitmap;
                        residual_base_row_q <= scan_base_row;
                    end else begin
                        residual_valid_q <= 1'b0;
                        residual_bitmap_q <= '0;
                    end
                end
            end else if (scan_accept) begin
                if (group_load_from_scan) begin
                    residual_valid_q <= |scan_remainder;
                    residual_bitmap_q <= scan_remainder;
                    residual_base_row_q <= scan_base_row;
                end else begin
                    residual_valid_q <= |scan_bitmap;
                    residual_bitmap_q <= scan_bitmap;
                    residual_base_row_q <= scan_base_row;
                end
            end
        end
    end
endmodule

`default_nettype wire
