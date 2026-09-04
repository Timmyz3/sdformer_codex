`timescale 1ns/1ps
`default_nettype none

// M175: matched 96-bit physical A/B point for the M174 128-bit frontend.
//
// Twelve rows are reduced in parallel to eight bank-present bits.  The same
// shared four-bank hierarchy, one-entry raw prefetch, group replay and
// same-cycle token rearm are retained so the M174/M175 DC comparison changes
// scan width rather than protocol or buffering.
module m175_fc2_bitmap96_hierarchical_k4_replay_frontend #(
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
    input  logic [95:0]                  scan_bitmap,
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
    logic [95:0] residual_bitmap_q;
    logic [BASE_ROW_BITS-1:0] residual_base_row_q;
    logic group_valid_q;
    logic [2:0] group_output_block_q;
    logic [2:0] group_source_count_q;
    logic [2:0] group_bank_id_q [0:3];
    logic [CHANNEL_BITS-1:0] group_source_channel_q [0:3];
    logic done_valid_q;
    logic [TAG_BITS-1:0] done_tag_q;
    logic done_had_event_q;

    logic [95:0] candidate_bitmap;
    logic [BASE_ROW_BITS-1:0] candidate_base_row;
    logic [7:0] bank_present;
    logic [7:0] bank_remaining [0:4];
    logic [7:0] selected_bank_onehot [0:3];
    logic [2:0] selected_bank_id [0:3];
    logic [3:0] selected_row [0:3];
    logic [CHANNEL_BITS-1:0] selected_channel [0:3];
    logic [2:0] candidate_count;
    logic [95:0] candidate_remainder;
    logic scan_shape_legal;
    logic scan_identity_legal;
    logic illegal_request;
    logic group_final_accept;
    logic group_slot_open;
    logic candidate_available;
    logic candidate_load;
    logic residual_will_clear;

    function automatic logic [7:0] lowest_onehot(input logic [7:0] value);
        lowest_onehot = value & (~value + 8'b1);
    endfunction

    function automatic logic [2:0] encode_bank(input logic [7:0] onehot);
        case (onehot)
            8'b0000_0001: encode_bank = 3'd0;
            8'b0000_0010: encode_bank = 3'd1;
            8'b0000_0100: encode_bank = 3'd2;
            8'b0000_1000: encode_bank = 3'd3;
            8'b0001_0000: encode_bank = 3'd4;
            8'b0010_0000: encode_bank = 3'd5;
            8'b0100_0000: encode_bank = 3'd6;
            8'b1000_0000: encode_bank = 3'd7;
            default:      encode_bank = 3'd0;
        endcase
    endfunction

    function automatic logic [3:0] first_row(
            input logic [95:0] bitmap, input logic [2:0] bank);
        logic found;
        begin
            first_row = 4'd0;
            found = 1'b0;
            for (int row = 0; row < 12; row++) begin
                if (!found && bitmap[(row*8)+bank]) begin
                    first_row = row[3:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    assign candidate_bitmap = residual_valid_q
        ? residual_bitmap_q : scan_bitmap;
    assign candidate_base_row = residual_valid_q
        ? residual_base_row_q : scan_base_row;

    always_comb begin : hierarchical_selector
        candidate_remainder = candidate_bitmap;
        candidate_count = '0;
        for (int bank = 0; bank < 8; bank++) begin
            bank_present[bank] = 1'b0;
            for (int row = 0; row < 12; row++)
                bank_present[bank] |= candidate_bitmap[(row*8)+bank];
        end
        bank_remaining[0] = bank_present;
        for (int slot = 0; slot < 4; slot++) begin
            selected_bank_onehot[slot]
                = lowest_onehot(bank_remaining[slot]);
            bank_remaining[slot+1] = bank_remaining[slot]
                & ~selected_bank_onehot[slot];
            selected_bank_id[slot]
                = encode_bank(selected_bank_onehot[slot]);
            if (selected_bank_onehot[slot] != 0) begin
                selected_row[slot]
                    = first_row(candidate_bitmap, selected_bank_id[slot]);
                selected_channel[slot]
                    = ((candidate_base_row + selected_row[slot]) << 3)
                        + selected_bank_id[slot];
                candidate_count = candidate_count + 1'b1;
                candidate_remainder[
                    (selected_row[slot]*8)+selected_bank_id[slot]
                ] = 1'b0;
            end else begin
                selected_row[slot] = '0;
                selected_channel[slot] = '0;
            end
        end
    end

    always_comb begin : scan_legality
        scan_shape_legal = ((scan_output_blocks == 4'd1)
                || (scan_output_blocks == 4'd2)
                || (scan_output_blocks == 4'd4)
                || (scan_output_blocks == 4'd8))
            && scan_base_row[1:0] == 2'b0;
        if (token_active_q && !token_done_accept) begin
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
    assign candidate_available = residual_valid_q || scan_accept;
    assign candidate_load = group_slot_open && candidate_available
        && candidate_count != 0;
    assign residual_will_clear = residual_valid_q && candidate_load
        && candidate_remainder == 96'b0;
    assign scan_ready = !fault_q
        && (!done_valid_q || token_done_accept)
        && (!token_last_seen_q || token_done_accept)
        && scan_shape_legal && scan_identity_legal
        && (!residual_valid_q || residual_will_clear);
    assign scan_accept = scan_valid && scan_ready;

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
                if (!token_active_q || token_done_accept) begin
                    token_active_q <= 1'b1;
                    token_tag_q <= scan_tag;
                    token_output_blocks_q <= scan_output_blocks;
                    token_had_event_q <= |scan_bitmap;
                end else if (|scan_bitmap) begin
                    token_had_event_q <= 1'b1;
                end
                expected_base_row_q <= scan_base_row + 12;
                if (scan_last)
                    token_last_seen_q <= 1'b1;
            end

            if (group_accept && !group_final_accept)
                group_output_block_q <= group_output_block_q + 1'b1;
            if (group_slot_open) begin
                if (candidate_load) begin
                    group_valid_q <= 1'b1;
                    group_output_block_q <= '0;
                    group_source_count_q <= candidate_count;
                    for (int slot = 0; slot < 4; slot++) begin
                        group_bank_id_q[slot] <= selected_bank_id[slot];
                        group_source_channel_q[slot]
                            <= selected_channel[slot];
                    end
                end else if (group_final_accept) begin
                    group_valid_q <= 1'b0;
                end
            end

            if (residual_valid_q) begin
                if (candidate_load) begin
                    if (candidate_remainder != 0) begin
                        residual_valid_q <= 1'b1;
                        residual_bitmap_q <= candidate_remainder;
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
                if (candidate_load) begin
                    residual_valid_q <= |candidate_remainder;
                    residual_bitmap_q <= candidate_remainder;
                end else begin
                    residual_valid_q <= |scan_bitmap;
                    residual_bitmap_q <= scan_bitmap;
                end
                residual_base_row_q <= scan_base_row;
            end
        end
    end
endmodule

`default_nettype wire
