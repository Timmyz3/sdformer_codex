`timescale 1ns/1ps
`default_nettype none

// M188: bounded dual-window K7 weakest-bank-defer FC2 event frontend.
//
// Structural slot b still belongs to weight bank b and no bank ID or packing
// crossbar is emitted.  When all eight banks are nonempty, one minimum-
// population bank is deferred and the seven largest populations issue; with
// seven or fewer nonempty banks every active bank issues.  This realizes the
// M187 exact K7 group bound with one weakest-bank selector rather than M180's
// four-winner global ranking network.
//
// The stage window depths are D={2,4,8,8} for output-block extents {1,2,4,8}.
// Header admission also proves that the nonzero descriptor count cannot exceed
// the stage's raw-beat extent {4,8,16,32}, closing the M180 extent hole.
module m188_fc2_dual_window_k7_weakest_bank_defer_frontend #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int MAX_WINDOW_DESCRIPTORS = 8
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         header_valid,
    output logic                         header_ready,
    input  logic [TAG_BITS-1:0]          header_tag,
    input  logic [3:0]                   header_output_blocks,
    input  logic [5:0]                   header_descriptor_count,
    output logic                         header_accept,

    input  logic                         descriptor_valid,
    output logic                         descriptor_ready,
    input  logic [4:0]                   descriptor_beat_index,
    input  logic [95:0]                  descriptor_bitmap,
    output logic                         descriptor_accept,

    output logic                         group_valid,
    input  logic                         group_ready,
    output logic [TAG_BITS-1:0]          group_tag,
    output logic [2:0]                   group_output_block,
    output logic [3:0]                   group_source_count,
    output logic [7:0]                   group_bank_valid,
    output logic [CHANNEL_BITS-1:0]      group_source_channel [0:7],
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
    logic token_has_index_q;
    logic [TAG_BITS-1:0] token_tag_q;
    logic [3:0] token_output_blocks_q;
    logic [5:0] token_descriptor_count_q;
    logic [5:0] descriptors_accepted_q;
    logic [4:0] last_beat_index_q;

    logic [95:0] bitmap_q [0:1][0:MAX_WINDOW_DESCRIPTORS-1];
    logic [4:0] beat_index_q [0:1][0:MAX_WINDOW_DESCRIPTORS-1];
    logic [3:0] entry_count_q [0:1];
    logic [6:0] bank_count_q [0:1][0:7];
    logic window_closed_q [0:1];
    logic fill_select_q;
    logic drain_select_q;

    logic group_valid_q;
    logic [2:0] group_output_block_q;
    logic [3:0] group_source_count_q;
    logic [7:0] group_bank_valid_q;
    logic [CHANNEL_BITS-1:0] group_source_channel_q [0:7];
    logic group_window_last_q;

    logic done_valid_q;
    logic [TAG_BITS-1:0] done_tag_q;
    logic done_had_event_q;

    logic header_shape_legal;
    logic descriptor_shape_legal;
    logic descriptor_extent_legal;
    logic illegal_request;
    logic [3:0] window_limit;
    logic [3:0] fill_entry_count_effective;
    logic fill_window_releasing;
    logic descriptor_closes_window;

    logic group_final_accept;
    logic group_slot_open;
    logic current_window_release;
    logic candidate_window;
    logic candidate_closed;
    logic candidate_load;
    logic candidate_window_last;
    logic [3:0] candidate_count;
    logic [9:0] candidate_total_events;
    logic [3:0] candidate_active_banks;
    logic [2:0] excluded_bank;
    logic [6:0] excluded_count;
    logic exclude_one_bank;
    logic [7:0] selected_valid;
    logic [2:0] selected_entry [0:7];
    logic [3:0] selected_row [0:7];
    logic [CHANNEL_BITS-1:0] selected_channel [0:7];
    logic [3:0] descriptor_bank_popcount [0:7];

    function automatic logic [3:0] popcount12(input logic [11:0] value);
        logic [3:0] count;
        begin
            count = '0;
            for (int bit_index = 0; bit_index < 12; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction

    always_comb begin : descriptor_popcounts
        logic [11:0] bank_bits;
        for (int bank = 0; bank < 8; bank++) begin
            for (int row = 0; row < 12; row++)
                bank_bits[row] = descriptor_bitmap[(row*8)+bank];
            descriptor_bank_popcount[bank] = popcount12(bank_bits);
        end
    end

    always_comb begin : shape_and_capacity
        header_shape_legal = 1'b0;
        case (header_output_blocks)
            4'd1: header_shape_legal = header_descriptor_count <= 6'd4;
            4'd2: header_shape_legal = header_descriptor_count <= 6'd8;
            4'd4: header_shape_legal = header_descriptor_count <= 6'd16;
            4'd8: header_shape_legal = header_descriptor_count <= 6'd32;
            default: header_shape_legal = 1'b0;
        endcase
        case (token_output_blocks_q)
            4'd1: begin
                window_limit = 4'd2;
                descriptor_extent_legal = descriptor_beat_index < 5'd4;
            end
            4'd2: begin
                window_limit = 4'd4;
                descriptor_extent_legal = descriptor_beat_index < 5'd8;
            end
            4'd4: begin
                window_limit = 4'd8;
                descriptor_extent_legal = descriptor_beat_index < 5'd16;
            end
            4'd8: begin
                window_limit = 4'd8;
                descriptor_extent_legal = 1'b1;
            end
            default: begin
                window_limit = 4'd0;
                descriptor_extent_legal = 1'b0;
            end
        endcase
        descriptor_shape_legal = token_active_q
            && descriptors_accepted_q < token_descriptor_count_q
            && descriptor_bitmap != 0
            && descriptor_extent_legal
            && (!token_has_index_q
                || descriptor_beat_index > last_beat_index_q);
        fill_entry_count_effective = fill_window_releasing
            ? 4'd0 : entry_count_q[fill_select_q];
        descriptor_closes_window = descriptor_accept
            && ((fill_entry_count_effective + 1'b1 >= window_limit)
                || (descriptors_accepted_q + 1'b1
                    == token_descriptor_count_q));
    end

    assign group_accept = group_valid_q && group_ready;
    assign group_final_accept = group_accept
        && ({1'b0, group_output_block_q} + 1'b1
            == token_output_blocks_q);
    assign group_slot_open = !group_valid_q || group_final_accept;
    assign current_window_release = group_final_accept
        && group_window_last_q;
    assign candidate_window = current_window_release
        ? ~drain_select_q : drain_select_q;
    assign candidate_closed = window_closed_q[candidate_window];
    assign fill_window_releasing = current_window_release
        && fill_select_q == drain_select_q;

    always_comb begin : weakest_bank_defer_group_selector
        logic source_found;
        candidate_total_events = '0;
        candidate_active_banks = '0;
        excluded_bank = '0;
        excluded_count = 7'h7f;
        selected_valid = '0;

        // K7 differs from K8 only in the all-eight-active case.  Excluding a
        // minimum-population bank is equivalent to serving the seven largest
        // queues and attains max(max_bank, ceil(total/7)) groups.
        for (int bank = 0; bank < 8; bank++) begin
            candidate_total_events = candidate_total_events
                + bank_count_q[candidate_window][bank];
            if (bank_count_q[candidate_window][bank] != 0) begin
                candidate_active_banks = candidate_active_banks + 1'b1;
                if (bank_count_q[candidate_window][bank]
                        < excluded_count) begin
                    excluded_count
                        = bank_count_q[candidate_window][bank];
                    excluded_bank = bank[2:0];
                end
            end
        end
        exclude_one_bank = candidate_active_banks == 4'd8;
        candidate_count = candidate_active_banks - exclude_one_bank;

        for (int bank = 0; bank < 8; bank++) begin
            selected_valid[bank]
                = bank_count_q[candidate_window][bank] != 0
                && (!exclude_one_bank || bank[2:0] != excluded_bank);
            selected_entry[bank] = '0;
            selected_row[bank] = '0;
            selected_channel[bank] = '0;
            source_found = 1'b0;
            for (int entry = 0; entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                for (int row = 0; row < 12; row++) begin
                    if (!source_found && selected_valid[bank]
                            && entry < entry_count_q[candidate_window]
                            && bitmap_q[candidate_window][entry]
                                [(row*8)+bank]) begin
                        selected_entry[bank] = entry[2:0];
                        selected_row[bank] = row[3:0];
                        source_found = 1'b1;
                    end
                end
            end
            if (selected_valid[bank] && source_found)
                selected_channel[bank]
                    = ((({{(CHANNEL_BITS-5){1'b0}},
                            beat_index_q[candidate_window]
                                [selected_entry[bank]]}
                        << 3)
                        + ({{(CHANNEL_BITS-5){1'b0}},
                            beat_index_q[candidate_window]
                                [selected_entry[bank]]}
                        << 2)
                        + selected_row[bank]) << 3)
                        + bank;
        end
        candidate_window_last = candidate_total_events == candidate_count;
    end

    assign candidate_load = group_slot_open && candidate_closed
        && candidate_count != 0;
    assign header_ready = !fault_q
        && (!token_active_q || token_done_accept)
        && header_shape_legal;
    assign header_accept = header_valid && header_ready;
    assign descriptor_ready = !fault_q && descriptor_shape_legal
        && ((!window_closed_q[fill_select_q]
                && entry_count_q[fill_select_q] < window_limit)
            || fill_window_releasing);
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign illegal_request = (header_valid && !header_shape_legal)
        || (descriptor_valid && !descriptor_shape_legal);

    assign group_valid = group_valid_q;
    assign group_tag = token_tag_q;
    assign group_output_block = group_output_block_q;
    assign group_source_count = group_source_count_q;
    assign group_bank_valid = group_bank_valid_q;
    assign token_done_valid = done_valid_q;
    assign token_done_accept = done_valid_q && token_done_ready;
    assign token_done_tag = done_tag_q;
    assign token_done_had_event = done_had_event_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = token_active_q || group_valid_q || done_valid_q
        || window_closed_q[0] || window_closed_q[1];

    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_group_output
            assign group_source_channel[bank]
                = group_source_channel_q[bank];
        end
    endgenerate

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 1'b0;
            token_active_q <= 1'b0;
            token_has_index_q <= 1'b0;
            token_tag_q <= '0;
            token_output_blocks_q <= '0;
            token_descriptor_count_q <= '0;
            descriptors_accepted_q <= '0;
            last_beat_index_q <= '0;
            fill_select_q <= 1'b0;
            drain_select_q <= 1'b0;
            group_valid_q <= 1'b0;
            group_output_block_q <= '0;
            group_source_count_q <= '0;
            group_bank_valid_q <= '0;
            group_window_last_q <= 1'b0;
            done_valid_q <= 1'b0;
            done_tag_q <= '0;
            done_had_event_q <= 1'b0;
            for (int buffer = 0; buffer < 2; buffer++) begin
                entry_count_q[buffer] <= '0;
                window_closed_q[buffer] <= 1'b0;
                for (int bank = 0; bank < 8; bank++)
                    bank_count_q[buffer][bank] <= '0;
                for (int entry = 0;
                        entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                    bitmap_q[buffer][entry] <= '0;
                    beat_index_q[buffer][entry] <= '0;
                end
            end
            for (int bank = 0; bank < 8; bank++)
                group_source_channel_q[bank] <= '0;
        end else begin
            if (illegal_request)
                fault_q <= 1'b1;
            if (token_done_accept) begin
                done_valid_q <= 1'b0;
                token_active_q <= 1'b0;
                token_has_index_q <= 1'b0;
                descriptors_accepted_q <= '0;
                last_beat_index_q <= '0;
            end

            if (header_accept) begin
                token_active_q <= 1'b1;
                token_has_index_q <= 1'b0;
                token_tag_q <= header_tag;
                token_output_blocks_q <= header_output_blocks;
                token_descriptor_count_q <= header_descriptor_count;
                descriptors_accepted_q <= '0;
                last_beat_index_q <= '0;
                fill_select_q <= 1'b0;
                drain_select_q <= 1'b0;
                group_valid_q <= 1'b0;
                group_output_block_q <= '0;
                group_source_count_q <= '0;
                group_bank_valid_q <= '0;
                group_window_last_q <= 1'b0;
                done_valid_q <= 1'b0;
                for (int buffer = 0; buffer < 2; buffer++) begin
                    entry_count_q[buffer] <= '0;
                    window_closed_q[buffer] <= 1'b0;
                    for (int bank = 0; bank < 8; bank++)
                        bank_count_q[buffer][bank] <= '0;
                    for (int entry = 0;
                            entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                        bitmap_q[buffer][entry] <= '0;
                        beat_index_q[buffer][entry] <= '0;
                    end
                end
            end else begin
                if (token_active_q
                        && descriptors_accepted_q
                            == token_descriptor_count_q
                        && !window_closed_q[0] && !window_closed_q[1]
                        && !group_valid_q && !done_valid_q) begin
                    done_valid_q <= 1'b1;
                    done_tag_q <= token_tag_q;
                    done_had_event_q <= token_descriptor_count_q != 0;
                end

                if (current_window_release) begin
                    window_closed_q[drain_select_q] <= 1'b0;
                    entry_count_q[drain_select_q] <= '0;
                    for (int bank = 0; bank < 8; bank++)
                        bank_count_q[drain_select_q][bank] <= '0;
                    for (int entry = 0;
                            entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                        bitmap_q[drain_select_q][entry] <= '0;
                        beat_index_q[drain_select_q][entry] <= '0;
                    end
                    drain_select_q <= ~drain_select_q;
                end

                if (descriptor_accept) begin
                    descriptors_accepted_q <= descriptors_accepted_q + 1'b1;
                    token_has_index_q <= 1'b1;
                    last_beat_index_q <= descriptor_beat_index;
                    if (fill_window_releasing) begin
                        entry_count_q[fill_select_q] <= 4'd1;
                        for (int bank = 0; bank < 8; bank++)
                            bank_count_q[fill_select_q][bank]
                                <= descriptor_bank_popcount[bank];
                        bitmap_q[fill_select_q][0] <= descriptor_bitmap;
                        beat_index_q[fill_select_q][0]
                            <= descriptor_beat_index;
                    end else begin
                        entry_count_q[fill_select_q]
                            <= entry_count_q[fill_select_q] + 1'b1;
                        for (int bank = 0; bank < 8; bank++)
                            bank_count_q[fill_select_q][bank]
                                <= bank_count_q[fill_select_q][bank]
                                    + descriptor_bank_popcount[bank];
                        bitmap_q[fill_select_q]
                            [entry_count_q[fill_select_q]]
                            <= descriptor_bitmap;
                        beat_index_q[fill_select_q]
                            [entry_count_q[fill_select_q]]
                            <= descriptor_beat_index;
                    end
                    if (descriptor_closes_window) begin
                        window_closed_q[fill_select_q] <= 1'b1;
                        fill_select_q <= ~fill_select_q;
                    end
                end

                if (candidate_load) begin
                    for (int bank = 0; bank < 8; bank++) begin
                        if (selected_valid[bank]) begin
                            bank_count_q[candidate_window][bank]
                                <= bank_count_q[candidate_window][bank] - 1'b1;
                            bitmap_q[candidate_window]
                                [selected_entry[bank]]
                                [(selected_row[bank]*8)+bank] <= 1'b0;
                        end
                    end
                end

                if (group_accept && !group_final_accept)
                    group_output_block_q <= group_output_block_q + 1'b1;
                if (group_slot_open) begin
                    if (candidate_load) begin
                        group_valid_q <= 1'b1;
                        group_output_block_q <= '0;
                        group_source_count_q <= candidate_count;
                        group_bank_valid_q <= selected_valid;
                        group_window_last_q <= candidate_window_last;
                        for (int bank = 0; bank < 8; bank++)
                            group_source_channel_q[bank]
                                <= selected_channel[bank];
                    end else if (group_final_accept) begin
                        group_valid_q <= 1'b0;
                        group_bank_valid_q <= '0;
                        group_window_last_q <= 1'b0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire

