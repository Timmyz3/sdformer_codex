`timescale 1ns/1ps
`default_nettype none

// M207: M204 paired-window K8 drain plus terminal-group token collapse.
// Two consecutive compact windows are treated as one bank-load job.  Within
// each bank, source order is oldest window, descriptor entry, then bitmap row.
// The module is self-delimiting: upstream_done closes an odd partial tail.
module m207_fc2_descriptor4_paired_window_terminal_collapse_frontend #(
    parameter int TAG_BITS = 24,
    parameter int CHANNEL_BITS = 12,
    parameter int MAX_WINDOW_DESCRIPTORS = 8
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, output logic header_ready,
    input logic [TAG_BITS-1:0] header_tag,
    input logic [3:0] header_output_blocks,
    output logic header_accept,

    input logic descriptor_valid, output logic descriptor_ready,
    input logic [2:0] descriptor_count,
    input logic [TAG_BITS-1:0] descriptor_token_tag,
    input logic [4:0] descriptor_beat_index [0:3],
    input logic [95:0] descriptor_bitmap [0:3],
    input logic [3:0] descriptor_window_last,
    output logic descriptor_accept,

    input logic upstream_done_valid, output logic upstream_done_ready,
    input logic [TAG_BITS-1:0] upstream_done_tag,
    input logic [5:0] upstream_done_descriptor_count,
    output logic upstream_done_accept,

    output logic group_valid, input logic group_ready,
    output logic [TAG_BITS-1:0] group_tag,
    output logic [2:0] group_output_block,
    output logic [3:0] group_source_count,
    output logic [7:0] group_bank_valid,
    output logic [CHANNEL_BITS-1:0] group_source_channel [0:7],
    output logic group_accept,

    output logic token_done_valid, input logic token_done_ready,
    output logic [TAG_BITS-1:0] token_done_tag,
    output logic [5:0] token_done_descriptor_count,
    output logic token_done_had_event,
    output logic token_done_accept,
    output logic protocol_error, output logic busy
);
    logic fault_q, token_active_q, upstream_done_seen_q;
    logic token_has_index_q;
    logic [TAG_BITS-1:0] token_tag_q;
    logic [3:0] output_blocks_q;
    logic [5:0] descriptors_accepted_q;
    logic [4:0] last_beat_index_q;

    logic [95:0] bitmap_q [0:1][0:MAX_WINDOW_DESCRIPTORS-1];
    logic [4:0] beat_index_q [0:1][0:MAX_WINDOW_DESCRIPTORS-1];
    logic [3:0] entry_count_q [0:1];
    logic [7:0] bank_count_q [0:1][0:7];
    logic window_closed_q [0:1];
    logic fill_select_q, drain_select_q;

    logic group_valid_q, group_pair_last_q, group_pair_has_two_q;
    logic [2:0] group_output_block_q;
    logic [3:0] group_source_count_q;
    logic [7:0] group_bank_valid_q;
    logic [CHANNEL_BITS-1:0] group_source_channel_q [0:7];

    logic header_legal, descriptor_shape_legal, descriptor_storage_legal;
    logic descriptor_fill_blocked, done_shape_legal;
    logic illegal_request, fill_window_releasing;
    logic [3:0] fill_entry_count_effective;
    logic descriptor_closes_window;
    logic descriptor_packet_last;
    logic [4:0] descriptor_last_index;
    logic [4:0] descriptor_bank_sum [0:7];
    logic [3:0] lane_bank_popcount [0:3][0:7];

    logic pair_available, pair_has_two, pair_first, pair_second;
    logic [10:0] candidate_total_events;
    logic [3:0] candidate_count;
    logic [7:0] selected_valid;
    logic buffer_selected_valid [0:1][0:7];
    logic [2:0] buffer_selected_entry [0:1][0:7];
    logic [3:0] buffer_selected_row [0:1][0:7];
    logic [CHANNEL_BITS-1:0] buffer_selected_channel [0:1][0:7];
    logic selected_buffer [0:7];
    logic [2:0] selected_entry [0:7];
    logic [3:0] selected_row [0:7];
    logic [CHANNEL_BITS-1:0] selected_channel [0:7];
    logic candidate_pair_last, candidate_load;
    logic group_final_accept, pair_release, terminal_pair_release;

    function automatic logic [3:0] popcount12(input logic [11:0] value);
        logic [3:0] count;
        begin
            count = 0;
            for (int bit_index = 0; bit_index < 12; bit_index++)
                count = count + value[bit_index];
            return count;
        end
    endfunction

    always_comb begin : descriptor_analysis
        logic [11:0] bank_bits;
        logic index_legal;
        descriptor_shape_legal = token_active_q && !upstream_done_seen_q
            && descriptor_count >= 1 && descriptor_count <= 4
            && descriptor_token_tag == token_tag_q;
        index_legal = !token_has_index_q
            || descriptor_beat_index[0] > last_beat_index_q;
        descriptor_packet_last = 0;
        case (descriptor_count)
            1: descriptor_packet_last = descriptor_window_last == 4'b0001;
            2: descriptor_packet_last = descriptor_window_last == 4'b0010;
            3: descriptor_packet_last = descriptor_window_last == 4'b0100;
            4: descriptor_packet_last = descriptor_window_last == 4'b1000;
            default: descriptor_packet_last = 0;
        endcase
        if (descriptor_window_last != 0 && !descriptor_packet_last)
            descriptor_shape_legal = 0;
        for (int bank = 0; bank < 8; bank++) begin
            descriptor_bank_sum[bank] = 0;
            for (int lane = 0; lane < 4; lane++) begin
                bank_bits = 0;
                for (int row = 0; row < 12; row++)
                    bank_bits[row] = descriptor_bitmap[lane][row*8+bank];
                lane_bank_popcount[lane][bank] = popcount12(bank_bits);
                if (lane < descriptor_count)
                    descriptor_bank_sum[bank] = descriptor_bank_sum[bank]
                        + lane_bank_popcount[lane][bank];
            end
        end
        for (int lane = 0; lane < 4; lane++) begin
            if (lane < descriptor_count) begin
                if (descriptor_bitmap[lane] == 0)
                    descriptor_shape_legal = 0;
                // Keep lane zero structurally separate.  Some synthesis
                // elaborators still expand lane-1 under a short-circuited
                // lane!=0 expression and report an out-of-bounds select.
                case (lane)
                    1: if (descriptor_beat_index[1]
                            <= descriptor_beat_index[0]) index_legal = 0;
                    2: if (descriptor_beat_index[2]
                            <= descriptor_beat_index[1]) index_legal = 0;
                    3: if (descriptor_beat_index[3]
                            <= descriptor_beat_index[2]) index_legal = 0;
                    default: index_legal = index_legal;
                endcase
            end else if (descriptor_window_last[lane]) begin
                descriptor_shape_legal = 0;
            end
        end
        descriptor_shape_legal = descriptor_shape_legal && index_legal;
        descriptor_last_index = 0;
        case (descriptor_count)
            1: descriptor_last_index = descriptor_beat_index[0];
            2: descriptor_last_index = descriptor_beat_index[1];
            3: descriptor_last_index = descriptor_beat_index[2];
            4: descriptor_last_index = descriptor_beat_index[3];
            default: descriptor_last_index = 0;
        endcase
        fill_entry_count_effective = fill_window_releasing
            ? 0 : entry_count_q[fill_select_q];
        descriptor_fill_blocked = window_closed_q[fill_select_q]
            && !fill_window_releasing;
        descriptor_storage_legal = !descriptor_fill_blocked
            && fill_entry_count_effective + descriptor_count
                <= MAX_WINDOW_DESCRIPTORS
            && !(fill_entry_count_effective + descriptor_count
                    == MAX_WINDOW_DESCRIPTORS && !descriptor_packet_last);
        descriptor_closes_window = descriptor_accept
            && descriptor_packet_last;
    end

    assign pair_first = drain_select_q;
    assign pair_second = ~drain_select_q;
    // Stage 0 has one output block and the frozen payload says W1 is faster;
    // stages 1--3 wait for and jointly drain a consecutive window pair.
    assign pair_has_two = output_blocks_q != 1
        && window_closed_q[pair_first] && window_closed_q[pair_second];
    assign pair_available = pair_has_two
        || (output_blocks_q == 1 && window_closed_q[pair_first])
        || (upstream_done_seen_q && window_closed_q[pair_first]);

    always_comb begin : per_window_fixed_bank_selector
        logic source_found;
        logic [CHANNEL_BITS-1:0] beat_extended;
        for (int buffer = 0; buffer < 2; buffer++) begin
            for (int bank = 0; bank < 8; bank++) begin
                buffer_selected_valid[buffer][bank] = 0;
                buffer_selected_entry[buffer][bank] = 0;
                buffer_selected_row[buffer][bank] = 0;
                buffer_selected_channel[buffer][bank] = 0;
                source_found = 0;
                for (int entry = 0;
                        entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                    for (int row = 0; row < 12; row++) begin
                        // Released windows are drained to an all-zero bitmap,
                        // and header reset also clears every entry.  Therefore
                        // the bitmap itself is the occupancy authority; keeping
                        // entry_count on this selector path only adds a long
                        // count-decode cone without changing the chosen source.
                        if (!source_found
                                && bitmap_q[buffer][entry][row*8+bank]) begin
                            buffer_selected_valid[buffer][bank] = 1;
                            buffer_selected_entry[buffer][bank] = entry;
                            buffer_selected_row[buffer][bank] = row;
                            beat_extended = {{(CHANNEL_BITS-5){1'b0}},
                                beat_index_q[buffer][entry]};
                            buffer_selected_channel[buffer][bank]
                                = (((beat_extended << 3)
                                    + (beat_extended << 2) + row) << 3)
                                    + bank;
                            source_found = 1;
                        end
                    end
                end
            end
        end
    end

    always_comb begin : paired_fixed_bank_merge
        candidate_total_events = 0;
        candidate_count = 0;
        selected_valid = 0;
        for (int bank = 0; bank < 8; bank++) begin
            candidate_total_events = candidate_total_events
                + bank_count_q[pair_first][bank]
                + (pair_has_two ? bank_count_q[pair_second][bank] : 0);
            selected_valid[bank] = buffer_selected_valid[pair_first][bank]
                || (pair_has_two
                    && buffer_selected_valid[pair_second][bank]);
            candidate_count = candidate_count + selected_valid[bank];
            selected_buffer[bank] = pair_first;
            if (!buffer_selected_valid[pair_first][bank]
                    && pair_has_two)
                selected_buffer[bank] = pair_second;
            selected_entry[bank]
                = buffer_selected_entry[selected_buffer[bank]][bank];
            selected_row[bank]
                = buffer_selected_row[selected_buffer[bank]][bank];
            selected_channel[bank]
                = buffer_selected_channel[selected_buffer[bank]][bank];
        end
        candidate_pair_last = candidate_total_events == candidate_count;
    end

    assign group_valid = group_valid_q;
    assign group_accept = group_valid_q && group_ready;
    assign group_final_accept = group_accept
        && ({1'b0, group_output_block_q} + 1 == output_blocks_q);
    assign pair_release = group_final_accept && group_pair_last_q;
    // The last replay group already proves that every event in the active
    // pair has been consumed.  When upstream_done has fenced the token and no
    // second stage-0 window remains, retire the token on that same handshake
    // instead of spending a dedicated empty-state cycle.
    assign terminal_pair_release = pair_release && upstream_done_seen_q
        && (group_pair_has_two_q
            || (!window_closed_q[pair_second]
                && entry_count_q[pair_second] == 0));
    assign candidate_load = pair_available && candidate_count != 0
        && (!group_valid_q || (group_final_accept && !group_pair_last_q));
    assign group_tag = token_tag_q;
    assign group_output_block = group_output_block_q;
    assign group_source_count = group_source_count_q;
    assign group_bank_valid = group_bank_valid_q;
    generate
        for (genvar bank = 0; bank < 8; bank++) begin : g_group_output
            assign group_source_channel[bank] = group_source_channel_q[bank];
        end
    endgenerate

    assign fill_window_releasing = pair_release
        && fill_select_q == pair_first;
    assign header_legal = header_output_blocks == 1
        || header_output_blocks == 2 || header_output_blocks == 4
        || header_output_blocks == 8;
    assign header_ready = !fault_q
        && (!token_active_q || token_done_accept) && header_legal;
    assign header_accept = header_valid && header_ready;
    assign descriptor_ready = !fault_q && descriptor_shape_legal
        && descriptor_storage_legal;
    assign descriptor_accept = descriptor_valid && descriptor_ready;
    assign done_shape_legal = token_active_q && !upstream_done_seen_q
        && upstream_done_tag == token_tag_q
        && upstream_done_descriptor_count == descriptors_accepted_q
        && !descriptor_valid;
    assign upstream_done_ready = !fault_q && done_shape_legal;
    assign upstream_done_accept = upstream_done_valid
        && upstream_done_ready;
    assign token_done_valid = token_active_q && upstream_done_seen_q
        && ((!window_closed_q[0] && !window_closed_q[1]
                && entry_count_q[0] == 0 && entry_count_q[1] == 0
                && !group_valid_q)
            || terminal_pair_release);
    assign token_done_accept = token_done_valid && token_done_ready;
    assign token_done_tag = token_tag_q;
    assign token_done_descriptor_count = descriptors_accepted_q;
    assign token_done_had_event = descriptors_accepted_q != 0;
    // A legal next-token header may be held while the current token drains.
    // Busy state is ordinary ready/valid backpressure, not a protocol attack.
    assign illegal_request = (header_valid && !header_legal)
        // A legal descriptor may remain asserted while both window buffers
        // are full.  Fullness is backpressure, not a protocol violation.
        || (descriptor_valid && (!descriptor_shape_legal
            || (!descriptor_fill_blocked && !descriptor_storage_legal)))
        || (upstream_done_valid && !done_shape_legal);
    assign protocol_error = fault_q || illegal_request;
    assign busy = token_active_q || group_valid_q
        || window_closed_q[0] || window_closed_q[1];

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 0; token_active_q <= 0; upstream_done_seen_q <= 0;
            token_has_index_q <= 0; token_tag_q <= 0; output_blocks_q <= 0;
            descriptors_accepted_q <= 0; last_beat_index_q <= 0;
            fill_select_q <= 0; drain_select_q <= 0;
            group_valid_q <= 0; group_pair_last_q <= 0;
            group_pair_has_two_q <= 0; group_output_block_q <= 0;
            group_source_count_q <= 0; group_bank_valid_q <= 0;
            for (int bank = 0; bank < 8; bank++)
                group_source_channel_q[bank] <= 0;
            for (int buffer = 0; buffer < 2; buffer++) begin
                entry_count_q[buffer] <= 0; window_closed_q[buffer] <= 0;
                for (int bank = 0; bank < 8; bank++)
                    bank_count_q[buffer][bank] <= 0;
                for (int entry = 0; entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                    bitmap_q[buffer][entry] <= 0;
                    beat_index_q[buffer][entry] <= 0;
                end
            end
        end else begin
            if (illegal_request) fault_q <= 1;
            if (token_done_accept) begin
                token_active_q <= 0; upstream_done_seen_q <= 0;
                token_has_index_q <= 0; descriptors_accepted_q <= 0;
                last_beat_index_q <= 0;
            end
            if (header_accept) begin
                token_active_q <= 1; upstream_done_seen_q <= 0;
                token_has_index_q <= 0; token_tag_q <= header_tag;
                output_blocks_q <= header_output_blocks;
                descriptors_accepted_q <= 0; last_beat_index_q <= 0;
                fill_select_q <= 0; drain_select_q <= 0;
                group_valid_q <= 0; group_output_block_q <= 0;
                group_bank_valid_q <= 0;
                for (int buffer = 0; buffer < 2; buffer++) begin
                    entry_count_q[buffer] <= 0; window_closed_q[buffer] <= 0;
                    for (int bank = 0; bank < 8; bank++)
                        bank_count_q[buffer][bank] <= 0;
                    for (int entry = 0; entry < MAX_WINDOW_DESCRIPTORS; entry++) begin
                        bitmap_q[buffer][entry] <= 0;
                        beat_index_q[buffer][entry] <= 0;
                    end
                end
            end else begin
                if (pair_release) begin
                    window_closed_q[pair_first] <= 0;
                    entry_count_q[pair_first] <= 0;
                    for (int bank = 0; bank < 8; bank++)
                        bank_count_q[pair_first][bank] <= 0;
                    if (group_pair_has_two_q) begin
                        window_closed_q[pair_second] <= 0;
                        entry_count_q[pair_second] <= 0;
                        for (int bank = 0; bank < 8; bank++)
                            bank_count_q[pair_second][bank] <= 0;
                    end
                    drain_select_q <= group_pair_has_two_q
                        ? drain_select_q : ~drain_select_q;
                end
                if (descriptor_accept) begin
                    descriptors_accepted_q <= descriptors_accepted_q
                        + descriptor_count;
                    token_has_index_q <= 1;
                    last_beat_index_q <= descriptor_last_index;
                    for (int lane = 0; lane < 4; lane++) begin
                        if (lane < descriptor_count) begin
                            bitmap_q[fill_select_q]
                                [fill_entry_count_effective + lane]
                                <= descriptor_bitmap[lane];
                            beat_index_q[fill_select_q]
                                [fill_entry_count_effective + lane]
                                <= descriptor_beat_index[lane];
                        end
                    end
                    entry_count_q[fill_select_q]
                        <= fill_entry_count_effective + descriptor_count;
                    for (int bank = 0; bank < 8; bank++)
                        bank_count_q[fill_select_q][bank]
                            <= (fill_window_releasing ? 0
                                : bank_count_q[fill_select_q][bank])
                                + descriptor_bank_sum[bank];
                    if (descriptor_closes_window) begin
                        window_closed_q[fill_select_q] <= 1;
                        fill_select_q <= ~fill_select_q;
                    end
                end
                if (upstream_done_accept) begin
                    upstream_done_seen_q <= 1;
                    if (entry_count_q[fill_select_q] != 0
                            && !window_closed_q[fill_select_q]) begin
                        window_closed_q[fill_select_q] <= 1;
                        fill_select_q <= ~fill_select_q;
                    end
                end
                if (candidate_load) begin
                    group_valid_q <= 1;
                    group_output_block_q <= 0;
                    group_source_count_q <= candidate_count;
                    group_bank_valid_q <= selected_valid;
                    group_pair_last_q <= candidate_pair_last;
                    group_pair_has_two_q <= pair_has_two;
                    for (int bank = 0; bank < 8; bank++) begin
                        group_source_channel_q[bank] <= selected_channel[bank];
                        if (selected_valid[bank]) begin
                            bank_count_q[selected_buffer[bank]][bank]
                                <= bank_count_q[selected_buffer[bank]][bank] - 1;
                            bitmap_q[selected_buffer[bank]][selected_entry[bank]]
                                [selected_row[bank]*8+bank] <= 0;
                        end
                    end
                end else if (group_final_accept) begin
                    if (group_pair_last_q) begin
                        group_valid_q <= 0; group_bank_valid_q <= 0;
                        group_pair_last_q <= 0;
                    end else begin
                        group_valid_q <= 0;
                    end
                end else if (group_accept) begin
                    group_output_block_q <= group_output_block_q + 1;
                end
            end
        end
    end
endmodule

`default_nettype wire
