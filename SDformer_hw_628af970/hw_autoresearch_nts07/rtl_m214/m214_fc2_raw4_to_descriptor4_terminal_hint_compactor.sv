`timescale 1ns/1ps
`default_nettype none

// M214: S4/F4 stable compactor with a fresh-arrival same-cycle bypass and an
// exact terminal-descriptor hint.  The hint accompanies the final nonzero
// descriptor packet when accepting that packet leaves no queued descriptor and
// the raw token is already complete (or completes on the same raw handshake).
// When the reservoir is empty, a legal raw packet may drive descriptors in
// the same cycle.  Descriptors after the first compact-window boundary are
// retained in the eight-entry queue.  Stalled visible output freezes raw input.
module m214_fc2_raw4_to_descriptor4_terminal_hint_compactor #(
    parameter int TAG_BITS = 24,
    parameter int QUEUE_DEPTH = 8
) (
    input logic clk_core, input logic rst_core,
    input logic header_valid, output logic header_ready,
    input logic [TAG_BITS-1:0] header_token_tag,
    input logic [5:0] header_raw_beat_count,
    input logic [3:0] header_window_depth,
    output logic header_accept,
    input logic raw_valid, output logic raw_ready,
    input logic [3:0] raw_lane_valid,
    input logic [4:0] raw_beat_index [0:3],
    input logic [95:0] raw_bitmap [0:3],
    input logic raw_last, output logic raw_accept,
    output logic descriptor_valid, input logic descriptor_ready,
    output logic [2:0] descriptor_count,
    output logic [TAG_BITS-1:0] descriptor_token_tag,
    output logic [4:0] descriptor_beat_index [0:3],
    output logic [95:0] descriptor_bitmap [0:3],
    output logic [3:0] descriptor_window_last,
    output logic descriptor_token_last,
    output logic descriptor_accept,
    output logic token_done_valid, input logic token_done_ready,
    output logic [TAG_BITS-1:0] token_done_tag,
    output logic [5:0] token_done_descriptor_count,
    output logic token_done_accept,
    output logic protocol_error, output logic busy
);
    localparam int COUNT_BITS = $clog2(QUEUE_DEPTH + 1);
    logic fault_q, token_active_q, raw_done_q, done_valid_q;
    logic [TAG_BITS-1:0] token_tag_q;
    logic [5:0] raw_beat_count_q, raw_beats_accepted_q;
    logic [3:0] window_depth_q, window_fill_q;
    logic [5:0] descriptor_total_q;
    logic [95:0] queue_bitmap_q [0:QUEUE_DEPTH-1];
    logic [4:0] queue_beat_index_q [0:QUEUE_DEPTH-1];
    logic queue_window_last_q [0:QUEUE_DEPTH-1];
    logic [COUNT_BITS-1:0] queue_count_q;
    logic [95:0] queue_bitmap_next [0:QUEUE_DEPTH-1];
    logic [4:0] queue_beat_index_next [0:QUEUE_DEPTH-1];
    logic queue_window_last_next [0:QUEUE_DEPTH-1];
    logic [COUNT_BITS-1:0] queue_count_next;
    logic header_legal, raw_prefix_legal, raw_index_legal;
    logic raw_last_legal, raw_packet_legal, illegal_request;
    logic [2:0] raw_lane_count, incoming_nonzero_count, push_count;
    logic [3:0] queue_pop_count, fresh_pop_count;
    logic [COUNT_BITS:0] queue_available_after_pop;
    logic [3:0] window_fill_after_push, push_window_last;
    logic [95:0] push_bitmap [0:3];
    logic [4:0] push_beat_index [0:3];
    logic fresh_mode, descriptor_drains_visible_source;

    function automatic logic is_prefix4(input logic [3:0] value);
        begin
            case (value)
                4'b0001, 4'b0011, 4'b0111, 4'b1111:
                    is_prefix4 = 1'b1;
                default: is_prefix4 = 1'b0;
            endcase
        end
    endfunction

    always_comb begin : input_analysis
        logic [3:0] local_fill;
        raw_lane_count = 0;
        incoming_nonzero_count = 0;
        raw_prefix_legal = is_prefix4(raw_lane_valid);
        raw_index_legal = token_active_q && !raw_done_q;
        push_count = 0;
        local_fill = window_fill_q;
        for (int lane = 0; lane < 4; lane++) begin
            push_bitmap[lane] = 0;
            push_beat_index[lane] = 0;
            push_window_last[lane] = 0;
            if (raw_lane_valid[lane]) begin
                raw_lane_count = raw_lane_count + 1;
                if (raw_beat_index[lane]
                        != raw_beats_accepted_q + lane)
                    raw_index_legal = 0;
                if (raw_bitmap[lane] != 0) begin
                    push_bitmap[push_count] = raw_bitmap[lane];
                    push_beat_index[push_count] = raw_beat_index[lane];
                    incoming_nonzero_count = incoming_nonzero_count + 1;
                    if (local_fill + 1 == window_depth_q) begin
                        push_window_last[push_count] = 1;
                        local_fill = 0;
                    end else begin
                        local_fill = local_fill + 1;
                    end
                    push_count = push_count + 1;
                end
            end
        end
        window_fill_after_push = local_fill;
        raw_last_legal = token_active_q && !raw_done_q
            && raw_lane_count != 0
            && raw_beats_accepted_q + raw_lane_count
                == raw_beat_count_q && raw_last;
        if (raw_beats_accepted_q + raw_lane_count
                != raw_beat_count_q)
            raw_last_legal = token_active_q && !raw_done_q
                && raw_lane_count != 0 && !raw_last;
        raw_packet_legal = token_active_q && !raw_done_q
            && raw_prefix_legal && raw_index_legal
            && raw_beats_accepted_q + raw_lane_count
                <= raw_beat_count_q && raw_last_legal;
    end

    assign fresh_mode = queue_count_q == 0 && raw_valid
        && raw_packet_legal && incoming_nonzero_count != 0 && !fault_q;

    always_comb begin : output_view
        logic boundary_found;
        logic [2:0] source_count;
        source_count = fresh_mode
            ? incoming_nonzero_count
            : (queue_count_q >= 4 ? 3'd4 : queue_count_q[2:0]);
        descriptor_count = 0;
        boundary_found = 0;
        if (fresh_mode || queue_count_q != 0) begin
            descriptor_count = source_count;
            for (int lane = 0; lane < 4; lane++) begin
                if (!boundary_found && lane < descriptor_count
                        && (fresh_mode ? push_window_last[lane]
                                       : queue_window_last_q[lane])) begin
                    descriptor_count = lane + 1;
                    boundary_found = 1;
                end
            end
        end
        descriptor_valid = (fresh_mode || queue_count_q != 0) && !fault_q;
        descriptor_drains_visible_source = fresh_mode
            ? descriptor_count == incoming_nonzero_count
            : descriptor_count == queue_count_q;
        // This is a property of the held descriptor packet, not of ready.  It
        // therefore remains stable under backpressure and is consumed atomically
        // with descriptor_accept by the downstream partial-window closer.
        descriptor_token_last = descriptor_valid
            && descriptor_drains_visible_source
            && (raw_done_q
                || (raw_valid && raw_packet_legal && raw_last
                    && (fresh_mode || incoming_nonzero_count == 0)));
        descriptor_accept = descriptor_valid && descriptor_ready;
        queue_pop_count = descriptor_accept && !fresh_mode
            ? {1'b0, descriptor_count} : 0;
        fresh_pop_count = descriptor_accept && fresh_mode
            ? {1'b0, descriptor_count} : 0;
        descriptor_token_tag = token_tag_q;
        for (int lane = 0; lane < 4; lane++) begin
            descriptor_beat_index[lane] = 0;
            descriptor_bitmap[lane] = 0;
            descriptor_window_last[lane] = 0;
            if (lane < descriptor_count) begin
                descriptor_beat_index[lane] = fresh_mode
                    ? push_beat_index[lane] : queue_beat_index_q[lane];
                descriptor_bitmap[lane] = fresh_mode
                    ? push_bitmap[lane] : queue_bitmap_q[lane];
                descriptor_window_last[lane] = fresh_mode
                    ? push_window_last[lane] : queue_window_last_q[lane];
            end
        end
    end

    always_comb begin : queue_transition
        integer remaining;
        integer enqueue_start;
        remaining = queue_count_q - queue_pop_count;
        enqueue_start = fresh_mode ? fresh_pop_count : 0;
        for (int slot = 0; slot < QUEUE_DEPTH; slot++) begin
            queue_bitmap_next[slot] = 0;
            queue_beat_index_next[slot] = 0;
            queue_window_last_next[slot] = 0;
            if (slot < remaining) begin
                queue_bitmap_next[slot]
                    = queue_bitmap_q[slot + queue_pop_count];
                queue_beat_index_next[slot]
                    = queue_beat_index_q[slot + queue_pop_count];
                queue_window_last_next[slot]
                    = queue_window_last_q[slot + queue_pop_count];
            end
        end
        if (raw_accept) begin
            for (int push = 0; push < 4; push++) begin
                if (push >= enqueue_start && push < push_count) begin
                    queue_bitmap_next[remaining + push - enqueue_start]
                        = push_bitmap[push];
                    queue_beat_index_next[remaining + push - enqueue_start]
                        = push_beat_index[push];
                    queue_window_last_next[remaining + push - enqueue_start]
                        = push_window_last[push];
                end
            end
        end
        queue_count_next = remaining + (raw_accept
            ? incoming_nonzero_count - enqueue_start : 0);
    end

    assign header_legal = header_raw_beat_count >= 1
        && header_raw_beat_count <= 32
        && (header_window_depth == 2 || header_window_depth == 4
            || header_window_depth == 8);
    assign header_ready = !fault_q && !token_active_q && header_legal;
    assign header_accept = header_valid && header_ready;
    assign queue_available_after_pop = QUEUE_DEPTH
        - queue_count_q + queue_pop_count;
    assign raw_ready = !fault_q && raw_packet_legal
        && !(descriptor_valid && !descriptor_ready)
        && incoming_nonzero_count - fresh_pop_count
            <= queue_available_after_pop;
    assign raw_accept = raw_valid && raw_ready;
    assign illegal_request = (header_valid
            && (!header_legal || token_active_q))
        || (raw_valid && !raw_packet_legal);
    assign token_done_valid = done_valid_q && !fault_q;
    assign token_done_accept = token_done_valid && token_done_ready;
    assign token_done_tag = token_tag_q;
    assign token_done_descriptor_count = descriptor_total_q;
    assign protocol_error = fault_q || illegal_request;
    assign busy = token_active_q || queue_count_q != 0 || done_valid_q;

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            fault_q <= 0; token_active_q <= 0; raw_done_q <= 0;
            done_valid_q <= 0; token_tag_q <= 0; raw_beat_count_q <= 0;
            raw_beats_accepted_q <= 0; window_depth_q <= 0;
            window_fill_q <= 0; descriptor_total_q <= 0;
            queue_count_q <= 0;
            for (int slot = 0; slot < QUEUE_DEPTH; slot++) begin
                queue_bitmap_q[slot] <= 0;
                queue_beat_index_q[slot] <= 0;
                queue_window_last_q[slot] <= 0;
            end
        end else begin
            if (illegal_request) fault_q <= 1;
            if (header_accept) begin
                token_active_q <= 1; raw_done_q <= 0; done_valid_q <= 0;
                token_tag_q <= header_token_tag;
                raw_beat_count_q <= header_raw_beat_count;
                raw_beats_accepted_q <= 0;
                window_depth_q <= header_window_depth;
                window_fill_q <= 0; descriptor_total_q <= 0;
                queue_count_q <= 0;
            end
            if (raw_accept || descriptor_accept) begin
                queue_count_q <= queue_count_next;
                for (int slot = 0; slot < QUEUE_DEPTH; slot++) begin
                    queue_bitmap_q[slot] <= queue_bitmap_next[slot];
                    queue_beat_index_q[slot] <= queue_beat_index_next[slot];
                    queue_window_last_q[slot]
                        <= queue_window_last_next[slot];
                end
            end
            if (raw_accept) begin
                raw_beats_accepted_q <= raw_beats_accepted_q
                    + raw_lane_count;
                window_fill_q <= window_fill_after_push;
                descriptor_total_q <= descriptor_total_q
                    + incoming_nonzero_count;
                if (raw_last) raw_done_q <= 1;
            end
            if ((raw_done_q || (raw_accept && raw_last))
                    && queue_count_next == 0)
                done_valid_q <= 1;
            if (token_done_accept) begin
                token_active_q <= 0; raw_done_q <= 0; done_valid_q <= 0;
                raw_beats_accepted_q <= 0; window_fill_q <= 0;
                descriptor_total_q <= 0;
            end
        end
    end
endmodule

`default_nettype wire
