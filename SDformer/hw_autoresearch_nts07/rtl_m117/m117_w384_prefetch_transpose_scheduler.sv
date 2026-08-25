`timescale 1ns/1ps
`default_nettype none

// M117 W384 source-key transpose with one-key-ahead weight prefetch.
//
// Natural-raster events set one (key,row) bitmap position in the fill bank.
// The drain bank walks keys and rows in ascending order, emitting exactly
// three weight-load slots followed by one event slot per set position.  Two
// banks allow fill and drain to overlap.  A separate whole-vector prefetch
// handshake is issued on descriptor dispatch and while the final event of the
// current key is visible.  With an always-ready lane-sliced weight SRAM, the
// next key's beat0 follows the prior key's final event without an extra group
// bubble.  Payload SRAM and integer accumulation remain explicit port cuts.
module m117_w384_prefetch_transpose_scheduler #(
    parameter int WIN_ROWS = 384,
    parameter int ROW_W = $clog2(WIN_ROWS),
    parameter int BASE_W = 12,
    parameter int CONTEXT_W = 16
) (
    input  logic                     clk_core,
    input  logic                     rst_core,

    input  logic                     event_valid,
    output logic                     event_ready,
    input  logic [3:0]               event_source,
    input  logic [2:0]               event_block,
    input  logic [ROW_W-1:0]         event_row_offset,
    input  logic                     event_negate,
    input  logic [BASE_W-1:0]        window_base_row,
    input  logic [CONTEXT_W-1:0]     window_context,
    output logic                     event_accept,

    input  logic                     window_close_valid,
    output logic                     window_close_ready,
    output logic                     window_close_accept,

    output logic                     service_valid,
    input  logic                     service_ready,
    output logic                     service_is_event,
    output logic [3:0]               service_source,
    output logic [2:0]               service_block,
    output logic [1:0]               service_load_beat,
    output logic [ROW_W-1:0]         service_row_offset,
    output logic [BASE_W-1:0]        service_destination_row,
    output logic                     service_negate,
    output logic                     service_last_for_key,
    output logic [CONTEXT_W-1:0]     service_context,
    output logic                     service_accept,

    output logic                     weight_prefetch_valid,
    input  logic                     weight_prefetch_ready,
    output logic [3:0]               weight_prefetch_source,
    output logic [2:0]               weight_prefetch_block,
    output logic [CONTEXT_W-1:0]     weight_prefetch_context,
    output logic                     weight_prefetch_accept,

    output logic                     descriptor_done,
    output logic                     descriptor_done_empty,
    output logic [BASE_W-1:0]        descriptor_done_base_row,
    output logic [CONTEXT_W-1:0]     descriptor_done_context,

    output logic                     fill_bank,
    output logic                     drain_bank,
    output logic [1:0]               bank_ready,
    output logic                     protocol_error,
    output logic                     busy
);
    localparam logic [1:0] BANK_EMPTY = 2'd0;
    localparam logic [1:0] BANK_FILL  = 2'd1;
    localparam logic [1:0] BANK_READY = 2'd2;
    localparam logic [1:0] BANK_DRAIN = 2'd3;

    logic [1:0] bank_state_q [0:1];
    logic fill_available_q, fill_bank_q, next_drain_bank_q;
    logic drain_active_q, drain_bank_q, drain_event_phase_q;
    logic drain_prefetch_wait_q, next_key_prefetched_q;
    logic [1:0] drain_load_beat_q;
    logic [6:0] drain_key_q;
    logic [ROW_W-1:0] drain_row_q;

    // One presence and one direction bit per possible (key,row) event.  The
    // fixed address is the transpose: no linked-list pointer or row payload.
    logic [WIN_ROWS-1:0] row_valid_q [0:1][0:127];
    logic [WIN_ROWS-1:0] row_negate_q [0:1][0:127];
    logic [127:0] active_key_q [0:1];
    logic [BASE_W-1:0] bank_base_q [0:1];
    logic [CONTEXT_W-1:0] bank_context_q [0:1];
    logic identity_valid_q [0:1];

    logic request_fault_q;
    logic event_semantically_valid, close_semantically_valid;
    logic event_identity_match, close_identity_match;
    logic row_in_range, duplicate_event, request_collision;
    logic event_violation, close_violation, illegal_request;

    // Exact accepted-request grace prevents a producer that deasserts valid
    // after observing the active edge from being mistaken for a new request.
    // Any identity mutation remains a combinational fail-closed violation.
    logic accepted_event_grace_q, accepted_close_grace_q;
    logic [3:0] accepted_event_source_q;
    logic [2:0] accepted_event_block_q;
    logic [ROW_W-1:0] accepted_event_row_q;
    logic accepted_event_negate_q;
    logic [BASE_W-1:0] accepted_event_base_q, accepted_close_base_q;
    logic [CONTEXT_W-1:0] accepted_event_context_q;
    logic [CONTEXT_W-1:0] accepted_close_context_q;
    logic accepted_event_grace_match, accepted_close_grace_match;

    logic [WIN_ROWS-1:0] current_rows, rows_after_current;
    logic [127:0] keys_after_current;
    logic [6:0] next_service_key;
    logic initial_prefetch_request, next_key_prefetch_request;

`ifndef SYNTHESIS
    initial begin
        if (WIN_ROWS < 2 || WIN_ROWS > (1 << ROW_W))
            $fatal(1, "M117 invalid frozen window geometry");
        if (WIN_ROWS != 384 || ROW_W != 9 || BASE_W != 12
                || CONTEXT_W != 16)
            $fatal(1, "M117 W384 production geometry drift");
    end
`endif

    function automatic logic [6:0] first_key(
        input logic [127:0] bitmap
    );
        logic found;
        begin
            first_key = '0;
            found = 1'b0;
            for (int key = 0; key < 128; key++) begin
                if (!found && bitmap[key]) begin
                    first_key = key[6:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [ROW_W-1:0] first_row(
        input logic [WIN_ROWS-1:0] bitmap
    );
        logic found;
        begin
            first_row = '0;
            found = 1'b0;
            for (int row = 0; row < WIN_ROWS; row++) begin
                if (!found && bitmap[row]) begin
                    first_row = row[ROW_W-1:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [WIN_ROWS-1:0] row_onehot(
        input logic [ROW_W-1:0] row
    );
        begin
            row_onehot = '0;
            if (row < WIN_ROWS)
                row_onehot[row] = 1'b1;
        end
    endfunction

    function automatic logic [127:0] key_onehot(input logic [6:0] key);
        begin
            key_onehot = '0;
            key_onehot[key] = 1'b1;
        end
    endfunction

    always_comb begin : request_audit
        row_in_range = event_row_offset < WIN_ROWS;
        event_identity_match = !identity_valid_q[fill_bank_q]
                             || (window_base_row == bank_base_q[fill_bank_q]
                                 && window_context
                                    == bank_context_q[fill_bank_q]);
        close_identity_match = event_identity_match;
        duplicate_event = fill_available_q && row_in_range
                        && row_valid_q[fill_bank_q]
                                      [{event_source, event_block}]
                                      [event_row_offset];
        event_semantically_valid = fill_available_q
                                 && bank_state_q[fill_bank_q] == BANK_FILL
                                 && row_in_range && event_identity_match
                                 && !duplicate_event;
        close_semantically_valid = fill_available_q
                                 && bank_state_q[fill_bank_q] == BANK_FILL
                                 && close_identity_match;
        accepted_event_grace_match = accepted_event_grace_q
                                  && event_source
                                     == accepted_event_source_q
                                  && event_block == accepted_event_block_q
                                  && event_row_offset
                                     == accepted_event_row_q
                                  && event_negate
                                     == accepted_event_negate_q
                                  && window_base_row
                                     == accepted_event_base_q
                                  && window_context
                                     == accepted_event_context_q;
        accepted_close_grace_match = accepted_close_grace_q
                                  && window_base_row
                                     == accepted_close_base_q
                                  && window_context
                                     == accepted_close_context_q;
        request_collision = event_valid && window_close_valid;
        // Standard ready/valid streaming permits a different legal payload on
        // the next edge without an intervening sampled-low cycle.  The exact
        // accepted identity alone is grace-held and cannot be accepted twice;
        // a changed payload is audited as a new transaction.
        event_violation = event_valid && !event_semantically_valid
                        && !accepted_event_grace_match;
        close_violation = window_close_valid && !close_semantically_valid
                        && !accepted_close_grace_match;
        illegal_request = request_collision || event_violation
                        || close_violation;
    end

    assign protocol_error = request_fault_q || illegal_request;
    assign event_ready = !protocol_error && !window_close_valid
                       && !accepted_event_grace_match
                       && event_semantically_valid;
    assign window_close_ready = !protocol_error && !event_valid
                              && !accepted_close_grace_match
                              && close_semantically_valid;
    assign event_accept = event_valid && event_ready;
    assign window_close_accept = window_close_valid && window_close_ready;

    always_comb begin : service_mapper
        current_rows = row_valid_q[drain_bank_q][drain_key_q];
        rows_after_current = current_rows & ~row_onehot(drain_row_q);
        keys_after_current = active_key_q[drain_bank_q]
                           & ~key_onehot(drain_key_q);

        service_valid = !protocol_error && drain_active_q
                      && !drain_prefetch_wait_q;
        service_is_event = drain_event_phase_q;
        service_source = drain_key_q[6:3];
        service_block = drain_key_q[2:0];
        service_load_beat = drain_event_phase_q ? '0 : drain_load_beat_q;
        service_row_offset = drain_event_phase_q ? drain_row_q : '0;
        service_destination_row = drain_event_phase_q
                                ? bank_base_q[drain_bank_q] + drain_row_q
                                : '0;
        service_negate = drain_event_phase_q
                       && row_negate_q[drain_bank_q][drain_key_q][drain_row_q];
        service_last_for_key = drain_event_phase_q
                            && rows_after_current == '0;
        service_context = bank_context_q[drain_bank_q];
        service_accept = service_valid && service_ready;

        initial_prefetch_request = !drain_active_q
                                 && bank_state_q[next_drain_bank_q]
                                    == BANK_READY
                                 && active_key_q[next_drain_bank_q] != '0;
        next_key_prefetch_request = drain_active_q
                                  && !drain_prefetch_wait_q
                                  && drain_event_phase_q
                                  && rows_after_current == '0
                                  && keys_after_current != '0
                                  && !next_key_prefetched_q;
        next_service_key = drain_key_q;
        if (initial_prefetch_request)
            next_service_key = first_key(active_key_q[next_drain_bank_q]);
        else if (next_key_prefetch_request)
            next_service_key = first_key(keys_after_current);

        weight_prefetch_valid = !protocol_error
                              && (initial_prefetch_request
                                  || drain_prefetch_wait_q
                                  || next_key_prefetch_request);
        weight_prefetch_source = next_service_key[6:3];
        weight_prefetch_block = next_service_key[2:0];
        weight_prefetch_context = initial_prefetch_request
                                ? bank_context_q[next_drain_bank_q]
                                : bank_context_q[drain_bank_q];
        weight_prefetch_accept = weight_prefetch_valid
                               && weight_prefetch_ready;
    end

    assign fill_bank = fill_bank_q;
    assign drain_bank = drain_bank_q;
    assign bank_ready = {
        bank_state_q[1] == BANK_READY,
        bank_state_q[0] == BANK_READY
    };
    assign busy = drain_active_q || bank_state_q[0] == BANK_READY
                || bank_state_q[1] == BANK_READY
                || identity_valid_q[0] || identity_valid_q[1];

    always_ff @(posedge clk_core) begin : state_update
        if (rst_core) begin
            bank_state_q[0] <= BANK_FILL;
            bank_state_q[1] <= BANK_EMPTY;
            fill_available_q <= 1'b1;
            fill_bank_q <= 1'b0;
            next_drain_bank_q <= 1'b0;
            drain_active_q <= 1'b0;
            drain_bank_q <= 1'b0;
            drain_event_phase_q <= 1'b0;
            drain_prefetch_wait_q <= 1'b0;
            next_key_prefetched_q <= 1'b0;
            drain_load_beat_q <= '0;
            drain_key_q <= '0;
            drain_row_q <= '0;
            active_key_q[0] <= '0;
            active_key_q[1] <= '0;
            bank_base_q[0] <= '0;
            bank_base_q[1] <= '0;
            bank_context_q[0] <= '0;
            bank_context_q[1] <= '0;
            identity_valid_q[0] <= 1'b0;
            identity_valid_q[1] <= 1'b0;
            request_fault_q <= 1'b0;
            accepted_event_grace_q <= 1'b0;
            accepted_close_grace_q <= 1'b0;
            accepted_event_source_q <= '0;
            accepted_event_block_q <= '0;
            accepted_event_row_q <= '0;
            accepted_event_negate_q <= 1'b0;
            accepted_event_base_q <= '0;
            accepted_event_context_q <= '0;
            accepted_close_base_q <= '0;
            accepted_close_context_q <= '0;
            descriptor_done <= 1'b0;
            descriptor_done_empty <= 1'b0;
            descriptor_done_base_row <= '0;
            descriptor_done_context <= '0;
            for (int bank = 0; bank < 2; bank++) begin
                for (int key = 0; key < 128; key++) begin
                    row_valid_q[bank][key] <= '0;
                    row_negate_q[bank][key] <= '0;
                end
            end
        end else begin
            descriptor_done <= 1'b0;
            descriptor_done_empty <= 1'b0;
            if (!event_valid || !accepted_event_grace_match)
                accepted_event_grace_q <= 1'b0;
            if (!window_close_valid || !accepted_close_grace_match)
                accepted_close_grace_q <= 1'b0;
            if (illegal_request)
                request_fault_q <= 1'b1;

            if (!protocol_error) begin
                if (!fill_available_q) begin
                    if (bank_state_q[0] == BANK_EMPTY) begin
                        bank_state_q[0] <= BANK_FILL;
                        fill_bank_q <= 1'b0;
                        fill_available_q <= 1'b1;
                        identity_valid_q[0] <= 1'b0;
                        active_key_q[0] <= '0;
                    end else if (bank_state_q[1] == BANK_EMPTY) begin
                        bank_state_q[1] <= BANK_FILL;
                        fill_bank_q <= 1'b1;
                        fill_available_q <= 1'b1;
                        identity_valid_q[1] <= 1'b0;
                        active_key_q[1] <= '0;
                    end
                end

                if (event_accept) begin
                    row_valid_q[fill_bank_q][{event_source, event_block}]
                               [event_row_offset] <= 1'b1;
                    row_negate_q[fill_bank_q][{event_source, event_block}]
                                [event_row_offset] <= event_negate;
                    active_key_q[fill_bank_q]
                                [{event_source, event_block}] <= 1'b1;
                    if (!identity_valid_q[fill_bank_q]) begin
                        identity_valid_q[fill_bank_q] <= 1'b1;
                        bank_base_q[fill_bank_q] <= window_base_row;
                        bank_context_q[fill_bank_q] <= window_context;
                    end
                    accepted_event_grace_q <= 1'b1;
                    accepted_event_source_q <= event_source;
                    accepted_event_block_q <= event_block;
                    accepted_event_row_q <= event_row_offset;
                    accepted_event_negate_q <= event_negate;
                    accepted_event_base_q <= window_base_row;
                    accepted_event_context_q <= window_context;
                end

                if (window_close_accept) begin
                    if (!identity_valid_q[fill_bank_q]) begin
                        identity_valid_q[fill_bank_q] <= 1'b1;
                        bank_base_q[fill_bank_q] <= window_base_row;
                        bank_context_q[fill_bank_q] <= window_context;
                    end
                    bank_state_q[fill_bank_q] <= BANK_READY;
                    if (bank_state_q[~fill_bank_q] == BANK_EMPTY) begin
                        bank_state_q[~fill_bank_q] <= BANK_FILL;
                        fill_bank_q <= ~fill_bank_q;
                        fill_available_q <= 1'b1;
                        identity_valid_q[~fill_bank_q] <= 1'b0;
                        active_key_q[~fill_bank_q] <= '0;
                    end else begin
                        fill_available_q <= 1'b0;
                    end
                    accepted_close_grace_q <= 1'b1;
                    accepted_close_base_q <= window_base_row;
                    accepted_close_context_q <= window_context;
                end

                if (!drain_active_q
                        && bank_state_q[next_drain_bank_q] == BANK_READY) begin
                    if (active_key_q[next_drain_bank_q] == '0) begin
                        bank_state_q[next_drain_bank_q] <= BANK_EMPTY;
                        identity_valid_q[next_drain_bank_q] <= 1'b0;
                        next_drain_bank_q <= ~next_drain_bank_q;
                        descriptor_done <= 1'b1;
                        descriptor_done_empty <= 1'b1;
                        descriptor_done_base_row
                            <= bank_base_q[next_drain_bank_q];
                        descriptor_done_context
                            <= bank_context_q[next_drain_bank_q];
                    end else if (weight_prefetch_accept) begin
                        bank_state_q[next_drain_bank_q] <= BANK_DRAIN;
                        drain_active_q <= 1'b1;
                        drain_bank_q <= next_drain_bank_q;
                        drain_event_phase_q <= 1'b0;
                        drain_prefetch_wait_q <= 1'b0;
                        next_key_prefetched_q <= 1'b0;
                        drain_load_beat_q <= '0;
                        drain_key_q <= first_key(
                            active_key_q[next_drain_bank_q]);
                        drain_row_q <= first_row(row_valid_q[
                            next_drain_bank_q][first_key(
                                active_key_q[next_drain_bank_q])]);
                    end
                end

                // A prefetch may complete while the final event is stalled.
                // Remember it so the event can retire later without issuing a
                // duplicate read.  If backpressure prevented the lookahead,
                // the next key waits off the service interface until the same
                // request is eventually accepted.
                if (weight_prefetch_accept && drain_active_q
                        && !drain_prefetch_wait_q && drain_event_phase_q
                        && rows_after_current == '0
                        && keys_after_current != '0 && !service_accept)
                    next_key_prefetched_q <= 1'b1;
                if (drain_prefetch_wait_q && weight_prefetch_accept)
                    drain_prefetch_wait_q <= 1'b0;

                if (service_accept) begin
                    if (!drain_event_phase_q) begin
                        if (drain_load_beat_q == 2) begin
                            drain_event_phase_q <= 1'b1;
                            drain_load_beat_q <= '0;
                        end else begin
                            drain_load_beat_q <= drain_load_beat_q + 1'b1;
                        end
                    end else begin
                        row_valid_q[drain_bank_q][drain_key_q]
                                   [drain_row_q] <= 1'b0;
                        if (rows_after_current != '0) begin
                            drain_row_q <= first_row(rows_after_current);
                        end else begin
                            active_key_q[drain_bank_q][drain_key_q] <= 1'b0;
                            if (keys_after_current != '0) begin
                                drain_key_q <= first_key(keys_after_current);
                                drain_row_q <= first_row(row_valid_q[
                                    drain_bank_q][first_key(
                                        keys_after_current)]);
                                drain_event_phase_q <= 1'b0;
                                drain_load_beat_q <= '0;
                                drain_prefetch_wait_q
                                    <= !(next_key_prefetched_q
                                         || weight_prefetch_accept);
                                next_key_prefetched_q <= 1'b0;
                            end else begin
                                drain_active_q <= 1'b0;
                                drain_prefetch_wait_q <= 1'b0;
                                next_key_prefetched_q <= 1'b0;
                                bank_state_q[drain_bank_q] <= BANK_EMPTY;
                                identity_valid_q[drain_bank_q] <= 1'b0;
                                next_drain_bank_q <= ~next_drain_bank_q;
                                descriptor_done <= 1'b1;
                                descriptor_done_empty <= 1'b0;
                                descriptor_done_base_row
                                    <= bank_base_q[drain_bank_q];
                                descriptor_done_context
                                    <= bank_context_q[drain_bank_q];
                            end
                        end
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
