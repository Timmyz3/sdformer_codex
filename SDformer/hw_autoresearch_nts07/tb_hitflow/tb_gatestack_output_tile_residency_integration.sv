`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_output_tile_residency_integration;
    localparam int HEADS = 2;
    localparam int CACHE_TERMS = 4;
    localparam int HEAD_ID_W = 1;
    localparam logic [31:0] GROUP_TAG = 32'ha200_0051;

    logic clk_core, rst_core;

    logic group_valid, group_ready, group_context_id;
    logic [31:0] group_tag;
    logic [1:0] group_head_count;
    logic [7:0] group_first_output_tile, group_output_tile_count;
    logic tile_start_valid, tile_start_ready;
    logic [31:0] tile_start_tag;
    logic [7:0] tile_start_output_tile;
    logic [1:0] tile_start_head_count;
    logic head_issue_valid, head_issue_ready, head_issue_context_id;
    logic [31:0] head_issue_tag;
    logic [HEAD_ID_W-1:0] head_issue_head_id;
    logic [1:0] head_issue_head_index;
    logic [9:0] head_issue_input_channel_base;
    logic [7:0] head_issue_output_tile;
    logic head_issue_last_head, head_issue_last_output_tile;
    logic head_done_valid, head_done_ready;
    logic [31:0] head_done_tag;
    logic [HEAD_ID_W-1:0] head_done_head_id;
    logic head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [31:0] tile_done_tag;
    logic tile_done_error;
    logic group_done_valid, group_done_ready;
    logic [31:0] group_done_tag;
    logic group_done_error, scheduler_protocol_error;
    logic [31:0] count_groups, count_tile_starts;
    logic [31:0] count_head_issues, count_group_errors;

    logic fill_begin_valid, fill_begin_ready, fill_context_id;
    logic [HEAD_ID_W-1:0] fill_head_id;
    logic [31:0] fill_tag;
    logic [7:0] fill_term_count;
    logic fill_begin_cacheable;
    logic fill_entry_valid, fill_entry_ready;
    logic [8:0] fill_gate_code;
    logic [4:0] fill_lane_id;
    logic [7:0] fill_destination_count;
    logic fill_entry_last;
    logic lookup_valid, lookup_ready, lookup_context_id;
    logic [HEAD_ID_W-1:0] lookup_head_id;
    logic [31:0] lookup_expected_tag;
    logic lookup_meta_valid, lookup_meta_ready, lookup_hit;
    logic [31:0] lookup_tag;
    logic [7:0] lookup_term_count;
    logic lookup_entry_valid, lookup_entry_ready;
    logic [8:0] lookup_gate_code;
    logic [4:0] lookup_lane_id;
    logic [7:0] lookup_destination_count;
    logic [1:0] lookup_term_index;
    logic lookup_entry_last;
    logic cache_release_valid, cache_release_ready, cache_release_context_id;
    logic [HEAD_ID_W-1:0] cache_release_head_id;
    logic [31:0] cache_release_payload_tag;
    logic [HEADS-1:0] cache_valid_flat;
    logic cache_protocol_error;
    logic [31:0] count_cached_heads, count_bypass_heads;
    logic [31:0] count_lookup_hits, count_lookup_misses, count_releases;
    logic [31:0] count_release_noops, count_release_tag_mismatches;

    logic session_valid, session_ready, session_context_id;
    logic [HEAD_ID_W-1:0] session_head_id;
    logic [31:0] session_payload_tag, session_execution_tag;
    logic session_cache_owned, session_last_output_tile;
    logic decoder_done_valid, decoder_done_ready;
    logic [31:0] decoder_done_payload_tag;
    logic decoder_done_error;
    logic backend_done_valid, backend_done_ready;
    logic [31:0] backend_done_execution_tag;
    logic backend_done_error;
    logic slot_release_valid, slot_release_ready, slot_release_context_id;
    logic [HEAD_ID_W-1:0] slot_release_head_id;
    logic lifecycle_cache_release_valid, lifecycle_cache_release_ready;
    logic lifecycle_cache_release_context_id;
    logic [HEAD_ID_W-1:0] lifecycle_cache_release_head_id;
    logic [31:0] lifecycle_cache_release_payload_tag;
    logic session_done_valid, session_done_ready;
    logic [31:0] session_done_payload_tag;
    logic [31:0] session_done_execution_tag;
    logic session_done_error, lifecycle_protocol_error;
    logic [31:0] count_sessions, count_final_tile_releases;
    logic [31:0] count_cache_releases, count_session_errors;

    logic [HEAD_ID_W-1:0] active_head_q;
    logic [31:0] active_execution_tag_q;
    int slot_release_count;
    int descriptor_entries_read;

    gatestack_output_tile_scheduler #(
        .HEADS(HEADS), .LANES(32), .HEAD_COUNT_W(2),
        .HEAD_ID_W(HEAD_ID_W)
    ) u_scheduler (
        .clk_core, .rst_core,
        .group_valid, .group_ready, .group_context_id, .group_tag,
        .group_head_count, .group_first_output_tile,
        .group_output_tile_count,
        .tile_start_valid, .tile_start_ready, .tile_start_tag,
        .tile_start_output_tile, .tile_start_head_count,
        .head_issue_valid, .head_issue_ready, .head_issue_context_id,
        .head_issue_tag, .head_issue_head_id, .head_issue_head_index,
        .head_issue_input_channel_base, .head_issue_output_tile,
        .head_issue_last_head, .head_issue_last_output_tile,
        .head_done_valid, .head_done_ready, .head_done_tag,
        .head_done_head_id, .head_done_error,
        .tile_done_valid, .tile_done_ready, .tile_done_tag,
        .tile_done_error, .group_done_valid, .group_done_ready,
        .group_done_tag, .group_done_error,
        .protocol_error(scheduler_protocol_error),
        .count_groups, .count_tile_starts, .count_head_issues,
        .count_group_errors
    );

    gatestack_descriptor_residency_cache #(
        .CONTEXTS(1), .HEADS(HEADS), .CACHE_TERMS(CACHE_TERMS),
        .CONTEXT_ID_W(1), .HEAD_ID_W(HEAD_ID_W)
    ) u_cache (
        .clk_core, .rst_core,
        .fill_begin_valid, .fill_begin_ready, .fill_context_id,
        .fill_head_id, .fill_tag, .fill_term_count,
        .fill_begin_cacheable, .fill_entry_valid, .fill_entry_ready,
        .fill_gate_code, .fill_lane_id, .fill_destination_count,
        .fill_entry_last, .lookup_valid, .lookup_ready,
        .lookup_context_id, .lookup_head_id, .lookup_expected_tag,
        .lookup_meta_valid, .lookup_meta_ready, .lookup_hit,
        .lookup_tag, .lookup_term_count, .lookup_entry_valid,
        .lookup_entry_ready, .lookup_gate_code, .lookup_lane_id,
        .lookup_destination_count, .lookup_term_index,
        .lookup_entry_last, .release_valid(cache_release_valid),
        .release_ready(cache_release_ready),
        .release_context_id(cache_release_context_id),
        .release_head_id(cache_release_head_id),
        .release_expected_tag(cache_release_payload_tag), .cache_valid_flat,
        .protocol_error(cache_protocol_error), .count_cached_heads,
        .count_bypass_heads, .count_lookup_hits, .count_lookup_misses,
        .count_releases, .count_release_noops,
        .count_release_tag_mismatches
    );

    gatestack_dualtag_replay_lifecycle_manager #(
        .CONTEXTS(1), .HEADS(HEADS), .CONTEXT_ID_W(1),
        .HEAD_ID_W(HEAD_ID_W)
    ) u_lifecycle (
        .clk_core, .rst_core, .session_valid, .session_ready,
        .session_context_id, .session_head_id, .session_payload_tag,
        .session_execution_tag,
        .session_cache_owned, .session_last_output_tile,
        .decoder_done_valid, .decoder_done_ready,
        .decoder_done_payload_tag,
        .decoder_done_error, .backend_done_valid, .backend_done_ready,
        .backend_done_execution_tag, .backend_done_error,
        .slot_release_valid,
        .slot_release_ready, .slot_release_context_id,
        .slot_release_head_id,
        .cache_release_valid(lifecycle_cache_release_valid),
        .cache_release_ready(lifecycle_cache_release_ready),
        .cache_release_context_id(lifecycle_cache_release_context_id),
        .cache_release_head_id(lifecycle_cache_release_head_id),
        .cache_release_payload_tag(lifecycle_cache_release_payload_tag),
        .session_done_valid, .session_done_ready,
        .session_done_payload_tag, .session_done_execution_tag,
        .session_done_error,
        .protocol_error(lifecycle_protocol_error), .count_sessions,
        .count_final_tile_releases, .count_cache_releases,
        .count_session_errors
    );

    assign cache_release_valid = lifecycle_cache_release_valid;
    assign lifecycle_cache_release_ready = cache_release_ready;
    assign cache_release_context_id = lifecycle_cache_release_context_id;
    assign cache_release_head_id = lifecycle_cache_release_head_id;
    assign cache_release_payload_tag = lifecycle_cache_release_payload_tag;
    assign slot_release_ready = 1'b1;
    assign head_done_valid = session_done_valid;
    assign session_done_ready = head_done_ready;
    assign head_done_tag = session_done_execution_tag;
    assign head_done_head_id = active_head_q;
    assign head_done_error = session_done_error ||
                             session_done_payload_tag != GROUP_TAG;

    always #5 clk_core <= ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            slot_release_count <= 0;
        end else if (slot_release_valid && slot_release_ready) begin
            if (slot_release_context_id != 0 ||
                slot_release_head_id != active_head_q)
                $fatal(1, "slot release identity mismatch");
            slot_release_count <= slot_release_count + 1;
        end
    end

    task automatic fill_head(input logic head_id, input int terms);
        begin
            @(negedge clk_core);
            fill_context_id = 1'b0;
            fill_head_id = head_id;
            fill_tag = GROUP_TAG;
            fill_term_count = 8'(terms);
            fill_begin_valid = 1'b1;
            do @(posedge clk_core); while (!fill_begin_ready);
            @(negedge clk_core);
            fill_begin_valid = 1'b0;
            for (int term = 0; term < terms; term = term + 1) begin
                fill_gate_code = 9'(16 * head_id + term + 1);
                fill_lane_id = 5'(4 * head_id + term);
                fill_destination_count = 8'(term + 2);
                fill_entry_last = term == terms - 1;
                fill_entry_valid = 1'b1;
                do @(posedge clk_core); while (!fill_entry_ready);
                @(negedge clk_core);
                fill_entry_valid = 1'b0;
                fill_entry_last = 1'b0;
            end
        end
    endtask

    task automatic lookup_and_consume(input logic head_id,
                                      input int terms);
        int consumed;
        int cycles;
        begin
            consumed = 0;
            cycles = 0;
            @(negedge clk_core);
            lookup_context_id = 1'b0;
            lookup_head_id = head_id;
            lookup_expected_tag = GROUP_TAG;
            lookup_valid = 1'b1;
            do @(posedge clk_core); while (!lookup_ready);
            @(negedge clk_core);
            lookup_valid = 1'b0;
            while (!lookup_meta_valid) @(posedge clk_core);
            if (!lookup_hit || lookup_tag != GROUP_TAG ||
                lookup_term_count != 8'(terms))
                $fatal(1, "descriptor cache miss during tile reuse");
            @(negedge clk_core);
            lookup_meta_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            lookup_meta_ready = 1'b0;
            while (consumed < terms) begin
                @(negedge clk_core);
                lookup_entry_ready = (cycles % 3) != 1;
                @(posedge clk_core);
                if (lookup_entry_valid && lookup_entry_ready) begin
                    if (lookup_gate_code != 9'(16 * head_id + consumed + 1) ||
                        lookup_lane_id != 5'(4 * head_id + consumed) ||
                        lookup_destination_count != 8'(consumed + 2) ||
                        lookup_term_index != 2'(consumed) ||
                        lookup_entry_last != (consumed == terms - 1))
                        $fatal(1, "resident descriptor payload mismatch");
                    consumed = consumed + 1;
                    descriptor_entries_read = descriptor_entries_read + 1;
                end
                cycles = cycles + 1;
            end
            @(negedge clk_core);
            lookup_entry_ready = 1'b0;
        end
    endtask

    task automatic send_decoder_done;
        begin
            @(negedge clk_core);
            decoder_done_payload_tag = GROUP_TAG;
            decoder_done_error = 1'b0;
            decoder_done_valid = 1'b1;
            do @(posedge clk_core); while (!decoder_done_ready);
            @(negedge clk_core);
            decoder_done_valid = 1'b0;
        end
    endtask

    task automatic send_backend_done;
        begin
            @(negedge clk_core);
            backend_done_execution_tag = active_execution_tag_q;
            backend_done_error = 1'b0;
            backend_done_valid = 1'b1;
            do @(posedge clk_core); while (!backend_done_ready);
            @(negedge clk_core);
            backend_done_valid = 1'b0;
        end
    endtask

    task automatic service_head(input int tile, input logic head_id);
        int terms;
        begin
            terms = head_id ? 3 : 2;
            while (!head_issue_valid) @(posedge clk_core);
            if (head_issue_context_id != 0 ||
                head_issue_tag != GROUP_TAG + 32'(tile - 4) ||
                head_issue_head_id != head_id ||
                head_issue_head_index != 2'(head_id) ||
                head_issue_input_channel_base != 10'(32 * head_id) ||
                head_issue_output_tile != 8'(tile) ||
                head_issue_last_head != head_id ||
                head_issue_last_output_tile != (tile == 6))
                $fatal(1, "scheduler head issue contract mismatch");
            active_head_q = head_id;
            active_execution_tag_q = head_issue_tag;
            @(negedge clk_core);
            head_issue_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            head_issue_ready = 1'b0;

            lookup_and_consume(head_id, terms);

            @(negedge clk_core);
            session_context_id = 1'b0;
            session_head_id = head_id;
            session_payload_tag = GROUP_TAG;
            session_execution_tag = active_execution_tag_q;
            session_cache_owned = 1'b1;
            session_last_output_tile = tile == 6;
            session_valid = 1'b1;
            do @(posedge clk_core); while (!session_ready);
            @(negedge clk_core);
            session_valid = 1'b0;

            if (head_id) begin
                send_backend_done();
                repeat (2) @(posedge clk_core);
                send_decoder_done();
            end else begin
                send_decoder_done();
                repeat (2) @(posedge clk_core);
                send_backend_done();
            end
            while (!session_done_valid) @(posedge clk_core);
            @(posedge clk_core);
            @(negedge clk_core);
            if (head_done_valid)
                $fatal(1, "head completion did not retire");
        end
    endtask

    task automatic accept_tile(input logic [7:0] tile);
        begin
            while (!tile_start_valid) @(posedge clk_core);
            if (tile_start_tag != GROUP_TAG + 32'(tile) - 32'd4 ||
                tile_start_output_tile != tile ||
                tile_start_head_count != 2)
                $fatal(1, "tile start mismatch");
            @(negedge clk_core);
            tile_start_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            tile_start_ready = 1'b0;
        end
    endtask

    task automatic finish_tile;
        begin
            while (!tile_done_ready) @(posedge clk_core);
            @(negedge clk_core);
            tile_done_tag = active_execution_tag_q;
            tile_done_error = 1'b0;
            tile_done_valid = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            tile_done_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_context_id = 1'b0;
        group_tag = GROUP_TAG;
        group_head_count = 2;
        group_first_output_tile = 4;
        group_output_tile_count = 3;
        tile_start_ready = 1'b0;
        head_issue_ready = 1'b0;
        tile_done_valid = 1'b0;
        tile_done_tag = GROUP_TAG;
        tile_done_error = 1'b0;
        group_done_ready = 1'b0;
        fill_begin_valid = 1'b0;
        fill_context_id = 1'b0;
        fill_head_id = '0;
        fill_tag = GROUP_TAG;
        fill_term_count = '0;
        fill_entry_valid = 1'b0;
        fill_gate_code = '0;
        fill_lane_id = '0;
        fill_destination_count = '0;
        fill_entry_last = 1'b0;
        lookup_valid = 1'b0;
        lookup_context_id = 1'b0;
        lookup_head_id = '0;
        lookup_expected_tag = GROUP_TAG;
        lookup_meta_ready = 1'b0;
        lookup_entry_ready = 1'b0;
        session_valid = 1'b0;
        session_context_id = 1'b0;
        session_head_id = '0;
        session_payload_tag = GROUP_TAG;
        session_execution_tag = GROUP_TAG;
        session_cache_owned = 1'b1;
        session_last_output_tile = 1'b0;
        decoder_done_valid = 1'b0;
        decoder_done_payload_tag = GROUP_TAG;
        decoder_done_error = 1'b0;
        backend_done_valid = 1'b0;
        backend_done_execution_tag = GROUP_TAG;
        backend_done_error = 1'b0;
        active_head_q = '0;
        active_execution_tag_q = GROUP_TAG;
        descriptor_entries_read = 0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        fill_head(1'b0, 2);
        fill_head(1'b1, 3);
        if (cache_valid_flat != 2'b11 || !fill_begin_cacheable)
            $fatal(1, "initial descriptor residency fill failed");

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;

        for (int tile = 4; tile < 7; tile = tile + 1) begin
            accept_tile(8'(tile));
            service_head(tile, 1'b0);
            if (tile != 6 && cache_valid_flat != 2'b11)
                $fatal(1, "head0 released before final output tile");
            if (tile == 6 && cache_valid_flat != 2'b10)
                $fatal(1, "head0 final release boundary mismatch");
            service_head(tile, 1'b1);
            if (tile != 6 && cache_valid_flat != 2'b11)
                $fatal(1, "head1 released before final output tile");
            if (tile == 6 && cache_valid_flat != 2'b00)
                $fatal(1, "head1 final release boundary mismatch");
            finish_tile();
        end

        while (!group_done_valid) @(posedge clk_core);
        if (group_done_tag != GROUP_TAG || group_done_error)
            $fatal(1, "group completion mismatch");
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;

        if (scheduler_protocol_error || cache_protocol_error ||
            lifecycle_protocol_error || count_groups != 1 ||
            count_tile_starts != 3 || count_head_issues != 6 ||
            count_group_errors != 0 || count_cached_heads != 2 ||
            count_bypass_heads != 0 || count_lookup_hits != 6 ||
            count_lookup_misses != 0 || count_releases != 2 ||
            count_release_noops != 0 || count_release_tag_mismatches != 0 ||
            count_sessions != 6 || count_final_tile_releases != 2 ||
            count_cache_releases != 2 || count_session_errors != 0 ||
            slot_release_count != 2 || descriptor_entries_read != 15)
            $fatal(1, "residency integration counters mismatch");
        $display("PASS: residency across output tiles hits=%0d reads=%0d releases=%0d sessions=%0d",
                 count_lookup_hits, descriptor_entries_read, count_releases,
                 count_sessions);
        $finish;
    end

    initial begin
        repeat (30000) @(posedge clk_core);
        $fatal(1, "output-tile residency integration timeout");
    end
endmodule

`default_nettype wire
