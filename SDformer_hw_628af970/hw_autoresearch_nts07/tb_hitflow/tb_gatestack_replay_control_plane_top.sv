`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_replay_control_plane_top;
    logic clk_core, rst_core;
    logic head_request_valid, head_request_ready;
    logic head_request_context_id;
    logic [4:0] head_request_head_id;
    logic [31:0] head_request_execution_tag;
    logic [5:0] head_request_head_index;
    logic [9:0] head_request_input_channel_base;
    logic [7:0] head_request_output_tile;
    logic head_request_last_head, head_request_last_output_tile;
    logic slot_inspect_valid, slot_inspect_ready, slot_inspect_context_id;
    logic [4:0] slot_inspect_head_id;
    logic slot_meta_valid, slot_meta_ready, slot_meta_exists;
    logic [31:0] slot_meta_tag;
    logic slot_meta_mode_is_csr;
    logic [1:0] slot_meta_format;
    logic [15:0] slot_meta_payload_bits, slot_meta_word_count;
    logic cache_lookup_valid, cache_lookup_ready, cache_lookup_context_id;
    logic [4:0] cache_lookup_head_id;
    logic [31:0] cache_lookup_expected_tag;
    logic cache_meta_valid, cache_meta_ready, cache_meta_hit;
    logic [31:0] cache_meta_tag;
    logic [7:0] cache_meta_term_count;
    logic projection_commit_pulse, projection_reserve_ready;
    logic projection_context_id;
    logic [4:0] projection_head_id;
    logic [31:0] projection_payload_tag, projection_execution_tag;
    logic [1:0] projection_route;
    logic [1:0] projection_format;
    logic [5:0] projection_head_index;
    logic [9:0] projection_input_channel_base;
    logic [7:0] projection_output_tile;
    logic projection_last_head;
    logic [7:0] projection_resident_term_count;
    logic [12:0] projection_resident_event_count;
    logic slot_commit_pulse, slot_reserve_ready, slot_replay_context_id;
    logic [4:0] slot_replay_head_id;
    logic [31:0] slot_replay_payload_tag;
    logic [6:0] slot_replay_start_word;
    logic decoder_done_valid, decoder_done_ready;
    logic [31:0] decoder_done_payload_tag;
    logic decoder_done_error;
    logic backend_done_valid, backend_done_ready;
    logic [31:0] backend_done_execution_tag;
    logic backend_done_error;
    logic slot_release_valid, slot_release_ready, slot_release_context_id;
    logic [4:0] slot_release_head_id;
    logic cache_release_valid, cache_release_ready;
    logic cache_release_context_id;
    logic [4:0] cache_release_head_id;
    logic [31:0] cache_release_payload_tag;
    logic head_complete_valid, head_complete_ready;
    logic head_complete_context_id;
    logic [4:0] head_complete_head_id;
    logic [5:0] head_complete_head_index;
    logic head_complete_last_head;
    logic [31:0] head_complete_payload_tag;
    logic [31:0] head_complete_execution_tag;
    logic head_complete_error, protocol_error;
    logic [31:0] count_requests, count_commits, count_rejects;
    logic [31:0] count_sessions;
    int projection_commits, slot_commits;

    gatestack_replay_control_plane_top dut (.*);

    always #5 clk_core <= ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            projection_commits <= 0;
            slot_commits <= 0;
        end else begin
            if (projection_commit_pulse)
                projection_commits <= projection_commits + 1;
            if (slot_commit_pulse)
                slot_commits <= slot_commits + 1;
        end
    end

    task automatic send_request(
        input int id,
        input logic [31:0] execution_tag,
        input logic last_head,
        input logic last_tile
    );
        begin
            @(negedge clk_core);
            head_request_context_id = id[0];
            head_request_head_id = 5'(id);
            head_request_execution_tag = execution_tag;
            head_request_head_index = 6'(id);
            head_request_input_channel_base = 10'(id * 32);
            head_request_output_tile = 8'(id + 1);
            head_request_last_head = last_head;
            head_request_last_output_tile = last_tile;
            head_request_valid = 1'b1;
            do @(posedge clk_core); while (!head_request_ready);
            @(negedge clk_core);
            head_request_valid = 1'b0;
        end
    endtask

    task automatic send_slot_meta(
        input logic exists,
        input logic csr,
        input logic [31:0] payload_tag,
        input logic [15:0] payload_bits,
        input logic [15:0] words
    );
        begin
            while (!slot_inspect_valid) @(posedge clk_core);
            if (slot_inspect_context_id != head_request_context_id ||
                slot_inspect_head_id != head_request_head_id)
                $fatal(1, "slot inspect identity mismatch");
            @(negedge clk_core);
            slot_inspect_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            slot_inspect_ready = 1'b0;
            slot_meta_exists = exists;
            slot_meta_mode_is_csr = csr;
            slot_meta_format = csr ? 2'd1 : 2'd0;
            slot_meta_tag = payload_tag;
            slot_meta_payload_bits = payload_bits;
            slot_meta_word_count = words;
            slot_meta_valid = 1'b1;
            do @(posedge clk_core); while (!slot_meta_ready);
            @(negedge clk_core);
            slot_meta_valid = 1'b0;
        end
    endtask

    task automatic send_cache_meta(
        input logic hit,
        input logic [31:0] payload_tag,
        input logic [7:0] terms
    );
        begin
            while (!cache_lookup_valid) @(posedge clk_core);
            if (cache_lookup_context_id != head_request_context_id ||
                cache_lookup_head_id != head_request_head_id ||
                cache_lookup_expected_tag != slot_meta_tag)
                $fatal(1, "cache lookup identity mismatch");
            @(negedge clk_core);
            cache_lookup_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            cache_lookup_ready = 1'b0;
            cache_meta_hit = hit;
            cache_meta_tag = payload_tag;
            cache_meta_term_count = terms;
            cache_meta_valid = 1'b1;
            do @(posedge clk_core); while (!cache_meta_ready);
            @(negedge clk_core);
            cache_meta_valid = 1'b0;
        end
    endtask

    task automatic check_commit(
        input int id,
        input logic [31:0] payload_tag,
        input logic [31:0] execution_tag,
        input logic [1:0] route,
        input logic last_head,
        input logic slot_required,
        input logic [7:0] terms,
        input logic [12:0] events,
        input logic [6:0] start_word
    );
        begin
            if (!projection_commit_pulse ||
                projection_context_id != id[0] ||
                projection_head_id != 5'(id) ||
                projection_payload_tag != payload_tag ||
                projection_execution_tag != execution_tag ||
                projection_route != route ||
                projection_format != (route == 2'd2 ? 2'd0 : 2'd1) ||
                projection_head_index != 6'(id) ||
                projection_input_channel_base != 10'(id * 32) ||
                projection_output_tile != 8'(id + 1) ||
                projection_last_head != last_head ||
                projection_resident_term_count != terms ||
                projection_resident_event_count != events ||
                slot_commit_pulse != slot_required)
                $fatal(1, "projection commit mismatch id=%0d", id);
            if (slot_required &&
                (slot_replay_context_id != id[0] ||
                 slot_replay_head_id != 5'(id) ||
                 slot_replay_payload_tag != payload_tag ||
                 slot_replay_start_word != start_word))
                $fatal(1, "slot commit mismatch id=%0d", id);
        end
    endtask

    task automatic wait_and_check_commit(
        input int id,
        input logic [31:0] payload_tag,
        input logic [31:0] execution_tag,
        input logic [1:0] route,
        input logic last_head,
        input logic slot_required,
        input logic [7:0] terms,
        input logic [12:0] events,
        input logic [6:0] start_word
    );
        begin
            while (!projection_commit_pulse) @(posedge clk_core);
            check_commit(id, payload_tag, execution_tag, route, last_head,
                         slot_required, terms, events, start_word);
            @(posedge clk_core);
        end
    endtask

    task automatic send_decoder_done(input logic [31:0] payload_tag);
        begin
            @(negedge clk_core);
            decoder_done_payload_tag = payload_tag;
            decoder_done_error = 1'b0;
            decoder_done_valid = 1'b1;
            do @(posedge clk_core); while (!decoder_done_ready);
            @(negedge clk_core);
            decoder_done_valid = 1'b0;
        end
    endtask

    task automatic send_backend_done(input logic [31:0] execution_tag);
        begin
            @(negedge clk_core);
            backend_done_execution_tag = execution_tag;
            backend_done_error = 1'b0;
            backend_done_valid = 1'b1;
            do @(posedge clk_core); while (!backend_done_ready);
            @(negedge clk_core);
            backend_done_valid = 1'b0;
        end
    endtask

    task automatic accept_completion(
        input int id,
        input logic [31:0] payload_tag,
        input logic [31:0] execution_tag,
        input logic last_head,
        input logic error_expected
    );
        begin
            while (!head_complete_valid) @(posedge clk_core);
            if (head_complete_context_id != id[0] ||
                head_complete_head_id != 5'(id) ||
                head_complete_head_index != 6'(id) ||
                head_complete_last_head != last_head ||
                head_complete_payload_tag != payload_tag ||
                head_complete_execution_tag != execution_tag ||
                head_complete_error != error_expected)
                $fatal(1, "head completion mismatch id=%0d", id);
            repeat (2) begin
                @(posedge clk_core);
                if (!head_complete_valid)
                    $fatal(1, "completion dropped under backpressure");
            end
            @(negedge clk_core);
            head_complete_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            head_complete_ready = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        head_request_valid = 1'b0;
        head_request_context_id = 1'b0;
        head_request_head_id = '0;
        head_request_execution_tag = '0;
        head_request_head_index = '0;
        head_request_input_channel_base = '0;
        head_request_output_tile = '0;
        head_request_last_head = 1'b0;
        head_request_last_output_tile = 1'b0;
        slot_inspect_ready = 1'b0;
        slot_meta_valid = 1'b0;
        slot_meta_exists = 1'b0;
        slot_meta_tag = '0;
        slot_meta_mode_is_csr = 1'b0;
        slot_meta_format = 2'd0;
        slot_meta_payload_bits = '0;
        slot_meta_word_count = '0;
        cache_lookup_ready = 1'b0;
        cache_meta_valid = 1'b0;
        cache_meta_hit = 1'b0;
        cache_meta_tag = '0;
        cache_meta_term_count = '0;
        projection_reserve_ready = 1'b0;
        slot_reserve_ready = 1'b0;
        decoder_done_valid = 1'b0;
        decoder_done_payload_tag = '0;
        decoder_done_error = 1'b0;
        backend_done_valid = 1'b0;
        backend_done_execution_tag = '0;
        backend_done_error = 1'b0;
        slot_release_ready = 1'b0;
        cache_release_ready = 1'b0;
        head_complete_ready = 1'b0;
        repeat (5) @(posedge clk_core);
        rst_core = 1'b0;

        // Resident nonfinal: PLAN has no side effects and COMMIT is atomic.
        send_request(2, 32'he700_0010, 1'b0, 1'b0);
        send_slot_meta(1'b1, 1'b1, 32'hd700_0002, 16'd296, 16'd5);
        send_cache_meta(1'b1, 32'hd700_0002, 8'd3);
        repeat (2) @(posedge clk_core);
        if (projection_commit_pulse || slot_commit_pulse ||
            projection_commits != 0 || slot_commits != 0)
            $fatal(1, "PLAN caused a side effect");
        @(negedge clk_core);
        slot_reserve_ready = 1'b1;
        repeat (2) @(posedge clk_core);
        if (projection_commit_pulse || slot_commit_pulse)
            $fatal(1, "partial commit with projection blocked");
        @(negedge clk_core);
        projection_reserve_ready = 1'b1;
        wait_and_check_commit(2, 32'hd700_0002, 32'he700_0010,
                              2'd0, 1'b0, 1'b1, 8'd3, 13'd5, 7'd4);

        // Builder is idle now, but the single-context owner must block overwrite.
        @(negedge clk_core);
        head_request_context_id = 1'b1;
        head_request_head_id = 5'd3;
        head_request_execution_tag = 32'he700_0011;
        head_request_head_index = 6'd3;
        head_request_input_channel_base = 10'd96;
        head_request_output_tile = 8'd4;
        head_request_last_head = 1'b0;
        head_request_last_output_tile = 1'b0;
        head_request_valid = 1'b1;
        repeat (3) begin
            @(posedge clk_core);
            if (head_request_ready)
                $fatal(1, "second request overwrote active lifecycle");
        end
        @(negedge clk_core);
        head_request_valid = 1'b0;

        send_backend_done(32'he700_0010);
        send_decoder_done(32'hd700_0002);
        repeat (2) @(posedge clk_core);
        if (slot_release_valid || cache_release_valid)
            $fatal(1, "nonfinal tile released persistent state");
        accept_completion(2, 32'hd700_0002, 32'he700_0010,
                          1'b0, 1'b0);

        // Resident final tile: both release channels must drain before retire.
        send_request(3, 32'he700_0011, 1'b1, 1'b1);
        send_slot_meta(1'b1, 1'b1, 32'hd700_0003, 16'd296, 16'd5);
        send_cache_meta(1'b1, 32'hd700_0003, 8'd3);
        wait_and_check_commit(3, 32'hd700_0003, 32'he700_0011,
                              2'd0, 1'b1, 1'b1, 8'd3, 13'd5, 7'd4);
        send_decoder_done(32'hd700_0003);
        send_backend_done(32'he700_0011);
        while (!slot_release_valid || !cache_release_valid)
            @(posedge clk_core);
        if (slot_release_context_id != 1'b1 ||
            slot_release_head_id != 5'd3 ||
            cache_release_context_id != 1'b1 ||
            cache_release_head_id != 5'd3 || head_complete_valid)
            $fatal(1, "final release metadata/order mismatch");
        @(negedge clk_core);
        cache_release_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        cache_release_ready = 1'b0;
        repeat (2) @(posedge clk_core);
        if (!slot_release_valid || cache_release_valid || head_complete_valid)
            $fatal(1, "independent release backpressure failed");
        @(negedge clk_core);
        slot_release_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        slot_release_ready = 1'b0;
        accept_completion(3, 32'hd700_0003, 32'he700_0011,
                          1'b1, 1'b0);

        // IPD cache miss remains a slot-backed exact replay session.
        send_request(4, 32'he700_0012, 1'b0, 1'b0);
        send_slot_meta(1'b1, 1'b1, 32'hd700_0004, 16'd6640, 16'd104);
        send_cache_meta(1'b0, 32'h0, 8'd0);
        wait_and_check_commit(4, 32'hd700_0004, 32'he700_0012,
                              2'd1, 1'b0, 1'b1, 8'd0, 13'd0, 7'd0);
        send_decoder_done(32'hd700_0004);
        send_backend_done(32'he700_0012);
        accept_completion(4, 32'hd700_0004, 32'he700_0012,
                          1'b0, 1'b0);

        // Missing slot rejects without projection/slot/lifecycle acquisition.
        send_request(5, 32'he700_0013, 1'b0, 1'b0);
        send_slot_meta(1'b0, 1'b0, 32'h0, 16'd0, 16'd0);
        accept_completion(5, 32'h0, 32'he700_0013, 1'b0, 1'b1);

        repeat (3) @(posedge clk_core);
        if (!protocol_error || count_requests != 4 || count_commits != 3 ||
            count_sessions != 3 || count_rejects != 1 ||
            projection_commits != 3 || slot_commits != 3)
            $fatal(1, "control-plane counters mismatch");
        $display("PASS: control requests=%0d commits=%0d sessions=%0d rejects=%0d",
                 count_requests, count_commits, count_sessions, count_rejects);
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(1, "replay control-plane timeout");
    end
endmodule

`default_nettype wire
