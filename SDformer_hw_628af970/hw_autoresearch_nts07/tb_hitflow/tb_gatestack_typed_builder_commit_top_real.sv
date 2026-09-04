`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_typed_builder_commit_top_real;
    localparam logic [1:0] FORMAT_RAW = 2'd0;
    localparam logic [1:0] FORMAT_IPD32W = 2'd1;
    localparam logic [1:0] FORMAT_FADC24 = 2'd2;

    logic clk_core, rst_core;
    logic begin_valid, begin_ready, begin_context_id;
    logic [4:0] begin_head_id;
    logic [31:0] begin_tag;
    logic [1:0] begin_format;
    logic [15:0] begin_expected_payload_bits;
    logic [3:0] begin_active_classes;
    logic [7:0] begin_active_tokens, begin_term_count;
    logic [12:0] begin_event_count;
    logic [7:0] begin_bitmap_term_count;
    logic [12:0] begin_fadc_destination_bytes;
    logic descriptor_valid, descriptor_ready;
    logic [8:0] descriptor_gate_code;
    logic [4:0] descriptor_lane_id;
    logic [7:0] descriptor_destination_count;
    logic descriptor_last;
    logic destination_valid, destination_ready;
    logic [7:0] destination_token_id;
    logic destination_last_for_term;
    logic destination_bitmap_valid, destination_bitmap_ready;
    logic [161:0] destination_bitmap;
    logic raw_token_valid, raw_token_ready;
    logic [7:0] raw_token_id;
    logic [8:0] raw_gate_code;
    logic [31:0] raw_k_bits;
    logic builder_done_valid, builder_done_ready;
    logic [31:0] builder_done_tag;
    logic [1:0] builder_done_format;
    logic builder_done_error;
    logic [7:0] builder_done_word_count;
    logic inspect_valid, inspect_ready, inspect_context_id;
    logic [4:0] inspect_head_id;
    logic inspect_meta_valid, inspect_meta_ready, inspect_exists;
    logic [31:0] inspect_tag;
    logic inspect_mode_is_csr;
    logic [1:0] inspect_format;
    logic [15:0] inspect_payload_bits, inspect_word_count;
    logic replay_begin_valid, replay_begin_ready, replay_context_id;
    logic [4:0] replay_head_id;
    logic [6:0] replay_start_word;
    logic replay_word_valid, replay_word_ready;
    logic [63:0] replay_word_data;
    logic [6:0] replay_word_index;
    logic replay_word_last;
    logic [31:0] replay_tag;
    logic replay_mode_is_csr;
    logic [1:0] replay_format;
    logic [15:0] replay_payload_bits;
    logic release_valid, release_ready, release_context_id;
    logic [4:0] release_head_id;
    logic [47:0] slot_valid_flat;
    logic commit_session_active, replay_session_active;
    logic serializer_protocol_error, slot_protocol_error;
    logic [31:0] count_builder_heads;
    logic [31:0] count_builder_committed_heads;
    logic [31:0] count_builder_aborted_heads;
    logic [31:0] count_builder_committed_words;
    logic [31:0] count_builder_input_stall_cycles;
    logic [31:0] count_builder_output_stall_cycles;
    logic [31:0] count_slot_commit_heads;
    logic [31:0] count_slot_replay_heads;
    logic [31:0] count_slot_release_heads;
    logic [31:0] count_slot_invalid_headers;
    logic [31:0] count_slot_commit_stall_cycles;
    logic [31:0] count_slot_replay_stall_cycles;

    logic [23:0] descriptor_mem [0:127];
    logic [7:0] destination_mem [0:8191];
    logic [40:0] raw_record_mem [0:161];
    logic [63:0] expected_word_mem [0:103];
    logic [31:0] prng_q;

    gatestack_typed_builder_commit_top dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic build_replay_release(
        input logic [1:0] format_value,
        input logic [31:0] tag_value,
        input logic [4:0] head_value,
        input logic [3:0] active_classes,
        input logic [7:0] active_tokens,
        input int terms,
        input logic [12:0] events,
        input logic [7:0] bitmap_terms,
        input logic [12:0] fadc_destination_bytes,
        input int payload_bits,
        input int word_count
    );
        int destination_index;
        int count;
        begin
            @(negedge clk_core);
            begin_valid = 1'b1;
            begin_context_id = 1'b0;
            begin_head_id = head_value;
            begin_tag = tag_value;
            begin_format = format_value;
            begin_expected_payload_bits = 16'(payload_bits);
            begin_active_classes = active_classes;
            begin_active_tokens = active_tokens;
            begin_term_count = 8'(terms);
            begin_event_count = 13'(events);
            begin_bitmap_term_count = bitmap_terms;
            begin_fadc_destination_bytes = fadc_destination_bytes;
            do @(posedge clk_core); while (!begin_ready);
            @(negedge clk_core);
            begin_valid = 1'b0;

            if (format_value == FORMAT_RAW) begin
                for (int token = 0; token < 162; token = token + 1) begin
                    @(negedge clk_core);
                    raw_token_valid = 1'b1;
                    raw_token_id = 8'(token);
                    raw_k_bits = raw_record_mem[token][31:0];
                    raw_gate_code = raw_record_mem[token][40:32];
                    do @(posedge clk_core); while (!raw_token_ready);
                    @(negedge clk_core);
                    raw_token_valid = 1'b0;
                end
            end else if (terms != 0) begin
                for (int term = 0; term < terms; term = term + 1) begin
                    @(negedge clk_core);
                    descriptor_valid = 1'b1;
                    descriptor_gate_code = descriptor_mem[term][8:0];
                    descriptor_lane_id = descriptor_mem[term][13:9];
                    descriptor_destination_count = descriptor_mem[term][21:14];
                    descriptor_last = term == terms - 1;
                    do @(posedge clk_core); while (!descriptor_ready);
                    @(negedge clk_core);
                    descriptor_valid = 1'b0;
                end
                destination_index = 0;
                for (int term = 0; term < terms; term = term + 1) begin
                    count = 32'(descriptor_mem[term][21:14]);
                    for (int item = 0; item < count; item = item + 1) begin
                        @(negedge clk_core);
                        destination_valid = 1'b1;
                        destination_token_id = destination_mem[destination_index];
                        destination_last_for_term = item == count - 1;
                        do @(posedge clk_core); while (!destination_ready);
                        @(negedge clk_core);
                        destination_valid = 1'b0;
                        destination_index = destination_index + 1;
                    end
                end
            end

            while (!builder_done_valid) @(posedge clk_core);
            if (builder_done_tag != tag_value ||
                builder_done_format != format_value || builder_done_error ||
                32'(builder_done_word_count) != word_count)
                $fatal(1, "builder done mismatch");
            @(negedge clk_core);
            builder_done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            builder_done_ready = 1'b0;

            inspect_valid = 1'b1;
            inspect_context_id = 1'b0;
            inspect_head_id = head_value;
            do @(posedge clk_core); while (!inspect_ready);
            @(negedge clk_core);
            inspect_valid = 1'b0;
            repeat (2) @(posedge clk_core);
            if (!inspect_meta_valid || !inspect_exists ||
                inspect_tag != tag_value || inspect_format != format_value ||
                inspect_mode_is_csr != (format_value != FORMAT_RAW) ||
                32'(inspect_payload_bits) != payload_bits ||
                32'(inspect_word_count) != word_count)
                $fatal(1, "slot inspect mismatch");
            @(negedge clk_core);
            inspect_meta_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            inspect_meta_ready = 1'b0;

            replay_begin_valid = 1'b1;
            replay_context_id = 1'b0;
            replay_head_id = head_value;
            replay_start_word = '0;
            do @(posedge clk_core); while (!replay_begin_ready);
            @(negedge clk_core);
            replay_begin_valid = 1'b0;
            for (int word = 0; word < word_count; word = word + 1) begin
                while (!replay_word_valid) @(posedge clk_core);
                advance_prng();
                repeat (32'(prng_q[1:0])) @(posedge clk_core);
                if (replay_word_data != expected_word_mem[word] ||
                    replay_word_index != 7'(word) ||
                    replay_word_last != (word == word_count - 1) ||
                    replay_tag != tag_value || replay_format != format_value ||
                    replay_mode_is_csr != (format_value != FORMAT_RAW) ||
                    32'(replay_payload_bits) != payload_bits)
                    $fatal(1, "slot replay mismatch word=%0d", word);
                @(negedge clk_core);
                replay_word_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                replay_word_ready = 1'b0;
            end

            release_valid = 1'b1;
            release_context_id = 1'b0;
            release_head_id = head_value;
            do @(posedge clk_core); while (!release_ready);
            @(negedge clk_core);
            release_valid = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        begin_valid = 1'b0;
        descriptor_valid = 1'b0;
        destination_valid = 1'b0;
        destination_bitmap_valid = 1'b0;
        raw_token_valid = 1'b0;
        builder_done_ready = 1'b0;
        inspect_valid = 1'b0;
        inspect_meta_ready = 1'b0;
        replay_begin_valid = 1'b0;
        replay_word_ready = 1'b0;
        release_valid = 1'b0;
        begin_context_id = '0;
        begin_head_id = '0;
        begin_tag = '0;
        begin_format = '0;
        begin_expected_payload_bits = '0;
        begin_active_classes = '0;
        begin_active_tokens = '0;
        begin_term_count = '0;
        begin_event_count = '0;
        begin_bitmap_term_count = '0;
        begin_fadc_destination_bytes = '0;
        descriptor_gate_code = '0;
        descriptor_lane_id = '0;
        descriptor_destination_count = '0;
        descriptor_last = 1'b0;
        destination_token_id = '0;
        destination_last_for_term = 1'b0;
        destination_bitmap = '0;
        raw_token_id = '0;
        raw_gate_code = '0;
        raw_k_bits = '0;
        inspect_context_id = '0;
        inspect_head_id = '0;
        replay_context_id = '0;
        replay_head_id = '0;
        replay_start_word = '0;
        release_context_id = '0;
        release_head_id = '0;
        prng_q = 32'ha70c_2027;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/descriptors.memh", descriptor_mem, 0, 31);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/destinations.memh", destination_mem, 0, 126);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/expected_words.memh", expected_word_mem, 0, 33);
        build_replay_release(FORMAT_IPD32W, 32'h6900_0000, 5'd0, 4'd2, 8'd56, 32, 127, 8'd1, 13'd124, 2168, 34);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/descriptors.memh", descriptor_mem, 0, 60);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/destinations.memh", destination_mem, 0, 813);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/expected_words.memh", expected_word_mem, 0, 101);
        build_replay_release(FORMAT_FADC24, 32'h6903_0004, 5'd4, 4'd3, 8'd153, 61, 814, 8'd15, 13'd616, 6520, 102);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_s0_h0/expected_words.memh", expected_word_mem, 0, 103);
        build_replay_release(FORMAT_RAW, 32'h69f0_0000, 5'd5, 4'd2, 8'd56, 32, 127, 8'd1, 13'd124, 6642, 104);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_empty_s1_h0/expected_words.memh", expected_word_mem, 0, 1);
        build_replay_release(FORMAT_IPD32W, 32'h6901_0000, 5'd6, 4'd0, 8'd0, 0, 0, 8'd0, 13'd0, 128, 2);

        repeat (3) @(posedge clk_core);
        if (slot_valid_flat != '0 || commit_session_active ||
            replay_session_active || serializer_protocol_error ||
            slot_protocol_error || destination_bitmap_ready ||
            count_builder_heads != 4 ||
            count_builder_committed_heads != 4 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 242 ||
            count_builder_input_stall_cycles != 0 ||
            count_builder_output_stall_cycles != 0 ||
            count_slot_commit_heads != 4 || count_slot_replay_heads != 4 ||
            count_slot_release_heads != 4 || count_slot_invalid_headers != 0 ||
            count_slot_commit_stall_cycles != 0)
            $fatal(1, "builder-slot integration counters/state mismatch");
        $display(
            "PASS: typed builder atomic slot integration heads=%0d words=%0d replays=%0d releases=%0d input_stalls=%0d output_stalls=%0d commit_stalls=%0d replay_stalls=%0d",
            count_builder_committed_heads, count_builder_committed_words,
            count_slot_replay_heads, count_slot_release_heads,
            count_builder_input_stall_cycles,
            count_builder_output_stall_cycles,
            count_slot_commit_stall_cycles,
            count_slot_replay_stall_cycles);
        $finish;
    end

endmodule

`default_nettype wire
