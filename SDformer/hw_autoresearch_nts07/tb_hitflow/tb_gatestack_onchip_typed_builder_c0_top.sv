`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_onchip_typed_builder_c0_top;
    localparam logic [1:0] FORMAT_RAW = 2'd0;
    localparam logic [1:0] FORMAT_IPD32W = 2'd1;
    localparam logic [1:0] FORMAT_FADC24 = 2'd2;

    logic clk_core, rst_core;
    logic head_begin_valid, head_begin_ready, head_context_id;
    logic [4:0] head_id;
    logic [31:0] head_tag;
    logic token_valid, token_ready;
    logic [7:0] token_id;
    logic [8:0] token_gate_code;
    logic [31:0] token_k_bits;
    logic token_last;
    logic done_valid, done_ready;
    logic [31:0] done_tag;
    logic [1:0] done_format;
    logic done_error;
    logic [7:0] done_word_count;
    logic [2:0] selected_reason;
    logic [15:0] selected_payload_bits;
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
    logic workspace_protocol_error, serializer_protocol_error, slot_protocol_error;
    logic [31:0] count_workspace_heads;
    logic [31:0] count_workspace_raw_fallback_heads;
    logic [31:0] count_workspace_terms, count_workspace_destinations;
    logic [31:0] count_workspace_scan_cycles;
    logic [31:0] count_workspace_output_stall_cycles;
    logic [31:0] count_builder_committed_heads;
    logic [31:0] count_builder_aborted_heads;
    logic [31:0] count_builder_committed_words;
    logic [31:0] count_slot_commit_heads;
    logic [31:0] count_slot_replay_heads;
    logic [31:0] count_slot_release_heads;

    logic [40:0] raw_record_mem [0:161];
    logic [63:0] expected_word_mem [0:103];
    logic [31:0] prng_q;

    gatestack_onchip_typed_builder_c0_top dut (.*);
    always #5 clk_core <= ~clk_core;

    task automatic advance_prng;
        begin
            prng_q = {prng_q[30:0],
                prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
        end
    endtask

    task automatic build_head(
        input logic [31:0] tag_value,
        input logic [4:0] head_value,
        input logic [1:0] format_value,
        input logic [2:0] reason_value,
        input int payload_bits,
        input int word_count
    );
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = head_value;
            head_tag = tag_value;
            do @(posedge clk_core); while (!head_begin_ready);
            @(negedge clk_core);
            head_begin_valid = 1'b0;

            for (int token = 0; token < 162; token = token + 1) begin
                advance_prng();
                repeat (32'(prng_q[1:0])) @(posedge clk_core);
                @(negedge clk_core);
                token_valid = 1'b1;
                token_id = 8'(token);
                token_k_bits = raw_record_mem[token][31:0];
                token_gate_code = raw_record_mem[token][40:32];
                token_last = token == 161;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
                token_valid = 1'b0;
            end

            while (!done_valid) @(posedge clk_core);
            if (done_tag != tag_value || done_format != format_value ||
                done_error || 32'(done_word_count) != word_count ||
                selected_reason != reason_value ||
                32'(selected_payload_bits) != payload_bits)
                $fatal(1, "onchip builder done mismatch tag=%h", tag_value);
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;
        end
    endtask

    task automatic inspect_replay_release(
        input logic [31:0] tag_value,
        input logic [4:0] head_value,
        input logic [1:0] format_value,
        input int payload_bits,
        input int word_count
    );
        begin
            @(negedge clk_core);
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
                $fatal(1, "onchip slot inspect mismatch");
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
                    $fatal(1, "onchip slot replay mismatch word=%0d", word);
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
        head_begin_valid = 1'b0;
        token_valid = 1'b0;
        done_ready = 1'b0;
        inspect_valid = 1'b0;
        inspect_meta_ready = 1'b0;
        replay_begin_valid = 1'b0;
        replay_word_ready = 1'b0;
        release_valid = 1'b0;
        head_context_id = '0;
        head_id = '0;
        head_tag = '0;
        token_id = '0;
        token_gate_code = '0;
        token_k_bits = '0;
        token_last = 1'b0;
        inspect_context_id = '0;
        inspect_head_id = '0;
        replay_context_id = '0;
        replay_head_id = '0;
        replay_start_word = '0;
        release_context_id = '0;
        release_head_id = '0;
        prng_q = 32'h0c00_2027;
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/expected_words.memh", expected_word_mem, 0, 33);
        build_head(32'h6900_0000, 5'd0, FORMAT_IPD32W, 3'd0, 2168, 34);
        inspect_replay_release(32'h6900_0000, 5'd0, FORMAT_IPD32W, 2168, 34);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/expected_words.memh", expected_word_mem, 0, 101);
        build_head(32'h6903_0004, 5'd4, FORMAT_FADC24, 3'd2, 6520, 102);
        inspect_replay_release(32'h6903_0004, 5'd4, FORMAT_FADC24, 6520, 102);

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_class_overflow_synth/raw_records.memh", raw_record_mem);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/raw_class_overflow_synth/expected_words.memh", expected_word_mem, 0, 103);
        build_head(32'h69ff_0001, 5'd7, FORMAT_RAW, 3'd4, 6642, 104);
        inspect_replay_release(32'h69ff_0001, 5'd7, FORMAT_RAW, 6642, 104);

        repeat (3) @(posedge clk_core);
        if (slot_valid_flat != '0 || workspace_protocol_error ||
            serializer_protocol_error || slot_protocol_error ||
            count_workspace_heads != 3 ||
            count_workspace_raw_fallback_heads != 1 ||
            count_workspace_terms != 93 ||
            count_workspace_destinations != 941 ||
            count_workspace_scan_cycles != 443 ||
            count_workspace_output_stall_cycles == 0 ||
            count_builder_committed_heads != 3 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 240 ||
            count_slot_commit_heads != 3 || count_slot_replay_heads != 3 ||
            count_slot_release_heads != 3)
            $fatal(1,
                "onchip builder final counters/state mismatch heads=%0d raw=%0d terms=%0d destinations=%0d scans=%0d words=%0d commits=%0d replays=%0d releases=%0d",
                count_workspace_heads, count_workspace_raw_fallback_heads,
                count_workspace_terms, count_workspace_destinations,
                count_workspace_scan_cycles, count_builder_committed_words,
                count_slot_commit_heads, count_slot_replay_heads,
                count_slot_release_heads);
        $display(
            "PASS: full C0 onchip builder heads=%0d formats=IPD/FADC/RAW words=%0d terms=%0d destinations=%0d scan_cycles=%0d",
            count_workspace_heads, count_builder_committed_words,
            count_workspace_terms, count_workspace_destinations,
            count_workspace_scan_cycles);
        $finish;
    end

endmodule

`default_nettype wire
