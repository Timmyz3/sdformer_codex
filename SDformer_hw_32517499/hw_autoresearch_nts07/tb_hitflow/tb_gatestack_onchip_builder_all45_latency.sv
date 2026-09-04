`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_onchip_builder_all45_latency;
    localparam int HEAD_COUNT = 45;
    localparam int RAW_RECORD_COUNT = HEAD_COUNT * 162;
    localparam int EXPECTED_WORD_COUNT = 861;

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

    logic [40:0] raw_records [0:RAW_RECORD_COUNT-1];
    logic [63:0] expected_words [0:EXPECTED_WORD_COUNT-1];
    logic [31:0] tags [0:HEAD_COUNT-1];
    logic [1:0] stages [0:HEAD_COUNT-1];
    logic [4:0] heads [0:HEAD_COUNT-1];
    logic [1:0] formats [0:HEAD_COUNT-1];
    logic [2:0] reasons [0:HEAD_COUNT-1];
    logic [15:0] payload_bits_rows [0:HEAD_COUNT-1];
    logic [7:0] word_counts [0:HEAD_COUNT-1];
    logic [15:0] word_offsets [0:HEAD_COUNT-1];
    logic [7:0] term_counts [0:HEAD_COUNT-1];
    logic [12:0] event_counts [0:HEAD_COUNT-1];
    logic [31:0] cycle_q;

    gatestack_onchip_typed_builder_c0_top dut (.*);
    always #5 clk_core <= ~clk_core;
    always_ff @(posedge clk_core) begin
        if (rst_core)
            cycle_q <= '0;
        else
            cycle_q <= cycle_q + 1'b1;
    end

    task automatic run_head(input int head_index);
        int start_cycle;
        int latency;
        int raw_offset;
        int word_offset;
        int word_count;
        begin
            raw_offset = head_index * 162;
            word_offset = 32'(word_offsets[head_index]);
            word_count = 32'(word_counts[head_index]);
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = heads[head_index];
            head_tag = tags[head_index];
            start_cycle = cycle_q;
            do @(posedge clk_core); while (!head_begin_ready);
            @(negedge clk_core);
            head_begin_valid = 1'b0;

            for (int token = 0; token < 162; token = token + 1) begin
                token_valid = 1'b1;
                token_id = 8'(token);
                token_k_bits = raw_records[raw_offset + token][31:0];
                token_gate_code = raw_records[raw_offset + token][40:32];
                token_last = token == 161;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
            end
            token_valid = 1'b0;

            while (!done_valid) @(posedge clk_core);
            latency = cycle_q - start_cycle;
            if (done_tag != tags[head_index] ||
                done_format != formats[head_index] || done_error ||
                done_word_count != word_counts[head_index] ||
                selected_reason != reasons[head_index] ||
                selected_payload_bits != payload_bits_rows[head_index])
                $fatal(1, "all45 done mismatch index=%0d", head_index);
            $display(
                "LATENCY index=%0d stage=%0d head=%0d format=%0d terms=%0d events=%0d words=%0d cycles=%0d",
                head_index, stages[head_index], heads[head_index],
                formats[head_index], term_counts[head_index],
                event_counts[head_index], word_counts[head_index], latency);
            @(negedge clk_core);
            done_ready = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            done_ready = 1'b0;

            replay_begin_valid = 1'b1;
            replay_context_id = 1'b0;
            replay_head_id = heads[head_index];
            replay_start_word = '0;
            do @(posedge clk_core); while (!replay_begin_ready);
            @(negedge clk_core);
            replay_begin_valid = 1'b0;
            for (int word = 0; word < word_count; word = word + 1) begin
                while (!replay_word_valid) @(posedge clk_core);
                if (replay_word_data != expected_words[word_offset + word] ||
                    replay_word_index != 7'(word) ||
                    replay_word_last != (word == word_count - 1) ||
                    replay_tag != tags[head_index] ||
                    replay_format != formats[head_index] ||
                    replay_mode_is_csr != (formats[head_index] != 0) ||
                    replay_payload_bits != payload_bits_rows[head_index])
                    $fatal(1, "all45 replay mismatch head=%0d word=%0d",
                        head_index, word);
                @(negedge clk_core);
                replay_word_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                replay_word_ready = 1'b0;
            end
            release_valid = 1'b1;
            release_context_id = 1'b0;
            release_head_id = heads[head_index];
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

        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/raw_records.memh", raw_records);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/expected_words.memh", expected_words);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/tags.memh", tags);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/stages.memh", stages);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/heads.memh", heads);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/formats.memh", formats);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/reasons.memh", reasons);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/payload_bits.memh", payload_bits_rows);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/word_counts.memh", word_counts);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/word_offsets.memh", word_offsets);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/term_counts.memh", term_counts);
        $readmemh("tb_hitflow/vectors/gatestack_all45_builder_20260720/event_counts.memh", event_counts);

        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        for (int item = 0; item < HEAD_COUNT; item = item + 1)
            run_head(item);

        repeat (3) @(posedge clk_core);
        $display(
            "AUDIT: unused inspect boundary ready=%b meta_valid=%b exists=%b tag=%h csr=%b format=%0d payload=%0d words=%0d",
            inspect_ready, inspect_meta_valid, inspect_exists, inspect_tag,
            inspect_mode_is_csr, inspect_format, inspect_payload_bits,
            inspect_word_count);
        if (slot_valid_flat != '0 || workspace_protocol_error ||
            serializer_protocol_error || slot_protocol_error ||
            count_workspace_heads != 45 ||
            count_workspace_raw_fallback_heads != 0 ||
            count_workspace_terms != 762 ||
            count_workspace_destinations != 3226 ||
            count_workspace_scan_cycles != 2728 ||
            count_workspace_output_stall_cycles == 0 ||
            count_builder_committed_heads != 45 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 861 ||
            count_slot_commit_heads != 45 || count_slot_replay_heads != 45 ||
            count_slot_release_heads != 45)
            $fatal(1,
                "all45 final counters/state mismatch heads=%0d words=%0d terms=%0d destinations=%0d scans=%0d stalls=%0d",
                count_workspace_heads, count_builder_committed_words,
                count_workspace_terms, count_workspace_destinations,
                count_workspace_scan_cycles,
                count_workspace_output_stall_cycles);
        $display(
            "PASS: all45 full C0 heads=%0d words=%0d terms=%0d destinations=%0d scan=%0d stalls=%0d",
            count_workspace_heads, count_builder_committed_words,
            count_workspace_terms, count_workspace_destinations,
            count_workspace_scan_cycles, count_workspace_output_stall_cycles);
        $finish;
    end

endmodule

`default_nettype wire
