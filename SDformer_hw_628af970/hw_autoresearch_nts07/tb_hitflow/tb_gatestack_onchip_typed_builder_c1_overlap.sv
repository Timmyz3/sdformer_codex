`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_onchip_typed_builder_c1_overlap;
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
    logic [31:0] done_sequence;
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
    logic workspace_protocol_error, serializer_protocol_error;
    logic slot_protocol_error;
    logic [31:0] count_workspace_heads;
    logic [31:0] count_workspace_raw_fallback_heads;
    logic [31:0] count_workspace_terms, count_workspace_destinations;
    logic [31:0] count_workspace_scan_cycles;
    logic [31:0] count_workspace_output_stall_cycles;
    logic [31:0] count_builder_committed_heads;
    logic [31:0] count_builder_aborted_heads;
    logic [31:0] count_builder_committed_words;
    logic [31:0] count_slot_commit_heads, count_slot_replay_heads;
    logic [31:0] count_slot_release_heads;
    logic [31:0] count_capture_blocked_cycles;
    logic [31:0] count_capture_service_overlap_cycles;
    logic [31:0] count_order_wait_cycles;

    logic [40:0] raw0 [0:161];
    logic [40:0] raw1 [0:161];
    logic [63:0] expected0 [0:33];
    logic [63:0] expected1 [0:101];
    logic [31:0] cycle_q;
    int start_cycle;
    int finish_cycle;

    gatestack_onchip_typed_builder_c1_top dut (.*);

    always #5 clk_core <= ~clk_core;
    always_ff @(posedge clk_core) begin
        if (rst_core)
            cycle_q <= '0;
        else
            cycle_q <= cycle_q + 1'b1;
    end

    task automatic send_head(input int which);
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = which == 0 ? 5'd0 : 5'd4;
            head_tag = which == 0 ? 32'h6900_0000 : 32'h6903_0004;
            do @(posedge clk_core); while (!head_begin_ready);
            @(negedge clk_core);
            head_begin_valid = 1'b0;
            for (int token = 0; token < 162; token = token + 1) begin
                token_valid = 1'b1;
                token_id = 8'(token);
                token_k_bits = which == 0 ? raw0[token][31:0] :
                                            raw1[token][31:0];
                token_gate_code = which == 0 ? raw0[token][40:32] :
                                               raw1[token][40:32];
                token_last = token == 161;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
            end
            token_valid = 1'b0;
        end
    endtask

    task automatic replay_release(input int which);
        int words;
        begin
            words = which == 0 ? 34 : 102;
            replay_begin_valid = 1'b1;
            replay_context_id = 1'b0;
            replay_head_id = which == 0 ? 5'd0 : 5'd4;
            replay_start_word = '0;
            do @(posedge clk_core); while (!replay_begin_ready);
            @(negedge clk_core);
            replay_begin_valid = 1'b0;
            for (int word = 0; word < words; word = word + 1) begin
                while (!replay_word_valid) @(posedge clk_core);
                if (replay_word_data !=
                        (which == 0 ? expected0[word] : expected1[word]) ||
                    replay_word_index != 7'(word) ||
                    replay_word_last != (word == words - 1) ||
                    replay_tag != (which == 0 ? 32'h6900_0000 :
                                               32'h6903_0004) ||
                    replay_format != (which == 0 ? FORMAT_IPD32W :
                                                  FORMAT_FADC24) ||
                    !replay_mode_is_csr ||
                    replay_payload_bits != (which == 0 ? 16'd2168 :
                                                          16'd6520))
                    $fatal(1, "C1 replay mismatch which=%0d word=%0d",
                        which, word);
                @(negedge clk_core);
                replay_word_ready = 1'b1;
                @(posedge clk_core);
                @(negedge clk_core);
                replay_word_ready = 1'b0;
            end
            release_valid = 1'b1;
            release_context_id = 1'b0;
            release_head_id = which == 0 ? 5'd0 : 5'd4;
            do @(posedge clk_core); while (!release_ready);
            @(negedge clk_core);
            release_valid = 1'b0;
        end
    endtask

    task automatic consume_done(input int which);
        logic [31:0] expected_tag;
        logic [1:0] expected_format;
        logic [7:0] expected_words;
        logic [2:0] expected_reason;
        logic [15:0] expected_bits;
        begin
            expected_tag = which == 0 ? 32'h6900_0000 : 32'h6903_0004;
            expected_format = which == 0 ? FORMAT_IPD32W : FORMAT_FADC24;
            expected_words = which == 0 ? 8'd34 : 8'd102;
            expected_reason = which == 0 ? 3'd0 : 3'd2;
            expected_bits = which == 0 ? 16'd2168 : 16'd6520;
            while (!done_valid) @(posedge clk_core);
            if (done_sequence != 32'(which) || done_tag != expected_tag ||
                done_format != expected_format || done_error ||
                done_word_count != expected_words ||
                selected_reason != expected_reason ||
                selected_payload_bits != expected_bits)
                $fatal(1, "C1 done mismatch which=%0d seq=%0d tag=%h",
                    which, done_sequence, done_tag);
            if (which == 1)
                finish_cycle = cycle_q;
            @(posedge clk_core);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        head_begin_valid = 1'b0;
        token_valid = 1'b0;
        done_ready = 1'b1;
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

        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/raw_records.memh", raw0);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/raw_records.memh", raw1);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/ipd_s0_h0/expected_words.memh", expected0);
        $readmemh("tb_hitflow/vectors/typed_serializer_real_20260719/fadc_s3_h4/expected_words.memh", expected1);

        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;
        start_cycle = cycle_q;
        fork
            begin
                send_head(0);
                send_head(1);
            end
            begin
                consume_done(0);
                consume_done(1);
            end
        join

        if (finish_cycle - start_cycle >= 1492 ||
            count_capture_service_overlap_cycles == 0)
            $fatal(1, "C1 did not overlap capture/service makespan=%0d overlap=%0d",
                finish_cycle - start_cycle,
                count_capture_service_overlap_cycles);

        replay_release(0);
        replay_release(1);
        repeat (3) @(posedge clk_core);
        $display(
            "AUDIT: C1 unused inspect ready=%b meta=%b exists=%b tag=%h csr=%b format=%0d payload=%0d words=%0d",
            inspect_ready, inspect_meta_valid, inspect_exists, inspect_tag,
            inspect_mode_is_csr, inspect_format, inspect_payload_bits,
            inspect_word_count);
        if (slot_valid_flat != '0 || workspace_protocol_error ||
            serializer_protocol_error || slot_protocol_error ||
            count_workspace_heads != 2 ||
            count_workspace_raw_fallback_heads != 0 ||
            count_workspace_terms != 93 ||
            count_workspace_destinations != 941 ||
            count_workspace_scan_cycles != 443 ||
            count_builder_committed_heads != 2 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 136 ||
            count_slot_commit_heads != 2 || count_slot_replay_heads != 2 ||
            count_slot_release_heads != 2 || count_order_wait_cycles != 0)
            $fatal(1,
                "C1 counters mismatch heads=%0d terms=%0d dest=%0d scans=%0d words=%0d commits=%0d replays=%0d releases=%0d blocked=%0d overlap=%0d orderwait=%0d",
                count_workspace_heads, count_workspace_terms,
                count_workspace_destinations, count_workspace_scan_cycles,
                count_builder_committed_words, count_slot_commit_heads,
                count_slot_replay_heads, count_slot_release_heads,
                count_capture_blocked_cycles,
                count_capture_service_overlap_cycles,
                count_order_wait_cycles);
        $display(
            "PASS: C1 ordered overlap makespan=%0d C0sum=1492 speedup=%f overlap=%0d stalls=%0d blocked=%0d",
            finish_cycle - start_cycle,
            1492.0 / (finish_cycle - start_cycle),
            count_capture_service_overlap_cycles,
            count_workspace_output_stall_cycles,
            count_capture_blocked_cycles);
        $finish;
    end

endmodule

`default_nettype wire
