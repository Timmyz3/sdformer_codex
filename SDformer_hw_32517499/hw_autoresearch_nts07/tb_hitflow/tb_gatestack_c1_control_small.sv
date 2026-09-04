`timescale 1ns/1ps
`default_nettype none

module tb_gatestack_c1_control_small;
    logic clk_core, rst_core;
    logic head_begin_valid, head_begin_ready, head_context_id;
    logic [1:0] head_id;
    logic [31:0] head_tag;
    logic token_valid, token_ready;
    logic [2:0] token_id;
    logic [8:0] token_gate_code;
    logic [3:0] token_k_bits;
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
    logic [1:0] inspect_head_id;
    logic inspect_meta_valid, inspect_meta_ready, inspect_exists;
    logic [31:0] inspect_tag;
    logic inspect_mode_is_csr;
    logic [1:0] inspect_format;
    logic [15:0] inspect_payload_bits, inspect_word_count;
    logic replay_begin_valid, replay_begin_ready, replay_context_id;
    logic [1:0] replay_head_id;
    logic [3:0] replay_start_word;
    logic replay_word_valid, replay_word_ready;
    logic [63:0] replay_word_data;
    logic [3:0] replay_word_index;
    logic replay_word_last;
    logic [31:0] replay_tag;
    logic replay_mode_is_csr;
    logic [1:0] replay_format;
    logic [15:0] replay_payload_bits;
    logic release_valid, release_ready, release_context_id;
    logic [1:0] release_head_id;
    logic [7:0] slot_valid_flat;
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

    gatestack_onchip_typed_builder_c1_top #(
        .TOKENS(8), .LANES(4), .CLASS_SLOTS(2), .HEADS(4),
        .SLOT_WORDS(16)
    ) dut (.*);

    always #5 clk_core <= ~clk_core;

    task automatic send_head(input int which);
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = 2'(which);
            head_tag = 32'h7100_0000 + 32'(which);
            do @(posedge clk_core); while (!head_begin_ready);
            @(negedge clk_core);
            head_begin_valid = 1'b0;
            for (int token = 0; token < 8; token = token + 1) begin
                token_valid = 1'b1;
                token_id = 3'(token);
                token_gate_code = which == 0 ? 9'd64 : 9'd128;
                token_k_bits = '0;
                if (which == 0 && token == 0)
                    token_k_bits = 4'b0001;
                else if (which == 0 && token == 1)
                    token_k_bits = 4'b0011;
                else if (which == 1 && (token == 0 || token == 2))
                    token_k_bits = 4'b1000;
                token_last = token == 7;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
            end
            token_valid = 1'b0;
        end
    endtask

    task automatic consume_done(input int which);
        begin
            while (!done_valid) @(posedge clk_core);
            if (done_sequence != 32'(which) ||
                done_tag != 32'h7100_0000 + 32'(which) ||
                done_format != 2'd1 || done_error ||
                done_word_count != 8'd4 || selected_reason != 3'd0 ||
                selected_payload_bits != (which == 0 ? 16'd216 : 16'd208))
                $fatal(1, "small C1 done mismatch which=%0d", which);
            @(posedge clk_core);
            @(negedge clk_core);
        end
    endtask

    task automatic release_head(input logic [1:0] which);
        begin
            release_valid = 1'b1;
            release_context_id = 1'b0;
            release_head_id = 2'(which);
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
        repeat (4) @(posedge clk_core);
        rst_core = 1'b0;

        send_head(0);
        send_head(1);

        while (!done_valid) @(posedge clk_core);
        repeat (3) begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_context_id = 1'b0;
            head_id = 2'd2;
            head_tag = 32'h7100_0002;
            @(posedge clk_core);
            if (head_begin_ready)
                $fatal(1, "third head must remain blocked while both workspaces are occupied");
        end
        @(negedge clk_core);
        head_begin_valid = 1'b0;
        done_ready = 1'b1;
        consume_done(0);
        consume_done(1);
        release_head(0);
        release_head(1);
        repeat (3) @(posedge clk_core);

        $display(
            "AUDIT: small C1 inspect=%b/%b/%b/%h/%b/%0d/%0d/%0d replay=%b/%b/%h/%0d/%b/%h/%b/%0d/%0d stalls=%0d blocked=%0d",
            inspect_ready, inspect_meta_valid, inspect_exists, inspect_tag,
            inspect_mode_is_csr, inspect_format, inspect_payload_bits,
            inspect_word_count, replay_begin_ready, replay_word_valid,
            replay_word_data, replay_word_index, replay_word_last, replay_tag,
            replay_mode_is_csr, replay_format, replay_payload_bits,
            count_workspace_output_stall_cycles,
            count_capture_blocked_cycles);
        if (slot_valid_flat != '0 || workspace_protocol_error ||
            serializer_protocol_error || slot_protocol_error ||
            count_workspace_heads != 2 ||
            count_workspace_raw_fallback_heads != 0 ||
            count_workspace_terms != 3 ||
            count_workspace_destinations != 5 ||
            count_workspace_scan_cycles != 5 ||
            count_builder_committed_heads != 2 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 8 ||
            count_slot_commit_heads != 2 || count_slot_replay_heads != 0 ||
            count_slot_release_heads != 2 || count_order_wait_cycles != 0 ||
            count_capture_blocked_cycles < 3 ||
            count_capture_service_overlap_cycles == 0)
            $fatal(1, "small C1 counters mismatch");
        $display(
            "PASS: small C1 control overlap=%0d",
            count_capture_service_overlap_cycles);
        $finish;
    end

endmodule

`default_nettype wire
