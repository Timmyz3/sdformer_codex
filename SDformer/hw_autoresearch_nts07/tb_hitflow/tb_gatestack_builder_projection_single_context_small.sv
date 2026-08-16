`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_builder_projection_single_context_small #(
    parameter int BUILDER_C1_ENABLE = 0
);
    localparam int TOKENS = 8;
    localparam int HEADS = 2;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    localparam logic [15:0] PAYLOAD_TAG = 16'h6800;
    localparam logic [15:0] GROUP_TAG = 16'h7800;

    logic clk_core, rst_core;
    logic head_begin_valid, head_begin_ready, head_id;
    logic [15:0] head_tag;
    logic token_valid, token_ready;
    logic [2:0] token_id;
    logic [8:0] token_gate_code;
    logic [31:0] token_k_bits;
    logic token_last;
    logic group_valid, group_ready;
    logic [15:0] group_tag;
    logic [3:0] group_first_output_tile, group_output_tile_count;
    logic group_done_valid, group_done_ready;
    logic [15:0] group_done_tag;
    logic group_done_error;
    logic batch_abort_valid, batch_abort_ready;
    logic weight_req_valid, weight_req_ready;
    logic [15:0] weight_req_tag;
    logic [5:0] weight_req_input_channel;
    logic [3:0] weight_req_output_tile;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [15:0] weight_rsp_tag;
    logic [5:0] weight_rsp_input_channel;
    logic [3:0] weight_rsp_output_tile;
    logic [15:0] weight_rsp_weights;
    logic bias_req_valid, bias_req_ready;
    logic [15:0] bias_req_tag, bias_rsp_tag;
    logic [3:0] bias_req_output_tile;
    logic [7:0] bias_req_token_id, bias_rsp_token_id;
    logic bias_req_allow, bias_rsp_valid, bias_rsp_ready;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values, bias_lookup_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [(BANKS*8)-1:0] final_token_ids;
    logic [15:0] final_tag;
    logic [(BANKS*OUT_TILE*ACC_W)-1:0] final_values;
    logic builder_done_pulse;
    logic [15:0] builder_done_tag;
    logic [1:0] builder_done_format;
    logic builder_done_error;
    logic [7:0] builder_done_word_count;
    logic [2:0] builder_selected_reason;
    logic [15:0] builder_selected_payload_bits;
    logic [1:0] batch_accepted_heads, batch_completed_heads;
    logic [HEADS-1:0] slot_valid_flat;
    logic slot_reset_pulse, protocol_error;
    logic [31:0] count_builder_heads, count_builder_raw_heads;
    logic [31:0] count_builder_terms, count_builder_destinations;
    logic [31:0] count_builder_scan_cycles;
    logic [31:0] count_builder_output_stalls;
    logic [31:0] count_builder_committed_heads;
    logic [31:0] count_builder_aborted_heads;
    logic [31:0] count_builder_committed_words;
    logic [31:0] count_slot_commit_heads, count_slot_replay_heads;
    logic [31:0] count_slot_release_heads;
    logic [31:0] count_builder_capture_blocked_cycles;
    logic [31:0] count_builder_overlap_cycles;
    logic [31:0] count_builder_order_wait_cycles;
    logic [31:0] count_payload_copy_words;
    logic [31:0] count_groups, count_tile_starts, count_head_issues;
    logic [31:0] count_control_requests, count_control_commits;
    logic [31:0] count_control_rejects, count_control_sessions;
    logic [31:0] count_projection_heads, count_projection_terms;
    logic [31:0] count_bias_commits, count_context_resets;
    logic [31:0] count_error_aborts, count_timeout_aborts;

    integer cycle_count;
    integer first_head_cycle, builders_done_cycle, group_done_cycle;
    integer done_count, final_count, tile0_final_count, tile1_final_count;
    integer signed final_signature;

    gatestack_builder_projection_single_context_top #(
        .BUILDER_C1_ENABLE(BUILDER_C1_ENABLE), .TOKENS(TOKENS),
        .HEADS(HEADS), .SLOT_WORDS(8), .MAX_TERMS(8),
        .RESIDENT_TERMS(4), .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .SEGMENT_TOKENS(4), .TAG_W(16), .INPUT_CH_W(6),
        .OUTPUT_TILE_W(4), .OUTPUT_TILE_COUNT_W(4), .HEAD_COUNT_W(2),
        .WORD_INDEX_W(3), .RES_TERM_IDX_W(2), .TOKEN_ID_W(8),
        .BUILD_TIMEOUT_CYCLES(256)
    ) dut (.*);

    gatestack_bias_sram_model #(
        .TAG_W(16), .OUTPUT_TILE_W(4), .TOKEN_ID_W(8),
        .OUT_TILE(OUT_TILE), .ACC_W(ACC_W)
    ) bias_sram (
        .clk_core, .rst_core, .req_allow(bias_req_allow),
        .bias_req_valid, .bias_req_ready, .bias_req_tag,
        .bias_req_output_tile,
        .bias_req_token_id, .lookup_values(bias_lookup_values),
        .bias_rsp_valid, .bias_rsp_ready, .bias_rsp_tag,
        .bias_rsp_token_id, .bias_rsp_values
    );

    always #5 clk_core <= ~clk_core;

    task automatic send_head(input integer which);
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_id = 1'(which);
            head_tag = PAYLOAD_TAG + 16'(which);
            do @(posedge clk_core); while (!head_begin_ready);
            if (which == 0)
                first_head_cycle = cycle_count;
            @(negedge clk_core);
            head_begin_valid = 1'b0;
            for (integer token = 0; token < TOKENS; token = token + 1) begin
                token_valid = 1'b1;
                token_id = 3'(token);
                token_gate_code = which == 0 ? 9'd2 : 9'd3;
                token_k_bits = '0;
                if (which == 0 && (token == 0 || token == 7))
                    token_k_bits[0] = 1'b1;
                if (which == 1 && (token == 1 || token == 6))
                    token_k_bits[1] = 1'b1;
                token_last = token == TOKENS - 1;
                do @(posedge clk_core); while (!token_ready);
                @(negedge clk_core);
            end
            token_valid = 1'b0;
            token_last = 1'b0;
        end
    endtask

    initial begin : weight_model
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_channel = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_weights = '0;
        wait (!rst_core);
        forever begin
            @(negedge clk_core);
            weight_req_ready = 1'b1;
            do @(posedge clk_core); while (!weight_req_valid);
            if ((weight_req_input_channel != 0 &&
                 weight_req_input_channel != 33) ||
                (weight_req_tag != GROUP_TAG &&
                 weight_req_tag != GROUP_TAG + 1'b1))
                $fatal(1, "weight request identity mismatch");
            weight_rsp_tag = weight_req_tag;
            weight_rsp_input_channel = weight_req_input_channel;
            weight_rsp_output_tile = weight_req_output_tile;
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            if (weight_rsp_input_channel == 0)
                weight_rsp_weights = {8'sd2, 8'sd1};
            else
                weight_rsp_weights = {8'sd4, -8'sd2};
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    end

    always_comb begin
        bias_req_allow = 1'b1;
        bias_lookup_values[31:0] = 32'(10 + 32'(bias_req_token_id));
        bias_lookup_values[63:32] = 32'(-20 - 32'(bias_req_token_id));
        final_ready = '1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core || slot_reset_pulse) begin
            cycle_count <= 0;
            done_count <= 0;
            final_count <= 0;
            tile0_final_count <= 0;
            tile1_final_count <= 0;
            final_signature <= 0;
            builders_done_cycle <= -1;
        end else begin
            integer fires;
            integer tile0_fires;
            integer tile1_fires;
            integer signed signature_delta;
            cycle_count <= cycle_count + 1;
            fires = 0;
            tile0_fires = 0;
            tile1_fires = 0;
            signature_delta = 0;
            if (dut.execution_group_valid &&
                (batch_accepted_heads != 2'd2 ||
                 batch_completed_heads != 2'd2 ||
                 slot_valid_flat != {HEADS{1'b1}} || protocol_error))
                $fatal(1, "execution launch before clean complete builder batch");
            if (builder_done_pulse) begin
                if (builder_done_error || builder_done_format != 2'd1 ||
                    builder_done_word_count != 8'd4 ||
                    builder_selected_payload_bits != 16'd208)
                    $fatal(1,
                        "builder done mismatch tag=%h format=%0d words=%0d bits=%0d reason=%0d",
                        builder_done_tag, builder_done_format,
                        builder_done_word_count,
                        builder_selected_payload_bits,
                        builder_selected_reason);
                done_count <= done_count + 1;
                if (done_count == HEADS - 1)
                    builders_done_cycle <= cycle_count;
            end
            if (bias_req_valid && bias_req_ready &&
                32'(bias_req_token_id) >= TOKENS)
                $fatal(1, "bias token id out of range");
            for (integer bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    integer token_value;
                    integer signed expected0, expected1;
                    integer signed actual0, actual1;
                    token_value = 32'(final_token_ids[(bank*8) +: 8]);
                    expected0 = 10 + token_value;
                    expected1 = -20 - token_value;
                    if (token_value == 0 || token_value == 7) begin
                        expected0 = expected0 + 2;
                        expected1 = expected1 + 4;
                    end
                    if (token_value == 1 || token_value == 6) begin
                        expected0 = expected0 - 6;
                        expected1 = expected1 + 12;
                    end
                    actual0 = $signed(final_values[(bank*64) +: 32]);
                    actual1 = $signed(final_values[(bank*64)+32 +: 32]);
                    if (actual0 != expected0 || actual1 != expected1)
                        $fatal(1,
                            "final mismatch tag=%h token=%0d got=(%0d,%0d) expected=(%0d,%0d)",
                            final_tag, token_value, actual0, actual1,
                            expected0, expected1);
                    if (final_tag == GROUP_TAG)
                        tile0_fires = tile0_fires + 1;
                    else if (final_tag == GROUP_TAG + 1'b1)
                        tile1_fires = tile1_fires + 1;
                    else
                        $fatal(1, "unexpected final tag %h", final_tag);
                    fires = fires + 1;
                    signature_delta = signature_delta + actual0 + actual1;
                end
            end
            final_count <= final_count + fires;
            tile0_final_count <= tile0_final_count + tile0_fires;
            tile1_final_count <= tile1_final_count + tile1_fires;
            final_signature <= final_signature + signature_delta;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        head_begin_valid = 1'b0;
        head_id = '0;
        head_tag = '0;
        token_valid = 1'b0;
        token_id = '0;
        token_gate_code = '0;
        token_k_bits = '0;
        token_last = 1'b0;
        group_valid = 1'b0;
        group_tag = GROUP_TAG;
        group_first_output_tile = 4'd2;
        group_output_tile_count = 4'd2;
        group_done_ready = 1'b0;
        batch_abort_valid = 1'b0;
        first_head_cycle = -1;
        group_done_cycle = -1;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        // A repeated head is fail-stop until the host performs an explicit
        // atomic batch abort. The following clean batch proves recovery.
        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        send_head(0);
        @(negedge clk_core);
        head_begin_valid = 1'b1;
        head_id = 1'b0;
        head_tag = PAYLOAD_TAG;
        @(posedge clk_core);
        if (head_begin_ready)
            $fatal(1, "duplicate head unexpectedly accepted");
        @(negedge clk_core);
        head_begin_valid = 1'b0;
        @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "duplicate head did not latch protocol error");
        wait (group_done_valid);
        if (!group_done_error || group_done_tag != GROUP_TAG)
            $fatal(1, "automatic builder abort did not return tagged error completion");
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        if (protocol_error || batch_accepted_heads != 0 ||
            batch_completed_heads != 0 || slot_valid_flat != 0 ||
            !group_ready || head_begin_ready)
            $fatal(1, "batch abort did not restore clean admission state");

        // Manual abort follows the same tagged-completion contract even when
        // no head has been submitted yet.
        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        batch_abort_valid = 1'b1;
        @(posedge clk_core);
        if (!batch_abort_ready || !slot_reset_pulse)
            $fatal(1, "manual batch abort handshake/reset missing");
        @(negedge clk_core);
        batch_abort_valid = 1'b0;
        wait (group_done_valid);
        if (!group_done_error || group_done_tag != GROUP_TAG)
            $fatal(1, "manual batch abort did not return tagged error completion");
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;
        repeat (2) @(posedge clk_core);

        // A stalled build is covered by a wrapper-level watchdog, before the
        // execution watchdog can become active.
        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        wait (group_done_valid);
        if (!group_done_error || group_done_tag != GROUP_TAG)
            $fatal(1, "build watchdog did not return tagged error completion");
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;
        repeat (2) @(posedge clk_core);

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        fork
            begin
                send_head(0);
                send_head(1);
            end
            begin
                repeat (6) @(posedge clk_core);
                if (group_done_valid)
                    $fatal(1, "group completed before all heads built");
            end
        join
        while (!group_done_valid) @(posedge clk_core);
        group_done_cycle = cycle_count;
        if (group_done_tag != GROUP_TAG || group_done_error ||
            protocol_error || slot_reset_pulse ||
            count_payload_copy_words != 0 || count_builder_heads != 2 ||
            count_builder_raw_heads != 0 || count_builder_terms != 2 ||
            count_builder_destinations != 4 ||
            count_builder_scan_cycles == 0 ||
            count_builder_committed_heads != 2 ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != 8 ||
            count_slot_commit_heads != 2 || count_slot_replay_heads != 4 ||
            count_slot_release_heads != 2 || count_groups != 1 ||
            count_tile_starts != 2 || count_head_issues != 4 ||
            count_control_requests != 4 || count_control_commits != 4 ||
            count_control_rejects != 0 || count_control_sessions != 4 ||
            count_projection_heads != 4 || count_projection_terms != 4 ||
            count_bias_commits != 16 || count_context_resets != 0 ||
            count_error_aborts != 0 || count_timeout_aborts != 0 ||
            count_builder_order_wait_cycles != 0 ||
            done_count != 2 || final_count != 16 ||
            tile0_final_count != 8 || tile1_final_count != 8 ||
            final_signature != -112)
            $fatal(1,
                "integration counters mismatch mode=%0d build=%0d/%0d terms=%0d dest=%0d words=%0d slots=%0d/%0d/%0d projection=%0d/%0d finals=%0d signature=%0d errors=%b",
                BUILDER_C1_ENABLE, count_builder_heads,
                count_builder_committed_heads, count_builder_terms,
                count_builder_destinations, count_builder_committed_words,
                count_slot_commit_heads, count_slot_replay_heads,
                count_slot_release_heads, count_projection_heads,
                count_projection_terms, final_count, final_signature,
                protocol_error);

        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;
        repeat (3) @(posedge clk_core);
        if (!group_ready || head_begin_ready || batch_accepted_heads != 0 ||
            batch_completed_heads != 0 || slot_valid_flat != 0)
            $fatal(1, "next batch did not reopen after group completion");
        $display(
            "RESULT mode=C%0d signature=%0d total_cycles=%0d build_cycles=%0d projection_cycles=%0d replays=%0d releases=%0d heads=%0d terms=%0d scan=%0d stalls=%0d blocked=%0d overlap=%0d",
            BUILDER_C1_ENABLE, final_signature,
            group_done_cycle - first_head_cycle,
            builders_done_cycle - first_head_cycle,
            group_done_cycle - builders_done_cycle,
            count_slot_replay_heads, count_slot_release_heads,
            count_projection_heads, count_projection_terms,
            count_builder_scan_cycles,
            count_builder_output_stalls,
            count_builder_capture_blocked_cycles,
            count_builder_overlap_cycles);
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk_core);
        $fatal(1, "builder projection integration timeout group=%b/%b done=%b/%b error=%b abort=%b/%b request=%b launched=%b accepted=%0d completed=%0d slots=%b head=%b/%b builder_done=%b exec_group=%b/%b",
               group_valid, group_ready, group_done_valid, group_done_ready,
               protocol_error, batch_abort_valid, batch_abort_ready,
               dut.group_request_active_q, dut.execution_launched_q,
               batch_accepted_heads, batch_completed_heads, slot_valid_flat,
               head_begin_valid, head_begin_ready, builder_done_pulse,
               dut.execution_group_valid, dut.execution_group_ready);
    end
endmodule

`default_nettype wire
