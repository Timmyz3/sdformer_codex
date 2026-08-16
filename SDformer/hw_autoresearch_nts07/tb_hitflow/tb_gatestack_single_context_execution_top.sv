`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_single_context_execution_top;
    localparam int TOKENS = 8;
    localparam int OUT_TILE = 2;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    localparam logic [15:0] PAYLOAD_TAG = 16'h6800;
    localparam logic [15:0] GROUP_TAG = 16'h7800;

    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [15:0] group_tag;
    logic [1:0] group_head_count;
    logic [3:0] group_first_output_tile, group_output_tile_count;
    logic group_done_valid, group_done_ready;
    logic [15:0] group_done_tag;
    logic group_done_error;
    logic payload_commit_begin_valid, payload_commit_begin_ready;
    logic payload_commit_head_id;
    logic [15:0] payload_commit_tag;
    logic payload_commit_mode_is_csr;
    logic [15:0] payload_commit_bits;
    logic payload_commit_word_valid, payload_commit_word_ready;
    logic [63:0] payload_commit_word_data;
    logic payload_commit_word_last;
    /* verilator lint_off UNUSEDSIGNAL */
    logic external_slot_inspect_valid, external_slot_inspect_ready;
    logic external_slot_inspect_context_id;
    logic external_slot_inspect_head_id;
    logic external_slot_inspect_meta_valid;
    logic external_slot_inspect_meta_ready, external_slot_inspect_exists;
    logic [15:0] external_slot_inspect_tag;
    logic external_slot_inspect_mode_is_csr;
    logic [1:0] external_slot_inspect_format;
    logic [15:0] external_slot_inspect_payload_bits;
    logic [15:0] external_slot_inspect_word_count;
    logic external_slot_replay_begin_valid;
    logic external_slot_replay_begin_ready;
    logic external_slot_replay_context_id;
    logic external_slot_replay_head_id;
    logic [15:0] external_slot_replay_payload_tag;
    logic [2:0] external_slot_replay_start_word;
    logic external_slot_replay_word_valid;
    logic external_slot_replay_word_ready;
    logic [63:0] external_slot_replay_word_data;
    logic [2:0] external_slot_replay_word_index;
    logic external_slot_replay_word_last;
    logic [15:0] external_slot_replay_tag;
    logic external_slot_replay_mode_is_csr;
    logic [1:0] external_slot_replay_format;
    logic [15:0] external_slot_replay_payload_bits;
    logic external_slot_release_valid, external_slot_release_ready;
    logic external_slot_release_context_id;
    logic external_slot_release_head_id;
    logic [1:0] external_slot_valid_flat;
    logic external_slot_protocol_error;
    logic [31:0] external_slot_count_replays;
    logic [31:0] external_slot_count_releases;
    logic external_slot_reset_pulse;
    /* verilator lint_on UNUSEDSIGNAL */
    logic descriptor_fill_begin_valid, descriptor_fill_begin_ready;
    logic descriptor_fill_head_id;
    logic [15:0] descriptor_fill_tag;
    logic [7:0] descriptor_fill_term_count;
    logic [1:0] descriptor_fill_format;
    logic descriptor_fill_begin_cacheable;
    logic descriptor_fill_entry_valid, descriptor_fill_entry_ready;
    logic [8:0] descriptor_fill_gate_code;
    logic [4:0] descriptor_fill_lane_id;
    logic [7:0] descriptor_fill_destination_count;
    logic descriptor_fill_entry_last;
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
    logic protocol_error;
    logic [31:0] count_groups, count_tile_starts, count_head_issues;
    logic [31:0] count_control_requests, count_control_commits;
    logic [31:0] count_control_rejects, count_control_sessions;
    logic [31:0] count_slot_replays, count_slot_releases;
    logic [31:0] count_cache_hits, count_cache_releases;
    logic [31:0] count_projection_heads, count_projection_terms;
    logic [31:0] count_bias_commits;
    logic [31:0] count_context_resets, count_error_aborts;
    logic [31:0] count_timeout_aborts;
    int cycle_count, final_count, tile0_final_count, tile1_final_count;
    int normal_group_cycle;

    gatestack_single_context_execution_top #(
        .TOKENS(TOKENS), .HEADS(2), .HEAD_BITS(384),
        .MAX_TERMS(8), .RESIDENT_TERMS(4), .OUT_TILE(OUT_TILE),
        .BANKS(BANKS), .SEGMENT_TOKENS(4), .TAG_W(16),
        .INPUT_CH_W(6), .OUTPUT_TILE_W(4),
        .OUTPUT_TILE_COUNT_W(4), .HEAD_COUNT_W(2),
        .WORD_INDEX_W(3), .RES_TERM_IDX_W(2)
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

    function automatic logic [31:0] descriptor(
        input logic [8:0] gate_value,
        input logic [4:0] lane_value,
        input logic [7:0] count_value
    );
        descriptor = 32'(gate_value) | (32'(lane_value) << 9) |
                     (32'(count_value) << 14);
    endfunction

    function automatic logic [63:0] header0(input logic [15:0] tag_value);
        header0 = (64'(tag_value) << 32) | (64'(1) << 20) |
                  (64'(1) << 16) | 64'h4753;
    endfunction

    function automatic logic [63:0] header1;
        header1 = 64'(208) | (64'(1) << 13) | (64'(2) << 21) |
                  (64'(1) << 34) | (64'(2) << 37) | (64'(24) << 45);
    endfunction

    task automatic commit_payload_word(
        input logic [63:0] data_value,
        input logic last_value
    );
        begin
            @(negedge clk_core);
            payload_commit_word_data = data_value;
            payload_commit_word_last = last_value;
            payload_commit_word_valid = 1'b1;
            do @(posedge clk_core); while (!payload_commit_word_ready);
            @(negedge clk_core);
            payload_commit_word_valid = 1'b0;
            payload_commit_word_last = 1'b0;
        end
    endtask

    task automatic commit_csr_payload(
        input logic head_id,
        input logic [15:0] payload_tag,
        input logic [8:0] gate_code,
        input logic [4:0] lane_id,
        input logic [63:0] token_word
    );
        begin
            @(negedge clk_core);
            payload_commit_head_id = head_id;
            payload_commit_tag = payload_tag;
            payload_commit_begin_valid = 1'b1;
            do @(posedge clk_core); while (!payload_commit_begin_ready);
            @(negedge clk_core);
            payload_commit_begin_valid = 1'b0;
            commit_payload_word(header0(payload_tag), 1'b0);
            commit_payload_word(header1(), 1'b0);
            commit_payload_word({32'd0, descriptor(gate_code, lane_id, 8'd2)},
                                1'b0);
            commit_payload_word(token_word, 1'b1);
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
        if (rst_core) begin
            cycle_count <= 0;
            final_count <= 0;
            tile0_final_count <= 0;
            tile1_final_count <= 0;
        end else begin
            integer fires;
            integer tile0_fires;
            integer tile1_fires;
            fires = 0;
            tile0_fires = 0;
            tile1_fires = 0;
            cycle_count <= cycle_count + 1;
            if (bias_req_valid && bias_req_ready &&
                32'(bias_req_token_id) >= TOKENS)
                $fatal(1, "bias token id out of range");
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
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
                        $fatal(1, "final mismatch tag=%h token=%0d got=(%0d,%0d) expected=(%0d,%0d)",
                               final_tag, token_value, actual0, actual1,
                               expected0, expected1);
                    if (final_tag == GROUP_TAG)
                        tile0_fires = tile0_fires + 1;
                    else if (final_tag == GROUP_TAG + 1'b1)
                        tile1_fires = tile1_fires + 1;
                    else
                        $fatal(1, "unexpected final execution tag %h", final_tag);
                    fires = fires + 1;
                end
            end
            final_count <= final_count + fires;
            tile0_final_count <= tile0_final_count + tile0_fires;
            tile1_final_count <= tile1_final_count + tile1_fires;
        end
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = GROUP_TAG;
        group_head_count = 2'd2;
        group_first_output_tile = 4'd2;
        group_output_tile_count = 4'd2;
        group_done_ready = 1'b0;
        payload_commit_begin_valid = 1'b0;
        payload_commit_head_id = 1'b0;
        payload_commit_tag = PAYLOAD_TAG;
        payload_commit_mode_is_csr = 1'b1;
        payload_commit_bits = 16'd208;
        payload_commit_word_valid = 1'b0;
        payload_commit_word_data = '0;
        payload_commit_word_last = 1'b0;
        external_slot_inspect_ready = 1'b0;
        external_slot_inspect_meta_valid = 1'b0;
        external_slot_inspect_exists = 1'b0;
        external_slot_inspect_tag = '0;
        external_slot_inspect_mode_is_csr = 1'b0;
        external_slot_inspect_format = '0;
        external_slot_inspect_payload_bits = '0;
        external_slot_inspect_word_count = '0;
        external_slot_replay_begin_ready = 1'b0;
        external_slot_replay_word_valid = 1'b0;
        external_slot_replay_word_data = '0;
        external_slot_replay_word_index = '0;
        external_slot_replay_word_last = 1'b0;
        external_slot_replay_tag = '0;
        external_slot_replay_mode_is_csr = 1'b0;
        external_slot_replay_format = '0;
        external_slot_replay_payload_bits = '0;
        external_slot_release_ready = 1'b0;
        external_slot_valid_flat = '0;
        external_slot_protocol_error = 1'b0;
        external_slot_count_replays = '0;
        external_slot_count_releases = '0;
        descriptor_fill_begin_valid = 1'b0;
        descriptor_fill_head_id = 1'b0;
        descriptor_fill_tag = PAYLOAD_TAG;
        descriptor_fill_term_count = 8'd1;
        descriptor_fill_format = 2'd1;
        descriptor_fill_entry_valid = 1'b0;
        descriptor_fill_gate_code = 9'd2;
        descriptor_fill_lane_id = 5'd0;
        descriptor_fill_destination_count = 8'd2;
        descriptor_fill_entry_last = 1'b1;
        normal_group_cycle = 0;
        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;

        commit_csr_payload(1'b0, PAYLOAD_TAG, 9'd2, 5'd0,
                           64'h0000_0000_0000_0700);
        commit_csr_payload(1'b1, PAYLOAD_TAG + 1'b1, 9'd3, 5'd1,
                           64'h0000_0000_0000_0601);

        @(negedge clk_core);
        descriptor_fill_begin_valid = 1'b1;
        do @(posedge clk_core); while (!descriptor_fill_begin_ready);
        if (!descriptor_fill_begin_cacheable)
            $fatal(1, "resident descriptor unexpectedly bypassed cache");
        @(negedge clk_core);
        descriptor_fill_begin_valid = 1'b0;
        descriptor_fill_entry_valid = 1'b1;
        do @(posedge clk_core); while (!descriptor_fill_entry_ready);
        @(negedge clk_core);
        descriptor_fill_entry_valid = 1'b0;

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;

        wait (group_done_valid);
        normal_group_cycle = cycle_count;
        if (group_done_tag != GROUP_TAG || group_done_error ||
            protocol_error || count_groups != 1 || count_tile_starts != 2 ||
            count_head_issues != 4 || count_control_requests != 4 ||
            count_control_commits != 4 || count_control_rejects != 0 ||
            count_control_sessions != 4 || count_slot_replays != 4 ||
            count_slot_releases != 2 || count_cache_hits != 3 ||
            count_cache_releases != 2 || count_projection_heads != 4 ||
            count_projection_terms != 4 || count_bias_commits != 16 ||
            count_context_resets != 0 || count_error_aborts != 0 ||
            count_timeout_aborts != 0 ||
            final_count != 16 || tile0_final_count != 8 ||
            tile1_final_count != 8)
            $fatal(1, "single-context execution counters mismatch tiles=%0d heads=%0d sessions=%0d replays=%0d cache_hits=%0d releases=(%0d,%0d) finals=%0d/%0d/%0d protocol=%b",
                   count_tile_starts, count_projection_heads,
                   count_control_sessions, count_slot_replays,
                   count_cache_hits, count_slot_releases,
                   count_cache_releases, final_count,
                   tile0_final_count, tile1_final_count, protocol_error);
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        group_done_ready = 1'b0;

        // Both slots were released by the final tile. Reissuing without refill
        // must produce one bounded full-context abort instead of deadlock.
        group_tag = 16'h7900;
        group_head_count = 2'd1;
        group_first_output_tile = 4'd4;
        group_output_tile_count = 4'd1;
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        while (!group_done_valid) @(posedge clk_core);
        if (group_done_tag != 16'h7900 || !group_done_error ||
            !protocol_error || count_context_resets != 1 ||
            count_error_aborts != 1 || count_timeout_aborts != 0)
            $fatal(1, "integrated context abort mismatch tag=%h error=%b resets=%0d/%0d/%0d",
                   group_done_tag, group_done_error, count_context_resets,
                   count_error_aborts, count_timeout_aborts);
        repeat (2) begin
            @(posedge clk_core);
            if (!group_done_valid || group_ready)
                $fatal(1, "abort response/admission backpressure mismatch");
        end
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: single-context execution normal=%0d total_with_abort=%0d",
                 normal_group_cycle, cycle_count);
        $finish;
    end

    initial begin
        repeat (40000) @(posedge clk_core);
        $fatal(1, "single-context execution timeout");
    end
endmodule

`default_nettype wire
