`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_single_context_execution_external_slot;
    localparam int TOKENS = 2;
    localparam int OUT_TILE = 1;
    localparam int BANKS = 1;
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
    logic external_slot_valid_flat;
    logic external_slot_protocol_error;
    logic [31:0] external_slot_count_replays;
    logic [31:0] external_slot_count_releases;
    logic external_slot_reset_pulse;

    /* verilator lint_off UNUSEDSIGNAL */
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
    logic [7:0] weight_rsp_weights;
    logic bias_req_valid, bias_req_ready;
    logic [15:0] bias_req_tag, bias_rsp_tag;
    logic [3:0] bias_req_output_tile;
    logic [7:0] bias_req_token_id, bias_rsp_token_id;
    logic bias_req_allow, bias_rsp_valid, bias_rsp_ready;
    logic [31:0] bias_rsp_values, bias_lookup_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [7:0] final_token_ids;
    logic [15:0] final_tag;
    logic [31:0] final_values;

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
    /* verilator lint_on UNUSEDSIGNAL */
    integer final_count, service_stage;

    gatestack_single_context_execution_top #(
        .TOKENS(TOKENS), .HEADS(1), .HEAD_BITS(384),
        .MAX_TERMS(8), .RESIDENT_TERMS(4), .ENABLE_RESIDENCY(0),
        .EXTERNAL_SLOT_SERVICE_ENABLE(1), .OUT_TILE(OUT_TILE),
        .BANKS(BANKS), .SEGMENT_TOKENS(2), .TAG_W(16),
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

    function automatic logic [31:0] descriptor;
        descriptor = 32'(9'd2) | (32'(8'd2) << 14);
    endfunction

    function automatic logic [63:0] payload_word(input integer index);
        case (index)
            0: payload_word = (64'(PAYLOAD_TAG) << 32) |
                              (64'(1) << 20) | (64'(1) << 16) | 64'h4753;
            1: payload_word = 64'(208) | (64'(1) << 13) |
                              (64'(2) << 21) | (64'(1) << 34) |
                              (64'(2) << 37) | (64'(24) << 45);
            2: payload_word = {32'd0, descriptor()};
            default: payload_word = 64'h0000_0000_0000_0100;
        endcase
    endfunction

    always_comb begin
        weight_req_ready = !weight_rsp_valid || weight_rsp_ready;
        bias_req_allow = 1'b1;
        bias_lookup_values = '0;
        final_ready = '1;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            weight_rsp_valid <= 1'b0;
            weight_rsp_tag <= '0;
            weight_rsp_input_channel <= '0;
            weight_rsp_output_tile <= '0;
            weight_rsp_weights <= '0;
            final_count <= 0;
        end else begin
            if (weight_rsp_valid && weight_rsp_ready)
                weight_rsp_valid <= 1'b0;
            if (weight_req_valid && weight_req_ready) begin
                weight_rsp_valid <= 1'b1;
                weight_rsp_tag <= weight_req_tag;
                weight_rsp_input_channel <= weight_req_input_channel;
                weight_rsp_output_tile <= weight_req_output_tile;
                weight_rsp_weights <= 8'd1;
            end
            if (final_valid[0] && final_ready[0])
                final_count <= final_count + 1;
        end
    end

    initial begin : external_slot_service
        external_slot_inspect_ready = 1'b0;
        external_slot_inspect_meta_valid = 1'b0;
        external_slot_inspect_exists = 1'b1;
        external_slot_inspect_tag = PAYLOAD_TAG;
        external_slot_inspect_mode_is_csr = 1'b1;
        external_slot_inspect_format = 2'd1;
        external_slot_inspect_payload_bits = 16'd208;
        external_slot_inspect_word_count = 16'd4;
        external_slot_replay_begin_ready = 1'b0;
        external_slot_replay_word_valid = 1'b0;
        external_slot_replay_word_data = '0;
        external_slot_replay_word_index = '0;
        external_slot_replay_word_last = 1'b0;
        external_slot_replay_tag = PAYLOAD_TAG;
        external_slot_replay_mode_is_csr = 1'b1;
        external_slot_replay_format = 2'd1;
        external_slot_replay_payload_bits = 16'd208;
        external_slot_release_ready = 1'b0;
        external_slot_valid_flat = 1'b1;
        external_slot_protocol_error = 1'b0;
        external_slot_count_replays = '0;
        external_slot_count_releases = '0;
        service_stage = 0;

        wait (!rst_core);
        do @(posedge clk_core); while (!external_slot_inspect_valid);
        service_stage = 1;
        if (external_slot_inspect_context_id ||
            external_slot_inspect_head_id)
            $fatal(1, "external inspect identity mismatch");
        repeat (2) begin
            @(posedge clk_core);
            if (!external_slot_inspect_valid)
                $fatal(1, "external inspect valid dropped under backpressure");
        end
        @(negedge clk_core);
        external_slot_inspect_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        external_slot_inspect_ready = 1'b0;
        external_slot_inspect_meta_valid = 1'b1;
        do @(posedge clk_core); while (!external_slot_inspect_meta_ready);
        service_stage = 2;
        @(negedge clk_core);
        external_slot_inspect_meta_valid = 1'b0;
        external_slot_replay_begin_ready = 1'b1;

        do @(posedge clk_core); while (!external_slot_replay_begin_valid);
        service_stage = 3;
        if (external_slot_replay_context_id ||
            external_slot_replay_head_id ||
            external_slot_replay_payload_tag != PAYLOAD_TAG ||
            external_slot_replay_start_word != '0)
            $fatal(1, "external replay begin identity mismatch");
        @(negedge clk_core);
        external_slot_replay_begin_ready = 1'b0;
        external_slot_count_replays = 32'd1;

        for (integer word = 0; word < 4; word = word + 1) begin
            external_slot_replay_word_valid = 1'b1;
            external_slot_replay_word_data = payload_word(word);
            external_slot_replay_word_index = 3'(word);
            external_slot_replay_word_last = word == 3;
            do @(posedge clk_core); while (!external_slot_replay_word_ready);
            @(negedge clk_core);
            external_slot_replay_word_valid = 1'b0;
            external_slot_replay_word_last = 1'b0;
        end
        service_stage = 4;

        do @(posedge clk_core); while (!external_slot_release_valid);
        service_stage = 5;
        if (external_slot_release_context_id ||
            external_slot_release_head_id)
            $fatal(1, "external release identity mismatch");
        repeat (2) begin
            @(posedge clk_core);
            if (!external_slot_release_valid)
                $fatal(1, "external release valid dropped under backpressure");
        end
        @(negedge clk_core);
        external_slot_release_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        external_slot_release_ready = 1'b0;
        external_slot_valid_flat = 1'b0;
        external_slot_count_releases = 32'd1;
        service_stage = 6;
    end

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = GROUP_TAG;
        group_head_count = 2'd1;
        group_first_output_tile = 4'd0;
        group_output_tile_count = 4'd1;
        group_done_ready = 1'b1;
        payload_commit_begin_valid = 1'b1;
        payload_commit_head_id = 1'b0;
        payload_commit_tag = PAYLOAD_TAG;
        payload_commit_mode_is_csr = 1'b1;
        payload_commit_bits = 16'd208;
        payload_commit_word_valid = 1'b1;
        payload_commit_word_data = '0;
        payload_commit_word_last = 1'b0;
        descriptor_fill_begin_valid = 1'b0;
        descriptor_fill_head_id = 1'b0;
        descriptor_fill_tag = '0;
        descriptor_fill_term_count = '0;
        descriptor_fill_format = '0;
        descriptor_fill_entry_valid = 1'b0;
        descriptor_fill_gate_code = '0;
        descriptor_fill_lane_id = '0;
        descriptor_fill_destination_count = '0;
        descriptor_fill_entry_last = 1'b0;

        repeat (6) @(posedge clk_core);
        rst_core = 1'b0;
        @(posedge clk_core);
        if (payload_commit_begin_ready || payload_commit_word_ready)
            $fatal(1, "payload commit ready must be zero in external mode");
        payload_commit_begin_valid = 1'b0;
        payload_commit_word_valid = 1'b0;

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;

        do @(posedge clk_core); while (!group_done_valid);
        if (group_done_tag != GROUP_TAG || group_done_error || protocol_error ||
            external_slot_reset_pulse ||
            count_slot_replays != 1 || count_slot_releases != 1 ||
            count_projection_heads != 1 || final_count != TOKENS)
            $fatal(1, "external slot completion mismatch tag=%h error=%b protocol=%b replay=%0d release=%0d heads=%0d finals=%0d",
                   group_done_tag, group_done_error, protocol_error,
                   count_slot_replays, count_slot_releases,
                   count_projection_heads, final_count);
        $display("PASS: external slot bridge inspect/replay/release handshakes");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk_core);
        $fatal(1, "external slot bridge timeout stage=%0d rst=%b/%b group=%b/%b issue=%b/%b tile=%b/%b plan=%b/%b inspect=%b/%b meta=%b/%b replay=%b/%b word=%b/%b release=%b/%b done=%b protocol=%b counts=%0d/%0d/%0d/%0d/%0d/%0d/%0d/%0d",
               service_stage, rst_core, dut.fabric_rst_core,
               group_valid, group_ready,
               dut.head_issue_valid, dut.head_issue_ready,
               dut.tile_start_valid, dut.tile_start_ready,
               dut.slot_commit_pulse, dut.slot_reserve_ready,
               external_slot_inspect_valid, external_slot_inspect_ready,
               external_slot_inspect_meta_valid,
               external_slot_inspect_meta_ready,
               external_slot_replay_begin_valid,
               external_slot_replay_begin_ready,
               external_slot_replay_word_valid,
               external_slot_replay_word_ready,
               external_slot_release_valid, external_slot_release_ready,
               group_done_valid, protocol_error,
               count_groups, count_head_issues, count_control_requests,
               count_control_commits, count_control_rejects,
               count_control_sessions, count_projection_heads, final_count);
    end
endmodule

`default_nettype wire
