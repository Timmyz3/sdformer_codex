`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_single_context_execution_scale162_trace #(
    parameter int HEADS = 24
);
    localparam int TOKENS = 162;
    localparam int OUT_TILE = 32;
    localparam int BANKS = 2;
    localparam int ACC_W = 32;
    localparam int WORDS_PER_HEAD = 104;
    localparam int OUTPUT_TILES = HEADS;
    localparam int DIM = HEADS * 32;
    localparam int HEAD_ID_W = (HEADS <= 1) ? 1 : $clog2(HEADS);
    localparam int EXPECTED_REPLAYS = HEADS * OUTPUT_TILES;
`ifdef GATESTACK_NO_RESIDENCY
    localparam int ENABLE_RESIDENCY = 0;
`else
    localparam int ENABLE_RESIDENCY = 1;
`endif
`ifdef GATESTACK_ADAPTIVE_CSR
    localparam int CSR_FORMAT_FADC24 = 2;
`elsif GATESTACK_FADC24
    localparam int CSR_FORMAT_FADC24 = 1;
`else
    localparam int CSR_FORMAT_FADC24 = 0;
`endif
    localparam int EXPECTED_FINALS = TOKENS * OUTPUT_TILES;
    localparam logic [31:0] PAYLOAD_TAG_BASE = 32'h6800_0000;
    localparam logic [31:0] GROUP_TAG = 32'h7800_0000;

    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [31:0] group_tag;
    logic [5:0] group_head_count;
    logic [7:0] group_first_output_tile, group_output_tile_count;
    logic group_done_valid, group_done_ready;
    logic [31:0] group_done_tag;
    logic group_done_error;
    logic payload_commit_begin_valid, payload_commit_begin_ready;
    logic [HEAD_ID_W-1:0] payload_commit_head_id;
    logic [31:0] payload_commit_tag;
    logic payload_commit_mode_is_csr;
    logic [15:0] payload_commit_bits;
    logic payload_commit_word_valid, payload_commit_word_ready;
    logic [63:0] payload_commit_word_data;
    logic payload_commit_word_last;
    /* verilator lint_off UNUSEDSIGNAL */
    logic external_slot_inspect_valid, external_slot_inspect_ready;
    logic external_slot_inspect_context_id;
    logic [HEAD_ID_W-1:0] external_slot_inspect_head_id;
    logic external_slot_inspect_meta_valid;
    logic external_slot_inspect_meta_ready, external_slot_inspect_exists;
    logic [31:0] external_slot_inspect_tag;
    logic external_slot_inspect_mode_is_csr;
    logic [1:0] external_slot_inspect_format;
    logic [15:0] external_slot_inspect_payload_bits;
    logic [15:0] external_slot_inspect_word_count;
    logic external_slot_replay_begin_valid;
    logic external_slot_replay_begin_ready;
    logic external_slot_replay_context_id;
    logic [HEAD_ID_W-1:0] external_slot_replay_head_id;
    logic [31:0] external_slot_replay_payload_tag;
    logic [6:0] external_slot_replay_start_word;
    logic external_slot_replay_word_valid;
    logic external_slot_replay_word_ready;
    logic [63:0] external_slot_replay_word_data;
    logic [6:0] external_slot_replay_word_index;
    logic external_slot_replay_word_last;
    logic [31:0] external_slot_replay_tag;
    logic external_slot_replay_mode_is_csr;
    logic [1:0] external_slot_replay_format;
    logic [15:0] external_slot_replay_payload_bits;
    logic external_slot_release_valid, external_slot_release_ready;
    logic external_slot_release_context_id;
    logic [HEAD_ID_W-1:0] external_slot_release_head_id;
    logic [HEADS-1:0] external_slot_valid_flat;
    logic external_slot_protocol_error;
    logic [31:0] external_slot_count_replays;
    logic [31:0] external_slot_count_releases;
    logic external_slot_reset_pulse;
    /* verilator lint_on UNUSEDSIGNAL */
    logic descriptor_fill_begin_valid, descriptor_fill_begin_ready;
    logic [HEAD_ID_W-1:0] descriptor_fill_head_id;
    logic [31:0] descriptor_fill_tag;
    logic [7:0] descriptor_fill_term_count;
    logic [1:0] descriptor_fill_format;
    logic descriptor_fill_begin_cacheable;
    logic descriptor_fill_entry_valid, descriptor_fill_entry_ready;
    logic [8:0] descriptor_fill_gate_code;
    logic [4:0] descriptor_fill_lane_id;
    logic [7:0] descriptor_fill_destination_count;
    logic descriptor_fill_entry_last;
    logic weight_req_valid, weight_req_ready;
    logic [31:0] weight_req_tag;
    logic [9:0] weight_req_input_channel;
    logic [7:0] weight_req_output_tile;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [31:0] weight_rsp_tag;
    logic [9:0] weight_rsp_input_channel;
    logic [7:0] weight_rsp_output_tile;
    logic [(OUT_TILE*8)-1:0] weight_rsp_weights;
    logic bias_req_valid, bias_req_ready;
    logic [31:0] bias_req_tag, bias_rsp_tag;
    logic [7:0] bias_req_output_tile;
    logic [7:0] bias_req_token_id, bias_rsp_token_id;
    logic bias_req_allow, bias_rsp_valid, bias_rsp_ready;
    logic [(OUT_TILE*ACC_W)-1:0] bias_rsp_values, bias_lookup_values;
    logic [BANKS-1:0] final_valid, final_ready;
    logic [(BANKS*8)-1:0] final_token_ids;
    logic [31:0] final_tag;
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

    logic [63:0] payload_words [0:(HEADS*WORDS_PER_HEAD)-1];
    logic [15:0] payload_bits_vector [0:HEADS-1];
    logic payload_modes [0:HEADS-1];
    logic [7:0] payload_word_counts [0:HEADS-1];
    logic [7:0] term_counts [0:HEADS-1];
    logic [12:0] event_counts [0:HEADS-1];
    logic [31:0] expected_gate_sum [0:TOKENS-1];
    logic [7:0] projection_weights [0:(DIM*DIM)-1];
    logic [31:0] projection_bias_acc [0:DIM-1];
    logic [31:0] expected_output_acc32 [0:(TOKENS*DIM)-1];
    integer cycle_count, final_count, mismatch_count;
    integer tile_final_count [0:OUTPUT_TILES-1];
    integer group_start_cycle, group_end_cycle;
    integer expected_slot_replays, expected_cache_hits;
    integer expected_cache_releases, expected_projection_terms;
    bit dump_enabled;
    bit real_trace_enabled;
    string vector_dir;

    gatestack_single_context_execution_top #(
        .TOKENS(TOKENS), .HEADS(HEADS), .HEAD_BITS(6642),
        .MAX_TERMS(128), .RESIDENT_TERMS(80),
        .ENABLE_RESIDENCY(ENABLE_RESIDENCY),
        .CSR_FORMAT_FADC24(CSR_FORMAT_FADC24), .OUT_TILE(OUT_TILE),
        .BANKS(BANKS), .SEGMENT_TOKENS(18), .TAG_W(32),
        .INPUT_CH_W(10), .OUTPUT_TILE_W(8),
        .OUTPUT_TILE_COUNT_W(8), .HEAD_COUNT_W(6),
        .WORD_INDEX_W(7), .RES_TERM_IDX_W(7),
        .ABORT_TIMEOUT_CYCLES(2000000)
    ) dut (.*);

    gatestack_bias_sram_model #(
        .TAG_W(32), .OUTPUT_TILE_W(8), .TOKEN_ID_W(8),
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

    task automatic commit_payload(input int head);
        int words;
        begin
            words = 32'(payload_word_counts[head]);
            @(negedge clk_core);
            payload_commit_head_id = HEAD_ID_W'(head);
            payload_commit_tag = PAYLOAD_TAG_BASE + 32'(head);
            payload_commit_mode_is_csr = payload_modes[head];
            payload_commit_bits = payload_bits_vector[head];
            payload_commit_begin_valid = 1'b1;
            do @(posedge clk_core); while (!payload_commit_begin_ready);
            @(negedge clk_core);
            payload_commit_begin_valid = 1'b0;
            for (int word = 0; word < words; word = word + 1) begin
                payload_commit_word_data = payload_words[head*WORDS_PER_HEAD + word];
                payload_commit_word_last = word == words - 1;
                payload_commit_word_valid = 1'b1;
                do @(posedge clk_core); while (!payload_commit_word_ready);
                @(negedge clk_core);
                payload_commit_word_valid = 1'b0;
                payload_commit_word_last = 1'b0;
            end
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
            if (32'(weight_req_input_channel) >= HEADS*32 ||
                32'(weight_req_output_tile) >= OUTPUT_TILES)
                $fatal(1, "weight request越界 channel=%0d tile=%0d",
                       weight_req_input_channel, weight_req_output_tile);
            weight_rsp_tag = weight_req_tag;
            weight_rsp_input_channel = weight_req_input_channel;
            weight_rsp_output_tile = weight_req_output_tile;
            for (int lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                if (real_trace_enabled)
                    weight_rsp_weights[(lane*8) +: 8] = projection_weights[
                        ((32'(weight_req_output_tile)*OUT_TILE + lane)*DIM) +
                        32'(weight_req_input_channel)
                    ];
                else
                    weight_rsp_weights[(lane*8) +: 8] = 8'sd1;
            end
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            if ((cycle_count % 7) == 3) @(posedge clk_core);
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    end

    always_comb begin
        bias_req_allow = (cycle_count % 5) != 1;
        for (int lane = 0; lane < OUT_TILE; lane = lane + 1) begin
            if (real_trace_enabled)
                bias_lookup_values[(lane*ACC_W) +: ACC_W] =
                    projection_bias_acc[
                        (32'(bias_req_output_tile)*OUT_TILE) + lane
                    ];
            else
                bias_lookup_values[(lane*ACC_W) +: ACC_W] =
                    32'(32'(bias_req_token_id) + lane);
        end
        final_ready[0] = (cycle_count % 7) != 2;
        final_ready[1] = (cycle_count % 5) != 3;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            final_count <= 0;
            mismatch_count <= 0;
            for (int tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
                tile_final_count[tile] <= 0;
        end else begin
            integer fires;
            integer tile_fires [0:OUTPUT_TILES-1];
            fires = 0;
            for (int tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
                tile_fires[tile] = 0;
            cycle_count <= cycle_count + 1;
            for (int bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    integer token_value;
                    integer tile_value;
                    token_value = 32'(final_token_ids[(bank*8) +: 8]);
                    tile_value = 32'(final_tag - GROUP_TAG);
                    if (token_value >= TOKENS || tile_value >= OUTPUT_TILES)
                        $fatal(1, "final identity越界 tag=%h token=%0d", final_tag, token_value);
                    for (int lane = 0; lane < OUT_TILE; lane = lane + 1) begin
                        integer signed expected_value;
                        integer signed actual_value;
                        if (real_trace_enabled)
                            expected_value = $signed(expected_output_acc32[
                                token_value*DIM + tile_value*OUT_TILE + lane
                            ]);
                        else
                            expected_value = 32'(expected_gate_sum[token_value]) + token_value + lane;
                        actual_value = $signed(final_values[
                            (bank*OUT_TILE*ACC_W) + (lane*ACC_W) +: ACC_W]);
                        if (actual_value != expected_value) begin
                            mismatch_count <= mismatch_count + 1;
                            $fatal(1, "final数值错误 tile=%0d token=%0d lane=%0d got=%0d expected=%0d",
                                   tile_value, token_value, lane, actual_value, expected_value);
                        end
                    end
                    tile_fires[tile_value] = tile_fires[tile_value] + 1;
                    fires = fires + 1;
                end
            end
            for (int tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
                if (tile_fires[tile] != 0)
                    tile_final_count[tile] <= tile_final_count[tile] + tile_fires[tile];
            final_count <= final_count + fires;
        end
    end

    initial begin
        real_trace_enabled = $value$plusargs("vector_dir=%s", vector_dir);
        if (real_trace_enabled) begin
            $readmemh({vector_dir, "/payload_words.memh"}, payload_words);
            $readmemh({vector_dir, "/payload_bits.memh"}, payload_bits_vector);
            $readmemh({vector_dir, "/payload_modes.memh"}, payload_modes);
            $readmemh({vector_dir, "/payload_word_counts.memh"}, payload_word_counts);
            $readmemh({vector_dir, "/term_counts.memh"}, term_counts);
            $readmemh({vector_dir, "/event_counts.memh"}, event_counts);
            $readmemh({vector_dir, "/projection_weights_int8.memh"}, projection_weights);
            $readmemh({vector_dir, "/projection_bias_acc.memh"}, projection_bias_acc);
            $readmemh({vector_dir, "/expected_output_acc32.memh"}, expected_output_acc32);
        end else begin
`ifdef GATESTACK_RAW_ONLY
            vector_dir = "tb_hitflow/vectors/gatestack_h67_stage3_sample0_b0_rawonly";
`else
            vector_dir = "tb_hitflow/vectors/gatestack_h67_stage3_sample0_b0";
`endif
            $readmemh({vector_dir, "/payload_words.memh"}, payload_words);
            $readmemh({vector_dir, "/payload_bits.memh"}, payload_bits_vector);
            $readmemh({vector_dir, "/payload_modes.memh"}, payload_modes);
            $readmemh({vector_dir, "/payload_word_counts.memh"}, payload_word_counts);
            $readmemh({vector_dir, "/term_counts.memh"}, term_counts);
            $readmemh({vector_dir, "/event_counts.memh"}, event_counts);
            $readmemh({vector_dir, "/expected_gate_sum.memh"}, expected_gate_sum);
        end
        begin
            integer ipd_heads;
            integer fadc_heads;
            integer raw_heads;
            integer nonempty_ipd_heads;
            ipd_heads = 0;
            fadc_heads = 0;
            raw_heads = 0;
            nonempty_ipd_heads = 0;
            expected_projection_terms = 0;
            for (int head = 0; head < HEADS; head = head + 1) begin
                if (payload_modes[head]) begin
                    if (payload_words[head*WORDS_PER_HEAD][15:0] ==
                        16'h4641) begin
                        fadc_heads = fadc_heads + 1;
                    end else begin
                        ipd_heads = ipd_heads + 1;
                        if (term_counts[head] != 0)
                            nonempty_ipd_heads = nonempty_ipd_heads + 1;
                    end
                    expected_projection_terms = expected_projection_terms +
                                                32'(term_counts[head]);
                end else begin
                    raw_heads = raw_heads + 1;
                    expected_projection_terms = expected_projection_terms +
                                                32'(event_counts[head]);
                end
            end
            expected_projection_terms = expected_projection_terms * OUTPUT_TILES;
            if (ENABLE_RESIDENCY == 0) begin
                expected_slot_replays = EXPECTED_REPLAYS;
                expected_cache_hits = 0;
                expected_cache_releases = 0;
            end else begin
                expected_slot_replays = HEADS + (OUTPUT_TILES - 1) *
                    (nonempty_ipd_heads + fadc_heads + raw_heads);
                expected_cache_hits = ipd_heads * (OUTPUT_TILES - 1);
                expected_cache_releases = ipd_heads;
            end
        end
        dump_enabled = $test$plusargs("dump_vcd");
        if (dump_enabled) begin
            $dumpfile("build_hitflow/gatestack_single_context_execution/scale162_trace.vcd");
            $dumpvars(0, dut);
            $dumpoff;
        end

        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = GROUP_TAG;
        group_head_count = 6'(HEADS);
        group_first_output_tile = 8'd0;
        group_output_tile_count = 8'(OUTPUT_TILES);
        group_done_ready = 1'b0;
        payload_commit_begin_valid = 1'b0;
        payload_commit_head_id = '0;
        payload_commit_tag = '0;
        payload_commit_mode_is_csr = 1'b0;
        payload_commit_bits = '0;
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
        descriptor_fill_head_id = '0;
        descriptor_fill_tag = '0;
        descriptor_fill_term_count = '0;
        descriptor_fill_format = 2'd1;
        descriptor_fill_entry_valid = 1'b0;
        descriptor_fill_gate_code = '0;
        descriptor_fill_lane_id = '0;
        descriptor_fill_destination_count = '0;
        descriptor_fill_entry_last = 1'b0;
        group_start_cycle = 0;
        group_end_cycle = 0;
        repeat (8) @(posedge clk_core);
        rst_core = 1'b0;

        for (int head = 0; head < HEADS; head = head + 1)
            commit_payload(head);

        @(negedge clk_core);
        if (dump_enabled) $dumpon;
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        group_start_cycle = cycle_count;
        @(negedge clk_core);
        group_valid = 1'b0;

        wait (group_done_valid);
        group_end_cycle = cycle_count;
        if (dump_enabled) $dumpoff;
        $display("TRACE CHECK tag=%h/%h error=%b protocol=%b groups=%0d tiles=%0d issues=%0d ctrl=%0d/%0d/%0d/%0d slot=%0d/%0d cache=%0d/%0d proj=%0d/%0d bias=%0d/%b abort=%0d/%0d/%0d final=%0d mismatch=%0d extfill=%b/%b/%b",
                 group_done_tag, GROUP_TAG, group_done_error, protocol_error,
                 count_groups, count_tile_starts, count_head_issues,
                 count_control_requests, count_control_commits,
                 count_control_rejects, count_control_sessions,
                 count_slot_replays, count_slot_releases,
                 count_cache_hits, count_cache_releases,
                 count_projection_heads, count_projection_terms,
                 count_bias_commits, bias_req_valid, count_context_resets,
                 count_error_aborts, count_timeout_aborts,
                 final_count, mismatch_count,
                 descriptor_fill_begin_ready,
                 descriptor_fill_begin_cacheable,
                 descriptor_fill_entry_ready);
        if (group_done_tag != GROUP_TAG || group_done_error || protocol_error ||
            count_groups != 1 || count_tile_starts != OUTPUT_TILES ||
            count_head_issues != EXPECTED_REPLAYS ||
            count_control_requests != EXPECTED_REPLAYS ||
            count_control_commits != EXPECTED_REPLAYS ||
            count_control_rejects != 0 ||
            count_control_sessions != EXPECTED_REPLAYS ||
            count_slot_replays != expected_slot_replays ||
            count_slot_releases != HEADS ||
            count_cache_hits != expected_cache_hits ||
            count_cache_releases != expected_cache_releases ||
            count_projection_heads != EXPECTED_REPLAYS ||
            count_projection_terms != expected_projection_terms ||
            count_bias_commits != EXPECTED_FINALS ||
            count_context_resets != 0 || count_error_aborts != 0 ||
            count_timeout_aborts != 0 || final_count != EXPECTED_FINALS ||
            mismatch_count != 0)
            $fatal(1, "scale162 trace计数错误 tiles=%0d issues=%0d sessions=%0d hits=%0d releases=%0d/%0d heads=%0d terms=%0d bias=%0d final=%0d protocol=%b",
                   count_tile_starts, count_head_issues, count_control_sessions,
                   count_cache_hits, count_slot_releases, count_cache_releases,
                   count_projection_heads, count_projection_terms,
                   count_bias_commits, final_count, protocol_error);
        for (int tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
            if (tile_final_count[tile] != TOKENS)
                $fatal(1, "tile final数量错误 tile=%0d count=%0d", tile, tile_final_count[tile]);
        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        $display("PASS: H67 stage3 trace-shaped full top heads=%0d tiles=%0d replays=%0d cache_hits=%0d terms=%0d finals=%0d group_cycles=%0d",
                 HEADS, OUTPUT_TILES, EXPECTED_REPLAYS, count_cache_hits,
                 count_projection_terms, final_count,
                 group_end_cycle - group_start_cycle);
        $finish;
    end

    initial begin
        repeat (3000000) @(posedge clk_core);
        $fatal(1, "scale162 trace full top timeout");
    end
endmodule

`default_nettype wire
