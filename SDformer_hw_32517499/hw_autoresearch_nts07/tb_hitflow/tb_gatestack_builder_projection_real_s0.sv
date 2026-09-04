`timescale 1ns/1ps
`include "tb_hitflow/gatestack_bias_sram_model.sv"
`default_nettype none

module tb_gatestack_builder_projection_real_s0 #(
    parameter int BUILDER_C1_ENABLE = 0,
    parameter int STAGE_ID = 0,
    parameter int OUT_TILE = 32,
    parameter bit BIAS_STATIONARY_ENABLE = 1'b0
);
    localparam int TOKENS = 162;
    localparam int HEADS = 3 << STAGE_ID;
    localparam int HEAD_OFFSET = 3 * ((1 << STAGE_ID) - 1);
    localparam int HEAD_ID_W = (HEADS <= 1) ? 1 : $clog2(HEADS);
    localparam int OUTPUT_TILES = (DIM + OUT_TILE - 1) / OUT_TILE;
    localparam int BANKS = 2;
    localparam int LANES = 32;
    localparam int DIM = HEADS * LANES;
    localparam int ACC_W = 32;
    localparam int EXPECTED_FINALS = TOKENS * OUTPUT_TILES;
    localparam int EXPECTED_ELEMENTS = TOKENS * DIM;
    localparam int EXPECTED_REPLAYS = HEADS * OUTPUT_TILES;
    localparam logic [31:0] GROUP_TAG = 32'h7800_0000;
    localparam string BUILDER_VECTOR_DIR =
        "tb_hitflow/vectors/gatestack_all45_builder_20260720";

    logic clk_core, rst_core;
    logic head_begin_valid, head_begin_ready;
    logic [HEAD_ID_W-1:0] head_id;
    logic [31:0] head_tag;
    logic token_valid, token_ready;
    logic [7:0] token_id;
    logic [8:0] token_gate_code;
    logic [31:0] token_k_bits;
    logic token_last;
    logic group_valid, group_ready;
    logic [31:0] group_tag;
    logic [7:0] group_first_output_tile, group_output_tile_count;
    logic group_done_valid, group_done_ready;
    logic [31:0] group_done_tag;
    logic group_done_error;
    logic batch_abort_valid, batch_abort_ready;
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
    logic builder_done_pulse;
    logic [31:0] builder_done_tag;
    logic [1:0] builder_done_format;
    logic builder_done_error;
    logic [7:0] builder_done_word_count;
    logic [2:0] builder_selected_reason;
    logic [15:0] builder_selected_payload_bits;
    logic [5:0] batch_accepted_heads, batch_completed_heads;
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

    logic [40:0] raw_records [0:7289];
    logic [31:0] tags_vector [0:44];
    logic [7:0] heads_vector [0:44];
    logic [1:0] formats_vector [0:44];
    logic [2:0] reasons_vector [0:44];
    logic [15:0] payload_bits_vector [0:44];
    logic [7:0] word_counts_vector [0:44];
    logic [7:0] term_counts_vector [0:44];
    logic [12:0] event_counts_vector [0:44];
    logic [7:0] projection_weights [0:(DIM*DIM)-1];
    logic [15:0] projection_weight_scale_exp2 [0:DIM-1];
    logic [31:0] projection_bias_acc [0:DIM-1];
    logic [31:0] expected_output_acc32 [0:(TOKENS*DIM)-1];

    integer cycle_count;
    integer first_head_cycle, builders_done_cycle, group_done_cycle;
    integer builder_done_count, final_count, compared_elements;
    integer mismatch_count;
    integer bias_request_handshakes, bias_response_handshakes;
    integer expected_builder_terms, expected_builder_destinations;
    integer expected_builder_words, expected_projection_terms;
    integer expected_event_sum;
    integer tile_final_count [0:OUTPUT_TILES-1];
    integer tile_bias_request_count [0:OUTPUT_TILES-1];
    logic signed [63:0] output_checksum;

    gatestack_builder_projection_single_context_top #(
        .BUILDER_C1_ENABLE(BUILDER_C1_ENABLE), .TOKENS(TOKENS),
        .HEADS(HEADS), .SLOT_WORDS(104), .MAX_TERMS(128),
        .RESIDENT_TERMS(80), .ENABLE_RESIDENCY(1),
        .CSR_FORMAT_FADC24(2), .OUT_TILE(OUT_TILE), .BANKS(BANKS),
        .BIAS_STATIONARY_ENABLE(BIAS_STATIONARY_ENABLE),
        .SEGMENT_TOKENS(18), .TAG_W(32), .INPUT_CH_W(10),
        .OUTPUT_TILE_W(8), .OUTPUT_TILE_COUNT_W(8), .HEAD_COUNT_W(6),
        .WORD_INDEX_W(7), .RES_TERM_IDX_W(7), .TOKEN_ID_W(8),
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

    task automatic send_head(input integer head_index);
        begin
            @(negedge clk_core);
            head_begin_valid = 1'b1;
            head_id = HEAD_ID_W'(heads_vector[HEAD_OFFSET + head_index]);
            head_tag = tags_vector[HEAD_OFFSET + head_index];
            do @(posedge clk_core); while (!head_begin_ready);
            if (head_index == 0)
                first_head_cycle = cycle_count;
            @(negedge clk_core);
            head_begin_valid = 1'b0;
            for (integer token = 0; token < TOKENS; token = token + 1) begin
                token_valid = 1'b1;
                token_id = 8'(token);
                token_k_bits = raw_records[
                    (HEAD_OFFSET + head_index)*TOKENS + token][31:0];
                token_gate_code =
                    raw_records[(HEAD_OFFSET + head_index)*TOKENS + token][40:32];
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
            if (32'(weight_req_input_channel) >= DIM ||
                32'(weight_req_output_tile) >= OUTPUT_TILES ||
                weight_req_tag != GROUP_TAG +
                                  32'(weight_req_output_tile))
                $fatal(1,
                    "weight request identity mismatch tag=%h channel=%0d tile=%0d",
                    weight_req_tag, weight_req_input_channel,
                    weight_req_output_tile);
            weight_rsp_tag = weight_req_tag;
            weight_rsp_input_channel = weight_req_input_channel;
            weight_rsp_output_tile = weight_req_output_tile;
            for (integer lane = 0; lane < OUT_TILE; lane = lane + 1)
                if (32'(weight_req_output_tile)*OUT_TILE + lane < DIM)
                    weight_rsp_weights[(lane*8) +: 8] = projection_weights[
                        ((32'(weight_req_output_tile)*OUT_TILE + lane)*DIM) +
                        32'(weight_req_input_channel)
                    ];
                else
                    weight_rsp_weights[(lane*8) +: 8] = '0;
            @(negedge clk_core);
            weight_req_ready = 1'b0;
            if ((cycle_count % 7) == 3)
                @(posedge clk_core);
            @(negedge clk_core);
            weight_rsp_valid = 1'b1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            @(negedge clk_core);
            weight_rsp_valid = 1'b0;
        end
    end

    always_comb begin
        bias_req_allow = (cycle_count % 5) != 1;
        for (integer lane = 0; lane < OUT_TILE; lane = lane + 1)
            if (32'(bias_req_output_tile)*OUT_TILE + lane < DIM)
                bias_lookup_values[(lane*ACC_W) +: ACC_W] =
                    projection_bias_acc[
                        (32'(bias_req_output_tile)*OUT_TILE) + lane];
            else
                bias_lookup_values[(lane*ACC_W) +: ACC_W] = '0;
        final_ready[0] = (cycle_count % 7) != 2;
        final_ready[1] = (cycle_count % 5) != 3;
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            cycle_count <= 0;
            builder_done_count <= 0;
            builders_done_cycle <= -1;
            final_count <= 0;
            compared_elements <= 0;
            mismatch_count <= 0;
            bias_request_handshakes <= 0;
            bias_response_handshakes <= 0;
            output_checksum <= '0;
            for (integer tile = 0; tile < OUTPUT_TILES; tile = tile + 1) begin
                tile_final_count[tile] <= 0;
                tile_bias_request_count[tile] <= 0;
            end
        end else begin
            integer fires;
            integer compared_delta;
            integer mismatch_delta;
            integer tile_fires [0:OUTPUT_TILES-1];
            logic signed [63:0] checksum_delta;
            cycle_count <= cycle_count + 1;
            fires = 0;
            compared_delta = 0;
            mismatch_delta = 0;
            checksum_delta = '0;
            for (integer tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
                tile_fires[tile] = 0;

            if (dut.execution_group_valid &&
                (batch_accepted_heads != 6'(HEADS) ||
                 batch_completed_heads != 6'(HEADS) ||
                 slot_valid_flat != {HEADS{1'b1}} || protocol_error))
                $fatal(1, "execution launch before complete clean slot batch");

            if (bias_req_valid && bias_req_ready &&
                32'(bias_req_token_id) >= TOKENS)
                $fatal(1, "bias token id out of range token=%0d",
                       bias_req_token_id);
            if (bias_req_valid && bias_req_ready) begin
                if (32'(bias_req_output_tile) >= OUTPUT_TILES)
                    $fatal(1, "bias output tile out of range tile=%0d",
                           bias_req_output_tile);
                if (BIAS_STATIONARY_ENABLE && bias_req_token_id != '0)
                    $fatal(1, "BSF bias request token must be zero");
                bias_request_handshakes <= bias_request_handshakes + 1;
                tile_bias_request_count[32'(bias_req_output_tile)] <=
                    tile_bias_request_count[32'(bias_req_output_tile)] + 1;
            end
            if (bias_rsp_valid && bias_rsp_ready)
                bias_response_handshakes <= bias_response_handshakes + 1;

            if (builder_done_pulse) begin
                if (builder_done_count >= HEADS || builder_done_error ||
                    builder_done_tag !=
                        tags_vector[HEAD_OFFSET + builder_done_count] ||
                    builder_done_format !=
                        formats_vector[HEAD_OFFSET + builder_done_count] ||
                    builder_done_word_count !=
                        word_counts_vector[HEAD_OFFSET + builder_done_count] ||
                    builder_selected_reason !=
                        reasons_vector[HEAD_OFFSET + builder_done_count] ||
                    builder_selected_payload_bits !=
                        payload_bits_vector[HEAD_OFFSET + builder_done_count])
                    $fatal(1,
                        "builder done mismatch index=%0d tag=%h format=%0d words=%0d bits=%0d error=%b",
                        builder_done_count, builder_done_tag,
                        builder_done_format, builder_done_word_count,
                        builder_selected_payload_bits, builder_done_error);
                if (builder_done_count == HEADS - 1)
                    builders_done_cycle <= cycle_count;
            end

            for (integer bank = 0; bank < BANKS; bank = bank + 1) begin
                if (final_valid[bank] && final_ready[bank]) begin
                    integer token_value;
                    integer tile_value;
                    token_value = 32'(final_token_ids[(bank*8) +: 8]);
                    tile_value = 32'(final_tag - GROUP_TAG);
                    if (token_value >= TOKENS || tile_value >= OUTPUT_TILES)
                        $fatal(1,
                            "final identity out of range tag=%h token=%0d",
                            final_tag, token_value);
                    for (integer lane = 0; lane < OUT_TILE;
                         lane = lane + 1) begin
                        integer signed expected_value;
                        integer signed actual_value;
                        actual_value = $signed(final_values[
                            (bank*OUT_TILE*ACC_W) + (lane*ACC_W) +: ACC_W]);
                        if (tile_value*OUT_TILE + lane < DIM) begin
                            expected_value = $signed(expected_output_acc32[
                                token_value*DIM + tile_value*OUT_TILE + lane]);
                            compared_delta = compared_delta + 1;
                            checksum_delta = checksum_delta + 64'(actual_value);
                            if (actual_value != expected_value)
                                mismatch_delta = mismatch_delta + 1;
                        end else if (actual_value != 0) begin
                            mismatch_delta = mismatch_delta + 1;
                        end
                    end
                    tile_fires[tile_value] = tile_fires[tile_value] + 1;
                    fires = fires + 1;
                end
            end
            builder_done_count <= builder_done_count +
                (builder_done_pulse ? 1 : 0);
            final_count <= final_count + fires;
            compared_elements <= compared_elements + compared_delta;
            mismatch_count <= mismatch_count + mismatch_delta;
            output_checksum <= output_checksum + checksum_delta;
            for (integer tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
                tile_final_count[tile] <=
                    tile_final_count[tile] + tile_fires[tile];
        end
    end

    initial begin
        $readmemh({BUILDER_VECTOR_DIR, "/raw_records.memh"}, raw_records);
        $readmemh({BUILDER_VECTOR_DIR, "/tags.memh"}, tags_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/heads.memh"}, heads_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/formats.memh"}, formats_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/reasons.memh"}, reasons_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/payload_bits.memh"},
                  payload_bits_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/word_counts.memh"},
                  word_counts_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/term_counts.memh"},
                  term_counts_vector);
        $readmemh({BUILDER_VECTOR_DIR, "/event_counts.memh"},
                  event_counts_vector);
        if (STAGE_ID == 0) begin
            $readmemh("tb_hitflow/vectors/real_sample0_s0_b0_capacity/projection_weights_int8.memh", projection_weights);
            $readmemh("tb_hitflow/vectors/real_sample0_s0_b0_capacity/projection_weight_scale_exp2.memh", projection_weight_scale_exp2);
            $readmemh("tb_hitflow/vectors/real_sample0_s0_b0_capacity/projection_bias_acc.memh", projection_bias_acc);
            $readmemh("tb_hitflow/vectors/real_sample0_s0_b0_capacity/expected_output_acc32.memh", expected_output_acc32);
        end else if (STAGE_ID == 1) begin
            $readmemh("tb_hitflow/vectors/real_sample0_s1_b0_capacity/projection_weights_int8.memh", projection_weights);
            $readmemh("tb_hitflow/vectors/real_sample0_s1_b0_capacity/projection_weight_scale_exp2.memh", projection_weight_scale_exp2);
            $readmemh("tb_hitflow/vectors/real_sample0_s1_b0_capacity/projection_bias_acc.memh", projection_bias_acc);
            $readmemh("tb_hitflow/vectors/real_sample0_s1_b0_capacity/expected_output_acc32.memh", expected_output_acc32);
        end else if (STAGE_ID == 2) begin
            $readmemh("tb_hitflow/vectors/real_sample0_s2_b0_capacity/projection_weights_int8.memh", projection_weights);
            $readmemh("tb_hitflow/vectors/real_sample0_s2_b0_capacity/projection_weight_scale_exp2.memh", projection_weight_scale_exp2);
            $readmemh("tb_hitflow/vectors/real_sample0_s2_b0_capacity/projection_bias_acc.memh", projection_bias_acc);
            $readmemh("tb_hitflow/vectors/real_sample0_s2_b0_capacity/expected_output_acc32.memh", expected_output_acc32);
        end else if (STAGE_ID == 3) begin
            $readmemh("tb_hitflow/vectors/real_sample0_s3_b0_capacity/projection_weights_int8.memh", projection_weights);
            $readmemh("tb_hitflow/vectors/real_sample0_s3_b0_capacity/projection_weight_scale_exp2.memh", projection_weight_scale_exp2);
            $readmemh("tb_hitflow/vectors/real_sample0_s3_b0_capacity/projection_bias_acc.memh", projection_bias_acc);
            $readmemh("tb_hitflow/vectors/real_sample0_s3_b0_capacity/expected_output_acc32.memh", expected_output_acc32);
        end else begin
            $fatal(1, "unsupported stage id=%0d", STAGE_ID);
        end

        expected_builder_terms = 0;
        expected_builder_destinations = 0;
        expected_builder_words = 0;
        expected_event_sum = 0;
        for (integer head = 0; head < HEADS; head = head + 1) begin
            if (32'(heads_vector[HEAD_OFFSET + head]) != head)
                $fatal(1, "builder stage head id mismatch stage=%0d index=%0d actual=%0d",
                       STAGE_ID, head, heads_vector[HEAD_OFFSET + head]);
            expected_builder_terms = expected_builder_terms +
                32'(term_counts_vector[HEAD_OFFSET + head]);
            expected_builder_destinations = expected_builder_destinations +
                32'(event_counts_vector[HEAD_OFFSET + head]);
            expected_builder_words = expected_builder_words +
                32'(word_counts_vector[HEAD_OFFSET + head]);
            expected_event_sum = expected_event_sum +
                32'(event_counts_vector[HEAD_OFFSET + head]);
        end
        expected_projection_terms = expected_builder_terms * OUTPUT_TILES;

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
        group_first_output_tile = 8'd0;
        group_output_tile_count = 8'(OUTPUT_TILES);
        group_done_ready = 1'b0;
        batch_abort_valid = 1'b0;
        first_head_cycle = -1;
        group_done_cycle = -1;
        repeat (8) @(posedge clk_core);
        rst_core = 1'b0;

        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;
        for (integer head = 0; head < HEADS; head = head + 1)
            send_head(head);
        wait (group_done_valid);
        group_done_cycle = cycle_count;

        if (group_done_tag != GROUP_TAG || group_done_error ||
            protocol_error || slot_reset_pulse || !batch_abort_ready ||
            builder_done_count != HEADS ||
            count_builder_heads != HEADS || count_builder_raw_heads != 0 ||
            count_builder_terms != expected_builder_terms ||
            count_builder_destinations != expected_builder_destinations ||
            count_builder_committed_heads != HEADS ||
            count_builder_aborted_heads != 0 ||
            count_builder_committed_words != expected_builder_words ||
            count_slot_commit_heads != HEADS ||
            count_slot_replay_heads < HEADS ||
            count_slot_replay_heads > EXPECTED_REPLAYS ||
            count_slot_release_heads != HEADS || slot_valid_flat != '0 ||
            count_payload_copy_words != 0 || count_groups != 1 ||
            count_tile_starts != OUTPUT_TILES ||
            count_head_issues != EXPECTED_REPLAYS ||
            count_control_requests != EXPECTED_REPLAYS ||
            count_control_commits != EXPECTED_REPLAYS ||
            count_control_rejects != 0 ||
            count_control_sessions != EXPECTED_REPLAYS ||
            count_projection_heads != EXPECTED_REPLAYS ||
            count_projection_terms != expected_projection_terms ||
            count_bias_commits != EXPECTED_FINALS ||
            bias_request_handshakes !=
                (BIAS_STATIONARY_ENABLE ? OUTPUT_TILES : EXPECTED_FINALS) ||
            bias_response_handshakes !=
                (BIAS_STATIONARY_ENABLE ? OUTPUT_TILES : EXPECTED_FINALS) ||
            count_context_resets != 0 || count_error_aborts != 0 ||
            count_timeout_aborts != 0 || final_count != EXPECTED_FINALS ||
            compared_elements != EXPECTED_ELEMENTS || mismatch_count != 0)
            $fatal(1,
                "real s0 counters mismatch mode=%0d builder=%0d/%0d terms=%0d dest=%0d words=%0d slot=%0d/%0d/%0d valid=%b proj=%0d/%0d bias=%0d final=%0d elements=%0d mismatch=%0d errors=%b",
                BUILDER_C1_ENABLE, count_builder_heads,
                count_builder_committed_heads, count_builder_terms,
                count_builder_destinations, count_builder_committed_words,
                count_slot_commit_heads, count_slot_replay_heads,
                count_slot_release_heads, slot_valid_flat,
                count_projection_heads, count_projection_terms,
                count_bias_commits, final_count, compared_elements,
                mismatch_count, protocol_error);

        for (integer tile = 0; tile < OUTPUT_TILES; tile = tile + 1)
            if (tile_final_count[tile] != TOKENS ||
                tile_bias_request_count[tile] !=
                    (BIAS_STATIONARY_ENABLE ? 1 : TOKENS))
                $fatal(1, "tile final count mismatch tile=%0d count=%0d",
                       tile, tile_final_count[tile]);

        @(negedge clk_core);
        group_done_ready = 1'b1;
        @(posedge clk_core);
        $display(
            "RESULT stage=S%0d mode=C%0d out_tile=%0d bsf=%0d status=PASS total_cycles=%0d build_cycles=%0d projection_cycles=%0d compared=%0d mismatches=%0d checksum=%0d replay=%0d release=%0d projection_heads=%0d projection_terms=%0d bias=%0d bias_req_hs=%0d bias_rsp_hs=%0d slot_commits=%0d payload_copy=%0d errors=0 scan=%0d stalls=%0d blocked=%0d overlap=%0d order_wait=%0d scale0=%h event_sum=%0d",
            STAGE_ID, BUILDER_C1_ENABLE, OUT_TILE, BIAS_STATIONARY_ENABLE,
            group_done_cycle - first_head_cycle,
            builders_done_cycle - first_head_cycle,
            group_done_cycle - builders_done_cycle, compared_elements,
            mismatch_count, output_checksum, count_slot_replay_heads,
            count_slot_release_heads, count_projection_heads,
            count_projection_terms, count_bias_commits,
            bias_request_handshakes, bias_response_handshakes,
            count_slot_commit_heads, count_payload_copy_words,
            count_builder_scan_cycles, count_builder_output_stalls,
            count_builder_capture_blocked_cycles,
            count_builder_overlap_cycles, count_builder_order_wait_cycles,
            projection_weight_scale_exp2[0], expected_event_sum);
        $finish;
    end

    initial begin
        repeat (3000000) @(posedge clk_core);
        $fatal(1, "real stage builder projection timeout stage=%0d", STAGE_ID);
    end
endmodule

`default_nettype wire
