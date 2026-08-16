`timescale 1ns/1ps
`default_nettype none

module tb_h67_zkqi_row_miter #(
    parameter bit CANDIDATE_BUNDLE_SKIP_ENABLE = 1'b1,
    parameter bit BASELINE_ZK_BYPASS_ENABLE = 1'b0,
    parameter int BASELINE_ACTIVE_SCORE_RESIDUAL_W = 0,
    parameter int CANDIDATE_ACTIVE_SCORE_RESIDUAL_W = 0
);
    localparam int HEAD_DIM = 32;
    localparam int PAIRS = 225;
    localparam int TOKENS = 450;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(TOKENS + 1);
    localparam int ACTIVE_FIFO_DEPTH = 1;
    localparam int FIFO_OCC_W = $clog2(ACTIVE_FIFO_DEPTH + 1);

    logic clk;
    logic rst;
    logic row_load_start;
    logic row_load_valid;
    logic row_load_ready_b;
    logic row_load_ready_z;
    logic [PAIR_ID_W-1:0] row_load_pair_id;
    logic [2*HEAD_DIM-1:0] row_load_q_pair;
    logic [2*HEAD_DIM-1:0] row_load_k_pair;
    logic row_loaded_b;
    logic row_loaded_z;
    logic window_start;
    logic seal_ready_b;
    logic seal_ready_z;
    logic window_done_b;
    logic window_done_z;
    logic descriptor_issue_enable;
    logic out_ready;

    logic out_valid_b;
    logic out_last_b;
    logic [TOKEN_W-1:0] out_token_b;
    logic [HEAD_DIM-1:0] out_k_b;
    logic [8:0] out_gate_b;
    logic [7:0] out_threshold_b;
    logic error_b;
    logic [31:0] score_pairs_b;
    logic [31:0] score_slots_b;
    logic [31:0] original_tokens_b;
    logic [31:0] equal_pairs_b;
    logic [31:0] seeded_tokens_b;
    logic [31:0] active_entries_b;
    logic [31:0] class_tx_b;
    logic [31:0] exp_tx_b;
    logic [31:0] emitted_b;
    logic [31:0] read_tx_b;
    logic [31:0] read_bits_b;
    logic [31:0] preload_cycles_b;
    logic [31:0] cycles_b;
    logic [31:0] score_stall_b;
    logic [31:0] output_stall_b;
    logic [31:0] preclass_b;
    logic [31:0] metadata_bits_b;
    logic [FIFO_OCC_W-1:0] fifo_occ_b;
    logic [FIFO_OCC_W-1:0] fifo_max_b;
    logic [31:0] tare_dense_b;

    logic out_valid_z;
    logic out_last_z;
    logic [TOKEN_W-1:0] out_token_z;
    logic [HEAD_DIM-1:0] out_k_z;
    logic [8:0] out_gate_z;
    logic [7:0] out_threshold_z;
    logic error_z;
    logic [31:0] score_pairs_z;
    logic [31:0] score_slots_z;
    logic [31:0] original_tokens_z;
    logic [31:0] equal_pairs_z;
    logic [31:0] seeded_tokens_z;
    logic [31:0] active_entries_z;
    logic [31:0] class_tx_z;
    logic [31:0] exp_tx_z;
    logic [31:0] emitted_z;
    logic [31:0] read_tx_z;
    logic [31:0] read_bits_z;
    logic [31:0] preload_cycles_z;
    logic [31:0] cycles_z;
    logic [31:0] score_stall_z;
    logic [31:0] output_stall_z;
    logic [31:0] preclass_z;
    logic [31:0] metadata_bits_z;
    logic [FIFO_OCC_W-1:0] fifo_occ_z;
    logic [FIFO_OCC_W-1:0] fifo_max_z;
    logic [31:0] tare_dense_z;

    logic [31:0] q_vector [0:TOKENS-1];
    logic [31:0] k_current_vector [0:TOKENS-1];
    logic [31:0] k_peer_vector [0:TOKENS-1];
    integer expected_gate [0:TOKENS-1];
    logic seen_b [0:TOKENS-1];
    logic seen_z [0:TOKENS-1];

    integer fd;
    integer scan_count;
    integer file_rows;
    integer file_tokens;
    integer row_index;
    integer row_tag;
    integer stage_tag;
    integer block_tag;
    integer head_tag;
    integer expected_outputs_header;
    integer expected_folded_header;
    integer row_limit;
    integer stall_mode;
    integer total_errors;
    integer total_outputs_b;
    integer total_outputs_z;
    longint total_cycles_b;
    longint total_cycles_z;
    longint total_preload_cycles_b;
    longint total_preload_cycles_z;
    longint total_read_bits_b;
    longint total_read_bits_z;
    longint total_tare_dense_b;
    longint total_tare_dense_z;
    string vector_path;

    assign row_load_q_pair = {
        q_vector[PAIRS + row_load_pair_id],
        q_vector[row_load_pair_id]
    };
    assign row_load_k_pair = {
        k_current_vector[PAIRS + row_load_pair_id],
        k_current_vector[row_load_pair_id]
    };

    h67_zkqi_row_shiftmax_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .ACTIVE_FIFO_DEPTH(ACTIVE_FIFO_DEPTH),
        .ZK_BYPASS_ENABLE(BASELINE_ZK_BYPASS_ENABLE),
        .BUNDLE_SKIP_ENABLE(1'b1),
        .ACTIVE_SCORE_RESIDUAL_W(BASELINE_ACTIVE_SCORE_RESIDUAL_W)
    ) dut_baseline (
        .clk_core(clk), .rst_core(rst),
        .row_load_start(row_load_start),
        .row_load_valid(row_load_valid),
        .row_load_ready(row_load_ready_b),
        .row_load_pair_id(row_load_pair_id),
        .row_load_q_pair(row_load_q_pair),
        .row_load_k_pair(row_load_k_pair),
        .row_loaded(row_loaded_b),
        .window_start(window_start),
        .window_seal(seal_ready_b),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1),
        .cfg_threshold_q8(8'd64),
        .seal_ready(seal_ready_b),
        .window_done(window_done_b),
        .out_valid(out_valid_b), .out_ready(out_ready),
        .out_last(out_last_b), .out_token_id(out_token_b),
        .out_k_bits(out_k_b), .out_gate_q17(out_gate_b),
        .out_threshold_q8(out_threshold_b),
        .protocol_error(error_b),
        .perf_score_pairs(score_pairs_b),
        .perf_score_slots(score_slots_b),
        .perf_original_tokens(original_tokens_b),
        .perf_equal_pairs(equal_pairs_b),
        .perf_seeded_tokens(seeded_tokens_b),
        .perf_active_entries(active_entries_b),
        .perf_class_transactions(class_tx_b),
        .perf_exp_transactions(exp_tx_b),
        .perf_emitted_tokens(emitted_b),
        .perf_row_read_transactions(read_tx_b),
        .perf_row_read_bits(read_bits_b),
        .perf_preload_cycles(preload_cycles_b),
        .perf_total_cycles(cycles_b),
        .perf_score_stall_cycles(score_stall_b),
        .perf_output_stall_cycles(output_stall_b),
        .perf_preclassified_pairs(preclass_b),
        .perf_metadata_bits(metadata_bits_b),
        .perf_fifo_occupancy(fifo_occ_b),
        .perf_fifo_max_occupancy(fifo_max_b),
        .perf_tare_dense_fallbacks(tare_dense_b)
    );

    h67_zkqi_row_shiftmax_top #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .ACTIVE_FIFO_DEPTH(ACTIVE_FIFO_DEPTH),
        .ZK_BYPASS_ENABLE(1'b1),
        .BUNDLE_SKIP_ENABLE(CANDIDATE_BUNDLE_SKIP_ENABLE),
        .ACTIVE_SCORE_RESIDUAL_W(CANDIDATE_ACTIVE_SCORE_RESIDUAL_W)
    ) dut_zkqi (
        .clk_core(clk), .rst_core(rst),
        .row_load_start(row_load_start),
        .row_load_valid(row_load_valid),
        .row_load_ready(row_load_ready_z),
        .row_load_pair_id(row_load_pair_id),
        .row_load_q_pair(row_load_q_pair),
        .row_load_k_pair(row_load_k_pair),
        .row_loaded(row_loaded_z),
        .window_start(window_start),
        .window_seal(seal_ready_z),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1),
        .cfg_threshold_q8(8'd64),
        .seal_ready(seal_ready_z),
        .window_done(window_done_z),
        .out_valid(out_valid_z), .out_ready(out_ready),
        .out_last(out_last_z), .out_token_id(out_token_z),
        .out_k_bits(out_k_z), .out_gate_q17(out_gate_z),
        .out_threshold_q8(out_threshold_z),
        .protocol_error(error_z),
        .perf_score_pairs(score_pairs_z),
        .perf_score_slots(score_slots_z),
        .perf_original_tokens(original_tokens_z),
        .perf_equal_pairs(equal_pairs_z),
        .perf_seeded_tokens(seeded_tokens_z),
        .perf_active_entries(active_entries_z),
        .perf_class_transactions(class_tx_z),
        .perf_exp_transactions(exp_tx_z),
        .perf_emitted_tokens(emitted_z),
        .perf_row_read_transactions(read_tx_z),
        .perf_row_read_bits(read_bits_z),
        .perf_preload_cycles(preload_cycles_z),
        .perf_total_cycles(cycles_z),
        .perf_score_stall_cycles(score_stall_z),
        .perf_output_stall_cycles(output_stall_z),
        .perf_preclassified_pairs(preclass_z),
        .perf_metadata_bits(metadata_bits_z),
        .perf_fifo_occupancy(fifo_occ_z),
        .perf_fifo_max_occupancy(fifo_max_z),
        .perf_tare_dense_fallbacks(tare_dense_z)
    );

    always #5 clk = ~clk;

    function automatic integer vector_index(input integer token_id);
        integer pair_id;
        integer time_id;
        begin
            pair_id = token_id >> 1;
            time_id = token_id & 1;
            vector_index = time_id * PAIRS + pair_id;
        end
    endfunction

    task automatic check_output(
        input bit is_zkqi,
        input integer token_id,
        input logic [31:0] k_bits,
        input integer gate
    );
        integer index;
        begin
            index = vector_index(token_id);
            if (index < 0 || index >= TOKENS) begin
                $display("ERROR row=%0d mode=%0d token_id out of range %0d",
                         row_tag, is_zkqi, token_id);
                total_errors = total_errors + 1;
            end else begin
                if (k_current_vector[index] == 0) begin
                    $display("ERROR row=%0d mode=%0d emitted zero-K index=%0d",
                             row_tag, is_zkqi, index);
                    total_errors = total_errors + 1;
                end
                if (k_bits !== k_current_vector[index]) begin
                    $display("ERROR row=%0d mode=%0d K index=%0d got=%08x exp=%08x",
                             row_tag, is_zkqi, index, k_bits,
                             k_current_vector[index]);
                    total_errors = total_errors + 1;
                end
                if (gate !== expected_gate[index]) begin
                    $display("ERROR row=%0d mode=%0d gate index=%0d got=%0d exp=%0d",
                             row_tag, is_zkqi, index, gate,
                             expected_gate[index]);
                    total_errors = total_errors + 1;
                end
                if (is_zkqi) begin
                    if (seen_z[index]) begin
                        $display("ERROR row=%0d ZKQI duplicate index=%0d", row_tag, index);
                        total_errors = total_errors + 1;
                    end
                    seen_z[index] = 1'b1;
                    total_outputs_z = total_outputs_z + 1;
                end else begin
                    if (seen_b[index]) begin
                        $display("ERROR row=%0d baseline duplicate index=%0d", row_tag, index);
                        total_errors = total_errors + 1;
                    end
                    seen_b[index] = 1'b1;
                    total_outputs_b = total_outputs_b + 1;
                end
            end
        end
    endtask

    task automatic run_loaded_row;
        integer pair;
        integer token;
        integer active_pairs;
        integer baseline_dense_expected;
        integer candidate_dense_expected;
        integer update_count;
        integer bit_index;
        integer timeout;
        integer row_outputs_b;
        integer row_outputs_z;
        begin
            active_pairs = 0;
            baseline_dense_expected = 0;
            candidate_dense_expected = 0;
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if (k_current_vector[pair] != 0
                    || k_current_vector[PAIRS + pair] != 0)
                    active_pairs = active_pairs + 1;
                update_count = 0;
                for (bit_index = 0; bit_index < HEAD_DIM; bit_index = bit_index + 1)
                    update_count = update_count + (
                        (q_vector[pair][bit_index] ^ q_vector[PAIRS + pair][bit_index])
                        || (k_current_vector[pair][bit_index]
                            ^ k_current_vector[PAIRS + pair][bit_index])
                    );
                if ((k_current_vector[pair] != 0
                     || k_current_vector[PAIRS + pair] != 0)
                    && BASELINE_ACTIVE_SCORE_RESIDUAL_W > 0
                    && update_count > BASELINE_ACTIVE_SCORE_RESIDUAL_W)
                    baseline_dense_expected = baseline_dense_expected + 1;
                if ((k_current_vector[pair] != 0
                     || k_current_vector[PAIRS + pair] != 0)
                    && CANDIDATE_ACTIVE_SCORE_RESIDUAL_W > 0
                    && update_count > CANDIDATE_ACTIVE_SCORE_RESIDUAL_W)
                    candidate_dense_expected = candidate_dense_expected + 1;
                if (k_peer_vector[pair] !== k_current_vector[PAIRS + pair]
                    || k_peer_vector[PAIRS + pair] !== k_current_vector[pair]) begin
                    $display("ERROR row=%0d pair=%0d peer-K mismatch", row_tag, pair);
                    total_errors = total_errors + 1;
                end
            end
            for (token = 0; token < TOKENS; token = token + 1) begin
                seen_b[token] = 1'b0;
                seen_z[token] = 1'b0;
            end

            @(negedge clk);
            row_load_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            row_load_start = 1'b0;

            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                while (!(row_load_ready_b && row_load_ready_z)) @(negedge clk);
                row_load_pair_id = pair[PAIR_ID_W-1:0];
                row_load_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                row_load_valid = 1'b0;
            end
            if (!row_loaded_b || !row_loaded_z) begin
                $display("ERROR row=%0d row_loaded baseline/ZKQI=%0d/%0d",
                         row_tag, row_loaded_b, row_loaded_z);
                total_errors = total_errors + 1;
            end
            if (preload_cycles_b != PAIRS || preload_cycles_z != PAIRS) begin
                $display("ERROR row=%0d preload baseline/ZKQI=%0d/%0d expected=%0d",
                         row_tag, preload_cycles_b, preload_cycles_z, PAIRS);
                total_errors = total_errors + 1;
            end

            window_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            window_start = 1'b0;

            row_outputs_b = 0;
            row_outputs_z = 0;
            timeout = 0;
            while (!(window_done_b && window_done_z) && timeout < 20000) begin
                case (stall_mode)
                    0: begin
                        descriptor_issue_enable = 1'b1;
                        out_ready = 1'b1;
                    end
                    1: begin
                        // 每行独立播种，避免候选早晚完成污染后续baseline刺激。
                        descriptor_issue_enable = ((timeout * 17 + row_tag * 3) % 11) >= 2;
                        out_ready = ((timeout * 13 + row_tag * 5) % 7) != 1;
                    end
                    2: begin
                        descriptor_issue_enable = timeout >= 96;
                        out_ready = 1'b1;
                    end
                    default: begin
                        descriptor_issue_enable = (timeout % 64) >= 32;
                        out_ready = (timeout % 48) >= 24;
                    end
                endcase

                if (out_valid_b && out_ready) begin
                    check_output(1'b0, out_token_b, out_k_b, out_gate_b);
                    row_outputs_b = row_outputs_b + 1;
                end
                if (out_valid_z && out_ready) begin
                    check_output(1'b1, out_token_z, out_k_z, out_gate_z);
                    row_outputs_z = row_outputs_z + 1;
                end

                @(posedge clk);
                @(negedge clk);
                timeout = timeout + 1;
            end
            out_ready = 1'b1;
            descriptor_issue_enable = 1'b1;
            // 等待window_active/perf寄存器完成最终一次非阻塞更新再取ledger。
            @(posedge clk);
            @(negedge clk);

            if (!window_done_b || !window_done_z) begin
                $display("ERROR row=%0d timeout done=%0d/%0d", row_tag,
                         window_done_b, window_done_z);
                total_errors = total_errors + 1;
            end
            if (error_b || error_z) begin
                $display("ERROR row=%0d protocol baseline/ZKQI=%0d/%0d",
                         row_tag, error_b, error_z);
                total_errors = total_errors + 1;
            end
            if (row_outputs_b != expected_outputs_header
                || row_outputs_z != expected_outputs_header) begin
                $display("ERROR row=%0d outputs baseline/ZKQI=%0d/%0d expected=%0d",
                         row_tag, row_outputs_b, row_outputs_z,
                         expected_outputs_header);
                total_errors = total_errors + 1;
            end
            if (score_pairs_b != (BASELINE_ZK_BYPASS_ENABLE ? active_pairs : PAIRS)
                || score_pairs_z != active_pairs) begin
                $display("ERROR row=%0d score pairs baseline/ZKQI=%0d/%0d expected=%0d/%0d",
                         row_tag, score_pairs_b, score_pairs_z,
                         BASELINE_ZK_BYPASS_ENABLE ? active_pairs : PAIRS,
                         active_pairs);
                total_errors = total_errors + 1;
            end
            if (original_tokens_b != TOKENS || original_tokens_z != TOKENS) begin
                $display("ERROR row=%0d original tokens baseline/ZKQI=%0d/%0d",
                         row_tag, original_tokens_b, original_tokens_z);
                total_errors = total_errors + 1;
            end
            if (seeded_tokens_b
                    != (BASELINE_ZK_BYPASS_ENABLE ? 2 * (PAIRS - active_pairs) : 0)
                || seeded_tokens_z != 2 * (PAIRS - active_pairs)) begin
                $display("ERROR row=%0d seeded baseline/ZKQI=%0d/%0d expected=%0d/%0d",
                         row_tag, seeded_tokens_b, seeded_tokens_z,
                         BASELINE_ZK_BYPASS_ENABLE ? 2 * (PAIRS - active_pairs) : 0,
                         2 * (PAIRS - active_pairs));
                total_errors = total_errors + 1;
            end
            if (emitted_b != expected_outputs_header
                || emitted_z != expected_outputs_header
                || class_tx_b != class_tx_z
                || active_entries_b != active_entries_z) begin
                $display("ERROR row=%0d ledger emit=%0d/%0d class=%0d/%0d active=%0d/%0d",
                         row_tag, emitted_b, emitted_z, class_tx_b, class_tx_z,
                         active_entries_b, active_entries_z);
                total_errors = total_errors + 1;
            end
            if (preclass_b != (BASELINE_ZK_BYPASS_ENABLE ? PAIRS : 0)
                || preclass_z != PAIRS
                || fifo_max_z > ACTIVE_FIFO_DEPTH || fifo_occ_z != 0) begin
                $display("ERROR row=%0d metadata preclass=%0d/%0d fifo=%0d/%0d",
                         row_tag, preclass_b, preclass_z, fifo_occ_z, fifo_max_z);
                total_errors = total_errors + 1;
            end
            if (tare_dense_b != baseline_dense_expected
                || tare_dense_z != candidate_dense_expected) begin
                $display("ERROR row=%0d TARE dense baseline/candidate=%0d/%0d expected=%0d/%0d",
                         row_tag, tare_dense_b, tare_dense_z,
                         baseline_dense_expected, candidate_dense_expected);
                total_errors = total_errors + 1;
            end
            for (token = 0; token < TOKENS; token = token + 1)
                if (k_current_vector[token] != 0
                    && (!seen_b[token] || !seen_z[token])) begin
                    $display("ERROR row=%0d missing active token=%0d seen=%0d/%0d",
                             row_tag, token, seen_b[token], seen_z[token]);
                    total_errors = total_errors + 1;
                end

            total_cycles_b = total_cycles_b + cycles_b;
            total_cycles_z = total_cycles_z + cycles_z;
            total_preload_cycles_b = total_preload_cycles_b + preload_cycles_b;
            total_preload_cycles_z = total_preload_cycles_z + preload_cycles_z;
            total_read_bits_b = total_read_bits_b + read_bits_b;
            total_read_bits_z = total_read_bits_z + read_bits_z;
            total_tare_dense_b = total_tare_dense_b + tare_dense_b;
            total_tare_dense_z = total_tare_dense_z + tare_dense_z;
            $display("ROW_RESULT row=%0d stage=%0d block=%0d head=%0d bundle_skip=%0d active_pairs=%0d outputs=%0d baseline_preload=%0d zkqi_preload=%0d baseline_cycles=%0d zkqi_cycles=%0d baseline_e2e_cycles=%0d zkqi_e2e_cycles=%0d baseline_slots=%0d zkqi_slots=%0d seeded=%0d baseline_read_bits=%0d zkqi_read_bits=%0d fifo_max=%0d",
                     row_tag, stage_tag, block_tag, head_tag,
                     CANDIDATE_BUNDLE_SKIP_ENABLE, active_pairs,
                     expected_outputs_header, preload_cycles_b, preload_cycles_z,
                     cycles_b, cycles_z,
                     preload_cycles_b + cycles_b, preload_cycles_z + cycles_z,
                     score_slots_b, score_slots_z, seeded_tokens_z,
                     read_bits_b, read_bits_z, fifo_max_z);
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        integer token;
        clk = 1'b0;
        rst = 1'b1;
        row_load_start = 1'b0;
        row_load_valid = 1'b0;
        row_load_pair_id = '0;
        window_start = 1'b0;
        descriptor_issue_enable = 1'b1;
        out_ready = 1'b1;
        row_limit = 0;
        stall_mode = 1;
        total_errors = 0;
        total_outputs_b = 0;
        total_outputs_z = 0;
        total_cycles_b = 0;
        total_cycles_z = 0;
        total_preload_cycles_b = 0;
        total_preload_cycles_z = 0;
        total_read_bits_b = 0;
        total_read_bits_z = 0;
        total_tare_dense_b = 0;
        total_tare_dense_z = 0;

        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS=<path>");
        void'($value$plusargs("ROW_LIMIT=%d", row_limit));
        void'($value$plusargs("STALL_MODE=%d", stall_mode));
        fd = $fopen(vector_path, "r");
        if (fd == 0)
            $fatal(1, "cannot open vectors: %s", vector_path);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_rows <= 0 || file_tokens != TOKENS)
            $fatal(1, "invalid vector header rows=%0d tokens=%0d",
                   file_rows, file_tokens);
        if (row_limit <= 0 || row_limit > file_rows)
            row_limit = file_rows;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);

        for (row_index = 0; row_index < row_limit; row_index = row_index + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs_header, expected_folded_header);
            if (scan_count != 6 || row_tag != row_index)
                $fatal(1, "invalid row header row=%0d scan=%0d tag=%0d",
                       row_index, scan_count, row_tag);
            for (token = 0; token < TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_current_vector[token],
                    k_peer_vector[token], expected_gate[token]);
                if (scan_count != 4)
                    $fatal(1, "invalid token row=%0d token=%0d", row_index, token);
            end
            run_loaded_row();
        end
        $fclose(fd);

        if (total_errors != 0)
            $fatal(1, "FAIL tb_h67_zkqi_row_miter rows=%0d errors=%0d",
                   row_limit, total_errors);
        $display("PASS tb_h67_zkqi_row_miter rows=%0d stall_mode=%0d bundle_skip=%0d outputs=%0d baseline_preload=%0d zkqi_preload=%0d baseline_cycles=%0d zkqi_cycles=%0d baseline_e2e_cycles=%0d zkqi_e2e_cycles=%0d baseline_read_bits=%0d zkqi_read_bits=%0d baseline_tare_dense=%0d candidate_tare_dense=%0d",
                 row_limit, stall_mode, CANDIDATE_BUNDLE_SKIP_ENABLE,
                 total_outputs_b,
                 total_preload_cycles_b, total_preload_cycles_z,
                 total_cycles_b, total_cycles_z,
                 total_preload_cycles_b + total_cycles_b,
                 total_preload_cycles_z + total_cycles_z,
                 total_read_bits_b, total_read_bits_z,
                 total_tare_dense_b, total_tare_dense_z);
        $finish;
    end

    initial begin
        integer watchdog_cycles;
        watchdog_cycles = 5000000;
        void'($value$plusargs("WATCHDOG_CYCLES=%d", watchdog_cycles));
        repeat (watchdog_cycles) @(posedge clk);
        $fatal(1, "FAIL tb_h67_zkqi_row_miter global watchdog");
    end
endmodule

`default_nettype wire
