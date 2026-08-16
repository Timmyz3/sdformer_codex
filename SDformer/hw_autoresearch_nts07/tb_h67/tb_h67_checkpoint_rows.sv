`timescale 1ns/1ps
`default_nettype none

module tb_h67_checkpoint_rows #(
    parameter int MAX_TOKENS = 450
);
    localparam int HEAD_DIM = 32;
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int CLASS_COUNT_W = $clog2(HEAD_DIM + 4);

    logic clk;
    logic rst_n;
    logic cfg_start;
    logic [TOKEN_W-1:0] cfg_n_tokens;
    logic cfg_preserve_mean;
    logic cfg_enable_score_fold;
    logic [7:0] cfg_threshold_q8;
    logic in_valid;
    logic in_ready;
    logic in_last;
    logic in_time_sel;
    logic [HEAD_DIM-1:0] in_q_bits;
    logic [2*HEAD_DIM-1:0] in_k_pair_bits;
    logic out_valid;
    logic out_ready;
    logic out_last;
    logic [TOKEN_W-1:0] out_token_idx;
    logic [HEAD_DIM-1:0] out_k_bits;
    logic [8:0] out_gate_q8;
    logic [7:0] out_threshold_q8;
    logic busy;
    logic done;
    logic [TOKEN_W-1:0] perf_tokens_loaded;
    logic [TOKEN_W-1:0] perf_kzero_folded;
    logic [TOKEN_W-1:0] perf_entries_emitted;
    logic [CLASS_COUNT_W-1:0] perf_fold_classes;
    logic [15:0] perf_exp_transactions;
    logic perf_score_range_error;

    logic [31:0] q_vector [0:MAX_TOKENS-1];
    logic [31:0] k_current_vector [0:MAX_TOKENS-1];
    logic [31:0] k_peer_vector [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];

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
    integer total_errors;
    integer total_checked_outputs;
    string vector_path;

    h67_score_class_row_engine #(
        .HEAD_DIM(HEAD_DIM),
        .MAX_TOKENS(MAX_TOKENS),
        .ENABLE_MOTION_XOR(1'b1),
        .TOKEN_W(TOKEN_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_start(cfg_start),
        .cfg_n_tokens(cfg_n_tokens),
        .cfg_preserve_mean(cfg_preserve_mean),
        .cfg_enable_score_fold(cfg_enable_score_fold),
        .cfg_threshold_q8(cfg_threshold_q8),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_last(in_last),
        .in_time_sel(in_time_sel),
        .in_q_bits(in_q_bits),
        .in_k_pair_bits(in_k_pair_bits),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_last(out_last),
        .out_token_idx(out_token_idx),
        .out_k_bits(out_k_bits),
        .out_gate_q8(out_gate_q8),
        .out_threshold_q8(out_threshold_q8),
        .busy(busy),
        .done(done),
        .perf_tokens_loaded(perf_tokens_loaded),
        .perf_kzero_folded(perf_kzero_folded),
        .perf_entries_emitted(perf_entries_emitted),
        .perf_fold_classes(perf_fold_classes),
        .perf_exp_transactions(perf_exp_transactions),
        .perf_score_range_error(perf_score_range_error)
    );

    always #5 clk = ~clk;

    task automatic run_loaded_row;
        integer token;
        integer output_count;
        integer active_count;
        integer folded_count;
        integer timeout_cycles;
        integer token_index;
        begin
            active_count = 0;
            folded_count = 0;
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                if (k_current_vector[token] == 0) folded_count = folded_count + 1;
                else active_count = active_count + 1;
            end
            if (active_count != expected_outputs_header
                || folded_count != expected_folded_header) begin
                $display("ERROR row=%0d S%0d.B%0d.H%0d header active/fold mismatch got=%0d/%0d expected=%0d/%0d",
                         row_tag, stage_tag, block_tag, head_tag,
                         active_count, folded_count,
                         expected_outputs_header, expected_folded_header);
                total_errors = total_errors + 1;
            end

            cfg_n_tokens = MAX_TOKENS[TOKEN_W-1:0];
            @(negedge clk);
            cfg_start = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cfg_start = 1'b0;

            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                if ((token % 13) == 7) begin
                    in_valid = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                end
                while (!in_ready) @(negedge clk);
                in_valid = 1'b1;
                in_last = (token == MAX_TOKENS - 1);
                in_time_sel = 1'b0;
                in_q_bits = q_vector[token];
                in_k_pair_bits = {k_peer_vector[token], k_current_vector[token]};
                @(posedge clk);
                @(negedge clk);
                in_valid = 1'b0;
                in_last = 1'b0;
            end

            output_count = 0;
            timeout_cycles = 0;
            while (!done && timeout_cycles < 5000) begin
                @(negedge clk);
                out_ready = ((timeout_cycles % 5) != 2);
                if (out_valid && out_ready) begin
                    token_index = out_token_idx;
                    if (token_index < 0 || token_index >= MAX_TOKENS) begin
                        $display("ERROR row=%0d output token out of range: %0d", row_tag, token_index);
                        total_errors = total_errors + 1;
                    end else begin
                        if (k_current_vector[token_index] == 0) begin
                            $display("ERROR row=%0d emitted folded zero-K token=%0d", row_tag, token_index);
                            total_errors = total_errors + 1;
                        end
                        if (out_k_bits !== k_current_vector[token_index]) begin
                            $display("ERROR row=%0d token=%0d K got=%08x expected=%08x",
                                     row_tag, token_index, out_k_bits, k_current_vector[token_index]);
                            total_errors = total_errors + 1;
                        end
                        if (out_gate_q8 !== expected_gate[token_index][8:0]) begin
                            $display("ERROR row=%0d token=%0d gate got=%0d expected=%0d",
                                     row_tag, token_index, out_gate_q8, expected_gate[token_index]);
                            total_errors = total_errors + 1;
                        end
                    end
                    output_count = output_count + 1;
                    total_checked_outputs = total_checked_outputs + 1;
                end
                timeout_cycles = timeout_cycles + 1;
            end
            out_ready = 1'b1;
            if (!done) begin
                $display("ERROR row=%0d timeout", row_tag);
                total_errors = total_errors + 1;
            end
            if (output_count != expected_outputs_header) begin
                $display("ERROR row=%0d outputs got=%0d expected=%0d",
                         row_tag, output_count, expected_outputs_header);
                total_errors = total_errors + 1;
            end
            if (perf_tokens_loaded != MAX_TOKENS[TOKEN_W-1:0]
                || perf_kzero_folded != expected_folded_header[TOKEN_W-1:0]
                || perf_entries_emitted != expected_outputs_header[TOKEN_W-1:0]) begin
                $display("ERROR row=%0d perf loaded/fold/emitted=%0d/%0d/%0d expected=%0d/%0d/%0d",
                         row_tag, perf_tokens_loaded, perf_kzero_folded,
                         perf_entries_emitted, MAX_TOKENS,
                         expected_folded_header, expected_outputs_header);
                total_errors = total_errors + 1;
            end
            if (perf_score_range_error) begin
                $display("ERROR row=%0d score range error", row_tag);
                total_errors = total_errors + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        integer token;
        clk = 1'b0;
        rst_n = 1'b0;
        cfg_start = 1'b0;
        cfg_n_tokens = '0;
        cfg_preserve_mean = 1'b1;
        cfg_enable_score_fold = 1'b1;
        cfg_threshold_q8 = 8'd64;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_time_sel = 1'b0;
        in_q_bits = '0;
        in_k_pair_bits = '0;
        out_ready = 1'b1;
        total_errors = 0;
        total_checked_outputs = 0;

        if (!$value$plusargs("VECTORS=%s", vector_path)) begin
            $fatal(1, "missing +VECTORS=<path>");
        end
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "cannot open vectors: %s", vector_path);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_rows <= 0 || file_tokens != MAX_TOKENS) begin
            $fatal(1, "invalid vector header rows=%0d tokens=%0d", file_rows, file_tokens);
        end

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        for (row_index = 0; row_index < file_rows; row_index = row_index + 1) begin
            scan_count = $fscanf(
                fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs_header, expected_folded_header
            );
            if (scan_count != 6 || row_tag != row_index) begin
                $fatal(1, "invalid row header at row=%0d scan=%0d tag=%0d", row_index, scan_count, row_tag);
            end
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(
                    fd, "%h %h %h %d",
                    q_vector[token], k_current_vector[token],
                    k_peer_vector[token], expected_gate[token]
                );
                if (scan_count != 4) begin
                    $fatal(1, "invalid token vector row=%0d token=%0d scan=%0d", row_index, token, scan_count);
                end
            end
            run_loaded_row();
        end
        $fclose(fd);

        if (total_errors != 0) begin
            $fatal(1, "FAIL tb_h67_checkpoint_rows rows=%0d checked=%0d errors=%0d",
                   file_rows, total_checked_outputs, total_errors);
        end
        $display("PASS tb_h67_checkpoint_rows rows=%0d tokens=%0d checked_outputs=%0d",
                 file_rows, MAX_TOKENS, total_checked_outputs);
        $finish;
    end

    initial begin
        repeat (2000000) @(posedge clk);
        $fatal(1, "FAIL tb_h67_checkpoint_rows global watchdog timeout");
    end
endmodule

`default_nettype wire
