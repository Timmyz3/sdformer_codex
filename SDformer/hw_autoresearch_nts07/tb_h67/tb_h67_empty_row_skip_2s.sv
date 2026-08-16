`timescale 1ns/1ps
`default_nettype none

module tb_h67_empty_row_skip_2s;
    localparam int MAX_TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SLOT_FIFO_DEPTH = 32;
    localparam int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1);

    logic clk = 1'b0;
    logic rst_core;
    logic window_start;
    logic row_k_present;
    logic window_seal;
    logic seal_ready;
    logic window_done;
    logic pair_valid;
    logic pair_ready;
    logic [PAIR_ID_W-1:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic out_valid;
    logic protocol_error;
    logic [31:0] perf_cycles;
    logic [31:0] perf_skipped;
    logic skipped_row;

    logic [31:0] q_vector [0:MAX_TOKENS-1];
    logic [31:0] k_vector [0:MAX_TOKENS-1];
    logic [31:0] peer_vector [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];

    integer fd, scan_count, file_rows, file_tokens, row_index;
    integer row_tag, stage_tag, block_tag, head_tag;
    integer expected_outputs, expected_folded, token, pair, wait_cycles;
    integer has_k, empty_rows, dense_rows, skip_cycles, run_cycles;
    string vector_path;

    always #1 clk = ~clk;

    h67_empty_row_skip_2s #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
        .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b1)
    ) u_dut (
        .clk_core(clk),
        .rst_core(rst_core),
        .window_start(window_start),
        .row_k_present(row_k_present),
        .window_seal(window_seal),
        .descriptor_issue_enable(1'b1),
        .cfg_preserve_mean(1'b1),
        .cfg_threshold_q8(8'd64),
        .seal_ready(seal_ready),
        .window_done(window_done),
        .pair_valid(pair_valid),
        .pair_ready(pair_ready),
        .pair_id(pair_id),
        .q_pair(q_pair),
        .k_pair(k_pair),
        .out_valid(out_valid),
        .out_ready(1'b1),
        .out_last(),
        .out_token_id(),
        .out_k_bits(),
        .out_gate_q17(),
        .protocol_error(protocol_error),
        .perf_total_cycles(perf_cycles),
        .perf_skipped_rows(perf_skipped),
        .skipped_row(skipped_row)
    );

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        rst_core = 1'b1;
        window_start = 1'b0;
        row_k_present = 1'b0;
        window_seal = 1'b0;
        pair_valid = 1'b0;
        empty_rows = 0;
        dense_rows = 0;
        skip_cycles = 0;
        run_cycles = 0;
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "cannot open");
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        repeat (4) @(negedge clk);
        rst_core = 1'b0;

        for (row_index = 0; row_index < file_rows; row_index = row_index + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs, expected_folded);
            has_k = 0;
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_vector[token],
                    peer_vector[token], expected_gate[token]);
                if (k_vector[token] != 0)
                    has_k = 1;
            end
            row_k_present = has_k[0];
            @(negedge clk);
            window_start = 1'b1;
            @(negedge clk);
            window_start = 1'b0;
            if (!has_k) begin
                wait_cycles = 1;
                while (!seal_ready && wait_cycles < 20) begin
                    @(negedge clk);
                    wait_cycles = wait_cycles + 1;
                end
                if (!seal_ready || !skipped_row || out_valid)
                    $fatal(1, "empty row did not skip row=%0d", row_tag);
                window_seal = 1'b1;
                @(negedge clk);
                window_seal = 1'b0;
                empty_rows = empty_rows + 1;
                skip_cycles = skip_cycles + wait_cycles + 1;
                $display("EMPTY_SKIP row=%0d stage=%0d block=%0d cycles=%0d",
                    row_tag, stage_tag, block_tag, wait_cycles + 1);
            end else begin
                for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                    pair_id = PAIR_ID_W'(pair);
                    q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                    k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                    pair_valid = 1'b1;
                    wait_cycles = 0;
                    @(posedge clk);
                    while (!pair_ready && wait_cycles < 8000) begin
                        wait_cycles = wait_cycles + 1;
                        @(posedge clk);
                    end
                    if (!pair_ready)
                        $fatal(1, "pair timeout row=%0d", row_tag);
                    @(negedge clk);
                    pair_valid = 1'b0;
                end
                wait_cycles = 0;
                while (!seal_ready && wait_cycles < 8000) begin
                    @(negedge clk);
                    wait_cycles = wait_cycles + 1;
                end
                window_seal = 1'b1;
                @(negedge clk);
                window_seal = 1'b0;
                wait_cycles = 0;
                while (!window_done && wait_cycles < 20000) begin
                    @(negedge clk);
                    wait_cycles = wait_cycles + 1;
                end
                if (!window_done || protocol_error || skipped_row)
                    $fatal(1, "dense row failed row=%0d err=%0d", row_tag, protocol_error);
                dense_rows = dense_rows + 1;
                run_cycles = run_cycles + perf_cycles;
                $display("EMPTY_KEEP row=%0d stage=%0d block=%0d cycles=%0d",
                    row_tag, stage_tag, block_tag, perf_cycles);
            end
        end
        $fclose(fd);
        if (perf_skipped != 32'(empty_rows))
            $fatal(1, "skip count %0d != %0d", perf_skipped, empty_rows);
        $display("EMPTY_SKIP_SUM empty=%0d dense=%0d skip_cycles=%0d run_cycles=%0d total=%0d",
            empty_rows, dense_rows, skip_cycles, run_cycles, skip_cycles + run_cycles);
        $display("PASS tb_h67_empty_row_skip_2s");
        $finish;
    end
endmodule

`default_nettype wire
