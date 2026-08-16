`timescale 1ns/1ps
`default_nettype none

module tb_h67_laws_shared_backend_2s #(
    parameter int MEMORY_IMPL = 0
);
    localparam int MAX_TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int MAX_ROWS = 138;

    logic clk = 1'b0;
    logic rst_core;
    logic window_start;
    logic row_k_present;
    logic window_seal;
    logic build_ready;
    logic seal_ready;
    logic last_row_done;
    logic pair_valid;
    logic pair_ready;
    logic [PAIR_ID_W-1:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic out_valid;
    logic out_last;
    logic [TOKEN_W-1:0] out_token;
    logic [31:0] out_k;
    logic [8:0] out_gate;
    logic protocol_error;
    logic [31:0] perf_pairs;
    logic [31:0] perf_slots;
    logic [31:0] perf_equal_pairs;

    logic [31:0] q_mem [0:MAX_ROWS-1][0:MAX_TOKENS-1];
    logic [31:0] k_mem [0:MAX_ROWS-1][0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_ROWS-1][0:MAX_TOKENS-1];
    integer expected_count [0:MAX_ROWS-1];
    integer signed expected_acc [0:31];
    integer signed got_acc [0:31];
    logic [31:0] scan_q;
    logic [31:0] scan_k;
    logic [31:0] dummy_peer;
    integer dummy_gate;
    integer dummy_i;
    integer file_rows;
    integer row;
    integer token;
    integer pair;
    integer wait_cycles;
    integer wall;
    integer done_rows;
    integer out_count;
    integer emit_row;
    integer trace;
    integer fd;
    integer scan_count;
    integer row_limit;
    integer skip_rows;
    integer sequential;
    integer skip_empty_pairs;
    integer pair_issues;
    integer total_equal;
    integer total_slots;
    integer empty_pairs;
    integer total_tokens;
    integer acc_lane;
    integer seal_pairs [0:MAX_ROWS-1];
    integer seal_slots [0:MAX_ROWS-1];
    integer seal_equal [0:MAX_ROWS-1];
    string vector_path;

    function automatic integer lane_weight(input integer lane);
        lane_weight = (lane % 17) - 8;
    endfunction

    always #1 clk = ~clk;

    h67_laws_shared_backend_2s_top #(
        .PAIRS(PAIRS),
        .PAIR_ID_W(PAIR_ID_W),
        .TOKEN_W(TOKEN_W),
        .QUOTIENT_ENABLE(1'b1),
        .MEMORY_IMPL(MEMORY_IMPL)
    ) dut (
        .clk_core(clk),
        .rst_core(rst_core),
        .window_start(window_start),
        .row_k_present(row_k_present),
        .window_seal(window_seal),
        .descriptor_issue_enable(1'b1),
        .cfg_preserve_mean(1'b1),
        .cfg_threshold_q8(8'd64),
        .build_ready(build_ready),
        .seal_ready(seal_ready),
        .emit_active(),
        .last_row_done(last_row_done),
        .pair_valid(pair_valid),
        .pair_ready(pair_ready),
        .pair_id(pair_id),
        .q_pair(q_pair),
        .k_pair(k_pair),
        .out_valid(out_valid),
        .out_ready(1'b1),
        .out_last(out_last),
        .out_token_id(out_token),
        .out_k_bits(out_k),
        .out_gate_q17(out_gate),
        .protocol_error(protocol_error),
        .perf_pairs(perf_pairs),
        .perf_slots(perf_slots),
        .perf_equal_pairs(perf_equal_pairs)
    );

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        if (!$value$plusargs("ROW_LIMIT=%d", row_limit))
            row_limit = MAX_ROWS;
        if (!$value$plusargs("SEQUENTIAL=%d", sequential))
            sequential = 0;
        if (!$value$plusargs("SKIP_EMPTY_PAIRS=%d", skip_empty_pairs))
            skip_empty_pairs = 0;
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "open failed");
        scan_count = $fscanf(fd, "%d %d", file_rows, dummy_i);
        if (file_rows != MAX_ROWS)
            $fatal(1, "rows");
        if (row_limit <= 0 || row_limit > file_rows)
            row_limit = file_rows;
        for (row = 0; row < file_rows; row = row + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                dummy_i, dummy_gate, dummy_gate, dummy_gate, dummy_gate, dummy_gate);
            expected_count[row] = 0;
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    scan_q, scan_k, dummy_peer, dummy_gate);
                q_mem[row][token] = scan_q;
                k_mem[row][token] = scan_k;
                expected_gate[row][token] = dummy_gate;
                if (scan_k != 0)
                    expected_count[row] = expected_count[row] + 1;
            end
        end
        $fclose(fd);

        rst_core = 1'b1;
        window_start = 1'b0;
        row_k_present = 1'b1;
        window_seal = 1'b0;
        pair_valid = 1'b0;
        wall = 0;
        done_rows = 0;
        out_count = 0;
        emit_row = 0;
        skip_rows = 0;
        pair_issues = 0;
        empty_pairs = 0;
        total_tokens = 0;
        total_equal = 0;
        total_slots = 0;
        for (row = 0; row < MAX_ROWS; row = row + 1) begin
            seal_pairs[row] = 0;
            seal_slots[row] = 0;
            seal_equal[row] = 0;
        end
        repeat (4) @(negedge clk);
        rst_core = 1'b0;
        for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1) begin
            expected_acc[acc_lane] = 0;
            got_acc[acc_lane] = 0;
        end

        fork
            begin : feed
                for (row = 0; row < row_limit; row = row + 1) begin
                    if (sequential) begin
                        wait_cycles = 0;
                        while (done_rows < row && wait_cycles < 40000) begin
                            @(negedge clk);
                            wait_cycles = wait_cycles + 1;
                        end
                    end
                    wait_cycles = 0;
                    while (!build_ready && wait_cycles < 20000) begin
                        @(negedge clk);
                        wait_cycles = wait_cycles + 1;
                    end
                    if (!build_ready)
                        $fatal(1, "build_ready timeout row=%0d", row);
                    row_k_present = (expected_count[row] != 0);
                    @(negedge clk);
                    window_start = 1'b1;
                    @(negedge clk);
                    window_start = 1'b0;
                    if (!row_k_present) begin
                        skip_rows = skip_rows + 1;
                        $display("SB_SKIP row=%0d", row);
                    end else begin
                        for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                            if (skip_empty_pairs
                                && k_mem[row][pair] == 0
                                && k_mem[row][pair + PAIRS] == 0)
                                empty_pairs = empty_pairs + 1;
                            else begin
                            pair_id = PAIR_ID_W'(pair);
                            q_pair = {q_mem[row][pair + PAIRS], q_mem[row][pair]};
                            k_pair = {k_mem[row][pair + PAIRS], k_mem[row][pair]};
                            pair_valid = 1'b1;
                            wait_cycles = 0;
                            @(posedge clk);
                            while (!pair_ready && wait_cycles < 8000) begin
                                @(posedge clk);
                                wait_cycles = wait_cycles + 1;
                            end
                            if (!pair_ready)
                                $fatal(1, "pair timeout row=%0d pair=%0d", row, pair);
                            pair_issues = pair_issues + 1;
                            if (k_mem[row][pair] == 0 && k_mem[row][pair + PAIRS] == 0)
                                empty_pairs = empty_pairs + 1;
                            @(negedge clk);
                            pair_valid = 1'b0;
                            end
                        end
                        wait_cycles = 0;
                        while (!seal_ready && wait_cycles < 8000) begin
                            @(negedge clk);
                            wait_cycles = wait_cycles + 1;
                        end
                        if (!seal_ready)
                            $fatal(1, "seal timeout row=%0d", row);
                        // Snapshot before the next overlapping window_start
                        // clears the shared encoder counters.
                        if (perf_pairs != 32'(PAIRS))
                            $fatal(1, "SEAL_PAIRS row=%0d got=%0d",
                                row, perf_pairs);
                        if ((perf_slots + perf_equal_pairs)
                            != (perf_pairs << 1))
                            $fatal(1, "SEAL_SLOT_ID row=%0d slots=%0d equal=%0d pairs=%0d",
                                row, perf_slots, perf_equal_pairs, perf_pairs);
                        seal_pairs[row] = perf_pairs;
                        seal_slots[row] = perf_slots;
                        seal_equal[row] = perf_equal_pairs;
                        window_seal = 1'b1;
                        @(negedge clk);
                        window_seal = 1'b0;
                        $display("SB_SEAL row=%0d pairs=%0d slots=%0d equal=%0d",
                            row, seal_pairs[row], seal_slots[row],
                            seal_equal[row]);
                    end
                end
            end
            begin : drain
                while (done_rows < row_limit) begin
                    @(posedge clk);
                    wall = wall + 1;
                    if (protocol_error) begin
                        if (skip_empty_pairs) begin
                            $display("EMPTY_PAIR_DROP_BREAKS_CONTRACT row=%0d wall=%0d",
                                emit_row, wall);
                            $display("PASS tb_h67_laws_shared_backend_2s empty_pair_drop_illegal");
                            $finish;
                        end
                        $fatal(1, "protocol error row=%0d", emit_row);
                    end
                    if (out_valid) begin
                        trace = out_token[0] ? (PAIRS + (out_token >> 1))
                                             : (out_token >> 1);
                        if (k_mem[emit_row][trace] !== out_k)
                            $fatal(1, "K mismatch row=%0d token=%0d", emit_row, out_token);
                        if (out_gate !== expected_gate[emit_row][trace][8:0])
                            $fatal(1, "gate mismatch row=%0d token=%0d", emit_row, out_token);
                        for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1)
                            if (out_k[acc_lane])
                                got_acc[acc_lane] = got_acc[acc_lane]
                                    + lane_weight(acc_lane) * out_gate;
                        out_count = out_count + 1;
                    end
                    if (last_row_done) begin
                        if (out_count != expected_count[emit_row])
                            $fatal(1, "count row=%0d got=%0d exp=%0d",
                                emit_row, out_count, expected_count[emit_row]);
                        for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1)
                            expected_acc[acc_lane] = 0;
                        for (token = 0; token < MAX_TOKENS; token = token + 1)
                            if (k_mem[emit_row][token] != 0)
                                for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1)
                                    if (k_mem[emit_row][token][acc_lane])
                                        expected_acc[acc_lane] = expected_acc[acc_lane]
                                            + lane_weight(acc_lane)
                                            * expected_gate[emit_row][token];
                        for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1)
                            if (got_acc[acc_lane] != expected_acc[acc_lane])
                                $fatal(1, "Acc32 row=%0d lane=%0d got=%0d exp=%0d",
                                    emit_row, acc_lane, got_acc[acc_lane],
                                    expected_acc[acc_lane]);
                        if (expected_count[emit_row] != 0) begin
                            total_equal = total_equal + seal_equal[emit_row];
                            total_slots = total_slots + seal_slots[emit_row];
                        end
                        $display("SB_DONE row=%0d outs=%0d skip=%0d pairs=%0d slots=%0d equal=%0d",
                            emit_row, out_count, expected_count[emit_row] == 0,
                            seal_pairs[emit_row],
                            seal_slots[emit_row],
                            seal_equal[emit_row]);
                        total_tokens = total_tokens + out_count;
                        emit_row = emit_row + 1;
                        out_count = 0;
                        for (acc_lane = 0; acc_lane < 32; acc_lane = acc_lane + 1)
                            got_acc[acc_lane] = 0;
                        done_rows = done_rows + 1;
                    end
                    if (wall > 400000)
                        $fatal(1, "drain timeout done=%0d", done_rows);
                end
            end
        join
        $display("SB_SUM rows=%0d wall=%0d skip=%0d tokens=%0d pairs=%0d empty_pairs=%0d sequential=%0d slots=%0d equal=%0d",
            row_limit, wall, skip_rows, total_tokens, pair_issues, empty_pairs,
            sequential, total_slots, total_equal);
        $display("PASS tb_h67_laws_shared_backend_2s");
        $finish;
    end
endmodule

`default_nettype wire
