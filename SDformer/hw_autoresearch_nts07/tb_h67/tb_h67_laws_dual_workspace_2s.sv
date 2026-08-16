`timescale 1ns/1ps
`default_nettype none

// Two RQTB2S engines, exclusive build and exclusive emit.
// This measures shared-encoder/shared-Shiftmax wall time with 2x area.
module tb_h67_laws_dual_workspace_2s;
    localparam int MAX_TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SLOT_FIFO_DEPTH = 32;
    localparam int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1);
    localparam int MAX_ROWS = 138;

    logic clk = 1'b0;
    logic rst_core;
    logic [1:0] start_w;
    logic [1:0] seal_w;
    logic [1:0] seal_ready_w;
    logic [1:0] done_w;
    logic [1:0] pair_valid_w;
    logic [1:0] pair_ready_w;
    logic [PAIR_ID_W-1:0] pair_id_w [0:1];
    logic [63:0] q_pair_w [0:1];
    logic [63:0] k_pair_w [0:1];
    logic [1:0] out_valid_w;
    logic [1:0] out_ready_w;
    logic [TOKEN_W-1:0] out_token_w [0:1];
    logic [31:0] out_k_w [0:1];
    logic [8:0] out_gate_w [0:1];
    logic [1:0] error_w;
    logic [31:0] emitted_w [0:1];

    logic [31:0] q_mem [0:MAX_ROWS-1][0:MAX_TOKENS-1];
    logic [31:0] k_mem [0:MAX_ROWS-1][0:MAX_TOKENS-1];
    integer expected_count [0:MAX_ROWS-1];
    integer stage_mem [0:MAX_ROWS-1];
    integer block_mem [0:MAX_ROWS-1];

    integer file_rows;
    integer next_row;
    integer build_eng;
    integer emit_eng;
    integer pair_idx [0:1];
    integer out_count [0:1];
    integer row_of [0:1];
    integer seen [0:1][0:MAX_TOKENS-1];
    integer wall;
    integer finished;
    integer token;
    integer fd;
    integer scan_count;
    integer dummy_gate;
    logic [31:0] dummy_peer;
    logic [31:0] scan_q;
    logic [31:0] scan_k;
    integer dummy_tag;
    integer dummy_head;
    integer dummy_folded;
    string vector_path;

    always #1 clk = ~clk;

    genvar gi;
    generate
        for (gi = 0; gi < 2; gi = gi + 1) begin : g_eng
            h67_temporal_slot_shiftmax_sync_k_2s_top #(
                .PAIRS(PAIRS),
                .PAIR_ID_W(PAIR_ID_W),
                .TOKEN_W(TOKEN_W),
                .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH),
                .FIFO_OCC_W(FIFO_OCC_W),
                .QUOTIENT_ENABLE(1'b1)
            ) u_rqtb (
                .clk_core(clk),
                .rst_core(rst_core),
                .window_start(start_w[gi]),
                .window_seal(seal_w[gi]),
                .descriptor_issue_enable(1'b1),
                .cfg_preserve_mean(1'b1),
                .cfg_threshold_q8(8'd64),
                .seal_ready(seal_ready_w[gi]),
                .window_done(done_w[gi]),
                .pair_valid(pair_valid_w[gi]),
                .pair_ready(pair_ready_w[gi]),
                .pair_id(pair_id_w[gi]),
                .q_pair(q_pair_w[gi]),
                .k_pair(k_pair_w[gi]),
                .out_valid(out_valid_w[gi]),
                .out_ready(out_ready_w[gi]),
                .out_last(),
                .out_token_id(out_token_w[gi]),
                .out_k_bits(out_k_w[gi]),
                .out_gate_q17(out_gate_w[gi]),
                .out_threshold_q8(),
                .protocol_error(error_w[gi]),
                .perf_pairs(),
                .perf_slots(),
                .perf_equal_pairs(),
                .perf_quotient_descriptors(),
                .perf_original_tokens(),
                .perf_active_entries(),
                .perf_class_transactions(),
                .perf_exp_transactions(),
                .perf_emitted_tokens(emitted_w[gi]),
                .perf_k_read_transactions(),
                .perf_k_read_bits(),
                .perf_total_cycles(),
                .perf_pair_stall_cycles(),
                .perf_descriptor_stall_cycles(),
                .perf_output_stall_cycles(),
                .perf_fifo_occupancy(),
                .perf_fifo_max_occupancy()
            );
        end
    endgenerate

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "open failed");
        scan_count = $fscanf(fd, "%d %d", file_rows, dummy_tag);
        if (file_rows != MAX_ROWS)
            $fatal(1, "rows=%0d", file_rows);
        for (next_row = 0; next_row < file_rows; next_row = next_row + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                dummy_tag, stage_mem[next_row], block_mem[next_row],
                dummy_head, expected_count[next_row], dummy_folded);
            expected_count[next_row] = 0;
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    scan_q, scan_k, dummy_peer, dummy_gate);
                q_mem[next_row][token] = scan_q;
                k_mem[next_row][token] = scan_k;
                if (scan_k != 0)
                    expected_count[next_row] = expected_count[next_row] + 1;
            end
        end
        $fclose(fd);

        rst_core = 1'b1;
        start_w = 2'b00;
        seal_w = 2'b00;
        pair_valid_w = 2'b00;
        out_ready_w = 2'b00;
        build_eng = -1;
        emit_eng = -1;
        next_row = 0;
        finished = 0;
        wall = 0;
        row_of[0] = -1;
        row_of[1] = -1;
        pair_idx[0] = 0;
        pair_idx[1] = 0;
        out_count[0] = 0;
        out_count[1] = 0;
        repeat (4) @(negedge clk);
        rst_core = 1'b0;

        while (finished < file_rows) begin
            @(negedge clk);
            start_w = 2'b00;
            seal_w = 2'b00;
            pair_valid_w = 2'b00;
            out_ready_w = 2'b00;

            if (emit_eng >= 0) begin
                out_ready_w[emit_eng] = 1'b1;
                if (done_w[emit_eng] && !start_w[emit_eng]) begin
                    if (out_count[emit_eng] != expected_count[row_of[emit_eng]])
                        $fatal(1, "row %0d emit count %0d != %0d",
                            row_of[emit_eng], out_count[emit_eng],
                            expected_count[row_of[emit_eng]]);
                    $display("LAWS_DW_DONE row=%0d eng=%0d outs=%0d",
                        row_of[emit_eng], emit_eng, out_count[emit_eng]);
                    row_of[emit_eng] = -1;
                    emit_eng = -1;
                    finished = finished + 1;
                end
            end else begin
                if (row_of[0] >= 0 && pair_idx[0] >= PAIRS && seal_ready_w[0] == 1'b0 && done_w[0])
                    ;
                if (row_of[0] >= 0 && pair_idx[0] >= PAIRS && !done_w[0] && seal_w[0] == 1'b0) begin
                    // waiting to become emit
                end
                if (row_of[0] >= 0 && pair_idx[0] >= PAIRS && !done_w[0]) begin
                    emit_eng = 0;
                    out_ready_w[0] = 1'b1;
                end else if (row_of[1] >= 0 && pair_idx[1] >= PAIRS && !done_w[1]) begin
                    emit_eng = 1;
                    out_ready_w[1] = 1'b1;
                end
            end

            if (build_eng >= 0) begin
                if (pair_idx[build_eng] < PAIRS) begin
                    pair_valid_w[build_eng] = 1'b1;
                    pair_id_w[build_eng] = PAIR_ID_W'(pair_idx[build_eng]);
                    q_pair_w[build_eng] = {
                        q_mem[row_of[build_eng]][pair_idx[build_eng] + PAIRS],
                        q_mem[row_of[build_eng]][pair_idx[build_eng]]
                    };
                    k_pair_w[build_eng] = {
                        k_mem[row_of[build_eng]][pair_idx[build_eng] + PAIRS],
                        k_mem[row_of[build_eng]][pair_idx[build_eng]]
                    };
                end else if (seal_ready_w[build_eng]) begin
                    seal_w[build_eng] = 1'b1;
                    if (emit_eng < 0)
                        emit_eng = build_eng;
                    build_eng = -1;
                end
            end else if (next_row < file_rows) begin
                if (row_of[0] < 0) begin
                    build_eng = 0;
                    row_of[0] = next_row;
                    pair_idx[0] = 0;
                    out_count[0] = 0;
                    start_w[0] = 1'b1;
                    next_row = next_row + 1;
                end else if (row_of[1] < 0) begin
                    build_eng = 1;
                    row_of[1] = next_row;
                    pair_idx[1] = 0;
                    out_count[1] = 0;
                    start_w[1] = 1'b1;
                    next_row = next_row + 1;
                end
            end
            wall = wall + 1;
            if (wall > 400000)
                $fatal(1, "timeout finished=%0d", finished);
        end
        $display("LAWS_DW_SUM rows=%0d wall=%0d", file_rows, wall);
        $display("PASS tb_h67_laws_dual_workspace_2s");
        $finish;
    end

    always @(posedge clk) begin
        integer eng;
        integer tok;
        if (!rst_core) begin
            for (eng = 0; eng < 2; eng = eng + 1) begin
                if (error_w[eng])
                    $fatal(1, "engine %0d protocol error", eng);
                if (pair_valid_w[eng] && pair_ready_w[eng] && pair_idx[eng] < PAIRS)
                    pair_idx[eng] <= pair_idx[eng] + 1;
                if (out_valid_w[eng] && out_ready_w[eng]) begin
                    tok = out_token_w[eng];
                    tok = tok[0] ? (PAIRS + (tok >> 1)) : (tok >> 1);
                    if (k_mem[row_of[eng]][tok] !== out_k_w[eng])
                        $fatal(1, "row %0d token %0d K mismatch", row_of[eng], tok);
                    if (seen[eng][tok])
                        $fatal(1, "row %0d duplicate token %0d", row_of[eng], tok);
                    seen[eng][tok] <= 1;
                    out_count[eng] <= out_count[eng] + 1;
                end
                if (start_w[eng]) begin
                    for (tok = 0; tok < MAX_TOKENS; tok = tok + 1)
                        seen[eng][tok] = 0;
                end
            end
        end
    end
endmodule

`default_nettype wire
