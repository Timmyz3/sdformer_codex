`timescale 1ns/1ps
`default_nettype none

// Measure RQTB2S build (start->seal) vs emit (seal->done) on ep35 rows.
module tb_h67_laws_phase_split_2s;
    localparam int HEAD_DIM = 32;
    localparam int MAX_TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SLOT_FIFO_DEPTH = 32;
    localparam int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1);

    logic clk = 1'b0;
    logic rst_core;
    logic window_start;
    logic window_seal;
    logic seal_ready;
    logic window_done;
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
    logic [31:0] perf_equal;
    logic [31:0] perf_desc;
    logic [31:0] perf_tokens;
    logic [31:0] perf_active;
    logic [31:0] perf_classes;
    logic [31:0] perf_exp;
    logic [31:0] perf_emitted;
    logic [31:0] perf_k_reads;
    logic [31:0] perf_k_bits;
    logic [31:0] perf_cycles;
    logic [31:0] perf_pair_stalls;
    logic [31:0] perf_desc_stalls;
    logic [31:0] perf_out_stalls;
    logic [FIFO_OCC_W-1:0] fifo_occ;
    logic [FIFO_OCC_W-1:0] fifo_max;

    logic [31:0] q_vector [0:MAX_TOKENS-1];
    logic [31:0] k_vector [0:MAX_TOKENS-1];
    logic [31:0] peer_vector [0:MAX_TOKENS-1];
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
    integer expected_outputs;
    integer expected_folded;
    integer token;
    integer pair;
    integer wait_cycles;
    integer build_cycles;
    integer emit_cycles;
    integer total_build;
    integer total_emit;
    integer total_seq;
    string vector_path;

    always #1 clk = ~clk;

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
        .window_start(window_start),
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
        .out_last(out_last),
        .out_token_id(out_token),
        .out_k_bits(out_k),
        .out_gate_q17(out_gate),
        .out_threshold_q8(),
        .protocol_error(protocol_error),
        .perf_pairs(perf_pairs),
        .perf_slots(perf_slots),
        .perf_equal_pairs(perf_equal),
        .perf_quotient_descriptors(perf_desc),
        .perf_original_tokens(perf_tokens),
        .perf_active_entries(perf_active),
        .perf_class_transactions(perf_classes),
        .perf_exp_transactions(perf_exp),
        .perf_emitted_tokens(perf_emitted),
        .perf_k_read_transactions(perf_k_reads),
        .perf_k_read_bits(perf_k_bits),
        .perf_total_cycles(perf_cycles),
        .perf_pair_stall_cycles(perf_pair_stalls),
        .perf_descriptor_stall_cycles(perf_desc_stalls),
        .perf_output_stall_cycles(perf_out_stalls),
        .perf_fifo_occupancy(fifo_occ),
        .perf_fifo_max_occupancy(fifo_max)
    );

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        rst_core = 1'b1;
        window_start = 1'b0;
        window_seal = 1'b0;
        pair_valid = 1'b0;
        pair_id = '0;
        q_pair = '0;
        k_pair = '0;
        total_build = 0;
        total_emit = 0;
        total_seq = 0;
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "cannot open %s", vector_path);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_tokens != MAX_TOKENS)
            $fatal(1, "bad header");
        repeat (4) @(negedge clk);
        rst_core = 1'b0;

        for (row_index = 0; row_index < file_rows; row_index = row_index + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs, expected_folded);
            if (scan_count != 6)
                $fatal(1, "bad row header");
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_vector[token],
                    peer_vector[token], expected_gate[token]);
                if (scan_count != 4)
                    $fatal(1, "bad token");
            end

            @(negedge clk);
            window_start = 1'b1;
            @(negedge clk);
            window_start = 1'b0;
            build_cycles = 0;
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
                    $fatal(1, "pair timeout row=%0d pair=%0d", row_tag, pair);
                @(negedge clk);
                pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!seal_ready && wait_cycles < 8000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!seal_ready)
                $fatal(1, "seal timeout row=%0d", row_tag);
            build_cycles = perf_cycles;
            window_seal = 1'b1;
            @(negedge clk);
            window_seal = 1'b0;
            wait_cycles = 0;
            while (!window_done && wait_cycles < 20000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!window_done)
                $fatal(1, "done timeout row=%0d", row_tag);
            if (protocol_error)
                $fatal(1, "protocol error row=%0d", row_tag);
            emit_cycles = perf_cycles - build_cycles;
            total_build = total_build + build_cycles;
            total_emit = total_emit + emit_cycles;
            total_seq = total_seq + perf_cycles;
            $display("LAWS_PHASE row=%0d stage=%0d block=%0d head=%0d build=%0d emit=%0d total=%0d slots=%0d active=%0d classes=%0d",
                row_tag, stage_tag, block_tag, head_tag,
                build_cycles, emit_cycles, perf_cycles,
                perf_slots, perf_active, perf_classes);
        end
        $fclose(fd);
        $display("LAWS_PHASE_SUM build=%0d emit=%0d seq=%0d rows=%0d",
            total_build, total_emit, total_seq, file_rows);
        $display("PASS tb_h67_laws_phase_split_2s");
        $finish;
    end
endmodule

`default_nettype wire
