`timescale 1ns/1ps
`default_nettype none

// Single frozen Motion wrapper replay for VCD/SAIF attribution.
module tb_h67_motion_dc_activity #(
    parameter bit RQTB = 1'b1,
    parameter int MAX_ROWS = 138,
    parameter int ROW_LIMIT = MAX_ROWS
);
    localparam int TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int HEAD_DIM = 32;

    logic clk_core = 1'b0;
    logic rst_core;
    logic window_start;
    logic window_seal;
    logic descriptor_issue_enable;
    logic [15:0] backpressure_lfsr;
    logic backpressure_reseed;
    logic cfg_preserve_mean;
    logic [7:0] cfg_threshold_q8;
    logic seal_ready;
    logic window_done;
    logic pair_valid;
    logic pair_ready;
    logic [7:0] pair_id;
    logic [63:0] q_pair;
    logic [63:0] k_pair;
    logic out_valid;
    logic out_ready;
    logic out_last;
    logic [8:0] out_token_id;
    logic [31:0] out_k_bits;
    logic [8:0] out_gate_q17;
    logic [7:0] out_threshold_q8;
    logic protocol_error;
    logic [31:0] perf_pairs;
    logic [31:0] perf_slots;
    logic [31:0] perf_equal_pairs;
    logic [31:0] perf_quotient_descriptors;
    logic [31:0] perf_original_tokens;
    logic [31:0] perf_active_entries;
    logic [31:0] perf_class_transactions;
    logic [31:0] perf_exp_transactions;
    logic [31:0] perf_emitted_tokens;
    logic [31:0] perf_k_read_transactions;
    logic [31:0] perf_k_read_bits;
    logic [31:0] perf_total_cycles;
    logic [31:0] perf_pair_stall_cycles;
    logic [31:0] perf_descriptor_stall_cycles;
    logic [31:0] perf_output_stall_cycles;
    logic [5:0] perf_fifo_occupancy;
    logic [5:0] perf_fifo_max_occupancy;

    logic [31:0] q_vector [0:TOKENS-1];
    logic [31:0] k_vector [0:TOKENS-1];
    logic [31:0] peer_vector [0:TOKENS-1];
    integer expected_gate [0:TOKENS-1];
    logic seen [0:TOKENS-1];
    integer fd;
    integer scan_count;
    integer file_rows;
    integer file_tokens;
    integer row_tag;
    integer stage_tag;
    integer block_tag;
    integer head_tag;
    integer expected_outputs;
    integer expected_folded;
    integer emitted;
    integer total_rows;
    integer total_cycles;
    integer measured_cycles;
    integer dump_start_row;
    integer dump_rows;
    bit dump_configured;
    bit dump_active;
    string vectors;
    string dump_file;

    generate
        if (RQTB) begin : g_rqtb
            h67_rqtb2s_mssb5_dc_top dut (.*);
            initial begin
                #0;
                if (dump_configured) begin
                    $dumpfile(dump_file);
                    $dumpvars(0, dut);
                    $dumpoff;
                end
            end
        end else begin : g_fixed
            h67_fixed2s_mssb5_dc_top dut (.*);
            initial begin
                #0;
                if (dump_configured) begin
                    $dumpfile(dump_file);
                    $dumpvars(0, dut);
                    $dumpoff;
                end
            end
        end
    endgenerate

    always #5 clk_core = ~clk_core;

    // Match the frozen fair three-way testbench exactly.
    always @(negedge clk_core) begin
        if (rst_core || backpressure_reseed) begin
            backpressure_lfsr <= 16'h1d3f;
            descriptor_issue_enable <= 1'b0;
            out_ready <= 1'b0;
        end else begin
            backpressure_lfsr <= {backpressure_lfsr[14:0],
                backpressure_lfsr[15] ^ backpressure_lfsr[13]
                ^ backpressure_lfsr[12] ^ backpressure_lfsr[10]};
            descriptor_issue_enable <= backpressure_lfsr[0]
                                    || backpressure_lfsr[5];
            out_ready <= backpressure_lfsr[2] || backpressure_lfsr[9];
        end
    end

    always_ff @(posedge clk_core) begin
        if (dump_active)
            measured_cycles <= measured_cycles + 1;
    end

    task automatic drive_pairs;
        integer pair;
        integer wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7)
                    @(negedge clk_core);
                pair_id = pair[7:0];
                q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                pair_valid = 1'b1;
                wait_cycles = 0;
                do begin
                    @(posedge clk_core);
                    wait_cycles = wait_cycles + 1;
                end while (!pair_ready && wait_cycles < 20000);
                if (!pair_ready)
                    $fatal(1, "pair timeout row=%0d pair=%0d", row_tag, pair);
                @(negedge clk_core);
                pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!seal_ready && wait_cycles < 20000) begin
                @(negedge clk_core);
                wait_cycles = wait_cycles + 1;
            end
            if (!seal_ready)
                $fatal(1, "seal timeout row=%0d", row_tag);
            window_seal = 1'b1;
            @(negedge clk_core);
            window_seal = 1'b0;
        end
    endtask

    task automatic run_row;
        integer timeout;
        integer trace_index;
        begin
            emitted = 0;
            for (integer token = 0; token < TOKENS; token = token + 1)
                seen[token] = 1'b0;
            @(posedge clk_core);
            backpressure_reseed = 1'b1;
            @(posedge clk_core);
            backpressure_reseed = 1'b0;
            @(negedge clk_core);
            window_start = 1'b1;
            @(negedge clk_core);
            window_start = 1'b0;
            fork
                drive_pairs();
                begin
                    timeout = 0;
                    while (!window_done && timeout < 50000) begin
                        @(posedge clk_core);
                        timeout = timeout + 1;
                        if (out_valid && out_ready) begin
                            if (out_token_id >= TOKENS || seen[out_token_id])
                                $fatal(1, "duplicate/out-of-range token row=%0d token=%0d",
                                       row_tag, out_token_id);
                            seen[out_token_id] = 1'b1;
                            trace_index = out_token_id[0]
                                ? PAIRS + (out_token_id >> 1)
                                : (out_token_id >> 1);
                            if (out_k_bits !== k_vector[trace_index]
                                || out_gate_q17 !== expected_gate[trace_index][8:0])
                                $fatal(1, "trace mismatch row=%0d token=%0d",
                                       row_tag, out_token_id);
                            emitted = emitted + 1;
                        end
                        @(negedge clk_core);
                    end
                    if (!window_done)
                        $fatal(1, "window timeout row=%0d", row_tag);
                    // In the frozen parallel fair TB, RQTB completes before Fixed2S
                    // and therefore observes one final active->idle retirement edge.
                    // Keep that real edge in the single-wrapper VCD so its activity
                    // and perf_total_cycles use the same accounting boundary.
                    if (RQTB) begin
                        @(posedge clk_core);
                        @(negedge clk_core);
                    end
                end
            join
            if (protocol_error || emitted != expected_outputs
                || perf_pairs != PAIRS || perf_original_tokens != TOKENS
                || perf_emitted_tokens != expected_outputs)
                $fatal(1, "row contract mismatch row=%0d emitted=%0d expected=%0d",
                       row_tag, emitted, expected_outputs);
            total_cycles = total_cycles + perf_total_cycles;
            total_rows = total_rows + 1;
            $display(
                "MOTION_ACTIVITY_ROW mode=%s row=%0d cycles=%0d slots=%0d equal=%0d emitted=%0d",
                RQTB ? "rqtb" : "fixed", row_tag, perf_total_cycles,
                perf_slots, perf_equal_pairs, emitted
            );
        end
    endtask

    initial begin
        dump_configured = $value$plusargs("DUMP_FILE=%s", dump_file);
        dump_start_row = 0;
        dump_rows = 1;
        void'($value$plusargs("DUMP_START_ROW=%d", dump_start_row));
        void'($value$plusargs("DUMP_ROWS=%d", dump_rows));
        if (!$value$plusargs("VECTORS=%s", vectors))
            $fatal(1, "missing +VECTORS");
        if (ROW_LIMIT <= 0 || ROW_LIMIT > MAX_ROWS || dump_rows <= 0
            || dump_start_row < 0 || dump_start_row + dump_rows > ROW_LIMIT)
            $fatal(1, "invalid row/dump limits");

        rst_core = 1'b1;
        window_start = 1'b0;
        window_seal = 1'b0;
        backpressure_reseed = 1'b0;
        descriptor_issue_enable = 1'b0;
        cfg_preserve_mean = 1'b1;
        cfg_threshold_q8 = 8'd64;
        pair_valid = 1'b0;
        pair_id = '0;
        q_pair = '0;
        k_pair = '0;
        out_ready = 1'b0;
        total_rows = 0;
        total_cycles = 0;
        measured_cycles = 0;
        dump_active = 1'b0;

        fd = $fopen(vectors, "r");
        if (fd == 0)
            $fatal(1, "cannot open vectors=%s", vectors);
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_rows < ROW_LIMIT || file_tokens != TOKENS)
            $fatal(1, "invalid vector header rows=%0d tokens=%0d", file_rows, file_tokens);
        repeat (5) @(negedge clk_core);
        rst_core = 1'b0;

        for (integer row = 0; row < file_rows; row = row + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs, expected_folded);
            if (scan_count != 6 || row_tag != row)
                $fatal(1, "invalid row header row=%0d", row);
            for (integer token = 0; token < TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_vector[token], peer_vector[token],
                    expected_gate[token]);
                if (scan_count != 4)
                    $fatal(1, "invalid token row=%0d token=%0d", row, token);
            end
            if (row < ROW_LIMIT) begin
                if (dump_configured && row == dump_start_row) begin
                    @(negedge clk_core);
                    dump_active = 1'b1;
                    $dumpon;
                end
                run_row();
                if (dump_configured && row + 1 == dump_start_row + dump_rows) begin
                    @(negedge clk_core);
                    dump_active = 1'b0;
                    $dumpoff;
                    $display(
                        "SAIF_MEASUREMENT design=%s start_group=%0d groups=%0d measured_cycles=%0d scope=fair_lfsr_row_execution",
                        RQTB ? "h67_rqtb2s_mssb5_dc_top" : "h67_fixed2s_mssb5_dc_top",
                        dump_start_row, dump_rows, measured_cycles
                    );
                end
                if (row + 1 < ROW_LIMIT) begin
                    @(negedge clk_core);
                end
            end
        end
        $fclose(fd);
        $display("PASS Motion wrapper activity mode=%s rows=%0d total_cycles=%0d",
                 RQTB ? "rqtb" : "fixed", total_rows, total_cycles);
        $finish;
    end

    initial begin
        repeat (10_000_000) @(posedge clk_core);
        $fatal(1, "Motion activity timeout");
    end
endmodule

`default_nettype wire
