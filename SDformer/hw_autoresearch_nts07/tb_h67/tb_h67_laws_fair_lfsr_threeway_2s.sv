`timescale 1ns/1ps
`default_nettype none

// Sequential-row LFSR fair package: Fixed2S / RQTB2S / SharedBackend+Skip.
// Same LFSR polynomial and seed as the sealed 2S TB (16'h1d3f).
module tb_h67_laws_fair_lfsr_threeway_2s;
    localparam int MAX_TOKENS = 450;
    localparam int PAIRS = 225;
    localparam int PAIR_ID_W = $clog2(PAIRS);
    localparam int TOKEN_W = $clog2(MAX_TOKENS + 1);
    localparam int SLOT_FIFO_DEPTH = 32;
    localparam int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1);
`ifdef MSSB5_SCORE_FRONT
    localparam bit USE_MSSB5 = 1'b1;
`else
    localparam bit USE_MSSB5 = 1'b0;
`endif

    logic clk = 1'b0;
    logic rst_core;
    logic descriptor_issue_enable;
    logic common_out_ready;
    logic [15:0] backpressure_lfsr;
    logic backpressure_reseed;

    logic fixed_start, fixed_seal, fixed_seal_ready, fixed_done;
    logic fixed_pair_valid, fixed_pair_ready, fixed_out_valid, fixed_out_last;
    logic [PAIR_ID_W-1:0] fixed_pair_id;
    logic [63:0] fixed_q_pair, fixed_k_pair;
    logic [TOKEN_W-1:0] fixed_out_token;
    logic [31:0] fixed_out_k;
    logic [8:0] fixed_out_gate;
    logic fixed_error;
    logic [31:0] fixed_cycles, fixed_emitted;

    logic rqtb_start, rqtb_seal, rqtb_seal_ready, rqtb_done;
    logic rqtb_pair_valid, rqtb_pair_ready, rqtb_out_valid, rqtb_out_last;
    logic [PAIR_ID_W-1:0] rqtb_pair_id;
    logic [63:0] rqtb_q_pair, rqtb_k_pair;
    logic [TOKEN_W-1:0] rqtb_out_token;
    logic [31:0] rqtb_out_k;
    logic [8:0] rqtb_out_gate;
    logic rqtb_error;
    logic [31:0] rqtb_cycles, rqtb_emitted;

    logic sb_start, sb_seal, sb_seal_ready, sb_done, sb_build_ready;
    logic sb_row_k_present;
    logic sb_pair_valid, sb_pair_ready, sb_out_valid, sb_out_last;
    logic [PAIR_ID_W-1:0] sb_pair_id;
    logic [63:0] sb_q_pair, sb_k_pair;
    logic [TOKEN_W-1:0] sb_out_token;
    logic [31:0] sb_out_k;
    logic [8:0] sb_out_gate;
    logic sb_error;

    logic [31:0] q_vector [0:MAX_TOKENS-1];
    logic [31:0] k_vector [0:MAX_TOKENS-1];
    logic [31:0] peer_vector [0:MAX_TOKENS-1];
    integer expected_gate [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] fixed_token_log [0:MAX_TOKENS-1];
    logic [31:0] fixed_k_log [0:MAX_TOKENS-1];
    logic [8:0] fixed_gate_log [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] rqtb_token_log [0:MAX_TOKENS-1];
    logic [31:0] rqtb_k_log [0:MAX_TOKENS-1];
    logic [8:0] rqtb_gate_log [0:MAX_TOKENS-1];
    logic [TOKEN_W-1:0] sb_token_log [0:MAX_TOKENS-1];
    logic [31:0] sb_k_log [0:MAX_TOKENS-1];
    logic [8:0] sb_gate_log [0:MAX_TOKENS-1];

    integer fd, scan_count, file_rows, file_tokens, row_limit, row_index;
    integer row_tag, stage_tag, block_tag, head_tag;
    integer expected_outputs_header, expected_folded_header;
    integer fixed_count, rqtb_count, sb_count, active_count;
    integer total_rows, skip_rows;
    integer total_fixed_cycles, total_rqtb_cycles, total_sb_cycles;
    integer total_fixed_pairs, total_fixed_slots, total_fixed_equal;
    integer total_rqtb_pairs, total_rqtb_slots, total_rqtb_equal;
    integer sb_row_cycles;
    logic sb_row_active;
    logic [31:0] fixed_pairs, fixed_slots, fixed_equal;
    logic [31:0] rqtb_pairs, rqtb_slots, rqtb_equal;
    logic [31:0] sb_pairs, sb_slots, sb_equal;
    string vector_path;

    always #1 clk = ~clk;

    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .PAIRS(PAIRS), .PAIR_ID_W(PAIR_ID_W), .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH), .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b0), .MSSB5_SCORE_FRONT(USE_MSSB5)
    ) u_fixed (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(fixed_start), .window_seal(fixed_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1), .cfg_threshold_q8(8'd64),
        .seal_ready(fixed_seal_ready), .window_done(fixed_done),
        .pair_valid(fixed_pair_valid), .pair_ready(fixed_pair_ready),
        .pair_id(fixed_pair_id), .q_pair(fixed_q_pair), .k_pair(fixed_k_pair),
        .out_valid(fixed_out_valid), .out_ready(common_out_ready),
        .out_last(fixed_out_last), .out_token_id(fixed_out_token),
        .out_k_bits(fixed_out_k), .out_gate_q17(fixed_out_gate),
        .out_threshold_q8(), .protocol_error(fixed_error),
        .perf_pairs(fixed_pairs), .perf_slots(fixed_slots),
        .perf_equal_pairs(fixed_equal),
        .perf_quotient_descriptors(), .perf_original_tokens(),
        .perf_active_entries(), .perf_class_transactions(),
        .perf_exp_transactions(), .perf_emitted_tokens(fixed_emitted),
        .perf_k_read_transactions(), .perf_k_read_bits(),
        .perf_total_cycles(fixed_cycles),
        .perf_pair_stall_cycles(), .perf_descriptor_stall_cycles(),
        .perf_output_stall_cycles(), .perf_fifo_occupancy(),
        .perf_fifo_max_occupancy()
    );

    h67_temporal_slot_shiftmax_sync_k_2s_top #(
        .PAIRS(PAIRS), .PAIR_ID_W(PAIR_ID_W), .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH), .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b1), .MSSB5_SCORE_FRONT(USE_MSSB5)
    ) u_rqtb (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(rqtb_start), .window_seal(rqtb_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1), .cfg_threshold_q8(8'd64),
        .seal_ready(rqtb_seal_ready), .window_done(rqtb_done),
        .pair_valid(rqtb_pair_valid), .pair_ready(rqtb_pair_ready),
        .pair_id(rqtb_pair_id), .q_pair(rqtb_q_pair), .k_pair(rqtb_k_pair),
        .out_valid(rqtb_out_valid), .out_ready(common_out_ready),
        .out_last(rqtb_out_last), .out_token_id(rqtb_out_token),
        .out_k_bits(rqtb_out_k), .out_gate_q17(rqtb_out_gate),
        .out_threshold_q8(), .protocol_error(rqtb_error),
        .perf_pairs(rqtb_pairs), .perf_slots(rqtb_slots),
        .perf_equal_pairs(rqtb_equal),
        .perf_quotient_descriptors(), .perf_original_tokens(),
        .perf_active_entries(), .perf_class_transactions(),
        .perf_exp_transactions(), .perf_emitted_tokens(rqtb_emitted),
        .perf_k_read_transactions(), .perf_k_read_bits(),
        .perf_total_cycles(rqtb_cycles),
        .perf_pair_stall_cycles(), .perf_descriptor_stall_cycles(),
        .perf_output_stall_cycles(), .perf_fifo_occupancy(),
        .perf_fifo_max_occupancy()
    );

    h67_laws_shared_backend_2s_top #(
        .PAIRS(PAIRS), .PAIR_ID_W(PAIR_ID_W), .TOKEN_W(TOKEN_W),
        .SLOT_FIFO_DEPTH(SLOT_FIFO_DEPTH), .FIFO_OCC_W(FIFO_OCC_W),
        .QUOTIENT_ENABLE(1'b1), .MSSB5_SCORE_FRONT(USE_MSSB5)
    ) u_shared (
        .clk_core(clk), .rst_core(rst_core),
        .window_start(sb_start), .row_k_present(sb_row_k_present),
        .window_seal(sb_seal),
        .descriptor_issue_enable(descriptor_issue_enable),
        .cfg_preserve_mean(1'b1), .cfg_threshold_q8(8'd64),
        .build_ready(sb_build_ready), .seal_ready(sb_seal_ready),
        .emit_active(), .last_row_done(sb_done),
        .pair_valid(sb_pair_valid), .pair_ready(sb_pair_ready),
        .pair_id(sb_pair_id), .q_pair(sb_q_pair), .k_pair(sb_k_pair),
        .out_valid(sb_out_valid), .out_ready(common_out_ready),
        .out_last(sb_out_last), .out_token_id(sb_out_token),
        .out_k_bits(sb_out_k), .out_gate_q17(sb_out_gate),
        .protocol_error(sb_error),
        .perf_pairs(sb_pairs), .perf_slots(sb_slots),
        .perf_equal_pairs(sb_equal)
    );

    always @(negedge clk) begin
        if (rst_core || backpressure_reseed) begin
            backpressure_lfsr <= 16'h1d3f;
            descriptor_issue_enable <= 1'b0;
            common_out_ready <= 1'b0;
        end else begin
            backpressure_lfsr <= {backpressure_lfsr[14:0],
                backpressure_lfsr[15] ^ backpressure_lfsr[13]
                ^ backpressure_lfsr[12] ^ backpressure_lfsr[10]};
            descriptor_issue_enable <= backpressure_lfsr[0]
                                    || backpressure_lfsr[5];
            common_out_ready <= backpressure_lfsr[2]
                             || backpressure_lfsr[9];
        end
    end

    always @(posedge clk) begin
        if (!rst_core) begin
            if (sb_row_active)
                sb_row_cycles = sb_row_cycles + 1;
            if (fixed_out_valid && common_out_ready) begin
                fixed_token_log[fixed_count] = fixed_out_token;
                fixed_k_log[fixed_count] = fixed_out_k;
                fixed_gate_log[fixed_count] = fixed_out_gate;
                fixed_count = fixed_count + 1;
            end
            if (rqtb_out_valid && common_out_ready) begin
                rqtb_token_log[rqtb_count] = rqtb_out_token;
                rqtb_k_log[rqtb_count] = rqtb_out_k;
                rqtb_gate_log[rqtb_count] = rqtb_out_gate;
                rqtb_count = rqtb_count + 1;
            end
            if (sb_out_valid && common_out_ready) begin
                sb_token_log[sb_count] = sb_out_token;
                sb_k_log[sb_count] = sb_out_k;
                sb_gate_log[sb_count] = sb_out_gate;
                sb_count = sb_count + 1;
            end
        end
    end

    task automatic drive_fixed_pairs;
        integer pair, wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7) @(negedge clk);
                fixed_pair_id = PAIR_ID_W'(pair);
                fixed_q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                fixed_k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                fixed_pair_valid = 1'b1;
                wait_cycles = 0;
                @(posedge clk);
                while (!fixed_pair_ready && wait_cycles < 8000) begin
                    wait_cycles = wait_cycles + 1;
                    @(posedge clk);
                end
                if (!fixed_pair_ready)
                    $fatal(1, "fixed pair timeout row=%0d pair=%0d", row_tag, pair);
                @(negedge clk);
                fixed_pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!fixed_seal_ready && wait_cycles < 8000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!fixed_seal_ready)
                $fatal(1, "fixed seal timeout row=%0d", row_tag);
            fixed_seal = 1'b1;
            @(negedge clk);
            fixed_seal = 1'b0;
        end
    endtask

    task automatic drive_rqtb_pairs;
        integer pair, wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7) @(negedge clk);
                rqtb_pair_id = PAIR_ID_W'(pair);
                rqtb_q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                rqtb_k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                rqtb_pair_valid = 1'b1;
                wait_cycles = 0;
                @(posedge clk);
                while (!rqtb_pair_ready && wait_cycles < 8000) begin
                    wait_cycles = wait_cycles + 1;
                    @(posedge clk);
                end
                if (!rqtb_pair_ready)
                    $fatal(1, "rqtb pair timeout row=%0d pair=%0d", row_tag, pair);
                @(negedge clk);
                rqtb_pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!rqtb_seal_ready && wait_cycles < 8000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!rqtb_seal_ready)
                $fatal(1, "rqtb seal timeout row=%0d", row_tag);
            rqtb_seal = 1'b1;
            @(negedge clk);
            rqtb_seal = 1'b0;
        end
    endtask

    task automatic drive_sb_pairs;
        integer pair, wait_cycles;
        begin
            for (pair = 0; pair < PAIRS; pair = pair + 1) begin
                if ((pair % 13) == 7) @(negedge clk);
                sb_pair_id = PAIR_ID_W'(pair);
                sb_q_pair = {q_vector[pair + PAIRS], q_vector[pair]};
                sb_k_pair = {k_vector[pair + PAIRS], k_vector[pair]};
                sb_pair_valid = 1'b1;
                wait_cycles = 0;
                @(posedge clk);
                while (!sb_pair_ready && wait_cycles < 8000) begin
                    wait_cycles = wait_cycles + 1;
                    @(posedge clk);
                end
                if (!sb_pair_ready)
                    $fatal(1, "shared pair timeout row=%0d pair=%0d", row_tag, pair);
                @(negedge clk);
                sb_pair_valid = 1'b0;
            end
            wait_cycles = 0;
            while (!sb_seal_ready && wait_cycles < 8000) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!sb_seal_ready)
                $fatal(1, "shared seal timeout row=%0d", row_tag);
            sb_seal = 1'b1;
            @(negedge clk);
            sb_seal = 1'b0;
        end
    endtask

    task automatic wait_fixed_done;
        integer timeout;
        begin
            timeout = 0;
            while (!fixed_done && timeout < 30000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!fixed_done)
                $fatal(1, "fixed row timeout row=%0d", row_tag);
        end
    endtask

    task automatic wait_rqtb_done;
        integer timeout;
        begin
            timeout = 0;
            while (!rqtb_done && timeout < 30000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!rqtb_done)
                $fatal(1, "rqtb row timeout row=%0d", row_tag);
        end
    endtask

    task automatic wait_sb_done;
        integer timeout;
        begin
            timeout = 0;
            while (!sb_done && timeout < 30000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!sb_done)
                $fatal(1, "shared row timeout row=%0d", row_tag);
            sb_row_active = 1'b0;
        end
    endtask

    initial begin
        integer token, index, trace_index;
        if (!$value$plusargs("VECTORS=%s", vector_path))
            $fatal(1, "missing +VECTORS");
        if (!$value$plusargs("ROW_LIMIT=%d", row_limit))
            row_limit = 0;
        fd = $fopen(vector_path, "r");
        if (fd == 0) $fatal(1, "open failed");
        scan_count = $fscanf(fd, "%d %d", file_rows, file_tokens);
        if (scan_count != 2 || file_tokens != MAX_TOKENS)
            $fatal(1, "bad header");
        if (row_limit <= 0 || row_limit > file_rows)
            row_limit = file_rows;

        rst_core = 1'b1;
        backpressure_reseed = 1'b0;
        descriptor_issue_enable = 1'b0;
        common_out_ready = 1'b0;
        fixed_start = 1'b0; fixed_seal = 1'b0; fixed_pair_valid = 1'b0;
        rqtb_start = 1'b0; rqtb_seal = 1'b0; rqtb_pair_valid = 1'b0;
        sb_start = 1'b0; sb_seal = 1'b0; sb_pair_valid = 1'b0;
        sb_row_k_present = 1'b1;
        sb_row_active = 1'b0;
        total_rows = 0; skip_rows = 0;
        total_fixed_cycles = 0; total_rqtb_cycles = 0; total_sb_cycles = 0;
        total_fixed_pairs = 0; total_fixed_slots = 0; total_fixed_equal = 0;
        total_rqtb_pairs = 0; total_rqtb_slots = 0; total_rqtb_equal = 0;
        repeat (4) @(negedge clk);
        rst_core = 1'b0;

        for (row_index = 0; row_index < row_limit; row_index = row_index + 1) begin
            scan_count = $fscanf(fd, "%d %d %d %d %d %d",
                row_tag, stage_tag, block_tag, head_tag,
                expected_outputs_header, expected_folded_header);
            if (scan_count != 6)
                $fatal(1, "row header");
            for (token = 0; token < MAX_TOKENS; token = token + 1) begin
                scan_count = $fscanf(fd, "%h %h %h %d",
                    q_vector[token], k_vector[token],
                    peer_vector[token], expected_gate[token]);
            end
            active_count = 0;
            for (token = 0; token < MAX_TOKENS; token = token + 1)
                if (k_vector[token] != 0)
                    active_count = active_count + 1;
            if (active_count != expected_outputs_header)
                $fatal(1, "active header row=%0d", row_tag);

            fixed_count = 0; rqtb_count = 0; sb_count = 0; sb_row_cycles = 0;
            @(posedge clk);
            backpressure_reseed = 1'b1;
            @(posedge clk);
            backpressure_reseed = 1'b0;
            @(negedge clk);
            wait (sb_build_ready);
            sb_row_k_present = (active_count != 0);
            fixed_start = 1'b1; rqtb_start = 1'b1; sb_start = 1'b1;
            sb_row_active = 1'b1;
            @(negedge clk);
            fixed_start = 1'b0; rqtb_start = 1'b0; sb_start = 1'b0;
            if (active_count == 0) begin
                skip_rows = skip_rows + 1;
                fork
                    drive_fixed_pairs();
                    drive_rqtb_pairs();
                    wait_fixed_done();
                    wait_rqtb_done();
                    wait_sb_done();
                join
            end else begin
                fork
                    drive_fixed_pairs();
                    drive_rqtb_pairs();
                    drive_sb_pairs();
                    wait_fixed_done();
                    wait_rqtb_done();
                    wait_sb_done();
                join
            end
            if (fixed_error || rqtb_error || sb_error)
                $fatal(1, "protocol row=%0d", row_tag);
            if (fixed_count != active_count || rqtb_count != active_count
                || sb_count != active_count)
                $fatal(1, "count row=%0d exp=%0d fixed=%0d rqtb=%0d sb=%0d",
                    row_tag, active_count, fixed_count, rqtb_count, sb_count);
            for (index = 0; index < active_count; index = index + 1) begin
                if (fixed_token_log[index] !== rqtb_token_log[index]
                    || rqtb_token_log[index] !== sb_token_log[index]
                    || fixed_k_log[index] !== rqtb_k_log[index]
                    || rqtb_k_log[index] !== sb_k_log[index]
                    || fixed_gate_log[index] !== rqtb_gate_log[index]
                    || rqtb_gate_log[index] !== sb_gate_log[index])
                    $fatal(1, "miter row=%0d idx=%0d", row_tag, index);
                trace_index = sb_token_log[index][0]
                    ? (PAIRS + (32'(sb_token_log[index]) >> 1))
                    : (32'(sb_token_log[index]) >> 1);
                if (sb_k_log[index] !== k_vector[trace_index]
                    || sb_gate_log[index] !== expected_gate[trace_index][8:0])
                    $fatal(1, "trace row=%0d idx=%0d", row_tag, index);
            end
            if (fixed_pairs != 32'(PAIRS) || rqtb_pairs != 32'(PAIRS))
                $fatal(1, "fair pairs row=%0d fixed=%0d rqtb=%0d",
                    row_tag, fixed_pairs, rqtb_pairs);
            if (fixed_slots != (fixed_pairs << 1))
                $fatal(1, "fixed slots row=%0d slots=%0d", row_tag, fixed_slots);
            if ((rqtb_slots + rqtb_equal) != (rqtb_pairs << 1))
                $fatal(1, "rqtb slot id row=%0d slots=%0d equal=%0d",
                    row_tag, rqtb_slots, rqtb_equal);
            if (fixed_equal != rqtb_equal)
                $fatal(1, "equal miter row=%0d fixed=%0d rqtb=%0d",
                    row_tag, fixed_equal, rqtb_equal);
            total_fixed_cycles = total_fixed_cycles + fixed_cycles;
            total_rqtb_cycles = total_rqtb_cycles + rqtb_cycles;
            total_sb_cycles = total_sb_cycles + sb_row_cycles;
            total_fixed_pairs = total_fixed_pairs + fixed_pairs;
            total_fixed_slots = total_fixed_slots + fixed_slots;
            total_fixed_equal = total_fixed_equal + fixed_equal;
            total_rqtb_pairs = total_rqtb_pairs + rqtb_pairs;
            total_rqtb_slots = total_rqtb_slots + rqtb_slots;
            total_rqtb_equal = total_rqtb_equal + rqtb_equal;
            total_rows = total_rows + 1;
            $display("FAIR_ROW row=%0d active=%0d skip=%0d fixed=%0d rqtb=%0d shared=%0d fslots=%0d rslots=%0d equal=%0d",
                row_tag, active_count, active_count == 0,
                fixed_cycles, rqtb_cycles, sb_row_cycles,
                fixed_slots, rqtb_slots, rqtb_equal);
            @(negedge clk);
        end
        $fclose(fd);
        $display("FAIR_SUM rows=%0d skip=%0d fixed=%0d rqtb=%0d shared=%0d fpairs=%0d fslots=%0d fequal=%0d rpairs=%0d rslots=%0d requal=%0d",
            total_rows, skip_rows, total_fixed_cycles, total_rqtb_cycles,
            total_sb_cycles,
            total_fixed_pairs, total_fixed_slots, total_fixed_equal,
            total_rqtb_pairs, total_rqtb_slots, total_rqtb_equal);
        $display("PASS tb_h67_laws_fair_lfsr_threeway_2s");
        $finish;
    end
endmodule

`default_nettype wire
