`timescale 1ns/1ps
`default_nettype none

module tb_h67_real_weight_projection_all_sidecar;
    localparam int ROWS = 138;
    localparam int TOKENS = 450;
    localparam int HEAD_DIM = 32;
    localparam int OUT_CHANNELS = 16;

    logic clk;
    logic rst_core;
    logic row_start;
    logic row_done;
    logic in_fire;
    logic in_last;
    logic [HEAD_DIM-1:0] in_k_bits;
    logic [8:0] in_gate_q17;
    logic [OUT_CHANNELS*HEAD_DIM*8-1:0] weight_flat;
    logic result_valid;
    logic [OUT_CHANNELS*32-1:0] result_acc32_flat;

    logic [31:0] k_row [0:TOKENS-1];
    logic [8:0] gate_row [0:TOKENS-1];
    integer signed expected_row [0:OUT_CHANNELS-1];

    integer base_fd;
    integer batch_fd;
    integer base_rows;
    integer base_tokens;
    integer batch_rows;
    integer batch_channels;
    integer batch_id;
    integer scan_count;
    integer row;
    integer token;
    integer channel;
    integer lane;
    integer base_row_id;
    integer batch_row_id;
    integer stage_read;
    integer block_read;
    integer head_read;
    integer batch_stage;
    integer batch_block;
    integer batch_head;
    integer expected_valid;
    integer unused0;
    integer unused1;
    integer q_read;
    integer k_read;
    integer peer_read;
    integer gate_read;
    integer weight_read;
    integer expected_read;
    integer last_active;
    integer checked;
    string base_path;
    string batch_path;

    always #5 clk = ~clk;

    h67_gated_k_projection16_acc #(.OUT_CHANNELS(OUT_CHANNELS)) dut (
        .clk, .rst_core, .row_start, .row_done, .in_fire, .in_last,
        .in_k_bits, .in_gate_q17, .weight_flat, .result_valid,
        .result_acc32_flat
    );

    initial begin
        clk = 1'b0;
        rst_core = 1'b1;
        row_start = 1'b0;
        row_done = 1'b0;
        in_fire = 1'b0;
        in_last = 1'b0;
        in_k_bits = '0;
        in_gate_q17 = '0;
        weight_flat = '0;
        checked = 0;
        if (!$value$plusargs("VECTORS=%s", base_path))
            $fatal(1, "missing +VECTORS=<path>");
        if (!$value$plusargs("REALW_BATCH=%s", batch_path))
            $fatal(1, "missing +REALW_BATCH=<path>");

        base_fd = $fopen(base_path, "r");
        batch_fd = $fopen(batch_path, "r");
        if (base_fd == 0 || batch_fd == 0)
            $fatal(1, "cannot open vector inputs");
        scan_count = $fscanf(base_fd, "%d %d", base_rows, base_tokens);
        if (scan_count != 2 || base_rows != ROWS || base_tokens != TOKENS)
            $fatal(1, "invalid base header rows=%0d tokens=%0d",
                base_rows, base_tokens);
        scan_count = $fscanf(
            batch_fd, "%d %d %d", batch_rows, batch_channels, batch_id
        );
        if (scan_count != 3 || batch_rows != ROWS
            || batch_channels != OUT_CHANNELS || batch_id < 0)
            $fatal(1, "invalid batch header rows=%0d channels=%0d batch=%0d",
                batch_rows, batch_channels, batch_id);

        repeat (4) @(negedge clk);
        rst_core = 1'b0;
        for (row = 0; row < ROWS; row = row + 1) begin
            scan_count = $fscanf(base_fd, "%d %d %d %d %d %d",
                base_row_id, stage_read, block_read, head_read,
                unused0, unused1);
            if (scan_count != 6 || base_row_id != row)
                $fatal(1, "invalid base row header row=%0d", row);
            for (token = 0; token < TOKENS; token = token + 1) begin
                scan_count = $fscanf(base_fd, "%h %h %h %d",
                    q_read, k_read, peer_read, gate_read);
                if (scan_count != 4 || gate_read < 0 || gate_read > 511)
                    $fatal(1, "invalid base payload row=%0d token=%0d",
                        row, token);
                k_row[token] = k_read[31:0];
                gate_row[token] = gate_read[8:0];
            end

            scan_count = $fscanf(batch_fd, "%d %d %d %d %d",
                batch_row_id, batch_stage, batch_block, batch_head,
                expected_valid);
            if (scan_count != 5 || batch_row_id != row
                || batch_stage != stage_read || batch_block != block_read
                || batch_head != head_read || expected_valid < 0
                || expected_valid > OUT_CHANNELS)
                $fatal(1, "batch identity mismatch row=%0d", row);
            for (channel = 0; channel < OUT_CHANNELS;
                 channel = channel + 1) begin
                scan_count = $fscanf(batch_fd, "%d", expected_read);
                if (scan_count != 1)
                    $fatal(1, "invalid expected row=%0d channel=%0d",
                        row, channel);
                expected_row[channel] = expected_read;
            end
            for (lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                for (channel = 0; channel < OUT_CHANNELS;
                     channel = channel + 1) begin
                    scan_count = $fscanf(batch_fd, "%d", weight_read);
                    if (scan_count != 1 || weight_read < -128
                        || weight_read > 127) begin
                        $fatal(1, "invalid weight row=%0d channel=%0d lane=%0d",
                            row, channel, lane);
                    end
                    weight_flat[(channel*HEAD_DIM + lane)*8 +: 8]
                        = weight_read[7:0];
                end
            end

            last_active = -1;
            for (token = 0; token < TOKENS; token = token + 1)
                if (k_row[token] != 0)
                    last_active = token;
            @(negedge clk);
            row_start = 1'b1;
            @(negedge clk);
            row_start = 1'b0;
            if (last_active >= 0) begin
                for (token = 0; token < TOKENS; token = token + 1) begin
                    if (k_row[token] != 0) begin
                        in_fire = 1'b1;
                        in_last = (token == last_active);
                        in_k_bits = k_row[token];
                        in_gate_q17 = gate_row[token];
                        @(negedge clk);
                        in_fire = 1'b0;
                        in_last = 1'b0;
                        in_k_bits = '0;
                        in_gate_q17 = '0;
                    end
                end
            end else begin
                row_done = 1'b1;
                @(negedge clk);
                row_done = 1'b0;
            end
            if (!result_valid)
                $fatal(1, "missing projection result batch=%0d row=%0d",
                    batch_id, row);
            for (channel = 0; channel < OUT_CHANNELS;
                 channel = channel + 1) begin
                if ($signed(result_acc32_flat[channel*32 +: 32])
                    != expected_row[channel])
                    $fatal(1, "all-output Acc32 mismatch batch=%0d row=%0d channel=%0d expected=%0d actual=%0d",
                        batch_id, row, channel, expected_row[channel],
                        $signed(result_acc32_flat[channel*32 +: 32]));
            end
            checked = checked + expected_valid;
            $display("REALWALL_ROW batch=%0d row=%0d stage=%0d block=%0d head=%0d valid=%0d",
                batch_id, row, stage_read, block_read, head_read,
                expected_valid);
        end
        $fclose(base_fd);
        $fclose(batch_fd);
        $display("PASS H67 real-weight all-output sidecar batch=%0d rows=%0d valid=%0d mismatch=0",
            batch_id, ROWS, checked);
        $finish;
    end
endmodule

`default_nettype wire
