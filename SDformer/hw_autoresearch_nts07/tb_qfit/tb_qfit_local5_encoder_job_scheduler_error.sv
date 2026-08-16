`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_encoder_job_scheduler_error;
    logic clk_core, rst_core, start_frame, frame_busy, frame_done;
    logic tile_start_valid, tile_start_ready;
    logic [23:0] tile_start_tag;
    logic [1:0] tile_start_stage;
    logic [2:0] tile_start_block;
    logic [8:0] tile_start_window;
    logic [4:0] tile_start_output_tile;
    logic [5:0] tile_start_head_count;
    logic head_job_valid, head_job_ready;
    logic [23:0] head_job_tag;
    logic [1:0] head_job_stage;
    logic [2:0] head_job_block;
    logic [8:0] head_job_window;
    logic [4:0] head_job_input_head;
    logic [9:0] head_job_input_channel_base;
    logic [4:0] head_job_output_tile;
    logic head_job_decode_required, head_job_cache_release;
    logic head_job_last_input_head, head_job_last_output_tile;
    logic head_done_valid, head_done_ready;
    logic [23:0] head_done_tag;
    logic [4:0] head_done_input_head;
    logic head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [23:0] tile_done_tag;
    logic tile_done_error, protocol_error;
    logic [31:0] perf_window_groups, perf_output_tiles, perf_head_replays;
    logic [31:0] perf_decode_intent_jobs, perf_release_intent_jobs;
    integer observed_tile_starts;
    integer observed_head_jobs;
    integer frozen_tile_starts;
    integer frozen_head_jobs;

    qfit_local5_encoder_job_scheduler dut (.*);
    always #5 clk_core = ~clk_core;

    always @(posedge clk_core) begin
        if (rst_core) begin
            observed_tile_starts = 0;
            observed_head_jobs = 0;
        end else begin
            if (tile_start_valid && tile_start_ready)
                observed_tile_starts = observed_tile_starts + 1;
            if (head_job_valid && head_job_ready)
                observed_head_jobs = observed_head_jobs + 1;
            if (protocol_error && frame_done)
                $fatal(1, "error frame produced frame_done");
        end
    end

    task automatic apply_reset;
        begin
            rst_core = 1;
            start_frame = 0;
            head_done_valid = 0;
            tile_done_valid = 0;
            repeat (4) @(posedge clk_core);
            @(negedge clk_core);
            rst_core = 0;
        end
    endtask

    task automatic pulse_start;
        begin
            @(negedge clk_core);
            start_frame = 1;
            @(posedge clk_core);
            @(negedge clk_core);
            start_frame = 0;
        end
    endtask

    initial begin
        clk_core = 0;
        rst_core = 1;
        start_frame = 0;
        tile_start_ready = 1;
        head_job_ready = 1;
        head_done_valid = 0;
        head_done_tag = 0;
        head_done_input_head = 0;
        head_done_error = 0;
        tile_done_valid = 0;
        tile_done_tag = 0;
        tile_done_error = 0;
        apply_reset();
        pulse_start();
        wait (head_job_valid && head_job_ready);
        @(negedge clk_core);
        head_done_tag = head_job_tag ^ 24'h1;
        head_done_input_head = head_job_input_head;
        head_done_valid = 1;
        do @(posedge clk_core); while (!head_done_ready);
        @(negedge clk_core);
        head_done_valid = 0;
        wait (protocol_error);
        frozen_tile_starts = observed_tile_starts;
        frozen_head_jobs = observed_head_jobs;
        repeat (12) @(posedge clk_core);
        if (!protocol_error)
            $fatal(1, "wrong completion tag did not fail closed");
        if (observed_tile_starts != frozen_tile_starts
            || observed_head_jobs != frozen_head_jobs)
            $fatal(1, "wrong completion allowed dispatch after error");

        apply_reset();
        pulse_start();
        wait (frame_busy);
        pulse_start();
        wait (protocol_error);
        frozen_tile_starts = observed_tile_starts;
        frozen_head_jobs = observed_head_jobs;
        repeat (12) @(posedge clk_core);
        if (observed_tile_starts != frozen_tile_starts
            || observed_head_jobs != frozen_head_jobs || frame_done)
            $fatal(1, "double-start was not isolated");

        $display("PASS Local5 encoder scheduler fail-closed wrong-tag and double-start");
        $finish;
    end

    initial begin
        repeat (1000) @(posedge clk_core);
        $fatal(1, "Local5 encoder scheduler error timeout");
    end
endmodule

`default_nettype wire
