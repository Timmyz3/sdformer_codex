`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_cross_head_tile_executor_error;
    logic clk_core, rst_core;
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
    logic tile_done_error;
    logic token_req_valid, token_req_ready;
    logic [23:0] token_req_tag;
    logic [4:0] token_req_input_head;
    logic [8:0] token_req_token_id;
    logic token_req_plane;
    logic [3:0] token_req_y, token_req_x;
    logic token_rsp_valid, token_rsp_ready;
    logic [23:0] token_rsp_tag;
    logic [4:0] token_rsp_input_head;
    logic [8:0] token_rsp_token_id;
    logic [31:0] token_rsp_q;
    logic [159:0] token_rsp_k;
    logic [4:0] token_rsp_valid_mask;
    logic token_rsp_error;
    logic weight_req_valid, weight_req_ready;
    logic [23:0] weight_req_tag;
    logic [4:0] weight_req_input_head;
    logic [4:0] weight_req_output_tile;
    logic [4:0] weight_req_lane, weight_req_out;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [23:0] weight_rsp_tag;
    logic [4:0] weight_rsp_input_head;
    logic [4:0] weight_rsp_output_tile;
    logic [4:0] weight_rsp_lane, weight_rsp_out;
    logic signed [7:0] weight_rsp_data;
    logic weight_rsp_error;
    logic tile_result_valid, tile_result_ready;
    logic [23:0] tile_result_tag;
    logic [4:0] tile_result_output_tile;
    logic tile_result_plane;
    logic [3:0] tile_result_y, tile_result_x;
    logic [4:0] tile_result_out;
    logic signed [31:0] tile_result_data;
    logic tile_result_last;
    logic protocol_error;
    logic [31:0] perf_tiles, perf_heads, perf_partial_results;
    logic [31:0] perf_accumulator_writes, perf_final_results;
    integer error_mode;

    qfit_local5_cross_head_tile_executor dut (.*);

    always #5 clk_core = ~clk_core;

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = 1'b0;
        tile_start_tag = 24'h5ce000;
        tile_start_stage = 2'd0;
        tile_start_block = 3'd0;
        tile_start_window = 9'd0;
        tile_start_output_tile = 5'd0;
        tile_start_head_count = 6'd3;
        head_job_valid = 1'b0;
        head_job_tag = tile_start_tag;
        head_job_stage = tile_start_stage;
        head_job_block = tile_start_block;
        head_job_window = tile_start_window;
        head_job_input_head = 5'd0;
        head_job_input_channel_base = 10'd0;
        head_job_output_tile = 5'd0;
        head_job_decode_required = 1'b1;
        head_job_cache_release = 1'b0;
        head_job_last_input_head = 1'b0;
        head_job_last_output_tile = 1'b0;
        head_done_ready = 1'b0;
        tile_done_ready = 1'b0;
        token_req_ready = 1'b0;
        token_rsp_valid = 1'b0;
        token_rsp_tag = '0;
        token_rsp_input_head = '0;
        token_rsp_token_id = '0;
        token_rsp_q = '0;
        token_rsp_k = '0;
        token_rsp_valid_mask = '0;
        token_rsp_error = 1'b0;
        weight_req_ready = 1'b0;
        weight_rsp_valid = 1'b0;
        weight_rsp_tag = '0;
        weight_rsp_input_head = '0;
        weight_rsp_output_tile = '0;
        weight_rsp_lane = '0;
        weight_rsp_out = '0;
        weight_rsp_data = '0;
        weight_rsp_error = 1'b0;
        tile_result_ready = 1'b1;
        error_mode = 0;
        if (!$value$plusargs("ERROR_MODE=%d", error_mode))
            error_mode = 0;

        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        tile_start_valid = 1'b1;
        do @(posedge clk_core); while (!tile_start_ready);
        @(negedge clk_core);
        tile_start_valid = 1'b0;
        if (error_mode == 0)
            head_job_tag = tile_start_tag ^ 24'd1;
        else
            head_job_input_head = 5'd1;
        head_job_valid = 1'b1;
        do @(posedge clk_core); while (!head_job_ready);
        @(negedge clk_core);
        head_job_valid = 1'b0;

        wait (head_done_valid);
        repeat (3) @(posedge clk_core);
        if (!head_done_valid || !head_done_error
            || head_done_tag != tile_start_tag
            || perf_partial_results != 0
            || perf_accumulator_writes != 0
            || token_req_valid || weight_req_valid || tile_result_valid)
            $fatal(1, "invalid head job was not held fail-closed");
        @(negedge clk_core);
        head_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        head_done_ready = 1'b0;
        repeat (12) @(posedge clk_core);
        if (!protocol_error || perf_tiles != 1 || perf_heads != 0
            || perf_partial_results != 0 || perf_accumulator_writes != 0
            || perf_final_results != 0 || token_req_valid
            || weight_req_valid || tile_result_valid || tile_done_valid)
            $fatal(1, "error path emitted partial work");
        $display("PASS Local5 cross-head invalid head fail-closed mode=%0d", error_mode);
        $finish;
    end

    initial begin
        repeat (1000) @(posedge clk_core);
        $fatal(1, "Local5 cross-head error timeout");
    end
endmodule

`default_nettype wire
