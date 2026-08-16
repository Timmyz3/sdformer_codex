`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_cross_head_partial_faults;
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
    integer fault_mode;

    qfit_local5_cross_head_tile_executor dut (.*);

    always #5 clk_core = ~clk_core;

    task drive_partial(
        input int flat_index,
        input bit last_value,
        input bit expect_write
    );
        int local_index;
        int plane;
        int y;
        int x;
        int out;
        begin
            local_index = flat_index;
            out = local_index % 32;
            local_index = local_index / 32;
            x = local_index % 15;
            local_index = local_index / 15;
            y = local_index % 15;
            plane = local_index / 15;
            @(negedge clk_core);
            force dut.child_result_valid = 1'b1;
            force dut.child_result_tag = tile_start_tag;
            force dut.child_result_input_head = 5'd0;
            force dut.child_result_output_tile = 5'd0;
            force dut.child_result_plane = 1'(plane);
            force dut.child_result_y = 4'(y);
            force dut.child_result_x = 4'(x);
            force dut.child_result_out = 5'(out);
            force dut.child_result_data = 32'sd17;
            force dut.child_result_last = last_value;
            #1;
            if (!dut.child_result_ready)
                $fatal(1, "partial injector did not receive ready");
            if (dut.memory_command_valid != expect_write)
                $fatal(1, "bad partial memory-command isolation failed");
            @(posedge clk_core);
            @(negedge clk_core);
            release dut.child_result_valid;
            release dut.child_result_tag;
            release dut.child_result_input_head;
            release dut.child_result_output_tile;
            release dut.child_result_plane;
            release dut.child_result_y;
            release dut.child_result_x;
            release dut.child_result_out;
            release dut.child_result_data;
            release dut.child_result_last;
        end
    endtask

    task drive_early_done;
        begin
            @(negedge clk_core);
            force dut.child_job_done_valid = 1'b1;
            force dut.child_job_done_tag = tile_start_tag;
            force dut.child_job_done_input_head = 5'd0;
            force dut.child_job_done_error = 1'b0;
            #1;
            if (!dut.child_job_done_ready || dut.memory_command_valid)
                $fatal(1, "early completion isolation failed");
            @(posedge clk_core);
            @(negedge clk_core);
            release dut.child_job_done_valid;
            release dut.child_job_done_tag;
            release dut.child_job_done_input_head;
            release dut.child_job_done_error;
        end
    endtask

    task drive_child_protocol_error;
        begin
            @(negedge clk_core);
            force dut.child_protocol_error = 1'b1;
            @(posedge clk_core);
            @(negedge clk_core);
            force dut.child_protocol_error = 1'b0;
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        tile_start_valid = 1'b0;
        tile_start_tag = 24'h5cf000;
        tile_start_stage = 2'd0;
        tile_start_block = 3'd0;
        tile_start_window = 9'd0;
        tile_start_output_tile = 5'd0;
        tile_start_head_count = 6'd1;
        head_job_valid = 1'b0;
        head_job_tag = tile_start_tag;
        head_job_stage = tile_start_stage;
        head_job_block = tile_start_block;
        head_job_window = tile_start_window;
        head_job_input_head = 5'd0;
        head_job_input_channel_base = 10'd0;
        head_job_output_tile = 5'd0;
        head_job_decode_required = 1'b1;
        head_job_cache_release = 1'b1;
        head_job_last_input_head = 1'b1;
        head_job_last_output_tile = 1'b1;
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
        fault_mode = 0;
        if (!$value$plusargs("FAULT_MODE=%d", fault_mode))
            fault_mode = 0;

        force dut.child_job_ready = 1'b1;
        force dut.child_protocol_error = 1'b0;
        force dut.child_result_valid = 1'b0;
        force dut.child_job_done_valid = 1'b0;
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        tile_start_valid = 1'b1;
        do @(posedge clk_core); while (!tile_start_ready);
        @(negedge clk_core);
        tile_start_valid = 1'b0;
        head_job_valid = 1'b1;
        do @(posedge clk_core); while (!head_job_ready);
        @(negedge clk_core);
        head_job_valid = 1'b0;

        if (fault_mode == 4) begin
            drive_child_protocol_error();
            repeat (8) @(posedge clk_core);
            if (!protocol_error || dut.tx_state_q != dut.TX_ERROR
                || head_done_valid || tile_done_valid || tile_result_valid
                || token_req_valid || weight_req_valid)
                $fatal(1, "child protocol error did not enter terminal state");
            $display("PASS Local5 child protocol error terminal fail-closed");
            release dut.child_job_ready;
            release dut.child_protocol_error;
            release dut.child_result_valid;
            release dut.child_job_done_valid;
            $finish;
        end else begin
        case (fault_mode)
            0: begin
                drive_partial(0, 1'b0, 1'b1);
                drive_partial(0, 1'b0, 1'b0);
            end
            1: drive_partial(1, 1'b0, 1'b0);
            2: drive_partial(0, 1'b1, 1'b0);
            3: begin
                drive_partial(0, 1'b0, 1'b1);
                drive_early_done();
            end
            default: $fatal(1, "unknown partial fault mode");
        endcase

        wait (head_done_valid);
        repeat (2) @(posedge clk_core);
        if (!head_done_valid || !head_done_error
            || head_done_tag != tile_start_tag
            || perf_partial_results != ((fault_mode == 0 || fault_mode == 3) ? 1 : 0)
            || perf_accumulator_writes != ((fault_mode == 0 || fault_mode == 3) ? 1 : 0)
            || tile_result_valid || tile_done_valid)
            $fatal(1, "partial fault was not held fail-closed");
        @(negedge clk_core);
        head_done_ready = 1'b1;
        @(posedge clk_core);
        @(negedge clk_core);
        head_done_ready = 1'b0;
        repeat (8) @(posedge clk_core);
        if (!protocol_error || token_req_valid || weight_req_valid
            || tile_result_valid || tile_done_valid)
            $fatal(1, "partial fault emitted work after error");
        $display(
            "PASS Local5 partial fault fail-closed mode=%0d writes=%0d",
            fault_mode, perf_accumulator_writes
        );
        release dut.child_job_ready;
        release dut.child_protocol_error;
        release dut.child_result_valid;
        release dut.child_job_done_valid;
        $finish;
        end
    end

    initial begin
        repeat (2000) @(posedge clk_core);
        $fatal(1, "Local5 partial fault timeout");
    end
endmodule

`default_nettype wire
