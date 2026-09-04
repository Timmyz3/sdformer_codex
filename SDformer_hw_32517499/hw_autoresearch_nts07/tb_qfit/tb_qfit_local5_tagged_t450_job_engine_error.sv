`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_tagged_t450_job_engine_error;
    logic clk_core, rst_core;
    logic job_valid, job_ready;
    logic [23:0] job_tag;
    logic [4:0] job_input_head, job_output_tile;
    logic job_done_valid, job_done_ready;
    logic [23:0] job_done_tag;
    logic [4:0] job_done_input_head;
    logic job_done_error;
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
    logic [4:0] weight_req_input_head, weight_req_output_tile;
    logic [4:0] weight_req_lane;
    logic weight_req_out;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [23:0] weight_rsp_tag;
    logic [4:0] weight_rsp_input_head, weight_rsp_output_tile;
    logic [4:0] weight_rsp_lane;
    logic weight_rsp_out;
    logic signed [7:0] weight_rsp_data;
    logic weight_rsp_error;
    logic result_valid, result_ready;
    logic [23:0] result_tag;
    logic [4:0] result_input_head, result_output_tile;
    logic result_plane;
    logic [3:0] result_y, result_x;
    logic result_out;
    logic signed [31:0] result_data;
    logic result_last;
    logic protocol_error;
    logic [31:0] perf_jobs, perf_token_requests, perf_token_responses;
    logic [31:0] perf_weight_requests, perf_weight_responses, perf_results;
    integer frozen_token_requests;
    integer frozen_weight_requests;
    integer error_mode;

    qfit_local5_tagged_t450_job_engine dut (.*);
    always #5 clk_core = ~clk_core;

    initial begin
        clk_core = 0;
        rst_core = 1;
        job_valid = 0;
        job_tag = 24'hbad500;
        job_input_head = 5'd1;
        job_output_tile = 0;
        job_done_ready = 1;
        token_req_ready = 1;
        token_rsp_valid = 0;
        token_rsp_tag = 0;
        token_rsp_input_head = 0;
        token_rsp_token_id = 0;
        token_rsp_q = 0;
        token_rsp_k = 0;
        token_rsp_valid_mask = 5'b11111;
        token_rsp_error = 0;
        weight_req_ready = 1;
        weight_rsp_valid = 0;
        weight_rsp_tag = 0;
        weight_rsp_input_head = 0;
        weight_rsp_output_tile = 0;
        weight_rsp_lane = 0;
        weight_rsp_out = 0;
        weight_rsp_data = 1;
        weight_rsp_error = 0;
        result_ready = 1;
        error_mode = 0;
        if (!$value$plusargs("ERROR_MODE=%d", error_mode))
            error_mode = 0;
        if (error_mode < 0 || error_mode > 11)
            $fatal(1, "ERROR_MODE must be in [0,11]");
        repeat (5) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 0;
        @(negedge clk_core);
        job_valid = 1;
        do @(posedge clk_core); while (!job_ready);
        @(negedge clk_core);
        job_valid = 0;

        if (error_mode == 10) begin
            @(negedge clk_core);
            token_rsp_valid = 1;
            token_rsp_tag = job_tag;
            token_rsp_input_head = job_input_head;
            token_rsp_token_id = 0;
            @(posedge clk_core);
        end else if (error_mode >= 4) begin
            do @(posedge clk_core);
            while (!(weight_req_valid && weight_req_ready));
            @(negedge clk_core);
            weight_rsp_tag = weight_req_tag;
            weight_rsp_input_head = weight_req_input_head;
            weight_rsp_output_tile = weight_req_output_tile;
            weight_rsp_lane = weight_req_lane;
            weight_rsp_out = weight_req_out;
            case (error_mode)
                4: weight_rsp_tag = weight_req_tag ^ 24'h1;
                5: weight_rsp_input_head = weight_req_input_head ^ 5'h1;
                6: weight_rsp_output_tile = weight_req_output_tile ^ 5'h1;
                7: weight_rsp_lane = weight_req_lane ^ 5'h1;
                8: weight_rsp_out = weight_req_out ^ 1'b1;
                9: weight_rsp_error = 1;
                default: begin end
            endcase
            weight_rsp_valid = 1;
            do @(posedge clk_core); while (!weight_rsp_ready);
            if (error_mode == 11)
                @(posedge clk_core);
            else begin
                @(negedge clk_core);
                weight_rsp_valid = 0;
            end
        end else begin
            for (int index = 0; index < 64; index = index + 1) begin
                do @(posedge clk_core);
                while (!(weight_req_valid && weight_req_ready));
                @(negedge clk_core);
                weight_rsp_tag = weight_req_tag;
                weight_rsp_input_head = weight_req_input_head;
                weight_rsp_output_tile = weight_req_output_tile;
                weight_rsp_lane = weight_req_lane;
                weight_rsp_out = weight_req_out;
                weight_rsp_valid = 1;
                do @(posedge clk_core); while (!weight_rsp_ready);
                @(negedge clk_core);
                weight_rsp_valid = 0;
            end

            do @(posedge clk_core);
            while (!(token_req_valid && token_req_ready));
            @(negedge clk_core);
            token_rsp_tag = token_req_tag;
            token_rsp_input_head = token_req_input_head;
            token_rsp_token_id = token_req_token_id;
            case (error_mode)
                0: token_rsp_tag = token_req_tag ^ 24'h1;
                1: token_rsp_input_head = token_req_input_head ^ 5'h1;
                2: token_rsp_token_id = token_req_token_id ^ 9'h1;
                3: token_rsp_error = 1;
                default: begin end
            endcase
            token_rsp_valid = 1;
            do @(posedge clk_core); while (!token_rsp_ready);
            @(negedge clk_core);
            token_rsp_valid = 0;
        end
        wait (job_done_valid);
        if (!job_done_error || !protocol_error || perf_token_responses != 0
            || perf_results != 0)
            $fatal(1, "response fault mode=%0d was not fail-closed", error_mode);
        frozen_token_requests = perf_token_requests;
        frozen_weight_requests = perf_weight_requests;
        repeat (12) @(posedge clk_core);
        if (perf_token_requests != frozen_token_requests
            || perf_weight_requests != frozen_weight_requests
            || result_valid)
            $fatal(1, "error state issued new external work");
        $display(
            "PASS Local5 tagged T450 error_mode=%0d fail-closed",
            error_mode
        );
        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk_core);
        $fatal(
            1,
            "tagged T450 error TB timeout state=%0d jobs=%0d wreq=%0d wrsp=%0d treq=%0d trsp=%0d results=%0d tile_error=%0b protocol_error=%0b",
            dut.state_q, perf_jobs, perf_weight_requests,
            perf_weight_responses, perf_token_requests,
            perf_token_responses, perf_results,
            dut.tile_protocol_error, protocol_error
        );
    end
endmodule

`default_nettype wire
