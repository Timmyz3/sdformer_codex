`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_tagged_t450_job_engine;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 2;
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam int TAG_W = 24;
    localparam int HEAD_W = 5;
    localparam int OUTPUT_TILE_W = 5;
    localparam int TOKEN_ID_W = 9;

    logic clk_core, rst_core;
    logic job_valid, job_ready;
    logic [TAG_W-1:0] job_tag;
    logic [HEAD_W-1:0] job_input_head;
    logic [OUTPUT_TILE_W-1:0] job_output_tile;
    logic job_accumulate, job_emit_results;
    logic job_done_valid, job_done_ready;
    logic [TAG_W-1:0] job_done_tag;
    logic [HEAD_W-1:0] job_done_input_head;
    logic job_done_error;
    logic token_req_valid, token_req_ready;
    logic [TAG_W-1:0] token_req_tag;
    logic [HEAD_W-1:0] token_req_input_head;
    logic [TOKEN_ID_W-1:0] token_req_token_id;
    logic token_req_plane;
    logic [3:0] token_req_y, token_req_x;
    logic token_rsp_valid, token_rsp_ready;
    logic [TAG_W-1:0] token_rsp_tag;
    logic [HEAD_W-1:0] token_rsp_input_head;
    logic [TOKEN_ID_W-1:0] token_rsp_token_id;
    logic [31:0] token_rsp_q;
    logic [159:0] token_rsp_k;
    logic [4:0] token_rsp_valid_mask;
    logic token_rsp_error;
    logic weight_req_valid, weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [HEAD_W-1:0] weight_req_input_head;
    logic [OUTPUT_TILE_W-1:0] weight_req_output_tile;
    logic [4:0] weight_req_lane;
    logic weight_req_out;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [HEAD_W-1:0] weight_rsp_input_head;
    logic [OUTPUT_TILE_W-1:0] weight_rsp_output_tile;
    logic [4:0] weight_rsp_lane;
    logic weight_rsp_out;
    logic signed [7:0] weight_rsp_data;
    logic weight_rsp_error;
    logic result_valid, result_ready;
    logic [TAG_W-1:0] result_tag;
    logic [HEAD_W-1:0] result_input_head;
    logic [OUTPUT_TILE_W-1:0] result_output_tile;
    logic result_plane;
    logic [3:0] result_y, result_x;
    logic result_out;
    logic signed [31:0] result_data;
    logic result_last;
    logic result_vector_valid, result_vector_ready;
    logic [OUT_DIM*32-1:0] result_vector_data;
    logic protocol_error;
    logic [31:0] perf_jobs, perf_token_requests, perf_token_responses;
    logic [31:0] perf_weight_requests, perf_weight_responses, perf_results;
    logic [31:0] perf_result_jobs;

    logic [31:0] q_mem [0:TOTAL_TOKENS-1];
    logic [31:0] k_mem [0:TOTAL_TOKENS-1][0:4];
    logic [4:0] mask_mem [0:TOTAL_TOKENS-1];
    integer signed expected_mem [0:TOTAL_RESULTS-1];

    logic [15:0] lfsr_q;
    logic [15:0] service_seed;
    logic [1:0] done_hold_q;
    logic token_pending_q, weight_pending_q;
    logic [2:0] token_delay_q, weight_delay_q;
    logic [TAG_W-1:0] token_tag_q, weight_tag_q;
    logic [HEAD_W-1:0] token_head_q, weight_head_q;
    logic [TOKEN_ID_W-1:0] token_id_q;
    logic [OUTPUT_TILE_W-1:0] weight_tile_q;
    logic [4:0] weight_lane_q;
    logic weight_out_q;
    integer result_count;
    integer completed_jobs;
    integer cycle_count;
    integer done_stall_cycles;
    string inputs_path;
    string expected_path;

    qfit_local5_tagged_t450_job_engine #(
`ifdef QFIT_SCORE_ACTIVE_FRONT
        .USE_SCORE_ACTIVE_FRONT(1'b1)
`else
        .USE_SCORE_ACTIVE_FRONT(1'b0)
`endif
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    function automatic integer signed weight_value(
        input integer lane,
        input integer out
    );
        weight_value = (((lane + 1) * 37 + (out + 1) * 53
                      + lane * out * 11) % 127) - 63;
    endfunction

    assign token_req_ready = !token_pending_q && lfsr_q[0];
    assign weight_req_ready = !weight_pending_q && lfsr_q[1];
    assign result_ready = lfsr_q[2];
    assign job_done_ready = !job_done_valid || done_hold_q == 2;

    assign token_rsp_valid = token_pending_q && token_delay_q == 0;
    assign token_rsp_tag = token_tag_q;
    assign token_rsp_input_head = token_head_q;
    assign token_rsp_token_id = token_id_q;
    assign token_rsp_q = q_mem[token_id_q];
    assign token_rsp_k = {
        k_mem[token_id_q][4], k_mem[token_id_q][3],
        k_mem[token_id_q][2], k_mem[token_id_q][1],
        k_mem[token_id_q][0]
    };
    assign token_rsp_valid_mask = mask_mem[token_id_q];
    assign token_rsp_error = 1'b0;

    assign weight_rsp_valid = weight_pending_q && weight_delay_q == 0;
    assign weight_rsp_tag = weight_tag_q;
    assign weight_rsp_input_head = weight_head_q;
    assign weight_rsp_output_tile = weight_tile_q;
    assign weight_rsp_lane = weight_lane_q;
    assign weight_rsp_out = weight_out_q;
    assign weight_rsp_data = 8'(
        (weight_tile_q == 0 ? 1 : -1)
        * weight_value(weight_lane_q, weight_out_q)
    );
    assign weight_rsp_error = 1'b0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= service_seed;
            cycle_count <= 0;
            done_stall_cycles <= 0;
            done_hold_q <= '0;
            token_pending_q <= 1'b0;
            weight_pending_q <= 1'b0;
            token_delay_q <= '0;
            weight_delay_q <= '0;
            token_tag_q <= '0;
            token_head_q <= '0;
            token_id_q <= '0;
            weight_tag_q <= '0;
            weight_head_q <= '0;
            weight_tile_q <= '0;
            weight_lane_q <= '0;
            weight_out_q <= '0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (job_done_valid && !job_done_ready)
                done_stall_cycles <= done_stall_cycles + 1;
            if (job_done_valid && !job_done_ready)
                done_hold_q <= done_hold_q + 1'b1;
            else if (job_done_valid && job_done_ready)
                done_hold_q <= '0;
            lfsr_q <= {
                lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]
            };
            if (token_pending_q && token_delay_q != 0)
                token_delay_q <= token_delay_q - 1'b1;
            if (weight_pending_q && weight_delay_q != 0)
                weight_delay_q <= weight_delay_q - 1'b1;

            if (token_req_valid && token_req_ready) begin
                if (token_req_token_id != TOKEN_ID_W'(
                    token_req_plane * 225 + token_req_y * 15 + token_req_x
                ))
                    $fatal(1, "token request geometry mismatch");
                token_pending_q <= 1'b1;
                token_delay_q <= 3'd1 + {1'b0, lfsr_q[4:3]};
                token_tag_q <= token_req_tag;
                token_head_q <= token_req_input_head;
                token_id_q <= token_req_token_id;
            end
            if (token_rsp_valid && token_rsp_ready)
                token_pending_q <= 1'b0;

            if (weight_req_valid && weight_req_ready) begin
                weight_pending_q <= 1'b1;
                weight_delay_q <= 3'd1 + {1'b0, lfsr_q[6:5]};
                weight_tag_q <= weight_req_tag;
                weight_head_q <= weight_req_input_head;
                weight_tile_q <= weight_req_output_tile;
                weight_lane_q <= weight_req_lane;
                weight_out_q <= weight_req_out;
            end
            if (weight_rsp_valid && weight_rsp_ready)
                weight_pending_q <= 1'b0;
        end
    end

    always @(posedge clk_core) begin
        integer index;
        integer signed expected_value;
        if (!rst_core && result_valid && result_ready) begin
            index = (((result_plane * HEIGHT + result_y) * WIDTH
                    + result_x) * OUT_DIM) + result_out;
            expected_value = expected_mem[index];
            if (result_output_tile != 0)
                expected_value = -expected_value;
            if (result_tag != (result_output_tile == 0
                    ? 24'h51a000 : 24'h51a001)
                || result_input_head != 5'd2
                || $signed(result_data) != expected_value
                || result_last != (index == TOTAL_RESULTS - 1))
                $fatal(
                    1,
                    "result mismatch tile=%0d index=%0d got=%0d exp=%0d",
                    result_output_tile, index, result_data, expected_value
                );
            result_count = result_count + 1;
        end
        if (!rst_core && job_done_valid && job_done_ready) begin
            if (job_done_error || job_done_input_head != 2)
                $fatal(1, "job completion error");
            completed_jobs = completed_jobs + 1;
        end
    end

    task automatic run_job(
        input logic [TAG_W-1:0] tag,
        input logic [OUTPUT_TILE_W-1:0] output_tile
    );
        begin
            @(negedge clk_core);
            job_tag = tag;
            job_input_head = 5'd2;
            job_output_tile = output_tile;
            job_valid = 1'b1;
            do @(posedge clk_core); while (!job_ready);
            @(negedge clk_core);
            job_valid = 1'b0;
            do @(posedge clk_core);
            while (!(job_done_valid && job_done_ready));
            if (job_done_tag != tag || job_done_error)
                $fatal(1, "job done identity mismatch");
        end
    endtask

    initial begin
        integer input_fd;
        integer expected_fd;
        integer rc;
        integer plane, y, x, out;
        integer mask;
        integer signed value;
        logic [31:0] q_value;
        logic [31:0] k0, k1, k2, k3, k4;
        inputs_path = "";
        expected_path = "";
        service_seed = 16'hace1;
        rc = $value$plusargs("SERVICE_SEED=%d", service_seed);
        if (!$value$plusargs("PY_INPUTS=%s", inputs_path)
            || !$value$plusargs("PY_EXPECTED=%s", expected_path))
            $fatal(1, "PY_INPUTS/PY_EXPECTED are required");
        input_fd = $fopen(inputs_path, "r");
        expected_fd = $fopen(expected_path, "r");
        if (input_fd == 0 || expected_fd == 0)
            $fatal(1, "cannot open Python oracle");
        for (int row = 0; row < TOTAL_TOKENS; row = row + 1) begin
            rc = $fscanf(
                input_fd, "%d %d %d %h %h %h %h %h %h %h\n",
                plane, y, x, q_value, k0, k1, k2, k3, k4, mask
            );
            if (rc != 10 || row != plane * 225 + y * 15 + x)
                $fatal(1, "invalid Python input row=%0d rc=%0d", row, rc);
            q_mem[row] = q_value;
            k_mem[row][0] = k0;
            k_mem[row][1] = k1;
            k_mem[row][2] = k2;
            k_mem[row][3] = k3;
            k_mem[row][4] = k4;
            mask_mem[row] = 5'(mask);
        end
        for (int row = 0; row < TOTAL_RESULTS; row = row + 1) begin
            rc = $fscanf(expected_fd, "%d %d %d %d %d\n",
                         plane, y, x, out, value);
            if (rc != 5
                || row != (((plane * HEIGHT + y) * WIDTH + x)
                           * OUT_DIM + out))
                $fatal(1, "invalid Python expected row=%0d rc=%0d", row, rc);
            expected_mem[row] = value;
        end
        $fclose(input_fd);
        $fclose(expected_fd);

        clk_core = 1'b0;
        rst_core = 1'b1;
        job_valid = 1'b0;
        job_tag = '0;
        job_input_head = '0;
        job_output_tile = '0;
        job_accumulate = 1'b0;
        job_emit_results = 1'b1;
        result_count = 0;
        result_vector_ready = 1'b0;
        completed_jobs = 0;
        repeat (6) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        run_job(24'h51a000, 5'd0);
        run_job(24'h51a001, 5'd1);
        repeat (4) @(posedge clk_core);
        if (protocol_error || completed_jobs != 2
            || result_count != 2 * TOTAL_RESULTS
            || perf_jobs != 2
            || perf_token_requests != 2 * TOTAL_TOKENS
            || perf_token_responses != 2 * TOTAL_TOKENS
            || perf_weight_requests != 2 * HEAD_DIM * OUT_DIM
            || perf_weight_responses != 2 * HEAD_DIM * OUT_DIM
            || perf_results != 2 * TOTAL_RESULTS
            || done_stall_cycles == 0)
            $fatal(
                1,
                "job ledger mismatch completed=%0d jobs=%0d token=%0d/%0d weight=%0d/%0d result=%0d done_stall=%0d",
                completed_jobs,
                perf_jobs, perf_token_requests, perf_token_responses,
                perf_weight_requests, perf_weight_responses, perf_results,
                done_stall_cycles
            );
        $display(
            "PASS Local5 tagged T450 seed=%0d cycles=%0d jobs=2 token=%0d weight=%0d result=%0d done_stall=%0d",
            service_seed, cycle_count, perf_token_responses,
            perf_weight_responses, perf_results, done_stall_cycles
        );
        $finish;
    end

    initial begin
        repeat (600000) @(posedge clk_core);
        $fatal(
            1,
            "Local5 tagged T450 job engine timeout state=%0d jobs=%0d token=%0d/%0d weight=%0d/%0d result=%0d pending=%0b/%0b tile_ready=%0b/%0b/%0b/%0b tile_err=%0b",
            dut.state_q, perf_jobs, perf_token_requests,
            perf_token_responses, perf_weight_requests,
            perf_weight_responses, perf_results,
            token_pending_q, weight_pending_q,
            dut.tile_weight_ready, dut.tile_projection_start_ready,
            dut.tile_plane_start_ready, dut.tile_read_ready,
            dut.tile_protocol_error
        );
    end
endmodule

`default_nettype wire
