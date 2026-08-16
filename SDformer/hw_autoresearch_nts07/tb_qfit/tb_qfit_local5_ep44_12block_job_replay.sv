`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_ep44_12block_job_replay;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 2;
    localparam int GROUPS = 100;
    localparam int JOBS = 12;
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam int TOTAL_WEIGHTS = HEAD_DIM * OUT_DIM;
    localparam int TAG_W = 24;

    logic clk_core, rst_core;
    logic job_valid, job_ready;
    logic [TAG_W-1:0] job_tag;
    logic [4:0] job_input_head;
    logic [4:0] job_output_tile;
    logic job_accumulate, job_emit_results;
    logic job_done_valid, job_done_ready;
    logic [TAG_W-1:0] job_done_tag;
    logic [4:0] job_done_input_head;
    logic job_done_error;
    logic token_req_valid, token_req_ready;
    logic [TAG_W-1:0] token_req_tag;
    logic [4:0] token_req_input_head;
    logic [8:0] token_req_token_id;
    logic token_req_plane;
    logic [3:0] token_req_y, token_req_x;
    logic token_rsp_valid, token_rsp_ready;
    logic [TAG_W-1:0] token_rsp_tag;
    logic [4:0] token_rsp_input_head;
    logic [8:0] token_rsp_token_id;
    logic [31:0] token_rsp_q;
    logic [159:0] token_rsp_k;
    logic [4:0] token_rsp_valid_mask;
    logic token_rsp_error;
    logic weight_req_valid, weight_req_ready;
    logic [TAG_W-1:0] weight_req_tag;
    logic [4:0] weight_req_input_head;
    logic [4:0] weight_req_output_tile;
    logic [4:0] weight_req_lane;
    logic weight_req_out;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [4:0] weight_rsp_input_head;
    logic [4:0] weight_rsp_output_tile;
    logic [4:0] weight_rsp_lane;
    logic weight_rsp_out;
    logic signed [7:0] weight_rsp_data;
    logic weight_rsp_error;
    logic result_valid, result_ready;
    logic [TAG_W-1:0] result_tag;
    logic [4:0] result_input_head;
    logic [4:0] result_output_tile;
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

    logic [31:0] q_mem [0:GROUPS*TOTAL_TOKENS-1];
    logic [159:0] k_mem [0:GROUPS*TOTAL_TOKENS-1];
    logic [4:0] valid_mem [0:GROUPS*TOTAL_TOKENS-1];
    logic signed [7:0] weight_mem [0:GROUPS*TOTAL_WEIGHTS-1];
    logic [31:0] expected_mem [0:GROUPS*TOTAL_RESULTS-1];
    logic [6:0] selected_group [0:JOBS-1];
    logic [1:0] selected_stage [0:JOBS-1];
    logic [2:0] selected_block [0:JOBS-1];
    logic [4:0] selected_head [0:JOBS-1];
    logic [4:0] selected_output_tile [0:JOBS-1];
    logic selected_empty [0:JOBS-1];

    logic [15:0] lfsr_q;
    logic [15:0] service_seed;
    logic token_pending_q, weight_pending_q;
    logic [2:0] token_delay_q, weight_delay_q;
    logic [TAG_W-1:0] token_tag_q, weight_tag_q;
    logic [4:0] token_head_q, weight_head_q;
    logic [8:0] token_id_q;
    logic [4:0] weight_tile_q, weight_lane_q;
    logic weight_out_q;
    integer current_job;
    integer current_group;
    integer job_result_count;
    integer total_result_count;
    integer completed_jobs;
    integer cycle_count;
    integer result_stall_cycles;
    integer token_stall_cycles;
    integer weight_stall_cycles;
    string vector_dir, plan_dir;

    qfit_local5_tagged_t450_job_engine #(
        .USE_SCORE_ACTIVE_FRONT(1'b1)
    ) dut (.*);

    always #5 clk_core = ~clk_core;

    assign token_req_ready = !token_pending_q && lfsr_q[0];
    assign weight_req_ready = !weight_pending_q && lfsr_q[1];
    assign result_ready = lfsr_q[2];
    assign job_done_ready = lfsr_q[3];

    assign token_rsp_valid = token_pending_q && token_delay_q == 0;
    assign token_rsp_tag = token_tag_q;
    assign token_rsp_input_head = token_head_q;
    assign token_rsp_token_id = token_id_q;
    assign token_rsp_q = q_mem[current_group * TOTAL_TOKENS + token_id_q];
    assign token_rsp_k = k_mem[current_group * TOTAL_TOKENS + token_id_q];
    assign token_rsp_valid_mask =
        valid_mem[current_group * TOTAL_TOKENS + token_id_q];
    assign token_rsp_error = 1'b0;

    assign weight_rsp_valid = weight_pending_q && weight_delay_q == 0;
    assign weight_rsp_tag = weight_tag_q;
    assign weight_rsp_input_head = weight_head_q;
    assign weight_rsp_output_tile = weight_tile_q;
    assign weight_rsp_lane = weight_lane_q;
    assign weight_rsp_out = weight_out_q;
    assign weight_rsp_data = weight_mem[
        current_group * TOTAL_WEIGHTS + weight_lane_q * OUT_DIM + weight_out_q
    ];
    assign weight_rsp_error = 1'b0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= service_seed;
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
            cycle_count <= 0;
            result_stall_cycles <= 0;
            token_stall_cycles <= 0;
            weight_stall_cycles <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            lfsr_q <= {
                lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]
            };
            if (result_valid && !result_ready)
                result_stall_cycles <= result_stall_cycles + 1;
            if (token_req_valid && !token_req_ready)
                token_stall_cycles <= token_stall_cycles + 1;
            if (weight_req_valid && !weight_req_ready)
                weight_stall_cycles <= weight_stall_cycles + 1;
            if (token_pending_q && token_delay_q != 0)
                token_delay_q <= token_delay_q - 1'b1;
            if (weight_pending_q && weight_delay_q != 0)
                weight_delay_q <= weight_delay_q - 1'b1;

            if (token_req_valid && token_req_ready) begin
                if (token_req_token_id != 9'(
                    token_req_plane * 225 + token_req_y * 15 + token_req_x
                ))
                    $fatal(1, "token request geometry mismatch");
                token_pending_q <= 1'b1;
                token_delay_q <= 3'd1 + {1'b0, lfsr_q[5:4]};
                token_tag_q <= token_req_tag;
                token_head_q <= token_req_input_head;
                token_id_q <= token_req_token_id;
            end
            if (token_rsp_valid && token_rsp_ready)
                token_pending_q <= 1'b0;

            if (weight_req_valid && weight_req_ready) begin
                weight_pending_q <= 1'b1;
                weight_delay_q <= 3'd1 + {1'b0, lfsr_q[7:6]};
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
        integer expected_index;
        if (!rst_core && result_valid && result_ready) begin
            index = (((result_plane * HEIGHT + result_y) * WIDTH + result_x)
                    * OUT_DIM) + result_out;
            expected_index = current_group * TOTAL_RESULTS + index;
            if (
                index != job_result_count
                || result_tag != job_tag
                || result_input_head != selected_head[current_job]
                || result_output_tile != selected_output_tile[current_job]
                || $signed(result_data) != $signed(expected_mem[expected_index])
                || result_last != (index == TOTAL_RESULTS - 1)
            )
                $fatal(
                    1,
                    "result mismatch job=%0d group=%0d index=%0d got=%0d exp=%0d",
                    current_job, current_group, index, result_data,
                    $signed(expected_mem[expected_index])
                );
            job_result_count = job_result_count + 1;
            total_result_count = total_result_count + 1;
        end
        if (!rst_core && job_done_valid && job_done_ready) begin
            if (
                job_done_error
                || job_done_tag != job_tag
                || job_done_input_head != selected_head[current_job]
            )
                $fatal(1, "job completion mismatch job=%0d", current_job);
            completed_jobs = completed_jobs + 1;
        end
    end

    task automatic run_job(input integer ordinal);
        integer start_cycle;
        integer start_tokens;
        integer start_weights;
        integer start_results;
        begin
            current_job = ordinal;
            current_group = selected_group[ordinal];
            job_result_count = 0;
            start_cycle = cycle_count;
            start_tokens = perf_token_responses;
            start_weights = perf_weight_responses;
            start_results = perf_results;
            @(negedge clk_core);
            job_tag = 24'he44000 + ordinal;
            job_input_head = selected_head[ordinal];
            job_output_tile = selected_output_tile[ordinal];
            job_valid = 1'b1;
            do @(posedge clk_core); while (!job_ready);
            @(negedge clk_core);
            job_valid = 1'b0;
            wait (completed_jobs == ordinal + 1);
            @(negedge clk_core);
            if (
                job_result_count != TOTAL_RESULTS
                || perf_token_responses - start_tokens != TOTAL_TOKENS
                || perf_weight_responses - start_weights != TOTAL_WEIGHTS
                || perf_results - start_results != TOTAL_RESULTS
            )
                $fatal(1, "job ledger mismatch ordinal=%0d", ordinal);
            $display(
                "BLOCK ordinal=%0d stage=%0d block=%0d group=%0d empty=%0d cycles=%0d results=%0d",
                ordinal, selected_stage[ordinal], selected_block[ordinal],
                selected_group[ordinal], selected_empty[ordinal],
                cycle_count - start_cycle, job_result_count
            );
        end
    endtask

    initial begin
        integer ignored;
        vector_dir = "";
        plan_dir = "";
        service_seed = 16'h5a5d;
        ignored = $value$plusargs("SERVICE_SEED=%d", service_seed);
        if (
            !$value$plusargs("VECTOR_DIR=%s", vector_dir)
            || !$value$plusargs("PLAN_DIR=%s", plan_dir)
        )
            $fatal(1, "VECTOR_DIR and PLAN_DIR are required");
        $readmemh({vector_dir, "/input_q.memh"}, q_mem);
        $readmemh({vector_dir, "/input_candidate_k.memh"}, k_mem);
        $readmemh({vector_dir, "/input_valid.memh"}, valid_mem);
        $readmemh({vector_dir, "/input_weights.memh"}, weight_mem);
        $readmemh({vector_dir, "/expected_acc.memh"}, expected_mem);
        $readmemh({plan_dir, "/selected_group.memh"}, selected_group);
        $readmemh({plan_dir, "/selected_stage.memh"}, selected_stage);
        $readmemh({plan_dir, "/selected_block.memh"}, selected_block);
        $readmemh({plan_dir, "/selected_head.memh"}, selected_head);
        $readmemh(
            {plan_dir, "/selected_output_tile.memh"}, selected_output_tile
        );
        $readmemh({plan_dir, "/selected_empty.memh"}, selected_empty);

        clk_core = 1'b0;
        rst_core = 1'b1;
        job_valid = 1'b0;
        job_tag = '0;
        job_input_head = '0;
        job_output_tile = '0;
        job_accumulate = 1'b0;
        job_emit_results = 1'b1;
        result_vector_ready = 1'b0;
        current_job = 0;
        current_group = 0;
        job_result_count = 0;
        total_result_count = 0;
        completed_jobs = 0;
        repeat (6) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;

        for (int ordinal = 0; ordinal < JOBS; ordinal = ordinal + 1)
            run_job(ordinal);
        repeat (4) @(posedge clk_core);
        if (
            protocol_error
            || completed_jobs != JOBS
            || total_result_count != JOBS * TOTAL_RESULTS
            || perf_jobs != JOBS
            || perf_token_requests != JOBS * TOTAL_TOKENS
            || perf_token_responses != JOBS * TOTAL_TOKENS
            || perf_weight_requests != JOBS * TOTAL_WEIGHTS
            || perf_weight_responses != JOBS * TOTAL_WEIGHTS
            || perf_results != JOBS * TOTAL_RESULTS
            || perf_result_jobs != JOBS
            || result_stall_cycles == 0
            || token_stall_cycles == 0
            || weight_stall_cycles == 0
        )
            $fatal(1, "12-block aggregate ledger mismatch");
        $display(
            "PASS Local5 ep44 12-block tagged jobs seed=%0d cycles=%0d jobs=%0d token=%0d weight=%0d result=%0d result_stall=%0d token_stall=%0d weight_stall=%0d",
            service_seed, cycle_count, completed_jobs,
            perf_token_responses, perf_weight_responses, perf_results,
            result_stall_cycles, token_stall_cycles, weight_stall_cycles
        );
        $finish;
    end

    initial begin
        repeat (8000000) @(posedge clk_core);
        $fatal(1, "Local5 ep44 12-block tagged replay timeout");
    end
endmodule

`default_nettype wire
