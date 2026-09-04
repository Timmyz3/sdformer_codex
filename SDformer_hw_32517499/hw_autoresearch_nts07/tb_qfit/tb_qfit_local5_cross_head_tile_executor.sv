`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_cross_head_tile_executor;
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 32;
    localparam int HEADS = 3;
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam int TAG_W = 24;

    logic clk_core, rst_core;
    logic group_valid, group_ready;
    logic [TAG_W-1:0] group_tag;
    logic tile_start_valid, tile_start_ready;
    logic [TAG_W-1:0] tile_start_tag;
    logic [4:0] tile_start_output_tile;
    logic [5:0] tile_start_head_count;
    logic head_job_valid, head_job_ready;
    logic [TAG_W-1:0] head_job_tag;
    logic [4:0] head_job_input_head;
    logic [5:0] head_job_index;
    logic [9:0] head_job_input_channel_base;
    logic [4:0] head_job_output_tile;
    logic head_job_last_input_head, head_job_last_output_tile;
    logic head_done_valid, head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [4:0] head_done_input_head;
    logic head_done_error;
    logic tile_done_valid, tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic tile_done_error;
    logic group_done_valid, group_done_ready;
    logic [TAG_W-1:0] group_done_tag;
    logic group_done_error;
    logic scheduler_error;
    logic [31:0] scheduler_groups, scheduler_tiles, scheduler_heads;
    logic [31:0] scheduler_errors;

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
    logic [4:0] weight_req_out;
    logic weight_rsp_valid, weight_rsp_ready;
    logic [TAG_W-1:0] weight_rsp_tag;
    logic [4:0] weight_rsp_input_head;
    logic [4:0] weight_rsp_output_tile;
    logic [4:0] weight_rsp_lane;
    logic [4:0] weight_rsp_out;
    logic signed [7:0] weight_rsp_data;
    logic weight_rsp_error;
    logic tile_result_valid, tile_result_ready;
    logic [TAG_W-1:0] tile_result_tag;
    logic [4:0] tile_result_output_tile;
    logic tile_result_plane;
    logic [3:0] tile_result_y, tile_result_x;
    logic [4:0] tile_result_out;
    logic signed [31:0] tile_result_data;
    logic tile_result_last;
    logic protocol_error;
    logic [31:0] perf_tiles, perf_heads, perf_partial_results;
    logic [31:0] perf_accumulator_writes, perf_final_results;

    logic [31:0] q_mem [0:HEADS-1][0:TOTAL_TOKENS-1];
    logic [31:0] k_mem [0:HEADS-1][0:TOTAL_TOKENS-1][0:4];
    logic [4:0] mask_mem [0:HEADS-1][0:TOTAL_TOKENS-1];
    integer signed expected_mem [0:TOTAL_RESULTS-1];
    logic [15:0] lfsr_q;
    logic [15:0] service_seed;
    integer service_seed_arg;
    logic token_pending_q, weight_pending_q;
    logic [2:0] token_delay_q, weight_delay_q;
    logic [TAG_W-1:0] token_tag_q, weight_tag_q;
    logic [4:0] token_head_q, weight_head_q;
    logic [8:0] token_id_q;
    logic [4:0] weight_tile_q, weight_lane_q, weight_out_q;
    integer result_count, head_done_count, token_count, weight_count;
    integer cycle_count, result_stall_count, group_done_stall_count;
    logic group_done_seen_q;
    string input_path [0:HEADS-1];
    string expected_path [0:HEADS-1];
    string input_path_h0, input_path_h1, input_path_h2;
    string expected_path_h0, expected_path_h1, expected_path_h2;

    gatestack_output_tile_scheduler #(
        .CONTEXTS(1), .HEADS(3), .LANES(32), .TAG_W(TAG_W),
        .INPUT_CH_W(10), .OUTPUT_TILE_W(5),
        .OUTPUT_TILE_COUNT_W(6), .HEAD_COUNT_W(6),
        .CONTEXT_ID_W(1), .HEAD_ID_W(5)
    ) u_scheduler (
        .clk_core(clk_core), .rst_core(rst_core),
        .group_valid(group_valid), .group_ready(group_ready),
        .group_context_id(1'b0), .group_tag(group_tag),
        .group_head_count(6'(HEADS)), .group_first_output_tile(5'd0),
        .group_output_tile_count(6'd1),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready),
        .tile_start_tag(tile_start_tag),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_issue_valid(head_job_valid),
        .head_issue_ready(head_job_ready), .head_issue_context_id(),
        .head_issue_tag(head_job_tag),
        .head_issue_head_id(head_job_input_head),
        .head_issue_head_index(head_job_index),
        .head_issue_input_channel_base(head_job_input_channel_base),
        .head_issue_output_tile(head_job_output_tile),
        .head_issue_last_head(head_job_last_input_head),
        .head_issue_last_output_tile(head_job_last_output_tile),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_head_id(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag), .tile_done_error(tile_done_error),
        .group_done_valid(group_done_valid),
        .group_done_ready(group_done_ready), .group_done_tag(group_done_tag),
        .group_done_error(group_done_error), .protocol_error(scheduler_error),
        .count_groups(scheduler_groups),
        .count_tile_starts(scheduler_tiles),
        .count_head_issues(scheduler_heads),
        .count_group_errors(scheduler_errors)
    );

    qfit_local5_cross_head_tile_executor #(
`ifdef QFIT_SCORE_ACTIVE_FRONT
        .USE_SCORE_ACTIVE_FRONT(1'b1)
`else
        .USE_SCORE_ACTIVE_FRONT(1'b0)
`endif
    ) u_executor (
        .clk_core(clk_core), .rst_core(rst_core),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_stage(2'd0), .tile_start_block(3'd0),
        .tile_start_window(9'd0),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_job_valid(head_job_valid), .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag), .head_job_stage(2'd0),
        .head_job_block(3'd0), .head_job_window(9'd0),
        .head_job_input_head(head_job_input_head),
        .head_job_input_channel_base(head_job_input_channel_base),
        .head_job_output_tile(head_job_output_tile),
        .head_job_decode_required(head_job_output_tile == 0),
        .head_job_cache_release(head_job_last_output_tile),
        .head_job_last_input_head(head_job_last_input_head),
        .head_job_last_output_tile(head_job_last_output_tile),
        .head_done_valid(head_done_valid), .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_input_head(head_done_input_head),
        .head_done_error(head_done_error),
        .tile_done_valid(tile_done_valid), .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag), .tile_done_error(tile_done_error),
        .token_req_valid(token_req_valid), .token_req_ready(token_req_ready),
        .token_req_tag(token_req_tag),
        .token_req_input_head(token_req_input_head),
        .token_req_token_id(token_req_token_id),
        .token_req_plane(token_req_plane), .token_req_y(token_req_y),
        .token_req_x(token_req_x),
        .token_rsp_valid(token_rsp_valid), .token_rsp_ready(token_rsp_ready),
        .token_rsp_tag(token_rsp_tag),
        .token_rsp_input_head(token_rsp_input_head),
        .token_rsp_token_id(token_rsp_token_id), .token_rsp_q(token_rsp_q),
        .token_rsp_k(token_rsp_k),
        .token_rsp_valid_mask(token_rsp_valid_mask),
        .token_rsp_error(token_rsp_error),
        .weight_req_valid(weight_req_valid),
        .weight_req_ready(weight_req_ready), .weight_req_tag(weight_req_tag),
        .weight_req_input_head(weight_req_input_head),
        .weight_req_output_tile(weight_req_output_tile),
        .weight_req_lane(weight_req_lane), .weight_req_out(weight_req_out),
        .weight_rsp_valid(weight_rsp_valid),
        .weight_rsp_ready(weight_rsp_ready),
        .weight_rsp_tag(weight_rsp_tag),
        .weight_rsp_input_head(weight_rsp_input_head),
        .weight_rsp_output_tile(weight_rsp_output_tile),
        .weight_rsp_lane(weight_rsp_lane), .weight_rsp_out(weight_rsp_out),
        .weight_rsp_data(weight_rsp_data),
        .weight_rsp_error(weight_rsp_error),
        .tile_result_valid(tile_result_valid),
        .tile_result_ready(tile_result_ready),
        .tile_result_tag(tile_result_tag),
        .tile_result_output_tile(tile_result_output_tile),
        .tile_result_plane(tile_result_plane), .tile_result_y(tile_result_y),
        .tile_result_x(tile_result_x), .tile_result_out(tile_result_out),
        .tile_result_data(tile_result_data),
        .tile_result_last(tile_result_last),
        .protocol_error(protocol_error), .perf_tiles(perf_tiles),
        .perf_heads(perf_heads),
        .perf_partial_results(perf_partial_results),
        .perf_accumulator_writes(perf_accumulator_writes),
        .perf_final_results(perf_final_results)
    );

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
    assign tile_result_ready = lfsr_q[2];
    assign group_done_ready = 1'b1;
    assign token_rsp_valid = token_pending_q && token_delay_q == 0;
    assign token_rsp_tag = token_tag_q;
    assign token_rsp_input_head = token_head_q;
    assign token_rsp_token_id = token_id_q;
    assign token_rsp_q = q_mem[token_head_q][token_id_q];
    assign token_rsp_k = {
        k_mem[token_head_q][token_id_q][4],
        k_mem[token_head_q][token_id_q][3],
        k_mem[token_head_q][token_id_q][2],
        k_mem[token_head_q][token_id_q][1],
        k_mem[token_head_q][token_id_q][0]
    };
    assign token_rsp_valid_mask = mask_mem[token_head_q][token_id_q];
    assign token_rsp_error = 1'b0;
    assign weight_rsp_valid = weight_pending_q && weight_delay_q == 0;
    assign weight_rsp_tag = weight_tag_q;
    assign weight_rsp_input_head = weight_head_q;
    assign weight_rsp_output_tile = weight_tile_q;
    assign weight_rsp_lane = weight_lane_q;
    assign weight_rsp_out = weight_out_q;
    assign weight_rsp_data = 8'(weight_value(weight_lane_q, weight_out_q));
    assign weight_rsp_error = 1'b0;

    // Testbench service model: token/weight counters are initialized by the
    // stimulus process and updated here, so this is intentionally not
    // always_ff. VCS correctly rejects multiple-process variables in
    // always_ff; Verilator previously accepted this testbench idiom.
    always @(posedge clk_core) begin
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
            result_stall_count <= 0;
            group_done_stall_count <= 0;
            group_done_seen_q <= 1'b0;
        end else begin
            lfsr_q <= {lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]};
            cycle_count <= cycle_count + 1;
            if (tile_result_valid && !tile_result_ready)
                result_stall_count <= result_stall_count + 1;
            if (group_done_valid && !group_done_ready)
                group_done_stall_count <= group_done_stall_count + 1;
            if (group_done_valid && group_done_ready)
                group_done_seen_q <= 1'b1;
            if (token_pending_q && token_delay_q != 0)
                token_delay_q <= token_delay_q - 1'b1;
            if (weight_pending_q && weight_delay_q != 0)
                weight_delay_q <= weight_delay_q - 1'b1;
            if (token_req_valid && token_req_ready) begin
                token_pending_q <= 1'b1;
                token_delay_q <= 3'd1 + {1'b0, lfsr_q[5:4]};
                token_tag_q <= token_req_tag;
                token_head_q <= token_req_input_head;
                token_id_q <= token_req_token_id;
                token_count <= token_count + 1;
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
                weight_count <= weight_count + 1;
            end
            if (weight_rsp_valid && weight_rsp_ready)
                weight_pending_q <= 1'b0;
        end
    end

    always @(posedge clk_core) begin
        integer index;
        if (!rst_core && head_done_valid && head_done_ready) begin
            if (head_done_error || head_done_input_head != head_done_count)
                $fatal(1, "head completion mismatch head=%0d", head_done_count);
            head_done_count = head_done_count + 1;
        end
        if (!rst_core && tile_result_valid && tile_result_ready) begin
            index = (((tile_result_plane * HEIGHT + tile_result_y) * WIDTH
                    + tile_result_x) * OUT_DIM) + tile_result_out;
            if (index != result_count || tile_result_tag != group_tag
                || tile_result_output_tile != 0
                || $signed(tile_result_data) != expected_mem[index]
                || tile_result_last != (index == TOTAL_RESULTS - 1))
                $fatal(1, "final result mismatch index=%0d got=%0d exp=%0d",
                       index, tile_result_data, expected_mem[index]);
            result_count = result_count + 1;
        end
    end

    task automatic load_head(input integer head);
        integer input_fd, expected_fd, rc;
        integer plane, y, x, out, mask;
        integer signed value;
        logic [31:0] q_value, k0, k1, k2, k3, k4;
        begin
            input_fd = $fopen(input_path[head], "r");
            expected_fd = $fopen(expected_path[head], "r");
            if (input_fd == 0 || expected_fd == 0)
                $fatal(1, "cannot open head %0d oracle", head);
            for (int row = 0; row < TOTAL_TOKENS; row = row + 1) begin
                rc = $fscanf(input_fd,
                    "%d %d %d %h %h %h %h %h %h %h\n",
                    plane, y, x, q_value, k0, k1, k2, k3, k4, mask);
                if (rc != 10 || row != plane * 225 + y * 15 + x)
                    $fatal(1, "invalid head input row head=%0d row=%0d", head, row);
                q_mem[head][row] = q_value;
                k_mem[head][row][0] = k0;
                k_mem[head][row][1] = k1;
                k_mem[head][row][2] = k2;
                k_mem[head][row][3] = k3;
                k_mem[head][row][4] = k4;
                mask_mem[head][row] = 5'(mask);
            end
            for (int row = 0; row < TOTAL_RESULTS; row = row + 1) begin
                rc = $fscanf(expected_fd, "%d %d %d %d %d\n",
                             plane, y, x, out, value);
                if (rc != 5 || row != (((plane * HEIGHT + y) * WIDTH + x)
                                      * OUT_DIM + out))
                    $fatal(1, "invalid head expected row head=%0d row=%0d", head, row);
                expected_mem[row] = expected_mem[row] + value;
            end
            $fclose(input_fd);
            $fclose(expected_fd);
        end
    endtask

    initial begin
        integer rc;
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = 24'h5c5000;
        service_seed_arg = 16'hace1;
        if (!$value$plusargs("SERVICE_SEED=%d", service_seed_arg))
            service_seed_arg = 16'hace1;
        service_seed = service_seed_arg[15:0];
        for (int head = 0; head < HEADS; head = head + 1) begin
            input_path[head] = "";
            expected_path[head] = "";
        end
        if (!$value$plusargs("PY_INPUTS_H0=%s", input_path_h0)
            || !$value$plusargs("PY_EXPECTED_H0=%s", expected_path_h0)
            || !$value$plusargs("PY_INPUTS_H1=%s", input_path_h1)
            || !$value$plusargs("PY_EXPECTED_H1=%s", expected_path_h1)
            || !$value$plusargs("PY_INPUTS_H2=%s", input_path_h2)
            || !$value$plusargs("PY_EXPECTED_H2=%s", expected_path_h2))
            $fatal(1, "all three head oracle paths are required");
        input_path[0] = input_path_h0;
        input_path[1] = input_path_h1;
        input_path[2] = input_path_h2;
        expected_path[0] = expected_path_h0;
        expected_path[1] = expected_path_h1;
        expected_path[2] = expected_path_h2;
        for (int row = 0; row < TOTAL_RESULTS; row = row + 1)
            expected_mem[row] = 0;
        for (int head = 0; head < HEADS; head = head + 1)
            load_head(head);
        result_count = 0;
        head_done_count = 0;
        token_count = 0;
        weight_count = 0;

        repeat (6) @(posedge clk_core);
        @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        group_valid = 1'b1;
        do @(posedge clk_core); while (!group_ready);
        @(negedge clk_core);
        group_valid = 1'b0;

        wait (group_done_seen_q);
        @(posedge clk_core);
        #1;
        if (scheduler_error || protocol_error || group_done_error
            || group_done_tag != group_tag || scheduler_groups != 1
            || scheduler_tiles != 1 || scheduler_heads != HEADS
            || scheduler_errors != 0 || perf_tiles != 1
            || perf_heads != HEADS
            || perf_partial_results != HEADS * TOTAL_RESULTS
            || perf_accumulator_writes != HEADS * TOTAL_RESULTS
            || perf_final_results != TOTAL_RESULTS
            || result_count != TOTAL_RESULTS || head_done_count != HEADS
            || token_count != HEADS * TOTAL_TOKENS
            || weight_count != HEADS * HEAD_DIM * OUT_DIM
            || result_stall_count == 0)
            $fatal(1, "cross-head ledger mismatch");
        $display("PASS Local5 cross-head OUT32 seed=%0d cycles=%0d heads=%0d partial=%0d final=%0d result_stall=%0d group_stall=%0d",
                 service_seed, cycle_count, perf_heads,
                 perf_partial_results, perf_final_results,
                 result_stall_count, group_done_stall_count);
        $finish;
    end

    initial begin
        repeat (2000000) @(posedge clk_core);
        $fatal(1, "Local5 cross-head executor timeout");
    end
endmodule

`default_nettype wire
