`timescale 1ns/1ps
`default_nettype none

module tb_qfit_local5_memo_multitile_cross_head #(
    parameter bit USE_MEMO = 1'b1,
    parameter bit USE_INPLACE = 1'b0,
    parameter int ACC_BACKEND_KIND = 0,
    parameter bit TRANSACTION_INDEXED_SERVICE = 1'b0,
    parameter bit IDENTITY_DERIVED_SERVICE = 1'b0,
    parameter int FORCE_WEIGHT_RESPONSE_HOLD_CYCLES = 0,
    parameter bit VECTOR_RESULT_MODE = 1'b0,
    parameter int HEADS = 3,
    parameter int OUTPUT_TILES = HEADS,
    parameter int STAGE_ID = 0,
    parameter int BLOCK_ID = 0,
    parameter int WINDOW_ID = 0,
    parameter int TIMEOUT_CYCLES = 1500000
);
    localparam int HEIGHT = 15;
    localparam int WIDTH = 15;
    localparam int TIME_PLANES = 2;
    localparam int HEAD_DIM = 32;
    localparam int OUT_DIM = 32;
    localparam int TOTAL_TOKENS = HEIGHT * WIDTH * TIME_PLANES;
    localparam int TOTAL_RESULTS = TOTAL_TOKENS * OUT_DIM;
    localparam int TAG_W = 24;
    localparam int RELATION_TABLE_ENTRIES = HEADS * TOTAL_TOKENS;
    localparam int WEIGHT_TABLE_ENTRIES =
        OUTPUT_TILES * HEADS * HEAD_DIM * OUT_DIM;
    localparam int FINAL_TABLE_ENTRIES = OUTPUT_TILES * TOTAL_RESULTS;
    localparam bit DETERMINISTIC_SERVICE =
        TRANSACTION_INDEXED_SERVICE || IDENTITY_DERIVED_SERVICE;

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
    logic weight_service_valid, weight_service_ready;
    logic dut_weight_rsp_valid, dut_weight_rsp_ready;
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
    logic signed [7:0] checkpoint_weight_mem
        [0:HEADS-1][0:OUTPUT_TILES-1][0:HEAD_DIM-1][0:OUT_DIM-1];
    integer signed expected_mem [0:OUTPUT_TILES-1][0:TOTAL_RESULTS-1];
    logic [1:0] relation_delay_mem [0:RELATION_TABLE_ENTRIES-1];
    logic [1:0] weight_delay_mem [0:WEIGHT_TABLE_ENTRIES-1];
    logic [1:0] final_delay_mem [0:FINAL_TABLE_ENTRIES-1];
    logic [15:0] lfsr_q;
    logic [15:0] service_seed;
    integer service_seed_arg;
    logic token_pending_q, weight_pending_q;
    logic [2:0] token_delay_q, weight_delay_q;
    logic [2:0] result_wait_q;
    logic [2:0] result_assigned_delay_q;
    logic [TAG_W-1:0] token_tag_q, weight_tag_q;
    logic [4:0] token_head_q, weight_head_q;
    logic [8:0] token_id_q;
    logic [4:0] weight_tile_q, weight_lane_q, weight_out_q;
    integer result_in_tile, completed_tiles, head_done_count;
    integer token_count, weight_count, cycle_count, result_stall_count;
    integer result_service_count;
    integer token_table_index_q, weight_table_index_q;
    integer token_accept_cycle_q, weight_accept_cycle_q;
    integer result_present_cycle_q;
    logic [2:0] token_assigned_delay_q, weight_assigned_delay_q;
    logic [2:0] weight_hold_q;
    logic token_response_seen_q, weight_response_seen_q;
    logic result_present_seen_q;
    logic [235:0] token_response_payload_q;
    logic [52:0] weight_response_payload_q;
    logic [75:0] final_response_payload_q;
    longint unsigned token_delay_sum, weight_delay_sum;
    logic [63:0] token_service_hash, weight_service_hash;
    logic [63:0] result_service_hash;
    integer phase_weight_cycles, phase_frontend_cycles;
    integer phase_readout_cycles, phase_release_cycles;
    integer phase_cross_rmw_cycles, phase_external_drain_cycles;
    integer phase_scheduler_cycles;
    integer previous_tx_state, previous_acc_state, previous_head_state;
    logic group_done_seen_q;
    string input_path_h0, input_path_h1, input_path_h2, expected_path;
    string combined_input_path;
    string weights_path, actual_path;
    string relation_delay_path, weight_delay_path, final_delay_path;
    string identity_trace_path, identity_manifest_sha, identity_receipt_sha;
    string input_path [0:HEADS-1];
    integer actual_fd, identity_trace_fd;
    logic no_acc_check, use_checkpoint_weights, use_combined_inputs;
    integer run_stage_id, run_block_id, run_window_id;

    initial begin
        run_stage_id = STAGE_ID;
        run_block_id = BLOCK_ID;
        run_window_id = WINDOW_ID;
        if (!$value$plusargs("STAGE_ID=%d", run_stage_id))
            run_stage_id = STAGE_ID;
        if (!$value$plusargs("BLOCK_ID=%d", run_block_id))
            run_block_id = BLOCK_ID;
        if (!$value$plusargs("WINDOW_ID=%d", run_window_id))
            run_window_id = WINDOW_ID;
        if (HEADS < 1 || HEADS > 32)
            $fatal(1, "HEADS must be in [1,32]");
        if (OUTPUT_TILES != HEADS)
            $fatal(1, "formal cross-head TB requires OUTPUT_TILES=HEADS");
        if (run_stage_id < 0 || run_stage_id > 3
            || run_block_id < 0
            || run_block_id >= (run_stage_id == 2 ? 6 : 2)
            || run_window_id < 0 || run_window_id >= 512)
            $fatal(1, "formal stage/block/window parameter is out of range");
        case (run_stage_id)
            0: if (HEADS != 3) $fatal(1, "stage0 requires HEADS=3");
            1: if (HEADS != 6) $fatal(1, "stage1 requires HEADS=6");
            2: if (HEADS != 12) $fatal(1, "stage2 requires HEADS=12");
            3: if (HEADS != 24) $fatal(1, "stage3 requires HEADS=24");
            default: $fatal(1, "invalid stage");
        endcase
        if (TIMEOUT_CYCLES <= 0)
            $fatal(1, "TIMEOUT_CYCLES must be positive");
        if (FORCE_WEIGHT_RESPONSE_HOLD_CYCLES < 0
            || FORCE_WEIGHT_RESPONSE_HOLD_CYCLES > 7)
            $fatal(1, "FORCE_WEIGHT_RESPONSE_HOLD_CYCLES must be in [0,7]");
        if (TRANSACTION_INDEXED_SERVICE && IDENTITY_DERIVED_SERVICE)
            $fatal(1, "transaction-indexed and identity-derived service are exclusive");
        if (IDENTITY_DERIVED_SERVICE && USE_MEMO)
            $fatal(1, "identity-derived service currently requires USE_MEMO=0");
    end

    gatestack_output_tile_scheduler #(
        .CONTEXTS(1), .HEADS(HEADS), .LANES(32), .TAG_W(TAG_W),
        .INPUT_CH_W(10), .OUTPUT_TILE_W(5),
        .OUTPUT_TILE_COUNT_W(6), .HEAD_COUNT_W(6),
        .CONTEXT_ID_W(1), .HEAD_ID_W(5)
    ) u_scheduler (
        .clk_core(clk_core), .rst_core(rst_core),
        .group_valid(group_valid), .group_ready(group_ready),
        .group_context_id(1'b0), .group_tag(group_tag),
        .group_head_count(6'(HEADS)), .group_first_output_tile(5'd0),
        .group_output_tile_count(6'(OUTPUT_TILES)),
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
        .USE_RELATION_MEMO(USE_MEMO),
        .USE_INPLACE_CROSS_HEAD_ACC(USE_INPLACE),
        .VECTOR_RESULT_MODE(VECTOR_RESULT_MODE),
        .ACC_BACKEND_KIND(ACC_BACKEND_KIND)
    ) u_executor (
        .clk_core(clk_core), .rst_core(rst_core),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(tile_start_ready), .tile_start_tag(tile_start_tag),
        .tile_start_stage(2'(run_stage_id)),
        .tile_start_block(3'(run_block_id)),
        .tile_start_window(9'(run_window_id)),
        .tile_start_output_tile(tile_start_output_tile),
        .tile_start_head_count(tile_start_head_count),
        .head_job_valid(head_job_valid), .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag), .head_job_stage(2'(run_stage_id)),
        .head_job_block(3'(run_block_id)),
        .head_job_window(9'(run_window_id)),
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
        .weight_rsp_valid(dut_weight_rsp_valid),
        .weight_rsp_ready(dut_weight_rsp_ready),
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
        input integer head,
        input integer tile,
        input integer lane,
        input integer out
    );
        weight_value = (((head + 1) * 29 + (tile + 1) * 43
                      + (lane + 1) * 37 + (out + 1) * 53
                      + lane * out * 11) % 127) - 63;
    endfunction

    function automatic logic [2:0] transaction_delay(
        input integer stream,
        input integer transaction
    );
        integer unsigned mixed;
        begin
            mixed = 32'(service_seed)
                  ^ (32'(transaction + 1) * 32'h9e37_79b9)
                  ^ (32'(stream + 1) * 32'h7f4a_7c15);
            mixed = mixed ^ (mixed >> 16);
            mixed = mixed * 32'h45d9_f3b;
            mixed = mixed ^ (mixed >> 16);
            transaction_delay = 3'(1 + (mixed & 3));
        end
    endfunction

    function automatic integer relation_table_index(
        input integer input_head,
        input integer source_id
    );
        relation_table_index = input_head * TOTAL_TOKENS + source_id;
    endfunction

    function automatic integer weight_table_index(
        input integer output_tile,
        input integer input_head,
        input integer lane,
        input integer out
    );
        weight_table_index =
            (((output_tile * HEADS + input_head) * HEAD_DIM + lane)
             * OUT_DIM) + out;
    endfunction

    function automatic integer final_table_index(
        input integer output_tile,
        input integer source_id,
        input integer out
    );
        final_table_index =
            (output_tile * TOTAL_TOKENS + source_id) * OUT_DIM + out;
    endfunction

    function automatic logic [63:0] service_hash_next(
        input logic [63:0] previous,
        input logic [63:0] identity,
        input integer transaction,
        input logic [2:0] delay
    );
        logic [63:0] mixed;
        begin
            mixed = identity
                  ^ (64'(transaction + 1) * 64'h9e37_79b9_7f4a_7c15)
                  ^ (64'(delay) << 57);
            service_hash_next = (previous ^ mixed) * 64'h0000_0100_0000_01b3;
        end
    endfunction

    assign token_req_ready = !token_pending_q
                           && (DETERMINISTIC_SERVICE || lfsr_q[0]);
    assign weight_req_ready = !weight_pending_q
                            && (DETERMINISTIC_SERVICE || lfsr_q[1]);
    assign tile_result_ready = DETERMINISTIC_SERVICE
                             ? result_wait_q == 0 : lfsr_q[2];
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
    assign weight_service_valid = weight_pending_q && weight_delay_q == 0;
    assign weight_rsp_valid = weight_service_valid;
    assign weight_rsp_ready = dut_weight_rsp_ready && weight_hold_q == 0;
    assign weight_service_ready = weight_rsp_ready;
    assign dut_weight_rsp_valid = weight_rsp_valid && weight_hold_q == 0;
    assign weight_rsp_tag = weight_tag_q;
    assign weight_rsp_input_head = weight_head_q;
    assign weight_rsp_output_tile = weight_tile_q;
    assign weight_rsp_lane = weight_lane_q;
    assign weight_rsp_out = weight_out_q;
    assign weight_rsp_data = use_checkpoint_weights
        ? checkpoint_weight_mem[
            weight_head_q][weight_tile_q][weight_lane_q][weight_out_q]
        : 8'(weight_value(
            weight_head_q, weight_tile_q, weight_lane_q, weight_out_q
        ));
    assign weight_rsp_error = 1'b0;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            lfsr_q <= service_seed;
            token_pending_q <= 1'b0;
            weight_pending_q <= 1'b0;
            token_delay_q <= '0;
            weight_delay_q <= '0;
            weight_hold_q <= '0;
            result_wait_q <= IDENTITY_DERIVED_SERVICE
                ? 3'(final_delay_mem[0]) + 3'd1 : transaction_delay(2, 0);
            result_assigned_delay_q <= IDENTITY_DERIVED_SERVICE
                ? {1'b0, final_delay_mem[0]} : transaction_delay(2, 0);
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
            result_service_count <= 0;
            token_table_index_q <= -1;
            weight_table_index_q <= -1;
            token_accept_cycle_q <= -1;
            weight_accept_cycle_q <= -1;
            result_present_cycle_q <= -1;
            token_assigned_delay_q <= '0;
            weight_assigned_delay_q <= '0;
            token_response_seen_q <= 1'b0;
            weight_response_seen_q <= 1'b0;
            result_present_seen_q <= 1'b0;
            token_response_payload_q <= '0;
            weight_response_payload_q <= '0;
            final_response_payload_q <= '0;
            token_delay_sum <= 0;
            weight_delay_sum <= 0;
            token_service_hash <= 64'hcbf2_9ce4_8422_2325;
            weight_service_hash <= 64'hcbf2_9ce4_8422_2325;
            result_service_hash <= 64'hcbf2_9ce4_8422_2325;
            group_done_seen_q <= 1'b0;
            phase_cross_rmw_cycles <= 0;
            phase_external_drain_cycles <= 0;
            phase_scheduler_cycles <= 0;
        end else begin
            lfsr_q <= {lfsr_q[14:0],
                lfsr_q[15] ^ lfsr_q[13] ^ lfsr_q[12] ^ lfsr_q[10]};
            cycle_count <= cycle_count + 1;
            if (tile_result_valid && !tile_result_ready)
                result_stall_count <= result_stall_count + 1;
            if (DETERMINISTIC_SERVICE
                && tile_result_valid && !tile_result_ready
                && result_wait_q != 0)
                result_wait_q <= result_wait_q - 1'b1;
            if (IDENTITY_DERIVED_SERVICE && token_response_seen_q
                && (!token_rsp_valid
                    || {token_rsp_tag, token_rsp_input_head,
                        token_rsp_token_id, token_rsp_q, token_rsp_k,
                        token_rsp_valid_mask, token_rsp_error}
                       != token_response_payload_q))
                $fatal(1, "identity relation response changed under backpressure");
            if (IDENTITY_DERIVED_SERVICE && weight_response_seen_q
                && (!weight_service_valid
                    || {weight_rsp_tag, weight_rsp_input_head,
                        weight_rsp_output_tile, weight_rsp_lane,
                        weight_rsp_out, weight_rsp_data, weight_rsp_error}
                       != weight_response_payload_q))
                $fatal(1, "identity weight response changed under backpressure");
            if (IDENTITY_DERIVED_SERVICE && result_present_seen_q
                && (!tile_result_valid
                    || {tile_result_tag, tile_result_output_tile,
                        tile_result_plane, tile_result_y, tile_result_x,
                        tile_result_out, tile_result_data, tile_result_last}
                       != final_response_payload_q))
                $fatal(1, "identity final response changed under backpressure");
            if (IDENTITY_DERIVED_SERVICE && tile_result_valid
                && !result_present_seen_q) begin
                integer source_id;
                integer expected_index;
                source_id = ((32'(tile_result_plane) * HEIGHT
                              + 32'(tile_result_y)) * WIDTH)
                            + 32'(tile_result_x);
                expected_index = final_table_index(
                    tile_result_output_tile, source_id, tile_result_out
                );
                if (expected_index != result_service_count)
                    $fatal(1, "identity final request order mismatch");
                result_present_seen_q <= 1'b1;
                result_present_cycle_q <= cycle_count;
                final_response_payload_q <= {
                    tile_result_tag, tile_result_output_tile,
                    tile_result_plane, tile_result_y, tile_result_x,
                    tile_result_out, tile_result_data, tile_result_last
                };
                if (identity_trace_fd != 0)
                    $fwrite(identity_trace_fd,
                        "%0d,final_request,%0d,-1,%0d,-1,%0d,%0d,%0d,rtl_handshake,%h\n",
                        cycle_count, tile_result_output_tile, source_id,
                        tile_result_out, result_assigned_delay_q,
                        result_service_count,
                        {tile_result_tag, tile_result_output_tile,
                         tile_result_plane, tile_result_y, tile_result_x,
                         tile_result_out, tile_result_data, tile_result_last});
            end
            if (DETERMINISTIC_SERVICE
                && tile_result_valid && tile_result_ready) begin
                integer source_id;
                integer expected_index;
                source_id = ((32'(tile_result_plane) * HEIGHT
                              + 32'(tile_result_y)) * WIDTH)
                            + 32'(tile_result_x);
                expected_index = final_table_index(
                    tile_result_output_tile, source_id, tile_result_out
                );
                if (IDENTITY_DERIVED_SERVICE) begin
                    if (expected_index != result_service_count)
                        $fatal(1,
                            "identity final order mismatch got=%0d expected=%0d",
                            expected_index, result_service_count);
                    if (!result_present_seen_q
                        || cycle_count - result_present_cycle_q
                           != 32'(result_assigned_delay_q) + 1)
                        $fatal(1, "identity final response latency mismatch");
                    if (identity_trace_fd != 0)
                        $fwrite(identity_trace_fd,
                            "%0d,final_accept,%0d,-1,%0d,-1,%0d,%0d,%0d,rtl_handshake,%h\n",
                            cycle_count, tile_result_output_tile, source_id,
                            tile_result_out, result_assigned_delay_q,
                            result_service_count,
                            {tile_result_tag, tile_result_output_tile,
                             tile_result_plane, tile_result_y, tile_result_x,
                             tile_result_out, tile_result_data, tile_result_last});
                    result_present_seen_q <= 1'b0;
                end
                result_service_count <= result_service_count + 1;
                result_service_hash <= service_hash_next(
                    result_service_hash,
                    64'({tile_result_tag, tile_result_output_tile,
                         tile_result_plane, tile_result_y,
                         tile_result_x, tile_result_out}),
                    result_service_count, result_assigned_delay_q
                );
                if (IDENTITY_DERIVED_SERVICE) begin
                    if (result_service_count + 1 < FINAL_TABLE_ENTRIES) begin
                        result_wait_q <= 3'(
                            final_delay_mem[result_service_count + 1]
                        ) + 3'd1;
                        result_assigned_delay_q <= {1'b0,
                            final_delay_mem[result_service_count + 1]};
                    end else begin
                        result_wait_q <= '0;
                        result_assigned_delay_q <= '0;
                    end
                end else begin
                    result_wait_q <= transaction_delay(
                        2, result_service_count + 1
                    );
                    result_assigned_delay_q <= transaction_delay(
                        2, result_service_count + 1
                    );
                end
            end
            if (u_executor.acc_state_q != 0)
                phase_cross_rmw_cycles <= phase_cross_rmw_cycles + 1;
            if (u_executor.tx_state_q >= 4
                && u_executor.tx_state_q <= 6)
                phase_external_drain_cycles <=
                    phase_external_drain_cycles + 1;
            if (u_executor.tx_state_q == 1
                || u_executor.tx_state_q == 3)
                phase_scheduler_cycles <= phase_scheduler_cycles + 1;
            if (group_done_valid && group_done_ready)
                group_done_seen_q <= 1'b1;
            if (token_pending_q && token_delay_q != 0)
                token_delay_q <= token_delay_q - 1'b1;
            if (weight_pending_q && weight_delay_q != 0)
                weight_delay_q <= weight_delay_q - 1'b1;
            if (weight_pending_q && weight_delay_q == 0 && weight_hold_q != 0)
                weight_hold_q <= weight_hold_q - 1'b1;
            if (token_req_valid && token_req_ready) begin
                integer expected_tile;
                integer expected_head;
                integer expected_source;
                integer table_index;
                token_pending_q <= 1'b1;
                if (IDENTITY_DERIVED_SERVICE) begin
                    expected_tile = token_count / (HEADS * TOTAL_TOKENS);
                    expected_head = (token_count / TOTAL_TOKENS) % HEADS;
                    expected_source = token_count % TOTAL_TOKENS;
                    table_index = relation_table_index(
                        token_req_input_head, token_req_token_id
                    );
                    if (expected_tile >= OUTPUT_TILES
                        || token_req_input_head != expected_head
                        || token_req_token_id != expected_source
                        || token_req_tag != group_tag + TAG_W'(expected_tile)
                        || token_req_plane != expected_source / (HEIGHT * WIDTH)
                        || token_req_y != (expected_source % (HEIGHT * WIDTH)) / WIDTH
                        || token_req_x != expected_source % WIDTH
                        || table_index < 0
                        || table_index >= RELATION_TABLE_ENTRIES)
                        $fatal(1, "identity relation runtime order mismatch");
                    token_delay_q <= {1'b0, relation_delay_mem[table_index]};
                    token_assigned_delay_q <= {1'b0,
                        relation_delay_mem[table_index]};
                    token_table_index_q <= table_index;
                    token_accept_cycle_q <= cycle_count;
                    token_response_seen_q <= 1'b0;
                    token_delay_sum <= token_delay_sum
                                     + relation_delay_mem[table_index];
                    if (identity_trace_fd != 0)
                        $fwrite(identity_trace_fd,
                            "%0d,relation_accept,%0d,%0d,%0d,-1,-1,%0d,%0d,rtl_handshake,-\n",
                            cycle_count, expected_tile,
                            token_req_input_head, token_req_token_id,
                            relation_delay_mem[table_index], table_index);
                end else if (TRANSACTION_INDEXED_SERVICE) begin
                    token_delay_q <= transaction_delay(0, token_count);
                    token_delay_sum <= token_delay_sum
                                     + transaction_delay(0, token_count);
                    token_service_hash <= service_hash_next(
                        token_service_hash,
                        64'({token_req_tag, token_req_input_head,
                             token_req_token_id}),
                        token_count, transaction_delay(0, token_count)
                    );
                end else begin
                    token_delay_q <= 3'd1 + {1'b0, lfsr_q[5:4]};
                    token_delay_sum <= token_delay_sum
                                     + 3'd1 + {1'b0, lfsr_q[5:4]};
                end
                token_tag_q <= token_req_tag;
                token_head_q <= token_req_input_head;
                token_id_q <= token_req_token_id;
                token_count <= token_count + 1;
            end
            if (IDENTITY_DERIVED_SERVICE && token_rsp_valid
                && !token_response_seen_q) begin
                if (cycle_count - token_accept_cycle_q
                    != 32'(token_assigned_delay_q) + 1)
                    $fatal(1, "identity relation response-valid latency mismatch");
                token_response_seen_q <= 1'b1;
                token_response_payload_q <= {
                    token_rsp_tag, token_rsp_input_head, token_rsp_token_id,
                    token_rsp_q, token_rsp_k, token_rsp_valid_mask,
                    token_rsp_error
                };
                if (identity_trace_fd != 0)
                    $fwrite(identity_trace_fd,
                        "%0d,relation_response_available,-1,%0d,%0d,-1,-1,%0d,%0d,rtl_handshake,%h\n",
                        cycle_count, token_head_q, token_id_q,
                        token_assigned_delay_q, token_table_index_q,
                        {token_rsp_tag, token_rsp_input_head,
                         token_rsp_token_id, token_rsp_q, token_rsp_k,
                         token_rsp_valid_mask, token_rsp_error});
            end
            if (token_rsp_valid && token_rsp_ready) begin
                if (IDENTITY_DERIVED_SERVICE) begin
                    if (cycle_count - token_accept_cycle_q
                        < 32'(token_assigned_delay_q) + 1)
                        $fatal(1, "identity relation response accepted too early");
                    if (identity_trace_fd != 0)
                        $fwrite(identity_trace_fd,
                            "%0d,relation_response_accept,-1,%0d,%0d,-1,-1,%0d,%0d,rtl_handshake,%h\n",
                            cycle_count, token_head_q, token_id_q,
                            token_assigned_delay_q, token_table_index_q,
                            {token_rsp_tag, token_rsp_input_head,
                             token_rsp_token_id, token_rsp_q, token_rsp_k,
                             token_rsp_valid_mask, token_rsp_error});
                end
                token_response_seen_q <= 1'b0;
                token_pending_q <= 1'b0;
            end
            if (weight_req_valid && weight_req_ready) begin
                integer expected_tile;
                integer expected_head;
                integer expected_lane;
                integer expected_out;
                integer table_index;
                weight_pending_q <= 1'b1;
                if (IDENTITY_DERIVED_SERVICE) begin
                    expected_tile = weight_count / (HEADS * HEAD_DIM * OUT_DIM);
                    expected_head = (weight_count / (HEAD_DIM * OUT_DIM)) % HEADS;
                    expected_lane = (weight_count / OUT_DIM) % HEAD_DIM;
                    expected_out = weight_count % OUT_DIM;
                    table_index = weight_table_index(
                        weight_req_output_tile, weight_req_input_head,
                        weight_req_lane, weight_req_out
                    );
                    if (expected_tile >= OUTPUT_TILES
                        || weight_req_output_tile != expected_tile
                        || weight_req_input_head != expected_head
                        || weight_req_lane != expected_lane
                        || weight_req_out != expected_out
                        || weight_req_tag != group_tag + TAG_W'(expected_tile)
                        || table_index != weight_count
                        || table_index < 0
                        || table_index >= WEIGHT_TABLE_ENTRIES)
                        $fatal(1, "identity weight runtime order mismatch");
                    weight_delay_q <= {1'b0, weight_delay_mem[table_index]};
                    weight_assigned_delay_q <= {1'b0,
                        weight_delay_mem[table_index]};
                    weight_table_index_q <= table_index;
                    weight_accept_cycle_q <= cycle_count;
                    weight_response_seen_q <= 1'b0;
                    weight_hold_q <= 3'(FORCE_WEIGHT_RESPONSE_HOLD_CYCLES);
                    weight_delay_sum <= weight_delay_sum
                                      + weight_delay_mem[table_index];
                    if (identity_trace_fd != 0)
                        $fwrite(identity_trace_fd,
                            "%0d,weight_accept,%0d,%0d,-1,%0d,%0d,%0d,%0d,rtl_handshake,-\n",
                            cycle_count, weight_req_output_tile,
                            weight_req_input_head, weight_req_lane,
                            weight_req_out, weight_delay_mem[table_index],
                            table_index);
                end else if (TRANSACTION_INDEXED_SERVICE) begin
                    weight_delay_q <= transaction_delay(1, weight_count);
                    weight_delay_sum <= weight_delay_sum
                                      + transaction_delay(1, weight_count);
                    weight_service_hash <= service_hash_next(
                        weight_service_hash,
                        64'({weight_req_tag, weight_req_input_head,
                             weight_req_output_tile, weight_req_lane,
                             weight_req_out}),
                        weight_count, transaction_delay(1, weight_count)
                    );
                end else begin
                    weight_delay_q <= 3'd1 + {1'b0, lfsr_q[7:6]};
                    weight_delay_sum <= weight_delay_sum
                                      + 3'd1 + {1'b0, lfsr_q[7:6]};
                end
                weight_tag_q <= weight_req_tag;
                weight_head_q <= weight_req_input_head;
                weight_tile_q <= weight_req_output_tile;
                weight_lane_q <= weight_req_lane;
                weight_out_q <= weight_req_out;
                weight_count <= weight_count + 1;
            end
            if (IDENTITY_DERIVED_SERVICE && weight_service_valid
                && !weight_response_seen_q) begin
                if (cycle_count - weight_accept_cycle_q
                    != 32'(weight_assigned_delay_q) + 1)
                    $fatal(1, "identity weight response-valid latency mismatch");
                weight_response_seen_q <= 1'b1;
                weight_response_payload_q <= {
                    weight_rsp_tag, weight_rsp_input_head,
                    weight_rsp_output_tile, weight_rsp_lane,
                    weight_rsp_out, weight_rsp_data, weight_rsp_error
                };
                if (identity_trace_fd != 0)
                    $fwrite(identity_trace_fd,
                        "%0d,weight_response_available,%0d,%0d,-1,%0d,%0d,%0d,%0d,rtl_handshake,%h\n",
                        cycle_count, weight_tile_q, weight_head_q,
                        weight_lane_q, weight_out_q,
                        weight_assigned_delay_q, weight_table_index_q,
                        {weight_rsp_tag, weight_rsp_input_head,
                         weight_rsp_output_tile, weight_rsp_lane,
                         weight_rsp_out, weight_rsp_data, weight_rsp_error});
            end
            if (IDENTITY_DERIVED_SERVICE
                && weight_rsp_valid && !weight_rsp_ready) begin
                if (identity_trace_fd != 0)
                    $fwrite(identity_trace_fd,
                        "%0d,weight_response_stall,%0d,%0d,-1,%0d,%0d,%0d,%0d,rtl_protocol_telemetry,%h\n",
                        cycle_count, weight_tile_q, weight_head_q,
                        weight_lane_q, weight_out_q,
                        weight_assigned_delay_q, weight_table_index_q,
                        {weight_rsp_tag, weight_rsp_input_head,
                         weight_rsp_output_tile, weight_rsp_lane,
                         weight_rsp_out, weight_rsp_data, weight_rsp_error});
            end
            if (weight_service_valid && weight_service_ready) begin
                if (IDENTITY_DERIVED_SERVICE) begin
                    if (cycle_count - weight_accept_cycle_q
                        < 32'(weight_assigned_delay_q) + 1)
                        $fatal(1, "identity weight response accepted too early");
                    if (identity_trace_fd != 0)
                        $fwrite(identity_trace_fd,
                            "%0d,weight_response_accept,%0d,%0d,-1,%0d,%0d,%0d,%0d,rtl_handshake,%h\n",
                            cycle_count, weight_tile_q, weight_head_q,
                            weight_lane_q, weight_out_q,
                            weight_assigned_delay_q, weight_table_index_q,
                            {weight_rsp_tag, weight_rsp_input_head,
                             weight_rsp_output_tile, weight_rsp_lane,
                             weight_rsp_out, weight_rsp_data, weight_rsp_error});
                end
                weight_response_seen_q <= 1'b0;
                weight_pending_q <= 1'b0;
            end
        end
    end

    generate
        if (USE_MEMO) begin : g_memo_phase_ledger
            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    phase_weight_cycles <= 0;
                    phase_frontend_cycles <= 0;
                    phase_readout_cycles <= 0;
                    phase_release_cycles <= 0;
                end else begin
                    case (u_executor.g_memo_head_engine.u_head_engine.state_q)
                        1, 2: phase_weight_cycles <=
                            phase_weight_cycles + 1;
                        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15:
                            phase_frontend_cycles <=
                                phase_frontend_cycles + 1;
                        16, 17, 18: phase_readout_cycles <=
                            phase_readout_cycles + 1;
                        19, 20: phase_release_cycles <=
                            phase_release_cycles + 1;
                        default: begin end
                    endcase
                end
            end
        end else begin : g_recompute_phase_ledger
            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    phase_weight_cycles <= 0;
                    phase_frontend_cycles <= 0;
                    phase_readout_cycles <= 0;
                    phase_release_cycles <= 0;
                end else begin
                    case (u_executor.g_baseline_head_engine.u_head_engine.state_q)
                        1, 2: phase_weight_cycles <=
                            phase_weight_cycles + 1;
                        3, 4, 5, 6, 7, 8, 9:
                            phase_frontend_cycles <=
                                phase_frontend_cycles + 1;
                        10, 11, 12: phase_readout_cycles <=
                            phase_readout_cycles + 1;
                        13, 14: phase_release_cycles <=
                            phase_release_cycles + 1;
                        default: begin end
                    endcase
                end
            end
        end
    endgenerate

    always @(posedge clk_core) begin
        integer index;
        integer tile;
        if (!rst_core && head_done_valid && head_done_ready) begin
            if (head_done_error
                || head_done_input_head != 5'(head_done_count % HEADS))
                $fatal(1, "head completion mismatch count=%0d", head_done_count);
            head_done_count = head_done_count + 1;
        end
        if (!rst_core && tile_result_valid && tile_result_ready) begin
            tile = tile_result_output_tile;
            index = (((tile_result_plane * HEIGHT + tile_result_y) * WIDTH
                    + tile_result_x) * OUT_DIM) + tile_result_out;
            if (tile < 0 || tile >= OUTPUT_TILES
                || index != result_in_tile
                || tile_result_tag != group_tag + TAG_W'(tile)
                || (!no_acc_check
                    && $signed(tile_result_data) != expected_mem[tile][index])
                || tile_result_last != (index == TOTAL_RESULTS - 1))
                $fatal(1,
                    "memo final mismatch tile=%0d index=%0d got=%0d exp=%0d",
                    tile, index, tile_result_data, expected_mem[tile][index]);
            if (actual_fd != 0)
                $fwrite(actual_fd, "%08x\n", tile_result_data);
            if (index == TOTAL_RESULTS - 1) begin
                result_in_tile = 0;
                completed_tiles = completed_tiles + 1;
            end else begin
                result_in_tile = result_in_tile + 1;
            end
        end
    end

    generate
        if (!USE_MEMO) begin : g_identity_phase_trace
            always @(posedge clk_core) begin
                if (rst_core) begin
                    previous_tx_state = -1;
                    previous_acc_state = -1;
                    previous_head_state = -1;
                end else if (IDENTITY_DERIVED_SERVICE
                             && identity_trace_fd != 0) begin
            if (group_valid && group_ready)
                $fwrite(identity_trace_fd,
                    "%0d,group_start,-1,-1,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count);
            if (tile_start_valid && tile_start_ready)
                $fwrite(identity_trace_fd,
                    "%0d,tile_start,%0d,-1,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count, tile_start_output_tile);
            if (head_job_valid && head_job_ready)
                $fwrite(identity_trace_fd,
                    "%0d,head_start,%0d,%0d,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count, head_job_output_tile, head_job_input_head);
            if (head_done_valid && head_done_ready)
                $fwrite(identity_trace_fd,
                    "%0d,head_done,%0d,%0d,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count, u_executor.tile_output_tile_q,
                    head_done_input_head);
            if (tile_done_valid && tile_done_ready)
                $fwrite(identity_trace_fd,
                    "%0d,tile_done,%0d,-1,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count, u_executor.tile_output_tile_q);
            if (group_done_valid && group_done_ready)
                $fwrite(identity_trace_fd,
                    "%0d,group_done,-1,-1,-1,-1,-1,-1,-1,rtl_boundary,-\n",
                    cycle_count);
            if (previous_tx_state != u_executor.tx_state_q) begin
                $fwrite(identity_trace_fd,
                    "%0d,tx_state,%0d,-1,-1,-1,-1,-1,%0d,rtl_internal_state,-\n",
                    cycle_count, u_executor.tile_output_tile_q,
                    u_executor.tx_state_q);
                previous_tx_state = u_executor.tx_state_q;
            end
            if (previous_acc_state != u_executor.acc_state_q) begin
                $fwrite(identity_trace_fd,
                    "%0d,acc_state,%0d,-1,-1,-1,-1,-1,%0d,rtl_internal_state,-\n",
                    cycle_count, u_executor.tile_output_tile_q,
                    u_executor.acc_state_q);
                previous_acc_state = u_executor.acc_state_q;
            end
                    if (previous_head_state
                        != u_executor.g_baseline_head_engine.u_head_engine.state_q) begin
                        $fwrite(identity_trace_fd,
                            "%0d,head_state,%0d,%0d,-1,-1,-1,-1,%0d,rtl_internal_state,-\n",
                            cycle_count, u_executor.tile_output_tile_q,
                            u_executor.expected_head_q,
                            u_executor.g_baseline_head_engine.u_head_engine.state_q);
                        previous_head_state =
                            u_executor.g_baseline_head_engine.u_head_engine.state_q;
                    end
                end
            end
        end
    endgenerate

    task automatic load_head(input integer head, input string path);
        integer fd, rc, plane, y, x, mask;
        logic [31:0] q_value, k0, k1, k2, k3, k4;
        begin
            fd = $fopen(path, "r");
            if (fd == 0)
                $fatal(1, "cannot open head input %0d", head);
            for (int row = 0; row < TOTAL_TOKENS; row = row + 1) begin
                rc = $fscanf(fd, "%d %d %d %h %h %h %h %h %h %h\n",
                    plane, y, x, q_value, k0, k1, k2, k3, k4, mask);
                if (rc != 10
                    || plane < 0 || plane >= TIME_PLANES
                    || y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH
                    || row != plane * 225 + y * 15 + x)
                    $fatal(1, "invalid memo input row head=%0d row=%0d", head, row);
                q_mem[head][row] = q_value;
                k_mem[head][row][0] = k0;
                k_mem[head][row][1] = k1;
                k_mem[head][row][2] = k2;
                k_mem[head][row][3] = k3;
                k_mem[head][row][4] = k4;
                mask_mem[head][row] = 5'(mask);
            end
            $fclose(fd);
        end
    endtask

    task automatic load_combined_heads(input string path);
        integer fd, rc, head, plane, y, x, mask;
        logic [31:0] q_value, k0, k1, k2, k3, k4;
        begin
            fd = $fopen(path, "r");
            if (fd == 0)
                $fatal(1, "cannot open combined head input file");
            for (int row = 0; row < HEADS * TOTAL_TOKENS; row = row + 1) begin
                rc = $fscanf(fd, "%d %d %d %d %h %h %h %h %h %h %h\n",
                    head, plane, y, x, q_value, k0, k1, k2, k3, k4, mask);
                if (rc != 11
                    || head < 0 || head >= HEADS
                    || plane < 0 || plane >= TIME_PLANES
                    || y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH
                    || row != head * TOTAL_TOKENS
                              + plane * 225 + y * 15 + x)
                    $fatal(1, "invalid combined input row=%0d", row);
                q_mem[head][plane * 225 + y * 15 + x] = q_value;
                k_mem[head][plane * 225 + y * 15 + x][0] = k0;
                k_mem[head][plane * 225 + y * 15 + x][1] = k1;
                k_mem[head][plane * 225 + y * 15 + x][2] = k2;
                k_mem[head][plane * 225 + y * 15 + x][3] = k3;
                k_mem[head][plane * 225 + y * 15 + x][4] = k4;
                mask_mem[head][plane * 225 + y * 15 + x] = 5'(mask);
            end
            $fclose(fd);
        end
    endtask

    task automatic load_checkpoint_weights(input string path);
        integer fd, rc, head, tile, lane, out_index, raw_value;
        begin
            fd = $fopen(path, "r");
            if (fd == 0)
                $fatal(1, "cannot open checkpoint weight file");
            for (int row = 0;
                 row < HEADS * OUTPUT_TILES * HEAD_DIM * OUT_DIM;
                 row = row + 1) begin
                rc = $fscanf(fd, "%d %d %d %d %h\n",
                    head, tile, lane, out_index, raw_value);
                if (rc != 5
                    || head < 0 || head >= HEADS
                    || tile < 0 || tile >= OUTPUT_TILES
                    || lane < 0 || lane >= HEAD_DIM
                    || out_index < 0 || out_index >= OUT_DIM
                    || row != (((head * OUTPUT_TILES + tile) * HEAD_DIM
                                 + lane) * OUT_DIM + out_index)) begin
                    $fatal(1, "invalid checkpoint weight row=%0d", row);
                end
                checkpoint_weight_mem[head][tile][lane][out_index]
                    = raw_value[7:0];
            end
            $fclose(fd);
        end
    endtask

    task automatic load_expected(input string path);
        integer fd, rc, tile, plane, y, x, out;
        integer signed value;
        begin
            fd = $fopen(path, "r");
            if (fd == 0)
                $fatal(1, "cannot open memo expected file");
            for (int row = 0; row < OUTPUT_TILES * TOTAL_RESULTS; row = row + 1) begin
                rc = $fscanf(fd, "%d %d %d %d %d %d\n",
                    tile, plane, y, x, out, value);
                if (rc != 6
                    || tile < 0 || tile >= OUTPUT_TILES
                    || plane < 0 || plane >= TIME_PLANES
                    || y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH
                    || out < 0 || out >= OUT_DIM
                    || row != tile * TOTAL_RESULTS
                              + (((plane * HEIGHT + y) * WIDTH + x)
                                 * OUT_DIM + out))
                    $fatal(1, "invalid memo expected row=%0d", row);
                expected_mem[tile][row % TOTAL_RESULTS] = value;
            end
            $fclose(fd);
        end
    endtask

    initial begin
        clk_core = 1'b0;
        rst_core = 1'b1;
        group_valid = 1'b0;
        group_tag = 24'h5d5000;
        actual_fd = 0;
        identity_trace_fd = 0;
        no_acc_check = $test$plusargs("NO_ACC_CHECK");
        use_checkpoint_weights =
            $value$plusargs("WEIGHTS=%s", weights_path);
        use_combined_inputs =
            $value$plusargs("INPUTS=%s", combined_input_path);
        service_seed_arg = 16'hace1;
        if (!$value$plusargs("SERVICE_SEED=%d", service_seed_arg))
            service_seed_arg = 16'hace1;
        service_seed = service_seed_arg[15:0];
        if (IDENTITY_DERIVED_SERVICE) begin
            if (service_seed_arg != 20260810
                || !$value$plusargs(
                    "RELATION_DELAY_MEMH=%s", relation_delay_path)
                || !$value$plusargs(
                    "WEIGHT_DELAY_MEMH=%s", weight_delay_path)
                || !$value$plusargs(
                    "FINAL_DELAY_MEMH=%s", final_delay_path)
                || !$value$plusargs(
                    "IDENTITY_TRACE=%s", identity_trace_path)
                || !$value$plusargs(
                    "IDENTITY_MANIFEST_SHA=%s", identity_manifest_sha)
                || !$value$plusargs(
                    "IDENTITY_RECEIPT_SHA=%s", identity_receipt_sha))
                $fatal(1, "identity-derived service plusarg contract is incomplete");
            if (identity_manifest_sha.len() != 64
                || identity_receipt_sha.len() != 64)
                $fatal(1, "identity package SHA must contain 64 characters");
            $readmemh(relation_delay_path, relation_delay_mem);
            $readmemh(weight_delay_path, weight_delay_mem);
            $readmemh(final_delay_path, final_delay_mem);
            for (int row = 0; row < RELATION_TABLE_ENTRIES; row = row + 1)
                if (^relation_delay_mem[row] === 1'bx)
                    $fatal(1, "relation delay table contains X at %0d", row);
            for (int row = 0; row < WEIGHT_TABLE_ENTRIES; row = row + 1)
                if (^weight_delay_mem[row] === 1'bx)
                    $fatal(1, "weight delay table contains X at %0d", row);
            for (int row = 0; row < FINAL_TABLE_ENTRIES; row = row + 1)
                if (^final_delay_mem[row] === 1'bx)
                    $fatal(1, "final delay table contains X at %0d", row);
            identity_trace_fd = $fopen(identity_trace_path, "w");
            if (identity_trace_fd == 0)
                $fatal(1, "cannot open identity trace output");
            $fwrite(identity_trace_fd,
                "cycle,event,tile,head,source,lane,out,delay,index,origin,payload\n");
            $fwrite(identity_trace_fd,
                "0,manifest_binding,-1,-1,-1,-1,-1,-1,-1,%s,-\n",
                identity_manifest_sha);
            $fwrite(identity_trace_fd,
                "0,receipt_binding,-1,-1,-1,-1,-1,-1,-1,%s,-\n",
                identity_receipt_sha);
        end
        if (!use_combined_inputs
            && (HEADS != 3
                || !$value$plusargs("INPUT_H0=%s", input_path_h0)
                || !$value$plusargs("INPUT_H1=%s", input_path_h1)
                || !$value$plusargs("INPUT_H2=%s", input_path_h2)))
            $fatal(1, "combined input or three legacy input paths are required");
        if (!no_acc_check
            && !$value$plusargs("EXPECTED=%s", expected_path))
            $fatal(1, "memo expected path is required");
        if (use_combined_inputs) begin
            load_combined_heads(combined_input_path);
        end else begin
            input_path[0] = input_path_h0;
            input_path[1] = input_path_h1;
            input_path[2] = input_path_h2;
            for (int head = 0; head < HEADS; head = head + 1)
                load_head(head, input_path[head]);
        end
        if (!no_acc_check)
            load_expected(expected_path);
        if (use_checkpoint_weights)
            load_checkpoint_weights(weights_path);
        if ($value$plusargs("ACTUAL_ACC_FILE=%s", actual_path)) begin
            actual_fd = $fopen(actual_path, "w");
            if (actual_fd == 0)
                $fatal(1, "cannot open actual Acc32 output");
        end
        result_in_tile = 0;
        completed_tiles = 0;
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
            || scheduler_tiles != OUTPUT_TILES
            || scheduler_heads != HEADS * OUTPUT_TILES
            || scheduler_errors != 0 || perf_tiles != OUTPUT_TILES
            || perf_heads != HEADS * OUTPUT_TILES
            || perf_partial_results
               != (USE_INPLACE
                   ? 0 : HEADS * OUTPUT_TILES
                         * (VECTOR_RESULT_MODE ? TOTAL_TOKENS : TOTAL_RESULTS))
            || perf_accumulator_writes
               != (USE_INPLACE
                   ? 0 : HEADS * OUTPUT_TILES
                         * (VECTOR_RESULT_MODE ? TOTAL_TOKENS : TOTAL_RESULTS))
            || perf_final_results != OUTPUT_TILES * TOTAL_RESULTS
            || u_executor.child_result_jobs
               != (USE_INPLACE ? OUTPUT_TILES : HEADS * OUTPUT_TILES)
            || completed_tiles != OUTPUT_TILES
            || head_done_count != HEADS * OUTPUT_TILES
            || weight_count != HEADS * OUTPUT_TILES * HEAD_DIM * OUT_DIM
            || u_executor.child_token_requests != token_count
            || (IDENTITY_DERIVED_SERVICE
                && result_service_count != FINAL_TABLE_ENTRIES)
            || result_stall_count == 0)
            $fatal(1, "multi-tile common ledger mismatch");
        if (USE_MEMO) begin
            if (token_count
                   != (HEADS + u_executor.perf_memo_fallbacks)
                      * TOTAL_TOKENS
                || u_executor.perf_memo_hits
                   + u_executor.perf_memo_fallbacks
                   != HEADS * (OUTPUT_TILES - 1)
                || u_executor.perf_memo_resident_builds
                   * (OUTPUT_TILES - 1)
                   != u_executor.perf_memo_hits
                || u_executor.perf_cache_release_intents != HEADS
                || (u_executor.perf_memo_hits != 0
                    && u_executor.perf_replay_records == 0))
                $fatal(1,
                    "memo generic ledger mismatch token=%0d hits=%0d fallback=%0d resident=%0d release=%0d replay=%0d",
                    token_count, u_executor.perf_memo_hits,
                    u_executor.perf_memo_fallbacks,
                    u_executor.perf_memo_resident_builds,
                    u_executor.perf_cache_release_intents,
                    u_executor.perf_replay_records);
            if (!use_checkpoint_weights && !use_combined_inputs
                && HEADS == 3 && OUTPUT_TILES == 3
                && (token_count != 2250
                    || u_executor.perf_memo_hits != 4
                    || u_executor.perf_memo_fallbacks != 2
                    || u_executor.perf_memo_resident_builds != 2
                    || u_executor.perf_replay_records != 28))
                $fatal(1, "legacy synthetic memo ledger mismatch");
        end else if (!USE_MEMO) begin
            if (token_count != HEADS * OUTPUT_TILES * TOTAL_TOKENS
                || u_executor.perf_memo_hits != 0
                || u_executor.perf_memo_fallbacks != 0
                || u_executor.perf_memo_resident_builds != 0
                || u_executor.perf_cache_release_intents != 0
                || u_executor.perf_replay_records != 0)
                $fatal(1, "recompute multi-tile ledger mismatch");
        end
        $display(
            "PASS Local5 multi-tile memo=%0d inplace=%0d acc_backend=%0d transaction_service=%0d identity_service=%0d seed=%0d stage=%0d block=%0d window=%0d cycles=%0d token=%0d token_delay_sum=%0d weight_delay_sum=%0d result_service=%0d hits=%0d fallback=%0d replay_records=%0d partial=%0d final=%0d child_results=%0d weight_cycles=%0d frontend_cycles=%0d readout_cycles=%0d release_cycles=%0d rmw_cycles=%0d drain_cycles=%0d scheduler_cycles=%0d vector=%0d token_service_hash=%016h weight_service_hash=%016h result_service_hash=%016h",
            USE_MEMO, USE_INPLACE, ACC_BACKEND_KIND,
            TRANSACTION_INDEXED_SERVICE, IDENTITY_DERIVED_SERVICE,
            service_seed_arg,
            run_stage_id, run_block_id, run_window_id,
            cycle_count, token_count,
            token_delay_sum, weight_delay_sum, result_service_count,
            u_executor.perf_memo_hits, u_executor.perf_memo_fallbacks,
            u_executor.perf_replay_records, perf_partial_results,
            perf_final_results, u_executor.unused_child_results,
            phase_weight_cycles, phase_frontend_cycles,
            phase_readout_cycles, phase_release_cycles,
            phase_cross_rmw_cycles, phase_external_drain_cycles,
            phase_scheduler_cycles, VECTOR_RESULT_MODE,
            token_service_hash, weight_service_hash, result_service_hash
        );
        if (actual_fd != 0) begin
            $fclose(actual_fd);
            actual_fd = 0;
        end
        if (identity_trace_fd != 0) begin
            $fclose(identity_trace_fd);
            identity_trace_fd = 0;
        end
        $finish;
    end

    initial begin
        repeat (TIMEOUT_CYCLES) @(posedge clk_core);
        $display(
            "DEBUG sched=%0d exec=%0d head=%0d tile=%0d token=%0d weight=%0d partial=%0d final=%0d memo_hit=%0d fallback=%0d",
            u_scheduler.state_q, u_executor.tx_state_q,
            head_done_count, completed_tiles, token_count, weight_count,
            perf_partial_results, perf_final_results,
            u_executor.perf_memo_hits, u_executor.perf_memo_fallbacks
        );
        $fatal(1, "Local5 memo multi-tile timeout");
    end
endmodule

`default_nettype wire
