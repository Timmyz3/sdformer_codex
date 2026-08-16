`timescale 1ns/1ps
`default_nettype none

// Scheduler + Q-silent score on the first S0.B0 window (3 heads).
module tb_qfit_local5_qsilent_scheduler_window;
    localparam int TAG_W = 24;
    localparam int HEADS = 3;
    localparam int SOURCES = 450;

    logic clk_core = 1'b0;
    logic rst_core;
    logic start_frame;
    logic frame_busy;
    logic frame_done;
    logic tile_start_valid;
    logic [TAG_W-1:0] tile_start_tag;
    logic head_job_valid;
    logic head_job_ready;
    logic [TAG_W-1:0] head_job_tag;
    logic [1:0] head_job_stage;
    logic [2:0] head_job_block;
    logic [8:0] head_job_window;
    logic [4:0] head_job_input_head;
    logic [4:0] head_job_output_tile;
    logic head_job_last_input_head;
    logic head_job_last_output_tile;
    logic head_done_valid;
    logic head_done_ready;
    logic [TAG_W-1:0] head_done_tag;
    logic [4:0] head_done_input_head;
    logic tile_done_valid;
    logic tile_done_ready;
    logic [TAG_W-1:0] tile_done_tag;
    logic protocol_error;

    logic score_in_valid;
    logic score_in_ready;
    logic [31:0] score_q;
    logic [159:0] score_k;
    logic [4:0] score_mask;
    logic score_out_valid;
    logic [79:0] score_out_q7;
    logic [44:0] score_out_gate;
    logic [31:0] qsilent_hits;

    logic [31:0] mem_q [0:HEADS*SOURCES-1];
    logic [159:0] mem_k [0:HEADS*SOURCES-1];
    logic [4:0] mem_valid [0:HEADS*SOURCES-1];
    logic [79:0] mem_score [0:HEADS*SOURCES-1];
    logic [44:0] mem_gate [0:HEADS*SOURCES-1];

    logic head_pending;
    logic tile_pending;
    logic score_busy;
    logic score_finished;
    logic [TAG_W-1:0] pend_tag;
    logic [4:0] pend_head;
    logic pend_last_head;

    integer real_heads;
    integer dummy_jobs;
    integer checked;
    integer score_cycles;
    integer dest;
    string vector_dir;

    always #1 clk_core = ~clk_core;

    qfit_local5_encoder_job_scheduler u_sched (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .start_frame(start_frame),
        .frame_busy(frame_busy),
        .frame_done(frame_done),
        .tile_start_valid(tile_start_valid),
        .tile_start_ready(1'b1),
        .tile_start_tag(tile_start_tag),
        .tile_start_stage(),
        .tile_start_block(),
        .tile_start_window(),
        .tile_start_output_tile(),
        .tile_start_head_count(),
        .head_job_valid(head_job_valid),
        .head_job_ready(head_job_ready),
        .head_job_tag(head_job_tag),
        .head_job_stage(head_job_stage),
        .head_job_block(head_job_block),
        .head_job_window(head_job_window),
        .head_job_input_head(head_job_input_head),
        .head_job_input_channel_base(),
        .head_job_output_tile(head_job_output_tile),
        .head_job_decode_required(),
        .head_job_cache_release(),
        .head_job_last_input_head(head_job_last_input_head),
        .head_job_last_output_tile(head_job_last_output_tile),
        .head_done_valid(head_done_valid),
        .head_done_ready(head_done_ready),
        .head_done_tag(head_done_tag),
        .head_done_input_head(head_done_input_head),
        .head_done_error(1'b0),
        .tile_done_valid(tile_done_valid),
        .tile_done_ready(tile_done_ready),
        .tile_done_tag(tile_done_tag),
        .tile_done_error(1'b0),
        .protocol_error(protocol_error),
        .perf_window_groups(),
        .perf_output_tiles(),
        .perf_head_replays(),
        .perf_decode_intent_jobs(),
        .perf_release_intent_jobs()
    );

    qfit_local5_qsilent_score_leaf #(
        .ENABLE_QSILENT(1'b1),
        .ARCH_QFSA(1'b1),
        .PIPE_COMPACTOR(1'b1),
        .XBF_BANKED(1'b1),
        .USE_THRESHOLD_ROUTE(1'b1),
        .ROUTE_THRESHOLD(8),
        .USE_BANK_PRESSURE_ROUTE(1'b1),
        .BANK_PRESSURE_THRESHOLD(2)
    ) u_score (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(score_in_valid),
        .in_ready(score_in_ready),
        .in_tag(16'(dest)),
        .in_q(score_q),
        .in_k(score_k),
        .in_valid_mask(score_mask),
        .out_valid(score_out_valid),
        .out_ready(1'b1),
        .out_tag(),
        .out_score_q7(score_out_q7),
        .out_gate_q17(score_out_gate),
        .out_k_self(),
        .out_valid_mask(),
        .perf_service_cycles(),
        .perf_route_direct_mask(),
        .perf_qsilent_rows(qsilent_hits),
        .perf_identk_rows(),
        .perf_overlap_accepts()
    );

    assign head_job_ready = !head_pending && !score_busy;
    assign head_done_valid = head_pending && score_finished;
    assign head_done_tag = pend_tag;
    assign head_done_input_head = pend_head;
    assign tile_done_valid = tile_pending && score_finished && pend_last_head;
    assign tile_done_tag = pend_tag;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            head_pending <= 1'b0;
            tile_pending <= 1'b0;
            pend_tag <= '0;
            pend_head <= '0;
            pend_last_head <= 1'b0;
        end else begin
            if (head_job_valid && head_job_ready) begin
                head_pending <= 1'b1;
                pend_tag <= head_job_tag;
                pend_head <= head_job_input_head;
                pend_last_head <= head_job_last_input_head;
                if (head_job_last_input_head)
                    tile_pending <= 1'b1;
                if (!(head_job_stage == 2'd0 && head_job_block == 3'd0
                      && head_job_window == 9'd0 && head_job_output_tile == '0
                      && real_heads < HEADS))
                    dummy_jobs <= dummy_jobs + 1;
            end
            if (head_done_valid && head_done_ready)
                head_pending <= 1'b0;
            if (tile_done_valid && tile_done_ready)
                tile_pending <= 1'b0;
        end
    end

    initial begin
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir))
            $fatal(1, "missing +VECTOR_DIR");
        $readmemh({vector_dir, "/input_q.memh"}, mem_q);
        $readmemh({vector_dir, "/input_candidate_k.memh"}, mem_k);
        $readmemh({vector_dir, "/input_valid.memh"}, mem_valid);
        $readmemh({vector_dir, "/expected_scores.memh"}, mem_score);
        $readmemh({vector_dir, "/expected_gates.memh"}, mem_gate);

        rst_core = 1'b1;
        start_frame = 1'b0;
        score_in_valid = 1'b0;
        score_busy = 1'b0;
        score_finished = 1'b1;
        real_heads = 0;
        dummy_jobs = 0;
        checked = 0;
        score_cycles = 0;
        dest = 0;
        repeat (4) @(negedge clk_core);
        rst_core = 1'b0;
        @(negedge clk_core);
        start_frame = 1'b1;
        @(negedge clk_core);
        start_frame = 1'b0;

        while (real_heads < HEADS) begin
            @(negedge clk_core);
            if (head_job_valid && head_job_ready
                && head_job_stage == 2'd0 && head_job_block == 3'd0
                && head_job_window == 9'd0 && head_job_output_tile == '0) begin
                score_busy = 1'b1;
                score_finished = 1'b0;
                $display("QS_SCHED_HEAD head=%0d tag=%0d",
                    head_job_input_head, head_job_tag);
                for (dest = 0; dest < SOURCES; dest = dest + 1) begin
                    @(negedge clk_core);
                    score_q = mem_q[real_heads*SOURCES + dest];
                    score_k = mem_k[real_heads*SOURCES + dest];
                    score_mask = mem_valid[real_heads*SOURCES + dest];
                    score_in_valid = 1'b1;
                    @(posedge clk_core);
                    while (!score_in_ready) @(posedge clk_core);
                    @(negedge clk_core);
                    score_in_valid = 1'b0;
                    while (!score_out_valid) @(posedge clk_core);
                    if (score_out_q7 !== mem_score[real_heads*SOURCES + dest]
                        || score_out_gate !== mem_gate[real_heads*SOURCES + dest])
                        $fatal(1, "mismatch head=%0d dest=%0d", real_heads, dest);
                    checked = checked + 1;
                    score_cycles = score_cycles + 2;
                end
                score_finished = 1'b1;
                score_busy = 1'b0;
                real_heads = real_heads + 1;
                $display("QS_SCHED_DONE head=%0d checked=%0d", real_heads-1, checked);
            end
        end
        $display("QS_SCHED_SUM real_heads=%0d checked=%0d dummy=%0d score_cycles=%0d wall=%0d qsilent_hits=%0d",
            real_heads, checked, dummy_jobs, score_cycles, score_cycles, qsilent_hits);
        $display("PASS tb_qfit_local5_qsilent_scheduler_window");
        $finish;
    end
endmodule

`default_nettype wire
