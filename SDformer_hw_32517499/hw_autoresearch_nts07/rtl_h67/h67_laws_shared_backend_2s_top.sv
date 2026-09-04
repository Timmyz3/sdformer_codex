`timescale 1ns/1ps
`default_nettype none

// Shared-encoder / shared-Shiftmax row pipeline.
// Two directory+K workspaces; at most one build and one emit.
module h67_laws_shared_backend_2s_top #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int TOKEN_W = $clog2(2 * PAIRS + 1),
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 2 * PAIRS,
    parameter int COUNT_W = $clog2(2 * PAIRS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1),
    parameter int SLOT_FIFO_DEPTH = 32,
    parameter int FIFO_OCC_W = $clog2(SLOT_FIFO_DEPTH + 1),
    parameter bit QUOTIENT_ENABLE = 1'b1,
    parameter bit MSSB5_SCORE_FRONT = 1'b0,
    parameter int MEMORY_IMPL = 0
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       row_k_present,
    input  logic                       window_seal,
    input  logic                       descriptor_issue_enable,
    input  logic                       cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]     cfg_threshold_q8,
    output logic                       build_ready,
    output logic                       seal_ready,
    output logic                       emit_active,
    output logic                       last_row_done,

    input  logic                       pair_valid,
    output logic                       pair_ready,
    input  logic [PAIR_ID_W-1:0]       pair_id,
    input  logic [2*HEAD_DIM-1:0]      q_pair,
    input  logic [2*HEAD_DIM-1:0]      k_pair,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic                       out_last,
    output logic [TOKEN_W-1:0]         out_token_id,
    output logic [HEAD_DIM-1:0]        out_k_bits,
    output logic [GATE_W-1:0]          out_gate_q17,
    output logic                       protocol_error,
    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_slots,
    output logic [31:0]                perf_equal_pairs
);
    localparam int PAIR_COUNT_W = $clog2(PAIRS + 1);

    typedef enum logic [1:0] {
        WS_IDLE  = 2'd0,
        WS_BUILD = 2'd1,
        WS_SEAL  = 2'd2,
        WS_EMIT  = 2'd3
    } ws_state_t;

    typedef enum logic [1:0] {
        EMIT_IDLE,
        EMIT_WAIT_K,
        EMIT_ACTIVE
    } emit_state_t;

    ws_state_t ws_state_q [0:1];
    logic build_sel_q;
    logic emit_sel_q;
    logic emit_valid_sel;
    logic [1:0] idle_oh;
    logic [1:0] build_oh;
    logic start_accept;
    logic start_reject;

    logic packet_valid;
    logic packet_ready;
    logic encoder_pair_ready;
    logic encoder_pair_valid;
    logic [1:0] packet_slot_count;
    logic [15:0] packet_slot0;
    logic [15:0] packet_slot1;
    logic encoder_pair_commit;
    logic encoder_error;
    logic fifo_valid;
    logic fifo_ready;
    logic [1:0] fifo_count;
    logic [15:0] fifo_slot0;
    logic [15:0] fifo_slot1;
    logic fifo_error;
    logic [FIFO_OCC_W-1:0] fifo_occ;

    logic [PAIR_COUNT_W-1:0] decoded_pairs_q;
    logic pair_open_q;
    logic [7:0] slot0_score;
    logic [1:0] slot0_temporal_mask;
    logic [1:0] slot0_active_mask;
    logic slot0_pair_last;
    logic slot0_shape_legal;
    logic slot0_legal;
    logic [7:0] slot1_score;
    logic [1:0] slot1_temporal_mask;
    logic [1:0] slot1_active_mask;
    logic slot1_pair_last;
    logic slot1_shape_legal;
    logic slot1_legal;
    logic open_after_slot0;
    logic open_after_slot1;
    logic [PAIR_COUNT_W-1:0] slot1_pair_id;
    logic [1:0] closed_pair_count;
    logic batch_legal;
    logic batch_fire;
    logic directory_in_valid;
    logic [1:0] dir_in_ready;
    logic [1:0] dir_window_start;
    logic [1:0] dir_window_seal;
    logic [1:0] dir_window_ready;
    logic [1:0] dir_window_done;
    logic [1:0] dir_class_valid;
    logic [CLASS_W-1:0] dir_class_score [0:1];
    logic [COUNT_W-1:0] dir_class_mult [0:1];
    logic [1:0] dir_class_last;
    logic [1:0] dir_active_valid;
    logic [PAIR_ID_W-1:0] dir_active_pair [0:1];
    logic signed [SCORE_W-1:0] dir_active_score [0:1];
    logic [1:0] dir_active_temporal [0:1];
    logic [1:0] dir_active_k [0:1];
    logic [1:0] dir_active_last;
    logic signed [SCORE_W-1:0] dir_row_max [0:1];
    logic [1:0] dir_error;
    logic [1:0] k_start;
    logic [1:0] k_write_valid;
    logic [1:0] k_error;
    logic [1:0] k_resp_valid [0:1];
    logic [HEAD_DIM-1:0] k_resp0 [0:1];
    logic [HEAD_DIM-1:0] k_resp1 [0:1];

    logic directory_in_ready;
    logic class_valid;
    logic [CLASS_W-1:0] class_score;
    logic [COUNT_W-1:0] class_multiplicity;
    logic class_last;
    logic active_valid;
    logic active_ready;
    logic [PAIR_ID_W-1:0] active_pair_id;
    logic signed [SCORE_W-1:0] active_score_q7;
    logic [1:0] active_temporal_mask;
    logic [1:0] active_k_mask;
    logic active_last;
    logic signed [SCORE_W-1:0] row_max_q7;
    logic [1:0] k_read_req_valid;
    logic [1:0] k_read_resp_valid;
    logic [HEAD_DIM-1:0] k_read_resp_k0;
    logic [HEAD_DIM-1:0] k_read_resp_k1;

    logic signed [SCORE_W-1:0] class_score_q7;
    logic signed [SCORE_W-1:0] class_delta_q7;
    logic [15:0] class_exp_q8;
    logic [31:0] class_sum_term;
    logic [31:0] row_sum_q8_q;
    emit_state_t emit_state_q;
    logic [1:0] emit_mask_q;
    logic [PAIR_ID_W-1:0] emit_pair_id_q;
    logic signed [SCORE_W-1:0] emit_score_q7_q;
    logic emit_active_last_q;
    logic [HEAD_DIM-1:0] emit_k0_q;
    logic [HEAD_DIM-1:0] emit_k1_q;
    logic emit_time_sel;
    logic signed [SCORE_W-1:0] emit_delta_q7;
    logic [15:0] emit_exp_q8;
    logic [GATE_W-1:0] emit_gate_q17;
    logic class_phase_done_q;
    logic preserve_mean_q;
    logic protocol_error_q;
    logic [31:0] emitted_tokens_q;
    logic last_row_done_q;
    logic [1:0] skip_ws_q;
    logic [0:0] retire_ws_q [0:1];
    logic retire_wr_q;
    logic retire_rd_q;
    logic [1:0] retire_count_q;
    logic start_skip;
    logic [0:0] start_ws;
    logic [0:0] retire_head;
    logic can_skip_retire;
    logic can_emit_start;
    logic start_any;
    logic emit_complete;
    logic retire_any;

    integer wi;

    assign idle_oh[0] = ws_state_q[0] == WS_IDLE;
    assign idle_oh[1] = ws_state_q[1] == WS_IDLE;
    assign build_oh[0] = ws_state_q[0] == WS_BUILD;
    assign build_oh[1] = ws_state_q[1] == WS_BUILD;
    assign build_ready = idle_oh[0] || idle_oh[1];
    assign start_ws = idle_oh[0] ? 1'b0 : 1'b1;
    assign start_accept = window_start && build_ready && row_k_present;
    assign start_skip = window_start && build_ready && !row_k_present;
    assign start_any = start_accept || start_skip;
    assign start_reject = window_start && !build_ready;
    assign retire_head = retire_ws_q[retire_rd_q];
    assign can_skip_retire = (retire_count_q != 2'd0)
                          && skip_ws_q[retire_head]
                          && (ws_state_q[retire_head] == WS_SEAL)
                          && (ws_state_q[0] != WS_EMIT)
                          && (ws_state_q[1] != WS_EMIT);
    assign can_emit_start = (retire_count_q != 2'd0)
                         && !skip_ws_q[retire_head]
                         && (ws_state_q[retire_head] == WS_SEAL)
                         && (ws_state_q[0] != WS_EMIT)
                         && (ws_state_q[1] != WS_EMIT);
    assign emit_complete = emit_valid_sel && dir_window_done[emit_sel_q]
                        && (emit_state_q == EMIT_IDLE) && class_phase_done_q;
    assign retire_any = can_skip_retire || emit_complete;
    assign pair_ready = encoder_pair_ready && |build_oh && !window_start;
    assign encoder_pair_valid = pair_valid && |build_oh && !window_start;

    generate
        if (MSSB5_SCORE_FRONT) begin : g_mssb5_encoder
            h67_mssb5_temporal_slot_encoder #(
                .HEAD_DIM(HEAD_DIM), .PAIRS(PAIRS), .SCORE_W(SCORE_W),
                .PAIR_ID_W(PAIR_ID_W), .QUOTIENT_ENABLE(QUOTIENT_ENABLE)
            ) u_encoder (
                .clk_core(clk_core), .rst_core(rst_core),
                .window_start(start_accept), .pair_valid(encoder_pair_valid),
                .pair_ready(encoder_pair_ready), .pair_id(pair_id),
                .q_pair(q_pair), .k_pair(k_pair), .packet_valid(packet_valid),
                .packet_ready(packet_ready), .packet_slot_count(packet_slot_count),
                .packet_slot0(packet_slot0), .packet_slot1(packet_slot1),
                .pair_commit(encoder_pair_commit), .protocol_error(encoder_error),
                .perf_pairs(perf_pairs), .perf_slots(perf_slots),
                .perf_equal_pairs(perf_equal_pairs)
            );
        end else begin : g_direct_encoder
            h67_temporal_slot_encoder #(
                .HEAD_DIM(HEAD_DIM), .PAIRS(PAIRS), .SCORE_W(SCORE_W),
                .PAIR_ID_W(PAIR_ID_W), .QUOTIENT_ENABLE(QUOTIENT_ENABLE)
            ) u_encoder (
                .clk_core(clk_core), .rst_core(rst_core),
                .window_start(start_accept), .pair_valid(encoder_pair_valid),
                .pair_ready(encoder_pair_ready), .pair_id(pair_id),
                .q_pair(q_pair), .k_pair(k_pair), .packet_valid(packet_valid),
                .packet_ready(packet_ready), .packet_slot_count(packet_slot_count),
                .packet_slot0(packet_slot0), .packet_slot1(packet_slot1),
                .pair_commit(encoder_pair_commit), .protocol_error(encoder_error),
                .perf_pairs(perf_pairs), .perf_slots(perf_slots),
                .perf_equal_pairs(perf_equal_pairs)
            );
        end
    endgenerate

    h67_temporal_slot_fifo_2s #(
        .DEPTH(SLOT_FIFO_DEPTH),
        .OCC_W(FIFO_OCC_W)
    ) u_slot_fifo (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(start_accept),
        .enq_valid(packet_valid),
        .enq_ready(packet_ready),
        .enq_count(packet_slot_count),
        .enq_slot0(packet_slot0),
        .enq_slot1(packet_slot1),
        .deq_valid(fifo_valid),
        .deq_ready(fifo_ready),
        .deq_count(fifo_count),
        .deq_slot0(fifo_slot0),
        .deq_slot1(fifo_slot1),
        .occupancy(fifo_occ),
        .max_occupancy(),
        .protocol_error(fifo_error)
    );

    assign slot0_score = fifo_slot0[7:0];
    assign slot0_temporal_mask = fifo_slot0[9:8];
    assign slot0_active_mask = fifo_slot0[11:10];
    assign slot0_pair_last = fifo_slot0[12];
    assign slot0_shape_legal = (!pair_open_q
                             && slot0_temporal_mask == 2'b01
                             && !slot0_pair_last)
                            || (pair_open_q
                             && slot0_temporal_mask == 2'b10
                             && slot0_pair_last)
                            || (!pair_open_q
                             && slot0_temporal_mask == 2'b11
                             && slot0_pair_last);
    assign slot0_legal = fifo_slot0[15:13] == 0
                      && slot0_temporal_mask != 0
                      && (slot0_active_mask & ~slot0_temporal_mask) == 0
                      && 32'(slot0_score) <= 32'(MAX_SCORE)
                      && 32'(decoded_pairs_q) < 32'(PAIRS)
                      && slot0_shape_legal;
    assign open_after_slot0 = slot0_temporal_mask == 2'b01 && !slot0_pair_last;
    assign slot1_pair_id = decoded_pairs_q + PAIR_COUNT_W'(slot0_pair_last);
    assign slot1_score = fifo_slot1[7:0];
    assign slot1_temporal_mask = fifo_slot1[9:8];
    assign slot1_active_mask = fifo_slot1[11:10];
    assign slot1_pair_last = fifo_slot1[12];
    assign slot1_shape_legal = (!open_after_slot0
                             && slot1_temporal_mask == 2'b01
                             && !slot1_pair_last)
                            || (open_after_slot0
                             && slot1_temporal_mask == 2'b10
                             && slot1_pair_last)
                            || (!open_after_slot0
                             && slot1_temporal_mask == 2'b11
                             && slot1_pair_last);
    assign slot1_legal = fifo_slot1[15:13] == 0
                      && slot1_temporal_mask != 0
                      && (slot1_active_mask & ~slot1_temporal_mask) == 0
                      && 32'(slot1_score) <= 32'(MAX_SCORE)
                      && 32'(slot1_pair_id) < 32'(PAIRS)
                      && slot1_shape_legal;
    assign open_after_slot1 = slot1_temporal_mask == 2'b01 && !slot1_pair_last;
    assign closed_pair_count = {1'b0, slot0_pair_last}
                             + {1'b0, fifo_count == 2 && slot1_pair_last};
    assign batch_legal = slot0_legal && (fifo_count == 1 || slot1_legal);
    assign directory_in_ready = dir_in_ready[build_sel_q];
    assign directory_in_valid = fifo_valid && batch_legal
                              && descriptor_issue_enable && |build_oh;
    assign fifo_ready = fifo_valid && !batch_legal
                      ? 1'b1
                      : directory_in_ready && descriptor_issue_enable && |build_oh;
    assign batch_fire = fifo_valid && fifo_ready;
    assign seal_ready = |build_oh
                     && decoded_pairs_q == PAIR_COUNT_W'(PAIRS)
                     && fifo_occ == 0
                     && !pair_open_q;

    genvar gi;
    generate
        for (gi = 0; gi < 2; gi = gi + 1) begin : g_ws
            assign dir_window_start[gi] = start_accept && (
                idle_oh[0] ? (gi == 0) : (gi == 1)
            );
            assign dir_window_seal[gi] = window_seal && seal_ready
                                       && build_sel_q == 1'(gi);
            assign k_start[gi] = dir_window_start[gi];
            assign k_write_valid[gi] = encoder_pair_commit && !window_start
                                     && build_sel_q == 1'(gi);

            h67_temporal_weighted_scs_directory_2s #(
                .MAX_SCORE(MAX_SCORE),
                .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
                .SCORE_W(SCORE_W),
                .PAIR_ID_W(PAIR_ID_W),
                .COUNT_W(COUNT_W),
                .CLASS_W(CLASS_W)
            ) u_directory (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .window_start(dir_window_start[gi]),
                .window_seal(dir_window_seal[gi]),
                .window_ready(dir_window_ready[gi]),
                .window_done(dir_window_done[gi]),
                .in_valid(directory_in_valid && build_sel_q == 1'(gi)),
                .in_ready(dir_in_ready[gi]),
                .in_count(fifo_count),
                .in0_pair_id(PAIR_ID_W'(decoded_pairs_q)),
                .in0_score_q7($signed({{(SCORE_W-8){1'b0}}, slot0_score})),
                .in0_temporal_mask(slot0_temporal_mask),
                .in0_active_mask(slot0_active_mask),
                .in1_pair_id(PAIR_ID_W'(slot1_pair_id)),
                .in1_score_q7($signed({{(SCORE_W-8){1'b0}}, slot1_score})),
                .in1_temporal_mask(slot1_temporal_mask),
                .in1_active_mask(slot1_active_mask),
                .class_valid(dir_class_valid[gi]),
                .class_ready(emit_valid_sel && emit_sel_q == 1'(gi)),
                .class_score(dir_class_score[gi]),
                .class_multiplicity(dir_class_mult[gi]),
                .class_last(dir_class_last[gi]),
                .active_valid(dir_active_valid[gi]),
                .active_ready(active_ready && emit_sel_q == 1'(gi)),
                .active_pair_id(dir_active_pair[gi]),
                .active_score_q7(dir_active_score[gi]),
                .active_temporal_mask(dir_active_temporal[gi]),
                .active_k_mask(dir_active_k[gi]),
                .active_last(dir_active_last[gi]),
                .row_max_q7(dir_row_max[gi]),
                .protocol_error(dir_error[gi]),
                .perf_quotient_descriptors(),
                .perf_original_tokens(),
                .perf_active_entries()
            );

            h67_sync_dual_bank_k_store #(
                .HEAD_DIM(HEAD_DIM),
                .PAIRS(PAIRS),
                .ADDR_W(PAIR_ID_W),
                .MEMORY_IMPL(MEMORY_IMPL)
            ) u_k_store (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .window_start(k_start[gi]),
                .write_valid(k_write_valid[gi]),
                .write_addr(pair_id),
                .write_k_pair(k_pair),
                .read_req_valid(k_read_req_valid & {2{emit_sel_q == 1'(gi)}}),
                .read_req_addr(active_pair_id),
                .read_resp_valid(k_resp_valid[gi]),
                .read_resp_k0(k_resp0[gi]),
                .read_resp_k1(k_resp1[gi]),
                .perf_read_transactions(),
                .perf_read_bits(),
                .protocol_error(k_error[gi])
            );
        end
    endgenerate

    assign emit_valid_sel = (ws_state_q[emit_sel_q] == WS_EMIT);
    assign class_valid = dir_class_valid[emit_sel_q] && emit_valid_sel;
    assign class_score = dir_class_score[emit_sel_q];
    assign class_multiplicity = dir_class_mult[emit_sel_q];
    assign class_last = dir_class_last[emit_sel_q];
    assign active_valid = dir_active_valid[emit_sel_q] && emit_valid_sel;
    assign active_pair_id = dir_active_pair[emit_sel_q];
    assign active_score_q7 = dir_active_score[emit_sel_q];
    assign active_temporal_mask = dir_active_temporal[emit_sel_q];
    assign active_k_mask = dir_active_k[emit_sel_q];
    assign active_last = dir_active_last[emit_sel_q];
    assign row_max_q7 = dir_row_max[emit_sel_q];
    assign k_read_req_valid = (active_valid && active_ready) ? active_k_mask : 2'b00;
    assign k_read_resp_valid = k_resp_valid[emit_sel_q];
    assign k_read_resp_k0 = k_resp0[emit_sel_q];
    assign k_read_resp_k1 = k_resp1[emit_sel_q];

    assign class_score_q7 = $signed({{(SCORE_W-CLASS_W){1'b0}}, class_score});
    assign class_delta_q7 = class_score_q7 - row_max_q7;

    ttx_exp2_lut_q8 #(.SCORE_W(SCORE_W), .SCORE_FRAC(7)) u_class_exp (
        .delta_q7(class_delta_q7),
        .exp_q8(class_exp_q8)
    );
    assign class_sum_term = 32'(class_exp_q8) * 32'(class_multiplicity);
    assign active_ready = emit_state_q == EMIT_IDLE && class_phase_done_q;
    assign emit_time_sel = !emit_mask_q[0];
    assign emit_delta_q7 = emit_score_q7_q - row_max_q7;

    ttx_exp2_lut_q8 #(.SCORE_W(SCORE_W), .SCORE_FRAC(7)) u_emit_exp (
        .delta_q7(emit_delta_q7),
        .exp_q8(emit_exp_q8)
    );
    ttx_gate_quant_q17 #(
        .TOKEN_W(TOKEN_W),
        .GATE_W(GATE_W),
        .GATE_FRAC(7)
    ) u_gate_quant (
        .exp_q8(emit_exp_q8),
        .row_sum_q8(row_sum_q8_q),
        .n_tokens(TOKEN_W'(2 * PAIRS)),
        .preserve_mean(preserve_mean_q),
        .gate_q17(emit_gate_q17)
    );

    assign out_valid = emit_state_q == EMIT_ACTIVE && emit_valid_sel;
    assign out_token_id = TOKEN_W'(2 * 32'(emit_pair_id_q) + 32'(emit_time_sel));
    assign out_k_bits = emit_time_sel ? emit_k1_q : emit_k0_q;
    assign out_gate_q17 = emit_gate_q17;
    assign out_last = out_valid && emit_active_last_q
                   && (emit_mask_q == 2'b01 || emit_mask_q == 2'b10);
    assign emit_active = emit_valid_sel;
    assign last_row_done = last_row_done_q;
    assign protocol_error = encoder_error || fifo_error || protocol_error_q
                         || dir_error[0] || dir_error[1]
                         || k_error[0] || k_error[1]
                         || start_reject;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            ws_state_q[0] <= WS_IDLE;
            ws_state_q[1] <= WS_IDLE;
            build_sel_q <= 1'b0;
            emit_sel_q <= 1'b0;
            decoded_pairs_q <= '0;
            pair_open_q <= 1'b0;
            emit_state_q <= EMIT_IDLE;
            emit_mask_q <= '0;
            class_phase_done_q <= 1'b0;
            row_sum_q8_q <= '0;
            preserve_mean_q <= 1'b1;
            protocol_error_q <= 1'b0;
            emitted_tokens_q <= '0;
            last_row_done_q <= 1'b0;
            skip_ws_q <= 2'b00;
            retire_ws_q[0] <= 1'b0;
            retire_ws_q[1] <= 1'b0;
            retire_wr_q <= 1'b0;
            retire_rd_q <= 1'b0;
            retire_count_q <= 2'd0;
        end else begin
            last_row_done_q <= 1'b0;
            if (start_reject)
                protocol_error_q <= 1'b1;
            if (start_skip) begin
                build_sel_q <= start_ws;
                ws_state_q[start_ws] <= WS_SEAL;
                skip_ws_q[start_ws] <= 1'b1;
                retire_ws_q[retire_wr_q] <= start_ws;
                retire_wr_q <= retire_wr_q + 1'b1;
            end
            if (start_accept) begin
                build_sel_q <= start_ws;
                ws_state_q[start_ws] <= WS_BUILD;
                skip_ws_q[start_ws] <= 1'b0;
                decoded_pairs_q <= '0;
                pair_open_q <= 1'b0;
                preserve_mean_q <= cfg_preserve_mean;
                retire_ws_q[retire_wr_q] <= start_ws;
                retire_wr_q <= retire_wr_q + 1'b1;
            end
            if (fifo_valid && !batch_legal)
                protocol_error_q <= 1'b1;
            else if (batch_fire) begin
                pair_open_q <= fifo_count == 2 ? open_after_slot1 : open_after_slot0;
                decoded_pairs_q <= decoded_pairs_q + PAIR_COUNT_W'(closed_pair_count);
            end
            if (window_seal && seal_ready && |build_oh) begin
                ws_state_q[build_sel_q] <= WS_SEAL;
            end

            if (can_skip_retire) begin
                ws_state_q[retire_head] <= WS_IDLE;
                skip_ws_q[retire_head] <= 1'b0;
                last_row_done_q <= 1'b1;
                retire_rd_q <= retire_rd_q + 1'b1;
            end else if (can_emit_start) begin
                emit_sel_q <= retire_head;
                ws_state_q[retire_head] <= WS_EMIT;
                emit_state_q <= EMIT_IDLE;
                class_phase_done_q <= 1'b0;
                row_sum_q8_q <= '0;
                emitted_tokens_q <= '0;
            end

            if (class_valid) begin
                row_sum_q8_q <= row_sum_q8_q + class_sum_term;
                if (class_last)
                    class_phase_done_q <= 1'b1;
            end

            case (emit_state_q)
                EMIT_IDLE: begin
                    if (emit_valid_sel && active_valid && active_ready) begin
                        if (!class_phase_done_q
                            || active_k_mask == 0
                            || (active_k_mask & ~active_temporal_mask) != 0)
                            protocol_error_q <= 1'b1;
                        else begin
                            emit_mask_q <= active_k_mask;
                            emit_pair_id_q <= active_pair_id;
                            emit_score_q7_q <= active_score_q7;
                            emit_active_last_q <= active_last;
                            emit_state_q <= EMIT_WAIT_K;
                        end
                    end
                end
                EMIT_WAIT_K: begin
                    if ((k_read_resp_valid & emit_mask_q) == emit_mask_q) begin
                        if (emit_mask_q[0])
                            emit_k0_q <= k_read_resp_k0;
                        if (emit_mask_q[1])
                            emit_k1_q <= k_read_resp_k1;
                        emit_state_q <= EMIT_ACTIVE;
                    end
                end
                EMIT_ACTIVE: begin
                    if (out_valid && out_ready) begin
                        emitted_tokens_q <= emitted_tokens_q + 1'b1;
                        if (emit_mask_q[0]) begin
                            emit_mask_q[0] <= 1'b0;
                            if (!emit_mask_q[1])
                                emit_state_q <= EMIT_IDLE;
                        end else begin
                            emit_mask_q[1] <= 1'b0;
                            emit_state_q <= EMIT_IDLE;
                        end
                    end
                end
                default: emit_state_q <= EMIT_IDLE;
            endcase

            if (emit_complete) begin
                ws_state_q[emit_sel_q] <= WS_IDLE;
                last_row_done_q <= 1'b1;
                retire_rd_q <= retire_rd_q + 1'b1;
            end

            unique case ({start_any, retire_any})
                2'b10: retire_count_q <= retire_count_q + 2'd1;
                2'b01: retire_count_q <= retire_count_q - 2'd1;
                default: ;
            endcase
        end
    end
endmodule

`default_nettype wire
