`timescale 1ns/1ps
`default_nettype none

// 共享row-SRAM的公平顶层：ZK_BYPASS_ENABLE=0为RQTB2S，=1为TTB8-ZKQI。
module h67_zkqi_row_shiftmax_top #(
    parameter int HEAD_DIM = 32,
    parameter int PAIRS = 225,
    parameter int BUNDLE_SIZE = 8,
    // active bitmap本身保存全部backlog；这里只需一个弹性bundle skid槽。
    parameter int ACTIVE_FIFO_DEPTH = 1,
    parameter int SCORE_W = 16,
    parameter int GATE_W = 9,
    parameter int THRESHOLD_W = 8,
    parameter int PAIR_ID_W = (PAIRS <= 1) ? 1 : $clog2(PAIRS),
    parameter int TOKEN_W = $clog2(2 * PAIRS + 1),
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 2 * PAIRS,
    parameter int COUNT_W = $clog2(2 * PAIRS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1),
    parameter int BUNDLE_COUNT = (PAIRS + BUNDLE_SIZE - 1) / BUNDLE_SIZE,
    parameter int BUNDLE_ID_W = (BUNDLE_COUNT <= 1) ? 1 : $clog2(BUNDLE_COUNT),
    parameter int FIFO_OCC_W = $clog2(ACTIVE_FIFO_DEPTH + 1),
    parameter bit ZK_BYPASS_ENABLE = 1'b1,
    // 0 uses two parallel Direct32 engines; 8/16 selects exact TARE residual.
    parameter int ACTIVE_SCORE_RESIDUAL_W = 0,
    // 0为普通225-bit pair bitmap逐pair扫描，1为TTB8分层跳扫。
    parameter bit BUNDLE_SKIP_ENABLE = 1'b1,
    // 0为bit-exact行为存储，1为Nangate45四宏物理代理。
    parameter int ROW_MEMORY_IMPL = 0,
    // active descriptor偶/奇双bank：0为行为模型，1为两个256x32开放宏。
    parameter int DIRECTORY_MEMORY_IMPL = 0
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       row_load_start,
    input  logic                       row_load_valid,
    output logic                       row_load_ready,
    input  logic [PAIR_ID_W-1:0]       row_load_pair_id,
    input  logic [2*HEAD_DIM-1:0]      row_load_q_pair,
    input  logic [2*HEAD_DIM-1:0]      row_load_k_pair,
    output logic                       row_loaded,

    input  logic                       window_start,
    input  logic                       window_seal,
    input  logic                       descriptor_issue_enable,
    input  logic                       cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]     cfg_threshold_q8,
    output logic                       seal_ready,
    output logic                       window_done,

    output logic                       out_valid,
    input  logic                       out_ready,
    output logic                       out_last,
    output logic [TOKEN_W-1:0]         out_token_id,
    output logic [HEAD_DIM-1:0]        out_k_bits,
    output logic [GATE_W-1:0]          out_gate_q17,
    output logic [THRESHOLD_W-1:0]     out_threshold_q8,

    output logic                       protocol_error,
    output logic [31:0]                perf_score_pairs,
    output logic [31:0]                perf_score_slots,
    output logic [31:0]                perf_original_tokens,
    output logic [31:0]                perf_equal_pairs,
    output logic [31:0]                perf_seeded_tokens,
    output logic [31:0]                perf_active_entries,
    output logic [31:0]                perf_class_transactions,
    output logic [31:0]                perf_exp_transactions,
    output logic [31:0]                perf_emitted_tokens,
    output logic [31:0]                perf_row_read_transactions,
    output logic [31:0]                perf_row_read_bits,
    output logic [31:0]                perf_preload_cycles,
    output logic [31:0]                perf_total_cycles,
    output logic [31:0]                perf_score_stall_cycles,
    output logic [31:0]                perf_output_stall_cycles,
    output logic [31:0]                perf_preclassified_pairs,
    output logic [31:0]                perf_metadata_bits,
    output logic [FIFO_OCC_W-1:0]      perf_fifo_occupancy,
    output logic [FIFO_OCC_W-1:0]      perf_fifo_max_occupancy,
    output logic [31:0]                perf_tare_dense_fallbacks
);
    typedef enum logic [1:0] {
        EMIT_IDLE,
        EMIT_WAIT_K,
        EMIT_ACTIVE
    } emit_state_t;

    localparam int LOAD_COUNT_W = $clog2(PAIRS + 1);

    logic window_active_q;
    logic window_start_accept;
    logic window_start_reject;
    logic load_fire;
    logic store_write_ready;
    logic store_error;

    logic baseline_row_loaded_q;
    logic [LOAD_COUNT_W-1:0] baseline_load_count_q;
    logic baseline_load_ready;
    logic baseline_load_error_q;

    logic meta_load_ready;
    logic meta_row_loaded;
    logic meta_scan_valid;
    logic meta_scan_ready;
    logic [BUNDLE_ID_W-1:0] meta_scan_bundle_id;
    logic [BUNDLE_SIZE-1:0] meta_scan_active_mask;
    logic meta_scan_done;
    logic meta_done_q;
    logic [COUNT_W-1:0] meta_zk_count0;
    logic [COUNT_W-1:0] meta_zk_count1;
    logic [COUNT_W-1:0] meta_zk_count2;
    logic [31:0] meta_preclassified_pairs;
    logic [31:0] meta_active_pairs;
    logic [31:0] meta_metadata_bits;
    logic meta_error;

    logic fifo_enq_valid;
    logic fifo_enq_ready;
    logic fifo_pair_valid;
    logic fifo_pair_ready;
    logic [PAIR_ID_W-1:0] fifo_pair_id;
    logic [FIFO_OCC_W-1:0] fifo_occupancy;
    logic [FIFO_OCC_W-1:0] fifo_max_occupancy;
    logic fifo_empty;
    logic fifo_error;

    logic [PAIR_ID_W-1:0] baseline_score_next_q;
    logic baseline_all_issued_q;
    logic score_source_valid;
    logic score_source_ready;
    logic [PAIR_ID_W-1:0] score_source_pair_id;
    logic score_request_fire;

    logic store_read_req_valid;
    logic store_read_req_ready;
    logic [PAIR_ID_W-1:0] store_read_req_addr;
    logic store_read_req_q;
    logic [1:0] store_read_req_k_mask;
    logic store_read_req_score_tag;
    logic store_read_resp_valid;
    logic store_read_resp_ready;
    logic [PAIR_ID_W-1:0] store_read_resp_addr;
    logic [2*HEAD_DIM-1:0] store_read_resp_q_pair;
    logic [2*HEAD_DIM-1:0] store_read_resp_k_pair;
    logic [1:0] store_read_resp_k_mask;
    logic store_read_resp_score_tag;

    logic signed [SCORE_W-1:0] score0_w;
    logic signed [SCORE_W-1:0] score1_w;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_overlap0;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_same_zero0;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_motion0;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_overlap1;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_same_zero1;
    logic [$clog2(HEAD_DIM+1)-1:0] unused_motion1;
    logic score_equal;
    logic score_active0;
    logic score_active1;
    logic [1:0] score_packet_count;
    logic score_front_valid;
    logic score_front_ready;
    logic [PAIR_ID_W-1:0] score_front_pair_id;
    logic signed [SCORE_W-1:0] score_front_score0;
    logic signed [SCORE_W-1:0] score_front_score1;
    logic [1:0] score_front_k_active;
    logic score_front_error;
    logic score_response_ready;
    logic tare_in_ready;
    logic [5:0] unused_tare_update_count;
    logic unused_tare_dense_fallback;
    logic signed [12:0] unused_tare_delta_raw16;

    logic directory_in_valid;
    logic directory_in_ready;
    logic directory_seal;
    logic directory_ready;
    logic directory_done;
    logic directory_error;
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
    logic [31:0] directory_slots;
    logic [31:0] directory_original_tokens;
    logic [31:0] directory_active_entries;
    logic [31:0] directory_seeded_tokens;

    logic build_complete;
    logic seal_issued_q;
    logic class_phase_done_q;
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

    logic protocol_error_q;
    logic [31:0] score_pairs_q;
    logic [31:0] equal_pairs_q;
    logic [31:0] class_transactions_q;
    logic [31:0] emitted_tokens_q;
    logic [31:0] total_cycles_q;
    logic [31:0] score_stall_cycles_q;
    logic [31:0] output_stall_cycles_q;
    logic [31:0] tare_dense_fallbacks_q;
    logic [THRESHOLD_W-1:0] threshold_q8_q;
    logic preserve_mean_q;
    logic preload_active_q;
    logic [31:0] preload_cycles_q;

    assign window_start_accept = window_start && row_loaded
                               && directory_ready
                               && (!window_active_q || window_done);
    assign window_start_reject = window_start && !window_start_accept;

    generate
        if (ZK_BYPASS_ENABLE && BUNDLE_SKIP_ENABLE) begin : g_metadata
            h67_ttb8_metadata_builder #(
                .HEAD_DIM(HEAD_DIM),
                .PAIRS(PAIRS),
                .BUNDLE_SIZE(BUNDLE_SIZE),
                .PAIR_ID_W(PAIR_ID_W),
                .BUNDLE_COUNT(BUNDLE_COUNT),
                .BUNDLE_ID_W(BUNDLE_ID_W),
                .COUNT_W(COUNT_W)
            ) u_metadata (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .row_load_start(row_load_start && !window_active_q),
                .window_start(window_start_accept),
                .load_valid(row_load_valid && store_write_ready && !window_active_q),
                .load_ready(meta_load_ready),
                .load_pair_id(row_load_pair_id),
                .load_q_pair(row_load_q_pair),
                .load_k_pair(row_load_k_pair),
                .row_loaded(meta_row_loaded),
                .scan_valid(meta_scan_valid),
                .scan_ready(meta_scan_ready),
                .scan_bundle_id(meta_scan_bundle_id),
                .scan_active_mask(meta_scan_active_mask),
                .scan_last(),
                .scan_done(meta_scan_done),
                .zk_count0(meta_zk_count0),
                .zk_count1(meta_zk_count1),
                .zk_count2(meta_zk_count2),
                .perf_preclassified_pairs(meta_preclassified_pairs),
                .perf_active_pairs(meta_active_pairs),
                .perf_metadata_bits(meta_metadata_bits),
                .protocol_error(meta_error)
            );

            h67_active_bundle_fifo #(
                .PAIRS(PAIRS),
                .BUNDLE_SIZE(BUNDLE_SIZE),
                .DEPTH(ACTIVE_FIFO_DEPTH),
                .BUNDLE_COUNT(BUNDLE_COUNT),
                .BUNDLE_ID_W(BUNDLE_ID_W),
                .PAIR_ID_W(PAIR_ID_W),
                .OCC_W(FIFO_OCC_W)
            ) u_active_fifo (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .window_start(window_start_accept),
                .enq_valid(fifo_enq_valid),
                .enq_ready(fifo_enq_ready),
                .enq_bundle_id(meta_scan_bundle_id),
                .enq_active_mask(meta_scan_active_mask),
                .pair_valid(fifo_pair_valid),
                .pair_ready(fifo_pair_ready),
                .pair_id(fifo_pair_id),
                .pair_bundle_id(),
                .pair_lane(),
                .occupancy(fifo_occupancy),
                .max_occupancy(fifo_max_occupancy),
                .empty(fifo_empty),
                .protocol_error(fifo_error)
            );
        end else if (ZK_BYPASS_ENABLE) begin : g_pair_bitmap_metadata
            h67_pair_bitmap_metadata_builder #(
                .HEAD_DIM(HEAD_DIM),
                .PAIRS(PAIRS),
                .PAIR_ID_W(PAIR_ID_W),
                .COUNT_W(COUNT_W)
            ) u_pair_bitmap (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .row_load_start(row_load_start && !window_active_q),
                .window_start(window_start_accept),
                .load_valid(row_load_valid && store_write_ready && !window_active_q),
                .load_ready(meta_load_ready),
                .load_pair_id(row_load_pair_id),
                .load_q_pair(row_load_q_pair),
                .load_k_pair(row_load_k_pair),
                .row_loaded(meta_row_loaded),
                .pair_valid(fifo_pair_valid),
                .pair_ready(fifo_pair_ready),
                .pair_id(fifo_pair_id),
                .scan_done(meta_scan_done),
                .zk_count0(meta_zk_count0),
                .zk_count1(meta_zk_count1),
                .zk_count2(meta_zk_count2),
                .perf_preclassified_pairs(meta_preclassified_pairs),
                .perf_active_pairs(meta_active_pairs),
                .perf_metadata_bits(meta_metadata_bits),
                .protocol_error(meta_error)
            );
            assign meta_scan_valid = 1'b0;
            assign meta_scan_bundle_id = '0;
            assign meta_scan_active_mask = '0;
            assign fifo_enq_ready = 1'b0;
            assign fifo_occupancy = '0;
            assign fifo_max_occupancy = '0;
            assign fifo_empty = !fifo_pair_valid;
            assign fifo_error = 1'b0;
        end else begin : g_no_metadata
            assign meta_load_ready = 1'b0;
            assign meta_row_loaded = 1'b0;
            assign meta_scan_valid = 1'b0;
            assign meta_scan_bundle_id = '0;
            assign meta_scan_active_mask = '0;
            assign meta_scan_done = 1'b0;
            assign meta_zk_count0 = '0;
            assign meta_zk_count1 = '0;
            assign meta_zk_count2 = '0;
            assign meta_preclassified_pairs = '0;
            assign meta_active_pairs = PAIRS;
            assign meta_metadata_bits = '0;
            assign meta_error = 1'b0;
            assign fifo_enq_ready = 1'b0;
            assign fifo_pair_valid = 1'b0;
            assign fifo_pair_id = '0;
            assign fifo_occupancy = '0;
            assign fifo_max_occupancy = '0;
            assign fifo_empty = 1'b1;
            assign fifo_error = 1'b0;
        end
    endgenerate

    assign baseline_load_ready = !baseline_row_loaded_q && !window_active_q;
    assign row_load_ready = store_write_ready
                          && (ZK_BYPASS_ENABLE ? meta_load_ready
                                               : baseline_load_ready);
    assign load_fire = row_load_valid && row_load_ready;
    assign row_loaded = ZK_BYPASS_ENABLE ? meta_row_loaded
                                         : baseline_row_loaded_q;

    generate
        if (ROW_MEMORY_IMPL == 0) begin : g_behavior_row_store
            h67_sync_qk_row_store #(
                .HEAD_DIM(HEAD_DIM),
                .PAIRS(PAIRS),
                .ADDR_W(PAIR_ID_W)
            ) u_row_store (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .row_reset(window_start_accept),
                .write_valid(load_fire),
                .write_ready(store_write_ready),
                .write_addr(row_load_pair_id),
                .write_q_pair(row_load_q_pair),
                .write_k_pair(row_load_k_pair),
                .read_req_valid(store_read_req_valid),
                .read_req_ready(store_read_req_ready),
                .read_req_addr(store_read_req_addr),
                .read_req_q(store_read_req_q),
                .read_req_k_mask(store_read_req_k_mask),
                .read_req_score_tag(store_read_req_score_tag),
                .read_resp_valid(store_read_resp_valid),
                .read_resp_ready(store_read_resp_ready),
                .read_resp_addr(store_read_resp_addr),
                .read_resp_q_pair(store_read_resp_q_pair),
                .read_resp_k_pair(store_read_resp_k_pair),
                .read_resp_k_mask(store_read_resp_k_mask),
                .read_resp_score_tag(store_read_resp_score_tag),
                .perf_read_transactions(perf_row_read_transactions),
                .perf_read_bits(perf_row_read_bits),
                .protocol_error(store_error)
            );
        end else begin : g_macro_row_store
            h67_fakeram45_qk_row_store #(
                .HEAD_DIM(HEAD_DIM),
                .PAIRS(PAIRS),
                .ADDR_W(PAIR_ID_W)
            ) u_row_store (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .row_reset(window_start_accept),
                .write_valid(load_fire),
                .write_ready(store_write_ready),
                .write_addr(row_load_pair_id),
                .write_q_pair(row_load_q_pair),
                .write_k_pair(row_load_k_pair),
                .read_req_valid(store_read_req_valid),
                .read_req_ready(store_read_req_ready),
                .read_req_addr(store_read_req_addr),
                .read_req_q(store_read_req_q),
                .read_req_k_mask(store_read_req_k_mask),
                .read_req_score_tag(store_read_req_score_tag),
                .read_resp_valid(store_read_resp_valid),
                .read_resp_ready(store_read_resp_ready),
                .read_resp_addr(store_read_resp_addr),
                .read_resp_q_pair(store_read_resp_q_pair),
                .read_resp_k_pair(store_read_resp_k_pair),
                .read_resp_k_mask(store_read_resp_k_mask),
                .read_resp_score_tag(store_read_resp_score_tag),
                .perf_read_transactions(perf_row_read_transactions),
                .perf_read_bits(perf_row_read_bits),
                .protocol_error(store_error)
            );
        end
    endgenerate

    assign fifo_enq_valid = ZK_BYPASS_ENABLE && meta_scan_valid
                          && meta_scan_active_mask != 0;
    assign meta_scan_ready = ZK_BYPASS_ENABLE
                           && (meta_scan_active_mask == 0 || fifo_enq_ready);
    assign score_source_valid = window_active_q && !seal_issued_q
                              && (ZK_BYPASS_ENABLE
                                  ? fifo_pair_valid
                                  : !baseline_all_issued_q);
    assign score_source_pair_id = ZK_BYPASS_ENABLE
                                ? fifo_pair_id : baseline_score_next_q;

    assign store_read_req_valid = score_source_valid
        ? 1'b1
        : (active_valid && emit_state_q == EMIT_IDLE && class_phase_done_q);
    assign store_read_req_addr = score_source_valid
        ? score_source_pair_id : active_pair_id;
    assign store_read_req_q = score_source_valid;
    assign store_read_req_k_mask = score_source_valid ? 2'b11 : active_k_mask;
    assign store_read_req_score_tag = score_source_valid;
    assign score_source_ready = score_source_valid && store_read_req_ready;
    assign score_request_fire = score_source_valid && score_source_ready;
    assign fifo_pair_ready = ZK_BYPASS_ENABLE && score_source_ready;

    assign active_ready = active_valid && !score_source_valid
                       && emit_state_q == EMIT_IDLE && class_phase_done_q
                       && store_read_req_ready;

    generate
        if (ACTIVE_SCORE_RESIDUAL_W == 0) begin : g_direct_score_front
            h67_motionxor_score_q7 #(
                .HEAD_DIM(HEAD_DIM),
                .SCORE_W(SCORE_W),
                .COUNT_W($clog2(HEAD_DIM + 1)),
                .ENABLE_MOTION_XOR(1'b1)
            ) u_score0 (
                .q_bits(store_read_resp_q_pair[HEAD_DIM-1:0]),
                .k_current_bits(store_read_resp_k_pair[HEAD_DIM-1:0]),
                .k_peer_bits(store_read_resp_k_pair[2*HEAD_DIM-1:HEAD_DIM]),
                .overlap(unused_overlap0),
                .same_zero(unused_same_zero0),
                .motion_xor(unused_motion0),
                .score_q7(score0_w)
            );

            h67_motionxor_score_q7 #(
                .HEAD_DIM(HEAD_DIM),
                .SCORE_W(SCORE_W),
                .COUNT_W($clog2(HEAD_DIM + 1)),
                .ENABLE_MOTION_XOR(1'b1)
            ) u_score1 (
                .q_bits(store_read_resp_q_pair[2*HEAD_DIM-1:HEAD_DIM]),
                .k_current_bits(store_read_resp_k_pair[2*HEAD_DIM-1:HEAD_DIM]),
                .k_peer_bits(store_read_resp_k_pair[HEAD_DIM-1:0]),
                .overlap(unused_overlap1),
                .same_zero(unused_same_zero1),
                .motion_xor(unused_motion1),
                .score_q7(score1_w)
            );

            assign score_front_valid = store_read_resp_valid
                                     && store_read_resp_score_tag
                                     && descriptor_issue_enable;
            assign score_front_pair_id = store_read_resp_addr;
            assign score_front_score0 = score0_w;
            assign score_front_score1 = score1_w;
            assign score_front_k_active = {
                |store_read_resp_k_pair[2*HEAD_DIM-1:HEAD_DIM],
                |store_read_resp_k_pair[HEAD_DIM-1:0]
            };
            assign score_response_ready = score_front_ready
                                        && descriptor_issue_enable;
            assign score_front_error = 1'b0;
            assign tare_in_ready = 1'b0;
            assign unused_tare_update_count = '0;
            assign unused_tare_dense_fallback = 1'b0;
            assign unused_tare_delta_raw16 = '0;
        end else begin : g_tare_score_front
            h67_tare_score_pair #(
                .HEAD_DIM(HEAD_DIM),
                .RESIDUAL_W(ACTIVE_SCORE_RESIDUAL_W),
                .TAG_W(PAIR_ID_W),
                .SCORE_W(SCORE_W)
            ) u_tare (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .window_start(window_start_accept),
                .in_valid(store_read_resp_valid
                          && store_read_resp_score_tag),
                .in_enable(descriptor_issue_enable),
                .in_ready(tare_in_ready),
                .in_tag(store_read_resp_addr),
                .in_q_pair(store_read_resp_q_pair),
                .in_k_pair(store_read_resp_k_pair),
                .out_valid(score_front_valid),
                .out_ready(score_front_ready),
                .out_tag(score_front_pair_id),
                .out_score0_q7(score_front_score0),
                .out_score1_q7(score_front_score1),
                .out_k_active(score_front_k_active),
                .out_update_count(unused_tare_update_count),
                .out_dense_fallback(unused_tare_dense_fallback),
                .out_delta_raw16(unused_tare_delta_raw16),
                .protocol_error(score_front_error)
            );
            assign score_response_ready = tare_in_ready;
        end
    endgenerate

    assign score_equal = score_front_score0 == score_front_score1;
    assign score_active0 = score_front_k_active[0];
    assign score_active1 = score_front_k_active[1];
    assign score_packet_count = score_equal ? 1 : 2;
    assign directory_in_valid = score_front_valid;
    assign score_front_ready = directory_in_ready;
    assign store_read_resp_ready = store_read_resp_score_tag
        ? score_response_ready
        : (emit_state_q == EMIT_WAIT_K);

    h67_temporal_weighted_scs_directory_seed_2s #(
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .EXPECTED_TOKENS(2 * PAIRS),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W),
        .ACTIVE_MEMORY_IMPL(DIRECTORY_MEMORY_IMPL)
    ) u_directory (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start_accept),
        .window_seal(directory_seal),
        .window_ready(directory_ready),
        .window_done(directory_done),
        .seed_count0(ZK_BYPASS_ENABLE ? meta_zk_count0 : '0),
        .seed_count1(ZK_BYPASS_ENABLE ? meta_zk_count1 : '0),
        .seed_count2(ZK_BYPASS_ENABLE ? meta_zk_count2 : '0),
        .in_valid(directory_in_valid),
        .in_ready(directory_in_ready),
        .in_count(score_packet_count),
        .in0_pair_id(score_front_pair_id),
        .in0_score_q7(score_front_score0),
        .in0_temporal_mask(score_equal ? 2'b11 : 2'b01),
        .in0_active_mask(score_equal
            ? {score_active1, score_active0} : {1'b0, score_active0}),
        .in1_pair_id(score_front_pair_id),
        .in1_score_q7(score_front_score1),
        .in1_temporal_mask(2'b10),
        .in1_active_mask({score_active1, 1'b0}),
        .class_valid(class_valid),
        .class_ready(1'b1),
        .class_score(class_score),
        .class_multiplicity(class_multiplicity),
        .class_last(class_last),
        .active_valid(active_valid),
        .active_ready(active_ready),
        .active_pair_id(active_pair_id),
        .active_score_q7(active_score_q7),
        .active_temporal_mask(active_temporal_mask),
        .active_k_mask(active_k_mask),
        .active_last(active_last),
        .row_max_q7(row_max_q7),
        .protocol_error(directory_error),
        .perf_quotient_descriptors(directory_slots),
        .perf_original_tokens(directory_original_tokens),
        .perf_active_entries(directory_active_entries),
        .perf_seeded_tokens(directory_seeded_tokens)
    );

    assign build_complete = ZK_BYPASS_ENABLE
        ? (meta_done_q && fifo_empty
           && score_pairs_q == meta_active_pairs
           && !(store_read_resp_valid && store_read_resp_score_tag))
        : (baseline_all_issued_q && score_pairs_q == 32'(PAIRS)
           && !(store_read_resp_valid && store_read_resp_score_tag));
    assign seal_ready = window_active_q && build_complete && !seal_issued_q;
    assign directory_seal = window_seal && seal_ready;

    assign class_score_q7 = $signed({{(SCORE_W-CLASS_W){1'b0}}, class_score});
    assign class_delta_q7 = class_score_q7 - row_max_q7;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(7)
    ) u_class_exp (
        .delta_q7(class_delta_q7),
        .exp_q8(class_exp_q8)
    );

    assign class_sum_term = 32'(class_exp_q8) * 32'(class_multiplicity);
    assign emit_time_sel = !emit_mask_q[0];
    assign emit_delta_q7 = emit_score_q7_q - row_max_q7;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(7)
    ) u_emit_exp (
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

    assign out_valid = emit_state_q == EMIT_ACTIVE && !window_start_accept;
    assign out_token_id = TOKEN_W'(2 * 32'(emit_pair_id_q)
                                + 32'(emit_time_sel));
    assign out_k_bits = emit_time_sel ? emit_k1_q : emit_k0_q;
    assign out_gate_q17 = emit_gate_q17;
    assign out_threshold_q8 = threshold_q8_q;
    assign out_last = out_valid && emit_active_last_q
                   && (emit_mask_q == 2'b01 || emit_mask_q == 2'b10);
    assign window_done = directory_done && emit_state_q == EMIT_IDLE;

    assign protocol_error = protocol_error_q || baseline_load_error_q
                         || store_error || meta_error || fifo_error
                         || directory_error || score_front_error;
    assign perf_score_pairs = score_pairs_q;
    assign perf_score_slots = directory_slots;
    assign perf_original_tokens = directory_original_tokens;
    assign perf_equal_pairs = equal_pairs_q;
    assign perf_seeded_tokens = directory_seeded_tokens;
    assign perf_active_entries = directory_active_entries;
    assign perf_class_transactions = class_transactions_q;
    assign perf_exp_transactions = class_transactions_q
                                 + directory_active_entries;
    assign perf_emitted_tokens = emitted_tokens_q;
    assign perf_preload_cycles = preload_cycles_q;
    assign perf_total_cycles = total_cycles_q;
    assign perf_score_stall_cycles = score_stall_cycles_q;
    assign perf_output_stall_cycles = output_stall_cycles_q;
    assign perf_preclassified_pairs = ZK_BYPASS_ENABLE
                                    ? meta_preclassified_pairs : 0;
    assign perf_metadata_bits = ZK_BYPASS_ENABLE ? meta_metadata_bits : 0;
    assign perf_fifo_occupancy = fifo_occupancy;
    assign perf_fifo_max_occupancy = fifo_max_occupancy;
    assign perf_tare_dense_fallbacks = tare_dense_fallbacks_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            baseline_row_loaded_q <= 1'b0;
            baseline_load_count_q <= '0;
            baseline_load_error_q <= 1'b0;
            baseline_score_next_q <= '0;
            baseline_all_issued_q <= 1'b0;
            meta_done_q <= 1'b0;
            window_active_q <= 1'b0;
            seal_issued_q <= 1'b0;
            class_phase_done_q <= 1'b0;
            emit_state_q <= EMIT_IDLE;
            emit_mask_q <= '0;
            emit_pair_id_q <= '0;
            emit_score_q7_q <= '0;
            emit_active_last_q <= 1'b0;
            emit_k0_q <= '0;
            emit_k1_q <= '0;
            row_sum_q8_q <= '0;
            protocol_error_q <= 1'b0;
            score_pairs_q <= '0;
            equal_pairs_q <= '0;
            class_transactions_q <= '0;
            emitted_tokens_q <= '0;
            total_cycles_q <= '0;
            score_stall_cycles_q <= '0;
            output_stall_cycles_q <= '0;
            tare_dense_fallbacks_q <= '0;
            threshold_q8_q <= '0;
            preserve_mean_q <= 1'b0;
            preload_active_q <= 1'b0;
            preload_cycles_q <= '0;
        end else begin
            if (row_load_start && !window_active_q) begin
                preload_active_q <= 1'b1;
                preload_cycles_q <= '0;
            end else if (preload_active_q) begin
                preload_cycles_q <= preload_cycles_q + 1'b1;
                if (load_fire && 32'(row_load_pair_id) == 32'(PAIRS - 1))
                    preload_active_q <= 1'b0;
            end
            if (row_load_start && !ZK_BYPASS_ENABLE) begin
                if (window_active_q) begin
                    baseline_load_error_q <= 1'b1;
                end else begin
                    baseline_row_loaded_q <= 1'b0;
                    baseline_load_count_q <= '0;
                    baseline_load_error_q <= 1'b0;
                end
            end
            if (row_load_start && window_active_q)
                protocol_error_q <= 1'b1;
            if (load_fire && !ZK_BYPASS_ENABLE) begin
                if (32'(row_load_pair_id) != 32'(baseline_load_count_q)) begin
                    baseline_load_error_q <= 1'b1;
                end else begin
                    baseline_load_count_q <= baseline_load_count_q + 1'b1;
                    if (32'(row_load_pair_id) == 32'(PAIRS - 1))
                        baseline_row_loaded_q <= 1'b1;
                end
            end

            if (window_start_accept) begin
                baseline_score_next_q <= '0;
                baseline_all_issued_q <= 1'b0;
                meta_done_q <= !ZK_BYPASS_ENABLE;
                window_active_q <= 1'b1;
                seal_issued_q <= 1'b0;
                class_phase_done_q <= 1'b0;
                emit_state_q <= EMIT_IDLE;
                emit_mask_q <= '0;
                row_sum_q8_q <= '0;
                protocol_error_q <= 1'b0;
                score_pairs_q <= '0;
                equal_pairs_q <= '0;
                class_transactions_q <= '0;
                emitted_tokens_q <= '0;
                total_cycles_q <= '0;
                score_stall_cycles_q <= '0;
                output_stall_cycles_q <= '0;
                tare_dense_fallbacks_q <= '0;
                threshold_q8_q <= cfg_threshold_q8;
                preserve_mean_q <= cfg_preserve_mean;
            end else begin
                if (window_start_reject)
                    protocol_error_q <= 1'b1;
                if (window_active_q)
                    total_cycles_q <= total_cycles_q + 1'b1;
                if (window_done)
                    window_active_q <= 1'b0;
                if (meta_scan_done)
                    meta_done_q <= 1'b1;
                if (window_seal && !seal_ready)
                    protocol_error_q <= 1'b1;
                if (directory_seal)
                    seal_issued_q <= 1'b1;

                if (score_request_fire && !ZK_BYPASS_ENABLE) begin
                    if (32'(baseline_score_next_q) == 32'(PAIRS - 1)) begin
                        baseline_all_issued_q <= 1'b1;
                    end else begin
                        baseline_score_next_q <= baseline_score_next_q + 1'b1;
                    end
                end

                if (store_read_resp_valid && store_read_resp_score_tag
                    && !descriptor_issue_enable)
                    score_stall_cycles_q <= score_stall_cycles_q + 1'b1;
                if (out_valid && !out_ready)
                    output_stall_cycles_q <= output_stall_cycles_q + 1'b1;

                if (directory_in_valid && directory_in_ready) begin
                    score_pairs_q <= score_pairs_q + 1'b1;
                    if (score_equal)
                        equal_pairs_q <= equal_pairs_q + 1'b1;
                    if (unused_tare_dense_fallback)
                        tare_dense_fallbacks_q <= tare_dense_fallbacks_q + 1'b1;
                end

                if (class_valid) begin
                    row_sum_q8_q <= row_sum_q8_q + class_sum_term;
                    class_transactions_q <= class_transactions_q + 1'b1;
                    if (class_last)
                        class_phase_done_q <= 1'b1;
                end

                if (active_valid && active_ready) begin
                    if (!class_phase_done_q || active_k_mask == 0
                        || (active_k_mask & ~active_temporal_mask) != 0) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        emit_mask_q <= active_k_mask;
                        emit_pair_id_q <= active_pair_id;
                        emit_score_q7_q <= active_score_q7;
                        emit_active_last_q <= active_last;
                        emit_state_q <= EMIT_WAIT_K;
                    end
                end

                if (store_read_resp_valid && !store_read_resp_score_tag
                    && emit_state_q == EMIT_WAIT_K) begin
                    if ((store_read_resp_k_mask & emit_mask_q) != emit_mask_q) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        if (emit_mask_q[0])
                            emit_k0_q <= store_read_resp_k_pair[HEAD_DIM-1:0];
                        if (emit_mask_q[1])
                            emit_k1_q <= store_read_resp_k_pair[2*HEAD_DIM-1:HEAD_DIM];
                        emit_state_q <= EMIT_ACTIVE;
                    end
                end

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
        end
    end
endmodule

`default_nettype wire
