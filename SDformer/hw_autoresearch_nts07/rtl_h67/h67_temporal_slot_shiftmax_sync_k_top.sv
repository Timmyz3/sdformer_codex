`timescale 1ns/1ps
`default_nettype none

// Fixed-TTB32与RQTB16/32的同约束物理流顶层。
// 两种模式共享slot FIFO、weighted-SCS/Shiftmax和一拍同步双bank K存储。
module h67_temporal_slot_shiftmax_sync_k_top #(
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
    parameter bit QUOTIENT_ENABLE = 1'b1
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       window_seal,
    input  logic                       descriptor_issue_enable,
    input  logic                       cfg_preserve_mean,
    input  logic [THRESHOLD_W-1:0]     cfg_threshold_q8,
    output logic                       seal_ready,
    output logic                       window_done,

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
    output logic [THRESHOLD_W-1:0]     out_threshold_q8,

    output logic                       protocol_error,
    output logic [31:0]                perf_pairs,
    output logic [31:0]                perf_slots,
    output logic [31:0]                perf_equal_pairs,
    output logic [31:0]                perf_quotient_descriptors,
    output logic [31:0]                perf_original_tokens,
    output logic [31:0]                perf_active_entries,
    output logic [31:0]                perf_class_transactions,
    output logic [31:0]                perf_exp_transactions,
    output logic [31:0]                perf_emitted_tokens,
    output logic [31:0]                perf_k_read_transactions,
    output logic [31:0]                perf_k_read_bits,
    output logic [31:0]                perf_total_cycles,
    output logic [31:0]                perf_pair_stall_cycles,
    output logic [31:0]                perf_descriptor_stall_cycles,
    output logic [31:0]                perf_output_stall_cycles,
    output logic [FIFO_OCC_W-1:0]      perf_fifo_occupancy,
    output logic [FIFO_OCC_W-1:0]      perf_fifo_max_occupancy
);
    localparam int PAIR_COUNT_W = $clog2(PAIRS + 1);

    typedef enum logic [1:0] {
        EMIT_IDLE,
        EMIT_WAIT_K,
        EMIT_ACTIVE
    } emit_state_t;

    logic packet_valid;
    logic packet_ready;
    logic [1:0] packet_slot_count;
    logic [15:0] packet_slot0;
    logic [15:0] packet_slot1;
    logic pair_commit;
    logic encoder_error;

    logic fifo_valid;
    logic fifo_ready;
    logic [15:0] fifo_slot;
    logic fifo_error;

    logic [PAIR_COUNT_W-1:0] decoded_pairs_q;
    logic pair_open_q;
    logic slot_legal;
    logic [7:0] slot_score;
    logic [1:0] slot_temporal_mask;
    logic [1:0] slot_active_mask;
    logic slot_pair_last;
    logic slot_shape_legal;
    logic slot_fire;

    logic directory_window_ready;
    logic directory_done;
    logic directory_error;
    logic directory_in_valid;
    logic directory_in_ready;
    logic directory_seal;
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
    logic k_store_error;

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
    logic [31:0] class_transactions_q;
    logic [31:0] emitted_tokens_q;
    logic [31:0] total_cycles_q;
    logic [31:0] pair_stall_cycles_q;
    logic [31:0] descriptor_stall_cycles_q;
    logic [31:0] output_stall_cycles_q;
    logic [THRESHOLD_W-1:0] threshold_q8_q;
    logic preserve_mean_q;
    logic class_phase_done_q;
    logic window_active_q;

    h67_temporal_slot_encoder #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .QUOTIENT_ENABLE(QUOTIENT_ENABLE)
    ) u_encoder (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .pair_valid(pair_valid),
        .pair_ready(pair_ready),
        .pair_id(pair_id),
        .q_pair(q_pair),
        .k_pair(k_pair),
        .packet_valid(packet_valid),
        .packet_ready(packet_ready),
        .packet_slot_count(packet_slot_count),
        .packet_slot0(packet_slot0),
        .packet_slot1(packet_slot1),
        .pair_commit(pair_commit),
        .protocol_error(encoder_error),
        .perf_pairs(perf_pairs),
        .perf_slots(perf_slots),
        .perf_equal_pairs(perf_equal_pairs)
    );

    h67_temporal_slot_fifo #(
        .DEPTH(SLOT_FIFO_DEPTH),
        .OCC_W(FIFO_OCC_W)
    ) u_slot_fifo (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .enq_valid(packet_valid),
        .enq_ready(packet_ready),
        .enq_count(packet_slot_count),
        .enq_slot0(packet_slot0),
        .enq_slot1(packet_slot1),
        .deq_valid(fifo_valid),
        .deq_ready(fifo_ready),
        .deq_slot(fifo_slot),
        .occupancy(perf_fifo_occupancy),
        .max_occupancy(perf_fifo_max_occupancy),
        .protocol_error(fifo_error)
    );

    assign slot_score = fifo_slot[7:0];
    assign slot_temporal_mask = fifo_slot[9:8];
    assign slot_active_mask = fifo_slot[11:10];
    assign slot_pair_last = fifo_slot[12];
    assign slot_shape_legal = (!pair_open_q
                            && slot_temporal_mask == 2'b01
                            && !slot_pair_last)
                           || (pair_open_q
                            && slot_temporal_mask == 2'b10
                            && slot_pair_last)
                           || (!pair_open_q
                            && slot_temporal_mask == 2'b11
                            && slot_pair_last);
    assign slot_legal = fifo_slot[15:13] == 0
                     && slot_temporal_mask != 0
                     && (slot_active_mask & ~slot_temporal_mask) == 0
                     && 32'(slot_score) <= 32'(MAX_SCORE)
                     && 32'(decoded_pairs_q) < 32'(PAIRS)
                     && slot_shape_legal;
    assign directory_in_valid = fifo_valid && slot_legal
                              && descriptor_issue_enable;
    assign fifo_ready = fifo_valid && !slot_legal
                      ? 1'b1
                      : directory_in_ready && descriptor_issue_enable;
    assign slot_fire = fifo_valid && fifo_ready;

    assign seal_ready = perf_pairs == 32'(PAIRS)
                     && perf_fifo_occupancy == 0
                     && decoded_pairs_q == PAIR_COUNT_W'(PAIRS)
                     && !pair_open_q;
    assign directory_seal = window_seal && seal_ready;

    h67_temporal_weighted_scs_directory #(
        .MAX_SCORE(MAX_SCORE),
        .MAX_DESCRIPTORS(MAX_DESCRIPTORS),
        .SCORE_W(SCORE_W),
        .PAIR_ID_W(PAIR_ID_W),
        .COUNT_W(COUNT_W),
        .CLASS_W(CLASS_W)
    ) u_directory (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .window_seal(directory_seal),
        .window_ready(directory_window_ready),
        .window_done(directory_done),
        .in_valid(directory_in_valid),
        .in_ready(directory_in_ready),
        .in_pair_id(PAIR_ID_W'(decoded_pairs_q)),
        .in_score_q7($signed({{(SCORE_W-8){1'b0}}, slot_score})),
        .in_temporal_mask(slot_temporal_mask),
        .in_active_mask(slot_active_mask),
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
        .perf_quotient_descriptors(perf_quotient_descriptors),
        .perf_original_tokens(perf_original_tokens),
        .perf_active_entries(perf_active_entries)
    );

    assign k_read_req_valid = (active_valid && active_ready)
                            ? active_k_mask : 2'b00;

    h67_sync_dual_bank_k_store #(
        .HEAD_DIM(HEAD_DIM),
        .PAIRS(PAIRS),
        .ADDR_W(PAIR_ID_W)
    ) u_k_store (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .write_valid(pair_commit),
        .write_addr(pair_id),
        .write_k_pair(k_pair),
        .read_req_valid(k_read_req_valid),
        .read_req_addr(active_pair_id),
        .read_resp_valid(k_read_resp_valid),
        .read_resp_k0(k_read_resp_k0),
        .read_resp_k1(k_read_resp_k1),
        .perf_read_transactions(perf_k_read_transactions),
        .perf_read_bits(perf_k_read_bits),
        .protocol_error(k_store_error)
    );

    assign class_score_q7 = $signed(
        {{(SCORE_W-CLASS_W){1'b0}}, class_score}
    );
    assign class_delta_q7 = class_score_q7 - row_max_q7;

    ttx_exp2_lut_q8 #(
        .SCORE_W(SCORE_W),
        .SCORE_FRAC(7)
    ) u_class_exp (
        .delta_q7(class_delta_q7),
        .exp_q8(class_exp_q8)
    );

    assign class_sum_term = 32'(class_exp_q8) * 32'(class_multiplicity);
    assign active_ready = emit_state_q == EMIT_IDLE && class_phase_done_q;
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

    assign out_valid = emit_state_q == EMIT_ACTIVE && !window_start;
    assign out_token_id = TOKEN_W'(2 * 32'(emit_pair_id_q)
                                + 32'(emit_time_sel));
    assign out_k_bits = emit_time_sel ? emit_k1_q : emit_k0_q;
    assign out_gate_q17 = emit_gate_q17;
    assign out_threshold_q8 = threshold_q8_q;
    assign out_last = out_valid && emit_active_last_q
                   && (emit_mask_q == 2'b01 || emit_mask_q == 2'b10);
    assign window_done = directory_done && emit_state_q == EMIT_IDLE;
    assign protocol_error = encoder_error || fifo_error || directory_error
                         || k_store_error || protocol_error_q;
    assign perf_class_transactions = class_transactions_q;
    assign perf_exp_transactions = class_transactions_q + perf_active_entries;
    assign perf_emitted_tokens = emitted_tokens_q;
    assign perf_total_cycles = total_cycles_q;
    assign perf_pair_stall_cycles = pair_stall_cycles_q;
    assign perf_descriptor_stall_cycles = descriptor_stall_cycles_q;
    assign perf_output_stall_cycles = output_stall_cycles_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            decoded_pairs_q <= '0;
            pair_open_q <= 1'b0;
            emit_state_q <= EMIT_IDLE;
            emit_mask_q <= '0;
            emit_pair_id_q <= '0;
            emit_score_q7_q <= '0;
            emit_active_last_q <= 1'b0;
            emit_k0_q <= '0;
            emit_k1_q <= '0;
            protocol_error_q <= 1'b0;
            row_sum_q8_q <= '0;
            class_transactions_q <= '0;
            emitted_tokens_q <= '0;
            total_cycles_q <= '0;
            pair_stall_cycles_q <= '0;
            descriptor_stall_cycles_q <= '0;
            output_stall_cycles_q <= '0;
            threshold_q8_q <= '0;
            preserve_mean_q <= 1'b0;
            class_phase_done_q <= 1'b0;
            window_active_q <= 1'b0;
        end else if (window_start) begin
            decoded_pairs_q <= '0;
            pair_open_q <= 1'b0;
            emit_state_q <= EMIT_IDLE;
            emit_mask_q <= '0;
            protocol_error_q <= !directory_window_ready;
            row_sum_q8_q <= '0;
            class_transactions_q <= '0;
            emitted_tokens_q <= '0;
            total_cycles_q <= '0;
            pair_stall_cycles_q <= '0;
            descriptor_stall_cycles_q <= '0;
            output_stall_cycles_q <= '0;
            threshold_q8_q <= cfg_threshold_q8;
            preserve_mean_q <= cfg_preserve_mean;
            class_phase_done_q <= 1'b0;
            window_active_q <= 1'b1;
        end else begin
            if (window_active_q)
                total_cycles_q <= total_cycles_q + 1'b1;
            if (window_done)
                window_active_q <= 1'b0;
            if (pair_valid && !pair_ready)
                pair_stall_cycles_q <= pair_stall_cycles_q + 1'b1;
            if (fifo_valid && slot_legal
                && (!descriptor_issue_enable || !directory_in_ready))
                descriptor_stall_cycles_q <= descriptor_stall_cycles_q + 1'b1;
            if (out_valid && !out_ready)
                output_stall_cycles_q <= output_stall_cycles_q + 1'b1;
            if (window_seal && !seal_ready)
                protocol_error_q <= 1'b1;

            if (fifo_valid && !slot_legal) begin
                protocol_error_q <= 1'b1;
            end else if (slot_fire) begin
                if (slot_temporal_mask == 2'b01)
                    pair_open_q <= 1'b1;
                else begin
                    pair_open_q <= 1'b0;
                    decoded_pairs_q <= decoded_pairs_q + 1'b1;
                end
            end

            if (class_valid) begin
                row_sum_q8_q <= row_sum_q8_q + class_sum_term;
                class_transactions_q <= class_transactions_q + 1'b1;
                if (class_last)
                    class_phase_done_q <= 1'b1;
            end

            case (emit_state_q)
                EMIT_IDLE: begin
                    if (active_valid && active_ready) begin
                        if (!class_phase_done_q
                            || active_k_mask == 0
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
                default: begin
                    emit_state_q <= EMIT_IDLE;
                    protocol_error_q <= 1'b1;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
