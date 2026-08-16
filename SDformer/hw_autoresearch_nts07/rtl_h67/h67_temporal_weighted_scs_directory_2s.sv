`timescale 1ns/1ps
`default_nettype none

// 每拍接收一或两条slot。两条slot命中同一score class时执行冲突合并更新。
module h67_temporal_weighted_scs_directory_2s #(
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 450,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 8,
    parameter int COUNT_W = $clog2(2 * MAX_DESCRIPTORS + 1),
    parameter int DESC_COUNT_W = $clog2(MAX_DESCRIPTORS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       window_seal,
    output logic                       window_ready,
    output logic                       window_done,

    input  logic                       in_valid,
    output logic                       in_ready,
    input  logic [1:0]                 in_count,
    input  logic [PAIR_ID_W-1:0]       in0_pair_id,
    input  logic signed [SCORE_W-1:0]  in0_score_q7,
    input  logic [1:0]                 in0_temporal_mask,
    input  logic [1:0]                 in0_active_mask,
    input  logic [PAIR_ID_W-1:0]       in1_pair_id,
    input  logic signed [SCORE_W-1:0]  in1_score_q7,
    input  logic [1:0]                 in1_temporal_mask,
    input  logic [1:0]                 in1_active_mask,

    output logic                       class_valid,
    input  logic                       class_ready,
    output logic [CLASS_W-1:0]         class_score,
    output logic [COUNT_W-1:0]         class_multiplicity,
    output logic                       class_last,

    output logic                       active_valid,
    input  logic                       active_ready,
    output logic [PAIR_ID_W-1:0]       active_pair_id,
    output logic signed [SCORE_W-1:0]  active_score_q7,
    output logic [1:0]                 active_temporal_mask,
    output logic [1:0]                 active_k_mask,
    output logic                       active_last,

    output logic signed [SCORE_W-1:0]  row_max_q7,
    output logic                       protocol_error,
    output logic [31:0]                perf_quotient_descriptors,
    output logic [31:0]                perf_original_tokens,
    output logic [31:0]                perf_active_entries
);
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_BUILD,
        ST_CLASS,
        ST_ACTIVE,
        ST_DONE
    } state_t;

    state_t state_q;
    logic [COUNT_W-1:0] class_hist [0:MAX_SCORE];
    logic [MAX_SCORE:0] class_present_q;
    logic [DESC_COUNT_W-1:0] active_count_q;
    logic [DESC_COUNT_W-1:0] active_read_q;
    logic [PAIR_ID_W-1:0] active_pair_store [0:MAX_DESCRIPTORS-1];
    logic signed [SCORE_W-1:0] active_score_store [0:MAX_DESCRIPTORS-1];
    logic [1:0] active_temporal_store [0:MAX_DESCRIPTORS-1];
    logic [1:0] active_mask_store [0:MAX_DESCRIPTORS-1];
    logic [CLASS_W-1:0] selected_class;
    logic selected_class_valid;
    logic protocol_error_q;
    logic signed [SCORE_W-1:0] row_max_q;
    logic [31:0] quotient_descriptors_q;
    logic [31:0] original_tokens_q;
    logic [31:0] active_entries_q;
    logic [1:0] multiplicity0;
    logic [1:0] multiplicity1;
    logic [2:0] batch_multiplicity;
    logic [1:0] active_add;
    logic input0_legal;
    logic input1_legal;
    logic batch_legal;
    logic same_score;
    logic signed [SCORE_W-1:0] batch_max_score;

    assign multiplicity0 = {1'b0, in0_temporal_mask[0]}
                         + {1'b0, in0_temporal_mask[1]};
    assign multiplicity1 = {1'b0, in1_temporal_mask[0]}
                         + {1'b0, in1_temporal_mask[1]};
    assign batch_multiplicity = 3'(multiplicity0)
                              + (in_count == 2 ? 3'(multiplicity1) : 0);
    assign active_add = {1'b0, in0_active_mask != 0}
                      + {1'b0, in_count == 2 && in1_active_mask != 0};
    assign input0_legal = in0_temporal_mask != 0
                       && (in0_active_mask & ~in0_temporal_mask) == 0
                       && in0_score_q7 >= 0
                       && in0_score_q7 <= $signed(SCORE_W'(MAX_SCORE));
    assign input1_legal = in1_temporal_mask != 0
                       && (in1_active_mask & ~in1_temporal_mask) == 0
                       && in1_score_q7 >= 0
                       && in1_score_q7 <= $signed(SCORE_W'(MAX_SCORE));
    assign batch_legal = (in_count == 1 && input0_legal)
                      || (in_count == 2 && input0_legal && input1_legal);
    assign same_score = in_count == 2 && in0_score_q7 == in1_score_q7;
    assign batch_max_score = in_count == 2 && in1_score_q7 > in0_score_q7
                           ? in1_score_q7 : in0_score_q7;

    assign window_ready = state_q == ST_IDLE || state_q == ST_DONE;
    assign window_done = state_q == ST_DONE;
    assign in_ready = state_q == ST_BUILD
                   && 32'(active_count_q) + 32'(active_add)
                      <= 32'(MAX_DESCRIPTORS);
    assign class_valid = state_q == ST_CLASS && selected_class_valid;
    assign class_score = selected_class;
    assign class_multiplicity = class_hist[selected_class];
    assign class_last = class_valid
                     && (class_present_q & (class_present_q - 1'b1)) == 0;
    assign active_valid = state_q == ST_ACTIVE
                       && active_read_q < active_count_q;
    assign active_pair_id = active_pair_store[active_read_q];
    assign active_score_q7 = active_score_store[active_read_q];
    assign active_temporal_mask = active_temporal_store[active_read_q];
    assign active_k_mask = active_mask_store[active_read_q];
    assign active_last = active_valid
                      && active_read_q + 1'b1 == active_count_q;
    assign row_max_q7 = row_max_q;
    assign protocol_error = protocol_error_q;
    assign perf_quotient_descriptors = quotient_descriptors_q;
    assign perf_original_tokens = original_tokens_q;
    assign perf_active_entries = active_entries_q;

    always_comb begin
        selected_class = '0;
        selected_class_valid = 1'b0;
        for (integer score = 0; score <= MAX_SCORE; score = score + 1)
            if (!selected_class_valid && class_present_q[score]) begin
                selected_class = CLASS_W'(score);
                selected_class_valid = 1'b1;
            end
    end

    always_ff @(posedge clk_core) begin : directory_state
        if (rst_core) begin
            state_q <= ST_IDLE;
            class_present_q <= '0;
            active_count_q <= '0;
            active_read_q <= '0;
            protocol_error_q <= 1'b0;
            row_max_q <= '0;
            quotient_descriptors_q <= '0;
            original_tokens_q <= '0;
            active_entries_q <= '0;
        end else if (window_start) begin
            if (!window_ready) begin
                protocol_error_q <= 1'b1;
            end else begin
                state_q <= ST_BUILD;
                class_present_q <= '0;
                active_count_q <= '0;
                active_read_q <= '0;
                protocol_error_q <= 1'b0;
                row_max_q <= '0;
                quotient_descriptors_q <= '0;
                original_tokens_q <= '0;
                active_entries_q <= '0;
            end
        end else begin
            if (in_valid && in_ready) begin
                quotient_descriptors_q <= quotient_descriptors_q + 32'(in_count);
                original_tokens_q <= original_tokens_q + 32'(batch_multiplicity);
                if (!batch_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    if (same_score) begin
                        if (class_present_q[in0_score_q7[CLASS_W-1:0]])
                            class_hist[in0_score_q7[CLASS_W-1:0]]
                                <= class_hist[in0_score_q7[CLASS_W-1:0]]
                                 + COUNT_W'(batch_multiplicity);
                        else
                            class_hist[in0_score_q7[CLASS_W-1:0]]
                                <= COUNT_W'(batch_multiplicity);
                        class_present_q[in0_score_q7[CLASS_W-1:0]] <= 1'b1;
                    end else begin
                        if (class_present_q[in0_score_q7[CLASS_W-1:0]])
                            class_hist[in0_score_q7[CLASS_W-1:0]]
                                <= class_hist[in0_score_q7[CLASS_W-1:0]]
                                 + COUNT_W'(multiplicity0);
                        else
                            class_hist[in0_score_q7[CLASS_W-1:0]]
                                <= COUNT_W'(multiplicity0);
                        class_present_q[in0_score_q7[CLASS_W-1:0]] <= 1'b1;
                        if (in_count == 2) begin
                            if (class_present_q[in1_score_q7[CLASS_W-1:0]])
                                class_hist[in1_score_q7[CLASS_W-1:0]]
                                    <= class_hist[in1_score_q7[CLASS_W-1:0]]
                                     + COUNT_W'(multiplicity1);
                            else
                                class_hist[in1_score_q7[CLASS_W-1:0]]
                                    <= COUNT_W'(multiplicity1);
                            class_present_q[in1_score_q7[CLASS_W-1:0]] <= 1'b1;
                        end
                    end

                    if (quotient_descriptors_q == 0 || batch_max_score > row_max_q)
                        row_max_q <= batch_max_score;

                    if (in0_active_mask != 0) begin
                        active_pair_store[active_count_q] <= in0_pair_id;
                        active_score_store[active_count_q] <= in0_score_q7;
                        active_temporal_store[active_count_q] <= in0_temporal_mask;
                        active_mask_store[active_count_q] <= in0_active_mask;
                    end
                    if (in_count == 2 && in1_active_mask != 0) begin
                        active_pair_store[active_count_q
                            + DESC_COUNT_W'(in0_active_mask != 0)] <= in1_pair_id;
                        active_score_store[active_count_q
                            + DESC_COUNT_W'(in0_active_mask != 0)] <= in1_score_q7;
                        active_temporal_store[active_count_q
                            + DESC_COUNT_W'(in0_active_mask != 0)] <= in1_temporal_mask;
                        active_mask_store[active_count_q
                            + DESC_COUNT_W'(in0_active_mask != 0)] <= in1_active_mask;
                    end
                    active_count_q <= active_count_q + DESC_COUNT_W'(active_add);
                    active_entries_q <= active_entries_q + 32'(active_add);
                end
            end else if (in_valid && state_q == ST_BUILD && !in_ready) begin
                protocol_error_q <= 1'b1;
            end

            if (window_seal) begin
                if (state_q != ST_BUILD || in_valid)
                    protocol_error_q <= 1'b1;
                else if (class_present_q != 0)
                    state_q <= ST_CLASS;
                else if (active_count_q != 0)
                    state_q <= ST_ACTIVE;
                else
                    state_q <= ST_DONE;
            end

            if (class_valid && class_ready) begin
                class_present_q[selected_class] <= 1'b0;
                if (class_last) begin
                    if (active_count_q != 0)
                        state_q <= ST_ACTIVE;
                    else
                        state_q <= ST_DONE;
                end
            end

            if (active_valid && active_ready) begin
                active_read_q <= active_read_q + 1'b1;
                if (active_last)
                    state_q <= ST_DONE;
            end
        end
    end
endmodule

`default_nettype wire
