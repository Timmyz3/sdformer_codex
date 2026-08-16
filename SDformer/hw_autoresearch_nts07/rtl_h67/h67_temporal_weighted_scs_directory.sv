`timescale 1ns/1ps
`default_nettype none

// Temporal quotient后的SCS前端。
// temporal_mask的popcount作为Shiftmax分母multiplicity；active_mask仅控制K目录驻留。
module h67_temporal_weighted_scs_directory #(
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 162,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 9,
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
    input  logic [PAIR_ID_W-1:0]       in_pair_id,
    input  logic signed [SCORE_W-1:0]  in_score_q7,
    input  logic [1:0]                 in_temporal_mask,
    input  logic [1:0]                 in_active_mask,

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
    logic [1:0] input_multiplicity;

    assign input_multiplicity = {1'b0, in_temporal_mask[0]}
                              + {1'b0, in_temporal_mask[1]};
    assign window_ready = state_q == ST_IDLE || state_q == ST_DONE;
    assign window_done = state_q == ST_DONE;
    assign in_ready = state_q == ST_BUILD;
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
        end else begin
            if (window_start) begin
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
                    quotient_descriptors_q <= quotient_descriptors_q + 1'b1;
                    original_tokens_q <= original_tokens_q
                        + 32'(input_multiplicity);
                    if (
                        in_temporal_mask == 0
                        || (in_active_mask & ~in_temporal_mask) != 0
                        || in_score_q7 < 0
                        || in_score_q7 > $signed(SCORE_W'(MAX_SCORE))
                    ) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        if (class_present_q[in_score_q7[CLASS_W-1:0]])
                            class_hist[in_score_q7[CLASS_W-1:0]]
                                <= class_hist[in_score_q7[CLASS_W-1:0]]
                                 + COUNT_W'(input_multiplicity);
                        else
                            class_hist[in_score_q7[CLASS_W-1:0]]
                                <= COUNT_W'(input_multiplicity);
                        class_present_q[in_score_q7[CLASS_W-1:0]] <= 1'b1;
                        if (
                            quotient_descriptors_q == 0
                            || in_score_q7 > row_max_q
                        )
                            row_max_q <= in_score_q7;
                        if (in_active_mask != 0) begin
                            if (active_count_q == DESC_COUNT_W'(MAX_DESCRIPTORS)) begin
                                protocol_error_q <= 1'b1;
                            end else begin
                                active_pair_store[active_count_q] <= in_pair_id;
                                active_score_store[active_count_q] <= in_score_q7;
                                active_temporal_store[active_count_q]
                                    <= in_temporal_mask;
                                active_mask_store[active_count_q] <= in_active_mask;
                                active_count_q <= active_count_q + 1'b1;
                                active_entries_q <= active_entries_q + 1'b1;
                            end
                        end
                    end
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
    end
endmodule

`default_nettype wire
