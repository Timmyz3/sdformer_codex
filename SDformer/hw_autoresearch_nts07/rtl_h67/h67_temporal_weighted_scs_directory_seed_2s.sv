`timescale 1ns/1ps
`default_nettype none

// 双slot weighted-SCS目录；seal时原子合并zero-K的0/1/2类multiplicity。
module h67_temporal_weighted_scs_directory_seed_2s #(
    parameter int MAX_SCORE = 162,
    parameter int MAX_DESCRIPTORS = 450,
    parameter int SCORE_W = 16,
    parameter int PAIR_ID_W = 8,
    parameter int EXPECTED_TOKENS = MAX_DESCRIPTORS,
    parameter int COUNT_W = $clog2(2 * MAX_DESCRIPTORS + 1),
    parameter int DESC_COUNT_W = $clog2(MAX_DESCRIPTORS + 1),
    parameter int CLASS_W = $clog2(MAX_SCORE + 1),
    parameter int ACTIVE_MEMORY_IMPL = 0
) (
    input  logic                       clk_core,
    input  logic                       rst_core,
    input  logic                       window_start,
    input  logic                       window_seal,
    output logic                       window_ready,
    output logic                       window_done,

    input  logic [COUNT_W-1:0]         seed_count0,
    input  logic [COUNT_W-1:0]         seed_count1,
    input  logic [COUNT_W-1:0]         seed_count2,

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
    output logic [31:0]                perf_active_entries,
    output logic [31:0]                perf_seeded_tokens
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
    logic [DESC_COUNT_W-1:0] active_issue_q;
    logic [CLASS_W-1:0] selected_class;
    logic selected_class_valid;
    logic protocol_error_q;
    logic signed [SCORE_W-1:0] row_max_q;
    logic [31:0] quotient_descriptors_q;
    logic [31:0] original_tokens_q;
    logic [31:0] active_entries_q;
    logic [31:0] seeded_tokens_q;
    logic [1:0] multiplicity0;
    logic [1:0] multiplicity1;
    logic [2:0] batch_multiplicity;
    logic [1:0] active_add;
    logic input0_legal;
    logic input1_legal;
    logic batch_legal;
    logic same_score;
    logic signed [SCORE_W-1:0] batch_max_score;
    logic [COUNT_W:0] seed_total;
    logic seed_any;
    logic [1:0] seed_max_class;
    // active score已由合法性检查限制在0..MAX_SCORE，只驻留CLASS_W位。
    localparam int ACTIVE_SCORE_W = CLASS_W;
    localparam int ACTIVE_DESC_W = PAIR_ID_W + ACTIVE_SCORE_W + 4;
    logic [1:0] active_store_write_count;
    logic [DESC_COUNT_W-1:0] active_store_write0_addr;
    logic [DESC_COUNT_W-1:0] active_store_write1_addr;
    logic [ACTIVE_DESC_W-1:0] active_store_write0_data;
    logic [ACTIVE_DESC_W-1:0] active_store_write1_data;
    logic active_store_read_req_valid;
    logic active_store_read_req_ready;
    logic [DESC_COUNT_W-1:0] active_store_read_req_addr;
    logic active_store_read_resp_valid;
    logic [DESC_COUNT_W-1:0] active_store_read_resp_addr;
    logic [ACTIVE_DESC_W-1:0] active_store_read_resp_data;
    logic active_store_error;
    logic active_store_read_fire;
    logic active_store_read_resp_fire;

    initial begin
        if (SCORE_W < ACTIVE_SCORE_W)
            $error("SCORE_W must cover compressed active score");
    end

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
    assign seed_total = {1'b0, seed_count0}
                      + {1'b0, seed_count1}
                      + {1'b0, seed_count2};
    assign seed_any = seed_total != 0;
    assign seed_max_class = seed_count2 != 0 ? 2
                          : seed_count1 != 0 ? 1 : 0;

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
    assign active_store_write_count = in_valid && in_ready ? active_add : 0;
    assign active_store_write0_addr = active_count_q;
    assign active_store_write1_addr = active_count_q + 1'b1;
    assign active_store_write0_data = in0_active_mask != 0
        ? {in0_active_mask, in0_temporal_mask,
           in0_score_q7[ACTIVE_SCORE_W-1:0], in0_pair_id}
        : {in1_active_mask, in1_temporal_mask,
           in1_score_q7[ACTIVE_SCORE_W-1:0], in1_pair_id};
    assign active_store_write1_data =
        {in1_active_mask, in1_temporal_mask,
         in1_score_q7[ACTIVE_SCORE_W-1:0], in1_pair_id};
    assign active_store_read_req_valid = state_q == ST_ACTIVE
                                       && active_issue_q < active_count_q;
    assign active_store_read_req_addr = active_issue_q;
    assign active_store_read_fire = active_store_read_req_valid
                                  && active_store_read_req_ready;
    assign active_valid = state_q == ST_ACTIVE && active_store_read_resp_valid;
    assign active_pair_id = active_store_read_resp_data[PAIR_ID_W-1:0];
    assign active_score_q7 = $signed({
        {(SCORE_W-ACTIVE_SCORE_W){1'b0}},
        active_store_read_resp_data[PAIR_ID_W +: ACTIVE_SCORE_W]
    });
    assign active_temporal_mask = active_store_read_resp_data[
        PAIR_ID_W + ACTIVE_SCORE_W +: 2];
    assign active_k_mask = active_store_read_resp_data[
        PAIR_ID_W + ACTIVE_SCORE_W + 2 +: 2];
    assign active_last = active_valid
                      && active_store_read_resp_addr + 1'b1 == active_count_q;
    assign active_store_read_resp_fire = active_valid && active_ready;
    assign row_max_q7 = row_max_q;
    assign protocol_error = protocol_error_q || active_store_error;
    assign perf_quotient_descriptors = quotient_descriptors_q;
    assign perf_original_tokens = original_tokens_q;
    assign perf_active_entries = active_entries_q;
    assign perf_seeded_tokens = seeded_tokens_q;

    h67_banked_active_descriptor_store #(
        .DEPTH(MAX_DESCRIPTORS),
        .DATA_W(ACTIVE_DESC_W),
        .ADDR_W(DESC_COUNT_W),
        .MEMORY_IMPL(ACTIVE_MEMORY_IMPL)
    ) u_active_store (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .window_start(window_start),
        .write_count(active_store_write_count),
        .write0_addr(active_store_write0_addr),
        .write0_data(active_store_write0_data),
        .write1_addr(active_store_write1_addr),
        .write1_data(active_store_write1_data),
        .read_req_valid(active_store_read_req_valid),
        .read_req_ready(active_store_read_req_ready),
        .read_req_addr(active_store_read_req_addr),
        .read_resp_valid(active_store_read_resp_valid),
        .read_resp_ready(active_ready),
        .read_resp_addr(active_store_read_resp_addr),
        .read_resp_data(active_store_read_resp_data),
        .protocol_error(active_store_error)
    );

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
            active_issue_q <= '0;
            protocol_error_q <= 1'b0;
            row_max_q <= '0;
            quotient_descriptors_q <= '0;
            original_tokens_q <= '0;
            active_entries_q <= '0;
            seeded_tokens_q <= '0;
        end else if (window_start) begin
            if (!window_ready) begin
                protocol_error_q <= 1'b1;
            end else begin
                state_q <= ST_BUILD;
                class_present_q <= '0;
                active_count_q <= '0;
                active_issue_q <= '0;
                protocol_error_q <= 1'b0;
                row_max_q <= '0;
                quotient_descriptors_q <= '0;
                original_tokens_q <= '0;
                active_entries_q <= '0;
                seeded_tokens_q <= '0;
            end
        end else begin
            if (in_valid && in_ready) begin
                quotient_descriptors_q <= quotient_descriptors_q + 32'(in_count);
                original_tokens_q <= original_tokens_q + 32'(batch_multiplicity);
                if (!batch_legal) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    if (same_score) begin
                        class_hist[in0_score_q7[CLASS_W-1:0]]
                            <= (class_present_q[in0_score_q7[CLASS_W-1:0]]
                                ? class_hist[in0_score_q7[CLASS_W-1:0]] : '0)
                             + COUNT_W'(batch_multiplicity);
                        class_present_q[in0_score_q7[CLASS_W-1:0]] <= 1'b1;
                    end else begin
                        class_hist[in0_score_q7[CLASS_W-1:0]]
                            <= (class_present_q[in0_score_q7[CLASS_W-1:0]]
                                ? class_hist[in0_score_q7[CLASS_W-1:0]] : '0)
                             + COUNT_W'(multiplicity0);
                        class_present_q[in0_score_q7[CLASS_W-1:0]] <= 1'b1;
                        if (in_count == 2) begin
                            class_hist[in1_score_q7[CLASS_W-1:0]]
                                <= (class_present_q[in1_score_q7[CLASS_W-1:0]]
                                    ? class_hist[in1_score_q7[CLASS_W-1:0]] : '0)
                                 + COUNT_W'(multiplicity1);
                            class_present_q[in1_score_q7[CLASS_W-1:0]] <= 1'b1;
                        end
                    end

                    if (quotient_descriptors_q == 0 || batch_max_score > row_max_q)
                        row_max_q <= batch_max_score;

                    active_count_q <= active_count_q + DESC_COUNT_W'(active_add);
                    active_entries_q <= active_entries_q + 32'(active_add);
                end
            end else if (in_valid && state_q == ST_BUILD && !in_ready) begin
                protocol_error_q <= 1'b1;
            end

            if (window_seal) begin
                if (state_q != ST_BUILD || in_valid) begin
                    protocol_error_q <= 1'b1;
                end else if (32'(original_tokens_q) + 32'(seed_total)
                             != 32'(EXPECTED_TOKENS)) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    if (seed_count0 != 0) begin
                        class_hist[0] <= (class_present_q[0] ? class_hist[0] : '0)
                                       + seed_count0;
                        class_present_q[0] <= 1'b1;
                    end
                    if (seed_count1 != 0) begin
                        class_hist[1] <= (class_present_q[1] ? class_hist[1] : '0)
                                       + seed_count1;
                        class_present_q[1] <= 1'b1;
                    end
                    if (seed_count2 != 0) begin
                        class_hist[2] <= (class_present_q[2] ? class_hist[2] : '0)
                                       + seed_count2;
                        class_present_q[2] <= 1'b1;
                    end
                    original_tokens_q <= original_tokens_q + 32'(seed_total);
                    seeded_tokens_q <= 32'(seed_total);
                    if (seed_any
                        && (quotient_descriptors_q == 0
                            || $signed(SCORE_W'(seed_max_class)) > row_max_q))
                        row_max_q <= $signed(SCORE_W'(seed_max_class));
                    if (class_present_q != 0 || seed_any)
                        state_q <= ST_CLASS;
                    else if (active_count_q != 0)
                        state_q <= ST_ACTIVE;
                    else
                        state_q <= ST_DONE;
                end
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

            if (active_store_read_fire)
                active_issue_q <= active_issue_q + 1'b1;

            if (active_store_read_resp_fire) begin
                if (active_last)
                    state_q <= ST_DONE;
            end
        end
    end
endmodule

`default_nettype wire
