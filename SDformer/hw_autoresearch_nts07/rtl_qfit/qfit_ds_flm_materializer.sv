`timescale 1ns/1ps
`default_nettype none

// Expand one factorized Local5 descriptor into exact projection terms.
// Mode 0 emits lane-major order; mode 1 emits gate-major order. The mode is
// captured with the descriptor and cannot affect an in-flight descriptor.
module qfit_ds_flm_materializer #(
    parameter int HEAD_DIM = 32,
    parameter int GATE_W = 9,
    parameter int SOURCE_ID_W = 9,
    parameter int Y_W = 4,
    parameter int X_W = 4,
    localparam int LANE_W =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    localparam int ACTIVE_COUNT_W =
        $clog2(HEAD_DIM + 1),
    localparam int TERM_COUNT_W =
        $clog2(HEAD_DIM * 5 + 1)
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       descriptor_valid,
    output logic                       descriptor_ready,
    input  logic                       descriptor_mode,
    input  logic [SOURCE_ID_W-1:0]     descriptor_source_id,
    input  logic [Y_W-1:0]             descriptor_y,
    input  logic [X_W-1:0]             descriptor_x,
    input  logic [HEAD_DIM-1:0]        descriptor_k,
    input  logic [5*GATE_W-1:0]        descriptor_incoming_gates,
    input  logic [4:0]                 descriptor_valid_mask,

    output logic                       term_valid,
    input  logic                       term_ready,
    output logic [SOURCE_ID_W-1:0]     term_source_id,
    output logic [Y_W-1:0]             term_source_y,
    output logic [X_W-1:0]             term_source_x,
    output logic [LANE_W-1:0]          term_lane,
    output logic [GATE_W-1:0]          term_gate,
    output logic [4:0]                 term_destination_mask,
    output logic                       term_last,

    output logic [31:0]                perf_descriptors,
    output logic [31:0]                perf_terms,
    output logic [31:0]                perf_destination_updates
);
    typedef enum logic {
        ST_IDLE = 1'b0,
        ST_SCAN = 1'b1
    } state_t;

    state_t state_q;
    logic mode_q;
    logic [SOURCE_ID_W-1:0] source_id_q;
    logic [Y_W-1:0] source_y_q;
    logic [X_W-1:0] source_x_q;
    logic [HEAD_DIM-1:0] active_k_q;
    logic [HEAD_DIM-1:0] remaining_k_q;
    logic [GATE_W-1:0] unique_gate_q [0:4];
    logic [4:0] unique_mask_q [0:4];
    logic [2:0] unique_count_q;
    logic [2:0] gate_index_q;
    logic [TERM_COUNT_W-1:0] terms_remaining_q;
    logic [LANE_W-1:0] selected_lane;
    logic selected_lane_valid;
    logic [HEAD_DIM-1:0] selected_lane_onehot;
    logic [HEAD_DIM-1:0] remaining_after_selected;
    logic [31:0] perf_descriptors_q;
    logic [31:0] perf_terms_q;
    logic [31:0] perf_updates_q;

    function automatic logic [2:0] popcount5(input logic [4:0] value);
        logic [2:0] count;
        count = '0;
        for (integer i = 0; i < 5; i = i + 1)
            count = count + 3'(value[i]);
        popcount5 = count;
    endfunction

    function automatic logic [ACTIVE_COUNT_W-1:0] popcount_k(
        input logic [HEAD_DIM-1:0] value
    );
        logic [ACTIVE_COUNT_W-1:0] count;
        count = '0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1)
            count = count + ACTIVE_COUNT_W'(value[lane]);
        popcount_k = count;
    endfunction

    function automatic logic [TERM_COUNT_W-1:0] term_count(
        input logic [ACTIVE_COUNT_W-1:0] active_lanes,
        input logic [2:0] unique_count
    );
        logic [TERM_COUNT_W-1:0] lanes;
        lanes = TERM_COUNT_W'(active_lanes);
        unique case (unique_count)
            3'd0: term_count = '0;
            3'd1: term_count = lanes;
            3'd2: term_count = lanes << 1;
            3'd3: term_count = (lanes << 1) + lanes;
            3'd4: term_count = lanes << 2;
            3'd5: term_count = (lanes << 2) + lanes;
            default: term_count = '0;
        endcase
    endfunction

    assign descriptor_ready = state_q == ST_IDLE;
    assign term_valid = state_q == ST_SCAN
                     && unique_count_q != 0
                     && selected_lane_valid;
    assign term_source_id = source_id_q;
    assign term_source_y = source_y_q;
    assign term_source_x = source_x_q;
    assign term_lane = selected_lane;
    assign term_gate = unique_gate_q[gate_index_q];
    assign term_destination_mask = unique_mask_q[gate_index_q];
    assign term_last = term_valid
                    && terms_remaining_q == TERM_COUNT_W'(1);
    assign perf_descriptors = perf_descriptors_q;
    assign perf_terms = perf_terms_q;
    assign perf_destination_updates = perf_updates_q;
    assign remaining_after_selected =
        remaining_k_q & ~selected_lane_onehot;

    always_comb begin
        selected_lane = '0;
        selected_lane_valid = 1'b0;
        selected_lane_onehot = '0;
        for (integer lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
            if (!selected_lane_valid && remaining_k_q[lane]) begin
                selected_lane = LANE_W'(lane);
                selected_lane_valid = 1'b1;
                selected_lane_onehot[lane] = 1'b1;
            end
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            mode_q <= 1'b0;
            source_id_q <= '0;
            source_y_q <= '0;
            source_x_q <= '0;
            active_k_q <= '0;
            remaining_k_q <= '0;
            unique_count_q <= '0;
            gate_index_q <= '0;
            terms_remaining_q <= '0;
            perf_descriptors_q <= '0;
            perf_terms_q <= '0;
            perf_updates_q <= '0;
            for (int i = 0; i < 5; i = i + 1) begin
                unique_gate_q[i] <= '0;
                unique_mask_q[i] <= '0;
            end
        end else begin
            if (descriptor_valid && descriptor_ready) begin : capture
                logic [GATE_W-1:0] gates [0:4];
                logic [GATE_W-1:0] unique_gate [0:4];
                logic [4:0] unique_mask [0:4];
                logic found;
                logic [2:0] found_index;
                int unique_count;
                logic [ACTIVE_COUNT_W-1:0] active_lanes;

                for (int i = 0; i < 5; i = i + 1) begin
                    gates[i] =
                        descriptor_incoming_gates[i*GATE_W +: GATE_W];
                    unique_gate[i] = '0;
                    unique_mask[i] = '0;
                end

                unique_count = 0;
                for (int role = 0; role < 5; role = role + 1) begin
                    if (
                        descriptor_valid_mask[role]
                        && gates[role] != '0
                    ) begin
                        found = 1'b0;
                        found_index = '0;
                        for (int u = 0; u < 5; u = u + 1) begin
                            if (
                                u < unique_count
                                && unique_gate[u] == gates[role]
                            ) begin
                                found = 1'b1;
                                found_index = 3'(u);
                            end
                        end
                        if (found) begin
                            unique_mask[found_index][role] = 1'b1;
                        end else begin
                            unique_gate[unique_count] = gates[role];
                            unique_mask[unique_count][role] = 1'b1;
                            unique_count = unique_count + 1;
                        end
                    end
                end

                active_lanes = popcount_k(descriptor_k);

                mode_q <= descriptor_mode;
                source_id_q <= descriptor_source_id;
                source_y_q <= descriptor_y;
                source_x_q <= descriptor_x;
                active_k_q <= descriptor_k;
                remaining_k_q <= descriptor_k;
                unique_count_q <= 3'(unique_count);
                gate_index_q <= '0;
                terms_remaining_q <= term_count(
                    active_lanes,
                    3'(unique_count)
                );
                perf_descriptors_q <= perf_descriptors_q + 1'b1;
                for (int i = 0; i < 5; i = i + 1) begin
                    unique_gate_q[i] <= unique_gate[i];
                    unique_mask_q[i] <= unique_mask[i];
                end
                if (active_lanes != 0 && unique_count != 0)
                    state_q <= ST_SCAN;
            end else if (
                state_q == ST_SCAN
                && term_valid
                && term_ready
            ) begin
                perf_terms_q <= perf_terms_q + 1'b1;
                perf_updates_q <=
                    perf_updates_q
                    + 32'(popcount5(term_destination_mask));
                terms_remaining_q <=
                    terms_remaining_q - TERM_COUNT_W'(1);

                if (terms_remaining_q == TERM_COUNT_W'(1)) begin
                    remaining_k_q <= remaining_after_selected;
                    state_q <= ST_IDLE;
                end else if (!mode_q) begin
                    // Lane-major: consume all gates before advancing lane.
                    if (gate_index_q + 3'd1 < unique_count_q) begin
                        gate_index_q <= gate_index_q + 3'd1;
                    end else begin
                        remaining_k_q <= remaining_after_selected;
                        gate_index_q <= '0;
                    end
                end else begin
                    // Gate-major: consume all lanes before advancing gate.
                    if (remaining_after_selected == '0) begin
                        gate_index_q <= gate_index_q + 3'd1;
                        remaining_k_q <= active_k_q;
                    end else begin
                        remaining_k_q <= remaining_after_selected;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
