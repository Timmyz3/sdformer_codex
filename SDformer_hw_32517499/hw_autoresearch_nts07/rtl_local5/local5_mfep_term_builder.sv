`timescale 1ns/1ps
`default_nettype none

// Multiplicity-Folded Edge Projection (MFEP) term builder.
// For one destination with up to N_CAND edges {gate,k}:
//   m[lane,gate] = |{e | K[e,lane] && gate[e]==g}|
// Emit nonempty terms {dest,lane,gate,multiplicity}.
// Zero-gate edges contribute no products.
module local5_mfep_term_builder #(
    parameter int HEAD_DIM  = 32,
    parameter int N_CAND    = 5,
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int DIR_W     = 3,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int COUNTER_W = 16,
    parameter int TERM_COUNT_W = $clog2(HEAD_DIM * N_CAND + 1)
) (
    input  logic                  clk_core,
    input  logic                  rst_core,

    input  logic                  dest_valid,
    output logic                  dest_ready,
    input  logic [TAG_W-1:0]      dest_tag,
    input  logic [DEST_W-1:0]     dest_id,

    input  logic                  edge_valid,
    output logic                  edge_ready,
    input  logic [DIR_W-1:0]      edge_dir,
    input  logic [GATE_W-1:0]     edge_gate_q17,
    input  logic [HEAD_DIM-1:0]   edge_k_bits,
    input  logic                  edge_last,

    output logic                  term_valid,
    input  logic                  term_ready,
    output logic [TAG_W-1:0]      term_tag,
    output logic [DEST_W-1:0]     term_dest_id,
    output logic [LANE_ID_W-1:0]  term_lane,
    output logic [GATE_W-1:0]     term_gate_q17,
    output logic [MULT_W-1:0]     term_multiplicity,
    output logic                  term_last,

    output logic                  dest_done_valid,
    input  logic                  dest_done_ready,
    output logic [TAG_W-1:0]      dest_done_tag,
    output logic                  protocol_error,

    output logic [COUNTER_W-1:0]  count_edges,
    output logic [COUNTER_W-1:0]  count_terms,
    output logic [COUNTER_W-1:0]  count_naive_products
);

    typedef enum logic [2:0] {
        ST_IDLE    = 3'd0,
        ST_COLLECT = 3'd1,
        ST_BUILD   = 3'd2,
        ST_SCAN    = 3'd3,
        ST_DONE    = 3'd4
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [2:0] n_edges_q;
    logic [N_CAND-1:0] seen_dir_q;
    logic [GATE_W-1:0] e_gate_q [0:N_CAND-1];
    logic [HEAD_DIM-1:0] e_k_q [0:N_CAND-1];
    logic [GATE_W-1:0] uniq_gate_q [0:N_CAND-1];
    logic [2:0] n_uniq_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [2:0] ug_idx_q;
    logic [TERM_COUNT_W-1:0] terms_remaining_q;
    logic protocol_error_q;
    logic [COUNTER_W-1:0] count_edges_q;
    logic [COUNTER_W-1:0] count_terms_q;
    logic [COUNTER_W-1:0] count_naive_q;

    logic [MULT_W-1:0] mult_w;
    logic [GATE_W-1:0] cur_gate_w;
    logic term_fire_ok;

    assign cur_gate_w = uniq_gate_q[ug_idx_q];

    always_comb begin
        mult_w = '0;
        for (int e = 0; e < N_CAND; e = e + 1) begin
            if ((3'(e) < n_edges_q) &&
                e_k_q[e][lane_q] &&
                (e_gate_q[e] == cur_gate_w) &&
                (cur_gate_w != '0)) begin
                mult_w = mult_w + MULT_W'(1);
            end
        end
    end

    assign term_fire_ok = (state_q == ST_SCAN) && (n_uniq_q != 0) && (mult_w != 0);
    assign term_valid = term_fire_ok;
    assign term_tag = tag_q;
    assign term_dest_id = dest_q;
    assign term_lane = lane_q;
    assign term_gate_q17 = cur_gate_w;
    assign term_multiplicity = mult_w;
    // Last refers to the last emitted nonzero term, not the last slot in the
    // lane x unique-gate scan. Sparse tails commonly have no lane-31 term.
    assign term_last = term_fire_ok
                     && (terms_remaining_q == TERM_COUNT_W'(1));

    assign dest_ready = (state_q == ST_IDLE);
    assign edge_ready = (state_q == ST_COLLECT);
    assign dest_done_valid = (state_q == ST_DONE);
    assign dest_done_tag = tag_q;
    assign protocol_error = protocol_error_q;
    assign count_edges = count_edges_q;
    assign count_terms = count_terms_q;
    assign count_naive_products = count_naive_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            dest_q <= '0;
            n_edges_q <= '0;
            seen_dir_q <= '0;
            n_uniq_q <= '0;
            lane_q <= '0;
            ug_idx_q <= '0;
            terms_remaining_q <= '0;
            protocol_error_q <= 1'b0;
            count_edges_q <= '0;
            count_terms_q <= '0;
            count_naive_q <= '0;
            for (int e = 0; e < N_CAND; e = e + 1) begin
                e_gate_q[e] <= '0;
                e_k_q[e] <= '0;
                uniq_gate_q[e] <= '0;
            end
        end else begin
            case (state_q)
                ST_IDLE: begin
                    protocol_error_q <= 1'b0;
                    n_edges_q <= '0;
                    seen_dir_q <= '0;
                    n_uniq_q <= '0;
                    terms_remaining_q <= '0;
                    if (dest_valid) begin
                        tag_q <= dest_tag;
                        dest_q <= dest_id;
                        state_q <= ST_COLLECT;
                    end
                end

                ST_COLLECT: begin
                    if (edge_valid) begin
                        if (n_edges_q >= 3'(N_CAND) ||
                            32'(edge_dir) >= N_CAND ||
                            seen_dir_q[edge_dir]) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            e_gate_q[n_edges_q] <= edge_gate_q17;
                            e_k_q[n_edges_q] <= edge_k_bits;
                            n_edges_q <= n_edges_q + 3'd1;
                            seen_dir_q[edge_dir] <= 1'b1;
                            count_edges_q <= count_edges_q + 1'b1;
                            begin
                                int add;
                                add = 0;
                                for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                                    if (edge_k_bits[lane] && (edge_gate_q17 != '0)) begin
                                        add = add + 1;
                                    end
                                end
                                count_naive_q <= count_naive_q + COUNTER_W'(add);
                            end
                            if (edge_last) begin
                                state_q <= ST_BUILD;
                            end
                        end
                    end
                end

                ST_BUILD: begin
                    // edges fully latched; freeze unique nonzero gates
                    // Procedural temps (Yosys-friendly: no 'automatic' keyword)
                    begin : build_unique
                        int u;
                        int e;
                        int j;
                        logic found;
                        logic [GATE_W-1:0] cand;
                        logic [GATE_W-1:0] tlist [0:N_CAND-1];
                        int term_total;
                        logic slot_nonzero;
                        u = 0;
                        for (j = 0; j < N_CAND; j = j + 1)
                            tlist[j] = '0;
                        for (e = 0; e < N_CAND; e = e + 1) begin
                            if (e < n_edges_q && e_gate_q[e] != '0) begin
                                cand = e_gate_q[e];
                                found = 1'b0;
                                for (j = 0; j < N_CAND; j = j + 1) begin
                                    if (j < u && tlist[j] == cand)
                                        found = 1'b1;
                                end
                                if (!found && u < N_CAND) begin
                                    tlist[u] = cand;
                                    u = u + 1;
                                end
                            end
                        end
                        term_total = 0;
                        for (int lane = 0; lane < HEAD_DIM; lane = lane + 1) begin
                            for (j = 0; j < N_CAND; j = j + 1) begin
                                slot_nonzero = 1'b0;
                                for (e = 0; e < N_CAND; e = e + 1) begin
                                    if (e < n_edges_q
                                        && e_k_q[e][lane]
                                        && e_gate_q[e] == tlist[j]
                                        && tlist[j] != '0) begin
                                        slot_nonzero = 1'b1;
                                    end
                                end
                                if (j < u && slot_nonzero)
                                    term_total = term_total + 1;
                            end
                        end
                        for (j = 0; j < N_CAND; j = j + 1)
                            uniq_gate_q[j] <= tlist[j];
                        n_uniq_q <= 3'(u);
                        terms_remaining_q <= TERM_COUNT_W'(term_total);
                        lane_q <= '0;
                        ug_idx_q <= '0;
                        if (u == 0) state_q <= ST_DONE;
                        else state_q <= ST_SCAN;
                    end
                end

                ST_SCAN: begin
                    if (!term_fire_ok || term_ready) begin
                        if (term_fire_ok && term_ready) begin
                            count_terms_q <= count_terms_q + 1'b1;
                            terms_remaining_q <=
                                terms_remaining_q - TERM_COUNT_W'(1);
                        end
                        if ((ug_idx_q + 3'd1) < n_uniq_q) begin
                            ug_idx_q <= ug_idx_q + 3'd1;
                        end else if (lane_q != LANE_ID_W'(HEAD_DIM - 1)) begin
                            lane_q <= lane_q + 1'b1;
                            ug_idx_q <= '0;
                        end else begin
                            state_q <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    if (dest_done_ready) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
