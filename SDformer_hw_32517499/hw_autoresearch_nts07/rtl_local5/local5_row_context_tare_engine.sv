`timescale 1ns/1ps
`default_nettype none

// Local5 row-context that scores neighbors via TARE-4 (topology mode).
// Instantiates Codex `local5_tare4_composite_top` READ-ONLY — does not modify
// rtl_delta sources.
//
// Protocol (same as local5_row_context_engine):
//   ANCHOR_LOAD(q_self, k_self, valid_mask)
//   PROBE(dir, k_neighbor)* for non-self valid dirs
//   RETIRE: Shiftmax5 + gated edge stream
//
// Self score: direct axnor (or TARE with k_target=k_self).
// Neighbor scores: one TARE transaction per probe with fixed Q=self, bias=0.
// Anchor K/Q latched once — fixes the "recompute anchor every edge" P0.
module local5_row_context_tare_engine #(
    parameter int HEAD_DIM = 32,
    parameter int N_CAND   = 5,
    parameter int SCORE_W  = 16,
    parameter int GATE_W   = 9,
    parameter int TAG_W    = 16,
    parameter int DEST_W   = 8,
    parameter int DIR_W    = 3
) (
    input  logic                      clk_core,
    input  logic                      rst_core,

    input  logic                      anchor_valid,
    output logic                      anchor_ready,
    input  logic [TAG_W-1:0]          anchor_tag,
    input  logic [DEST_W-1:0]         anchor_dest_id,
    input  logic [HEAD_DIM-1:0]       anchor_q_bits,
    input  logic [HEAD_DIM-1:0]       anchor_k_bits,
    input  logic [N_CAND-1:0]         anchor_valid_mask,

    input  logic                      probe_valid,
    output logic                      probe_ready,
    input  logic [DIR_W-1:0]          probe_dir,
    input  logic [HEAD_DIM-1:0]       probe_k_bits,
    input  logic                      probe_last,

    output logic                      edge_valid,
    input  logic                      edge_ready,
    output logic [TAG_W-1:0]          edge_tag,
    output logic [DEST_W-1:0]         edge_dest_id,
    output logic [DIR_W-1:0]          edge_dir,
    output logic [HEAD_DIM-1:0]       edge_k_bits,
    output logic [GATE_W-1:0]         edge_gate_q17,
    output logic signed [SCORE_W-1:0] edge_score_q7,
    output logic                      edge_last,

    output logic                      row_done_valid,
    input  logic                      row_done_ready,
    output logic [TAG_W-1:0]          row_done_tag,
    output logic [2:0]                row_done_degree,
    output logic                      protocol_error,

    output logic [15:0]               perf_probe_count,
    output logic [15:0]               perf_tare_issue_count,
    output logic [15:0]               perf_edge_emit_count,
    output logic [15:0]               perf_tare_zero_count,
    output logic [15:0]               perf_tare_sparse_count,
    output logic [15:0]               perf_tare_dense_count
);

    typedef enum logic [2:0] {
        ST_IDLE      = 3'd0,
        ST_PROBE     = 3'd1,
        ST_SCORE_TARE= 3'd2,
        ST_SHIFTMAX  = 3'd3,
        ST_EMIT      = 3'd4,
        ST_DONE      = 3'd5
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [HEAD_DIM-1:0] q_q, k_self_q;
    logic [N_CAND-1:0] valid_mask_q;
    logic [HEAD_DIM-1:0] k_cand_q [0:N_CAND-1];
    logic signed [SCORE_W-1:0] score_q [0:N_CAND-1];
    logic [GATE_W-1:0] gate_q [0:N_CAND-1];
    logic [DIR_W-1:0] emit_idx_q;
    logic [DIR_W-1:0] tare_dir_q;
    logic [2:0] degree_q, expected_probe_q, probes_seen_q;
    logic [N_CAND-1:0] probe_seen_mask_q;
    logic protocol_error_q;
    logic [15:0] perf_probe_q, perf_tare_q, perf_edge_q;
    logic [15:0] perf_zero_q, perf_sparse_q, perf_dense_q;
    logic tare_issue_pending_q;

    // TARE-4 Local5 wrapper (Codex module, instantiate only)
    // Self also goes through TARE (k_target=k_self → ZERO path) so quant
    // matches neighbor residual path (not local5_axnor leaf).
    logic tare_in_valid, tare_in_ready;
    logic [15:0] tare_in_tag;
    logic [31:0] tare_q, tare_k_self, tare_k_nb;
    logic tare_out_valid, tare_out_ready;
    logic [15:0] tare_out_tag;
    logic tare_out_mode;
    logic [1:0] tare_out_kind;
    logic [5:0] tare_out_count;
    logic [12:0] tare_out_raw;
    logic [8:0] tare_out_score;

    local5_tare4_composite_top u_tare (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .in_valid(tare_in_valid),
        .in_ready(tare_in_ready),
        .in_tag(tare_in_tag),
        .in_q_self(tare_q),
        .in_k_self(tare_k_self),
        .in_k_neighbor(tare_k_nb),
        .out_valid(tare_out_valid),
        .out_ready(tare_out_ready),
        .out_tag(tare_out_tag),
        .out_mode_local5(tare_out_mode),
        .out_kind(tare_out_kind),
        .out_update_count(tare_out_count),
        .out_raw16(tare_out_raw),
        .out_score_q7(tare_out_score)
    );

    // Shiftmax over latched scores
    logic [N_CAND*SCORE_W-1:0] sm_scores;
    logic [N_CAND-1:0]         sm_valid;
    logic [N_CAND*GATE_W-1:0]  sm_gates;
    integer vi;
    always_comb begin
        for (vi = 0; vi < N_CAND; vi = vi + 1) begin
            sm_scores[vi*SCORE_W +: SCORE_W] = score_q[vi];
            sm_valid[vi]  = valid_mask_q[vi];
        end
    end
    local5_shiftmax5_q17 #(.N_CAND(N_CAND), .SCORE_W(SCORE_W), .GATE_W(GATE_W)) u_sm (
        .score_q7(sm_scores), .valid(sm_valid), .gate_q17(sm_gates)
    );

    function automatic logic [2:0] pop5(input logic [N_CAND-1:0] m);
        logic [2:0] c; c = 3'd0;
        for (int i = 0; i < N_CAND; i++) c = c + {2'b0, m[i]};
        pop5 = c;
    endfunction
    function automatic logic [2:0] nprobe(input logic [N_CAND-1:0] m);
        nprobe = pop5(m & 5'b11110);
    endfunction
    function automatic logic [DIR_W-1:0] first_nb(input logic [N_CAND-1:0] m);
        logic [DIR_W-1:0] d; logic f; d = DIR_W'(1); f = 1'b0;
        for (int i = 1; i < N_CAND; i++)
            if (!f && m[i]) begin d = DIR_W'(i); f = 1'b1; end
        first_nb = d;
    endfunction
    function automatic logic has_after(input logic [N_CAND-1:0] m, input logic [DIR_W-1:0] c);
        logic f; f = 1'b0;
        for (int i = 0; i < N_CAND; i++)
            if (DIR_W'(i) > c && m[i]) f = 1'b1;
        has_after = f;
    endfunction
    function automatic logic [DIR_W-1:0] next_v(input logic [N_CAND-1:0] m, input logic [DIR_W-1:0] c);
        logic [DIR_W-1:0] d; logic f; d = c; f = 1'b0;
        for (int i = 0; i < N_CAND; i++)
            if (!f && DIR_W'(i) > c && m[i]) begin d = DIR_W'(i); f = 1'b1; end
        next_v = d;
    endfunction

    // Sign-extend TARE 9b Q7 (values in practice non-negative small ints)
    function automatic logic signed [SCORE_W-1:0] sext9(input logic [8:0] s);
        sext9 = SCORE_W'(signed'({ {7{s[8]}}, s }));
    endfunction

    assign anchor_ready = (state_q == ST_IDLE);
    assign probe_ready  = (state_q == ST_PROBE);
    assign edge_valid   = (state_q == ST_EMIT);
    assign edge_tag     = tag_q;
    assign edge_dest_id = dest_q;
    assign edge_dir     = emit_idx_q;
    assign edge_k_bits  = k_cand_q[emit_idx_q];
    assign edge_gate_q17 = gate_q[emit_idx_q];
    assign edge_score_q7 = score_q[emit_idx_q];
    assign edge_last    = edge_valid && !has_after(valid_mask_q, emit_idx_q);
    assign row_done_valid = (state_q == ST_DONE);
    assign row_done_tag = tag_q;
    assign row_done_degree = degree_q;
    assign protocol_error = protocol_error_q;
    assign perf_probe_count = perf_probe_q;
    assign perf_tare_issue_count = perf_tare_q;
    assign perf_edge_emit_count = perf_edge_q;
    assign perf_tare_zero_count = perf_zero_q;
    assign perf_tare_sparse_count = perf_sparse_q;
    assign perf_tare_dense_count = perf_dense_q;

    always_comb begin
        tare_in_valid = 1'b0;
        tare_in_tag = 16'(tare_dir_q);
        tare_q = q_q;
        tare_k_self = k_self_q;
        tare_k_nb = k_cand_q[tare_dir_q];
        tare_out_ready = 1'b0;
        if (state_q == ST_SCORE_TARE && tare_issue_pending_q) begin
            tare_in_valid = 1'b1;
        end
        if (state_q == ST_SCORE_TARE && !tare_issue_pending_q) begin
            tare_out_ready = 1'b1;
        end
    end

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0; dest_q <= '0; q_q <= '0; k_self_q <= '0;
            valid_mask_q <= '0;
            for (int i = 0; i < N_CAND; i++) begin
                k_cand_q[i] <= '0; score_q[i] <= '0; gate_q[i] <= '0;
            end
            emit_idx_q <= '0; tare_dir_q <= '0;
            degree_q <= '0; expected_probe_q <= '0; probes_seen_q <= '0;
            probe_seen_mask_q <= '0;
            protocol_error_q <= 1'b0;
            perf_probe_q <= '0; perf_tare_q <= '0; perf_edge_q <= '0;
            perf_zero_q <= '0; perf_sparse_q <= '0; perf_dense_q <= '0;
            tare_issue_pending_q <= 1'b0;
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    protocol_error_q <= 1'b0;
                    if (anchor_valid) begin
                        tag_q <= anchor_tag;
                        dest_q <= anchor_dest_id;
                        q_q <= anchor_q_bits;
                        k_self_q <= anchor_k_bits;
                        valid_mask_q <= anchor_valid_mask;
                        k_cand_q[0] <= anchor_k_bits;
                        for (int i = 1; i < N_CAND; i++) k_cand_q[i] <= '0;
                        degree_q <= pop5(anchor_valid_mask);
                        expected_probe_q <= nprobe(anchor_valid_mask);
                        probes_seen_q <= '0;
                        probe_seen_mask_q <= '0;
                        // An empty candidate set retires without issuing TARE work.
                        if (anchor_valid_mask == '0) begin
                            state_q <= ST_DONE;
                        end else if (nprobe(anchor_valid_mask) == 0) begin
                            tare_dir_q <= '0;
                            tare_issue_pending_q <= 1'b1;
                            state_q <= ST_SCORE_TARE;
                        end else state_q <= ST_PROBE;
                    end
                end

                ST_PROBE: begin
                    if (probe_valid) begin
                        if (probe_dir == 0 || probe_dir >= DIR_W'(N_CAND) ||
                            !valid_mask_q[probe_dir] ||
                            probe_seen_mask_q[probe_dir]) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else if (
                            probe_last
                            != ((probes_seen_q + 3'd1) == expected_probe_q)
                        ) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            k_cand_q[probe_dir] <= probe_k_bits;
                            probes_seen_q <= probes_seen_q + 3'd1;
                            probe_seen_mask_q[probe_dir] <= 1'b1;
                            perf_probe_q <= perf_probe_q + 16'd1;
                            if (probe_last) begin
                                // start TARE from first valid including self
                                tare_dir_q <= '0;
                                // if self invalid (rare), jump to first_nb
                                if (!valid_mask_q[0])
                                    tare_dir_q <= first_nb(valid_mask_q);
                                tare_issue_pending_q <= 1'b1;
                                state_q <= ST_SCORE_TARE;
                            end
                        end
                    end
                end

                ST_SCORE_TARE: begin
                    if (tare_issue_pending_q) begin
                        if (tare_in_valid && tare_in_ready) begin
                            tare_issue_pending_q <= 1'b0;
                            perf_tare_q <= perf_tare_q + 16'd1;
                        end
                    end else if (tare_out_valid && tare_out_ready) begin
                        score_q[tare_dir_q] <= sext9(tare_out_score);
                        if (tare_out_kind == 2'd0) perf_zero_q <= perf_zero_q + 1'b1;
                        else if (tare_out_kind == 2'd1) perf_sparse_q <= perf_sparse_q + 1'b1;
                        else perf_dense_q <= perf_dense_q + 1'b1;
                        // next valid candidate (self or neighbor)
                        if (has_after(valid_mask_q, tare_dir_q)) begin
                            tare_dir_q <= next_v(valid_mask_q, tare_dir_q);
                            tare_issue_pending_q <= 1'b1;
                        end else begin
                            state_q <= ST_SHIFTMAX;
                        end
                    end
                end

                ST_SHIFTMAX: begin
                    for (int i = 0; i < N_CAND; i++) begin
                        score_q[i] <= score_q[i];
                        gate_q[i]  <= sm_gates[i*GATE_W +: GATE_W];
                        if (!valid_mask_q[i]) begin
                            score_q[i] <= SCORE_W'(-256);
                            gate_q[i]  <= '0;
                        end
                    end
                    if (valid_mask_q == 0) state_q <= ST_DONE;
                    else begin
                        emit_idx_q <= first_nb(valid_mask_q | 5'b00001);
                        // first valid including self
                        begin
                            logic found; logic [DIR_W-1:0] dsel;
                            found = 1'b0; dsel = '0;
                            for (int d = 0; d < N_CAND; d++)
                                if (!found && valid_mask_q[d]) begin
                                    dsel = DIR_W'(d); found = 1'b1;
                                end
                            emit_idx_q <= dsel;
                            state_q <= found ? ST_EMIT : ST_DONE;
                        end
                    end
                end

                ST_EMIT: begin
                    if (edge_ready) begin
                        perf_edge_q <= perf_edge_q + 16'd1;
                        if (has_after(valid_mask_q, emit_idx_q))
                            emit_idx_q <= next_v(valid_mask_q, emit_idx_q);
                        else
                            state_q <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    if (row_done_ready) state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
