`timescale 1ns/1ps
`default_nettype none

// Local5 row-context score/gate engine (ANCHOR_LOAD + PROBE + RETIRE).
// Candidate order: 0=self, 1=N, 2=S, 3=E, 4=W.
// Self anchor is loaded once; neighbor K arrive on PROBE beats. Scores use
// the same alpha-XNOR Q7 leaf as local5_stencil_token (bit-exact path).
// Neighbor residual scoring via TARE-4 is available on the dual-mode top and
// is exercised by the dual-line score substrate, not re-duplicated here.
module local5_row_context_engine #(
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
    output logic [15:0]               perf_edge_emit_count
);

    typedef enum logic [2:0] {
        ST_IDLE    = 3'd0,
        ST_PROBE   = 3'd1,
        ST_COMPUTE = 3'd2,
        ST_EMIT    = 3'd3,
        ST_DONE    = 3'd4
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [HEAD_DIM-1:0] q_q;
    logic [N_CAND-1:0] valid_mask_q;
    logic [HEAD_DIM-1:0] k_cand_q [0:N_CAND-1];
    logic signed [SCORE_W-1:0] score_q [0:N_CAND-1];
    logic [GATE_W-1:0] gate_q [0:N_CAND-1];
    logic [DIR_W-1:0] emit_idx_q;
    logic [2:0] degree_q;
    logic [2:0] expected_probe_q;
    logic [2:0] probes_seen_q;
    logic [N_CAND-1:0] probe_seen_mask_q;
    logic protocol_error_q;
    logic [15:0] perf_probe_q;
    logic [15:0] perf_edge_q;

    // Combinational scores for all candidates from latched Q/K
    logic [N_CAND*SCORE_W-1:0] score_comb;
    logic [N_CAND*GATE_W-1:0]  gate_comb;
    logic [N_CAND-1:0]         sm_valid;

    genvar gi;
    generate
        for (gi = 0; gi < N_CAND; gi = gi + 1) begin : g_score
            logic [$clog2(HEAD_DIM+1)-1:0] ov_unused;
            logic [$clog2(HEAD_DIM+1)-1:0] sz_unused;
            local5_axnor_score_q7 #(
                .HEAD_DIM(HEAD_DIM),
                .SCORE_W(SCORE_W)
            ) u_score (
                .q_bits(q_q),
                .k_bits(k_cand_q[gi]),
                .overlap(ov_unused),
                .same_zero(sz_unused),
                .score_q7(score_comb[gi*SCORE_W +: SCORE_W])
            );
        end
    endgenerate

    integer vi;
    always_comb begin
        for (vi = 0; vi < N_CAND; vi = vi + 1) begin
            sm_valid[vi] = valid_mask_q[vi];
        end
    end

    local5_shiftmax5_q17 #(
        .N_CAND(N_CAND),
        .SCORE_W(SCORE_W),
        .GATE_W(GATE_W)
    ) u_sm (
        .score_q7(score_comb),
        .valid(sm_valid),
        .gate_q17(gate_comb)
    );

    function automatic logic [2:0] popcount5(input logic [N_CAND-1:0] mask);
        logic [2:0] c;
        c = 3'd0;
        for (int i = 0; i < N_CAND; i = i + 1) begin
            c = c + {2'b0, mask[i]};
        end
        popcount5 = c;
    endfunction

    function automatic logic [2:0] expected_probes(input logic [N_CAND-1:0] mask);
        expected_probes = popcount5(mask & 5'b11110);
    endfunction

    function automatic logic [DIR_W-1:0] first_valid_dir(
        input logic [N_CAND-1:0] mask
    );
        logic [DIR_W-1:0] dsel;
        logic found;
        dsel = '0;
        found = 1'b0;
        for (int d = 0; d < N_CAND; d = d + 1) begin
            if (!found && mask[d]) begin
                dsel = DIR_W'(d);
                found = 1'b1;
            end
        end
        first_valid_dir = dsel;
    endfunction

    function automatic logic has_valid_after(
        input logic [N_CAND-1:0] mask,
        input logic [DIR_W-1:0] cur
    );
        logic found;
        found = 1'b0;
        for (int d = 0; d < N_CAND; d = d + 1) begin
            if (DIR_W'(d) > cur && mask[d]) begin
                found = 1'b1;
            end
        end
        has_valid_after = found;
    endfunction

    function automatic logic [DIR_W-1:0] next_valid_dir(
        input logic [N_CAND-1:0] mask,
        input logic [DIR_W-1:0] cur
    );
        logic [DIR_W-1:0] dsel;
        logic found;
        dsel = cur;
        found = 1'b0;
        for (int d = 0; d < N_CAND; d = d + 1) begin
            if (!found && DIR_W'(d) > cur && mask[d]) begin
                dsel = DIR_W'(d);
                found = 1'b1;
            end
        end
        next_valid_dir = dsel;
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
    assign edge_last    = edge_valid && !has_valid_after(valid_mask_q, emit_idx_q);

    assign row_done_valid = (state_q == ST_DONE);
    assign row_done_tag = tag_q;
    assign row_done_degree = degree_q;
    assign protocol_error = protocol_error_q;
    assign perf_probe_count = perf_probe_q;
    assign perf_edge_emit_count = perf_edge_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            tag_q <= '0;
            dest_q <= '0;
            q_q <= '0;
            valid_mask_q <= '0;
            for (int i = 0; i < N_CAND; i = i + 1) begin
                k_cand_q[i] <= '0;
                score_q[i] <= '0;
                gate_q[i] <= '0;
            end
            emit_idx_q <= '0;
            degree_q <= '0;
            expected_probe_q <= '0;
            probes_seen_q <= '0;
            probe_seen_mask_q <= '0;
            protocol_error_q <= 1'b0;
            perf_probe_q <= '0;
            perf_edge_q <= '0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    protocol_error_q <= 1'b0;
                    if (anchor_valid) begin
                        tag_q <= anchor_tag;
                        dest_q <= anchor_dest_id;
                        q_q <= anchor_q_bits;
                        valid_mask_q <= anchor_valid_mask;
                        k_cand_q[0] <= anchor_k_bits;
                        for (int i = 1; i < N_CAND; i = i + 1) begin
                            k_cand_q[i] <= '0;
                        end
                        degree_q <= popcount5(anchor_valid_mask);
                        expected_probe_q <= expected_probes(anchor_valid_mask);
                        probes_seen_q <= '0;
                        probe_seen_mask_q <= '0;
                        if (expected_probes(anchor_valid_mask) == 3'd0) begin
                            state_q <= ST_COMPUTE;
                        end else begin
                            state_q <= ST_PROBE;
                        end
                    end
                end

                ST_PROBE: begin
                    if (probe_valid) begin
                        if (probe_dir == '0 || probe_dir >= DIR_W'(N_CAND) ||
                            !valid_mask_q[probe_dir] ||
                            probe_seen_mask_q[probe_dir]) begin
                            protocol_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else if (
                            probe_last
                            != ((probes_seen_q + 3'd1) == expected_probe_q)
                        ) begin
                            // A malformed row is aborted atomically. Never
                            // expose scores computed from a partial probe set.
                            protocol_error_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            k_cand_q[probe_dir] <= probe_k_bits;
                            probes_seen_q <= probes_seen_q + 3'd1;
                            probe_seen_mask_q[probe_dir] <= 1'b1;
                            perf_probe_q <= perf_probe_q + 16'd1;
                            if (probe_last) begin
                                state_q <= ST_COMPUTE;
                            end
                        end
                    end
                end

                ST_COMPUTE: begin
                    for (int i = 0; i < N_CAND; i = i + 1) begin
                        score_q[i] <= score_comb[i*SCORE_W +: SCORE_W];
                        gate_q[i]  <= gate_comb[i*GATE_W +: GATE_W];
                        if (!valid_mask_q[i]) begin
                            score_q[i] <= SCORE_W'(-256);
                            gate_q[i]  <= '0;
                        end
                    end
                    if (valid_mask_q == '0) begin
                        state_q <= ST_DONE;
                    end else begin
                        emit_idx_q <= first_valid_dir(valid_mask_q);
                        state_q <= ST_EMIT;
                    end
                end

                ST_EMIT: begin
                    if (edge_ready) begin
                        perf_edge_q <= perf_edge_q + 16'd1;
                        if (has_valid_after(valid_mask_q, emit_idx_q)) begin
                            emit_idx_q <= next_valid_dir(valid_mask_q, emit_idx_q);
                        end else begin
                            state_q <= ST_DONE;
                        end
                    end
                end

                ST_DONE: begin
                    if (row_done_ready) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
