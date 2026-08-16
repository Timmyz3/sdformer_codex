`timescale 1ns/1ps
`default_nettype none

// Local5 score -> Shiftmax5 gate -> MFEP term -> DCTF-style cmd chain.
// One destination stencil at a time (row-context). Parity counterpart of
// Motion SCS -> NMF/G1 -> DCTF term path, with multiset multiplicity.
module local5_score_gate_term_top #(
    parameter int HEAD_DIM  = 32,
    parameter int N_CAND    = 5,
    parameter int SCORE_W   = 16,
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int DIR_W     = 3,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter int CMD_SEQ_W = 16,
    parameter int PERF_W    = 32,
    parameter bit USE_TARE  = 1'b0
) (
    input  logic                      clk_core,
    input  logic                      rst_core,

    // Stencil load (same as row-context ANCHOR + PROBE)
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

    // Projection command stream
    output logic                      cmd_valid,
    input  logic                      cmd_ready,
    output logic [TAG_W-1:0]          cmd_group_tag,
    output logic [CMD_SEQ_W-1:0]      cmd_sequence,
    output logic [GATE_W-1:0]         cmd_gate_code,
    output logic [LANE_ID_W-1:0]      cmd_lane_id,
    output logic [DEST_W-1:0]         cmd_destination_token,
    output logic [MULT_W-1:0]         cmd_multiplicity,
    output logic                      cmd_term_first,
    output logic                      cmd_term_last,
    output logic                      cmd_head_last,

    output logic                      stencil_done_valid,
    input  logic                      stencil_done_ready,
    output logic [TAG_W-1:0]          stencil_done_tag,
    output logic                      protocol_error,

    output logic [PERF_W-1:0]         perf_edges,
    output logic [PERF_W-1:0]         perf_terms,
    output logic [PERF_W-1:0]         perf_naive_products,
    output logic [15:0]               perf_tare_issues,
    output logic [15:0]               perf_tare_zero,
    output logic [15:0]               perf_tare_sparse,
    output logic [15:0]               perf_tare_dense
);

    // Row context edges
    logic edge_valid;
    logic edge_ready;
    logic [TAG_W-1:0] edge_tag;
    logic [DEST_W-1:0] edge_dest_id;
    logic [DIR_W-1:0] edge_dir;
    logic [HEAD_DIM-1:0] edge_k_bits;
    logic [GATE_W-1:0] edge_gate_q17;
    logic signed [SCORE_W-1:0] edge_score_q7;
    logic edge_last;
    logic row_done_valid;
    logic row_done_ready;
    logic [TAG_W-1:0] row_done_tag;
    logic [2:0] row_done_degree;
    logic row_protocol_error;
    logic [15:0] perf_probe;
    logic [15:0] perf_edge_emit;
    logic [15:0] row_perf_tare_issue;
    logic [15:0] row_perf_tare_zero;
    logic [15:0] row_perf_tare_sparse;
    logic [15:0] row_perf_tare_dense;
    logic row_anchor_valid;
    logic row_anchor_ready;

    generate
        if (USE_TARE) begin : g_row_tare
            local5_row_context_tare_engine #(
                .HEAD_DIM(HEAD_DIM),
                .N_CAND(N_CAND),
                .SCORE_W(SCORE_W),
                .GATE_W(GATE_W),
                .TAG_W(TAG_W),
                .DEST_W(DEST_W),
                .DIR_W(DIR_W)
            ) u_row (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .anchor_valid(row_anchor_valid),
                .anchor_ready(row_anchor_ready),
                .anchor_tag(anchor_tag),
                .anchor_dest_id(anchor_dest_id),
                .anchor_q_bits(anchor_q_bits),
                .anchor_k_bits(anchor_k_bits),
                .anchor_valid_mask(anchor_valid_mask),
                .probe_valid(probe_valid),
                .probe_ready(probe_ready),
                .probe_dir(probe_dir),
                .probe_k_bits(probe_k_bits),
                .probe_last(probe_last),
                .edge_valid(edge_valid),
                .edge_ready(edge_ready),
                .edge_tag(edge_tag),
                .edge_dest_id(edge_dest_id),
                .edge_dir(edge_dir),
                .edge_k_bits(edge_k_bits),
                .edge_gate_q17(edge_gate_q17),
                .edge_score_q7(edge_score_q7),
                .edge_last(edge_last),
                .row_done_valid(row_done_valid),
                .row_done_ready(row_done_ready),
                .row_done_tag(row_done_tag),
                .row_done_degree(row_done_degree),
                .protocol_error(row_protocol_error),
                .perf_probe_count(perf_probe),
                .perf_tare_issue_count(row_perf_tare_issue),
                .perf_edge_emit_count(perf_edge_emit),
                .perf_tare_zero_count(row_perf_tare_zero),
                .perf_tare_sparse_count(row_perf_tare_sparse),
                .perf_tare_dense_count(row_perf_tare_dense)
            );
        end else begin : g_row_direct
            local5_row_context_engine #(
                .HEAD_DIM(HEAD_DIM),
                .N_CAND(N_CAND),
                .SCORE_W(SCORE_W),
                .GATE_W(GATE_W),
                .TAG_W(TAG_W),
                .DEST_W(DEST_W),
                .DIR_W(DIR_W)
            ) u_row (
                .clk_core(clk_core),
                .rst_core(rst_core),
                .anchor_valid(row_anchor_valid),
                .anchor_ready(row_anchor_ready),
                .anchor_tag(anchor_tag),
                .anchor_dest_id(anchor_dest_id),
                .anchor_q_bits(anchor_q_bits),
                .anchor_k_bits(anchor_k_bits),
                .anchor_valid_mask(anchor_valid_mask),
                .probe_valid(probe_valid),
                .probe_ready(probe_ready),
                .probe_dir(probe_dir),
                .probe_k_bits(probe_k_bits),
                .probe_last(probe_last),
                .edge_valid(edge_valid),
                .edge_ready(edge_ready),
                .edge_tag(edge_tag),
                .edge_dest_id(edge_dest_id),
                .edge_dir(edge_dir),
                .edge_k_bits(edge_k_bits),
                .edge_gate_q17(edge_gate_q17),
                .edge_score_q7(edge_score_q7),
                .edge_last(edge_last),
                .row_done_valid(row_done_valid),
                .row_done_ready(row_done_ready),
                .row_done_tag(row_done_tag),
                .row_done_degree(row_done_degree),
                .protocol_error(row_protocol_error),
                .perf_probe_count(perf_probe),
                .perf_edge_emit_count(perf_edge_emit)
            );

            assign row_perf_tare_issue  = '0;
            assign row_perf_tare_zero   = '0;
            assign row_perf_tare_sparse = '0;
            assign row_perf_tare_dense  = '0;
        end
    endgenerate

    // MFEP side: open destination when first edge arrives
    typedef enum logic [2:0] {
        ST_WAIT_EDGE = 3'd0,
        ST_FEED      = 3'd1,
        ST_WAIT_MFEP = 3'd2,
        ST_WAIT_ROW  = 3'd3,
        ST_FINISH    = 3'd4
    } ctrl_t;

    ctrl_t ctrl_q;
    logic dest_valid;
    logic dest_ready;
    logic [TAG_W-1:0] dest_tag_r;
    logic [DEST_W-1:0] dest_id_r;
    logic mfep_edge_valid;
    logic mfep_edge_ready;
    logic mfep_edge_last;
    logic term_valid;
    logic term_ready;
    logic [TAG_W-1:0] term_tag;
    logic [DEST_W-1:0] term_dest_id;
    logic [LANE_ID_W-1:0] term_lane;
    logic [GATE_W-1:0] term_gate;
    logic [MULT_W-1:0] term_mult;
    logic term_last;
    logic mfep_done_valid;
    logic mfep_done_ready;
    logic [TAG_W-1:0] mfep_done_tag;
    logic mfep_protocol_error;
    logic [PERF_W-1:0] count_edges;
    logic [PERF_W-1:0] count_terms;
    logic [PERF_W-1:0] count_naive;

    local5_mfep_term_builder #(
        .HEAD_DIM(HEAD_DIM),
        .N_CAND(N_CAND),
        .GATE_W(GATE_W),
        .TAG_W(TAG_W),
        .DEST_W(DEST_W),
        .DIR_W(DIR_W),
        .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W),
        .COUNTER_W(PERF_W)
    ) u_mfep (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .dest_valid(dest_valid),
        .dest_ready(dest_ready),
        .dest_tag(dest_tag_r),
        .dest_id(dest_id_r),
        .edge_valid(mfep_edge_valid),
        .edge_ready(mfep_edge_ready),
        .edge_dir(edge_dir),
        .edge_gate_q17(edge_gate_q17),
        .edge_k_bits(edge_k_bits),
        .edge_last(mfep_edge_last),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_tag(term_tag),
        .term_dest_id(term_dest_id),
        .term_lane(term_lane),
        .term_gate_q17(term_gate),
        .term_multiplicity(term_mult),
        .term_last(term_last),
        .dest_done_valid(mfep_done_valid),
        .dest_done_ready(mfep_done_ready),
        .dest_done_tag(mfep_done_tag),
        .protocol_error(mfep_protocol_error),
        .count_edges(count_edges),
        .count_terms(count_terms),
        .count_naive_products(count_naive)
    );

    logic adapter_error;
    local5_mfep_dctf_cmd_adapter #(
        .GATE_W(GATE_W),
        .TAG_W(TAG_W),
        .DEST_W(DEST_W),
        .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W),
        .CMD_SEQ_W(CMD_SEQ_W),
        .EXPLODE_MODE(1'b0)
    ) u_adapter (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(term_valid),
        .term_ready(term_ready),
        .term_tag(term_tag),
        .term_dest_id(term_dest_id),
        .term_lane(term_lane),
        .term_gate_q17(term_gate),
        .term_multiplicity(term_mult),
        .term_last(term_last),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_group_tag(cmd_group_tag),
        .cmd_sequence(cmd_sequence),
        .cmd_gate_code(cmd_gate_code),
        .cmd_lane_id(cmd_lane_id),
        .cmd_destination_token(cmd_destination_token),
        .cmd_multiplicity(cmd_multiplicity),
        .cmd_term_first(cmd_term_first),
        .cmd_term_last(cmd_term_last),
        .cmd_head_last(cmd_head_last),
        .protocol_error(adapter_error)
    );

    // Glue: open MFEP dest, feed edges, retire both engines
    logic glue_error_q;
    logic [TAG_W-1:0] done_tag_q;
    assign protocol_error = row_protocol_error | mfep_protocol_error |
                            adapter_error | glue_error_q;
    assign perf_edges = count_edges;
    assign perf_terms = count_terms;
    assign perf_naive_products = count_naive;
    assign perf_tare_issues = row_perf_tare_issue;
    assign perf_tare_zero = row_perf_tare_zero;
    assign perf_tare_sparse = row_perf_tare_sparse;
    assign perf_tare_dense = row_perf_tare_dense;

    // The row engine can return to IDLE before the externally visible done
    // handshake. Keep the next anchor blocked until the previous stencil is
    // retired so its tag and lifecycle cannot be overwritten.
    assign row_anchor_valid = anchor_valid && (ctrl_q == ST_WAIT_EDGE);
    assign anchor_ready = row_anchor_ready && (ctrl_q == ST_WAIT_EDGE);
    assign dest_valid = (ctrl_q == ST_WAIT_EDGE) && edge_valid;
    assign dest_tag_r = edge_tag;
    assign dest_id_r = edge_dest_id;

    // Feed edges only after dest opened
    assign mfep_edge_valid = (ctrl_q == ST_FEED) && edge_valid;
    assign mfep_edge_last = edge_last;
    assign edge_ready = (ctrl_q == ST_WAIT_EDGE) ? 1'b0 :
                        (ctrl_q == ST_FEED) ? mfep_edge_ready : 1'b0;

    assign mfep_done_ready = (ctrl_q == ST_WAIT_MFEP);
    assign row_done_ready  = (ctrl_q == ST_WAIT_ROW);
    assign stencil_done_valid = (ctrl_q == ST_FINISH);
    assign stencil_done_tag = done_tag_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            ctrl_q <= ST_WAIT_EDGE;
            glue_error_q <= 1'b0;
            done_tag_q <= '0;
        end else begin
            case (ctrl_q)
                ST_WAIT_EDGE: begin
                    if (edge_valid && dest_ready) begin
                        // dest accepted this cycle; next cycle feed including
                        // the first edge (row engine holds edge until ready)
                        // Problem: edge not accepted yet. Open dest first cycle
                        // without consuming edge, then FEED.
                        ctrl_q <= ST_FEED;
                    end else if (row_done_valid && !edge_valid) begin
                        // zero edges path
                        ctrl_q <= ST_WAIT_ROW;
                    end
                end
                ST_FEED: begin
                    if (edge_valid && mfep_edge_ready && edge_last) begin
                        ctrl_q <= ST_WAIT_MFEP;
                    end
                end
                ST_WAIT_MFEP: begin
                    if (mfep_done_valid) begin
                        ctrl_q <= ST_WAIT_ROW;
                    end
                end
                ST_WAIT_ROW: begin
                    if (row_done_valid) begin
                        done_tag_q <= row_done_tag;
                        ctrl_q <= ST_FINISH;
                    end
                end
                ST_FINISH: begin
                    if (stencil_done_ready) begin
                        ctrl_q <= ST_WAIT_EDGE;
                    end
                end
                default: ctrl_q <= ST_WAIT_EDGE;
            endcase
        end
    end

endmodule

`default_nettype wire
