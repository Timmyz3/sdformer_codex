`timescale 1ns/1ps
`default_nettype none

// Multi-destination Local5 window attention top (Local5-only modules).
// Flow per destination:
//   STT issue → ANCHOR_LOAD → PROBEs → score_gate_term → multiset bridge → banklocal
// Window finishes when last destination's last cmd sets cmd_window_last.
module local5_window_attention_top #(
    parameter int HEAD_DIM  = 32,
    parameter int N_CAND    = 5,
    parameter int OUT_DIM   = 4,
    parameter int MAX_DEST  = 16,
    parameter int SCORE_W   = 16,
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int DIR_W     = 3,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter bit EXPLODE_MULT = 1'b0,
    parameter bit USE_TARE = 1'b0
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    input  logic                       w_load_valid,
    output logic                       w_load_ready,
    input  logic [LANE_ID_W-1:0]       w_load_lane,
    input  logic [$clog2(OUT_DIM)-1:0] w_load_out,
    input  logic signed [7:0]          w_load_data,
    input  logic                       w_load_last,

    input  logic                       run_start,
    output logic                       run_busy,
    output logic                       run_done,

    input  logic                       dest_valid,
    output logic                       dest_ready,
    input  logic [TAG_W-1:0]           dest_tag,
    input  logic [DEST_W-1:0]          dest_id,
    input  logic [HEAD_DIM-1:0]        dest_q,
    input  logic [HEAD_DIM-1:0]        dest_k_self,
    input  logic [N_CAND-1:0]          dest_valid_mask,
    input  logic [HEAD_DIM-1:0]        dest_k_n,
    input  logic [HEAD_DIM-1:0]        dest_k_s,
    input  logic [HEAD_DIM-1:0]        dest_k_e,
    input  logic [HEAD_DIM-1:0]        dest_k_w,
    input  logic                       dest_last_in_window,

    input  logic                       acc_read_valid,
    output logic                       acc_read_ready,
    input  logic [DEST_W-1:0]          acc_read_dest,
    input  logic [$clog2(OUT_DIM)-1:0] acc_read_out,
    output logic                       acc_data_valid,
    output logic signed [31:0]         acc_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_dest_count,
    output logic [31:0]                perf_cmd_count,
    output logic [31:0]                perf_cycle_count
);

    typedef enum logic [3:0] {
        ST_IDLE      = 4'd0,
        ST_ARM_PROJ  = 4'd1,
        ST_ACCEPT    = 4'd2,
        ST_ANCHOR    = 4'd3,
        ST_PROBE     = 4'd4,
        ST_WAIT_SGT  = 4'd5,
        ST_WAIT_PROJ = 4'd6,
        ST_FINISH    = 4'd7
    } state_t;

    state_t state_q;

    // Held destination operands
    logic [TAG_W-1:0]  h_tag;
    logic [DEST_W-1:0] h_id;
    logic [HEAD_DIM-1:0] h_q, h_ks, h_kn, h_ksou, h_ke, h_kw;
    logic [N_CAND-1:0] h_mask;
    logic h_last;

    logic [2:0] probe_dir_q;
    logic [2:0] probes_left_q;
    logic [31:0] dest_count_q;
    logic [31:0] cycle_q;
    logic glue_err_q;
    logic weights_loaded_q;

    // SGT
    logic anchor_valid, anchor_ready;
    logic probe_valid, probe_ready;
    logic [DIR_W-1:0] probe_dir;
    logic [HEAD_DIM-1:0] probe_k;
    logic probe_last;
    logic sgt_cmd_valid, sgt_cmd_ready;
    logic [TAG_W-1:0] sgt_cmd_tag;
    logic [15:0] sgt_cmd_seq;
    logic [GATE_W-1:0] sgt_cmd_gate;
    logic [LANE_ID_W-1:0] sgt_cmd_lane;
    logic [DEST_W-1:0] sgt_cmd_dest;
    logic [MULT_W-1:0] sgt_cmd_mult;
    logic sgt_term_first, sgt_term_last, sgt_head_last;
    logic stencil_done_valid, stencil_done_ready;
    logic [TAG_W-1:0] stencil_done_tag;
    logic sgt_err;
    logic [31:0] pe, pt, pn;
    logic [15:0] sgt_tare_issues, sgt_tare_zero;
    logic [15:0] sgt_tare_sparse, sgt_tare_dense;

    local5_score_gate_term_top #(
        .HEAD_DIM(HEAD_DIM), .N_CAND(N_CAND), .SCORE_W(SCORE_W), .GATE_W(GATE_W),
        .TAG_W(TAG_W), .DEST_W(DEST_W), .DIR_W(DIR_W), .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W), .USE_TARE(USE_TARE)
    ) u_sgt (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .anchor_valid(anchor_valid),
        .anchor_ready(anchor_ready),
        .anchor_tag(h_tag),
        .anchor_dest_id(h_id),
        .anchor_q_bits(h_q),
        .anchor_k_bits(h_ks),
        .anchor_valid_mask(h_mask),
        .probe_valid(probe_valid),
        .probe_ready(probe_ready),
        .probe_dir(probe_dir),
        .probe_k_bits(probe_k),
        .probe_last(probe_last),
        .cmd_valid(sgt_cmd_valid),
        .cmd_ready(sgt_cmd_ready),
        .cmd_group_tag(sgt_cmd_tag),
        .cmd_sequence(sgt_cmd_seq),
        .cmd_gate_code(sgt_cmd_gate),
        .cmd_lane_id(sgt_cmd_lane),
        .cmd_destination_token(sgt_cmd_dest),
        .cmd_multiplicity(sgt_cmd_mult),
        .cmd_term_first(sgt_term_first),
        .cmd_term_last(sgt_term_last),
        .cmd_head_last(sgt_head_last),
        .stencil_done_valid(stencil_done_valid),
        .stencil_done_ready(stencil_done_ready),
        .stencil_done_tag(stencil_done_tag),
        .protocol_error(sgt_err),
        .perf_edges(pe),
        .perf_terms(pt),
        .perf_naive_products(pn),
        .perf_tare_issues(sgt_tare_issues),
        .perf_tare_zero(sgt_tare_zero),
        .perf_tare_sparse(sgt_tare_sparse),
        .perf_tare_dense(sgt_tare_dense)
    );

    // Bridge
    logic b_cmd_valid, b_cmd_ready;
    logic [TAG_W-1:0] b_tag;
    logic [15:0] b_seq;
    logic [GATE_W-1:0] b_gate;
    logic [LANE_ID_W-1:0] b_lane;
    logic [DEST_W-1:0] b_dest;
    logic [MULT_W-1:0] b_mult;
    logic [12:0] b_issue;
    logic b_first, b_last, b_head_last;
    logic b_err;
    logic [31:0] b_cc, b_ce;

    local5_dctf_multiset_bridge #(
        .GATE_W(GATE_W), .TAG_W(TAG_W), .DEST_W(DEST_W), .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W), .EXPLODE(EXPLODE_MULT)
    ) u_bridge (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .term_valid(sgt_cmd_valid),
        .term_ready(sgt_cmd_ready),
        .term_tag(sgt_cmd_tag),
        .term_dest_id(sgt_cmd_dest),
        .term_lane(sgt_cmd_lane),
        .term_gate(sgt_cmd_gate),
        .term_mult(sgt_cmd_mult),
        .term_last(sgt_term_last),
        .term_head_last(sgt_term_last && h_last),
        .cmd_valid(b_cmd_valid),
        .cmd_ready(b_cmd_ready),
        .cmd_group_tag(b_tag),
        .cmd_sequence(b_seq),
        .cmd_gate_code(b_gate),
        .cmd_lane_id(b_lane),
        .cmd_destination_token(b_dest),
        .cmd_multiplicity(b_mult),
        .cmd_term_issue_seq(b_issue),
        .cmd_term_first(b_first),
        .cmd_term_last(b_last),
        .cmd_head_last(b_head_last),
        .protocol_error(b_err),
        .count_cmds(b_cc),
        .count_exploded(b_ce)
    );

    logic proj_start;
    logic proj_busy, proj_done, proj_err;
    logic proj_acc_read_valid, proj_acc_read_ready, proj_acc_data_valid;
    logic signed [31:0] proj_acc_data;
    logic [31:0] proj_cmds, proj_prods;
    logic window_last_cmd;

    // Attach end-of-window to the last real command. If the last destination
    // produces no term, emit an ordered command-less close after both stream
    // adapters are drained.
    assign window_last_cmd =
        (b_cmd_valid && b_head_last)
        || ((state_q == ST_WAIT_PROJ) && !b_cmd_valid
            && !sgt_cmd_valid && !proj_done);

    local5_banklocal_projection_top #(
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .MAX_DEST(MAX_DEST),
        .GATE_W(GATE_W), .TAG_W(TAG_W), .DEST_W(DEST_W), .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W)
    ) u_proj (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .w_load_valid(w_load_valid),
        .w_load_ready(w_load_ready),
        .w_load_lane(w_load_lane),
        .w_load_out(w_load_out),
        .w_load_data(w_load_data),
        .w_load_last(w_load_last),
        .run_start(proj_start),
        .run_busy(proj_busy),
        .run_done(proj_done),
        .cmd_valid(b_cmd_valid),
        .cmd_ready(b_cmd_ready),
        .cmd_group_tag(b_tag),
        .cmd_gate_code(b_gate),
        .cmd_lane_id(b_lane),
        .cmd_destination_token(b_dest),
        .cmd_multiplicity(b_mult),
        .cmd_head_last(b_head_last),
        .cmd_window_last(window_last_cmd),
        .acc_read_valid(proj_acc_read_valid),
        .acc_read_ready(proj_acc_read_ready),
        .acc_read_dest(acc_read_dest),
        .acc_read_out(acc_read_out),
        .acc_data_valid(proj_acc_data_valid),
        .acc_data(proj_acc_data),
        .protocol_error(proj_err),
        .perf_cmd_count(proj_cmds),
        .perf_product_count(proj_prods)
    );

    // STT (observability / phase sideband)
    logic stt_ready, stt_err;
    local5_stt_descriptor #(.TAG_W(TAG_W), .DEST_W(DEST_W)) u_stt (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .issue_valid(anchor_valid && anchor_ready),
        .issue_ready(stt_ready),
        .issue_tag(h_tag),
        .issue_dest_id(h_id),
        .issue_valid_mask(h_mask),
        .issue_boundary(1'b0),
        .issue_delta_class(3'd0),
        .live_valid(),
        .live_tag(),
        .live_dest_id(),
        .live_valid_mask(),
        .live_boundary(),
        .live_delta_class(),
        .live_degree(),
        .live_phase(),
        .mark_score_start(anchor_valid && anchor_ready),
        .mark_term_start(state_q == ST_WAIT_SGT),
        .mark_commit(stencil_done_valid && stencil_done_ready),
        .retire_ready(state_q == ST_ACCEPT
                      || state_q == ST_WAIT_PROJ
                      || state_q == ST_FINISH),
        .protocol_error(stt_err)
    );

    assign run_busy = (state_q != ST_IDLE && state_q != ST_FINISH);
    assign run_done = (state_q == ST_FINISH);
    assign proj_acc_read_valid = acc_read_valid && (state_q == ST_FINISH);
    assign acc_read_ready = (state_q == ST_FINISH) && proj_acc_read_ready;
    assign acc_data_valid = (state_q == ST_FINISH) && proj_acc_data_valid;
    assign acc_data = acc_data_valid ? proj_acc_data : '0;
    assign dest_ready = (state_q == ST_ACCEPT);
    assign anchor_valid = (state_q == ST_ANCHOR);
    assign probe_valid = (state_q == ST_PROBE);
    assign probe_last = (probes_left_q == 3'd1);
    assign stencil_done_ready = (state_q == ST_WAIT_SGT);
    assign protocol_error = sgt_err | b_err | proj_err | stt_err | glue_err_q;
    assign perf_dest_count = dest_count_q;
    assign perf_cmd_count = proj_cmds;
    assign perf_cycle_count = cycle_q;

    always_comb begin
        probe_k = '0;
        probe_dir = DIR_W'(probe_dir_q);
        unique case (probe_dir_q)
            3'd1: probe_k = h_kn;
            3'd2: probe_k = h_ksou;
            3'd3: probe_k = h_ke;
            3'd4: probe_k = h_kw;
            default: probe_k = '0;
        endcase
    end

    function automatic logic [2:0] count_probes(input logic [N_CAND-1:0] m);
        logic [2:0] c;
        c = 3'd0;
        for (int d = 1; d < N_CAND; d = d + 1) c = c + {2'b0, m[d]};
        return c;
    endfunction

    function automatic logic [2:0] first_probe(input logic [N_CAND-1:0] m);
        logic [2:0] dsel;
        logic found;
        dsel = 3'd1;
        found = 1'b0;
        for (int d = 1; d < N_CAND; d = d + 1) begin
            if (!found && m[d]) begin
                dsel = 3'(d);
                found = 1'b1;
            end
        end
        return dsel;
    endfunction

    function automatic logic [2:0] next_probe(
        input logic [N_CAND-1:0] m,
        input logic [2:0] cur
    );
        logic [2:0] dsel;
        logic found;
        dsel = cur;
        found = 1'b0;
        for (int d = 1; d < N_CAND; d = d + 1) begin
            if (!found && 3'(d) > cur && m[d]) begin
                dsel = 3'(d);
                found = 1'b1;
            end
        end
        return dsel;
    endfunction

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            h_tag <= '0; h_id <= '0;
            h_q <= '0; h_ks <= '0; h_kn <= '0; h_ksou <= '0; h_ke <= '0; h_kw <= '0;
            h_mask <= '0; h_last <= 1'b0;
            probe_dir_q <= '0; probes_left_q <= '0;
            dest_count_q <= '0; cycle_q <= '0; glue_err_q <= 1'b0;
            weights_loaded_q <= 1'b0;
            proj_start <= 1'b0;
        end else begin
            proj_start <= 1'b0;

            if (w_load_valid && w_load_ready && w_load_last)
                weights_loaded_q <= 1'b1;

            if (state_q != ST_IDLE && state_q != ST_FINISH)
                cycle_q <= cycle_q + 1'b1;
            if (sgt_err || b_err || proj_err || stt_err)
                glue_err_q <= 1'b1;

            unique case (state_q)
                ST_IDLE: begin
                    if (run_start) begin
                        if (weights_loaded_q) begin
                            cycle_q <= '0;
                            dest_count_q <= '0;
                            glue_err_q <= 1'b0;
                            proj_start <= 1'b1;
                            state_q <= ST_ARM_PROJ;
                        end else begin
                            glue_err_q <= 1'b1;
                        end
                    end
                end
                ST_ARM_PROJ: begin
                    state_q <= ST_ACCEPT;
                end
                ST_ACCEPT: begin
                    if (dest_valid) begin
                        h_tag <= dest_tag;
                        h_id <= dest_id;
                        h_q <= dest_q;
                        h_ks <= dest_k_self;
                        h_kn <= dest_k_n;
                        h_ksou <= dest_k_s;
                        h_ke <= dest_k_e;
                        h_kw <= dest_k_w;
                        h_mask <= dest_valid_mask;
                        h_last <= dest_last_in_window;
                        dest_count_q <= dest_count_q + 1'b1;
                        state_q <= ST_ANCHOR;
                    end
                end
                ST_ANCHOR: begin
                    if (anchor_ready) begin
                        if (!stt_ready)
                            glue_err_q <= 1'b1;
                        probes_left_q <= count_probes(h_mask);
                        if (count_probes(h_mask) == 3'd0) begin
                            state_q <= ST_WAIT_SGT;
                        end else begin
                            probe_dir_q <= first_probe(h_mask);
                            state_q <= ST_PROBE;
                        end
                    end
                end
                ST_PROBE: begin
                    if (probe_ready) begin
                        if (probes_left_q == 3'd1) begin
                            state_q <= ST_WAIT_SGT;
                        end else begin
                            probes_left_q <= probes_left_q - 3'd1;
                            probe_dir_q <= next_probe(h_mask, probe_dir_q);
                        end
                    end
                end
                ST_WAIT_SGT: begin
                    if (stencil_done_valid) begin
                        if (h_last) state_q <= ST_WAIT_PROJ;
                        else state_q <= ST_ACCEPT;
                    end
                end
                ST_WAIT_PROJ: begin
                    if (proj_done)
                        state_q <= ST_FINISH;
                end
                ST_FINISH: begin
                    if (run_start) begin
                        if (weights_loaded_q) begin
                            cycle_q <= '0;
                            dest_count_q <= '0;
                            glue_err_q <= 1'b0;
                            proj_start <= 1'b1;
                            state_q <= ST_ARM_PROJ;
                        end else begin
                            glue_err_q <= 1'b1;
                        end
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
