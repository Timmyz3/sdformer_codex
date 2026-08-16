`timescale 1ns/1ps
`default_nettype none

// Full Local5 row-stream window:
//   push 3 rows into line buffer → fetch stencils along curr row →
//   window attention (score→MFEP→bridge→multibank proj)
// Local5-only; does not edit Motion RTL.
module local5_linebuf_window_top #(
    parameter int HEAD_DIM   = 32,
    parameter int ROW_TOKENS = 8,
    parameter int OUT_DIM    = 4,
    parameter int MAX_DEST   = 16,
    parameter int NUM_BANKS  = 3,
    parameter int GATE_W     = 9,
    parameter int TAG_W      = 16,
    parameter int DEST_W     = 8,
    parameter int MULT_W     = 3,
    parameter int LANE_ID_W  = 5,
    parameter int TOKEN_W    = (ROW_TOKENS <= 1) ? 1 : $clog2(ROW_TOKENS),
    parameter bit USE_TARE   = 1'b0
) (
    input  logic                       clk_core,
    input  logic                       rst_core,

    // weight load
    input  logic                       w_load_valid,
    output logic                       w_load_ready,
    input  logic [LANE_ID_W-1:0]       w_load_lane,
    input  logic [$clog2(OUT_DIM)-1:0] w_load_out,
    input  logic signed [7:0]          w_load_data,
    input  logic                       w_load_last,

    // push three rows (prev, curr, next) then start
    input  logic                       row_push_valid,
    output logic                       row_push_ready,
    input  logic [TAG_W-1:0]           row_push_tag,
    input  logic [HEAD_DIM-1:0]        row_push_q [0:ROW_TOKENS-1],
    input  logic [HEAD_DIM-1:0]        row_push_k [0:ROW_TOKENS-1],
    input  logic [ROW_TOKENS-1:0]      row_push_valid_mask,

    input  logic                       run_start, // after 3 rows loaded
    output logic                       run_busy,
    output logic                       run_done,

    // Acc readback
    input  logic                       acc_read_valid,
    output logic                       acc_read_ready,
    input  logic [DEST_W-1:0]          acc_read_dest,
    input  logic [$clog2(OUT_DIM)-1:0] acc_read_out,
    output logic                       acc_data_valid,
    output logic signed [31:0]         acc_data,

    output logic                       protocol_error,
    output logic [31:0]                perf_dest_count,
    output logic [31:0]                perf_cmd_count,
    output logic [31:0]                perf_cycle_count,
    output logic [31:0]                perf_bank_conflict_count
);

    // Line buffer
    logic lbuf_rd_valid, lbuf_rd_ready;
    logic [1:0] lbuf_rd_row;
    logic [TOKEN_W-1:0] lbuf_rd_x;
    logic lbuf_rsp_valid, lbuf_rsp_ready;
    logic [HEAD_DIM-1:0] lbuf_rq, lbuf_rk;
    logic lbuf_rv;
    logic [TAG_W-1:0] lbuf_tag;
    logic [1:0] lbuf_filled;
    logic lbuf_err;

    local5_line_buffer_3row #(
        .HEAD_DIM(HEAD_DIM), .ROW_TOKENS(ROW_TOKENS), .TAG_W(TAG_W)
    ) u_lbuf (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .row_push_valid(row_push_valid),
        .row_push_ready(row_push_ready),
        .row_push_tag(row_push_tag),
        .row_push_q(row_push_q),
        .row_push_k(row_push_k),
        .row_push_valid_mask(row_push_valid_mask),
        .rd_valid(lbuf_rd_valid),
        .rd_ready(lbuf_rd_ready),
        .rd_row_sel(lbuf_rd_row),
        .rd_x(lbuf_rd_x),
        .rd_rsp_valid(lbuf_rsp_valid),
        .rd_rsp_ready(lbuf_rsp_ready),
        .rd_q_bits(lbuf_rq),
        .rd_k_bits(lbuf_rk),
        .rd_token_valid(lbuf_rv),
        .curr_row_tag(lbuf_tag),
        .rows_filled(lbuf_filled),
        .protocol_error(lbuf_err)
    );

    // Fetcher
    logic fetch_req_valid, fetch_req_ready;
    logic [TAG_W-1:0] fetch_tag;
    logic [DEST_W-1:0] fetch_dest;
    logic [TOKEN_W-1:0] fetch_x;
    logic fetch_last;
    logic st_valid, st_ready;
    logic [TAG_W-1:0] st_tag;
    logic [DEST_W-1:0] st_dest;
    logic [HEAD_DIM-1:0] st_q, st_ks, st_kn, st_ksou, st_ke, st_kw;
    logic [4:0] st_mask;
    logic st_last;
    logic fetch_err;

    local5_stencil_linebuf_fetcher #(
        .HEAD_DIM(HEAD_DIM), .ROW_TOKENS(ROW_TOKENS), .TAG_W(TAG_W), .DEST_W(DEST_W)
    ) u_fetch (
        .clk_core(clk_core),
        .rst_core(rst_core),
        .req_valid(fetch_req_valid),
        .req_ready(fetch_req_ready),
        .req_tag(fetch_tag),
        .req_dest_id(fetch_dest),
        .req_x(fetch_x),
        .req_last(fetch_last),
        .rd_valid(lbuf_rd_valid),
        .rd_ready(lbuf_rd_ready),
        .rd_row_sel(lbuf_rd_row),
        .rd_x(lbuf_rd_x),
        .rd_rsp_valid(lbuf_rsp_valid),
        .rd_rsp_ready(lbuf_rsp_ready),
        .rd_q_bits(lbuf_rq),
        .rd_k_bits(lbuf_rk),
        .rd_token_valid(lbuf_rv),
        .stencil_valid(st_valid),
        .stencil_ready(st_ready),
        .stencil_tag(st_tag),
        .stencil_dest_id(st_dest),
        .stencil_q(st_q),
        .stencil_k_self(st_ks),
        .stencil_k_n(st_kn),
        .stencil_k_s(st_ksou),
        .stencil_k_e(st_ke),
        .stencil_k_w(st_kw),
        .stencil_valid_mask(st_mask),
        .stencil_last(st_last),
        .protocol_error(fetch_err)
    );

    // Reuse window attention score path pieces via score_gate_term + bridge + multibank
    // Scheduler FSM
    typedef enum logic [3:0] {
        ST_IDLE     = 4'd0,
        ST_ARM      = 4'd1,
        ST_ISSUE_X  = 4'd2,
        ST_WAIT_ST  = 4'd3,
        ST_ANCHOR   = 4'd4,
        ST_PROBE    = 4'd5,
        ST_WAIT_SGT = 4'd6,
        ST_NEXT_X   = 4'd7,
        ST_WAIT_PROJ= 4'd8,
        ST_FINISH   = 4'd9
    } state_t;

    state_t state_q;
    logic [TOKEN_W-1:0] x_q;
    logic [31:0] dest_count_q, cycle_q;
    logic glue_err_q;

    // held stencil
    logic [TAG_W-1:0] h_tag;
    logic [DEST_W-1:0] h_id;
    logic [HEAD_DIM-1:0] h_q, h_ks, h_kn, h_ksou, h_ke, h_kw;
    logic [4:0] h_mask;
    logic h_last;
    logic [2:0] probe_dir_q, probes_left_q;

    // SGT
    logic anchor_valid, anchor_ready;
    logic probe_valid, probe_ready;
    logic [2:0] probe_dir;
    logic [HEAD_DIM-1:0] probe_k;
    logic probe_last;
    logic sgt_cmd_valid, sgt_cmd_ready;
    logic [TAG_W-1:0] sgt_tag;
    logic [15:0] sgt_seq;
    logic [GATE_W-1:0] sgt_gate;
    logic [LANE_ID_W-1:0] sgt_lane;
    logic [DEST_W-1:0] sgt_dest;
    logic [MULT_W-1:0] sgt_mult;
    logic sgt_tf, sgt_tl, sgt_hl;
    logic sgt_done_valid, sgt_done_ready;
    logic [TAG_W-1:0] sgt_done_tag;
    logic sgt_err;
    logic [31:0] pe, pt, pn;
    logic [15:0] sgt_tare_issues, sgt_tare_zero;
    logic [15:0] sgt_tare_sparse, sgt_tare_dense;

    local5_score_gate_term_top #(
        .HEAD_DIM(HEAD_DIM), .GATE_W(GATE_W), .TAG_W(TAG_W), .DEST_W(DEST_W),
        .MULT_W(MULT_W), .LANE_ID_W(LANE_ID_W), .USE_TARE(USE_TARE)
    ) u_sgt (
        .clk_core(clk_core), .rst_core(rst_core),
        .anchor_valid(anchor_valid), .anchor_ready(anchor_ready),
        .anchor_tag(h_tag), .anchor_dest_id(h_id),
        .anchor_q_bits(h_q), .anchor_k_bits(h_ks), .anchor_valid_mask(h_mask),
        .probe_valid(probe_valid), .probe_ready(probe_ready),
        .probe_dir(probe_dir), .probe_k_bits(probe_k), .probe_last(probe_last),
        .cmd_valid(sgt_cmd_valid), .cmd_ready(sgt_cmd_ready),
        .cmd_group_tag(sgt_tag), .cmd_sequence(sgt_seq),
        .cmd_gate_code(sgt_gate), .cmd_lane_id(sgt_lane),
        .cmd_destination_token(sgt_dest), .cmd_multiplicity(sgt_mult),
        .cmd_term_first(sgt_tf), .cmd_term_last(sgt_tl), .cmd_head_last(sgt_hl),
        .stencil_done_valid(sgt_done_valid), .stencil_done_ready(sgt_done_ready),
        .stencil_done_tag(sgt_done_tag), .protocol_error(sgt_err),
        .perf_edges(pe), .perf_terms(pt), .perf_naive_products(pn),
        .perf_tare_issues(sgt_tare_issues),
        .perf_tare_zero(sgt_tare_zero),
        .perf_tare_sparse(sgt_tare_sparse),
        .perf_tare_dense(sgt_tare_dense)
    );

    logic b_valid, b_ready;
    logic [TAG_W-1:0] b_tag;
    logic [15:0] b_seq;
    logic [GATE_W-1:0] b_gate;
    logic [LANE_ID_W-1:0] b_lane;
    logic [DEST_W-1:0] b_dest;
    logic [MULT_W-1:0] b_mult;
    logic [12:0] b_issue;
    logic b_first, b_last, b_hl;
    logic b_err;
    logic [31:0] b_cc, b_ce;

    local5_dctf_multiset_bridge #(
        .GATE_W(GATE_W), .TAG_W(TAG_W), .DEST_W(DEST_W), .MULT_W(MULT_W),
        .LANE_ID_W(LANE_ID_W), .EXPLODE(1'b0)
    ) u_bridge (
        .clk_core(clk_core), .rst_core(rst_core),
        .term_valid(sgt_cmd_valid), .term_ready(sgt_cmd_ready),
        .term_tag(sgt_tag), .term_dest_id(sgt_dest), .term_lane(sgt_lane),
        .term_gate(sgt_gate), .term_mult(sgt_mult), .term_last(sgt_tl),
        .term_head_last(sgt_tl && h_last),
        .cmd_valid(b_valid), .cmd_ready(b_ready),
        .cmd_group_tag(b_tag), .cmd_sequence(b_seq),
        .cmd_gate_code(b_gate), .cmd_lane_id(b_lane),
        .cmd_destination_token(b_dest), .cmd_multiplicity(b_mult),
        .cmd_term_issue_seq(b_issue),
        .cmd_term_first(b_first), .cmd_term_last(b_last), .cmd_head_last(b_hl),
        .protocol_error(b_err), .count_cmds(b_cc), .count_exploded(b_ce)
    );

    logic proj_start, proj_busy, proj_done, proj_err;
    logic [31:0] proj_cmds, proj_prods, proj_conflicts;
    logic window_last_cmd;
    assign window_last_cmd =
        (b_valid && b_hl)
        || ((state_q == ST_WAIT_PROJ) && !b_valid
            && !sgt_cmd_valid && !proj_done);

    local5_multibank_projection_top #(
        .HEAD_DIM(HEAD_DIM), .OUT_DIM(OUT_DIM), .MAX_DEST(MAX_DEST),
        .NUM_BANKS(NUM_BANKS), .GATE_W(GATE_W), .TAG_W(TAG_W), .DEST_W(DEST_W),
        .MULT_W(MULT_W), .LANE_ID_W(LANE_ID_W)
    ) u_proj (
        .clk_core(clk_core), .rst_core(rst_core),
        .w_load_valid(w_load_valid), .w_load_ready(w_load_ready),
        .w_load_lane(w_load_lane), .w_load_out(w_load_out),
        .w_load_data(w_load_data), .w_load_last(w_load_last),
        .run_start(proj_start), .run_busy(proj_busy), .run_done(proj_done),
        .cmd_valid(b_valid), .cmd_ready(b_ready),
        .cmd_group_tag(b_tag), .cmd_gate_code(b_gate), .cmd_lane_id(b_lane),
        .cmd_destination_token(b_dest), .cmd_multiplicity(b_mult),
        .cmd_head_last(b_hl), .cmd_window_last(window_last_cmd),
        .acc_read_valid(acc_read_valid), .acc_read_ready(acc_read_ready),
        .acc_read_dest(acc_read_dest), .acc_read_out(acc_read_out),
        .acc_data_valid(acc_data_valid), .acc_data(acc_data),
        .protocol_error(proj_err),
        .perf_cmd_count(proj_cmds), .perf_product_count(proj_prods),
        .perf_bank_conflict_count(proj_conflicts)
    );

    assign run_busy = (state_q != ST_IDLE && state_q != ST_FINISH);
    assign run_done = (state_q == ST_FINISH);
    assign protocol_error = lbuf_err | fetch_err | sgt_err | b_err | proj_err | glue_err_q;
    assign perf_dest_count = dest_count_q;
    assign perf_cmd_count = proj_cmds;
    assign perf_cycle_count = cycle_q;
    assign perf_bank_conflict_count = proj_conflicts;

    // stencil accept into held regs
    assign st_ready = (state_q == ST_WAIT_ST);

    always_comb begin
        probe_k = '0;
        probe_dir = probe_dir_q;
        unique case (probe_dir_q)
            3'd1: probe_k = h_kn;
            3'd2: probe_k = h_ksou;
            3'd3: probe_k = h_ke;
            3'd4: probe_k = h_kw;
            default: probe_k = '0;
        endcase
    end

    function automatic logic [2:0] count_probes(input logic [4:0] m);
        logic [2:0] c; c = 3'd0;
        for (int d = 1; d < 5; d++) c = c + {2'b0, m[d]};
        return c;
    endfunction

    function automatic logic [2:0] first_probe(input logic [4:0] m);
        logic [2:0] dsel; logic found;
        dsel = 3'd1; found = 1'b0;
        for (int d = 1; d < 5; d++)
            if (!found && m[d]) begin dsel = 3'(d); found = 1'b1; end
        return dsel;
    endfunction

    function automatic logic [2:0] next_probe(input logic [4:0] m, input logic [2:0] cur);
        logic [2:0] dsel; logic found;
        dsel = cur; found = 1'b0;
        for (int d = 1; d < 5; d++)
            if (!found && 3'(d) > cur && m[d]) begin dsel = 3'(d); found = 1'b1; end
        return dsel;
    endfunction

    assign anchor_valid = (state_q == ST_ANCHOR);
    assign probe_valid = (state_q == ST_PROBE);
    assign probe_last = (probes_left_q == 3'd1);
    assign sgt_done_ready = (state_q == ST_WAIT_SGT);

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= ST_IDLE;
            x_q <= '0;
            dest_count_q <= '0;
            cycle_q <= '0;
            glue_err_q <= 1'b0;
            proj_start <= 1'b0;
            fetch_req_valid <= 1'b0;
            fetch_tag <= '0; fetch_dest <= '0; fetch_x <= '0; fetch_last <= 1'b0;
            h_tag <= '0; h_id <= '0;
            h_q <= '0; h_ks <= '0; h_kn <= '0; h_ksou <= '0; h_ke <= '0; h_kw <= '0;
            h_mask <= '0; h_last <= 1'b0;
            probe_dir_q <= '0; probes_left_q <= '0;
        end else begin
            proj_start <= 1'b0;
            fetch_req_valid <= 1'b0;

            if (state_q != ST_IDLE && state_q != ST_FINISH)
                cycle_q <= cycle_q + 1'b1;
            if (lbuf_err || fetch_err || sgt_err || b_err || proj_err)
                glue_err_q <= 1'b1;

            unique case (state_q)
                ST_IDLE: begin
                    if (run_start) begin
                        if (lbuf_filled < 2'd3) glue_err_q <= 1'b1;
                        cycle_q <= '0;
                        dest_count_q <= '0;
                        x_q <= '0;
                        proj_start <= 1'b1;
                        state_q <= ST_ARM;
                    end
                end
                ST_ARM: state_q <= ST_ISSUE_X;
                ST_ISSUE_X: begin
                    fetch_req_valid <= 1'b1;
                    fetch_tag <= TAG_W'(x_q);
                    fetch_dest <= DEST_W'(x_q); // dest id = x
                    fetch_x <= x_q;
                    fetch_last <= (32'(x_q) == ROW_TOKENS - 1);
                    if (fetch_req_ready) state_q <= ST_WAIT_ST;
                end
                ST_WAIT_ST: begin
                    if (st_valid) begin
                        if (!st_mask[0]) begin
                            // Padding/invalid centers do not create a fake
                            // self edge. They retire without score or terms.
                            if (st_last)
                                state_q <= ST_WAIT_PROJ;
                            else begin
                                x_q <= x_q + TOKEN_W'(1);
                                state_q <= ST_ISSUE_X;
                            end
                        end else begin
                            h_tag <= st_tag;
                            h_id <= st_dest;
                            h_q <= st_q;
                            h_ks <= st_ks;
                            h_kn <= st_kn;
                            h_ksou <= st_ksou;
                            h_ke <= st_ke;
                            h_kw <= st_kw;
                            h_mask <= st_mask;
                            h_last <= st_last;
                            dest_count_q <= dest_count_q + 1'b1;
                            state_q <= ST_ANCHOR;
                        end
                    end
                end
                ST_ANCHOR: begin
                    if (anchor_ready) begin
                        probes_left_q <= count_probes(h_mask);
                        if (count_probes(h_mask) == 0) state_q <= ST_WAIT_SGT;
                        else begin
                            probe_dir_q <= first_probe(h_mask);
                            state_q <= ST_PROBE;
                        end
                    end
                end
                ST_PROBE: begin
                    if (probe_ready) begin
                        if (probes_left_q == 3'd1) state_q <= ST_WAIT_SGT;
                        else begin
                            probes_left_q <= probes_left_q - 3'd1;
                            probe_dir_q <= next_probe(h_mask, probe_dir_q);
                        end
                    end
                end
                ST_WAIT_SGT: begin
                    if (sgt_done_valid) begin
                        if (h_last) state_q <= ST_WAIT_PROJ;
                        else begin
                            x_q <= x_q + TOKEN_W'(1);
                            state_q <= ST_ISSUE_X;
                        end
                    end
                end
                ST_WAIT_PROJ: begin
                    if (proj_done)
                        state_q <= ST_FINISH;
                end
                ST_FINISH: begin
                    // re-run after TB pushes next 3 rows
                    if (run_start) begin
                        cycle_q <= '0;
                        dest_count_q <= '0;
                        x_q <= '0;
                        glue_err_q <= 1'b0;
                        if (lbuf_filled < 2'd3) glue_err_q <= 1'b1;
                        proj_start <= 1'b1;
                        state_q <= ST_ARM;
                    end
                end
                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
