`timescale 1ns/1ps
`default_nettype none

// Stencil-Time Tile (STT) descriptor — Local5 counterpart of Motion TAB.
// Packs boundary / valid-direction / occupancy sideband for one destination.
// Lifecycle is issue→score→term commit markers only (not full SCS/NMF join yet).
module local5_stt_descriptor #(
    parameter int TAG_W  = 16,
    parameter int DEST_W = 8,
    parameter int N_CAND = 5
) (
    input  logic                 clk_core,
    input  logic                 rst_core,

    input  logic                 issue_valid,
    output logic                 issue_ready,
    input  logic [TAG_W-1:0]     issue_tag,
    input  logic [DEST_W-1:0]    issue_dest_id,
    input  logic [N_CAND-1:0]    issue_valid_mask, // self,N,S,E,W
    input  logic                 issue_boundary,   // corner/edge tile
    input  logic [2:0]           issue_delta_class, // 0 exact-heavy .. 7 dense

    output logic                 live_valid,
    output logic [TAG_W-1:0]     live_tag,
    output logic [DEST_W-1:0]    live_dest_id,
    output logic [N_CAND-1:0]    live_valid_mask,
    output logic                 live_boundary,
    output logic [2:0]           live_delta_class,
    output logic [2:0]           live_degree,
    output logic [1:0]           live_phase, // 0 idle-issued,1 scoring,2 terming,3 done

    // phase advances from downstream
    input  logic                 mark_score_start,
    input  logic                 mark_term_start,
    input  logic                 mark_commit,
    input  logic                 retire_ready,

    output logic                 protocol_error
);

    typedef enum logic [1:0] {
        PH_IDLE  = 2'd0,
        PH_SCORE = 2'd1,
        PH_TERM  = 2'd2,
        PH_DONE  = 2'd3
    } phase_t;

    phase_t phase_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [N_CAND-1:0] mask_q;
    logic boundary_q;
    logic [2:0] delta_q;
    logic [2:0] degree_q;
    logic protocol_error_q;

    function automatic logic [2:0] pop5(input logic [N_CAND-1:0] m);
        logic [2:0] c;
        c = 3'd0;
        for (int i = 0; i < N_CAND; i = i + 1) c = c + {2'b0, m[i]};
        return c;
    endfunction

    assign issue_ready = (phase_q == PH_IDLE);
    assign live_valid = (phase_q != PH_IDLE);
    assign live_tag = tag_q;
    assign live_dest_id = dest_q;
    assign live_valid_mask = mask_q;
    assign live_boundary = boundary_q;
    assign live_delta_class = delta_q;
    assign live_degree = degree_q;
    assign live_phase = phase_q;
    assign protocol_error = protocol_error_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            phase_q <= PH_IDLE;
            tag_q <= '0;
            dest_q <= '0;
            mask_q <= '0;
            boundary_q <= 1'b0;
            delta_q <= '0;
            degree_q <= '0;
            protocol_error_q <= 1'b0;
        end else begin
            unique case (phase_q)
                PH_IDLE: begin
                    if (issue_valid) begin
                        tag_q <= issue_tag;
                        dest_q <= issue_dest_id;
                        mask_q <= issue_valid_mask;
                        boundary_q <= issue_boundary;
                        delta_q <= issue_delta_class;
                        degree_q <= pop5(issue_valid_mask);
                        phase_q <= PH_SCORE;
                    end
                end
                PH_SCORE: begin
                    if (mark_term_start) phase_q <= PH_TERM;
                    else if (mark_score_start) phase_q <= PH_SCORE;
                    if (mark_commit) protocol_error_q <= 1'b1; // too early
                end
                PH_TERM: begin
                    if (mark_commit) phase_q <= PH_DONE;
                end
                PH_DONE: begin
                    if (retire_ready) phase_q <= PH_IDLE;
                end
                default: phase_q <= PH_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
