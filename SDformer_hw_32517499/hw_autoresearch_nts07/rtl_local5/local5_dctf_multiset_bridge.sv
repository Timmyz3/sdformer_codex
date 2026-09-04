`timescale 1ns/1ps
`default_nettype none

// Bridge MFEP multiset terms → DCTF-compatible unit commands.
// Strategy A (default EXPLODE=1): emit multiplicity copies of set-style cmds
//   so an unmodified set-multicast bank executor remains bit-exact.
// Strategy B (EXPLODE=0): single cmd with multiplicity sideband for
//   Local5 banklocal / future multiset-aware executor.
// This file only lives under rtl_local5; it does not edit rtl_hitflow.
module local5_dctf_multiset_bridge #(
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter int CMD_SEQ_W = 16,
    parameter int ISSUE_W   = 13,
    parameter bit EXPLODE   = 1'b1
) (
    input  logic                  clk_core,
    input  logic                  rst_core,

    input  logic                  term_valid,
    output logic                  term_ready,
    input  logic [TAG_W-1:0]      term_tag,
    input  logic [DEST_W-1:0]     term_dest_id,
    input  logic [LANE_ID_W-1:0]  term_lane,
    input  logic [GATE_W-1:0]     term_gate,
    input  logic [MULT_W-1:0]     term_mult,
    input  logic                  term_last,      // last MFEP item of destination
    input  logic                  term_head_last, // last MFEP item of head/window

    // DCTF-style unit command (set fabric compatible when EXPLODE=1)
    output logic                  cmd_valid,
    input  logic                  cmd_ready,
    output logic [TAG_W-1:0]      cmd_group_tag,
    output logic [CMD_SEQ_W-1:0]  cmd_sequence,
    output logic [GATE_W-1:0]     cmd_gate_code,
    output logic [LANE_ID_W-1:0]  cmd_lane_id,
    output logic [DEST_W-1:0]     cmd_destination_token,
    output logic [MULT_W-1:0]     cmd_multiplicity, // 1 if EXPLODE else term_mult
    output logic [ISSUE_W-1:0]    cmd_term_issue_seq,
    output logic                  cmd_term_first,
    output logic                  cmd_term_last,
    output logic                  cmd_head_last,

    output logic                  protocol_error,
    output logic [31:0]           count_cmds,
    output logic [31:0]           count_exploded
);

    logic held_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [GATE_W-1:0] gate_q;
    logic [MULT_W-1:0] mult_q;
    logic [MULT_W-1:0] remain_q;
    logic last_q;
    logic head_last_q;
    logic first_beat_q;
    logic [CMD_SEQ_W-1:0] seq_q;
    logic [ISSUE_W-1:0] issue_q;
    logic protocol_error_q;
    logic [31:0] count_cmds_q;
    logic [31:0] count_exploded_q;

    assign protocol_error = protocol_error_q;
    assign count_cmds = count_cmds_q;
    assign count_exploded = count_exploded_q;

    assign cmd_valid = held_q;
    assign cmd_group_tag = tag_q;
    assign cmd_sequence = seq_q;
    assign cmd_gate_code = gate_q;
    assign cmd_lane_id = lane_q;
    assign cmd_destination_token = dest_q;
    assign cmd_multiplicity = EXPLODE ? MULT_W'(1) : mult_q;
    assign cmd_term_issue_seq = issue_q;
    assign cmd_term_first = first_beat_q;
    // Every input MFEP item is one DCTF term because gate/lane/multiplicity
    // identify one reusable product. EXPLODE only changes its destination-beat
    // count; it must not merge adjacent MFEP items into one executor term.
    assign cmd_term_last = EXPLODE ? (remain_q == MULT_W'(1)) : 1'b1;
    assign cmd_head_last = head_last_q && cmd_term_last;

    generate
        if (EXPLODE) begin : g_explode
            assign term_ready = !held_q || (cmd_ready && remain_q == MULT_W'(1));
        end else begin : g_sideband
            assign term_ready = !held_q || cmd_ready;
        end
    endgenerate

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            held_q <= 1'b0;
            tag_q <= '0;
            dest_q <= '0;
            lane_q <= '0;
            gate_q <= '0;
            mult_q <= '0;
            remain_q <= '0;
            last_q <= 1'b0;
            head_last_q <= 1'b0;
            first_beat_q <= 1'b1;
            seq_q <= '0;
            issue_q <= '0;
            protocol_error_q <= 1'b0;
            count_cmds_q <= '0;
            count_exploded_q <= '0;
        end else begin
            if (held_q && cmd_ready) begin
                count_cmds_q <= count_cmds_q + 1'b1;
                seq_q <= seq_q + 1'b1;
                first_beat_q <= 1'b0;
                if (EXPLODE) begin
                    if (remain_q > MULT_W'(1)) begin
                        remain_q <= remain_q - MULT_W'(1);
                    end else begin
                        held_q <= 1'b0;
                        first_beat_q <= 1'b1;
                        issue_q <= issue_q + 1'b1;
                    end
                end else begin
                    held_q <= 1'b0;
                    first_beat_q <= 1'b1;
                    issue_q <= issue_q + 1'b1;
                end
            end

            if (term_valid && term_ready) begin
                if (term_mult == 0 || term_mult > MULT_W'(5) ||
                    (term_head_last && !term_last)) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    held_q <= 1'b1;
                    tag_q <= term_tag;
                    dest_q <= term_dest_id;
                    lane_q <= term_lane;
                    gate_q <= term_gate;
                    mult_q <= term_mult;
                    remain_q <= term_mult;
                    last_q <= term_last;
                    head_last_q <= term_head_last;
                    first_beat_q <= 1'b1;
                    if (EXPLODE && term_mult > MULT_W'(1)) begin
                        count_exploded_q <=
                            count_exploded_q + 32'(term_mult) - 32'd1;
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
