`timescale 1ns/1ps
`default_nettype none

// Expand MFEP multiset terms into DCTF-compatible projection commands.
// Each term {lane, gate, multiplicity, dest} becomes one command with a
// multiplicity sideband (x1..x5). Downstream bank executor can scale the
// product as mult * gate * W[lane].
// Optional explode_mode emits multiplicity single-dest set events for
// backends that only understand set multicast.
module local5_mfep_dctf_cmd_adapter #(
    parameter int GATE_W    = 9,
    parameter int TAG_W     = 16,
    parameter int DEST_W    = 8,
    parameter int MULT_W    = 3,
    parameter int LANE_ID_W = 5,
    parameter int CMD_SEQ_W = 16,
    parameter bit EXPLODE_MODE = 1'b0
) (
    input  logic                  clk_core,
    input  logic                  rst_core,

    input  logic                  term_valid,
    output logic                  term_ready,
    input  logic [TAG_W-1:0]      term_tag,
    input  logic [DEST_W-1:0]     term_dest_id,
    input  logic [LANE_ID_W-1:0]  term_lane,
    input  logic [GATE_W-1:0]     term_gate_q17,
    input  logic [MULT_W-1:0]     term_multiplicity,
    input  logic                  term_last,

    output logic                  cmd_valid,
    input  logic                  cmd_ready,
    output logic [TAG_W-1:0]      cmd_group_tag,
    output logic [CMD_SEQ_W-1:0]  cmd_sequence,
    output logic [GATE_W-1:0]     cmd_gate_code,
    output logic [LANE_ID_W-1:0]  cmd_lane_id,
    output logic [DEST_W-1:0]     cmd_destination_token,
    output logic [MULT_W-1:0]     cmd_multiplicity,
    output logic                  cmd_term_first,
    output logic                  cmd_term_last,
    output logic                  cmd_head_last,

    output logic                  protocol_error
);

    typedef enum logic [0:0] {
        ST_PASS = 1'b0,
        ST_EXPL = 1'b1
    } state_t;

    state_t state_q;
    logic [TAG_W-1:0] tag_q;
    logic [DEST_W-1:0] dest_q;
    logic [LANE_ID_W-1:0] lane_q;
    logic [GATE_W-1:0] gate_q;
    logic [MULT_W-1:0] mult_q;
    logic [MULT_W-1:0] remain_q;
    logic last_q;
    logic first_q;
    logic [CMD_SEQ_W-1:0] seq_q;
    logic protocol_error_q;
    logic held_q;

    assign protocol_error = protocol_error_q;

    generate
        if (!EXPLODE_MODE) begin : g_pass
            assign term_ready = !held_q || cmd_ready;
            assign cmd_valid = held_q;
            assign cmd_group_tag = tag_q;
            assign cmd_sequence = seq_q;
            assign cmd_gate_code = gate_q;
            assign cmd_lane_id = lane_q;
            assign cmd_destination_token = dest_q;
            assign cmd_multiplicity = mult_q;
            assign cmd_term_first = first_q;
            assign cmd_term_last = last_q;
            assign cmd_head_last = last_q;

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    held_q <= 1'b0;
                    tag_q <= '0;
                    dest_q <= '0;
                    lane_q <= '0;
                    gate_q <= '0;
                    mult_q <= '0;
                    last_q <= 1'b0;
                    first_q <= 1'b1;
                    seq_q <= '0;
                    protocol_error_q <= 1'b0;
                end else begin
                    if (held_q && cmd_ready) begin
                        held_q <= 1'b0;
                        first_q <= last_q ? 1'b1 : 1'b0;
                        if (term_valid && term_ready) begin
                            // back-to-back accept handled below
                        end
                    end
                    if (term_valid && (!held_q || cmd_ready)) begin
                        if (term_multiplicity == 0 || term_multiplicity > MULT_W'(5)) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            held_q <= 1'b1;
                            tag_q <= term_tag;
                            dest_q <= term_dest_id;
                            lane_q <= term_lane;
                            gate_q <= term_gate_q17;
                            mult_q <= term_multiplicity;
                            last_q <= term_last;
                            seq_q <= seq_q + 1'b1;
                        end
                    end
                end
            end
        end else begin : g_explode
            // Emit multiplicity unit cmds with mult=1
            assign term_ready = (state_q == ST_PASS) && (!held_q || (remain_q == 1 && cmd_ready));
            assign cmd_valid = held_q;
            assign cmd_group_tag = tag_q;
            assign cmd_sequence = seq_q;
            assign cmd_gate_code = gate_q;
            assign cmd_lane_id = lane_q;
            assign cmd_destination_token = dest_q;
            assign cmd_multiplicity = MULT_W'(1);
            assign cmd_term_first = first_q;
            assign cmd_term_last = last_q && (remain_q == 1);
            assign cmd_head_last = cmd_term_last;

            always_ff @(posedge clk_core) begin
                if (rst_core) begin
                    state_q <= ST_PASS;
                    held_q <= 1'b0;
                    remain_q <= '0;
                    tag_q <= '0;
                    dest_q <= '0;
                    lane_q <= '0;
                    gate_q <= '0;
                    mult_q <= '0;
                    last_q <= 1'b0;
                    first_q <= 1'b1;
                    seq_q <= '0;
                    protocol_error_q <= 1'b0;
                end else begin
                    if (held_q && cmd_ready) begin
                        seq_q <= seq_q + 1'b1;
                        first_q <= 1'b0;
                        if (remain_q > 1) begin
                            remain_q <= remain_q - 1'b1;
                        end else begin
                            held_q <= 1'b0;
                            first_q <= last_q ? 1'b1 : 1'b0;
                            state_q <= ST_PASS;
                        end
                    end
                    if (term_valid && term_ready) begin
                        if (term_multiplicity == 0 || term_multiplicity > MULT_W'(5)) begin
                            protocol_error_q <= 1'b1;
                        end else begin
                            held_q <= 1'b1;
                            remain_q <= term_multiplicity;
                            tag_q <= term_tag;
                            dest_q <= term_dest_id;
                            lane_q <= term_lane;
                            gate_q <= term_gate_q17;
                            mult_q <= term_multiplicity;
                            last_q <= term_last;
                            first_q <= 1'b1;
                            state_q <= ST_EXPL;
                            seq_q <= seq_q + 1'b1;
                        end
                    end
                end
            end
        end
    endgenerate

endmodule

`default_nettype wire
