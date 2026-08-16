`timescale 1ns/1ps
`default_nettype none

module local5_banklocal_projection_assertions #(
    parameter int MAX_DEST = 16,
    parameter int GATE_W = 9,
    parameter int TAG_W = 16,
    parameter int DEST_W = 8,
    parameter int MULT_W = 3,
    parameter int LANE_ID_W = 5
) (
    input logic clk_core,
    input logic rst_core,
    input logic [1:0] state_q,
    input logic [((MAX_DEST <= 1) ? 1 : $clog2(MAX_DEST))-1:0] clear_dest_q,
    input logic run_busy,
    input logic run_done,
    input logic acc_read_valid,
    input logic acc_read_ready,
    input logic acc_data_valid,
    input logic cmd_valid,
    input logic cmd_ready,
    input logic [TAG_W-1:0] cmd_group_tag,
    input logic [GATE_W-1:0] cmd_gate_code,
    input logic [LANE_ID_W-1:0] cmd_lane_id,
    input logic [DEST_W-1:0] cmd_destination_token,
    input logic [MULT_W-1:0] cmd_multiplicity,
    input logic cmd_head_last,
    input logic cmd_window_last
);

    localparam logic [1:0] ST_CLEAR = 2'd1;
    localparam logic [1:0] ST_RUN = 2'd2;

    default clocking cb @(posedge clk_core); endclocking

    property p_clear_blocks_commands;
        disable iff (rst_core)
        state_q == ST_CLEAR |-> run_busy && !run_done && !cmd_ready
                              && !acc_read_ready && !acc_data_valid;
    endproperty

    property p_clear_starts_at_zero;
        disable iff (rst_core)
        state_q == ST_CLEAR && $past(state_q) != ST_CLEAR |-> clear_dest_q == '0;
    endproperty

    property p_clear_progresses_one_row;
        disable iff (rst_core)
        state_q == ST_CLEAR && 32'(clear_dest_q) < MAX_DEST-1 |=>
            state_q == ST_CLEAR && clear_dest_q == $past(clear_dest_q) + 1'b1;
    endproperty

    property p_clear_last_enters_run;
        disable iff (rst_core)
        state_q == ST_CLEAR && 32'(clear_dest_q) == MAX_DEST-1 |=>
            state_q == ST_RUN && cmd_ready;
    endproperty

    property p_command_stable_while_stalled;
        disable iff (rst_core)
        cmd_valid && !cmd_ready |=>
            cmd_valid && $stable({cmd_group_tag, cmd_gate_code, cmd_lane_id,
                                  cmd_destination_token, cmd_multiplicity,
                                  cmd_head_last, cmd_window_last});
    endproperty

    property p_reads_only_when_done;
        disable iff (rst_core)
        acc_read_ready |-> run_done;
    endproperty

    property p_invalid_read_has_no_response;
        disable iff (rst_core)
        acc_read_valid && !acc_read_ready |-> !acc_data_valid;
    endproperty

    assert property (p_clear_blocks_commands);
    assert property (p_clear_starts_at_zero);
    assert property (p_clear_progresses_one_row);
    assert property (p_clear_last_enters_run);
    assert property (p_command_stable_while_stalled);
    assert property (p_reads_only_when_done);
    assert property (p_invalid_read_has_no_response);
endmodule

bind local5_banklocal_projection_top local5_banklocal_projection_assertions #(
    .MAX_DEST(MAX_DEST),
    .GATE_W(GATE_W),
    .TAG_W(TAG_W),
    .DEST_W(DEST_W),
    .MULT_W(MULT_W),
    .LANE_ID_W(LANE_ID_W)
) u_local5_banklocal_projection_assertions (
    .clk_core(clk_core),
    .rst_core(rst_core),
    .state_q(state_q),
    .clear_dest_q(clear_dest_q),
    .run_busy(run_busy),
    .run_done(run_done),
    .acc_read_valid(acc_read_valid),
    .acc_read_ready(acc_read_ready),
    .acc_data_valid(acc_data_valid),
    .cmd_valid(cmd_valid),
    .cmd_ready(cmd_ready),
    .cmd_group_tag(cmd_group_tag),
    .cmd_gate_code(cmd_gate_code),
    .cmd_lane_id(cmd_lane_id),
    .cmd_destination_token(cmd_destination_token),
    .cmd_multiplicity(cmd_multiplicity),
    .cmd_head_last(cmd_head_last),
    .cmd_window_last(cmd_window_last)
);

`default_nettype wire
