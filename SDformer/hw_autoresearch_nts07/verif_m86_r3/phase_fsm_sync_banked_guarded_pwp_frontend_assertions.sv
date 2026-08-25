`timescale 1ns/1ps
`default_nettype none

module phase_fsm_sync_banked_guarded_pwp_frontend_assertions (
    input logic clk_core,
    input logic rst_core,
    input logic payload_load_valid,
    input logic payload_load_ready,
    input logic payload_load_accept,
    input logic phase_load_valid,
    input logic phase_load_ready,
    input logic phase_load_accept,
    input logic descriptor_valid,
    input logic descriptor_ready,
    input logic descriptor_accept,
    input logic output_valid,
    input logic output_ready,
    input logic output_accept,
    input logic protocol_error,
    input logic busy,
    input logic payload_selected,
    input logic phase_selected,
    input logic descriptor_selected,
    input logic [2:0] fsm_state,
    input logic [8:0] accepted_rows,
    input logic [8:0] accepted_descriptors
);
    ap_onehot_selection: assert property (@(posedge clk_core)
        disable iff (rst_core)
        $onehot0({payload_selected, phase_selected, descriptor_selected}));
    ap_accepts_exclusive: assert property (@(posedge clk_core)
        disable iff (rst_core)
        $onehot0({payload_load_accept, phase_load_accept, descriptor_accept}));
    ap_load_state_only_payload: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 0 |-> !phase_load_ready && !descriptor_ready);
    ap_commit_state_only_phase: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 1 |-> !payload_load_ready && !descriptor_ready && busy);
    ap_execute_state_only_descriptor: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 2 |-> !payload_load_ready && !phase_load_ready && busy);
    ap_drain_no_request_ready: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 3 |-> !payload_load_ready && !phase_load_ready
                           && !descriptor_ready && busy);
    ap_load_triple_contention_progress: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 0 && payload_load_valid && phase_load_valid
        && descriptor_valid |-> payload_selected && payload_load_ready);
    ap_commit_triple_contention_progress: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 1 && payload_load_valid && phase_load_valid
        && descriptor_valid |-> phase_selected && phase_load_ready);
    ap_execute_triple_contention_progress: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        fsm_state == 2 && payload_load_valid && phase_load_valid
        && descriptor_valid |-> descriptor_selected);
    ap_output_accept: assert property (@(posedge clk_core)
        disable iff (rst_core) output_accept == (output_valid && output_ready));
    ap_fsm_bounds: assert property (@(posedge clk_core)
        disable iff (rst_core) fsm_state <= 4 && accepted_rows <= 460);

    cp_load_triple: cover property (@(posedge clk_core)
        fsm_state == 0 && payload_load_valid && phase_load_valid
        && descriptor_valid && payload_load_accept);
    cp_commit_triple: cover property (@(posedge clk_core)
        fsm_state == 1 && payload_load_valid && phase_load_valid
        && descriptor_valid && phase_load_accept);
    cp_execute_triple: cover property (@(posedge clk_core)
        fsm_state == 2 && payload_load_valid && phase_load_valid
        && descriptor_valid && descriptor_accept);
    cp_descriptor_128: cover property (@(posedge clk_core)
        fsm_state == 2 && accepted_descriptors == 127 && descriptor_accept);
    cp_return_to_load: cover property (@(posedge clk_core)
        fsm_state == 3 ##1 fsm_state == 0);
endmodule

`default_nettype wire
