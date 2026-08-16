`default_nettype none

module qfit_vl_gs_ttb_motion_dvco_assertions #(
    parameter int SLOTS = 4,
    parameter int GATE_W = 9,
    parameter int PAYLOAD_W = 32,
    parameter int SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
    input logic clk_core,
    input logic rst_core,
    input logic build_start_valid,
    input logic build_start_ready,
    input logic build_update_valid,
    input logic build_update_ready,
    input logic build_commit_valid,
    input logic build_commit_ready,
    input logic body_valid,
    input logic body_ready,
    input logic body_last,
    input logic out_valid,
    input logic out_ready,
    input logic [GATE_W-1:0] out_gate,
    input logic [PAYLOAD_W-1:0] out_payload,
    input logic out_last
);
    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            if ($past(!rst_core && out_valid && !out_ready))
                assert(out_valid && $stable({out_gate, out_payload, out_last}));
            assert(!(build_update_valid && build_update_ready
                && build_commit_valid && build_commit_ready));
            if (body_valid && body_ready && body_last)
                assert(out_ready || !out_valid);
            if (build_start_valid && !build_start_ready)
                assert(!build_update_ready && !build_commit_ready);
        end
    end
endmodule

module qfit_vl_gs_ttb_abic_decoder_assertions #(
    parameter int SETS = 32,
    parameter int SLOTS = 6,
    parameter int GATE_W = 9,
    parameter int PAYLOAD_W = 32,
    parameter int SET_W = (SETS <= 1) ? 1 : $clog2(SETS),
    parameter int SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
    input logic clk_core,
    input logic rst_core,
    input logic lifecycle_active,
    input logic update_valid,
    input logic update_ready,
    input logic [SET_W-1:0] update_set,
    input logic [SLOT_W-1:0] update_slot,
    input logic primary_valid,
    input logic primary_ready,
    input logic [SET_W-1:0] primary_set,
    input logic [SLOT_W-1:0] primary_slot,
    input logic primary_use_exception,
    input logic exception_valid,
    input logic exception_ready,
    input logic out_valid,
    input logic out_ready,
    input logic [GATE_W-1:0] out_gate,
    input logic [PAYLOAD_W-1:0] out_payload,
    input logic out_last
);
    always_ff @(posedge clk_core) begin
        if (!rst_core) begin
            if ($past(!rst_core && out_valid && !out_ready))
                assert(out_valid && $stable({out_gate, out_payload, out_last}));
            if (primary_valid && primary_ready && !primary_use_exception
                && update_valid && update_ready
                && update_set == primary_set && update_slot == primary_slot)
                assert(lifecycle_active);
            if (exception_valid && exception_ready)
                assert(lifecycle_active);
        end
    end
endmodule

bind qfit_vl_gs_ttb_motion_dvco
    qfit_vl_gs_ttb_motion_dvco_assertions #(
        .SLOTS(SLOTS), .GATE_W(GATE_W), .PAYLOAD_W(PAYLOAD_W),
        .SLOT_W(SLOT_W)
    ) u_qfit_vl_gs_ttb_motion_dvco_assertions (.*);

bind qfit_vl_gs_ttb_abic_decoder
    qfit_vl_gs_ttb_abic_decoder_assertions #(
        .SETS(SETS), .SLOTS(SLOTS), .GATE_W(GATE_W),
        .PAYLOAD_W(PAYLOAD_W), .SET_W(SET_W), .SLOT_W(SLOT_W)
    ) u_qfit_vl_gs_ttb_abic_decoder_assertions (.*);

`default_nettype wire
