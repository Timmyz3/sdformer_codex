`timescale 1ns/1ps
`default_nettype none

// Motion前端：SCS已知完整词表，header先建立slot，body再查表编码。
module qfit_vl_gs_ttb_motion_encoder #(
    parameter int SLOTS = 4,
    parameter int GATE_W = 9,
    parameter int PAYLOAD_W = 32,
    parameter int SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
    input  logic clk_core,
    input  logic rst_core,
    input  logic lifecycle_start,
    input  logic lifecycle_end,
    input  logic lifecycle_raw_mode,
    output logic lifecycle_active,
    output logic lifecycle_done,

    input  logic header_valid,
    output logic header_ready,
    input  logic [SLOT_W-1:0] header_slot,
    input  logic [GATE_W-1:0] header_gate,
    output logic update_valid,
    input  logic update_ready,
    output logic [SLOT_W-1:0] update_slot,
    output logic [GATE_W-1:0] update_gate,

    input  logic in_valid,
    output logic in_ready,
    input  logic [GATE_W-1:0] in_gate,
    input  logic [PAYLOAD_W-1:0] in_payload,
    input  logic in_last,

    output logic primary_valid,
    input  logic primary_ready,
    output logic [SLOT_W-1:0] primary_slot,
    output logic primary_use_exception,
    output logic [PAYLOAD_W-1:0] primary_payload,
    output logic primary_last,
    output logic exception_valid,
    input  logic exception_ready,
    output logic [GATE_W-1:0] exception_gate,
    output logic protocol_error
);
    logic slot_valid_q [0:SLOTS-1];
    logic [GATE_W-1:0] slot_gate_q [0:SLOTS-1];
    logic raw_mode_q;
    logic protocol_error_q;
    logic lookup_hit;
    logic [SLOT_W-1:0] lookup_slot;
    logic use_exception;
    logic primary_pending_q;
    logic exception_pending_q;
    logic [SLOT_W-1:0] hold_slot_q;
    logic [GATE_W-1:0] hold_gate_q;
    logic [PAYLOAD_W-1:0] hold_payload_q;
    logic hold_last_q;
    logic hold_use_exception_q;
    logic active_q;

    always_comb begin
        lookup_hit = 1'b0;
        lookup_slot = '0;
        for (integer slot = 0; slot < SLOTS; slot++) begin
            if (
                !lookup_hit
                && slot_valid_q[slot]
                && slot_gate_q[slot] == in_gate
            ) begin
                lookup_hit = 1'b1;
                lookup_slot = SLOT_W'(slot);
            end
        end
    end

    assign header_ready =
        active_q && !raw_mode_q && !lifecycle_end && update_ready;
    assign update_valid = header_valid;
    assign update_slot = header_slot;
    assign update_gate = header_gate;

    assign use_exception = raw_mode_q;
    assign in_ready =
        active_q
        && !lifecycle_start
        && !lifecycle_end
        && !primary_pending_q
        && !exception_pending_q;
    assign primary_valid = primary_pending_q;
    assign exception_valid = exception_pending_q;
    assign primary_slot = hold_slot_q;
    assign primary_use_exception = hold_use_exception_q;
    assign primary_payload = hold_payload_q;
    assign primary_last = hold_last_q;
    assign exception_gate = hold_gate_q;
    assign protocol_error = protocol_error_q;
    assign lifecycle_active = active_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            raw_mode_q <= 1'b0;
            active_q <= 1'b0;
            lifecycle_done <= 1'b0;
            protocol_error_q <= 1'b0;
            primary_pending_q <= 1'b0;
            exception_pending_q <= 1'b0;
            hold_slot_q <= '0;
            hold_gate_q <= '0;
            hold_payload_q <= '0;
            hold_last_q <= 1'b0;
            hold_use_exception_q <= 1'b0;
            for (integer slot = 0; slot < SLOTS; slot++) begin
                slot_valid_q[slot] <= 1'b0;
                slot_gate_q[slot] <= '0;
            end
        end else begin
            lifecycle_done <= 1'b0;
            if (lifecycle_start) begin
                if (active_q) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    raw_mode_q <= lifecycle_raw_mode;
                    protocol_error_q <= 1'b0;
                    primary_pending_q <= 1'b0;
                    exception_pending_q <= 1'b0;
                    for (integer slot = 0; slot < SLOTS; slot++) begin
                        slot_valid_q[slot] <= 1'b0;
                        slot_gate_q[slot] <= '0;
                    end
                end
            end
            if (header_valid && header_ready) begin
                if (
                    32'(header_slot) >= SLOTS
                    || header_gate == '0
                    || slot_valid_q[header_slot]
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    slot_valid_q[header_slot] <= 1'b1;
                    slot_gate_q[header_slot] <= header_gate;
                end
            end
            if (primary_valid && primary_ready)
                primary_pending_q <= 1'b0;
            if (exception_valid && exception_ready)
                exception_pending_q <= 1'b0;
            if (in_valid && in_ready) begin
                if (
                    in_gate == '0
                    || (!raw_mode_q && !lookup_hit)
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    hold_slot_q <= lookup_slot;
                    hold_gate_q <= in_gate;
                    hold_payload_q <= in_payload;
                    hold_last_q <= in_last;
                    hold_use_exception_q <= use_exception;
                    primary_pending_q <= 1'b1;
                    exception_pending_q <= use_exception;
                end
            end
            if (lifecycle_end) begin
                if (
                    !active_q
                    || primary_pending_q
                    || exception_pending_q
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b0;
                    lifecycle_done <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
