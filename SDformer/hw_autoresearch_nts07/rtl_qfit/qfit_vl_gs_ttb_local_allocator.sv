`timescale 1ns/1ps
`default_nettype none

// Local5前端：每lane first-bind、无替换；满表后raw exception精确旁路。
module qfit_vl_gs_ttb_local_allocator #(
    parameter int SETS = 32,
    parameter int SLOTS = 6,
    parameter int GATE_W = 9,
    parameter int PAYLOAD_W = 32,
    parameter int SET_W = (SETS <= 1) ? 1 : $clog2(SETS),
    parameter int SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
    input  logic clk_core,
    input  logic rst_core,
    input  logic lifecycle_start,
    input  logic lifecycle_end,
    output logic lifecycle_active,
    output logic lifecycle_done,

    input  logic in_valid,
    output logic in_ready,
    input  logic [SET_W-1:0] in_set,
    input  logic [GATE_W-1:0] in_gate,
    input  logic [PAYLOAD_W-1:0] in_payload,
    input  logic in_last,

    output logic update_valid,
    input  logic update_ready,
    output logic [SET_W-1:0] update_set,
    output logic [SLOT_W-1:0] update_slot,
    output logic [GATE_W-1:0] update_gate,

    output logic primary_valid,
    input  logic primary_ready,
    output logic [SET_W-1:0] primary_set,
    output logic [SLOT_W-1:0] primary_slot,
    output logic primary_use_exception,
    output logic [PAYLOAD_W-1:0] primary_payload,
    output logic primary_last,
    output logic exception_valid,
    input  logic exception_ready,
    output logic [GATE_W-1:0] exception_gate,

    output logic protocol_error,
    output logic [31:0] perf_fills,
    output logic [31:0] perf_hits,
    output logic [31:0] perf_bypasses
);
    typedef enum logic [1:0] {
        S_IDLE,
        S_UPDATE,
        S_PRIMARY,
        S_BYPASS
    } state_t;

    logic slot_valid_q [0:SETS-1][0:SLOTS-1];
    logic [GATE_W-1:0] slot_gate_q [0:SETS-1][0:SLOTS-1];
    state_t state_q;
    logic [SET_W-1:0] hold_set_q;
    logic [SLOT_W-1:0] hold_slot_q;
    logic [GATE_W-1:0] hold_gate_q;
    logic [PAYLOAD_W-1:0] hold_payload_q;
    logic hold_last_q;
    logic hold_is_fill_q;
    logic bypass_primary_pending_q;
    logic bypass_exception_pending_q;
    logic protocol_error_q;
    logic [31:0] perf_fills_q;
    logic [31:0] perf_hits_q;
    logic [31:0] perf_bypasses_q;
    logic active_q;

    logic lookup_hit;
    logic free_found;
    logic [SLOT_W-1:0] lookup_slot;
    logic [SLOT_W-1:0] free_slot;

    always_comb begin
        lookup_hit = 1'b0;
        free_found = 1'b0;
        lookup_slot = '0;
        free_slot = '0;
        if (32'(in_set) < SETS) begin
            for (integer slot = 0; slot < SLOTS; slot++) begin
                if (
                    !lookup_hit
                    && slot_valid_q[in_set][slot]
                    && slot_gate_q[in_set][slot] == in_gate
                ) begin
                    lookup_hit = 1'b1;
                    lookup_slot = SLOT_W'(slot);
                end
                if (!free_found && !slot_valid_q[in_set][slot]) begin
                    free_found = 1'b1;
                    free_slot = SLOT_W'(slot);
                end
            end
        end
    end

    assign in_ready =
        active_q
        && state_q == S_IDLE
        && !lifecycle_start
        && !lifecycle_end;
    assign update_valid = state_q == S_UPDATE;
    assign update_set = hold_set_q;
    assign update_slot = hold_slot_q;
    assign update_gate = hold_gate_q;

    assign primary_valid = state_q == S_PRIMARY
        || (state_q == S_BYPASS && bypass_primary_pending_q);
    assign primary_set = hold_set_q;
    assign primary_slot = hold_slot_q;
    assign primary_use_exception = state_q == S_BYPASS;
    assign primary_payload = hold_payload_q;
    assign primary_last = hold_last_q;
    assign exception_valid =
        state_q == S_BYPASS && bypass_exception_pending_q;
    assign exception_gate = hold_gate_q;

    assign protocol_error = protocol_error_q;
    assign perf_fills = perf_fills_q;
    assign perf_hits = perf_hits_q;
    assign perf_bypasses = perf_bypasses_q;
    assign lifecycle_active = active_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            state_q <= S_IDLE;
            active_q <= 1'b0;
            lifecycle_done <= 1'b0;
            hold_set_q <= '0;
            hold_slot_q <= '0;
            hold_gate_q <= '0;
            hold_payload_q <= '0;
            hold_last_q <= 1'b0;
            hold_is_fill_q <= 1'b0;
            bypass_primary_pending_q <= 1'b0;
            bypass_exception_pending_q <= 1'b0;
            protocol_error_q <= 1'b0;
            perf_fills_q <= '0;
            perf_hits_q <= '0;
            perf_bypasses_q <= '0;
            for (integer set_id = 0; set_id < SETS; set_id++)
                for (integer slot = 0; slot < SLOTS; slot++) begin
                    slot_valid_q[set_id][slot] <= 1'b0;
                    slot_gate_q[set_id][slot] <= '0;
                end
        end else begin
            lifecycle_done <= 1'b0;
            if (lifecycle_start) begin
                if (active_q) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    state_q <= S_IDLE;
                    bypass_primary_pending_q <= 1'b0;
                    bypass_exception_pending_q <= 1'b0;
                    protocol_error_q <= 1'b0;
                    perf_fills_q <= '0;
                    perf_hits_q <= '0;
                    perf_bypasses_q <= '0;
                    for (integer set_id = 0; set_id < SETS; set_id++)
                        for (integer slot = 0; slot < SLOTS; slot++) begin
                            slot_valid_q[set_id][slot] <= 1'b0;
                            slot_gate_q[set_id][slot] <= '0;
                        end
                end
            end else begin
                case (state_q)
                    S_IDLE: begin
                        if (in_valid) begin
                            if (
                                32'(in_set) >= SETS
                                || in_gate == '0
                            ) begin
                                protocol_error_q <= 1'b1;
                            end else begin
                                hold_set_q <= in_set;
                                hold_gate_q <= in_gate;
                                hold_payload_q <= in_payload;
                                hold_last_q <= in_last;
                                if (lookup_hit) begin
                                    hold_slot_q <= lookup_slot;
                                    hold_is_fill_q <= 1'b0;
                                    state_q <= S_PRIMARY;
                                end else if (free_found) begin
                                    hold_slot_q <= free_slot;
                                    hold_is_fill_q <= 1'b1;
                                    state_q <= S_UPDATE;
                                end else begin
                                    hold_slot_q <= '0;
                                    hold_is_fill_q <= 1'b0;
                                    bypass_primary_pending_q <= 1'b1;
                                    bypass_exception_pending_q <= 1'b1;
                                    state_q <= S_BYPASS;
                                end
                            end
                        end
                    end
                    S_UPDATE: begin
                        if (update_valid && update_ready) begin
                            slot_valid_q[hold_set_q][hold_slot_q] <= 1'b1;
                            slot_gate_q[hold_set_q][hold_slot_q] <= hold_gate_q;
                            perf_fills_q <= perf_fills_q + 1'b1;
                            state_q <= S_PRIMARY;
                        end
                    end
                    S_PRIMARY: begin
                        if (primary_valid && primary_ready) begin
                            if (!hold_is_fill_q)
                                perf_hits_q <= perf_hits_q + 1'b1;
                            state_q <= S_IDLE;
                        end
                    end
                    default: begin
                        if (primary_valid && primary_ready)
                            bypass_primary_pending_q <= 1'b0;
                        if (exception_valid && exception_ready)
                            bypass_exception_pending_q <= 1'b0;
                        if (
                            (!bypass_primary_pending_q || primary_ready)
                            && (!bypass_exception_pending_q || exception_ready)
                        ) begin
                            perf_bypasses_q <= perf_bypasses_q + 1'b1;
                            state_q <= S_IDLE;
                        end
                    end
                endcase
            end
            if (lifecycle_end) begin
                if (!active_q || state_q != S_IDLE) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b0;
                    lifecycle_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (SETS < 1 || SLOTS < 2 || GATE_W < 1 || PAYLOAD_W < 1)
            $fatal(1, "VL-GS-TTB allocator参数非法");
    end
endmodule

`default_nettype wire
