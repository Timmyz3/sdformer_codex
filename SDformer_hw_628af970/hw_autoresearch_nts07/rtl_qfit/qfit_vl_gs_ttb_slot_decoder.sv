`timescale 1ns/1ps
`default_nettype none

// VL-GS-TTB公共消费端。Motion使用SETS=1，Local5使用SETS=HEAD_DIM。
// update在primary之前建立稳定slot；exception仅为raw-gate精确旁路。
module qfit_vl_gs_ttb_slot_decoder #(
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

    input  logic update_valid,
    output logic update_ready,
    input  logic [SET_W-1:0] update_set,
    input  logic [SLOT_W-1:0] update_slot,
    input  logic [GATE_W-1:0] update_gate,

    input  logic primary_valid,
    output logic primary_ready,
    input  logic [SET_W-1:0] primary_set,
    input  logic [SLOT_W-1:0] primary_slot,
    input  logic primary_use_exception,
    input  logic [PAYLOAD_W-1:0] primary_payload,
    input  logic primary_last,

    input  logic exception_valid,
    output logic exception_ready,
    input  logic [GATE_W-1:0] exception_gate,

    output logic out_valid,
    input  logic out_ready,
    output logic [GATE_W-1:0] out_gate,
    output logic [PAYLOAD_W-1:0] out_payload,
    output logic out_last,

    output logic protocol_error,
    output logic [31:0] perf_updates,
    output logic [31:0] perf_slot_terms,
    output logic [31:0] perf_exception_terms,
    output logic [31:0] perf_output_stalls
);
    logic slot_valid_q [0:SETS-1][0:SLOTS-1];
    logic [GATE_W-1:0] slot_gate_q [0:SETS-1][0:SLOTS-1];

    logic active_q;
    logic protocol_error_q;
    logic primary_buf_valid_q;
    logic [SET_W-1:0] primary_set_q;
    logic [SLOT_W-1:0] primary_slot_q;
    logic primary_exception_q;
    logic [PAYLOAD_W-1:0] primary_payload_q;
    logic primary_last_q;
    logic exception_buf_valid_q;
    logic [GATE_W-1:0] exception_gate_q;
    logic [31:0] perf_updates_q;
    logic [31:0] perf_slot_q;
    logic [31:0] perf_exception_q;
    logic [31:0] perf_stalls_q;

    logic update_contract_valid;
    logic primary_contract_valid;
    logic slot_reference_valid;
    logic primary_fire;
    logic exception_fire;
    logic out_fire;

    assign update_contract_valid =
        32'(update_set) < SETS
        && 32'(update_slot) < SLOTS
        && update_gate != '0;
    assign primary_contract_valid =
        32'(primary_set) < SETS
        && 32'(primary_slot) < SLOTS;
    assign slot_reference_valid =
        slot_valid_q[primary_set_q][primary_slot_q];

    assign update_ready = active_q && !lifecycle_end;
    assign primary_ready = active_q && !primary_buf_valid_q && !lifecycle_end;
    assign exception_ready = active_q && !exception_buf_valid_q && !lifecycle_end;
    assign primary_fire = primary_valid && primary_ready;
    assign exception_fire = exception_valid && exception_ready;

    assign out_valid =
        primary_buf_valid_q
        && (
            primary_exception_q
                ? exception_buf_valid_q
                : slot_reference_valid
        );
    assign out_gate = primary_exception_q
        ? exception_gate_q
        : slot_gate_q[primary_set_q][primary_slot_q];
    assign out_payload = primary_payload_q;
    assign out_last = primary_last_q;
    assign out_fire = out_valid && out_ready;

    assign lifecycle_active = active_q;
    assign protocol_error = protocol_error_q;
    assign perf_updates = perf_updates_q;
    assign perf_slot_terms = perf_slot_q;
    assign perf_exception_terms = perf_exception_q;
    assign perf_output_stalls = perf_stalls_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            lifecycle_done <= 1'b0;
            protocol_error_q <= 1'b0;
            primary_buf_valid_q <= 1'b0;
            primary_set_q <= '0;
            primary_slot_q <= '0;
            primary_exception_q <= 1'b0;
            primary_payload_q <= '0;
            primary_last_q <= 1'b0;
            exception_buf_valid_q <= 1'b0;
            exception_gate_q <= '0;
            perf_updates_q <= '0;
            perf_slot_q <= '0;
            perf_exception_q <= '0;
            perf_stalls_q <= '0;
            for (integer set_id = 0; set_id < SETS; set_id++)
                for (integer slot = 0; slot < SLOTS; slot++) begin
                    slot_valid_q[set_id][slot] <= 1'b0;
                    slot_gate_q[set_id][slot] <= '0;
                end
        end else begin
            lifecycle_done <= 1'b0;

            if (lifecycle_start) begin
                if (active_q)
                    protocol_error_q <= 1'b1;
                else begin
                    active_q <= 1'b1;
                    protocol_error_q <= 1'b0;
                    primary_buf_valid_q <= 1'b0;
                    exception_buf_valid_q <= 1'b0;
                    perf_updates_q <= '0;
                    perf_slot_q <= '0;
                    perf_exception_q <= '0;
                    perf_stalls_q <= '0;
                    for (integer set_id = 0; set_id < SETS; set_id++)
                        for (integer slot = 0; slot < SLOTS; slot++) begin
                            slot_valid_q[set_id][slot] <= 1'b0;
                            slot_gate_q[set_id][slot] <= '0;
                        end
                end
            end

            if (update_valid && update_ready) begin
                if (!update_contract_valid) begin
                    protocol_error_q <= 1'b1;
                end else if (
                    slot_valid_q[update_set][update_slot]
                    && slot_gate_q[update_set][update_slot] != update_gate
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    slot_valid_q[update_set][update_slot] <= 1'b1;
                    slot_gate_q[update_set][update_slot] <= update_gate;
                    perf_updates_q <= perf_updates_q + 1'b1;
                end
            end

            if (primary_fire) begin
                if (
                    !primary_contract_valid
                    || (
                        !primary_use_exception
                        && !slot_valid_q[primary_set][primary_slot]
                    )
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    primary_buf_valid_q <= 1'b1;
                    primary_set_q <= primary_set;
                    primary_slot_q <= primary_slot;
                    primary_exception_q <= primary_use_exception;
                    primary_payload_q <= primary_payload;
                    primary_last_q <= primary_last;
                end
            end

            if (exception_fire) begin
                if (exception_gate == '0)
                    protocol_error_q <= 1'b1;
                else begin
                    exception_buf_valid_q <= 1'b1;
                    exception_gate_q <= exception_gate;
                end
            end

            if (out_valid && !out_ready)
                perf_stalls_q <= perf_stalls_q + 1'b1;
            if (out_fire) begin
                primary_buf_valid_q <= 1'b0;
                if (primary_exception_q) begin
                    exception_buf_valid_q <= 1'b0;
                    perf_exception_q <= perf_exception_q + 1'b1;
                end else begin
                    perf_slot_q <= perf_slot_q + 1'b1;
                end
            end

            if (lifecycle_end) begin
                if (
                    !active_q
                    || primary_buf_valid_q
                    || exception_buf_valid_q
                ) begin
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
            $fatal(1, "VL-GS-TTB decoder参数非法");
    end
endmodule

`default_nettype wire
