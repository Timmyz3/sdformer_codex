`timescale 1ns/1ps
`default_nettype none

// Local5原子bind-and-issue消费端：同拍slot commit可转发给对应primary。
module qfit_vl_gs_ttb_abic_decoder #(
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
    output logic [31:0] perf_commit_forwards,
    output logic [31:0] perf_output_stalls
);
    localparam int ENTRIES = SETS * SLOTS;
    localparam int ADDR_W = (ENTRIES <= 1) ? 1 : $clog2(ENTRIES);

    logic [ENTRIES-1:0] slot_valid_q;
    logic [GATE_W-1:0] slot_gate_q [0:ENTRIES-1];
    logic active_q;
    logic protocol_error_q;
    logic primary_q_valid;
    logic primary_exception_q;
    logic [GATE_W-1:0] primary_gate_q;
    logic [PAYLOAD_W-1:0] primary_payload_q;
    logic primary_last_q;
    logic exception_q_valid;
    logic [GATE_W-1:0] exception_gate_q;
    logic [31:0] forward_q;
    logic [31:0] stall_q;

    logic update_contract_valid;
    logic update_fire;
    logic primary_contract_valid;
    logic primary_slot_valid;
    logic primary_forward_valid;
    logic primary_fire;
    logic exception_fire;
    logic out_fire;
    logic output_space;
    logic [ADDR_W-1:0] update_addr;
    logic [ADDR_W-1:0] primary_addr;

    assign update_addr = ADDR_W'(
        32'(update_set) * SLOTS + 32'(update_slot)
    );
    assign primary_addr = ADDR_W'(
        32'(primary_set) * SLOTS + 32'(primary_slot)
    );

    assign update_contract_valid =
        32'(update_set) < SETS
        && 32'(update_slot) < SLOTS
        && update_gate != '0;
    assign update_ready = active_q && !lifecycle_end;
    assign update_fire = update_valid && update_ready && update_contract_valid;
    assign primary_contract_valid =
        32'(primary_set) < SETS && 32'(primary_slot) < SLOTS;
    assign primary_slot_valid = primary_contract_valid
        && slot_valid_q[primary_addr];
    assign primary_forward_valid = update_fire
        && update_set == primary_set
        && update_slot == primary_slot;
    assign output_space = !primary_q_valid || out_ready;
    assign primary_ready = active_q
        && !lifecycle_end
        && output_space
        && primary_contract_valid
        && (
            primary_use_exception
            || primary_slot_valid
            || primary_forward_valid
        );
    assign exception_ready = active_q
        && !lifecycle_end
        && (
            !exception_q_valid
            || (out_fire && primary_exception_q)
        );
    assign primary_fire = primary_valid && primary_ready;
    assign exception_fire = exception_valid && exception_ready;

    assign out_valid = primary_q_valid
        && (!primary_exception_q || exception_q_valid);
    assign out_gate = primary_exception_q
        ? exception_gate_q
        : primary_gate_q;
    assign out_payload = primary_payload_q;
    assign out_last = primary_last_q;
    assign out_fire = out_valid && out_ready;
    assign lifecycle_active = active_q;
    assign protocol_error = protocol_error_q;
    assign perf_commit_forwards = forward_q;
    assign perf_output_stalls = stall_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_q <= 1'b0;
            lifecycle_done <= 1'b0;
            protocol_error_q <= 1'b0;
            primary_q_valid <= 1'b0;
            primary_exception_q <= 1'b0;
            primary_gate_q <= '0;
            primary_payload_q <= '0;
            primary_last_q <= 1'b0;
            exception_q_valid <= 1'b0;
            exception_gate_q <= '0;
            forward_q <= '0;
            stall_q <= '0;
            slot_valid_q <= '0;
        end else begin
            lifecycle_done <= 1'b0;
            if (lifecycle_start) begin
                if (active_q) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    active_q <= 1'b1;
                    protocol_error_q <= 1'b0;
                    primary_q_valid <= 1'b0;
                    primary_exception_q <= 1'b0;
                    exception_q_valid <= 1'b0;
                    forward_q <= '0;
                    stall_q <= '0;
                    slot_valid_q <= '0;
                end
            end else begin
                if (update_valid && update_ready) begin
                    if (!update_contract_valid) begin
                        protocol_error_q <= 1'b1;
                    end else if (
                        slot_valid_q[update_addr]
                        && slot_gate_q[update_addr] != update_gate
                    ) begin
                        protocol_error_q <= 1'b1;
                    end else begin
                        slot_valid_q[update_addr] <= 1'b1;
                        slot_gate_q[update_addr] <= update_gate;
                    end
                end

                if (out_valid && !out_ready)
                    stall_q <= stall_q + 1'b1;
                if (out_fire) begin
                    primary_q_valid <= 1'b0;
                    if (primary_exception_q)
                        exception_q_valid <= 1'b0;
                end

                if (primary_valid && active_q && !lifecycle_end
                    && !primary_contract_valid)
                    protocol_error_q <= 1'b1;
                if (primary_fire) begin
                    primary_q_valid <= 1'b1;
                    primary_exception_q <= primary_use_exception;
                    if (!primary_use_exception && primary_forward_valid) begin
                        primary_gate_q <= update_gate;
                        forward_q <= forward_q + 1'b1;
                    end else if (!primary_use_exception)
                        primary_gate_q <= slot_gate_q[primary_addr];
                    primary_payload_q <= primary_payload;
                    primary_last_q <= primary_last;
                end

                if (exception_fire) begin
                    if (exception_gate == '0)
                        protocol_error_q <= 1'b1;
                    else begin
                        exception_q_valid <= 1'b1;
                        exception_gate_q <= exception_gate;
                    end
                end
            end

            if (lifecycle_end) begin
                if (!active_q || primary_q_valid || exception_q_valid)
                    protocol_error_q <= 1'b1;
                else begin
                    active_q <= 1'b0;
                    lifecycle_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (SETS < 1 || SLOTS < 2 || GATE_W < 2 || PAYLOAD_W < 1)
            $fatal(1, "ABIC decoder参数非法");
    end
endmodule

`default_nettype wire
