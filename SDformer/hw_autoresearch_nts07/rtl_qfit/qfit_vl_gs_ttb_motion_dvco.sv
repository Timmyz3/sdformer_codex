`timescale 1ns/1ps
`default_nettype none

// Motion双词表上下文重叠：header在一bank构建时，body可从另一bank消费。
module qfit_vl_gs_ttb_motion_dvco #(
    parameter int SLOTS = 4,
    parameter int GATE_W = 9,
    parameter int PAYLOAD_W = 32,
    parameter int SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
    input  logic clk_core,
    input  logic rst_core,

    input  logic build_start_valid,
    output logic build_start_ready,
    input  logic build_raw_mode,
    input  logic build_update_valid,
    output logic build_update_ready,
    input  logic [SLOT_W-1:0] build_update_slot,
    input  logic [GATE_W-1:0] build_update_gate,
    input  logic build_commit_valid,
    output logic build_commit_ready,

    output logic body_context_valid,
    input  logic body_context_ready,
    output logic body_context_raw_mode,
    input  logic body_valid,
    output logic body_ready,
    input  logic [SLOT_W-1:0] body_slot,
    input  logic [GATE_W-1:0] body_raw_gate,
    input  logic [PAYLOAD_W-1:0] body_payload,
    input  logic body_last,

    output logic out_valid,
    input  logic out_ready,
    output logic [GATE_W-1:0] out_gate,
    output logic [PAYLOAD_W-1:0] out_payload,
    output logic out_last,

    output logic protocol_error,
    output logic [31:0] perf_overlap_cycles,
    output logic [31:0] perf_build_wait_bank,
    output logic [31:0] perf_body_wait_context
);
    typedef enum logic [2:0] {
        B_EMPTY,
        B_BUILD,
        B_COMMITTED,
        B_ACTIVE
    } bank_state_t;

    bank_state_t bank_state_q [0:1];
    logic bank_raw_q [0:1];
    logic slot_valid_q [0:1][0:SLOTS-1];
    logic [GATE_W-1:0] slot_gate_q [0:1][0:SLOTS-1];
    logic build_bank_q;
    logic body_bank_q;
    logic out_valid_q;
    logic [GATE_W-1:0] out_gate_q;
    logic [PAYLOAD_W-1:0] out_payload_q;
    logic out_last_q;
    logic protocol_error_q;
    logic [31:0] overlap_q;
    logic [31:0] build_wait_q;
    logic [31:0] body_wait_q;

    logic build_active;
    logic body_active;
    logic update_contract_valid;
    logic body_contract_valid;
    logic body_fire;

    assign build_active = bank_state_q[build_bank_q] == B_BUILD;
    assign body_active = bank_state_q[body_bank_q] == B_ACTIVE;
    assign build_start_ready = bank_state_q[build_bank_q] == B_EMPTY;
    assign build_update_ready = build_active && !build_commit_valid;
    assign build_commit_ready = build_active && !build_update_valid;
    assign body_context_valid = bank_state_q[body_bank_q] == B_COMMITTED;
    assign body_context_raw_mode = bank_raw_q[body_bank_q];
    assign body_contract_valid = bank_raw_q[body_bank_q]
        ? body_raw_gate != '0
        : (
            32'(body_slot) < SLOTS
            && slot_valid_q[body_bank_q][body_slot]
        );
    assign body_ready = body_active
        && (!out_valid_q || out_ready)
        && body_contract_valid;
    assign body_fire = body_valid && body_ready;
    assign update_contract_valid =
        32'(build_update_slot) < SLOTS && build_update_gate != '0;

    assign out_valid = out_valid_q;
    assign out_gate = out_gate_q;
    assign out_payload = out_payload_q;
    assign out_last = out_last_q;
    assign protocol_error = protocol_error_q;
    assign perf_overlap_cycles = overlap_q;
    assign perf_build_wait_bank = build_wait_q;
    assign perf_body_wait_context = body_wait_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            build_bank_q <= 1'b0;
            body_bank_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_gate_q <= '0;
            out_payload_q <= '0;
            out_last_q <= 1'b0;
            protocol_error_q <= 1'b0;
            overlap_q <= '0;
            build_wait_q <= '0;
            body_wait_q <= '0;
            for (integer bank = 0; bank < 2; bank++) begin
                bank_state_q[bank] <= B_EMPTY;
                bank_raw_q[bank] <= 1'b0;
                for (integer slot = 0; slot < SLOTS; slot++) begin
                    slot_valid_q[bank][slot] <= 1'b0;
                end
            end
        end else begin
            if (build_active && body_active)
                overlap_q <= overlap_q + 1'b1;
            if (build_start_valid && !build_start_ready)
                build_wait_q <= build_wait_q + 1'b1;
            if (body_context_ready && !body_context_valid && !body_active)
                body_wait_q <= body_wait_q + 1'b1;

            if (out_valid_q && out_ready)
                out_valid_q <= 1'b0;

            if (build_start_valid && build_start_ready) begin
                bank_state_q[build_bank_q] <= B_BUILD;
                bank_raw_q[build_bank_q] <= build_raw_mode;
                for (integer slot = 0; slot < SLOTS; slot++) begin
                    slot_valid_q[build_bank_q][slot] <= 1'b0;
                end
            end

            if (build_update_valid && build_update_ready) begin
                if (!update_contract_valid || bank_raw_q[build_bank_q]) begin
                    protocol_error_q <= 1'b1;
                end else if (
                    slot_valid_q[build_bank_q][build_update_slot]
                    && slot_gate_q[build_bank_q][build_update_slot]
                        != build_update_gate
                ) begin
                    protocol_error_q <= 1'b1;
                end else begin
                    slot_valid_q[build_bank_q][build_update_slot] <= 1'b1;
                    slot_gate_q[build_bank_q][build_update_slot]
                        <= build_update_gate;
                end
            end

            if (build_commit_valid && build_commit_ready) begin
                bank_state_q[build_bank_q] <= B_COMMITTED;
                build_bank_q <= ~build_bank_q;
            end

            if (body_context_valid && body_context_ready)
                bank_state_q[body_bank_q] <= B_ACTIVE;

            if (body_valid && body_active && !body_contract_valid)
                protocol_error_q <= 1'b1;

            if (body_fire) begin
                out_valid_q <= 1'b1;
                out_gate_q <= bank_raw_q[body_bank_q]
                    ? body_raw_gate
                    : slot_gate_q[body_bank_q][body_slot];
                out_payload_q <= body_payload;
                out_last_q <= body_last;
                if (body_last) begin
                    bank_state_q[body_bank_q] <= B_EMPTY;
                    body_bank_q <= ~body_bank_q;
                end
            end
        end
    end

    initial begin
        if (SLOTS < 2 || GATE_W < 1 || PAYLOAD_W < 1)
            $fatal(1, "Motion DVCO参数非法");
    end
endmodule

`default_nettype wire
