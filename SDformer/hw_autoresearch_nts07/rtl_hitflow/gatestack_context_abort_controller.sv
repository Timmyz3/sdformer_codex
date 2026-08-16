`timescale 1ns/1ps
`default_nettype none

// Bounded single-context recovery. Any pre-retire fabric error or watchdog
// timeout flushes the complete execution context and returns one tagged error.
module gatestack_context_abort_controller #(
    parameter int TAG_W = 32,
    parameter int TIMEOUT_CYCLES = 1000000,
    parameter int COUNTER_W = 32
) (
    input  logic                     clk_core,
    input  logic                     rst_core,
    input  logic                     group_accept_pulse,
    input  logic [TAG_W-1:0]         group_accept_tag,
    input  logic                     normal_done_fire,
    input  logic                     normal_done_error,
    input  logic                     fabric_error,
    output logic                     fabric_reset_pulse,
    output logic                     abort_done_valid,
    input  logic                     abort_done_ready,
    output logic [TAG_W-1:0]         abort_done_tag,
    output logic                     abort_done_error,
    output logic                     admission_blocked,
    output logic                     group_active,
    output logic                     protocol_error,
    output logic [COUNTER_W-1:0]     count_context_resets,
    output logic [COUNTER_W-1:0]     count_error_aborts,
    output logic [COUNTER_W-1:0]     count_timeout_aborts
);
    logic [TAG_W-1:0] active_tag_q;
    logic [COUNTER_W-1:0] watchdog_q;
    logic timeout_hit, abort_fire;

    assign timeout_hit = group_active &&
        32'(watchdog_q) + 32'd1 >= 32'(TIMEOUT_CYCLES);
    assign abort_done_error = 1'b1;
    assign admission_blocked = fabric_reset_pulse || abort_done_valid;
    assign abort_fire = abort_done_valid && abort_done_ready;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            active_tag_q <= '0;
            watchdog_q <= '0;
            fabric_reset_pulse <= 1'b0;
            abort_done_valid <= 1'b0;
            abort_done_tag <= '0;
            group_active <= 1'b0;
            protocol_error <= 1'b0;
            count_context_resets <= '0;
            count_error_aborts <= '0;
            count_timeout_aborts <= '0;
        end else begin
            fabric_reset_pulse <= 1'b0;
            if (abort_fire)
                abort_done_valid <= 1'b0;

            if (group_accept_pulse) begin
                active_tag_q <= group_accept_tag;
                watchdog_q <= '0;
                group_active <= 1'b1;
            end else if (group_active) begin
                watchdog_q <= watchdog_q + 1'b1;
            end

            if (group_active && fabric_error && !normal_done_fire) begin
                fabric_reset_pulse <= 1'b1;
                abort_done_valid <= 1'b1;
                abort_done_tag <= active_tag_q;
                group_active <= 1'b0;
                watchdog_q <= '0;
                protocol_error <= 1'b1;
                count_context_resets <= count_context_resets + 1'b1;
                count_error_aborts <= count_error_aborts + 1'b1;
            end else if (group_active && timeout_hit && !normal_done_fire) begin
                fabric_reset_pulse <= 1'b1;
                abort_done_valid <= 1'b1;
                abort_done_tag <= active_tag_q;
                group_active <= 1'b0;
                watchdog_q <= '0;
                protocol_error <= 1'b1;
                count_context_resets <= count_context_resets + 1'b1;
                count_timeout_aborts <= count_timeout_aborts + 1'b1;
            end else if (normal_done_fire) begin
                group_active <= 1'b0;
                watchdog_q <= '0;
                if (normal_done_error || fabric_error) begin
                    fabric_reset_pulse <= 1'b1;
                    protocol_error <= 1'b1;
                    count_context_resets <= count_context_resets + 1'b1;
                    count_error_aborts <= count_error_aborts + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
