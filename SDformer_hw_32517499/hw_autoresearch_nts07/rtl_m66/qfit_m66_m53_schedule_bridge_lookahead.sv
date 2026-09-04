`timescale 1ns/1ps
`default_nettype none

// Additive telemetry wrapper around the M66 lookahead derivative.  It still
// contains no trace scheduler; only the final-response/next-launch seam is
// changed relative to M57/M54.
module qfit_m66_m53_schedule_bridge_lookahead #(
    parameter int TILE_BITS = 256,
    parameter int BANKS = 8,
    parameter int LANES = 96,
    parameter int CONTEXTS = 16,
    parameter int MAX_K = 4,
    parameter int TAG_W = 48,
    parameter int RESPONSE_TAG_W = 16,
    parameter int W_W = 8,
    parameter int ACC_W = 19,
    parameter int BANK_ADDR_W = 5,
    parameter int COUNT_W = 9,
    parameter int CONTEXT_W = 4,
    parameter int GROUP_COUNT_W = 3
) (
    input  logic                                  clk_core,
    input  logic                                  rst_core,
    input  logic                                  command_valid,
    output logic                                  command_ready,
    input  logic [TAG_W-1:0]                      command_tag,
    input  logic [TILE_BITS-1:0]                  command_add_bits,
    input  logic [TILE_BITS-1:0]                  command_subtract_bits,
    input  logic [LANES*ACC_W-1:0]                command_seed_acc,
    output logic                                  command_accept,
    output logic [CONTEXT_W-1:0]                  command_accept_context,
    input  logic                                  launch_valid,
    output logic                                  launch_ready,
    input  logic [GROUP_COUNT_W-1:0]              launch_context_count,
    input  logic [MAX_K*CONTEXT_W-1:0]            launch_contexts,
    output logic                                  launch_accept,
    output logic                                  weight_request_valid,
    input  logic                                  weight_request_ready,
    output logic [RESPONSE_TAG_W-1:0]             weight_request_tag,
    output logic [GROUP_COUNT_W-1:0]              weight_request_context_count,
    output logic [MAX_K*CONTEXT_W-1:0]            weight_request_contexts,
    output logic [BANKS-1:0]                      weight_request_bank_valid,
    output logic [BANKS*BANK_ADDR_W-1:0]          weight_request_bank_addr,
    output logic [MAX_K*BANKS-1:0]                weight_request_context_valid,
    output logic [MAX_K*BANKS-1:0]                weight_request_context_subtract,
    output logic                                  weight_request_last,
    output logic                                  request_accept,
    input  logic                                  weight_response_valid,
    output logic                                  weight_response_ready,
    input  logic [RESPONSE_TAG_W-1:0]             weight_response_tag,
    input  logic [GROUP_COUNT_W-1:0]              weight_response_context_count,
    input  logic [MAX_K*CONTEXT_W-1:0]            weight_response_contexts,
    input  logic [BANKS-1:0]                      weight_response_bank_valid,
    input  logic [BANKS*LANES*W_W-1:0]           weight_response_data,
    output logic                                  response_accept,
    output logic                                  output_valid,
    input  logic                                  output_ready,
    output logic [TAG_W-1:0]                      output_tag,
    output logic [COUNT_W-1:0]                    output_source_count,
    output logic [LANES*ACC_W-1:0]                output_acc,
    output logic                                  output_accept,
    output logic                                  protocol_error,
    output logic                                  busy,
    output logic [CONTEXT_W:0]                    context_occupancy,
    output logic [4:0]                            response_metadata_occupancy,
    output logic [4:0]                            complete_occupancy,
    output logic                                  group_active,
    output logic [63:0]                           telemetry_cycles,
    output logic [63:0]                           telemetry_commands,
    output logic [63:0]                           telemetry_launches,
    output logic [63:0]                           telemetry_requests,
    output logic [63:0]                           telemetry_responses,
    output logic [63:0]                           telemetry_outputs,
    output logic [63:0]                           telemetry_command_stalls,
    output logic [63:0]                           telemetry_launch_stalls,
    output logic [63:0]                           telemetry_request_stalls,
    output logic [63:0]                           telemetry_response_stalls,
    output logic [63:0]                           telemetry_output_stalls,
    output logic [63:0]                           telemetry_context_reuses,
    output logic [63:0]                           telemetry_response_tag_wraps,
    output logic [4:0]                            telemetry_max_context_occupancy,
    output logic [4:0]                            telemetry_max_metadata_occupancy,
    output logic [4:0]                            telemetry_max_complete_occupancy
);
    logic [CONTEXTS-1:0] context_seen_q;

    qfit_k4_parent_delta_p8_l96_ctx16_lookahead core (
        .clk_core, .rst_core,
        .command_valid, .command_ready, .command_tag,
        .command_add_bits, .command_subtract_bits, .command_seed_acc,
        .command_accept, .command_accept_context,
        .launch_valid, .launch_ready, .launch_context_count,
        .launch_contexts, .launch_accept,
        .weight_request_valid, .weight_request_ready, .weight_request_tag,
        .weight_request_context_count, .weight_request_contexts,
        .weight_request_bank_valid, .weight_request_bank_addr,
        .weight_request_context_valid, .weight_request_context_subtract,
        .weight_request_last, .request_accept,
        .weight_response_valid, .weight_response_ready, .weight_response_tag,
        .weight_response_context_count, .weight_response_contexts,
        .weight_response_bank_valid, .weight_response_data, .response_accept,
        .output_valid, .output_ready, .output_tag, .output_source_count,
        .output_acc, .output_accept, .protocol_error, .busy,
        .context_occupancy, .response_metadata_occupancy,
        .complete_occupancy, .group_active
    );

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            telemetry_cycles <= '0;
            telemetry_commands <= '0;
            telemetry_launches <= '0;
            telemetry_requests <= '0;
            telemetry_responses <= '0;
            telemetry_outputs <= '0;
            telemetry_command_stalls <= '0;
            telemetry_launch_stalls <= '0;
            telemetry_request_stalls <= '0;
            telemetry_response_stalls <= '0;
            telemetry_output_stalls <= '0;
            telemetry_context_reuses <= '0;
            telemetry_response_tag_wraps <= '0;
            telemetry_max_context_occupancy <= '0;
            telemetry_max_metadata_occupancy <= '0;
            telemetry_max_complete_occupancy <= '0;
            context_seen_q <= '0;
        end else begin
            telemetry_cycles <= telemetry_cycles + 1'b1;
            if (command_accept) begin
                telemetry_commands <= telemetry_commands + 1'b1;
                if (context_seen_q[command_accept_context])
                    telemetry_context_reuses <= telemetry_context_reuses + 1'b1;
                context_seen_q[command_accept_context] <= 1'b1;
            end
            if (launch_accept) telemetry_launches <= telemetry_launches + 1'b1;
            if (request_accept) begin
                telemetry_requests <= telemetry_requests + 1'b1;
                if (weight_request_tag == {RESPONSE_TAG_W{1'b1}})
                    telemetry_response_tag_wraps
                        <= telemetry_response_tag_wraps + 1'b1;
            end
            if (response_accept) telemetry_responses <= telemetry_responses + 1'b1;
            if (output_accept) telemetry_outputs <= telemetry_outputs + 1'b1;
            if (command_valid && !command_ready)
                telemetry_command_stalls <= telemetry_command_stalls + 1'b1;
            if (launch_valid && !launch_ready)
                telemetry_launch_stalls <= telemetry_launch_stalls + 1'b1;
            if (weight_request_valid && !weight_request_ready)
                telemetry_request_stalls <= telemetry_request_stalls + 1'b1;
            if (weight_response_valid && !weight_response_ready)
                telemetry_response_stalls <= telemetry_response_stalls + 1'b1;
            if (output_valid && !output_ready)
                telemetry_output_stalls <= telemetry_output_stalls + 1'b1;
            if (context_occupancy > telemetry_max_context_occupancy)
                telemetry_max_context_occupancy <= context_occupancy;
            if (response_metadata_occupancy > telemetry_max_metadata_occupancy)
                telemetry_max_metadata_occupancy <= response_metadata_occupancy;
            if (complete_occupancy > telemetry_max_complete_occupancy)
                telemetry_max_complete_occupancy <= complete_occupancy;
        end
    end
endmodule

`default_nettype wire
