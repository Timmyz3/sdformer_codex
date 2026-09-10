// Test wrapper only. External packet storage and repair responses are driven
// independently by the C++ scoreboard; neither is a fabricated production leaf.
module packet_test_top #(
    parameter int B=32, K=4, W=32, T=10,
    parameter int PW=B+2*(W+1)+2+K*(2*(W+1)+$clog2(B)+1)+82
) (
    input logic clk, rst_n,
    input logic e_start_valid, output logic e_start_ready,
    input logic [31:0] e_context, e_theta,
    input logic [15:0] e_epoch,
    input logic e_reverse,
    input logic signed [W-1:0] e_prediction,
    input logic e_in_valid, output logic e_in_ready,
    input logic signed [W-1:0] e_lo, e_hi,
    output logic e_packet_valid, input logic e_packet_ready,
    output logic [PW-1:0] e_packet,
    input logic v_valid, output logic v_ready,
    input logic [PW-1:0] v_packet,
    input logic [31:0] v_context, input logic [15:0] v_epoch,
    input logic signed [W-1:0] v_tau_lo, v_tau_hi,
    output logic v_result_valid, input logic v_result_ready,
    output logic [1:0] v_status, output logic [B-1:0] v_bits,
    output logic [31:0] v_theta, v_out_context,
    output logic [15:0] v_out_epoch,
    input logic g_start_valid, output logic g_start_ready,
    input logic [31:0] g_base, g_theta,
    input logic [15:0] g_epoch,
    input logic g_packet_valid, output logic g_packet_ready,
    input logic [1:0] g_packet_status,
    input logic [B-1:0] g_packet_bits,
    input logic [31:0] g_packet_context, g_packet_theta,
    input logic [15:0] g_packet_epoch,
    output logic g_replay_valid, input logic g_replay_ready,
    output logic [T-1:0] g_replay_mask,
    input logic g_repair_valid, output logic g_repair_ready,
    input logic [T*B-1:0] g_repair_bits,
    input logic [31:0] g_repair_context, g_repair_theta,
    input logic [15:0] g_repair_epoch,
    output logic g_result_valid, input logic g_result_ready,
    output logic [1:0] g_status,
    output logic [T*B-1:0] g_bits,
    output logic [31:0] g_out_context, g_out_theta,
    output logic [15:0] g_out_epoch
);
    atlif_threshold_packet_encoder #(.B(B),.K(K),.W(W)) encoder (
        .clk(clk),.rst_n(rst_n),.start_valid_i(e_start_valid),.start_ready_o(e_start_ready),
        .context_i(e_context),.epoch_i(e_epoch),.theta_payload_i(e_theta),.reverse_i(e_reverse),
        .predicted_tau_i(e_prediction),.in_valid_i(e_in_valid),.in_ready_o(e_in_ready),
        .u_lo_i(e_lo),.u_hi_i(e_hi),.packet_valid_o(e_packet_valid),
        .packet_ready_i(e_packet_ready),.packet_o(e_packet)
    );
    atlif_threshold_packet_verifier #(.B(B),.K(K),.W(W)) verifier (
        .clk(clk),.rst_n(rst_n),.request_valid_i(v_valid),.request_ready_o(v_ready),
        .packet_i(v_packet),.context_i(v_context),.epoch_i(v_epoch),
        .tau_lo_i(v_tau_lo),.tau_hi_i(v_tau_hi),.result_valid_o(v_result_valid),
        .result_ready_i(v_result_ready),.status_o(v_status),.bits_o(v_bits),
        .theta_payload_o(v_theta),.context_o(v_out_context),.epoch_o(v_out_epoch)
    );
    atlif_threshold_group_commit #(.B(B),.T(T)) group_commit (
        .clk(clk),.rst_n(rst_n),.start_valid_i(g_start_valid),.start_ready_o(g_start_ready),
        .context_base_i(g_base),.epoch_i(g_epoch),.theta_payload_i(g_theta),
        .packet_valid_i(g_packet_valid),.packet_ready_o(g_packet_ready),
        .packet_status_i(g_packet_status),.packet_bits_i(g_packet_bits),
        .packet_context_i(g_packet_context),.packet_epoch_i(g_packet_epoch),
        .packet_theta_i(g_packet_theta),.replay_valid_o(g_replay_valid),
        .replay_ready_i(g_replay_ready),.replay_mask_o(g_replay_mask),
        .repair_valid_i(g_repair_valid),.repair_ready_o(g_repair_ready),
        .repair_bits_i(g_repair_bits),.repair_context_base_i(g_repair_context),
        .repair_epoch_i(g_repair_epoch),.repair_theta_i(g_repair_theta),
        .result_valid_o(g_result_valid),.result_ready_i(g_result_ready),
        .result_status_o(g_status),.result_bits_o(g_bits),.context_base_o(g_out_context),
        .epoch_o(g_out_epoch),.theta_payload_o(g_out_theta)
    );
endmodule
