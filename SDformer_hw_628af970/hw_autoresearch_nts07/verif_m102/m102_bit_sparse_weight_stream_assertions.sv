`timescale 1ns/1ps
`default_nettype none

module m102_bit_sparse_weight_stream_assertions #(
    parameter int ROW_W = 10,
    parameter int TAG_W = 32,
    parameter int LANES = 96,
    parameter int OUT_W = 12
) (
    input logic                         clk_core,
    input logic                         rst_core,
    input logic                         lookup_valid,
    input logic                         lookup_ready,
    input logic [3:0]                   lookup_source,
    input logic [2:0]                   lookup_block,
    input logic [1:0]                   lookup_beat,
    input logic [TAG_W-1:0]             lookup_tag,
    input logic [255:0]                 bank_words,
    input logic [8*ROW_W-1:0]           bank_row_addresses,
    input logic                         output_valid,
    input logic                         output_ready,
    input logic [TAG_W-1:0]             output_tag,
    input logic [3:0]                   output_width,
    input logic                         output_escape,
    input logic [LANES*OUT_W-1:0]       output_values,
    input logic                         output_accept,
    input logic                         protocol_error,
    input logic                         busy,
    input logic                         request_violation,
    input logic                         request_fault,
    input logic                         m82_output_valid,
    input logic                         accepted_grace_match
);
    logic shadow_active_q;
    logic [1:0] shadow_expected_beat_q;
    logic [3:0] shadow_source_q;
    logic [2:0] shadow_block_q;
    logic [TAG_W-1:0] shadow_tag_q;
    logic shadow_request_legal;
    logic [6:0] expected_vector_index;
    logic [11:0] expected_logical_word;
    logic [2:0] expected_base_bank;
    logic [ROW_W-1:0] expected_base_row;

    wire lookup_accept = lookup_valid && lookup_ready;

    always_comb begin
        if (!shadow_active_q) begin
            shadow_request_legal = lookup_beat == 0;
        end else begin
            shadow_request_legal = lookup_beat == shadow_expected_beat_q
                                  && lookup_source == shadow_source_q
                                  && lookup_block == shadow_block_q
                                  && lookup_tag == shadow_tag_q;
        end
        shadow_request_legal = shadow_request_legal && lookup_beat < 3;

        expected_vector_index = {lookup_source, lookup_block};
        expected_logical_word = {5'd0, expected_vector_index} * 12'd24
                              + ({10'd0, lookup_beat} << 3);
        expected_base_bank = expected_logical_word[2:0];
        expected_base_row = expected_logical_word[11:3];
    end

    always_ff @(posedge clk_core) begin
        if (rst_core || protocol_error) begin
            shadow_active_q <= 1'b0;
            shadow_expected_beat_q <= '0;
            shadow_source_q <= '0;
            shadow_block_q <= '0;
            shadow_tag_q <= '0;
        end else if (lookup_accept) begin
            if (!shadow_active_q) begin
                shadow_active_q <= 1'b1;
                shadow_expected_beat_q <= 2'd1;
                shadow_source_q <= lookup_source;
                shadow_block_q <= lookup_block;
                shadow_tag_q <= lookup_tag;
            end else if (lookup_beat == 2) begin
                shadow_active_q <= 1'b0;
                shadow_expected_beat_q <= '0;
            end else begin
                shadow_expected_beat_q <= shadow_expected_beat_q + 1'b1;
            end
        end
    end

    ap_output_handshake: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_accept == (output_valid && output_ready));

    ap_accept_only_legal_sequence: assert property (@(posedge clk_core)
        disable iff (rst_core)
        lookup_accept |-> shadow_request_legal);

    ap_illegal_valid_fails_closed_same_cycle: assert property (@(posedge clk_core)
        disable iff (rst_core)
        request_violation |-> protocol_error && !lookup_ready
            && !output_valid && !output_accept);

    ap_accepted_request_grace_is_not_a_fault: assert property (
        @(posedge clk_core) disable iff (rst_core || request_fault)
        lookup_valid && accepted_grace_match
        |-> !request_violation && !protocol_error && !lookup_ready);

    ap_exact_third_beat_completion: assert property (@(posedge clk_core)
        disable iff (rst_core)
        lookup_accept && lookup_beat == 2 |=> output_valid || protocol_error);

    ap_output_stable_under_stall: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid && !output_ready
        |=> protocol_error
            || (output_valid && $stable({output_tag, output_width,
                                         output_escape, output_values})));

    ap_fixed_output_format: assert property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid |-> output_width == 8 && !output_escape);

    ap_escape_is_always_low: assert property (@(posedge clk_core)
        disable iff (rst_core) !output_escape);

    ap_fault_is_sticky: assert property (@(posedge clk_core)
        disable iff (rst_core) request_fault |=> request_fault);

    ap_fault_blocks_interface: assert property (@(posedge clk_core)
        disable iff (rst_core)
        protocol_error |-> !lookup_ready && !output_valid && !output_accept);

    ap_reset_clears_fault: assert property (@(posedge clk_core)
        rst_core |=> !protocol_error);

    ap_busy_while_partial_vector: assert property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        shadow_active_q |-> busy);

    ap_frozen_vectors_are_bank_aligned: assert property (@(posedge clk_core)
        disable iff (rst_core)
        lookup_accept |-> expected_base_bank == 0);

    for (genvar bank = 0; bank < 8; bank++) begin : g_bank_address_assertions
        ap_bank_row_mapping: assert property (@(posedge clk_core)
            disable iff (rst_core)
            lookup_accept |-> bank_row_addresses[bank*ROW_W +: ROW_W]
                == expected_base_row + (bank < expected_base_bank));
    end

    for (genvar lane = 0; lane < LANES; lane++) begin : g_sign_extension_assertions
        ap_signed8_to_signed12: assert property (@(posedge clk_core)
            disable iff (rst_core)
            output_valid |-> output_values[lane*OUT_W + 8 +: 4]
                == {4{output_values[lane*OUT_W + 7]}});
    end

    cp_exact_ii3: cover property (@(posedge clk_core)
        disable iff (rst_core)
        (lookup_accept && lookup_beat == 0) ##1
        (lookup_accept && lookup_beat == 1) ##1
        (lookup_accept && lookup_beat == 2) ##1
        (lookup_accept && lookup_beat == 0));

    cp_output_stall: cover property (@(posedge clk_core)
        disable iff (rst_core) output_valid && !output_ready);

    cp_signed_boundaries: cover property (@(posedge clk_core)
        disable iff (rst_core)
        output_valid
        && output_values[0*OUT_W +: OUT_W] == 12'hf80
        && output_values[1*OUT_W +: OUT_W] == 12'h07f);

    cp_protocol_fault: cover property (@(posedge clk_core)
        disable iff (rst_core) protocol_error);

    cp_same_cycle_release_quarantine: cover property (@(posedge clk_core)
        disable iff (rst_core)
        request_violation && output_ready && m82_output_valid
        && !output_valid && !output_accept);

    cp_accepted_request_grace: cover property (@(posedge clk_core)
        disable iff (rst_core || protocol_error)
        lookup_valid && accepted_grace_match && output_valid);

    cp_fault_quarantines_buffered_output: cover property (@(posedge clk_core)
        disable iff (rst_core)
        protocol_error && m82_output_valid && !output_valid && !output_accept);

    cp_fault_reset_recovery: cover property (@(posedge clk_core)
        protocol_error ##1 rst_core[*1:4] ##1
        (!rst_core && !protocol_error));
endmodule

`default_nettype wire
