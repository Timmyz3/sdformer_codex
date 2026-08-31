`timescale 1ns/1ps
`default_nettype none

// Exact K1 fusion probe for the admitted M430/M433 data path.
//
// A fused request concurrently consumes one already-existing PWP-cache read
// and one already-existing signed-INT8 correction-weight read.  It produces
// only update_delta = PWP + signed_correction.  The persistent accumulator is
// downstream and must still perform new_psum = old_psum + update_delta.
// This block neither adds a memory port nor claims that concurrent reads are
// power-free; those assumptions are priced separately.
module m451_exact_k1_fused_pwp_correction_adapter #(
    parameter int TAG_BITS = 24
) (
    input  logic                    clk_core,
    input  logic                    reset_n,
    input  logic                    config_reload,

    input  logic                    request_valid,
    output logic                    request_ready,
    output logic                    request_accept,
    input  logic [TAG_BITS-1:0]     low_tag,
    input  logic                    low_tile,
    input  logic [4:0]              low_center_id,
    input  logic [2:0]              low_output_block,
    input  logic                    request_narrow,
    input  logic [767:0]            low_data,
    input  logic [TAG_BITS-1:0]     high_tag,
    input  logic                    high_tile,
    input  logic [4:0]              high_center_id,
    input  logic [2:0]              high_output_block,
    input  logic [511:0]            high_data,

    input  logic                    request_fuse_correction,
    input  logic                    correction_subtract,
    input  logic [TAG_BITS-1:0]     correction_tag,
    input  logic                    correction_tile,
    input  logic [2:0]              correction_output_block,
    // One existing correction read: 96 signed-INT8 weights.  The single
    // correction_subtract bit realizes a source residual of -1; otherwise +1.
    input  logic [767:0]            correction_data,

    output logic                    contribution_valid,
    input  logic                    contribution_ready,
    output logic                    contribution_accept,
    output logic [TAG_BITS-1:0]     contribution_tag,
    output logic                    contribution_tile,
    output logic [4:0]              contribution_center_id,
    output logic [2:0]              contribution_output_block,
    output logic                    contribution_narrow,
    output logic                    contribution_fused,
    output logic [1247:0]           contribution_data,

    output logic                    protocol_error,
    output logic                    busy,
    output logic [31:0]             debug_request_accepts,
    output logic [31:0]             debug_plain_accepts,
    output logic [31:0]             debug_fused_accepts,
    output logic [31:0]             debug_contributions,
    output logic [31:0]             debug_protocol_faults
);
    logic fault_q;
    logic output_valid_q;
    logic [TAG_BITS-1:0] output_tag_q;
    logic output_tile_q;
    logic [4:0] output_center_q;
    logic [2:0] output_block_q;
    logic output_narrow_q;
    logic output_fused_q;
    logic [1247:0] output_data_q;

    logic [31:0] request_accepts_q;
    logic [31:0] plain_accepts_q;
    logic [31:0] fused_accepts_q;
    logic [31:0] contributions_q;
    logic [31:0] protocol_faults_q;

    logic wide_metadata_legal_w;
    logic wide_padding_legal_w;
    logic narrow_high_side_zero_w;
    logic pwp_payload_legal_w;
    logic fused_correction_legal_w;
    logic plain_correction_zero_w;
    logic correction_payload_legal_w;
    logic illegal_request_w;
    logic illegal_reload_w;
    logic illegal_now_w;
    logic pop_w;

    assign wide_metadata_legal_w =
        (high_tag == low_tag) &&
        (high_tile == low_tile) &&
        (high_center_id == low_center_id) &&
        (high_output_block == low_output_block);
    assign wide_padding_legal_w = (high_data[511:384] == 128'b0);
    assign narrow_high_side_zero_w =
        (high_tag == {TAG_BITS{1'b0}}) &&
        (high_tile == 1'b0) &&
        (high_center_id == 5'b0) &&
        (high_output_block == 3'b0) &&
        (high_data == 512'b0);
    assign pwp_payload_legal_w = request_narrow ?
        narrow_high_side_zero_w :
        (wide_metadata_legal_w && wide_padding_legal_w);

    assign fused_correction_legal_w =
        (correction_tag == low_tag) &&
        (correction_tile == low_tile) &&
        (correction_output_block == low_output_block);
    assign plain_correction_zero_w =
        (correction_subtract == 1'b0) &&
        (correction_tag == {TAG_BITS{1'b0}}) &&
        (correction_tile == 1'b0) &&
        (correction_output_block == 3'b0) &&
        (correction_data == 768'b0);
    assign correction_payload_legal_w = request_fuse_correction ?
        fused_correction_legal_w : plain_correction_zero_w;

    assign illegal_request_w = request_valid &&
        !(pwp_payload_legal_w && correction_payload_legal_w);
    assign illegal_reload_w = config_reload &&
        (output_valid_q || request_valid);
    assign illegal_now_w = illegal_request_w || illegal_reload_w;

    assign contribution_valid = !fault_q && !illegal_now_w && output_valid_q;
    assign contribution_accept = contribution_valid && contribution_ready;
    assign pop_w = contribution_accept;
    assign request_ready = !fault_q && !illegal_now_w && !config_reload &&
        (!output_valid_q || pop_w);
    assign request_accept = request_valid && request_ready;

    assign contribution_tag = contribution_valid ? output_tag_q : '0;
    assign contribution_tile = contribution_valid && output_tile_q;
    assign contribution_center_id = contribution_valid ? output_center_q : '0;
    assign contribution_output_block = contribution_valid ? output_block_q : '0;
    assign contribution_narrow = contribution_valid && output_narrow_q;
    assign contribution_fused = contribution_valid && output_fused_q;
    assign contribution_data = contribution_valid ? output_data_q : '0;

    assign protocol_error = fault_q;
    assign busy = output_valid_q;
    assign debug_request_accepts = request_accepts_q;
    assign debug_plain_accepts = plain_accepts_q;
    assign debug_fused_accepts = fused_accepts_q;
    assign debug_contributions = contributions_q;
    assign debug_protocol_faults = protocol_faults_q;

    integer lane;
    always_ff @(posedge clk_core or negedge reset_n) begin
        if (!reset_n) begin
            fault_q <= 1'b0;
            output_valid_q <= 1'b0;
            output_tag_q <= '0;
            output_tile_q <= 1'b0;
            output_center_q <= '0;
            output_block_q <= '0;
            output_narrow_q <= 1'b0;
            output_fused_q <= 1'b0;
            output_data_q <= '0;
            request_accepts_q <= '0;
            plain_accepts_q <= '0;
            fused_accepts_q <= '0;
            contributions_q <= '0;
            protocol_faults_q <= '0;
        end else begin
            if (illegal_now_w) begin
                fault_q <= 1'b1;
                if (!fault_q)
                    protocol_faults_q <= protocol_faults_q + 1'b1;
            end

            if (request_accept) begin
                output_tag_q <= low_tag;
                output_tile_q <= low_tile;
                output_center_q <= low_center_id;
                output_block_q <= low_output_block;
                output_narrow_q <= request_narrow;
                output_fused_q <= request_fuse_correction;
                request_accepts_q <= request_accepts_q + 1'b1;
                if (request_fuse_correction)
                    fused_accepts_q <= fused_accepts_q + 1'b1;
                else
                    plain_accepts_q <= plain_accepts_q + 1'b1;

                for (lane = 0; lane < 96; lane = lane + 1) begin
                    if (request_narrow) begin
                        if (!request_fuse_correction)
                            output_data_q[lane*13 +: 13] <= $signed({
                                {5{low_data[lane*8+7]}},
                                low_data[lane*8 +: 8]});
                        else if (correction_subtract)
                            output_data_q[lane*13 +: 13] <= $signed({
                                {5{low_data[lane*8+7]}},
                                low_data[lane*8 +: 8]}) - $signed({
                                {5{correction_data[lane*8+7]}},
                                correction_data[lane*8 +: 8]});
                        else
                            output_data_q[lane*13 +: 13] <= $signed({
                                {5{low_data[lane*8+7]}},
                                low_data[lane*8 +: 8]}) + $signed({
                                {5{correction_data[lane*8+7]}},
                                correction_data[lane*8 +: 8]});
                    end else begin
                        if (!request_fuse_correction)
                            output_data_q[lane*13 +: 13] <= $signed({
                                high_data[lane*4+3],
                                high_data[lane*4 +: 4],
                                low_data[lane*8 +: 8]});
                        else if (correction_subtract)
                            output_data_q[lane*13 +: 13] <= $signed({
                                high_data[lane*4+3],
                                high_data[lane*4 +: 4],
                                low_data[lane*8 +: 8]}) - $signed({
                                {5{correction_data[lane*8+7]}},
                                correction_data[lane*8 +: 8]});
                        else
                            output_data_q[lane*13 +: 13] <= $signed({
                                high_data[lane*4+3],
                                high_data[lane*4 +: 4],
                                low_data[lane*8 +: 8]}) + $signed({
                                {5{correction_data[lane*8+7]}},
                                correction_data[lane*8 +: 8]});
                    end
                end
            end

            if (contribution_accept)
                contributions_q <= contributions_q + 1'b1;

            case ({request_accept, pop_w})
                2'b10: output_valid_q <= 1'b1;
                2'b01: output_valid_q <= 1'b0;
                2'b11: output_valid_q <= 1'b1;
                default: output_valid_q <= output_valid_q;
            endcase
        end
    end
endmodule

`default_nettype wire
