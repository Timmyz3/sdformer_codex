`timescale 1ns/1ps
`default_nettype none

// Dual-row 512-bit elastic PWP assembler.
//
// This is the executable service-port cut for M132.  It consumes two logical
// 256-bit rows per accepted beat and reconstructs one signed 96-lane vector in
// 2/2/2/3 cycles for signed 8/9/10/11-bit payloads.  The upstream 16-word bank
// mapper and foundry macro are intentionally outside this standalone module.
module m133_dualrow512_elastic_pwp_stream #(
    parameter int LANES = 96,
    parameter int BEAT_W = 512,
    parameter int MAX_BEATS = 3,
    parameter int OUT_W = 12,
    parameter int TAG_W = 32
) (
    input  logic                         clk_core,
    input  logic                         rst_core,

    input  logic                         beat_valid,
    output logic                         beat_ready,
    input  logic                         beat_start,
    input  logic                         beat_last,
    input  logic [3:0]                   beat_width,
    input  logic [TAG_W-1:0]             beat_tag,
    input  logic [BEAT_W-1:0]            beat_data,
    output logic                         beat_accept,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [TAG_W-1:0]             output_tag,
    output logic [3:0]                   output_width,
    output logic                         output_escape,
    output logic [LANES*OUT_W-1:0]       output_values,
    output logic                         output_accept,

    output logic                         protocol_error,
    output logic                         collecting,
    output logic                         busy
);
    localparam int BUFFER_W = BEAT_W * MAX_BEATS;

    logic collecting_q;
    logic faulted_q;
    logic [1:0] accepted_beats_q;
    logic [1:0] beats_needed_q;
    logic [3:0] active_width_q;
    logic [TAG_W-1:0] active_tag_q;
    logic [BUFFER_W-1:0] buffer_q;

    logic output_valid_q;
    logic [TAG_W-1:0] output_tag_q;
    logic [3:0] output_width_q;
    logic output_escape_q;
    logic [LANES*OUT_W-1:0] output_values_q;

    logic [BUFFER_W-1:0] buffer_with_beat;
    logic [LANES*OUT_W-1:0] assembled_values;
    logic expected_last;
    logic final_padding_zero;
    logic request_violation;
    logic quarantine;
    logic input_capacity;

`ifndef SYNTHESIS
    initial begin
        if (LANES != 96 || BEAT_W != 512 || MAX_BEATS != 3
                || OUT_W != 12)
            $fatal(1, "M133 production geometry drift");
    end
`endif

    assign expected_last = (accepted_beats_q + 1'b1 == beats_needed_q);

    always_comb begin : assemble_and_audit
        buffer_with_beat = buffer_q;
        case (accepted_beats_q)
            2'd1: buffer_with_beat[1*BEAT_W +: BEAT_W] = beat_data;
            2'd2: buffer_with_beat[2*BEAT_W +: BEAT_W] = beat_data;
            default: buffer_with_beat = buffer_q;
        endcase

        assembled_values = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            case (active_width_q)
                4'd8: assembled_values[lane*OUT_W +: OUT_W] = {
                    {4{buffer_with_beat[lane*8 + 7]}},
                    buffer_with_beat[lane*8 +: 8]};
                4'd9: assembled_values[lane*OUT_W +: OUT_W] = {
                    {3{buffer_with_beat[lane*9 + 8]}},
                    buffer_with_beat[lane*9 +: 9]};
                4'd10: assembled_values[lane*OUT_W +: OUT_W] = {
                    {2{buffer_with_beat[lane*10 + 9]}},
                    buffer_with_beat[lane*10 +: 10]};
                4'd11: assembled_values[lane*OUT_W +: OUT_W] = {
                    buffer_with_beat[lane*11 + 10],
                    buffer_with_beat[lane*11 +: 11]};
                default: assembled_values[lane*OUT_W +: OUT_W] = '0;
            endcase
        end
        case (active_width_q)
            4'd8: final_padding_zero = !(|buffer_with_beat[1023:768]);
            4'd9: final_padding_zero = !(|buffer_with_beat[1023:864]);
            4'd10: final_padding_zero = !(|buffer_with_beat[1023:960]);
            4'd11: final_padding_zero = !(|buffer_with_beat[1535:1056]);
            default: final_padding_zero = 1'b0;
        endcase

        request_violation = 1'b0;
        if (beat_valid) begin
            if (!collecting_q) begin
                if (!beat_start) begin
                    request_violation = 1'b1;
                end else if (beat_width == 4'd12) begin
                    request_violation = !beat_last || beat_data != '0;
                end else if (beat_width inside
                             {4'd8, 4'd9, 4'd10, 4'd11}) begin
                    request_violation = beat_last;
                end else begin
                    request_violation = 1'b1;
                end
            end else begin
                request_violation = beat_start || beat_width != 0
                    || beat_tag != 0 || beat_last != expected_last;
                if (expected_last && !final_padding_zero)
                    request_violation = 1'b1;
            end
        end
    end

    assign quarantine = faulted_q || request_violation;
    assign input_capacity = !rst_core && !faulted_q
                          && (!output_valid_q || output_ready);
    // With valid low, ready is capacity-only and payload independent.
    assign beat_ready = input_capacity
                      && (!beat_valid || !request_violation);
    assign beat_accept = beat_valid && beat_ready;
    assign protocol_error = !rst_core && quarantine;

    assign output_valid = !rst_core && output_valid_q && !quarantine;
    assign output_tag = output_valid_q ? output_tag_q : '0;
    assign output_width = output_valid_q ? output_width_q : '0;
    assign output_escape = output_valid_q && output_escape_q;
    assign output_values = output_valid_q ? output_values_q : '0;
    assign output_accept = output_valid && output_ready;
    assign collecting = collecting_q;
    assign busy = collecting_q || output_valid_q;

    always_ff @(posedge clk_core) begin
        if (rst_core) begin
            collecting_q <= 1'b0;
            faulted_q <= 1'b0;
            accepted_beats_q <= '0;
            beats_needed_q <= '0;
            active_width_q <= '0;
            active_tag_q <= '0;
            buffer_q <= '0;
            output_valid_q <= 1'b0;
            output_tag_q <= '0;
            output_width_q <= '0;
            output_escape_q <= 1'b0;
            output_values_q <= '0;
        end else begin
            if (request_violation)
                faulted_q <= 1'b1;

            if (!quarantine) begin
                if (output_accept)
                    output_valid_q <= 1'b0;

                if (beat_accept) begin
                    if (!collecting_q) begin
                        if (beat_width == 4'd12) begin
                            output_valid_q <= 1'b1;
                            output_tag_q <= beat_tag;
                            output_width_q <= beat_width;
                            output_escape_q <= 1'b1;
                            output_values_q <= '0;
                        end else begin
                            collecting_q <= 1'b1;
                            accepted_beats_q <= 2'd1;
                            active_width_q <= beat_width;
                            active_tag_q <= beat_tag;
                            buffer_q <= '0;
                            buffer_q[0*BEAT_W +: BEAT_W] <= beat_data;
                            case (beat_width)
                                4'd8, 4'd9, 4'd10: beats_needed_q <= 2;
                                default: beats_needed_q <= 3;
                            endcase
                        end
                    end else if (expected_last) begin
                        collecting_q <= 1'b0;
                        accepted_beats_q <= '0;
                        output_valid_q <= 1'b1;
                        output_tag_q <= active_tag_q;
                        output_width_q <= active_width_q;
                        output_escape_q <= 1'b0;
                        output_values_q <= assembled_values;
                    end else begin
                        accepted_beats_q <= accepted_beats_q + 1'b1;
                        buffer_q <= buffer_with_beat;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
